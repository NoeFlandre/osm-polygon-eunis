from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box, mapping

from osm_polygon_eunis import eea, pool_runner
from osm_polygon_eunis.eea import (
    ClassificationTaxonomyCacheLineage,
    ClassificationTaxonomyLineagePin,
    parse_classification_taxonomy_rows,
    verify_classification_taxonomy_lineage,
)
from osm_polygon_eunis.eunis_asset_manifest import (
    EEA_2021_MAP_RELEASE,
    EEA_2021_NATIVE_ASSET_INVENTORY_SHA256,
    EEA_2021_NATIVE_MANIFEST_SHA256,
    EunisRasterAssetManifest,
    EunisRasterAssetPin,
    SourceOutputSemanticsEvidence,
)
from osm_polygon_eunis.pool_adapter import (
    EUNIS_POOL_SOURCE_LAYOUTS,
    EunisTaxonomy,
    EunisTaxonomyPath,
    SourceTreeFile,
    SourceTreeGroupPin,
    eunis_area_measurement_pin,
    make_eunis_taxonomy_from_classification,
    source_tree_inventory_sha256,
    validate_source_tree_group,
)
from osm_polygon_eunis.pool_contract import (
    ArtifactManifestPin,
    FilesystemArtifactReader,
    FilesystemStreamingSink,
    ReferencePin,
    RelevanceStatus,
    SourceSnapshot,
    StreamingLimits,
    SupportSemantics,
    TextLinkStatus,
    TextSource,
    native_keyset_sha256,
)
from osm_polygon_eunis.pool_contract.builder import canonical_native_key
from osm_polygon_eunis.pool_runner import (
    DuplicateNativeKeyError,
    FrozenPolygonSource,
    PinnedPolygonShard,
    PoolAdapterCodePin,
    export_eunis_full_pool,
    export_eunis_pool,
    iter_attempt_checkpoint,
    iter_eunis_attempt_inventory_rows,
    iter_eunis_wiki_downstream_records,
    read_projected_geometry_wkb,
    read_raw_geometry,
    run_eunis_label_attempts,
    scan_raw_inventory,
)
from osm_polygon_eunis.raster_class_support import ClassificationAreas, ScoreDiagnostics
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference
from osm_polygon_eunis.text_join import (
    BoundedJoinedTextRecords,
    JoinAllocationBounds,
    PinnedSourceRow,
    iter_eunis_join_records,
)

_TAXONOMY = EunisTaxonomy(
    taxonomy_id="fixture-taxonomy",
    taxonomy_revision="fixture-taxonomy-v1",
    taxonomy_sha256="c" * 64,
    paths_by_direct_code={"C1": EunisTaxonomyPath("A", "A1", "C1")},
)


@pytest.fixture(autouse=True)
def _verified_fixture_taxonomy(monkeypatch: pytest.MonkeyPatch) -> None:
    classification = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "A", "Root A"],
            [2, "A1", "Branch A1"],
            [3, "A11", "Class A11"],
        ]
    )
    cache_payload = b"fixture taxonomy cache bytes"
    workbook_payload = b"fixture workbook bytes"
    lineage_pin = ClassificationTaxonomyLineagePin(
        hashlib.sha256(cache_payload).hexdigest(),
        "fixture-classification-record",
        hashlib.sha256(workbook_payload).hexdigest(),
    )
    monkeypatch.setattr(
        eea,
        "parse_classification_taxonomy_workbook",
        lambda _payload, *, supplemental_labels=None: classification,
    )
    verified = verify_classification_taxonomy_lineage(
        classification,
        cache_payload,
        workbook_payload,
        ClassificationTaxonomyCacheLineage(
            lineage_pin.classification_record_id,
            lineage_pin.workbook_sha256,
        ),
        lineage_pin,
    )
    fixture_taxonomy = make_eunis_taxonomy_from_classification(
        verified,
        ("A11",),
        taxonomy_id="fixture-taxonomy",
        taxonomy_revision="fixture-taxonomy-v1",
        taxonomy_sha256=lineage_pin.cache_sha256,
    )
    monkeypatch.setattr(pool_runner, "EEA_2021_CLASSIFICATION_LINEAGE_PIN", lineage_pin)
    monkeypatch.setitem(globals(), "_TAXONOMY", fixture_taxonomy)


@dataclass
class FakeReference:
    """Small deterministic replacement for the expensive production raster set."""

    fail: bool = False
    query_count: int = 0
    source_version: str = "EEA test map release"

    @property
    def layer_codes(self) -> tuple[str, ...]:
        return ("A11",)

    def classification_areas(
        self,
        polygon: Any,
        _classes: dict[tuple[str, str], tuple[str, ...]],
    ) -> ClassificationAreas:
        self.query_count += 1
        if self.fail:
            raise RuntimeError("fixture query error")
        support = float(polygon.area) * 0.8
        return ClassificationAreas(
            {("L1", "A"): support, ("L2", "A1"): support, ("L3", "A11"): support},
            ScoreDiagnostics(
                unknown_area_m2=0.0,
                zero_max_score_area_m2=float(polygon.area) * 0.2,
                ambiguous_area_m2_by_level={"L1": 0.0, "L2": 0.0, "L3": 0.0},
                maximum_source_score_min=0.8,
                maximum_source_score_max=0.8,
                maximum_source_score_area_weighted_mean=0.8,
            ),
        )


def _source_row(osm_id: int, geometry: str | None, *, source_pbf: str = "region.osm.pbf") -> dict:
    return {
        "source_pbf": source_pbf,
        "osm_type": "way",
        "osm_id": osm_id,
        "geometry": geometry,
    }


def _geojson_polygon() -> str:
    return json.dumps(mapping(box(0.0, 40.0, 0.001, 40.001)), separators=(",", ":"))


def _source_snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        snapshot_id="website-fixture-snapshot",
        repository="NoeFlandre/osm-polygon-website-tag",
        revision=EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE].raw_revision,
        pbf_uri="https://source.example/pbf-manifest.json",
        pbf_sha256="a" * 64,
    )


def _write_source_shard(path: Path, rows: list[dict]) -> PinnedPolygonShard:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema(
            [
                pa.field("source_pbf", pa.string()),
                pa.field("osm_type", pa.string()),
                pa.field("osm_id", pa.int64()),
                pa.field("geometry", pa.string()),
            ]
        ),
    )
    pq.write_table(table, path)
    content = path.read_bytes()
    relative_path = f"polygons/{path.name}"
    blob_id = hashlib.sha1(
        f"blob {len(content)}\0".encode() + content,
        usedforsecurity=False,
    ).hexdigest()
    return PinnedPolygonShard(
        path_in_repo=relative_path,
        revision=_source_snapshot().revision,
        sha256=hashlib.sha256(content).hexdigest(),
        size_bytes=len(content),
        blob_id=blob_id,
        lfs_oid=None,
        lfs_size=len(content),
        local_path=path,
    )


def _fixture_layout(shard: PinnedPolygonShard):
    base = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE]
    file = SourceTreeFile(
        path=shard.path_in_repo,
        size_bytes=shard.size_bytes,
        blob_id=shard.blob_id,
        lfs_oid=shard.lfs_oid,
        lfs_size=shard.lfs_size,
    )
    pin = SourceTreeGroupPin(
        repository=base.repository,
        revision=base.raw_revision,
        role="raw_geometry",
        path_prefix="polygons/",
        file_count=1,
        total_size_bytes=shard.size_bytes,
        inventory_sha256=source_tree_inventory_sha256((file,)),
    )
    return replace(base, raw_polygon_tree_pin=pin)


def _frozen_source(tmp_path: Path, rows: list[dict]) -> FrozenPolygonSource:
    shard = _write_source_shard(tmp_path / "staged" / "region-a.parquet", rows)
    source_artifact_manifest = _artifact_manifest_pin(
        tmp_path,
        "source-artifacts",
        (shard.local_path,),
    )
    return FrozenPolygonSource(
        layout=_fixture_layout(shard),
        source_snapshot=replace(
            _source_snapshot(),
            source_artifact_manifest=source_artifact_manifest,
        ),
        manifest_uri="https://source.example/raw-source-manifest.json",
        manifest_sha256="b" * 64,
        manifest_paths=(shard.path_in_repo, "dataset-card.md"),
        polygon_shards=(shard,),
    )


def _artifact_manifest_pin(
    tmp_path: Path,
    name: str,
    artifact_paths: tuple[Path, ...],
) -> ArtifactManifestPin:
    rows = [
        {
            "artifact_uri": path.resolve().as_uri(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(artifact_paths, key=lambda item: item.resolve().as_uri())
    ]
    content = b"".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode() + b"\n" for row in rows
    )
    manifest_path = tmp_path / f"{name}.jsonl"
    manifest_path.write_bytes(content)
    return ArtifactManifestPin(
        format_version="artifact-list-jsonl-v1",
        manifest_uri=manifest_path.resolve().as_uri(),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        expected_artifact_count=len(rows),
    )


def _reference_pin(tmp_path: Path | None = None) -> ReferencePin:
    reference_artifact_manifest = None
    if tmp_path is not None:
        raster_path = tmp_path / "reference-fixture" / "Prob_A11_10m.tif"
        raster_path.parent.mkdir(parents=True, exist_ok=True)
        scores = np.asarray([[0.8]], dtype=np.float32)
        with rasterio.open(
            raster_path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype=scores.dtype,
            crs="EPSG:3035",
            transform=from_origin(0, 10, 10, 10),
            nodata=-9999.0,
        ) as dataset:
            dataset.write(scores, 1)
            dataset.scales = (1.0,)
            dataset.offsets = (0.0,)
        reference_artifact_manifest = _artifact_manifest_pin(
            tmp_path,
            "reference-artifacts",
            (raster_path,),
        )
    return ReferencePin(
        release_id="EEA test map release",
        source_repository="https://reference.example/eunis",
        source_revision="reference-manifest-sha256:" + "d" * 64,
        taxonomy_id=_TAXONOMY.taxonomy_id,
        taxonomy_revision=_TAXONOMY.pinned_revision,
        support_semantics=SupportSemantics.MODELED_SCORE_DERIVED_CLASSIFICATION,
        support_unit="m2",
        threshold=0.8,
        support_union_policy="highest_source_score_per_cell_then_unique_class_area_per_level:v1",
        reference_artifact_manifest=reference_artifact_manifest,
        area_measurement=eunis_area_measurement_pin(),
    )


def _byte_pinned_fixture_reference(tmp_path: Path) -> tuple[RasterReference, ReferencePin]:
    reference_pin = _reference_pin(tmp_path)
    raster_path = tmp_path / "reference-fixture" / "Prob_A11_10m.tif"
    evidence = SourceOutputSemanticsEvidence(
        "synthetic-suitability-score",
        "test-fixture://eunis/output-semantics",
        "b" * 64,
    )
    asset = EunisRasterAssetPin(
        code="A11",
        filename="Prob_A11_10m.tif",
        size_bytes=raster_path.stat().st_size,
        sha256=hashlib.sha256(raster_path.read_bytes()).hexdigest(),
        output_evidence=evidence,
    )
    reference = RasterReference(
        (
            RasterLayer(
                "A11",
                "fixture class",
                raster_path,
                "EEA test map release",
                "Prob_A11_10m.tif",
            ),
        ),
        asset_manifest=EunisRasterAssetManifest("d" * 64, (asset,)),
    )
    return reference, reference_pin


def test_label_runner_fails_closed_without_byte_verified_taxonomy_lineage(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    raw_scan = scan_raw_inventory(source, tmp_path / "raw-checkpoints", _code_pin())
    unverified = EunisTaxonomy(
        taxonomy_id=_TAXONOMY.taxonomy_id,
        taxonomy_revision=_TAXONOMY.taxonomy_revision,
        taxonomy_sha256=_TAXONOMY.taxonomy_sha256,
        paths_by_direct_code=dict(_TAXONOMY.paths_by_direct_code),
    )
    attempts_root = tmp_path / "attempts-not-created"

    with pytest.raises(ValueError, match="requires verified workbook lineage"):
        run_eunis_label_attempts(
            raw_scan,
            cast(RasterReference, FakeReference()),
            _reference_pin(),
            unverified,
            attempts_root,
        )

    assert not attempts_root.exists()


def _code_pin() -> PoolAdapterCodePin:
    return PoolAdapterCodePin(
        adapter_revision="a" * 40,
        builder_revision="b" * 40,
    )


def test_raw_scan_is_complete_and_uses_core_canonical_key_digest(tmp_path: Path) -> None:
    source = _frozen_source(
        tmp_path,
        [
            _source_row(11, _geojson_polygon()),
            _source_row(12, None, source_pbf="other-region.osm.pbf"),
        ],
    )

    scan = scan_raw_inventory(
        source,
        tmp_path / "checkpoints",
        _code_pin(),
        batch_size=1,
        key_chunk_size=1,
    )

    keys = (
        canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 11}),
        canonical_native_key(
            {"source_pbf": "other-region.osm.pbf", "osm_type": "way", "osm_id": 12}
        ),
    )
    assert scan.evidence.expected_key_count == 2
    assert scan.evidence.expected_keyset_sha256 == native_keyset_sha256(keys)
    assert scan.evidence.enumeration_rule.startswith("enumerate every geometry row")
    raw_rows = [
        json.loads(line)
        for line in Path(scan.shards[0].checkpoint_uri.removeprefix("file://"))
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(raw_rows) == 2
    assert raw_rows[1]["geometry_payload"] == {"kind": "missing"}
    assert all("source_row_locator" in row for row in raw_rows)


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("source_sha256", "0" * 64),
        ("source_size_bytes", 0),
        ("row_count", 0),
        ("checkpoint_sha256", "1" * 64),
    ],
)
def test_raw_shard_report_receipt_checks_each_pinned_field(
    tmp_path: Path,
    field: str,
    replacement: Any,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, None)])
    shard = scan_raw_inventory(source, tmp_path / "checkpoints", _code_pin()).shards[0]
    row = asdict(shard)

    pool_runner._validate_raw_shard_receipt(row, shard)
    row[field] = replacement
    with pytest.raises(ValueError, match="receipt changed"):
        pool_runner._validate_raw_shard_receipt(row, shard)


def test_scan_and_code_pins_reject_nonpositive_limits_and_invalid_revisions(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, None)])
    with pytest.raises(ValueError, match="chunk sizes must be positive"):
        scan_raw_inventory(source, tmp_path / "invalid", _code_pin(), batch_size=0)
    with pytest.raises(ValueError, match="adapter revision must be a full lowercase Git SHA"):
        PoolAdapterCodePin("not-a-git-revision", "b" * 40)


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("threshold", 0.7, "exact 0.8 area threshold"),
        (
            "support_union_policy",
            "sum_by_class",
            "pinned highest-source-score area policy",
        ),
        (
            "support_semantics",
            SupportSemantics.MAPPED_COVER,
            "modeled score-derived classification",
        ),
        ("support_unit", "km2", "must use m2"),
    ],
)
def test_eunis_support_policy_rejects_each_unpinned_choice(
    field: str,
    value: Any,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=error):
        pool_runner._validate_eunis_support_policy(replace(_reference_pin(), **{field: value}))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("source_repository", " "),
        ("source_revision", " "),
        ("release_id", " "),
        ("taxonomy_id", " "),
        ("taxonomy_revision", " "),
    ],
)
def test_eunis_reference_provenance_requires_every_pinned_field(field: str, value: str) -> None:
    with pytest.raises(ValueError, match="must be pinned"):
        pool_runner._validate_eunis_provenance(replace(_reference_pin(), **{field: value}))


def test_eunis_reference_provenance_checks_optional_area_method() -> None:
    reference_pin = _reference_pin()
    area_pin = reference_pin.area_measurement
    assert area_pin is not None
    pool_runner._validate_eunis_provenance(replace(reference_pin, area_measurement=None))
    bad_numerator = replace(area_pin.numerator, method_id="wrong-method")
    bad_area_pin = replace(area_pin, numerator=bad_numerator)

    with pytest.raises(ValueError, match="area measurement pin does not match"):
        pool_runner._validate_eunis_provenance(
            replace(reference_pin, area_measurement=bad_area_pin)
        )


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, {"kind": "missing"}),
        (b"binary", {"kind": "base64", "value": "YmluYXJ5"}),
        (bytearray(b"binary"), {"kind": "base64", "value": "YmluYXJ5"}),
        (memoryview(b"binary"), {"kind": "base64", "value": "YmluYXJ5"}),
        ("WKT polygon", {"kind": "text", "value": "WKT polygon"}),
        ({"coordinates": [1, 2]}, {"kind": "json", "value": {"coordinates": [1, 2]}}),
        ([1, 2], {"kind": "json", "value": [1, 2]}),
        (4, {"kind": "json", "value": 4}),
        (4.5, {"kind": "json", "value": 4.5}),
        (False, {"kind": "json", "value": False}),
        (object(), {"kind": "unsupported", "value": "object"}),
    ],
)
def test_geometry_payload_encodings_are_round_trip_safe(
    value: Any, expected: dict[str, Any]
) -> None:
    encoded = pool_runner._encode_geometry_payload(value)
    assert encoded == expected
    decoded = pool_runner._decode_geometry_payload(encoded)
    assert decoded == (None if expected["kind"] == "unsupported" else value)


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "unknown"},
        {"kind": "base64", "value": "not base64!"},
    ],
)
def test_geometry_payload_decoder_rejects_bad_encodings(payload: dict[str, Any]) -> None:
    with pytest.raises((ValueError, KeyError)):
        pool_runner._decode_geometry_payload(payload)


def test_empty_accumulator_flushes_cleanly_and_rejects_an_oversized_polygon() -> None:
    accumulator = pool_runner._PoolBatchAccumulator(
        limits=StreamingLimits(max_batch_rows=1),
        max_polygons_per_batch=1,
    )
    assert accumulator.flush() is None
    with pytest.raises(ValueError, match="one polygon attempt has"):
        accumulator.add(cast(Any, None), None, (), (), cast(Any, None), ())


def test_raw_inventory_digest_merges_more_than_one_bounded_key_run(tmp_path: Path) -> None:
    rows = [_source_row(osm_id, None) for osm_id in range(1, 66)]
    source = _frozen_source(tmp_path, rows)

    scan = scan_raw_inventory(
        source,
        tmp_path / "merge-checkpoints",
        _code_pin(),
        batch_size=7,
        key_chunk_size=1,
    )

    keys = (
        canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": osm_id})
        for osm_id in range(1, 66)
    )
    assert scan.evidence.expected_key_count == 65
    assert scan.evidence.expected_keyset_sha256 == native_keyset_sha256(keys)


def test_raw_scan_resumes_verified_shards_without_reopening_source(
    tmp_path: Path, monkeypatch
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    root = tmp_path / "checkpoints"
    first = scan_raw_inventory(source, root, _code_pin())

    def unexpected_rescan(_shard: PinnedPolygonShard) -> None:
        raise AssertionError("valid raw shard checkpoint should be resumed")

    monkeypatch.setattr("osm_polygon_eunis.pool_runner._verify_local_shard", unexpected_rescan)
    second = scan_raw_inventory(source, root, _code_pin())

    assert second.evidence == first.evidence
    assert second.shards == first.shards


def test_handoff_source_tree_inventory_pins_match_checked_in_provenance() -> None:
    manifest_path = (
        Path(__file__).resolve().parents[2] / "config/eunis-source-provenance-20261021.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema"]["metadata_handoff_sha256"] == (
        "d1c4bf9e60468a85b17c535be17ae4efeba7284b718d60235e19a39a9e65a072"
    )
    assert manifest["schema"]["raw_native_key_counts"] is None
    assert "/srv/storage/" not in manifest_path.read_text(encoding="utf-8")

    groups = {
        (tree["repository"], tree["revision"], group["path_prefix"]): group
        for section in ("raw_source_file_trees", "relevance_label_file_trees")
        for tree in manifest[section]
        for group in tree["groups"]
    }
    wiki_tree = manifest["wiki_document_link_file_trees"]
    groups.update(
        {
            (wiki_tree["repository"], wiki_tree["revision"], group["path_prefix"]): group
            for group in wiki_tree["groups"]
        }
    )
    pins = [
        pin
        for layout in EUNIS_POOL_SOURCE_LAYOUTS.values()
        for pin in (layout.raw_polygon_tree_pin, *layout.additional_tree_pins)
    ]
    assert len(groups) == len(pins) == 14
    for pin in pins:
        group = groups[(pin.repository, pin.revision, pin.path_prefix)]
        files = tuple(
            SourceTreeFile(
                path=file["path"],
                size_bytes=file["size"],
                blob_id=file["blob_id"],
                lfs_oid=file["lfs_oid"],
                lfs_size=file["lfs_size"],
            )
            for file in group["files"]
        )
        assert group["file_count"] == pin.file_count
        assert group["total_size_bytes"] == pin.total_size_bytes
        assert group["inventory_sha256"] == pin.inventory_sha256
        validate_source_tree_group(pin, files)
        altered_first = replace(files[0], blob_id="0" * 40)
        if altered_first.blob_id == files[0].blob_id:
            altered_first = replace(files[0], blob_id="1" * 40)
        with pytest.raises(ValueError, match="inventory digest differs"):
            validate_source_tree_group(pin, (altered_first, *files[1:]))

    reference = manifest["reference"]
    assert reference["native_manifest_sha256"] == (
        "4a91e553b2e0271e7ca935e542dd092fe45288c5f9bb4a09f40f6b74da4eae22"
    )
    assert reference["native_asset_count"] == len(reference["native_assets"]) == 136
    assert (
        sum(row["size_bytes"] for row in reference["native_assets"])
        == (reference["native_asset_total_bytes"])
    )
    assert reference["taxonomy"]["workbook_bytes"] == 1_439_940
    assert reference["taxonomy"]["workbook_direct_url"].endswith(
        "EUNIS%20terrestrial%20habitat%20classification%202021_1%20including%20crosswalks.xlsx"
    )
    assert reference["taxonomy"]["crosswalk_rows"] is None


def test_manifest_pin_must_cover_every_polygon_shard(tmp_path: Path) -> None:
    shard = _write_source_shard(tmp_path / "staged" / "region-a.parquet", [_source_row(1, None)])

    with pytest.raises(ValueError, match="exactly equal"):
        FrozenPolygonSource(
            layout=EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE],
            source_snapshot=_source_snapshot(),
            manifest_uri="https://source.example/manifest.json",
            manifest_sha256="b" * 64,
            manifest_paths=("polygons/region-b.parquet",),
            polygon_shards=(shard,),
        )


def test_source_snapshot_revision_must_be_an_immutable_git_sha(tmp_path: Path) -> None:
    shard = _write_source_shard(tmp_path / "staged" / "region-a.parquet", [_source_row(1, None)])
    snapshot = _source_snapshot()
    invalid_snapshot = SourceSnapshot(
        snapshot_id=snapshot.snapshot_id,
        repository=snapshot.repository,
        revision="main",
        pbf_uri=snapshot.pbf_uri,
        pbf_sha256=snapshot.pbf_sha256,
    )

    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        FrozenPolygonSource(
            layout=EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE],
            source_snapshot=invalid_snapshot,
            manifest_uri="https://source.example/manifest.json",
            manifest_sha256="b" * 64,
            manifest_paths=(shard.path_in_repo,),
            polygon_shards=(shard,),
        )


def test_source_snapshot_revision_must_match_its_pinned_eunis_layout(tmp_path: Path) -> None:
    shard = _write_source_shard(tmp_path / "staged" / "region-a.parquet", [_source_row(1, None)])
    unexpected_revision = "e" * 40
    snapshot = _source_snapshot()
    mismatched_snapshot = SourceSnapshot(
        snapshot_id=snapshot.snapshot_id,
        repository=snapshot.repository,
        revision=unexpected_revision,
        pbf_uri=snapshot.pbf_uri,
        pbf_sha256=snapshot.pbf_sha256,
    )
    mismatched_shard = PinnedPolygonShard(
        path_in_repo=shard.path_in_repo,
        revision=unexpected_revision,
        sha256=shard.sha256,
        size_bytes=shard.size_bytes,
        blob_id=shard.blob_id,
        lfs_oid=shard.lfs_oid,
        lfs_size=shard.lfs_size,
        local_path=shard.local_path,
    )

    with pytest.raises(ValueError, match="does not match the pinned EUNIS layout"):
        FrozenPolygonSource(
            layout=EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE],
            source_snapshot=mismatched_snapshot,
            manifest_uri="https://source.example/manifest.json",
            manifest_sha256="b" * 64,
            manifest_paths=(shard.path_in_repo,),
            polygon_shards=(mismatched_shard,),
        )


def test_duplicate_raw_native_keys_fail_instead_of_silently_deduplicating(
    tmp_path: Path,
) -> None:
    source = _frozen_source(
        tmp_path,
        [_source_row(11, _geojson_polygon()), _source_row(11, _geojson_polygon())],
    )

    with pytest.raises(DuplicateNativeKeyError, match="duplicate raw source-native"):
        scan_raw_inventory(source, tmp_path / "checkpoints", _code_pin(), key_chunk_size=1)


def test_label_attempts_follow_raw_inventory_and_keep_missing_geometry_outcome(
    tmp_path: Path,
) -> None:
    source = _frozen_source(
        tmp_path,
        [_source_row(11, _geojson_polygon()), _source_row(12, None)],
    )
    root = tmp_path / "checkpoints"
    raw = scan_raw_inventory(source, root, _code_pin())
    reference = FakeReference()

    attempts = run_eunis_label_attempts(
        raw,
        cast(RasterReference, reference),
        _reference_pin(tmp_path),
        _TAXONOMY,
        root,
    )

    assert reference.query_count == 1
    assert attempts.shards[0].row_count == raw.evidence.expected_key_count == 2
    assert dict(attempts.shards[0].attempt_counts) == {"completed": 1, "excluded": 1}
    assert dict(attempts.shards[0].outcome_counts) == {
        "L1:assigned": 1,
        "L2:assigned": 1,
        "L1:excluded": 1,
        "L2:excluded": 1,
        "L3:assigned": 1,
        "L3:excluded": 1,
    }
    rows = list(iter_attempt_checkpoint(attempts.shards[0].checkpoint_uri))
    assert len(rows) == 2
    assert len({row["inventory"]["inventory_row_id"] for row in rows}) == 2
    missing = next(row for row in rows if row["inventory"]["exclusion_reason"])
    assert missing["inventory"]["exclusion_reason"] == "missing_geometry"
    assert missing["attempt_receipt"]["status"] == "excluded"
    assert missing["attempt_receipt"]["reason"] == "missing_geometry"
    assert missing["attempt_receipt"]["attempted_reference_ids"] == []
    assert [item["status"] for item in missing["outcomes"]] == ["excluded"] * 3
    usable = next(row for row in rows if row["polygon"] is not None)
    assert usable["attempt_receipt"]["status"] == "completed"
    assert usable["attempt_receipt"]["attempted_reference_ids"] == ["A11"]
    assert usable["score_diagnostics"]["method"] == ("highest-source-score-derived-classification")
    assert usable["score_diagnostics"]["maximum_source_score_min"] == 0.8
    assert usable["score_diagnostics"]["maximum_source_score_area_weighted_mean"] == 0.8
    assert len(usable["support_evidence"]) == len(usable["candidates"]) == 3
    assert {tuple(item["reference_ids"]) for item in usable["support_evidence"]} == {("A11",)}
    assert usable["polygon"]["geometry_uri"].startswith("file://")
    assert usable["projected_geometry_wkb_base64"]
    projected_wkb = read_projected_geometry_wkb(usable["polygon"]["geometry_uri"])
    assert hashlib.sha256(projected_wkb).hexdigest() == usable["polygon"]["geometry_sha256"]
    assert read_raw_geometry(usable["raw_geometry_checkpoint_uri"]) == _geojson_polygon()
    assert {item["assignment_kind"] for item in usable["candidates"]} == {
        "direct",
        "ancestor_derived",
    }
    report = json.loads(Path(attempts.report_uri.removeprefix("file://")).read_text())
    assert report["total_attempted_polygons"] == 2
    assert report["attempt_counts_by_status"] == dict(attempts.shards[0].attempt_counts)
    assert report["outcome_counts_by_level_status"] == dict(attempts.shards[0].outcome_counts)


def test_incomplete_raster_inventory_fails_before_attempt_checkpoint_writes(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    raw_scan = scan_raw_inventory(source, tmp_path / "raw-checkpoints", _code_pin())
    paths = {}
    for code in ("A11", "A12"):
        path = tmp_path / f"{code}.tif"
        values = np.asarray([[0.8]], dtype=np.float32)
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=1,
            width=1,
            count=1,
            dtype=values.dtype,
            crs="EPSG:3035",
            transform=from_origin(0, 10, 10, 10),
        ) as dataset:
            dataset.write(values, 1)
        paths[code] = path
    output_evidence = SourceOutputSemanticsEvidence(
        "synthetic-suitability-score",
        "test-fixture://eunis/output-semantics",
        "b" * 64,
    )
    manifest = EunisRasterAssetManifest(
        "a" * 64,
        tuple(
            EunisRasterAssetPin(
                code=code,
                filename=f"Prob_{code}_100m.tif",
                size_bytes=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                output_evidence=output_evidence,
            )
            for code, path in paths.items()
        ),
    )
    reference = RasterReference(
        (
            RasterLayer(
                "A11",
                "fixture class",
                paths["A11"],
                "EEA test map release",
                "Prob_A11_100m.tif",
            ),
        ),
        asset_manifest=manifest,
    )
    attempts_root = tmp_path / "attempts-not-created"

    with pytest.raises(ValueError, match=r"missing=\['A12'\]"):
        run_eunis_label_attempts(
            raw_scan,
            reference,
            _reference_pin(tmp_path),
            _TAXONOMY,
            attempts_root,
        )

    assert not attempts_root.exists()


def test_reference_artifact_list_must_match_the_verified_raster_inventory(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    raw_scan = scan_raw_inventory(source, tmp_path / "raw-checkpoints", _code_pin())
    reference, reference_pin = _byte_pinned_fixture_reference(tmp_path)
    unrelated = tmp_path / "unrelated-valid-artifact.bin"
    unrelated.write_bytes(b"valid, but not an EUNIS raster")
    unrelated_manifest = _artifact_manifest_pin(tmp_path, "unrelated-artifacts", (unrelated,))
    unrelated_reference_pin = replace(
        reference_pin,
        reference_artifact_manifest=unrelated_manifest,
    )
    failed_attempt_root = tmp_path / "attempts-with-unrelated-reference"

    with pytest.raises(ValueError, match="outside the verified raster inventory"):
        run_eunis_label_attempts(
            raw_scan,
            reference,
            unrelated_reference_pin,
            _TAXONOMY,
            failed_attempt_root,
        )

    assert not failed_attempt_root.exists()

    attempt_root = tmp_path / "attempts-with-pinned-reference"
    attempts = run_eunis_label_attempts(
        raw_scan,
        reference,
        reference_pin,
        _TAXONOMY,
        attempt_root,
    )
    assert attempts.verified_reference_inventory is not None

    # Re-pin the portable report to a different, internally valid artifact list.
    # Export must still compare that list with the inventory used for attempts.
    tampered_reference_pin = replace(
        attempts.reference_pin,
        reference_artifact_manifest=unrelated_manifest,
    )
    tampered_signature = pool_runner._attempt_signature(
        attempts.raw_scan,
        tampered_reference_pin,
        attempts.taxonomy,
        reference.layer_codes,
        attempts.code_pin,
    )
    report_path = pool_runner._local_file_uri_path(attempts.report_uri)
    report = json.loads(report_path.read_bytes())
    report["reference_pin"] = asdict(tampered_reference_pin)
    report["signature"] = tampered_signature
    report_bytes = pool_runner._canonical_json_bytes(report)
    report_path.write_bytes(report_bytes)
    tampered_attempts = replace(
        attempts,
        reference_pin=tampered_reference_pin,
        signature=tampered_signature,
        report_sha256=hashlib.sha256(report_bytes).hexdigest(),
    )
    sink_root = tmp_path / "pool-from-unrelated-reference"

    with pytest.raises(ValueError, match="outside the verified raster inventory"):
        export_eunis_pool(tampered_attempts, FilesystemStreamingSink(sink_root))

    assert not sink_root.exists()


@pytest.mark.parametrize(
    ("line", "message"),
    [
        (b"not-json\n", "invalid JSON at line 7"),
        (
            b'{"artifact_uri":"file:///a.tif","sha256":"' + b"a" * 64 + b'","extra":true}\n',
            "invalid schema",
        ),
        (
            b'{ "artifact_uri": "file:///a.tif", "sha256": "' + b"a" * 64 + b'"}\n',
            "not canonical JSONL",
        ),
        (
            pool_runner._canonical_json_bytes({"artifact_uri": 5, "sha256": "a" * 64}),
            "invalid artifact pin",
        ),
        (
            pool_runner._canonical_json_bytes(
                {"artifact_uri": "relative/a.tif", "sha256": "a" * 64}
            ),
            "invalid artifact pin",
        ),
        (
            pool_runner._canonical_json_bytes({"artifact_uri": "file:///a.tif", "sha256": 5}),
            "invalid artifact pin",
        ),
        (
            pool_runner._canonical_json_bytes(
                {"artifact_uri": "file:///a.tif", "sha256": "A" * 64}
            ),
            "invalid artifact pin",
        ),
    ],
)
def test_reference_artifact_manifest_parser_rejects_malformed_rows(
    line: bytes,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        pool_runner._parse_reference_artifact_row(line, 7)


def test_reference_artifact_manifest_parser_accepts_canonical_pin() -> None:
    row = {"artifact_uri": "file:///a.tif", "sha256": "a" * 64}
    assert pool_runner._parse_reference_artifact_row(pool_runner._canonical_json_bytes(row), 1) == (
        "file:///a.tif",
        "a" * 64,
    )


def _write_reference_manifest_rows(
    tmp_path: Path,
    name: str,
    rows: tuple[dict[str, Any], ...],
    *,
    expected_count: int | None = None,
) -> ArtifactManifestPin:
    content = b"".join(pool_runner._canonical_json_bytes(row) for row in rows)
    path = tmp_path / f"{name}.jsonl"
    path.write_bytes(content)
    return ArtifactManifestPin(
        format_version="artifact-list-jsonl-v1",
        manifest_uri=path.resolve().as_uri(),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        expected_artifact_count=len(rows) if expected_count is None else expected_count,
    )


def test_reference_artifact_inventory_verifier_rejects_incomplete_and_duplicate_lists(
    tmp_path: Path,
) -> None:
    reference, reference_pin = _byte_pinned_fixture_reference(tmp_path)
    inventory = reference.validate_classification_inventory()
    asset = inventory.assets[0]
    second_asset = replace(asset, code="A12", filename="Prob_A12_10m.tif")
    two_asset_inventory = EunisRasterAssetManifest(
        inventory.native_manifest_sha256,
        (asset, second_asset),
    )
    first_uri = "file:///a/Prob_A11_10m.tif"
    second_uri = "file:///z/Prob_A12_10m.tif"
    first_row = {"artifact_uri": first_uri, "sha256": asset.sha256}
    second_row = {"artifact_uri": second_uri, "sha256": second_asset.sha256}

    missing_pin = reference_pin.reference_artifact_manifest
    assert missing_pin is not None
    with pytest.raises(ValueError, match="reference artifact manifest pin is missing"):
        pool_runner._verify_reference_artifact_inventory(None, inventory)

    with pytest.raises(ValueError, match="count differs"):
        pool_runner._verify_reference_artifact_inventory(
            replace(missing_pin, expected_artifact_count=2), inventory
        )

    wrong_hash_pin = _write_reference_manifest_rows(
        tmp_path,
        "wrong-reference-hash",
        ({"artifact_uri": first_uri, "sha256": "e" * 64},),
    )
    with pytest.raises(ValueError, match="hash differs from verified raster"):
        pool_runner._verify_reference_artifact_inventory(wrong_hash_pin, inventory)

    incomplete_pin = _write_reference_manifest_rows(
        tmp_path,
        "incomplete-reference-list",
        (first_row,),
        expected_count=2,
    )
    with pytest.raises(ValueError, match="does not cover the verified raster inventory"):
        pool_runner._verify_reference_artifact_inventory(incomplete_pin, two_asset_inventory)

    duplicate_pin = _write_reference_manifest_rows(
        tmp_path,
        "duplicate-reference-list",
        (
            {"artifact_uri": "file:///a/Prob_A11_10m.tif", "sha256": asset.sha256},
            {"artifact_uri": "file:///z/Prob_A11_10m.tif", "sha256": asset.sha256},
        ),
        expected_count=2,
    )
    with pytest.raises(ValueError, match="repeats a verified raster"):
        pool_runner._verify_reference_artifact_inventory(duplicate_pin, two_asset_inventory)

    unsorted_pin = _write_reference_manifest_rows(
        tmp_path,
        "unsorted-reference-list",
        (second_row, first_row),
        expected_count=2,
    )
    with pytest.raises(ValueError, match="not uniquely sorted"):
        pool_runner._verify_reference_artifact_inventory(unsorted_pin, two_asset_inventory)


def test_production_eunis_attempts_fail_closed_without_output_semantics_pin(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    raw_scan = scan_raw_inventory(source, tmp_path / "raw-checkpoints", _code_pin())
    test_pin = _reference_pin(tmp_path)
    reference_pin = replace(
        test_pin,
        release_id=EEA_2021_MAP_RELEASE,
        source_revision=(
            f"native-manifest-sha256:{EEA_2021_NATIVE_MANIFEST_SHA256};"
            f"asset-inventory-sha256:{EEA_2021_NATIVE_ASSET_INVENTORY_SHA256};"
            "source-output-semantics-evidence-sha256:unverified"
        ),
    )
    attempts_root = tmp_path / "attempts-not-created"

    with pytest.raises(ValueError, match="verified per-asset source-output semantics evidence"):
        run_eunis_label_attempts(
            raw_scan,
            cast(RasterReference, FakeReference()),
            reference_pin,
            _TAXONOMY,
            attempts_root,
        )

    assert not attempts_root.exists()


def test_production_eunis_attempts_reject_unbound_reference_objects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    raw_scan = scan_raw_inventory(source, tmp_path / "raw-checkpoints", _code_pin())
    semantics_sha256 = "e" * 64
    monkeypatch.setattr(pool_runner, "EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256", semantics_sha256)
    reference_pin = replace(
        _reference_pin(tmp_path),
        release_id=EEA_2021_MAP_RELEASE,
        source_revision=(
            f"native-manifest-sha256:{EEA_2021_NATIVE_MANIFEST_SHA256};"
            f"asset-inventory-sha256:{EEA_2021_NATIVE_ASSET_INVENTORY_SHA256};"
            f"source-output-semantics-evidence-sha256:{semantics_sha256}"
        ),
    )
    reference = cast(
        RasterReference,
        FakeReference(source_version=EEA_2021_MAP_RELEASE),
    )
    attempts_root = tmp_path / "attempts-not-created"

    with pytest.raises(ValueError, match="requires a RasterReference bound"):
        run_eunis_label_attempts(
            raw_scan,
            reference,
            reference_pin,
            _TAXONOMY,
            attempts_root,
        )

    assert not attempts_root.exists()


def test_production_eunis_reference_pin_checks_all_official_revision_hashes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantics_sha256 = "e" * 64
    monkeypatch.setattr(pool_runner, "EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256", semantics_sha256)
    valid_revision = (
        f"native-manifest-sha256:{EEA_2021_NATIVE_MANIFEST_SHA256};"
        f"asset-inventory-sha256:{EEA_2021_NATIVE_ASSET_INVENTORY_SHA256};"
        f"source-output-semantics-evidence-sha256:{semantics_sha256}"
    )
    base_pin = replace(
        _reference_pin(tmp_path),
        release_id=EEA_2021_MAP_RELEASE,
    )
    pool_runner._validate_eunis_reference_pin(replace(base_pin, source_revision=valid_revision))

    for revision, error in (
        (
            valid_revision.replace(EEA_2021_NATIVE_MANIFEST_SHA256, "f" * 64),
            "native asset manifest",
        ),
        (
            valid_revision.replace(EEA_2021_NATIVE_ASSET_INVENTORY_SHA256, "f" * 64),
            "complete asset inventory",
        ),
        (valid_revision.replace(semantics_sha256, "f" * 64), "source-output evidence"),
    ):
        with pytest.raises(ValueError, match=error):
            pool_runner._validate_eunis_reference_pin(replace(base_pin, source_revision=revision))


def test_query_error_is_excluded_with_reason_at_every_supported_level(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    root = tmp_path / "checkpoints"
    raw = scan_raw_inventory(source, root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw,
        cast(RasterReference, FakeReference(fail=True)),
        _reference_pin(tmp_path),
        _TAXONOMY,
        root,
    )
    row = next(iter_attempt_checkpoint(attempts.shards[0].checkpoint_uri))

    assert [item["status"] for item in row["outcomes"]] == ["excluded"] * 3
    assert {item["reason"] for item in row["outcomes"]} == {"reference_query_error:RuntimeError"}
    assert row["attempt_receipt"]["status"] == "failed"
    assert row["attempt_receipt"]["reason"] == "reference_query_error:RuntimeError"


def test_eunis_attempts_export_through_bounded_shared_streaming_contract(
    tmp_path: Path,
) -> None:
    source = _frozen_source(
        tmp_path,
        [_source_row(11, _geojson_polygon()), _source_row(12, None)],
    )
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    sink_root = tmp_path / "streamed-pool"

    exported = export_eunis_pool(
        attempts,
        FilesystemStreamingSink(sink_root),
        limits=StreamingLimits(max_batch_rows=16),
        staging_directory=tmp_path,
    )

    receipt = exported.receipt
    assert receipt.total_source_keys == raw_scan.evidence.expected_key_count == 2
    assert receipt.inventory_keyset_sha256 == raw_scan.evidence.expected_keyset_sha256
    accounting = {item.level: item for item in receipt.level_accounting}
    for level in ("L1", "L2", "L3"):
        assert accounting[level].supported
        assert accounting[level].processed_decisions == 2
        assert accounting[level].assigned == 1
        assert accounting[level].excluded == 1
    for level in ("L4", "L5", "L6", "L7", "L8"):
        assert not accounting[level].supported
        assert accounting[level].processed_decisions == 0

    manifest_path = Path(receipt.manifest_uri.removeprefix("file://"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["pool_complete"] is True
    assert manifest["raw_inventory_evidence"]["scan_report_uri"] == (
        exported.raw_inventory_report_uri
    )
    assert Path(exported.raw_inventory_report_uri.removeprefix("file://")).is_file()
    assert Path(exported.label_attempt_report_uri.removeprefix("file://")).is_file()
    assert len(exported.attempt_checkpoint_uris) == 1

    shard_manifest_path = Path(receipt.shard_manifest_uri.removeprefix("file://"))
    shard_rows = [
        json.loads(line) for line in shard_manifest_path.read_text(encoding="utf-8").splitlines()
    ]
    polygon_shard = next(row for row in shard_rows if row["table"] == "polygon_snapshots")
    polygon_table = pq.read_table(sink_root / polygon_shard["path"])
    geometry_uri = polygon_table["geometry_uri"][0].as_py()
    assert geometry_uri.startswith(exported.attempt_checkpoint_uris[0][1] + "#row=")
    projected_wkb = read_projected_geometry_wkb(geometry_uri)
    assert hashlib.sha256(projected_wkb).hexdigest() == polygon_table["geometry_sha256"][0].as_py()
    attempt_shard = next(row for row in shard_rows if row["table"] == "polygon_attempts")
    attempt_table = pq.read_table(sink_root / attempt_shard["path"])
    assert attempt_table.num_rows == raw_scan.evidence.expected_key_count
    attempt_metadata = FilesystemArtifactReader().parquet_metadata(
        (sink_root / attempt_shard["path"]).as_uri()
    )
    assert attempt_metadata.row_count == raw_scan.evidence.expected_key_count
    assert attempt_metadata.schema == attempt_table.schema
    support_shard = next(row for row in shard_rows if row["table"] == "label_support_evidence")
    support_table = pq.read_table(sink_root / support_shard["path"])
    support_metadata = FilesystemArtifactReader().parquet_metadata(
        (sink_root / support_shard["path"]).as_uri()
    )
    assert support_metadata.row_count == sum(
        count for _level, count in attempts.shards[0].candidate_counts
    )
    assert support_metadata.schema == support_table.schema


@pytest.mark.parametrize(
    ("missing_pin", "expected"),
    [
        ("source_snapshot.source_artifact_manifest", "source_snapshot.source_artifact_manifest"),
        ("reference_pin.reference_artifact_manifest", "reference_pin.reference_artifact_manifest"),
        ("reference_pin.area_measurement", "reference_pin.area_measurement"),
    ],
)
def test_contract_v12_export_fails_before_writes_without_required_provenance(
    tmp_path: Path,
    missing_pin: str,
    expected: str,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    if missing_pin == "source_snapshot.source_artifact_manifest":
        raw_source = attempts.raw_scan.source
        raw_source = replace(
            raw_source,
            source_snapshot=replace(raw_source.source_snapshot, source_artifact_manifest=None),
        )
        attempts = replace(attempts, raw_scan=replace(attempts.raw_scan, source=raw_source))
    elif missing_pin == "reference_pin.reference_artifact_manifest":
        attempts = replace(
            attempts,
            reference_pin=replace(attempts.reference_pin, reference_artifact_manifest=None),
        )
    else:
        attempts = replace(
            attempts,
            reference_pin=replace(attempts.reference_pin, area_measurement=None),
        )
    sink_root = tmp_path / "missing-v12-provenance-pool"

    with pytest.raises(ValueError) as error:
        export_eunis_pool(attempts, FilesystemStreamingSink(sink_root))

    assert expected in str(error.value)
    assert not sink_root.exists()


@pytest.mark.parametrize(
    ("target", "field", "value", "expected"),
    [
        ("source", "format_version", "artifact-list-v2", "unsupported format"),
        ("reference", "manifest_uri", " ", "needs a manifest URI"),
        ("source", "expected_artifact_count", 0, "at least one artifact"),
        ("reference", "manifest_sha256", "bad-digest", "must be lowercase hexadecimal"),
        ("area", "method_id", "wrong-method", "area measurement pin does not match"),
    ],
)
def test_contract_v12_export_rejects_incomplete_provenance_before_writes(
    tmp_path: Path,
    target: str,
    field: str,
    value: object,
    expected: str,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    if target == "source":
        artifact_pin = attempts.raw_scan.source.source_snapshot.source_artifact_manifest
        assert artifact_pin is not None
        artifact_pin = replace(artifact_pin, **{field: value})
        source_snapshot = replace(
            attempts.raw_scan.source.source_snapshot,
            source_artifact_manifest=artifact_pin,
        )
        source = replace(attempts.raw_scan.source, source_snapshot=source_snapshot)
        attempts = replace(attempts, raw_scan=replace(attempts.raw_scan, source=source))
    elif target == "reference":
        artifact_pin = attempts.reference_pin.reference_artifact_manifest
        assert artifact_pin is not None
        artifact_pin = replace(artifact_pin, **{field: value})
        attempts = replace(
            attempts,
            reference_pin=replace(attempts.reference_pin, reference_artifact_manifest=artifact_pin),
        )
    else:
        area_pin = attempts.reference_pin.area_measurement
        assert area_pin is not None
        numerator = replace(area_pin.numerator, **{field: value})
        attempts = replace(
            attempts,
            reference_pin=replace(
                attempts.reference_pin,
                area_measurement=replace(area_pin, numerator=numerator),
            ),
        )
    sink_root = tmp_path / "invalid-v12-provenance-pool"

    with pytest.raises(ValueError) as error:
        export_eunis_pool(attempts, FilesystemStreamingSink(sink_root))

    assert expected in str(error.value)
    assert not sink_root.exists()


def test_full_pool_export_streams_downstream_text_and_relevance_rows(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    unbounded_sink = tmp_path / "unbounded-full-export"
    with pytest.raises(TypeError, match="bounds-checked pinned join stream"):
        export_eunis_full_pool(
            attempts,
            cast(BoundedJoinedTextRecords, iter(())),
            FilesystemStreamingSink(unbounded_sink),
        )
    assert not unbounded_sink.exists()
    inventory = list(iter_eunis_attempt_inventory_rows(attempts))
    text = "A useful visitor description."
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    limits = StreamingLimits(max_batch_rows=16)
    downstream = iter_eunis_join_records(
        TextSource.WEBSITE,
        [
            PinnedSourceRow(
                "polygons/region-a.parquet",
                0,
                {
                    "polygon_id": "region:way:11",
                    "source_pbf": "region.osm.pbf",
                    "osm_type": "way",
                    "osm_id": 11,
                    "website_sentences": [text],
                    "contact_website_sentences": [],
                    "website_sentence_status": "success",
                    "contact_website_sentence_status": "absent",
                    "website_text": text,
                    "contact_website_text": None,
                    "website_text_status": "success",
                    "contact_website_text_status": "absent",
                },
            )
        ],
        [
            PinnedSourceRow(
                "labels/polygons/region-a.parquet",
                0,
                {
                    "label_id": "fixture-website-yes",
                    "polygon_id": "region:way:11",
                    "field": "website",
                    "sentence_index": 0,
                    "text_sha256": digest,
                    "decision": "YES",
                },
            )
        ],
        inventory,
        website_field_to_column={
            "website": "website_sentences",
            "contact_website": "contact_website_sentences",
        },
        allocation_bounds=JoinAllocationBounds.from_streaming_limits(limits),
    )
    sink_root = tmp_path / "full-pool-with-text"

    exported = export_eunis_full_pool(
        attempts,
        downstream,
        FilesystemStreamingSink(sink_root),
        limits=limits,
        staging_directory=tmp_path,
    )

    assert exported.downstream_join_included
    manifest_path = Path(exported.receipt.shard_manifest_uri.removeprefix("file://"))
    shards = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    counts = {row["table"]: row["row_count"] for row in shards}
    assert counts["source_documents"] == 2
    assert counts["document_relevance"] == 2
    assert counts["text_polygon_links"] == 2
    document_shard = next(row for row in shards if row["table"] == "source_documents")
    documents = pq.read_table(sink_root / document_shard["path"])
    assert documents["content_sha256"][0].as_py() == digest
    relevance_shard = next(row for row in shards if row["table"] == "document_relevance")
    relevance = pq.read_table(sink_root / relevance_shard["path"])
    assert relevance["status"][0].as_py() == RelevanceStatus.YES.value
    link_shard = next(row for row in shards if row["table"] == "text_polygon_links")
    links = pq.read_table(sink_root / link_shard["path"])
    assert links["status"][0].as_py() == TextLinkStatus.LINKED.value


def test_lazy_downstream_failure_leaves_attempt_evidence_without_completion_manifest(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    text = "A useful visitor description."
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    limits = StreamingLimits(max_batch_rows=16, max_row_bytes=1024)
    downstream = iter_eunis_join_records(
        TextSource.WEBSITE,
        [
            PinnedSourceRow(
                "polygons/region-a.parquet",
                0,
                {
                    "polygon_id": "region:way:11",
                    "source_pbf": "region.osm.pbf",
                    "osm_type": "way",
                    "osm_id": 11,
                    "website_sentences": [text],
                    "contact_website_sentences": [],
                    "website_sentence_status": "success",
                    "contact_website_sentence_status": "absent",
                    "website_text": text,
                    "contact_website_text": None,
                    "website_text_status": "success",
                    "contact_website_text_status": "absent",
                },
            )
        ],
        [
            PinnedSourceRow(
                "labels/polygons/region-a.parquet",
                0,
                {
                    "label_id": f"oversized-{'x' * 2048}",
                    "polygon_id": "region:way:11",
                    "field": "website",
                    "sentence_index": 0,
                    "text_sha256": digest,
                    "decision": "YES",
                },
            )
        ],
        list(iter_eunis_attempt_inventory_rows(attempts)),
        website_field_to_column={
            "website": "website_sentences",
            "contact_website": "contact_website_sentences",
        },
        allocation_bounds=JoinAllocationBounds.from_streaming_limits(limits),
    )
    sink_root = tmp_path / "failed-full-export"

    with pytest.raises(ValueError, match="exceeds the maximum output row size"):
        export_eunis_full_pool(
            attempts,
            downstream,
            FilesystemStreamingSink(sink_root),
            limits=limits,
            staging_directory=tmp_path,
        )

    assert list(sink_root.rglob("raw-inventory-scan.json"))
    assert list(sink_root.rglob("label-attempts-report.json"))
    assert list(sink_root.rglob("*.jsonl"))
    assert not list(sink_root.rglob("manifest.json"))


def test_wiki_downstream_wrapper_streams_both_corpora(monkeypatch) -> None:
    calls = []

    def fake_join(_attempts, text_table, relevance_table, **kwargs):
        calls.append((text_table, relevance_table, kwargs))
        return iter_eunis_join_records(
            TextSource.WIKI,
            [],
            [],
            [],
            wiki_corpus=kwargs["wiki_corpus"],
        )

    monkeypatch.setattr(pool_runner, "iter_eunis_downstream_records", fake_join)
    attempts = cast(Any, None)
    wikipedia_text, wikipedia_labels = cast(Any, "wiki-text"), cast(Any, "wiki-labels")
    wikivoyage_text, wikivoyage_labels = cast(Any, "voyage-text"), cast(Any, "voyage-labels")
    links = cast(Any, "links")

    combined = iter_eunis_wiki_downstream_records(
        attempts,
        wikipedia_text,
        wikipedia_labels,
        wikivoyage_text,
        wikivoyage_labels,
        links,
        batch_size=17,
    )

    assert list(combined) == []
    corpora = [call[2]["wiki_corpus"] for call in calls]
    assert corpora == ["wikipedia", "wikivoyage"]
    assert [call[0] for call in calls] == [wikipedia_text, wikivoyage_text]
    assert all(call[2]["wiki_link_table"] is links for call in calls)
    assert all(call[2]["batch_size"] == 17 for call in calls)


def test_attempt_inventory_iteration_checks_shard_and_global_counts(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    checkpoint_root = tmp_path / "checkpoints"
    raw_scan = scan_raw_inventory(source, checkpoint_root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw_scan,
        cast(RasterReference, FakeReference()),
        _reference_pin(tmp_path),
        _TAXONOMY,
        checkpoint_root,
    )
    shard = attempts.shards[0]

    [checkpoint_row] = list(iter_attempt_checkpoint(shard.checkpoint_uri))
    assert checkpoint_row["inventory"]["inventory_row_id"]

    missing_checkpoint = replace(shard, checkpoint_uri=(tmp_path / "absent.jsonl").as_uri())
    with pytest.raises(FileNotFoundError, match="not available"):
        list(iter_eunis_attempt_inventory_rows(replace(attempts, shards=(missing_checkpoint,))))
    changed_checksum = replace(shard, checkpoint_sha256="0" * 64)
    with pytest.raises(ValueError, match="checksum differs"):
        list(iter_eunis_attempt_inventory_rows(replace(attempts, shards=(changed_checksum,))))

    with pytest.raises(ValueError, match="attempt inventory changed"):
        list(
            iter_eunis_attempt_inventory_rows(
                replace(attempts, shards=(replace(shard, row_count=2),))
            )
        )
    changed_evidence = replace(attempts.raw_scan.evidence, expected_key_count=2)
    changed_scan = replace(attempts.raw_scan, evidence=changed_evidence)
    with pytest.raises(ValueError, match="raw source key count"):
        list(iter_eunis_attempt_inventory_rows(replace(attempts, raw_scan=changed_scan)))


def test_local_checkpoint_readers_reject_remote_and_malformed_fragments(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="local file URIs only"):
        list(iter_attempt_checkpoint("https://example.org/checkpoint.jsonl"))
    checkpoint = tmp_path / "fragment.jsonl"
    checkpoint.write_text('{"other":"value"}\n', encoding="utf-8")
    with pytest.raises(TypeError, match="no projected WKB geometry"):
        read_projected_geometry_wkb(checkpoint.as_uri() + "#row=0/projected_geometry_wkb")


def test_invalid_attempt_signature_is_recomputed_from_cached_raw_inventory(
    tmp_path: Path,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    root = tmp_path / "checkpoints"
    raw = scan_raw_inventory(source, root, _code_pin())
    first_reference = FakeReference()
    first = run_eunis_label_attempts(
        raw,
        cast(RasterReference, first_reference),
        _reference_pin(tmp_path),
        _TAXONOMY,
        root,
    )
    next_reference = FakeReference()
    second = run_eunis_label_attempts(
        raw,
        cast(RasterReference, next_reference),
        _reference_pin(tmp_path),
        _TAXONOMY,
        root,
    )

    assert second.signature == first.signature
    assert next_reference.query_count == 0


def test_source_manifest_does_not_allow_an_unpinned_local_shard(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    shard = source.polygon_shards[0]
    changed = shard.local_path.with_name("changed.parquet")
    changed.write_bytes(shard.local_path.read_bytes() + b"tamper")
    changed_source = FrozenPolygonSource(
        layout=source.layout,
        source_snapshot=source.source_snapshot,
        manifest_uri=source.manifest_uri,
        manifest_sha256=source.manifest_sha256,
        manifest_paths=source.manifest_paths,
        polygon_shards=(
            PinnedPolygonShard(
                path_in_repo=shard.path_in_repo,
                revision=shard.revision,
                sha256=shard.sha256,
                size_bytes=shard.size_bytes,
                blob_id=shard.blob_id,
                lfs_oid=shard.lfs_oid,
                lfs_size=shard.lfs_size,
                local_path=changed,
            ),
        ),
    )

    with pytest.raises(ValueError, match="size differs from its manifest"):
        scan_raw_inventory(changed_source, tmp_path / "new-checkpoints", _code_pin())


def test_raw_shard_content_must_match_its_git_tree_blob_id(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    shard = source.polygon_shards[0]
    content = shard.local_path.read_bytes()
    changed = shard.local_path.with_name("same-size-changed.parquet")
    changed.write_bytes(bytes([content[0] ^ 1]) + content[1:])
    changed_shard = replace(shard, sha256=hashlib.sha256(changed.read_bytes()).hexdigest())
    changed_shard = replace(changed_shard, local_path=changed)
    changed_source = replace(source, polygon_shards=(changed_shard,))

    with pytest.raises(ValueError, match="Git blob id differs"):
        scan_raw_inventory(changed_source, tmp_path / "new-checkpoints", _code_pin())


def test_raw_lfs_shard_must_match_its_tree_oid_and_size(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    shard = source.polygon_shards[0]
    lfs_shard = replace(shard, lfs_oid=shard.sha256, lfs_size=shard.size_bytes)

    pool_runner._verify_local_shard(lfs_shard)
    with pytest.raises(ValueError, match="raw LFS shard differs from its tree pin"):
        pool_runner._verify_local_shard(replace(lfs_shard, lfs_oid="0" * 64))
    with pytest.raises(ValueError, match="raw LFS shard differs from its tree pin"):
        pool_runner._verify_local_shard(replace(lfs_shard, lfs_size=shard.size_bytes + 1))


def test_raw_source_revision_must_match_repository_and_frozen_layout(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, None)])
    bad_repository = SimpleNamespace(
        source_snapshot=replace(source.source_snapshot, repository="another/repository"),
        layout=source.layout,
    )
    with pytest.raises(ValueError, match="repository does not match"):
        pool_runner._validate_source_repository_revision(cast(FrozenPolygonSource, bad_repository))

    bad_sha = SimpleNamespace(
        source_snapshot=replace(source.source_snapshot, revision="not-a-revision"),
        layout=source.layout,
    )
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        pool_runner._validate_source_repository_revision(cast(FrozenPolygonSource, bad_sha))

    bad_pin = SimpleNamespace(
        source_snapshot=replace(source.source_snapshot, revision="c" * 40),
        layout=source.layout,
    )
    with pytest.raises(ValueError, match="does not match the pinned"):
        pool_runner._validate_source_repository_revision(cast(FrozenPolygonSource, bad_pin))


@pytest.mark.parametrize(
    ("raw_geometry", "reason"),
    [
        (None, "missing_geometry"),
        ("not a valid geometry payload", "invalid_or_empty_polygon_geometry"),
        ({"type": "Point", "coordinates": [0, 0]}, "invalid_or_empty_polygon_geometry"),
        ("{}", "geometry_transform_error"),
    ],
)
def test_raw_geometry_projection_preserves_each_invalid_reason(
    raw_geometry: Any,
    reason: str,
) -> None:
    projected, exclusion = pool_runner._project_raw_geometry(raw_geometry)

    assert projected is None
    assert exclusion is not None and exclusion.startswith(reason)


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (FakeReference(source_version="another release"), "does not match the pinned"),
    ],
)
def test_map_inputs_must_match_release_pin(
    tmp_path: Path,
    reference: FakeReference,
    error: str,
) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    root = tmp_path / "checkpoints"
    raw = scan_raw_inventory(source, root, _code_pin())

    with pytest.raises(ValueError, match=error):
        run_eunis_label_attempts(
            raw,
            cast(RasterReference, reference),
            _reference_pin(tmp_path),
            _TAXONOMY,
            root,
        )
