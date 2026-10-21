from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from shapely.geometry import box, mapping

from osm_polygon_eunis.pool_adapter import (
    EUNIS_POOL_SOURCE_LAYOUTS,
    EunisTaxonomy,
    EunisTaxonomyPath,
    SourceTreeFile,
    SourceTreeGroupPin,
    source_tree_inventory_sha256,
    validate_source_tree_group,
)
from osm_polygon_eunis.pool_contract import (
    FilesystemArtifactReader,
    FilesystemStreamingSink,
    ReferencePin,
    SourceSnapshot,
    StreamingLimits,
    SupportSemantics,
    TextSource,
    native_keyset_sha256,
)
from osm_polygon_eunis.pool_contract.builder import canonical_native_key
from osm_polygon_eunis.pool_runner import (
    DuplicateNativeKeyError,
    FrozenPolygonSource,
    PinnedPolygonShard,
    PoolAdapterCodePin,
    export_eunis_pool,
    iter_attempt_checkpoint,
    read_projected_geometry_wkb,
    read_raw_geometry,
    run_eunis_label_attempts,
    scan_raw_inventory,
)
from osm_polygon_eunis.raster_reference import RasterReference

_TAXONOMY = EunisTaxonomy(
    taxonomy_id="fixture-taxonomy",
    taxonomy_revision="fixture-taxonomy-v1",
    taxonomy_sha256="c" * 64,
    paths_by_direct_code={"C1": EunisTaxonomyPath("A", "A1", "C1")},
)


@dataclass
class FakeReference:
    """Small deterministic replacement for the expensive production raster set."""

    fail: bool = False
    query_count: int = 0
    presence_threshold: int = 0
    source_version: str = "EEA test map release"

    @property
    def layer_codes(self) -> tuple[str, ...]:
        return ("C1",)

    def support_areas(
        self,
        polygon: Any,
        _classes: dict[tuple[str, str], tuple[str, ...]],
    ) -> dict[tuple[str, str], float]:
        self.query_count += 1
        if self.fail:
            raise RuntimeError("fixture query error")
        support = float(polygon.area) * 0.8
        return {("L1", "A"): support, ("L2", "A1"): support, ("L3", "C1"): support}


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
    return FrozenPolygonSource(
        layout=_fixture_layout(shard),
        source_snapshot=_source_snapshot(),
        manifest_uri="https://source.example/raw-source-manifest.json",
        manifest_sha256="b" * 64,
        manifest_paths=(shard.path_in_repo, "dataset-card.md"),
        polygon_shards=(shard,),
    )


def _reference_pin() -> ReferencePin:
    return ReferencePin(
        release_id="EEA test map release",
        source_repository="https://reference.example/eunis",
        source_revision="reference-manifest-sha256:" + "d" * 64,
        taxonomy_id=_TAXONOMY.taxonomy_id,
        taxonomy_revision=_TAXONOMY.pinned_revision,
        support_semantics=SupportSemantics.MODELED_HABITAT_SUPPORT,
        support_unit="m2",
        threshold=0.8,
        support_union_policy="union_by_class",
    )


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
        "eef1267cca7cbf455f09d4df9d41cff592cf841b313ea0d9c71262d24ad5f107"
    )
    assert manifest["schema"]["raw_native_key_counts"] is None
    assert "/srv/storage/" not in manifest_path.read_text(encoding="utf-8")

    groups = {
        (tree["repository"], tree["revision"], group["path_prefix"]): group
        for section in ("raw_source_file_trees", "relevance_label_file_trees")
        for tree in manifest[section]
        for group in tree["groups"]
    }
    pins = [
        pin
        for layout in EUNIS_POOL_SOURCE_LAYOUTS.values()
        for pin in (layout.raw_polygon_tree_pin, *layout.additional_tree_pins)
    ]
    assert len(groups) == len(pins) == 10
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
    assert reference["taxonomy"]["workbook_direct_url"] is None
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
        _reference_pin(),
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
    assert usable["attempt_receipt"]["attempted_reference_ids"] == ["C1"]
    assert len(usable["support_evidence"]) == len(usable["candidates"]) == 3
    assert {tuple(item["reference_ids"]) for item in usable["support_evidence"]} == {("C1",)}
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


def test_query_error_is_excluded_with_reason_at_every_supported_level(tmp_path: Path) -> None:
    source = _frozen_source(tmp_path, [_source_row(11, _geojson_polygon())])
    root = tmp_path / "checkpoints"
    raw = scan_raw_inventory(source, root, _code_pin())
    attempts = run_eunis_label_attempts(
        raw,
        cast(RasterReference, FakeReference(fail=True)),
        _reference_pin(),
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
        _reference_pin(),
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
        _reference_pin(),
        _TAXONOMY,
        root,
    )
    next_reference = FakeReference()
    second = run_eunis_label_attempts(
        raw,
        cast(RasterReference, next_reference),
        _reference_pin(),
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


@pytest.mark.parametrize(
    ("reference", "error"),
    [
        (FakeReference(presence_threshold=1), "greater than zero"),
        (FakeReference(source_version="another release"), "does not match the pinned"),
    ],
)
def test_map_inputs_must_match_threshold_and_release_pin(
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
            _reference_pin(),
            _TAXONOMY,
            root,
        )
