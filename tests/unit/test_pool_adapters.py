from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import shapely
from shapely.geometry import LineString, box

from osm_polygon_eunis.pool_adapters import (
    WORLDCOVER_AREA_MEASUREMENT,
    WORLDCOVER_LEVELS,
    AdapterStreamingOptions,
    LucasCode,
    LucasFeatureQuery,
    LucasFootprint,
    LucasGeoPackageReader,
    LucasLayerManifest,
    LucasReferenceManifest,
    LucasTaxonomy,
    PoolPolygonInput,
    SupportArea,
    SupportAttempt,
    WorldCoverEvidence,
    build_raw_inventory_evidence,
    export_labeled_polygons,
    iter_labeled_polygon_batches,
    iter_lucas_attempts,
    iter_lucas_rows,
    iter_polygon_batches,
    iter_worldcover_attempts,
    iter_worldcover_rows,
    iter_worldcover_source_attempts,
    label_polygon,
    lucas_2022_copernicus_manifest,
    lucas_2022_reference_pin,
    measure_lucas_supports,
    polygon_geometry_tree_pins,
    raw_source_tree_pins,
    verify_lucas_file,
    verify_lucas_taxonomy_file,
    worldcover_supports,
)
from osm_polygon_eunis.pool_adapters import lucas as lucas_module
from osm_polygon_eunis.pool_adapters import source_pins as source_pins_module
from osm_polygon_eunis.pool_contract import (
    AssignmentKind,
    AttemptStatus,
    BodyStatus,
    FilesystemArtifactReader,
    FilesystemStreamingSink,
    GeometryStatus,
    LabelLevel,
    OutcomeStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBatch,
    PoolBuildInput,
    PoolKey,
    ReferencePin,
    ReferenceSystem,
    SourceDocument,
    SourceSnapshot,
    StreamingLimits,
    SupportSemantics,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
    canonical_native_key,
    inventory_row_identifier,
    native_keyset_sha256,
    polygon_snapshot_identifier,
    validate_nine_streamed_pool_gate,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


@dataclass(frozen=True)
class SamplePolygon:
    pool_polygon: PoolPolygonInput
    snapshot: PolygonSnapshot


def _polygon(index: int, *, area_unit: str = "m^2", crs: str = "EPSG:4326") -> SamplePolygon:
    source = SourceSnapshot(
        snapshot_id="raw-description",
        repository="NoeFlandre/source-polygons",
        revision="source-commit-1",
        pbf_uri="https://artifacts.example/source.osm.pbf",
        pbf_sha256=_SHA_A,
    )
    native_key = canonical_native_key(
        {"source_pbf": "source.osm.pbf", "osm_type": "way", "osm_id": index}
    )
    geometry = box(index * 20, 0, index * 20 + 10, 10)
    geometry_hash = hashlib.sha256(shapely.to_wkb(geometry)).hexdigest()
    inventory_id = inventory_row_identifier(source.snapshot_id, native_key)
    snapshot_id = polygon_snapshot_identifier(source.snapshot_id, native_key, geometry_hash)
    inventory = PolygonInventoryRow(
        inventory_row_id=inventory_id,
        native_key=native_key,
        source_row_locator=f"polygons/region.parquet#row={index}",
        polygon_snapshot_id=snapshot_id,
    )
    snapshot = PolygonSnapshot(
        polygon_snapshot_id=snapshot_id,
        inventory_row_id=inventory_id,
        native_key=native_key,
        geometry_uri=f"https://artifacts.example/geometries/{index}.wkb",
        geometry_sha256=geometry_hash,
        geometry_format="WKB",
        crs=crs,
        coordinate_transform="identity",
        geometry_status=GeometryStatus.USABLE,
        original_area=100.0,
        area_unit=area_unit,
    )
    return SamplePolygon(PoolPolygonInput(f"way/{index}", inventory, snapshot, geometry), snapshot)


def _reference_pin(reference: ReferenceSystem) -> ReferencePin:
    unit = "m^2"
    semantics = (
        SupportSemantics.MAPPED_COVER
        if reference is ReferenceSystem.WORLDCOVER
        else SupportSemantics.SURVEYED_FOOTPRINT
    )
    return ReferencePin(
        release_id=f"{reference.value}-release-1",
        source_repository="NoeFlandre/reference-data",
        source_revision="reference-commit-1",
        taxonomy_id=f"{reference.value}-taxonomy",
        taxonomy_revision="taxonomy-commit-1",
        support_semantics=semantics,
        support_unit=unit,
        threshold=0.8,
        support_union_policy="union_by_class",
    )


def _lucas_taxonomy(*, sha256: str = _SHA_B) -> LucasTaxonomy:
    return LucasTaxonomy(
        taxonomy_id="fixture-lucas",
        revision="fixture-codebook-1",
        sha256=sha256,
        codes=(
            LucasCode("A", "A1", "A11"),
            LucasCode("A", "A2", "A21"),
        ),
    )


def test_lucas_2022_manifest_and_c3_taxonomy_are_fully_pinned() -> None:
    manifest = lucas_2022_copernicus_manifest()
    taxonomy = LucasTaxonomy.lucas_2022_c3()
    reference_pin = lucas_2022_reference_pin(manifest)

    assert manifest.file_size_bytes == 1_121_144_832
    assert len(manifest.file_sha256) == 64
    assert manifest.layer.table == "l2022_cop_radpoly_radqs_allatr_wcol"
    assert manifest.layer.srs_id == 3035
    assert len(manifest.taxonomy_sha256) == 64
    assert taxonomy.taxonomy_id == manifest.taxonomy_id
    assert taxonomy.revision == manifest.taxonomy_revision
    assert taxonomy.sha256 == manifest.taxonomy_sha256
    assert reference_pin.source_revision == manifest.source_revision
    assert reference_pin.support_unit == "m^2"
    assert len({row.l1_code for row in taxonomy.codes}) == 8
    assert len({row.l2_code for row in taxonomy.codes}) == 29
    assert len({row.l3_code for row in taxonomy.codes}) == 74
    assert taxonomy.path_for_observed_code("C10") == LucasCode("C00", "C10", "C10")
    assert taxonomy.path_for_observed_code("C1") is None


def test_raw_inventory_evidence_rejects_incomplete_scan_pins() -> None:
    with pytest.raises(ValueError, match="enumeration rule"):
        build_raw_inventory_evidence(
            enumeration_rule=" ",
            scan_report_uri="file:///tmp/scan.json",
            scan_report=b"scan report",
            expected_key_count=1,
            expected_keyset_sha256=_SHA_A,
        )


def test_label_polygon_rejects_inconsistent_pins_and_support_rows() -> None:
    sample = _polygon(19)
    polygon = sample.pool_polygon
    reference_pin = _reference_pin(ReferenceSystem.WORLDCOVER)
    with pytest.raises(ValueError, match=r"0\.8 threshold"):
        label_polygon(
            polygon,
            SupportAttempt(()),
            WORLDCOVER_LEVELS,
            replace(reference_pin, threshold=0.9),
            reference_id_namespace="fixture-reference",
        )
    with pytest.raises(ValueError, match="assignment-rule version"):
        label_polygon(
            polygon,
            SupportAttempt(()),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
            assignment_rule_version=" ",
        )
    with pytest.raises(ValueError, match="does not match its raw inventory"):
        label_polygon(
            replace(
                polygon,
                inventory=replace(polygon.inventory, polygon_snapshot_id="other-snapshot"),
            ),
            SupportAttempt(()),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    with pytest.raises(ValueError, match="at least one supported level"):
        label_polygon(
            polygon,
            SupportAttempt(()),
            (LabelLevel("class", supported=False),),
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    with pytest.raises(ValueError, match="names must be unique"):
        label_polygon(
            polygon,
            SupportAttempt(()),
            (LabelLevel("class", True), LabelLevel("class", True)),
            reference_pin,
            reference_id_namespace="fixture-reference",
        )

    unsupported = SupportAttempt((SupportArea("unsupported", "10", 1.0, AssignmentKind.DIRECT),))
    with pytest.raises(ValueError, match="undeclared or unsupported level"):
        label_polygon(
            polygon,
            unsupported,
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    blank_code = SupportAttempt((SupportArea("class", " ", 1.0, AssignmentKind.DIRECT),))
    with pytest.raises(ValueError, match="code must not be empty"):
        label_polygon(
            polygon,
            blank_code,
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    duplicate = SupportArea("class", "10", 20.0, AssignmentKind.DIRECT)
    with pytest.raises(ValueError, match="must be unioned before labeling"):
        label_polygon(
            polygon,
            SupportAttempt((duplicate, duplicate)),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    with pytest.raises(ValueError, match="finite and non-negative"):
        label_polygon(
            polygon,
            SupportAttempt((SupportArea("class", "10", float("nan"), AssignmentKind.DIRECT),)),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )
    with pytest.raises(ValueError, match="exceeds full polygon area"):
        label_polygon(
            polygon,
            SupportAttempt((SupportArea("class", "10", 101.0, AssignmentKind.DIRECT),)),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )


def test_label_polygon_preserves_exclusion_and_partial_reasons() -> None:
    sample = _polygon(20)
    polygon = sample.pool_polygon
    reference_pin = _reference_pin(ReferenceSystem.WORLDCOVER)
    with pytest.raises(ValueError, match="area units do not match"):
        label_polygon(
            replace(polygon, snapshot=replace(sample.snapshot, area_unit="hectares")),
            SupportAttempt(()),
            WORLDCOVER_LEVELS,
            reference_pin,
            reference_id_namespace="fixture-reference",
        )

    invalid_status = replace(sample.snapshot, geometry_status=GeometryStatus.INVALID)
    excluded = label_polygon(
        replace(polygon, snapshot=invalid_status),
        SupportAttempt(()),
        WORLDCOVER_LEVELS,
        reference_pin,
        reference_id_namespace="fixture-reference",
    )
    assert excluded.attempt_evidence.status == "excluded"
    assert excluded.label_outcomes[0].reason == "invalid"

    invalid_area = replace(sample.snapshot, original_area=0.0)
    excluded_area = label_polygon(
        replace(polygon, snapshot=invalid_area),
        SupportAttempt(()),
        WORLDCOVER_LEVELS,
        reference_pin,
        reference_id_namespace="fixture-reference",
    )
    assert excluded_area.label_outcomes[0].reason == "geometry_area_unusable"

    unusable_geometry = label_polygon(
        replace(polygon, geometry=None),
        SupportAttempt(()),
        WORLDCOVER_LEVELS,
        reference_pin,
        reference_id_namespace="fixture-reference",
    )
    assert unusable_geometry.label_outcomes[0].reason == "geometry_unusable"

    partial = label_polygon(
        polygon,
        SupportAttempt((), partial_reason="reference scan incomplete"),
        WORLDCOVER_LEVELS,
        reference_pin,
        reference_id_namespace="fixture-reference",
    )
    assert partial.attempt_evidence.status == "partial"
    assert partial.attempt_evidence.reason == "reference scan incomplete"
    with pytest.raises(ValueError, match="scan-report URI"):
        build_raw_inventory_evidence(
            enumeration_rule="raw polygons before text joins",
            scan_report_uri="",
            scan_report=b"scan report",
            expected_key_count=1,
            expected_keyset_sha256=_SHA_A,
        )
    with pytest.raises(ValueError, match="scan report"):
        build_raw_inventory_evidence(
            enumeration_rule="raw polygons before text joins",
            scan_report_uri="file:///tmp/scan.json",
            scan_report=b"",
            expected_key_count=1,
            expected_keyset_sha256=_SHA_A,
        )
    with pytest.raises(ValueError, match="cannot be negative"):
        build_raw_inventory_evidence(
            enumeration_rule="raw polygons before text joins",
            scan_report_uri="file:///tmp/scan.json",
            scan_report=b"scan report",
            expected_key_count=-1,
            expected_keyset_sha256=_SHA_A,
        )
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        build_raw_inventory_evidence(
            enumeration_rule="raw polygons before text joins",
            scan_report_uri="file:///tmp/scan.json",
            scan_report=b"scan report",
            expected_key_count=1,
            expected_keyset_sha256="A" * 64,
        )


def test_source_tree_pins_verify_frozen_raw_files_without_claiming_key_counts() -> None:
    pins = raw_source_tree_pins()
    assert sum(pin.file_count for pin in pins) == 2305
    assert {pin.text_source.value for pin in pins} == {"description", "website", "wiki"}
    assert all(len(pin.inventory_sha256) == 64 for pin in pins)
    assert all(len(pin.files) == pin.file_count for pin in pins)

    description = polygon_geometry_tree_pins(TextSource.DESCRIPTION)
    website = polygon_geometry_tree_pins(TextSource.WEBSITE)
    wiki = polygon_geometry_tree_pins(TextSource.WIKI)
    assert {pin.path_prefix for pin in description} == {"data/", "language-v1/data/"}
    assert {pin.path_prefix for pin in website} == {"polygons/"}
    assert {pin.path_prefix for pin in wiki} == {"polygons/"}
    assert all(pin.purpose == "polygon_geometry" for pin in (*description, *website, *wiki))
    assert all(
        pin.purpose == "downstream_text_documents"
        for pin in pins
        if pin.path_prefix.endswith("/sentences/")
    )


def test_source_tree_pin_validation_rejects_tampered_tree_metadata() -> None:
    pin = polygon_geometry_tree_pins(TextSource.WEBSITE)[0]
    raw_files = [
        {
            "path": row.path,
            "size": row.size_bytes,
            "blob_id": row.blob_id,
            "lfs_oid": row.lfs_oid,
            "lfs_size": row.lfs_size_bytes,
        }
        for row in pin.files
    ]
    with pytest.raises(ValueError, match="file count"):
        source_pins_module._validate_tree_pin(
            replace(pin, file_count=pin.file_count - 1), raw_files
        )
    with pytest.raises(ValueError, match="byte total"):
        source_pins_module._validate_tree_pin(
            replace(pin, total_size_bytes=pin.total_size_bytes - 1), raw_files
        )
    outside_prefix = (replace(pin.files[0], path="other/file.parquet"), *pin.files[1:])
    with pytest.raises(ValueError, match="out-of-prefix"):
        source_pins_module._validate_tree_pin(replace(pin, files=outside_prefix), raw_files)
    duplicate_path = (
        replace(pin.files[0], path=pin.files[1].path),
        *pin.files[1:],
    )
    with pytest.raises(ValueError, match="duplicate or out-of-prefix"):
        source_pins_module._validate_tree_pin(replace(pin, files=duplicate_path), raw_files)
    changed_files = list(raw_files)
    assert isinstance(changed_files[0]["size"], int)
    changed_files[0] = {
        **changed_files[0],
        "size": changed_files[0]["size"] + 1,
    }
    with pytest.raises(ValueError, match="inventory digest"):
        source_pins_module._validate_tree_pin(pin, changed_files)

    with pytest.raises(ValueError, match="unclassified pinned source tree"):
        source_pins_module._tree_purpose("unfrozen/")
    with pytest.raises(ValueError, match="raw file count"):
        source_pins_module._validate_source_tree_set(())
    with pytest.raises(ValueError, match="repository or revision"):
        source_pins_module._load_source_trees(
            [{"name": "website", "repo": "wrong", "resolved_sha": "wrong", "groups": []}]
        )


@pytest.mark.parametrize("bad_field", ("handoff_sha256", "source_key_counts"))
def test_source_tree_loader_fails_closed_on_unverified_handoff(monkeypatch, bad_field: str) -> None:
    handoff_path = (
        Path(__file__).parents[2]
        / "src/osm_polygon_eunis/pool_adapters/data/raw-source-tree-pins.json"
    )
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    if bad_field == "handoff_sha256":
        handoff[bad_field] = "0" * 64
    else:
        handoff[bad_field] = 1

    class FakeResource:
        def joinpath(self, _path: str) -> FakeResource:
            return self

        def read_text(self, encoding: str) -> str:
            assert encoding == "utf-8"
            return json.dumps(handoff)

    monkeypatch.setattr(source_pins_module, "files", lambda _package: FakeResource())
    source_pins_module.raw_source_tree_pins.cache_clear()
    try:
        with pytest.raises(ValueError, match=r"verified file SHA-256|unverified raw-key counts"):
            source_pins_module.raw_source_tree_pins()
    finally:
        source_pins_module.raw_source_tree_pins.cache_clear()


def test_worldcover_keeps_nodata_and_missing_tiles_in_full_area_denominator() -> None:
    polygon = _polygon(1)
    result = worldcover_supports(
        polygon.pool_polygon,
        WorldCoverEvidence(
            source_polygon_id="way/1",
            polygon_area_m2=100.0,
            class_areas_m2=((10, 80.0),),
            area_measurement=WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=("N00E000", "N00E003"),
            missing_tiles=("N00E003",),
            class_evidence_ids=((10, ("N00E000",)),),
        ),
        _reference_pin(ReferenceSystem.WORLDCOVER),
    )

    (candidate,) = result.label_candidates
    assert candidate.numerator == 80.0
    assert candidate.denominator == 100.0
    assert candidate.ratio == 0.8
    assert candidate.assignment_kind is AssignmentKind.DIRECT
    assert result.label_outcomes[0].status is OutcomeStatus.ASSIGNED
    assert result.attempt_evidence.attempted_tile_ids == ("N00E000", "N00E003")
    assert result.attempt_evidence.missing_tile_ids == ("N00E003",)
    assert result.attempt_evidence.support_evidence == (("class", "10", ("N00E000",)),)


def test_worldcover_rejects_mismatched_reference_and_geometry_pins() -> None:
    sample = _polygon(21)
    polygon = sample.pool_polygon
    pin = _reference_pin(ReferenceSystem.WORLDCOVER)
    good = WorldCoverEvidence(
        polygon.source_polygon_id,
        100.0,
        ((10, 80.0),),
        WORLDCOVER_AREA_MEASUREMENT,
        class_evidence_ids=((10, ("tile-1",)),),
    )
    with pytest.raises(ValueError, match="does not match the raw polygon"):
        worldcover_supports(polygon, replace(good, source_polygon_id="other"), pin)
    with pytest.raises(ValueError, match="physical square metres"):
        worldcover_supports(polygon, good, replace(pin, support_unit="hectares"))
    with pytest.raises(ValueError, match="not the pinned equal-area method"):
        worldcover_supports(polygon, replace(good, area_measurement="square degrees"), pin)
    with pytest.raises(ValueError, match="EPSG:4326"):
        worldcover_supports(
            replace(polygon, snapshot=replace(sample.snapshot, crs="EPSG:3857")),
            good,
            pin,
        )
    with pytest.raises(ValueError, match="physical square metres"):
        worldcover_supports(
            replace(polygon, snapshot=replace(sample.snapshot, area_unit="hectares")),
            good,
            pin,
        )


def test_worldcover_records_exclusion_and_area_error_outcomes() -> None:
    polygon = _polygon(22).pool_polygon
    pin = _reference_pin(ReferenceSystem.WORLDCOVER)
    for error, expected in (
        ("geometry_unusable", "geometry_unusable"),
        ("source_polygon_id_missing", "source_polygon_id_missing"),
        ("raster_read_error:OSError", "worldcover_extraction_failed:raster_read_error:OSError"),
    ):
        result = worldcover_supports(
            polygon,
            WorldCoverEvidence(
                polygon.source_polygon_id,
                None,
                (),
                WORLDCOVER_AREA_MEASUREMENT,
                extraction_error=error,
                attempted_tiles=("tile-attempted",),
            ),
            pin,
        )
        assert result.label_outcomes[0].reason == expected
        assert result.attempt_evidence.attempted_tile_ids == ("tile-attempted",)

    invalid_area = worldcover_supports(
        polygon,
        WorldCoverEvidence(
            polygon.source_polygon_id,
            0.0,
            (),
            WORLDCOVER_AREA_MEASUREMENT,
        ),
        pin,
    )
    assert invalid_area.label_outcomes[0].reason == "geometry_area_unusable"
    with pytest.raises(ValueError, match="differs from full equal-area"):
        worldcover_supports(
            polygon,
            WorldCoverEvidence(
                polygon.source_polygon_id,
                101.0,
                (),
                WORLDCOVER_AREA_MEASUREMENT,
            ),
            pin,
        )


def test_worldcover_validates_flat_class_areas_and_tile_evidence() -> None:
    polygon = _polygon(23).pool_polygon
    pin = _reference_pin(ReferenceSystem.WORLDCOVER)

    def evidence(
        class_areas: tuple[tuple[int, float], ...],
        class_evidence_ids: tuple[tuple[int, tuple[str, ...]], ...],
    ) -> WorldCoverEvidence:
        return WorldCoverEvidence(
            polygon.source_polygon_id,
            100.0,
            class_areas,
            WORLDCOVER_AREA_MEASUREMENT,
            class_evidence_ids=class_evidence_ids,
        )

    invalid_cases = (
        (evidence(((101, 10.0),), ((101, ("tile-1",)),)), "unknown class code"),
        (
            evidence(((10, 10.0), (10, 10.0)), ((10, ("tile-1",)),)),
            "must be unioned before adaptation",
        ),
        (evidence(((10, -1.0),), ((10, ("tile-1",)),)), "within polygon area"),
        (evidence(((10, float("nan")),), ((10, ("tile-1",)),)), "within polygon area"),
        (evidence(((10, 101.0),), ((10, ("tile-1",)),)), "within polygon area"),
        (evidence(((10, 20.0),), ()), "missing contributing tile IDs"),
        (
            evidence(
                ((10, 20.0),),
                ((10, ("tile-1",)), (20, ("tile-2",))),
            ),
            "no class area",
        ),
        (evidence(((10, 20.0),), ((10, ("tile-1", "tile-1")),)), "repeats a support tile ID"),
        (
            evidence(((10, 60.0), (20, 60.0)), ((10, ("tile-1",)), (20, ("tile-2",)))),
            "exceed the full equal-area denominator",
        ),
    )
    for invalid, message in invalid_cases:
        with pytest.raises(ValueError, match=message):
            worldcover_supports(polygon, invalid, pin)

    with pytest.raises(ValueError, match="repeats its support evidence"):
        worldcover_supports(
            polygon,
            evidence(((10, 20.0),), ((10, ("tile-1",)), (10, ("tile-2",)))),
            pin,
        )
    with pytest.raises(ValueError, match="no class area"):
        worldcover_supports(
            polygon,
            evidence((), ((10, ("tile-1",)),)),
            pin,
        )

    zero_support = worldcover_supports(
        polygon,
        evidence(((10, 0.0),), ()),
        pin,
    )
    assert zero_support.label_outcomes[0].status is OutcomeStatus.UNASSIGNED


def test_worldcover_attempts_every_raw_polygon_without_text_filters() -> None:
    polygons = (_polygon(1), _polygon(2), _polygon(3))
    areas = {"way/1": ((10, 80.0),), "way/2": ((10, 79.0),), "way/3": ()}
    attempted: list[str] = []

    def evidence_for(polygon: PoolPolygonInput) -> WorldCoverEvidence:
        attempted.append(polygon.source_polygon_id)
        return WorldCoverEvidence(
            polygon.source_polygon_id,
            100.0,
            areas[polygon.source_polygon_id],
            WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=(f"tile-{polygon.source_polygon_id}",),
            class_evidence_ids=(
                ((10, (f"tile-{polygon.source_polygon_id}",)),)
                if areas[polygon.source_polygon_id]
                else ()
            ),
        )

    results = tuple(
        iter_worldcover_attempts(
            (row.pool_polygon for row in polygons),
            evidence_for,
            _reference_pin(ReferenceSystem.WORLDCOVER),
        )
    )
    inventory = tuple(row.pool_polygon.inventory for row in polygons)
    scan = build_raw_inventory_evidence(
        enumeration_rule=(
            "enumerate every raw polygon row before text, relevance, or evaluation joins"
        ),
        scan_report_uri="https://artifacts.example/worldcover-scan.json",
        scan_report=b"synthetic raw inventory scan\n",
        expected_key_count=len(inventory),
        expected_keyset_sha256=native_keyset_sha256(row.native_key for row in inventory),
    )

    assert attempted == ["way/1", "way/2", "way/3"]
    assert [result.label_outcomes[0].status for result in results] == [
        OutcomeStatus.ASSIGNED,
        OutcomeStatus.UNASSIGNED,
        OutcomeStatus.UNASSIGNED,
    ]
    assert len({result.inventory_row_id for result in results}) == len(inventory)
    assert scan.expected_key_count == len(inventory)
    assert scan.expected_keyset_sha256 == native_keyset_sha256(row.native_key for row in inventory)


def test_worldcover_source_evidence_is_joined_one_to_one_in_raw_order() -> None:
    polygons = (_polygon(1), _polygon(2))
    evidence = (
        WorldCoverEvidence(
            "way/1",
            100.0,
            ((10, 80.0),),
            WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=("N00E000",),
            class_evidence_ids=((10, ("N00E000",)),),
        ),
        WorldCoverEvidence(
            "way/2",
            100.0,
            ((20, 80.0),),
            WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=("N00E003",),
            class_evidence_ids=((20, ("N00E003",)),),
        ),
    )

    results = tuple(
        iter_worldcover_source_attempts(
            (row.pool_polygon for row in polygons),
            evidence,
            _reference_pin(ReferenceSystem.WORLDCOVER),
        )
    )

    assert [row.label_outcomes[0].assigned_code for row in results] == ["10", "20"]
    with pytest.raises(ValueError, match="counts differ"):
        tuple(
            iter_worldcover_source_attempts(
                (row.pool_polygon for row in polygons),
                evidence[:1],
                _reference_pin(ReferenceSystem.WORLDCOVER),
            )
        )


def test_pool_adapter_batches_are_bounded_and_consecutive() -> None:
    polygons = tuple(_polygon(index) for index in (1, 2, 3))
    attempts = iter_worldcover_attempts(
        (row.pool_polygon for row in polygons),
        lambda row: WorldCoverEvidence(
            row.source_polygon_id,
            100.0,
            ((10, 80.0),),
            WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=(f"tile-{row.source_polygon_id}", f"missing-{row.source_polygon_id}"),
            missing_tiles=(f"missing-{row.source_polygon_id}",),
            class_evidence_ids=((10, (f"tile-{row.source_polygon_id}",)),),
        ),
        _reference_pin(ReferenceSystem.WORLDCOVER),
    )

    batches = tuple(
        iter_polygon_batches((row.pool_polygon for row in polygons), attempts, max_batch_rows=6)
    )

    assert [batch.batch_index for batch in batches] == [0, 1, 2]
    assert all(batch.row_count == 6 for batch in batches)
    assert all(batch.attempt_receipts[0].status is AttemptStatus.PARTIAL for batch in batches)
    assert all(
        batch.attempt_receipts[0].missing_reference_ids == (f"missing-way/{index}",)
        for batch, index in zip(batches, (1, 2, 3), strict=True)
    )
    assert all(
        batch.support_evidence[0].reference_ids == (f"tile-way/{index}",)
        for batch, index in zip(batches, (1, 2, 3), strict=True)
    )


def test_lucas_unions_same_class_before_full_area_threshold() -> None:
    polygon = _polygon(0, crs="EPSG:3035")
    taxonomy = _lucas_taxonomy()
    query = LucasFeatureQuery(
        footprints=(
            LucasFootprint("1", "A", "A1", "A11", box(0, 0, 6, 10)),
            LucasFootprint("2", "A", "A1", "A11", box(4, 0, 8, 10)),
            LucasFootprint("3", "A", "A2", "A21", box(9, 0, 10, 10)),
            LucasFootprint("4", "A", "A2", "A21", box(10, 0, 11, 10)),
        ),
        attempted_feature_ids=("1", "2", "3", "4"),
    )

    result = measure_lucas_supports(
        polygon.pool_polygon,
        query,
        taxonomy,
        _reference_pin(ReferenceSystem.LUCAS),
    )
    by_level_code = {
        (candidate.level, candidate.code): candidate for candidate in result.label_candidates
    }

    assert by_level_code[("L3", "A11")].numerator == 80.0
    assert by_level_code[("L3", "A11")].ratio == 0.8
    assert by_level_code[("L2", "A1")].numerator == 80.0
    assert by_level_code[("L1", "A")].numerator == 90.0
    assert by_level_code[("L1", "A")].assignment_kind is AssignmentKind.ANCESTOR_DERIVED
    assert result.label_outcomes[0].status is OutcomeStatus.ASSIGNED
    assert result.label_outcomes[1].status is OutcomeStatus.ASSIGNED
    assert result.label_outcomes[2].status is OutcomeStatus.ASSIGNED
    assert result.attempt_evidence.attempted_reference_ids == ("1", "2", "3", "4")
    assert ("L1", "A", ("1", "2", "3")) in result.attempt_evidence.support_evidence
    assert ("L3", "A11", ("1", "2")) in result.attempt_evidence.support_evidence
    assert ("L2", "A2", ("3",)) in result.attempt_evidence.support_evidence
    batch = next(iter_labeled_polygon_batches(((polygon.pool_polygon, result),)))
    assert len(batch.support_evidence) == len(result.label_candidates)
    assert all(row.reference_id_namespace == "lucas-footprint-id" for row in batch.support_evidence)
    assert all(row.reference_ids for row in batch.support_evidence)


def test_lucas_ambiguous_no_coverage_and_missing_geometry_are_explicit() -> None:
    polygon = _polygon(0, crs="EPSG:3035")
    taxonomy = _lucas_taxonomy()
    ambiguous = measure_lucas_supports(
        polygon.pool_polygon,
        LucasFeatureQuery(
            (
                LucasFootprint("x", "A", "A1", "A11", box(0, 0, 8, 10)),
                LucasFootprint("y", "A", "A2", "A21", box(2, 0, 10, 10)),
            ),
            ("x", "y"),
        ),
        taxonomy,
        _reference_pin(ReferenceSystem.LUCAS),
    )
    no_coverage = measure_lucas_supports(
        polygon.pool_polygon,
        LucasFeatureQuery((), ()),
        taxonomy,
        _reference_pin(ReferenceSystem.LUCAS),
    )
    missing_inventory = PolygonInventoryRow(
        inventory_row_id="missing-geometry-row",
        native_key=canonical_native_key({"missing": 1}),
        source_row_locator="synthetic/missing-geometry",
        polygon_snapshot_id=None,
        exclusion_reason="geometry_artifact_missing",
    )
    missing_geometry = PoolPolygonInput("way/missing", missing_inventory, None, None)
    excluded = measure_lucas_supports(
        missing_geometry,
        LucasFeatureQuery((), ()),
        taxonomy,
        _reference_pin(ReferenceSystem.LUCAS),
    )

    assert [row.status for row in ambiguous.label_outcomes] == [
        OutcomeStatus.ASSIGNED,
        OutcomeStatus.AMBIGUOUS,
        OutcomeStatus.AMBIGUOUS,
    ]
    assert all(row.status is OutcomeStatus.UNASSIGNED for row in no_coverage.label_outcomes)
    assert no_coverage.label_outcomes[0].reason == "no_reference_coverage"
    assert all(row.status is OutcomeStatus.EXCLUDED for row in excluded.label_outcomes)
    assert all(row.reason == "geometry_artifact_missing" for row in excluded.label_outcomes)


def _gpkg_blob(geometry) -> bytes:
    return b"GP\x00\x01" + struct.pack("<i", 3035) + shapely.to_wkb(geometry)


def _create_lucas_fixture(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE gpkg_contents (table_name TEXT, data_type TEXT, srs_id INTEGER);
        INSERT INTO gpkg_contents VALUES ('survey', 'features', 3035);
        CREATE TABLE gpkg_spatial_ref_sys (
            srs_id INTEGER, organization TEXT, organization_coordsys_id INTEGER
        );
        INSERT INTO gpkg_spatial_ref_sys VALUES (3035, 'EPSG', 3035);
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT, column_name TEXT, geometry_type_name TEXT,
            srs_id INTEGER, z INTEGER, m INTEGER
        );
        INSERT INTO gpkg_geometry_columns VALUES ('survey', 'geom', 'MULTISURFACE', 3035, 0, 0);
        CREATE TABLE survey (
            fid INTEGER PRIMARY KEY, surveycprnlc TEXT, geom BLOB,
            survey_year INTEGER, point_copernicus INTEGER,
            surveycprncando INTEGER, survey_obs_type TEXT
        );
        CREATE TABLE rtree_survey_geom (
            id INTEGER PRIMARY KEY, minx REAL, maxx REAL, miny REAL, maxy REAL
        );
        """
    )
    geometry = box(0, 0, 8, 10)
    connection.execute(
        "INSERT INTO survey VALUES (?, ?, ?, ?, ?, ?, ?)",
        (1, "A11", _gpkg_blob(geometry), 2022, 1, 1, "Field Survey, point visible <= 100 m"),
    )
    connection.execute("INSERT INTO rtree_survey_geom VALUES (?, ?, ?, ?, ?)", (1, 0, 8, 0, 10))
    connection.execute(
        "INSERT INTO survey VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            2,
            "A21",
            _gpkg_blob(box(8, 0, 10, 10)),
            2021,
            1,
            1,
            "Field Survey, point visible <= 100 m",
        ),
    )
    connection.execute("INSERT INTO rtree_survey_geom VALUES (?, ?, ?, ?, ?)", (2, 8, 10, 0, 10))
    rejected_rows = (
        (3, "A21", 2022, 0, 1, "Field Survey, point visible <= 100 m"),
        (4, "A21", 2022, 1, 0, "Field Survey, point visible <= 100 m"),
        (5, "A21", 2022, 1, 1, "Photo-interpretation in the field"),
        (6, None, 2022, 1, 1, "Field Survey, point visible <= 100 m"),
    )
    for feature_id, support_class, year, copernicus, can_do, observation_type in rejected_rows:
        connection.execute(
            "INSERT INTO survey VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                feature_id,
                support_class,
                _gpkg_blob(geometry),
                year,
                copernicus,
                can_do,
                observation_type,
            ),
        )
        connection.execute(
            "INSERT INTO rtree_survey_geom VALUES (?, ?, ?, ?, ?)",
            (feature_id, 0, 8, 0, 10),
        )
    connection.commit()
    connection.close()


def test_lucas_reader_uses_exact_file_pins_and_records_feature_ids(tmp_path: Path) -> None:
    path = tmp_path / "lucas-fixture.gpkg"
    taxonomy_path = tmp_path / "lucas-codebook.fixture"
    taxonomy_bytes = b"synthetic fixture taxonomy\n"
    taxonomy_path.write_bytes(taxonomy_bytes)
    taxonomy = _lucas_taxonomy(sha256=hashlib.sha256(taxonomy_bytes).hexdigest())
    _create_lucas_fixture(path)
    manifest = LucasReferenceManifest(
        release_id="synthetic-fixture-only",
        source_uri="https://example.test/fixture.gpkg",
        source_revision="fixture-source-1",
        file_size_bytes=path.stat().st_size,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        taxonomy_id="fixture-lucas",
        taxonomy_uri="https://example.test/codebook.csv",
        taxonomy_revision="fixture-codebook-1",
        taxonomy_sha256=taxonomy.sha256,
        layer=LucasLayerManifest(
            table="survey",
            geometry_column="geom",
            primary_key="fid",
            rtree_table="rtree_survey_geom",
            support_class_column="surveycprnlc",
            geometry_type="MULTISURFACE",
        ),
    )
    reader = LucasGeoPackageReader(path, taxonomy_path, manifest, taxonomy, fetch_size=1)
    with reader:
        result = reader.query(box(0, 0, 10, 10))
        attempts = tuple(
            iter_lucas_attempts(
                (
                    _polygon(0, crs="EPSG:3035").pool_polygon,
                    _polygon(1, crs="EPSG:3035").pool_polygon,
                ),
                reader,
                taxonomy,
                _reference_pin(ReferenceSystem.LUCAS),
            )
        )

    assert result.attempted_feature_ids == ("1", "2", "3", "4", "5", "6")
    assert result.attempt_reason == (
        "filtered:point_copernicus=1,survey_obs_type=1,survey_year=1,"
        "surveycprncando=1,surveycprnlc=1"
    )
    assert result.footprints == ()
    assert result.preaggregated_supports is not None
    assert result.preaggregated_supports == (
        SupportArea("L1", "A", 80.0, AssignmentKind.ANCESTOR_DERIVED, ("1",)),
        SupportArea("L2", "A1", 80.0, AssignmentKind.ANCESTOR_DERIVED, ("1",)),
        SupportArea("L3", "A11", 80.0, AssignmentKind.DIRECT, ("1",)),
    )
    assert result.exclusion_reason is None
    assert len(attempts) == 2
    assert attempts[0].attempt_evidence.attempted_reference_ids == ("1", "2", "3", "4", "5", "6")
    assert attempts[1].label_outcomes[0].status is OutcomeStatus.UNASSIGNED

    taxonomy_path.write_bytes(b"changed codebook")
    with pytest.raises(ValueError, match="taxonomy SHA-256"):
        verify_lucas_taxonomy_file(taxonomy_path, manifest)

    with path.open("ab") as output:
        output.write(b"changed")
    with (
        pytest.raises(ValueError, match="byte size"),
        LucasGeoPackageReader(path, taxonomy_path, manifest, taxonomy),
    ):
        pass


def test_lucas_failed_bounded_scan_preserves_ids_already_attempted(monkeypatch) -> None:
    taxonomy = _lucas_taxonomy()
    layer = LucasLayerManifest(
        table="survey",
        geometry_column="geom",
        primary_key="fid",
        rtree_table="rtree_survey_geom",
        support_class_column="surveycprnlc",
        geometry_type="MULTISURFACE",
    )
    manifest = LucasReferenceManifest(
        release_id="synthetic-fixture-only",
        source_uri="https://example.test/fixture.gpkg",
        source_revision="fixture-source-1",
        file_size_bytes=1,
        file_sha256=_SHA_A,
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_uri="https://example.test/codebook.csv",
        taxonomy_revision=taxonomy.revision,
        taxonomy_sha256=taxonomy.sha256,
        layer=layer,
    )
    reader = LucasGeoPackageReader(Path("unused.gpkg"), Path("unused.pdf"), manifest, taxonomy)
    reader._connection = sqlite3.connect(":memory:")

    class BrokenCursor:
        calls = 0

        def fetchmany(self, _size):
            self.calls += 1
            if self.calls == 1:
                return [
                    (
                        17,
                        "A11",
                        _gpkg_blob(box(0, 0, 5, 10)),
                        2022,
                        1,
                        1,
                        "Field Survey, point visible <= 100 m",
                    )
                ]
            raise sqlite3.DatabaseError("synthetic failure after one row")

    monkeypatch.setattr(
        lucas_module,
        "_query_rtree_candidates",
        lambda *_args: BrokenCursor(),
    )
    try:
        result = reader.query(box(0, 0, 10, 10))
    finally:
        reader.__exit__()

    assert result.exclusion_reason == "lucas_reference_query_failed:DatabaseError"
    assert result.attempted_feature_ids == ("17",)


def test_bounded_export_preserves_adapter_rows_and_unmatched_text_link(tmp_path: Path) -> None:
    sample = _polygon(1)
    evidence = WorldCoverEvidence(
        "way/1",
        100.0,
        ((10, 80.0),),
        WORLDCOVER_AREA_MEASUREMENT,
        attempted_tiles=("N00E000",),
        class_evidence_ids=((10, ("N00E000",)),),
    )
    source = SourceSnapshot(
        snapshot_id="raw-description",
        repository="NoeFlandre/source-polygons",
        revision="source-commit-1",
        pbf_uri="https://artifacts.example/source.osm.pbf",
        pbf_sha256=_SHA_A,
    )
    inventory = (sample.pool_polygon.inventory,)
    scan_report = b"complete one-row raw scan"
    scan_report_path = tmp_path / "raw-scan.json"
    scan_report_path.write_bytes(scan_report)
    document = SourceDocument(
        "unmatched-document",
        canonical_native_key({"document": "unmatched-document"}),
        "text-revision-1",
        None,
        BodyStatus.MISSING,
    )
    link = TextPolygonLink(
        document_id=document.document_id,
        inventory_row_id=None,
        native_link_key=canonical_native_key({"polygon_id": "not-in-source"}),
        source_revision="text-revision-1",
        status=TextLinkStatus.UNMATCHED,
        evidence="source-native link key has no polygon target",
    )
    metadata = PoolBuildInput(
        pool_key=PoolKey(TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER),
        source_snapshot=source,
        raw_inventory_evidence=build_raw_inventory_evidence(
            enumeration_rule="all polygons before text/relevance/evaluation joins",
            scan_report_uri=scan_report_path.as_uri(),
            scan_report=scan_report,
            expected_key_count=1,
            expected_keyset_sha256=native_keyset_sha256(row.native_key for row in inventory),
        ),
        reference_pin=_reference_pin(ReferenceSystem.WORLDCOVER),
        adapter_id="worldcover-source-wide-adapter",
        adapter_revision="adapter-commit-1",
        builder_revision="6ceddf3477c282b366bed4d502a1aaa68e8c194d",
        inventory=(),
        polygons=(),
        levels=WORLDCOVER_LEVELS,
        label_candidates=(),
        label_outcomes=(),
        documents=(),
        text_polygon_links=(),
    )
    receipt = export_labeled_polygons(
        metadata,
        iter_worldcover_rows(
            iter((sample.pool_polygon,)),
            lambda _polygon: evidence,
            _reference_pin(ReferenceSystem.WORLDCOVER),
        ),
        FilesystemStreamingSink(tmp_path),
        options=AdapterStreamingOptions(
            limits=StreamingLimits(max_batch_rows=6),
            staging_directory=tmp_path,
            supplemental_batches=(
                PoolBatch(
                    batch_index=77,
                    documents=(document,),
                    text_polygon_links=(link,),
                ),
            ),
        ),
    )

    assert receipt.total_source_keys == 1
    assert receipt.inventory_keyset_sha256 == native_keyset_sha256(
        row.native_key for row in inventory
    )
    manifests = list(tmp_path.rglob("manifest.json"))
    assert len(manifests) == 1
    exported = json.loads(manifests[0].read_text())
    assert exported["storage_layout"]["tables"]["source_documents"]["row_count"] == 1
    assert exported["storage_layout"]["tables"]["text_polygon_links"]["row_count"] == 1
    assert exported["storage_layout"]["tables"]["polygon_attempts"]["row_count"] == 1
    assert exported["storage_layout"]["tables"]["label_support_evidence"]["row_count"] == 1
    gate = validate_nine_streamed_pool_gate((receipt,), FilesystemArtifactReader())
    assert gate.missing_pools == tuple(
        sorted(
            f"{text_source}-{reference_system}"
            for text_source in TextSource
            for reference_system in ReferenceSystem
            if (text_source, reference_system)
            != (TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER)
        )
    )
    assert gate.duplicate_pools == ()
    assert gate.invalid_pools == ()


def test_lucas_taxonomy_and_manifest_validation_is_exact() -> None:
    taxonomy = _lucas_taxonomy()
    with pytest.raises(ValueError, match="identity and revision"):
        replace(taxonomy, taxonomy_id=" ")
    with pytest.raises(ValueError, match="lowercase 64-digit"):
        replace(taxonomy, sha256="A" * 64)
    with pytest.raises(ValueError, match="contain code paths"):
        replace(taxonomy, codes=())
    with pytest.raises(ValueError, match="contain L1, L2, and L3"):
        replace(taxonomy, codes=(LucasCode("A", "", "A11"),))
    with pytest.raises(ValueError, match="must be unique"):
        replace(taxonomy, codes=(*taxonomy.codes, taxonomy.codes[0]))
    with pytest.raises(ValueError, match="multiple L1 parents"):
        replace(taxonomy, codes=(*taxonomy.codes, LucasCode("B", "A1", "B11")))
    with pytest.raises(ValueError, match="multiple parents"):
        replace(taxonomy, codes=(*taxonomy.codes, LucasCode("B", "B1", "A11")))

    from_records = LucasTaxonomy.from_records(
        ({"L1": " A ", "L2": " A1 ", "L3": " A11 "},),
        taxonomy_id="fixture",
        revision="fixture-revision",
        sha256=_SHA_B,
    )
    assert from_records.codes == (LucasCode("A", "A1", "A11"),)
    with pytest.raises(KeyError):
        LucasTaxonomy.from_records(
            ({"L1": "A", "L2": "A1"},),
            taxonomy_id="fixture",
            revision="fixture-revision",
            sha256=_SHA_B,
        )

    for args in ((None, "A1", None), ("Z", None, None), ("A", None, "A11")):
        with pytest.raises(ValueError):
            taxonomy.validate_path(*args)
    with pytest.raises(ValueError, match="unknown LUCAS L2"):
        taxonomy.validate_path("A", "Z", None)
    with pytest.raises(ValueError, match="unknown LUCAS L3"):
        taxonomy.validate_path("A", "A1", "A21")
    taxonomy.validate_path("A", "A1", "A11")

    with pytest.raises(ValueError, match="EPSG:3035"):
        LucasLayerManifest("survey", "geom", "fid", "rtree", "class", srs_id=4326)
    with pytest.raises(ValueError, match="identifiers must be non-empty"):
        LucasLayerManifest("survey\x00bad", "geom", "fid", "rtree", "class")
    manifest = lucas_2022_copernicus_manifest()
    for changes, message in (
        ({"release_id": " "}, "pins must be present"),
        ({"file_size_bytes": 0}, "size must be positive"),
        ({"file_sha256": "partial"}, "full SHA-256"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(manifest, **changes)


def test_lucas_file_pins_and_reader_preconditions_fail_closed(tmp_path: Path) -> None:
    taxonomy = _lucas_taxonomy()
    manifest = lucas_2022_copernicus_manifest()
    reader_args = (tmp_path / "not-downloaded.gpkg", tmp_path / "codebook.pdf", manifest, taxonomy)
    with pytest.raises(ValueError, match="fetch size must be positive"):
        LucasGeoPackageReader(*reader_args, fetch_size=0)
    with pytest.raises(ValueError, match="taxonomy ID"):
        LucasGeoPackageReader(
            reader_args[0], reader_args[1], manifest, replace(taxonomy, taxonomy_id="other")
        )
    with pytest.raises(ValueError, match="taxonomy revision"):
        LucasGeoPackageReader(
            reader_args[0],
            reader_args[1],
            manifest,
            replace(taxonomy, taxonomy_id=manifest.taxonomy_id, revision="other"),
        )
    with pytest.raises(ValueError, match="taxonomy hash"):
        LucasGeoPackageReader(
            reader_args[0],
            reader_args[1],
            manifest,
            replace(
                taxonomy,
                taxonomy_id=manifest.taxonomy_id,
                revision=manifest.taxonomy_revision,
                sha256=_SHA_A,
            ),
        )
    with pytest.raises(FileNotFoundError, match="reference is not present"):
        verify_lucas_file(reader_args[0], manifest)
    with pytest.raises(FileNotFoundError, match="taxonomy is not present"):
        verify_lucas_taxonomy_file(reader_args[1], manifest)

    reference_path = tmp_path / "reference.bin"
    reference_path.write_bytes(b"synthetic reference")
    same_size_bad_hash = replace(
        manifest,
        file_size_bytes=reference_path.stat().st_size,
        file_sha256=_SHA_A,
    )
    with pytest.raises(ValueError, match="SHA-256"):
        verify_lucas_file(reference_path, same_size_bad_hash)
    with pytest.raises(ValueError, match="byte size"):
        verify_lucas_file(reference_path, replace(manifest, file_size_bytes=1))


def test_lucas_reader_query_and_schema_failures_are_reported(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "fixture.gpkg"
    taxonomy_path = tmp_path / "fixture.csv"
    taxonomy_bytes = b"synthetic fixture taxonomy\n"
    taxonomy_path.write_bytes(taxonomy_bytes)
    taxonomy = _lucas_taxonomy(sha256=hashlib.sha256(taxonomy_bytes).hexdigest())
    _create_lucas_fixture(path)
    manifest = LucasReferenceManifest(
        release_id="fixture",
        source_uri="https://example.test/fixture.gpkg",
        source_revision="fixture-revision",
        file_size_bytes=path.stat().st_size,
        file_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_uri="https://example.test/fixture.csv",
        taxonomy_revision=taxonomy.revision,
        taxonomy_sha256=taxonomy.sha256,
        layer=LucasLayerManifest("survey", "geom", "fid", "rtree_survey_geom", "surveycprnlc"),
    )
    reader = LucasGeoPackageReader(path, taxonomy_path, manifest, taxonomy)
    with pytest.raises(RuntimeError, match="opened before querying"):
        reader.query(box(0, 0, 1, 1))
    reader._connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="valid and non-empty"):
            reader.query(shapely.GeometryCollection())
        monkeypatch.setattr(
            lucas_module,
            "_query_rtree_candidates",
            lambda *_args: (_ for _ in ()).throw(sqlite3.DatabaseError("fixture query error")),
        )
        failed = reader.query(box(0, 0, 1, 1))
        assert failed.exclusion_reason == "lucas_reference_query_failed:DatabaseError"
    finally:
        reader.__exit__()

    closed = LucasGeoPackageReader(path, taxonomy_path, manifest, taxonomy)
    with pytest.raises(RuntimeError, match="connection is not open"):
        closed._validate_schema()


def test_lucas_row_parser_preserves_unusable_and_unmatched_features() -> None:
    taxonomy = _lucas_taxonomy()
    layer = LucasLayerManifest("survey", "geom", "fid", "rtree", "class")
    query_geometry = box(0, 0, 10, 10)
    base = (2022, 1, 1, "Field Survey, point visible <= 100 m")

    non_text = lucas_module._parse_reference_row(
        (1, 5, _gpkg_blob(box(0, 0, 2, 2)), *base), query_geometry, taxonomy, layer
    )
    assert non_text[2:] == (None, "support_class_type_invalid")
    unknown = lucas_module._parse_reference_row(
        (2, "NOT-A-CODE", _gpkg_blob(box(0, 0, 2, 2)), *base),
        query_geometry,
        taxonomy,
        layer,
    )
    assert unknown[2] == "support_class_not_in_c3"
    malformed = lucas_module._parse_reference_row(
        (3, "A11", b"bad", *base), query_geometry, taxonomy, layer
    )
    assert malformed[3] == "reference_geometry_unusable"
    wrong_shape = lucas_module._parse_reference_row(
        (4, "A11", _gpkg_blob(LineString([(0, 0), (1, 1)])), *base),
        query_geometry,
        taxonomy,
        layer,
    )
    assert wrong_shape[3] == "reference_geometry_unusable"
    outside = lucas_module._parse_reference_row(
        (5, "A11", _gpkg_blob(box(20, 20, 22, 22)), *base),
        query_geometry,
        taxonomy,
        layer,
    )
    assert outside[1:] == (None, None, None)

    for malformed_blob, expected in (
        ("not-bytes", "not a GeoPackage"),
        (b"GP", "invalid GeoPackage"),
        (b"GP\x00\x01" + struct.pack("<i", 4326) + shapely.to_wkb(box(0, 0, 1, 1)), "CRS"),
        (
            b"GP\x00\x0b" + struct.pack("<i", 3035) + shapely.to_wkb(box(0, 0, 1, 1)),
            "envelope type",
        ),
    ):
        with pytest.raises((TypeError, ValueError), match=expected):
            lucas_module._decode_gpkg_geometry(malformed_blob, 3035)


def test_lucas_support_statuses_units_and_bounded_union() -> None:
    polygon = _polygon(0, crs="EPSG:3035")
    taxonomy = _lucas_taxonomy()
    pin = _reference_pin(ReferenceSystem.LUCAS)
    with pytest.raises(ValueError, match="square metres"):
        measure_lucas_supports(
            polygon.pool_polygon,
            LucasFeatureQuery((), ()),
            taxonomy,
            replace(pin, support_unit="ha"),
        )

    excluded = measure_lucas_supports(
        polygon.pool_polygon,
        LucasFeatureQuery((), ("seen",), exclusion_reason="lucas_reference_query_failed:OSError"),
        taxonomy,
        pin,
    )
    assert excluded.attempt_evidence.status == "excluded"
    assert excluded.attempt_evidence.reason == "lucas_reference_query_failed:OSError"
    partial = measure_lucas_supports(
        polygon.pool_polygon,
        LucasFeatureQuery((), ("seen",), partial_reason="reference_geometry_unusable"),
        taxonomy,
        pin,
    )
    assert partial.attempt_evidence.status == "partial"
    assert partial.attempt_evidence.reason == "reference_geometry_unusable"

    unusable_area = replace(
        polygon.pool_polygon, snapshot=replace(polygon.snapshot, original_area=0.0)
    )
    no_area = measure_lucas_supports(unusable_area, LucasFeatureQuery((), ("seen",)), taxonomy, pin)
    assert no_area.label_outcomes[0].reason == "geometry_area_unusable"
    wrong_crs = replace(polygon.pool_polygon, snapshot=replace(polygon.snapshot, crs="EPSG:4326"))
    with pytest.raises(ValueError, match="EPSG:3035 square metres"):
        measure_lucas_supports(wrong_crs, LucasFeatureQuery((), ()), taxonomy, pin)
    wrong_area = replace(
        polygon.pool_polygon, snapshot=replace(polygon.snapshot, original_area=99.0)
    )
    with pytest.raises(ValueError, match="full original polygon area"):
        measure_lucas_supports(wrong_area, LucasFeatureQuery((), ()), taxonomy, pin)

    forest = lucas_module._SupportUnionForest()
    for index in range(129):
        forest.add(box(index, 0, index + 1, 1))
    assert shapely.area(forest.finish()) == 129.0


def test_lucas_streaming_union_retains_logarithmic_geometry_state() -> None:
    taxonomy = _lucas_taxonomy()
    accumulator = lucas_module._LucasSupportAccumulator(box(0, 0, 10, 10), taxonomy)
    footprint_count = 4095
    for index in range(footprint_count):
        accumulator.add(LucasFootprint(str(index), "A", "A1", "A11", box(0, 0, 8, 10)))

    supports = accumulator.finish()
    expected_state_bound = 3 * footprint_count.bit_length()
    assert accumulator.peak_retained_union_geometry_count <= expected_state_bound
    assert accumulator.peak_retained_union_geometry_count < footprint_count
    assert [(row.level, row.code, row.numerator) for row in supports] == [
        ("L1", "A", 80.0),
        ("L2", "A1", 80.0),
        ("L3", "A11", 80.0),
    ]
    assert all(len(row.evidence_ids) == footprint_count for row in supports)


def test_source_wide_runners_attempt_raw_rows_and_report_query_errors(monkeypatch) -> None:
    polygon = _polygon(51)
    unusable = replace(polygon.pool_polygon, geometry=None)
    errors = tuple(
        iter_worldcover_rows(
            (polygon.pool_polygon, unusable),
            lambda _row: (_ for _ in ()).throw(OSError("raster fixture error")),
            _reference_pin(ReferenceSystem.WORLDCOVER),
        )
    )
    assert len(errors) == 2
    assert errors[0][1].label_outcomes[0].reason == (
        "worldcover_extraction_failed:worldcover_reference_query_failed:OSError"
    )
    assert errors[1][1].label_outcomes[0].reason == "geometry_unusable"

    extra_evidence = (
        WorldCoverEvidence("way/51", 100.0, (), WORLDCOVER_AREA_MEASUREMENT),
        WorldCoverEvidence("way/52", 100.0, (), WORLDCOVER_AREA_MEASUREMENT),
    )
    with pytest.raises(ValueError, match="counts differ"):
        tuple(
            iter_worldcover_source_attempts(
                (polygon.pool_polygon,), extra_evidence, _reference_pin(ReferenceSystem.WORLDCOVER)
            )
        )
    with pytest.raises(ValueError, match="does not match"):
        tuple(
            iter_worldcover_source_attempts(
                (polygon.pool_polygon,),
                (replace(extra_evidence[0], source_polygon_id="way/other"),),
                _reference_pin(ReferenceSystem.WORLDCOVER),
            )
        )

    taxonomy = _lucas_taxonomy()
    layer = LucasLayerManifest("survey", "geom", "fid", "rtree", "class")
    manifest = LucasReferenceManifest(
        "fixture",
        "https://example.test/gpkg",
        "rev",
        1,
        _SHA_A,
        taxonomy.taxonomy_id,
        "https://example.test/codebook",
        taxonomy.revision,
        taxonomy.sha256,
        layer,
    )
    reader = LucasGeoPackageReader(Path("unused"), Path("unused"), manifest, taxonomy)
    with pytest.raises(RuntimeError, match="open, verified"):
        tuple(
            iter_lucas_rows(
                (polygon.pool_polygon,), reader, taxonomy, _reference_pin(ReferenceSystem.LUCAS)
            )
        )
    reader._connection = sqlite3.connect(":memory:")
    try:
        with pytest.raises(ValueError, match="do not match"):
            tuple(
                iter_lucas_rows(
                    (polygon.pool_polygon,),
                    reader,
                    _lucas_taxonomy(sha256=_SHA_A),
                    _reference_pin(ReferenceSystem.LUCAS),
                )
            )
        monkeypatch.setattr(
            reader,
            "query",
            lambda _geometry: (_ for _ in ()).throw(RuntimeError("synthetic query error")),
        )
        failed = tuple(
            iter_lucas_rows(
                (polygon.pool_polygon,), reader, taxonomy, _reference_pin(ReferenceSystem.LUCAS)
            )
        )
        assert failed[0][1].label_outcomes[0].reason == "lucas_reference_query_failed:RuntimeError"
    finally:
        reader.__exit__()


def test_adapter_batch_guardrails_keep_records_aligned() -> None:
    polygon = _polygon(61).pool_polygon
    record = worldcover_supports(
        polygon,
        WorldCoverEvidence(
            polygon.source_polygon_id,
            100.0,
            ((10, 80.0),),
            WORLDCOVER_AREA_MEASUREMENT,
            class_evidence_ids=((10, ("tile",)),),
        ),
        _reference_pin(ReferenceSystem.WORLDCOVER),
    )
    with pytest.raises(ValueError, match="max_batch_rows"):
        tuple(iter_polygon_batches((polygon,), (record,), max_batch_rows=5))
    with pytest.raises(ValueError, match=r"zip\(\) argument 2 is shorter"):
        tuple(iter_polygon_batches((polygon, polygon), (record,), max_batch_rows=12))
    mismatched = replace(record, inventory_row_id="other")
    with pytest.raises(ValueError, match="does not match the next raw polygon"):
        tuple(iter_polygon_batches((polygon,), (mismatched,), max_batch_rows=12))
