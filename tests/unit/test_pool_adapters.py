from __future__ import annotations

import hashlib
import json
import sqlite3
import struct
from dataclasses import dataclass
from pathlib import Path

import pytest
import shapely
from shapely.geometry import box

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
    WorldCoverEvidence,
    build_raw_inventory_evidence,
    export_labeled_polygons,
    iter_lucas_attempts,
    iter_polygon_batches,
    iter_worldcover_attempts,
    iter_worldcover_rows,
    iter_worldcover_source_attempts,
    measure_lucas_supports,
    verify_lucas_taxonomy_file,
    worldcover_supports,
)
from osm_polygon_eunis.pool_contract import (
    AssignmentKind,
    BodyStatus,
    FilesystemStreamingSink,
    GeometryStatus,
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
    assert result.attempt_evidence.support_evidence == (("class", "10", ("N00E000", "N00E003")),)


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
        ),
        WorldCoverEvidence(
            "way/2",
            100.0,
            ((20, 80.0),),
            WORLDCOVER_AREA_MEASUREMENT,
            attempted_tiles=("N00E003",),
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
        ),
        _reference_pin(ReferenceSystem.WORLDCOVER),
    )

    batches = tuple(
        iter_polygon_batches((row.pool_polygon for row in polygons), attempts, max_batch_rows=4)
    )

    assert [batch.batch_index for batch in batches] == [0, 1, 2]
    assert all(batch.row_count == 4 for batch in batches)


def test_lucas_unions_same_class_before_full_area_threshold() -> None:
    polygon = _polygon(0, crs="EPSG:3035")
    taxonomy = _lucas_taxonomy()
    query = LucasFeatureQuery(
        footprints=(
            LucasFootprint("1", "A", "A1", "A11", box(0, 0, 6, 10)),
            LucasFootprint("2", "A", "A1", "A11", box(4, 0, 8, 10)),
            LucasFootprint("3", "A", "A2", "A21", box(9, 0, 10, 10)),
        ),
        attempted_feature_ids=("1", "2", "3"),
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
    assert result.attempt_evidence.attempted_reference_ids == ("1", "2", "3")
    assert ("L3", "A11", ("1", "2")) in result.attempt_evidence.support_evidence


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
        CREATE TABLE gpkg_geometry_columns (
            table_name TEXT, column_name TEXT, geometry_type_name TEXT,
            srs_id INTEGER, z INTEGER, m INTEGER
        );
        INSERT INTO gpkg_geometry_columns VALUES ('survey', 'geom', 'POLYGON', 3035, 0, 0);
        CREATE TABLE survey (
            fid INTEGER PRIMARY KEY, l1 TEXT, l2 TEXT, l3 TEXT, geom BLOB
        );
        CREATE TABLE rtree_survey_geom (
            id INTEGER PRIMARY KEY, minx REAL, maxx REAL, miny REAL, maxy REAL
        );
        """
    )
    geometry = box(0, 0, 8, 10)
    connection.execute(
        "INSERT INTO survey VALUES (?, ?, ?, ?, ?)", (1, "A", "A1", "A11", _gpkg_blob(geometry))
    )
    connection.execute("INSERT INTO rtree_survey_geom VALUES (?, ?, ?, ?, ?)", (1, 0, 8, 0, 10))
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
        layer=LucasLayerManifest("survey", "geom", "fid", "rtree_survey_geom", "l1", "l2", "l3"),
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

    assert result.attempted_feature_ids == ("1",)
    assert len(result.footprints) == 1
    assert result.footprints[0].l3_code == "A11"
    assert result.exclusion_reason is None
    assert len(attempts) == 2
    assert attempts[0].attempt_evidence.attempted_reference_ids == ("1",)
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


def test_bounded_export_preserves_adapter_rows_and_unmatched_text_link(tmp_path: Path) -> None:
    sample = _polygon(1)
    evidence = WorldCoverEvidence("way/1", 100.0, ((10, 80.0),), WORLDCOVER_AREA_MEASUREMENT)
    source = SourceSnapshot(
        snapshot_id="raw-description",
        repository="NoeFlandre/source-polygons",
        revision="source-commit-1",
        pbf_uri="https://artifacts.example/source.osm.pbf",
        pbf_sha256=_SHA_A,
    )
    inventory = (sample.pool_polygon.inventory,)
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
            scan_report_uri="https://artifacts.example/raw-scan.json",
            scan_report=b"complete one-row raw scan",
            expected_key_count=1,
            expected_keyset_sha256=native_keyset_sha256(row.native_key for row in inventory),
        ),
        reference_pin=_reference_pin(ReferenceSystem.WORLDCOVER),
        adapter_id="worldcover-source-wide-adapter",
        adapter_revision="adapter-commit-1",
        builder_revision="26a6ce9bf934530e2bb9a5ca51eb5147d2adb9a7",
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
            limits=StreamingLimits(max_batch_rows=4),
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
