from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
import shapely
from osm_worldcover.adapters.worldcover.geodesic_coverage import (
    AREA_MEASUREMENT,
    _equal_area_geometry,
)
from osm_worldcover.adapters.worldcover.pool_attempts import (
    SourceWideCoverageAttempt,
    iter_sourcewide_coverage,
)
from osm_worldcover.domain.tiling import Tile
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis.pool_adapters import (
    WORLDCOVER_AREA_MEASUREMENT,
    WORLDCOVER_LEVELS,
    PoolPolygonInput,
    build_raw_inventory_evidence,
    iter_worldcover_source_rows,
    worldcover_area_measurement_pin,
)
from osm_polygon_eunis.pool_adapters.streaming import (
    AdapterStreamingOptions,
    export_labeled_polygons,
)
from osm_polygon_eunis.pool_adapters.workers import (
    _load_pinned_worldcover_producer,
    _worldcover_install_revision,
)
from osm_polygon_eunis.pool_contract import (
    ArtifactManifestPin,
    FilesystemArtifactReader,
    FilesystemStreamingSink,
    GeometryStatus,
    OutcomeStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBuildInput,
    PoolKey,
    ReferencePin,
    ReferenceSystem,
    SourceSnapshot,
    StreamingExportReceipt,
    SupportSemantics,
    TextSource,
    canonical_native_key,
    native_keyset_sha256,
    validate_nine_streamed_evaluation_ready_gate,
)
from osm_polygon_eunis.pool_contract.builder import (
    inventory_row_identifier,
    polygon_snapshot_identifier,
)

_SOURCE_SNAPSHOT_ID = "tiny-worldcover-raster-fixture"
_PRODUCER_REVISION = "18c75517c74be22ab2f90edce2b65d3600461048"


@dataclass(frozen=True, slots=True)
class _ExportFixture:
    attempts: tuple[SourceWideCoverageAttempt, ...]
    pool_polygons: tuple[PoolPolygonInput, ...]
    exact_area: float
    below_area: float
    receipt: StreamingExportReceipt
    sink_root: Path


class _RasterTiles:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.attempted: list[str] = []
        self.discarded: list[str] = []

    def ensure(self, tile: Tile) -> Path:
        self.attempted.append(tile.name)
        return self.path

    def discard(self, tile: Tile) -> None:
        self.discarded.append(tile.name)


def _write_raster(path: Path, values: np.ndarray, transform) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype=values.dtype,
        crs="EPSG:4326",
        transform=transform,
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)


def _canonical_attempt(source_id: str, geometry, raster: Path):
    raw_rows = pd.DataFrame([{"polygon_id": source_id, "geometry": geometry}])
    tile_source = _RasterTiles(raster)
    attempt = next(iter_sourcewide_coverage(raw_rows, tile_source))
    assert tile_source.attempted == tile_source.discarded == ["N00E000"]
    return attempt


def _pool_polygon(
    tmp_path: Path,
    index: int,
    source_id: str,
    geometry,
    area_m2: float,
) -> PoolPolygonInput:
    native_key = canonical_native_key(
        {"source_pbf": "tiny-worldcover.osm.pbf", "osm_type": "way", "osm_id": index}
    )
    geometry_sha256 = hashlib.sha256(shapely.to_wkb(geometry)).hexdigest()
    inventory_id = inventory_row_identifier(_SOURCE_SNAPSHOT_ID, native_key)
    snapshot_id = polygon_snapshot_identifier(_SOURCE_SNAPSHOT_ID, native_key, geometry_sha256)
    inventory = PolygonInventoryRow(
        inventory_row_id=inventory_id,
        native_key=native_key,
        source_row_locator=f"polygons/tiny-worldcover.parquet#row={index - 1}",
        polygon_snapshot_id=snapshot_id,
    )
    geometry_path = tmp_path / f"worldcover-fixture-{index}.wkb"
    geometry_path.write_bytes(shapely.to_wkb(geometry))
    snapshot = PolygonSnapshot(
        polygon_snapshot_id=snapshot_id,
        inventory_row_id=inventory_id,
        native_key=native_key,
        geometry_uri=geometry_path.as_uri(),
        geometry_sha256=geometry_sha256,
        geometry_format="WKB",
        crs="EPSG:4326",
        coordinate_transform="identity",
        geometry_status=GeometryStatus.USABLE,
        original_area=area_m2,
        area_unit="m^2",
    )
    return PoolPolygonInput(source_id, inventory, snapshot, geometry)


def _reference_pin() -> ReferencePin:
    return ReferencePin(
        release_id="worldcover-v200-tiny-fixture",
        source_repository="NoeFlandre/osm-worldcover",
        source_revision=_PRODUCER_REVISION,
        taxonomy_id="worldcover-v200",
        taxonomy_revision="v200",
        support_semantics=SupportSemantics.MAPPED_COVER,
        support_unit="m^2",
        threshold=0.8,
        support_union_policy="union_by_class",
    )


def _artifact_manifest_pin(
    tmp_path: Path,
    name: str,
    artifact_paths: tuple[Path, ...],
) -> ArtifactManifestPin:
    rows = [
        {
            "artifact_uri": path.as_uri(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in sorted(artifact_paths, key=lambda item: item.as_uri())
    ]
    content = b"".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode() + b"\n"
        for row in rows
    )
    manifest_path = tmp_path / f"{name}-artifact-manifest.jsonl"
    manifest_path.write_bytes(content)
    return ArtifactManifestPin(
        format_version="artifact-list-jsonl-v1",
        manifest_uri=manifest_path.as_uri(),
        manifest_sha256=hashlib.sha256(content).hexdigest(),
        expected_artifact_count=len(rows),
    )


def _metadata(
    tmp_path: Path,
    polygons: tuple[PoolPolygonInput, ...],
    reference_artifacts: tuple[Path, ...],
) -> PoolBuildInput:
    scan_report = b"tiny raster integration raw polygon inventory\n"
    scan_path = tmp_path / "raw-inventory.json"
    scan_path.write_bytes(scan_report)
    source_path = tmp_path / "tiny-source-inventory.jsonl"
    source_inventory_rows = []
    for polygon in polygons:
        if polygon.snapshot is None:
            raise ValueError("WorldCover source fixture needs a pinned polygon snapshot")
        source_inventory_rows.append(
            json.dumps(
                {
                    "native_key": polygon.inventory.native_key,
                    "geometry_sha256": polygon.snapshot.geometry_sha256,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    source_path.write_text("\n".join(source_inventory_rows) + "\n", encoding="utf-8")
    source_manifest = _artifact_manifest_pin(tmp_path, "source", (source_path,))
    reference_manifest = _artifact_manifest_pin(tmp_path, "reference", reference_artifacts)
    reference_pin = replace(
        _reference_pin(),
        reference_artifact_manifest=reference_manifest,
        area_measurement=worldcover_area_measurement_pin(_PRODUCER_REVISION),
    )
    return PoolBuildInput(
        pool_key=PoolKey(TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER),
        source_snapshot=SourceSnapshot(
            snapshot_id=_SOURCE_SNAPSHOT_ID,
            repository="NoeFlandre/osm-polygon-eunis",
            revision="tiny-fixture-source-v1",
            source_artifact_manifest=source_manifest,
        ),
        raw_inventory_evidence=build_raw_inventory_evidence(
            enumeration_rule="enumerate every raw polygon before text and evaluation joins",
            scan_report_uri=scan_path.as_uri(),
            scan_report=scan_report,
            expected_key_count=len(polygons),
            expected_keyset_sha256=native_keyset_sha256(
                polygon.inventory.native_key for polygon in polygons
            ),
        ),
        reference_pin=reference_pin,
        adapter_id="osm-polygon-eunis/worldcover-integration-fixture",
        adapter_revision="fixture-adapter-v1",
        builder_revision="fixture-builder-v1",
        inventory=(),
        polygons=(),
        levels=WORLDCOVER_LEVELS,
        label_candidates=(),
        label_outcomes=(),
    )


def _exported_table(receipt, sink_root: Path, table_name: str) -> pa.Table:
    shard_manifest = Path(receipt.shard_manifest_uri.removeprefix("file://"))
    shards = [json.loads(line) for line in shard_manifest.read_text().splitlines()]
    shard = next(row for row in shards if row["table"] == table_name)
    return pq.read_table(sink_root / shard["path"])


def _export_fixture(
    tmp_path: Path,
    *,
    include_unresolved: bool = True,
) -> _ExportFixture:
    raster = tmp_path / "four_of_five_cells.tif"
    _write_raster(
        raster,
        np.array([[10, 10, 10, 10, 0]], dtype=np.uint8),
        from_origin(1.0, 1.125, 0.125, 0.125),
    )
    all_nodata_raster = tmp_path / "all_nodata.tif"
    _write_raster(
        all_nodata_raster,
        np.zeros((1, 1), dtype=np.uint8),
        from_origin(1, 2, 1, 1),
    )
    exact_geometry = box(1.0, 1.0, 1.625, 1.125)
    below_geometry = box(1.0, 1.0, 1.625001, 1.125)
    unresolved_geometry = shapely.Polygon(
        ((1.1, 1.1), (1.9, 1.9), (1.9, 1.9 + 1e-14), (1.1, 1.1 + 1e-14))
    )
    attempts = (
        _canonical_attempt("way/1", exact_geometry, raster),
        _canonical_attempt("way/2", below_geometry, raster),
        _canonical_attempt("way/3", unresolved_geometry, all_nodata_raster),
    )
    exact_area = attempts[0].polygon_area_m2
    below_area = attempts[1].polygon_area_m2
    assert exact_area is not None and below_area is not None
    pool_polygons = (
        _pool_polygon(tmp_path, 1, "way/1", exact_geometry, exact_area),
        _pool_polygon(tmp_path, 2, "way/2", below_geometry, below_area),
        _pool_polygon(
            tmp_path,
            3,
            "way/3",
            unresolved_geometry,
            float(shapely.area(_equal_area_geometry(unresolved_geometry, densify_degrees=0.0001))),
        ),
    )
    if not include_unresolved:
        attempts = attempts[:2]
        pool_polygons = pool_polygons[:2]
    sink_root = tmp_path / "exported"
    sink = FilesystemStreamingSink(sink_root)
    metadata = _metadata(tmp_path, pool_polygons, (raster, all_nodata_raster))
    source_rows = iter_worldcover_source_rows(pool_polygons, attempts, metadata.reference_pin)
    receipt = export_labeled_polygons(
        metadata,
        source_rows,
        sink,
        options=AdapterStreamingOptions(staging_directory=tmp_path),
    )
    return _ExportFixture(attempts, pool_polygons, exact_area, below_area, receipt, sink_root)


def _assert_label_decisions(
    fixture: _ExportFixture,
    candidate_by_id: dict[str, dict[str, Any]],
    outcome_by_id: dict[str, dict[str, Any]],
) -> None:
    exact_id = fixture.pool_polygons[0].inventory.inventory_row_id
    exact_candidate = candidate_by_id[exact_id]
    assert exact_candidate["ratio"] == pytest.approx(0.8, abs=1e-12)
    assert exact_candidate["numerator"] == pytest.approx(fixture.attempts[0].class_areas_m2[0][1])
    assert exact_candidate["denominator"] == pytest.approx(fixture.exact_area)
    assert exact_candidate["denominator"] > exact_candidate["numerator"]
    assert outcome_by_id[exact_id]["status"] == OutcomeStatus.ASSIGNED.value

    below_id = fixture.pool_polygons[1].inventory.inventory_row_id
    assert candidate_by_id[below_id]["ratio"] < 0.8
    assert outcome_by_id[below_id]["status"] == OutcomeStatus.UNASSIGNED.value
    assert outcome_by_id[below_id]["assigned_code"] is None


def _assert_unresolved_is_excluded(
    fixture: _ExportFixture,
    candidate_by_id: dict[str, dict[str, Any]],
    outcome_by_id: dict[str, dict[str, Any]],
    attempt_by_id: dict[str, dict[str, Any]],
) -> None:
    unresolved_id = fixture.pool_polygons[2].inventory.inventory_row_id
    assert unresolved_id not in candidate_by_id
    assert outcome_by_id[unresolved_id]["status"] == OutcomeStatus.EXCLUDED.value
    reason = attempt_by_id[unresolved_id]["reason"]
    assert reason is not None
    assert "coverage_accuracy_bound_unresolved_at_0.8:class=10" in reason
    assert "class_support_error_bound_m2=" in reason


def _assert_diagnostics_and_stale_version_are_preserved(fixture: _ExportFixture) -> None:
    exact_id = fixture.pool_polygons[0].inventory.inventory_row_id
    diagnostics = fixture.attempts[0].measurement_diagnostics
    assert diagnostics is not None
    assert "observed_range=" in diagnostics
    assert "ratio_enclosure=" in diagnostics
    assert "projected_chord_boundary_error_bound_m2=" in diagnostics
    assert "class_support_error_m2=" in diagnostics
    attempts = _exported_table(fixture.receipt, fixture.sink_root, "polygon_attempts")
    attempt_by_id = {row["inventory_row_id"]: row for row in attempts.to_pylist()}
    assert attempt_by_id[exact_id]["reason"] == diagnostics

    stale_v3 = replace(fixture.attempts[0], area_measurement=f"{AREA_MEASUREMENT[:-2]}v3")
    with pytest.raises(ValueError, match="not the pinned equal-area method"):
        next(
            iter_worldcover_source_rows((fixture.pool_polygons[0],), (stale_v3,), _reference_pin())
        )


def test_pinned_worldcover_iterator_adapter_and_export_contract(tmp_path: Path) -> None:
    producer_requirement = (
        Path(__file__).with_name("requirements-worldcover.txt").read_text().strip()
    )
    assert producer_requirement.endswith(f"@{_PRODUCER_REVISION}")
    fixture = _export_fixture(tmp_path)

    assert AREA_MEASUREMENT == WORLDCOVER_AREA_MEASUREMENT
    assert "GEOS overlay and floating-point accumulation error are excluded" in AREA_MEASUREMENT
    assert fixture.attempts[0].class_areas_m2[0][1] / fixture.exact_area == pytest.approx(0.8)
    assert fixture.attempts[1].class_areas_m2[0][1] / fixture.below_area < 0.8
    assert fixture.attempts[2].class_areas_m2 == ()
    assert fixture.attempts[2].extraction_error is not None
    assert "coverage_accuracy_bound_unresolved_at_0.8:class=10" in (
        fixture.attempts[2].extraction_error
    )
    assert fixture.receipt.total_source_keys == 3

    candidates = _exported_table(fixture.receipt, fixture.sink_root, "label_candidates").to_pylist()
    outcomes = _exported_table(fixture.receipt, fixture.sink_root, "label_outcomes").to_pylist()
    candidate_by_id = {row["inventory_row_id"]: row for row in candidates}
    outcome_by_id = {row["inventory_row_id"]: row for row in outcomes}
    attempt_by_id = {
        row["inventory_row_id"]: row
        for row in _exported_table(
            fixture.receipt, fixture.sink_root, "polygon_attempts"
        ).to_pylist()
    }
    _assert_label_decisions(fixture, candidate_by_id, outcome_by_id)
    _assert_unresolved_is_excluded(fixture, candidate_by_id, outcome_by_id, attempt_by_id)
    _assert_diagnostics_and_stale_version_are_preserved(fixture)

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (fixture.receipt,), FilesystemArtifactReader()
    )
    assert readiness.missing_pools == tuple(
        sorted(
            f"{text_source.value}-{reference_system.value}"
            for text_source in TextSource
            for reference_system in ReferenceSystem
            if (text_source, reference_system)
            != (TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER)
        )
    )
    assert readiness.duplicate_pools == ()
    assert len(readiness.invalid_pools) == 1
    failed_pool, issues = readiness.invalid_pools[0]
    assert failed_pool == "description-worldcover"
    assert any("FAILED polygon attempts" in issue for issue in issues)
    unresolved_id = fixture.pool_polygons[2].inventory.inventory_row_id
    assert any(unresolved_id in issue for issue in issues)


def test_builtin_worker_accepts_only_the_live_pinned_worldcover_producer() -> None:
    producer = _load_pinned_worldcover_producer()

    assert _worldcover_install_revision() == _PRODUCER_REVISION
    assert all(
        callable(function)
        for function in (
            producer.iter_sourcewide_coverage,
            producer.is_usable_polygon,
            producer.tiles_for_bbox,
        )
    )
    assert producer.tile_not_published_error.__name__ == "TileNotPublishedError"
    assert len(producer.taxonomy_sha256) == 64


def test_v12_readiness_accepts_fully_pinned_completed_worldcover_attempts(
    tmp_path: Path,
) -> None:
    fixture = _export_fixture(tmp_path, include_unresolved=False)

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (fixture.receipt,), FilesystemArtifactReader()
    )

    assert len(readiness.missing_pools) == 8
    assert readiness.duplicate_pools == ()
    assert readiness.invalid_pools == ()
