"""Opt-in synthetic raster, sidecar, and GeoPackage performance benchmarks."""

from __future__ import annotations

import hashlib
import json
import os
import sys

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows.
    resource = None

import sqlite3
import struct
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import Point
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry

from osm_polygon_eunis import transform as transform_module
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.geopackage_reference import GeoPackageReference
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference

_SEED = 20260926
_RASTER_LAYERS = 10
_FULL_RASTER_SIZE = 2000
_FULL_POLYGON_COUNT = 1000
_PIXEL_SIZE_METRES = 100.0
_SOURCE_VERSION = "EEA-synthetic-benchmark"
_EUNIS_LABELS = {"R11": "Steppe", "R12": "Dry grasslands"}


@dataclass(frozen=True, slots=True)
class SyntheticCase:
    layers: tuple[RasterLayer, ...]
    polygons_wgs84: tuple[BaseGeometry, ...]
    polygons_projected: tuple[BaseGeometry, ...]
    positions: tuple[tuple[float, float], ...]
    source_parquet: Path
    geopackage: Path


def _scale() -> float:
    value = float(os.environ.get("EUNIS_BENCHMARK_SCALE", "1"))
    if not 0 < value <= 1:
        raise ValueError("EUNIS_BENCHMARK_SCALE must be greater than 0 and no more than 1")
    return value


def _write_geopackage(path: Path, geometries: tuple[BaseGeometry, ...]) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE gpkg_geometry_columns (
                table_name TEXT PRIMARY KEY,
                column_name TEXT NOT NULL,
                geometry_type_name TEXT NOT NULL,
                srs_id INTEGER NOT NULL,
                z TINYINT NOT NULL,
                m TINYINT NOT NULL
            );
            CREATE TABLE habitats (
                fid INTEGER PRIMARY KEY,
                geom BLOB NOT NULL,
                code TEXT NOT NULL
            );
            INSERT INTO gpkg_geometry_columns VALUES
                ('habitats', 'geom', 'POLYGON', 3035, 0, 0);
            CREATE VIRTUAL TABLE rtree_habitats_geom USING rtree(
                id, minx, maxx, miny, maxy
            );
            """
        )
        for row_id, geometry in enumerate(geometries, start=1):
            code = "R11" if row_id % 2 else "R12"
            blob = b"GP" + bytes((0, 1)) + struct.pack("<i", 3035) + geometry.wkb
            connection.execute(
                "INSERT INTO habitats VALUES (?, ?, ?)",
                (row_id, blob, code),
            )
            min_x, min_y, max_x, max_y = geometry.bounds
            connection.execute(
                "INSERT INTO rtree_habitats_geom VALUES (?, ?, ?, ?, ?)",
                (row_id, min_x, max_x, min_y, max_y),
            )


@pytest.fixture(scope="module")
def synthetic_case(tmp_path_factory: pytest.TempPathFactory) -> SyntheticCase:
    scale = _scale()
    size = max(40, int(_FULL_RASTER_SIZE * scale) // 10 * 10)
    polygon_count = max(25, int(_FULL_POLYGON_COUNT * scale))
    root = tmp_path_factory.mktemp("eunis-synthetic-benchmark")
    rng = np.random.default_rng(_SEED)
    pattern_size = size // 10
    layers: list[RasterLayer] = []
    projected_extent = (3_900_000.0, 2_900_000.0, 4_100_000.0, 3_100_000.0)
    xmin, ymin, xmax, ymax = projected_extent
    pixel_size = (xmax - xmin) / size
    to_wgs84 = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)

    for layer_index in range(_RASTER_LAYERS):
        values = rng.integers(0, 2, (pattern_size, pattern_size), dtype=np.uint8)
        values = values.repeat(10, axis=0).repeat(10, axis=1)
        path = root / f"Prob_R{layer_index + 10:02d}_100m.tif"
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            width=size,
            height=size,
            count=1,
            dtype="uint8",
            crs="EPSG:3035",
            transform=from_origin(xmin, ymax, pixel_size, pixel_size),
            nodata=0,
            tiled=True,
            blockxsize=64,
            blockysize=64,
            compress="deflate",
        ) as dataset:
            dataset.write(values, 1)
        layers.append(RasterLayer(f"R{layer_index + 10:02d}", "synthetic", path, _SOURCE_VERSION))

    margin = 1_500.0 * scale + 100.0
    positions = tuple(
        (
            float(rng.uniform(xmin + margin, xmax - margin)),
            float(rng.uniform(ymin + margin, ymax - margin)),
        )
        for _ in range(polygon_count)
    )
    radius_min = 50.0 * scale
    radius_max = 1_500.0 * scale
    polygons_projected = tuple(
        Point(x, y).buffer(float(rng.uniform(radius_min, radius_max)), quad_segs=8)
        for x, y in positions
    )
    polygons = tuple(
        transform_geometry(to_wgs84.transform, polygon) for polygon in polygons_projected
    )

    source_parquet = root / "synthetic-shard.parquet"
    pq.write_table(pa.table({"geometry": [polygon.wkb for polygon in polygons]}), source_parquet)
    geopackage = root / "synthetic-reference.gpkg"
    _write_geopackage(geopackage, polygons_projected)
    return SyntheticCase(
        tuple(layers),
        polygons,
        polygons_projected,
        positions,
        source_parquet,
        geopackage,
    )


@pytest.fixture(scope="module")
def benchmark_result(request: pytest.FixtureRequest) -> dict[str, object]:
    result: dict[str, object] = {
        "parameters": {
            "seed": _SEED,
            "scale": _scale(),
            "layers": _RASTER_LAYERS,
            "raster_size": max(40, int(_FULL_RASTER_SIZE * _scale()) // 10 * 10),
            "polygon_count": max(25, int(_FULL_POLYGON_COUNT * _scale())),
        },
        "metrics": {},
        "invariants": {},
    }

    def write_result() -> None:
        destination = os.environ.get("EUNIS_BENCHMARK_RESULT")
        if destination:
            metrics = result["metrics"]
            assert isinstance(metrics, dict)
            metrics["peak_rss_bytes"] = _process_peak_rss_bytes()
            path = Path(destination)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")

    request.addfinalizer(write_result)
    return result


def _signature(result: EunisResult) -> tuple[object, ...]:
    return (
        result.code,
        result.name,
        result.overlap_percentage,
        result.source_version,
    )


def _peak_rss_bytes_from_platform_units(value: float, *, platform: str) -> int:
    """Normalize ``ru_maxrss`` to bytes on macOS and Linux benchmark hosts."""

    return int(value if platform == "darwin" else value * 1024)


def _process_peak_rss_bytes() -> int:
    """Read this benchmark process's high-water RSS in bytes."""

    if resource is None:
        raise RuntimeError("peak RSS measurements require the Unix resource module")
    return _peak_rss_bytes_from_platform_units(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        platform=sys.platform,
    )


def _update_sidecar_compatibly(
    module: Any,
    source: Path,
    destination: Path,
    reference: Any,
    batch_size: int,
) -> Any:
    """Call the sidecar updater across the pre-options and options APIs."""

    options_factory = getattr(module, "SidecarUpdateOptions", None)
    if options_factory is None:
        return module.update_label_sidecar(
            source,
            destination,
            reference=reference,
            batch_size=batch_size,
        )
    options = options_factory(reference=reference, batch_size=batch_size)
    return module.update_label_sidecar(source, destination, options)


def _measure_raster_order(
    case: SyntheticCase,
    order: tuple[int, ...],
) -> tuple[float, tuple[tuple[object, ...], ...], float | None, int | None]:
    reference = RasterReference(case.layers)
    results: list[tuple[object, ...] | None] = [None] * len(order)
    started = time.perf_counter()
    with reference:
        for index in order:
            results[index] = _signature(reference.overlap(case.polygons_projected[index]))
    seconds = time.perf_counter() - started
    miss_rate = getattr(reference, "tile_cache_miss_rate", None)
    cache_budget = getattr(reference, "tile_cache_byte_budget", None)
    return (
        seconds,
        tuple(result for result in results if result is not None),
        miss_rate,
        cache_budget,
    )


def _timed_repeats(function: Callable[[], object], *, repeats: int, iterations: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        for _ in range(iterations):
            function()
        samples.append((time.perf_counter() - started) / iterations)
    return float(np.median(samples))


def test_raster_overlap_random_and_spatial_order_benchmark(
    synthetic_case: SyntheticCase,
    benchmark_result: dict[str, object],
) -> None:
    case = synthetic_case
    count = len(case.polygons_projected)
    random_order = tuple(range(count))
    spatial_order = tuple(sorted(random_order, key=lambda index: case.positions[index]))
    random_seconds, random_results, random_miss_rate, cache_budget = _measure_raster_order(
        case,
        random_order,
    )
    spatial_seconds, spatial_results, spatial_miss_rate, _ = _measure_raster_order(
        case,
        spatial_order,
    )

    assert random_results == spatial_results
    assert any(result[0] is not None for result in random_results)
    metrics = benchmark_result["metrics"]
    assert isinstance(metrics, dict)
    metrics["raster_random_seconds"] = random_seconds
    metrics["raster_spatial_seconds"] = spatial_seconds
    metrics["raster_random_cache_miss_rate"] = random_miss_rate
    metrics["raster_spatial_cache_miss_rate"] = spatial_miss_rate
    metrics["raster_tile_cache_budget_bytes"] = cache_budget
    invariants = benchmark_result["invariants"]
    assert isinstance(invariants, dict)
    invariants["raster_order_independent"] = True


def test_update_label_sidecar_benchmark(
    synthetic_case: SyntheticCase,
    benchmark_result: dict[str, object],
    tmp_path: Path,
) -> None:
    destination = tmp_path / "synthetic-labels.parquet"
    reference = RasterReference(synthetic_case.layers)
    started = time.perf_counter()
    with reference:
        rows = _update_sidecar_compatibly(
            transform_module,
            synthetic_case.source_parquet,
            destination,
            reference,
            batch_size=256,
        )
    seconds = time.perf_counter() - started
    assert rows == len(synthetic_case.polygons_wgs84)
    metrics = benchmark_result["metrics"]
    invariants = benchmark_result["invariants"]
    assert isinstance(metrics, dict) and isinstance(invariants, dict)
    metrics["sidecar_update_seconds"] = seconds
    metrics["sidecar_cache_miss_rate"] = getattr(reference, "tile_cache_miss_rate", None)
    invariants["sidecar_rows_match_input"] = True
    invariants["sidecar_sha256"] = hashlib.sha256(destination.read_bytes()).hexdigest()


def test_geopackage_overlap_benchmark(
    synthetic_case: SyntheticCase,
    benchmark_result: dict[str, object],
) -> None:
    reference = GeoPackageReference(
        synthetic_case.geopackage,
        _EUNIS_LABELS,
        source_version=_SOURCE_VERSION,
    )
    started = time.perf_counter()
    with reference:
        results = tuple(
            _signature(reference.overlap(polygon)) for polygon in synthetic_case.polygons_projected
        )
    seconds = time.perf_counter() - started
    assert len(results) == len(synthetic_case.polygons_projected)
    assert any(result[0] is not None for result in results)
    metrics = benchmark_result["metrics"]
    invariants = benchmark_result["invariants"]
    assert isinstance(metrics, dict) and isinstance(invariants, dict)
    metrics["geopackage_overlap_seconds"] = seconds
    invariants["geopackage_result_count"] = len(results)
