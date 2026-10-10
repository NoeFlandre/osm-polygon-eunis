import logging
import sqlite3
import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import rasterio
from pyproj import CRS, Transformer
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform as transform_geometry
from shapely.wkb import dumps

import osm_polygon_eunis.geopackage_reference as geopackage_module
import osm_polygon_eunis.geopackage_tiles as tiles_module
import osm_polygon_eunis.raster_reference as raster_module
from osm_polygon_eunis.geopackage_reference import GeoPackageReference
from osm_polygon_eunis.raster_reference import (
    RasterLayer,
    RasterReference,
    _wgs84_dataset_extent,
    parse_layer_code,
)
from osm_polygon_eunis.transform import SidecarUpdateOptions, update_label_sidecar


def _write_geopackage(path: Path, rows: tuple[tuple[str, BaseGeometry], ...]) -> None:
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
            INSERT INTO gpkg_geometry_columns VALUES ('habitats', 'geom', 'POLYGON', 3035, 0, 0);
            CREATE VIRTUAL TABLE rtree_habitats_geom USING rtree(
                id, minx, maxx, miny, maxy
            );
            """
        )
        for index, (code, geometry) in enumerate(rows, start=1):
            connection.execute(
                "INSERT INTO habitats VALUES (?, ?, ?)",
                (index, b"GP" + bytes((0, 1)) + struct.pack("<i", 3035) + dumps(geometry), code),
            )
            min_x, min_y, max_x, max_y = geometry.bounds
            connection.execute(
                "INSERT INTO rtree_habitats_geom VALUES (?, ?, ?, ?, ?)",
                (index, min_x, max_x, min_y, max_y),
            )


def _tile_bytes(values: list[list[int]]) -> bytes:
    with MemoryFile() as memory:
        with memory.open(driver="GTiff", width=2, height=2, count=1, dtype="uint8") as dataset:
            dataset.write(np.asarray(values, dtype="uint8"), 1)
        return memory.read()


def _write_tile_geopackage(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.executescript(
            """
            CREATE TABLE gpkg_contents (
                table_name TEXT PRIMARY KEY,
                data_type TEXT NOT NULL,
                identifier TEXT,
                description TEXT,
                last_change DATETIME NOT NULL,
                min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
                srs_id INTEGER
            );
            CREATE TABLE gpkg_tile_matrix_set (
                table_name TEXT PRIMARY KEY,
                srs_id INTEGER NOT NULL,
                min_x DOUBLE NOT NULL, min_y DOUBLE NOT NULL,
                max_x DOUBLE NOT NULL, max_y DOUBLE NOT NULL
            );
            CREATE TABLE gpkg_tile_matrix (
                table_name TEXT NOT NULL,
                zoom_level INTEGER NOT NULL,
                matrix_width INTEGER NOT NULL, matrix_height INTEGER NOT NULL,
                tile_width INTEGER NOT NULL, tile_height INTEGER NOT NULL,
                pixel_x_size DOUBLE NOT NULL, pixel_y_size DOUBLE NOT NULL,
                PRIMARY KEY (table_name, zoom_level)
            );
            CREATE TABLE R11 (
                id INTEGER PRIMARY KEY,
                zoom_level INTEGER NOT NULL,
                tile_column INTEGER NOT NULL,
                tile_row INTEGER NOT NULL,
                tile_data BLOB NOT NULL
            );
            """
        )
        connection.execute(
            "INSERT INTO gpkg_contents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("R11", "tiles", "R11", "", "2026-01-01", 0, 0, 20, 20, 3035),
        )
        connection.execute(
            "INSERT INTO gpkg_tile_matrix_set VALUES (?, ?, ?, ?, ?, ?)",
            ("R11", 3035, 0, 0, 20, 20),
        )
        connection.execute(
            "INSERT INTO gpkg_tile_matrix VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("R11", 0, 1, 1, 2, 2, 10, 10),
        )
        connection.execute(
            "INSERT INTO R11 VALUES (?, ?, ?, ?, ?)",
            (1, 0, 0, 0, _tile_bytes([[1, 0], [0, 0]])),
        )


def _write_raster(path: Path, values: list[list[int]]) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 20, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(np.asarray(values, dtype="uint8"), 1)
    return path


def test_parse_layer_code() -> None:
    assert parse_layer_code("Prob_R11_100m.tif") == "R11"


def test_parse_layer_code_rejects_non_reference_files() -> None:
    with pytest.raises(ValueError, match="reference layer code"):
        parse_layer_code("README.md")


def test_actual_cell_intersection_drives_percentage(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    reference = RasterReference(
        (RasterLayer("R11", "steppe", raster, "EEA-test"),),
    )

    result = reference.overlap(box(1, 11, 19, 19))

    assert result.code == "R11"
    assert result.overlap_percentage == 50.0


def test_bbox_touch_without_actual_intersection_is_null(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    reference = RasterReference(
        (RasterLayer("R11", "steppe", raster, "EEA-test"),),
    )

    result = reference.overlap(box(10, 10, 20, 20))

    assert result.is_empty


def test_largest_actual_overlap_wins_across_raster_layers(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    second = _write_raster(tmp_path / "Prob_R12_100m.tif", [[0, 1], [0, 0]])
    reference = RasterReference(
        (
            RasterLayer("R11", "first", first, "EEA-test"),
            RasterLayer("R12", "second", second, "EEA-test"),
        ),
    )

    result = reference.overlap(box(1, 11, 18, 19))

    assert result.code == "R11"
    assert result.overlap_percentage == pytest.approx(9 * 8 / (17 * 8) * 100)


def test_window_rounds_outwards_to_cover_polygon_bounds(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [1, 1]])
    reference = RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),))
    with reference as opened:
        _, dataset = opened._datasets[0]
        window = opened._window(dataset, box(4.5, 4.5, 15.5, 15.5))

    assert window is not None
    assert (window.col_off, window.row_off, window.width, window.height) == (0, 0, 2, 2)


def test_overlap_keeps_positive_cell_at_next_tile_boundary(tmp_path: Path) -> None:
    path = tmp_path / "Prob_R11_100m.tif"
    values = np.zeros((4, 128), dtype="uint8")
    values[0, 64] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=128,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 40, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)

    reference = RasterReference((RasterLayer("R11", "steppe", path, "EEA-test"),))
    result = reference.overlap(box(623.3, 33.3, 643.8, 39.0))
    assert result.code == "R11"
    assert result.overlap_percentage is not None
    assert result.overlap_percentage > 0.0


def test_raster_reference_reuses_exact_tile_geometry(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])
    reference = RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),))

    with reference:
        first = reference.overlap(box(1, 11, 9, 19))
        second = reference.overlap(box(11, 11, 19, 19))
        assert len(reference._stack_cache) == 1

    assert first.code == second.code == "R11"
    assert len(reference._stack_cache) == 0


def test_raster_tile_cache_byte_budget_preserves_sidecar_bytes_and_reports_misses(
    tmp_path: Path,
) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])
    polygon = box(1, 11, 19, 19)
    inverse = Transformer.from_crs(3035, 4326, always_xy=True)
    geographic = transform_geometry(inverse.transform, polygon)
    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"geometry": [dumps(geographic), dumps(geographic)]}), source)
    references = (
        RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),)),
        RasterReference(
            (RasterLayer("R11", "steppe", raster, "EEA-test"),),
            tile_cache_bytes=1,
        ),
    )
    sidecars = (tmp_path / "default.parquet", tmp_path / "small.parquet")

    for reference, sidecar in zip(references, sidecars, strict=True):
        with reference:
            update_label_sidecar(source, sidecar, SidecarUpdateOptions(reference=reference))
            assert reference.tile_cache_bytes <= reference.tile_cache_byte_budget
            if reference.tile_cache_byte_budget == 1:
                assert reference.tile_cache_bytes == 0
            else:
                assert reference.tile_cache_bytes > 0

    assert sidecars[0].read_bytes() == sidecars[1].read_bytes()
    assert references[0].tile_cache_hits == 1
    assert references[0].tile_cache_misses == 1
    assert references[0].tile_cache_miss_rate == 0.5
    assert references[1].tile_cache_hits == 0
    assert references[1].tile_cache_misses == 2
    assert references[1].tile_cache_miss_rate == 1.0


def test_raster_tile_cache_evicts_oldest_tile_at_its_byte_budget(tmp_path: Path) -> None:
    path = tmp_path / "Prob_R11_100m.tif"
    values = np.ones((4, 256), dtype="uint8")
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=256,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 40, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)
    tile_bytes = 128 * 128 + 256
    reference = RasterReference(
        (RasterLayer("R11", "steppe", path, "EEA-test"),), tile_cache_bytes=tile_bytes
    )

    with reference:
        datasets = reference._datasets
        reference._tile_stack(datasets, 0, 0)
        reference._tile_stack(datasets, 0, 1)

        assert list(reference._stack_cache) == [(0, 1)]
        assert reference.tile_cache_bytes == tile_bytes


def test_raster_overlap_returns_no_result_for_invalid_and_outside_polygons(
    tmp_path: Path,
) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])
    reference = RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),))

    invalid = reference.overlap(None)
    with reference:
        outside = reference.overlap(box(100, 100, 110, 110))
        touching = reference.overlap(box(20, 5, 25, 10))

    assert invalid.code is None
    assert outside.code is None
    assert touching.code is None


def test_raster_reference_rejects_a_nonpositive_tile_cache_budget(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])

    with pytest.raises(ValueError, match="byte budget must be positive"):
        RasterReference(
            (RasterLayer("R11", "steppe", raster, "EEA-test"),),
            tile_cache_bytes=0,
        )


def test_reference_rejects_mixed_source_versions(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    second = _write_raster(tmp_path / "Prob_R12_100m.tif", [[0, 1], [0, 0]])

    with pytest.raises(ValueError, match="source version"):
        RasterReference(
            (
                RasterLayer("R11", "first", first, "one"),
                RasterLayer("R12", "second", second, "two"),
            ),
        )


def test_raster_reference_exposes_cached_wgs84_extent_while_open(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    second = _write_raster(tmp_path / "Prob_R12_100m.tif", [[0, 1], [0, 0]])
    reference = RasterReference(
        (
            RasterLayer("R11", "first", first, "EEA-test"),
            RasterLayer("R12", "second", second, "EEA-test"),
        )
    )

    assert reference.source_extent_wgs84 is None
    with reference:
        extent = reference.source_extent_wgs84
        assert extent is not None
        assert extent[0] < extent[2]
        assert extent[1] < extent[3]
        assert reference.source_extent_wgs84 == extent

    assert reference.source_extent_wgs84 is None


def test_wgs84_dataset_extent_combines_bounds_before_projection() -> None:
    first_bounds = SimpleNamespace(left=10.0, bottom=20.0, right=30.0, top=40.0)
    second_bounds = SimpleNamespace(left=-5.0, bottom=15.0, right=25.0, top=35.0)
    first = cast(rasterio.DatasetReader, SimpleNamespace(bounds=first_bounds))
    second = cast(rasterio.DatasetReader, SimpleNamespace(bounds=second_bounds))
    datasets = (
        (RasterLayer("R11", "first", Path("first.tif"), "EEA-test"), first),
        (RasterLayer("R12", "second", Path("second.tif"), "EEA-test"), second),
    )

    extent = _wgs84_dataset_extent(datasets)

    assert extent == raster_module.wgs84_envelope((-5.0, 15.0, 30.0, 40.0))


def test_wgs84_dataset_extent_returns_none_for_no_datasets() -> None:
    assert _wgs84_dataset_extent(()) is None


def test_wgs84_dataset_extent_projects_one_dataset() -> None:
    bounds = SimpleNamespace(left=10.0, bottom=20.0, right=30.0, top=40.0)
    dataset = cast(rasterio.DatasetReader, SimpleNamespace(bounds=bounds))
    item = (RasterLayer("R11", "first", Path("first.tif"), "EEA-test"), dataset)

    assert _wgs84_dataset_extent((item,)) == raster_module.wgs84_envelope((10.0, 20.0, 30.0, 40.0))


def test_raster_accepts_catalog_equivalent_epsg_3035_wkt() -> None:
    class Dataset:
        crs = CRS.from_epsg(3035).to_wkt()

    RasterReference._validate_crs(cast(rasterio.DatasetReader, Dataset()), Path("official.tif"))


def test_geopackage_reference_uses_rtree_as_candidate_filter(tmp_path: Path) -> None:
    database = tmp_path / "habitats.gpkg"
    reference = GeoPackageReference(
        database,
        {"Q11": "Raised bog"},
        source_version="EEA-test",
    )
    _write_geopackage(
        database,
        (("Q11", box(0, 0, 10, 10)), ("Q11", box(20, 20, 30, 30))),
    )

    with reference:
        result = reference.overlap(box(1, 1, 9, 9))

    assert result.code == "Q11"
    assert result.overlap_percentage == 100.0


def test_geopackage_bbox_touch_is_not_an_intersection(tmp_path: Path) -> None:
    database = tmp_path / "habitats.gpkg"
    reference = GeoPackageReference(database, {"Q11": "Raised bog"}, source_version="EEA-test")
    _write_geopackage(database, (("Q11", box(0, 0, 10, 10)),))

    with reference:
        result = reference.overlap(box(10, 10, 20, 20))

    assert result.is_empty


def test_geopackage_repairs_invalid_reference_geometry(tmp_path: Path) -> None:
    database = tmp_path / "habitats.gpkg"
    bowtie = Polygon([(0, 0), (10, 0), (0, 10), (10, 10)])
    _write_geopackage(database, (("R11", bowtie),))
    reference = GeoPackageReference(database, {"R11": "steppe"}, source_version="EEA-test")

    with reference:
        result = reference.overlap(box(1, 0.1, 9, 1))

    assert result.code == "R11"
    assert result.overlap_percentage == 100.0
    assert reference.intersection_errors == 0


def test_geopackage_counts_and_logs_unrepairable_reference_geometry(
    tmp_path: Path,
    caplog,
) -> None:
    database = tmp_path / "habitats.gpkg"
    collapsed = Polygon([(0, 0), (5, 5), (10, 10), (0, 0)])
    assert not collapsed.is_valid
    _write_geopackage(database, (("R11", collapsed),))
    reference = GeoPackageReference(database, {"R11": "steppe"}, source_version="EEA-test")

    with caplog.at_level(logging.WARNING, logger=geopackage_module.__name__), reference:
        result = reference.overlap(box(1, 1, 9, 9))

    assert result.is_empty
    assert reference.intersection_errors == 1
    assert "unrepairable EUNIS reference geometry for R11" in caplog.text


def test_geopackage_geometry_header_round_trip() -> None:
    geometry = box(1, 2, 3, 4)
    blob = b"GP" + bytes((0, 1)) + struct.pack("<i", 3035) + dumps(geometry)

    assert GeoPackageReference.decode_geometry(blob).equals(geometry)


def test_geopackage_tile_reference_uses_exact_positive_pixels(tmp_path: Path) -> None:
    database = tmp_path / "tiles.gpkg"
    _write_tile_geopackage(database)
    reference = GeoPackageReference(database, {"R11": "steppe"}, source_version="EEA-test")

    result = reference.overlap(box(1, 11, 19, 19))

    assert result.code == "R11"
    assert result.overlap_percentage == 50.0


def test_geopackage_tile_lookup_returns_nothing_for_a_missing_tile() -> None:
    layer = tiles_module.TileLayer(
        "R11",
        0.0,
        0.0,
        20.0,
        20.0,
        1,
        1,
        2,
        2,
        10.0,
        10.0,
        0,
    )

    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE R11 (zoom_level INTEGER, tile_column INTEGER, "
            "tile_row INTEGER, tile_data BLOB)"
        )
        assert GeoPackageReference._tile_blob(connection, layer, 5, 5) is None


def test_reference_metadata_and_raster_lifecycle_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="non-negative"):
        RasterReference((), threshold=-1)

    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[0, 0], [0, 0]])
    reference = RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),))
    assert reference.overlap(box(1, 11, 19, 19)).is_empty
    reference.close()
    with reference, pytest.raises(RuntimeError, match="already open"):
        reference.__enter__()
    reference.close()

    invalid = _write_raster(tmp_path / "invalid-crs.tif", [[1, 0], [0, 0]])
    with rasterio.open(invalid, "r+") as dataset:
        dataset.crs = "EPSG:4326"
    with pytest.raises(ValueError, match="not EPSG:3035"):
        RasterReference((RasterLayer("R11", "steppe", invalid, "EEA-test"),)).__enter__()

    class BrokenCrs:
        def to_epsg(self):
            raise TypeError("broken CRS")

    class BrokenDataset:
        crs = BrokenCrs()

    with pytest.raises(ValueError, match="not EPSG:3035"):
        RasterReference._validate_crs(cast(rasterio.DatasetReader, BrokenDataset()), Path("broken"))


def test_geopackage_validation_and_geometry_headers() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        GeoPackageReference(Path("missing.gpkg"), {}, source_version="EEA-test")
    with pytest.raises(ValueError, match="non-negative"):
        GeoPackageReference(
            Path("missing.gpkg"), {"R11": "steppe"}, source_version="EEA-test", threshold=-1
        )
    with pytest.raises(ValueError, match="invalid GeoPackage"):
        GeoPackageReference.decode_geometry(b"bad")
    with pytest.raises(ValueError, match="EPSG:3035"):
        GeoPackageReference.decode_geometry(b"GP" + bytes((1, 1)) + struct.pack("<i", 4326))
    with pytest.raises(ValueError, match="envelope"):
        GeoPackageReference.decode_geometry(b"GP" + bytes((1, 15)) + struct.pack("<i", 3035) + b"")
    with pytest.raises(ValueError, match="identifier"):
        geopackage_module.sql_identifier("")
    with pytest.raises(ValueError, match="identifier"):
        geopackage_module.sql_identifier("bad\x00name")
    with pytest.raises(ValueError, match="EPSG:3035"):
        geopackage_module._validate_vector_header("habitats", "geom", 4326)
    with pytest.raises(ValueError, match="metadata"):
        geopackage_module._validate_vector_header(1, "geom", 3035)
    with pytest.raises(ValueError, match="binary"):
        geopackage_module._vector_blob("not-binary")
    with pytest.raises(ValueError, match="unknown EUNIS"):
        geopackage_module._vector_code("R12", {"R11": "steppe"})
    with pytest.raises(ValueError, match="no RTree"):
        geopackage_module._require_rtree(
            sqlite3.connect(":memory:"), "habitats", "rtree_habitats_geom"
        )
    with pytest.raises(ValueError, match="unexpected shape"):
        tiles_module._tile_values(("R11",))
    with pytest.raises(ValueError, match="unsupported"):
        geopackage_module._envelope_size(5)


def test_geopackage_empty_and_vector_metadata_errors(tmp_path: Path) -> None:
    empty = tmp_path / "empty.gpkg"
    sqlite3.connect(empty).close()
    reference = GeoPackageReference(empty, {"R11": "steppe"}, source_version="EEA-test")
    with pytest.raises(ValueError, match="no EPSG"):
        reference.__enter__()
    reference.close()

    connection = sqlite3.connect(":memory:")
    assert GeoPackageReference._geometry_rows(connection) == []
    assert GeoPackageReference._tile_rows(connection) == []
    with pytest.raises(ValueError, match="geometry table is missing"):
        GeoPackageReference._vector_columns(connection, "missing")
    with pytest.raises(ValueError, match="no EUNIS code"):
        GeoPackageReference._required_code_column(
            [(0, "fid", "INTEGER", 0, None, 1), (1, "geom", "BLOB", 0, None, 0)],
            "habitats",
        )
    connection.close()

    database = tmp_path / "vector-invalid.gpkg"
    _write_geopackage(database, (("Q11", box(0, 0, 1, 1)),))
    invalid_reference = GeoPackageReference(database, {"R11": "steppe"}, source_version="EEA-test")
    with pytest.raises(ValueError, match="unknown EUNIS"):
        invalid_reference.overlap(box(0, 0, 1, 1))


def test_geopackage_tile_cache_and_metadata_guards(tmp_path: Path) -> None:
    database = tmp_path / "tiles.gpkg"
    _write_tile_geopackage(database)
    reference = GeoPackageReference(database, {"R11": "steppe"}, source_version="EEA-test")
    with reference:
        first = reference.overlap(box(1, 11, 19, 19))
        second = reference.overlap(box(1, 11, 19, 19))
        outside = reference.overlap(box(30, 30, 40, 40))
    assert first.code == second.code == "R11"
    assert outside.is_empty

    with pytest.raises(ValueError, match="not EPSG"):
        tiles_module._validate_tile_header("R11", 4326, {"R11": "steppe"})
    with pytest.raises(ValueError, match="unknown EUNIS"):
        tiles_module._validate_tile_header("R12", 3035, {"R11": "steppe"})
    with pytest.raises(ValueError, match="invalid matrix"):
        tiles_module._validate_tile_types("R11", (1.0,), cast(tuple[int, ...], (1, "bad")))
    with pytest.raises(ValueError, match="invalid dimensions"):
        tiles_module._validate_tile_dimensions("R11", 0, 1, 1, 1)

    layer = tiles_module.TileLayer("R11", 0, 0, 20, 20, 1, 1, 2, 2, 10.0, 10.0, 0)
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE R11 (zoom_level INTEGER, tile_column INTEGER, "
            "tile_row INTEGER, tile_data BLOB)"
        )
        connection.execute("INSERT INTO R11 VALUES (0, 0, 0, 'not-binary')")
        with pytest.raises(ValueError, match="binary tile"):
            GeoPackageReference._tile_blob(connection, layer, 0, 0)


def test_geopackage_tile_discovery_fails_closed_on_corrupt_database(tmp_path: Path) -> None:
    corrupt = tmp_path / "corrupt.gpkg"
    corrupt.write_bytes(b"SQLite format 3\x00" + b"\xff" * 4096)
    connection = sqlite3.connect(corrupt)
    try:
        with pytest.raises(ValueError, match="not a readable GeoPackage"):
            GeoPackageReference._tile_rows(connection)
    finally:
        connection.close()


def test_geopackage_tile_discovery_fails_closed_on_broken_tile_schema() -> None:
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE gpkg_contents (table_name TEXT)")
    connection.execute("CREATE TABLE gpkg_tile_matrix_set (table_name TEXT)")
    connection.execute("CREATE TABLE gpkg_tile_matrix (table_name TEXT)")
    with pytest.raises(ValueError, match="not a readable GeoPackage"):
        GeoPackageReference._tile_rows(connection)


def test_references_count_intersection_errors(tmp_path: Path, monkeypatch) -> None:
    from osm_polygon_eunis import matching

    def broken(*_args):
        raise RuntimeError("GEOS TopologyException")

    database = tmp_path / "habitats.gpkg"
    _write_geopackage(database, (("Q11", box(0, 0, 10, 10)),))
    vector = GeoPackageReference(database, {"Q11": "Raised bog"}, source_version="EEA-test")
    raster = RasterReference(
        (
            RasterLayer(
                "R11",
                "steppe",
                _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]]),
                "EEA-test",
            ),
        ),
    )
    assert vector.intersection_errors == raster.intersection_errors == 0
    assert vector.overlap(box(1, 1, 9, 9)).code == "Q11"
    assert vector.intersection_errors == 0

    monkeypatch.setattr(matching, "_exact_intersection_area", broken)
    monkeypatch.setattr(raster_module, "weighted_cells", broken)
    assert vector.overlap(box(1, 1, 9, 9)).code is None
    assert vector.overlap(box(1, 1, 9, 9)).code is None
    assert raster.overlap(box(1, 11, 19, 19)).code is None
    assert vector.intersection_errors == 2
    assert raster.intersection_errors == 1


def test_raster_window_error_fallback_is_logged(monkeypatch, caplog) -> None:
    def broken(*_args, **_kwargs):
        raise raster_module.WindowError("invalid bounds")

    monkeypatch.setattr(raster_module, "from_bounds", broken)
    dataset = SimpleNamespace(transform=from_origin(0, 10, 1, 1), width=10, height=10)

    with caplog.at_level(logging.WARNING, logger=raster_module.__name__):
        assert (
            RasterReference._window(cast(rasterio.DatasetReader, dataset), box(0, 0, 1, 1)) is None
        )

    assert any(
        record.levelno == logging.WARNING and "window" in record.getMessage().lower()
        for record in caplog.records
    )


def test_raster_reference_rejects_layers_on_different_grids(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])
    second = tmp_path / "Prob_R12_100m.tif"
    with rasterio.open(
        second,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(10, 20, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(np.ones((2, 2), dtype="uint8"), 1)
    reference = RasterReference(
        (
            RasterLayer("R11", "steppe", first, "EEA-test"),
            RasterLayer("R12", "heath", second, "EEA-test"),
        )
    )

    with reference, pytest.raises(ValueError, match="share one square-cell grid"):
        reference.overlap(box(1, 11, 9, 19))


def test_raster_reference_picks_the_lower_code_on_equal_overlap(tmp_path: Path) -> None:
    layers = tuple(
        RasterLayer(
            code, code.lower(), _write_raster(tmp_path / f"Prob_{code}_100m.tif", [[1]]), "v"
        )
        for code in ("R12", "R11")
    )

    with RasterReference(layers) as reference:
        assert reference.overlap(box(1, 1, 9, 9)).code == "R11"


def _tiny_raster(directory: Path, code: str) -> Path:
    return _write_raster(directory / f"Prob_{code}_100m.tif", [[1, 0], [0, 0]])


def test_square_grid_detection_rejects_rotated_and_stretched_transforms() -> None:
    from rasterio.transform import Affine

    from osm_polygon_eunis.grid_overlap import is_square_north_up as _is_square_north_up

    assert _is_square_north_up(Affine(10, 0, 0, 0, -10, 0))
    assert not _is_square_north_up(Affine(10, 0, 0, 0, -20, 0))
    assert not _is_square_north_up(Affine(10, 1, 0, 0, -10, 0))
    assert not _is_square_north_up(Affine(10, 0, 0, 1, -10, 0))


def test_raster_overlap_is_100_percent_for_an_all_100_grid_with_a_touching_hole(
    tmp_path: Path,
) -> None:
    path = tmp_path / "Prob_R11_5m.tif"
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=8,
        height=8,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 40, 5, 5),
        nodata=0,
    ) as dataset:
        dataset.write(np.full((8, 8), 100, dtype="uint8"), 1)
    polygon = Polygon(
        [(0, 0), (40, 0), (40, 40), (0, 40)],
        [[(10, 10), (21, 21), (22, 15), (24, 40)]],
    )
    assert polygon.is_valid
    reference = RasterReference((RasterLayer("R11", "Steppe", path, "EEA-test"),))

    with reference:
        result = reference.overlap(polygon)

    assert result.code == "R11"
    assert result.overlap_percentage == pytest.approx(100.0)
