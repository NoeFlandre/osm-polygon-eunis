import json
import sqlite3
import struct
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rasterio
from pyproj import CRS
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry
from shapely.wkb import dumps

import osm_polygon_eunis.reference as reference_module
from osm_polygon_eunis.reference import (
    GeoPackageReference,
    RasterLayer,
    RasterReference,
    load_label_table,
    parse_layer_code,
    raster_layer_from_file,
    resolve_eea_layers,
)


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


def test_parse_layer_code_and_label_table(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text('{"R11": "Pannonian and Pontic sandy steppe"}', encoding="utf-8")

    assert parse_layer_code("Prob_R11_100m.tif") == "R11"
    assert load_label_table(labels) == {"R11": "Pannonian and Pontic sandy steppe"}


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


def test_raster_reference_reuses_exact_tile_geometry(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 1], [0, 0]])
    reference = RasterReference((RasterLayer("R11", "steppe", raster, "EEA-test"),))

    with reference:
        first = reference.overlap(box(1, 11, 9, 19))
        second = reference.overlap(box(11, 11, 19, 19))
        assert len(reference._tile_cache) == 1

    assert first.code == second.code == "R11"
    assert len(reference._tile_cache) == 0


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


def test_raster_cell_collection_preserves_exact_intersection_area() -> None:
    cells = reference_module._geometry_collection([box(0, 0, 1, 1), box(1, 0, 2, 1)])

    assert cells is not None
    assert box(0.5, 0.5, 1.5, 1.5).intersection(cells).area == 0.5


def test_reference_metadata_and_raster_lifecycle_fail_closed(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_label_table(labels)
    labels.write_text('{"R11": ""}', encoding="utf-8")
    with pytest.raises(ValueError, match="empty name"):
        load_label_table(labels)
    with pytest.raises(ValueError, match="missing EUNIS"):
        raster_layer_from_file(tmp_path / "Prob_R11_100m.tif", {}, "EEA-test")
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
        reference_module._sql_identifier("")
    with pytest.raises(ValueError, match="identifier"):
        reference_module._sql_identifier("bad\x00name")
    with pytest.raises(ValueError, match="EPSG:3035"):
        reference_module._validate_vector_header("habitats", "geom", 4326)
    with pytest.raises(ValueError, match="metadata"):
        reference_module._validate_vector_header(1, "geom", 3035)
    with pytest.raises(ValueError, match="binary"):
        reference_module._vector_blob("not-binary")
    with pytest.raises(ValueError, match="unknown EUNIS"):
        reference_module._vector_code("R12", {"R11": "steppe"})
    with pytest.raises(ValueError, match="no RTree"):
        reference_module._require_rtree(
            sqlite3.connect(":memory:"), "habitats", "rtree_habitats_geom"
        )
    with pytest.raises(ValueError, match="unexpected shape"):
        reference_module._tile_values(("R11",))
    with pytest.raises(ValueError, match="unsupported"):
        reference_module._envelope_size(5)


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
        reference_module._validate_tile_header("R11", 4326, {"R11": "steppe"})
    with pytest.raises(ValueError, match="unknown EUNIS"):
        reference_module._validate_tile_header("R12", 3035, {"R11": "steppe"})
    with pytest.raises(ValueError, match="invalid matrix"):
        reference_module._validate_tile_types("R11", (1.0,), cast(tuple[int, ...], (1, "bad")))
    with pytest.raises(ValueError, match="invalid dimensions"):
        reference_module._validate_tile_dimensions("R11", 0, 1, 1, 1)

    layer = reference_module._TileLayer("R11", 0, 0, 20, 20, 1, 1, 2, 2, 10.0, 10.0, 0)
    with sqlite3.connect(":memory:") as connection:
        connection.execute(
            "CREATE TABLE R11 (zoom_level INTEGER, tile_column INTEGER, "
            "tile_row INTEGER, tile_data BLOB)"
        )
        connection.execute("INSERT INTO R11 VALUES (0, 0, 0, 'not-binary')")
        with pytest.raises(ValueError, match="binary tile"):
            reference_module.GeoPackageReference._candidate_tile_rows(
                connection, layer, box(1, 1, 19, 19)
            )


def test_resolved_layer_manifest_validation(tmp_path: Path) -> None:
    config = tmp_path / "resolved.json"
    with pytest.raises(ValueError, match="no resolved layers"):
        config.write_text(json.dumps({"source_version": "EEA-test"}), encoding="utf-8")
        resolve_eea_layers(config, tmp_path)
    with pytest.raises(ValueError, match="must be an object"):
        config.write_text(
            json.dumps({"source_version": "EEA-test", "layers": ["bad"]}), encoding="utf-8"
        )
        resolve_eea_layers(config, tmp_path)
    with pytest.raises(FileNotFoundError):
        config.write_text(
            json.dumps(
                {
                    "source_version": "EEA-test",
                    "layers": [{"path": "r.tif", "code": "R11", "name": "steppe"}],
                }
            ),
            encoding="utf-8",
        )
        resolve_eea_layers(config, tmp_path)
    with pytest.raises(ValueError, match="missing path"):
        config.write_text(
            json.dumps(
                {"source_version": "EEA-test", "layers": [{"code": "R11", "name": "steppe"}]}
            ),
            encoding="utf-8",
        )
        resolve_eea_layers(config, tmp_path)

    (tmp_path / "r.tif").write_bytes(b"placeholder")
    config.write_text(
        json.dumps(
            {
                "source_version": "EEA-test",
                "layers": [{"path": "r.tif", "code": "R11", "name": "steppe"}],
            }
        ),
        encoding="utf-8",
    )
    assert resolve_eea_layers(config, tmp_path)[0].code == "R11"
