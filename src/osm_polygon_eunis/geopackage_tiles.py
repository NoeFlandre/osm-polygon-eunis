"""Tile-matrix discovery and exact pixel decoding for GeoPackage readers."""

from __future__ import annotations

import math
import sqlite3
import warnings
from collections import OrderedDict
from dataclasses import dataclass
from typing import cast

import numpy as np
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from .domain import EPSG_LAEA_EUROPE, SchemaError
from .geopackage_sql import sql_identifier
from .grid_overlap import WeightedCells, band_row_ranges, weighted_cells

_GEOPACKAGE_TILE_CACHE_SIZE = 4096
_ALPHA_BAND = 4
_TILE_METADATA_FIELDS = 13

_TileMetadata = tuple[str, int, float, float, float, float, int, int, int, int, float, float, int]


@dataclass(frozen=True, slots=True)
class TileLayer:
    """Tile-matrix geometry of one GeoPackage tile layer."""

    table: str
    min_x: float
    min_y: float
    max_x: float
    max_y: float
    matrix_width: int
    matrix_height: int
    tile_width: int
    tile_height: int
    pixel_x_size: float
    pixel_y_size: float
    zoom_level: int


def _tile_values(row: tuple[object, ...]) -> _TileMetadata:
    if len(row) != _TILE_METADATA_FIELDS:
        raise ValueError("GeoPackage tile metadata has an unexpected shape")
    return cast(_TileMetadata, row)


def _validate_tile_header(
    table: str,
    content_srs_id: int,
    labels: dict[str, str],
) -> None:
    if content_srs_id != EPSG_LAEA_EUROPE:
        raise ValueError(f"GeoPackage tile layer {table} is not EPSG:3035")
    if table not in labels:
        raise ValueError(f"GeoPackage tile layer has unknown EUNIS code {table!r}")


def _validate_tile_types(
    table: str,
    numeric: tuple[float, ...],
    dimensions: tuple[int, ...],
) -> None:
    if not all(isinstance(value, (int, float)) for value in numeric) or not all(
        isinstance(value, int) for value in dimensions
    ):
        raise ValueError(f"GeoPackage tile layer {table} has invalid matrix metadata")


def _validate_tile_dimensions(
    table: str,
    pixel_x_size: float,
    pixel_y_size: float,
    matrix_width: int,
    matrix_height: int,
) -> None:
    if any(value <= 0 for value in (pixel_x_size, pixel_y_size, matrix_width, matrix_height)):
        raise ValueError(f"GeoPackage tile layer {table} has invalid dimensions")


class GeoPackageTileMethods:
    """Tile-cache lookups shared by GeoPackage reference readers."""

    _labels: dict[str, str]
    _threshold: int
    _tile_layers: tuple[TileLayer, ...]
    _tile_cache: OrderedDict[tuple[str, int, int], np.ndarray | None]

    def _discover_tile_layers(self, connection: sqlite3.Connection) -> tuple[TileLayer, ...]:
        rows = self._tile_rows(connection)
        return tuple(self._tile_layer(row) for row in rows)

    @staticmethod
    def _tile_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
        try:
            return connection.execute(
                """
                SELECT c.table_name, c.srs_id,
                       s.min_x, s.min_y, s.max_x, s.max_y,
                       m.matrix_width, m.matrix_height,
                       m.tile_width, m.tile_height,
                       m.pixel_x_size, m.pixel_y_size, m.zoom_level
                FROM gpkg_contents AS c
                JOIN gpkg_tile_matrix_set AS s ON s.table_name = c.table_name
                JOIN gpkg_tile_matrix AS m ON m.table_name = c.table_name
                WHERE c.data_type = 'tiles'
                  AND m.zoom_level = (
                      SELECT max(m2.zoom_level)
                      FROM gpkg_tile_matrix AS m2
                      WHERE m2.table_name = c.table_name
                  )
                ORDER BY c.table_name
                """
            ).fetchall()
        except sqlite3.DatabaseError as error:
            # Tile tables are optional (vector-only GeoPackage); any other database
            # error (corrupt or truncated file) must fail closed, as _geometry_rows does.
            if "no such table" in str(error):
                return []
            raise ValueError("not a readable GeoPackage") from error

    def _tile_layer(self, row: tuple[object, ...]) -> TileLayer:
        values = _tile_values(row)
        table = values[0]
        self._validate_tile_values(values)
        return TileLayer(
            table,
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
            values[6],
            values[7],
            values[8],
            values[9],
            float(values[10]),
            float(values[11]),
            values[12],
        )

    def _validate_tile_values(self, values: _TileMetadata) -> None:
        (
            table,
            content_srs_id,
            min_x,
            min_y,
            max_x,
            max_y,
            matrix_width,
            matrix_height,
            tile_width,
            tile_height,
            pixel_x_size,
            pixel_y_size,
            zoom_level,
        ) = values
        _validate_tile_header(table, content_srs_id, self._labels)
        numeric = (min_x, min_y, max_x, max_y, pixel_x_size, pixel_y_size)
        dimensions = (matrix_width, matrix_height, tile_width, tile_height, zoom_level)
        _validate_tile_types(table, numeric, dimensions)
        _validate_tile_dimensions(table, pixel_x_size, pixel_y_size, matrix_width, matrix_height)

    def _tile_layer_area(
        self,
        connection: sqlite3.Connection,
        layer: TileLayer,
        polygon: BaseGeometry,
        bands: dict[tuple[object, ...], list[WeightedCells]],
    ) -> float:
        """Return the exact polygon area covered by positive pixels of one tile layer.

        ``bands`` memoizes the polygon's weighted cells per grid, so layers that
        share a tile matrix rasterize the polygon only once.
        """

        if not polygon.intersects(box(layer.min_x, layer.min_y, layer.max_x, layer.max_y)):
            return 0.0
        pixel_x, pixel_y = layer.pixel_x_size, layer.pixel_y_size
        min_x, min_y, max_x, max_y = polygon.bounds
        cols = (
            max(0, math.floor((min_x - layer.min_x) / pixel_x)),
            min(layer.matrix_width * layer.tile_width, math.ceil((max_x - layer.min_x) / pixel_x)),
        )
        rows = (
            max(0, math.floor((layer.max_y - max_y) / pixel_y)),
            min(
                layer.matrix_height * layer.tile_height, math.ceil((layer.max_y - min_y) / pixel_y)
            ),
        )
        grid = (layer.min_x, layer.max_y, pixel_x, pixel_y, cols, rows)
        if grid not in bands:
            transform = from_origin(layer.min_x, layer.max_y, pixel_x, pixel_y)
            bands[grid] = [
                weighted_cells(polygon, transform, band, cols)
                for band in band_row_ranges(rows[0], rows[1], cols[0], cols[1])
            ]
        return sum(self._valid_cell_area(connection, layer, cells) for cells in bands[grid])

    def _valid_cell_area(
        self,
        connection: sqlite3.Connection,
        layer: TileLayer,
        cells: WeightedCells,
    ) -> float:
        if not len(cells.rows):
            return 0.0
        tile_columns = cells.cols // layer.tile_width
        tile_rows = cells.rows // layer.tile_height
        tile_ids = tile_rows * 1_000_003 + tile_columns
        order = np.argsort(tile_ids, kind="stable")
        cuts = np.flatnonzero(np.diff(tile_ids[order])) + 1
        total = 0.0
        for group in np.split(order, cuts):
            column, row = int(tile_columns[group[0]]), int(tile_rows[group[0]])
            valid = self._tile_mask(connection, layer, column, row)
            if valid is not None:
                local_rows = cells.rows[group] % layer.tile_height
                local_cols = cells.cols[group] % layer.tile_width
                total += float(cells.areas[group][valid[local_rows, local_cols]].sum())
        return total

    def _tile_mask(
        self,
        connection: sqlite3.Connection,
        layer: TileLayer,
        tile_column: int,
        tile_row: int,
    ) -> np.ndarray | None:
        key = (layer.table, tile_column, tile_row)
        if key in self._tile_cache:
            self._tile_cache.move_to_end(key)
            return self._tile_cache[key]
        blob = self._tile_blob(connection, layer, tile_column, tile_row)
        valid = None if blob is None else self._decode_tile(blob)
        self._tile_cache[key] = valid
        if len(self._tile_cache) > _GEOPACKAGE_TILE_CACHE_SIZE:
            self._tile_cache.popitem(last=False)
        return valid

    @staticmethod
    def _tile_blob(
        connection: sqlite3.Connection,
        layer: TileLayer,
        tile_column: int,
        tile_row: int,
    ) -> bytes | memoryview | None:
        table = sql_identifier(layer.table)
        row = connection.execute(
            # Identifier escaped by sql_identifier; all values are bound parameters.
            f"SELECT tile_data FROM {table} "  # noqa: S608
            "WHERE zoom_level = ? AND tile_column = ? AND tile_row = ?",
            (layer.zoom_level, tile_column, tile_row),
        ).fetchone()
        if row is None:
            return None
        if not isinstance(row[0], (bytes, memoryview)):
            raise SchemaError(f"GeoPackage tile {layer.table} has no binary tile data")
        return row[0]

    def _decode_tile(self, blob: bytes | memoryview) -> np.ndarray | None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(bytes(blob)) as memory, memory.open() as dataset:
                data = dataset.read(1, masked=True)
                values = np.asarray(data)
                valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
                if dataset.count >= _ALPHA_BAND:
                    valid &= dataset.read(_ALPHA_BAND) > 0
        return valid if valid.any() else None
