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
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from shapely.geometry import box, shape
from shapely.geometry.base import BaseGeometry

from .domain import SchemaError
from .geopackage_sql import _sql_identifier
from .reference_geometry import _geometry_collection, _merge_tile_cells

EPSG_LAEA_EUROPE = 3035

_GEOPACKAGE_TILE_CACHE_SIZE = 256
_ALPHA_BAND = 4
_TILE_METADATA_FIELDS = 13

_TileMetadata = tuple[str, int, float, float, float, float, int, int, int, int, float, float, int]


@dataclass(frozen=True, slots=True)
class _TileLayer:
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


class _GeoPackageTileMethods:
    _labels: dict[str, str]
    _threshold: int
    _tile_layers: tuple[_TileLayer, ...]
    _tile_cache: OrderedDict[tuple[str, int, int], BaseGeometry | None]

    def _discover_tile_layers(self, connection: sqlite3.Connection) -> tuple[_TileLayer, ...]:
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

    def _tile_layer(self, row: tuple[object, ...]) -> _TileLayer:
        values = _tile_values(row)
        table = values[0]
        self._validate_tile_values(values)
        return _TileLayer(
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

    def _positive_tile_geometry(
        self,
        connection: sqlite3.Connection,
        layer: _TileLayer,
        polygon: BaseGeometry,
    ) -> BaseGeometry | None:
        extent = box(layer.min_x, layer.min_y, layer.max_x, layer.max_y)
        if not polygon.intersects(extent):
            return None
        cells: list[BaseGeometry] = []
        for tile_column, tile_row, blob in self._candidate_tile_rows(connection, layer, polygon):
            tile_geometry = self._cached_tile(
                layer,
                tile_column,
                tile_row,
                blob,
            )
            if tile_geometry is not None:
                cells.append(tile_geometry)
        return _merge_tile_cells(cells)

    def _cached_tile(
        self,
        layer: _TileLayer,
        tile_column: int,
        tile_row: int,
        blob: bytes | memoryview,
    ) -> BaseGeometry | None:
        key = (layer.table, tile_column, tile_row)
        if key in self._tile_cache:
            tile_geometry = self._tile_cache[key]
            self._tile_cache.move_to_end(key)
            return tile_geometry
        tile_geometry = self._decode_tile(layer, tile_column, tile_row, blob)
        self._tile_cache[key] = tile_geometry
        if len(self._tile_cache) > _GEOPACKAGE_TILE_CACHE_SIZE:
            self._tile_cache.popitem(last=False)
        return tile_geometry

    def _decode_tile(
        self,
        layer: _TileLayer,
        tile_column: int,
        tile_row: int,
        blob: bytes | memoryview,
    ) -> BaseGeometry | None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", NotGeoreferencedWarning)
            with MemoryFile(bytes(blob)) as memory, memory.open() as dataset:
                data = dataset.read(1, masked=True)
                values = np.asarray(data)
                valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
                if dataset.count >= _ALPHA_BAND:
                    valid &= dataset.read(_ALPHA_BAND) > 0
                if not valid.any():
                    return None
                origin_x = layer.min_x + tile_column * layer.tile_width * layer.pixel_x_size
                origin_y = layer.max_y - tile_row * layer.tile_height * layer.pixel_y_size
                transform = from_origin(
                    origin_x,
                    origin_y,
                    layer.pixel_x_size,
                    layer.pixel_y_size,
                )
                cells = [
                    shape(geometry)
                    for geometry, _ in shapes(
                        valid.astype("uint8"),
                        mask=valid,
                        transform=transform,
                    )
                ]
        return _geometry_collection(cells)

    @staticmethod
    def _candidate_tile_rows(
        connection: sqlite3.Connection,
        layer: _TileLayer,
        polygon: BaseGeometry,
    ) -> list[tuple[int, int, bytes | memoryview]]:
        tile_span_x = layer.tile_width * layer.pixel_x_size
        tile_span_y = layer.tile_height * layer.pixel_y_size
        min_x, min_y, max_x, max_y = polygon.bounds
        min_column = max(0, math.floor((min_x - layer.min_x) / tile_span_x))
        max_column = min(layer.matrix_width - 1, math.floor((max_x - layer.min_x) / tile_span_x))
        min_row = max(0, math.floor((layer.max_y - max_y) / tile_span_y))
        max_row = min(layer.matrix_height - 1, math.floor((layer.max_y - min_y) / tile_span_y))
        if min_column > max_column or min_row > max_row:
            return []
        table = _sql_identifier(layer.table)
        rows = connection.execute(
            # Identifier escaped by _sql_identifier; all values are bound parameters.
            f"SELECT tile_column, tile_row, tile_data FROM {table} "  # noqa: S608
            "WHERE zoom_level = ? AND tile_column BETWEEN ? AND ? "
            "AND tile_row BETWEEN ? AND ?",
            (layer.zoom_level, min_column, max_column, min_row, max_row),
        ).fetchall()
        typed_rows: list[tuple[int, int, bytes | memoryview]] = []
        for tile_column, tile_row, blob in rows:
            if not isinstance(blob, (bytes, memoryview)):
                raise SchemaError(f"GeoPackage tile {layer.table} has no binary tile data")
            typed_rows.append((int(tile_column), int(tile_row), blob))
        return typed_rows
