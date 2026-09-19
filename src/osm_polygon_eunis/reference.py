"""Official EEA raster reference access with exact cell intersections."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import struct
from collections import OrderedDict
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar, cast

import numpy as np
import rasterio
from pyproj import CRS
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.windows import Window, WindowError, from_bounds
from shapely.geometry import GeometryCollection, box, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import loads as load_wkb

from .domain import EunisResult, OverlapCandidate
from .matching import choose_winner

_LAYER_CODE = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")
_RASTER_TILE_SIZE = 64
_RASTER_TILE_CACHE_SIZE = 1024
_GEOPACKAGE_TILE_CACHE_SIZE = 128


@dataclass(frozen=True, slots=True)
class RasterLayer:
    """One EEA probability raster and its stable EUNIS metadata."""

    code: str
    name: str
    path: Path
    source_version: str


def parse_layer_code(filename: str) -> str:
    """Extract the EUNIS code from an official probability-map filename."""

    match = _LAYER_CODE.fullmatch(Path(filename).name)
    if match is None:
        raise ValueError(f"could not parse reference layer code from {filename!r}")
    return match.group("code")


def load_label_table(path: Path) -> dict[str, str]:
    """Load a compact code-to-name JSON table and reject incomplete values."""

    value: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("EUNIS label table must be a JSON object")
    return _validated_labels(value)


def _validated_labels(value: dict[object, object]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for code, name in value.items():
        if not isinstance(code, str) or not isinstance(name, str) or not name.strip():
            raise ValueError("EUNIS label table contains a non-string or empty name")
        labels[code] = name
    return labels


def raster_layer_from_file(
    path: Path,
    labels: dict[str, str],
    source_version: str,
) -> RasterLayer:
    """Create a layer only when its parsed code has a verified name."""

    code = parse_layer_code(path.name)
    try:
        name = labels[code]
    except KeyError as error:
        raise ValueError(f"missing EUNIS name for reference code {code}") from error
    return RasterLayer(code, name, path, source_version)


class RasterReference:
    """Read only the raster windows needed for a projected input polygon."""

    def __init__(self, layers: tuple[RasterLayer, ...], threshold: int = 0) -> None:
        versions = {layer.source_version for layer in layers}
        if len(versions) > 1:
            raise ValueError("all reference layers must use one source version")
        if threshold < 0:
            raise ValueError("raster threshold must be non-negative")
        self._layers = layers
        self._threshold = threshold
        self._source_version = next(iter(versions), None)
        self._stack: ExitStack | None = None
        self._datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...] = ()
        self._tile_cache: OrderedDict[tuple[str, int, int], BaseGeometry | None] = OrderedDict()

    def __enter__(self) -> RasterReference:
        if self._stack is not None:
            raise RuntimeError("raster reference is already open")
        stack = ExitStack()
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=256))
        datasets: list[tuple[RasterLayer, rasterio.DatasetReader]] = []
        for layer in self._layers:
            dataset = stack.enter_context(rasterio.open(layer.path))
            self._validate_crs(dataset, layer.path)
            datasets.append((layer, dataset))
        self._stack = stack
        self._datasets = tuple(datasets)
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._stack is not None:
            self._stack.close()
            self._stack = None
            self._datasets = ()
            self._tile_cache.clear()

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        """Return the label with the largest actual intersection area."""

        if polygon is None or polygon.is_empty or not polygon.is_valid:
            return choose_winner(polygon, (), source_version=self._source_version)

        if self._stack is not None:
            return self._overlap_open(polygon, self._datasets)
        return self._overlap_closed(polygon)

    def _overlap_closed(self, polygon: BaseGeometry) -> EunisResult:
        with ExitStack() as stack:
            stack.enter_context(rasterio.Env(GDAL_CACHEMAX=256))
            datasets = self._open_datasets(stack)
            return self._overlap_open(polygon, tuple(datasets))

    def _open_datasets(
        self,
        stack: ExitStack,
    ) -> list[tuple[RasterLayer, rasterio.DatasetReader]]:
        datasets: list[tuple[RasterLayer, rasterio.DatasetReader]] = []
        for layer in self._layers:
            dataset = stack.enter_context(rasterio.open(layer.path))
            self._validate_crs(dataset, layer.path)
            datasets.append((layer, dataset))
        return datasets

    def _overlap_open(
        self,
        polygon: BaseGeometry,
        datasets: tuple[tuple[RasterLayer, rasterio.DatasetReader], ...],
    ) -> EunisResult:
        candidates: list[OverlapCandidate] = []
        for layer, dataset in datasets:
            cell_geometry = self._positive_cell_geometry(layer, dataset, polygon)
            if cell_geometry is not None:
                candidates.append(
                    OverlapCandidate(
                        layer.code,
                        layer.name,
                        cell_geometry,
                        components_are_disjoint=_has_disjoint_components(cell_geometry),
                    )
                )
        return choose_winner(polygon, candidates, source_version=self._source_version)

    @staticmethod
    def _validate_crs(dataset: rasterio.DatasetReader, path: Path) -> None:
        try:
            crs = dataset.crs
            valid = _is_epsg_3035(crs)
        except (TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError(f"reference raster {path} is not EPSG:3035")

    def _positive_cell_geometry(
        self,
        layer: RasterLayer,
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> BaseGeometry | None:
        raster_bounds = box(*dataset.bounds)
        if not polygon.intersects(raster_bounds):
            return None
        window = self._window(dataset, polygon)
        if window is None:
            return None
        return _merge_tile_cells(self._raster_tile_cells(layer, dataset, window))

    def _raster_tile_cells(
        self,
        layer: RasterLayer,
        dataset: rasterio.DatasetReader,
        window: Window,
    ) -> list[BaseGeometry]:
        return [
            geometry
            for row in _raster_tile_indices(window.row_off, window.height)
            for column in _raster_tile_indices(window.col_off, window.width)
            if (geometry := self._cached_tile_geometry(layer, dataset, row, column)) is not None
        ]

    def _cached_tile_geometry(
        self,
        layer: RasterLayer,
        dataset: rasterio.DatasetReader,
        row: int,
        column: int,
    ) -> BaseGeometry | None:
        key = (layer.code, row, column)
        if key in self._tile_cache:
            geometry = self._tile_cache[key]
            self._tile_cache.move_to_end(key)
            return geometry
        window = _raster_tile_window(dataset, row, column)
        valid = self._positive_mask(dataset, window)
        geometry = (
            None if valid is None else _mask_geometry(valid, dataset.window_transform(window))
        )
        self._tile_cache[key] = geometry
        if len(self._tile_cache) > _RASTER_TILE_CACHE_SIZE:
            self._tile_cache.popitem(last=False)
        return geometry

    @staticmethod
    def _window(
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None:
        try:
            window = from_bounds(*polygon.bounds, transform=dataset.transform)
            full_window = Window.from_slices(
                (0, dataset.height),
                (0, dataset.width),
            )
            window = window.intersection(full_window).round_offsets().round_lengths()
        except WindowError:
            return None
        return window if window.width > 0 and window.height > 0 else None

    def _positive_mask(
        self,
        dataset: rasterio.DatasetReader,
        window: Window,
    ) -> np.ndarray | None:
        data = dataset.read(1, window=window, masked=True)
        values = np.asarray(data)
        valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
        return valid if valid.any() else None


def _mask_geometry(valid: np.ndarray, transform: Any) -> BaseGeometry | None:
    cells = [
        shape(geometry)
        for geometry, _ in shapes(
            valid.astype("uint8"),
            mask=valid,
            transform=transform,
        )
    ]
    return _geometry_collection(cells)


def _geometry_collection(cells: list[BaseGeometry]) -> BaseGeometry | None:
    if not cells:
        return None
    if len(cells) == 1:
        return cells[0]
    return GeometryCollection(cells)


def _raster_tile_indices(offset: float, length: float) -> range:
    first = max(0, math.floor(offset / _RASTER_TILE_SIZE))
    last = math.ceil((offset + length) / _RASTER_TILE_SIZE)
    return range(first, last)


def _raster_tile_window(
    dataset: rasterio.DatasetReader,
    row: int,
    column: int,
) -> Window:
    row_offset = row * _RASTER_TILE_SIZE
    column_offset = column * _RASTER_TILE_SIZE
    width = min(_RASTER_TILE_SIZE, dataset.width - column_offset)
    height = min(_RASTER_TILE_SIZE, dataset.height - row_offset)
    return Window.from_slices(
        (row_offset, row_offset + height),
        (column_offset, column_offset + width),
    )


def _is_epsg_3035(crs: object) -> bool:
    if crs is None:
        return False
    if getattr(crs, "to_epsg", lambda: None)() == 3035:
        return True
    return _is_equivalent_crs(crs)


def _is_equivalent_crs(crs: object) -> bool:
    if CRS.from_user_input(crs).equals(CRS.from_epsg(3035)):
        return True
    to_wkt = getattr(crs, "to_wkt", None)
    wkt = to_wkt() if callable(to_wkt) else str(crs)
    return _has_catalog_crs_wkt(wkt)


def _has_catalog_crs_wkt(wkt: str) -> bool:
    return 'AUTHORITY["EPSG","3035"]' in wkt and "ETRS89-extended / LAEA Europe" in wkt


@dataclass(frozen=True, slots=True)
class _VectorLayer:
    table: str
    geometry_column: str
    primary_key: str
    code_column: str | None
    rtree_table: str


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


_TileMetadata = tuple[str, int, float, float, float, float, int, int, int, int, float, float, int]


def _sql_identifier(value: str) -> str:
    if not value or "\x00" in value:
        raise ValueError("invalid GeoPackage identifier")
    return '"' + value.replace('"', '""') + '"'


def _validate_vector_header(
    table: object,
    geometry_column: object,
    srs_id: object,
) -> None:
    if srs_id != 3035:
        raise ValueError(f"GeoPackage layer {table} is not EPSG:3035")
    if not isinstance(table, str) or not isinstance(geometry_column, str):
        raise ValueError("GeoPackage geometry metadata is invalid")


def _require_rtree(
    connection: sqlite3.Connection,
    table: str,
    rtree_table: str,
) -> None:
    exists = connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
        (rtree_table,),
    ).fetchone()
    if exists is None:
        raise ValueError(f"GeoPackage layer {table} has no RTree index")


def _vector_code(value: object, labels: dict[str, str]) -> str:
    if not isinstance(value, str) or value not in labels:
        raise ValueError(f"GeoPackage feature has unknown EUNIS code {value!r}")
    return value


def _vector_blob(value: object) -> bytes | memoryview:
    if not isinstance(value, (bytes, memoryview)):
        raise ValueError("GeoPackage geometry is not binary")
    return value


class GeoPackageReference:
    """Query official EPSG:3035 GeoPackage habitat polygons by RTree bounds."""

    _CODE_COLUMNS: ClassVar[frozenset[str]] = frozenset(
        {"code", "eunis_code", "prob", "habitat_code", "eunis_habitat_code"}
    )

    def __init__(
        self,
        path: Path,
        labels: dict[str, str],
        *,
        source_version: str,
        threshold: int = 0,
    ) -> None:
        if not labels:
            raise ValueError("GeoPackage reference labels must not be empty")
        if threshold < 0:
            raise ValueError("GeoPackage threshold must be non-negative")
        self._path = path
        self._labels = labels
        self._source_version = source_version
        self._threshold = threshold
        self._connection: sqlite3.Connection | None = None
        self._layers: tuple[_VectorLayer, ...] = ()
        self._tile_layers: tuple[_TileLayer, ...] = ()
        self._tile_cache: OrderedDict[tuple[str, int, int], BaseGeometry | None] = OrderedDict()

    def __enter__(self) -> GeoPackageReference:
        if self._connection is not None:
            raise RuntimeError("GeoPackage reference is already open")
        self._connection = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        self._layers = self._discover_layers(self._connection)
        self._tile_layers = self._discover_tile_layers(self._connection)
        if not self._layers and not self._tile_layers:
            raise ValueError("GeoPackage has no EPSG:3035 geometry or tile layers")
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._layers = ()
            self._tile_layers = ()
            self._tile_cache.clear()

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        """Return the largest exact polygon intersection from indexed features."""

        if polygon is None or polygon.is_empty or not polygon.is_valid:
            return choose_winner(polygon, (), source_version=self._source_version)
        if self._connection is None:
            with self:
                return self._overlap_open(polygon)
        return self._overlap_open(polygon)

    def _overlap_open(self, polygon: BaseGeometry) -> EunisResult:
        assert self._connection is not None
        candidates = self._vector_candidates(self._connection, polygon)
        candidates.extend(self._tile_candidates(self._connection, polygon))
        return choose_winner(polygon, candidates, source_version=self._source_version)

    def _vector_candidates(
        self,
        connection: sqlite3.Connection,
        polygon: BaseGeometry,
    ) -> list[OverlapCandidate]:
        candidates: list[OverlapCandidate] = []
        for layer in self._layers:
            grouped: dict[str, list[BaseGeometry]] = {}
            for row in self._candidate_rows(connection, layer, polygon):
                decoded = self._decode_vector_row(row)
                if decoded is not None:
                    code, geometry = decoded
                    grouped.setdefault(code, []).append(geometry)
            candidates.extend(
                OverlapCandidate(code, self._labels[code], unary_union(geometries))
                for code, geometries in grouped.items()
            )
        return candidates

    def _decode_vector_row(
        self,
        row: tuple[object, object],
    ) -> tuple[str, BaseGeometry] | None:
        code = _vector_code(row[0], self._labels)
        blob = _vector_blob(row[1])
        geometry = self.decode_geometry(blob)
        return (code, geometry) if not geometry.is_empty and geometry.is_valid else None

    def _tile_candidates(
        self,
        connection: sqlite3.Connection,
        polygon: BaseGeometry,
    ) -> list[OverlapCandidate]:
        candidates: list[OverlapCandidate] = []
        for layer in self._tile_layers:
            geometry = self._positive_tile_geometry(connection, layer, polygon)
            if geometry is not None:
                candidates.append(
                    OverlapCandidate(
                        layer.table,
                        self._labels[layer.table],
                        geometry,
                        components_are_disjoint=_has_disjoint_components(geometry),
                    )
                )
        return candidates

    @staticmethod
    def decode_geometry(blob: bytes | memoryview) -> BaseGeometry:
        """Decode a GeoPackage binary geometry and require ETRS89 LAEA Europe."""

        data = bytes(blob)
        if len(data) < 8 or data[:2] != b"GP":
            raise ValueError("invalid GeoPackage geometry header")
        flags = data[3]
        byte_order = "<" if flags & 1 else ">"
        srs_id = struct.unpack_from(f"{byte_order}i", data, 4)[0]
        if srs_id != 3035:
            raise ValueError(f"GeoPackage geometry is not EPSG:3035: {srs_id}")
        envelope_type = (flags >> 1) & 0b111
        envelope_size = _envelope_size(envelope_type)
        return load_wkb(data[8 + envelope_size :])


    @classmethod
    def _discover_layers(cls, connection: sqlite3.Connection) -> tuple[_VectorLayer, ...]:
        rows = cls._geometry_rows(connection)
        return tuple(cls._vector_layer(connection, row) for row in rows)

    @staticmethod
    def _geometry_rows(connection: sqlite3.Connection) -> list[tuple[object, ...]]:
        try:
            return connection.execute(
                "SELECT table_name, column_name, srs_id FROM gpkg_geometry_columns"
            ).fetchall()
        except sqlite3.DatabaseError as error:
            if "no such table: gpkg_geometry_columns" in str(error):
                return []
            raise ValueError("not a readable GeoPackage") from error

    @classmethod
    def _vector_layer(
        cls,
        connection: sqlite3.Connection,
        row: tuple[object, ...],
    ) -> _VectorLayer:
        table, geometry_column, srs_id = row
        _validate_vector_header(table, geometry_column, srs_id)
        assert isinstance(table, str)
        assert isinstance(geometry_column, str)
        primary_key, code_column, rtree_table = cls._vector_details(
            connection,
            table,
            geometry_column,
        )
        return _VectorLayer(table, geometry_column, primary_key, code_column, rtree_table)

    @classmethod
    def _vector_details(
        cls,
        connection: sqlite3.Connection,
        table: str,
        geometry_column: str,
    ) -> tuple[str, str, str]:
        primary_key, code_column = cls._vector_columns(connection, table)
        rtree_table = f"rtree_{table}_{geometry_column}"
        _require_rtree(connection, table, rtree_table)
        return primary_key, code_column, rtree_table

    @classmethod
    def _vector_columns(
        cls,
        connection: sqlite3.Connection,
        table: str,
    ) -> tuple[str, str]:
        columns = connection.execute(
            f"PRAGMA table_info({_sql_identifier(table)})"
        ).fetchall()
        if not columns:
            raise ValueError(f"GeoPackage geometry table is missing: {table}")
        primary_key = cast(str, next((column[1] for column in columns if column[5]), "rowid"))
        return primary_key, cls._required_code_column(columns, table)

    @classmethod
    def _required_code_column(
        cls,
        columns: list[tuple[object, ...]],
        table: str,
    ) -> str:
        code_column = cls._code_column(columns)
        if code_column is None:
            raise ValueError(f"GeoPackage layer {table} has no EUNIS code column")
        return code_column

    @classmethod
    def _code_column(cls, columns: list[tuple[object, ...]]) -> str | None:
        return next(
            (
                row[1]
                for row in columns
                if isinstance(row[1], str) and row[1].lower() in cls._CODE_COLUMNS
            ),
            None,
        )

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
        except sqlite3.DatabaseError:
            return []

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
        with MemoryFile(bytes(blob)) as memory, memory.open() as dataset:
            data = dataset.read(1, masked=True)
            values = np.asarray(data)
            valid = (~np.ma.getmaskarray(data)) & (values > self._threshold)
            if dataset.count >= 4:
                valid &= dataset.read(4) > 0
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
            f"SELECT tile_column, tile_row, tile_data FROM {table} "
            "WHERE zoom_level = ? AND tile_column BETWEEN ? AND ? "
            "AND tile_row BETWEEN ? AND ?",
            (layer.zoom_level, min_column, max_column, min_row, max_row),
        ).fetchall()
        typed_rows: list[tuple[int, int, bytes | memoryview]] = []
        for tile_column, tile_row, blob in rows:
            if not isinstance(blob, (bytes, memoryview)):
                raise ValueError(f"GeoPackage tile {layer.table} has no binary tile data")
            typed_rows.append((int(tile_column), int(tile_row), blob))
        return typed_rows

    @staticmethod
    def _candidate_rows(
        connection: sqlite3.Connection,
        layer: _VectorLayer,
        polygon: BaseGeometry,
    ) -> list[tuple[object, object]]:
        table = _sql_identifier(layer.table)
        geometry = _sql_identifier(layer.geometry_column)
        code = _sql_identifier(layer.code_column or "")
        rtree = _sql_identifier(layer.rtree_table)
        key = (
            "t.rowid"
            if layer.primary_key == "rowid"
            else f"t.{_sql_identifier(layer.primary_key)}"
        )
        query = (
            f"SELECT t.{code}, t.{geometry} FROM {table} AS t "
            f"JOIN {rtree} AS r ON r.id = {key} "
            "WHERE r.maxx > ? AND r.minx < ? AND r.maxy > ? AND r.miny < ?"
        )
        min_x, min_y, max_x, max_y = polygon.bounds
        return connection.execute(query, (min_x, max_x, min_y, max_y)).fetchall()


def _merge_tile_cells(cells: list[BaseGeometry]) -> BaseGeometry | None:
    return _geometry_collection(cells)


def _has_disjoint_components(geometry: BaseGeometry) -> bool:
    return isinstance(geometry, GeometryCollection)


def _tile_values(row: tuple[object, ...]) -> _TileMetadata:
    if len(row) != 13:
        raise ValueError("GeoPackage tile metadata has an unexpected shape")
    return cast(_TileMetadata, row)


def _validate_tile_header(
    table: str,
    content_srs_id: int,
    labels: dict[str, str],
) -> None:
    if content_srs_id != 3035:
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


def _envelope_size(envelope_type: int) -> int:
    try:
        return {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_type]
    except KeyError as error:
        raise ValueError("unsupported GeoPackage envelope type") from error


def resolve_eea_layers(config_path: Path, workspace: Path) -> tuple[RasterLayer, ...]:
    """Load a run-resolved local layer manifest produced from the EEA catalog."""

    config: Any = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ValueError(
            "reference config has no resolved layers; run the EEA catalog resolver first",
        )
    layer_specs = _resolved_layer_specs(config)
    source_version = str(config["source_version"])
    return tuple(
        _resolved_layer(spec, workspace, source_version) for spec in layer_specs
    )


def _resolved_layer_specs(config: dict[object, object]) -> list[object]:
    layer_specs = config.get("layers")
    if not isinstance(layer_specs, list) or not layer_specs:
        raise ValueError(
            "reference config has no resolved layers; run the EEA catalog resolver first",
        )
    return layer_specs


def _resolved_layer(spec: object, workspace: Path, source_version: str) -> RasterLayer:
    if not isinstance(spec, dict):
        raise ValueError("reference layer specification must be an object")
    relative_path, code, name = _layer_values(spec)
    path = workspace / relative_path
    if not path.is_file():
        raise FileNotFoundError(path)
    return RasterLayer(code=code, name=name, path=path, source_version=source_version)


def _layer_values(spec: dict[object, object]) -> tuple[str, str, str]:
    values = tuple(_required_layer_text(spec, field) for field in ("path", "code", "name"))
    return cast(tuple[str, str, str], values)


def _required_layer_text(spec: dict[object, object], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError("reference layer specification is missing path, code, or name")
    return value
