"""Exact-overlap queries for vector GeoPackage reference layers."""

from __future__ import annotations

import logging
import sqlite3
import struct
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, cast

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import loads as load_wkb

from .domain import EPSG_LAEA_EUROPE, EunisResult, OverlapCandidate, SchemaError
from .geometry import is_usable, repair_polygonal
from .geopackage_sql import _sql_identifier
from .geopackage_tiles import _GeoPackageTileMethods, _TileLayer
from .grid_overlap import WeightedCells
from .matching import choose_winner

_GPKG_HEADER_SIZE = 8

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _VectorLayer:
    table: str
    geometry_column: str
    primary_key: str
    code_column: str | None
    rtree_table: str


def _validate_vector_header(
    table: object,
    geometry_column: object,
    srs_id: object,
) -> None:
    if srs_id != EPSG_LAEA_EUROPE:
        raise ValueError(f"GeoPackage layer {table} is not EPSG:3035")
    if not isinstance(table, str) or not isinstance(geometry_column, str):
        raise SchemaError("GeoPackage geometry metadata is invalid")


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
        raise SchemaError("GeoPackage geometry is not binary")
    return value


def _envelope_size(envelope_type: int) -> int:
    try:
        return {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_type]
    except KeyError as error:
        raise ValueError("unsupported GeoPackage envelope type") from error


class GeoPackageReference(_GeoPackageTileMethods):
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
        """Prepare a GeoPackage reference without opening its SQLite handle.

        Args:
            path: Read-only GeoPackage path.
            labels: EUNIS code to display-name mapping.
            source_version: Immutable version recorded in output labels.
            threshold: Minimum raster value counted as habitat presence.

        Raises:
            ValueError: If labels are empty or the threshold is negative.
        """
        if not labels:
            raise ValueError("GeoPackage reference labels must not be empty")
        if threshold < 0:
            raise ValueError("GeoPackage threshold must be non-negative")
        self._path = path
        self._labels = labels
        self._source_version = source_version
        self.intersection_errors = 0
        self._threshold = threshold
        self._connection: sqlite3.Connection | None = None
        self._layers: tuple[_VectorLayer, ...] = ()
        self._tile_layers: tuple[_TileLayer, ...] = ()
        self._tile_cache: OrderedDict[tuple[str, int, int], BaseGeometry | None] = OrderedDict()

    def __enter__(self) -> GeoPackageReference:
        """Open the GeoPackage and validate its geometry or tile layers."""
        if self._connection is not None:
            raise RuntimeError("GeoPackage reference is already open")
        self._connection = sqlite3.connect(f"file:{self._path}?mode=ro", uri=True)
        self._layers = self._discover_layers(self._connection)
        self._tile_layers = self._discover_tile_layers(self._connection)
        if not self._layers and not self._tile_layers:
            raise ValueError("GeoPackage has no EPSG:3035 geometry or tile layers")
        return self

    def __exit__(self, *_args: object) -> None:
        """Close the SQLite connection and release cached geometries."""
        self.close()

    def close(self) -> None:
        """Close the SQLite connection and clear this reference's tile cache."""
        if self._connection is not None:
            self._connection.close()
            self._connection = None
            self._layers = ()
            self._tile_layers = ()
            self._tile_cache.clear()

    def _count_intersection_error(self) -> None:
        self.intersection_errors += 1

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        """Return the largest exact polygon intersection from indexed features."""

        if not is_usable(polygon):
            return choose_winner(polygon, (), source_version=self._source_version)
        if self._connection is None:
            with self:
                return self._overlap_open(polygon)
        return self._overlap_open(polygon)

    def _overlap_open(self, polygon: BaseGeometry) -> EunisResult:
        connection = self._connection
        if connection is None:
            raise RuntimeError("GeoPackage reference is not open")
        candidates = self._vector_candidates(connection, polygon)
        return choose_winner(
            polygon,
            candidates,
            source_version=self._source_version,
            on_error=self._count_intersection_error,
            extra_areas=self._tile_areas(connection, polygon),
        )

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
        geometry: BaseGeometry | None = self.decode_geometry(blob)
        if not is_usable(geometry):
            geometry = repair_polygonal(geometry)
        if is_usable(geometry):
            return (code, geometry)
        logger.warning("skipping unrepairable EUNIS reference geometry for %s", code)
        self._count_intersection_error()
        return None

    def _tile_areas(
        self,
        connection: sqlite3.Connection,
        polygon: BaseGeometry,
    ) -> list[tuple[float, str, str]]:
        areas: list[tuple[float, str, str]] = []
        bands: dict[tuple[object, ...], list[WeightedCells]] = {}
        for layer in self._tile_layers:
            try:
                area = self._tile_layer_area(connection, layer, polygon, bands)
            except (ValueError, RuntimeError, shapely.errors.GEOSException) as error:
                logger.warning("skipping EUNIS tile layer %s: %s", layer.table, error)
                self._count_intersection_error()
                continue
            if area > 0.0:
                areas.append((area, layer.table, self._labels[layer.table]))
        return areas

    @staticmethod
    def decode_geometry(blob: bytes | memoryview) -> BaseGeometry:
        """Decode a GeoPackage binary geometry and require ETRS89 LAEA Europe."""

        data = bytes(blob)
        if len(data) < _GPKG_HEADER_SIZE or data[:2] != b"GP":
            raise ValueError("invalid GeoPackage geometry header")
        flags = data[3]
        byte_order = "<" if flags & 1 else ">"
        srs_id = struct.unpack_from(f"{byte_order}i", data, 4)[0]
        if srs_id != EPSG_LAEA_EUROPE:
            raise ValueError(f"GeoPackage geometry is not EPSG:3035: {srs_id}")
        envelope_type = (flags >> 1) & 0b111
        envelope_size = _envelope_size(envelope_type)
        return load_wkb(data[_GPKG_HEADER_SIZE + envelope_size :])

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
        table_name = cast(str, table)
        geometry_column_name = cast(str, geometry_column)
        primary_key, code_column, rtree_table = cls._vector_details(
            connection,
            table_name,
            geometry_column_name,
        )
        return _VectorLayer(
            table_name,
            geometry_column_name,
            primary_key,
            code_column,
            rtree_table,
        )

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
        columns = connection.execute(f"PRAGMA table_info({_sql_identifier(table)})").fetchall()
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
            "t.rowid" if layer.primary_key == "rowid" else f"t.{_sql_identifier(layer.primary_key)}"
        )
        query = (
            # Identifiers escaped by _sql_identifier; all values are bound parameters.
            f"SELECT t.{code}, t.{geometry} FROM {table} AS t "  # noqa: S608
            f"JOIN {rtree} AS r ON r.id = {key} "
            "WHERE r.maxx > ? AND r.minx < ? AND r.maxy > ? AND r.miny < ?"
        )
        min_x, min_y, max_x, max_y = polygon.bounds
        return connection.execute(query, (min_x, max_x, min_y, max_y)).fetchall()
