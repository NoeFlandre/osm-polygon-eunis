"""Geometry parsing and projection helpers kept separate from I/O."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import TypeGuard

from pyproj import CRS, Transformer
from shapely import from_wkt, segmentize
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform, unary_union
from shapely.validation import make_valid
from shapely.wkb import loads as load_wkb

from .domain import EPSG_LAEA_EUROPE_CRS, WGS84_CRS, WGS84_EPSG_CODE

WGS84_MAX_SEGMENT_LENGTH_DEGREES = 0.01
ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES = 180.0
OVERLAP_KERNEL_VERSION = 3
__all__ = [
    "ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES",
    "GEOMETRY_POLICY",
    "OVERLAP_KERNEL_VERSION",
    "WGS84_EPSG_CODE",
    "WGS84_MAX_SEGMENT_LENGTH_DEGREES",
    "has_antimeridian_span",
    "is_usable",
    "parse_geometry",
    "safe_area",
    "to_equal_area",
]
GEOMETRY_POLICY: dict[str, object] = {
    "accepted_geometry_types": ["Polygon", "MultiPolygon"],
    "geometry_collection_policy": "retain polygonal parts; ignore points and lines",
    "collapsed_polygon_policy": "reject and count as invalid geometry",
    "wgs84_max_segment_length_degrees": WGS84_MAX_SEGMENT_LENGTH_DEGREES,
    "antimeridian_policy": "reject polygons whose longitude span exceeds 180 degrees",
    "overlap_kernel_version": OVERLAP_KERNEL_VERSION,
}


def parse_geometry(value: object) -> BaseGeometry | None:
    """Parse an areal geometry, retaining only polygon parts after repair."""

    if value is None:
        return None
    try:
        geometry = _decode_geometry(value)
        return _valid_polygonal_geometry(geometry)
    except (GEOSException, TypeError, ValueError, json.JSONDecodeError):
        return None


def _decode_geometry(value: object) -> BaseGeometry | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return load_wkb(bytes(value))
    if isinstance(value, str):
        try:
            payload: object = json.loads(value)
        except json.JSONDecodeError:
            return from_wkt(value)
    else:
        payload = value
    if not isinstance(payload, Mapping):
        return None
    return shape(payload)


def _valid_polygonal_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return None
    polygonal = _combine_polygonal_parts(_polygonal_parts(geometry))
    if polygonal is None:
        return None
    if not _is_positive_polygonal(polygonal):
        return None
    return polygonal


def _combine_polygonal_parts(parts: list[Polygon]) -> BaseGeometry | None:
    if not parts:
        return None
    return parts[0] if len(parts) == 1 else unary_union(parts)


def _is_positive_polygonal(geometry: BaseGeometry) -> bool:
    return all(
        (
            geometry.geom_type in {"Polygon", "MultiPolygon"},
            not geometry.is_empty,
            geometry.is_valid,
            geometry.area > 0.0,
        )
    )


def _polygonal_parts(geometry: BaseGeometry) -> list[Polygon]:
    if geometry.is_empty:
        return []
    if isinstance(geometry, Polygon):
        return _polygon_parts_from_polygon(geometry)
    if isinstance(geometry, (MultiPolygon, GeometryCollection)):
        return _polygon_parts_from_collection(geometry)
    return []


def _polygon_parts_from_polygon(geometry: Polygon) -> list[Polygon]:
    repaired = geometry if geometry.is_valid else make_valid(geometry)
    if isinstance(repaired, Polygon):
        return [repaired] if repaired.area > 0.0 else []
    return _polygonal_parts(repaired)


def _polygon_parts_from_collection(
    geometry: MultiPolygon | GeometryCollection,
) -> list[Polygon]:
    return [part for child in geometry.geoms for part in _polygonal_parts(child)]


@lru_cache(maxsize=8)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def has_antimeridian_span(geometry: BaseGeometry) -> bool:
    """Return whether WGS84 longitude bounds span more than 180 degrees."""

    min_longitude, _, max_longitude, _ = geometry.bounds
    return max_longitude - min_longitude > ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES


def to_equal_area(
    geometry: BaseGeometry | None,
    source_crs: str = WGS84_CRS,
    target_crs: str = EPSG_LAEA_EUROPE_CRS,
) -> BaseGeometry | None:
    """Densify WGS84 edges, then project areal input to the EEA equal-area CRS.

    Longitude spans greater than 180 degrees are rejected as antimeridian-
    spanning input. WGS84 edges are segmented to at most 0.01 degrees before
    projection so their projected curves are represented by short chords.
    """

    polygonal = _valid_polygonal_geometry(geometry)
    if polygonal is None:
        return None
    if CRS.from_user_input(source_crs).to_epsg() == WGS84_EPSG_CODE:
        if has_antimeridian_span(polygonal):
            return None
        polygonal = segmentize(
            polygonal,
            max_segment_length=WGS84_MAX_SEGMENT_LENGTH_DEGREES,
        )
    projected = transform(_transformer(source_crs, target_crs).transform, polygonal)
    return _valid_polygonal_geometry(projected)


def is_usable(geometry: BaseGeometry | None) -> TypeGuard[BaseGeometry]:
    """Return whether a geometry is non-null, non-empty and valid."""

    return geometry is not None and not geometry.is_empty and geometry.is_valid


def safe_area(geometry: BaseGeometry | None) -> float:
    """Return a positive area only for a usable geometry."""

    if not is_usable(geometry):
        return 0.0
    return float(geometry.area)
