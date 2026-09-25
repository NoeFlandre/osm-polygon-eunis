"""Geometry parsing and projection helpers kept separate from I/O."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any, TypeGuard

from pyproj import Transformer
from shapely.errors import GEOSException
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import make_valid
from shapely.wkb import loads as load_wkb


def parse_geometry(value: object) -> BaseGeometry | None:
    """Parse a GeoJSON value and return a valid, non-empty Shapely geometry."""

    if value is None:
        return None
    try:
        geometry = _decode_geometry(value)
    except (GEOSException, TypeError, ValueError, json.JSONDecodeError):
        return None
    return _valid_geometry(geometry)


def _decode_geometry(value: object) -> BaseGeometry | None:
    if isinstance(value, (bytes, bytearray, memoryview)):
        return load_wkb(bytes(value))
    payload: Any = json.loads(value) if isinstance(value, str) else value
    if not isinstance(payload, Mapping):
        return None
    return shape(payload)


def _valid_geometry(geometry: BaseGeometry | None) -> BaseGeometry | None:
    if geometry is None or geometry.is_empty:
        return None
    return _repair_geometry(geometry)


def _repair_geometry(geometry: BaseGeometry) -> BaseGeometry | None:
    if not geometry.is_valid:
        geometry = make_valid(geometry)
    return None if geometry.is_empty or not geometry.is_valid else geometry


@lru_cache(maxsize=8)
def _transformer(source_crs: str, target_crs: str) -> Transformer:
    return Transformer.from_crs(source_crs, target_crs, always_xy=True)


def to_equal_area(
    geometry: BaseGeometry | None,
    source_crs: str = "EPSG:4326",
    target_crs: str = "EPSG:3035",
) -> BaseGeometry | None:
    """Project a geometry to the EEA equal-area CRS."""

    if not is_usable(geometry):
        return None
    projected = transform(_transformer(source_crs, target_crs).transform, geometry)
    return _valid_geometry(projected)


def is_usable(geometry: BaseGeometry | None) -> TypeGuard[BaseGeometry]:
    """Return whether a geometry is non-null, non-empty and valid."""

    return geometry is not None and not geometry.is_empty and geometry.is_valid


def safe_area(geometry: BaseGeometry | None) -> float:
    """Return a positive area only for a usable geometry."""

    if not is_usable(geometry):
        return 0.0
    area = float(geometry.area)
    return area if area > 0.0 else 0.0
