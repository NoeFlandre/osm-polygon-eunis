"""Geometry parsing and projection helpers kept separate from I/O."""

from __future__ import annotations

import json
from collections.abc import Mapping
from functools import lru_cache
from typing import Any

from pyproj import Transformer
from shapely.geometry import shape
from shapely.geometry.base import BaseGeometry
from shapely.ops import transform
from shapely.validation import make_valid


def parse_geometry(value: object) -> BaseGeometry | None:
    """Parse a GeoJSON value and return a valid, non-empty Shapely geometry."""

    if value is None:
        return None
    try:
        payload: Any = json.loads(value) if isinstance(value, str) else value
        if not isinstance(payload, Mapping):
            return None
        geometry = shape(payload)
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    if geometry.is_empty:
        return None
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

    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return None
    projected = transform(_transformer(source_crs, target_crs).transform, geometry)
    return None if projected.is_empty or not projected.is_valid else projected


def safe_area(geometry: BaseGeometry | None) -> float:
    """Return a positive area only for a usable geometry."""

    if geometry is None or geometry.is_empty or not geometry.is_valid:
        return 0.0
    area = float(geometry.area)
    return area if area > 0.0 else 0.0
