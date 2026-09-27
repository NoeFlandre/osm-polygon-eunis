"""Geometry collection helpers shared by the reference readers."""

from __future__ import annotations

from shapely.geometry import GeometryCollection
from shapely.geometry.base import BaseGeometry


def _geometry_collection(cells: list[BaseGeometry]) -> BaseGeometry | None:
    if not cells:
        return None
    if len(cells) == 1:
        return cells[0]
    return GeometryCollection(cells)


def _merge_tile_cells(cells: list[BaseGeometry]) -> BaseGeometry | None:
    return _geometry_collection(cells)


def _has_disjoint_components(geometry: BaseGeometry) -> bool:
    return isinstance(geometry, GeometryCollection)
