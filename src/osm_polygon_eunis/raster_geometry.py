"""Geometry construction and CRS helpers for raster references."""

from __future__ import annotations

from typing import Protocol

from pyproj import CRS, Transformer

from .domain import EPSG_LAEA_EUROPE, EPSG_LAEA_EUROPE_CRS, WGS84_CRS

# 128-pixel cache windows meet the synthetic overlap miss-rate budget with
# less polygonisation overhead than 256-pixel windows.
_RASTER_TILE_SIZE = 128
_CACHE_ENTRY_OVERHEAD_BYTES = 256
_CACHE_GEOMETRY_SIZE_MULTIPLIER = 2


class _CrsLike(Protocol):
    """CRS objects (pyproj or rasterio) that report an EPSG code and WKT."""

    def to_epsg(self) -> int | None: ...

    def to_wkt(self) -> str: ...


def _is_epsg_3035(crs: _CrsLike | str | None) -> bool:
    if crs is None:
        return False
    if not isinstance(crs, str) and crs.to_epsg() == EPSG_LAEA_EUROPE:
        return True
    return _is_equivalent_crs(crs)


def _is_equivalent_crs(crs: _CrsLike | str) -> bool:
    if CRS.from_user_input(crs).equals(CRS.from_epsg(EPSG_LAEA_EUROPE)):
        return True
    wkt = crs if isinstance(crs, str) else crs.to_wkt()
    return _has_catalog_crs_wkt(wkt)


def _has_catalog_crs_wkt(wkt: str) -> bool:
    return 'AUTHORITY["EPSG","3035"]' in wkt and "ETRS89-extended / LAEA Europe" in wkt


def wgs84_envelope(
    bounds: tuple[float, float, float, float],
    source_crs: str = EPSG_LAEA_EUROPE_CRS,
    pad: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a padded WGS84 envelope for a projected extent."""

    transformer = Transformer.from_crs(source_crs, WGS84_CRS, always_xy=True)
    left, bottom, right, top = transformer.transform_bounds(*bounds, densify_pts=101)
    return (left - pad, bottom - pad, right + pad, top + pad)
