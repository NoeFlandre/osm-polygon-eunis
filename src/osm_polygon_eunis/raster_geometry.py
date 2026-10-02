"""Geometry construction and CRS helpers for raster references."""

from __future__ import annotations

from pyproj import CRS, Transformer

# 128-pixel cache windows meet the synthetic overlap miss-rate budget with
# less polygonisation overhead than 256-pixel windows.
_RASTER_TILE_SIZE = 128
_CACHE_ENTRY_OVERHEAD_BYTES = 256
_CACHE_GEOMETRY_SIZE_MULTIPLIER = 2
EPSG_LAEA_EUROPE = 3035


def _is_epsg_3035(crs: object) -> bool:
    if crs is None:
        return False
    if getattr(crs, "to_epsg", lambda: None)() == EPSG_LAEA_EUROPE:
        return True
    return _is_equivalent_crs(crs)


def _is_equivalent_crs(crs: object) -> bool:
    if CRS.from_user_input(crs).equals(CRS.from_epsg(EPSG_LAEA_EUROPE)):
        return True
    to_wkt = getattr(crs, "to_wkt", None)
    wkt = to_wkt() if callable(to_wkt) else str(crs)
    return _has_catalog_crs_wkt(wkt)


def _has_catalog_crs_wkt(wkt: str) -> bool:
    return 'AUTHORITY["EPSG","3035"]' in wkt and "ETRS89-extended / LAEA Europe" in wkt


def wgs84_envelope(
    bounds: tuple[float, float, float, float],
    source_crs: str = "EPSG:3035",
    pad: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return a padded WGS84 envelope for a projected extent."""

    transformer = Transformer.from_crs(source_crs, "EPSG:4326", always_xy=True)
    left, bottom, right, top = transformer.transform_bounds(*bounds, densify_pts=101)
    return (left - pad, bottom - pad, right + pad, top + pad)
