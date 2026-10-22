"""Small grid and extent helpers shared by EUNIS raster queries."""

from __future__ import annotations

import logging
import math
from typing import Protocol

import rasterio
from rasterio.windows import Window, WindowError, from_bounds
from shapely.geometry.base import BaseGeometry

from .grid_overlap import is_square_north_up
from .raster_geometry import wgs84_envelope


class _RasterLayerLike(Protocol):
    """The stable code field needed by raster ordering helpers."""

    @property
    def code(self) -> str: ...


RasterLayerLike = _RasterLayerLike
logger = logging.getLogger(__name__)


def outward_raster_window(
    dataset: rasterio.DatasetReader,
    polygon: BaseGeometry,
) -> Window | None:
    """Return the raster window enclosing a polygon's bounding box."""

    try:
        window = from_bounds(*polygon.bounds, transform=dataset.transform)
    except WindowError as error:
        logger.warning("raster window fallback after WindowError: %s", error)
        return None
    # Round outwards so partial edge cells are not silently dropped.
    col_off = max(0, math.floor(window.col_off))
    row_off = max(0, math.floor(window.row_off))
    col_end = min(dataset.width, math.ceil(window.col_off + window.width))
    row_end = min(dataset.height, math.ceil(window.row_off + window.height))
    if col_end <= col_off or row_end <= row_off:
        return None
    return Window.from_slices((row_off, row_end), (col_off, col_end))


def shares_square_grid(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> bool:
    """Return whether all rasters share one square north-up grid."""

    grids = {
        (tuple(dataset.transform)[:6], dataset.width, dataset.height) for _, dataset in datasets
    }
    return len(grids) == 1 and is_square_north_up(datasets[0][1].transform)


def code_order_on_shared_grid(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> tuple[int, ...] | None:
    """Return stable code order when all unique layers share one grid."""

    codes = [layer.code for layer, _ in datasets]
    if len(set(codes)) != len(codes) or not shares_square_grid(datasets):
        return None
    return tuple(sorted(range(len(datasets)), key=codes.__getitem__))


def wgs84_dataset_extent(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> tuple[float, float, float, float] | None:
    """Return the envelope covering every raster dataset in WGS84."""

    iterator = iter(datasets)
    first = next(iterator, None)
    if first is None:
        return None
    bounds = first[1].bounds
    left, bottom, right, top = bounds.left, bounds.bottom, bounds.right, bounds.top
    for _, dataset in iterator:
        bounds = dataset.bounds
        left = min(left, bounds.left)
        bottom = min(bottom, bounds.bottom)
        right = max(right, bounds.right)
        top = max(top, bounds.top)
    return wgs84_envelope((left, bottom, right, top))
