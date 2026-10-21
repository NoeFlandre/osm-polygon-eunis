"""Small grid and extent helpers shared by EUNIS raster queries."""

from __future__ import annotations

from typing import Protocol

import rasterio

from .grid_overlap import is_square_north_up
from .raster_geometry import wgs84_envelope


class _RasterLayerLike(Protocol):
    """The stable code field needed by raster ordering helpers."""

    @property
    def code(self) -> str: ...


RasterLayerLike = _RasterLayerLike


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
