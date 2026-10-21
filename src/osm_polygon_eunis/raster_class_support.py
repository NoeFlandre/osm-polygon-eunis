"""Unioned class-support areas for independent EUNIS hierarchy decisions."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from contextlib import ExitStack
from typing import Protocol, TypeGuard

import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry.base import BaseGeometry

from .geometry import is_usable
from .grid_overlap import (
    TILE_SIZE,
    WeightedCells,
    band_row_ranges,
    weighted_cells,
)
from .raster_reference_helpers import RasterLayerLike
from .raster_reference_helpers import shares_square_grid as _shares_square_grid

_GDAL_CACHE_MAX_MB = 256
_SUPPORT_CLASS_KEY_FIELDS = 2


class _Reference(Protocol):
    @property
    def _stack(self) -> ExitStack | None: ...

    @property
    def _datasets(self) -> tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...]: ...

    def _open_datasets(
        self,
        stack: ExitStack,
    ) -> Iterable[tuple[RasterLayerLike, rasterio.DatasetReader]]: ...

    def _covering_window(
        self,
        dataset: rasterio.DatasetReader,
        polygon: BaseGeometry,
    ) -> Window | None: ...

    def _tile_stack(
        self,
        datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
        row: int,
        column: int,
    ) -> np.ndarray | None: ...


def support_areas(
    reference: _Reference,
    polygon: BaseGeometry,
    classes: Mapping[tuple[str, str], Iterable[str]],
) -> dict[tuple[str, str], float]:
    """Measure unioned map support for every requested hierarchy class.

    Areas use the projected polygon's square CRS units. NoData and uncovered
    areas contribute zero support and remain in the caller's denominator.
    Invalid polygons, incompatible grids, absent map codes, and query errors
    raise instead of being converted to a legacy empty winner.
    """

    if not is_usable(polygon):
        raise ValueError("support query requires a usable projected polygon")
    if not math.isfinite(float(polygon.area)) or polygon.area <= 0.0:
        raise ValueError("support query requires a positive full polygon area")
    normalized = _normalize_class_groups(classes)
    if reference._stack is not None:
        return _support_areas_open(reference, polygon, reference._datasets, normalized)
    with ExitStack() as stack:
        stack.enter_context(rasterio.Env(GDAL_CACHEMAX=_GDAL_CACHE_MAX_MB))
        datasets = tuple(reference._open_datasets(stack))
        return _support_areas_open(reference, polygon, datasets, normalized)


def _support_areas_open(
    reference: _Reference,
    polygon: BaseGeometry,
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
    classes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...],
) -> dict[tuple[str, str], float]:
    first = _support_grid_dataset(datasets)
    class_indexes = _class_layer_indexes(classes, datasets)
    window = reference._covering_window(first, polygon)
    areas = {class_key: 0.0 for class_key, _indexes in class_indexes}
    if window is None:
        return areas
    rows = (int(window.row_off), int(window.row_off + window.height))
    cols = (int(window.col_off), int(window.col_off + window.width))
    for band in band_row_ranges(rows[0], rows[1], cols[0], cols[1]):
        cells = weighted_cells(polygon, first.transform, band, cols)
        _accumulate_support_cells(reference, areas, class_indexes, cells, datasets)
    _validate_support_areas(areas)
    return areas


def _support_grid_dataset(
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> rasterio.DatasetReader:
    if not datasets or not _shares_square_grid(datasets):
        raise ValueError("support reference layers must share one square-cell grid")
    return datasets[0][1]


def _validate_support_areas(areas: Mapping[tuple[str, str], float]) -> None:
    if any(not math.isfinite(area) or area < 0.0 for area in areas.values()):
        raise ArithmeticError("support query produced a non-finite or negative area")


def _accumulate_support_cells(
    reference: _Reference,
    areas: dict[tuple[str, str], float],
    class_indexes: tuple[tuple[tuple[str, str], tuple[int, ...]], ...],
    cells: WeightedCells,
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> None:
    if not len(cells.rows):
        return
    tile_rows = cells.rows // TILE_SIZE
    tile_cols = cells.cols // TILE_SIZE
    tile_ids = tile_rows * 1_000_003 + tile_cols
    order = np.argsort(tile_ids, kind="stable")
    cuts = np.flatnonzero(np.diff(tile_ids[order])) + 1
    for group in np.split(order, cuts):
        stack = reference._tile_stack(
            datasets,
            int(tile_rows[group[0]]),
            int(tile_cols[group[0]]),
        )
        if stack is None:
            continue
        local_rows = cells.rows[group] % TILE_SIZE
        local_cols = cells.cols[group] % TILE_SIZE
        weights = cells.areas[group]
        for class_key, layer_indexes in class_indexes:
            union_mask = np.any(stack[list(layer_indexes)], axis=0)
            areas[class_key] += float(
                union_mask[local_rows, local_cols].astype(np.float64) @ weights
            )


def _normalize_class_groups(
    classes: Mapping[tuple[str, str], Iterable[str]],
) -> tuple[tuple[tuple[str, str], tuple[str, ...]], ...]:
    if not classes:
        raise ValueError("at least one support class must be requested")
    normalized = []
    for class_key, raw_codes in classes.items():
        if not _is_support_class_key(class_key):
            raise ValueError("support class keys must be non-empty (level, code) pairs")
        normalized.append((class_key, _normalize_class_codes(class_key, raw_codes)))
    return tuple(sorted(normalized))


def _is_support_class_key(value: object) -> TypeGuard[tuple[str, str]]:
    return (
        isinstance(value, tuple)
        and len(value) == _SUPPORT_CLASS_KEY_FIELDS
        and all(isinstance(part, str) and part.strip() for part in value)
    )


def _normalize_class_codes(
    class_key: tuple[str, str],
    raw_codes: Iterable[str],
) -> tuple[str, ...]:
    codes = tuple(sorted(set(raw_codes)))
    if not codes or any(not isinstance(code, str) or not code.strip() for code in codes):
        raise ValueError(f"support class {class_key} needs non-empty raster codes")
    return codes


def _class_layer_indexes(
    classes: tuple[tuple[tuple[str, str], tuple[str, ...]], ...],
    datasets: tuple[tuple[RasterLayerLike, rasterio.DatasetReader], ...],
) -> tuple[tuple[tuple[str, str], tuple[int, ...]], ...]:
    indexes_by_code: dict[str, list[int]] = {}
    for index, (layer, _dataset) in enumerate(datasets):
        indexes_by_code.setdefault(layer.code, []).append(index)
    return tuple(
        (class_key, _indexes_for_class(class_key, member_codes, indexes_by_code))
        for class_key, member_codes in classes
    )


def _indexes_for_class(
    class_key: tuple[str, str],
    member_codes: tuple[str, ...],
    indexes_by_code: Mapping[str, list[int]],
) -> tuple[int, ...]:
    missing = sorted(set(member_codes) - indexes_by_code.keys())
    if missing:
        raise ValueError(f"support class {class_key} references missing raster codes: {missing}")
    return tuple(index for code in member_codes for index in indexes_by_code[code])
