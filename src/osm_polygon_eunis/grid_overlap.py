"""Exact polygon/cell overlap areas on a regular raster grid, without vectorizing cells.

For a polygon and a grid of equal square cells, every cell the polygon covers
completely contributes the whole cell area, and only the cells crossed by the
polygon boundary need a geometric intersection. The boundary cells are the same
for every habitat layer on the grid, so their exact areas are computed once per
polygon and each layer only selects the cells it marks as habitat.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from rasterio.features import rasterize
from rasterio.io import DatasetReader
from rasterio.transform import Affine
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry

TILE_SIZE = 128
_CACHE_ENTRY_BYTES = 256
# A boundary cell block is clipped once so the overlay never sees the whole polygon.
_BLOCK_CELLS = 16
_STRIP_CELLS = 64
# Bands bound the rasterized window (cells) so a continent-sized polygon stays in memory.
MAX_BAND_CELLS = 16_000_000


@dataclass(frozen=True, slots=True)
class WeightedCells:
    """Cells of one grid band with the polygon area each of them contains."""

    rows: np.ndarray
    cols: np.ndarray
    areas: np.ndarray


def band_row_ranges(
    row_start: int, row_end: int, col_start: int, col_end: int
) -> list[tuple[int, int]]:
    """Split a cell window into tile-aligned row bands of bounded size."""

    width = max(1, col_end - col_start)
    tiles_per_band = max(1, MAX_BAND_CELLS // (width * TILE_SIZE))
    step = tiles_per_band * TILE_SIZE
    first = (row_start // TILE_SIZE) * TILE_SIZE
    return [
        (max(start, row_start), min(start + step, row_end)) for start in range(first, row_end, step)
    ]


def weighted_cells(
    polygon: BaseGeometry,
    transform: Affine,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> WeightedCells:
    """Return every cell in the window that overlaps the polygon, with its overlap area.

    ``transform`` maps global (col, row) cell indices to coordinates. Rows and
    columns are half-open global cell index ranges.
    """

    row_start, row_end = rows
    col_start, col_end = cols
    height = row_end - row_start
    width = col_end - col_start
    if height <= 0 or width <= 0:
        return _empty_cells()
    window_transform = transform @ Affine.translation(col_start, row_start)
    cell_area = abs(transform.a * transform.e)
    inside = _burn(polygon, window_transform, (height, width), all_touched=False)
    touched = _burn(polygon.boundary, window_transform, (height, width), all_touched=True)
    boundary = _dilate(touched)
    full = inside & ~boundary
    full_rows, full_cols = np.nonzero(full)
    edge_rows, edge_cols = np.nonzero(boundary)
    edge_areas = _boundary_areas(polygon, window_transform, edge_rows, edge_cols)
    keep = edge_areas > 0.0
    return WeightedCells(
        np.concatenate((full_rows, edge_rows[keep])) + row_start,
        np.concatenate((full_cols, edge_cols[keep])) + col_start,
        np.concatenate((np.full(len(full_rows), cell_area), edge_areas[keep])),
    )


def _empty_cells() -> WeightedCells:
    return WeightedCells(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    )


def _burn(
    geometry: BaseGeometry,
    transform: Affine,
    shape: tuple[int, int],
    *,
    all_touched: bool,
) -> np.ndarray:
    if geometry.is_empty:
        return np.zeros(shape, dtype=bool)
    burned = rasterize(
        [(geometry, 1)],
        out_shape=shape,
        transform=transform,
        fill=0,
        all_touched=all_touched,
        dtype="uint8",
    )
    return burned.astype(bool, copy=False)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """Grow a boolean mask by one cell in every direction (including diagonals)."""

    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    side = grown.copy()
    grown[:, 1:] |= side[:, :-1]
    grown[:, :-1] |= side[:, 1:]
    return grown


def _boundary_areas(
    polygon: BaseGeometry,
    transform: Affine,
    rows: np.ndarray,
    cols: np.ndarray,
) -> np.ndarray:
    """Exact polygon area inside each listed cell.

    The polygon is narrowed hierarchically (row strip, then block) so each
    exact intersection only sees the vertices near its cells.
    """

    areas = np.zeros(len(rows), dtype=np.float64)
    if not len(rows):
        return areas
    allow_rect_clip = _simple_polygon_without_holes(polygon)
    for strip in _groups(rows // _STRIP_CELLS):
        areas[strip] = _strip_areas(
            polygon, transform, rows[strip], cols[strip], allow_rect_clip=allow_rect_clip
        )
    return areas


def _strip_areas(
    polygon: BaseGeometry,
    transform: Affine,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    allow_rect_clip: bool,
) -> np.ndarray:
    areas = np.zeros(len(rows), dtype=np.float64)
    strip_polygon = _clip_to_cells(polygon, transform, rows, cols, allow_rect_clip=allow_rect_clip)
    for block in _groups(cols // _BLOCK_CELLS):
        block_polygon = _clip_to_cells(
            strip_polygon,
            transform,
            rows[block],
            cols[block],
            allow_rect_clip=allow_rect_clip,
        )
        areas[block] = _cell_areas(
            block_polygon,
            transform,
            rows[block],
            cols[block],
            allow_rect_clip=allow_rect_clip,
        )
    return areas


def _groups(keys: np.ndarray) -> list[np.ndarray]:
    """Return index groups of equal keys."""

    order = np.argsort(keys, kind="stable")
    return np.split(order, np.flatnonzero(np.diff(keys[order])) + 1)


def _clip_to_cells(
    polygon: BaseGeometry,
    transform: Affine,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    allow_rect_clip: bool = False,
) -> BaseGeometry:
    bounds = shapely.bounds(_cell_boxes(transform, rows, cols))
    x_min = float(bounds[:, 0].min())
    y_min = float(bounds[:, 1].min())
    x_max = float(bounds[:, 2].max())
    y_max = float(bounds[:, 3].max())
    polygon_bounds = shapely.bounds(polygon)
    if (
        polygon_bounds[0] >= x_min
        and polygon_bounds[1] >= y_min
        and polygon_bounds[2] <= x_max
        and polygon_bounds[3] <= y_max
    ):
        return polygon
    if allow_rect_clip:
        clipped = shapely.clip_by_rect(polygon, x_min, y_min, x_max, y_max)
        if clipped.is_valid:
            return clipped
    clip_box = shapely.box(x_min, y_min, x_max, y_max)
    return shapely.intersection(polygon, clip_box)


def _cell_areas(
    polygon: BaseGeometry,
    transform: Affine,
    rows: np.ndarray,
    cols: np.ndarray,
    *,
    allow_rect_clip: bool = False,
) -> np.ndarray:
    """Polygon area per cell: whole cells by predicate, crossing cells by exact overlay."""

    boxes = _cell_boxes(transform, rows, cols)
    if polygon.is_empty:
        return np.zeros(len(rows), dtype=np.float64)
    shapely.prepare(polygon)
    inside = shapely.contains(polygon, boxes)
    areas = np.where(inside, shapely.area(boxes), 0.0)
    crossing = np.flatnonzero(shapely.intersects(polygon, boxes) & ~inside)
    if len(crossing):
        if allow_rect_clip:
            bounds = shapely.bounds(boxes[crossing])
            for index, (x_min, y_min, x_max, y_max) in zip(crossing, bounds, strict=True):
                clipped = shapely.clip_by_rect(polygon, x_min, y_min, x_max, y_max)
                if clipped.is_valid:
                    areas[index] = shapely.area(clipped)
                else:
                    areas[index] = shapely.area(shapely.intersection(boxes[index], polygon))
        else:
            areas[crossing] = shapely.area(shapely.intersection(boxes[crossing], polygon))
    return np.asarray(areas, dtype=np.float64)


def _simple_polygon_without_holes(polygon: BaseGeometry) -> bool:
    """Identify a valid single-ring polygon for the rectangle-clipping fast path."""

    return isinstance(polygon, Polygon) and polygon.is_valid and not polygon.interiors


def _cell_boxes(transform: Affine, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    x0 = transform.c + cols * transform.a
    x1 = transform.c + (cols + 1) * transform.a
    y0 = transform.f + rows * transform.e
    y1 = transform.f + (rows + 1) * transform.e
    return shapely.box(
        np.minimum(x0, x1), np.minimum(y0, y1), np.maximum(x0, x1), np.maximum(y0, y1)
    )


def stack_bytes(stack: np.ndarray | None) -> int:
    """Estimate the memory a cached tile stack (or an empty marker) retains."""

    return _CACHE_ENTRY_BYTES + (0 if stack is None else int(stack.nbytes))


def tile_window(dataset: DatasetReader, row: int, column: int) -> Window:
    """Return the raster window of one tile, clipped at the raster edge."""

    row_start, col_start = row * TILE_SIZE, column * TILE_SIZE
    return Window.from_slices(
        (row_start, min(row_start + TILE_SIZE, dataset.height)),
        (col_start, min(col_start + TILE_SIZE, dataset.width)),
    )


def is_square_north_up(transform: Affine) -> bool:
    """Return whether cells are axis-aligned squares (north-up, no rotation)."""

    return abs(transform.a) == abs(transform.e) and transform.b == 0 and transform.d == 0
