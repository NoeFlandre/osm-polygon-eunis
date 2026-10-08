from __future__ import annotations

import os
from unittest.mock import patch

import numpy as np
import pytest
import shapely
from hypothesis import given, settings
from hypothesis import strategies as st
from rasterio.transform import from_origin
from shapely.geometry import CAP_STYLE, LineString, MultiPoint, Polygon, box
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis import grid_overlap
from osm_polygon_eunis.grid_overlap import WeightedCells, band_row_ranges, weighted_cells

ROWS = 12
COLS = 16
TRANSFORM = from_origin(0.0, float(ROWS), 1.0, 1.0)
# Areas below this are floating-point noise from edges that only touch a cell.
NOISE = 1e-9
# The default pytest run has a twelve-second budget (.github/workflows/qa.yml), so outside
# the ci profile these properties run 50 examples. `make test` (HYPOTHESIS_PROFILE=ci) keeps 300.
_PROPERTY_SETTINGS = (
    settings() if os.environ.get("HYPOTHESIS_PROFILE") == "ci" else settings(max_examples=50)
)

# Coordinates are drawn in eighths of a cell so edges often land on cell borders.
_X_EIGHTHS = st.integers(min_value=-8, max_value=136)
_Y_EIGHTHS = st.integers(min_value=-8, max_value=104)


def _small_bands() -> list[tuple[int, int]]:
    """Split the test window into bands of 8 and 4 rows, so polygons cross a band edge."""

    with (
        patch.object(grid_overlap, "TILE_SIZE", 4),
        patch.object(grid_overlap, "MAX_BAND_CELLS", 2 * 4 * COLS),
    ):
        return band_row_ranges(0, ROWS, 0, COLS)


_BANDS = _small_bands()


@st.composite
def _rectangles(draw: st.DrawFn) -> Polygon:
    x0 = draw(_X_EIGHTHS) / 8
    y0 = draw(_Y_EIGHTHS) / 8
    width = draw(st.integers(min_value=1, max_value=160)) / 8
    height = draw(st.integers(min_value=1, max_value=120)) / 8
    return box(x0, y0, x0 + width, y0 + height)


@st.composite
def _slivers(draw: st.DrawFn) -> Polygon:
    """Thin axis-aligned strips, narrower than a cell."""

    x0 = draw(_X_EIGHTHS) / 8
    y0 = draw(_Y_EIGHTHS) / 8
    width = draw(st.integers(min_value=1, max_value=8)) / 64
    height = draw(st.integers(min_value=8, max_value=120)) / 8
    return box(x0, y0, x0 + width, y0 + height)


@st.composite
def _diagonal_slivers(draw: st.DrawFn) -> Polygon:
    """Thin buffered segments that cross cells at an angle."""

    x0, y0 = draw(_X_EIGHTHS) / 8, draw(_Y_EIGHTHS) / 8
    x1, y1 = draw(_X_EIGHTHS) / 8, draw(_Y_EIGHTHS) / 8
    half_width = draw(st.integers(min_value=1, max_value=4)) / 64
    return LineString([(x0, y0), (x1, y1)]).buffer(half_width, cap_style=CAP_STYLE.flat)


@st.composite
def _polygons_with_holes(draw: st.DrawFn) -> BaseGeometry:
    """A rectangle minus an inner rectangle, which may touch the outer edge."""

    x0, y0 = draw(_X_EIGHTHS), draw(_Y_EIGHTHS)
    width = draw(st.integers(min_value=4, max_value=64))
    height = draw(st.integers(min_value=4, max_value=48))
    hole_x0 = draw(st.integers(min_value=0, max_value=width - 2))
    hole_x1 = draw(st.integers(min_value=hole_x0 + 1, max_value=width))
    hole_y0 = draw(st.integers(min_value=0, max_value=height - 2))
    hole_y1 = draw(st.integers(min_value=hole_y0 + 1, max_value=height))
    outer = box(x0 / 8, y0 / 8, (x0 + width) / 8, (y0 + height) / 8)
    hole = box((x0 + hole_x0) / 8, (y0 + hole_y0) / 8, (x0 + hole_x1) / 8, (y0 + hole_y1) / 8)
    return outer.difference(hole)


@st.composite
def _unions(draw: st.DrawFn) -> BaseGeometry:
    """Two rectangles that may be disjoint, touching or overlapping (a multipolygon)."""

    return shapely.union(draw(_rectangles()), draw(_rectangles()))


@st.composite
def _hulls(draw: st.DrawFn) -> BaseGeometry:
    points = [
        (draw(_X_EIGHTHS) / 8, draw(_Y_EIGHTHS) / 8)
        for _ in range(draw(st.integers(min_value=3, max_value=6)))
    ]
    return MultiPoint(points).convex_hull


def _is_usable(geometry: BaseGeometry) -> bool:
    return (
        geometry.geom_type in {"Polygon", "MultiPolygon"}
        and geometry.is_valid
        and geometry.area > 0.0
    )


_POLYGONS = st.one_of(
    _rectangles(),
    _slivers(),
    _diagonal_slivers(),
    _polygons_with_holes(),
    _unions(),
    _hulls(),
).filter(_is_usable)


@st.composite
def _windows(draw: st.DrawFn) -> tuple[tuple[int, int], tuple[int, int]]:
    row_start = draw(st.integers(min_value=0, max_value=ROWS - 1))
    row_end = draw(st.integers(min_value=row_start + 1, max_value=ROWS))
    col_start = draw(st.integers(min_value=0, max_value=COLS - 1))
    col_end = draw(st.integers(min_value=col_start + 1, max_value=COLS))
    return (row_start, row_end), (col_start, col_end)


def _cell_box_areas(
    polygon: BaseGeometry, rows: tuple[int, int], cols: tuple[int, int]
) -> dict[tuple[int, int], float]:
    """Intersect each explicit cell box with the polygon and keep the non-noise areas."""

    row_ids, col_ids = np.meshgrid(np.arange(*rows), np.arange(*cols), indexing="ij")
    row_ids, col_ids = row_ids.ravel(), col_ids.ravel()
    left = TRANSFORM.c + col_ids * TRANSFORM.a
    top = TRANSFORM.f + row_ids * TRANSFORM.e
    boxes = shapely.box(left, top + TRANSFORM.e, left + TRANSFORM.a, top)
    areas = shapely.area(shapely.intersection(boxes, polygon))
    return {
        (int(row), int(col)): float(area)
        for row, col, area in zip(row_ids, col_ids, areas, strict=True)
        if area > NOISE
    }


def _significant(cells: WeightedCells) -> dict[tuple[int, int], float]:
    return {
        (int(row), int(col)): float(area)
        for row, col, area in zip(cells.rows, cells.cols, cells.areas, strict=True)
        if area > NOISE
    }


def _assert_matches_cell_boxes(
    polygon: BaseGeometry,
    cells: WeightedCells,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> None:
    assert (cells.areas > -NOISE).all()
    assert ((cells.rows >= rows[0]) & (cells.rows < rows[1])).all()
    assert ((cells.cols >= cols[0]) & (cells.cols < cols[1])).all()

    found = _significant(cells)
    expected = _cell_box_areas(polygon, rows, cols)
    assert found.keys() == expected.keys()
    for key, area in expected.items():
        assert found[key] == pytest.approx(area, abs=NOISE)

    window = box(cols[0], ROWS - rows[1], cols[1], ROWS - rows[0])
    assert sum(found.values()) == pytest.approx(polygon.intersection(window).area, abs=NOISE)


@pytest.mark.property
@_PROPERTY_SETTINGS
@given(polygon=_POLYGONS, window=_windows())
def test_weighted_cells_match_explicit_cell_box_intersections(
    polygon: BaseGeometry,
    window: tuple[tuple[int, int], tuple[int, int]],
) -> None:
    rows, cols = window

    cells = weighted_cells(polygon, TRANSFORM, rows, cols)

    _assert_matches_cell_boxes(polygon, cells, rows, cols)


@pytest.mark.property
@_PROPERTY_SETTINGS
@given(polygon=_POLYGONS)
def test_weighted_cells_match_explicit_cell_box_intersections_in_each_band(
    polygon: BaseGeometry,
) -> None:
    for rows in _BANDS:
        cells = weighted_cells(polygon, TRANSFORM, rows, (0, COLS))

        _assert_matches_cell_boxes(polygon, cells, rows, (0, COLS))


def test_small_bands_split_the_window_so_polygons_can_cross_a_boundary() -> None:
    assert _BANDS == [(0, 8), (8, 12)]
