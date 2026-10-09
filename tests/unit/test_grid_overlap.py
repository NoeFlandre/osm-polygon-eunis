from itertools import pairwise
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import shapely
from rasterio.io import DatasetReader
from rasterio.transform import Affine, from_origin
from shapely.geometry import MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis import grid_overlap
from osm_polygon_eunis.grid_overlap import (
    _burn,
    _cell_areas,
    _clip_to_cells,
    _dilate,
    _groups,
    band_row_ranges,
    is_square_north_up,
    stack_bytes,
    tile_window,
    weighted_cells,
)

TRANSFORM = from_origin(1000.0, 5000.0, 100.0, 100.0)
SHAPE = (50, 60)


def _brute_force_areas(polygon: Polygon) -> dict[tuple[int, int], float]:
    areas = {}
    for row in range(SHAPE[0]):
        for col in range(SHAPE[1]):
            cell = box(
                1000 + col * 100, 5000 - (row + 1) * 100, 1000 + (col + 1) * 100, 5000 - row * 100
            )
            area = polygon.intersection(cell).area
            if area > 0:
                areas[(row, col)] = area
    return areas


@pytest.mark.slow
@pytest.mark.parametrize(
    "polygon",
    [
        Point(3500, 3000).buffer(1300),
        box(1230, 2210, 4470, 4890),
        Polygon([(1100, 2000), (5000, 2300), (2200, 4900)]),
        Point(3000, 3500).buffer(2000).difference(Point(3000, 3500).buffer(700)),
    ],
)
def test_weighted_cells_match_cell_by_cell_intersection(polygon: Polygon) -> None:
    cells = weighted_cells(polygon, TRANSFORM, (0, SHAPE[0]), (0, SHAPE[1]))

    found = {
        (int(r), int(c)): float(a)
        for r, c, a in zip(cells.rows, cells.cols, cells.areas, strict=True)
    }
    expected = _brute_force_areas(polygon)
    assert found.keys() == expected.keys()
    assert all(found[key] == pytest.approx(expected[key], rel=1e-9) for key in expected)
    assert sum(found.values()) == pytest.approx(
        polygon.intersection(box(1000, -1000, 9000, 5000)).area
    )


def test_weighted_cells_of_an_empty_window_are_empty() -> None:
    cells = weighted_cells(box(0, 0, 1, 1), TRANSFORM, (0, 0), (0, 5))

    assert len(cells.rows) == 0


def test_band_row_ranges_cover_the_window_without_gaps() -> None:
    ranges = band_row_ranges(130, 100_000, 0, 50_000)

    assert ranges[0][0] == 130
    assert ranges[-1][1] == 100_000
    assert all(left[1] == right[0] for left, right in pairwise(ranges))


def _cells(cells) -> dict[tuple[int, int], float]:
    return {
        (int(row), int(col)): float(area)
        for row, col, area in zip(cells.rows, cells.cols, cells.areas, strict=True)
    }


def test_weighted_cells_return_exact_areas_for_an_offset_window() -> None:
    # The polygon covers 2x3 whole cells, plus half-cells on every side and quarter corners.
    polygon = box(1650, 4250, 2050, 4550)

    cells = weighted_cells(polygon, TRANSFORM, (4, 8), (6, 11))

    expected = {(row, col): 10_000.0 for row in (5, 6) for col in (7, 8, 9)}
    expected |= {(row, col): 5_000.0 for row in (4, 7) for col in (7, 8, 9)}
    expected |= {(row, col): 5_000.0 for row in (5, 6) for col in (6, 10)}
    expected |= {(row, col): 2_500.0 for row in (4, 7) for col in (6, 10)}
    assert _cells(cells) == pytest.approx(expected)
    assert cells.rows.dtype == np.int64
    assert cells.cols.dtype == np.int64
    assert cells.areas.dtype == np.float64


def test_weighted_cells_drop_cells_that_only_touch_the_polygon() -> None:
    cells = weighted_cells(box(1000, 4900, 1100, 5000), TRANSFORM, (0, 5), (0, 5))

    assert _cells(cells) == {(0, 0): 10_000.0}


def test_weighted_cells_keep_slivers_smaller_than_one_unit() -> None:
    cells = weighted_cells(box(1000, 4900, 1000.5, 4900.5), TRANSFORM, (0, 3), (0, 3))

    assert _cells(cells) == pytest.approx({(0, 0): 0.25})


@pytest.mark.parametrize(
    ("polygon", "rows", "cols", "expected"),
    [
        (box(1710, 4410, 1790, 4490), (5, 6), (7, 8), {(5, 7): 6_400.0}),
        (
            box(1710, 4410, 2290, 4490),
            (5, 6),
            (7, 10),
            {(5, 7): 7_200.0, (5, 8): 8_000.0, (5, 9): 8_000.0},
        ),
        (
            box(1710, 4110, 1790, 4490),
            (5, 8),
            (7, 8),
            {(5, 7): 7_200.0, (6, 7): 8_000.0, (7, 7): 8_000.0},
        ),
    ],
    ids=["one-cell", "one-row", "one-column"],
)
def test_weighted_cells_cover_degenerate_windows(
    polygon: Polygon,
    rows: tuple[int, int],
    cols: tuple[int, int],
    expected: dict[tuple[int, int], float],
) -> None:
    cells = weighted_cells(polygon, TRANSFORM, rows, cols)

    assert _cells(cells) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("rows", "cols"),
    [((3, 3), (0, 5)), ((0, 5), (3, 3)), ((4, 2), (0, 5)), ((0, 5), (4, 2))],
    ids=["no-rows", "no-columns", "negative-rows", "negative-columns"],
)
def test_weighted_cells_of_empty_windows_have_typed_empty_arrays(
    rows: tuple[int, int], cols: tuple[int, int]
) -> None:
    cells = weighted_cells(box(0, 0, 9000, 9000), TRANSFORM, rows, cols)

    assert [len(cells.rows), len(cells.cols), len(cells.areas)] == [0, 0, 0]
    assert cells.rows.dtype == np.int64
    assert cells.cols.dtype == np.int64
    assert cells.areas.dtype == np.float64


def _vectorized_areas(polygon, transform, shape) -> dict[tuple[int, int], float]:
    rows, cols = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing="ij")
    rows, cols = rows.ravel(), cols.ravel()
    x0 = transform.c + cols * transform.a
    y0 = transform.f + rows * transform.e
    boxes = shapely.box(x0, y0 + transform.e, x0 + transform.a, y0)
    # Exact per-cell areas without clipping every cell: a cell inside the polygon keeps its
    # full area, a cell outside has none, and only cells on the boundary need an intersection.
    shapely.prepare(polygon)
    inside = shapely.contains(polygon, boxes)
    on_boundary = shapely.intersects(polygon, boxes) & ~inside
    areas = np.where(inside, shapely.area(boxes), 0.0)
    areas[on_boundary] = shapely.area(shapely.intersection(boxes[on_boundary], polygon))
    return {
        (int(row), int(col)): float(area)
        for row, col, area in zip(rows, cols, areas, strict=True)
        if area > 0
    }


def test_weighted_cells_match_a_vectorized_overlay_across_strips_and_blocks() -> None:
    # 150 x 170 cells span three row strips (64) and eleven column blocks (16).
    shape = (150, 170)
    transform = from_origin(0.0, 15_000.0, 100.0, 100.0)
    rng = np.random.default_rng(7)
    polygons = [
        Point(8_500, 7_500).buffer(6_000, quad_segs=24),
        Point(8_500, 7_500)
        .buffer(6_000, quad_segs=24)
        .difference(Point(8_000, 7_000).buffer(2_500)),
        *(
            Polygon(
                [(x, y) for x, y in rng.uniform([500, 500], [16_500, 14_500], size=(5, 2))]
            ).convex_hull
            for _ in range(6)
        ),
    ]

    for polygon in polygons:
        cells = weighted_cells(polygon, transform, (0, shape[0]), (0, shape[1]))
        found = _cells(cells)
        expected = _vectorized_areas(polygon, transform, shape)
        assert found.keys() == expected.keys()
        assert all(found[key] == pytest.approx(expected[key], rel=1e-9) for key in expected)


def test_band_row_ranges_are_exact() -> None:
    assert band_row_ranges(0, 1000, 0, 200_000) == [
        (0, 128),
        (128, 256),
        (256, 384),
        (384, 512),
        (512, 640),
        (640, 768),
        (768, 896),
        (896, 1000),
    ]
    assert band_row_ranges(130, 300, 0, 200_000) == [(130, 256), (256, 300)]
    assert band_row_ranges(0, 256, 0, 200_000) == [(0, 128), (128, 256)]
    assert band_row_ranges(0, 20_000_000, 5, 5) == [(0, 16_000_000), (16_000_000, 20_000_000)]
    assert band_row_ranges(0, 20_000_000, 0, 1) == [(0, 16_000_000), (16_000_000, 20_000_000)]
    assert band_row_ranges(0, 20_000_000, 3, 1) == [(0, 16_000_000), (16_000_000, 20_000_000)]
    assert band_row_ranges(10, 20_000, 0, 1000) == [
        (10, 16_000 + 0),
        (16_000, 20_000),
    ]


def test_band_row_ranges_are_plain_integers() -> None:
    ranges = band_row_ranges(10, 20_000, 0, 1000)

    assert ranges == [(10, 16_000), (16_000, 20_000)]
    assert all(type(bound) is int for band in ranges for bound in band)


def test_boundary_cells_are_found_for_circles_that_a_thin_line_raster_misses() -> None:
    shape = (60, 70)
    transform = from_origin(0.0, 6_000.0, 100.0, 100.0)
    for polygon in (
        Point(3198.045, 4609.925).buffer(1422.6, quad_segs=10),
        Point(3221.016, 4093.949).buffer(1909.85, quad_segs=10),
    ):
        found = _cells(weighted_cells(polygon, transform, (0, shape[0]), (0, shape[1])))
        expected = _vectorized_areas(polygon, transform, shape)
        assert found.keys() == expected.keys()
        assert all(found[key] == pytest.approx(expected[key], rel=1e-9) for key in expected)


def test_groups_keep_equal_keys_in_original_order_for_large_inputs() -> None:
    keys = np.random.default_rng(3).integers(0, 7, size=5_000)

    groups = _groups(keys)

    assert len(groups) == 7
    assert sum(len(group) for group in groups) == 5_000
    for key, group in enumerate(groups):
        assert (keys[group] == key).all()
        assert (np.diff(group) > 0).all()


def test_groups_partition_indices_by_equal_keys_in_stable_order() -> None:
    groups = _groups(np.array([2, 0, 2, 1, 0]))

    assert [group.tolist() for group in groups] == [[1, 4], [3], [0, 2]]


def test_dilate_grows_a_mask_by_one_cell_in_every_direction() -> None:
    mask = np.zeros((5, 6), dtype=bool)
    mask[2, 3] = True
    corner = np.zeros((5, 6), dtype=bool)
    corner[0, 0] = True
    far_corner = np.zeros((5, 6), dtype=bool)
    far_corner[4, 5] = True

    expected = np.zeros((5, 6), dtype=bool)
    expected[1:4, 2:5] = True
    expected_corner = np.zeros((5, 6), dtype=bool)
    expected_corner[:2, :2] = True
    expected_far = np.zeros((5, 6), dtype=bool)
    expected_far[3:, 4:] = True
    assert (_dilate(mask) == expected).all()
    assert (_dilate(corner) == expected_corner).all()
    assert (_dilate(far_corner) == expected_far).all()
    assert mask.sum() == 1


def test_burn_rasterizes_geometry_into_a_boolean_mask() -> None:
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    square = box(1, 1, 3, 3)

    inside = _burn(square, transform, (4, 4), all_touched=False)
    touched = _burn(square.boundary, transform, (4, 4), all_touched=True)
    thin = _burn(square.boundary, transform, (4, 4), all_touched=False)
    empty = _burn(Polygon(), transform, (4, 4), all_touched=True)

    expected = np.zeros((4, 4), dtype=bool)
    expected[1:3, 1:3] = True
    assert inside.dtype == np.bool_
    assert (inside == expected).all()
    assert touched.tolist() == [
        [False, False, False, False],
        [False, True, True, True],
        [False, True, False, True],
        [False, True, True, False],
    ]
    assert thin.tolist() == [
        [False, False, False, False],
        [False, True, True, True],
        [False, True, False, True],
        [False, True, True, True],
    ]
    assert empty.dtype == np.bool_
    assert empty.shape == (4, 4)
    assert not empty.any()


def test_cell_areas_split_whole_crossing_and_missed_cells() -> None:
    transform = from_origin(0.0, 3.0, 1.0, 1.0)
    polygon = box(0, 0, 1.5, 3)
    rows = np.array([0, 1, 2, 0])
    cols = np.array([0, 0, 1, 2])

    areas = _cell_areas(polygon, transform, rows, cols)
    empty = _cell_areas(Polygon(), transform, rows, cols)

    assert areas.dtype == np.float64
    assert areas.tolist() == [1.0, 1.0, 0.5, 0.0]
    assert empty.dtype == np.float64
    assert empty.tolist() == [0.0, 0.0, 0.0, 0.0]


def test_cell_areas_batches_exact_intersections_without_rectangle_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    polygon = box(-0.1, -0.1, 3.1, 3.1)
    rows = np.repeat(np.arange(4), 4)
    cols = np.tile(np.arange(4), 4)
    expected = [
        polygon.intersection(box(col, 3 - row, col + 1, 4 - row)).area
        for row, col in zip(rows, cols, strict=True)
    ]
    exact_intersection = shapely.intersection
    intersection_arrays: list[np.ndarray] = []

    def unexpected_rectangle_clip(*_args: object) -> Polygon:
        pytest.fail("cell areas must use exact intersections")

    def counted_intersection(left: object, right: object) -> object:
        if isinstance(left, np.ndarray):
            intersection_arrays.append(left)
        if isinstance(right, np.ndarray):
            intersection_arrays.append(right)
        return exact_intersection(left, right)

    monkeypatch.setattr(shapely, "clip_by_rect", unexpected_rectangle_clip)
    monkeypatch.setattr(shapely, "intersection", counted_intersection)

    areas = _cell_areas(polygon, transform, rows, cols)

    np.testing.assert_allclose(areas, expected, rtol=1e-9, atol=0.0)
    assert len(intersection_arrays) == 1
    assert intersection_arrays[0].size == 7
    assert shapely.is_prepared(polygon)


def test_cell_areas_falls_back_when_fast_clip_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = from_origin(0, 10, 1, 1)
    polygon = Polygon([(-1, -1), (12, 2), (7, 12), (-1, 12)])
    invalid_clip = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert polygon.is_valid and not invalid_clip.is_valid
    rows = np.repeat(np.arange(10), 10)
    cols = np.tile(np.arange(10), 10)
    clip_calls = 0

    def clip_to_invalid_geometry(*_args: object) -> Polygon:
        nonlocal clip_calls
        clip_calls += 1
        return invalid_clip

    monkeypatch.setattr(shapely, "clip_by_rect", clip_to_invalid_geometry)

    areas = _cell_areas(polygon, transform, rows, cols, source_allows_rect_clip=True)
    expected = [
        polygon.intersection(box(col, 9 - row, col + 1, 10 - row)).area
        for row, col in zip(rows, cols, strict=True)
    ]

    assert clip_calls > 0
    np.testing.assert_allclose(areas, expected, rtol=1e-9, atol=0.0)


def test_boundary_areas_keeps_original_clipping_eligibility_across_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    polygon = box(-1.0, -65.0, 20.0, 3.0)
    rows = np.array([0, 0, 64, 64])
    cols = np.array([0, 16, 0, 16])
    rectangle_clip = shapely.clip_by_rect
    clip_input_types: list[str] = []

    def return_multipolygon_after_source_clip(
        geometry: BaseGeometry, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> BaseGeometry:
        clip_input_types.append(geometry.geom_type)
        if isinstance(geometry, Polygon):
            return MultiPolygon(
                [
                    box(x_min + 0.1, y_min + 0.1, x_min + 0.4, y_max - 0.1),
                    box(x_min + 0.6, y_min + 0.1, x_min + 0.9, y_max - 0.1),
                ]
            )
        return rectangle_clip(geometry, x_min, y_min, x_max, y_max)

    monkeypatch.setattr(shapely, "clip_by_rect", return_multipolygon_after_source_clip)

    grid_overlap._boundary_areas(polygon, transform, rows, cols)

    assert clip_input_types.count("Polygon") == 2
    assert clip_input_types.count("MultiPolygon") >= 2


def test_boundary_areas_matches_oracle_after_simple_clip_splits_multipolygon(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = from_origin(0.0, 4.0, 1.0, 1.0)
    polygon = Polygon([(0, 0), (0, 4), (1.5, 4), (1.5, 1), (3.5, 1), (3.5, 4), (5, 4), (5, 0)])
    rows = np.array([0, 0, 1, 1])
    cols = np.array([1, 3, 1, 3])
    rectangle_clip = shapely.clip_by_rect
    input_types: list[str] = []
    output_types: list[str] = []

    def record_clips(
        geometry: BaseGeometry, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> BaseGeometry:
        input_types.append(geometry.geom_type)
        clipped = rectangle_clip(geometry, x_min, y_min, x_max, y_max)
        output_types.append(clipped.geom_type)
        return clipped

    monkeypatch.setattr(shapely, "clip_by_rect", record_clips)

    areas = grid_overlap._boundary_areas(polygon, transform, rows, cols)
    expected = np.array(
        [
            polygon.intersection(box(col, 3 - row, col + 1, 4 - row)).area
            for row, col in zip(rows, cols, strict=True)
        ]
    )

    np.testing.assert_allclose(areas, expected, rtol=1e-9, atol=0.0)
    assert output_types[0] == "MultiPolygon"
    assert "MultiPolygon" in input_types


def test_boundary_areas_reuses_strip_and_block_clips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transform = from_origin(0.0, 2.0, 1.0, 1.0)
    polygon = box(-0.5, -63.5, 17.5, 1.5)
    rows = np.repeat(np.array([0, 1, 64, 65]), 4)
    cols = np.tile(np.array([0, 1, 16, 17]), 4)
    rectangle_clip = shapely.clip_by_rect
    clip_bounds: list[tuple[float, float, float, float]] = []

    def count_rectangle_clips(
        geometry: BaseGeometry, x_min: float, y_min: float, x_max: float, y_max: float
    ) -> BaseGeometry:
        clip_bounds.append((x_min, y_min, x_max, y_max))
        return rectangle_clip(geometry, x_min, y_min, x_max, y_max)

    monkeypatch.setattr(shapely, "clip_by_rect", count_rectangle_clips)

    areas = grid_overlap._boundary_areas(polygon, transform, rows, cols)
    expected = [
        polygon.intersection(box(col, 1 - row, col + 1, 2 - row)).area
        for row, col in zip(rows, cols, strict=True)
    ]

    np.testing.assert_allclose(areas, expected, rtol=1e-9, atol=0.0)
    group_clips = [bounds for bounds in clip_bounds if bounds[2] - bounds[0] > 1]
    cell_clips = [bounds for bounds in clip_bounds if bounds[2] - bounds[0] == 1]
    assert len(group_clips) == 6
    assert cell_clips


def test_stack_bytes_adds_the_array_size_to_the_entry_overhead() -> None:
    assert stack_bytes(None) == 256
    assert stack_bytes(np.zeros(10, dtype=np.float64)) == 256 + 80


def test_tile_window_is_clipped_at_the_raster_edge() -> None:
    dataset = SimpleNamespace(height=300, width=500)

    inner = tile_window(cast(DatasetReader, dataset), 1, 2)
    corner = tile_window(cast(DatasetReader, dataset), 2, 3)

    assert (inner.row_off, inner.col_off, inner.height, inner.width) == (128, 256, 128, 128)
    assert (corner.row_off, corner.col_off, corner.height, corner.width) == (256, 384, 44, 116)


@pytest.mark.parametrize(
    ("transform", "expected"),
    [
        (Affine(10, 0, 0, 0, -10, 0), True),
        (Affine(10, 0, 0, 0, 10, 0), True),
        (Affine(10, 0, 0, 0, -20, 0), False),
        (Affine(10, 0.5, 0, 0, -10, 0), False),
        (Affine(10, 0, 0, 0.5, -10, 0), False),
        (Affine(10, 1, 0, 0, -10, 0), False),
        (Affine(10, 0, 0, 1, -10, 0), False),
        (Affine(10, 1, 0, 1, -10, 0), False),
    ],
)
def test_is_square_north_up(transform: Affine, expected: bool) -> None:
    assert is_square_north_up(transform) is expected


def _exact_cell_areas(
    polygon: BaseGeometry, transform: Affine, rows: tuple[int, int], cols: tuple[int, int]
) -> dict[tuple[int, int], float]:
    row_ids, col_ids = np.meshgrid(np.arange(*rows), np.arange(*cols), indexing="ij")
    row_ids, col_ids = row_ids.ravel(), col_ids.ravel()
    x0 = transform.c + col_ids * transform.a
    x1 = transform.c + (col_ids + 1) * transform.a
    y0 = transform.f + row_ids * transform.e
    y1 = transform.f + (row_ids + 1) * transform.e
    cells = shapely.box(
        np.minimum(x0, x1), np.minimum(y0, y1), np.maximum(x0, x1), np.maximum(y0, y1)
    )
    areas = shapely.area(shapely.intersection(polygon, cells))
    return {
        (int(row), int(col)): float(area)
        for row, col, area in zip(row_ids, col_ids, areas, strict=True)
        if area > 0.0
    }


@pytest.mark.parametrize(
    ("polygon", "transform", "rows", "cols"),
    [
        (
            Polygon(
                [(0, 0), (40, 0), (40, 40), (0, 40)],
                [[(10, 10), (21, 21), (22, 15), (24, 40)]],
            ),
            from_origin(0, 40, 5, 5),
            (0, 8),
            (0, 8),
        ),
        (
            Polygon(
                [(0, 0), (40, 0), (40, 100), (0, 100)],
                [[(10, 30), (21, 41), (22, 35), (24, 100)]],
            ),
            from_origin(0, 100, 1, 1),
            (0, 100),
            (0, 40),
        ),
        (
            MultiPolygon([box(0, 0, 4, 4), box(4, 4, 8, 8)]),
            from_origin(0, 8, 1, 1),
            (0, 8),
            (0, 8),
        ),
        (
            Polygon([(0.25, 0.5), (8.75, 1.25), (7.5, 8.5), (1.5, 7.75)]),
            from_origin(0, 10, 1, 1),
            (0, 10),
            (0, 10),
        ),
        (
            Polygon([(4, 4), (28, 4), (28, 92), (20, 92), (20, 20), (12, 20), (12, 92), (4, 92)]),
            from_origin(0, 96, 1, 1),
            (4, 92),
            (4, 28),
        ),
        (box(100, 100, 101, 101), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (
            box(126.25, 126.25, 129.75, 129.75),
            from_origin(0, 256, 1, 1),
            (126, 131),
            (126, 131),
        ),
        (box(0, 2, 4, 8), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(2, 0, 8, 4), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(6, 2, 10, 8), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(2, 6, 8, 10), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(0, 0, 10, 10), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(-3, 1, 6, 9), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(3, 1, 13, 9), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(1, -3, 9, 6), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
        (box(1, 3, 9, 13), from_origin(0, 10, 1, 1), (0, 10), (0, 10)),
    ],
    ids=[
        "touching-hole",
        "touching-hole-strip-and-block-clips",
        "touching-multipolygon-parts",
        "ordinary",
        "ordinary-concave",
        "empty",
        "tile-boundary",
        "equal-left-bound",
        "equal-bottom-bound",
        "equal-right-bound",
        "equal-top-bound",
        "equal-all-bounds",
        "left-protrusion",
        "right-protrusion",
        "bottom-protrusion",
        "top-protrusion",
    ],
)
def test_weighted_cells_conserve_exact_intersection_area(
    polygon: BaseGeometry,
    transform: Affine,
    rows: tuple[int, int],
    cols: tuple[int, int],
) -> None:
    assert polygon.is_valid
    cells = weighted_cells(polygon, transform, rows, cols)
    found = _cells(cells)
    expected = _exact_cell_areas(polygon, transform, rows, cols)

    assert found.keys() == expected.keys()
    assert all(found[key] == pytest.approx(expected[key], rel=1e-9) for key in expected)
    x0 = transform.c + cols[0] * transform.a
    x1 = transform.c + cols[1] * transform.a
    y0 = transform.f + rows[0] * transform.e
    y1 = transform.f + rows[1] * transform.e
    grid = box(min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    assert sum(found.values()) == pytest.approx(polygon.intersection(grid).area, rel=1e-9)


def test_simple_polygon_fast_clips_match_independent_intersection_oracle() -> None:
    transform = from_origin(0, 32, 1, 1)
    rows = (2, 30)
    cols = (2, 30)
    angles = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
    rng = np.random.default_rng(129)

    for _ in range(12):
        radii = rng.uniform(2.0, 12.0, size=len(angles))
        polygon = Polygon(
            np.column_stack((16.0 + radii * np.cos(angles), 16.0 + radii * np.sin(angles)))
        )
        assert polygon.is_valid and not polygon.interiors

        found = _cells(weighted_cells(polygon, transform, rows, cols))
        expected = _exact_cell_areas(polygon, transform, rows, cols)

        assert found.keys() == expected.keys()
        assert all(found[key] == pytest.approx(expected[key], rel=1e-9) for key in expected)


def test_invalid_rectangle_clips_fall_back_to_exact_intersections(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = Polygon([(-1, -1), (12, 2), (7, 12), (-1, 12)])
    transform = from_origin(0, 10, 1, 1)
    invalid_clip = Polygon([(0, 0), (2, 2), (0, 2), (2, 0)])
    assert polygon.is_valid
    assert not invalid_clip.is_valid
    clip_calls = 0

    def clip_to_invalid_geometry(*_args: object) -> Polygon:
        nonlocal clip_calls
        clip_calls += 1
        return invalid_clip

    monkeypatch.setattr(shapely, "clip_by_rect", clip_to_invalid_geometry)

    clipped = _clip_to_cells(
        polygon,
        transform,
        np.array([0, 9]),
        np.array([0, 9]),
        source_allows_rect_clip=True,
    )
    window = box(0, 0, 10, 10)
    expected_clip = polygon.intersection(window)
    assert clip_calls == 1
    assert clipped.symmetric_difference(expected_clip).area == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("polygon", "transform", "expected"),
    [
        (box(-1, 0.5, 1, 1.5), Affine(2, 0, 0, 0, -2, 2), box(0, 0.5, 1, 1.5)),
        (box(0.5, 1.5, 1.5, 3), Affine(2, 0, 0, 0, -2, 2), box(0.5, 1.5, 1.5, 2)),
        (box(0.5, -1, 1.5, 1), Affine(2, 0, 0, 0, -2, 2), box(0.5, 0, 1.5, 1)),
        (box(1, 0.5, 3, 1.5), Affine(2, 0, 0, 0, -2, 2), box(1, 0.5, 2, 1.5)),
    ],
    ids=["left", "top", "bottom", "right"],
)
def test_clip_to_cells_clips_each_protruding_bound(
    polygon: BaseGeometry,
    transform: Affine,
    expected: BaseGeometry,
) -> None:
    clipped = _clip_to_cells(polygon, transform, np.array([0]), np.array([0]))

    assert clipped.symmetric_difference(expected).area == pytest.approx(0.0)


def test_clip_to_cells_skips_overlay_when_all_bounds_are_equal(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    polygon = box(10, 0, 12, 2)

    def unexpected_intersection(*_args: object) -> BaseGeometry:
        pytest.fail("a polygon on the cell envelope should not need overlay")

    monkeypatch.setattr(shapely, "intersection", unexpected_intersection)

    clipped = _clip_to_cells(polygon, Affine(2, 0, 10, 0, 2, 0), np.array([0]), np.array([0]))

    assert clipped.equals(polygon)


def test_strip_and_block_clips_preserve_touching_hole_intersections() -> None:
    transform = from_origin(0, 12, 1, 1)
    polygon = Polygon(
        [(0, 0), (12, 0), (12, 12), (0, 12)],
        [[(6, 2), (10, 6), (6, 10), (2, 6)]],
    )
    strip = _clip_to_cells(polygon, transform, np.array([0, 0, 11, 11]), np.array([5, 9, 5, 9]))
    expected_strip = polygon.intersection(box(5, 0, 10, 12))
    block = _clip_to_cells(strip, transform, np.array([9, 9, 11, 11]), np.array([5, 6, 5, 6]))
    expected_block = polygon.intersection(box(5, 0, 7, 3))

    assert strip.is_valid
    assert strip.symmetric_difference(expected_strip).area == pytest.approx(0.0)
    assert block.is_valid
    assert block.symmetric_difference(expected_block).area == pytest.approx(0.0)
