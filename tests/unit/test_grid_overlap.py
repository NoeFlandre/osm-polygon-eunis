from itertools import pairwise


import pytest
import shapely
from rasterio.transform import from_origin
from shapely.geometry import Point, Polygon, box

from osm_polygon_eunis.grid_overlap import band_row_ranges, weighted_cells

TRANSFORM = from_origin(1000.0, 5000.0, 100.0, 100.0)
SHAPE = (60, 80)


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


@pytest.mark.parametrize(
    "polygon",
    [
        Point(3500, 3000).buffer(1300),
        box(1230, 2210, 4470, 4890),
        Polygon([(1100, 2000), (5000, 2300), (2200, 4900)]),
        Point(3000, 3500).buffer(2000).difference(Point(3000, 3500).buffer(700)),
        box(1000, 5000 - 6000, 1000 + 8000, 5000),
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


def test_shapely_is_the_exact_reference() -> None:
    assert shapely.area(box(0, 0, 2, 3)) == 6.0
