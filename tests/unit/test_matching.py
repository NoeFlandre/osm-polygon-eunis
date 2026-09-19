import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import box

from osm_polygon_eunis.domain import OverlapCandidate
from osm_polygon_eunis.matching import choose_winner


def test_largest_actual_intersection_and_percentage() -> None:
    polygon = box(0, 0, 10, 10)
    candidates = (
        OverlapCandidate("R11", "Pannonian steppe", box(0, 0, 8, 10)),
        OverlapCandidate("R12", "Other grassland", box(0, 0, 7, 10)),
    )

    result = choose_winner(polygon, candidates, source_version="test")

    assert result.code == "R11"
    assert result.name == "Pannonian steppe"
    assert result.overlap_percentage == 80.0


def test_bbox_touch_without_geometry_intersection_is_null() -> None:
    result = choose_winner(
        box(0, 0, 1, 1),
        (OverlapCandidate("R11", "steppe", box(1, 1, 2, 2)),),
        source_version="test",
    )

    assert result.is_empty


def test_equal_area_tie_uses_ascending_code() -> None:
    polygon = box(0, 0, 10, 10)
    candidates = (
        OverlapCandidate("R12", "second", box(5, 0, 10, 10)),
        OverlapCandidate("R11", "first", box(0, 0, 5, 10)),
    )

    assert choose_winner(polygon, candidates, source_version="test").code == "R11"


def test_empty_polygon_returns_all_null_fields() -> None:
    result = choose_winner(box(0, 0, 0, 0), (), source_version="test")

    assert result.is_empty
    assert result.code is None
    assert result.name is None
    assert result.overlap_percentage is None
    assert result.source_version is None


@st.composite
def _rectangles(draw: st.DrawFn):
    x = draw(st.integers(-20, 20))
    y = draw(st.integers(-20, 20))
    width = draw(st.integers(1, 20))
    height = draw(st.integers(1, 20))
    return box(x, y, x + width, y + height)


@pytest.mark.property
@given(_rectangles(), st.lists(_rectangles(), min_size=1, max_size=5))
def test_overlap_percentage_is_bounded(polygon, geometries) -> None:
    candidates = tuple(
        OverlapCandidate(f"R{index:02d}", f"habitat-{index}", geometry)
        for index, geometry in enumerate(geometries)
    )

    result = choose_winner(polygon, candidates, source_version="test")

    assert result.overlap_percentage is None or 0.0 <= result.overlap_percentage <= 100.0


@pytest.mark.property
@given(_rectangles(), st.lists(_rectangles(), min_size=1, max_size=5))
def test_candidate_permutation_does_not_change_result(polygon, geometries) -> None:
    candidates = tuple(
        OverlapCandidate(f"R{index:02d}", f"habitat-{index}", geometry)
        for index, geometry in enumerate(geometries)
    )

    assert choose_winner(polygon, candidates, source_version="test") == choose_winner(
        polygon, tuple(reversed(candidates)), source_version="test"
    )
