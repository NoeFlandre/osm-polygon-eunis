import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import GeometryCollection, Polygon, box

from osm_polygon_eunis.domain import EunisResult, OverlapCandidate
from osm_polygon_eunis.matching import choose_winner, prefer_result


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
    assert result.source_version == "test"


def test_disjoint_cell_collection_sums_exact_intersections() -> None:
    polygon = box(0.5, 0.5, 2.5, 1.5)
    cells = (box(0, 0, 1, 2), box(2, 0, 3, 2))

    result = choose_winner(
        polygon,
        (
            OverlapCandidate(
                "R11",
                "steppe",
                GeometryCollection(cells),
                components_are_disjoint=True,
            ),
        ),
        source_version="test",
    )

    assert result.code == "R11"
    assert result.overlap_percentage == 50.0


def test_empty_disjoint_cell_collection_has_no_overlap() -> None:
    candidate = OverlapCandidate(
        "R11",
        "steppe",
        GeometryCollection(),
        components_are_disjoint=True,
    )

    result = choose_winner(
        box(0, 0, 1, 1),
        (candidate,),
        source_version="test",
    )

    assert result.is_empty


def test_overlapping_generic_collection_keeps_union_semantics() -> None:
    cells = GeometryCollection((box(0, 0, 2, 2), box(1, 0, 3, 2)))

    result = choose_winner(
        box(0, 0, 3, 2),
        (OverlapCandidate("R11", "steppe", cells),),
        source_version="test",
    )

    assert result.overlap_percentage == 100.0


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


@pytest.mark.parametrize(
    "polygon",
    [
        None,
        box(0, 0, 0, 0),
        Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)]),
        box(0, 0, 1, 1).boundary,
    ],
    ids=["none", "zero-area", "invalid", "line"],
)
def test_unusable_polygon_returns_all_null_fields(polygon) -> None:
    result = choose_winner(
        polygon,
        (OverlapCandidate("R11", "steppe", box(0, 0, 2, 2)),),
        source_version="test",
    )

    assert result.is_empty
    assert result.code is None
    assert result.name is None
    assert result.overlap_percentage is None
    assert result.source_version is None


def test_small_and_full_intersections_are_retained_and_capped() -> None:
    small = choose_winner(
        box(0, 0, 1, 1),
        (OverlapCandidate("R11", "small", box(0, 0, 0.5, 0.5)),),
        source_version="test",
    )
    full = choose_winner(
        box(0, 0, 1, 1),
        (OverlapCandidate("R11", "full", box(-1, -1, 2, 2)),),
        source_version="test",
    )

    assert small.overlap_percentage == 25.0
    assert full.overlap_percentage == 100.0


def test_invalid_candidate_and_sub_unit_overlap_are_ignored() -> None:
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    result = choose_winner(
        box(0, 0, 1, 1),
        (
            OverlapCandidate("R11", "invalid", invalid),
            OverlapCandidate("R12", "small", box(0, 0, 0.5, 0.5)),
        ),
        source_version="test",
    )

    assert result.code == "R12"


def test_prefer_result_merges_reference_groups_by_percentage_then_code() -> None:
    first = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R12", "second", box(0, 0, 5, 10)),),
        source_version="test",
    )
    second = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R11", "first", box(0, 0, 5, 10)),),
        source_version="test",
    )

    assert prefer_result(first, second).code == "R11"


def test_prefer_result_handles_unequal_percentages_and_version_mismatch() -> None:
    current = EunisResult("R11", "current", 25.0, "test")
    candidate = EunisResult("R12", "candidate", 50.0, "test")

    assert prefer_result(current, candidate) is candidate
    assert prefer_result(candidate, current) is candidate
    same_code_tie = EunisResult("R11", "tie", 25.0, "test")
    assert prefer_result(current, same_code_tie) is current
    with pytest.raises(ValueError) as error:
        prefer_result(current, EunisResult("R12", "other", 50.0, "other"))
    assert str(error.value) == "cannot merge EUNIS results from different source versions"


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
