import logging
import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.geometry import GeometryCollection, Polygon, box

from osm_polygon_eunis.domain import EunisResult, OverlapCandidate
from osm_polygon_eunis.matching import (
    _exact_intersection_area,
    _percentage,
    _rank,
    _usable_polygon,
    choose_winner,
    prefer_result,
)


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


def test_exact_intersection_area_preserves_union_semantics_for_overlapping_parts() -> None:
    polygon = box(0, 0, 3, 2)
    candidate = OverlapCandidate(
        "R11",
        "steppe",
        GeometryCollection((box(0, 0, 2, 2), box(1, 0, 3, 2))),
        components_are_disjoint=False,
    )

    assert _exact_intersection_area(polygon, candidate) == 6.0


def test_exact_intersection_area_of_empty_disjoint_parts_is_zero() -> None:
    candidate = OverlapCandidate(
        "R11", "steppe", GeometryCollection(), components_are_disjoint=True
    )

    assert _exact_intersection_area(box(0, 0, 1, 1), candidate) == 0.0


@pytest.mark.parametrize("area", [0.0, 200.0])
def test_percentage_is_clamped_at_both_bounds(area: float) -> None:
    expected = 0.0 if area == 0.0 else 100.0

    assert _percentage(area, 100.0) == expected


def test_zero_area_geometry_is_not_usable() -> None:
    assert not _usable_polygon(box(0, 0, 0, 0))


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


def test_nonempty_zero_area_line_is_not_usable() -> None:
    assert not _usable_polygon(box(0, 0, 1, 1).boundary)


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


@pytest.mark.parametrize("percentages", [(None, 25.0), (25.0, None), (None, None)])
def test_prefer_result_rejects_missing_overlap(percentages) -> None:
    current = EunisResult("R11", "current", percentages[0], "test")
    candidate = EunisResult("R12", "candidate", percentages[1], "test")

    with pytest.raises(ValueError) as error:
        prefer_result(current, candidate)

    assert str(error.value) == "EUNIS rank requires a code and overlap percentage"


@pytest.mark.parametrize("percentage", [None, 25.0])
def test_rank_rejects_missing_code(percentage) -> None:
    with pytest.raises(ValueError) as error:
        _rank(EunisResult(None, None, percentage, "test"))

    assert str(error.value) == "EUNIS rank requires a code and overlap percentage"


def test_prefer_result_rejects_missing_overlap_with_optimized_python() -> None:
    script = """
import sys
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.matching import prefer_result

if sys.flags.optimize != 1:
    raise RuntimeError("optimized Python is required")
for current_percentage, candidate_percentage in [(None, 25.0), (25.0, None), (None, None)]:
    current = EunisResult("R11", "current", current_percentage, "test")
    candidate = EunisResult("R12", "candidate", candidate_percentage, "test")
    try:
        prefer_result(current, candidate)
    except ValueError as error:
        if str(error) != "EUNIS rank requires a code and overlap percentage":
            raise
    else:
        raise RuntimeError("accepted a result without an overlap percentage")
print("rejected all three malformed pairs")
"""
    result = subprocess.run(  # noqa: S603
        [sys.executable, "-O", "-c", script], capture_output=True, text=True, check=False
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "rejected all three malformed pairs\n"


def test_prefer_result_preserves_zero_overlap_and_empty_fast_paths() -> None:
    zero = EunisResult("R11", "zero", 0.0, "test")
    higher_code = EunisResult("R12", "tie", 0.0, "test")
    empty = EunisResult(None, None, None, None)
    malformed = EunisResult("R13", "malformed", None, "test")

    assert prefer_result(zero, higher_code) is zero
    assert prefer_result(higher_code, zero) is zero
    assert prefer_result(empty, malformed) is malformed
    assert prefer_result(malformed, empty) is malformed
    assert prefer_result(empty, empty) is empty


def test_prefer_result_checks_source_version_before_rank() -> None:
    current = EunisResult("R11", "current", None, "test")
    candidate = EunisResult("R12", "candidate", None, "other")

    with pytest.raises(ValueError) as error:
        prefer_result(current, candidate)

    assert str(error.value) == "cannot merge EUNIS results from different source versions"


@st.composite
def _rectangles(draw: st.DrawFn):
    x = draw(st.integers(-20, 20))
    y = draw(st.integers(-20, 20))
    width = draw(st.integers(1, 20))
    height = draw(st.integers(1, 20))
    return box(x, y, x + width, y + height)


@pytest.mark.property
@pytest.mark.slow
@given(_rectangles(), st.lists(_rectangles(), min_size=1, max_size=5))
def test_overlap_percentage_is_bounded(polygon, geometries) -> None:
    candidates = tuple(
        OverlapCandidate(f"R{index:02d}", f"habitat-{index}", geometry)
        for index, geometry in enumerate(geometries)
    )

    result = choose_winner(polygon, candidates, source_version="test")

    assert result.overlap_percentage is None or 0.0 <= result.overlap_percentage <= 100.0


@pytest.mark.property
@pytest.mark.slow
@given(_rectangles(), st.lists(_rectangles(), min_size=1, max_size=5))
def test_candidate_permutation_does_not_change_result(polygon, geometries) -> None:
    candidates = tuple(
        OverlapCandidate(f"R{index:02d}", f"habitat-{index}", geometry)
        for index, geometry in enumerate(geometries)
    )

    assert choose_winner(polygon, candidates, source_version="test") == choose_winner(
        polygon, tuple(reversed(candidates)), source_version="test"
    )


def test_intersection_error_is_reported_and_leaves_result_unchanged(monkeypatch, caplog) -> None:
    from osm_polygon_eunis import matching

    polygon = box(0, 0, 10, 10)
    good = OverlapCandidate("A1", "good", box(0, 0, 4, 10))
    bad = OverlapCandidate("B1", "bad", box(0, 0, 10, 10))
    exact = matching._exact_intersection_area

    def flaky(poly, candidate):
        if candidate.code == "B1":
            raise RuntimeError("GEOS TopologyException")
        return exact(poly, candidate)

    monkeypatch.setattr(matching, "_exact_intersection_area", flaky)
    errors: list[None] = []
    with caplog.at_level(logging.WARNING, logger=matching.__name__):
        counted = choose_winner(
            polygon, [bad, good, bad], source_version="test", on_error=lambda: errors.append(None)
        )
    counted_log_count = sum("skipping" in record.getMessage().lower() for record in caplog.records)
    silent = choose_winner(polygon, [bad, good, bad], source_version="test")

    assert len(errors) == 2
    assert counted == silent == EunisResult("A1", "good", 40.0, "test")
    assert counted_log_count == 2
    assert all("B1" in record.getMessage() for record in caplog.records)
    assert all("GEOS TopologyException" in record.getMessage() for record in caplog.records)
    assert all(
        record.getMessage().startswith("skipping EUNIS candidate") for record in caplog.records
    )


def test_value_error_intersection_is_counted_and_no_error_is_not(monkeypatch) -> None:
    from osm_polygon_eunis import matching

    polygon = box(0, 0, 10, 10)
    candidate = OverlapCandidate("A1", "a", box(0, 0, 5, 10))
    errors: list[None] = []
    assert choose_winner(
        polygon, [candidate], source_version="t", on_error=lambda: errors.append(None)
    ) == EunisResult("A1", "a", 50.0, "t")
    assert errors == []

    def broken(*_args):
        raise ValueError("bad geometry")

    monkeypatch.setattr(matching, "_exact_intersection_area", broken)
    result = choose_winner(
        polygon, [candidate], source_version="t", on_error=lambda: errors.append(None)
    )
    assert result == EunisResult(None, None, None, None)
    assert errors == [None]


def test_choose_winner_ranks_precomputed_areas_with_candidates() -> None:
    polygon = box(0, 0, 10, 10)
    candidate = OverlapCandidate("B1", "b", box(0, 0, 5, 10))

    result = choose_winner(
        polygon,
        [candidate],
        source_version="v",
        extra_areas=[(80.0, "A1", "a"), (50.0, "A0", "z")],
    )

    assert (result.code, result.name, result.overlap_percentage) == ("A1", "a", 80.0)
    tie = choose_winner(
        polygon,
        [candidate],
        source_version="v",
        extra_areas=[(50.0, "A0", "z")],
    )
    assert tie.code == "A0"
