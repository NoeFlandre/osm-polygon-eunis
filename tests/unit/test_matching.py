import logging
import math
import subprocess
import sys

import pytest
from hypothesis import given
from hypothesis import strategies as st
from shapely.errors import GEOSException
from shapely.geometry import GeometryCollection, Polygon, box

from osm_polygon_eunis.domain import EunisResult, EunisTaxonomy, OverlapCandidate
from osm_polygon_eunis.matching import (
    _exact_intersection_area,
    _intersection_area,
    _overlap_areas_by_node,
    _percentage,
    _qualifying_classes_conflict,
    _rank,
    _usable_polygon,
    choose_winner,
    prefer_result,
    result_from_class_areas,
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


def test_dominant_below_eighty_percent_is_withheld() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        [OverlapCandidate("R11", "steppe", box(0, 0, 7, 10))],
        source_version="test",
    )

    assert result == EunisResult(None, None, 70.0, "test")


def test_withheld_result_reports_the_largest_class_union_percentage() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("R11", "steppe", box(0, 0, 7, 10)),
            OverlapCandidate("R12", "bog", box(0, 0, 7.5, 10)),
        ),
        source_version="test",
    )

    assert result == EunisResult(None, None, 75.0, "test")


def test_dominant_at_exactly_eighty_percent_is_assigned() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        [OverlapCandidate("R11", "steppe", box(0, 0, 8, 10))],
        source_version="test",
    )

    assert result == EunisResult("R11", "steppe", 80.0, "test")


@pytest.mark.parametrize("polygon_area", [0.0, -1.0])
def test_precomputed_class_areas_reject_nonpositive_polygon_area(polygon_area: float) -> None:
    assert result_from_class_areas([(80.0, "R11", "steppe")], polygon_area, "test").is_empty


def test_precomputed_class_areas_use_full_polygon_denominator_and_require_unions() -> None:
    assigned = result_from_class_areas([(80.0, "R11", "steppe")], 100.0, "test")

    assert assigned == EunisResult("R11", "steppe", 80.0, "test")
    with pytest.raises(ValueError, match="must be unioned before selection"):
        result_from_class_areas([(50.0, "R11", "steppe"), (40.0, "R11", "steppe")], 100.0, "test")


def test_precomputed_nonpositive_support_areas_do_not_create_a_class() -> None:
    result = result_from_class_areas([(0.0, "R11", "steppe"), (-5.0, "R12", "bog")], 100.0, "test")

    assert result.is_empty


@pytest.mark.parametrize("area", [math.nan, math.inf, -math.inf])
def test_precomputed_nonfinite_support_areas_are_rejected(area: float) -> None:
    with pytest.raises(ValueError) as error:
        result_from_class_areas([(area, "R11", "steppe")], 100.0, "test")

    assert str(error.value) == (
        "class support and polygon areas must be finite, with a positive polygon area"
    )


@pytest.mark.parametrize("polygon_area", [math.nan, math.inf])
def test_precomputed_nonfinite_polygon_areas_are_rejected(polygon_area: float) -> None:
    with pytest.raises(ValueError) as error:
        result_from_class_areas([(80.0, "R11", "steppe")], polygon_area, "test")

    assert str(error.value) == (
        "class support and polygon areas must be finite, with a positive polygon area"
    )


def test_precomputed_support_area_cannot_exceed_the_full_polygon_area() -> None:
    with pytest.raises(ValueError) as error:
        result_from_class_areas([(100.1, "R11", "steppe")], 100.0, "test")

    assert str(error.value) == "class support area exceeds polygon area"


@pytest.mark.parametrize(
    ("support_area", "polygon_area", "codes"),
    [
        pytest.param(
            195.1521603930019,
            195.1521603882745,
            ("T1H", "T1F", "T1E"),
            id="thueringen",
        ),
        pytest.param(
            40.825887750524174,
            40.82588774909813,
            ("T1H", "T1E", "T1B"),
            id="alsace",
        ),
    ],
)
def test_overlay_rounding_overage_preserves_ambiguous_full_coverage(
    support_area: float,
    polygon_area: float,
    codes: tuple[str, ...],
) -> None:
    result = result_from_class_areas(
        [(support_area, code, code) for code in codes], polygon_area, "EEA-test"
    )

    assert result == EunisResult(None, None, 100.0, "EEA-test")


def test_precomputed_nonpositive_support_does_not_hide_a_later_class() -> None:
    result = result_from_class_areas([(0.0, "R12", "bog"), (80.0, "R11", "steppe")], 100.0, "test")

    assert result == EunisResult("R11", "steppe", 80.0, "test")


def test_precomputed_coverage_within_boundary_tolerance_is_normalized() -> None:
    result = result_from_class_areas([(79.9999999995, "R11", "steppe")], 100.0, "test")

    assert result == EunisResult("R11", "steppe", 80.0, "test")


def test_boundary_tolerance_includes_its_exact_float_limit(monkeypatch) -> None:
    from osm_polygon_eunis import matching

    percentage = math.nextafter(80.0, math.inf)
    tolerance = percentage - 80.0
    monkeypatch.setattr(matching, "_BOUNDARY_TOLERANCE_PERCENTAGE", tolerance)

    result = result_from_class_areas([(percentage, "R11", "steppe")], 100.0, "test")

    assert result == EunisResult("R11", "steppe", 80.0, "test")


def test_same_class_unions_disjoint_supports_before_the_eighty_percent_gate() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q111", "fine habitat", box(0, 0, 4.5, 10)),
            OverlapCandidate("Q111", "fine habitat", box(5.5, 0, 9.5, 10)),
        ),
        source_version="test",
    )

    assert result == EunisResult("Q111", "fine habitat", 85.0, "test")


def test_same_class_supports_reject_conflicting_labels() -> None:
    with pytest.raises(ValueError) as error:
        choose_winner(
            box(0, 0, 10, 10),
            (
                OverlapCandidate("Q111", "fine habitat", box(0, 0, 4, 10)),
                OverlapCandidate("Q111", "different label", box(6, 0, 10, 10)),
            ),
            source_version="test",
        )

    assert str(error.value) == "EUNIS class Q111 has conflicting names"


def test_same_class_unions_overlapping_supports_without_double_counting() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q111", "fine habitat", box(0, 0, 9, 5)),
            OverlapCandidate("Q111", "fine habitat", box(2, 4, 10, 9)),
        ),
        source_version="test",
    )

    assert result == EunisResult(None, None, 78.0, "test")


def test_class_union_error_drops_each_support(monkeypatch, caplog) -> None:
    from osm_polygon_eunis import matching

    def broken_union(_geometries):
        raise ValueError("invalid support union")

    monkeypatch.setattr(matching, "unary_union", broken_union)
    errors: list[None] = []
    with caplog.at_level(logging.WARNING, logger=matching.__name__):
        result = choose_winner(
            box(0, 0, 10, 10),
            (
                OverlapCandidate("Q111", "fine habitat", box(0, 0, 4, 10)),
                OverlapCandidate("Q111", "fine habitat", box(6, 0, 10, 10)),
            ),
            source_version="test",
            on_error=lambda: errors.append(None),
        )

    assert result.is_empty
    assert len(errors) == 2
    assert caplog.messages == [
        "skipping EUNIS class Q111 after support union error: invalid support union"
    ]


@pytest.mark.parametrize(
    "error",
    [GEOSException("GEOS failure"), RuntimeError("runtime failure")],
    ids=["geos", "runtime"],
)
def test_class_union_handles_supported_geometry_errors(monkeypatch, error) -> None:
    from osm_polygon_eunis import matching

    def broken_union(_geometries):
        raise error

    monkeypatch.setattr(matching, "unary_union", broken_union)
    errors: list[None] = []
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q111", "fine habitat", box(0, 0, 4, 10)),
            OverlapCandidate("Q111", "fine habitat", box(6, 0, 10, 10)),
        ),
        source_version="test",
        on_error=lambda: errors.append(None),
    )

    assert result.is_empty
    assert errors == [None, None]


def test_unusable_class_union_drops_each_support(monkeypatch, caplog) -> None:
    from osm_polygon_eunis import matching

    monkeypatch.setattr(matching, "unary_union", lambda _geometries: GeometryCollection())
    errors: list[None] = []
    with caplog.at_level(logging.WARNING, logger=matching.__name__):
        result = choose_winner(
            box(0, 0, 10, 10),
            (
                OverlapCandidate("Q111", "fine habitat", box(0, 0, 4, 10)),
                OverlapCandidate("Q111", "fine habitat", box(6, 0, 10, 10)),
            ),
            source_version="test",
            on_error=lambda: errors.append(None),
        )

    assert result.is_empty
    assert len(errors) == 2
    assert caplog.messages == ["skipping unusable EUNIS support union for Q111"]


def test_failed_node_union_reports_errors_and_keeps_later_node(monkeypatch, caplog) -> None:
    from osm_polygon_eunis import matching

    def broken_union(_geometries):
        raise ValueError("invalid node union")

    monkeypatch.setattr(matching, "unary_union", broken_union)
    supports = {
        "Q111": [
            OverlapCandidate("Q111", "fine habitat", box(0, 0, 1, 1), error_count=2),
            OverlapCandidate("Q111", "fine habitat", box(1, 0, 2, 1), error_count=3),
        ],
        "R11": [OverlapCandidate("R11", "steppe", box(0, 0, 1, 1))],
    }
    errors: list[None] = []

    with caplog.at_level(logging.WARNING, logger=matching.__name__):
        result = _overlap_areas_by_node(box(0, 0, 1, 1), supports, lambda: errors.append(None))

    assert result == [(1.0, "R11", "steppe")]
    assert errors == [None, None, None, None, None]
    assert caplog.messages == [
        "skipping EUNIS class Q111 after support union error: invalid node union"
    ]


def test_distinct_qualifying_hierarchy_codes_remain_ambiguous() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q11", "parent class", box(0, 0, 9, 10)),
            OverlapCandidate("Q111", "fine class", box(0, 0, 8, 10)),
        ),
        source_version="test",
    )

    assert result == EunisResult(None, None, 90.0, "test")


def test_deepest_qualifying_taxonomy_node_wins_over_its_ancestor() -> None:
    taxonomy = EunisTaxonomy(
        labels={"Q11": "parent", "Q111": "fine habitat"},
        levels={"Q11": 2, "Q111": 3},
        parents={"Q111": "Q11"},
    )
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q11", "parent", box(0, 0, 10, 10)),
            OverlapCandidate("Q111", "input label", box(0, 0, 8, 10)),
        ),
        source_version="test",
        taxonomy=taxonomy,
    )

    assert result == EunisResult("Q111", "fine habitat", 80.0, "test")


def test_precomputed_class_area_uses_taxonomy_label_and_preserves_unknown_label() -> None:
    taxonomy = EunisTaxonomy(
        labels={"Q11": "parent", "Q111": "canonical fine habitat"},
        levels={"Q11": 2, "Q111": 3},
        parents={"Q111": "Q11"},
    )

    assigned = result_from_class_areas(
        [(100.0, "Q11", "input parent"), (80.0, "Q111", "input child")],
        100.0,
        "test",
        taxonomy=taxonomy,
    )
    isolated = result_from_class_areas(
        [(80.0, "Q999", "input fallback")], 100.0, "test", taxonomy=taxonomy
    )

    assert assigned == EunisResult("Q111", "canonical fine habitat", 80.0, "test")
    assert isolated == EunisResult("Q999", "input fallback", 80.0, "test")


def test_unknown_taxonomy_root_keeps_the_candidate_label() -> None:
    taxonomy = EunisTaxonomy(
        labels={"MA223": "configured saltmarsh root"},
        levels={},
        parents={},
    )
    result = choose_winner(
        box(0, 0, 10, 10),
        [OverlapCandidate("MA221", "Atlantic saltmarsh driftlines", box(0, 0, 10, 10))],
        source_version="test",
        taxonomy=taxonomy,
    )

    assert result == EunisResult("MA221", "Atlantic saltmarsh driftlines", 100.0, "test")


def test_taxonomy_does_not_accept_precomputed_scalar_support_areas() -> None:
    taxonomy = EunisTaxonomy({"Q11": "parent"}, {"Q11": 2}, {})

    with pytest.raises(ValueError) as error:
        choose_winner(
            box(0, 0, 10, 10),
            (),
            source_version="test",
            extra_areas=[(80.0, "Q11", "parent")],
            taxonomy=taxonomy,
        )

    assert str(error.value) == "scalar support areas cannot be rolled up through an EUNIS hierarchy"


def test_sibling_supports_union_to_parent_and_count_overlaps_once() -> None:
    taxonomy = EunisTaxonomy(
        labels={"Q11": "parent", "Q111": "first child", "Q112": "second child"},
        levels={"Q11": 2, "Q111": 3, "Q112": 3},
        parents={"Q111": "Q11", "Q112": "Q11"},
    )
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q111", "first child", box(0, 0, 5, 10)),
            OverlapCandidate("Q112", "second child", box(3, 0, 8, 10)),
        ),
        source_version="test",
        taxonomy=taxonomy,
    )

    assert result == EunisResult("Q11", "parent", 80.0, "test")


def test_qualifying_sibling_classes_remain_ambiguous_after_parent_rollup() -> None:
    taxonomy = EunisTaxonomy(
        labels={"Q11": "parent", "Q111": "first child", "Q112": "second child"},
        levels={"Q11": 2, "Q111": 3, "Q112": 3},
        parents={"Q111": "Q11", "Q112": "Q11"},
    )
    result = choose_winner(
        box(0, 0, 10, 10),
        (
            OverlapCandidate("Q111", "first child", box(0, 0, 9, 10)),
            OverlapCandidate("Q112", "second child", box(1, 0, 10, 10)),
        ),
        source_version="test",
        taxonomy=taxonomy,
    )

    assert result == EunisResult(None, None, 90.0, "test")


def test_dominant_just_below_eighty_percent_is_withheld() -> None:
    result = choose_winner(
        box(0, 0, 10, 10),
        [OverlapCandidate("R11", "steppe", box(0, 0, 7.99, 10))],
        source_version="test",
    )

    assert result.code is None
    assert result.name is None
    assert result.overlap_percentage == pytest.approx(79.9)
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

    assert result == EunisResult(None, None, 50.0, "test")


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


def test_exact_intersection_area_sums_nonempty_disjoint_parts() -> None:
    polygon = box(0, 0, 3, 2)
    candidate = OverlapCandidate(
        "R11",
        "steppe",
        GeometryCollection((box(0, 0, 1, 2), box(2, 0, 3, 2))),
        components_are_disjoint=True,
    )

    assert _exact_intersection_area(polygon, candidate) == 4.0


@pytest.mark.parametrize("area", [-1.0, 0.0])
def test_intersection_area_omits_nonpositive_results(monkeypatch, area: float) -> None:
    from osm_polygon_eunis import matching

    monkeypatch.setattr(matching, "_exact_intersection_area", lambda *_args: area)
    candidate = OverlapCandidate("R11", "steppe", box(0, 0, 1, 1))

    assert _intersection_area(box(0, 0, 1, 1), candidate, lambda: None) is None


def test_intersection_area_reports_unusable_candidate(caplog) -> None:
    errors: list[None] = []
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    candidate = OverlapCandidate("R11", "invalid", invalid)

    with caplog.at_level(logging.WARNING):
        area = _intersection_area(
            box(0, 0, 1, 1), candidate, lambda: errors.append(None), error_count=2
        )

    assert area is None
    assert errors == [None, None]
    assert caplog.messages == ["skipping unusable EUNIS candidate R11"]


def test_intersection_error_reports_one_source_by_default(monkeypatch) -> None:
    from osm_polygon_eunis import matching

    def broken_intersection(*_args):
        raise GEOSException("intersection failure")

    monkeypatch.setattr(matching, "_exact_intersection_area", broken_intersection)
    errors: list[None] = []
    candidate = OverlapCandidate("R11", "steppe", box(0, 0, 1, 1))

    assert _intersection_area(box(0, 0, 1, 1), candidate, lambda: errors.append(None)) is None
    assert errors == [None]


def test_intersection_error_reports_every_unioned_support(monkeypatch) -> None:
    from osm_polygon_eunis import matching

    def broken_intersection(*_args):
        raise GEOSException("intersection failure")

    monkeypatch.setattr(matching, "_exact_intersection_area", broken_intersection)
    errors: list[None] = []
    candidate = OverlapCandidate("Q111", "fine habitat", box(0, 0, 1, 1))

    assert (
        _intersection_area(box(0, 0, 1, 1), candidate, lambda: errors.append(None), error_count=3)
        is None
    )
    assert errors == [None, None, None]


def test_percentage_is_clamped_at_zero() -> None:
    assert _percentage(0.0, 100.0) == 0.0


def test_percentage_clamps_only_roundoff_above_full_polygon_area() -> None:
    assert _percentage(math.nextafter(100.0, math.inf), 100.0) == 100.0


def test_percentage_clamps_at_the_exact_full_area_tolerance(monkeypatch) -> None:
    from osm_polygon_eunis import matching

    area = math.nextafter(100.0, math.inf)
    tolerance = area - 100.0
    monkeypatch.setattr(matching, "_MAX_PERCENTAGE_ROUNDING_TOLERANCE", tolerance)

    assert _percentage(area, 100.0) == 100.0


def test_percentage_rejects_material_area_overrun() -> None:
    with pytest.raises(ValueError) as error:
        _percentage(200.0, 100.0)

    assert str(error.value) == "class support area exceeds polygon area"


def test_percentage_rejects_overflow_from_finite_areas() -> None:
    with pytest.raises(ValueError) as error:
        _percentage(1e308, 1e-308)

    assert str(error.value) == "class overlap percentage must be finite"


def test_percentage_rejects_zero_polygon_area() -> None:
    with pytest.raises(ValueError) as error:
        _percentage(80.0, 0.0)

    assert str(error.value) == (
        "class support and polygon areas must be finite, with a positive polygon area"
    )


def test_percentage_is_clamped_at_zero_for_negative_area() -> None:
    assert _percentage(-1.0, 100.0) == 0.0


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


def test_equal_area_qualifying_classes_are_ambiguous() -> None:
    polygon = box(0, 0, 10, 10)
    candidates = (
        OverlapCandidate("R12", "second", box(0, 0, 9, 10)),
        OverlapCandidate("R11", "first", box(1, 0, 10, 10)),
    )

    assert choose_winner(polygon, candidates, source_version="test") == EunisResult(
        None, None, 90.0, "test"
    )


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

    assert result == EunisResult(None, None, 25.0, "test")


def test_unusable_candidate_is_reported_through_error_hook(caplog) -> None:
    errors: list[str] = []
    invalid = Polygon([(0, 0), (2, 2), (0, 2), (2, 0), (0, 0)])
    with caplog.at_level(logging.WARNING):
        result = choose_winner(
            box(0, 0, 1, 1),
            (OverlapCandidate("R11", "invalid", invalid),),
            source_version="test",
            on_error=lambda: errors.append("R11"),
        )

    assert result.is_empty
    assert errors == ["R11"]
    assert "skipping unusable EUNIS candidate R11" in caplog.messages


def test_prefer_result_keeps_multiple_qualifying_reference_classes_ambiguous() -> None:
    first = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R12", "second", box(0, 0, 9, 10)),),
        source_version="test",
    )
    second = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R11", "first", box(0, 0, 9, 10)),),
        source_version="test",
    )

    assert prefer_result(first, second) == EunisResult(None, None, 90.0, "test")


def test_prefer_result_ranks_withheld_measurements_by_percentage() -> None:
    sixty = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R12", "grassland", box(0, 0, 6, 10)),),
        source_version="test",
    )
    seventy = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R11", "steppe", box(0, 0, 7, 10)),),
        source_version="test",
    )

    assert prefer_result(sixty, seventy) == EunisResult(None, None, 70.0, "test")
    assert prefer_result(seventy, sixty) == EunisResult(None, None, 70.0, "test")


def test_prefer_result_assigned_eighty_beats_withheld_seventy_five() -> None:
    withheld = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R12", "grassland", box(0, 0, 7.5, 10)),),
        source_version="test",
    )
    assigned = choose_winner(
        box(0, 0, 10, 10),
        (OverlapCandidate("R11", "steppe", box(0, 0, 8, 10)),),
        source_version="test",
    )

    assert prefer_result(withheld, assigned) == EunisResult("R11", "steppe", 80.0, "test")
    assert prefer_result(assigned, withheld) == EunisResult("R11", "steppe", 80.0, "test")


def test_exactly_eighty_percent_ambiguous_result_stays_ambiguous_when_merged() -> None:
    ambiguous = EunisResult(None, None, 80.0, "test")
    larger_assignment = EunisResult("R11", "steppe", 90.0, "test")

    assert prefer_result(ambiguous, larger_assignment) == EunisResult(None, None, 90.0, "test")


def test_ambiguity_conflict_flag_survives_a_below_threshold_class() -> None:
    ambiguous = EunisResult(None, None, 85.0, "test")
    below_threshold = EunisResult("R11", "steppe", 70.0, "test")

    assert _qualifying_classes_conflict(ambiguous, below_threshold)


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

    assert str(error.value) == "EUNIS rank requires an overlap percentage"


@pytest.mark.parametrize("percentage", [None])
def test_rank_rejects_missing_overlap(percentage) -> None:
    with pytest.raises(ValueError) as error:
        _rank(EunisResult(None, None, percentage, "test"))

    assert str(error.value) == "EUNIS rank requires an overlap percentage"


def test_rank_uses_an_empty_code_for_a_withheld_result() -> None:
    assert _rank(EunisResult(None, None, 50.0, "test")) == (50.0, "")


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
        if str(error) != "EUNIS rank requires an overlap percentage":
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

    # The failed class intersection drops both same-code source candidates.
    assert len(errors) == 2
    assert counted == silent == EunisResult(None, None, 40.0, "test")
    assert counted_log_count == 1
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
    ) == EunisResult(None, None, 50.0, "t")
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
        [OverlapCandidate("B1", "b", box(0, 0, 9, 10))],
        source_version="v",
        extra_areas=[(90.0, "A0", "z")],
    )
    assert tie == EunisResult(None, None, 90.0, "v")


def test_choose_winner_rejects_duplicate_or_unrollable_precomputed_support() -> None:
    polygon = box(0, 0, 10, 10)
    candidate = OverlapCandidate("Q111", "fine habitat", box(0, 0, 8, 10))

    with pytest.raises(ValueError, match="must be unioned before selection"):
        choose_winner(
            polygon,
            [candidate],
            source_version="test",
            extra_areas=[(10.0, "Q111", "fine habitat")],
        )

    taxonomy = EunisTaxonomy({"Q111": "fine habitat"}, {"Q111": 3}, {})
    with pytest.raises(ValueError, match="cannot be rolled up"):
        choose_winner(
            polygon,
            [],
            source_version="test",
            extra_areas=[(80.0, "Q111", "fine habitat")],
            taxonomy=taxonomy,
        )
