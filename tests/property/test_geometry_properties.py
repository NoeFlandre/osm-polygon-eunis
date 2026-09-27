from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pyproj import Transformer
from shapely import to_wkt
from shapely.affinity import rotate
from shapely.geometry import Polygon, box
from shapely.ops import transform

from osm_polygon_eunis.domain import EunisResult, OverlapCandidate
from osm_polygon_eunis.geometry import is_usable, parse_geometry, safe_area, to_equal_area
from osm_polygon_eunis.matching import choose_winner, prefer_result


@st.composite
def _small_boxes(draw: st.DrawFn):
    left = draw(st.integers(min_value=-5_000, max_value=25_000)) / 1_000
    bottom = draw(st.integers(min_value=35_000, max_value=65_000)) / 1_000
    width = draw(st.integers(min_value=1, max_value=20)) / 1_000
    height = draw(st.integers(min_value=1, max_value=20)) / 1_000
    return box(left, bottom, left + width, bottom + height)


@st.composite
def _results(draw: st.DrawFn):
    code = draw(st.sampled_from(("A1", "B2", "C3")))
    percentage = draw(st.integers(min_value=0, max_value=100))
    return EunisResult(code, f"Habitat {code}", float(percentage), "EEA-test")


@pytest.mark.property
@given(
    x=st.floats(min_value=-1_000, max_value=1_000, allow_nan=False, allow_infinity=False),
    y=st.floats(min_value=-1_000, max_value=1_000, allow_nan=False, allow_infinity=False),
    width=st.floats(min_value=0, max_value=1_000, allow_nan=False, allow_infinity=False),
    height=st.floats(min_value=0, max_value=1_000, allow_nan=False, allow_infinity=False),
)
def test_safe_area_never_returns_a_negative_value(
    x: float, y: float, width: float, height: float
) -> None:
    geometry = box(x, y, x + width, y + height)

    assert safe_area(geometry) >= 0.0
    assert safe_area(None) == 0.0
    assert safe_area(Polygon()) == 0.0


@pytest.mark.property
@given(_small_boxes())
def test_parse_geometry_round_trips_wkb_and_wkt_and_is_idempotent(polygon: Polygon) -> None:
    encoded_values = (polygon.wkb, to_wkt(polygon, rounding_precision=18, trim=False))
    for encoded in encoded_values:
        parsed = parse_geometry(encoded)
        assert parsed is not None
        assert is_usable(parsed)
        assert parsed.equals(polygon)
        repeated = parse_geometry(parsed.wkb)
        assert repeated is not None
        assert repeated.equals(parsed)


@pytest.mark.property
@given(_small_boxes())
def test_equal_area_projection_round_trips_small_eea_polygons(polygon: Polygon) -> None:
    projected = to_equal_area(polygon)
    assert projected is not None
    assert projected.area > 0.0

    inverse = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    restored = transform(inverse.transform, projected)
    assert polygon.hausdorff_distance(restored) <= 1e-6


@pytest.mark.property
@given(_results(), _results(), _results())
def test_prefer_result_is_associative_commutative_and_idempotent(
    first: EunisResult,
    second: EunisResult,
    third: EunisResult,
) -> None:
    assert prefer_result(first, second) == prefer_result(second, first)
    assert prefer_result(prefer_result(first, second), third) == prefer_result(
        first, prefer_result(second, third)
    )
    assert prefer_result(first, first) == first


@pytest.mark.property
@given(
    angle=st.floats(min_value=1.0, max_value=89.0, allow_nan=False, allow_infinity=False),
    cut=st.floats(min_value=1.0, max_value=9.0, allow_nan=False, allow_infinity=False),
)
def test_choose_winner_shares_on_disjoint_parts_of_a_rotated_polygon(
    angle: float,
    cut: float,
) -> None:
    polygon = rotate(box(0.0, 0.0, 10.0, 10.0), angle, origin="center")
    left = polygon.intersection(box(-100.0, -100.0, cut, 100.0))
    right = polygon.difference(left)
    candidates = (
        OverlapCandidate("A1", "left", left),
        OverlapCandidate("B2", "right", right),
    )
    shares = tuple(100.0 * candidate.geometry.area / polygon.area for candidate in candidates)

    winner = choose_winner(polygon, candidates, source_version="EEA-test")

    assert winner.overlap_percentage is not None
    assert winner.overlap_percentage >= max(shares) - 1e-8
    assert sum(shares) <= 100.0 + 1e-8


@pytest.mark.property
@given(_results(), _results())
def test_prefer_result_rejects_different_reference_versions(
    first: EunisResult,
    second: EunisResult,
) -> None:
    second = EunisResult(second.code, second.name, second.overlap_percentage, "other")
    with pytest.raises(ValueError, match="different source versions"):
        prefer_result(first, second)
