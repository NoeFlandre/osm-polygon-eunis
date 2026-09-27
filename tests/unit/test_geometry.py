import json

import pytest
from pyproj import Transformer
from shapely import segmentize
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
)
from shapely.ops import transform
from shapely.wkb import dumps

from osm_polygon_eunis import geometry as geometry_module
from osm_polygon_eunis.geometry import (
    _valid_polygonal_geometry,
    has_antimeridian_span,
    parse_geometry,
    safe_area,
    to_equal_area,
)


def test_parse_geometry_accepts_json_text_and_returns_valid_geometry() -> None:
    value = json.dumps(
        {
            "type": "Polygon",
            "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]],
        }
    )

    geometry = parse_geometry(value)

    assert geometry is not None
    assert geometry.is_valid
    assert geometry.geom_type == "Polygon"


def test_parse_geometry_accepts_wkt_text() -> None:
    geometry = parse_geometry("POLYGON ((0 0, 1 0, 1 1, 0 0))")

    assert geometry is not None
    assert geometry.equals(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]))


@pytest.mark.parametrize("srid", [None, 4326], ids=["wkb", "ewkb"])
def test_parse_geometry_accepts_binary_wkb_and_ewkb(srid: int | None) -> None:
    geometry = parse_geometry(dumps(Polygon([(0, 0), (1, 0), (1, 1), (0, 0)]), srid=srid))

    assert geometry is not None
    assert geometry.geom_type == "Polygon"


@pytest.mark.parametrize(
    ("geometry", "expected_type", "expected_area"),
    [
        (MultiPolygon([box(0, 0, 1, 1), box(2, 0, 3, 1)]), "MultiPolygon", 2.0),
        (
            Polygon(
                [(0, 0), (4, 0), (4, 4), (0, 4), (0, 0)],
                holes=[[(1, 1), (3, 1), (3, 3), (1, 3), (1, 1)]],
            ),
            "Polygon",
            12.0,
        ),
    ],
    ids=["multipolygon", "polygon-with-hole"],
)
def test_parse_geometry_preserves_areal_parts(
    geometry: Polygon | MultiPolygon,
    expected_type: str,
    expected_area: float,
) -> None:
    parsed = parse_geometry(mapping(geometry))

    assert parsed is not None
    assert parsed.geom_type == expected_type
    assert parsed.area == expected_area


def test_parse_geometry_repairs_a_recoverable_self_intersection() -> None:
    geometry = parse_geometry(
        {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]],
        }
    )

    assert geometry is not None
    assert geometry.is_valid
    assert safe_area(geometry) > 0


def test_parse_geometry_rejects_missing_or_malformed_values() -> None:
    assert parse_geometry(None) is None
    assert parse_geometry("not-json") is None
    assert parse_geometry({"type": "Point", "coordinates": [None, None]}) is None


def test_parse_geometry_rejects_non_areal_and_collapsed_values() -> None:
    point = Point(1, 2)
    line = LineString([(0, 0), (1, 1)])
    collapsed = Polygon([(0, 0), (1, 1), (2, 2), (0, 0)])

    assert parse_geometry(mapping(point)) is None
    assert parse_geometry(mapping(line)) is None
    assert parse_geometry(mapping(collapsed)) is None


def test_valid_polygonal_geometry_rejects_missing_empty_and_non_areal_values() -> None:
    assert _valid_polygonal_geometry(None) is None
    assert _valid_polygonal_geometry(Polygon()) is None
    assert _valid_polygonal_geometry(LineString([(0, 0), (1, 1)])) is None


def test_parse_geometry_keeps_polygonal_parts_of_mixed_collection() -> None:
    polygon = box(1, 2, 3, 4)
    overlapping_polygon = box(2, 3, 4, 5)
    mixed = GeometryCollection(
        (polygon, overlapping_polygon, LineString([(0, 0), (1, 1)]), Point(5, 5))
    )

    parsed = parse_geometry(mapping(mixed))

    assert parsed is not None
    assert parsed.geom_type in {"Polygon", "MultiPolygon"}
    assert parsed.area == 7.0


def test_combine_polygonal_parts_returns_empty_single_and_union_results() -> None:
    first = box(1, 2, 3, 4)
    second = box(2, 3, 4, 5)

    assert geometry_module._combine_polygonal_parts([]) is None
    assert geometry_module._combine_polygonal_parts([first]) is first
    combined = geometry_module._combine_polygonal_parts([first, second])

    assert combined is not None
    assert combined.area == 7.0


def test_to_equal_area_projects_wgs84_geometry() -> None:
    geometry = to_equal_area(Polygon([(2, 48), (3, 48), (3, 49), (2, 48)]))

    assert geometry is not None
    assert geometry.is_valid
    assert safe_area(geometry) > 0


def test_to_equal_area_densifies_large_wgs84_edges_before_projection() -> None:
    polygon = box(-10.0, 35.0, 30.0, 70.0)
    actual = to_equal_area(polygon)
    reference = transform(
        Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform,
        segmentize(polygon, max_segment_length=0.0005),
    )

    assert actual is not None
    assert len(actual.exterior.coords) > len(polygon.exterior.coords)
    assert abs(actual.area - reference.area) / reference.area < 0.001


def test_to_equal_area_preserves_small_polygon_projection() -> None:
    polygon = box(2.0, 48.0, 2.005, 48.004)
    expected = transform(
        Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform,
        polygon,
    )

    actual = to_equal_area(polygon)

    assert actual is not None
    assert actual.equals_exact(expected, tolerance=1e-9)


def test_to_equal_area_rejects_antimeridian_spanning_polygons() -> None:
    assert to_equal_area(box(-179.0, 50.0, 179.0, 60.0)) is None


def test_antimeridian_span_rejects_only_spans_greater_than_180_degrees() -> None:
    assert not has_antimeridian_span(box(-90.0, 0.0, 90.0, 1.0))
    assert has_antimeridian_span(box(-90.1, 0.0, 90.1, 1.0))


def test_safe_area_is_zero_for_invalid_geometry_and_exact_for_small_polygons() -> None:
    small_polygon = box(0.0, 0.0, 0.5, 0.5)

    assert safe_area(None) == 0.0
    assert safe_area(Polygon()) == 0.0
    assert safe_area(Point(0.0, 0.0)) == 0.0
    assert safe_area(small_polygon) == small_polygon.area
