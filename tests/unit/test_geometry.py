import json

from shapely.geometry import Polygon

from osm_polygon_eunis.geometry import parse_geometry, safe_area, to_equal_area


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


def test_to_equal_area_projects_wgs84_geometry() -> None:
    geometry = to_equal_area(Polygon([(2, 48), (3, 48), (3, 49), (2, 48)]))

    assert geometry is not None
    assert geometry.is_valid
    assert safe_area(geometry) > 0
