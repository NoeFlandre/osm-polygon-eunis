from __future__ import annotations

import json
import os
from pathlib import Path

import h3
import pytest
from shapely.geometry import (
    GeometryCollection,
    LineString,
    MultiPolygon,
    Point,
    Polygon,
    box,
    mapping,
)

from osm_polygon_eunis.cards import (
    DatasetCardAccumulator,
    _extract_source_configs,
    _extract_source_license,
)
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.geometry import GEOMETRY_POLICY, parse_geometry


def _geometry(longitude: float, latitude: float) -> str:
    return json.dumps(mapping(box(longitude, latitude, longitude + 1, latitude + 1)))


def test_card_artifacts_have_distribution_table_and_static_world_map(tmp_path: Path) -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(3.0, 48.0))
    card.observe(EunisResult(None, None, None, None), None)

    first = card.write_artifacts(
        tmp_path / "first",
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )
    second = card.write_artifacts(
        tmp_path / "second",
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )

    assert [
        (summary.code, summary.name, summary.rows, summary.percentage)
        for summary in card.summaries()
    ] == [
        ("R11", "Steppe", 2, pytest.approx(200 / 3)),
        (None, "No EUNIS label", 1, pytest.approx(100 / 3)),
    ]
    assert "eunis/world-map.svg" in first.files["README.md"].read_text(encoding="utf-8")
    assert first.manifest["total_rows"] == 3
    assert first.manifest["geometry_policy"] == GEOMETRY_POLICY
    for path in ("README.md", "eunis/world-map.svg"):
        assert first.files[path].read_bytes() == second.files[path].read_bytes()
    assert first.hashes == {
        "README.md": first.manifest["readme_sha256"],
        "eunis/world-map.svg": first.manifest["map_sha256"],
    }


def test_card_preserves_source_viewer_configs_without_source_card_body(tmp_path: Path) -> None:
    source_readme = """---
license: odbl
configs:
  - config_name: polygon_document_links_by_language__lang_ab
    data_files:
      - split: train
        path: language_splits/polygon_document_links_by_language/lang-ab/part-*.parquet
dataset_info:
  features: []
---
# Source card body must not replace the EUNIS card.
"""
    card = DatasetCardAccumulator(source_readme=source_readme)
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="wikidata",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )

    readme = artifacts.files["README.md"].read_text(encoding="utf-8")
    assert "config_name: polygon_document_links_by_language__lang_ab" in readme
    assert "split: train" in readme
    assert (
        "path: language_splits/polygon_document_links_by_language/lang-ab/part-*.parquet" in readme
    )
    assert "# Source card body must not replace the EUNIS card." not in readme
    assert "# Wikidata" in readme
    assert artifacts.hashes["README.md"] == artifacts.manifest["readme_sha256"]
    boundary_cases = (
        (None, None),
        ("", None),
        ("not frontmatter", None),
        ("---\nconfigs:\n", None),
        ("---\nlicense: odbl\n---", None),
        ("---\nmetadata:\n  configs:\n    - name: nested\n---", None),
        (
            "---\nlicense: odbl\nconfigs:\n  - config_name: polygons\n---",
            "configs:\n  - config_name: polygons",
        ),
    )
    for source, expected_configs in boundary_cases:
        assert _extract_source_configs(source) == expected_configs


def test_card_preserves_source_data_license_metadata(tmp_path: Path) -> None:
    card = DatasetCardAccumulator(source_readme="---\nlicense: odbl\n---\n")
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="description",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )

    readme = artifacts.files["README.md"].read_text(encoding="utf-8")
    frontmatter = readme.split("---", 2)[1]
    assert "license: odbl" in frontmatter
    assert "license: apache-2.0" not in frontmatter


@pytest.mark.parametrize(
    ("source_readme", "expected"),
    [
        (None, None),
        ("", None),
        ("not frontmatter", None),
        ("---\nlicense: odbl", None),
        ("---\nlicense: odbl\n---\nlicense: apache-2.0\n", "odbl"),
        ("---\nlicense: cc-by-4.0\n---", "cc-by-4.0"),
        ("---\n  license: odbl\n---", None),
        ("---\nlicense: CC BY 4.0\n---", None),
        ("---\n---\nlicense: odbl\n", None),
    ],
)
def test_extract_source_license_reads_only_simple_frontmatter(
    source_readme: str | None, expected: str | None
) -> None:
    assert _extract_source_license(source_readme) == expected


def test_card_separates_code_license_from_dataset_data_terms(tmp_path: Path) -> None:
    card = DatasetCardAccumulator(source_readme="---\nlicense: odbl\n---\n")
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="description",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )

    readme = artifacts.files["README.md"].read_text(encoding="utf-8")
    assert "## Data terms" in readme
    assert "OpenStreetMap data is available under the [Open Database License]" in readme
    assert "EEA reference assets can carry item-specific reuse terms" in readme
    assert "The Apache-2.0 license applies to project code only" in readme


def test_card_accumulator_rejects_conflicting_names() -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), _geometry(0.0, 0.0))
    with pytest.raises(ValueError, match="conflicting names"):
        card.observe(EunisResult("R11", "Grassland", 50.0, "EEA-test"), _geometry(0.0, 0.0))


def test_card_counts_rows_even_when_geometry_is_invalid_or_outside_world() -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), "not-json")
    card.observe(
        EunisResult("R12", "Other", 50.0, "EEA-test"),
        _geometry(200.0, 0.0),
    )

    assert card.total_rows == 2
    assert [summary.rows for summary in card.summaries()] == [1, 1]


def test_card_map_counts_one_row_per_h3_cell_across_all_labels() -> None:
    geometry = _geometry(2.0, 48.0)
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), geometry)
    card.observe(EunisResult("R12", "Other", 25.0, "EEA-test"), geometry)
    card.observe(EunisResult(None, None, None, None), geometry)
    card.observe(EunisResult(None, None, None, None), None)

    cell = h3.latlng_to_cell(48.5, 2.5, 3)
    assert card.snapshot()["map_counts"] == [[cell, 3]]
    assert card.total_rows == 4
    assert [summary.rows for summary in card.summaries()] == [1, 1, 2]


def test_card_map_uses_a_representative_point_instead_of_the_centroid() -> None:
    geometry = MultiPolygon((box(-40.0, 1.0, -39.0, 2.0), box(40.0, 1.0, 41.0, 2.0)))
    polygonal = parse_geometry(mapping(geometry))
    assert polygonal is not None
    representative_point = polygonal.representative_point()
    representative_cell = h3.latlng_to_cell(representative_point.y, representative_point.x, 3)
    centroid = polygonal.centroid
    assert representative_cell != h3.latlng_to_cell(centroid.y, centroid.x, 3)

    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), mapping(geometry))

    assert card.snapshot()["map_counts"] == [[representative_cell, 1]]


def test_card_map_omits_missing_invalid_and_out_of_range_geometries() -> None:
    card = DatasetCardAccumulator()
    result = EunisResult(None, None, None, None)
    card.observe(result, None)
    card.observe(result, "not-json")
    card.observe(result, _geometry(200.0, 0.0))
    card.observe(result, _geometry(2.0, 48.0))

    assert card.snapshot()["map_counts"] == [[h3.latlng_to_cell(48.5, 2.5, 3), 1]]
    assert card.total_rows == 4


def test_card_counts_undecodable_geometries_separately_from_missing_ones(
    tmp_path: Path,
) -> None:
    card = DatasetCardAccumulator()
    result = EunisResult(None, None, None, None)
    card.observe(result, "not-json")
    card.observe(result, b"\x00not-wkb")
    card.observe(result, None)
    card.observe(result, _geometry(0.0, 0.0))

    assert card.invalid_geometries == 2
    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )
    assert artifacts.manifest["invalid_geometries"] == 2
    assert artifacts.manifest["total_rows"] == 4


def test_card_counts_non_areal_and_collapsed_geometry_as_invalid(tmp_path: Path) -> None:
    card = DatasetCardAccumulator()
    result = EunisResult(None, None, None, None)
    values = (
        mapping(Point(1, 2)),
        mapping(LineString([(0, 0), (1, 1)])),
        mapping(Polygon([(0, 0), (1, 1), (2, 2), (0, 0)])),
        mapping(box(-179.0, 50.0, 179.0, 60.0)),
        mapping(GeometryCollection((box(1, 2, 3, 4), LineString([(0, 0), (1, 1)])))),
    )
    for value in values:
        card.observe(result, value)

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )

    assert artifacts.manifest["invalid_geometries"] == 4
    assert artifacts.manifest["total_rows"] == 5


def _golden_card() -> DatasetCardAccumulator:
    card = DatasetCardAccumulator()
    for index in range(25):
        card.observe(
            EunisResult(f"R{index:02d}", f"Label number {index} with a long name", 80.0, "EEA"),
            _geometry(-170.0 + 13 * index, -80.0 + 6 * index),
        )
    card.observe(EunisResult(None, None, None, None), _geometry(10.0, 10.0))
    return card


def test_default_card_artifacts_are_byte_identical_to_golden_hashes(tmp_path: Path) -> None:
    """Assert golden hashes and document their intentional refresh command.

    Card and map hashes feed the release manifest and no-op detection. To
    refresh after an intentional output change, run:
    ``EUNIS_UPDATE_CARD_GOLDEN_HASHES=1 uv run pytest
    tests/unit/test_cards.py::test_default_card_artifacts_are_byte_identical_to_golden_hashes -s``.
    The opt-in prints replacement values; ``-s`` makes them visible.
    """
    artifacts = _golden_card().write_artifacts(
        tmp_path,
        dataset_name="website-polygons",
        source_repo="o/s",
        target_repo="o/t",
        source_revision="rev",
        reference_version="EEA",
    )
    if os.environ.get("EUNIS_UPDATE_CARD_GOLDEN_HASHES") == "1":
        print(json.dumps(dict(artifacts.hashes), indent=2, sort_keys=True))  # noqa: T201

    assert dict(artifacts.hashes) == {
        "README.md": "8d6b6409665764cd955ebcd983e9eba93e95d364eee4d5a39fe31fc07ea1f536",
        "eunis/world-map.svg": "387cc2b80335eebc80b0052169378d85fe6d8d587e36acab52ee2fe13fe08d3e",
    }


def test_card_map_manifest_readme_and_legend_define_fixed_h3_aggregation(
    tmp_path: Path,
) -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))
    card.observe(EunisResult(None, None, None, None), _geometry(2.0, 48.0))

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="rev",
        reference_version="EEA-test",
    )

    expected_map_metadata = {
        "unit": "dataset rows per occupied H3 cell",
        "aggregation": "one representative point per valid polygon-bearing dataset row",
        "includes_unlabeled_rows": True,
        "h3_resolution": 3,
        "count_bands": [
            {"minimum": 1, "maximum": 9, "label": "1-9", "color": "#d7ece3"},
            {"minimum": 10, "maximum": 99, "label": "10-99", "color": "#a4d1c2"},
            {"minimum": 100, "maximum": 999, "label": "100-999", "color": "#67a796"},
            {"minimum": 1000, "maximum": None, "label": "1,000+", "color": "#2f6f66"},
        ],
        "mapped_rows": 2,
        "omitted_rows": 0,
        "omission_policy": (
            "Missing or invalid polygon geometry, and representative points outside longitude "
            "[-180, 180] or latitude [-90, 90], are omitted."
        ),
    }
    assert artifacts.manifest["map_aggregation"] == expected_map_metadata

    readme = artifacts.files["README.md"].read_text(encoding="utf-8")
    assert "![World map of dataset-row coverage by H3 cell]" in readme
    assert "one representative point per valid polygon-bearing dataset row" in readme
    assert "H3 resolution 3" in readme
    assert "Counts include every EUNIS label, including unlabeled rows." in readme
    assert "1-9 rows, 10-99 rows, 100-999 rows, and 1,000+ rows" in readme
    assert "Missing or invalid polygon geometry is omitted." in readme
    assert (
        "Representative points outside longitude [-180, 180] or latitude [-90, 90] are omitted."
        in readme
    )

    svg = artifacts.files["eunis/world-map.svg"].read_text(encoding="utf-8")
    assert 'aria-labelledby="map-title map-description"' in svg
    assert '<title id="map-title">Website H3 coverage map</title>' in svg
    assert '<desc id="map-description">' in svg
    assert "Rows per H3 cell" in svg
    assert "H3 resolution 3" in svg
    assert "Counts include unlabeled rows" in svg
    assert all(label in svg for label in ("1-9", "10-99", "100-999", "1,000+"))
    assert "Missing or invalid geometry is omitted" in svg
    assert "Points outside longitude [-180, 180]" in svg
    assert "or latitude [-90, 90] are omitted" in svg


@pytest.mark.parametrize(
    ("geometry", "point"),
    [
        (_geometry(179.0, 0.0), (0.5, 179.5)),
        (json.dumps(mapping(box(-180.0, 0.0, -179.9, 1.0))), (0.5, -179.95)),
    ],
)
def test_card_map_splits_antimeridian_cells_and_clips_to_plot(
    tmp_path: Path,
    geometry: str,
    point: tuple[float, float],
) -> None:
    cell = h3.latlng_to_cell(*point, 3)
    boundary = h3.cell_to_boundary(cell)
    assert any(
        abs(boundary[index][1] - boundary[index - 1][1]) > 180.0 for index in range(len(boundary))
    )

    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), geometry)
    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="rev",
        reference_version="EEA-test",
    )

    svg = artifacts.files["eunis/world-map.svg"].read_text(encoding="utf-8")
    assert svg.count('fill="#d7ece3" stroke="none"') == 2
    assert '<clipPath id="map-plot-clip">' in svg
    assert '<g clip-path="url(#map-plot-clip)">' in svg


def test_card_map_uses_fixed_absolute_count_bands(tmp_path: Path) -> None:
    card = DatasetCardAccumulator()
    result = EunisResult(None, None, None, None)
    cells = (
        (1, -100.0, 40.0, "#d7ece3"),
        (10, 0.0, 0.0, "#a4d1c2"),
        (100, 70.0, 20.0, "#67a796"),
        (1000, 150.0, -20.0, "#2f6f66"),
    )
    for count, longitude, latitude, _color in cells:
        geometry = _geometry(longitude, latitude)
        for _ in range(count):
            card.observe(result, geometry)

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="rev",
        reference_version="EEA-test",
    )

    svg = artifacts.files["eunis/world-map.svg"].read_text(encoding="utf-8")
    assert all(f'fill="{color}" stroke="none"' in svg for _count, _lon, _lat, color in cells)
    map_metadata = artifacts.manifest["map_aggregation"]
    assert isinstance(map_metadata, dict)
    assert map_metadata["mapped_rows"] == 1111


def test_card_manifest_records_intersection_errors(tmp_path: Path) -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult(None, None, None, None), _geometry(0.0, 0.0))
    assert card.intersection_errors == 0
    card.record_intersection_errors(0)
    card.record_intersection_errors(2)
    card.record_intersection_errors(3)
    with pytest.raises(ValueError, match="non-negative"):
        card.record_intersection_errors(-1)

    assert card.intersection_errors == 5
    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )
    assert artifacts.manifest["intersection_errors"] == 5
    assert artifacts.manifest["invalid_geometries"] == 0


def test_card_accumulator_snapshot_restore_and_merge_round_trip() -> None:
    first = DatasetCardAccumulator()
    first.observe(EunisResult("R11", "steppe", 80.0, "v"), _geometry(1.0, 1.0))
    second = first.spawn()
    second.observe(EunisResult("R11", "steppe", 60.0, "v"), _geometry(1.0, 1.0))
    second.observe(EunisResult(None, None, None, None), None)
    second.record_intersection_errors(2)

    first.merge(second)
    restored = DatasetCardAccumulator()
    restored.restore(json.loads(json.dumps(first.snapshot())))

    assert second.total_rows == 2
    assert first.total_rows == restored.total_rows == 3
    assert restored.intersection_errors == 2
    assert restored.summaries() == first.summaries()
    assert restored.snapshot()["map_snapshot_schema_version"] == 1
    assert restored.snapshot() == first.snapshot()


def test_card_accumulator_rejects_legacy_two_degree_snapshot_without_partial_restore() -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), _geometry(2.0, 48.0))
    before = card.snapshot()
    old_snapshot = {
        "counts": [["R12", 4]],
        "names": [["R12", "Other"]],
        "bins": [["R12", 91, 69, 4]],
        "total_rows": 4,
        "invalid_geometries": 0,
        "intersection_errors": 0,
    }

    with pytest.raises(ValueError, match="legacy 2-degree map snapshot"):
        card.restore(old_snapshot)

    assert card.snapshot() == before


def test_card_accumulator_rejects_invalid_h3_snapshot_counts_before_restore() -> None:
    card = DatasetCardAccumulator()
    card.observe(EunisResult("R11", "Steppe", 50.0, "EEA-test"), _geometry(2.0, 48.0))
    before = card.snapshot()
    valid_cell = h3.latlng_to_cell(48.5, 2.5, 3)
    other_resolution = h3.latlng_to_cell(48.5, 2.5, 2)
    invalid_states = (
        (None, TypeError, "map_counts must be a list"),
        (["bad entry"], TypeError, "entries must contain"),
        ([[None, 1]], TypeError, "H3 cells must be strings"),
        ([["not-an-h3-cell", 1]], ValueError, "invalid resolution-3 H3 cell"),
        ([[other_resolution, 1]], ValueError, "invalid resolution-3 H3 cell"),
        ([[valid_cell, True]], TypeError, "row counts must be integers"),
        ([[valid_cell, 0]], ValueError, "row counts must be positive"),
        ([[valid_cell, 1], [valid_cell, 2]], ValueError, "duplicate H3 cells"),
    )

    for map_counts, error_type, message in invalid_states:
        with pytest.raises(error_type, match=message):
            card.restore({**before, "map_counts": map_counts})
        assert card.snapshot() == before

    with pytest.raises(ValueError, match="map snapshot row counts exceed total_rows"):
        card.restore(
            {
                **before,
                "map_counts": [[valid_cell, before["total_rows"] + 1]],
            }
        )
    assert card.snapshot() == before


def test_card_accumulator_merge_rejects_conflicting_names() -> None:
    first = DatasetCardAccumulator()
    first.observe(EunisResult("R11", "steppe", 80.0, "v"), None)
    other = first.spawn()
    other.observe(EunisResult("R11", "other", 80.0, "v"), None)

    with pytest.raises(ValueError, match="conflicting names"):
        first.merge(other)
