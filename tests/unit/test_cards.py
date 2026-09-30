from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from shapely.geometry import GeometryCollection, LineString, Point, Polygon, box, mapping
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis.cards import (
    DatasetCardAccumulator,
    _extract_source_configs,
    _extract_source_license,
    _map_cell_coordinates,
)
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.geometry import GEOMETRY_POLICY


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


def test_card_accumulator_rejects_invalid_cell_size_and_conflicting_names() -> None:
    with pytest.raises(ValueError, match="cell_size"):
        DatasetCardAccumulator(cell_size=0.0)

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


def test_map_cell_coordinates_uses_representative_point_and_rejects_outside_world() -> None:
    polygon: BaseGeometry = box(2.0, 48.0, 3.0, 49.0)
    outside: BaseGeometry = box(200.0, 0.0, 201.0, 1.0)

    assert _map_cell_coordinates(polygon, 2.0) == (91, 69)
    assert _map_cell_coordinates(outside, 2.0) is None


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
        "README.md": "0ae2d67b0d5d2a85c7bbee1aaaef918c3d27c7ad78b009ad5fbf321dbbd56848",
        "eunis/world-map.svg": "adb1d18e0876692cb71143402828c0aaa5a2df98762e9b43bcb6353d8fdbf731",
    }


def test_card_map_subtitle_and_readme_follow_cell_size(tmp_path: Path) -> None:
    card = DatasetCardAccumulator(cell_size=0.5)
    card.observe(EunisResult("R11", "Steppe", 80.0, "EEA-test"), _geometry(2.0, 48.0))

    artifacts = card.write_artifacts(
        tmp_path,
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="rev",
        reference_version="EEA-test",
    )

    assert "· 0.5° bins ·" in artifacts.files["eunis/world-map.svg"].read_text(encoding="utf-8")
    assert "0.5-degree geographic bins" in artifacts.files["README.md"].read_text(encoding="utf-8")


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
