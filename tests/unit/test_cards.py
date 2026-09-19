from __future__ import annotations

import json
from pathlib import Path

import pytest
from shapely.geometry import box, mapping

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.domain import EunisResult


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

    readme = first.files["README.md"].read_text(encoding="utf-8")
    world_map = first.files["eunis/world-map.svg"].read_text(encoding="utf-8")
    assert "./eunis/world-map.svg" in readme
    assert "| `R11` | Steppe | 2 | 66.67% |" in readme
    assert "| `—` | No EUNIS label | 1 | 33.33% |" in readme
    assert world_map.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert "R11" in world_map
    assert first.manifest["total_rows"] == 3
    assert (
        first.files["eunis/world-map.svg"].read_bytes()
        == second.files["eunis/world-map.svg"].read_bytes()
    )
    assert first.hashes == {
        "README.md": first.manifest["readme_sha256"],
        "eunis/world-map.svg": first.manifest["map_sha256"],
    }


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
