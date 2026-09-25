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

    assert [
        (summary.code, summary.name, summary.rows, summary.percentage)
        for summary in card.summaries()
    ] == [
        ("R11", "Steppe", 2, pytest.approx(200 / 3)),
        (None, "No EUNIS label", 1, pytest.approx(100 / 3)),
    ]
    assert "eunis/world-map.svg" in first.files["README.md"].read_text(encoding="utf-8")
    assert first.manifest["total_rows"] == 3
    for path in ("README.md", "eunis/world-map.svg"):
        assert first.files[path].read_bytes() == second.files[path].read_bytes()
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
    # Card and map hashes feed the release manifest and no-op detection.
    artifacts = _golden_card().write_artifacts(
        tmp_path,
        dataset_name="website-polygons",
        source_repo="o/s",
        target_repo="o/t",
        source_revision="rev",
        reference_version="EEA",
    )

    assert dict(artifacts.hashes) == {
        "README.md": "bfedb82974521e86cbec5eec9de77d88adfd3e15a10c5cdf28b80050eabedc7e",
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
