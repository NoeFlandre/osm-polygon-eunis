from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, parsers, scenarios, then, when
from shapely.geometry import box, mapping

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.domain import EunisResult, OverlapCandidate
from osm_polygon_eunis.geometry import to_equal_area
from osm_polygon_eunis.matching import choose_winner
from osm_polygon_eunis.publish import build_manifest
from osm_polygon_eunis.transform import enrich_parquet_shard

scenarios("features/enrich_dataset.feature")


class _GeometryReference:
    def __init__(self, candidates: tuple[OverlapCandidate, ...]) -> None:
        self.candidates = candidates

    def overlap(self, polygon):
        return choose_winner(polygon, self.candidates, source_version="EEA-test")


@given("a source shard with one polygon and two overlapping EUNIS geometries")
def source_shard(tmp_path: Path, acceptance_state) -> None:
    polygon = box(2.0, 48.0, 2.05, 48.05)
    polygon_projected = to_equal_area(polygon)
    first = to_equal_area(box(2.0, 48.0, 2.04, 48.05))
    second = to_equal_area(box(2.0, 48.0, 2.02, 48.05))
    assert polygon_projected is not None
    assert first is not None
    assert second is not None
    state = acceptance_state
    state["source"] = tmp_path / "source.parquet"
    state["destination"] = tmp_path / "output.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["france-test"],
                "name": ["example"],
                "geometry": [json.dumps(mapping(polygon))],
            }
        ),
        state["source"],
    )
    state["reference"] = _GeometryReference(
        (
            OverlapCandidate("R11", "larger", first),
            OverlapCandidate("R12", "smaller", second),
        )
    )
    state["polygon_area"] = polygon_projected.area


@when(parsers.parse("I run the shard enrichment with batch size {batch_size:d}"))
def run_shard(state, batch_size: int) -> None:
    enrich_parquet_shard(
        state["source"],
        state["destination"],
        reference=state["reference"],
        batch_size=batch_size,
    )
    state["output"] = pq.read_table(state["destination"])


@then("the source columns and row count are unchanged")
def source_columns_and_rows(state) -> None:
    source = pq.read_table(state["source"])
    output = state["output"]
    assert output.num_rows == source.num_rows
    assert output.column_names[: len(source.column_names)] == source.column_names
    assert output["polygon_id"].to_pylist() == ["france-test"]


@then("the larger actual intersection supplies the label")
def larger_intersection_label(state) -> None:
    assert state["output"]["eunis_code"].to_pylist() == ["R11"]
    assert state["output"]["eunis_name"].to_pylist() == ["larger"]


@then("the stored overlap percentage is the intersection percentage")
def overlap_percentage(state) -> None:
    percentage = state["output"]["eunis_overlap_percentage"].to_pylist()[0]
    assert percentage == pytest.approx(80.0, abs=0.01)


@given("a Wikidata source file list with polygon and document tables")
def wikidata_files(state) -> None:
    state["source_files"] = (
        "polygons/france.parquet",
        "polygon_document_links/france.parquet",
        "wikipedia/france.parquet",
    )


@when("I build the enrichment manifest")
def enrichment_manifest(state) -> None:
    state["manifest"] = build_manifest(
        source_repo="NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        target_repo="NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis",
        source_revision="source-revision",
        source_paths=state["source_files"],
        changed_paths=(
            "polygons/france.parquet",
            "polygon_document_links/france.parquet",
        ),
        reference_manifest={"source_version": "EEA-test"},
        rows_by_path={
            "polygons/france.parquet": 1,
            "polygon_document_links/france.parquet": 1,
        },
    )


@then("only polygon tables are changed")
def only_polygon_tables_changed(state) -> None:
    assert state["manifest"]["changed_paths"] == [
        "polygon_document_links/france.parquet",
        "polygons/france.parquet",
    ]


@then("document tables remain shared")
def document_tables_shared(state) -> None:
    assert state["manifest"]["shared_paths"] == ["wikipedia/france.parquet"]


@given("a completed label summary")
def completed_label_summary(state, tmp_path: Path) -> None:
    card = DatasetCardAccumulator()
    card.observe(
        EunisResult("R11", "steppe", 75.0, "EEA-test"),
        json.dumps(mapping(box(2.0, 48.0, 2.05, 48.05))),
    )
    state["card"] = card
    state["card_root"] = tmp_path / "card"


@when("I build the dataset card")
def build_dataset_card(state) -> None:
    state["artifacts"] = state["card"].write_artifacts(
        state["card_root"],
        dataset_name="website",
        source_repo="org/source",
        target_repo="org/target",
        source_revision="source-revision",
        reference_version="EEA-test",
    )


@then("the card contains a static map and percentage table")
def card_contains_map_and_table(state) -> None:
    artifacts = state["artifacts"]
    readme = artifacts.files["README.md"].read_text(encoding="utf-8")
    assert "./eunis/world-map.svg" in readme
    assert "| `R11` | steppe | 1 | 100.00% |" in readme
    assert artifacts.files["eunis/world-map.svg"].read_text(encoding="utf-8").startswith("<?xml")


@pytest.fixture
def acceptance_state() -> dict[str, object]:
    return {}


@pytest.fixture
def state(acceptance_state: dict[str, object]) -> dict[str, object]:
    return acceptance_state
