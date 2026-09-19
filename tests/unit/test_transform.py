import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_eunis.domain import EUNIS_FIELDS, EunisResult
from osm_polygon_eunis.transform import enrich_link_shard, enrich_parquet_shard


class _FakeReference:
    def __init__(self, results: list[EunisResult]) -> None:
        self._results = iter(results)

    def overlap(self, polygon) -> EunisResult:
        del polygon
        return next(self._results)


def _polygon_json(x: int) -> str:
    return json.dumps(
        {
            "type": "Polygon",
            "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 0]]],
        }
    )


def test_enrich_shard_preserves_rows_and_appends_nullable_fields(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "output.parquet"
    source_table = pa.table(
        {
            "polygon_id": ["a", "b", "c"],
            "name": ["A", "B", "C"],
            "geometry": [_polygon_json(0), None, _polygon_json(2)],
        }
    )
    pq.write_table(source_table, source)

    rows = enrich_parquet_shard(
        source,
        destination,
        reference=_FakeReference(
            [
                EunisResult("R11", "steppe", 25.0, "test"),
                EunisResult(None, None, None, None),
                EunisResult("R12", "grassland", 75.0, "test"),
            ]
        ),
        batch_size=2,
    )

    result = pq.read_table(destination)
    assert rows == 3
    assert result.num_rows == source_table.num_rows
    assert result.column_names[-4:] == list(EUNIS_FIELDS)
    assert result["polygon_id"].to_pylist() == ["a", "b", "c"]
    assert result["name"].to_pylist() == ["A", "B", "C"]
    assert result["eunis_code"].to_pylist() == ["R11", None, "R12"]
    assert result["eunis_overlap_percentage"].to_pylist() == [25.0, None, 75.0]


def test_enrich_link_shard_uses_polygon_id_map(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    destination = tmp_path / "links-output.parquet"
    pq.write_table(pa.table({"polygon_id": ["a", "missing"]}), source)

    rows = enrich_link_shard(
        source,
        destination,
        labels_by_polygon_id={"a": EunisResult("R11", "steppe", 50.0, "test")},
        batch_size=1,
    )

    result = pq.read_table(destination)
    assert rows == 2
    assert result["polygon_id"].to_pylist() == ["a", "missing"]
    assert result["eunis_code"].to_pylist() == ["R11", None]
    assert result["eunis_source_version"].to_pylist() == ["test", None]
