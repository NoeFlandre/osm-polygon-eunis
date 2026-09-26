import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_eunis.domain import EUNIS_FIELDS, EunisResult
from osm_polygon_eunis.transform import (
    append_label_sidecar,
    build_label_map,
    enrich_link_shard,
    enrich_parquet_shard,
    update_label_sidecar,
)


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


def test_label_sidecar_merges_groups_and_appends_without_geometry_loss(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    first_sidecar = tmp_path / "first.parquet"
    second_sidecar = tmp_path / "second.parquet"
    output = tmp_path / "output.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["a", "b"],
                "geometry": [_polygon_json(0), _polygon_json(2)],
            }
        ),
        source,
    )

    class FirstReference:
        def overlap(self, polygon) -> EunisResult:
            del polygon
            return EunisResult("R12", "second", 25.0, "EEA-test")

    class SecondReference:
        def overlap(self, polygon) -> EunisResult:
            del polygon
            return EunisResult("R11", "first", 50.0, "EEA-test")

    update_label_sidecar(source, first_sidecar, reference=FirstReference(), batch_size=1)
    update_label_sidecar(
        source,
        second_sidecar,
        reference=SecondReference(),
        current=first_sidecar,
        batch_size=1,
    )
    append_label_sidecar(source, second_sidecar, output, batch_size=1)
    observed: list[tuple[EunisResult, object]] = []
    append_label_sidecar(
        source,
        second_sidecar,
        tmp_path / "observed-output.parquet",
        batch_size=1,
        observe=lambda result, value: observed.append((result, value)),
    )

    result = pq.read_table(output)
    assert result.column_names == ["polygon_id", "geometry", *EUNIS_FIELDS]
    assert result["eunis_code"].to_pylist() == ["R11", "R11"]
    assert result["polygon_id"].to_pylist() == ["a", "b"]
    assert [item[0].code for item in observed] == ["R11", "R11"]
    assert all(isinstance(item[1], str) for item in observed)


def test_label_sidecar_streams_one_geometry_through_all_references(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    output = tmp_path / "labels.parquet"
    pq.write_table(
        pa.table({"geometry": [_polygon_json(0), _polygon_json(2)]}),
        source,
    )
    seen: list[object] = []

    class Reference:
        def __init__(self, code: str) -> None:
            self.code = code

        def overlap(self, polygon) -> EunisResult:
            seen.append(polygon)
            percentage = 50.0 if self.code == "R11" else 75.0
            return EunisResult(self.code, self.code, percentage, "EEA-test")

    update_label_sidecar(
        source,
        output,
        references=(Reference("R11"), Reference("R12")),
        batch_size=2,
    )

    assert seen[0] is seen[1]
    assert seen[2] is seen[3]
    assert pq.read_table(output)["eunis_code"].to_pylist() == ["R12", "R12"]


def test_build_label_map_joins_polygon_ids_to_a_sidecar_in_batches(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    sidecar = tmp_path / "sidecar.parquet"
    pq.write_table(pa.table({"polygon_id": ["a", "b"]}), source)
    pq.write_table(
        pa.table(
            {
                "eunis_code": ["R11", None],
                "eunis_name": ["steppe", None],
                "eunis_overlap_percentage": [50.0, None],
                "eunis_source_version": ["test", None],
            }
        ),
        sidecar,
    )

    assert build_label_map(source, sidecar, batch_size=1) == {
        "a": EunisResult("R11", "steppe", 50.0, "test"),
        "b": EunisResult(None, None, None, None),
    }


class _ErroringReference:
    """Fake reference that reports ``per_call[i]`` new intersection errors on call i."""

    def __init__(self, code: str, per_call: list[int]) -> None:
        self.code = code
        self.intersection_errors = 0
        self._per_call = iter(per_call)

    def overlap(self, polygon) -> EunisResult:
        del polygon
        self.intersection_errors += next(self._per_call)
        return EunisResult(self.code, self.code, 10.0, "EEA-test")


def test_sidecar_carries_intersection_errors_across_passes(tmp_path: Path) -> None:
    from osm_polygon_eunis.transform import INTERSECTION_ERRORS_FIELD

    source = tmp_path / "source.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["a", "b", "c"],
                "geometry": [_polygon_json(0), _polygon_json(2), _polygon_json(4)],
            }
        ),
        source,
    )
    legacy = tmp_path / "legacy.parquet"
    pq.write_table(
        pa.table(
            {
                "eunis_code": ["R11", None, None],
                "eunis_name": ["R11", None, None],
                "eunis_overlap_percentage": [90.0, None, None],
                "eunis_source_version": ["EEA-test", None, None],
            }
        ),
        legacy,
    )
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    update_label_sidecar(
        source,
        first,
        references=(_ErroringReference("R12", [0, 2, 0]), _ErroringReference("R13", [1, 0, 0])),
        current=legacy,
        batch_size=2,
    )
    assert pq.read_table(first)[INTERSECTION_ERRORS_FIELD].to_pylist() == [1, 2, 0]
    update_label_sidecar(
        source,
        second,
        reference=_ErroringReference("R14", [3, 0, 1]),
        current=first,
        batch_size=2,
    )
    assert pq.read_table(second)[INTERSECTION_ERRORS_FIELD].to_pylist() == [4, 2, 1]
    assert pq.read_table(second)["eunis_code"].to_pylist() == ["R11", "R12", "R12"]

    counts: list[int] = []
    output = tmp_path / "output.parquet"
    append_label_sidecar(source, second, output, batch_size=2, count_errors=counts.append)
    assert counts == [6, 1]
    assert pq.read_table(output).column_names == ["polygon_id", "geometry", *EUNIS_FIELDS]
    labels = build_label_map(source, second, batch_size=2)
    assert labels["a"].code == "R11"

    legacy_counts: list[int] = []
    legacy_output = tmp_path / "legacy-output.parquet"
    append_label_sidecar(
        source, legacy, legacy_output, batch_size=2, count_errors=legacy_counts.append
    )
    assert legacy_counts == [0, 0]
    assert pq.read_table(legacy_output).column_names == pq.read_table(output).column_names


def test_sidecar_without_counter_on_reference_counts_zero_and_bad_schema_fails(
    tmp_path: Path,
) -> None:
    import pytest

    from osm_polygon_eunis.transform import INTERSECTION_ERRORS_FIELD

    source = tmp_path / "source.parquet"
    pq.write_table(pa.table({"geometry": [_polygon_json(0)]}), source)
    sidecar = tmp_path / "labels.parquet"
    update_label_sidecar(
        source,
        sidecar,
        reference=_FakeReference([EunisResult("R11", "a", 5.0, "v")]),
        batch_size=1,
    )
    assert pq.read_table(sidecar)[INTERSECTION_ERRORS_FIELD].to_pylist() == [0]
    wrong = tmp_path / "wrong.parquet"
    pq.write_table(pq.read_table(sidecar).append_column("extra", pa.array([1])), wrong)
    with pytest.raises(ValueError, match="unexpected schema"):
        append_label_sidecar(source, wrong, tmp_path / "out.parquet", batch_size=1)
