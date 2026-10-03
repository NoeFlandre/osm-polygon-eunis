"""Independent examples of the published label and sidecar contracts."""

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis import transform
from osm_polygon_eunis.domain import EUNIS_FIELDS, EunisResult
from osm_polygon_eunis.transform import (
    SidecarAppendOptions,
    SidecarUpdateOptions,
    append_label_sidecar,
    enrich_link_shard,
    enrich_parquet_shard,
    update_label_sidecar,
)

# Keep the expected contract independent of the writer's field definition.
_EXPECTED_LABEL_SCHEMA = pa.schema(
    [
        pa.field("eunis_code", pa.string(), nullable=True),
        pa.field("eunis_name", pa.string(), nullable=True),
        pa.field("eunis_overlap_percentage", pa.float64(), nullable=True),
        pa.field("eunis_source_version", pa.string(), nullable=True),
    ]
)
_EXPECTED_SIDECAR_SCHEMA = _EXPECTED_LABEL_SCHEMA.append(
    pa.field("eunis_intersection_errors", pa.int64(), nullable=False)
)
_CASES = [
    pytest.param([], [[], [], [], []], id="empty"),
    pytest.param([EunisResult(None, None, None, None)], [[None]] * 4, id="null"),
    pytest.param(
        [
            EunisResult("R11", "steppe", 25.0, "EEA-test"),
            EunisResult(None, None, None, None),
            EunisResult("R12", "prairie côtière", 75.0, "second-version"),
            EunisResult("R13", None, 0.0, None),
        ],
        [
            ["R11", None, "R12", "R13"],
            ["steppe", None, "prairie côtière", None],
            [25.0, None, 75.0, 0.0],
            ["EEA-test", None, "second-version", None],
        ],
        id="populated-and-null",
    ),
]


class _Reference:
    def __init__(self, results):
        self.results = iter(results)

    def overlap(self, polygon):
        del polygon
        return next(self.results)


def _expected_labels(columns):
    return pa.Table.from_arrays(columns, schema=_EXPECTED_LABEL_SCHEMA)


def _source_table(rows):
    schema = pa.schema(
        [
            pa.field("polygon_id", pa.string(), nullable=False, metadata={b"source": b"id"}),
            pa.field("geometry", pa.binary(), nullable=True),
            pa.field("rank", pa.int32(), nullable=False),
        ],
        metadata={b"dataset": b"source-metadata", b"custom": b"preserve-me"},
    )
    return pa.Table.from_arrays(
        [[f"polygon-{row}" for row in range(rows)], [None] * rows, list(range(rows))],
        schema=schema,
    )


def _expected_output(source, labels):
    for field in _EXPECTED_LABEL_SCHEMA:
        source = source.append_column(field, labels[field.name])
    return source


def _assert_batches(path, row_count):
    metadata = pq.ParquetFile(path).metadata
    sizes = [metadata.row_group(index).num_rows for index in range(metadata.num_row_groups)]
    assert sum(sizes) == row_count
    assert all(size <= 2 for size in sizes)


@pytest.mark.parametrize(("results", "columns"), _CASES)
def test_explicit_label_schema_and_result_round_trip(results, columns):
    errors = list(range(len(results)))
    expected = _expected_labels(columns).append_column(
        _EXPECTED_SIDECAR_SCHEMA.field("eunis_intersection_errors"),
        pa.array(errors, type=pa.int64()),
    )
    actual = transform._results_table(results, errors)

    assert list(EUNIS_FIELDS) == [
        "eunis_code",
        "eunis_name",
        "eunis_overlap_percentage",
        "eunis_source_version",
    ]
    assert actual.schema.equals(_EXPECTED_SIDECAR_SCHEMA, check_metadata=True)
    assert actual.equals(expected, check_metadata=True)
    assert transform._table_results(actual) == results
    assert transform._table_results(_expected_labels(columns)) == results
    assert transform._table_errors(actual) == errors
    assert transform._table_errors(_expected_labels(columns)) == [0] * len(results)


@pytest.mark.parametrize(("results", "columns"), _CASES)
def test_direct_and_sidecar_paths_preserve_the_same_contract(tmp_path, results, columns):
    source = _source_table(len(results))
    source_path = tmp_path / "source.parquet"
    direct = tmp_path / "direct.parquet"
    link = tmp_path / "link.parquet"
    sidecar = tmp_path / "sidecar.parquet"
    appended = tmp_path / "appended.parquet"
    pq.write_table(source, source_path)

    assert enrich_parquet_shard(
        source_path, direct, reference=_Reference(results), batch_size=2
    ) == len(results)
    assert update_label_sidecar(
        source_path, sidecar, SidecarUpdateOptions(reference=_Reference(results), batch_size=2)
    ) == len(results)
    assert enrich_link_shard(
        source_path,
        link,
        labels_by_polygon_id=dict(zip(source["polygon_id"].to_pylist(), results, strict=True)),
        batch_size=2,
    ) == len(results)
    counts = []
    observed = []
    assert append_label_sidecar(
        source_path,
        sidecar,
        appended,
        SidecarAppendOptions(
            batch_size=2,
            count_errors=counts.append,
            observe=lambda result, geometry: observed.append(result),
        ),
    ) == len(results)

    expected = _expected_output(source, _expected_labels(columns))
    assert pq.read_table(direct).equals(expected, check_metadata=True)
    assert pq.read_table(link).equals(expected, check_metadata=True)
    assert pq.read_table(appended).equals(expected, check_metadata=True)
    assert pq.read_table(sidecar).schema.equals(_EXPECTED_SIDECAR_SCHEMA, check_metadata=True)
    assert pq.read_table(sidecar)["eunis_intersection_errors"].to_pylist() == [0] * len(results)
    assert observed == results
    assert counts == [0] * ((len(results) + 1) // 2)
    for path in (direct, link, sidecar, appended):
        _assert_batches(path, len(results))


@pytest.mark.parametrize(("results", "columns"), _CASES)
def test_legacy_sidecars_preserve_labels_and_default_error_counts(tmp_path, results, columns):
    source = _source_table(len(results))
    labels = _expected_labels(columns)
    source_path = tmp_path / "source.parquet"
    legacy = tmp_path / "legacy.parquet"
    output = tmp_path / "output.parquet"
    updated = tmp_path / "updated.parquet"
    pq.write_table(source, source_path)
    pq.write_table(labels, legacy)
    counts = []
    observed = []

    append_label_sidecar(
        source_path,
        legacy,
        output,
        SidecarAppendOptions(
            batch_size=2,
            count_errors=counts.append,
            observe=lambda result, geometry: observed.append(result),
        ),
    )
    update_label_sidecar(
        source_path,
        updated,
        SidecarUpdateOptions(
            reference=_Reference([EunisResult(None, None, None, None)] * len(results)),
            batch_size=2,
            current=legacy,
        ),
    )

    assert pq.read_table(output).equals(_expected_output(source, labels), check_metadata=True)
    assert observed == results
    assert counts == [0] * ((len(results) + 1) // 2)
    upgraded = pq.read_table(updated)
    assert upgraded.schema.equals(_EXPECTED_SIDECAR_SCHEMA, check_metadata=True)
    assert upgraded.select(_EXPECTED_LABEL_SCHEMA.names).equals(labels, check_metadata=True)
    assert upgraded["eunis_intersection_errors"].to_pylist() == [0] * len(results)


@pytest.mark.parametrize("field", list(_EXPECTED_LABEL_SCHEMA))
def test_duplicate_labels_are_rejected_before_creating_output(tmp_path: Path, field):
    source = tmp_path / "source.parquet"
    destination = tmp_path / "output.parquet"
    table = _source_table(0).append_column(field, pa.array([], type=field.type))
    pq.write_table(table, source)

    with pytest.raises(ValueError, match="source already contains EUNIS fields"):
        enrich_parquet_shard(source, destination, reference=_Reference([]), batch_size=2)

    assert not destination.exists()
