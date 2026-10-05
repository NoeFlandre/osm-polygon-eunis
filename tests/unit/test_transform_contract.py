"""Exact-value contract tests for the streaming transforms (mutation-testing targets)."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis import transform
from osm_polygon_eunis.domain import EunisResult, SchemaError
from osm_polygon_eunis.geometry import parse_geometry, to_equal_area
from osm_polygon_eunis.transform import (
    SidecarAppendOptions,
    SidecarUpdateOptions,
    append_label_sidecar,
    build_label_map,
    enrich_link_shard,
    enrich_parquet_shard,
    update_label_sidecar,
)

_LABEL_COLUMNS = ("eunis_code", "eunis_name", "eunis_overlap_percentage", "eunis_source_version")


def _polygon_json(x: int) -> str:
    return json.dumps(
        {"type": "Polygon", "coordinates": [[[x, 0], [x + 1, 0], [x + 1, 1], [x, 0]]]}
    )


class _Reference:
    def __init__(self, code: str = "R11") -> None:
        self.code = code
        self.polygons: list[object] = []

    def overlap(self, polygon) -> EunisResult:
        self.polygons.append(polygon)
        return EunisResult(self.code, "name", 50.0, "v")


def _message(call: Callable[[], object], kind: type[Exception] = ValueError) -> str:
    with pytest.raises(kind) as error:
        call()
    return str(error.value)


def _geometry_shard(path: Path, rows: int = 2, row_group_size: int | None = None) -> None:
    table = pa.table(
        {
            "polygon_id": [f"p{i}" for i in range(rows)],
            "geometry": [_polygon_json(i * 2) for i in range(rows)],
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _sidecar(path: Path, rows: int, row_group_size: int | None = None) -> None:
    table = pa.table(
        {
            "eunis_code": ["R11"] * rows,
            "eunis_name": ["name"] * rows,
            "eunis_overlap_percentage": [50.0] * rows,
            "eunis_source_version": ["v"] * rows,
        }
    )
    pq.write_table(table, path, row_group_size=row_group_size)


def _expected_geometry(x: int) -> object:
    return to_equal_area(parse_geometry(_polygon_json(x)))


def test_reference_without_a_counter_reports_zero_errors() -> None:
    assert transform._reference_errors(cast(transform.OverlapReference, object())) == 0


def test_every_reference_receives_the_equal_area_geometry(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source)
    enrich_reference = _Reference()
    enrich_parquet_shard(
        source, tmp_path / "enriched.parquet", reference=enrich_reference, batch_size=1
    )
    update_reference = _Reference()
    update_label_sidecar(
        source,
        tmp_path / "labels.parquet",
        SidecarUpdateOptions(reference=update_reference, batch_size=1),
    )

    for reference in (enrich_reference, update_reference):
        assert len(reference.polygons) == 2
        for polygon, x in zip(reference.polygons, (0, 2), strict=True):
            assert polygon.equals(_expected_geometry(x))  # ty: ignore[unresolved-attribute]


def test_outputs_use_zstd_compression(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source)
    enriched = tmp_path / "enriched.parquet"
    labels = tmp_path / "labels.parquet"
    appended = tmp_path / "appended.parquet"
    enrich_parquet_shard(source, enriched, reference=_Reference(), batch_size=2)
    update_label_sidecar(source, labels, SidecarUpdateOptions(reference=_Reference(), batch_size=2))
    append_label_sidecar(source, labels, appended, SidecarAppendOptions(batch_size=2))

    for path in (enriched, labels, appended):
        assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"


def test_batch_size_message_is_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source)

    message = _message(
        lambda: enrich_parquet_shard(
            source, tmp_path / "out.parquet", reference=_Reference(), batch_size=0
        )
    )

    assert message == "batch_size must be positive"


def test_reference_selection_messages_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source)
    out = tmp_path / "out.parquet"

    both = SidecarUpdateOptions(reference=_Reference(), references=(_Reference(),))
    neither = SidecarUpdateOptions()

    assert (
        _message(lambda: update_label_sidecar(source, out, both))
        == "provide reference or references, not both"
    )
    assert (
        _message(lambda: update_label_sidecar(source, out, neither))
        == "at least one overlap reference is required"
    )


def test_current_sidecar_must_differ_from_the_destination(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source)
    labels = tmp_path / "labels.parquet"
    update_label_sidecar(source, labels, SidecarUpdateOptions(reference=_Reference()))
    other = tmp_path / "other.parquet"

    message = _message(
        lambda: update_label_sidecar(
            source, labels, SidecarUpdateOptions(reference=_Reference(), current=labels)
        )
    )
    update_label_sidecar(
        source, other, SidecarUpdateOptions(reference=_Reference(), current=labels)
    )

    assert message == "current sidecar and destination must differ"
    assert pq.read_table(other).num_rows == 2


def test_current_sidecar_row_count_mismatches_have_exact_messages(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source, rows=2)
    out = tmp_path / "out.parquet"
    short = tmp_path / "short.parquet"
    long = tmp_path / "long.parquet"
    _sidecar(short, rows=1)
    _sidecar(long, rows=3)

    def update(current: Path) -> Callable[[], object]:
        options = SidecarUpdateOptions(reference=_Reference(), current=current, batch_size=2)
        return lambda: update_label_sidecar(source, out, options)

    assert _message(update(short)) == "current sidecar batch boundaries do not match source"
    assert _message(update(long)) == "current sidecar has more rows than source"

    _geometry_shard(source, rows=4)
    _sidecar(short, rows=2)
    assert _message(update(short)) == "current sidecar has fewer rows than source"


def test_sidecar_schema_and_alignment_messages_are_exact(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    _geometry_shard(source, rows=4)
    out = tmp_path / "out.parquet"
    short = tmp_path / "short.parquet"
    long = tmp_path / "long.parquet"
    skewed = tmp_path / "skewed.parquet"
    wrong = tmp_path / "wrong.parquet"
    _sidecar(short, rows=2)
    _sidecar(long, rows=6)
    _sidecar(skewed, rows=4)
    pq.write_table(pa.table({"unexpected": [1, 2, 3, 4]}), wrong)
    options = SidecarAppendOptions(batch_size=2)

    assert (
        _message(lambda: append_label_sidecar(source, wrong, out, options))
        == "sidecar has an unexpected schema"
    )
    assert (
        _message(lambda: append_label_sidecar(source, short, out, options))
        == "sidecar has fewer rows than source"
    )
    assert (
        _message(lambda: append_label_sidecar(source, long, out, options))
        == "sidecar has more rows than source"
    )
    odd_source = tmp_path / "odd.parquet"
    _geometry_shard(odd_source, rows=3)
    assert (
        _message(lambda: append_label_sidecar(odd_source, skewed, out, options))
        == "sidecar batch boundaries do not match source"
    )
    assert (
        _message(lambda: build_label_map(source, short, batch_size=2))
        == "sidecar has fewer rows than source"
    )
    assert (
        _message(lambda: build_label_map(source, long, batch_size=2))
        == "sidecar has more rows than source"
    )


def test_build_label_map_reads_only_the_polygon_id_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source.parquet"
    labels = tmp_path / "labels.parquet"
    _geometry_shard(source)
    _sidecar(labels, rows=2)
    requested: list[object] = []

    class Spy(pq.ParquetFile):
        def iter_batches(self, *args, **kwargs):
            requested.append(kwargs.get("columns", "all columns"))
            return super().iter_batches(*args, **kwargs)

    monkeypatch.setattr(transform.pq, "ParquetFile", Spy)

    build_label_map(source, labels, batch_size=2)

    assert requested == [["polygon_id"], "all columns"]


def test_build_label_map_rejects_missing_columns_and_bad_ids(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    labels = tmp_path / "labels.parquet"
    _geometry_shard(source)
    _sidecar(labels, rows=2)
    numeric = tmp_path / "numeric.parquet"
    duplicate = tmp_path / "duplicate.parquet"
    pq.write_table(pa.table({"polygon_id": [1, 2]}), numeric)
    pq.write_table(pa.table({"polygon_id": ["a", "a"]}), duplicate)

    assert (
        _message(lambda: build_label_map(source, labels, batch_size=1, polygon_id_column="nope"))
        == "polygon id column 'nope' is missing"
    )
    assert (
        _message(lambda: build_label_map(numeric, labels, batch_size=2), SchemaError)
        == "polygon id column contains a non-string value"
    )
    assert (
        _message(lambda: build_label_map(duplicate, labels, batch_size=2))
        == "duplicate polygon id 'a'"
    )


def test_length_mismatches_are_rejected_instead_of_truncated() -> None:
    result = EunisResult("R11", "name", 50.0, "v")
    geometry = _polygon_json(0)
    table = pa.table({"geometry": [geometry, geometry]})
    sidecar = pa.table(
        {
            "eunis_code": ["R11"],
            "eunis_name": ["name"],
            "eunis_overlap_percentage": [50.0],
            "eunis_source_version": ["v"],
        }
    )

    with pytest.raises(ValueError, match="zip"):
        transform._updated_results([geometry, geometry], [result], [0], (_Reference(),))
    with pytest.raises(ValueError, match="zip"):
        transform._add_labels({}, ["a", "b"], [result])
    with pytest.raises(ValueError, match="zip"):
        transform._observe_batch(table, sidecar, "geometry", lambda *_args: None)


def test_link_shard_leaves_unknown_polygons_unlabelled(tmp_path: Path) -> None:
    source = tmp_path / "links.parquet"
    pq.write_table(pa.table({"polygon_id": ["known", "unknown"]}), source)
    out = tmp_path / "out.parquet"

    enrich_link_shard(
        source,
        out,
        labels_by_polygon_id={"known": EunisResult("R11", "name", 50.0, "v")},
        batch_size=2,
    )

    table = pq.read_table(out)
    assert [table[column].to_pylist() for column in _LABEL_COLUMNS] == [
        ["R11", None],
        ["name", None],
        [50.0, None],
        ["v", None],
    ]
