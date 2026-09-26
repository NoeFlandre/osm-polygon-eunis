"""Bounded Arrow/Parquet transformations at the pipeline's I/O boundary."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from .domain import EUNIS_FIELDS, EunisResult, SchemaError
from .geometry import parse_geometry, to_equal_area
from .matching import prefer_result


class OverlapReference(Protocol):
    """Minimal reference interface needed by the batch transformer."""

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult: ...


# Sidecar-only column: per-row count of candidates dropped because the exact GEOS
# intersection raised, summed across reference passes. It is never appended to
# published shards. Sidecars written before it existed read as zero.
INTERSECTION_ERRORS_FIELD = "eunis_intersection_errors"


def _reference_errors(reference: OverlapReference) -> int:
    return int(getattr(reference, "intersection_errors", 0))


def _counted_overlap(
    reference: OverlapReference, geometry: BaseGeometry | None
) -> tuple[EunisResult, int]:
    before = _reference_errors(reference)
    result = reference.overlap(geometry)
    return result, _reference_errors(reference) - before


def _validate_batch_size(batch_size: int) -> None:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")


def _output_schema(schema: pa.Schema) -> pa.Schema:
    existing = set(schema.names)
    duplicate_fields = existing.intersection(EUNIS_FIELDS)
    if duplicate_fields:
        raise ValueError(f"source already contains EUNIS fields: {sorted(duplicate_fields)}")
    fields = [
        pa.field("eunis_code", pa.string(), nullable=True),
        pa.field("eunis_name", pa.string(), nullable=True),
        pa.field("eunis_overlap_percentage", pa.float64(), nullable=True),
        pa.field("eunis_source_version", pa.string(), nullable=True),
    ]
    return schema.append(fields[0]).append(fields[1]).append(fields[2]).append(fields[3])


def _append_results(table: pa.Table, results: list[EunisResult]) -> pa.Table:
    for name, data_type, values in _result_columns(results):
        table = table.append_column(name, pa.array(values, type=data_type))
    return table


def _result_columns(
    results: list[EunisResult],
) -> tuple[tuple[str, pa.DataType, list[object]], ...]:
    return (
        ("eunis_code", pa.string(), [result.code for result in results]),
        ("eunis_name", pa.string(), [result.name for result in results]),
        (
            "eunis_overlap_percentage",
            pa.float64(),
            [result.overlap_percentage for result in results],
        ),
        ("eunis_source_version", pa.string(), [result.source_version for result in results]),
    )


def _sidecar_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("eunis_code", pa.string(), nullable=True),
            pa.field("eunis_name", pa.string(), nullable=True),
            pa.field("eunis_overlap_percentage", pa.float64(), nullable=True),
            pa.field("eunis_source_version", pa.string(), nullable=True),
            pa.field(INTERSECTION_ERRORS_FIELD, pa.int64(), nullable=False),
        ]
    )


def _results_table(results: list[EunisResult], errors: list[int]) -> pa.Table:
    return pa.table(
        {
            "eunis_code": [result.code for result in results],
            "eunis_name": [result.name for result in results],
            "eunis_overlap_percentage": [result.overlap_percentage for result in results],
            "eunis_source_version": [result.source_version for result in results],
            INTERSECTION_ERRORS_FIELD: errors,
        },
        schema=_sidecar_schema(),
    )


def _table_results(table: pa.Table) -> list[EunisResult]:
    columns = [table[name].to_pylist() for name in EUNIS_FIELDS]
    return [EunisResult(*values) for values in zip(*columns, strict=True)]


def _table_errors(table: pa.Table) -> list[int]:
    if INTERSECTION_ERRORS_FIELD not in table.column_names:
        return [0] * table.num_rows
    return [int(value) for value in table[INTERSECTION_ERRORS_FIELD].to_pylist()]


def _writer(destination: Path, schema: pa.Schema) -> pq.ParquetWriter:
    return pq.ParquetWriter(destination, schema, compression="zstd")


def enrich_parquet_shard(
    source: Path,
    destination: Path,
    *,
    reference: OverlapReference,
    batch_size: int,
    geometry_column: str = "geometry",
) -> int:
    """Enrich a geometry shard while retaining every source row and field."""

    _validate_batch_size(batch_size)
    parquet_file = pq.ParquetFile(source)
    source_schema = parquet_file.schema_arrow
    if geometry_column not in source_schema.names:
        raise ValueError(f"geometry column {geometry_column!r} is missing")
    output_schema = _output_schema(source_schema)
    rows = 0
    with _writer(destination, output_schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=source_schema)
            results = []
            for value in table[geometry_column].to_pylist():
                geometry = to_equal_area(parse_geometry(value))
                results.append(reference.overlap(geometry))
            writer.write_table(_append_results(table, results))
            rows += batch.num_rows
    return rows


def enrich_link_shard(
    source: Path,
    destination: Path,
    *,
    labels_by_polygon_id: Mapping[str, EunisResult],
    batch_size: int,
    polygon_id_column: str = "polygon_id",
) -> int:
    """Enrich a link shard using only the current region's polygon label map."""

    _validate_batch_size(batch_size)
    parquet_file = pq.ParquetFile(source)
    source_schema = parquet_file.schema_arrow
    if polygon_id_column not in source_schema.names:
        raise ValueError(f"polygon id column {polygon_id_column!r} is missing")
    output_schema = _output_schema(source_schema)
    rows = 0
    empty = EunisResult(None, None, None, None)
    with _writer(destination, output_schema) as writer:
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch], schema=source_schema)
            results = [
                labels_by_polygon_id.get(polygon_id, empty)
                for polygon_id in table[polygon_id_column].to_pylist()
            ]
            writer.write_table(_append_results(table, results))
            rows += batch.num_rows
    return rows


def update_label_sidecar(
    source: Path,
    destination: Path,
    *,
    reference: OverlapReference | None = None,
    references: tuple[OverlapReference, ...] = (),
    batch_size: int,
    current: Path | None = None,
    geometry_column: str = "geometry",
) -> int:
    """Update labels in one source pass while keeping one batch live."""

    _validate_batch_size(batch_size)
    selected_references = _selected_references(reference, references)
    source_file, current_file = _sidecar_inputs(source, destination, current, geometry_column)
    source_schema = source_file.schema_arrow
    empty = EunisResult(None, None, None, None)
    current_batches = (
        iter(current_file.iter_batches(batch_size=batch_size)) if current_file else None
    )
    rows = 0
    with pq.ParquetWriter(destination, _sidecar_schema(), compression="zstd") as writer:
        for batch in source_file.iter_batches(batch_size=batch_size):
            source_table = pa.Table.from_batches([batch], schema=source_schema)
            previous, previous_errors = _previous_results(current_batches, batch.num_rows, empty)
            updated, errors = _updated_results(
                source_table[geometry_column].to_pylist(),
                previous,
                previous_errors,
                selected_references,
            )
            writer.write_table(_results_table(updated, errors))
            rows += batch.num_rows
    _ensure_no_extra_batches(current_batches, "current sidecar has more rows than source")
    return rows


def _selected_references(
    reference: OverlapReference | None,
    references: tuple[OverlapReference, ...],
) -> tuple[OverlapReference, ...]:
    if reference is not None:
        if references:
            raise ValueError("provide reference or references, not both")
        return (reference,)
    if not references:
        raise ValueError("at least one overlap reference is required")
    return references


def _sidecar_inputs(
    source: Path,
    destination: Path,
    current: Path | None,
    geometry_column: str,
) -> tuple[pq.ParquetFile, pq.ParquetFile | None]:
    _validate_sidecar_paths(current, destination)
    source_file = pq.ParquetFile(source)
    _validate_geometry_column(source_file, geometry_column)
    current_file = pq.ParquetFile(current) if current is not None else None
    if current_file is not None:
        _validate_sidecar_schema(current_file)
    return source_file, current_file


def _validate_sidecar_paths(current: Path | None, destination: Path) -> None:
    if current is not None and current == destination:
        raise ValueError("current sidecar and destination must differ")


def _validate_geometry_column(source_file: pq.ParquetFile, geometry_column: str) -> None:
    if geometry_column not in source_file.schema_arrow.names:
        raise ValueError(f"geometry column {geometry_column!r} is missing")


def _previous_results(
    batches: Iterator[pa.RecordBatch] | None,
    row_count: int,
    empty: EunisResult,
) -> tuple[list[EunisResult], list[int]]:
    if batches is None:
        return [empty] * row_count, [0] * row_count
    try:
        previous_batch = next(batches)
    except StopIteration as error:
        raise ValueError("current sidecar has fewer rows than source") from error
    if previous_batch.num_rows != row_count:
        raise ValueError("current sidecar batch boundaries do not match source")
    previous_table = pa.Table.from_batches([previous_batch])
    return _table_results(previous_table), _table_errors(previous_table)


def _updated_results(
    geometries: list[object],
    previous: list[EunisResult],
    previous_errors: list[int],
    references: tuple[OverlapReference, ...],
) -> tuple[list[EunisResult], list[int]]:
    results: list[EunisResult] = []
    errors: list[int] = []
    for value, existing, existing_errors in zip(geometries, previous, previous_errors, strict=True):
        geometry = to_equal_area(parse_geometry(value))
        result = existing
        row_errors = existing_errors
        for reference in references:
            candidate, new_errors = _counted_overlap(reference, geometry)
            result = prefer_result(result, candidate)
            row_errors += new_errors
        results.append(result)
        errors.append(row_errors)
    return results, errors


def _ensure_no_extra_batches(
    batches: Iterator[pa.RecordBatch] | None,
    message: str,
) -> None:
    if batches is None:
        return
    try:
        next(batches)
    except StopIteration:
        return
    raise ValueError(message)


def _validate_sidecar_schema(sidecar_file: pq.ParquetFile) -> None:
    names = sidecar_file.schema_arrow.names
    if names not in (list(EUNIS_FIELDS), [*EUNIS_FIELDS, INTERSECTION_ERRORS_FIELD]):
        raise ValueError("sidecar has an unexpected schema")


def _next_matching_batch(
    batches: Iterator[pa.RecordBatch],
    row_count: int,
    fewer_message: str,
) -> pa.RecordBatch:
    try:
        batch = next(batches)
    except StopIteration as error:
        raise ValueError(fewer_message) from error
    if batch.num_rows != row_count:
        raise ValueError("sidecar batch boundaries do not match source")
    return batch


def append_label_sidecar(
    source: Path,
    sidecar: Path,
    destination: Path,
    *,
    batch_size: int,
    observe: Callable[[EunisResult, object], None] | None = None,
    count_errors: Callable[[int], None] | None = None,
    geometry_column: str = "geometry",
) -> int:
    """Append a completed label sidecar to a source shard in bounded batches.

    ``count_errors`` receives each batch's summed intersection-error count; the
    counter column itself is never written to the published shard.
    """

    _validate_batch_size(batch_size)
    source_file = pq.ParquetFile(source)
    sidecar_file = pq.ParquetFile(sidecar)
    source_schema = source_file.schema_arrow
    if geometry_column not in source_schema.names:
        raise ValueError(f"geometry column {geometry_column!r} is missing")
    _validate_sidecar_schema(sidecar_file)
    output_schema = _output_schema(source_schema)
    sidecar_batches = iter(sidecar_file.iter_batches(batch_size=batch_size))
    rows = 0
    with _writer(destination, output_schema) as writer:
        for batch in source_file.iter_batches(batch_size=batch_size):
            sidecar_batch = _next_matching_batch(
                sidecar_batches,
                batch.num_rows,
                "sidecar has fewer rows than source",
            )
            table = pa.Table.from_batches([batch], schema=source_schema)
            sidecar_table = pa.Table.from_batches([sidecar_batch])
            _observe_batch(table, sidecar_table, geometry_column, observe)
            if count_errors is not None:
                count_errors(sum(_table_errors(sidecar_table)))
            for name in EUNIS_FIELDS:
                table = table.append_column(name, sidecar_table[name])
            writer.write_table(table)
            rows += batch.num_rows
    _ensure_no_extra_batches(sidecar_batches, "sidecar has more rows than source")
    return rows


def _observe_batch(
    source_table: pa.Table,
    sidecar_table: pa.Table,
    geometry_column: str,
    observe: Callable[[EunisResult, object], None] | None,
) -> None:
    if observe is None:
        return
    for value, result in zip(
        source_table[geometry_column].to_pylist(),
        _table_results(sidecar_table),
        strict=True,
    ):
        observe(result, value)


def build_label_map(
    source: Path,
    sidecar: Path,
    *,
    batch_size: int,
    polygon_id_column: str = "polygon_id",
) -> dict[str, EunisResult]:
    """Join one polygon shard to its labels without loading source geometry."""

    _validate_batch_size(batch_size)
    source_file = pq.ParquetFile(source)
    sidecar_file = pq.ParquetFile(sidecar)
    if polygon_id_column not in source_file.schema_arrow.names:
        raise ValueError(f"polygon id column {polygon_id_column!r} is missing")
    _validate_sidecar_schema(sidecar_file)
    source_batches = iter(
        source_file.iter_batches(columns=[polygon_id_column], batch_size=batch_size)
    )
    sidecar_batches = iter(sidecar_file.iter_batches(batch_size=batch_size))
    labels: dict[str, EunisResult] = {}
    for source_batch in source_batches:
        sidecar_batch = _next_matching_batch(
            sidecar_batches,
            source_batch.num_rows,
            "sidecar has fewer rows than source",
        )
        ids = source_batch.column(0).to_pylist()
        results = _table_results(pa.Table.from_batches([sidecar_batch]))
        _add_labels(labels, ids, results)
    _ensure_no_extra_batches(sidecar_batches, "sidecar has more rows than source")
    return labels


def _add_labels(
    labels: dict[str, EunisResult],
    ids: list[object],
    results: list[EunisResult],
) -> None:
    for polygon_id, result in zip(ids, results, strict=True):
        if not isinstance(polygon_id, str):
            raise SchemaError("polygon id column contains a non-string value")
        if polygon_id in labels:
            raise ValueError(f"duplicate polygon id {polygon_id!r}")
        labels[polygon_id] = result
