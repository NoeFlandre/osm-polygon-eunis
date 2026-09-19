"""Bounded Arrow/Parquet transformations at the pipeline's I/O boundary."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq
from shapely.geometry.base import BaseGeometry

from .domain import EUNIS_FIELDS, EunisResult
from .geometry import parse_geometry, to_equal_area


class OverlapReference(Protocol):
    """Minimal reference interface needed by the batch transformer."""

    def overlap(self, polygon: BaseGeometry | None) -> EunisResult: ...


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
    columns = (
        ("eunis_code", pa.string(), [result.code for result in results]),
        ("eunis_name", pa.string(), [result.name for result in results]),
        (
            "eunis_overlap_percentage",
            pa.float64(),
            [result.overlap_percentage for result in results],
        ),
        ("eunis_source_version", pa.string(), [result.source_version for result in results]),
    )
    for name, data_type, values in columns:
        table = table.append_column(name, pa.array(values, type=data_type))
    return table


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
