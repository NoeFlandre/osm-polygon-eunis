"""Bounded geometry-row decoding for static map aggregation."""

from __future__ import annotations

from typing import Any

from .cards import DatasetCardAccumulator

GEOMETRY_MAP_BATCH_ROWS = 8


def _observe_batch(
    batch: Any,
    accumulator: DatasetCardAccumulator,
) -> int:
    batch_rows = int(batch.num_rows)
    if batch_rows < 0 or batch_rows > GEOMETRY_MAP_BATCH_ROWS:
        raise ValueError("Parquet geometry batch exceeded the configured row bound")
    geometries = batch.column(0).to_pylist()
    if len(geometries) != batch_rows:
        raise ValueError("Parquet geometry batch row count changed during decoding")
    for geometry in geometries:
        accumulator.observe_map_row(geometry)
    return batch_rows


def accumulate_geometry_row_group(
    parquet_file: Any,
    row_group_index: int,
) -> tuple[int, dict[str, Any]]:
    """Count map cells from one row group without materializing the full group."""

    accumulator = DatasetCardAccumulator()
    row_count = 0
    batches = parquet_file.iter_batches(
        batch_size=GEOMETRY_MAP_BATCH_ROWS,
        row_groups=[row_group_index],
        columns=["geometry"],
        use_threads=False,
    )
    for batch in batches:
        row_count += _observe_batch(batch, accumulator)
        del batch
    return row_count, accumulator.snapshot()
