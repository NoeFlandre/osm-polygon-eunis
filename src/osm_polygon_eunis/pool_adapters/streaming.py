"""Bounded handoff from per-polygon adapters to the shared pool exporter."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path

from osm_polygon_eunis.pool_contract import (
    FileArtifactSink,
    PoolBatch,
    PoolBuildInput,
    PoolStreamSpec,
    StreamingExportReceipt,
    StreamingLimits,
    export_pool_batches,
)

from .common import PoolPolygonInput, PoolPolygonLabelRecords


@dataclass(frozen=True, slots=True)
class AdapterStreamingOptions:
    """Bounded-export settings and downstream rows for a source-specific runner."""

    limits: StreamingLimits = field(default_factory=StreamingLimits)
    staging_directory: Path | None = None
    supplemental_batches: Iterable[PoolBatch] = ()


def export_polygon_attempts(
    metadata_input: PoolBuildInput,
    raw_polygons: Iterable[PoolPolygonInput],
    label_records: Iterable[PoolPolygonLabelRecords],
    sink: FileArtifactSink,
    *,
    options: AdapterStreamingOptions | None = None,
) -> StreamingExportReceipt:
    """Stream two aligned source iterators through the shared pool validator.

    ``metadata_input`` carries only immutable pool metadata and independently
    scanned raw-inventory evidence. Source documents and link rows can arrive as
    bounded supplemental batches, including unresolved links with no polygon
    target. The shared exporter validates the complete cross-batch inventory.
    """
    return export_labeled_polygons(
        metadata_input,
        zip(raw_polygons, label_records, strict=True),
        sink,
        options=options,
    )


def export_labeled_polygons(
    metadata_input: PoolBuildInput,
    labeled_polygons: Iterable[tuple[PoolPolygonInput, PoolPolygonLabelRecords]],
    sink: FileArtifactSink,
    *,
    options: AdapterStreamingOptions | None = None,
) -> StreamingExportReceipt:
    """Stream each raw polygon together with its source-reference result once."""
    _require_metadata_only(metadata_input)
    stream_options = options or AdapterStreamingOptions()
    spec = PoolStreamSpec.from_build_input(metadata_input)
    batches = iter_labeled_polygon_batches(
        labeled_polygons,
        max_batch_rows=stream_options.limits.max_batch_rows,
        supplemental_batches=stream_options.supplemental_batches,
    )
    return export_pool_batches(
        spec,
        batches,
        sink,
        limits=stream_options.limits,
        staging_directory=stream_options.staging_directory,
    )


def iter_polygon_batches(
    raw_polygons: Iterable[PoolPolygonInput],
    label_records: Iterable[PoolPolygonLabelRecords],
    *,
    max_batch_rows: int = StreamingLimits().max_batch_rows,
    supplemental_batches: Iterable[PoolBatch] = (),
) -> Iterator[PoolBatch]:
    """Align two iterators strictly and yield consecutive bounded batches."""
    yield from iter_labeled_polygon_batches(
        zip(raw_polygons, label_records, strict=True),
        max_batch_rows=max_batch_rows,
        supplemental_batches=supplemental_batches,
    )


def iter_labeled_polygon_batches(
    labeled_polygons: Iterable[tuple[PoolPolygonInput, PoolPolygonLabelRecords]],
    *,
    max_batch_rows: int = StreamingLimits().max_batch_rows,
    supplemental_batches: Iterable[PoolBatch] = (),
) -> Iterator[PoolBatch]:
    """Yield bounded batches from one-pass raw polygon and label-result pairs."""
    if max_batch_rows <= 0:
        raise ValueError("max_batch_rows must be positive")
    batch_index = 0
    inventory = []
    polygons = []
    candidates = []
    outcomes = []

    def current_batch(index: int) -> PoolBatch:
        return PoolBatch(
            batch_index=index,
            inventory=tuple(inventory),
            polygons=tuple(polygons),
            label_candidates=tuple(candidates),
            label_outcomes=tuple(outcomes),
        )

    for polygon, record in labeled_polygons:
        _validate_polygon_record(polygon, record)
        contribution = _polygon_row_count(polygon, record, max_batch_rows)
        if inventory and current_batch(batch_index).row_count + contribution > max_batch_rows:
            yield current_batch(batch_index)
            batch_index += 1
            inventory.clear()
            polygons.clear()
            candidates.clear()
            outcomes.clear()
        inventory.append(polygon.inventory)
        if polygon.snapshot is not None:
            polygons.append(polygon.snapshot)
        candidates.extend(record.label_candidates)
        outcomes.extend(record.label_outcomes)
    if inventory:
        yield current_batch(batch_index)
        batch_index += 1
    for supplemental in supplemental_batches:
        yield replace(supplemental, batch_index=batch_index)
        batch_index += 1


def _validate_polygon_record(polygon: PoolPolygonInput, record: PoolPolygonLabelRecords) -> None:
    if polygon.inventory.inventory_row_id != record.inventory_row_id:
        raise ValueError("label decision does not match the next raw polygon key")
    if any(
        outcome.inventory_row_id != record.inventory_row_id for outcome in record.label_outcomes
    ):
        raise ValueError("label outcome references a different raw polygon")


def _polygon_row_count(
    polygon: PoolPolygonInput,
    record: PoolPolygonLabelRecords,
    max_batch_rows: int,
) -> int:
    count = (
        1
        + int(polygon.snapshot is not None)
        + len(record.label_candidates)
        + len(record.label_outcomes)
    )
    if count > max_batch_rows:
        raise ValueError("one polygon's label rows exceed max_batch_rows")
    return count


def _require_metadata_only(metadata_input: PoolBuildInput) -> None:
    row_groups = (
        metadata_input.inventory,
        metadata_input.polygons,
        metadata_input.label_candidates,
        metadata_input.label_outcomes,
        metadata_input.documents,
        metadata_input.text_polygon_links,
        metadata_input.entity_links,
        metadata_input.eligibility,
    )
    if any(row_groups):
        raise ValueError("stream metadata input must not contain materialized pool table rows")
