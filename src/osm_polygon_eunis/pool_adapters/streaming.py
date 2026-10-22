"""Bounded handoff from per-polygon adapters to the shared pool exporter."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
from urllib.parse import urlparse

from osm_polygon_eunis.pool_contract import (
    AreaMeasureSpec,
    AttemptStatus,
    FileArtifactSink,
    LabelSupportEvidence,
    PolygonAttemptReceipt,
    PoolBatch,
    PoolBuildInput,
    PoolStreamSpec,
    ReferenceSystem,
    StreamingExportReceipt,
    StreamingLimits,
    export_pool_batches,
)

from .common import (
    MeasurementSidecar,
    PoolPolygonInput,
    PoolPolygonLabelRecords,
)
from .lucas import LUCAS_AREA_MEASUREMENT_ID
from .worldcover import WORLDCOVER_AREA_MEASUREMENT

_AREA_METHOD_BY_REFERENCE = {
    ReferenceSystem.WORLDCOVER: WORLDCOVER_AREA_MEASUREMENT,
    ReferenceSystem.LUCAS: LUCAS_AREA_MEASUREMENT_ID,
}


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


def persist_measurement_sidecars(
    labeled_polygons: Iterable[tuple[PoolPolygonInput, PoolPolygonLabelRecords]],
    sink: FileArtifactSink,
    pool_name: str,
) -> Iterator[tuple[PoolPolygonInput, PoolPolygonLabelRecords]]:
    """Persist and pin adapter evidence files before a checkpoint advances."""

    yield from _persist_measurement_sidecars(labeled_polygons, sink, pool_name)


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
    _require_contract_v12_provenance(spec)
    batches = iter_labeled_polygon_batches(
        _persist_measurement_sidecars(labeled_polygons, sink, metadata_input.pool_key.name),
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
    attempt_receipts = []
    support_evidence = []

    def current_batch(index: int) -> PoolBatch:
        return PoolBatch(
            batch_index=index,
            inventory=tuple(inventory),
            polygons=tuple(polygons),
            label_candidates=tuple(candidates),
            label_outcomes=tuple(outcomes),
            attempt_receipts=tuple(attempt_receipts),
            support_evidence=tuple(support_evidence),
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
            attempt_receipts.clear()
            support_evidence.clear()
        inventory.append(polygon.inventory)
        if polygon.snapshot is not None:
            polygons.append(polygon.snapshot)
        candidates.extend(record.label_candidates)
        outcomes.extend(record.label_outcomes)
        attempt_receipts.append(_attempt_receipt(record))
        support_evidence.extend(_candidate_support_evidence(record))
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
    if record.attempt_evidence.inventory_row_id != record.inventory_row_id:
        raise ValueError("reference attempt receipt does not match the raw polygon")


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
        + 1
        + len(record.label_candidates)
    )
    if count > max_batch_rows:
        raise ValueError("one polygon's label rows exceed max_batch_rows")
    return count


def _attempt_receipt(record: PoolPolygonLabelRecords) -> PolygonAttemptReceipt:
    evidence = record.attempt_evidence
    attempted_reference_ids = evidence.attempted_reference_ids or evidence.attempted_tile_ids
    missing_reference_ids = evidence.missing_tile_ids
    reason = evidence.reason
    if evidence.status in {"completed", "partial"}:
        if missing_reference_ids:
            status = AttemptStatus.PARTIAL
            reason = reason or "one_or_more_reference_tiles_unavailable"
        elif evidence.status == "partial":
            status = AttemptStatus.PARTIAL
            reason = reason or "reference_measurement_incomplete"
        else:
            status = AttemptStatus.COMPLETED
    elif evidence.status == "excluded":
        status = (
            AttemptStatus.FAILED
            if reason is not None
            and any(
                marker in reason
                for marker in (
                    "reference_query_failed",
                    "extraction_failed",
                    "read_failed",
                    "measurement_failed",
                )
            )
            else AttemptStatus.EXCLUDED
        )
    else:
        raise ValueError(f"unknown adapter attempt status {evidence.status!r}")
    sidecar = evidence.attempted_reference_sidecar
    if sidecar is not None and sidecar.uri is None:
        raise ValueError("attempt measurement sidecar was not persisted before export")
    return PolygonAttemptReceipt(
        inventory_row_id=evidence.inventory_row_id,
        status=status,
        reference_id_namespace=evidence.reference_id_namespace,
        attempted_reference_ids=attempted_reference_ids,
        missing_reference_ids=missing_reference_ids,
        measurement_sidecar_uri=sidecar.uri if sidecar else None,
        measurement_sidecar_sha256=sidecar.sha256 if sidecar else None,
        measurement_record_count=sidecar.record_count if sidecar else None,
        reason=reason,
    )


def _candidate_support_evidence(
    record: PoolPolygonLabelRecords,
) -> tuple[LabelSupportEvidence, ...]:
    evidence = record.attempt_evidence
    ids_by_candidate = {
        (level, code): reference_ids for level, code, reference_ids in evidence.support_evidence
    }
    if len(ids_by_candidate) != len(evidence.support_evidence):
        raise ValueError("adapter emitted duplicate candidate support evidence")
    output = []
    sidecars_by_candidate = {
        (level, code): sidecar for level, code, sidecar in evidence.support_evidence_sidecars
    }
    if len(sidecars_by_candidate) != len(evidence.support_evidence_sidecars):
        raise ValueError("adapter emitted duplicate candidate support sidecars")
    for candidate in record.label_candidates:
        key = candidate.level, candidate.code
        reference_ids = ids_by_candidate.get(key, ())
        sidecar = sidecars_by_candidate.get(key)
        if bool(reference_ids) == (sidecar is not None):
            raise ValueError(f"candidate {key} is missing its exact reference evidence IDs")
        if sidecar is not None and sidecar.uri is None:
            raise ValueError(f"candidate {key} support sidecar was not persisted before export")
        output.append(
            LabelSupportEvidence(
                inventory_row_id=record.inventory_row_id,
                level=candidate.level,
                code=candidate.code,
                reference_id_namespace=evidence.reference_id_namespace,
                reference_ids=reference_ids,
                measurement_sidecar_uri=sidecar.uri if sidecar else None,
                measurement_sidecar_sha256=sidecar.sha256 if sidecar else None,
                measurement_record_count=sidecar.record_count if sidecar else None,
            )
        )
    candidate_keys = {(row.level, row.code) for row in record.label_candidates}
    extra = (set(ids_by_candidate) | set(sidecars_by_candidate)) - candidate_keys
    if extra:
        raise ValueError(f"adapter emitted support evidence without a candidate: {sorted(extra)}")
    return tuple(output)


def _persist_measurement_sidecars(
    labeled_polygons: Iterable[tuple[PoolPolygonInput, PoolPolygonLabelRecords]],
    sink: FileArtifactSink,
    pool_name: str,
) -> Iterator[tuple[PoolPolygonInput, PoolPolygonLabelRecords]]:
    """Persist one polygon's streamed ID files before advancing the source scan."""
    for polygon, record in labeled_polygons:
        evidence = record.attempt_evidence
        row_key = hashlib.sha256(evidence.inventory_row_id.encode("utf-8")).hexdigest()
        attempted = _persist_sidecar(
            evidence.attempted_reference_sidecar,
            sink,
            f"measurement-sidecars/{pool_name}/{row_key}-attempted",
        )
        support_sidecars = tuple(
            (
                level,
                code,
                _persist_sidecar(
                    sidecar,
                    sink,
                    f"measurement-sidecars/{pool_name}/{row_key}-support-"
                    f"{hashlib.sha256(f'{level}:{code}'.encode()).hexdigest()}",
                ),
            )
            for level, code, sidecar in evidence.support_evidence_sidecars
        )
        persisted = replace(
            record,
            attempt_evidence=replace(
                evidence,
                attempted_reference_sidecar=attempted,
                support_evidence_sidecars=support_sidecars,
            ),
        )
        yield polygon, persisted


def _persist_sidecar(
    sidecar: MeasurementSidecar | None,
    sink: FileArtifactSink,
    artifact_stem: str,
) -> MeasurementSidecar | None:
    if sidecar is None or sidecar.uri is not None:
        return sidecar
    if sidecar.path is None:
        raise ValueError("measurement sidecar has neither a local file nor a persisted URI")
    uri = sink.put_file(f"{artifact_stem}-{sidecar.sha256}.jsonl", sidecar.path)
    sidecar.path.unlink(missing_ok=True)
    return MeasurementSidecar(
        sha256=sidecar.sha256,
        record_count=sidecar.record_count,
        uri=uri,
    )


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
        metadata_input.attempt_receipts,
        metadata_input.support_evidence,
        metadata_input.relevance_assessments,
    )
    if any(row_groups):
        raise ValueError("stream metadata input must not contain materialized pool table rows")


def _require_contract_v12_provenance(spec: PoolStreamSpec) -> None:
    """Refuse adapter exports that cannot pass the v1.2 readiness gate."""
    if spec.contract_version != "1.2.0":
        raise ValueError("pool adapters require contract version 1.2.0")
    _require_raw_inventory_evidence(spec)
    _require_artifact_manifest_pins(spec)
    _require_area_measurement(spec)


def _require_raw_inventory_evidence(spec: PoolStreamSpec) -> None:
    evidence = spec.raw_inventory_evidence
    if evidence.expected_key_count <= 0:
        raise ValueError("contract 1.2 adapter export needs a verified non-empty raw inventory")
    if not _valid_sha256(evidence.expected_keyset_sha256):
        raise ValueError("contract 1.2 adapter export needs the raw inventory key-set SHA-256")
    if not _valid_uri(evidence.scan_report_uri) or not _valid_sha256(evidence.scan_report_sha256):
        raise ValueError("contract 1.2 adapter export needs a pinned raw inventory scan report")


def _require_artifact_manifest_pins(spec: PoolStreamSpec) -> None:
    artifact_pins = (
        (
            "source_snapshot.source_artifact_manifest",
            spec.source_snapshot.source_artifact_manifest,
        ),
        (
            "reference_pin.reference_artifact_manifest",
            spec.reference_pin.reference_artifact_manifest,
        ),
    )
    missing = [
        name
        for name, pin in (
            *artifact_pins,
            ("reference_pin.area_measurement", spec.reference_pin.area_measurement),
        )
        if pin is None
    ]
    if missing:
        fields = ", ".join(missing)
        raise ValueError(
            "contract 1.2 adapter export requires declared provenance pins before scanning: "
            f"{fields}"
        )
    for label, pin in artifact_pins:
        if pin is None:
            continue
        if (
            pin.format_version != "artifact-list-jsonl-v1"
            or not _valid_uri(pin.manifest_uri)
            or not _valid_sha256(pin.manifest_sha256)
            or pin.expected_artifact_count <= 0
        ):
            raise ValueError(f"contract 1.2 {label} pin is incomplete")


def _require_area_measurement(spec: PoolStreamSpec) -> None:
    area_pin = spec.reference_pin.area_measurement
    if area_pin is None:
        raise ValueError("contract 1.2 adapter export requires reference_pin.area_measurement")
    expected_method = _AREA_METHOD_BY_REFERENCE.get(spec.pool_key.reference_system)
    if expected_method is None:
        raise ValueError(
            "WorldCover/LUCAS adapter export cannot declare area provenance for "
            f"{spec.pool_key.reference_system.value}"
        )
    if any(
        not _area_spec_matches(measure, expected_method, spec.reference_pin.support_unit)
        for measure in (area_pin.numerator, area_pin.denominator)
    ):
        raise ValueError(
            "contract 1.2 area method pins do not match the adapter's exact "
            f"{spec.pool_key.reference_system.value} method"
        )
    if (
        spec.pool_key.reference_system is ReferenceSystem.WORLDCOVER
        and area_pin.implementation_version != spec.reference_pin.source_revision
    ):
        raise ValueError(
            "WorldCover area method implementation version must match the pinned producer revision"
        )


def _area_spec_matches(
    measure: AreaMeasureSpec,
    expected_method: str,
    expected_unit: str,
) -> bool:
    return (
        measure.method_id == expected_method
        and measure.output_unit == expected_unit
        and bool(measure.boundary_convention.strip())
        and bool(measure.densification_method.strip())
        and _has_canonical_json_object(measure.densification_parameters_json)
        and _has_canonical_json_object(measure.error_settings_json)
        and bool(measure.coordinate_system or measure.ellipsoid)
    )


def _valid_uri(value: str) -> bool:
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.path))


def _valid_sha256(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None


def _has_canonical_json_object(value: str) -> bool:
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and bool(parsed) and canonical == value
