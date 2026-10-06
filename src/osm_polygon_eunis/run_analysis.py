"""Summarize local geometry worker logs and resume checkpoints."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TypedDict

from .release_plan import DATASET_NAMES


class RunAnalysisError(ValueError):
    """A run-analysis input cannot be read or its options are invalid."""


@dataclass(frozen=True, slots=True)
class _TimingEvent:
    dataset: str
    path: str
    reference_batch: int
    seconds: float


@dataclass(frozen=True, slots=True)
class _RunPlan:
    dataset: str
    reference_batch_ids: frozenset[int]
    checkpoint_signatures: Mapping[str, str]


@dataclass(slots=True)
class _InputCounts:
    malformed_json_lines: int = 0
    malformed_records: int = 0
    repeated_plans: int = 0
    malformed_checkpoints: int = 0
    stale_checkpoints: int = 0


@dataclass(slots=True)
class _LogData:
    events: list[_TimingEvent] = field(default_factory=list)
    plans: dict[str, _RunPlan] = field(default_factory=dict)
    conflicts: set[str] = field(default_factory=set)
    invalid_plans: set[str] = field(default_factory=set)
    declared_datasets: set[str] = field(default_factory=set)


class _ProgressSummary(TypedDict):
    geometry_shards_completed: int  # noqa: V107 - emitted as a JSON result key
    geometry_shards_total: int  # noqa: V107 - emitted as a JSON result key
    reference_batches_completed: int  # noqa: V107 - emitted as a JSON result key
    reference_batches_total: int  # noqa: V107 - emitted as a JSON result key
    reference_batches_remaining: int  # noqa: V107 - emitted as a JSON result key
    percent: float  # noqa: V107 - emitted as a JSON result key


class _SlowShardSummary(TypedDict):
    path: str  # noqa: V107 - emitted as a JSON result key
    event_count: int  # noqa: V107 - emitted as a JSON result key
    worker_seconds: float  # noqa: V107 - emitted as a JSON result key


class _DatasetSummary(TypedDict):
    dataset: str  # noqa: V107 - emitted as a JSON result key
    event_count: int  # noqa: V107 - emitted as a JSON result key
    unique_event_count: int  # noqa: V107 - emitted as a JSON result key
    repeated_event_count: int  # noqa: V107 - emitted as a JSON result key
    worker_seconds: float  # noqa: V107 - emitted as a JSON result key
    throughput_batches_per_worker_hour: float | None  # noqa: V107 - emitted as a JSON result key
    checkpoint_batches_observed: int  # noqa: V107 - emitted as a JSON result key
    progress: _ProgressSummary | None  # noqa: V107 - emitted as a JSON result key
    slowest_shards: list[_SlowShardSummary]  # noqa: V107 - emitted as a JSON result key


class _InputSummary(TypedDict):
    log_files: int  # noqa: V107 - emitted as a JSON result key
    malformed_json_lines: int  # noqa: V107 - emitted as a JSON result key
    malformed_records: int  # noqa: V107 - emitted as a JSON result key
    repeated_plans: int  # noqa: V107 - emitted as a JSON result key
    malformed_checkpoints: int  # noqa: V107 - emitted as a JSON result key
    stale_checkpoints: int  # noqa: V107 - emitted as a JSON result key
    missing_metadata_datasets: list[str]  # noqa: V107 - emitted as a JSON result key


class _RunSummary(TypedDict):
    datasets: list[_DatasetSummary]  # noqa: V107 - emitted as a JSON result key
    inputs: _InputSummary  # noqa: V107 - emitted as a JSON result key


def summarize_run(
    log_paths: Sequence[Path],
    sidecar_root: Path,
    *,
    slowest_limit: int = 5,
) -> _RunSummary:
    """Summarize timing records and matching ``.done`` files without network calls."""

    _validate_options(log_paths, slowest_limit)
    counts = _InputCounts()
    log_data = _read_logs(log_paths, counts)
    missing_metadata = _missing_metadata_datasets(log_data)
    datasets = _dataset_summaries(log_data, sidecar_root, slowest_limit, counts)
    return {
        "datasets": datasets,
        "inputs": {
            "log_files": len(log_paths),
            "malformed_json_lines": counts.malformed_json_lines,
            "malformed_records": counts.malformed_records,
            "repeated_plans": counts.repeated_plans,
            "malformed_checkpoints": counts.malformed_checkpoints,
            "stale_checkpoints": counts.stale_checkpoints,
            "missing_metadata_datasets": missing_metadata,
        },
    }


def _validate_options(log_paths: Sequence[Path], slowest_limit: int) -> None:
    if not log_paths:
        raise RunAnalysisError("at least one --log path is required")
    if slowest_limit <= 0:
        raise RunAnalysisError("--slowest must be a positive integer")


def _missing_metadata_datasets(data: _LogData) -> list[str]:
    dataset_names = (
        set(data.plans) | data.declared_datasets | {event.dataset for event in data.events}
    )
    return sorted(dataset for dataset in dataset_names if _plan_metadata_missing(data, dataset))


def _plan_metadata_missing(data: _LogData, dataset: str) -> bool:
    return dataset not in data.plans or dataset in data.conflicts or dataset in data.invalid_plans


def _dataset_summaries(
    data: _LogData,
    sidecar_root: Path,
    slowest_limit: int,
    counts: _InputCounts,
) -> list[_DatasetSummary]:
    events_by_dataset: dict[str, list[_TimingEvent]] = defaultdict(list)
    for event in data.events:
        events_by_dataset[event.dataset].append(event)
    dataset_names = sorted(set(events_by_dataset) | set(data.plans) | data.declared_datasets)
    return [
        _dataset_summary(
            dataset,
            events_by_dataset[dataset],
            _dataset_plan(data, dataset),
            sidecar_root,
            slowest_limit,
            counts,
        )
        for dataset in dataset_names
    ]


def _dataset_plan(data: _LogData, dataset: str) -> _RunPlan | None:
    return (
        None
        if dataset in data.conflicts or dataset in data.invalid_plans
        else data.plans.get(dataset)
    )


def _read_logs(log_paths: Sequence[Path], counts: _InputCounts) -> _LogData:
    data = _LogData()

    for log_path in log_paths:
        try:
            lines = log_path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError) as error:
            raise RunAnalysisError(f"cannot read log {log_path}: {error}") from error
        for line in lines:
            record = _json_record(line, counts)
            if record is not None:
                _collect_record(record, data, counts)
    return data


def _collect_record(record: Mapping[str, object], data: _LogData, counts: _InputCounts) -> None:
    event_type = record.get("event")
    if event_type == "geometry_batch_done":
        event = _timing_event(record)
        if event is None:
            counts.malformed_records += 1
        else:
            data.events.append(event)
    elif event_type == "geometry_run_plan":
        _collect_plan(record, data, counts)


def _collect_plan(record: Mapping[str, object], data: _LogData, counts: _InputCounts) -> None:
    plan = _run_plan(record)
    if plan is None:
        _record_missing_plan(record, data, counts)
        return
    data.declared_datasets.add(plan.dataset)
    previous = data.plans.get(plan.dataset)
    if previous is None:
        data.plans[plan.dataset] = plan
    elif previous == plan:
        counts.repeated_plans += 1
    else:
        _record_conflicting_plan(plan, data, counts)


def _record_missing_plan(
    record: Mapping[str, object], data: _LogData, counts: _InputCounts
) -> None:
    counts.malformed_records += 1
    dataset = record.get("dataset")
    if isinstance(dataset, str) and dataset in DATASET_NAMES:
        data.declared_datasets.add(dataset)
        data.invalid_plans.add(dataset)


def _record_conflicting_plan(plan: _RunPlan, data: _LogData, counts: _InputCounts) -> None:
    data.conflicts.add(plan.dataset)
    counts.malformed_records += 1


def _json_record(line: str, counts: _InputCounts) -> Mapping[str, object] | None:
    if not line.strip():
        return None
    try:
        payload = json.loads(line)
    except ValueError:
        counts.malformed_json_lines += 1
        return None
    return payload if isinstance(payload, Mapping) else None


def _timing_event(record: Mapping[str, object]) -> _TimingEvent | None:
    dataset, path, batch = _timing_identity(record)
    if dataset is None or path is None or batch is None:
        return None
    seconds = record.get("seconds")
    elapsed = _event_seconds(seconds)
    if elapsed is None:
        return None
    return _TimingEvent(dataset, path, batch, elapsed)


def _timing_identity(
    record: Mapping[str, object],
) -> tuple[str | None, str | None, int | None]:
    dataset = _dataset_name(record.get("dataset"))
    path = _source_path(record.get("path"))
    batch = _batch_number(record.get("reference_batch"))
    if dataset is None or path is None or batch is None:
        return None, None, None
    return dataset, path, batch


def _dataset_name(value: object) -> str | None:
    return value if isinstance(value, str) and value in DATASET_NAMES else None


def _source_path(value: object) -> str | None:
    return value if isinstance(value, str) and value else None


def _batch_number(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _event_seconds(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return _integer_seconds(value)
    if isinstance(value, float):
        return _valid_seconds(value)
    return None


def _integer_seconds(value: int) -> float | None:
    try:
        elapsed = float(value)
    except OverflowError:
        return None
    return _valid_seconds(elapsed)


def _valid_seconds(value: float) -> float | None:
    return value if math.isfinite(value) and value >= 0 else None


def _run_plan(record: Mapping[str, object]) -> _RunPlan | None:
    dataset = record.get("dataset")
    raw_batch_ids = record.get("reference_batch_ids")
    raw_signatures = record.get("checkpoint_signatures")
    if not isinstance(dataset, str) or dataset not in DATASET_NAMES:
        return None
    batch_ids = _reference_batch_ids(raw_batch_ids)
    signatures = _checkpoint_signatures(raw_signatures)
    if batch_ids is None or signatures is None:
        return None
    return _RunPlan(dataset, batch_ids, signatures)


def _reference_batch_ids(value: object) -> frozenset[int] | None:
    if not isinstance(value, list):
        return None
    return _unique_batch_ids(value)


def _unique_batch_ids(values: list[object]) -> frozenset[int] | None:
    batch_ids: set[int] = set()
    for value in values:
        batch = _batch_number(value)
        if batch is None or batch in batch_ids:
            return None
        batch_ids.add(batch)
    return frozenset(batch_ids)


def _checkpoint_signatures(value: object) -> Mapping[str, str] | None:
    if not isinstance(value, Mapping):
        return None
    signatures: dict[str, str] = {}
    for path, signature in value.items():
        entry = _checkpoint_signature_entry(path, signature)
        if entry is None:
            return None
        signatures[entry[0]] = entry[1]
    return signatures


def _checkpoint_signature_entry(path: object, signature: object) -> tuple[str, str] | None:
    if not isinstance(path, str) or not path:
        return None
    if not isinstance(signature, str) or not signature:
        return None
    return path, signature


def _dataset_summary(
    dataset: str,
    events: list[_TimingEvent],
    plan: _RunPlan | None,
    sidecar_root: Path,
    slowest_limit: int,
    counts: _InputCounts,
) -> _DatasetSummary:
    events = _events_in_plan(events, plan, counts)
    if plan is None:
        progress = None
        checkpoint_batches = _observed_checkpoint_batches(dataset, sidecar_root, counts)
    else:
        progress, checkpoint_batches = _checkpoint_progress(dataset, plan, sidecar_root, counts)
    events, seconds = _finite_event_total(events, counts)
    unique_events = {(event.path, event.reference_batch) for event in events}
    return {
        "dataset": dataset,
        "event_count": len(events),
        "unique_event_count": len(unique_events),
        "repeated_event_count": len(events) - len(unique_events),
        "worker_seconds": round(seconds, 3),
        "throughput_batches_per_worker_hour": _throughput_rate(len(events), seconds),
        "checkpoint_batches_observed": checkpoint_batches,
        "progress": progress,
        "slowest_shards": _slowest_shards(events, slowest_limit),
    }


def _throughput_rate(batch_count: int, seconds: float) -> float | None:
    if seconds <= 0:
        return None
    rate = batch_count * 3600 / seconds
    return round(rate, 3) if math.isfinite(rate) else None


def _finite_event_total(
    events: list[_TimingEvent], counts: _InputCounts
) -> tuple[list[_TimingEvent], float]:
    try:
        seconds = math.fsum(event.seconds for event in events)
    except OverflowError:
        counts.malformed_records += len(events)
        return [], 0.0
    return events, seconds


def _events_in_plan(
    events: list[_TimingEvent], plan: _RunPlan | None, counts: _InputCounts
) -> list[_TimingEvent]:
    if plan is None:
        return events
    declared_events = []
    for event in events:
        is_undeclared = (
            event.path not in plan.checkpoint_signatures
            or event.reference_batch not in plan.reference_batch_ids
        )
        if is_undeclared:
            counts.malformed_records += 1
        else:
            declared_events.append(event)
    return declared_events


def _checkpoint_progress(
    dataset: str,
    plan: _RunPlan,
    sidecar_root: Path,
    counts: _InputCounts,
) -> tuple[_ProgressSummary, int]:
    expected_batches = plan.reference_batch_ids
    completed_batches = 0
    completed_shards = 0
    for shard_path, signature in plan.checkpoint_signatures.items():
        marker = _checkpoint_marker(sidecar_root, dataset, shard_path)
        batches = _read_checkpoint_batches(
            marker,
            counts,
            expected_signature=signature,
            expected_batches=expected_batches,
        )
        if batches is None:
            continue
        declared_batches = batches & expected_batches
        completed_batches += len(declared_batches)
        if expected_batches <= declared_batches:
            completed_shards += 1

    expected_total = len(plan.checkpoint_signatures) * len(expected_batches)
    remaining = expected_total - completed_batches
    percent = 100.0 if expected_total == 0 else round(completed_batches * 100 / expected_total, 2)
    return (
        {
            "geometry_shards_completed": completed_shards,
            "geometry_shards_total": len(plan.checkpoint_signatures),
            "reference_batches_completed": completed_batches,
            "reference_batches_total": expected_total,
            "reference_batches_remaining": remaining,
            "percent": percent,
        },
        completed_batches,
    )


def _observed_checkpoint_batches(
    dataset: str,
    sidecar_root: Path,
    counts: _InputCounts,
) -> int:
    directory = sidecar_root / dataset
    if not directory.is_dir():
        return 0
    total = 0
    for marker in sorted(directory.glob("*.labels.parquet.done")):
        batches = _read_checkpoint_batches(marker, counts)
        if batches is not None:
            total += len(batches)
    return total


def _read_checkpoint_batches(
    marker: Path,
    counts: _InputCounts,
    *,
    expected_signature: str | None = None,
    expected_batches: frozenset[int] | None = None,
) -> set[int] | None:
    payload = _checkpoint_payload(marker, counts)
    if payload is None:
        return None
    return _checkpoint_ids(payload, counts, expected_signature, expected_batches)


def _checkpoint_payload(marker: Path, counts: _InputCounts) -> Mapping[str, object] | None:
    if not marker.is_file():
        return None
    try:
        payload = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        _mark_malformed_checkpoint(counts)
        return None
    if not isinstance(payload, Mapping):
        _mark_malformed_checkpoint(counts)
        return None
    return payload


def _checkpoint_ids(
    payload: Mapping[str, object],
    counts: _InputCounts,
    expected_signature: str | None,
    expected_batches: frozenset[int] | None,
) -> set[int] | None:
    if not _checkpoint_signature_matches(payload, expected_signature, counts):
        return None
    return _decode_checkpoint_batch_ids(payload.get("batches"), counts, expected_batches)


def _checkpoint_signature_matches(
    payload: Mapping[str, object],
    expected_signature: str | None,
    counts: _InputCounts,
) -> bool:
    signature = payload.get("signature")
    if not isinstance(signature, str) or not signature:
        _mark_malformed_checkpoint(counts)
        return False
    if expected_signature is not None and signature != expected_signature:
        counts.stale_checkpoints += 1
        return False
    return True


def _decode_checkpoint_batch_ids(
    value: object,
    counts: _InputCounts,
    expected_batches: frozenset[int] | None,
) -> set[int] | None:
    if not isinstance(value, list):
        _mark_malformed_checkpoint(counts)
        return None
    batches, is_malformed = _checkpoint_batch_set(value, expected_batches)
    if is_malformed:
        counts.malformed_checkpoints += 1
    return batches


def _checkpoint_batch_set(
    values: list[object], expected_batches: frozenset[int] | None
) -> tuple[set[int], bool]:
    batch_ids: set[int] = set()
    is_malformed = False
    for value in values:
        batch = _batch_number(value)
        if (
            batch is not None
            and batch not in batch_ids
            and _is_expected_batch(batch, expected_batches)
        ):
            batch_ids.add(batch)
        else:
            is_malformed = True
    return batch_ids, is_malformed


def _is_expected_batch(batch: int, expected_batches: frozenset[int] | None) -> bool:
    return expected_batches is None or batch in expected_batches


def _mark_malformed_checkpoint(counts: _InputCounts) -> None:
    counts.malformed_checkpoints += 1


def _checkpoint_marker(sidecar_root: Path, dataset: str, shard_path: str) -> Path:
    filename = f"{shard_path.replace('/', '__')}.labels.parquet.done"
    return sidecar_root / dataset / filename


def _slowest_shards(events: list[_TimingEvent], limit: int) -> list[_SlowShardSummary]:
    aggregates: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for event in events:
        aggregates[event.path][0] += 1
        aggregates[event.path][1] += event.seconds
    slowest = sorted(aggregates.items(), key=lambda item: (-item[1][1], item[0]))[:limit]
    return [
        {
            "path": path,
            "event_count": int(values[0]),
            "worker_seconds": round(values[1], 3),
        }
        for path, values in slowest
    ]
