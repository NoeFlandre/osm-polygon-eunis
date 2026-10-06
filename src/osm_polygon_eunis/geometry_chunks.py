# ruff: noqa: D100
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from ._protocols import HubApi, StreamClient
from .eea import EeaGroup
from .options import BatchLimits
from .release_plan import DatasetPlan, Progress

_DEFAULT_BATCH_LIMITS = BatchLimits()


@dataclass(frozen=True, slots=True)
class _GeometryChunk:
    """Pickleable work unit for bounded source micro-batches."""

    groups: tuple[EeaGroup, ...]
    reference_directory: Path
    plans: tuple[DatasetPlan, ...]
    jobs: tuple[tuple[str, str], ...]
    sidecar_root: Path
    source_root: Path
    threshold: int
    limits: BatchLimits
    endpoint: str
    token: str | bool | None
    reference_signature: str


@dataclass(frozen=True, slots=True)
class _GeometryRunOptions:
    api: HubApi
    plans: tuple[DatasetPlan, ...]
    groups: tuple[EeaGroup, ...]
    sidecar_root: Path
    source_root: Path
    workdir: Path
    threshold: int
    checksums: dict[str, str]
    limits: BatchLimits
    progress: Progress | None
    http_client: StreamClient


def _geometry_jobs(plans: tuple[DatasetPlan, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (plan.spec.name, source_path) for plan in plans for source_path in plan.geometry_paths
    )


def _geometry_work_units(
    options: _GeometryRunOptions,
    jobs: tuple[tuple[str, str], ...],
    reference_directory: Path,
    reference_signature: str,
) -> tuple[_GeometryChunk, ...]:
    endpoint = str(getattr(options.api, "endpoint", None) or "https://huggingface.co")
    token = getattr(options.api, "token", None)
    return tuple(
        _GeometryChunk(
            groups=options.groups,
            reference_directory=reference_directory,
            plans=options.plans,
            jobs=chunk,
            sidecar_root=options.sidecar_root,
            source_root=options.source_root,
            threshold=options.threshold,
            limits=options.limits,
            endpoint=endpoint,
            token=token,
            reference_signature=reference_signature,
        )
        for chunk in _geometry_chunks(jobs, options.limits)
    )


def _geometry_chunks(
    jobs: tuple[tuple[str, str], ...],
    limits: BatchLimits = _DEFAULT_BATCH_LIMITS,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not jobs:
        return ()
    worker_count = min(max(limits.workers, 1), len(jobs))
    task_count = min(len(jobs), worker_count * limits.geometry_tasks_per_worker)
    chunk_size = max(1, (len(jobs) + task_count - 1) // task_count)
    return tuple(jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size))


def _geometry_micro_batches(
    jobs: tuple[tuple[str, str], ...],
    limits: BatchLimits = _DEFAULT_BATCH_LIMITS,
) -> Iterator[tuple[tuple[str, str], ...]]:
    for start in range(0, len(jobs), limits.retained_source_shards_per_worker):
        yield jobs[start : start + limits.retained_source_shards_per_worker]
