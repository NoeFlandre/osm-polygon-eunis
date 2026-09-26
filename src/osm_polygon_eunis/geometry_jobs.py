"""Serial and process-pool geometry work over bounded source micro-batches."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import httpx
from huggingface_hub import HfApi

from ._protocols import HubApi, StreamClient
from .eea import EeaGroup
from .fileio import DOWNLOAD_TIMEOUT
from .references import (
    _http_client,
    _indexed_reference_group_batches,
    _open_reference_batch,
    _reference_group_batches,
    _stage_reference_batch,
    _stage_reference_groups,
    _worker_reference_batch,
)
from .release_plan import (
    DatasetPlan,
    Progress,
    _cached_geometry_path,
    _sidecar_path,
)
from .sources import (
    download_to_temp,
)
from .transform import (
    OverlapReference,
    update_label_sidecar,
)

_SOURCE_MICRO_BATCH_SIZE = 128


_GEOMETRY_TASKS_PER_WORKER = 4


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
    batch_size: int
    endpoint: str
    token: str | bool | None


def process_geometry_paths(
    api: HubApi,
    plan: DatasetPlan,
    *,
    reference: OverlapReference,
    sidecar_root: Path,
    source_root: Path,
    batch_size: int,
    progress: Progress | None = None,
    http_client: StreamClient | None = None,
    retain_source: bool = False,
) -> None:
    """Merge one reference group into every geometry shard's compact sidecar."""

    with _http_client(http_client) as reusable_client:
        source_root.mkdir(parents=True, exist_ok=True)
        for source_path in plan.geometry_paths:
            _process_geometry_path(
                api,
                plan,
                source_path=source_path,
                references=(reference,),
                sidecar_root=sidecar_root,
                source_root=source_root,
                batch_size=batch_size,
                progress=progress,
                http_client=reusable_client,
                retain_source=retain_source,
            )


def _process_geometry_path(
    api: HubApi,
    plan: DatasetPlan,
    *,
    source_path: str,
    references: tuple[OverlapReference, ...],
    sidecar_root: Path,
    source_root: Path,
    batch_size: int,
    progress: Progress | None,
    http_client: StreamClient,
    retain_source: bool,
) -> None:
    local_source = _download_geometry_source(
        api,
        plan,
        source_path,
        source_root,
        retain_source=retain_source,
        client=http_client,
    )
    sidecar = _sidecar_path(sidecar_root, plan.spec, source_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    next_sidecar = sidecar.with_name(f"{sidecar.name}.next")
    update_label_sidecar(
        local_source,
        next_sidecar,
        references=references,
        current=sidecar if sidecar.is_file() else None,
        batch_size=batch_size,
    )
    next_sidecar.replace(sidecar)
    if not retain_source:
        local_source.unlink(missing_ok=True)
    if progress is not None:
        progress({"event": "sidecar_updated", "dataset": plan.spec.name, "path": source_path})


def _download_geometry_source(
    api: HubApi,
    plan: DatasetPlan,
    source_path: str,
    source_root: Path,
    *,
    retain_source: bool,
    client: StreamClient,
) -> Path:
    directory = source_root / plan.spec.name if retain_source else source_root
    cached = _cached_geometry_path(source_root, plan, source_path)
    if retain_source and cached.is_file():
        return cached
    return download_to_temp(
        api,
        plan.spec.source_repo,
        source_path,
        plan.source_revision,
        directory,
        client=client,
    )


def _process_reference_groups(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    groups: tuple[EeaGroup, ...],
    *,
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: StreamClient,
    parallelism: int = 1,
) -> None:
    """Process bounded reference batches with resumable compact sidecars."""

    if parallelism > 1:
        _process_reference_groups_parallel(
            api,
            plans,
            groups=groups,
            sidecar_root=sidecar_root,
            source_root=source_root,
            workdir=workdir,
            threshold=threshold,
            checksums=checksums,
            batch_size=batch_size,
            progress=progress,
            http_client=http_client,
            parallelism=parallelism,
        )
        return
    for batch in _reference_group_batches(groups):
        _process_reference_batch_serial(
            api,
            plans,
            groups=batch,
            sidecar_root=sidecar_root,
            source_root=source_root,
            workdir=workdir,
            threshold=threshold,
            checksums=checksums,
            batch_size=batch_size,
            progress=progress,
            http_client=http_client,
        )


def _process_reference_groups_parallel(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    *,
    groups: tuple[EeaGroup, ...],
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: StreamClient,
    parallelism: int,
) -> None:
    """Stream bounded source micro-batches through every reference batch."""

    jobs = _geometry_jobs(plans)
    if not jobs:
        return
    with _stage_reference_groups(
        groups,
        workdir=workdir,
        checksums=checksums,
        client=http_client,
    ) as reference_directory:
        work = _geometry_work_units(
            api,
            plans,
            groups,
            jobs,
            reference_directory=reference_directory,
            sidecar_root=sidecar_root,
            source_root=source_root,
            threshold=threshold,
            batch_size=batch_size,
            parallelism=parallelism,
        )
        _run_geometry_workers(work, progress, max_workers=parallelism)


def _process_reference_batch_serial(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    *,
    groups: tuple[EeaGroup, ...],
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: StreamClient,
) -> None:
    with _open_reference_batch(
        groups,
        workdir=workdir,
        threshold=threshold,
        checksums=checksums,
        client=http_client,
    ) as references:
        for plan in plans:
            for source_path in plan.geometry_paths:
                _process_geometry_path(
                    api,
                    plan,
                    source_path=source_path,
                    references=references,
                    sidecar_root=sidecar_root,
                    source_root=source_root,
                    batch_size=batch_size,
                    progress=progress,
                    http_client=http_client,
                    retain_source=True,
                )


def _process_reference_batch_parallel(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    *,
    groups: tuple[EeaGroup, ...],
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: StreamClient,
    parallelism: int,
) -> None:
    jobs = _geometry_jobs(plans)
    if not jobs:
        return
    with _stage_reference_batch(
        groups,
        workdir=workdir,
        checksums=checksums,
        client=http_client,
    ) as reference_directory:
        work = _geometry_work_units(
            api,
            plans,
            groups,
            jobs,
            reference_directory=reference_directory,
            sidecar_root=sidecar_root,
            source_root=source_root,
            threshold=threshold,
            batch_size=batch_size,
            parallelism=parallelism,
        )
        _run_geometry_workers(work, progress, max_workers=parallelism)


def _geometry_jobs(plans: tuple[DatasetPlan, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (plan.spec.name, source_path) for plan in plans for source_path in plan.geometry_paths
    )


def _geometry_work_units(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    groups: tuple[EeaGroup, ...],
    jobs: tuple[tuple[str, str], ...],
    *,
    reference_directory: Path,
    sidecar_root: Path,
    source_root: Path,
    threshold: int,
    batch_size: int,
    parallelism: int,
) -> tuple[_GeometryChunk, ...]:
    endpoint = str(getattr(api, "endpoint", None) or "https://huggingface.co")
    token = getattr(api, "token", None)
    return tuple(
        _GeometryChunk(
            groups=groups,
            reference_directory=reference_directory,
            plans=plans,
            jobs=chunk,
            sidecar_root=sidecar_root,
            source_root=source_root,
            threshold=threshold,
            batch_size=batch_size,
            endpoint=endpoint,
            token=token,
        )
        for chunk in _geometry_chunks(jobs, parallelism)
    )


def _run_geometry_workers(
    work: tuple[_GeometryChunk, ...],
    progress: Progress | None,
    *,
    max_workers: int,
) -> None:
    worker_count = min(max(max_workers, 1), len(work))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        for completed in executor.map(_process_geometry_chunk, work):
            _report_completed_geometry(completed, progress)


def _report_completed_geometry(
    completed: tuple[tuple[str, str], ...],
    progress: Progress | None,
) -> None:
    if progress is None:
        return
    for dataset, source_path in completed:
        progress(
            {
                "event": "sidecar_updated",
                "dataset": dataset,
                "path": source_path,
            }
        )


def _geometry_chunks(
    jobs: tuple[tuple[str, str], ...],
    parallelism: int,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not jobs:
        return ()
    worker_count = min(max(parallelism, 1), len(jobs))
    task_count = min(len(jobs), worker_count * _GEOMETRY_TASKS_PER_WORKER)
    chunk_size = max(1, (len(jobs) + task_count - 1) // task_count)
    return tuple(jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size))


def _process_geometry_chunk(chunk: _GeometryChunk) -> tuple[tuple[str, str], ...]:
    plans = {plan.spec.name: plan for plan in chunk.plans}
    api = HfApi(endpoint=chunk.endpoint, token=chunk.token)
    reference_batches = _indexed_reference_group_batches(chunk.groups)
    with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as client:
        for jobs in _geometry_micro_batches(chunk.jobs):
            _cache_geometry_jobs(api, plans, jobs, chunk.source_root, client)
            try:
                for start_index, groups in reference_batches:
                    references = _worker_reference_batch(
                        groups,
                        chunk.reference_directory,
                        chunk.threshold,
                        start_index=start_index,
                    )
                    for dataset, source_path in jobs:
                        _process_geometry_path(
                            api,
                            plans[dataset],
                            source_path=source_path,
                            references=references,
                            sidecar_root=chunk.sidecar_root,
                            source_root=chunk.source_root,
                            batch_size=chunk.batch_size,
                            progress=None,
                            http_client=client,
                            retain_source=True,
                        )
            finally:
                _remove_cached_geometry_jobs(plans, jobs, chunk.source_root)
    return chunk.jobs


def _cache_geometry_jobs(
    api: HubApi,
    plans: Mapping[str, DatasetPlan],
    jobs: tuple[tuple[str, str], ...],
    source_root: Path,
    client: StreamClient,
) -> None:
    for dataset, source_path in jobs:
        _download_geometry_source(
            api,
            plans[dataset],
            source_path,
            source_root,
            retain_source=True,
            client=client,
        )


def _remove_cached_geometry_jobs(
    plans: Mapping[str, DatasetPlan],
    jobs: tuple[tuple[str, str], ...],
    source_root: Path,
) -> None:
    for dataset, source_path in jobs:
        _cached_geometry_path(source_root, plans[dataset], source_path).unlink(missing_ok=True)


def _geometry_micro_batches(
    jobs: tuple[tuple[str, str], ...],
) -> Iterator[tuple[tuple[str, str], ...]]:
    for start in range(0, len(jobs), _SOURCE_MICRO_BATCH_SIZE):
        yield jobs[start : start + _SOURCE_MICRO_BATCH_SIZE]
