"""Serial and process-pool geometry work over bounded source micro-batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import cast

import httpx
from huggingface_hub import HfApi

from ._protocols import HubApi, StreamClient
from .eea import EeaGroup
from .fileio import DOWNLOAD_TIMEOUT
from .geometry import OVERLAP_KERNEL_VERSION
from .options import BatchLimits, GeometryPathOptions
from .reference_staging import (
    _close_worker_reference_cache,
    _http_client,
    _indexed_reference_group_batches,
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
    SidecarUpdateOptions,
    update_label_sidecar,
)

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
    """Shared inputs for processing every geometry shard in one release."""

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


@dataclass(frozen=True, slots=True)
class _GeometryWorker:
    """Per-process state shared by a bounded set of geometry jobs."""

    chunk: _GeometryChunk
    plans: Mapping[str, DatasetPlan]
    api: HubApi
    reference_batches: tuple[tuple[int, tuple[EeaGroup, ...]], ...]
    batch_indexes: set[int]
    client: StreamClient


def process_geometry_paths(
    api: HubApi,
    plan: DatasetPlan,
    *,
    reference: OverlapReference,
    options: GeometryPathOptions,
) -> None:
    """Merge one reference group into every geometry shard's compact sidecar."""

    with _http_client(options.http_client) as reusable_client:
        options.source_root.mkdir(parents=True, exist_ok=True)
        for source_path in plan.geometry_paths:
            _process_geometry_path(
                api,
                plan,
                source_path,
                (reference,),
                GeometryPathOptions(
                    sidecar_root=options.sidecar_root,
                    source_root=options.source_root,
                    limits=options.limits,
                    progress=options.progress,
                    http_client=reusable_client,
                    retain_source=options.retain_source,
                    reset_sidecar=options.reset_sidecar,
                ),
            )


def _process_geometry_path(
    api: HubApi,
    plan: DatasetPlan,
    source_path: str,
    references: tuple[OverlapReference, ...],
    options: GeometryPathOptions,
) -> None:
    local_source = _download_geometry_source(
        api,
        plan,
        source_path,
        options.source_root,
        retain_source=options.retain_source,
        client=options.http_client,
    )
    sidecar = _sidecar_path(options.sidecar_root, plan.spec, source_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    # Keep any pre-existing .next file as recovery evidence. This fresh staging
    # path is consumed atomically after a complete sidecar pass.
    next_sidecar = sidecar.with_name(f"{sidecar.name}.merge.next")
    update_label_sidecar(
        local_source,
        next_sidecar,
        SidecarUpdateOptions(
            batch_size=options.limits.parquet_batch_size,
            references=references,
            current=sidecar if sidecar.is_file() and not options.reset_sidecar else None,
        ),
    )
    next_sidecar.replace(sidecar)
    if not options.retain_source:
        local_source.unlink(missing_ok=True)
    if options.progress is not None:
        options.progress(
            {"event": "sidecar_updated", "dataset": plan.spec.name, "path": source_path}
        )


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


def _process_reference_groups(options: _GeometryRunOptions) -> None:
    """Process bounded reference batches with resumable compact sidecars."""
    _process_reference_groups_parallel(options)


def _process_reference_groups_parallel(options: _GeometryRunOptions) -> None:
    """Stream bounded source micro-batches through every reference batch."""

    jobs = _geometry_jobs(options.plans)
    if not jobs:
        return
    reference_checksums: dict[str, str] = {}
    with _stage_reference_groups(
        options.groups,
        workdir=options.workdir,
        checksums=reference_checksums,
        client=options.http_client,
    ) as reference_directory:
        options.checksums.update(reference_checksums)
        work = _geometry_work_units(
            options,
            jobs,
            reference_directory,
            _reference_signature(reference_checksums, options.threshold, options.groups),
        )
        _run_geometry_workers(work, options.progress, max_workers=options.limits.workers)


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


def _run_geometry_workers(
    work: tuple[_GeometryChunk, ...],
    progress: Progress | None,
    *,
    max_workers: int,
) -> None:
    worker_count = min(max(max_workers, 1), len(work))
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=get_context("spawn"),
    ) as executor:
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
    limits: BatchLimits = _DEFAULT_BATCH_LIMITS,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    if not jobs:
        return ()
    worker_count = min(max(limits.workers, 1), len(jobs))
    task_count = min(len(jobs), worker_count * limits.geometry_tasks_per_worker)
    chunk_size = max(1, (len(jobs) + task_count - 1) // task_count)
    return tuple(jobs[start : start + chunk_size] for start in range(0, len(jobs), chunk_size))


def _process_geometry_chunk(chunk: _GeometryChunk) -> tuple[tuple[str, str], ...]:
    plans = {plan.spec.name: plan for plan in chunk.plans}
    api = cast(HubApi, HfApi(endpoint=chunk.endpoint, token=chunk.token))
    reference_batches = _indexed_reference_group_batches(
        chunk.groups,
        limits=chunk.limits,
    )
    try:
        with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as client:
            _process_geometry_micro_batches(
                _GeometryWorker(
                    chunk=chunk,
                    plans=plans,
                    api=api,
                    reference_batches=reference_batches,
                    batch_indexes={start_index for start_index, _ in reference_batches},
                    client=client,
                )
            )
    finally:
        _close_worker_reference_cache()
    return chunk.jobs


def _process_geometry_micro_batches(worker: _GeometryWorker) -> None:
    for micro_batch in _geometry_micro_batches(worker.chunk.jobs, worker.chunk.limits):
        _process_geometry_micro_batch(worker, micro_batch)


def _process_geometry_micro_batch(
    worker: _GeometryWorker,
    micro_batch: tuple[tuple[str, str], ...],
) -> None:
    chunk = worker.chunk
    pending = _pending_geometry_batches(chunk, worker.plans, micro_batch)
    jobs = _unfinished_geometry_jobs(micro_batch, worker.batch_indexes, pending)
    if not jobs:
        return
    reset_sidecars = _reset_geometry_sidecars(chunk, worker.plans, jobs, pending)
    _cache_geometry_jobs(worker.api, worker.plans, jobs, chunk.source_root, worker.client)
    try:
        _process_geometry_reference_batches(
            worker,
            jobs,
            pending,
            reset_sidecars,
        )
    finally:
        _remove_cached_geometry_jobs(worker.plans, jobs, chunk.source_root)


def _pending_geometry_batches(
    chunk: _GeometryChunk,
    plans: Mapping[str, DatasetPlan],
    jobs: tuple[tuple[str, str], ...],
) -> dict[tuple[str, str], set[int]]:
    return {
        job: _completed_batches(
            _sidecar_path(chunk.sidecar_root, plans[job[0]].spec, job[1]),
            _geometry_checkpoint_signature(
                chunk.reference_signature,
                plans[job[0]],
                job[1],
            ),
        )
        for job in jobs
    }


def _unfinished_geometry_jobs(
    jobs: tuple[tuple[str, str], ...],
    batch_indexes: set[int],
    pending: Mapping[tuple[str, str], set[int]],
) -> tuple[tuple[str, str], ...]:
    return tuple(job for job in jobs if not batch_indexes <= pending[job])


def _reset_geometry_sidecars(
    chunk: _GeometryChunk,
    plans: Mapping[str, DatasetPlan],
    jobs: tuple[tuple[str, str], ...],
    pending: Mapping[tuple[str, str], set[int]],
) -> set[tuple[str, str]]:
    return {
        job
        for job in jobs
        if not pending[job]
        and _sidecar_path(chunk.sidecar_root, plans[job[0]].spec, job[1]).is_file()
    }


def _process_geometry_reference_batches(
    worker: _GeometryWorker,
    jobs: tuple[tuple[str, str], ...],
    pending: Mapping[tuple[str, str], set[int]],
    reset_sidecars: set[tuple[str, str]],
) -> None:
    for start_index, groups in worker.reference_batches:
        _process_geometry_reference_batch(
            worker,
            groups,
            jobs,
            pending,
            reset_sidecars,
            start_index,
        )


def _process_geometry_reference_batch(
    worker: _GeometryWorker,
    groups: tuple[EeaGroup, ...],
    jobs: tuple[tuple[str, str], ...],
    pending: Mapping[tuple[str, str], set[int]],
    reset_sidecars: set[tuple[str, str]],
    start_index: int,
) -> None:
    outstanding = tuple(job for job in jobs if start_index not in pending[job])
    if not outstanding:
        return
    references = _worker_reference_batch(
        groups,
        worker.chunk.reference_directory,
        worker.chunk.threshold,
        start_index=start_index,
    )
    for job in outstanding:
        _process_geometry_job(
            worker,
            job,
            references,
            start_index,
            pending[job],
            reset_sidecars,
        )


def _process_geometry_job(
    worker: _GeometryWorker,
    job: tuple[str, str],
    references: tuple[OverlapReference, ...],
    start_index: int,
    completed: set[int],
    reset_sidecars: set[tuple[str, str]],
) -> None:
    dataset, source_path = job
    chunk = worker.chunk
    _process_geometry_path(
        worker.api,
        worker.plans[dataset],
        source_path,
        references,
        GeometryPathOptions(
            sidecar_root=chunk.sidecar_root,
            source_root=chunk.source_root,
            limits=chunk.limits,
            http_client=worker.client,
            retain_source=True,
            reset_sidecar=job in reset_sidecars,
        ),
    )
    reset_sidecars.discard(job)
    completed.add(start_index)
    sidecar = _sidecar_path(chunk.sidecar_root, worker.plans[dataset].spec, source_path)
    signature = _geometry_checkpoint_signature(
        chunk.reference_signature,
        worker.plans[dataset],
        source_path,
    )
    _record_completed_batch(sidecar, signature, completed)


def _reference_signature(
    checksums: Mapping[str, str],
    threshold: int,
    groups: tuple[EeaGroup, ...] = (),
) -> str:
    """Fingerprint staged references and overlap policy for resumable sidecars."""

    payload = json.dumps(
        {
            "checksums": dict(sorted(checksums.items())),
            "threshold": threshold,
            "kernel": OVERLAP_KERNEL_VERSION,
            "reference_groups": [_reference_group_signature(group) for group in groups],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_group_signature(group: EeaGroup) -> dict[str, object]:
    assets = group.raster_assets
    if group.vector_asset is not None:
        assets += (group.vector_asset,)
    return {
        "record_id": group.record_id,
        "labels": dict(sorted(group.labels.items())),
        "assets": [
            {
                "path": asset.path,
                "record_id": asset.record_id,
                "source_version": asset.source_version,
                "code": asset.code,
                "name": asset.name,
            }
            for asset in assets
        ],
    }


def _geometry_checkpoint_signature(
    reference_signature: str,
    plan: DatasetPlan,
    source_path: str,
) -> str:
    """Bind a sidecar checkpoint to its immutable source shard and references."""

    payload = json.dumps(
        {
            "references": reference_signature,
            "dataset": plan.spec.name,
            "source_repo": plan.spec.source_repo,
            "source_revision": plan.source_revision,
            "source_path": source_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(sidecar: Path) -> Path:
    return sidecar.with_name(f"{sidecar.name}.done")


def _completed_batches(sidecar: Path, signature: str) -> set[int]:
    """Return completed reference batches for the matching reference signature."""

    payload = _read_checkpoint(_checkpoint_path(sidecar))
    return _checkpoint_batches(payload, signature)


def _read_checkpoint(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _checkpoint_batches(payload: object, signature: str) -> set[int]:
    if not isinstance(payload, Mapping):
        return set()
    if payload.get("signature") != signature:
        return set()
    return _integer_batch_ids(payload.get("batches"))


def _integer_batch_ids(value: object) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if type(item) is int}


def _record_completed_batch(sidecar: Path, signature: str, completed: set[int]) -> None:
    """Atomically record merged reference batches after the sidecar is durable."""

    path = _checkpoint_path(sidecar)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps({"signature": signature, "batches": sorted(completed)}),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)


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
    limits: BatchLimits = _DEFAULT_BATCH_LIMITS,
) -> Iterator[tuple[tuple[str, str], ...]]:
    for start in range(0, len(jobs), limits.retained_source_shards_per_worker):
        yield jobs[start : start + limits.retained_source_shards_per_worker]
