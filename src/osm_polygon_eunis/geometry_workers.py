# ruff: noqa: D100
from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Mapping
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
from .geometry_checkpoints import (
    _completed_batches,
    _geometry_checkpoint_signature,
    _record_completed_batch,
)
from .geometry_chunks import _geometry_micro_batches, _GeometryChunk
from .options import GeometryPathOptions
from .reference_staging import (
    _close_worker_reference_cache,
    _indexed_reference_group_batches,
    _worker_reference_batch,
)
from .release_plan import DatasetPlan, Progress, _cached_geometry_path, _sidecar_path
from .sources import download_to_temp
from .transform import OverlapReference, SidecarUpdateOptions, update_label_sidecar


@dataclass(frozen=True, slots=True)
class _GeometryWorker:
    chunk: _GeometryChunk
    plans: Mapping[str, DatasetPlan]
    api: HubApi
    reference_batches: tuple[tuple[int, tuple[EeaGroup, ...]], ...]
    batch_indexes: set[int]
    client: StreamClient


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
    merge_staging_sidecar = sidecar.with_name(f"{sidecar.name}.merge.next")
    update_label_sidecar(
        local_source,
        merge_staging_sidecar,
        SidecarUpdateOptions(
            batch_size=options.limits.parquet_batch_size,
            references=references,
            current=sidecar if sidecar.is_file() and not options.reset_sidecar else None,
        ),
    )
    merge_staging_sidecar.replace(sidecar)
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
    started = time.monotonic()
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
    _log_geometry_timing(job, start_index, references, time.monotonic() - started)
    sidecar = _sidecar_path(chunk.sidecar_root, worker.plans[dataset].spec, source_path)
    signature = _geometry_checkpoint_signature(
        chunk.reference_signature,
        worker.plans[dataset],
        source_path,
    )
    _record_completed_batch(sidecar, signature, completed)


def _log_geometry_timing(
    job: tuple[str, str],
    reference_batch: int,
    references: tuple[OverlapReference, ...],
    seconds: float,
) -> None:
    hits = sum(int(getattr(reference, "tile_cache_hits", 0)) for reference in references)
    misses = sum(int(getattr(reference, "tile_cache_misses", 0)) for reference in references)
    record = {
        "event": "geometry_batch_done",
        "dataset": job[0],
        "path": job[1],
        "reference_batch": reference_batch,
        "seconds": round(seconds, 3),
        "pid": os.getpid(),
        "tile_cache_hits": hits,
        "tile_cache_misses": misses,
    }
    _log_geometry_event(record)


def _log_geometry_run_plan(
    plans: tuple[DatasetPlan, ...],
    reference_batch_ids: tuple[int, ...],
    reference_signature: str,
) -> None:
    for plan in plans:
        record = {
            "event": "geometry_run_plan",
            "dataset": plan.spec.name,
            "reference_batch_ids": list(reference_batch_ids),
            "checkpoint_signatures": {
                path: _geometry_checkpoint_signature(reference_signature, plan, path)
                for path in plan.geometry_paths
            },
        }
        _log_geometry_event(record)


def _log_geometry_event(record: Mapping[str, object]) -> None:
    sys.stderr.write(json.dumps(record, sort_keys=True) + "\n")
    sys.stderr.flush()


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
