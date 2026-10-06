# ruff: noqa: D100
from __future__ import annotations

from . import geometry_checkpoints, geometry_chunks, geometry_workers, reference_staging
from ._protocols import HubApi
from .options import GeometryPathOptions
from .release_plan import DatasetPlan
from .transform import OverlapReference


def process_geometry_paths(
    api: HubApi,
    plan: DatasetPlan,
    *,
    reference: OverlapReference,
    options: GeometryPathOptions,
) -> None:
    """Merge one reference group into every geometry shard's compact sidecar."""

    with reference_staging._http_client(options.http_client) as reusable_client:
        options.source_root.mkdir(parents=True, exist_ok=True)
        for source_path in plan.geometry_paths:
            geometry_workers._process_geometry_path(
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


def _process_reference_groups(options: geometry_chunks._GeometryRunOptions) -> None:
    jobs = geometry_chunks._geometry_jobs(options.plans)
    if not jobs:
        return
    reference_checksums: dict[str, str] = {}
    with reference_staging._stage_reference_groups(
        options.groups,
        workdir=options.workdir,
        checksums=reference_checksums,
        client=options.http_client,
    ) as reference_directory:
        options.checksums.update(reference_checksums)
        reference_signature = geometry_checkpoints._reference_signature(
            reference_checksums, options.threshold, options.groups
        )
        reference_batch_ids = tuple(
            start_index
            for start_index, _ in reference_staging._indexed_reference_group_batches(
                options.groups, limits=options.limits
            )
        )
        work = geometry_chunks._geometry_work_units(
            options,
            jobs,
            reference_directory,
            reference_signature,
        )
        geometry_workers._log_geometry_run_plan(
            options.plans, reference_batch_ids, reference_signature
        )
        geometry_workers._run_geometry_workers(
            work, options.progress, max_workers=options.limits.workers
        )
