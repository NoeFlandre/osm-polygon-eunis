"""Enrich and publish individual geometry and link shards."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._protocols import HubApi, StreamClient
from .cards import DatasetCardAccumulator
from .domain import EunisResult
from .options import ShardContext
from .publish import ShardExpectation, parquet_signature, upload_replacement
from .reference_staging import _http_client
from .release_plan import DatasetPlan, Progress, _cached_geometry_path, _sidecar_path
from .sources import capture_revision, download_to_temp
from .transform import (
    SidecarAppendOptions,
    append_label_sidecar,
    build_label_map,
    enrich_link_shard,
)


@dataclass(frozen=True, slots=True)
class _LinkOutput:
    path: str
    source: Path
    output: Path
    expectation: ShardExpectation


@dataclass(frozen=True, slots=True)
class FinalizeOptions:
    """Settings that bound and identify one dataset finalization pass."""

    sidecar_root: Path
    local_root: Path
    batch_size: int
    parent_commit: str
    progress: Progress | None
    card: DatasetCardAccumulator
    source_cache_root: Path | None
    http_client: StreamClient | None = None


def _commit_id(result: Any) -> str | None:
    for attribute in ("oid", "commit_oid"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _advance_commit(api: HubApi, target_repo: str, result: Any) -> str:
    return _commit_id(result) or capture_revision(api, target_repo)


def finalize_dataset(
    api: HubApi,
    plan: DatasetPlan,
    options: FinalizeOptions,
) -> tuple[tuple[ShardExpectation, ...], str]:
    """Append labels, upload changed shards, and clean successful staging files."""

    with _http_client(options.http_client) as reusable_client:
        return _finalize_dataset_with_client(
            api,
            plan,
            options=options,
            http_client=reusable_client,
        )


def _finalize_dataset_with_client(
    api: HubApi,
    plan: DatasetPlan,
    *,
    options: FinalizeOptions,
    http_client: StreamClient,
) -> tuple[tuple[ShardExpectation, ...], str]:
    context = ShardContext(api, plan, options, http_client)
    options.local_root.mkdir(parents=True, exist_ok=True)
    link_by_filename = {Path(link_path).name: link_path for link_path in plan.link_paths}
    expectations: list[ShardExpectation] = []
    current_commit = options.parent_commit
    for geometry_path in plan.geometry_paths:
        shard_expectations, current_commit = _finalize_shard(
            replace(context, options=replace(options, parent_commit=current_commit)),
            geometry_path,
            link_by_filename.get(Path(geometry_path).name),
        )
        expectations.extend(shard_expectations)
        if options.progress is not None:
            options.progress(
                {"event": "shards_uploaded", "dataset": plan.spec.name, "path": geometry_path}
            )
    return tuple(expectations), current_commit


def _finalize_shard(
    context: ShardContext[HubApi, DatasetPlan, FinalizeOptions, StreamClient],
    geometry_path: str,
    link_path: str | None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    api = context.api
    plan = context.plan
    options = context.options
    http_client = context.http_client
    local_source, reused_source = _final_source(
        api,
        plan,
        geometry_path,
        source_cache_root=options.source_cache_root,
        local_root=options.local_root,
        http_client=http_client,
    )
    sidecar, local_output, geometry_expectation = _enrich_geometry_shard(
        context,
        geometry_path,
        local_source,
    )
    link_output = _build_link_output(
        context,
        link_path,
        local_source,
        sidecar,
    )
    current_commit = _upload_shard_outputs(
        api,
        plan,
        geometry_path,
        local_output,
        link_output,
        options.parent_commit,
    )
    _cleanup_shard(
        local_source,
        local_output,
        sidecar,
        link_output,
        delete_source=not reused_source,
    )
    if link_output is None:
        return (geometry_expectation,), current_commit
    return (geometry_expectation, link_output.expectation), current_commit


def _enrich_geometry_shard(
    context: ShardContext[HubApi, DatasetPlan, FinalizeOptions, StreamClient],
    geometry_path: str,
    local_source: Path,
) -> tuple[Path, Path, ShardExpectation]:
    plan = context.plan
    options = context.options
    sidecar = _sidecar_path(options.sidecar_root, plan.spec, geometry_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing completed label sidecar: {sidecar}")
    local_output = options.local_root / f"{geometry_path.replace('/', '__')}.enriched.parquet"
    local_output.unlink(missing_ok=True)
    rows = append_label_sidecar(
        local_source,
        sidecar,
        local_output,
        SidecarAppendOptions(
            batch_size=options.batch_size,
            observe=options.card.observe,
            count_errors=options.card.record_intersection_errors,
        ),
    )
    return (
        sidecar,
        local_output,
        ShardExpectation(geometry_path, rows, _schema_signature(local_output)),
    )


def _final_source(
    api: HubApi,
    plan: DatasetPlan,
    geometry_path: str,
    *,
    source_cache_root: Path | None,
    local_root: Path,
    http_client: StreamClient,
) -> tuple[Path, bool]:
    if source_cache_root is not None:
        cached = _cached_geometry_path(source_cache_root, plan, geometry_path)
        if cached.is_file():
            return cached, True
    return (
        download_to_temp(
            api,
            plan.spec.source_repo,
            geometry_path,
            plan.source_revision,
            local_root,
            client=http_client,
        ),
        False,
    )


def _upload_shard_outputs(
    api: HubApi,
    plan: DatasetPlan,
    geometry_path: str,
    local_output: Path,
    link_output: _LinkOutput | None,
    parent_commit: str,
) -> str:
    current_commit = _upload_file(
        api,
        plan.spec.output_repo,
        geometry_path,
        local_output,
        parent_commit,
    )
    if link_output is None:
        return current_commit
    return _upload_file(
        api,
        plan.spec.output_repo,
        link_output.path,
        link_output.output,
        current_commit,
    )


def _build_link_output(
    context: ShardContext[HubApi, DatasetPlan, FinalizeOptions, StreamClient],
    link_path: str | None,
    local_source: Path,
    sidecar: Path,
) -> _LinkOutput | None:
    api = context.api
    plan = context.plan
    options = context.options
    http_client = context.http_client
    if link_path is None:
        return None
    labels = build_label_map(local_source, sidecar, batch_size=options.batch_size)
    local_link_source = download_to_temp(
        api,
        plan.spec.source_repo,
        link_path,
        plan.source_revision,
        options.local_root,
        client=http_client,
    )
    local_link_output = options.local_root / f"{link_path.replace('/', '__')}.enriched.parquet"
    local_link_output.unlink(missing_ok=True)
    link_rows = _enrich_link(
        local_link_source,
        local_link_output,
        labels,
        batch_size=options.batch_size,
    )
    return _LinkOutput(
        link_path,
        local_link_source,
        local_link_output,
        ShardExpectation(link_path, link_rows, _schema_signature(local_link_output)),
    )


def _upload_file(
    api: HubApi,
    target_repo: str,
    path: str,
    local_path: Path,
    parent_commit: str,
    commit_message: str | None = None,
) -> str:
    if commit_message is None:
        result = upload_replacement(
            api,
            target_repo,
            path,
            local_path,
            parent_commit=parent_commit,
        )
    else:
        result = upload_replacement(
            api,
            target_repo,
            path,
            local_path,
            parent_commit=parent_commit,
            commit_message=commit_message,
        )
    return _advance_commit(api, target_repo, result)


def _cleanup_shard(
    local_source: Path,
    local_output: Path,
    sidecar: Path,
    link_output: _LinkOutput | None,
    *,
    delete_source: bool = True,
) -> None:
    if delete_source:
        local_source.unlink(missing_ok=True)
    local_output.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    if link_output is not None:
        link_output.source.unlink(missing_ok=True)
        link_output.output.unlink(missing_ok=True)


def _schema_signature(path: Path) -> str:
    return parquet_signature(path)[1]


def _enrich_link(
    source: Path,
    destination: Path,
    labels: Mapping[str, EunisResult],
    *,
    batch_size: int,
) -> int:
    return enrich_link_shard(
        source,
        destination,
        labels_by_polygon_id=labels,
        batch_size=batch_size,
    )
