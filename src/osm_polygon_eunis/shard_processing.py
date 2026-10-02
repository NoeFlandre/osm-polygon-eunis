"""Enrich and publish individual geometry and link shards."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from ._protocols import HubApi, StreamClient
from .cards import DatasetCardAccumulator
from .domain import EunisResult
from .options import ShardContext
from .publish import ShardExpectation, parquet_signature, upload_replacement, upload_replacements
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


_COMMIT_MAX_SHARDS = 24
_COMMIT_MAX_BYTES = 3 * 1024**3


@dataclass(slots=True)
class _PendingCommit:
    """Enriched shards waiting for one grouped Hub commit."""

    card: DatasetCardAccumulator
    files: list[tuple[str, Path]]
    expectations: list[ShardExpectation]
    geometry_paths: list[str]
    scratch: list[Path]
    size: int = 0

    def is_full(self) -> bool:
        return len(self.geometry_paths) >= _COMMIT_MAX_SHARDS or self.size >= _COMMIT_MAX_BYTES


def finalize_dataset(
    api: HubApi,
    plan: DatasetPlan,
    options: FinalizeOptions,
) -> tuple[tuple[ShardExpectation, ...], str]:
    """Append labels and upload changed shards in grouped, resumable Hub commits.

    Label sidecars are the resumable checkpoints and are kept. Each committed
    group is recorded in a progress file, so an interrupted pass resumes after
    the last committed group instead of repeating (or failing on) earlier shards.
    """

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
    options.local_root.mkdir(parents=True, exist_ok=True)
    link_by_filename = {Path(link_path).name: link_path for link_path in plan.link_paths}
    expectations, committed = _load_progress(options, plan)
    current_commit = options.parent_commit
    pending = _new_pending(options.card)
    for geometry_path in plan.geometry_paths:
        if geometry_path in committed:
            _report_uploaded(options, plan, geometry_path)
            continue
        context = ShardContext(api, plan, replace(options, card=pending.card), http_client)
        link_path = link_by_filename.get(Path(geometry_path).name)
        _prepare_shard(context, pending, geometry_path, link_path)
        if pending.is_full():
            current_commit = _commit_pending(
                api, plan, options, pending, expectations, current_commit
            )
            pending = _new_pending(options.card)
    current_commit = _commit_remaining(api, plan, options, pending, expectations, current_commit)
    return tuple(expectations), current_commit


def _commit_remaining(
    api: HubApi,
    plan: DatasetPlan,
    options: FinalizeOptions,
    pending: _PendingCommit,
    expectations: list[ShardExpectation],
    current_commit: str,
) -> str:
    if not pending.geometry_paths:
        return current_commit
    return _commit_pending(api, plan, options, pending, expectations, current_commit)


def _new_pending(card: DatasetCardAccumulator) -> _PendingCommit:
    return _PendingCommit(card.spawn(), [], [], [], [])


def _prepare_shard(
    context: ShardContext[HubApi, DatasetPlan, FinalizeOptions, StreamClient],
    pending: _PendingCommit,
    geometry_path: str,
    link_path: str | None,
) -> None:
    options = context.options
    local_source, reused_source = _final_source(
        context.api,
        context.plan,
        geometry_path,
        source_cache_root=options.source_cache_root,
        local_root=options.local_root,
        http_client=context.http_client,
    )
    sidecar, local_output, geometry_expectation = _enrich_geometry_shard(
        context,
        geometry_path,
        local_source,
    )
    link_output = _build_link_output(context, link_path, local_source, sidecar)
    pending.files.append((geometry_path, local_output))
    pending.expectations.append(geometry_expectation)
    pending.geometry_paths.append(geometry_path)
    pending.size += local_output.stat().st_size
    pending.scratch.append(local_output)
    if link_output is not None:
        pending.files.append((link_output.path, link_output.output))
        pending.expectations.append(link_output.expectation)
        pending.size += link_output.output.stat().st_size
        pending.scratch.extend((link_output.output, link_output.source))
    if not reused_source:
        pending.scratch.append(local_source)


def _commit_pending(
    api: HubApi,
    plan: DatasetPlan,
    options: FinalizeOptions,
    pending: _PendingCommit,
    expectations: list[ShardExpectation],
    parent_commit: str,
) -> str:
    result = upload_replacements(
        api,
        plan.spec.output_repo,
        pending.files,
        parent_commit=parent_commit,
    )
    current_commit = _advance_commit(api, plan.spec.output_repo, result)
    options.card.merge(pending.card)
    expectations.extend(pending.expectations)
    _save_progress(options, plan, expectations)
    for path in pending.scratch:
        path.unlink(missing_ok=True)
    for geometry_path in pending.geometry_paths:
        _report_uploaded(options, plan, geometry_path)
    return current_commit


def _report_uploaded(options: FinalizeOptions, plan: DatasetPlan, geometry_path: str) -> None:
    if options.progress is not None:
        options.progress(
            {"event": "shards_uploaded", "dataset": plan.spec.name, "path": geometry_path}
        )


def _progress_path(options: FinalizeOptions, plan: DatasetPlan) -> Path:
    return options.local_root / f"{plan.spec.name}.finalize.json"


def _save_progress(
    options: FinalizeOptions,
    plan: DatasetPlan,
    expectations: list[ShardExpectation],
) -> None:
    payload = {
        "source_revision": plan.source_revision,
        "expectations": [[e.path, e.rows, e.schema] for e in expectations],
        "card": options.card.snapshot(),
    }
    path = _progress_path(options, plan)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _progress_payload(options: FinalizeOptions, plan: DatasetPlan) -> dict[str, Any] | None:
    path = _progress_path(options, plan)
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if payload.get("source_revision") == plan.source_revision else None


def _load_progress(
    options: FinalizeOptions,
    plan: DatasetPlan,
) -> tuple[list[ShardExpectation], set[str]]:
    payload = _progress_payload(options, plan)
    if payload is None:
        return [], set()
    options.card.restore(payload["card"])
    expectations = [ShardExpectation(*entry) for entry in payload["expectations"]]
    return expectations, {e.path for e in expectations}


def clear_finalize_progress(options: FinalizeOptions, plan: DatasetPlan) -> None:
    """Forget resumable upload progress once the dataset is published and verified."""

    _progress_path(options, plan).unlink(missing_ok=True)


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
