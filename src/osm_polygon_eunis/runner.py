"""Storage-bounded orchestration for the three EUNIS dataset releases."""

from __future__ import annotations

import json
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

from ._protocols import HubApi, StreamClient
from .cards import CardArtifacts, DatasetCardAccumulator
from .domain import EunisResult
from .eea import EeaGroup, resolve_config_data
from .geometry_jobs import _process_reference_groups, process_geometry_paths
from .manifest_state import (
    _compatible_manifests,
    _ExistingManifest,
    _load_existing_manifests,
    _reference_manifest,
    _shared_blobs,
    _try_no_op_release,
    _verify_final_dataset,
    _verify_no_op_dataset,
)
from .publish import (
    ShardExpectation,
    VerificationError,
    build_manifest,
    duplicate_source,
    parquet_signature,
    target_exists,
    upload_manifest,
    upload_replacement,
)
from .references import _http_client, open_reference_group
from .release_plan import (
    DATASET_NAMES,
    DatasetPlan,
    DatasetReceipt,
    Progress,
    ReleaseReceipt,
    _cached_geometry_path,
    _sidecar_path,
    plan_datasets,
    selected_dataset_names,
)
from .sources import (
    capture_revision,
    download_to_temp,
)
from .transform import (
    append_label_sidecar,
    build_label_map,
    enrich_link_shard,
)

__all__ = [
    "DATASET_NAMES",
    "DEFAULT_WORKERS",
    "ConfigError",
    "DatasetPlan",
    "DatasetReceipt",
    "DryRunDataset",
    "DryRunReport",
    "Progress",
    "ReleaseReceipt",
    "VerificationError",
    "finalize_dataset",
    "open_reference_group",
    "plan_datasets",
    "plan_release",
    "process_geometry_paths",
    "run_release",
    "selected_dataset_names",
    "validate_reference_config",
    "verify_release",
]

_SOURCE_WORKERS = 8
DEFAULT_WORKERS = _SOURCE_WORKERS


@dataclass(frozen=True, slots=True)
class _LinkOutput:
    path: str
    source: Path
    output: Path
    expectation: ShardExpectation


@dataclass(frozen=True, slots=True)
class _ReferenceSettings:
    """Validated top-level reference config fields plus the parsed document."""

    source_version: str
    crs: str
    threshold: int
    config: Mapping[str, object]


class ConfigError(ValueError):
    """The reference config or a command input is missing or invalid."""


@dataclass(frozen=True, slots=True)
class DryRunDataset:
    """What a release would do for one dataset, computed without Hub writes."""

    plan: DatasetPlan
    target_exists: bool
    shards_to_upload: tuple[str, ...]

    @property
    def would_duplicate(self) -> bool:
        return not self.target_exists


@dataclass(frozen=True, slots=True)
class DryRunReport:
    """Read-only preview of a release."""

    datasets: tuple[DryRunDataset, ...]
    no_op: bool
    reference_assets: int


def validate_reference_config(config_path: Path) -> None:
    """Fail fast when the reference config is missing, unreadable or invalid."""

    if not config_path.is_file():
        raise ConfigError(f"reference config not found: {config_path}")
    try:
        _settings(config_path)
    except json.JSONDecodeError as error:
        raise ConfigError(f"reference config is not valid JSON: {config_path}: {error}") from error
    except (OSError, ValueError) as error:
        raise ConfigError(f"{config_path}: {error}") from error


def _settings(config_path: Path) -> _ReferenceSettings:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("reference config must be an object")
    return _validated_settings(cast(Mapping[str, object], config))


def _validated_settings(config: Mapping[str, object]) -> _ReferenceSettings:
    source_version = _required_setting(config, "source_version")
    crs = _required_setting(config, "crs")
    threshold = _threshold_setting(config)
    return _ReferenceSettings(source_version, crs, threshold, config)


def _required_setting(config: Mapping[str, object], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference config is missing {field}")
    return value


def _threshold_setting(config: Mapping[str, object]) -> int:
    value = config.get("threshold")
    if not isinstance(value, int) or value < 0:
        raise ValueError("reference config has invalid threshold")
    return value


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
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    progress: Progress | None = None,
    http_client: StreamClient | None = None,
    card: DatasetCardAccumulator | None = None,
    source_cache_root: Path | None = None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    """Append labels, upload changed shards, and clean successful staging files."""

    with _http_client(http_client) as reusable_client:
        return _finalize_dataset_with_client(
            api,
            plan,
            sidecar_root=sidecar_root,
            local_root=local_root,
            batch_size=batch_size,
            parent_commit=parent_commit,
            progress=progress,
            http_client=reusable_client,
            card=card,
            source_cache_root=source_cache_root,
        )


def _finalize_dataset_with_client(
    api: HubApi,
    plan: DatasetPlan,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    progress: Progress | None,
    http_client: StreamClient,
    card: DatasetCardAccumulator | None,
    source_cache_root: Path | None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    local_root.mkdir(parents=True, exist_ok=True)
    card_accumulator = card if card is not None else DatasetCardAccumulator()
    link_by_filename = {Path(link_path).name: link_path for link_path in plan.link_paths}
    expectations: list[ShardExpectation] = []
    current_commit = parent_commit
    for geometry_path in plan.geometry_paths:
        shard_expectations, current_commit = _finalize_shard(
            api,
            plan,
            geometry_path,
            link_by_filename.get(Path(geometry_path).name),
            sidecar_root=sidecar_root,
            local_root=local_root,
            batch_size=batch_size,
            parent_commit=current_commit,
            http_client=http_client,
            card=card_accumulator,
            source_cache_root=source_cache_root,
        )
        expectations.extend(shard_expectations)
        if progress is not None:
            progress({"event": "shards_uploaded", "dataset": plan.spec.name, "path": geometry_path})
    return tuple(expectations), current_commit


def _finalize_shard(
    api: HubApi,
    plan: DatasetPlan,
    geometry_path: str,
    link_path: str | None,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    http_client: StreamClient,
    card: DatasetCardAccumulator,
    source_cache_root: Path | None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    local_source, reused_source = _final_source(
        api,
        plan,
        geometry_path,
        source_cache_root=source_cache_root,
        local_root=local_root,
        http_client=http_client,
    )
    sidecar, local_output, geometry_expectation = _enrich_geometry_shard(
        plan,
        geometry_path,
        local_source,
        sidecar_root=sidecar_root,
        local_root=local_root,
        batch_size=batch_size,
        card=card,
    )
    link_output = _build_link_output(
        api,
        plan,
        link_path,
        local_source,
        sidecar,
        local_root,
        batch_size,
        http_client,
    )
    current_commit = _upload_shard_outputs(
        api,
        plan,
        geometry_path,
        local_output,
        link_output,
        parent_commit,
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
    plan: DatasetPlan,
    geometry_path: str,
    local_source: Path,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    card: DatasetCardAccumulator,
) -> tuple[Path, Path, ShardExpectation]:
    sidecar = _sidecar_path(sidecar_root, plan.spec, geometry_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing completed label sidecar: {sidecar}")
    local_output = local_root / f"{geometry_path.replace('/', '__')}.enriched.parquet"
    local_output.unlink(missing_ok=True)
    rows = append_label_sidecar(
        local_source,
        sidecar,
        local_output,
        batch_size=batch_size,
        observe=card.observe,
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
    api: HubApi,
    plan: DatasetPlan,
    link_path: str | None,
    local_source: Path,
    sidecar: Path,
    local_root: Path,
    batch_size: int,
    http_client: StreamClient,
) -> _LinkOutput | None:
    if link_path is None:
        return None
    labels = build_label_map(local_source, sidecar, batch_size=batch_size)
    local_link_source = download_to_temp(
        api,
        plan.spec.source_repo,
        link_path,
        plan.source_revision,
        local_root,
        client=http_client,
    )
    local_link_output = local_root / f"{link_path.replace('/', '__')}.enriched.parquet"
    local_link_output.unlink(missing_ok=True)
    link_rows = _enrich_link(
        local_link_source,
        local_link_output,
        labels,
        batch_size=batch_size,
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


def run_release(
    api: HubApi,
    *,
    reference_config: Path,
    workdir: Path,
    batch_size: int,
    token: str | None = None,
    progress: Progress | None = None,
    datasets: Sequence[str] | None = None,
    workers: int = _SOURCE_WORKERS,
) -> ReleaseReceipt:
    """Run, publish, and independently verify the selected (default: all) datasets."""

    if workers <= 0:
        raise ValueError("workers must be positive")
    settings = _settings(reference_config)
    workdir.mkdir(parents=True, exist_ok=True)
    plans = plan_datasets(api, datasets)
    groups = resolve_config_data(settings.config)
    sidecar_root = workdir / "sidecars"
    checksums: dict[str, str] = {}
    with _source_cache(workdir) as source_root, _http_client(None) as reusable_client:
        # Detect a verified no-op before any Hub write so a rerun stays read-only.
        no_op_receipt = _try_no_op_release(
            api,
            plans,
            _reference_info(groups, {}, settings),
            workdir=workdir,
            client=reusable_client,
            progress=progress,
        )
        if no_op_receipt is not None:
            return no_op_receipt
        _duplicate_outputs(api, plans, token)
        _process_reference_groups(
            api,
            plans,
            groups,
            sidecar_root=sidecar_root,
            source_root=source_root,
            workdir=workdir,
            threshold=settings.threshold,
            checksums=checksums,
            batch_size=batch_size,
            progress=progress,
            http_client=reusable_client,
            parallelism=workers,
        )
        reference_info = _reference_info(groups, checksums, settings)
        receipts = tuple(
            _finalize_plan(
                api,
                plan,
                sidecar_root=sidecar_root,
                workdir=workdir,
                batch_size=batch_size,
                reference_info=reference_info,
                progress=progress,
                http_client=reusable_client,
                source_cache_root=source_root,
            )
            for plan in plans
        )
    return ReleaseReceipt(receipts, reference_info)


def plan_release(
    api: HubApi,
    *,
    reference_config: Path,
    workdir: Path,
    datasets: Sequence[str] | None = None,
) -> DryRunReport:
    """Resolve plans, references and no-op status without writing to the Hub."""

    settings = _settings(reference_config)
    workdir.mkdir(parents=True, exist_ok=True)
    plans = plan_datasets(api, datasets)
    groups = resolve_config_data(settings.config)
    reference = _reference_info(groups, {}, settings)
    with _http_client(None) as client:
        existing = _load_existing_manifests(api, plans, workdir / "dry-run", client)
    no_op = _compatible_manifests(plans, existing, reference)
    assets = reference.get("assets")
    return DryRunReport(
        tuple(_dry_run_dataset(api, plan, no_op=no_op) for plan in plans),
        no_op,
        len(assets) if isinstance(assets, list) else 0,
    )


def verify_release(
    api: HubApi,
    *,
    workdir: Path,
    datasets: Sequence[str] | None = None,
) -> ReleaseReceipt:
    """Re-verify published targets against their own manifests without any Hub write.

    Raises ``VerificationError`` when a target has no EUNIS manifest or does not match it.
    """

    workdir.mkdir(parents=True, exist_ok=True)
    plans = plan_datasets(api, datasets)
    with _http_client(None) as client:
        existing = _load_existing_manifests(api, plans, workdir / "verify-load", client)
        receipts = tuple(
            _verify_published(api, plan, item, workdir=workdir, client=client)
            for plan, item in zip(plans, existing, strict=True)
        )
    manifest = (receipts[0].verification.manifest if receipts else None) or {}
    reference = manifest.get("reference")
    return ReleaseReceipt(receipts, reference if isinstance(reference, Mapping) else {})


def _verify_published(
    api: HubApi,
    plan: DatasetPlan,
    existing: _ExistingManifest | None,
    *,
    workdir: Path,
    client: StreamClient,
) -> DatasetReceipt:
    if existing is None:
        raise VerificationError(f"{plan.spec.output_repo} has no EUNIS manifest to verify")
    pinned = _manifest_plan(plan, existing.manifest)
    receipt = _verify_no_op_dataset(api, pinned, existing, workdir=workdir, client=client)
    return replace(receipt, no_op=False)


def _manifest_plan(plan: DatasetPlan, manifest: Mapping[str, object]) -> DatasetPlan:
    """Pin a plan to the source revision and paths the published manifest recorded."""

    revision = manifest.get("source_revision")
    paths = manifest.get("source_paths")
    if not isinstance(revision, str) or not isinstance(paths, list):
        raise VerificationError(f"{plan.spec.output_repo} manifest lacks source revision or paths")
    return replace(plan, source_revision=revision, source_files=tuple(str(p) for p in paths))


def _dry_run_dataset(api: HubApi, plan: DatasetPlan, *, no_op: bool) -> DryRunDataset:
    shards = () if no_op else (*plan.geometry_paths, *plan.link_paths)
    return DryRunDataset(plan, target_exists(api, plan.spec.output_repo), tuple(shards))


def _reference_info(
    groups: tuple[EeaGroup, ...],
    checksums: Mapping[str, str],
    settings: _ReferenceSettings,
) -> dict[str, object]:
    return _reference_manifest(
        groups,
        checksums,
        source_version=settings.source_version,
        crs=settings.crs,
        threshold=settings.threshold,
        config=settings.config,
    )


def _duplicate_outputs(api: HubApi, plans: tuple[DatasetPlan, ...], token: str | None) -> None:
    for plan in plans:
        duplicate_source(
            api,
            plan.spec.source_repo,
            plan.spec.output_repo,
            token=token,
        )


def _finalize_plan(
    api: HubApi,
    plan: DatasetPlan,
    *,
    sidecar_root: Path,
    workdir: Path,
    batch_size: int,
    reference_info: Mapping[str, object],
    progress: Progress | None,
    http_client: StreamClient,
    source_cache_root: Path | None = None,
) -> DatasetReceipt:
    target_revision = capture_revision(api, plan.spec.output_repo)
    card = DatasetCardAccumulator()
    expectations, current_commit = finalize_dataset(
        api,
        plan,
        sidecar_root=sidecar_root,
        local_root=workdir / "final",
        batch_size=batch_size,
        parent_commit=target_revision,
        progress=progress,
        http_client=http_client,
        card=card,
        source_cache_root=source_cache_root,
    )
    card_artifacts = _write_card_artifacts(card, workdir, plan, reference_info)
    current_commit = _upload_card_artifacts(
        api,
        plan.spec.output_repo,
        card_artifacts,
        current_commit,
    )
    manifest, changed_paths, added_paths = _publish_dataset_manifest(
        api,
        plan,
        reference_info,
        expectations,
        card_artifacts,
        parent_commit=current_commit,
    )
    shared_blobs = _shared_blobs(api, plan, set(changed_paths))
    verification = _verify_final_dataset(
        api,
        plan,
        expectations=expectations,
        added_paths=added_paths,
        expected_shared_blobs=shared_blobs,
        expected_manifest=manifest,
        expected_artifacts=card_artifacts.hashes,
        temp_dir=workdir / "verify",
        http_client=http_client,
    )
    _cleanup_card_artifacts(card_artifacts)
    return DatasetReceipt(plan, expectations, verification)


def _write_card_artifacts(
    card: DatasetCardAccumulator,
    workdir: Path,
    plan: DatasetPlan,
    reference_info: Mapping[str, object],
) -> CardArtifacts:
    return card.write_artifacts(
        workdir / "cards" / plan.spec.name,
        dataset_name=plan.spec.name,
        source_repo=plan.spec.source_repo,
        target_repo=plan.spec.output_repo,
        source_revision=plan.source_revision,
        reference_version=str(reference_info["source_version"]),
    )


def _publish_dataset_manifest(
    api: HubApi,
    plan: DatasetPlan,
    reference_info: Mapping[str, object],
    expectations: tuple[ShardExpectation, ...],
    card_artifacts: CardArtifacts,
    *,
    parent_commit: str,
) -> tuple[dict[str, Any], tuple[str, ...], tuple[str, ...]]:
    data_paths = tuple(expectation.path for expectation in expectations)
    changed_paths, added_paths = _card_paths(card_artifacts, plan.source_files, data_paths)
    manifest = _build_dataset_manifest(
        plan,
        reference_info,
        expectations,
        changed_paths,
        added_paths,
        card_artifacts,
    )
    manifest_commit = upload_manifest(
        api,
        plan.spec.output_repo,
        manifest,
        parent_commit=parent_commit,
    )
    _advance_commit(api, plan.spec.output_repo, manifest_commit)
    return manifest, changed_paths, added_paths


def _cleanup_card_artifacts(artifacts: CardArtifacts) -> None:
    for path in artifacts.files.values():
        path.unlink(missing_ok=True)


def _upload_card_artifacts(
    api: HubApi,
    target_repo: str,
    artifacts: CardArtifacts,
    parent_commit: str,
) -> str:
    current_commit = parent_commit
    for path, local_path in artifacts.files.items():
        current_commit = _upload_file(
            api,
            target_repo,
            path,
            local_path,
            current_commit,
            commit_message=f"Add EUNIS dataset card artifact {path}",
        )
    return current_commit


def _card_paths(
    artifacts: CardArtifacts,
    source_paths: tuple[str, ...],
    data_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    changed = tuple(path for path in artifacts.files if path in source_paths)
    added = tuple(path for path in artifacts.files if path not in source_paths)
    return (*data_paths, *changed), added


def _build_dataset_manifest(
    plan: DatasetPlan,
    reference_info: Mapping[str, object],
    expectations: tuple[ShardExpectation, ...],
    changed_paths: tuple[str, ...],
    added_paths: tuple[str, ...],
    card_artifacts: CardArtifacts,
) -> dict[str, Any]:
    return build_manifest(
        source_repo=plan.spec.source_repo,
        target_repo=plan.spec.output_repo,
        source_revision=plan.source_revision,
        source_paths=plan.source_files,
        changed_paths=changed_paths,
        added_paths=added_paths,
        reference_manifest=reference_info,
        card_manifest=card_artifacts.manifest,
        rows_by_path={expectation.path: expectation.rows for expectation in expectations},
        schema_by_path={expectation.path: expectation.schema for expectation in expectations},
    )


def _cleanup_source_cache(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _source_cache(workdir: Path) -> Iterator[Path]:
    root = workdir / "source"
    try:
        yield root
    finally:
        _cleanup_source_cache(root)
