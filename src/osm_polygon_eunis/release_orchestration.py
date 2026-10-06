"""Coordinate planning, shard processing, publication, and release verification."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import cast

from ._protocols import HubApi, StreamClient
from .card_publishing import _finalize_plan, _PlanOptions
from .domain import SchemaError
from .eea import EeaGroup, resolve_config_data
from .geometry_chunks import _GeometryRunOptions
from .geometry_jobs import _process_reference_groups
from .manifest_state import (
    _compatible_manifests,
    _ExistingManifest,
    _load_existing_manifests,
    _reference_manifest,
    _try_no_op_release,
    _verify_no_op_dataset,
)
from .options import ReleaseOptions, resolve_sidecar_root
from .publish import VerificationError, duplicate_source, target_exists
from .reference_staging import _http_client
from .release_plan import (
    DatasetPlan,
    DatasetReceipt,
    ReleaseReceipt,
    plan_datasets,
)


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
        """Return whether the planned target dataset already exists."""
        return not self.target_exists


@dataclass(frozen=True, slots=True)
class DryRunReport:
    """Read-only preview of a release."""

    datasets: tuple[DryRunDataset, ...]
    no_op: bool
    reference_assets: int


def validate_reference_config(config_path: Path) -> None:
    """Fail fast when the reference config is missing, unreadable or invalid.

    Raises:
        ConfigError: If the path cannot be read or does not match the config schema.
    """

    _settings(config_path)


def _settings(config_path: Path) -> _ReferenceSettings:
    config = _read_config_document(config_path)
    try:
        if not isinstance(config, Mapping):
            raise SchemaError("reference config must be an object")
        return _validated_settings(cast(Mapping[str, object], config))
    except ValueError as error:
        raise ConfigError(f"{config_path}: {error}") from error


def _read_config_document(config_path: Path) -> object:
    if not config_path.is_file():
        raise ConfigError(f"reference config not found: {config_path}")
    try:
        contents = config_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConfigError(f"reference config cannot be read: {config_path}: {error}") from error
    try:
        return json.loads(contents)
    except json.JSONDecodeError as error:
        raise ConfigError(f"reference config is not valid JSON: {config_path}: {error}") from error


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


def run_release(api: HubApi, options: ReleaseOptions) -> ReleaseReceipt:
    """Run, publish, and independently verify the selected (default: all) datasets.

    ``options.max_intersection_errors`` fails a dataset before its manifest is published
    when more overlap candidates than this were dropped by GEOS intersection
    errors; ``None`` (the default) disables the check.

    Raises:
        ConfigError: If the EEA reference configuration cannot be loaded.
        IntersectionErrorLimitError: If a dataset exceeds the configured error limit.
    """

    _validate_release_options(options.limits.workers, options.max_intersection_errors)
    settings = _settings(options.reference_config)
    workdir = options.workdir
    workdir.mkdir(parents=True, exist_ok=True)
    plans = plan_datasets(api, options.datasets)
    groups = resolve_config_data(settings.config)
    sidecar_root = _sidecar_root(workdir)
    checksums: dict[str, str] = {}
    with _source_cache(workdir) as source_root, _http_client(None) as reusable_client:
        # Detect a verified no-op before any Hub write so a rerun stays read-only.
        no_op_receipt = _try_no_op_release(
            api,
            plans,
            _reference_info(groups, {}, settings),
            workdir=workdir,
            client=reusable_client,
            progress=options.progress,
        )
        if no_op_receipt is not None:
            return no_op_receipt
        _duplicate_outputs(api, plans, options.token)
        _process_reference_groups(
            _GeometryRunOptions(
                api=api,
                plans=plans,
                groups=groups,
                sidecar_root=sidecar_root,
                source_root=source_root,
                workdir=workdir,
                threshold=settings.threshold,
                checksums=checksums,
                limits=options.limits,
                progress=options.progress,
                http_client=reusable_client,
            )
        )
        reference_info = _reference_info(groups, checksums, settings)
        receipts = tuple(
            _finalize_plan(
                api,
                plan,
                options=_PlanOptions(
                    sidecar_root=sidecar_root,
                    workdir=workdir,
                    batch_size=options.limits.parquet_batch_size,
                    reference_info=reference_info,
                    progress=options.progress,
                    http_client=reusable_client,
                    source_cache_root=source_root,
                    max_intersection_errors=options.max_intersection_errors,
                ),
            )
            for plan in plans
        )
    return ReleaseReceipt(receipts, reference_info)


def _validate_release_options(workers: int, max_intersection_errors: int | None) -> None:
    if workers <= 0:
        raise ValueError("workers must be positive")
    if max_intersection_errors is not None and max_intersection_errors < 0:
        raise ValueError("max_intersection_errors must be non-negative")


def plan_release(
    api: HubApi,
    *,
    reference_config: Path,
    workdir: Path,
    datasets: Sequence[str] | None = None,
) -> DryRunReport:
    """Resolve plans, references and no-op status without writing to the Hub.

    Raises:
        ConfigError: If the EEA reference configuration cannot be loaded.
    """

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


def _cleanup_source_cache(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


def _sidecar_root(workdir: Path) -> Path:
    """Resolve the resumable sidecar location, including Grid persistent storage."""

    root = resolve_sidecar_root(workdir)
    root.mkdir(parents=True, exist_ok=True)
    return root


@contextmanager
def _source_cache(workdir: Path) -> Iterator[Path]:
    configured = os.environ.get("EUNIS_SOURCE_DIR")
    if configured:
        scratch_root = Path(configured)
        scratch_root.mkdir(parents=True, exist_ok=True)
        with TemporaryDirectory(dir=scratch_root, prefix="source-") as directory:
            yield Path(directory)
        return

    root = workdir / "source"
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        _cleanup_source_cache(root)
