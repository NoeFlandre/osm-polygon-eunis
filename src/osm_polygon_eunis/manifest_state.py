"""Release manifest identity, existing-manifest loading and no-op detection."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from huggingface_hub.errors import RepositoryNotFoundError

from ._protocols import HubApi, StreamClient
from .eea import EeaGroup
from .publish import (
    ShardExpectation,
    VerificationReceipt,
    verify_dataset,
)
from .references import (
    _asset_key,
)
from .release_plan import (
    DatasetPlan,
    DatasetReceipt,
    Progress,
    ReleaseReceipt,
)
from .sources import (
    capture_revision,
    download_to_temp,
    list_repo_files,
)


@dataclass(frozen=True, slots=True)
class _ExistingManifest:
    """A target manifest and the immutable target revision that contains it."""

    revision: str
    manifest: Mapping[str, object]


def _reference_manifest(
    groups: tuple[EeaGroup, ...],
    checksums: Mapping[str, str],
    *,
    source_version: str,
    crs: str,
    threshold: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    assets: list[dict[str, object]] = []
    for group in groups:
        group_assets = [*group.raster_assets]
        if group.vector_asset is not None:
            group_assets.append(group.vector_asset)
        for asset in group_assets:
            assets.append(
                {
                    "record_id": asset.record_id,
                    "path": asset.path,
                    "url": asset.url,
                    "size": asset.size,
                    "etag": asset.etag,
                    "code": asset.code,
                    "name": asset.name,
                    "sha256": checksums.get(_asset_key(group, asset)),
                }
            )
    return {
        "source_version": source_version,
        "crs": crs,
        "threshold": threshold,
        "classification_record": config.get("classification_record"),
        "assets": sorted(assets, key=lambda asset: (str(asset["record_id"]), str(asset["path"]))),
    }


def _reference_identity(reference: Mapping[str, object]) -> Mapping[str, object]:
    """Remove downloaded-byte hashes before comparing pinned reference metadata."""

    normalized_assets = _reference_assets(reference.get("assets"))
    if normalized_assets is None:
        return reference
    normalized = dict(reference)
    normalized["assets"] = normalized_assets
    return normalized


def _reference_assets(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    return [_reference_asset(asset) for asset in value if isinstance(asset, Mapping)]


def _reference_asset(asset: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in asset.items() if key != "sha256"}


def _manifest_matches_inputs(
    plan: DatasetPlan,
    manifest: Mapping[str, object],
    reference: Mapping[str, object],
) -> bool:
    """Check whether a manifest describes exactly the current pinned inputs."""

    return all(
        (
            manifest.get("manifest_version") == 3,
            manifest.get("source_repo") == plan.spec.source_repo,
            manifest.get("target_repo") == plan.spec.output_repo,
            manifest.get("source_revision") == plan.source_revision,
            manifest.get("source_paths") == list(plan.source_files),
            _reference_identity(_mapping_field(manifest, "reference"))
            == _reference_identity(reference),
        )
    )


def _mapping_field(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = payload.get(field)
    return value if isinstance(value, Mapping) else {}


def _load_existing_manifest(
    api: HubApi,
    plan: DatasetPlan,
    directory: Path,
    client: StreamClient,
) -> _ExistingManifest | None:
    """Load a tiny target manifest without using the persistent Hub cache."""

    try:
        target_revision = capture_revision(api, plan.spec.output_repo)
    except RepositoryNotFoundError:
        return None
    paths = {entry.path for entry in list_repo_files(api, plan.spec.output_repo, target_revision)}
    if "eunis/manifest.json" not in paths:
        return None
    local = download_to_temp(
        api,
        plan.spec.output_repo,
        "eunis/manifest.json",
        target_revision,
        directory,
        client=client,
    )
    try:
        payload = json.loads(local.read_text(encoding="utf-8"))
    finally:
        local.unlink(missing_ok=True)
    if not isinstance(payload, Mapping):
        return None
    return _ExistingManifest(target_revision, payload)


def _manifest_expectations(manifest: Mapping[str, object]) -> tuple[ShardExpectation, ...] | None:
    fields = _manifest_table_fields(manifest)
    if fields is None:
        return None
    rows, schemas = fields
    try:
        return tuple(
            _manifest_expectation(path, rows[path], schemas[path]) for path in sorted(rows, key=str)
        )
    except (KeyError, TypeError, ValueError):
        return None


def _manifest_table_fields(
    manifest: Mapping[str, object],
) -> tuple[Mapping[object, object], Mapping[object, object]] | None:
    rows = manifest.get("rows_by_path")
    schemas = manifest.get("schema_by_path")
    if not isinstance(rows, Mapping) or not isinstance(schemas, Mapping):
        return None
    if set(rows) != set(schemas):
        return None
    return rows, schemas


def _manifest_expectation(path: object, rows: Any, schema: Any) -> ShardExpectation:
    if not isinstance(path, str):
        raise TypeError("manifest shard path must be a string")
    return ShardExpectation(path, int(rows), str(schema))


def _manifest_artifacts(manifest: Mapping[str, object]) -> Mapping[str, str] | None:
    card = manifest.get("card")
    if not isinstance(card, Mapping):
        return None
    artifacts: dict[str, str] = {}
    for path_field, hash_field in (
        ("readme_path", "readme_sha256"),
        ("map_path", "map_sha256"),
    ):
        path = card.get(path_field)
        digest = card.get(hash_field)
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        artifacts[path] = digest
    return artifacts


def _required_manifest_paths(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        _required_string_list(manifest, "changed_paths"),
        _required_string_list(manifest, "added_paths"),
    )


def _required_string_list(manifest: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = manifest.get(field)
    if not isinstance(value, list):
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    paths = tuple(value)
    if not all(isinstance(path, str) for path in paths):
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    return cast(tuple[str, ...], paths)


def _required_no_op_parts(
    manifest: Mapping[str, object],
) -> tuple[tuple[ShardExpectation, ...], Mapping[str, str], tuple[str, ...], tuple[str, ...]]:
    expectations = _manifest_expectations(manifest)
    artifacts = _manifest_artifacts(manifest)
    if expectations is None or artifacts is None:
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    changed_paths, added_paths = _required_manifest_paths(manifest)
    return expectations, artifacts, changed_paths, added_paths


def _verify_no_op_dataset(
    api: HubApi,
    plan: DatasetPlan,
    existing: _ExistingManifest,
    *,
    workdir: Path,
    client: StreamClient,
) -> DatasetReceipt:
    manifest = existing.manifest
    expectations, artifacts, changed_paths, added_paths = _required_no_op_parts(manifest)
    shared_blobs = _shared_blobs(api, plan, set(changed_paths))
    verification = _verify_final_dataset(
        api,
        plan,
        expectations=expectations,
        added_paths=tuple(added_paths),
        expected_shared_blobs=shared_blobs,
        expected_manifest=manifest,
        expected_artifacts=artifacts,
        temp_dir=workdir / "verify-no-op",
        http_client=client,
    )
    return DatasetReceipt(plan, expectations, verification, no_op=True)


def _load_existing_manifests(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    directory: Path,
    client: StreamClient,
) -> tuple[_ExistingManifest | None, ...]:
    return tuple(_load_existing_manifest(api, plan, directory, client) for plan in plans)


def _matches_existing_manifest(
    plan: DatasetPlan,
    existing: _ExistingManifest | None,
    reference: Mapping[str, object],
) -> bool:
    return existing is not None and _manifest_matches_inputs(plan, existing.manifest, reference)


def _compatible_manifests(
    plans: tuple[DatasetPlan, ...],
    existing: tuple[_ExistingManifest | None, ...],
    reference: Mapping[str, object],
) -> bool:
    return all(
        _matches_existing_manifest(plan, item, reference)
        for plan, item in zip(plans, existing, strict=True)
    )


def _no_op_receipts(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    existing: tuple[_ExistingManifest | None, ...],
    *,
    workdir: Path,
    client: StreamClient,
) -> tuple[DatasetReceipt, ...]:
    return tuple(
        _verify_no_op_dataset(
            api,
            plan,
            cast(_ExistingManifest, item),
            workdir=workdir,
            client=client,
        )
        for plan, item in zip(plans, existing, strict=True)
    )


def _try_no_op_release(
    api: HubApi,
    plans: tuple[DatasetPlan, ...],
    reference: Mapping[str, object],
    *,
    workdir: Path,
    client: StreamClient,
    progress: Progress | None,
) -> ReleaseReceipt | None:
    existing = _load_existing_manifests(api, plans, workdir / "noop", client)
    if not _compatible_manifests(plans, existing, reference):
        return None
    receipts = _no_op_receipts(api, plans, existing, workdir=workdir, client=client)
    if progress is not None:
        for receipt in receipts:
            progress({"event": "dataset_no_op", "dataset": receipt.plan.spec.name})
    manifest = receipts[0].verification.manifest or {}
    return ReleaseReceipt(receipts, _mapping_field(manifest, "reference"))


def _verify_final_dataset(
    api: HubApi,
    plan: DatasetPlan,
    *,
    expectations: tuple[ShardExpectation, ...],
    added_paths: tuple[str, ...],
    expected_shared_blobs: Mapping[str, str],
    expected_manifest: Mapping[str, Any],
    expected_artifacts: Mapping[str, str],
    temp_dir: Path,
    http_client: StreamClient,
) -> VerificationReceipt:
    return verify_dataset(
        api,
        plan.spec.output_repo,
        expectations=expectations,
        expected_tree_paths=(*plan.source_files, *added_paths, "eunis/manifest.json"),
        expected_shared_blobs=expected_shared_blobs,
        expected_manifest=expected_manifest,
        expected_artifacts=expected_artifacts,
        temp_dir=temp_dir,
        http_client=http_client,
    )


def _shared_blobs(
    api: HubApi,
    plan: DatasetPlan,
    changed_paths: set[str],
) -> dict[str, str]:
    return {
        entry.path: entry.blob_id
        for entry in list_repo_files(api, plan.spec.source_repo, plan.source_revision)
        if entry.path not in changed_paths and isinstance(getattr(entry, "blob_id", None), str)
    }
