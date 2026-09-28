"""Build, publish, and verify EUNIS dataset cards and manifests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from ._protocols import HubApi, StreamClient
from .cards import CardArtifacts, DatasetCardAccumulator
from .manifest_state import (
    _FinalDatasetVerificationOptions,
    _shared_blobs,
    _verify_final_dataset,
)
from .publish import ManifestBuildOptions, ShardExpectation, build_manifest, upload_manifest
from .release_plan import DatasetPlan, DatasetReceipt, Progress
from .shard_processing import FinalizeOptions, _advance_commit, _upload_file, finalize_dataset
from .sources import capture_revision, download_to_temp


@dataclass(frozen=True, slots=True)
class _PlanOptions:
    sidecar_root: Path
    workdir: Path
    batch_size: int
    reference_info: Mapping[str, object]
    progress: Progress | None
    http_client: StreamClient
    source_cache_root: Path | None
    max_intersection_errors: int | None


def _finalize_plan(
    api: HubApi,
    plan: DatasetPlan,
    *,
    options: _PlanOptions,
) -> DatasetReceipt:
    target_revision = capture_revision(api, plan.spec.output_repo)
    source_readme = _download_source_readme(api, plan, options)
    card = DatasetCardAccumulator(source_readme=source_readme)
    expectations, current_commit = finalize_dataset(
        api,
        plan,
        FinalizeOptions(
            sidecar_root=options.sidecar_root,
            local_root=options.workdir / "final",
            batch_size=options.batch_size,
            parent_commit=target_revision,
            progress=options.progress,
            card=card,
            source_cache_root=options.source_cache_root,
            http_client=options.http_client,
        ),
    )
    _enforce_intersection_error_limit(card, plan, options.max_intersection_errors)
    card_artifacts = _write_card_artifacts(card, options.workdir, plan, options.reference_info)
    current_commit = _upload_card_artifacts(
        api,
        plan.spec.output_repo,
        card_artifacts,
        current_commit,
    )
    manifest, changed_paths, added_paths = _publish_dataset_manifest(
        api,
        plan,
        options.reference_info,
        expectations,
        card_artifacts,
        parent_commit=current_commit,
    )
    shared_blobs = _shared_blobs(api, plan, set(changed_paths))
    verification = _verify_final_dataset(
        api,
        plan,
        _FinalDatasetVerificationOptions(
            expectations=expectations,
            added_paths=added_paths,
            expected_shared_blobs=shared_blobs,
            expected_manifest=manifest,
            expected_artifacts=card_artifacts.hashes,
            temp_dir=options.workdir / "verify",
            http_client=options.http_client,
        ),
    )
    _cleanup_card_artifacts(card_artifacts)
    return DatasetReceipt(plan, expectations, verification)


class IntersectionErrorLimitError(RuntimeError):
    """A dataset dropped more overlap candidates to GEOS errors than allowed."""


def _enforce_intersection_error_limit(
    card: DatasetCardAccumulator,
    plan: DatasetPlan,
    limit: int | None,
) -> None:
    if limit is None or card.intersection_errors <= limit:
        return
    raise IntersectionErrorLimitError(
        f"{plan.spec.name}: {card.intersection_errors} intersection errors exceed "
        f"the limit of {limit}; the dataset manifest was not published"
    )


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


def _download_source_readme(
    api: HubApi,
    plan: DatasetPlan,
    options: _PlanOptions,
) -> str | None:
    """Fetch the small pinned source card so its Viewer config is preserved."""

    if "README.md" not in plan.source_files:
        return None
    options.workdir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="eunis-source-card-", dir=options.workdir) as temporary:
        path = download_to_temp(
            api,
            plan.spec.source_repo,
            "README.md",
            plan.source_revision,
            Path(temporary),
            client=options.http_client,
        )
        return path.read_text(encoding="utf-8")


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
        ManifestBuildOptions(
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
    )
