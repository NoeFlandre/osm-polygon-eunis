"""Narrow Hugging Face publication and independent verification operations."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import duplicate_repo
from huggingface_hub.utils import RepositoryNotFoundError

from ._protocols import HubApi, StreamClient
from .fileio import sha256_file
from .sources import capture_revision, download_to_temp

MANIFEST_VERSION = 5


class VerificationError(ValueError):
    """A published target does not match what the release expected."""


@dataclass(frozen=True, slots=True)
class ShardExpectation:
    """Expected remote row count and schema fingerprint for one Parquet file."""

    path: str
    rows: int
    schema: str


@dataclass(frozen=True, slots=True)
class VerificationReceipt:
    """Independent target verification result."""

    target_repo: str
    target_revision: str
    rows_by_path: Mapping[str, int]
    shared_paths: tuple[str, ...]
    manifest: Mapping[str, Any] | None


@dataclass(frozen=True, slots=True)
class ManifestBuildOptions:
    """Inputs used to build a deterministic dataset manifest."""

    source_repo: str
    target_repo: str
    source_revision: str
    source_paths: Iterable[str]
    changed_paths: Iterable[str]
    reference_manifest: Mapping[str, Any]
    rows_by_path: Mapping[str, int]
    schema_by_path: Mapping[str, str] | None = None
    added_paths: Iterable[str] = ()
    card_manifest: Mapping[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class DatasetVerificationOptions:
    """Expected remote state used for independent dataset verification."""

    expectations: Iterable[ShardExpectation]
    expected_tree_paths: Iterable[str]
    expected_shared_blobs: Mapping[str, str] | None = None
    expected_manifest: Mapping[str, Any] | None = None
    expected_artifacts: Mapping[str, str] | None = None
    temp_dir: Path | None = None
    http_client: StreamClient | None = None


def target_exists(api: HubApi, target_repo: str) -> bool:
    """Return whether a target dataset repository already exists (read-only)."""

    try:
        api.repo_info(target_repo, repo_type="dataset")
    except RepositoryNotFoundError:
        return False
    return True


def duplicate_source(
    api: HubApi,
    source_repo: str,
    target_repo: str,
    *,
    token: str | None = None,
    duplicate: Callable[..., Any] = duplicate_repo,
) -> bool:
    """Duplicate a dataset server-side only when the target does not exist."""

    if not target_exists(api, target_repo):
        duplicate(
            source_repo,
            target_repo,
            repo_type="dataset",
            private=False,
            token=token,
            exist_ok=False,
        )
        return True
    return False


def upload_replacement(
    api: HubApi,
    target_repo: str,
    path: str,
    local_path: Path,
    *,
    parent_commit: str | None = None,
    commit_message: str | None = None,
) -> Any:
    """Upload one replacement shard as one explicit Hub commit."""

    if not local_path.is_file():
        raise FileNotFoundError(local_path)
    return api.upload_file(
        path_or_fileobj=local_path,
        path_in_repo=path,
        repo_id=target_repo,
        repo_type="dataset",
        revision="main",
        parent_commit=parent_commit,
        commit_message=commit_message or f"Add EUNIS labels to {path}",
    )


def _manifest_bytes(manifest: Mapping[str, Any]) -> bytes:
    return (json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n").encode()


def upload_manifest(
    api: HubApi,
    target_repo: str,
    manifest: Mapping[str, Any],
    *,
    parent_commit: str | None = None,
) -> Any:
    """Upload the deterministic run manifest at its fixed target path."""

    return api.upload_file(
        path_or_fileobj=_manifest_bytes(manifest),
        path_in_repo="eunis/manifest.json",
        repo_id=target_repo,
        repo_type="dataset",
        revision="main",
        parent_commit=parent_commit,
        commit_message="Record EUNIS enrichment manifest",
    )


def parquet_signature(path: Path) -> tuple[int, str]:
    """Return deterministic row count and Arrow-schema fingerprint."""

    parquet = pq.ParquetFile(path)
    serialized = parquet.schema_arrow.serialize().to_pybytes()
    return parquet.metadata.num_rows, hashlib.sha256(serialized).hexdigest()


def _manifest_geometry_policy(
    card_manifest: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if card_manifest is None:
        return None
    policy = card_manifest.get("geometry_policy")
    return dict(policy) if isinstance(policy, Mapping) else None


def _software_provenance() -> dict[str, str]:
    """Return the installed package version and the best available source commit."""

    commit = _environment_source_commit() or _git_source_commit()
    return {
        "name": "osm-polygon-eunis",
        "version": version("osm-polygon-eunis"),
        "commit": _validated_source_commit(commit),
    }


def _environment_source_commit() -> str | None:
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        commit = os.environ.get(name)
        if commit:
            return commit
    return None


def _git_source_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # noqa: S603 - resolved Git executable and fixed arguments
            (git, "rev-parse", "HEAD"),
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            check=False,
            text=True,
        )
    except OSError:
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _validated_source_commit(commit: str | None) -> str:
    if commit is None:
        raise ValueError("source commit unavailable; set EUNIS_SOURCE_COMMIT to a full commit SHA")
    if re.fullmatch(r"[0-9a-fA-F]{40}|[0-9a-fA-F]{64}", commit) is None:
        raise ValueError("source commit must be a 40- or 64-character hexadecimal SHA")
    return commit.lower()


def build_manifest(options: ManifestBuildOptions) -> dict[str, Any]:
    """Build a stable manifest that makes source conservation explicit."""

    source = sorted(set(options.source_paths))
    changed = sorted(set(options.changed_paths))
    added = sorted(set(options.added_paths))
    schemas = options.schema_by_path or {}
    _validate_manifest_paths(source, changed, added)
    return {
        "manifest_version": MANIFEST_VERSION,
        "source_repo": options.source_repo,
        "target_repo": options.target_repo,
        "source_revision": options.source_revision,
        "source_paths": source,
        "changed_paths": changed,
        "added_paths": added,
        "shared_paths": sorted(set(source) - set(changed)),
        "rows_by_path": {path: options.rows_by_path[path] for path in sorted(options.rows_by_path)},
        "schema_by_path": {path: schemas[path] for path in sorted(schemas)},
        "software": _software_provenance(),
        "reference": dict(options.reference_manifest),
        "geometry_policy": _manifest_geometry_policy(options.card_manifest),
        "card": dict(options.card_manifest) if options.card_manifest is not None else None,
    }


def _validate_manifest_paths(
    source: list[str],
    changed: list[str],
    added: list[str],
) -> None:
    source_paths = set(source)
    changed_paths = set(changed)
    added_paths = set(added)
    if not changed_paths.issubset(source_paths):
        raise ValueError("changed paths are not a subset of source paths")
    if added_paths & changed_paths:
        raise ValueError("a path cannot be both changed and added")
    if added_paths & source_paths:
        raise ValueError("added paths already exist in source paths")


def _remote_files(api: HubApi, repo_id: str, revision: str) -> dict[str, Any]:
    entries = api.list_repo_tree(
        repo_id,
        path_in_repo="",
        recursive=True,
        revision=revision,
        repo_type="dataset",
    )
    return {
        entry.path: entry
        for entry in entries
        if not hasattr(entry, "tree_id") and isinstance(getattr(entry, "path", None), str)
    }


def verify_dataset(
    api: HubApi,
    target_repo: str,
    options: DatasetVerificationOptions,
) -> VerificationReceipt:
    """Verify target tree, unchanged blob identities, Parquet schemas, and manifest."""

    revision = capture_revision(api, target_repo)
    remote = _remote_files(api, target_repo, revision)
    _validate_tree(remote, options.expected_tree_paths)
    _verify_shared_blobs(remote, options.expected_shared_blobs)

    if options.temp_dir is None:
        temporary = TemporaryDirectory(prefix="osm-polygon-eunis-verify-")
        directory = Path(temporary.name)
    else:
        temporary = None
        directory = options.temp_dir
    rows_by_path: dict[str, int] = {}
    try:
        rows_by_path = _verify_expectations(
            api,
            target_repo,
            revision,
            options.expectations,
            directory,
            options.http_client,
        )
        manifest = _verify_manifest(
            api,
            target_repo,
            revision,
            options.expected_manifest,
            directory,
            options.http_client,
        )
        _verify_artifacts(
            api,
            target_repo,
            revision,
            options.expected_artifacts,
            directory,
            options.http_client,
        )
        shared_paths = tuple(sorted(options.expected_shared_blobs or {}))
        return VerificationReceipt(target_repo, revision, rows_by_path, shared_paths, manifest)
    finally:
        if temporary is not None:
            temporary.cleanup()


def _validate_tree(remote: Mapping[str, Any], expected_tree_paths: Iterable[str]) -> None:
    expected_paths = set(expected_tree_paths)
    missing = sorted(expected_paths - set(remote))
    extra = sorted(set(remote) - expected_paths)
    if missing:
        raise VerificationError(f"missing target paths: {missing}")
    if extra:
        raise VerificationError(f"unexpected target paths: {extra}")


def _verify_shared_blobs(
    remote: Mapping[str, Any],
    expected_shared_blobs: Mapping[str, str] | None,
) -> None:
    for path, expected_blob in (expected_shared_blobs or {}).items():
        actual_blob = getattr(remote[path], "blob_id", None)
        if actual_blob != expected_blob:
            raise VerificationError(f"shared path changed: {path}")


def _verify_expectations(
    api: HubApi,
    target_repo: str,
    revision: str,
    expectations: Iterable[ShardExpectation],
    directory: Path,
    http_client: StreamClient | None,
) -> dict[str, int]:
    rows_by_path: dict[str, int] = {}
    for expectation in expectations:
        local = download_to_temp(
            api,
            target_repo,
            expectation.path,
            revision,
            directory,
            client=http_client,
        )
        rows, schema = parquet_signature(local)
        if rows != expectation.rows or schema != expectation.schema:
            raise VerificationError(f"remote Parquet mismatch: {expectation.path}")
        rows_by_path[expectation.path] = rows
    return rows_by_path


def _verify_manifest(
    api: HubApi,
    target_repo: str,
    revision: str,
    expected_manifest: Mapping[str, Any] | None,
    directory: Path,
    http_client: StreamClient | None,
) -> Mapping[str, Any] | None:
    if expected_manifest is None:
        return None
    manifest_path = download_to_temp(
        api,
        target_repo,
        "eunis/manifest.json",
        revision,
        directory,
        client=http_client,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest != dict(expected_manifest):
        raise VerificationError("remote EUNIS manifest does not match expected manifest")
    return manifest


def _verify_artifacts(
    api: HubApi,
    target_repo: str,
    revision: str,
    expected_artifacts: Mapping[str, str] | None,
    directory: Path,
    http_client: StreamClient | None,
) -> None:
    for path, expected_hash in (expected_artifacts or {}).items():
        local = download_to_temp(
            api,
            target_repo,
            path,
            revision,
            directory,
            client=http_client,
        )
        actual_hash = sha256_file(local)
        if actual_hash != expected_hash:
            raise VerificationError(f"remote artifact mismatch: {path}")
