"""Narrow Hugging Face publication and independent verification operations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pyarrow.parquet as pq
from huggingface_hub import duplicate_repo
from huggingface_hub.utils import RepositoryNotFoundError

from .sources import capture_revision, download_to_temp


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


def duplicate_source(
    api: Any,
    source_repo: str,
    target_repo: str,
    *,
    token: str | None = None,
    duplicate: Callable[..., Any] = duplicate_repo,
) -> bool:
    """Duplicate a dataset server-side only when the target does not exist."""

    try:
        api.repo_info(target_repo, repo_type="dataset")
    except RepositoryNotFoundError:
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
    api: Any,
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
    api: Any,
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


def build_manifest(
    *,
    source_repo: str,
    target_repo: str,
    source_revision: str,
    source_paths: Iterable[str],
    changed_paths: Iterable[str],
    reference_manifest: Mapping[str, Any],
    rows_by_path: Mapping[str, int],
    schema_by_path: Mapping[str, str] | None = None,
    added_paths: Iterable[str] = (),
    card_manifest: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a stable manifest that makes source conservation explicit."""

    source = sorted(set(source_paths))
    changed = sorted(set(changed_paths))
    added = sorted(set(added_paths))
    schemas = schema_by_path or {}
    _validate_manifest_paths(source, changed, added)
    return {
        "manifest_version": 3,
        "source_repo": source_repo,
        "target_repo": target_repo,
        "source_revision": source_revision,
        "source_paths": source,
        "changed_paths": changed,
        "added_paths": added,
        "shared_paths": sorted(set(source) - set(changed)),
        "rows_by_path": {path: rows_by_path[path] for path in sorted(rows_by_path)},
        "schema_by_path": {
            path: schemas[path]
            for path in sorted(schemas)
        },
        "reference": dict(reference_manifest),
        "card": dict(card_manifest) if card_manifest is not None else None,
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
    if added_paths & source_paths:
        raise ValueError("added paths already exist in source paths")
    if added_paths & changed_paths:
        raise ValueError("a path cannot be both changed and added")


def _remote_files(api: Any, repo_id: str, revision: str) -> dict[str, Any]:
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
    api: Any,
    target_repo: str,
    *,
    expectations: Iterable[ShardExpectation],
    expected_tree_paths: Iterable[str],
    expected_shared_blobs: Mapping[str, str] | None = None,
    expected_manifest: Mapping[str, Any] | None = None,
    expected_artifacts: Mapping[str, str] | None = None,
    temp_dir: Path | None = None,
    http_client: Any | None = None,
) -> VerificationReceipt:
    """Verify target tree, unchanged blob identities, Parquet schemas, and manifest."""

    revision = capture_revision(api, target_repo)
    remote = _remote_files(api, target_repo, revision)
    _validate_tree(remote, expected_tree_paths)
    _verify_shared_blobs(remote, expected_shared_blobs)

    if temp_dir is None:
        temporary = TemporaryDirectory(prefix="osm-polygon-eunis-verify-")
        directory = Path(temporary.name)
    else:
        temporary = None
        directory = temp_dir
    rows_by_path: dict[str, int] = {}
    try:
        rows_by_path = _verify_expectations(
            api,
            target_repo,
            revision,
            expectations,
            directory,
            http_client,
        )
        manifest = _verify_manifest(
            api,
            target_repo,
            revision,
            expected_manifest,
            directory,
            http_client,
        )
        _verify_artifacts(
            api,
            target_repo,
            revision,
            expected_artifacts,
            directory,
            http_client,
        )
        shared_paths = tuple(sorted(expected_shared_blobs or {}))
        return VerificationReceipt(target_repo, revision, rows_by_path, shared_paths, manifest)
    finally:
        if temporary is not None:
            temporary.cleanup()


def _validate_tree(remote: Mapping[str, Any], expected_tree_paths: Iterable[str]) -> None:
    expected_paths = set(expected_tree_paths)
    missing = sorted(expected_paths - set(remote))
    extra = sorted(set(remote) - expected_paths)
    if missing:
        raise ValueError(f"missing target paths: {missing}")
    if extra:
        raise ValueError(f"unexpected target paths: {extra}")


def _verify_shared_blobs(
    remote: Mapping[str, Any],
    expected_shared_blobs: Mapping[str, str] | None,
) -> None:
    for path, expected_blob in (expected_shared_blobs or {}).items():
        actual_blob = getattr(remote[path], "blob_id", None)
        if actual_blob != expected_blob:
            raise ValueError(f"shared path changed: {path}")


def _verify_expectations(
    api: Any,
    target_repo: str,
    revision: str,
    expectations: Iterable[ShardExpectation],
    directory: Path,
    http_client: Any | None,
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
            raise ValueError(f"remote Parquet mismatch: {expectation.path}")
        rows_by_path[expectation.path] = rows
    return rows_by_path


def _verify_manifest(
    api: Any,
    target_repo: str,
    revision: str,
    expected_manifest: Mapping[str, Any] | None,
    directory: Path,
    http_client: Any | None,
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
        raise ValueError("remote EUNIS manifest does not match expected manifest")
    return manifest


def _verify_artifacts(
    api: Any,
    target_repo: str,
    revision: str,
    expected_artifacts: Mapping[str, str] | None,
    directory: Path,
    http_client: Any | None,
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
        actual_hash = _sha256_file(local)
        if actual_hash != expected_hash:
            raise ValueError(f"remote artifact mismatch: {path}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
