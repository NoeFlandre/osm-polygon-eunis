"""Checksum-verified reuse of staged EEA reference assets."""

from __future__ import annotations

import json
from pathlib import Path

from ._protocols import StreamClient
from .eea import RemoteAsset, download_asset
from .fileio import sha256_file


def _asset_cache_identity(asset: RemoteAsset) -> dict[str, object]:
    return {
        "etag": asset.etag,
        "size": asset.size,
        "source_version": asset.source_version,
        "url": asset.url,
    }


def _asset_cache_receipt_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.stage.json")


def _read_staged_receipt(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return receipt if isinstance(receipt, dict) else None


def _expected_staged_digest(
    asset: RemoteAsset,
    receipt: dict[str, object],
) -> str | None:
    if receipt.get("asset") != _asset_cache_identity(asset):
        return None
    digest = receipt.get("sha256")
    return digest if isinstance(digest, str) else None


def _staged_file_has_size(path: Path, expected_size: int) -> bool:
    return path.is_file() and path.stat().st_size == expected_size


def _staged_asset_digest(asset: RemoteAsset, path: Path) -> str | None:
    """Revalidate a staged file against its metadata and recorded SHA-256."""

    receipt_path = _asset_cache_receipt_path(path)
    if not _staged_file_has_size(path, asset.size):
        return None
    receipt = _read_staged_receipt(receipt_path)
    if receipt is None:
        return None
    expected_digest = _expected_staged_digest(asset, receipt)
    if expected_digest is None:
        return None
    actual_digest = sha256_file(path)
    return actual_digest if actual_digest == expected_digest else None


def _write_staged_asset_receipt(asset: RemoteAsset, path: Path, digest: str) -> None:
    receipt_path = _asset_cache_receipt_path(path)
    temporary_path = receipt_path.with_name(f"{receipt_path.name}.tmp")
    payload = {"asset": _asset_cache_identity(asset), "sha256": digest}
    temporary_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    temporary_path.replace(receipt_path)


def stage_or_verify_asset(client: StreamClient, asset: RemoteAsset, path: Path) -> str:
    """Return a verified staged digest, downloading the asset when necessary."""

    digest = _staged_asset_digest(asset, path)
    if digest is not None:
        return digest
    digest = download_asset(client, asset, path)
    _write_staged_asset_receipt(asset, path, digest)
    return digest
