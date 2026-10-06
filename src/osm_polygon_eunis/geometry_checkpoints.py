# ruff: noqa: D100
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

from .eea import EeaGroup
from .geometry import OVERLAP_KERNEL_VERSION
from .release_plan import DatasetPlan


def _reference_signature(
    checksums: Mapping[str, str],
    threshold: int,
    groups: tuple[EeaGroup, ...] = (),
) -> str:
    payload = json.dumps(
        {
            "checksums": dict(sorted(checksums.items())),
            "threshold": threshold,
            "kernel": OVERLAP_KERNEL_VERSION,
            "reference_groups": [_reference_group_signature(group) for group in groups],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _reference_group_signature(group: EeaGroup) -> dict[str, object]:
    assets = group.raster_assets
    if group.vector_asset is not None:
        assets += (group.vector_asset,)
    return {
        "record_id": group.record_id,
        "labels": dict(sorted(group.labels.items())),
        "assets": [
            {
                "path": asset.path,
                "record_id": asset.record_id,
                "source_version": asset.source_version,
                "code": asset.code,
                "name": asset.name,
            }
            for asset in assets
        ],
    }


def _geometry_checkpoint_signature(
    reference_signature: str,
    plan: DatasetPlan,
    source_path: str,
) -> str:
    payload = json.dumps(
        {
            "references": reference_signature,
            "dataset": plan.spec.name,
            "source_repo": plan.spec.source_repo,
            "source_revision": plan.source_revision,
            "source_path": source_path,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _checkpoint_path(sidecar: Path) -> Path:
    return sidecar.with_name(f"{sidecar.name}.done")


def _completed_batches(sidecar: Path, signature: str) -> set[int]:
    payload = _read_checkpoint(_checkpoint_path(sidecar))
    return _checkpoint_batches(payload, signature)


def _read_checkpoint(path: Path) -> object | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _checkpoint_batches(payload: object, signature: str) -> set[int]:
    if not isinstance(payload, Mapping):
        return set()
    if payload.get("signature") != signature:
        return set()
    return _integer_batch_ids(payload.get("batches"))


def _integer_batch_ids(value: object) -> set[int]:
    if not isinstance(value, list):
        return set()
    return {item for item in value if type(item) is int}


def _record_completed_batch(sidecar: Path, signature: str, completed: set[int]) -> None:
    path = _checkpoint_path(sidecar)
    temporary = path.with_name(f"{path.name}.tmp")
    try:
        temporary.write_text(
            json.dumps({"signature": signature, "batches": sorted(completed)}),
            encoding="utf-8",
        )
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
