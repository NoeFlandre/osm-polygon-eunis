"""Pinned dataset plans and release receipts shared by the runner modules."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ._protocols import HubApi
from .publish import (
    ShardExpectation,
    VerificationReceipt,
)
from .sources import (
    DatasetSpec,
    capture_revision,
    dataset_spec,
    flatten_repo_path,
    list_repo_files,
    matches_layout,
    pair_region_paths,
)

Progress = Callable[[Mapping[str, object]], None]


@dataclass(frozen=True, slots=True)
class DatasetPlan:
    """Pinned source layout for one dataset release."""

    spec: DatasetSpec
    source_revision: str
    source_files: tuple[str, ...]
    geometry_paths: tuple[str, ...]
    link_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DatasetReceipt:
    """Published and independently verified result for one dataset."""

    plan: DatasetPlan
    expectations: tuple[ShardExpectation, ...]
    verification: VerificationReceipt
    no_op: bool = False


@dataclass(frozen=True, slots=True)
class ReleaseReceipt:
    """Complete release receipt returned by the runner."""

    datasets: tuple[DatasetReceipt, ...]
    reference: Mapping[str, object]


def _sidecar_path(root: Path, spec: DatasetSpec, source_path: str) -> Path:
    return root / spec.name / f"{source_path.replace('/', '__')}.labels.parquet"


def _cached_geometry_path(root: Path, plan: DatasetPlan, source_path: str) -> Path:
    return root / plan.spec.name / flatten_repo_path(source_path)


def plan_datasets(api: HubApi) -> tuple[DatasetPlan, ...]:
    """Capture source commits and validate all declared dataset layouts."""

    return tuple(_plan_dataset(api, name) for name in ("website", "wikidata", "description"))


def _plan_dataset(api: HubApi, name: str) -> DatasetPlan:
    spec = dataset_spec(name)
    revision = capture_revision(api, spec.source_repo)
    source_files = tuple(entry.path for entry in list_repo_files(api, spec.source_repo, revision))
    geometry_paths, link_paths = _layout_paths(spec, source_files)
    return DatasetPlan(spec, revision, source_files, geometry_paths, link_paths)


def _layout_paths(
    spec: DatasetSpec,
    source_files: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    geometry_paths = _geometry_paths(spec, source_files)
    link_paths = _link_paths(spec, source_files)
    if spec.link_glob is not None:
        pair_region_paths(geometry_paths, link_paths)
    return geometry_paths, link_paths


def _geometry_paths(spec: DatasetSpec, source_files: tuple[str, ...]) -> tuple[str, ...]:
    paths = tuple(sorted(path for path in source_files if matches_layout(path, spec.geometry_glob)))
    if not paths:
        raise ValueError(f"source has no geometry shards for {spec.name}")
    return paths


def _link_paths(spec: DatasetSpec, source_files: tuple[str, ...]) -> tuple[str, ...]:
    if spec.link_glob is None:
        return ()
    return tuple(sorted(path for path in source_files if matches_layout(path, spec.link_glob)))
