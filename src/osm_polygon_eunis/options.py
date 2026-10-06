"""Immutable release and worker configuration values."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

DEFAULT_BATCH_SIZE = 256
DEFAULT_WORKERS = 8
DEFAULT_WORKDIR = ".eunis-run"


def resolve_workdir() -> Path:
    """Return the staging directory from OSM_EUNIS_WORKDIR or the default."""
    return Path(os.environ.get("OSM_EUNIS_WORKDIR") or DEFAULT_WORKDIR)


def resolve_sidecar_root(workdir: Path | None = None) -> Path:
    """Return the sidecar directory from EUNIS_SIDECAR_DIR or under the workdir."""
    configured = os.environ.get("EUNIS_SIDECAR_DIR")
    if configured:
        return Path(configured)
    return (workdir if workdir is not None else resolve_workdir()) / "sidecars"


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """Bound memory and concurrency for release and geometry processing."""

    workers: int = DEFAULT_WORKERS
    parquet_batch_size: int = DEFAULT_BATCH_SIZE
    retained_source_shards_per_worker: int = 128
    raster_groups_per_batch: int = 2
    geometry_tasks_per_worker: int = 4

    def __post_init__(self) -> None:
        """Reject resource limits that could disable or stall processing."""
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class ReleaseOptions:
    """Settings that define one release invocation."""

    reference_config: Path
    workdir: Path
    limits: BatchLimits = field(default_factory=BatchLimits)
    token: str | None = None
    progress: Callable[[Mapping[str, object]], None] | None = None
    datasets: Sequence[str] | None = None
    max_intersection_errors: int | None = None


@dataclass(frozen=True, slots=True)
class ShardContext[ApiT, PlanT, OptionsT, ClientT]:
    """API, plan, settings and HTTP client shared while processing one shard."""

    api: ApiT
    plan: PlanT
    options: OptionsT
    http_client: ClientT


@dataclass(frozen=True, slots=True)
class GeometryPathOptions:
    """Local staging and batch settings for one geometry source pass."""

    sidecar_root: Path
    source_root: Path
    limits: BatchLimits = field(default_factory=BatchLimits)
    progress: Callable[[Mapping[str, object]], None] | None = None
    http_client: Any = None
    retain_source: bool = False
    reset_sidecar: bool = False
