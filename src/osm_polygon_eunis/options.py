"""Immutable release and worker configuration values."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class BatchLimits:
    """Bound memory and concurrency for release and geometry processing."""

    workers: int = 8
    parquet_batch_size: int = 256
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
