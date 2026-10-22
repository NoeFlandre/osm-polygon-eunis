"""Bounded, resumable shard controller for WorldCover and LUCAS pool attempts.

This controller stores raw rows once and replays that immutable checkpoint into
independent map adapters. A shard receipt is a task checkpoint, never a pool
completion manifest. Final pool export remains the shared contract exporter's
responsibility after the complete source keyset and all required shards pass.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib
import json
import os
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlparse

from osm_polygon_eunis.pool_adapters.source_pins import (
    RawSourceTreePin,
    SourceTreeFile,
    polygon_geometry_tree_pins,
)
from osm_polygon_eunis.pool_contract import TextSource

from .lucas import LUCAS_AREA_MEASUREMENT_ID
from .streaming import persist_measurement_sidecars
from .worldcover import WORLDCOVER_AREA_MEASUREMENT

_FORMAT_VERSION = "bounded-pool-controller-v1"
_CENSUS_FORMAT_VERSION = "raw-inventory-keyset-v1"
_MAPS = frozenset({"worldcover", "lucas"})
_SOURCES = frozenset({source.value for source in TextSource})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_BLOB_RE = re.compile(r"^[0-9a-f]{40}$")
_MAX_ROW_BYTES = 2 * 1024 * 1024
_NATIVE_KEY_PAIR_SIZE = 2
_POOL_THRESHOLD = 0.8
_WORLDCOVER_V4_PRODUCER_REVISION = "18c75517c74be22ab2f90edce2b65d3600461048"


class AdapterWorker(Protocol):
    """Consume the same raw rows and yield one serialized adapter record per row."""

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        """Return complete attempt, outcome, candidate and evidence records."""


@dataclass(frozen=True, slots=True)
class ShardCensus:
    """Independent row/key receipt for one shard in a full-source census."""

    path: str
    size_bytes: int
    blob_id: str
    lfs_oid: str | None
    row_count: int
    ordered_key_sha256: str


@dataclass(frozen=True, slots=True)
class RawInventoryCensus:
    """Verified complete source manifest and canonical full-source keyset receipt."""

    text_source: str
    repository: str
    revision: str
    path_prefix: str
    tree_inventory_sha256: str
    shard_count: int
    raw_row_count: int
    unique_key_count: int
    unique_keyset_sha256: str
    shards: tuple[ShardCensus, ...]
    manifest_sha256: str

    @classmethod
    def load(
        cls,
        path: Path,
        expected_sha256: str,
        source_pin: RawSourceTreePin,
    ) -> RawInventoryCensus:
        """Load a caller-pinned census and bind it to the checked-in full tree."""

        value, observed_sha256 = _read_pinned_census(path, expected_sha256)
        census = _census_from_document(cls, value, observed_sha256)
        census.validate(source_pin)
        return census

    def validate(self, source_pin: RawSourceTreePin) -> None:
        """Require exact full-tree coverage and reconcile all counts."""

        _validate_census_identity(self, source_pin)
        observed = _validate_census_shards(self, source_pin)
        _validate_census_totals(self, source_pin, observed)

    def shard(self, path: str) -> ShardCensus:
        """Return one exact path from the complete census."""

        for row in self.shards:
            if row.path == path:
                return row
        raise ValueError(f"selected shard {path!r} is not in the complete source census")

    @property
    def source_snapshot_id(self) -> str:
        """Return the stable source/tree identity used for native row IDs."""

        payload = f"{self.repository}\n{self.revision}\n{self.tree_inventory_sha256}".encode()
        return hashlib.sha256(payload).hexdigest()


def _read_pinned_census(path: Path, expected_sha256: str) -> tuple[dict[str, Any], str]:
    _require_sha256(expected_sha256, "inventory manifest pin")
    raw = path.read_bytes()
    observed_sha256 = hashlib.sha256(raw).hexdigest()
    if observed_sha256 != expected_sha256:
        raise ValueError("raw inventory manifest bytes do not match the supplied SHA-256 pin")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("raw inventory manifest is not valid JSON") from error
    if not isinstance(value, dict) or value.get("format_version") != _CENSUS_FORMAT_VERSION:
        raise ValueError("raw inventory manifest format is unsupported")
    return value, observed_sha256


def _census_from_document(
    census_type: type[RawInventoryCensus],
    value: Mapping[str, Any],
    manifest_sha256: str,
) -> RawInventoryCensus:
    shards_value = value.get("shards")
    if not isinstance(shards_value, list):
        raise TypeError("raw inventory manifest must include the complete shard list")
    shards = tuple(_decode_shard_census(row) for row in shards_value)
    return census_type(
        text_source=str(value.get("text_source", "")),
        repository=str(value.get("repository", "")),
        revision=str(value.get("revision", "")),
        path_prefix=str(value.get("path_prefix", "")),
        tree_inventory_sha256=str(value.get("tree_inventory_sha256", "")),
        shard_count=_non_negative_int(value.get("shard_count"), "shard_count"),
        raw_row_count=_non_negative_int(value.get("raw_row_count"), "raw_row_count"),
        unique_key_count=_non_negative_int(value.get("unique_key_count"), "unique_key_count"),
        unique_keyset_sha256=str(value.get("unique_keyset_sha256", "")),
        shards=shards,
        manifest_sha256=manifest_sha256,
    )


def _validate_census_identity(census: RawInventoryCensus, source_pin: RawSourceTreePin) -> None:
    if census.text_source != source_pin.text_source.value:
        raise ValueError("raw inventory manifest source does not match the selected source")
    if (census.repository, census.revision, census.path_prefix) != (
        source_pin.repository,
        source_pin.revision,
        source_pin.path_prefix,
    ):
        raise ValueError("raw inventory manifest repository, revision, or tree differs")
    if census.tree_inventory_sha256 != source_pin.inventory_sha256:
        raise ValueError("raw inventory manifest tree digest differs from the public source pin")
    _require_sha256(census.manifest_sha256, "raw inventory manifest SHA-256")
    _require_sha256(census.unique_keyset_sha256, "full-source keyset digest")


def _validate_census_shards(
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
) -> dict[str, ShardCensus]:
    expected = {row.path: row for row in source_pin.files}
    observed = {row.path: row for row in census.shards}
    _validate_census_shard_set(census, source_pin, expected, observed)
    _validate_census_shard_rows(observed, expected)
    return observed


def _validate_census_shard_set(
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
    expected: Mapping[str, SourceTreeFile],
    observed: Mapping[str, ShardCensus],
) -> None:
    _validate_census_shard_coverage(census.shards, expected, observed)
    _validate_census_shard_count(census, source_pin)
    _validate_census_source_bytes(source_pin, census.shards)


def _validate_census_shard_coverage(
    shards: tuple[ShardCensus, ...],
    expected: Mapping[str, SourceTreeFile],
    observed: Mapping[str, ShardCensus],
) -> None:
    if len(observed) != len(shards) or set(observed) != set(expected):
        raise ValueError("raw inventory manifest does not cover the complete pinned shard set")


def _validate_census_shard_count(census: RawInventoryCensus, source_pin: RawSourceTreePin) -> None:
    if census.shard_count != source_pin.file_count or census.shard_count != len(census.shards):
        raise ValueError("raw inventory manifest shard count differs from the complete source tree")


def _validate_census_source_bytes(
    source_pin: RawSourceTreePin, shards: tuple[ShardCensus, ...]
) -> None:
    if sum(row.size_bytes for row in shards) != source_pin.total_size_bytes:
        raise ValueError("raw inventory manifest shard sizes differ from the pinned source tree")


def _validate_census_shard_rows(
    observed: Mapping[str, ShardCensus],
    expected: Mapping[str, SourceTreeFile],
) -> None:
    for path, shard in observed.items():
        file_pin = expected[path]
        _validate_shard_file_pin(path, shard, file_pin)
        _require_sha256(shard.ordered_key_sha256, f"ordered key digest for {path}")


def _validate_shard_file_pin(path: str, shard: ShardCensus, file_pin: SourceTreeFile) -> None:
    if (shard.size_bytes, shard.blob_id, shard.lfs_oid) != (
        file_pin.size_bytes,
        file_pin.blob_id,
        file_pin.lfs_oid,
    ):
        raise ValueError(f"raw inventory shard pin differs for {path}")


def _validate_census_totals(
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
    observed: Mapping[str, ShardCensus],
) -> None:
    if sum(row.row_count for row in observed.values()) != census.raw_row_count:
        raise ValueError("raw inventory manifest raw row count does not reconcile")
    if census.unique_key_count != census.raw_row_count:
        raise ValueError("raw inventory manifest contains duplicate or missing native keys")
    if census.shard_count != source_pin.file_count:
        raise ValueError("raw inventory manifest omits one or more pinned source shards")


@dataclass(frozen=True, slots=True)
class MapPin:
    """Immutable map implementation and reference identities for one task."""

    map_name: str
    reference_manifest_sha256: str
    taxonomy_sha256: str
    area_method_id: str
    adapter_revision: str
    builder_revision: str
    resource_limits_sha256: str
    support_union_policy: str
    threshold: float = 0.8
    producer_revision: str | None = None

    def __post_init__(self) -> None:
        """Require stable map-specific reference and implementation pins."""

        _validate_map_pin_name(self.map_name)
        _validate_map_pin_hashes(self)
        _validate_map_pin_policy(self)
        _validate_map_producer_revision(self)


def _validate_map_pin_name(map_name: str) -> None:
    if map_name not in _MAPS:
        raise ValueError("pool controller supports only WorldCover and LUCAS")


def _validate_map_pin_hashes(pin: MapPin) -> None:
    for name, value in (
        ("reference manifest", pin.reference_manifest_sha256),
        ("taxonomy", pin.taxonomy_sha256),
        ("adapter", pin.adapter_revision),
        ("builder", pin.builder_revision),
        ("resource limits", pin.resource_limits_sha256),
    ):
        _require_sha256(value, f"{pin.map_name} {name} SHA-256")
    if not pin.area_method_id.strip():
        raise ValueError("map area method identity is required")


def _validate_map_pin_policy(pin: MapPin) -> None:
    expected = {
        "worldcover": "union_by_class",
        "lucas": "independent_union_per_level_and_class:v1",
    }[pin.map_name]
    expected_area = {
        "worldcover": WORLDCOVER_AREA_MEASUREMENT,
        "lucas": LUCAS_AREA_MEASUREMENT_ID,
    }[pin.map_name]
    if (
        pin.support_union_policy != expected
        or pin.threshold != _POOL_THRESHOLD
        or pin.area_method_id != expected_area
    ):
        raise ValueError("pool controller preserves the canonical inclusive 80% map policy")


def _validate_map_producer_revision(pin: MapPin) -> None:
    if pin.map_name == "worldcover":
        if pin.producer_revision != _WORLDCOVER_V4_PRODUCER_REVISION:
            raise ValueError("WorldCover requires the pinned v4 producer Git revision")
    elif pin.producer_revision is not None:
        raise ValueError("LUCAS does not accept a WorldCover producer revision")


@dataclass(frozen=True, slots=True)
class PoolControllerRequest:
    """Explicit source, map and shard selector with a hard output budget."""

    text_source: str
    shard_paths: tuple[str, ...]
    maps: tuple[str, ...]
    max_workspace_bytes: int
    max_shards_per_invocation: int = 1
    max_row_bytes: int = _MAX_ROW_BYTES

    def __post_init__(self) -> None:
        """Require bounded source/map selectors and the fixed 80-percent rule."""

        _validate_request_source(self.text_source)
        _validate_request_shards(self.shard_paths)
        _validate_request_maps(self.maps)
        _validate_request_limits(
            self.max_workspace_bytes,
            self.max_shards_per_invocation,
            self.max_row_bytes,
        )


def _validate_request_source(text_source: str) -> None:
    if text_source not in _SOURCES:
        raise ValueError(f"unsupported text source {text_source!r}")


def _validate_request_shards(shard_paths: tuple[str, ...]) -> None:
    if not shard_paths or len(set(shard_paths)) != len(shard_paths):
        raise ValueError("select one or more unique frozen source shard paths")


def _validate_request_maps(maps: tuple[str, ...]) -> None:
    if not maps or len(set(maps)) != len(maps):
        raise ValueError("select one or more unique label maps")
    if not set(maps).issubset(_MAPS):
        raise ValueError("EUNIS and unknown maps cannot block or enter this controller")


def _validate_request_limits(
    max_workspace_bytes: int,
    max_shards_per_invocation: int,
    max_row_bytes: int,
) -> None:
    if max_workspace_bytes <= 0 or max_shards_per_invocation <= 0:
        raise ValueError("workspace quota and per-invocation shard cap must be positive")
    if not 0 < max_row_bytes <= _MAX_ROW_BYTES:
        raise ValueError(f"raw or attempt records may not exceed {_MAX_ROW_BYTES} bytes")


@dataclass(frozen=True, slots=True)
class PoolControllerPlan:
    """Resolved bounded work, carrying no claim that any pool is complete."""

    text_source: str
    repository: str
    revision: str
    census_sha256: str
    selected_shards: tuple[str, ...]
    selected_maps: tuple[str, ...]
    omitted_shard_count: int  # noqa: V107 - emitted as the plan JSON field
    available_shards: tuple[str, ...]  # noqa: V107 - emitted as the plan JSON field
    missing_shards: tuple[str, ...]  # noqa: V107 - emitted as the plan JSON field
    expected_input_bytes: int  # noqa: V107 - emitted as the plan JSON field
    max_workspace_bytes: int
    scope: str = "selected-shards"
    complete_pool: bool = False  # noqa: V107 - explicit safety field in the plan JSON
    evaluation_started: bool = False  # noqa: V107 - explicit safety field in the plan JSON


@dataclass(frozen=True, slots=True)
class PoolControllerReceipt:
    """Immutable checkpoint status for one selected source-shard unit."""

    plan: PoolControllerPlan
    completed_maps: tuple[str, ...]  # noqa: V107 - emitted as the receipt JSON field
    reused_maps: tuple[str, ...]  # noqa: V107 - emitted as the receipt JSON field
    raw_row_count: int
    raw_checkpoint_sha256: str
    map_checkpoint_sha256: tuple[tuple[str, str], ...]  # noqa: V107 - emitted as receipt JSON
    workspace_bytes: int  # noqa: V107 - emitted as the receipt JSON field
    status: str


class LocalControllerArtifactSink:
    """Immutable local sink for large adapter evidence sidecars under the quota."""

    def __init__(self, artifact_root: Path, workspace: Path, max_workspace_bytes: int) -> None:
        """Set the immutable artifact root and total workspace byte cap."""

        self.artifact_root = artifact_root
        self.workspace = workspace
        self.max_workspace_bytes = max_workspace_bytes

    def put(self, relative_path: str, content: bytes) -> str:
        """Persist a small immutable adapter sidecar or return an identical retry."""

        target = self._target(relative_path)
        if target.exists():
            if target.read_bytes() != content:
                raise ValueError(f"refusing to replace immutable evidence artifact {target}")
            return target.resolve().as_uri()
        _ensure_workspace_quota(self.workspace, self.max_workspace_bytes, len(content))
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        try:
            with temporary.open("wb") as output:
                output.write(content)
                output.flush()
                os.fsync(output.fileno())
            temporary.chmod(0o444)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target.resolve().as_uri()

    def put_file(self, relative_path: str, source_path: Path) -> str:
        """Copy and hash a sidecar before its temporary adapter source is removed."""

        target = self._target(relative_path)
        expected_sha = _sidecar_sha_from_filename(target)
        if target.exists():
            return _verify_existing_sidecar(target, expected_sha)
        expected_size = source_path.stat().st_size
        _ensure_workspace_quota(self.workspace, self.max_workspace_bytes, expected_size)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(f".{target.name}.partial")
        try:
            _copy_hash_checked_sidecar(source_path, temporary, expected_size, expected_sha)
            temporary.chmod(0o444)
            temporary.replace(target)
        except Exception:
            temporary.unlink(missing_ok=True)
            raise
        return target.resolve().as_uri()

    def _target(self, relative_path: str) -> Path:
        candidate = Path(relative_path)
        if candidate.is_absolute() or ".." in candidate.parts or not relative_path:
            raise ValueError("evidence artifact path must be safe and relative")
        return self.artifact_root / candidate


def canonical_adapter_checkpoint_rows(
    labeled_polygons: Iterable[Any],
    sink: LocalControllerArtifactSink,
    pool_name: str,
) -> Iterator[dict[str, Any]]:
    """Serialize existing adapter records after durable sidecar persistence.

    ``labeled_polygons`` is the pair iterator returned by
    ``iter_worldcover_source_rows`` or ``iter_lucas_rows``. The helper preserves
    the shared adapter's 80-percent decisions and binds its attempt and support
    evidence to immutable local artifacts before the reader advances.
    """

    for polygon, record in persist_measurement_sidecars(labeled_polygons, sink, pool_name):
        yield {
            "inventory_row_id": record.inventory_row_id,
            "inventory": asdict(polygon.inventory),
            "polygon": asdict(polygon.snapshot) if polygon.snapshot is not None else None,
            "adapter_record": asdict(record),
        }


def _sidecar_sha_from_filename(target: Path) -> str:
    match = re.search(r"([0-9a-f]{64})\.jsonl$", target.name)
    if match is None:
        raise ValueError("measurement sidecar filename must carry its expected SHA-256")
    return match.group(1)


def _verify_existing_sidecar(target: Path, expected_sha: str) -> str:
    if _sha256_file(target) != expected_sha:
        raise ValueError(f"stored evidence sidecar changed: {target}")
    return target.resolve().as_uri()


def _copy_hash_checked_sidecar(
    source_path: Path,
    temporary: Path,
    expected_size: int,
    expected_sha: str,
) -> None:
    digest = hashlib.sha256()
    written = 0
    with source_path.open("rb") as source, temporary.open("wb") as output:
        while chunk := source.read(1024 * 1024):
            written = _copy_sidecar_chunk(chunk, written, expected_size, digest, output)
        if written != expected_size:
            raise ValueError("measurement sidecar size changed while it was being copied")
        output.flush()
        os.fsync(output.fileno())
    if digest.hexdigest() != expected_sha:
        raise ValueError("measurement sidecar bytes differ from the adapter receipt")


def _copy_sidecar_chunk(
    chunk: bytes,
    written: int,
    expected_size: int,
    digest: Any,
    output: Any,
) -> int:
    written += len(chunk)
    if written > expected_size:
        raise ValueError("measurement sidecar grew while it was being copied")
    digest.update(chunk)
    output.write(chunk)
    return written


RawRowFactory = Callable[[Path], Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ShardRunInputs:
    """Verified census, local roots and source-specific adapter functions."""

    census: RawInventoryCensus
    source_pin: RawSourceTreePin
    input_root: Path
    workspace: Path
    map_pins: Mapping[str, MapPin]
    map_workers: Mapping[str, AdapterWorker]
    raw_row_factory: RawRowFactory


@dataclass(frozen=True, slots=True)
class RawCheckpointContext:
    """Inputs to one atomic, resumable raw-row checkpoint."""

    request: PoolControllerRequest
    census_row: ShardCensus
    staged_path: Path
    source_sha256: str
    source_snapshot_id: str
    raw_row_factory: RawRowFactory
    shard_scope: Path
    workspace: Path


@dataclass(frozen=True, slots=True)
class MapCheckpointContext:
    """Inputs to one per-map immutable attempt/evidence checkpoint."""

    map_name: str
    worker: AdapterWorker
    raw_path: Path
    census_row: ShardCensus
    signature: str
    shard_scope: Path
    workspace: Path
    max_workspace_bytes: int
    max_row_bytes: int


@dataclass(frozen=True, slots=True)
class MapTaskContext:
    """Common verified inputs to one map's resumable checkpoint."""

    request: PoolControllerRequest
    inputs: ShardRunInputs
    plan: PoolControllerPlan
    source_row: ShardCensus
    source_sha256: str
    raw_sha256: str
    raw_path: Path
    shard_scope: Path


@dataclass(slots=True)
class SourceCopyContext:
    """Mutable streaming hash state for one staged source file."""

    expected_size: int
    git_digest: Any
    content_digest: Any
    output: Any
    workspace: Path
    max_workspace_bytes: int
    total: int = 0


def plan_run(
    request: PoolControllerRequest,
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
    input_root: Path,
) -> PoolControllerPlan:
    """Resolve exact source inputs without reading geometry or invoking maps."""

    census.validate(source_pin)
    _validate_source_choice(request, source_pin)
    selected_rows = _selected_census_rows(request, census)
    available = _available_selected_shards(selected_rows, source_pin, input_root)
    return PoolControllerPlan(
        text_source=request.text_source,
        repository=census.repository,
        revision=census.revision,
        census_sha256=census.manifest_sha256,
        selected_shards=tuple(row.path for row in selected_rows),
        selected_maps=tuple(sorted(request.maps)),
        omitted_shard_count=census.shard_count - len(selected_rows),
        available_shards=available,
        missing_shards=tuple(row.path for row in selected_rows if row.path not in set(available)),
        expected_input_bytes=_selected_input_size(selected_rows, source_pin),
        max_workspace_bytes=request.max_workspace_bytes,
    )


def _selected_census_rows(
    request: PoolControllerRequest,
    census: RawInventoryCensus,
) -> tuple[ShardCensus, ...]:
    rows = tuple(census.shard(path) for path in sorted(request.shard_paths))
    for row in rows:
        _safe_relative_path(row.path)
    return rows


def _available_selected_shards(
    selected_rows: tuple[ShardCensus, ...],
    source_pin: RawSourceTreePin,
    input_root: Path,
) -> tuple[str, ...]:
    return tuple(
        row.path
        for row in selected_rows
        if _local_source_size_matches(input_root, source_pin, row.path)
    )


def _local_source_size_matches(input_root: Path, source_pin: RawSourceTreePin, path: str) -> bool:
    source_path = input_root / path
    return source_path.is_file() and source_path.stat().st_size == _expected_local_size(
        _source_file(source_pin, path)
    )


def _selected_input_size(
    selected_rows: tuple[ShardCensus, ...], source_pin: RawSourceTreePin
) -> int:
    return sum(_expected_local_size(_source_file(source_pin, row.path)) for row in selected_rows)


def run_shard(request: PoolControllerRequest, inputs: ShardRunInputs) -> PoolControllerReceipt:
    """Stage one verified shard, checkpoint shared raw rows, then run selected maps.

    ``raw_row_factory`` reads the staged Parquet shard in bounded Arrow batches
    and yields rows with ``native_key_json``, ``inventory_row_id``,
    ``source_row_locator``, and ``geometry_payload`` fields. Each map worker
    must call the matching canonical adapter and yield one JSON-ready complete
    attempt/evidence record per raw row, including ``inventory_row_id``.
    """

    plan, source_row = _validate_run_request(request, inputs)
    workspace = inputs.workspace
    _ensure_workspace_quota(workspace, request.max_workspace_bytes, 0)
    file_pin = _source_file(inputs.source_pin, source_row.path)
    shard_scope = _task_root(workspace, request.text_source, source_row.path)
    shard_scope.mkdir(parents=True, exist_ok=True)
    lock_descriptor = _acquire_lock(shard_scope)
    try:
        return _run_locked_shard(
            request,
            inputs,
            plan,
            source_row,
            file_pin,
            shard_scope,
        )
    finally:
        _release_lock(lock_descriptor)


def _validate_run_request(
    request: PoolControllerRequest,
    inputs: ShardRunInputs,
) -> tuple[PoolControllerPlan, ShardCensus]:
    if len(request.shard_paths) > request.max_shards_per_invocation:
        raise ValueError("selected shard count exceeds max_shards_per_invocation")
    plan = plan_run(request, inputs.census, inputs.source_pin, inputs.input_root)
    if len(plan.selected_shards) != 1:
        raise ValueError("run_shard processes exactly one selected shard per invocation")
    _validate_run_map_workers(request, inputs)
    return plan, inputs.census.shard(plan.selected_shards[0])


def _validate_run_map_workers(request: PoolControllerRequest, inputs: ShardRunInputs) -> None:
    if set(inputs.map_pins) != set(request.maps) or set(inputs.map_workers) != set(request.maps):
        raise ValueError("selected maps, map pins, and map workers must match exactly")
    for map_name, pin in inputs.map_pins.items():
        if pin.map_name != map_name:
            raise ValueError(f"map pin identity mismatch for {map_name}")


def _run_locked_shard(
    request: PoolControllerRequest,
    inputs: ShardRunInputs,
    plan: PoolControllerPlan,
    source_row: ShardCensus,
    file_pin: SourceTreeFile,
    shard_scope: Path,
) -> PoolControllerReceipt:
    workspace = inputs.workspace
    staged_path, actual_sha256 = _stage_source_file(
        inputs.input_root / source_row.path,
        file_pin,
        workspace / "inputs",
        request.max_workspace_bytes,
        workspace,
    )
    raw_path, raw_sha256 = _build_or_reuse_raw_checkpoint(
        RawCheckpointContext(
            request,
            source_row,
            staged_path,
            actual_sha256,
            inputs.census.source_snapshot_id,
            inputs.raw_row_factory,
            shard_scope,
            workspace,
        )
    )
    digests, completed, reused = _run_map_checkpoints(
        MapTaskContext(
            request,
            inputs,
            plan,
            source_row,
            actual_sha256,
            raw_sha256,
            raw_path,
            shard_scope,
        )
    )
    _ensure_workspace_quota(workspace, request.max_workspace_bytes, 0)
    return PoolControllerReceipt(
        plan=plan,
        completed_maps=completed,
        reused_maps=reused,
        raw_row_count=source_row.row_count,
        raw_checkpoint_sha256=raw_sha256,
        map_checkpoint_sha256=digests,
        workspace_bytes=_workspace_size(workspace),
        status="selected_shard_checkpoints_complete",
    )


def _run_map_checkpoints(
    context: MapTaskContext,
) -> tuple[tuple[tuple[str, str], ...], tuple[str, ...], tuple[str, ...]]:
    digests: list[tuple[str, str]] = []
    completed: list[str] = []
    reused: list[str] = []
    for map_name in context.plan.selected_maps:
        digest, was_reused = _run_one_map_checkpoint(context, map_name)
        digests.append((map_name, digest))
        (reused if was_reused else completed).append(map_name)
    return tuple(digests), tuple(completed), tuple(reused)


def _run_one_map_checkpoint(
    context: MapTaskContext,
    map_name: str,
) -> tuple[str, bool]:
    request = context.request
    inputs = context.inputs
    plan = context.plan
    source_row = context.source_row
    pin = inputs.map_pins[map_name]
    signature = _map_signature(
        plan,
        source_row,
        context.source_sha256,
        context.raw_sha256,
        pin,
        request.max_row_bytes,
    )
    receipt_path = context.shard_scope / map_name / signature / "receipt.json"
    reused_digest = _valid_map_receipt(
        receipt_path, signature, source_row.row_count, request.max_row_bytes
    )
    if reused_digest is not None:
        return reused_digest, True
    digest = _write_map_checkpoint(
        MapCheckpointContext(
            map_name,
            inputs.map_workers[map_name],
            context.raw_path,
            source_row,
            signature,
            context.shard_scope,
            inputs.workspace,
            request.max_workspace_bytes,
            request.max_row_bytes,
        )
    )
    return digest, False


def read_raw_checkpoint(path: Path) -> Iterator[dict[str, Any]]:
    """Replay the shared immutable raw rows one JSONL record at a time."""

    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid raw checkpoint JSON at {path}:{line_number}") from error
            if not isinstance(value, dict):
                raise TypeError(f"raw checkpoint row {line_number} is not an object")
            yield value


def _build_or_reuse_raw_checkpoint(
    context: RawCheckpointContext,
) -> tuple[Path, str]:
    signature = _raw_checkpoint_signature(context)
    raw_dir = context.shard_scope / "raw" / signature
    raw_path = raw_dir / "rows.jsonl"
    receipt_path = raw_dir / "receipt.json"
    old = _read_receipt(receipt_path)
    if receipt_path.exists() and old is None:
        raise ValueError("raw checkpoint receipt exists but is not valid JSON")
    if old is not None:
        if old.get("signature") != signature:
            raise ValueError("raw checkpoint receipt has a different semantic signature")
        raw_sha = _verify_checkpoint_file(raw_path, old, context.census_row.row_count)
        return raw_path, raw_sha

    return _create_raw_checkpoint(context, raw_path, receipt_path, signature)


def _raw_checkpoint_signature(context: RawCheckpointContext) -> str:
    request = context.request
    census_row = context.census_row
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "source": request.text_source,
                "path": census_row.path,
                "source_sha256": context.source_sha256,
                "source_ordered_key_sha256": census_row.ordered_key_sha256,
                "census_row_count": census_row.row_count,
                "format_version": _FORMAT_VERSION,
            }
        )
    ).hexdigest()


def _create_raw_checkpoint(
    context: RawCheckpointContext,
    raw_path: Path,
    receipt_path: Path,
    signature: str,
) -> tuple[Path, str]:
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = raw_path.with_name(".rows.jsonl.partial")
    try:
        row_count, key_digest = _write_raw_checkpoint_rows(context, temporary)
        _install_unreceipted_checkpoint(temporary, raw_path, "raw")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    raw_sha = _sha256_file(raw_path)
    _write_receipt(
        receipt_path,
        {
            "format_version": _FORMAT_VERSION,
            "signature": signature,
            "path": raw_path.name,
            "sha256": raw_sha,
            "size_bytes": raw_path.stat().st_size,
            "row_count": row_count,
            "ordered_key_sha256": key_digest,
        },
        context.workspace,
        context.request.max_workspace_bytes,
    )
    return raw_path, raw_sha


def _write_raw_checkpoint_rows(
    context: RawCheckpointContext,
    temporary: Path,
) -> tuple[int, str]:
    request = context.request
    census_row = context.census_row
    key_hash = hashlib.sha256()
    row_count = 0
    with temporary.open("wb") as output:
        for row in context.raw_row_factory(context.staged_path):
            record = _validate_raw_row(
                row,
                row_count,
                census_row.path,
                request.max_row_bytes,
                context.source_snapshot_id,
            )
            _update_key_digest(key_hash, str(record["native_key_json"]), row_count)
            _write_bounded_line(
                output,
                record,
                request.max_row_bytes,
                context.workspace,
                request.max_workspace_bytes,
            )
            row_count += 1
        output.flush()
        os.fsync(output.fileno())
    if row_count != census_row.row_count:
        raise ValueError("staged raw row count differs from the verified source census")
    if key_hash.hexdigest() != census_row.ordered_key_sha256:
        raise ValueError("staged raw native keys differ from the verified shard keyset")
    return row_count, key_hash.hexdigest()


def _update_key_digest(digest: Any, key_json: str, row_count: int) -> None:
    if row_count:
        digest.update(b"\n")
    digest.update(key_json.encode("utf-8"))


def _write_map_checkpoint(
    context: MapCheckpointContext,
) -> str:
    map_dir = context.shard_scope / context.map_name / context.signature
    map_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = map_dir / "attempts.jsonl"
    receipt_path = map_dir / "receipt.json"
    temporary = checkpoint_path.with_name(".attempts.jsonl.partial")
    try:
        row_count = _write_map_attempt_rows(context, temporary)
        _install_unreceipted_checkpoint(temporary, checkpoint_path, "map")
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    checkpoint_sha = _sha256_file(checkpoint_path)
    _validate_map_checkpoint_artifacts(checkpoint_path)
    _write_receipt(
        receipt_path,
        {
            "format_version": _FORMAT_VERSION,
            "signature": context.signature,
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": checkpoint_sha,
            "size_bytes": checkpoint_path.stat().st_size,
            "row_count": row_count,
            "pool_complete": False,
            "evaluation_started": False,
        },
        context.workspace,
        context.max_workspace_bytes,
    )
    return checkpoint_sha


def _write_map_attempt_rows(context: MapCheckpointContext, temporary: Path) -> int:
    raw_iter = read_raw_checkpoint(context.raw_path)
    expected = next(raw_iter, None)
    row_count = 0
    checkpoint_runner = getattr(context.worker, "run_checkpoint", None)
    results = (
        checkpoint_runner(context.raw_path)
        if callable(checkpoint_runner)
        else context.worker(read_raw_checkpoint(context.raw_path))
    )
    with temporary.open("wb") as output:
        for result in results:
            wrapper = _map_attempt_record(result, expected, context.map_name)
            expected = next(raw_iter, None)
            _write_bounded_line(
                output,
                wrapper,
                context.max_row_bytes,
                context.workspace,
                context.max_workspace_bytes,
            )
            row_count += 1
        if expected is not None:
            raise ValueError(f"{context.map_name} adapter omitted raw source rows")
        output.flush()
        os.fsync(output.fileno())
    if row_count != context.census_row.row_count:
        raise ValueError(f"{context.map_name} attempt count differs from the raw source census")
    return row_count


def _map_attempt_record(
    result: object,
    expected: Mapping[str, Any] | None,
    map_name: str,
) -> dict[str, Any]:
    if not isinstance(result, Mapping):
        raise TypeError(f"{map_name} adapter record must be a mapping")
    if expected is None:
        raise ValueError(f"{map_name} adapter emitted more rows than the raw shard")
    record = dict(result)
    if record.get("inventory_row_id") != expected["inventory_row_id"]:
        raise ValueError(
            f"{map_name} adapter row order or inventory identity differs from raw input"
        )
    return {
        "inventory_row_id": expected["inventory_row_id"],
        "native_key_json": expected["native_key_json"],
        "source_row_locator": expected["source_row_locator"],
        "map": map_name,
        "result": record,
    }


def _validate_raw_row(
    row: Mapping[str, Any],
    ordinal: int,
    shard_path: str,
    max_row_bytes: int,
    source_snapshot_id: str,
) -> dict[str, Any]:
    _require_raw_row_fields(row, ordinal)
    key_json = _validate_native_key_json(row["native_key_json"], ordinal)
    inventory_id = _validate_inventory_row_id(
        row["inventory_row_id"], key_json, ordinal, source_snapshot_id
    )
    _validate_source_row_locator(row["source_row_locator"], ordinal, shard_path)
    record = {
        "native_key_json": key_json,
        "inventory_row_id": inventory_id,
        "source_row_locator": row["source_row_locator"],
        "geometry_payload": row["geometry_payload"],
    }
    _validate_serialized_raw_row(record, ordinal, max_row_bytes)
    return record


def _require_raw_row_fields(row: Mapping[str, Any], ordinal: int) -> None:
    required = {"native_key_json", "inventory_row_id", "source_row_locator", "geometry_payload"}
    missing = required - set(row)
    if missing:
        raise ValueError(f"raw row {ordinal} is missing fields: {sorted(missing)}")


def _validate_native_key_json(value: object, ordinal: int) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"raw row {ordinal} native key must be canonical JSON text")
    parsed_key = json.loads(value)
    if _canonical_json_bytes(parsed_key).decode("utf-8") != value:
        raise ValueError(f"raw row {ordinal} native key is not canonical JSON")
    if not _is_typed_composite_key(parsed_key):
        raise ValueError(f"raw row {ordinal} native key is not a typed composite key")
    return value


def _is_typed_composite_key(value: object) -> bool:
    if not isinstance(value, list) or not value:
        return False
    names: list[str] = []
    for pair in value:
        if not _is_typed_native_key_pair(pair):
            return False
        names.append(pair[0])
    return names == sorted(set(names))


def _is_typed_native_key_pair(value: object) -> bool:
    return (
        isinstance(value, list)
        and len(value) == _NATIVE_KEY_PAIR_SIZE
        and _is_native_key_name(value[0])
        and _is_native_key_value(value[1])
    )


def _is_native_key_name(value: object) -> bool:
    return isinstance(value, str) and bool(value)


def _is_native_key_value(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (str, int))


def _validate_inventory_row_id(
    value: object,
    key_json: str,
    ordinal: int,
    source_snapshot_id: str,
) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ValueError(f"raw row {ordinal} inventory row ID must be lowercase SHA-256")
    expected = hashlib.sha256(f"{source_snapshot_id}\n{key_json}".encode()).hexdigest()
    if value != expected:
        raise ValueError(
            f"raw row {ordinal} inventory ID is not derived from the pinned source key"
        )
    return value


def _validate_source_row_locator(value: object, ordinal: int, shard_path: str) -> None:
    if not isinstance(value, str) or not value.startswith(f"{shard_path}#row="):
        raise ValueError(f"raw row {ordinal} locator does not match its source shard")


def _validate_serialized_raw_row(
    record: Mapping[str, Any], ordinal: int, max_row_bytes: int
) -> None:
    if len(_canonical_json_bytes(record)) > max_row_bytes:
        raise ValueError(f"raw row {ordinal} exceeds the configured record byte cap")


def _write_bounded_line(
    output: Any,
    record: Mapping[str, Any],
    max_row_bytes: int,
    workspace: Path,
    max_workspace_bytes: int,
) -> None:
    line = _canonical_json_bytes(record) + b"\n"
    if len(line) > max_row_bytes + 1:
        raise ValueError("serialized row exceeds the configured record byte cap")
    _ensure_workspace_quota(workspace, max_workspace_bytes, len(line))
    output.write(line)


def _stage_source_file(
    source_path: Path,
    file_pin: SourceTreeFile,
    staging_root: Path,
    max_workspace_bytes: int,
    workspace: Path,
) -> tuple[Path, str]:
    """Copy one pinned source file to an owned snapshot and verify bytes in flight."""

    expected_size = _expected_local_size(file_pin)
    if not source_path.is_file() or source_path.stat().st_size != expected_size:
        raise ValueError("selected source shard is absent or has the wrong pinned byte size")
    staging_root.mkdir(parents=True, exist_ok=True)
    destination = _staged_source_path(staging_root, source_path, file_pin.path)
    if destination.exists():
        return _verify_existing_staged_source(destination, file_pin)
    _ensure_workspace_quota(workspace, max_workspace_bytes, expected_size)
    return _copy_verified_source(
        source_path, destination, file_pin, expected_size, max_workspace_bytes, workspace
    )


def _staged_source_path(staging_root: Path, source_path: Path, source_path_pin: str) -> Path:
    digest = hashlib.sha256(source_path_pin.encode()).hexdigest()
    return staging_root / f"{digest}-{source_path.name}"


def _verify_existing_staged_source(destination: Path, file_pin: SourceTreeFile) -> tuple[Path, str]:
    git_sha, content_sha = _source_hashes(destination)
    if not _source_hash_matches(file_pin, destination.stat().st_size, git_sha, content_sha):
        raise ValueError("existing staged source bytes differ from the immutable source pin")
    return destination, content_sha


def _copy_verified_source(
    source_path: Path,
    destination: Path,
    file_pin: SourceTreeFile,
    expected_size: int,
    max_workspace_bytes: int,
    workspace: Path,
) -> tuple[Path, str]:
    temporary = destination.with_name(f".{destination.name}.partial")
    git_digest = hashlib.sha1(usedforsecurity=False)
    content_digest = hashlib.sha256()
    try:
        with source_path.open("rb") as source, temporary.open("wb") as output:
            git_digest.update(f"blob {expected_size}\0".encode())
            copy_context = SourceCopyContext(
                expected_size,
                git_digest,
                content_digest,
                output,
                workspace,
                max_workspace_bytes,
            )
            while chunk := source.read(1024 * 1024):
                _write_source_chunk(chunk, copy_context)
            output.flush()
            os.fsync(output.fileno())
        if copy_context.total != expected_size or not _source_hash_matches(
            file_pin, copy_context.total, git_digest.hexdigest(), content_digest.hexdigest()
        ):
            raise ValueError("selected source shard bytes do not match the immutable source pin")  # noqa: TRY301
        temporary.chmod(0o444)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination, content_digest.hexdigest()


def _write_source_chunk(chunk: bytes, context: SourceCopyContext) -> None:
    context.total += len(chunk)
    if context.total > context.expected_size:
        raise ValueError("source shard grew beyond its pinned size while staging")
    context.git_digest.update(chunk)
    context.content_digest.update(chunk)
    _ensure_workspace_quota(context.workspace, context.max_workspace_bytes, len(chunk))
    context.output.write(chunk)


def _source_hashes(path: Path) -> tuple[str, str]:
    size = path.stat().st_size
    git_digest = hashlib.sha1(usedforsecurity=False)
    git_digest.update(f"blob {size}\0".encode())
    content_digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            git_digest.update(chunk)
            content_digest.update(chunk)
    return git_digest.hexdigest(), content_digest.hexdigest()


def _source_hash_matches(
    file_pin: SourceTreeFile,
    size_bytes: int,
    git_blob_sha1: str,
    content_sha256: str,
) -> bool:
    if file_pin.verified_lfs_content_sha256 is not None:
        return (
            size_bytes == file_pin.lfs_size_bytes
            and content_sha256 == file_pin.verified_lfs_content_sha256
        )
    if file_pin.lfs_oid is None:
        return size_bytes == file_pin.size_bytes and git_blob_sha1 == file_pin.blob_id
    expected = file_pin.lfs_oid.removeprefix("sha256:")
    return size_bytes == file_pin.lfs_size_bytes and content_sha256 == expected


def _valid_map_receipt(
    receipt_path: Path,
    signature: str,
    expected_rows: int,
    max_row_bytes: int,
) -> str | None:
    receipt = _read_receipt(receipt_path)
    if receipt is None:
        if receipt_path.exists():
            raise ValueError("map checkpoint receipt exists but is not valid JSON")
        return None
    _validate_map_receipt_signature(receipt, signature)
    checkpoint = _map_checkpoint_path(receipt_path, receipt)
    digest = _validate_map_checkpoint_digest(checkpoint, receipt.get("checkpoint_sha256"))
    _validate_map_receipt_scope(receipt, expected_rows)
    _validate_jsonl_checkpoint(checkpoint, expected_rows, max_row_bytes)
    _validate_map_checkpoint_artifacts(checkpoint)
    return digest


def _validate_map_checkpoint_artifacts(checkpoint: Path) -> None:
    """Revalidate geometry and streamed evidence files before checkpoint reuse."""

    with checkpoint.open("rb") as stream:
        for line_number, line in enumerate(stream, start=1):
            record = json.loads(line)
            if not isinstance(record, dict):
                raise TypeError("map checkpoint rows must be JSON objects")
            result = record.get("result")
            if isinstance(result, dict):
                _verify_artifacts_recursive(result, f"map checkpoint row {line_number}")


def _verify_artifacts_recursive(value: Any, context: str) -> None:
    if isinstance(value, dict):
        if "geometry_uri" in value and "geometry_sha256" in value:
            _verify_local_artifact(
                value["geometry_uri"], value["geometry_sha256"], context, expected_records=None
            )
        if {"uri", "sha256", "record_count"}.issubset(value) and value.get("uri") is not None:
            _verify_local_artifact(
                value["uri"],
                value["sha256"],
                context,
                expected_records=value["record_count"],
            )
        for child in value.values():
            _verify_artifacts_recursive(child, context)
    elif isinstance(value, list):
        for child in value:
            _verify_artifacts_recursive(child, context)


def _verify_local_artifact(
    uri: object,
    expected_sha256: object,
    context: str,
    *,
    expected_records: object,
) -> None:
    if (
        not isinstance(uri, str)
        or not isinstance(expected_sha256, str)
        or not _SHA256_RE.fullmatch(expected_sha256)
    ):
        raise ValueError(f"{context} artifact has invalid URI or SHA-256 metadata")
    parsed = urlparse(uri)
    if parsed.scheme != "file" or parsed.netloc not in {"", "localhost"}:
        raise ValueError(f"{context} artifact URI is not a local file URI")
    path = Path(unquote(parsed.path))
    if not path.is_file() or _sha256_file(path) != expected_sha256:
        raise ValueError(f"{context} artifact bytes differ from their checkpoint pin")
    if expected_records is None:
        return
    if (
        isinstance(expected_records, bool)
        or not isinstance(expected_records, int)
        or expected_records <= 0
    ):
        raise ValueError(f"{context} sidecar record count is invalid")
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            json.loads(line)
            count += 1
    if count != expected_records:
        raise ValueError(f"{context} sidecar record count differs from its checkpoint pin")


def _validate_map_receipt_signature(receipt: Mapping[str, Any], signature: str) -> None:
    if receipt.get("signature") != signature:
        raise ValueError("map checkpoint receipt exists under the wrong semantic signature")


def _map_checkpoint_path(receipt_path: Path, receipt: Mapping[str, Any]) -> Path:
    checkpoint = receipt.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint or Path(checkpoint).name != checkpoint:
        raise ValueError("map checkpoint receipt has an unsafe checkpoint path")
    return receipt_path.parent / checkpoint


def _validate_map_checkpoint_digest(checkpoint: Path, digest: object) -> str:
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("map checkpoint receipt has an invalid checkpoint digest")
    if not checkpoint.is_file() or _sha256_file(checkpoint) != digest:
        raise ValueError("map checkpoint bytes differ from their immutable receipt")
    return digest


def _validate_map_receipt_scope(receipt: Mapping[str, Any], expected_rows: int) -> None:
    if receipt.get("row_count") != expected_rows or receipt.get("pool_complete") is not False:
        raise ValueError("map checkpoint receipt does not describe this selected shard task")


def _verify_checkpoint_file(path: Path, receipt: Mapping[str, Any], expected_rows: int) -> str:
    digest = receipt.get("sha256")
    if not isinstance(digest, str) or not _SHA256_RE.fullmatch(digest):
        raise ValueError("raw checkpoint receipt has no valid SHA-256")
    if not path.is_file() or _sha256_file(path) != digest:
        raise ValueError("raw checkpoint bytes no longer match their receipt")
    _validate_jsonl_checkpoint(path, expected_rows, _MAX_ROW_BYTES)
    return digest


def _validate_jsonl_checkpoint(path: Path, expected_rows: int, max_row_bytes: int) -> None:
    count = 0
    with path.open("rb") as stream:
        for line in stream:
            _validate_jsonl_checkpoint_line(line, max_row_bytes)
            count += 1
    if count != expected_rows:
        raise ValueError("checkpoint row count differs from its signed receipt")


def _validate_jsonl_checkpoint_line(line: bytes, max_row_bytes: int) -> None:
    if len(line) > max_row_bytes + 1:
        raise ValueError("checkpoint contains a row above its configured byte cap")
    if not line.endswith(b"\n"):
        raise ValueError("checkpoint row is not newline terminated")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise TypeError("checkpoint row must be a JSON object")


def _install_unreceipted_checkpoint(temporary: Path, target: Path, kind: str) -> None:
    """Make retries deterministic after a crash between checkpoint and receipt."""

    if target.exists():
        if _sha256_file(target) != _sha256_file(temporary):
            raise ValueError(f"unreceipted {kind} checkpoint conflicts with regenerated bytes")
        temporary.unlink()
        return
    temporary.replace(target)


def _map_signature(
    plan: PoolControllerPlan,
    shard: ShardCensus,
    source_sha256: str,
    raw_checkpoint_sha256: str,
    pin: MapPin,
    max_row_bytes: int,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "format_version": _FORMAT_VERSION,
                "source": plan.text_source,
                "repository": plan.repository,
                "revision": plan.revision,
                "census_sha256": plan.census_sha256,
                "shard_path": shard.path,
                "source_sha256": source_sha256,
                "ordered_key_sha256": shard.ordered_key_sha256,
                "raw_checkpoint_sha256": raw_checkpoint_sha256,
                "map_pin": asdict(pin),
                "max_row_bytes": max_row_bytes,
                "assignment_rule": "unique-class-at-or-above-0.8/1.0",
                "evaluation_started": False,
            }
        )
    ).hexdigest()


def _validate_source_choice(request: PoolControllerRequest, source_pin: RawSourceTreePin) -> None:
    if request.text_source != source_pin.text_source.value:
        raise ValueError("selected source differs from the pinned raw geometry tree")


def _source_file(source_pin: RawSourceTreePin, path: str) -> SourceTreeFile:
    try:
        return next(row for row in source_pin.files if row.path == path)
    except StopIteration as error:
        raise ValueError(f"selected shard {path!r} is not in the pinned source tree") from error


def _expected_local_size(file_pin: SourceTreeFile) -> int:
    if file_pin.lfs_oid is not None or file_pin.verified_lfs_content_sha256 is not None:
        if file_pin.lfs_size_bytes is None:
            raise ValueError("pinned Git LFS source file has no content size")
        return file_pin.lfs_size_bytes
    return file_pin.size_bytes


def _decode_shard_census(value: object) -> ShardCensus:
    if not isinstance(value, dict):
        raise TypeError("raw inventory shard receipt must be an object")
    row = ShardCensus(
        path=_census_text(value, "path"),
        size_bytes=_non_negative_int(value.get("size_bytes"), "shard size"),
        blob_id=_census_text(value, "blob_id"),
        lfs_oid=_census_lfs_oid(value),
        row_count=_non_negative_int(value.get("row_count"), "shard row count"),
        ordered_key_sha256=_census_text(value, "ordered_key_sha256"),
    )
    _validate_decoded_shard_identity(row)
    return row


def _census_text(value: Mapping[str, Any], field: str) -> str:
    result = value.get(field)
    if not isinstance(result, str):
        raise TypeError(f"raw inventory shard {field} must be text")
    return result


def _census_lfs_oid(value: Mapping[str, Any]) -> str | None:
    lfs_oid = value.get("lfs_oid")
    if lfs_oid is not None and not isinstance(lfs_oid, str):
        raise TypeError("raw inventory shard LFS object ID must be text or null")
    return lfs_oid


def _validate_decoded_shard_identity(row: ShardCensus) -> None:
    if not row.path or not _GIT_BLOB_RE.fullmatch(row.blob_id):
        raise ValueError("raw inventory shard path or Git blob ID is invalid")


def _task_root(workspace: Path, source: str, shard_path: str) -> Path:
    digest = hashlib.sha256(shard_path.encode()).hexdigest()[:20]
    return workspace / "tasks" / source / digest


def _safe_relative_path(path: str) -> None:
    value = Path(path)
    if value.is_absolute() or ".." in value.parts or not path:
        raise ValueError("selected shard path must be a safe repository-relative path")


def _acquire_lock(directory: Path) -> int:
    lock = directory / ".controller.lock"
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        os.close(descriptor)
        raise RuntimeError(f"another controller owns {directory}") from error
    os.ftruncate(descriptor, 0)
    os.write(descriptor, f"pid={os.getpid()}\n".encode())
    return descriptor


def _release_lock(descriptor: int) -> None:
    fcntl.flock(descriptor, fcntl.LOCK_UN)
    os.close(descriptor)


def _write_receipt(
    path: Path,
    record: Mapping[str, Any],
    workspace: Path,
    max_workspace_bytes: int,
) -> None:
    data = _canonical_json_bytes(record) + b"\n"
    if path.exists():
        if path.read_bytes() == data:
            return
        raise ValueError(f"refusing to replace immutable controller receipt {path}")
    _ensure_workspace_quota(workspace, max_workspace_bytes, len(data))
    temporary = path.with_name(f".{path.name}.partial")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as output:
            output.write(data)
            output.flush()
            os.fsync(output.fileno())
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _read_receipt(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _ensure_workspace_quota(workspace: Path, cap: int, reservation: int) -> None:
    if reservation < 0:
        raise ValueError("quota reservation cannot be negative")
    if _workspace_size(workspace) + reservation > cap:
        raise WorkspaceQuotaExceededError(
            f"workspace cap {cap} bytes would be exceeded by a {reservation}-byte write"
        )


def _workspace_size(workspace: Path) -> int:
    if not workspace.exists():
        return 0
    total = 0
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise ValueError("workspace must not contain symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str, description: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{description} must be a lowercase 64-digit SHA-256")


def _non_negative_int(value: object, description: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer")
    return value


class WorkspaceQuotaExceededError(RuntimeError):
    """Raised before a write that would exceed the controller's disk cap."""


def _load_factory(spec: str) -> Callable[..., Any]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("worker factory must use module.path:callable syntax")
    value = getattr(importlib.import_module(module_name), attribute)
    if not callable(value):
        raise TypeError("worker factory target is not callable")
    return value


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osm-polygon-eunis-pool-controller",
        description="Plan or run one resumable WorldCover/LUCAS source-shard task.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run-shard"):
        command = commands.add_parser(name)
        command.add_argument("--source", choices=sorted(_SOURCES), required=True)
        command.add_argument("--shard", action="append", required=True)
        command.add_argument(
            "--map", dest="maps", action="append", choices=sorted(_MAPS), required=True
        )
        command.add_argument("--input-root", type=Path, required=True)
        command.add_argument("--inventory-manifest", type=Path, required=True)
        command.add_argument("--inventory-sha256", required=True)
        command.add_argument("--workspace", type=Path, required=True)
        command.add_argument("--max-workspace-bytes", type=int, required=True)
        command.add_argument("--max-row-bytes", type=int, default=_MAX_ROW_BYTES)
        command.add_argument("--max-shards-per-invocation", type=int, default=1)
        command.add_argument(
            "--worker-factory", help="module.path:callable returning workers and raw-row reader"
        )
        command.add_argument("--worker-config", type=Path)
        command.add_argument("--worker-config-sha256")
        command.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Print a validated plan or call an installed adapter worker factory."""

    args = _parser().parse_args(argv)
    _validate_cli_limits(args)
    census, source_pin, request, plan = _resolve_cli_plan(args)
    return _dispatch_cli(args, census, source_pin, request, plan)


def _validate_cli_limits(args: argparse.Namespace) -> None:
    if args.max_workspace_bytes <= 0 or args.max_shards_per_invocation <= 0:
        raise SystemExit("workspace and shard limits must be positive")
    if not 0 < args.max_row_bytes <= _MAX_ROW_BYTES:
        raise SystemExit(f"row byte limit must be between 1 and {_MAX_ROW_BYTES}")


def _dispatch_cli(
    args: argparse.Namespace,
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
    request: PoolControllerRequest,
    plan: PoolControllerPlan,
) -> int:
    if args.command == "plan" or args.dry_run:
        return _print_cli_json(plan)
    if not args.worker_factory:
        raise SystemExit("run-shard requires --worker-factory module.path:callable")
    factory = _load_factory(args.worker_factory)
    workers, map_pins, raw_row_factory = factory(args, census, source_pin)
    receipt = run_shard(
        request,
        ShardRunInputs(
            census,
            source_pin,
            args.input_root,
            args.workspace,
            map_pins,
            workers,
            raw_row_factory,
        ),
    )
    return _print_cli_json(receipt)


def _resolve_cli_plan(
    args: argparse.Namespace,
) -> tuple[RawInventoryCensus, RawSourceTreePin, PoolControllerRequest, PoolControllerPlan]:
    source = TextSource(args.source)
    source_pin = polygon_geometry_tree_pins(source)[0]
    census = RawInventoryCensus.load(args.inventory_manifest, args.inventory_sha256, source_pin)
    request = PoolControllerRequest(
        text_source=args.source,
        shard_paths=tuple(args.shard),
        maps=tuple(args.maps),
        max_workspace_bytes=args.max_workspace_bytes,
        max_shards_per_invocation=args.max_shards_per_invocation,
        max_row_bytes=args.max_row_bytes,
    )
    plan = plan_run(request, census, source_pin, args.input_root)
    return census, source_pin, request, plan


def _print_cli_json(value: PoolControllerPlan | PoolControllerReceipt) -> int:
    print(json.dumps(asdict(value), sort_keys=True, indent=2))  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
