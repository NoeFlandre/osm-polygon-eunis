"""Concrete bounded workers for the frozen WorldCover and LUCAS pool inputs."""

from __future__ import annotations

import argparse
import base64
import hashlib
import importlib
import importlib.metadata
import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import tee, zip_longest
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import shapely
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis.geometry import repair_polygonal, to_equal_area
from osm_polygon_eunis.pool_adapters.common import PoolPolygonInput
from osm_polygon_eunis.pool_adapters.controller import (
    AdapterWorker,
    LocalControllerArtifactSink,
    MapPin,
    RawInventoryCensus,
    RawRowFactory,
    _ensure_workspace_quota,
    _workspace_size,
    canonical_adapter_checkpoint_rows,
    read_raw_checkpoint,
)
from osm_polygon_eunis.pool_adapters.lucas import (
    DEFAULT_LUCAS_RESOURCE_LIMITS,
    LUCAS_AREA_MEASUREMENT_ID,
    LucasGeoPackageReader,
    LucasReaderConfig,
    LucasReferenceManifest,
    LucasScratchBudget,
    LucasTaxonomy,
    lucas_2022_copernicus_manifest,
    lucas_2022_reference_pin,
)
from osm_polygon_eunis.pool_adapters.runner import iter_lucas_rows, iter_worldcover_source_rows
from osm_polygon_eunis.pool_adapters.source_pins import RawSourceTreePin
from osm_polygon_eunis.pool_adapters.worldcover import (
    WORLDCOVER_AREA_MEASUREMENT,
    worldcover_area_measurement_pin,
)
from osm_polygon_eunis.pool_adapters.worldcover_tiles import (
    VerifiedWorldCoverTileSource,
    WorldCoverShardScope,
    WorldCoverTileCatalog,
)
from osm_polygon_eunis.pool_contract import (
    GeometryStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    ReferencePin,
    SupportSemantics,
    canonical_native_key,
    inventory_row_identifier,
    native_key_json,
    polygon_snapshot_identifier,
)

_WORLD_COVER_PRODUCER_REVISION = "18c75517c74be22ab2f90edce2b65d3600461048"
_WORLD_COVER_BATCH_ROWS = 16
_DEFAULT_PARQUET_BATCH_ROWS = 16
_MAX_RAW_ROW_BYTES = 2 * 1024 * 1024
_MAX_WORKER_CONFIG_BYTES = 1024 * 1024
_SHA256_LENGTH = 64
_STAGED_PREFIX_LENGTH = 65
_GIT_COMMAND_TIMEOUT_SECONDS = 3
_NATIVE_KEY_PAIR_SIZE = 2
_GEOMETRY_TRANSFORM_POLICY = (
    "repair polygonal source geometry; EPSG:4326 edges densified at 0.01 degrees; "
    "antimeridian spans above 180 degrees rejected; EPSG:3035; geometry policy v4"
)


@dataclass(frozen=True, slots=True)
class WorkerRuntime:
    """Immutable task inputs shared by every map-specific worker."""

    source_snapshot_id: str
    workspace: Path
    max_workspace_bytes: int
    text_source: str
    catalog: WorldCoverTileCatalog | None = None

    def __post_init__(self) -> None:
        """Reject an unpinned source snapshot or an unbounded workspace."""

        if not self.source_snapshot_id or self.max_workspace_bytes <= 0:
            raise ValueError("worker runtime needs a source snapshot ID and positive workspace cap")


class _WorkspaceScratchBudget(LucasScratchBudget):
    """Reserve temporary LUCAS evidence bytes against the shared workspace cap."""

    def __init__(self, workspace: Path, maximum_bytes: int) -> None:
        self.workspace = workspace
        self.maximum_bytes = maximum_bytes
        self._reserved_bytes = 0

    def reserve(self, size_bytes: int) -> int:
        if size_bytes <= 0:
            raise ValueError("scratch reservation must be positive")
        available = self.maximum_bytes - _workspace_size(self.workspace) - self._reserved_bytes
        if available <= 0:
            _ensure_workspace_quota(self.workspace, self.maximum_bytes, size_bytes)
        granted = min(size_bytes, available)
        self._reserved_bytes += granted
        return granted

    def consume(self, size_bytes: int) -> None:
        self._settle(size_bytes)

    def release(self, size_bytes: int) -> None:
        self._settle(size_bytes)

    def _settle(self, size_bytes: int) -> None:
        if size_bytes < 0 or size_bytes > self._reserved_bytes:
            raise ValueError("scratch reservation settlement exceeds outstanding bytes")
        self._reserved_bytes -= size_bytes


def build_workers(
    args: argparse.Namespace,
    census: RawInventoryCensus,
    source_pin: RawSourceTreePin,
) -> tuple[Mapping[str, AdapterWorker], Mapping[str, MapPin], RawRowFactory]:
    """Build concrete map workers from a caller-hashed local input configuration."""

    config, config_sha256 = _load_worker_config(args)
    runtime = WorkerRuntime(
        census.source_snapshot_id,
        Path(args.workspace),
        args.max_workspace_bytes,
        census.text_source,
    )
    raw_row_factory = make_raw_row_factory(
        source_pin,
        runtime.source_snapshot_id,
        max_row_bytes=getattr(args, "max_row_bytes", _MAX_RAW_ROW_BYTES),
    )
    selected = set(args.maps)
    workers: dict[str, AdapterWorker] = {}
    pins: dict[str, MapPin] = {}

    if "worldcover" in selected:
        worldcover = _mapping(config.get("worldcover"), "worldcover worker configuration")
        catalog = WorldCoverTileCatalog.load(
            _path(worldcover, "tile_catalog_path"), _hash(worldcover, "tile_catalog_sha256")
        )
        if len(args.shard) != 1:
            raise ValueError(
                "the checked first-shard WorldCover catalog requires exactly one shard"
            )
        selected_shard = census.shard(args.shard[0])
        catalog.validate_scope(
            WorldCoverShardScope(
                source_repository=source_pin.repository,
                source_revision=source_pin.revision,
                shard_path=selected_shard.path,
                shard_blob_id=selected_shard.blob_id,
                shard_size_bytes=selected_shard.size_bytes,
                shard_row_count=selected_shard.row_count,
                shard_content_sha256=_pinned_content_sha(source_pin, selected_shard.path),
            )
        )
        producer = _load_pinned_worldcover_producer()
        tile_source = VerifiedWorldCoverTileSource(
            catalog,
            _path(worldcover, "tile_root"),
            tile_not_published_error=producer.tile_not_published_error,
            snapshot_root=runtime.workspace / "references" / "worldcover",
            workspace=runtime.workspace,
            reserve_workspace_bytes=_workspace_reserver(runtime),
        )
        reference_pin = _worldcover_reference_pin(producer.taxonomy_sha256)
        runtime = WorkerRuntime(
            runtime.source_snapshot_id,
            runtime.workspace,
            runtime.max_workspace_bytes,
            runtime.text_source,
            catalog,
        )
        workers["worldcover"] = WorldCoverAdapterWorker(
            WorldCoverWorkerOptions(
                runtime,
                tile_source,
                producer.iter_sourcewide_coverage,
                producer.is_usable_polygon,
                producer.tiles_for_bbox,
                reference_pin,
                _artifact_sink(runtime),
            )
        )
        pins["worldcover"] = _worldcover_map_pin(
            catalog,
            producer.taxonomy_sha256,
            _resource_limits_digest(args, catalog, config_sha256),
        )

    if "lucas" in selected:
        lucas_config = _mapping(config.get("lucas"), "LUCAS worker configuration")
        manifest = lucas_2022_copernicus_manifest()
        taxonomy = LucasTaxonomy.lucas_2022_c3()
        reader_path = _path(lucas_config, "gpkg_path")
        taxonomy_path = _path(lucas_config, "taxonomy_path")
        reference_pin = lucas_2022_reference_pin(manifest)
        workers["lucas"] = LucasAdapterWorker(
            LucasWorkerOptions(
                runtime,
                reader_path,
                taxonomy_path,
                manifest,
                taxonomy,
                reference_pin,
                _artifact_sink(runtime),
            )
        )
        pins["lucas"] = _lucas_map_pin(
            manifest,
            taxonomy,
            _resource_limits_digest(args, None, config_sha256),
        )
    return workers, pins, raw_row_factory


def make_raw_row_factory(
    source_pin: RawSourceTreePin,
    source_snapshot_id: str,
    *,
    batch_size: int = _DEFAULT_PARQUET_BATCH_ROWS,
    max_row_bytes: int = _MAX_RAW_ROW_BYTES,
) -> RawRowFactory:
    """Read only the native key and geometry columns from bounded Parquet batches."""

    if source_pin.purpose != "polygon_geometry":
        raise ValueError("raw-row factory requires a pinned polygon geometry tree")
    if batch_size <= 0 or max_row_bytes <= 0 or max_row_bytes > _MAX_RAW_ROW_BYTES:
        raise ValueError("Parquet batch and row byte limits must be positive and bounded")
    source_paths_by_name = {Path(row.path).name: row.path for row in source_pin.files}
    if len(source_paths_by_name) != len(source_pin.files):
        raise ValueError("pinned source tree has duplicate shard basenames")

    def read_rows(path: Path) -> Iterator[Mapping[str, Any]]:
        shard_path = source_paths_by_name.get(path.name.split("-", 1)[-1])
        if shard_path is None:
            raise ValueError("staged Parquet shard does not match a pinned source path")
        parquet = pq.ParquetFile(path)
        required = {"source_pbf", "osm_type", "osm_id", "geometry"}
        missing = required - set(parquet.schema_arrow.names)
        if missing:
            raise ValueError(f"pinned Parquet shard lacks native fields: {sorted(missing)}")
        ordinal = 0
        for batch in parquet.iter_batches(
            batch_size=batch_size,
            columns=["source_pbf", "osm_type", "osm_id", "geometry"],
        ):
            for row in batch.to_pylist():
                key = canonical_native_key(
                    {
                        "source_pbf": row["source_pbf"],
                        "osm_type": row["osm_type"],
                        "osm_id": row["osm_id"],
                    }
                )
                yield {
                    "native_key_json": native_key_json(key),
                    "inventory_row_id": inventory_row_identifier(source_snapshot_id, key),
                    "source_row_locator": f"{shard_path}#row={ordinal}",
                    "geometry_payload": _encode_geometry(row["geometry"], max_row_bytes),
                }
                ordinal += 1

    return read_rows


@dataclass(frozen=True, slots=True)
class WorldCoverWorkerOptions:
    """Fully pinned inputs for one canonical WorldCover worker."""

    runtime: WorkerRuntime
    tile_source: Any
    attempt_factory: Callable[..., Iterable[Any]]
    usable_polygon: Callable[[BaseGeometry | None], bool]
    tiles_for_bbox: Callable[[tuple[float, float, float, float]], Iterable[Any]]
    reference_pin: ReferencePin
    sink: LocalControllerArtifactSink


class WorldCoverAdapterWorker:
    """Join the pinned v4 producer's rows to the shared canonical adapter."""

    def __init__(self, options: WorldCoverWorkerOptions) -> None:
        """Retain verified producer, tile catalog, and artifact sink inputs."""

        self.runtime = options.runtime
        self.tile_source = options.tile_source
        self.attempt_factory = options.attempt_factory
        self.usable_polygon = options.usable_polygon
        self.tiles_for_bbox = options.tiles_for_bbox
        self.reference_pin = options.reference_pin
        self.sink = options.sink

    def run_checkpoint(self, raw_path: Path) -> Iterable[Mapping[str, Any]]:
        """Preflight complete source-tile coverage and bytes before any labels."""

        catalog = self.runtime.catalog
        if catalog is not None:
            usable_count, request_counts = self._tile_request_counts(
                read_raw_checkpoint(raw_path), catalog.max_entries
            )
            if (
                usable_count != catalog.usable_geometry_count
                or tuple(sorted(request_counts.items())) != catalog.tile_requests_by_usable_geometry
            ):
                raise ValueError(
                    "WorldCover catalog geometry counts differ from the verified raw checkpoint"
                )
            self.tile_source.preflight(set(request_counts))
        return self(read_raw_checkpoint(raw_path))

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        """Run the v4 producer and preserve canonical decisions and sidecars."""

        producer_rows, polygon_rows = tee(iter(rows))
        attempts = iter(self.attempt_factory(_worldcover_frames(producer_rows), self.tile_source))
        area_attempts, adapter_attempts = tee(attempts)
        polygons = self._polygons_with_measured_area(polygon_rows, area_attempts)
        pairs = iter_worldcover_source_rows(polygons, adapter_attempts, self.reference_pin)
        return canonical_adapter_checkpoint_rows(
            pairs,
            self.sink,
            f"{self.runtime.text_source}-worldcover",
        )

    def _tile_request_counts(
        self,
        rows: Iterable[Mapping[str, Any]],
        maximum: int,
    ) -> tuple[int, dict[str, int]]:
        counts: defaultdict[str, int] = defaultdict(int)
        usable_count = 0
        for row in rows:
            geometry = _decode_geometry_payload(row["geometry_payload"])
            if geometry is None or not self.usable_polygon(geometry):
                continue
            usable_count += 1
            for tile in self.tiles_for_bbox(geometry.bounds):
                counts[tile.name] += 1
                if len(counts) > maximum:
                    raise ValueError("WorldCover source shard exceeds the tile catalog entry cap")
        return usable_count, dict(counts)

    def _polygons_with_measured_area(
        self,
        rows: Iterable[Mapping[str, Any]],
        attempts: Iterable[Any],
    ) -> Iterator[Any]:
        for ordinal, (row, attempt) in enumerate(zip_longest(rows, attempts)):
            if row is None or attempt is None:
                raise ValueError("WorldCover raw row and producer attempt counts differ")
            inventory_id = str(row["inventory_row_id"])
            if attempt.source_polygon_id != inventory_id or attempt.source_row_ordinal != ordinal:
                raise ValueError(
                    "WorldCover producer attempt identity or order differs from raw input"
                )
            native_key = _native_key_from_json(str(row["native_key_json"]))
            geometry = _decode_geometry_payload(row["geometry_payload"])
            yield _pool_polygon(
                row,
                native_key,
                geometry,
                _PoolPolygonContext(
                    source_snapshot_id=self.runtime.source_snapshot_id,
                    sink=self.sink,
                    crs="EPSG:4326",
                    transform="identity; unchanged CRS84 source geometry; WorldCover producer v4",
                    original_area=(
                        float(attempt.polygon_area_m2)
                        if attempt.polygon_area_m2 is not None
                        else None
                    ),
                    accepted=self.usable_polygon(geometry),
                ),
            )


@dataclass(frozen=True, slots=True)
class LucasWorkerOptions:
    """Pinned LUCAS files and the canonical worker context."""

    runtime: WorkerRuntime
    gpkg_path: Path
    taxonomy_path: Path
    manifest: LucasReferenceManifest
    taxonomy: LucasTaxonomy
    reference_pin: ReferencePin
    sink: LocalControllerArtifactSink


class LucasAdapterWorker:
    """Open the pinned LUCAS release inside a quota-bound task worker."""

    def __init__(self, options: LucasWorkerOptions) -> None:
        """Retain the pinned local reference and export inputs."""

        self.runtime = options.runtime
        self.gpkg_path = options.gpkg_path
        self.taxonomy_path = options.taxonomy_path
        self.manifest = options.manifest
        self.taxonomy = options.taxonomy
        self.reference_pin = options.reference_pin
        self.sink = options.sink

    def run_checkpoint(self, raw_path: Path) -> Iterable[Mapping[str, Any]]:
        """Read the immutable raw checkpoint once for a selected LUCAS shard."""

        return self(read_raw_checkpoint(raw_path))

    def __call__(self, rows: Iterable[Mapping[str, Any]]) -> Iterable[Mapping[str, Any]]:
        """Project each unchanged source polygon and emit canonical LUCAS attempts."""

        scratch_parent = self.runtime.workspace / "scratch" / "lucas"
        reader_config = LucasReaderConfig(
            resource_limits=DEFAULT_LUCAS_RESOURCE_LIMITS,
            scratch_parent=scratch_parent,
            scratch_budget=_WorkspaceScratchBudget(
                self.runtime.workspace,
                self.runtime.max_workspace_bytes,
            ),
        )
        with LucasGeoPackageReader(
            self.gpkg_path,
            self.taxonomy_path,
            self.manifest,
            self.taxonomy,
            config=reader_config,
        ) as reader:
            polygons = _lucas_pool_polygons(
                rows,
                self.runtime.source_snapshot_id,
                self.sink,
            )
            pairs = iter_lucas_rows(polygons, reader, self.taxonomy, self.reference_pin)
            yield from canonical_adapter_checkpoint_rows(
                pairs,
                self.sink,
                f"{self.runtime.text_source}-lucas",
            )


def _worldcover_frames(rows: Iterable[Mapping[str, Any]]) -> Iterator[Any]:
    pd = importlib.import_module("pandas")

    frame: list[dict[str, Any]] = []
    for row in rows:
        frame.append(
            {
                "polygon_id": str(row["inventory_row_id"]),
                "geometry": _decode_geometry_payload(row["geometry_payload"]),
            }
        )
        if len(frame) >= _WORLD_COVER_BATCH_ROWS:
            yield pd.DataFrame(frame, columns=["polygon_id", "geometry"])
            frame.clear()
    if frame:
        yield pd.DataFrame(frame, columns=["polygon_id", "geometry"])


def _worldcover_reference_pin(taxonomy_sha256: str) -> ReferencePin:
    return ReferencePin(
        release_id="esa-worldcover-2021-v200",
        source_repository="ESA WorldCover",
        source_revision="2021-v200; class map; 10 m",
        taxonomy_id="worldcover-v200",
        taxonomy_revision=f"18c75517c74be22ab2f90edce2b65d3600461048:{taxonomy_sha256}",
        support_semantics=SupportSemantics.MAPPED_COVER,
        support_unit="m^2",
        threshold=0.8,
        support_union_policy="union_by_class",
        area_measurement=worldcover_area_measurement_pin(_WORLD_COVER_PRODUCER_REVISION),
    )


def _worldcover_map_pin(
    catalog: WorldCoverTileCatalog,
    taxonomy_sha256: str,
    resource_limits_sha256: str,
) -> MapPin:
    return MapPin(
        map_name="worldcover",
        reference_manifest_sha256=catalog.sha256,
        taxonomy_sha256=taxonomy_sha256,
        area_method_id=WORLDCOVER_AREA_MEASUREMENT,
        adapter_revision=_implementation_sha256(
            ("common.py", "runner.py", "worldcover.py", "workers.py", "worldcover_tiles.py")
        ),
        builder_revision=_implementation_sha256(("controller.py", "streaming.py")),
        resource_limits_sha256=resource_limits_sha256,
        support_union_policy="union_by_class",
        producer_revision=_WORLD_COVER_PRODUCER_REVISION,
    )


def _lucas_map_pin(
    manifest: LucasReferenceManifest,
    taxonomy: LucasTaxonomy,
    resource_limits_sha256: str,
) -> MapPin:
    manifest_sha = _canonical_sha256(asdict(manifest))
    return MapPin(
        map_name="lucas",
        reference_manifest_sha256=manifest_sha,
        taxonomy_sha256=taxonomy.sha256,
        area_method_id=LUCAS_AREA_MEASUREMENT_ID,
        adapter_revision=_implementation_sha256(
            ("common.py", "runner.py", "lucas.py", "workers.py", "../geometry.py")
        ),
        builder_revision=_implementation_sha256(("controller.py", "streaming.py")),
        resource_limits_sha256=resource_limits_sha256,
        support_union_policy="independent_union_per_level_and_class:v1",
    )


def _resource_limits_digest(
    args: argparse.Namespace,
    catalog: WorldCoverTileCatalog | None,
    worker_config_sha256: str,
) -> str:
    return _canonical_sha256(
        {
            "max_workspace_bytes": args.max_workspace_bytes,
            "max_row_bytes": getattr(args, "max_row_bytes", 2 * 1024 * 1024),
            "max_shards_per_invocation": args.max_shards_per_invocation,
            "worker_config_sha256": worker_config_sha256,
            "worldcover_batch_rows": _WORLD_COVER_BATCH_ROWS,
            "worldcover_catalog_limits": (
                {
                    "max_entries": catalog.max_entries,
                    "max_tile_bytes": catalog.max_tile_bytes,
                    "max_total_bytes": catalog.max_total_bytes,
                }
                if catalog is not None
                else None
            ),
            "lucas": asdict(DEFAULT_LUCAS_RESOURCE_LIMITS),
            "lucas_scratch_parent": "<controller-workspace>/scratch/lucas",
        }
    )


def _artifact_sink(runtime: WorkerRuntime) -> LocalControllerArtifactSink:
    return LocalControllerArtifactSink(
        runtime.workspace / "artifacts",
        runtime.workspace,
        runtime.max_workspace_bytes,
    )


def _workspace_reserver(runtime: WorkerRuntime) -> Callable[[int], None]:
    def reserve(size_bytes: int) -> None:
        _ensure_workspace_quota(runtime.workspace, runtime.max_workspace_bytes, size_bytes)

    return reserve


def _lucas_pool_polygons(
    rows: Iterable[Mapping[str, Any]],
    source_snapshot_id: str,
    sink: LocalControllerArtifactSink,
) -> Iterator[Any]:
    for row in rows:
        raw_geometry = _decode_geometry_payload(row["geometry_payload"])
        native_key = _native_key_from_json(str(row["native_key_json"]))
        source_geometry = repair_polygonal(raw_geometry)
        projected = to_equal_area(source_geometry) if source_geometry is not None else None
        accepted = projected is not None and _usable(projected)
        yield _pool_polygon(
            row,
            native_key,
            projected if accepted else None,
            _PoolPolygonContext(
                source_snapshot_id=source_snapshot_id,
                sink=sink,
                crs="EPSG:3035",
                transform=_GEOMETRY_TRANSFORM_POLICY,
                original_area=(
                    float(projected.area) if accepted and projected is not None else None
                ),
                accepted=accepted,
                geometry_error="geometry_transform_failed_or_antimeridian_rejected",
            ),
        )


@dataclass(frozen=True, slots=True)
class _PoolPolygonContext:
    source_snapshot_id: str
    sink: LocalControllerArtifactSink
    crs: str
    transform: str
    original_area: float | None
    accepted: bool
    geometry_error: str = "geometry_unusable"


def _pool_polygon(
    row: Mapping[str, Any],
    native_key: tuple[tuple[str, str | int], ...],
    geometry: BaseGeometry | None,
    context: _PoolPolygonContext,
) -> Any:
    inventory_id = str(row["inventory_row_id"])
    locator = str(row["source_row_locator"])
    snapshot = None
    exclusion_reason = None
    if not context.accepted or geometry is None:
        exclusion_reason = context.geometry_error
    else:
        payload = shapely.to_wkb(geometry, byte_order=1, output_dimension=2)
        geometry_sha256 = hashlib.sha256(payload).hexdigest()
        snapshot_id = polygon_snapshot_identifier(
            context.source_snapshot_id, native_key, geometry_sha256
        )
        geometry_uri = context.sink.put(
            f"geometry/{inventory_id}-{geometry_sha256}.wkb",
            payload,
        )
        snapshot = PolygonSnapshot(
            polygon_snapshot_id=snapshot_id,
            inventory_row_id=inventory_id,
            native_key=native_key,
            geometry_uri=geometry_uri,
            geometry_sha256=geometry_sha256,
            geometry_format="WKB",
            crs=context.crs,
            coordinate_transform=context.transform,
            geometry_status=GeometryStatus.USABLE,
            original_area=context.original_area,
            area_unit="m^2",
        )
    inventory = PolygonInventoryRow(
        inventory_row_id=inventory_id,
        native_key=native_key,
        source_row_locator=locator,
        polygon_snapshot_id=snapshot.polygon_snapshot_id if snapshot is not None else None,
        exclusion_reason=exclusion_reason,
    )
    source_polygon_id = inventory_id
    return PoolPolygonInput(
        source_polygon_id,
        inventory,
        snapshot,
        geometry if context.accepted else None,
    )


def _load_worker_config(args: argparse.Namespace) -> tuple[dict[str, Any], str]:
    path = getattr(args, "worker_config", None)
    expected_sha = getattr(args, "worker_config_sha256", None)
    if path is None or expected_sha is None:
        raise ValueError("built-in worker factory requires a SHA-256-pinned --worker-config")
    if len(expected_sha) != _SHA256_LENGTH or any(
        character not in "0123456789abcdef" for character in expected_sha
    ):
        raise ValueError("worker config SHA-256 must be lowercase hexadecimal")
    config_path = Path(path)
    if config_path.stat().st_size > _MAX_WORKER_CONFIG_BYTES:
        raise ValueError("worker config must not exceed 1 MiB")
    raw = config_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("worker config bytes differ from the supplied SHA-256")
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError("worker config is not valid JSON") from error
    if not isinstance(config, dict):
        raise TypeError("worker config must be a JSON object")
    if config.get("format_version") != "bounded-pool-worker-config-v1":
        raise ValueError("worker config format is unsupported")
    return config, expected_sha


def _load_pinned_worldcover_producer() -> Any:
    """Import producer functions only after proving the installed commit identity."""

    revision = _worldcover_install_revision()
    if revision != _WORLD_COVER_PRODUCER_REVISION:
        raise ValueError("installed osm-worldcover is not the pinned v4 producer commit")
    pool_attempts = importlib.import_module("osm_worldcover.adapters.worldcover.pool_attempts")
    tiling = importlib.import_module("osm_worldcover.domain.tiling")
    nomenclature = importlib.import_module("osm_worldcover.domain.nomenclature")
    taxonomy_bytes = json.dumps(
        {
            "nodata": nomenclature.NODATA,
            "classes": [[code, label] for code, label in sorted(nomenclature.CLASS_LABELS.items())],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    tiles = importlib.import_module("osm_worldcover.adapters.worldcover.tiles")
    return _WorldCoverProducer(
        pool_attempts.iter_sourcewide_coverage,
        importlib.import_module("osm_worldcover.domain.geometry").is_usable_polygon,
        tiling.tiles_for_bbox,
        tiles.TileNotPublishedError,
        hashlib.sha256(taxonomy_bytes).hexdigest(),
    )


@dataclass(frozen=True, slots=True)
class _WorldCoverProducer:
    iter_sourcewide_coverage: Callable[..., Iterable[Any]]
    is_usable_polygon: Callable[[BaseGeometry | None], bool]
    tiles_for_bbox: Callable[[tuple[float, float, float, float]], Iterable[Any]]
    tile_not_published_error: type[Exception]
    taxonomy_sha256: str


def _worldcover_install_revision() -> str | None:
    package_path = _imported_worldcover_package_path()
    if package_path is None:
        return None
    direct_url_revision = _worldcover_distribution_revision(package_path)
    if direct_url_revision is not None:
        return direct_url_revision
    return _worldcover_git_revision(package_path)


def _imported_worldcover_package_path() -> Path | None:
    try:
        package = importlib.import_module("osm_worldcover")
    except ImportError:
        return None
    package_file = getattr(package, "__file__", None)
    return Path(package_file).resolve() if isinstance(package_file, str) else None


def _worldcover_distribution_revision(package_path: Path) -> str | None:
    try:
        distribution = importlib.metadata.distribution("osm-worldcover")
    except importlib.metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if direct_url is None:
        return None
    try:
        origin = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    return _verified_direct_url_revision(distribution, origin, package_path)


def _worldcover_git_revision(package_path: Path) -> str | None:
    repository = package_path.parents[2]
    if not package_path.is_relative_to(repository):
        return None
    git = shutil.which("git")
    if git is None:
        return None
    try:
        dirty = subprocess.run(  # noqa: S603 -- fixed git executable and argument vector
            [git, "-C", str(repository), "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
        ).stdout
        revision = subprocess.run(  # noqa: S603 -- fixed git executable and argument vector
            [git, "-C", str(repository), "rev-parse", "--verify", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=_GIT_COMMAND_TIMEOUT_SECONDS,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return revision if not dirty else None


def _verified_direct_url_revision(
    distribution: Any,
    origin: object,
    package_path: Path,
) -> str | None:
    if not isinstance(origin, dict) or not isinstance(origin.get("vcs_info"), dict):
        return None
    vcs = origin["vcs_info"]
    repository_url = origin.get("url")
    if (
        not isinstance(repository_url, str)
        or _canonical_repository_url(repository_url)
        != "https://github.com/noeflandre/osm-worldcover"
    ):
        return None
    commit_id = vcs.get("commit_id")
    dirty = vcs.get("dirty")
    if not isinstance(commit_id, str) or (dirty is not None and dirty is not False):
        return None
    directory_info = origin.get("dir_info", {})
    editable_location = (
        directory_info.get("editable_project_location")
        if isinstance(directory_info, dict)
        else None
    )
    if isinstance(editable_location, str):
        if dirty is not False:
            return None
        editable_root = Path(editable_location).resolve()
        if package_path.is_relative_to(editable_root):
            # PEP 610 metadata can outlive source edits. Bind editable imports to
            # their live clean checkout instead of trusting install-time metadata.
            live_revision = _worldcover_git_revision(package_path)
            module_matches_distribution = live_revision == commit_id
        else:
            module_matches_distribution = False
    else:
        installed_package = Path(distribution.locate_file("osm_worldcover/__init__.py")).resolve()
        module_matches_distribution = package_path == installed_package
    return commit_id if module_matches_distribution else None


def _native_key_from_json(value: str) -> tuple[tuple[str, str | int], ...]:
    decoded = json.loads(value)
    if not isinstance(decoded, list):
        raise TypeError("raw checkpoint native key is not a canonical pair list")
    if any(
        not isinstance(pair, list)
        or len(pair) != _NATIVE_KEY_PAIR_SIZE
        or not isinstance(pair[0], str)
        or isinstance(pair[1], bool)
        or not isinstance(pair[1], (str, int))
        for pair in decoded
    ):
        raise TypeError("raw checkpoint native key contains an invalid pair")
    return tuple((str(pair[0]), pair[1]) for pair in decoded)


def _encode_geometry(value: object, max_row_bytes: int) -> dict[str, Any]:
    if value is None:
        return {"kind": "missing"}
    if isinstance(value, (bytes, bytearray, memoryview)):
        raw = bytes(value)
        if len(raw) > max_row_bytes:
            raise ValueError("raw geometry exceeds the configured row-byte limit")
        return {"kind": "base64", "value": base64.b64encode(raw).decode("ascii")}
    if isinstance(value, str):
        if len(value.encode("utf-8")) > max_row_bytes:
            raise ValueError("raw geometry exceeds the configured row-byte limit")
        return {"kind": "text", "value": value}
    if isinstance(value, (dict, list, int, float, bool)):
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > max_row_bytes:
            raise ValueError("raw geometry exceeds the configured row-byte limit")
        return {"kind": "json", "value": value}
    raise TypeError(f"unsupported raw geometry type {type(value).__name__}")


def _decode_geometry_payload(payload: Mapping[str, Any]) -> BaseGeometry | None:
    kind = payload.get("kind")
    if kind in {"missing", "unsupported"}:
        return None
    try:
        if kind == "base64":
            return shapely.from_wkb(base64.b64decode(str(payload["value"]), validate=True))
        if kind == "text":
            return shapely.from_geojson(str(payload["value"]), on_invalid="ignore")
        if kind == "json":
            return shapely.from_geojson(
                json.dumps(payload["value"], ensure_ascii=False), on_invalid="ignore"
            )
    except (ValueError, TypeError, shapely.errors.ShapelyError):
        return None
    raise ValueError("raw geometry checkpoint has an unknown encoding")


def _usable(geometry: BaseGeometry) -> bool:
    return (
        geometry.geom_type in {"Polygon", "MultiPolygon"}
        and not geometry.is_empty
        and geometry.is_valid
        and geometry.area > 0.0
    )


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_sha256(file_names: tuple[str, ...]) -> str:
    package_root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in file_names:
        path = package_root / name
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    return value


def _path(value: Mapping[str, Any], field: str) -> Path:
    raw = value.get(field)
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"worker configuration field {field!r} must be a local path")
    return Path(raw)


def _hash(value: Mapping[str, Any], field: str) -> str:
    raw = value.get(field)
    if (
        not isinstance(raw, str)
        or len(raw) != _SHA256_LENGTH
        or any(c not in "0123456789abcdef" for c in raw)
    ):
        raise ValueError(f"worker configuration field {field!r} must be a lowercase SHA-256")
    return raw


def _canonical_repository_url(value: str) -> str:
    return value.removeprefix("git+").removesuffix(".git").rstrip("/").casefold()


def _pinned_content_sha(source_pin: RawSourceTreePin, path: str) -> str:
    file_pin = next((row for row in source_pin.files if row.path == path), None)
    if file_pin is None:
        raise ValueError("WorldCover catalog shard is absent from the selected source tree")
    if file_pin.verified_lfs_content_sha256 is None:
        raise ValueError("WorldCover catalog requires a verified content hash for its raw shard")
    return file_pin.verified_lfs_content_sha256
