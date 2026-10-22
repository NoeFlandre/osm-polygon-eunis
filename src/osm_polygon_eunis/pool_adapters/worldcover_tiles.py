"""Pinned, byte-verified local WorldCover tiles for bounded pool attempts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import rasterio

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TILE_RE = re.compile(r"^([NS])(\d{2})([EW])(\d{3})$")
_PRODUCER_REVISION = "18c75517c74be22ab2f90edce2b65d3600461048"
_WORLD_COVER_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/"
_MAX_CATALOG_BYTES = 1024 * 1024
_TILE_WIDTH_PIXELS = 36_000
_TILE_HEIGHT_PIXELS = 36_000
_TILE_CRS_EPSG = 4_326
_TILE_BAND_COUNT = 1
_TILE_NODATA = 0
_TILE_GRID_DEGREES = 3


@dataclass(frozen=True, slots=True)
class WorldCoverTileEntry:
    """One published tile and its pinned local byte identity."""

    tile_id: str
    state: str
    filename: str
    relative_path: str
    size_bytes: int
    sha256: str
    source_uri: str
    absence_evidence_uri: str | None = None


@dataclass(frozen=True, slots=True)
class WorldCoverShardScope:
    """Verified source-shard identity required by a tile catalog."""

    source_repository: str
    source_revision: str
    shard_path: str
    shard_blob_id: str
    shard_size_bytes: int
    shard_row_count: int
    shard_content_sha256: str


@dataclass(frozen=True, slots=True)
class WorldCoverTileCatalog:
    """Hash-verified tile list bounded by entry count and aggregate bytes."""

    sha256: str
    producer_revision: str
    source_repository: str
    source_revision: str
    shard_path: str
    shard_git_blob_id: str
    shard_content_sha256: str
    shard_size_bytes: int
    shard_row_count: int
    usable_geometry_count: int
    max_entries: int
    max_tile_bytes: int
    max_total_bytes: int
    requested_tile_ids: tuple[str, ...]
    tile_requests_by_usable_geometry: tuple[tuple[str, int], ...]
    entries: tuple[WorldCoverTileEntry, ...]

    @classmethod
    def load(cls, path: Path, expected_sha256: str) -> WorldCoverTileCatalog:
        """Load a small caller-pinned catalog; never accept unhashed JSON."""

        _require_sha256(expected_sha256, "WorldCover tile catalog")
        size = path.stat().st_size
        if size <= 0 or size > _MAX_CATALOG_BYTES:
            raise ValueError("WorldCover tile catalog must be non-empty and at most 1 MiB")
        raw = path.read_bytes()
        observed = hashlib.sha256(raw).hexdigest()
        if observed != expected_sha256:
            raise ValueError("WorldCover tile catalog bytes differ from the supplied SHA-256")
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError("WorldCover tile catalog is not valid JSON") from error
        return _catalog_from_document(cls, document, observed)

    @property
    def entries_by_id(self) -> dict[str, WorldCoverTileEntry]:
        """Index the already-bounded tile entries by exact ESA tile ID."""

        return {entry.tile_id: entry for entry in self.entries}

    @property
    def total_bytes(self) -> int:
        """Total bytes the catalog could expose to the producer."""

        return sum(entry.size_bytes for entry in self.entries if entry.state == "published")

    def validate_scope(self, scope: WorldCoverShardScope) -> None:
        """Bind this shard-scoped tile catalog to the exact selected raw shard."""

        if (
            self.source_repository,
            self.source_revision,
            self.shard_path,
            self.shard_git_blob_id,
            self.shard_size_bytes,
            self.shard_row_count,
            self.shard_content_sha256,
        ) != (
            scope.source_repository,
            scope.source_revision,
            scope.shard_path,
            scope.shard_blob_id,
            scope.shard_size_bytes,
            scope.shard_row_count,
            scope.shard_content_sha256,
        ):
            raise ValueError("WorldCover tile catalog does not match the selected source shard")


def _catalog_from_document(
    catalog_type: type[WorldCoverTileCatalog],
    document: object,
    catalog_sha256: str,
) -> WorldCoverTileCatalog:
    document, scope = _catalog_document_and_scope(document)
    (
        source_repository,
        source_revision,
        shard_path,
        shard_blob_id,
        shard_content_sha,
        shard_size,
        shard_rows,
        usable_rows,
    ) = _scope_identity(scope)
    max_entries, max_tile_bytes, max_total_bytes = _resource_limits(document)
    requested_tile_ids, tile_request_counts = _tile_requests(scope)
    entries = _catalog_entries(document, max_entries, requested_tile_ids)
    if usable_rows > shard_rows:
        raise ValueError("WorldCover catalog geometry count exceeds the selected shard rows")
    if any(entry.size_bytes > max_tile_bytes for entry in entries if entry.state == "published"):
        raise ValueError("WorldCover tile exceeds the per-file byte limit")
    total_bytes = sum(entry.size_bytes for entry in entries if entry.state == "published")
    if total_bytes > max_total_bytes:
        raise ValueError("WorldCover tile catalog exceeds its aggregate byte limit")
    return catalog_type(
        sha256=catalog_sha256,
        producer_revision=_text(document, "producer_revision"),
        source_repository=source_repository,
        source_revision=source_revision,
        shard_path=shard_path,
        shard_git_blob_id=shard_blob_id,
        shard_content_sha256=shard_content_sha,
        shard_size_bytes=shard_size,
        shard_row_count=shard_rows,
        usable_geometry_count=usable_rows,
        max_entries=max_entries,
        max_tile_bytes=max_tile_bytes,
        max_total_bytes=max_total_bytes,
        requested_tile_ids=requested_tile_ids,
        tile_requests_by_usable_geometry=tile_request_counts,
        entries=entries,
    )


def _catalog_document_and_scope(document: object) -> tuple[dict[str, Any], dict[str, Any]]:
    if (
        not isinstance(document, dict)
        or document.get("format_version") != "worldcover-tile-catalog-v1"
    ):
        raise ValueError("WorldCover tile catalog format is unsupported")
    if document.get("producer_revision") != _PRODUCER_REVISION:
        raise ValueError("WorldCover tile catalog is not bound to the pinned v4 producer")
    product = document.get("product")
    if not isinstance(product, dict) or (
        product.get("publisher"),
        product.get("year"),
        product.get("version"),
        product.get("class_map"),
        product.get("crs"),
        product.get("band_count"),
        product.get("dtype"),
        product.get("nodata"),
        product.get("tile_grid_degrees"),
        product.get("width"),
        product.get("height"),
    ) != (
        "ESA WorldCover",
        2021,
        "v200",
        "Map",
        "EPSG:4326",
        _TILE_BAND_COUNT,
        "uint8",
        _TILE_NODATA,
        _TILE_GRID_DEGREES,
        _TILE_WIDTH_PIXELS,
        _TILE_HEIGHT_PIXELS,
    ):
        raise ValueError("WorldCover tile catalog does not describe the pinned 2021 v200 product")
    scope = document.get("coverage_scope")
    if not isinstance(scope, dict) or scope.get("text_source") not in {
        "website",
        "description",
        "wiki",
    }:
        raise ValueError("WorldCover tile catalog lacks a recognized source scope")
    return document, scope


def _scope_identity(scope: Mapping[str, Any]) -> tuple[str, str, str, str, str, int, int, int]:
    source_repository = _text(scope, "source_repository")
    source_revision = _text(scope, "source_revision")
    shard_path = _text(scope, "shard_path")
    shard_blob_id = _text(scope, "shard_git_blob_id")
    shard_content_sha = _text(scope, "shard_content_sha256")
    shard_size = _positive_int(scope.get("shard_size_bytes"), "source shard size")
    shard_rows = _positive_int(scope.get("shard_row_count"), "source shard row count")
    usable_rows = _non_negative_int(scope.get("usable_geometry_count"), "usable geometry count")
    _require_sha256(shard_content_sha, "source shard content")
    if not re.fullmatch(r"[0-9a-f]{40}", shard_blob_id):
        raise ValueError("source shard Git blob ID must be a full lowercase SHA-1")
    return (
        source_repository,
        source_revision,
        shard_path,
        shard_blob_id,
        shard_content_sha,
        shard_size,
        shard_rows,
        usable_rows,
    )


def _resource_limits(document: Mapping[str, Any]) -> tuple[int, int, int]:
    limits = document.get("resource_limits")
    if not isinstance(limits, dict):
        raise TypeError("WorldCover tile catalog must include explicit byte and entry caps")
    max_entries = _positive_int(limits.get("max_entries"), "tile-entry limit")
    max_tile_bytes = _positive_int(limits.get("max_tile_bytes"), "tile byte limit")
    max_total_bytes = _positive_int(limits.get("max_total_bytes"), "total tile byte limit")
    return max_entries, max_tile_bytes, max_total_bytes


def _catalog_entries(
    document: Mapping[str, Any],
    max_entries: int,
    requested_tile_ids: tuple[str, ...],
) -> tuple[WorldCoverTileEntry, ...]:
    raw_entries = document.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries or len(raw_entries) > max_entries:
        raise ValueError("WorldCover tile catalog entry count exceeds its declared bound")
    entries = tuple(_entry_from_document(row) for row in raw_entries)
    ids = tuple(entry.tile_id for entry in entries)
    if ids != tuple(sorted(set(ids))):
        raise ValueError("WorldCover tile entries must have unique, sorted IDs")
    if ids != requested_tile_ids:
        raise ValueError("WorldCover catalog entries do not exactly cover its source geometry scan")
    return entries


def _tile_requests(
    scope: Mapping[str, Any],
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    raw_tile_requests = scope.get("tile_requests_by_usable_geometry")
    if not isinstance(raw_tile_requests, dict) or not raw_tile_requests:
        raise ValueError(
            "WorldCover tile catalog must record the bounded source geometry tile scan"
        )
    if any(not isinstance(tile_id, str) for tile_id in raw_tile_requests):
        raise ValueError("WorldCover geometry tile scan contains an invalid tile ID")
    requested_tile_ids = tuple(sorted(raw_tile_requests))
    tile_request_counts: list[tuple[str, int]] = []
    for tile_id, count in raw_tile_requests.items():
        if not isinstance(tile_id, str) or not _TILE_RE.fullmatch(tile_id):
            raise ValueError("WorldCover geometry tile scan contains an invalid tile ID")
        tile_request_counts.append(
            (tile_id, _positive_int(count, f"geometry request count for {tile_id}"))
        )
    return requested_tile_ids, tuple(sorted(tile_request_counts))


def _entry_from_document(value: object) -> WorldCoverTileEntry:
    if not isinstance(value, dict):
        raise TypeError("WorldCover catalog entries must be objects")
    tile_id = _text(value, "tile_id")
    if not _TILE_RE.fullmatch(tile_id):
        raise ValueError("WorldCover tile ID has an invalid ESA grid name")
    state = _text(value, "state")
    if state not in {"published", "unpublished"}:
        raise ValueError("WorldCover tile state must be published or explicitly unpublished")
    filename = _text(value, "filename")
    expected_filename = f"ESA_WorldCover_10m_2021_v200_{tile_id}_Map.tif"
    if filename != expected_filename:
        raise ValueError("WorldCover filename does not match the exact tile ID and product")
    relative_path = _text(value, "relative_path")
    path = Path(relative_path)
    if path.is_absolute() or len(path.parts) != 1 or path.name != filename:
        raise ValueError("WorldCover catalog path must be a safe relative product filename")
    source_uri = _text(value, "source_uri")
    parsed_uri = urlparse(source_uri)
    if parsed_uri.scheme != "https" or source_uri != f"{_WORLD_COVER_URL}{filename}":
        raise ValueError("WorldCover tile source URI is outside the pinned official product")
    absence_uri = value.get("absence_evidence_uri")
    if state == "unpublished" and (not isinstance(absence_uri, str) or not absence_uri.strip()):
        raise ValueError("unpublished WorldCover tiles need an exact source absence evidence URI")
    if state == "published" and absence_uri is not None:
        raise ValueError("published WorldCover tiles cannot carry absence evidence")
    size_bytes = _positive_int(value.get("size_bytes"), "tile size")
    sha256 = _text(value, "sha256")
    _require_sha256(sha256, "WorldCover tile")
    return WorldCoverTileEntry(
        tile_id,
        state,
        filename,
        relative_path,
        size_bytes,
        sha256,
        source_uri,
        absence_uri,
    )


class VerifiedWorldCoverTileSource:
    """Supply only catalog-listed tile bytes whose size, hash, and raster profile match."""

    def __init__(
        self,
        catalog: WorldCoverTileCatalog,
        reference_root: Path,
        *,
        tile_not_published_error: type[Exception],
        snapshot_root: Path | None = None,
        workspace: Path | None = None,
        reserve_workspace_bytes: Callable[[int], None] | None = None,
    ) -> None:
        """Retain catalog scope and caller-owned tile files without downloads."""

        self.catalog = catalog
        self.reference_root = reference_root.resolve(strict=True)
        self._tile_not_published_error = tile_not_published_error
        if snapshot_root is not None and (workspace is None or reserve_workspace_bytes is None):
            raise ValueError("verified tile snapshots require a bounded workspace")
        self.workspace = workspace.resolve() if workspace is not None else None
        self.reserve_workspace_bytes = reserve_workspace_bytes
        self.snapshot_root = snapshot_root.resolve() if snapshot_root is not None else None
        if (
            self.snapshot_root is not None
            and self.workspace is not None
            and not self.snapshot_root.is_relative_to(self.workspace)
        ):
            raise ValueError("WorldCover tile snapshots must be inside the bounded workspace")
        self._verified: dict[str, tuple[Path, tuple[int, int, int, int]]] = {}

    def preflight(self, required_tile_ids: set[str]) -> None:
        """Verify all selected-shard tiles before starting any raster attempt."""

        entries = self.catalog.entries_by_id
        if required_tile_ids != set(self.catalog.requested_tile_ids):
            raise ValueError("WorldCover tile catalog coverage differs from the raw shard geometry")
        missing_catalog_rows = required_tile_ids - entries.keys()
        if missing_catalog_rows:
            raise ValueError(
                f"WorldCover tile catalog omits required tiles: {sorted(missing_catalog_rows)}"
            )
        for tile_id in sorted(required_tile_ids):
            entry = entries[tile_id]
            if entry.state != "published":
                continue
            self._verified_path(entry)

    def ensure(self, tile: Any) -> Path:
        """Return a local verified tile or fail as an operational error."""

        entry = self.catalog.entries_by_id.get(tile.name)
        if entry is None:
            raise ValueError(f"WorldCover tile {tile.name} is absent from the pinned catalog")
        if entry.state != "published":
            raise self._tile_not_published_error(
                "WorldCover source explicitly records "
                f"{entry.tile_id} as unpublished; absence evidence: "
                f"{entry.absence_evidence_uri}"
            )
        return self._verified_path(entry)

    def discard(self, _tile: Any) -> None:
        """Keep the caller-owned immutable raster available for neighboring polygons."""

    def _verified_path(self, entry: WorldCoverTileEntry) -> Path:
        path = self.reference_root / entry.relative_path
        resolved = path.resolve(strict=True)
        if resolved.parent != self.reference_root or path.is_symlink() or not resolved.is_file():
            raise ValueError(f"WorldCover tile path is not a regular catalog-local file: {path}")
        if self.snapshot_root is not None:
            resolved = self._verified_snapshot(entry, resolved)
        stat = resolved.stat()
        signature = (stat.st_size, stat.st_mtime_ns, stat.st_ino, stat.st_ctime_ns)
        cached = self._verified.get(entry.tile_id)
        if cached is None or cached[0] != resolved or cached[1] != signature:
            if stat.st_size != entry.size_bytes or _sha256_file(resolved) != entry.sha256:
                raise ValueError(
                    f"WorldCover tile bytes differ from pinned catalog: {entry.tile_id}"
                )
            _verify_raster_profile(resolved, entry.tile_id)
            self._verified[entry.tile_id] = (resolved, signature)
        return resolved

    def _verified_snapshot(self, entry: WorldCoverTileEntry, source: Path) -> Path:
        root, destination = self._snapshot_destination(entry)
        if destination.is_file() and not destination.is_symlink():
            return destination.resolve(strict=True)
        if source.stat().st_size != entry.size_bytes or _sha256_file(source) != entry.sha256:
            raise ValueError(f"WorldCover tile bytes differ from pinned catalog: {entry.tile_id}")
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{entry.filename}-",
            suffix=".partial",
            dir=root,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        try:
            _copy_pinned_tile(entry, source, temporary_path, self.reserve_workspace_bytes)
            temporary_path.chmod(0o444)
            temporary_path.replace(destination)
        except Exception:
            temporary_path.unlink(missing_ok=True)
            raise
        return destination.resolve(strict=True)

    def _snapshot_destination(self, entry: WorldCoverTileEntry) -> tuple[Path, Path]:
        if self.snapshot_root is None or self.workspace is None:
            raise RuntimeError("WorldCover snapshot configuration was lost")
        root = self.snapshot_root / self.catalog.sha256
        if root.is_symlink():
            raise ValueError("WorldCover snapshot directory must not be a symlink")
        root.mkdir(parents=True, exist_ok=True)
        resolved_root = root.resolve(strict=True)
        if not resolved_root.is_relative_to(self.workspace):
            raise ValueError("WorldCover tile snapshot escaped the bounded workspace")
        return resolved_root, resolved_root / entry.relative_path


def _copy_pinned_tile(
    entry: WorldCoverTileEntry,
    source: Path,
    destination: Path,
    reserve_workspace_bytes: Callable[[int], None] | None,
) -> None:
    if reserve_workspace_bytes is None:
        raise RuntimeError("WorldCover snapshot workspace cap was lost")
    digest = hashlib.sha256()
    copied = 0
    with source.open("rb") as input_stream, destination.open("wb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            if copied + len(chunk) > entry.size_bytes:
                raise ValueError(f"WorldCover tile grew beyond its catalog size: {entry.tile_id}")
            reserve_workspace_bytes(len(chunk))
            output_stream.write(chunk)
            copied += len(chunk)
            digest.update(chunk)
        output_stream.flush()
        os.fsync(output_stream.fileno())
    if copied != entry.size_bytes or digest.hexdigest() != entry.sha256:
        raise ValueError(f"WorldCover tile copy differs from pinned catalog: {entry.tile_id}")


def _verify_raster_profile(path: Path, tile_id: str) -> None:
    with rasterio.open(path) as dataset:
        if (
            dataset.count != _TILE_BAND_COUNT
            or dataset.width != _TILE_WIDTH_PIXELS
            or dataset.height != _TILE_HEIGHT_PIXELS
            or dataset.dtypes != ("uint8",)
            or dataset.crs is None
            or dataset.crs.to_epsg() != _TILE_CRS_EPSG
            or dataset.nodata != _TILE_NODATA
        ):
            raise ValueError(
                f"WorldCover raster profile is not the pinned 10 m class map: {tile_id}"
            )
        match = _TILE_RE.fullmatch(tile_id)
        if match is None:
            raise ValueError(f"invalid WorldCover tile ID {tile_id!r}")
        lat = int(match.group(2)) * (1 if match.group(1) == "N" else -1)
        lon = int(match.group(4)) * (1 if match.group(3) == "E" else -1)
        bounds = dataset.bounds
        pixel_size = _TILE_GRID_DEGREES / _TILE_WIDTH_PIXELS
        expected_transform = (
            pixel_size,
            0.0,
            float(lon),
            0.0,
            -pixel_size,
            float(lat + _TILE_GRID_DEGREES),
        )
        if any(
            not math.isclose(actual, want, abs_tol=1e-12)
            for actual, want in zip(dataset.transform[:6], expected_transform, strict=True)
        ):
            raise ValueError(
                f"WorldCover tile affine transform differs from its grid ID: {tile_id}"
            )
        expected = (
            float(lon),
            float(lat),
            float(lon + _TILE_GRID_DEGREES),
            float(lat + _TILE_GRID_DEGREES),
        )
        if any(
            not math.isclose(actual, want, abs_tol=1e-6)
            for actual, want in zip(bounds, expected, strict=True)
        ):
            raise ValueError(f"WorldCover tile extent differs from its catalog ID: {tile_id}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _text(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"WorldCover tile catalog field {key!r} must be non-empty text")
    return value


def _positive_int(value: object, field: str) -> int:
    result = _non_negative_int(value, field)
    if result <= 0:
        raise ValueError(f"{field} must be positive")
    return result


def _non_negative_int(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


def _require_sha256(value: str, field: str) -> None:
    if not _SHA256_RE.fullmatch(value):
        raise ValueError(f"{field} SHA-256 must be lowercase hexadecimal")
