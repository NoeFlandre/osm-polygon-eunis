"""Stage, open and cache EEA reference groups for geometry processing."""

from __future__ import annotations

import atexit
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import unquote

import httpx

from ._protocols import StreamClient
from .eea import EeaGroup, RemoteAsset, download_asset
from .fileio import DOWNLOAD_TIMEOUT
from .reference import GeoPackageReference, RasterLayer, RasterReference
from .transform import (
    OverlapReference,
)

_RASTER_GROUP_BATCH_SIZE = 2


_WORKER_REFERENCE_STACKS: dict[tuple[str, int, int, tuple[str, ...]], ExitStack] = {}


_WORKER_REFERENCES: dict[tuple[str, int, int, tuple[str, ...]], tuple[OverlapReference, ...]] = {}


def _asset_filename(asset: RemoteAsset) -> str:
    suffix = Path(unquote(asset.path)).suffix.lower()
    stem = asset.code or Path(unquote(asset.path)).stem or "reference"
    return f"{stem}{suffix}"


def _asset_key(group: EeaGroup, asset: RemoteAsset) -> str:
    return f"{group.record_id}:{asset.path}"


@contextmanager
def open_reference_group(
    group: EeaGroup,
    directory: Path,
    *,
    threshold: int,
    checksums: dict[str, str] | None = None,
    client: StreamClient | None = None,
) -> Iterator[OverlapReference]:
    """Download one EEA group and expose a closed, exact-overlap reference."""

    directory.mkdir(parents=True, exist_ok=True)
    with (
        _http_client(client) as reusable_client,
        _reference_group_with_client(
            reusable_client,
            group,
            directory,
            threshold,
            checksums,
        ) as reference,
    ):
        yield reference


@contextmanager
def _http_client(client: StreamClient | None) -> Iterator[StreamClient]:
    if client is not None:
        yield client
        return
    with httpx.Client(follow_redirects=True, timeout=DOWNLOAD_TIMEOUT) as owned_client:
        yield owned_client


@contextmanager
def _reference_group_with_client(
    client: StreamClient,
    group: EeaGroup,
    directory: Path,
    threshold: int,
    checksums: dict[str, str] | None,
) -> Iterator[OverlapReference]:
    if group.raster_assets:
        with _raster_group_reference(client, group, directory, threshold, checksums) as reference:
            yield reference
        return
    if group.vector_asset is None:
        raise ValueError(f"EEA group has no reference asset: {group.record_id}")
    with _vector_group_reference(client, group, directory, threshold, checksums) as reference:
        yield reference


def _stage_reference_group(
    client: StreamClient,
    group: EeaGroup,
    directory: Path,
    checksums: dict[str, str],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for asset in _reference_assets_for_staging(group):
        path = directory / _asset_filename(asset)
        checksums[_asset_key(group, asset)] = download_asset(client, asset, path)


def _reference_assets_for_staging(group: EeaGroup) -> tuple[RemoteAsset, ...]:
    if group.raster_assets:
        return group.raster_assets
    if group.vector_asset is None:
        raise ValueError(f"EEA group has no reference asset: {group.record_id}")
    return (group.vector_asset,)


@contextmanager
def _open_local_reference_group(
    group: EeaGroup,
    directory: Path,
    threshold: int,
) -> Iterator[OverlapReference]:
    if group.raster_assets:
        with _open_local_raster_reference(group, directory, threshold) as reference:
            yield reference
        return
    with _open_local_vector_reference(group, directory, threshold) as reference:
        yield reference


@contextmanager
def _open_local_raster_reference(
    group: EeaGroup,
    directory: Path,
    threshold: int,
) -> Iterator[RasterReference]:
    layers = tuple(
        RasterLayer(
            code=asset.code or "",
            name=asset.name or "",
            path=directory / _asset_filename(asset),
            source_version=asset.source_version,
        )
        for asset in group.raster_assets
    )
    with RasterReference(layers, threshold=threshold) as reference:
        yield reference


@contextmanager
def _open_local_vector_reference(
    group: EeaGroup,
    directory: Path,
    threshold: int,
) -> Iterator[GeoPackageReference]:
    if group.vector_asset is None:
        raise ValueError(f"EEA group has no reference asset: {group.record_id}")
    asset = group.vector_asset
    with GeoPackageReference(
        directory / _asset_filename(asset),
        dict(group.labels),
        source_version=asset.source_version,
        threshold=threshold,
    ) as reference:
        yield reference


def _reference_group_directory(root: Path, index: int, group: EeaGroup) -> Path:
    return root / f"{index:02d}-{group.record_id[:8]}"


@contextmanager
def _stage_reference_groups(
    groups: tuple[EeaGroup, ...],
    *,
    workdir: Path,
    checksums: dict[str, str],
    client: StreamClient,
) -> Iterator[Path]:
    first_record = groups[0].record_id[:8]
    with TemporaryDirectory(
        dir=workdir,
        prefix=f"reference-{first_record}-",
    ) as directory:
        root = Path(directory)
        for index, group in enumerate(groups):
            _stage_reference_group(
                client,
                group,
                _reference_group_directory(root, index, group),
                checksums,
            )
        yield root


@contextmanager
def _stage_reference_batch(
    groups: tuple[EeaGroup, ...],
    *,
    workdir: Path,
    checksums: dict[str, str],
    client: StreamClient,
) -> Iterator[Path]:
    """Stage one reference batch for the legacy single-batch worker path."""

    with _stage_reference_groups(
        groups,
        workdir=workdir,
        checksums=checksums,
        client=client,
    ) as root:
        yield root


@contextmanager
def _open_local_reference_batch(
    groups: tuple[EeaGroup, ...],
    root: Path,
    threshold: int,
    *,
    start_index: int = 0,
) -> Iterator[tuple[OverlapReference, ...]]:
    with ExitStack() as stack:
        references = tuple(
            stack.enter_context(
                _open_local_reference_group(
                    group,
                    _reference_group_directory(root, start_index + index, group),
                    threshold,
                )
            )
            for index, group in enumerate(groups)
        )
        yield references


def _worker_reference_batch(
    groups: tuple[EeaGroup, ...],
    root: Path,
    threshold: int,
    start_index: int,
) -> tuple[OverlapReference, ...]:
    """Reuse one opened reference batch for the lifetime of a worker process."""

    key = (str(root), start_index, threshold, tuple(group.record_id for group in groups))
    cached = _WORKER_REFERENCES.get(key)
    if cached is not None:
        return cached
    stack = ExitStack()
    try:
        references = tuple(
            stack.enter_context(
                _open_local_reference_group(
                    group,
                    _reference_group_directory(root, start_index + index, group),
                    threshold,
                )
            )
            for index, group in enumerate(groups)
        )
    except BaseException:
        stack.close()
        raise
    _WORKER_REFERENCE_STACKS[key] = stack
    _WORKER_REFERENCES[key] = references
    return references


def _close_worker_reference_cache() -> None:
    for stack in _WORKER_REFERENCE_STACKS.values():
        stack.close()
    _WORKER_REFERENCE_STACKS.clear()
    _WORKER_REFERENCES.clear()


atexit.register(_close_worker_reference_cache)


@contextmanager
def _raster_group_reference(
    client: StreamClient,
    group: EeaGroup,
    directory: Path,
    threshold: int,
    checksums: dict[str, str] | None,
) -> Iterator[RasterReference]:
    layers = []
    for asset in group.raster_assets:
        path = directory / _asset_filename(asset)
        digest = download_asset(client, asset, path)
        if checksums is not None:
            checksums[_asset_key(group, asset)] = digest
        layers.append(
            RasterLayer(
                code=asset.code or "",
                name=asset.name or "",
                path=path,
                source_version=asset.source_version,
            )
        )
    with RasterReference(tuple(layers), threshold=threshold) as reference:
        yield reference


@contextmanager
def _vector_group_reference(
    client: StreamClient,
    group: EeaGroup,
    directory: Path,
    threshold: int,
    checksums: dict[str, str] | None,
) -> Iterator[GeoPackageReference]:
    assert group.vector_asset is not None
    asset = group.vector_asset
    path = directory / _asset_filename(asset)
    digest = download_asset(client, asset, path)
    if checksums is not None:
        checksums[_asset_key(group, asset)] = digest
    with GeoPackageReference(
        path,
        dict(group.labels),
        source_version=asset.source_version,
        threshold=threshold,
    ) as reference:
        yield reference


def _reference_group_batches(
    groups: tuple[EeaGroup, ...],
) -> tuple[tuple[EeaGroup, ...], ...]:
    """Bound raster groups while coalescing adjacent vector groups."""

    batches: list[list[EeaGroup]] = []
    for group in groups:
        if _starts_reference_batch(batches, group):
            batches.append([])
        batches[-1].append(group)
    return tuple(tuple(batch) for batch in batches)


def _indexed_reference_group_batches(
    groups: tuple[EeaGroup, ...],
) -> tuple[tuple[int, tuple[EeaGroup, ...]], ...]:
    indexed: list[tuple[int, tuple[EeaGroup, ...]]] = []
    start_index = 0
    for batch in _reference_group_batches(groups):
        indexed.append((start_index, batch))
        start_index += len(batch)
    return tuple(indexed)


def _starts_reference_batch(batches: list[list[EeaGroup]], group: EeaGroup) -> bool:
    if not batches:
        return True
    current = batches[-1]
    if group.raster_assets:
        return not current[0].raster_assets or len(current) >= _RASTER_GROUP_BATCH_SIZE
    return bool(current[0].raster_assets)


@contextmanager
def _open_reference_batch(
    groups: tuple[EeaGroup, ...],
    *,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    client: StreamClient,
) -> Iterator[tuple[OverlapReference, ...]]:
    first_record = groups[0].record_id[:8]
    with (
        TemporaryDirectory(
            dir=workdir,
            prefix=f"reference-{first_record}-",
        ) as directory,
        ExitStack() as stack,
    ):
        references = tuple(
            stack.enter_context(
                open_reference_group(
                    group,
                    _reference_group_directory(Path(directory), index, group),
                    threshold=threshold,
                    checksums=checksums,
                    client=client,
                )
            )
            for index, group in enumerate(groups)
        )
        yield references
