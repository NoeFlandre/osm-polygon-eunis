"""Storage-bounded orchestration for the three EUNIS dataset releases."""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Iterator, Mapping
from concurrent.futures import ProcessPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import unquote

import httpx
from huggingface_hub import HfApi

from .cards import CardArtifacts, DatasetCardAccumulator
from .domain import EunisResult
from .eea import EeaGroup, RemoteAsset, download_asset, resolve_config
from .publish import (
    ShardExpectation,
    VerificationReceipt,
    build_manifest,
    duplicate_source,
    upload_manifest,
    upload_replacement,
    verify_dataset,
)
from .reference import GeoPackageReference, RasterLayer, RasterReference
from .sources import (
    DatasetSpec,
    capture_revision,
    dataset_spec,
    download_to_temp,
    list_repo_files,
    matches_layout,
    pair_region_paths,
)
from .transform import (
    OverlapReference,
    append_label_sidecar,
    build_label_map,
    update_label_sidecar,
)

Progress = Callable[[Mapping[str, object]], None]
_RASTER_GROUP_BATCH_SIZE = 2
_SOURCE_WORKERS = 4


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


@dataclass(frozen=True, slots=True)
class _LinkOutput:
    path: str
    source: Path
    output: Path
    expectation: ShardExpectation


@dataclass(frozen=True, slots=True)
class _ExistingManifest:
    """A target manifest and the immutable target revision that contains it."""

    revision: str
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _GeometryChunk:
    """Pickleable work unit for one bounded reference batch."""

    groups: tuple[EeaGroup, ...]
    reference_directory: Path
    plans: tuple[DatasetPlan, ...]
    jobs: tuple[tuple[str, str], ...]
    sidecar_root: Path
    source_root: Path
    threshold: int
    batch_size: int
    endpoint: str
    token: str | bool | None


def _settings(config_path: Path) -> tuple[str, str, int, Mapping[str, object]]:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("reference config must be an object")
    return _validated_settings(cast(Mapping[str, object], config))


def _validated_settings(
    config: Mapping[str, object],
) -> tuple[str, str, int, Mapping[str, object]]:
    source_version = _required_setting(config, "source_version")
    crs = _required_setting(config, "crs")
    threshold = _threshold_setting(config)
    return source_version, crs, threshold, config


def _required_setting(config: Mapping[str, object], field: str) -> str:
    value = config.get(field)
    if not isinstance(value, str) or not value:
        raise ValueError(f"reference config is missing {field}")
    return value


def _threshold_setting(config: Mapping[str, object]) -> int:
    value = config.get("threshold")
    if not isinstance(value, int) or value < 0:
        raise ValueError("reference config has invalid threshold")
    return value


def _sidecar_path(root: Path, spec: DatasetSpec, source_path: str) -> Path:
    return root / spec.name / f"{source_path.replace('/', '__')}.labels.parquet"


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
    client: Any | None = None,
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
def _http_client(client: Any | None) -> Iterator[Any]:
    if client is not None:
        yield client
        return
    with httpx.Client(follow_redirects=True, timeout=None) as owned_client:
        yield owned_client


@contextmanager
def _reference_group_with_client(
    client: Any,
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
    client: Any,
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
def _stage_reference_batch(
    groups: tuple[EeaGroup, ...],
    *,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    client: Any,
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
def _open_local_reference_batch(
    groups: tuple[EeaGroup, ...],
    root: Path,
    threshold: int,
) -> Iterator[tuple[OverlapReference, ...]]:
    with ExitStack() as stack:
        references = tuple(
            stack.enter_context(
                _open_local_reference_group(
                    group,
                    _reference_group_directory(root, index, group),
                    threshold,
                )
            )
            for index, group in enumerate(groups)
        )
        yield references


@contextmanager
def _raster_group_reference(
    client: Any,
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
    client: Any,
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


def process_geometry_paths(
    api: Any,
    plan: DatasetPlan,
    *,
    reference: OverlapReference,
    sidecar_root: Path,
    source_root: Path,
    batch_size: int,
    progress: Progress | None = None,
    http_client: Any | None = None,
    retain_source: bool = False,
) -> None:
    """Merge one reference group into every geometry shard's compact sidecar."""

    with _http_client(http_client) as reusable_client:
        source_root.mkdir(parents=True, exist_ok=True)
        for source_path in plan.geometry_paths:
            _process_geometry_path(
                api,
                plan,
                source_path=source_path,
                references=(reference,),
                sidecar_root=sidecar_root,
                source_root=source_root,
                batch_size=batch_size,
                progress=progress,
                http_client=reusable_client,
                retain_source=retain_source,
            )


def _process_geometry_path(
    api: Any,
    plan: DatasetPlan,
    *,
    source_path: str,
    references: tuple[OverlapReference, ...],
    sidecar_root: Path,
    source_root: Path,
    batch_size: int,
    progress: Progress | None,
    http_client: Any,
    retain_source: bool,
) -> None:
    local_source = _download_geometry_source(
        api,
        plan,
        source_path,
        source_root,
        retain_source=retain_source,
        client=http_client,
    )
    sidecar = _sidecar_path(sidecar_root, plan.spec, source_path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    next_sidecar = sidecar.with_name(f"{sidecar.name}.next")
    update_label_sidecar(
        local_source,
        next_sidecar,
        references=references,
        current=sidecar if sidecar.is_file() else None,
        batch_size=batch_size,
    )
    next_sidecar.replace(sidecar)
    if not retain_source:
        local_source.unlink(missing_ok=True)
    if progress is not None:
        progress({"event": "sidecar_updated", "dataset": plan.spec.name, "path": source_path})


def _commit_id(result: Any) -> str | None:
    for attribute in ("oid", "commit_oid"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _advance_commit(api: Any, target_repo: str, result: Any) -> str:
    return _commit_id(result) or capture_revision(api, target_repo)


def _cached_geometry_path(root: Path, plan: DatasetPlan, source_path: str) -> Path:
    return root / plan.spec.name / source_path.replace("/", "__")


def _download_geometry_source(
    api: Any,
    plan: DatasetPlan,
    source_path: str,
    source_root: Path,
    *,
    retain_source: bool,
    client: Any,
) -> Path:
    directory = source_root / plan.spec.name if retain_source else source_root
    cached = _cached_geometry_path(source_root, plan, source_path)
    if retain_source and cached.is_file():
        return cached
    return download_to_temp(
        api,
        plan.spec.source_repo,
        source_path,
        plan.source_revision,
        directory,
        client=client,
    )


def finalize_dataset(
    api: Any,
    plan: DatasetPlan,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    progress: Progress | None = None,
    http_client: Any | None = None,
    card: DatasetCardAccumulator | None = None,
    source_cache_root: Path | None = None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    """Append labels, upload changed shards, and clean successful staging files."""

    with _http_client(http_client) as reusable_client:
        return _finalize_dataset_with_client(
            api,
            plan,
            sidecar_root=sidecar_root,
            local_root=local_root,
            batch_size=batch_size,
            parent_commit=parent_commit,
            progress=progress,
            http_client=reusable_client,
            card=card,
            source_cache_root=source_cache_root,
        )


def _finalize_dataset_with_client(
    api: Any,
    plan: DatasetPlan,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    progress: Progress | None,
    http_client: Any,
    card: DatasetCardAccumulator | None,
    source_cache_root: Path | None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    local_root.mkdir(parents=True, exist_ok=True)
    card_accumulator = card if card is not None else DatasetCardAccumulator()
    link_by_filename = {Path(link_path).name: link_path for link_path in plan.link_paths}
    expectations: list[ShardExpectation] = []
    current_commit = parent_commit
    for geometry_path in plan.geometry_paths:
        shard_expectations, current_commit = _finalize_shard(
            api,
            plan,
            geometry_path,
            link_by_filename.get(Path(geometry_path).name),
            sidecar_root=sidecar_root,
            local_root=local_root,
            batch_size=batch_size,
            parent_commit=current_commit,
            http_client=http_client,
            card=card_accumulator,
            source_cache_root=source_cache_root,
        )
        expectations.extend(shard_expectations)
        if progress is not None:
            progress({"event": "shards_uploaded", "dataset": plan.spec.name, "path": geometry_path})
    return tuple(expectations), current_commit


def _finalize_shard(
    api: Any,
    plan: DatasetPlan,
    geometry_path: str,
    link_path: str | None,
    *,
    sidecar_root: Path,
    local_root: Path,
    batch_size: int,
    parent_commit: str,
    http_client: Any,
    card: DatasetCardAccumulator,
    source_cache_root: Path | None,
) -> tuple[tuple[ShardExpectation, ...], str]:
    local_source, reused_source = _final_source(
        api,
        plan,
        geometry_path,
        source_cache_root=source_cache_root,
        local_root=local_root,
        http_client=http_client,
    )
    sidecar = _sidecar_path(sidecar_root, plan.spec, geometry_path)
    if not sidecar.is_file():
        raise FileNotFoundError(f"missing completed label sidecar: {sidecar}")
    local_output = local_root / f"{geometry_path.replace('/', '__')}.enriched.parquet"
    local_output.unlink(missing_ok=True)
    rows = append_label_sidecar(
        local_source,
        sidecar,
        local_output,
        batch_size=batch_size,
        observe=card.observe,
    )
    geometry_expectation = ShardExpectation(geometry_path, rows, _schema_signature(local_output))
    link_output = _build_link_output(
        api,
        plan,
        link_path,
        local_source,
        sidecar,
        local_root,
        batch_size,
        http_client,
    )
    current_commit = _upload_shard_outputs(
        api,
        plan,
        geometry_path,
        local_output,
        link_output,
        parent_commit,
    )
    _cleanup_shard(
        local_source,
        local_output,
        sidecar,
        link_output,
        delete_source=not reused_source,
    )
    expectations = [geometry_expectation]
    if link_output is not None:
        expectations.append(link_output.expectation)
    return tuple(expectations), current_commit


def _final_source(
    api: Any,
    plan: DatasetPlan,
    geometry_path: str,
    *,
    source_cache_root: Path | None,
    local_root: Path,
    http_client: Any,
) -> tuple[Path, bool]:
    if source_cache_root is not None:
        cached = _cached_geometry_path(source_cache_root, plan, geometry_path)
        if cached.is_file():
            return cached, True
    return (
        download_to_temp(
            api,
            plan.spec.source_repo,
            geometry_path,
            plan.source_revision,
            local_root,
            client=http_client,
        ),
        False,
    )


def _upload_shard_outputs(
    api: Any,
    plan: DatasetPlan,
    geometry_path: str,
    local_output: Path,
    link_output: _LinkOutput | None,
    parent_commit: str,
) -> str:
    current_commit = _upload_file(
        api,
        plan.spec.output_repo,
        geometry_path,
        local_output,
        parent_commit,
    )
    if link_output is None:
        return current_commit
    return _upload_file(
        api,
        plan.spec.output_repo,
        link_output.path,
        link_output.output,
        current_commit,
    )


def _build_link_output(
    api: Any,
    plan: DatasetPlan,
    link_path: str | None,
    local_source: Path,
    sidecar: Path,
    local_root: Path,
    batch_size: int,
    http_client: Any,
) -> _LinkOutput | None:
    if link_path is None:
        return None
    labels = build_label_map(local_source, sidecar, batch_size=batch_size)
    local_link_source = download_to_temp(
        api,
        plan.spec.source_repo,
        link_path,
        plan.source_revision,
        local_root,
        client=http_client,
    )
    local_link_output = local_root / f"{link_path.replace('/', '__')}.enriched.parquet"
    local_link_output.unlink(missing_ok=True)
    link_rows = _enrich_link(
        local_link_source,
        local_link_output,
        labels,
        batch_size=batch_size,
    )
    return _LinkOutput(
        link_path,
        local_link_source,
        local_link_output,
        ShardExpectation(link_path, link_rows, _schema_signature(local_link_output)),
    )


def _upload_file(
    api: Any,
    target_repo: str,
    path: str,
    local_path: Path,
    parent_commit: str,
    commit_message: str | None = None,
) -> str:
    if commit_message is None:
        result = upload_replacement(
            api,
            target_repo,
            path,
            local_path,
            parent_commit=parent_commit,
        )
    else:
        result = upload_replacement(
            api,
            target_repo,
            path,
            local_path,
            parent_commit=parent_commit,
            commit_message=commit_message,
        )
    return _advance_commit(api, target_repo, result)


def _cleanup_shard(
    local_source: Path,
    local_output: Path,
    sidecar: Path,
    link_output: _LinkOutput | None,
    *,
    delete_source: bool = True,
) -> None:
    if delete_source:
        local_source.unlink(missing_ok=True)
    local_output.unlink(missing_ok=True)
    sidecar.unlink(missing_ok=True)
    if link_output is not None:
        link_output.source.unlink(missing_ok=True)
        link_output.output.unlink(missing_ok=True)


def _schema_signature(path: Path) -> str:
    from .publish import parquet_signature

    return parquet_signature(path)[1]


def _enrich_link(
    source: Path,
    destination: Path,
    labels: Mapping[str, EunisResult],
    *,
    batch_size: int,
) -> int:
    from .transform import enrich_link_shard

    return enrich_link_shard(
        source,
        destination,
        labels_by_polygon_id=labels,
        batch_size=batch_size,
    )


def plan_datasets(api: Any) -> tuple[DatasetPlan, ...]:
    """Capture source commits and validate all declared dataset layouts."""

    return tuple(_plan_dataset(api, name) for name in ("website", "wikidata", "description"))


def _plan_dataset(api: Any, name: str) -> DatasetPlan:
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


def _reference_manifest(
    groups: tuple[EeaGroup, ...],
    checksums: Mapping[str, str],
    *,
    source_version: str,
    crs: str,
    threshold: int,
    config: Mapping[str, object],
) -> dict[str, object]:
    assets: list[dict[str, object]] = []
    for group in groups:
        group_assets = [*group.raster_assets]
        if group.vector_asset is not None:
            group_assets.append(group.vector_asset)
        for asset in group_assets:
            assets.append(
                {
                    "record_id": asset.record_id,
                    "path": asset.path,
                    "url": asset.url,
                    "size": asset.size,
                    "etag": asset.etag,
                    "code": asset.code,
                    "name": asset.name,
                    "sha256": checksums.get(_asset_key(group, asset)),
                }
            )
    return {
        "source_version": source_version,
        "crs": crs,
        "threshold": threshold,
        "classification_record": config.get("classification_record"),
        "assets": sorted(assets, key=lambda asset: (str(asset["record_id"]), str(asset["path"]))),
    }


def _reference_identity(reference: Mapping[str, object]) -> Mapping[str, object]:
    """Remove downloaded-byte hashes before comparing pinned reference metadata."""

    normalized_assets = _reference_assets(reference.get("assets"))
    if normalized_assets is None:
        return reference
    normalized = dict(reference)
    normalized["assets"] = normalized_assets
    return normalized


def _reference_assets(value: object) -> list[dict[str, object]] | None:
    if not isinstance(value, list):
        return None
    return [_reference_asset(asset) for asset in value if isinstance(asset, Mapping)]


def _reference_asset(asset: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in asset.items() if key != "sha256"}


def _manifest_matches_inputs(
    plan: DatasetPlan,
    manifest: Mapping[str, object],
    reference: Mapping[str, object],
) -> bool:
    """Check whether a manifest describes exactly the current pinned inputs."""

    return all(
        (
            manifest.get("manifest_version") == 3,
            manifest.get("source_repo") == plan.spec.source_repo,
            manifest.get("target_repo") == plan.spec.output_repo,
            manifest.get("source_revision") == plan.source_revision,
            manifest.get("source_paths") == list(plan.source_files),
            _reference_identity(_mapping_field(manifest, "reference"))
            == _reference_identity(reference),
        )
    )


def _mapping_field(payload: Mapping[str, object], field: str) -> Mapping[str, object]:
    value = payload.get(field)
    return value if isinstance(value, Mapping) else {}


def _load_existing_manifest(
    api: Any,
    plan: DatasetPlan,
    directory: Path,
    client: Any,
) -> _ExistingManifest | None:
    """Load a tiny target manifest without using the persistent Hub cache."""

    target_revision = capture_revision(api, plan.spec.output_repo)
    paths = {entry.path for entry in list_repo_files(api, plan.spec.output_repo, target_revision)}
    if "eunis/manifest.json" not in paths:
        return None
    local = download_to_temp(
        api,
        plan.spec.output_repo,
        "eunis/manifest.json",
        target_revision,
        directory,
        client=client,
    )
    try:
        payload = json.loads(local.read_text(encoding="utf-8"))
    finally:
        local.unlink(missing_ok=True)
    if not isinstance(payload, Mapping):
        return None
    return _ExistingManifest(target_revision, payload)


def _manifest_expectations(manifest: Mapping[str, object]) -> tuple[ShardExpectation, ...] | None:
    fields = _manifest_table_fields(manifest)
    if fields is None:
        return None
    rows, schemas = fields
    try:
        return tuple(
            _manifest_expectation(path, rows[path], schemas[path])
            for path in sorted(rows, key=str)
        )
    except (KeyError, TypeError, ValueError):
        return None


def _manifest_table_fields(
    manifest: Mapping[str, object],
) -> tuple[Mapping[object, object], Mapping[object, object]] | None:
    rows = manifest.get("rows_by_path")
    schemas = manifest.get("schema_by_path")
    if not isinstance(rows, Mapping) or not isinstance(schemas, Mapping):
        return None
    if set(rows) != set(schemas):
        return None
    return rows, schemas


def _manifest_expectation(path: object, rows: Any, schema: Any) -> ShardExpectation:
    if not isinstance(path, str):
        raise TypeError("manifest shard path must be a string")
    return ShardExpectation(path, int(rows), str(schema))


def _manifest_artifacts(manifest: Mapping[str, object]) -> Mapping[str, str] | None:
    card = manifest.get("card")
    if not isinstance(card, Mapping):
        return None
    artifacts: dict[str, str] = {}
    for path_field, hash_field in (
        ("readme_path", "readme_sha256"),
        ("map_path", "map_sha256"),
    ):
        path = card.get(path_field)
        digest = card.get(hash_field)
        if not isinstance(path, str) or not isinstance(digest, str):
            return None
        artifacts[path] = digest
    return artifacts


def _required_manifest_paths(
    manifest: Mapping[str, object],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return (
        _required_string_list(manifest, "changed_paths"),
        _required_string_list(manifest, "added_paths"),
    )


def _required_string_list(manifest: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = manifest.get(field)
    if not isinstance(value, list):
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    paths = tuple(value)
    if not all(isinstance(path, str) for path in paths):
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    return cast(tuple[str, ...], paths)


def _required_no_op_parts(
    manifest: Mapping[str, object],
) -> tuple[tuple[ShardExpectation, ...], Mapping[str, str], tuple[str, ...], tuple[str, ...]]:
    expectations = _manifest_expectations(manifest)
    artifacts = _manifest_artifacts(manifest)
    if expectations is None or artifacts is None:
        raise ValueError("matching EUNIS manifest is incomplete for no-op verification")
    changed_paths, added_paths = _required_manifest_paths(manifest)
    return expectations, artifacts, changed_paths, added_paths


def _verify_no_op_dataset(
    api: Any,
    plan: DatasetPlan,
    existing: _ExistingManifest,
    *,
    workdir: Path,
    client: Any,
) -> DatasetReceipt:
    manifest = existing.manifest
    expectations, artifacts, changed_paths, added_paths = _required_no_op_parts(manifest)
    shared_blobs = _shared_blobs(api, plan, set(changed_paths))
    verification = _verify_final_dataset(
        api,
        plan,
        expectations=expectations,
        added_paths=tuple(added_paths),
        expected_shared_blobs=shared_blobs,
        expected_manifest=manifest,
        expected_artifacts=artifacts,
        temp_dir=workdir / "verify-no-op",
        http_client=client,
    )
    return DatasetReceipt(plan, expectations, verification, no_op=True)


def _load_existing_manifests(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    directory: Path,
    client: Any,
) -> tuple[_ExistingManifest | None, ...]:
    return tuple(_load_existing_manifest(api, plan, directory, client) for plan in plans)


def _matches_existing_manifest(
    plan: DatasetPlan,
    existing: _ExistingManifest | None,
    reference: Mapping[str, object],
) -> bool:
    return existing is not None and _manifest_matches_inputs(plan, existing.manifest, reference)


def _compatible_manifests(
    plans: tuple[DatasetPlan, ...],
    existing: tuple[_ExistingManifest | None, ...],
    reference: Mapping[str, object],
) -> bool:
    return all(
        _matches_existing_manifest(plan, item, reference)
        for plan, item in zip(plans, existing, strict=True)
    )


def _no_op_receipts(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    existing: tuple[_ExistingManifest | None, ...],
    *,
    workdir: Path,
    client: Any,
) -> tuple[DatasetReceipt, ...]:
    return tuple(
        _verify_no_op_dataset(
            api,
            plan,
            cast(_ExistingManifest, item),
            workdir=workdir,
            client=client,
        )
        for plan, item in zip(plans, existing, strict=True)
    )


def _try_no_op_release(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    reference: Mapping[str, object],
    *,
    workdir: Path,
    client: Any,
    progress: Progress | None,
) -> ReleaseReceipt | None:
    existing = _load_existing_manifests(api, plans, workdir / "noop", client)
    if not _compatible_manifests(plans, existing, reference):
        return None
    receipts = _no_op_receipts(api, plans, existing, workdir=workdir, client=client)
    if progress is not None:
        for receipt in receipts:
            progress({"event": "dataset_no_op", "dataset": receipt.plan.spec.name})
    manifest = receipts[0].verification.manifest or {}
    return ReleaseReceipt(receipts, _mapping_field(manifest, "reference"))


def run_release(
    api: Any,
    *,
    reference_config: Path,
    workdir: Path,
    batch_size: int,
    token: str | None = None,
    progress: Progress | None = None,
) -> ReleaseReceipt:
    """Run, publish, and independently verify all three datasets."""

    source_version, crs, threshold, config = _settings(reference_config)
    workdir.mkdir(parents=True, exist_ok=True)
    plans = plan_datasets(api)
    _duplicate_outputs(api, plans, token)
    groups = resolve_config(reference_config)
    sidecar_root = workdir / "sidecars"
    checksums: dict[str, str] = {}
    with _source_cache(workdir) as source_root, _http_client(None) as reusable_client:
        reference_identity = _reference_manifest(
            groups,
            {},
            source_version=source_version,
            crs=crs,
            threshold=threshold,
            config=config,
        )
        no_op_receipt = _try_no_op_release(
            api,
            plans,
            reference_identity,
            workdir=workdir,
            client=reusable_client,
            progress=progress,
        )
        if no_op_receipt is not None:
            return no_op_receipt
        _process_reference_groups(
            api,
            plans,
            groups,
            sidecar_root=sidecar_root,
            source_root=source_root,
            workdir=workdir,
            threshold=threshold,
            checksums=checksums,
            batch_size=batch_size,
            progress=progress,
            http_client=reusable_client,
            parallelism=_SOURCE_WORKERS,
        )
        reference_info = _reference_manifest(
            groups,
            checksums,
            source_version=source_version,
            crs=crs,
            threshold=threshold,
            config=config,
        )
        receipts = tuple(
            _finalize_plan(
                api,
                plan,
                sidecar_root=sidecar_root,
                workdir=workdir,
                batch_size=batch_size,
                reference_info=reference_info,
                progress=progress,
                http_client=reusable_client,
                source_cache_root=source_root,
            )
            for plan in plans
        )
    return ReleaseReceipt(receipts, reference_info)


def _duplicate_outputs(api: Any, plans: tuple[DatasetPlan, ...], token: str | None) -> None:
    for plan in plans:
        duplicate_source(
            api,
            plan.spec.source_repo,
            plan.spec.output_repo,
            token=token,
        )


def _process_reference_groups(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    groups: tuple[EeaGroup, ...],
    *,
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: Any,
    parallelism: int = 1,
) -> None:
    """Process bounded reference batches while reusing downloaded sources."""

    for batch in _reference_group_batches(groups):
        if parallelism > 1:
            _process_reference_batch_parallel(
                api,
                plans,
                groups=batch,
                sidecar_root=sidecar_root,
                source_root=source_root,
                workdir=workdir,
                threshold=threshold,
                checksums=checksums,
                batch_size=batch_size,
                progress=progress,
                http_client=http_client,
                parallelism=parallelism,
            )
            continue
        _process_reference_batch_serial(
            api,
            plans,
            groups=batch,
            sidecar_root=sidecar_root,
            source_root=source_root,
            workdir=workdir,
            threshold=threshold,
            checksums=checksums,
            batch_size=batch_size,
            progress=progress,
            http_client=http_client,
        )


def _process_reference_batch_serial(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    *,
    groups: tuple[EeaGroup, ...],
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: Any,
) -> None:
    with _open_reference_batch(
        groups,
        workdir=workdir,
        threshold=threshold,
        checksums=checksums,
        client=http_client,
    ) as references:
        for plan in plans:
            for source_path in plan.geometry_paths:
                _process_geometry_path(
                    api,
                    plan,
                    source_path=source_path,
                    references=references,
                    sidecar_root=sidecar_root,
                    source_root=source_root,
                    batch_size=batch_size,
                    progress=progress,
                    http_client=http_client,
                    retain_source=True,
                )


def _process_reference_batch_parallel(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    *,
    groups: tuple[EeaGroup, ...],
    sidecar_root: Path,
    source_root: Path,
    workdir: Path,
    threshold: int,
    checksums: dict[str, str],
    batch_size: int,
    progress: Progress | None,
    http_client: Any,
    parallelism: int,
) -> None:
    jobs = _geometry_jobs(plans)
    if not jobs:
        return
    with _stage_reference_batch(
        groups,
        workdir=workdir,
        threshold=threshold,
        checksums=checksums,
        client=http_client,
    ) as reference_directory:
        work = _geometry_work_units(
            api,
            plans,
            groups,
            jobs,
            reference_directory=reference_directory,
            sidecar_root=sidecar_root,
            source_root=source_root,
            threshold=threshold,
            batch_size=batch_size,
            parallelism=parallelism,
        )
        _run_geometry_workers(work, progress)


def _geometry_jobs(plans: tuple[DatasetPlan, ...]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (plan.spec.name, source_path)
        for plan in plans
        for source_path in plan.geometry_paths
    )


def _geometry_work_units(
    api: Any,
    plans: tuple[DatasetPlan, ...],
    groups: tuple[EeaGroup, ...],
    jobs: tuple[tuple[str, str], ...],
    *,
    reference_directory: Path,
    sidecar_root: Path,
    source_root: Path,
    threshold: int,
    batch_size: int,
    parallelism: int,
) -> tuple[_GeometryChunk, ...]:
    endpoint = str(getattr(api, "endpoint", None) or "https://huggingface.co")
    token = getattr(api, "token", None)
    return tuple(
        _GeometryChunk(
            groups=groups,
            reference_directory=reference_directory,
            plans=plans,
            jobs=chunk,
            sidecar_root=sidecar_root,
            source_root=source_root,
            threshold=threshold,
            batch_size=batch_size,
            endpoint=endpoint,
            token=token,
        )
        for chunk in _geometry_chunks(jobs, parallelism)
    )


def _run_geometry_workers(
    work: tuple[_GeometryChunk, ...],
    progress: Progress | None,
) -> None:
    with ProcessPoolExecutor(max_workers=len(work)) as executor:
        for completed in executor.map(_process_geometry_chunk, work):
            _report_completed_geometry(completed, progress)


def _report_completed_geometry(
    completed: tuple[tuple[str, str], ...],
    progress: Progress | None,
) -> None:
    if progress is None:
        return
    for dataset, source_path in completed:
        progress(
            {
                "event": "sidecar_updated",
                "dataset": dataset,
                "path": source_path,
            }
        )


def _geometry_chunks(
    jobs: tuple[tuple[str, str], ...],
    parallelism: int,
) -> tuple[tuple[tuple[str, str], ...], ...]:
    worker_count = min(max(parallelism, 1), len(jobs))
    chunks: list[list[tuple[str, str]]] = [[] for _ in range(worker_count)]
    for index, job in enumerate(jobs):
        chunks[index % worker_count].append(job)
    return tuple(tuple(chunk) for chunk in chunks)


def _process_geometry_chunk(chunk: _GeometryChunk) -> tuple[tuple[str, str], ...]:
    plans = {plan.spec.name: plan for plan in chunk.plans}
    api = HfApi(endpoint=chunk.endpoint, token=chunk.token)
    with httpx.Client(follow_redirects=True, timeout=None) as client, _open_local_reference_batch(
        chunk.groups,
        chunk.reference_directory,
        chunk.threshold,
    ) as references:
        for dataset, source_path in chunk.jobs:
            _process_geometry_path(
                api,
                plans[dataset],
                source_path=source_path,
                references=references,
                sidecar_root=chunk.sidecar_root,
                source_root=chunk.source_root,
                batch_size=chunk.batch_size,
                progress=None,
                http_client=client,
                retain_source=True,
            )
    return chunk.jobs


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
    client: Any,
) -> Iterator[tuple[OverlapReference, ...]]:
    first_record = groups[0].record_id[:8]
    with TemporaryDirectory(
        dir=workdir,
        prefix=f"reference-{first_record}-",
    ) as directory, ExitStack() as stack:
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


def _finalize_plan(
    api: Any,
    plan: DatasetPlan,
    *,
    sidecar_root: Path,
    workdir: Path,
    batch_size: int,
    reference_info: Mapping[str, object],
    progress: Progress | None,
    http_client: Any,
    source_cache_root: Path | None = None,
) -> DatasetReceipt:
    target_revision = capture_revision(api, plan.spec.output_repo)
    card = DatasetCardAccumulator()
    expectations, current_commit = finalize_dataset(
        api,
        plan,
        sidecar_root=sidecar_root,
        local_root=workdir / "final",
        batch_size=batch_size,
        parent_commit=target_revision,
        progress=progress,
        http_client=http_client,
        card=card,
        source_cache_root=source_cache_root,
    )
    card_artifacts = card.write_artifacts(
        workdir / "cards" / plan.spec.name,
        dataset_name=plan.spec.name,
        source_repo=plan.spec.source_repo,
        target_repo=plan.spec.output_repo,
        source_revision=plan.source_revision,
        reference_version=str(reference_info["source_version"]),
    )
    data_paths = tuple(expectation.path for expectation in expectations)
    current_commit = _upload_card_artifacts(
        api,
        plan.spec.output_repo,
        card_artifacts,
        current_commit,
    )
    changed_paths, added_paths = _card_paths(card_artifacts, plan.source_files, data_paths)
    manifest = _build_dataset_manifest(
        plan,
        reference_info,
        expectations,
        changed_paths,
        added_paths,
        card_artifacts,
    )
    manifest_commit = upload_manifest(
        api,
        plan.spec.output_repo,
        manifest,
        parent_commit=current_commit,
    )
    target_revision = _advance_commit(api, plan.spec.output_repo, manifest_commit)
    shared_blobs = _shared_blobs(api, plan, set(changed_paths))
    verification = _verify_final_dataset(
        api,
        plan,
        expectations=expectations,
        added_paths=added_paths,
        expected_shared_blobs=shared_blobs,
        expected_manifest=manifest,
        expected_artifacts=card_artifacts.hashes,
        temp_dir=workdir / "verify",
        http_client=http_client,
    )
    _cleanup_card_artifacts(card_artifacts)
    return DatasetReceipt(plan, expectations, verification)


def _cleanup_card_artifacts(artifacts: CardArtifacts) -> None:
    for path in artifacts.files.values():
        path.unlink(missing_ok=True)


def _upload_card_artifacts(
    api: Any,
    target_repo: str,
    artifacts: CardArtifacts,
    parent_commit: str,
) -> str:
    current_commit = parent_commit
    for path, local_path in artifacts.files.items():
        current_commit = _upload_file(
            api,
            target_repo,
            path,
            local_path,
            current_commit,
            commit_message=f"Add EUNIS dataset card artifact {path}",
        )
    return current_commit


def _card_paths(
    artifacts: CardArtifacts,
    source_paths: tuple[str, ...],
    data_paths: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    changed = tuple(path for path in artifacts.files if path in source_paths)
    added = tuple(path for path in artifacts.files if path not in source_paths)
    return (*data_paths, *changed), added


def _build_dataset_manifest(
    plan: DatasetPlan,
    reference_info: Mapping[str, object],
    expectations: tuple[ShardExpectation, ...],
    changed_paths: tuple[str, ...],
    added_paths: tuple[str, ...],
    card_artifacts: CardArtifacts,
) -> dict[str, Any]:
    return build_manifest(
        source_repo=plan.spec.source_repo,
        target_repo=plan.spec.output_repo,
        source_revision=plan.source_revision,
        source_paths=plan.source_files,
        changed_paths=changed_paths,
        added_paths=added_paths,
        reference_manifest=reference_info,
        card_manifest=card_artifacts.manifest,
        rows_by_path={expectation.path: expectation.rows for expectation in expectations},
        schema_by_path={expectation.path: expectation.schema for expectation in expectations},
    )


def _verify_final_dataset(
    api: Any,
    plan: DatasetPlan,
    *,
    expectations: tuple[ShardExpectation, ...],
    added_paths: tuple[str, ...],
    expected_shared_blobs: Mapping[str, str],
    expected_manifest: Mapping[str, Any],
    expected_artifacts: Mapping[str, str],
    temp_dir: Path,
    http_client: Any,
) -> VerificationReceipt:
    return verify_dataset(
        api,
        plan.spec.output_repo,
        expectations=expectations,
        expected_tree_paths=(*plan.source_files, *added_paths, "eunis/manifest.json"),
        expected_shared_blobs=expected_shared_blobs,
        expected_manifest=expected_manifest,
        expected_artifacts=expected_artifacts,
        temp_dir=temp_dir,
        http_client=http_client,
    )


def _shared_blobs(
    api: Any,
    plan: DatasetPlan,
    changed_paths: set[str],
) -> dict[str, str]:
    return {
        entry.path: entry.blob_id
        for entry in list_repo_files(api, plan.spec.source_repo, plan.source_revision)
        if entry.path not in changed_paths and isinstance(getattr(entry, "blob_id", None), str)
    }


def _cleanup_source_cache(root: Path) -> None:
    shutil.rmtree(root, ignore_errors=True)


@contextmanager
def _source_cache(workdir: Path) -> Iterator[Path]:
    root = workdir / "source"
    try:
        yield root
    finally:
        _cleanup_source_cache(root)
