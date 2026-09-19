"""Storage-bounded orchestration for the three EUNIS dataset releases."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, cast
from urllib.parse import unquote

import httpx

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
) -> None:
    """Merge one reference group into every geometry shard's compact sidecar."""

    with _http_client(http_client) as reusable_client:
        source_root.mkdir(parents=True, exist_ok=True)
        for source_path in plan.geometry_paths:
            local_source = download_to_temp(
                api,
                plan.spec.source_repo,
                source_path,
                plan.source_revision,
                source_root,
                client=reusable_client,
            )
            sidecar = _sidecar_path(sidecar_root, plan.spec, source_path)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            next_sidecar = sidecar.with_name(f"{sidecar.name}.next")
            update_label_sidecar(
                local_source,
                next_sidecar,
                reference=reference,
                current=sidecar if sidecar.is_file() else None,
                batch_size=batch_size,
            )
            next_sidecar.replace(sidecar)
            local_source.unlink(missing_ok=True)
            if progress is not None:
                progress(
                    {"event": "sidecar_updated", "dataset": plan.spec.name, "path": source_path}
                )


def _commit_id(result: Any) -> str | None:
    for attribute in ("oid", "commit_oid"):
        value = getattr(result, attribute, None)
        if isinstance(value, str) and value:
            return value
    return None


def _advance_commit(api: Any, target_repo: str, result: Any) -> str:
    return _commit_id(result) or capture_revision(api, target_repo)


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
) -> tuple[tuple[ShardExpectation, ...], str]:
    local_source = download_to_temp(
        api,
        plan.spec.source_repo,
        geometry_path,
        plan.source_revision,
        local_root,
        client=http_client,
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
    current_commit = _upload_file(
        api,
        plan.spec.output_repo,
        geometry_path,
        local_output,
        parent_commit,
    )
    if link_output is not None:
        current_commit = _upload_file(
            api,
            plan.spec.output_repo,
            link_output.path,
            link_output.output,
            current_commit,
        )
    _cleanup_shard(local_source, local_output, sidecar, link_output)
    expectations = [geometry_expectation]
    if link_output is not None:
        expectations.append(link_output.expectation)
    return tuple(expectations), current_commit


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
) -> None:
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
    source_root = workdir / "source"
    checksums: dict[str, str] = {}
    with _http_client(None) as reusable_client:
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
) -> None:
    """Reuse each downloaded reference group across all source shards."""

    for group in groups:
        with (
            TemporaryDirectory(
                dir=workdir,
                prefix=f"reference-{group.record_id[:8]}-",
            ) as raw_dir,
            open_reference_group(
                group,
                Path(raw_dir),
                threshold=threshold,
                checksums=checksums,
                client=http_client,
            ) as reference,
        ):
            for plan in plans:
                process_geometry_paths(
                    api,
                    plan,
                    reference=reference,
                    sidecar_root=sidecar_root,
                    source_root=source_root,
                    batch_size=batch_size,
                    progress=progress,
                    http_client=http_client,
                )


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
