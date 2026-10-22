from __future__ import annotations

import hashlib
import json
import shutil
from argparse import Namespace
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import shapely
from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis.pool_adapters import source_pins
from osm_polygon_eunis.pool_adapters import workers as workers_module
from osm_polygon_eunis.pool_adapters import worldcover_tiles as worldcover_tiles_module
from osm_polygon_eunis.pool_adapters.controller import (
    RawInventoryCensus,
    ShardCensus,
    WorkspaceQuotaExceededError,
)
from osm_polygon_eunis.pool_adapters.lucas import LucasFeatureQuery, _JsonlReferenceIdWriter
from osm_polygon_eunis.pool_adapters.source_pins import RawSourceTreePin, SourceTreeFile
from osm_polygon_eunis.pool_adapters.workers import (
    LucasAdapterWorker,
    LucasWorkerOptions,
    WorkerRuntime,
    WorldCoverAdapterWorker,
    WorldCoverWorkerOptions,
    _encode_geometry,
    _lucas_pool_polygons,
    _WorkspaceScratchBudget,
    build_workers,
    make_raw_row_factory,
)
from osm_polygon_eunis.pool_adapters.workspace_quota import WorkspaceQuota
from osm_polygon_eunis.pool_adapters.worldcover_tiles import (
    VerifiedWorldCoverTileSource,
    WorldCoverShardScope,
    WorldCoverTileCatalog,
)
from osm_polygon_eunis.pool_contract import (
    TextSource,
    canonical_native_key,
    native_key_json,
)

_PRODUCER_REVISION = "18c75517c74be22ab2f90edce2b65d3600461048"
_SOURCE_REVISION = "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd"
_SOURCE_REPOSITORY = "NoeFlandre/osm-polygon-website-tag"
_CONTENT_SHA256 = "796965a9013b2ed590d1797a77a1dfa86dcda60e75fceb8d05d460d37e70c1b2"
_SHARD_BLOB_ID = "702eb4afb4c63502f7f598141c7f97f120a57f02"
_SHARD_PATH = "polygons/afghanistan-latest.parquet"
_TILE_ID = "N00E000"


def _catalog_document(tmp_path: Path, tile_bytes: bytes = b"tile") -> tuple[Path, str, Path]:
    filename = f"ESA_WorldCover_10m_2021_v200_{_TILE_ID}_Map.tif"
    tile_path = tmp_path / filename
    tile_path.write_bytes(tile_bytes)
    source_path = (
        Path(__file__).parents[2] / "config/worldcover-afghanistan-first-shard-tile-catalog.json"
    )
    document = json.loads(source_path.read_text(encoding="utf-8"))
    scope = document["coverage_scope"]
    scope["usable_geometry_count"] = 2
    scope["tile_requests_by_usable_geometry"] = {_TILE_ID: 2}
    document["resource_limits"] = {
        "max_entries": 1,
        "max_tile_bytes": 16,
        "max_total_bytes": 16,
    }
    document["entries"] = [
        {
            "tile_id": _TILE_ID,
            "state": "published",
            "filename": filename,
            "relative_path": filename,
            "size_bytes": len(tile_bytes),
            "sha256": hashlib.sha256(tile_bytes).hexdigest(),
            "source_uri": (
                f"https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/{filename}"
            ),
        }
    ]
    catalog_path = tmp_path / "catalog.json"
    catalog_path.write_text(
        json.dumps(document, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    return catalog_path, digest, tile_path


def _scope() -> WorldCoverShardScope:
    return WorldCoverShardScope(
        source_repository=_SOURCE_REPOSITORY,
        source_revision=_SOURCE_REVISION,
        shard_path=_SHARD_PATH,
        shard_blob_id=_SHARD_BLOB_ID,
        shard_size_bytes=145_293,
        shard_row_count=42,
        shard_content_sha256=_CONTENT_SHA256,
    )


def test_worker_runtime_and_configuration_field_guards_fail_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive workspace cap"):
        WorkerRuntime("", tmp_path, 1, "website")
    budget = _WorkspaceScratchBudget(tmp_path, 10)
    with pytest.raises(ValueError, match="reservation must be positive"):
        budget.reserve(0)
    with pytest.raises(ValueError, match="settlement exceeds"):
        budget.release(1)
    with pytest.raises(TypeError, match="must be an object"):
        workers_module._mapping([], "worker section")
    with pytest.raises(ValueError, match="must be a local path"):
        workers_module._path({"root": "  "}, "root")
    with pytest.raises(ValueError, match="lowercase SHA-256"):
        workers_module._hash({"sha": "A" * 64}, "sha")


def test_first_shard_worldcover_catalog_is_bound_to_full_raw_content_identity() -> None:
    catalog_path = (
        Path(__file__).parents[2] / "config/worldcover-afghanistan-first-shard-tile-catalog.json"
    )
    catalog_sha256 = "eb66d2ad10897ba869e0bfc48831151f592c36dfee7d84ce399c6dd866e22b88"
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha256)
    source_pin = source_pins.polygon_geometry_tree_pins(TextSource.WEBSITE)[0]
    pinned_shard = next(row for row in source_pin.files if row.path == _SHARD_PATH)

    catalog.validate_scope(_scope())

    assert pinned_shard.verified_lfs_content_sha256 == _CONTENT_SHA256
    assert catalog.producer_revision == _PRODUCER_REVISION
    assert catalog.shard_content_sha256 == _CONTENT_SHA256
    assert catalog.shard_row_count == catalog.usable_geometry_count == 42
    assert catalog.tile_requests_by_usable_geometry == (
        ("N30E063", 5),
        ("N33E060", 4),
        ("N33E066", 2),
        ("N33E069", 24),
        ("N36E063", 1),
        ("N36E066", 6),
    )
    assert catalog.total_bytes == 574_726_120
    with pytest.raises(ValueError, match="selected source shard"):
        catalog.validate_scope(replace(_scope(), shard_content_sha256="0" * 64))


def test_verified_worldcover_tile_source_checks_hashes_and_exact_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog_sha256, tile_path = _catalog_document(tmp_path)
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha256)
    monkeypatch.setattr(worldcover_tiles_module, "_verify_raster_profile", lambda *_args: None)
    source = VerifiedWorldCoverTileSource(
        catalog,
        tmp_path,
        tile_not_published_error=FileNotFoundError,
    )

    source.preflight({_TILE_ID})
    assert source.ensure(SimpleNamespace(name=_TILE_ID)) == tile_path
    with pytest.raises(ValueError, match="coverage differs"):
        source.preflight(set())

    tile_path.write_bytes(b"t1le")
    with pytest.raises(ValueError, match="bytes differ from pinned catalog"):
        source.ensure(SimpleNamespace(name=_TILE_ID))


def test_verified_worldcover_snapshot_is_quota_bound_and_isolates_consumed_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog_path, catalog_sha256, tile_path = _catalog_document(tmp_path)
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha256)
    monkeypatch.setattr(worldcover_tiles_module, "_verify_raster_profile", lambda *_args: None)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    source = VerifiedWorldCoverTileSource(
        catalog,
        tmp_path,
        tile_not_published_error=FileNotFoundError,
        snapshot_root=workspace / "references" / "worldcover",
        workspace=workspace,
        workspace_quota=WorkspaceQuota(workspace, 4),
    )

    source.preflight({_TILE_ID})
    snapshot = source.ensure(SimpleNamespace(name=_TILE_ID))
    assert snapshot != tile_path
    assert snapshot.read_bytes() == b"tile"
    tile_path.write_bytes(b"t1le")
    assert source.ensure(SimpleNamespace(name=_TILE_ID)) == snapshot
    assert snapshot.read_bytes() == b"tile"
    tile_path.write_bytes(b"tile")

    blocked_workspace = tmp_path / "blocked-workspace"
    blocked_workspace.mkdir()
    blocked = VerifiedWorldCoverTileSource(
        catalog,
        tmp_path,
        tile_not_published_error=FileNotFoundError,
        snapshot_root=blocked_workspace / "references" / "worldcover",
        workspace=blocked_workspace,
        workspace_quota=WorkspaceQuota(blocked_workspace, 3),
    )
    with pytest.raises(WorkspaceQuotaExceededError):
        blocked.preflight({_TILE_ID})
    assert not tuple(blocked_workspace.rglob("*.partial"))


def test_worldcover_catalog_rejects_malformed_scope_entries_and_limits(tmp_path: Path) -> None:
    cases = (
        ("format_version", "wrong", "format is unsupported"),
        ("producer_revision", "0" * 40, "not bound to the pinned v4 producer"),
        ("product.year", 2020, "does not describe the pinned 2021 v200 product"),
        ("coverage_scope.text_source", "unknown", "recognized source scope"),
        ("coverage_scope.tile_requests_by_usable_geometry.N00E000", 0, "must be positive"),
        ("entries[0].state", "missing", "must be published or explicitly unpublished"),
        ("entries[0].relative_path", "../escape.tif", "safe relative product filename"),
        (
            "entries[0].source_uri",
            "https://example.test/tile.tif",
            "outside the pinned official product",
        ),
        ("resource_limits.max_tile_bytes", 3, "per-file byte limit"),
    )
    for index, (field, value, reason) in enumerate(cases):
        case_root = tmp_path / str(index)
        case_root.mkdir()
        catalog_path, _catalog_sha, _tile_path = _catalog_document(case_root)
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
        parts = field.replace("][", ".").replace("[", ".").replace("]", "").split(".")
        target = document
        for part in parts[:-1]:
            target = target[int(part)] if part.isdigit() else target[part]
        target[parts[-1]] = value
        catalog_path.write_text(json.dumps(document), encoding="utf-8")
        digest = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
        with pytest.raises((TypeError, ValueError), match=reason):
            WorldCoverTileCatalog.load(catalog_path, digest)


def test_worldcover_unpublished_tile_error_carries_catalog_evidence(
    tmp_path: Path,
) -> None:
    catalog_path, _catalog_sha, _tile_path = _catalog_document(tmp_path)
    document = json.loads(catalog_path.read_text(encoding="utf-8"))
    document["entries"][0]["state"] = "unpublished"
    document["entries"][0]["absence_evidence_uri"] = "https://example.test/absence-proof"
    catalog_path.write_text(json.dumps(document), encoding="utf-8")
    catalog_sha = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha)
    source = VerifiedWorldCoverTileSource(
        catalog,
        tmp_path,
        tile_not_published_error=FileNotFoundError,
    )

    with pytest.raises(FileNotFoundError, match="absence-proof"):
        source.ensure(SimpleNamespace(name=_TILE_ID))


def test_worldcover_raster_profile_requires_pinned_crs_transform_and_extent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pixel_size = 3 / 36_000

    class Raster:
        count = 1
        width = 36_000
        height = 36_000
        dtypes = ("uint8",)
        crs = SimpleNamespace(to_epsg=lambda: 4_326)
        nodata = 0
        bounds = (63.0, 30.0, 66.0, 33.0)

        def __init__(self) -> None:
            self.transform = (pixel_size, 0.0, 63.0, 0.0, -pixel_size, 33.0)

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    raster = Raster()
    monkeypatch.setattr(worldcover_tiles_module.rasterio, "open", lambda _path: raster)
    worldcover_tiles_module._verify_raster_profile(Path("tile.tif"), "N30E063")

    raster.transform = (pixel_size, 0.0, 64.0, 0.0, -pixel_size, 33.0)
    with pytest.raises(ValueError, match="affine transform"):
        worldcover_tiles_module._verify_raster_profile(Path("tile.tif"), "N30E063")
    raster.transform = (pixel_size, 0.0, 63.0, 0.0, -pixel_size, 33.0)
    raster.bounds = (62.0, 30.0, 65.0, 33.0)
    with pytest.raises(ValueError, match="extent"):
        worldcover_tiles_module._verify_raster_profile(Path("tile.tif"), "N30E063")


def test_catalog_scope_rejects_stale_content_and_worker_config_is_hashed(tmp_path: Path) -> None:
    catalog_path, catalog_sha256, _ = _catalog_document(tmp_path)
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha256)
    with pytest.raises(ValueError, match="does not match the selected source shard"):
        catalog.validate_scope(
            WorldCoverShardScope(
                _SOURCE_REPOSITORY,
                _SOURCE_REVISION,
                _SHARD_PATH,
                _SHARD_BLOB_ID,
                145_293,
                42,
                "0" * 64,
            )
        )

    source_path = "polygons/lucas-fixture.parquet"
    source_bytes = b"pinned"
    blob_id = hashlib.sha1(
        f"blob {len(source_bytes)}\0".encode() + source_bytes, usedforsecurity=False
    ).hexdigest()
    source_file = SourceTreeFile(source_path, len(source_bytes), blob_id, None, len(source_bytes))
    source_pin = RawSourceTreePin(
        TextSource.WEBSITE,
        "NoeFlandre/osm-polygon-website-tag",
        "1" * 40,
        "polygons/",
        1,
        len(source_bytes),
        "a" * 64,
        (source_file,),
        "polygon_geometry",
    )
    census = RawInventoryCensus(
        "website",
        source_pin.repository,
        source_pin.revision,
        source_pin.path_prefix,
        source_pin.inventory_sha256,
        1,
        1,
        1,
        "b" * 64,
        (ShardCensus(source_path, len(source_bytes), blob_id, None, 1, "c" * 64),),
        "d" * 64,
    )
    config = {
        "format_version": "bounded-pool-worker-config-v1",
        "lucas": {
            "gpkg_path": str(tmp_path / "reference.gpkg"),
            "taxonomy_path": str(tmp_path / "taxonomy.pdf"),
        },
    }
    config_path = tmp_path / "worker-config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    args = Namespace(
        worker_config=config_path,
        worker_config_sha256=config_sha256,
        workspace=str(tmp_path / "workspace"),
        max_workspace_bytes=10_000,
        max_row_bytes=2 * 1024 * 1024,
        max_shards_per_invocation=1,
        maps=["lucas"],
        shard=[source_path],
    )

    workers, map_pins, _raw_rows = build_workers(args, census, source_pin)

    assert tuple(workers) == ("lucas",)
    assert tuple(map_pins) == ("lucas",)
    assert map_pins["lucas"].threshold == 0.8
    args.worker_config_sha256 = "0" * 64
    with pytest.raises(ValueError, match="bytes differ from the supplied SHA-256"):
        build_workers(args, census, source_pin)


def test_builtin_worldcover_factory_binds_the_real_first_shard_catalog(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_pin = source_pins.polygon_geometry_tree_pins(TextSource.WEBSITE)[0]
    selected = next(row for row in source_pin.files if row.path == _SHARD_PATH)
    census = RawInventoryCensus(
        "website",
        source_pin.repository,
        source_pin.revision,
        source_pin.path_prefix,
        source_pin.inventory_sha256,
        source_pin.file_count,
        42,
        42,
        "e" * 64,
        (
            ShardCensus(
                selected.path, selected.size_bytes, selected.blob_id, selected.lfs_oid, 42, "f" * 64
            ),
        ),
        "a" * 64,
    )
    catalog_path = (
        Path(__file__).parents[2] / "config/worldcover-afghanistan-first-shard-tile-catalog.json"
    )
    catalog_sha256 = hashlib.sha256(catalog_path.read_bytes()).hexdigest()
    config_path = tmp_path / "worldcover-worker-config.json"
    tile_root = tmp_path / "tiles"
    tile_root.mkdir()
    config_path.write_text(
        json.dumps(
            {
                "format_version": "bounded-pool-worker-config-v1",
                "worldcover": {
                    "tile_catalog_path": str(catalog_path),
                    "tile_catalog_sha256": catalog_sha256,
                    "tile_root": str(tile_root),
                },
            }
        ),
        encoding="utf-8",
    )
    producer = SimpleNamespace(
        iter_sourcewide_coverage=lambda *_args: (),
        is_usable_polygon=lambda geometry: geometry is not None,
        tiles_for_bbox=lambda _bounds: (),
        tile_not_published_error=FileNotFoundError,
        taxonomy_sha256="b" * 64,
    )
    monkeypatch.setattr(workers_module, "_load_pinned_worldcover_producer", lambda: producer)
    args = Namespace(
        worker_config=config_path,
        worker_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        workspace=str(tmp_path / "workspace"),
        max_workspace_bytes=1_000_000,
        max_row_bytes=2 * 1024 * 1024,
        max_shards_per_invocation=1,
        maps=["worldcover"],
        shard=[_SHARD_PATH],
    )

    workers, pins, _raw_rows = build_workers(args, census, source_pin)

    assert tuple(workers) == ("worldcover",)
    assert pins["worldcover"].producer_revision == _PRODUCER_REVISION
    worldcover_worker = workers["worldcover"]
    assert isinstance(worldcover_worker, WorldCoverAdapterWorker)
    assert worldcover_worker.runtime.catalog is not None
    assert worldcover_worker.runtime.catalog.sha256 == catalog_sha256


def test_lucas_sidecar_reserves_workspace_quota_before_writing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    budget = _WorkspaceScratchBudget(workspace, 10)
    scratch = workspace / "scratch" / "lucas-reference-evidence-test"
    scratch.mkdir(parents=True)
    budget.track_temporary_path(scratch)
    first = _JsonlReferenceIdWriter(scratch, "first", budget)
    second = _JsonlReferenceIdWriter(scratch, "second", budget)

    first.write_id("1")
    with pytest.raises(WorkspaceQuotaExceededError):
        second.write_id("2")
    assert second.path.stat().st_size == 0
    second.discard()
    first.finish()
    assert budget._reserved_bytes == 0

    retry = _JsonlReferenceIdWriter(scratch, "retry", budget)
    retry.write_id("3")
    sidecar = retry.finish()
    assert sidecar is not None
    sidecar_path = sidecar.path
    assert sidecar_path is not None
    assert sidecar_path.read_text(encoding="utf-8") == '"3"\n'
    assert budget._reserved_bytes == 0
    assert sum(path.stat().st_size for path in workspace.rglob("*") if path.is_file()) == 8
    shutil.rmtree(scratch)
    budget.forget_temporary_path(scratch)
    budget.close()


def test_worldcover_direct_url_requires_the_live_clean_pinned_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = tmp_path / "verified-producer"
    shadow = tmp_path / "shadow" / "osm_worldcover" / "__init__.py"
    expected_module = project / "src/osm_worldcover/__init__.py"
    expected_module.parent.mkdir(parents=True)
    expected_module.touch()
    shadow.parent.mkdir(parents=True)
    shadow.touch()
    origin = {
        "url": "git+https://github.com/NoeFlandre/osm-worldcover.git",
        "vcs_info": {"vcs": "git", "commit_id": _PRODUCER_REVISION, "dirty": False},
        "dir_info": {"editable_project_location": str(project)},
    }
    distribution = SimpleNamespace()
    monkeypatch.setattr(
        workers_module,
        "_worldcover_git_revision",
        lambda package_path: _PRODUCER_REVISION if package_path == expected_module else None,
    )

    assert (
        workers_module._verified_direct_url_revision(distribution, origin, expected_module)
        == _PRODUCER_REVISION
    )
    assert workers_module._verified_direct_url_revision(distribution, origin, shadow) is None
    origin_without_dirty_field = {
        "url": "git+https://github.com/NoeFlandre/osm-worldcover.git",
        "vcs_info": {"vcs": "git", "commit_id": _PRODUCER_REVISION},
        "dir_info": {"editable_project_location": str(project)},
    }
    installed_origin = {
        "url": "git+https://github.com/NoeFlandre/osm-worldcover.git",
        "vcs_info": {"vcs": "git", "commit_id": _PRODUCER_REVISION},
        "dir_info": {},
    }
    installed_distribution = SimpleNamespace(
        locate_file=lambda _path: expected_module,
    )
    assert (
        workers_module._verified_direct_url_revision(
            installed_distribution, installed_origin, expected_module
        )
        == _PRODUCER_REVISION
    )
    assert (
        workers_module._verified_direct_url_revision(
            distribution, origin_without_dirty_field, expected_module
        )
        is None
    )
    monkeypatch.setattr(workers_module, "_worldcover_git_revision", lambda _path: "0" * 40)
    assert (
        workers_module._verified_direct_url_revision(distribution, origin, expected_module) is None
    )
    origin["vcs_info"]["dirty"] = True
    assert (
        workers_module._verified_direct_url_revision(distribution, origin, expected_module) is None
    )


def test_worldcover_worker_preflights_and_serializes_canonical_attempts(tmp_path: Path) -> None:
    catalog_path, catalog_sha256, _tile_path = _catalog_document(tmp_path)
    catalog = WorldCoverTileCatalog.load(catalog_path, catalog_sha256)
    source_snapshot_id = "source-snapshot"
    raw_rows = tuple(_raw_worker_row(index, source_snapshot_id) for index in range(2))
    raw_checkpoint = tmp_path / "raw.jsonl"
    raw_checkpoint.write_text(
        "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in raw_rows),
        encoding="utf-8",
    )
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = workers_module.LocalControllerArtifactSink(workspace / "artifacts", workspace, 10_000)

    class TileSource:
        required: set[str] | None = None

        def preflight(self, required_tile_ids: set[str]) -> None:
            self.required = required_tile_ids

    tile_source = TileSource()

    def attempt_factory(frames, _tiles):
        ordinal = 0
        for frame in frames:
            for polygon_id in frame["polygon_id"]:
                yield SimpleNamespace(
                    source_polygon_id=polygon_id,
                    source_row_ordinal=ordinal,
                    polygon_area_m2=100.0,
                    class_areas_m2=((10, 80.0),),
                    class_evidence_ids=((10, (_TILE_ID,)),),
                    measurement_diagnostics="synthetic geometry fixture",
                    area_measurement=workers_module.WORLDCOVER_AREA_MEASUREMENT,
                    attempted_tiles=(_TILE_ID,),
                    missing_tiles=(),
                    extraction_error=None,
                )
                ordinal += 1

    worker = WorldCoverAdapterWorker(
        WorldCoverWorkerOptions(
            runtime=WorkerRuntime(source_snapshot_id, workspace, 10_000, "website", catalog),
            tile_source=tile_source,
            attempt_factory=attempt_factory,
            usable_polygon=lambda geometry: geometry is not None and geometry.is_valid,
            tiles_for_bbox=lambda _bounds: (SimpleNamespace(name=_TILE_ID),),
            reference_pin=workers_module._worldcover_reference_pin("b" * 64),
            sink=sink,
        )
    )

    records = tuple(worker.run_checkpoint(raw_checkpoint))

    assert tile_source.required == {_TILE_ID}
    assert len(records) == 2
    assert all(record["adapter_record"]["label_outcomes"] for record in records)
    assert all(record["polygon"]["original_area"] == 100.0 for record in records)
    assert all(
        record["adapter_record"]["attempt_evidence"]["status"] == "completed" for record in records
    )


def test_worker_geometry_codec_and_worldcover_batching_are_bounded() -> None:
    geometry = shapely.to_wkb(box(1, 2, 3, 4))
    payloads = (
        _encode_geometry(None, 128),
        _encode_geometry(geometry, 128),
        _encode_geometry('{"type":"Point","coordinates":[1,2]}', 128),
        _encode_geometry({"type": "Point", "coordinates": [1, 2]}, 128),
    )

    assert workers_module._decode_geometry_payload(payloads[0]) is None
    wkb_geometry = workers_module._decode_geometry_payload(payloads[1])
    text_geometry = workers_module._decode_geometry_payload(payloads[2])
    json_geometry = workers_module._decode_geometry_payload(payloads[3])
    assert wkb_geometry is not None and wkb_geometry.geom_type == "Polygon"
    assert text_geometry is not None and text_geometry.geom_type == "Point"
    assert json_geometry is not None and json_geometry.geom_type == "Point"
    assert workers_module._decode_geometry_payload({"kind": "base64", "value": "!"}) is None
    with pytest.raises(ValueError, match="unknown encoding"):
        workers_module._decode_geometry_payload({"kind": "unexpected"})
    with pytest.raises(TypeError, match="unsupported raw geometry type"):
        _encode_geometry(object(), 128)

    rows = tuple(
        {"inventory_row_id": str(index), "geometry_payload": payloads[1]} for index in range(17)
    )
    assert [len(frame) for frame in workers_module._worldcover_frames(rows)] == [16, 1]


def test_worker_config_and_native_key_validators_reject_malformed_inputs(tmp_path: Path) -> None:
    args = Namespace(worker_config=None, worker_config_sha256=None)
    with pytest.raises(ValueError, match="requires a SHA-256-pinned"):
        workers_module._load_worker_config(args)

    config_path = tmp_path / "bad-config.json"
    config_path.write_bytes(b"{")
    args.worker_config = config_path
    args.worker_config_sha256 = hashlib.sha256(config_path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not valid JSON"):
        workers_module._load_worker_config(args)

    with pytest.raises(TypeError, match="invalid pair"):
        workers_module._native_key_from_json('[{"source_pbf":"not-a-pair"}]')
    with pytest.raises(TypeError, match="not a canonical pair list"):
        workers_module._native_key_from_json("{}")


def _raw_worker_row(index: int, source_snapshot_id: str) -> dict[str, object]:
    key = canonical_native_key(
        {"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": index + 1}
    )
    return {
        "native_key_json": native_key_json(key),
        "inventory_row_id": workers_module.inventory_row_identifier(source_snapshot_id, key),
        "source_row_locator": f"{_SHARD_PATH}#row={index}",
        "geometry_payload": _encode_geometry(shapely.to_wkb(box(index, 0, index + 1, 1)), 128),
    }


def test_raw_row_factory_keeps_native_key_locator_and_bounds_geometry(
    tmp_path: Path,
) -> None:
    source_path = "polygons/fixture.parquet"
    source_file = SourceTreeFile(source_path, 1, "0" * 40, None, 1)
    source_pin = RawSourceTreePin(
        TextSource.WEBSITE,
        "fixture/website",
        "1" * 40,
        "polygons/",
        1,
        1,
        "a" * 64,
        (source_file,),
        "polygon_geometry",
    )
    geometry = shapely.to_wkb(box(1, 2, 3, 4))
    staged = tmp_path / ("a" * 64 + "-fixture.parquet")
    pq.write_table(
        pa.table(
            {
                "source_pbf": ["region.osm.pbf"],
                "osm_type": ["way"],
                "osm_id": [71],
                "geometry": [geometry],
            }
        ),
        staged,
    )

    (row,) = tuple(make_raw_row_factory(source_pin, "snapshot-1")(staged))

    key = canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 71})
    assert row["native_key_json"] == native_key_json(key)
    assert row["source_row_locator"] == f"{source_path}#row=0"
    assert row["geometry_payload"] == _encode_geometry(geometry, 2 * 1024 * 1024)
    with pytest.raises(ValueError, match="raw geometry exceeds"):
        _encode_geometry(geometry, 1)


def test_lucas_worker_polygon_conversion_preserves_valid_source_geometry(tmp_path: Path) -> None:
    source_snapshot_id = "source-snapshot"
    geometry = box(-1, 50, 0, 51)
    key = canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 17})
    inventory_id = workers_module.inventory_row_identifier(source_snapshot_id, key)
    row = {
        "native_key_json": native_key_json(key),
        "inventory_row_id": inventory_id,
        "source_row_locator": "polygons/region.parquet#row=0",
        "geometry_payload": _encode_geometry(shapely.to_wkb(geometry), 2 * 1024 * 1024),
    }
    sink = workers_module.LocalControllerArtifactSink(tmp_path / "artifacts", tmp_path, 1_000_000)

    (polygon,) = tuple(_lucas_pool_polygons((row,), source_snapshot_id, sink))

    assert polygon.geometry is not None and polygon.geometry.is_valid
    assert polygon.snapshot is not None
    assert polygon.snapshot.crs == "EPSG:3035"
    assert polygon.snapshot.original_area is not None and polygon.snapshot.original_area > 0


def test_lucas_worker_retains_missing_geometry_as_an_excluded_raw_row(tmp_path: Path) -> None:
    source_snapshot_id = "source-snapshot"
    key = canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 18})
    row = {
        "native_key_json": native_key_json(key),
        "inventory_row_id": workers_module.inventory_row_identifier(source_snapshot_id, key),
        "source_row_locator": "polygons/region.parquet#row=0",
        "geometry_payload": {"kind": "missing"},
    }
    sink = workers_module.LocalControllerArtifactSink(tmp_path / "artifacts", tmp_path, 1_000_000)

    (polygon,) = tuple(_lucas_pool_polygons((row,), source_snapshot_id, sink))

    assert polygon.geometry is None
    assert polygon.snapshot is None
    assert (
        polygon.inventory.exclusion_reason == "geometry_transform_failed_or_antimeridian_rejected"
    )


def test_lucas_worker_entrypoint_uses_quota_bound_reader_and_canonical_attempts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_snapshot_id = "source-snapshot"
    key = canonical_native_key({"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 19})
    row = {
        "native_key_json": native_key_json(key),
        "inventory_row_id": workers_module.inventory_row_identifier(source_snapshot_id, key),
        "source_row_locator": "polygons/region.parquet#row=0",
        "geometry_payload": {"kind": "missing"},
    }
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = workers_module.LocalControllerArtifactSink(workspace / "artifacts", workspace, 10_000)
    manifest = workers_module.lucas_2022_copernicus_manifest()
    taxonomy = workers_module.LucasTaxonomy.lucas_2022_c3()
    reference_pin = workers_module.lucas_2022_reference_pin(manifest)
    observed: dict[str, object] = {}

    class Reader:
        is_open = True

        def __init__(self, _gpkg, _taxonomy_path, _manifest, reader_taxonomy, *, config) -> None:
            self.taxonomy = reader_taxonomy
            self.config = config
            observed["config"] = config

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            observed["closed"] = True

    monkeypatch.setattr(workers_module, "LucasGeoPackageReader", Reader)
    worker = LucasAdapterWorker(
        LucasWorkerOptions(
            runtime=WorkerRuntime(source_snapshot_id, workspace, 10_000, "website"),
            gpkg_path=tmp_path / "lucas.gpkg",
            taxonomy_path=tmp_path / "taxonomy.csv",
            manifest=manifest,
            taxonomy=taxonomy,
            reference_pin=reference_pin,
            sink=sink,
        )
    )

    (record,) = tuple(worker((row,)))

    config = observed["config"]
    assert isinstance(config, workers_module.LucasReaderConfig)
    assert config.scratch_parent == workspace / "scratch" / "lucas"
    assert isinstance(config.scratch_budget, _WorkspaceScratchBudget)
    assert observed["closed"] is True
    assert record["adapter_record"]["attempt_evidence"]["status"] == "excluded"
    assert record["inventory"]["exclusion_reason"] == (
        "geometry_transform_failed_or_antimeridian_rejected"
    )


def test_lucas_worker_projects_valid_geometry_before_querying_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_snapshot_id = "source-snapshot"
    row = _raw_worker_row(0, source_snapshot_id)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    sink = workers_module.LocalControllerArtifactSink(workspace / "artifacts", workspace, 10_000)
    manifest = workers_module.lucas_2022_copernicus_manifest()
    taxonomy = workers_module.LucasTaxonomy.lucas_2022_c3()
    reference_pin = workers_module.lucas_2022_reference_pin(manifest)
    observed: dict[str, object] = {}

    class Reader:
        is_open = True

        def __init__(self, _gpkg, _taxonomy_path, _manifest, reader_taxonomy, *, config) -> None:
            self.taxonomy = reader_taxonomy
            self.config = config

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def query(self, geometry):
            observed["geometry"] = geometry
            return LucasFeatureQuery((), (), exclusion_reason="synthetic_reference_not_loaded")

    monkeypatch.setattr(workers_module, "LucasGeoPackageReader", Reader)
    worker = LucasAdapterWorker(
        LucasWorkerOptions(
            runtime=WorkerRuntime(source_snapshot_id, workspace, 10_000, "website"),
            gpkg_path=tmp_path / "lucas.gpkg",
            taxonomy_path=tmp_path / "taxonomy.csv",
            manifest=manifest,
            taxonomy=taxonomy,
            reference_pin=reference_pin,
            sink=sink,
        )
    )

    (record,) = tuple(worker((row,)))

    projected = observed["geometry"]
    assert isinstance(projected, BaseGeometry)
    assert projected.is_valid
    assert not projected.equals_exact(box(0, 0, 1, 1), 0)
    assert record["polygon"]["crs"] == "EPSG:3035"
    assert record["inventory"]["exclusion_reason"] is None
    assert record["adapter_record"]["attempt_evidence"]["reason"] == (
        "synthetic_reference_not_loaded"
    )
