from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from osm_polygon_eunis.pool_adapters import controller as controller_module
from osm_polygon_eunis.pool_adapters.controller import (
    LocalControllerArtifactSink,
    MapPin,
    PoolControllerRequest,
    RawCheckpointContext,
    RawInventoryCensus,
    ShardCensus,
    ShardRunInputs,
    WorkspaceQuotaExceededError,
    _build_or_reuse_raw_checkpoint,
    _load_factory,
    _raw_checkpoint_signature,
    _validate_map_checkpoint_digest,
    _validate_raw_row,
    _write_bounded_line,
    canonical_adapter_checkpoint_rows,
    plan_run,
    read_raw_checkpoint,
    run_shard,
)
from osm_polygon_eunis.pool_adapters.lucas import LUCAS_AREA_MEASUREMENT_ID
from osm_polygon_eunis.pool_adapters.source_pins import RawSourceTreePin, SourceTreeFile
from osm_polygon_eunis.pool_adapters.worldcover import WORLDCOVER_AREA_MEASUREMENT
from osm_polygon_eunis.pool_contract import PolygonInventoryRow, TextSource


@dataclass(frozen=True)
class _SerializableAdapterRecord:
    inventory_row_id: str
    result: str


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _fixture(
    tmp_path: Path,
) -> tuple[RawSourceTreePin, RawInventoryCensus, Path, list[dict[str, Any]]]:
    path = "polygons/test-region.parquet"
    data = b"frozen-parquet-shard"
    blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data, usedforsecurity=False).hexdigest()
    keys = (
        '[["osm_id",1],["osm_type","way"],["source_pbf","region.osm.pbf"]]',
        '[["osm_id",2],["osm_type","way"],["source_pbf","region.osm.pbf"]]',
    )
    ordered_digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
    file_pin = SourceTreeFile(path, len(data), blob, None, len(data))
    source_pin = RawSourceTreePin(
        TextSource.WEBSITE,
        "owner/website",
        "1" * 40,
        "polygons/",
        1,
        len(data),
        _SHA_A,
        (file_pin,),
        "polygon_geometry",
    )
    shard = ShardCensus(path, len(data), blob, None, 2, ordered_digest)
    census = RawInventoryCensus(
        "website",
        "owner/website",
        "1" * 40,
        "polygons/",
        _SHA_A,
        1,
        2,
        2,
        _SHA_B,
        (shard,),
        _SHA_C,
    )
    input_root = tmp_path / "input"
    local_shard = input_root / path
    local_shard.parent.mkdir(parents=True)
    local_shard.write_bytes(data)
    raw_rows = [
        {
            "native_key_json": key,
            "inventory_row_id": hashlib.sha256(
                f"{census.source_snapshot_id}\n{key}".encode()
            ).hexdigest(),
            "source_row_locator": f"{path}#row={ordinal}",
            "geometry_payload": "synthetic-wkb",
        }
        for ordinal, key in enumerate(keys)
    ]
    return source_pin, census, input_root, raw_rows


def _request(*, max_workspace_bytes: int = 10_000) -> PoolControllerRequest:
    return PoolControllerRequest(
        text_source="website",
        shard_paths=("polygons/test-region.parquet",),
        maps=("lucas",),
        max_workspace_bytes=max_workspace_bytes,
    )


def _lucas_pin() -> MapPin:
    return MapPin(
        "lucas",
        _SHA_A,
        _SHA_B,
        LUCAS_AREA_MEASUREMENT_ID,
        _SHA_C,
        _SHA_D,
        _SHA_A,
        "independent_union_per_level_and_class:v1",
    )


def _worker(rows):
    for row in rows:
        yield {
            "inventory_row_id": row["inventory_row_id"],
            "attempt": {"status": "completed", "outcomes": [], "evidence": []},
        }


def test_run_shard_checkpoints_shared_rows_and_reuses_complete_map_on_restart(
    tmp_path: Path,
) -> None:
    source_pin, census, input_root, raw_rows = _fixture(tmp_path)
    raw_reads = 0
    map_calls = 0

    def read_rows(_path: Path):
        nonlocal raw_reads
        raw_reads += 1
        yield from raw_rows

    def worker(rows):
        nonlocal map_calls
        map_calls += 1
        yield from _worker(rows)

    workspace = tmp_path / "workspace"
    first = run_shard(
        _request(),
        ShardRunInputs(
            census,
            source_pin,
            input_root,
            workspace,
            {"lucas": _lucas_pin()},
            {"lucas": worker},
            read_rows,
        ),
    )
    second = run_shard(
        _request(),
        ShardRunInputs(
            census,
            source_pin,
            input_root,
            workspace,
            {"lucas": _lucas_pin()},
            {"lucas": worker},
            read_rows,
        ),
    )

    assert first.completed_maps == ("lucas",)
    assert second.reused_maps == ("lucas",)
    assert raw_reads == 1
    assert map_calls == 1
    assert first.raw_row_count == second.raw_row_count == 2
    assert first.plan.complete_pool is second.plan.complete_pool is False
    assert first.plan.evaluation_started is second.plan.evaluation_started is False
    assert not tuple(workspace.rglob("manifest.json"))


def test_plan_and_run_reject_source_mismatch_before_writing(tmp_path: Path) -> None:
    source_pin, census, input_root, _ = _fixture(tmp_path)
    mismatched = PoolControllerRequest(
        text_source="description",
        shard_paths=("polygons/test-region.parquet",),
        maps=("lucas",),
        max_workspace_bytes=10_000,
    )
    with pytest.raises(ValueError, match="selected source differs"):
        plan_run(mismatched, census, source_pin, input_root)

    workspace = tmp_path / "unused-workspace"
    with pytest.raises(ValueError, match="selected source differs"):
        run_shard(
            mismatched,
            ShardRunInputs(
                census,
                source_pin,
                input_root,
                workspace,
                {"lucas": _lucas_pin()},
                {"lucas": _worker},
                lambda _path: (),
            ),
        )
    assert not workspace.exists()


def test_storage_cap_stops_before_attempts_and_leaves_no_completion_receipt(
    tmp_path: Path,
) -> None:
    source_pin, census, input_root, raw_rows = _fixture(tmp_path)
    workspace = tmp_path / "small-workspace"

    with pytest.raises(WorkspaceQuotaExceededError):
        run_shard(
            _request(max_workspace_bytes=8),
            ShardRunInputs(
                census,
                source_pin,
                input_root,
                workspace,
                {"lucas": _lucas_pin()},
                {"lucas": _worker},
                lambda _path: iter(raw_rows),
            ),
        )

    assert not tuple(workspace.rglob("receipt.json"))
    assert not tuple(workspace.rglob("manifest.json"))


def test_partial_worker_attempts_are_restarted_without_rewriting_raw_rows(
    tmp_path: Path,
) -> None:
    source_pin, census, input_root, raw_rows = _fixture(tmp_path)
    raw_reads = 0

    def read_rows(_path: Path):
        nonlocal raw_reads
        raw_reads += 1
        yield from raw_rows

    def interrupted(rows):
        iterator = iter(rows)
        first = next(iterator)
        yield {
            "inventory_row_id": first["inventory_row_id"],
            "attempt": {"status": "completed"},
        }
        raise RuntimeError("simulated interruption")

    workspace = tmp_path / "resume-workspace"

    def inputs(worker):
        return ShardRunInputs(
            census,
            source_pin,
            input_root,
            workspace,
            {"lucas": _lucas_pin()},
            {"lucas": worker},
            read_rows,
        )

    with pytest.raises(RuntimeError, match="simulated interruption"):
        run_shard(_request(), inputs(interrupted))

    receipt = run_shard(_request(), inputs(_worker))
    assert receipt.completed_maps == ("lucas",)
    assert raw_reads == 1
    assert not tuple(workspace.rglob("*.partial"))


def test_unreceipted_checkpoint_from_crash_cannot_be_replaced_on_retry(
    tmp_path: Path,
) -> None:
    source_pin, census, input_root, raw_rows = _fixture(tmp_path)
    workspace = tmp_path / "orphan-checkpoint-workspace"
    rows_to_emit = list(raw_rows)

    def read_rows(_path: Path):
        yield from rows_to_emit

    inputs = ShardRunInputs(
        census,
        source_pin,
        input_root,
        workspace,
        {"lucas": _lucas_pin()},
        {"lucas": _worker},
        read_rows,
    )
    run_shard(_request(), inputs)

    raw_receipt = next(
        path for path in workspace.rglob("receipt.json") if path.parent.parent.name == "raw"
    )
    raw_checkpoint = raw_receipt.parent / "rows.jsonl"
    original_checkpoint = raw_checkpoint.read_bytes()
    raw_receipt.unlink()
    changed_rows = [dict(row, geometry_payload="changed-wkb") for row in raw_rows]
    rows_to_emit[:] = changed_rows

    with pytest.raises(ValueError, match="unreceipted raw checkpoint conflicts"):
        run_shard(
            _request(),
            ShardRunInputs(
                census,
                source_pin,
                input_root,
                workspace,
                {"lucas": _lucas_pin()},
                {"lucas": _worker},
                read_rows,
            ),
        )

    assert raw_checkpoint.read_bytes() == original_checkpoint


def test_census_rejects_selected_path_absent_from_complete_source_tree(tmp_path: Path) -> None:
    source_pin, census, input_root, _ = _fixture(tmp_path)
    request = PoolControllerRequest(
        text_source="website",
        shard_paths=("polygons/foreign.parquet",),
        maps=("lucas",),
        max_workspace_bytes=10_000,
    )
    with pytest.raises(ValueError, match="not in the complete source census"):
        plan_run(request, census, source_pin, input_root)


def test_inventory_manifest_bytes_and_full_source_identity_are_pinned(tmp_path: Path) -> None:
    source_pin, census, _, _ = _fixture(tmp_path)
    shard = census.shards[0]
    document = {
        "format_version": "raw-inventory-keyset-v1",
        "text_source": census.text_source,
        "repository": census.repository,
        "revision": census.revision,
        "path_prefix": census.path_prefix,
        "tree_inventory_sha256": census.tree_inventory_sha256,
        "shard_count": census.shard_count,
        "raw_row_count": census.raw_row_count,
        "unique_key_count": census.unique_key_count,
        "unique_keyset_sha256": census.unique_keyset_sha256,
        "shards": [
            {
                "path": shard.path,
                "size_bytes": shard.size_bytes,
                "blob_id": shard.blob_id,
                "lfs_oid": shard.lfs_oid,
                "row_count": shard.row_count,
                "ordered_key_sha256": shard.ordered_key_sha256,
            }
        ],
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    manifest_path = tmp_path / "raw-inventory.json"
    manifest_path.write_bytes(payload)
    manifest_sha = hashlib.sha256(payload).hexdigest()

    loaded = RawInventoryCensus.load(manifest_path, manifest_sha, source_pin)
    assert loaded.manifest_sha256 == manifest_sha
    assert loaded.unique_keyset_sha256 == census.unique_keyset_sha256
    with pytest.raises(ValueError, match="supplied SHA-256 pin"):
        RawInventoryCensus.load(manifest_path, _SHA_D, source_pin)


def test_map_pin_rejects_noncanonical_threshold() -> None:
    with pytest.raises(ValueError, match="inclusive 80% map policy"):
        MapPin(
            "lucas",
            _SHA_A,
            _SHA_B,
            "lucas-full-area-v1",
            _SHA_C,
            _SHA_D,
            _SHA_A,
            "independent_union_per_level_and_class:v1",
            threshold=0.79,
        )


def test_eunis_is_not_a_dependency_of_worldcover_lucas_controller() -> None:
    with pytest.raises(ValueError, match="EUNIS and unknown maps"):
        PoolControllerRequest(
            text_source="website",
            shard_paths=("polygons/test.parquet",),
            maps=("eunis",),
            max_workspace_bytes=10_000,
        )


def test_census_validation_rejects_bad_manifest_identity_and_row_totals(
    tmp_path: Path,
) -> None:
    source_pin, census, _, _ = _fixture(tmp_path)

    with pytest.raises(ValueError, match="manifest SHA-256"):
        replace(census, manifest_sha256="invalid").validate(source_pin)
    with pytest.raises(ValueError, match="raw row count does not reconcile"):
        replace(census, raw_row_count=3).validate(source_pin)
    with pytest.raises(ValueError, match="duplicate or missing native keys"):
        replace(census, unique_key_count=1).validate(source_pin)
    with pytest.raises(ValueError, match="repository, revision, or tree differs"):
        replace(census, revision="2" * 40).validate(source_pin)


def test_worldcover_pin_requires_exact_producer_revision_and_union_policy() -> None:
    pin = MapPin(
        "worldcover",
        _SHA_A,
        _SHA_B,
        WORLDCOVER_AREA_MEASUREMENT,
        _SHA_C,
        _SHA_D,
        _SHA_A,
        "union_by_class",
        producer_revision="18c75517c74be22ab2f90edce2b65d3600461048",
    )
    assert pin.producer_revision == "18c75517c74be22ab2f90edce2b65d3600461048"
    with pytest.raises(ValueError, match="pinned v4 producer Git revision"):
        replace(pin, producer_revision="1" * 40)
    with pytest.raises(ValueError, match="preserves the canonical inclusive 80% map policy"):
        replace(pin, support_union_policy="renormalize_observed")
    with pytest.raises(ValueError, match="preserves the canonical inclusive 80% map policy"):
        replace(pin, area_method_id="another-area-method")


def test_plan_reports_missing_selected_input_without_claiming_it_is_verified(
    tmp_path: Path,
) -> None:
    source_pin, census, input_root, _ = _fixture(tmp_path)
    local_shard = input_root / census.shards[0].path
    local_shard.unlink()

    plan = plan_run(_request(), census, source_pin, input_root)

    assert plan.available_shards == ()
    assert plan.missing_shards == (census.shards[0].path,)
    assert plan.expected_input_bytes == census.shards[0].size_bytes
    assert plan.omitted_shard_count == 0
    assert plan.complete_pool is False
    assert plan.evaluation_started is False


def test_local_artifact_sink_is_immutable_bounded_and_path_safe(tmp_path: Path) -> None:
    workspace = tmp_path / "sink-workspace"
    sink = LocalControllerArtifactSink(workspace / "evidence", workspace, 100)

    uri = sink.put("worldcover/attempt.json", b"attempt")
    assert uri == sink.put("worldcover/attempt.json", b"attempt")
    with pytest.raises(ValueError, match="refusing to replace immutable"):
        sink.put("worldcover/attempt.json", b"changed")
    with pytest.raises(ValueError, match="safe and relative"):
        sink.put("../outside.json", b"bad")

    sidecar = b'{"id":"tile-1"}\n'
    sidecar_sha = hashlib.sha256(sidecar).hexdigest()
    source_path = tmp_path / "sidecar.jsonl"
    source_path.write_bytes(sidecar)
    sidecar_uri = sink.put_file(f"worldcover/refs-{sidecar_sha}.jsonl", source_path)
    assert Path(sidecar_uri.removeprefix("file://")).read_bytes() == sidecar

    small_sink = LocalControllerArtifactSink(tmp_path / "small", tmp_path / "small", 1)
    with pytest.raises(WorkspaceQuotaExceededError):
        small_sink.put("too-large", b"12")


def test_buffered_jsonl_writes_reserve_before_the_buffer_can_exceed_workspace_cap(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "buffered-workspace"
    workspace.mkdir()
    value_size = 99 - len(b'{"value":""}\n')
    record = {"value": "x" * value_size}
    temporary = workspace / ".attempts.jsonl.partial"

    with controller_module.WorkspaceQuota(workspace, 99).lease() as lease:
        lease.track_temporary_path(temporary)
        with lease.writer(temporary) as output:
            _write_bounded_line(output, record, 200)
            with pytest.raises(WorkspaceQuotaExceededError):
                _write_bounded_line(output, record, 200)
            assert temporary.stat().st_size == 99

    assert not temporary.exists()


def test_raw_checkpoint_identity_changes_with_snapshot_and_factory(tmp_path: Path) -> None:
    _, census, input_root, raw_rows = _fixture(tmp_path)
    staged_path = input_root / census.shards[0].path

    def factory_for(snapshot_id: str):
        rows = [
            dict(
                row,
                inventory_row_id=hashlib.sha256(
                    f"{snapshot_id}\n{row['native_key_json']}".encode()
                ).hexdigest(),
            )
            for row in raw_rows
        ]

        def read(_path: Path):
            yield from rows

        return read

    first_snapshot = census.source_snapshot_id
    second_snapshot = replace(census, revision="2" * 40).source_snapshot_id
    first_factory = factory_for(first_snapshot)
    second_factory = factory_for(second_snapshot)

    def different_factory(_path: Path):
        yield from raw_rows

    first = RawCheckpointContext(
        _request(),
        census.shards[0],
        staged_path,
        _SHA_A,
        first_snapshot,
        first_factory,
        tmp_path / "workspace" / "scope",
        tmp_path / "workspace",
    )
    same_snapshot_different_factory = replace(first, raw_row_factory=different_factory)
    second = replace(
        first,
        source_snapshot_id=second_snapshot,
        raw_row_factory=second_factory,
    )

    assert _raw_checkpoint_signature(first) != _raw_checkpoint_signature(
        same_snapshot_different_factory
    )
    assert _raw_checkpoint_signature(first) != _raw_checkpoint_signature(second)
    first_path, _ = _build_or_reuse_raw_checkpoint(first)
    second_path, _ = _build_or_reuse_raw_checkpoint(second)
    second_ids = [row["inventory_row_id"] for row in read_raw_checkpoint(second_path)]

    assert first_path != second_path
    assert second_ids == [
        row["inventory_row_id"] for row in factory_for(second_snapshot)(staged_path)
    ]


def test_raw_row_validation_binds_locator_typed_key_and_derived_id(tmp_path: Path) -> None:
    _, census, _, raw_rows = _fixture(tmp_path)
    snapshot_id = census.source_snapshot_id
    row = raw_rows[0]

    validated = _validate_raw_row(
        row,
        0,
        census.shards[0].path,
        10_000,
        snapshot_id,
    )
    assert validated["native_key_json"] == row["native_key_json"]
    with pytest.raises(ValueError, match="locator does not match"):
        _validate_raw_row(
            dict(row, source_row_locator="foreign.parquet#row=0"),
            0,
            census.shards[0].path,
            10_000,
            snapshot_id,
        )
    with pytest.raises(ValueError, match="inventory ID is not derived"):
        _validate_raw_row(
            dict(row, inventory_row_id="f" * 64),
            0,
            census.shards[0].path,
            10_000,
            snapshot_id,
        )
    with pytest.raises(ValueError, match="typed composite key"):
        _validate_raw_row(
            dict(row, native_key_json='[["osm_id",true]]'),
            0,
            census.shards[0].path,
            10_000,
            snapshot_id,
        )


def test_worker_factory_loader_requires_module_callable_syntax() -> None:
    assert _load_factory("osm_polygon_eunis.pool_adapters.controller:plan_run") is plan_run
    with pytest.raises(ValueError, match=r"module\.path:callable"):
        _load_factory("plan_run")
    with pytest.raises(AttributeError):
        _load_factory("osm_polygon_eunis.pool_adapters.controller:not_a_factory")


def test_map_checkpoint_digest_requires_matching_existing_bytes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "attempts.jsonl"
    checkpoint.write_bytes(b"{}\n")
    expected_sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()

    assert _validate_map_checkpoint_digest(checkpoint, expected_sha) == expected_sha
    with pytest.raises(ValueError, match="invalid checkpoint digest"):
        _validate_map_checkpoint_digest(checkpoint, "invalid")
    with pytest.raises(ValueError, match="bytes differ"):
        _validate_map_checkpoint_digest(checkpoint, _SHA_A)
    with pytest.raises(ValueError, match="bytes differ"):
        _validate_map_checkpoint_digest(tmp_path / "missing.jsonl", expected_sha)


def test_cli_plan_and_run_dry_run_print_only_a_noncompletion_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    source_pin, census, input_root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        controller_module,
        "polygon_geometry_tree_pins",
        lambda _source: (source_pin,),
    )
    monkeypatch.setattr(
        RawInventoryCensus,
        "load",
        classmethod(lambda _cls, _path, _sha, _pin: census),
    )
    common = [
        "--source",
        "website",
        "--shard",
        census.shards[0].path,
        "--map",
        "lucas",
        "--input-root",
        str(input_root),
        "--inventory-manifest",
        str(tmp_path / "census.json"),
        "--inventory-sha256",
        _SHA_A,
        "--workspace",
        str(tmp_path / "workspace"),
        "--max-workspace-bytes",
        "10000",
    ]

    assert controller_module.main(["plan", *common]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["selected_maps"] == ["lucas"]
    assert plan["complete_pool"] is False
    assert plan["evaluation_started"] is False

    assert controller_module.main(["run-shard", *common, "--dry-run"]) == 0
    dry_run = json.loads(capsys.readouterr().out)
    assert dry_run == plan


def test_cli_run_requires_an_explicit_worker_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_pin, census, input_root, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        controller_module,
        "polygon_geometry_tree_pins",
        lambda _source: (source_pin,),
    )
    monkeypatch.setattr(
        RawInventoryCensus,
        "load",
        classmethod(lambda _cls, _path, _sha, _pin: census),
    )
    with pytest.raises(SystemExit, match="requires --worker-factory"):
        controller_module.main(
            [
                "run-shard",
                "--source",
                "website",
                "--shard",
                census.shards[0].path,
                "--map",
                "lucas",
                "--input-root",
                str(input_root),
                "--inventory-manifest",
                str(tmp_path / "census.json"),
                "--inventory-sha256",
                _SHA_A,
                "--workspace",
                str(tmp_path / "workspace"),
                "--max-workspace-bytes",
                "10000",
            ]
        )


def test_adapter_checkpoint_serializer_keeps_inventory_and_decision_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inventory = PolygonInventoryRow(
        inventory_row_id="a" * 64,
        native_key=(("osm_id", 7),),
        source_row_locator="region.parquet#row=0",
        polygon_snapshot_id=None,
    )
    polygon = type("PolygonRecord", (), {"inventory": inventory, "snapshot": None})()
    adapter_record = _SerializableAdapterRecord(inventory.inventory_row_id, "completed")
    monkeypatch.setattr(
        controller_module,
        "persist_measurement_sidecars",
        lambda rows, _sink, _pool_name: iter(rows),
    )
    sink = LocalControllerArtifactSink(tmp_path / "artifacts", tmp_path / "workspace", 100)

    result = next(canonical_adapter_checkpoint_rows([(polygon, adapter_record)], sink, "pool"))

    assert result["inventory_row_id"] == inventory.inventory_row_id
    assert result["inventory"]["native_key"] == (("osm_id", 7),)
    assert result["polygon"] is None
    assert result["adapter_record"] == {
        "inventory_row_id": inventory.inventory_row_id,
        "result": "completed",
    }
