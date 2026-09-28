"""Resume and process-lifetime guarantees for geometry jobs."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_eunis import geometry_jobs, reference_staging
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.options import BatchLimits, GeometryPathOptions
from osm_polygon_eunis.release_plan import DatasetPlan, _cached_geometry_path, _sidecar_path
from osm_polygon_eunis.sources import DatasetSpec


def test_reference_signature_is_order_independent_and_uses_current_kernel() -> None:
    expected_payload = json.dumps(
        {
            "checksums": {"a": "1", "b": "2"},
            "threshold": 4,
            "kernel": 3,
            "reference_groups": [],
        },
        sort_keys=True,
    )

    assert (
        geometry_jobs._reference_signature({"b": "2", "a": "1"}, 4)
        == hashlib.sha256(expected_payload.encode("utf-8")).hexdigest()
    )


def test_reference_signature_includes_reference_version_and_labels() -> None:
    asset = _asset("/habitats.gpkg", code=None)
    group = EeaGroup("record", "title", "folder", "service", {"R11": "steppe"}, (), asset)
    revised_asset = replace(asset, source_version="EEA-next")
    relabeled_group = replace(group, labels={"R11": "new name"})

    initial = geometry_jobs._reference_signature({}, 4, (group,))

    assert (
        geometry_jobs._reference_signature({}, 4, (replace(group, vector_asset=revised_asset),))
        != initial
    )
    assert geometry_jobs._reference_signature({}, 4, (relabeled_group,)) != initial


def test_geometry_checkpoint_signature_includes_source_revision_and_shard() -> None:
    spec = DatasetSpec("website", "source", "target", "polygons/*.parquet")
    plan = DatasetPlan(spec, "revision-a", (), (), ())
    initial = geometry_jobs._geometry_checkpoint_signature("references", plan, "polygons/a.parquet")

    assert (
        geometry_jobs._geometry_checkpoint_signature(
            "references", replace(plan, source_revision="revision-b"), "polygons/a.parquet"
        )
        != initial
    )
    assert (
        geometry_jobs._geometry_checkpoint_signature("references", plan, "polygons/b.parquet")
        != initial
    )


def test_completed_batches_reject_old_kernel_marker_without_deleting_it(tmp_path: Path) -> None:
    sidecar = tmp_path / "labels.parquet"
    marker = sidecar.with_name(f"{sidecar.name}.done")
    marker.write_text(json.dumps({"signature": "kernel-v2", "batches": [0, 1]}))

    assert geometry_jobs._completed_batches(sidecar, "kernel-v3") == set()
    assert marker.is_file()
    assert json.loads(marker.read_text()) == {
        "signature": "kernel-v2",
        "batches": [0, 1],
    }


def test_completed_batches_rejects_a_non_list_batch_value(tmp_path: Path) -> None:
    sidecar = tmp_path / "labels.parquet"
    marker = sidecar.with_name(f"{sidecar.name}.done")
    marker.write_text(json.dumps({"signature": "kernel-v3", "batches": "0,1"}))

    assert geometry_jobs._completed_batches(sidecar, "kernel-v3") == set()


def test_checkpoint_batches_keeps_only_exact_integer_batch_ids() -> None:
    payload = {
        "signature": "kernel-v3",
        "batches": [0, True, "1", 2, 1.5],
    }

    assert geometry_jobs._checkpoint_batches(payload, "kernel-v3") == {0, 2}


@pytest.mark.parametrize(
    "payload",
    [None, [], {"signature": "kernel-v2", "batches": [0]}],
)
def test_checkpoint_batches_rejects_missing_or_stale_payload(
    payload: object,
) -> None:
    assert geometry_jobs._checkpoint_batches(payload, "kernel-v3") == set()


def test_completed_batches_returns_empty_for_missing_or_invalid_marker(tmp_path: Path) -> None:
    sidecar = tmp_path / "labels.parquet"

    assert geometry_jobs._completed_batches(sidecar, "kernel-v3") == set()
    sidecar.with_name(f"{sidecar.name}.done").write_text("{")
    assert geometry_jobs._completed_batches(sidecar, "kernel-v3") == set()


def test_geometry_path_reset_discards_stale_sidecar_and_preserves_old_next(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = DatasetSpec("website", "source", "target", "polygons/*.parquet")
    plan = DatasetPlan(spec, "source-revision", (), (), ())
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    monkeypatch.setattr(geometry_jobs, "download_to_temp", lambda *_args, **_kwargs: source)
    sidecar = _sidecar_path(tmp_path / "sidecars", spec, "polygons/a.parquet")
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"stale labels")
    old_next = sidecar.with_name(f"{sidecar.name}.next")
    old_next.write_bytes(b"preserve me")
    current_paths: list[Path | None] = []

    def fake_update(_source, destination, options) -> None:
        current_paths.append(options.current)
        destination.write_bytes(b"fresh labels")

    monkeypatch.setattr(geometry_jobs, "update_label_sidecar", fake_update)
    geometry_jobs._process_geometry_path(
        cast(HubApi, object()),
        plan,
        "polygons/a.parquet",
        (),
        GeometryPathOptions(
            sidecar_root=tmp_path / "sidecars",
            source_root=tmp_path / "sources",
            limits=BatchLimits(parquet_batch_size=8),
            http_client=cast(StreamClient, object()),
            retain_source=True,
            reset_sidecar=True,
        ),
    )

    assert current_paths == [None]
    assert sidecar.read_bytes() == b"fresh labels"
    assert old_next.read_bytes() == b"preserve me"


@pytest.mark.parametrize(
    ("existing_signature", "existing_batches", "expected_processed", "expected_reset"),
    [
        ("kernel-v2", [0], [0, 1], [True, False]),
        ("kernel-v3", [0], [1], [False]),
        ("kernel-v3", [0, 1], [], []),
    ],
)
def test_process_geometry_chunk_resumes_only_matching_checkpoint_batches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_signature: str,
    existing_batches: list[int],
    expected_processed: list[int],
    expected_reset: list[bool],
) -> None:
    spec = DatasetSpec("website", "source", "target", "polygons/*.parquet")
    plan = DatasetPlan(
        spec,
        "source-revision",
        ("polygons/a.parquet",),
        ("polygons/a.parquet",),
        (),
    )
    job = ("website", "polygons/a.parquet")
    sidecar_root = tmp_path / "sidecars"
    sidecar = _sidecar_path(sidecar_root, spec, job[1])
    sidecar.parent.mkdir(parents=True)
    sidecar.write_text("old-kernel-labels")
    marker = sidecar.with_name(f"{sidecar.name}.done")
    current_signature = geometry_jobs._geometry_checkpoint_signature("kernel-v3", plan, job[1])
    marker_signature = "kernel-v2" if existing_signature == "kernel-v2" else current_signature
    marker.write_text(json.dumps({"signature": marker_signature, "batches": existing_batches}))

    events: list[tuple[object, ...]] = []
    group_batches = ((0, ("first",)), (1, ("second",)))
    monkeypatch.setattr(geometry_jobs, "HfApi", lambda **_kwargs: object())
    monkeypatch.setattr(
        geometry_jobs,
        "_indexed_reference_group_batches",
        lambda _groups, **_kwargs: group_batches,
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_geometry_micro_batches",
        lambda _jobs, _limits: ((job,),),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_cache_geometry_jobs",
        lambda _api, _plans, jobs, _root, _client: events.append(("cache", jobs)),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_remove_cached_geometry_jobs",
        lambda _plans, jobs, _root: events.append(("remove", jobs)),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_worker_reference_batch",
        lambda _groups, _root, _threshold, *, start_index: (start_index,),
    )

    def process(_api, _plan, _path, selected_references, options) -> None:
        batch_index = cast(int, selected_references[0])
        events.append(("process", batch_index, options.reset_sidecar))
        if options.reset_sidecar:
            sidecar.write_text("new-kernel-labels")
        else:
            sidecar.write_text(sidecar.read_text() + f"+batch-{batch_index}")

    monkeypatch.setattr(geometry_jobs, "_process_geometry_path", process)
    chunk = geometry_jobs._GeometryChunk(
        groups=(),
        reference_directory=tmp_path,
        plans=(plan,),
        jobs=(job,),
        sidecar_root=sidecar_root,
        source_root=tmp_path / "sources",
        threshold=4,
        limits=BatchLimits(parquet_batch_size=8),
        endpoint="https://hub.test",
        token=None,
        reference_signature="kernel-v3",
    )

    assert geometry_jobs._process_geometry_chunk(chunk) == (job,)
    assert [event[1] for event in events if event[0] == "process"] == expected_processed
    assert [event[2] for event in events if event[0] == "process"] == expected_reset
    assert json.loads(marker.read_text()) == {
        "signature": current_signature,
        "batches": [0, 1],
    }
    if existing_signature == "kernel-v2":
        assert sidecar.read_text() == "new-kernel-labels+batch-1"
    elif expected_processed:
        assert sidecar.read_text() == "old-kernel-labels+batch-1"
    else:
        assert sidecar.read_text() == "old-kernel-labels"


def test_run_geometry_workers_uses_spawn_context(monkeypatch: pytest.MonkeyPatch) -> None:
    methods: list[str] = []

    class FakePool:
        def __init__(self, *, max_workers: int, mp_context) -> None:
            assert max_workers == 1
            methods.append(mp_context.get_start_method())

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def map(self, function, work):
            return (function(item) for item in work)

    monkeypatch.setattr(geometry_jobs, "ProcessPoolExecutor", FakePool)
    monkeypatch.setattr(geometry_jobs, "_process_geometry_chunk", lambda chunk: chunk)
    geometry_jobs._run_geometry_workers(
        cast(tuple[geometry_jobs._GeometryChunk, ...], ("work",)),
        None,
        max_workers=1,
    )

    assert methods == ["spawn"]


def test_geometry_chunks_respect_configured_worker_fanout() -> None:
    jobs = tuple(("website", f"polygons/{index}.parquet") for index in range(5))

    chunks = geometry_jobs._geometry_chunks(
        jobs,
        BatchLimits(workers=2, geometry_tasks_per_worker=1),
    )

    assert len(chunks) == 2
    assert tuple(job for chunk in chunks for job in chunk) == jobs


def test_geometry_micro_batches_respect_retained_shard_limit() -> None:
    jobs = tuple(("website", f"polygons/{index}.parquet") for index in range(5))

    batches = geometry_jobs._geometry_micro_batches(
        jobs,
        BatchLimits(retained_source_shards_per_worker=2),
    )

    assert tuple(map(len, batches)) == (2, 2, 1)


def test_reference_batches_respect_configured_raster_group_limit() -> None:
    groups = tuple(
        EeaGroup(
            record_id=str(index),
            title=str(index),
            folder_url="folder",
            service_url="service",
            labels={},
            raster_assets=(_asset(f"/{index}.tif"),),
            vector_asset=None,
        )
        for index in range(3)
    )

    batches = reference_staging._reference_group_batches(
        groups,
        limits=BatchLimits(raster_groups_per_batch=1),
    )

    assert tuple(map(len, batches)) == (1, 1, 1)


def _asset(path: str, *, code: str | None = "R11") -> RemoteAsset:
    return RemoteAsset(
        path=path,
        url=f"https://example.test/{path.lstrip('/')}",
        size=1,
        etag="etag",
        code=code,
        name="steppe" if code else None,
        record_id="record",
        source_version="EEA-test",
    )


@dataclass(frozen=True, slots=True)
class _RunOverrides:
    api: HubApi | None = None
    workdir: Path | None = None
    sidecar_root: Path | None = None
    source_root: Path | None = None
    threshold: int = 0
    checksums: dict[str, str] | None = None
    batch_size: int = 256
    workers: int = 1
    progress: Callable[[Mapping[str, object]], None] | None = None
    http_client: StreamClient | None = None


_DEFAULT_RUN_OVERRIDES = _RunOverrides()


def _run_options(
    tmp_path: Path,
    plans: tuple[DatasetPlan, ...],
    groups: tuple[EeaGroup, ...],
    overrides: _RunOverrides = _DEFAULT_RUN_OVERRIDES,
) -> geometry_jobs._GeometryRunOptions:
    return geometry_jobs._GeometryRunOptions(
        api=overrides.api or cast(HubApi, object()),
        plans=plans,
        groups=groups,
        sidecar_root=overrides.sidecar_root or tmp_path / "sidecars",
        source_root=overrides.source_root or tmp_path / "source",
        workdir=overrides.workdir or tmp_path,
        threshold=overrides.threshold,
        checksums=overrides.checksums if overrides.checksums is not None else {},
        limits=BatchLimits(
            workers=overrides.workers,
            parquet_batch_size=overrides.batch_size,
        ),
        progress=overrides.progress,
        http_client=overrides.http_client or cast(StreamClient, object()),
    )


def test_process_reference_groups_uses_checkpointed_path_with_one_worker(
    tmp_path: Path,
    monkeypatch,
) -> None:
    group = EeaGroup(
        "record", "title", "folder", "service", {}, (), _asset("/habitats.gpkg", code=None)
    )
    plans = (
        DatasetPlan(
            DatasetSpec("website", "source", "target", "polygons/*.parquet"), "rev", (), ("a",), ()
        ),
    )
    seen: list[tuple[tuple[str, ...], int]] = []
    monkeypatch.setattr(
        geometry_jobs,
        "_process_reference_groups_parallel",
        lambda options: seen.append(
            (tuple(item.record_id for item in options.groups), options.limits.workers)
        ),
    )
    geometry_jobs._process_reference_groups(
        _run_options(
            tmp_path,
            plans,
            (group,),
            _RunOverrides(batch_size=2, workers=1),
        )
    )

    assert seen == [(("record",), 1)]


def test_process_reference_groups_dispatches_streaming_parallel_batches(
    tmp_path: Path,
    monkeypatch,
) -> None:
    group = EeaGroup(
        "record", "title", "folder", "service", {}, (), _asset("/habitats.gpkg", code=None)
    )
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"), "rev", (), ("a",), ()
    )
    seen: list[tuple[tuple[str, ...], int]] = []

    def fake_parallel(options):
        seen.append((tuple(group.record_id for group in options.groups), options.limits.workers))

    monkeypatch.setattr(geometry_jobs, "_process_reference_groups_parallel", fake_parallel)
    geometry_jobs._process_reference_groups(
        _run_options(tmp_path, (plan,), (group,), _RunOverrides(batch_size=2, workers=2))
    )

    assert seen == [(("record",), 2)]


def test_process_geometry_chunk_runs_each_micro_batch_against_each_reference_batch(
    tmp_path: Path, monkeypatch
) -> None:
    events: list[tuple] = []
    plan = DatasetPlan(
        DatasetSpec("dataset", "source", "target", "*.parquet"),
        "source-revision",
        (),
        ("a.parquet", "b.parquet"),
        (),
    )
    monkeypatch.setattr(geometry_jobs, "HfApi", lambda **kwargs: ("api", kwargs))
    monkeypatch.setattr(
        geometry_jobs, "_indexed_reference_group_batches", lambda groups: ((0, groups), (5, groups))
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_indexed_reference_group_batches",
        lambda groups, **_kwargs: ((0, groups), (5, groups)),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_geometry_micro_batches",
        lambda jobs, _limits: (jobs[:1], jobs[1:]),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_cache_geometry_jobs",
        lambda api, plans, jobs, root, client: events.append(("cache", jobs)),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_remove_cached_geometry_jobs",
        lambda plans, jobs, root: events.append(("remove", jobs)),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_worker_reference_batch",
        lambda groups, root, threshold, *, start_index: (f"refs-{start_index}",),
    )

    def fake_process(_api, plan_arg, source_path, references, options):
        assert plan_arg is plan
        assert options.retain_source is True
        assert options.progress is None
        events.append(("process", source_path, references))

    monkeypatch.setattr(geometry_jobs, "_process_geometry_path", fake_process)
    jobs = (("dataset", "a.parquet"), ("dataset", "b.parquet"))
    chunk = geometry_jobs._GeometryChunk(
        groups=(),
        reference_directory=tmp_path,
        plans=(plan,),
        jobs=jobs,
        sidecar_root=tmp_path,
        source_root=tmp_path,
        threshold=0,
        limits=BatchLimits(parquet_batch_size=8),
        endpoint="https://hub.test",
        token=None,
        reference_signature="test-signature",
    )

    assert geometry_jobs._process_geometry_chunk(chunk) == jobs
    assert events == [
        ("cache", jobs[:1]),
        ("process", "a.parquet", ("refs-0",)),
        ("process", "a.parquet", ("refs-5",)),
        ("remove", jobs[:1]),
        ("cache", jobs[1:]),
        ("process", "b.parquet", ("refs-0",)),
        ("process", "b.parquet", ("refs-5",)),
        ("remove", jobs[1:]),
    ]


def test_process_geometry_path_reports_completed_sidecar(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/a.parquet",),
        ("polygons/a.parquet",),
        (),
    )
    source = tmp_path / "source.parquet"
    source.write_bytes(b"source")
    monkeypatch.setattr(
        geometry_jobs,
        "_download_geometry_source",
        lambda *_args, **_kwargs: source,
    )
    monkeypatch.setattr(
        geometry_jobs,
        "update_label_sidecar",
        lambda _source, destination, _options: destination.write_bytes(b"sidecar"),
    )
    events: list[Mapping[str, object]] = []

    geometry_jobs._process_geometry_path(
        cast(HubApi, object()),
        plan,
        "polygons/a.parquet",
        (),
        GeometryPathOptions(
            sidecar_root=tmp_path / "sidecars",
            source_root=tmp_path / "sources",
            limits=BatchLimits(parquet_batch_size=2),
            progress=events.append,
            http_client=cast(StreamClient, object()),
            retain_source=True,
        ),
    )

    assert events == [
        {"event": "sidecar_updated", "dataset": "website", "path": "polygons/a.parquet"}
    ]


def test_parallel_reference_processing_returns_without_geometry_jobs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        (),
        (),
        (),
    )
    monkeypatch.setattr(
        geometry_jobs,
        "_stage_reference_groups",
        lambda **_kwargs: pytest.fail("empty plans must skip reference downloads"),
    )

    geometry_jobs._process_reference_groups_parallel(
        _run_options(tmp_path, (plan,), (), _RunOverrides(batch_size=2, workers=2))
    )


def test_geometry_job_helpers_handle_empty_and_nonempty_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = DatasetSpec("website", "source", "target", "polygons/*.parquet")
    plan = DatasetPlan(spec, "revision", (), ("polygons/a.parquet",), ())
    plans = {"website": plan}
    jobs = (("website", "polygons/a.parquet"),)
    limits = BatchLimits(workers=2)
    assert geometry_jobs._geometry_chunks((), limits) == ()
    assert tuple(geometry_jobs._geometry_micro_batches(())) == ()
    assert tuple(geometry_jobs._geometry_micro_batches(jobs)) == (jobs,)

    downloaded: list[tuple[object, str, Path, bool, object]] = []

    def fake_download(api, selected_plan, source_path, root, *, retain_source, client):
        del api
        downloaded.append((selected_plan, source_path, root, retain_source, client))
        return _cached_geometry_path(root, selected_plan, source_path)

    monkeypatch.setattr(geometry_jobs, "_download_geometry_source", fake_download)
    source_root = tmp_path / "sources"
    source = _cached_geometry_path(source_root, plan, jobs[0][1])
    source.parent.mkdir(parents=True)
    source.write_bytes(b"cached")
    client = cast(StreamClient, object())

    geometry_jobs._cache_geometry_jobs(cast(HubApi, object()), plans, jobs, source_root, client)
    geometry_jobs._remove_cached_geometry_jobs(plans, jobs, source_root)

    assert downloaded == [(plan, jobs[0][1], source_root, True, client)]
    assert not source.exists()


def test_parallel_reference_processing_stages_and_dispatches_geometry_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/a.parquet",),
        ("polygons/a.parquet",),
        (),
    )
    staged: list[tuple[object, ...]] = []
    staged_checksums = {"record:/layer.gpkg": "sha256"}

    @contextmanager
    def fake_stage(groups, *, workdir, checksums, client):
        staged.append((groups, workdir, client))
        checksums.update(staged_checksums)
        yield tmp_path / "references"

    work_units: list[
        tuple[geometry_jobs._GeometryRunOptions, tuple[tuple[str, str], ...], Path, str]
    ] = []

    def fake_work_units(
        options: geometry_jobs._GeometryRunOptions,
        jobs: tuple[tuple[str, str], ...],
        reference_directory: Path,
        reference_signature: str,
    ):
        work_units.append((options, jobs, reference_directory, reference_signature))
        return ("work-unit",)

    worker_calls: list[tuple[object, ...]] = []
    monkeypatch.setattr(geometry_jobs, "_stage_reference_groups", fake_stage)
    monkeypatch.setattr(geometry_jobs, "_geometry_work_units", fake_work_units)
    monkeypatch.setattr(
        geometry_jobs,
        "_run_geometry_workers",
        lambda work, progress, *, max_workers: worker_calls.append((work, progress, max_workers)),
    )
    groups = (
        EeaGroup("record", "title", "folder", "service", {}, (), _asset("/layer.gpkg", code=None)),
    )
    checksums: dict[str, str] = {}
    api = cast(HubApi, object())
    client = cast(StreamClient, object())

    def progress(_event: object) -> None:
        return None

    geometry_jobs._process_reference_groups_parallel(
        _run_options(
            tmp_path,
            (plan,),
            groups,
            _RunOverrides(
                api=api,
                threshold=2,
                checksums=checksums,
                batch_size=4,
                workers=3,
                progress=progress,
                http_client=client,
            ),
        )
    )

    assert staged == [(groups, tmp_path, client)]
    assert checksums == staged_checksums
    assert len(work_units) == 1
    options, jobs, reference_directory, signature = work_units[0]
    assert options.api is api
    assert options.plans == (plan,)
    assert options.limits.workers == 3
    assert jobs == (("website", "polygons/a.parquet"),)
    assert reference_directory == tmp_path / "references"
    assert signature == geometry_jobs._reference_signature(staged_checksums, 2, groups)
    assert worker_calls == [(("work-unit",), progress, 3)]


def test_geometry_chunking_and_progress_reporting_cover_work_items() -> None:
    jobs = (("website", "a.parquet"), ("website", "b.parquet"))
    assert geometry_jobs._geometry_chunks(jobs, BatchLimits(workers=2)) == (
        (jobs[0],),
        (jobs[1],),
    )

    events: list[Mapping[str, object]] = []
    geometry_jobs._report_completed_geometry(jobs, events.append)

    assert events == [
        {"event": "sidecar_updated", "dataset": "website", "path": "a.parquet"},
        {"event": "sidecar_updated", "dataset": "website", "path": "b.parquet"},
    ]


@pytest.mark.parametrize(
    ("api", "expected_endpoint", "expected_token"),
    [
        (
            SimpleNamespace(endpoint="https://hub.test", token="secret"),
            "https://hub.test",
            "secret",
        ),
        (SimpleNamespace(), "https://huggingface.co", None),
    ],
)
def test_geometry_work_units_capture_hub_connection_settings(
    api: object,
    expected_endpoint: str,
    expected_token: str | None,
) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        (),
        ("polygons/a.parquet",),
        (),
    )

    options = _run_options(
        Path("run"),
        (plan,),
        (),
        _RunOverrides(api=cast(HubApi, api), threshold=2, batch_size=4),
    )
    work = geometry_jobs._geometry_work_units(
        options,
        (("website", "polygons/a.parquet"),),
        Path("references"),
        "signature",
    )

    assert len(work) == 1
    assert work[0].endpoint == expected_endpoint
    assert work[0].token == expected_token
