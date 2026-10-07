import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis import (
    card_publishing,
    geometry_workers,
    manifest_state,
    release_orchestration,
    release_plan,
    shard_processing,
)
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.cards import CardArtifacts, DatasetCardAccumulator
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.geometry_jobs import process_geometry_paths
from osm_polygon_eunis.options import BatchLimits, GeometryPathOptions, ReleaseOptions
from osm_polygon_eunis.publish import ShardExpectation, VerificationReceipt
from osm_polygon_eunis.release_plan import DatasetPlan, DatasetReceipt
from osm_polygon_eunis.sources import DatasetSpec

# Remaining private ``release_orchestration._*`` references are limited to seams with no
# public entry point: monkeypatch targets that stub network/HF side effects for
# ``run_release``, and the orchestration/manifest helpers that ``run_release`` only
# reaches after live I/O.


class _Reference:
    def __init__(self, result: EunisResult) -> None:
        self.result = result

    def overlap(self, polygon) -> EunisResult:
        del polygon
        return self.result


def test_process_geometry_paths_keeps_only_compact_sidecar_state(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["a", "b"],
                "geometry": [
                    '{"type":"Point","coordinates":[0,0]}',
                    '{"type":"Point","coordinates":[1,1]}',
                ],
            }
        ),
        source,
    )
    downloads = {"polygons/test.parquet": source}

    clients: list[object] = []

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision
        clients.append(client)
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(downloads[path].read_bytes())
        return destination

    monkeypatch.setattr(geometry_workers, "download_to_temp", fake_download)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/test.parquet",),
        ("polygons/test.parquet",),
        (),
    )
    sidecar_root = tmp_path / "sidecars"
    sidecar = sidecar_root / "website" / "polygons__test.parquet.labels.parquet"
    sidecar.parent.mkdir(parents=True)
    pending_next_sidecar = sidecar.with_name(f"{sidecar.name}.next")
    pending_next_sidecar.write_bytes(b"preserved recovery evidence")
    reusable_client = cast(StreamClient, object())

    process_geometry_paths(
        cast(HubApi, object()),
        plan,
        reference=_Reference(EunisResult("R11", "steppe", 25.0, "test")),
        options=GeometryPathOptions(
            sidecar_root=sidecar_root,
            source_root=tmp_path / "source",
            limits=BatchLimits(parquet_batch_size=1),
            http_client=reusable_client,
        ),
    )

    assert clients == [reusable_client]
    assert pq.read_table(sidecar)["eunis_code"].to_pylist() == ["R11", "R11"]
    assert pending_next_sidecar.read_bytes() == b"preserved recovery evidence"
    assert list((tmp_path / "source").iterdir()) == []


def test_process_geometry_paths_reuses_retained_source_shard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "input.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["a"],
                "geometry": ['{"type":"Point","coordinates":[0,0]}'],
            }
        ),
        source,
    )
    calls: list[str] = []

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision, client
        calls.append(path)
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr(geometry_workers, "download_to_temp", fake_download)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/test.parquet",),
        ("polygons/test.parquet",),
        (),
    )

    for _ in range(2):
        process_geometry_paths(
            cast(HubApi, object()),
            plan,
            reference=_Reference(EunisResult("R11", "steppe", 25.0, "test")),
            options=GeometryPathOptions(
                sidecar_root=tmp_path / "sidecars",
                source_root=tmp_path / "source",
                limits=BatchLimits(parquet_batch_size=1),
                retain_source=True,
            ),
        )

    assert calls == ["polygons/test.parquet"]
    assert (tmp_path / "source" / "website" / "revision" / "polygons__test.parquet").is_file()


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


@pytest.mark.parametrize(
    ("contents", "message"),
    [
        (None, "not found"),
        (b"\xff", "cannot be read"),
        ("[]", "must be an object"),
        ('{"source_version":"","crs":"EPSG:3035","threshold":0}', "missing source_version"),
        ('{"source_version":"ok","crs":null,"threshold":0}', "missing crs"),
        ('{"source_version":"ok","crs":"EPSG:3035","threshold":-1}', "invalid threshold"),
        ("{", "not valid JSON"),
    ],
)
def test_run_release_rejects_invalid_reference_config(
    tmp_path: Path, contents: str | bytes | None, message: str
) -> None:
    config = tmp_path / "config.json"
    if isinstance(contents, bytes):
        config.write_bytes(contents)
    elif contents is not None:
        config.write_text(contents, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        release_orchestration.run_release(
            cast(HubApi, object()),
            ReleaseOptions(
                reference_config=config,
                workdir=tmp_path / "run",
                limits=BatchLimits(parquet_batch_size=1),
            ),
        )
    assert not (tmp_path / "run").exists()


def test_finalize_dataset_enriches_polygon_and_link_shards_and_cleans_staging(
    tmp_path: Path,
    monkeypatch,
) -> None:
    geometry_bytes = pa.table(
        {
            "polygon_id": ["a", "b"],
            "geometry": [
                '{"type":"Point","coordinates":[0,0]}',
                '{"type":"Point","coordinates":[1,1]}',
            ],
        }
    )
    link_bytes = pa.table({"polygon_id": ["a", "missing"]})
    source_files = {
        "polygons/region.parquet": geometry_bytes,
        "polygon_document_links/region.parquet": link_bytes,
    }

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision, client
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(source_files[path], destination)
        return destination

    uploaded: list[tuple[str, Path]] = []

    def fake_upload(api, repo_id, files, *, parent_commit=None):
        del api, repo_id, parent_commit
        uploaded.extend(files)
        return SimpleNamespace(oid="commit-1")

    monkeypatch.setattr(shard_processing, "download_to_temp", fake_download)
    monkeypatch.setattr(shard_processing, "upload_replacements", fake_upload)
    plan = DatasetPlan(
        DatasetSpec(
            "wikidata", "source", "target", "polygons/*.parquet", "polygon_document_links/*.parquet"
        ),
        "source-revision",
        ("polygons/region.parquet", "polygon_document_links/region.parquet"),
        ("polygons/region.parquet",),
        ("polygon_document_links/region.parquet",),
    )
    sidecar = release_plan._sidecar_path(tmp_path / "sidecars", plan.spec, plan.geometry_paths[0])
    sidecar.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "eunis_code": ["R11", None],
                "eunis_name": ["steppe", None],
                "eunis_overlap_percentage": [75.0, None],
                "eunis_source_version": ["EEA-test", None],
            }
        ),
        sidecar,
    )

    progress: list[dict[str, object]] = []

    def capture_progress(event) -> None:
        progress.append(dict(event))

    expectations, commit = shard_processing.finalize_dataset(
        cast(HubApi, object()),
        plan,
        shard_processing.FinalizeOptions(
            sidecar_root=tmp_path / "sidecars",
            local_root=tmp_path / "final",
            batch_size=1,
            parent_commit="base",
            progress=capture_progress,
            card=DatasetCardAccumulator(),
            source_cache_root=None,
            http_client=cast(StreamClient, object()),
            input_identity="test-input-identity",
        ),
    )

    assert commit == "commit-1"
    assert [expectation.path for expectation in expectations] == [
        "polygons/region.parquet",
        "polygon_document_links/region.parquet",
    ]
    assert [path for path, _ in uploaded] == [
        "polygons/region.parquet",
        "polygon_document_links/region.parquet",
    ]
    assert progress == [
        {"event": "shards_uploaded", "dataset": "wikidata", "path": "polygons/region.parquet"}
    ]
    assert sidecar.exists()
    assert [path.name for path in (tmp_path / "final").iterdir()] == ["wikidata.finalize.json"]


def test_finalize_dataset_resumes_after_the_last_committed_group(
    tmp_path: Path,
    monkeypatch,
) -> None:
    sources = {}
    for name in ("a", "b"):
        sources[f"polygons/{name}.parquet"] = pa.table(
            {"polygon_id": [name], "geometry": ['{"type":"Point","coordinates":[0,0]}']}
        )

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision, client
        destination = directory / path.replace("/", "__")
        pq.write_table(sources[path], destination)
        return destination

    commits: list[list[str]] = []

    def fake_upload(api, repo_id, files, *, parent_commit=None):
        del api, repo_id, parent_commit
        commits.append([path for path, _ in files])
        if len(commits) == 2:
            raise RuntimeError("interrupted")
        return SimpleNamespace(oid=f"commit-{len(commits)}")

    monkeypatch.setattr(shard_processing, "download_to_temp", fake_download)
    monkeypatch.setattr(shard_processing, "upload_replacements", fake_upload)
    monkeypatch.setattr(shard_processing, "_COMMIT_MAX_SHARDS", 1)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        tuple(sources),
        tuple(sources),
        (),
    )
    for path in plan.geometry_paths:
        sidecar = release_plan._sidecar_path(tmp_path / "sidecars", plan.spec, path)
        sidecar.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(
            pa.table(
                {
                    "eunis_code": ["R11"],
                    "eunis_name": ["steppe"],
                    "eunis_overlap_percentage": [75.0],
                    "eunis_source_version": ["EEA-test"],
                }
            ),
            sidecar,
        )

    def run() -> tuple[DatasetCardAccumulator, tuple]:
        card = DatasetCardAccumulator()
        expectations, _ = shard_processing.finalize_dataset(
            cast(HubApi, object()),
            plan,
            shard_processing.FinalizeOptions(
                sidecar_root=tmp_path / "sidecars",
                local_root=tmp_path / "final",
                batch_size=10,
                parent_commit="base",
                progress=None,
                card=card,
                source_cache_root=None,
                http_client=cast(StreamClient, object()),
                input_identity="test-input-identity",
            ),
        )
        return card, expectations

    with pytest.raises(RuntimeError, match="interrupted"):
        run()
    card, expectations = run()

    assert commits == [["polygons/a.parquet"], ["polygons/b.parquet"], ["polygons/b.parquet"]]
    assert [e.path for e in expectations] == ["polygons/a.parquet", "polygons/b.parquet"]
    assert card.total_rows == 2
    assert all(sidecar.exists() for sidecar in (tmp_path / "sidecars").rglob("*.parquet"))


def _write_finalize_sidecar(sidecar_root: Path, plan: DatasetPlan, path: str, code: str) -> None:
    sidecar = release_plan._sidecar_path(sidecar_root, plan.spec, path)
    sidecar.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.table(
            {
                "eunis_code": [code],
                "eunis_name": [f"name-{code}"],
                "eunis_overlap_percentage": [75.0],
                "eunis_source_version": ["EEA-test"],
            }
        ),
        sidecar,
    )


def _track_card_restores(monkeypatch) -> list[Mapping[str, Any]]:
    snapshots: list[Mapping[str, Any]] = []
    original_restore = DatasetCardAccumulator.restore

    def capture_restore(card: DatasetCardAccumulator, state: Mapping[str, Any]) -> None:
        snapshots.append(state)
        original_restore(card, state)

    monkeypatch.setattr(DatasetCardAccumulator, "restore", capture_restore)
    return snapshots


def _finalize_resume_harness(tmp_path: Path, monkeypatch):
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("polygons/a.parquet", "polygons/b.parquet"),
        ("polygons/a.parquet", "polygons/b.parquet"),
        (),
    )
    source_tables = {
        path: pa.table(
            {
                "polygon_id": [Path(path).stem],
                "geometry": ['{"type":"Point","coordinates":[0,0]}'],
            }
        )
        for path in plan.geometry_paths
    }
    sidecar_root = tmp_path / "sidecars"
    for path in plan.geometry_paths:
        _write_finalize_sidecar(sidecar_root, plan, path, "R11")

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision, client
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(source_tables[path], destination)
        return destination

    monkeypatch.setattr(shard_processing, "download_to_temp", fake_download)
    monkeypatch.setattr(shard_processing, "_COMMIT_MAX_SHARDS", 1)

    progress_path = tmp_path / "final" / "website.finalize.json"
    upload_paths: list[str] = []
    uploaded: dict[str, bytes] = {}
    fail_next_path: list[str] = ["polygons/b.parquet"]
    checkpoint_seen_by_upload: list[bytes | None] = []

    def fake_upload(api, repo_id, files, *, parent_commit=None):
        del api, repo_id, parent_commit
        checkpoint_seen_by_upload.append(
            progress_path.read_bytes() if progress_path.is_file() else None
        )
        staged = [(path, local_path.read_bytes()) for path, local_path in files]
        upload_paths.extend(path for path, _ in staged)
        if fail_next_path and any(path == fail_next_path[0] for path, _ in staged):
            fail_next_path.clear()
            raise RuntimeError("interrupted upload")
        uploaded.update(staged)
        return SimpleNamespace(oid=f"target-commit-{len(upload_paths)}")

    monkeypatch.setattr(shard_processing, "upload_replacements", fake_upload)

    events_by_run: list[list[dict[str, object]]] = []
    card_rows: list[int] = []
    restore_snapshots = _track_card_restores(monkeypatch)
    target_revisions: list[str] = []

    def capture_revision(api, repo_id):
        del api, repo_id
        revision = f"target-head-{len(target_revisions) + 1}"
        target_revisions.append(revision)
        return revision

    def write_card(card, workdir, selected_plan, reference_info):
        del workdir, selected_plan, reference_info
        card_rows.append(card.total_rows)
        return CardArtifacts({}, {}, {})

    monkeypatch.setattr(card_publishing, "capture_revision", capture_revision)
    monkeypatch.setattr(card_publishing, "_download_source_readme", lambda *args: None)
    monkeypatch.setattr(card_publishing, "_write_card_artifacts", write_card)
    monkeypatch.setattr(card_publishing, "_upload_card_artifacts", lambda *args: "card-commit")
    monkeypatch.setattr(
        card_publishing,
        "_publish_dataset_manifest",
        lambda *args, **kwargs: ({}, (), ()),
    )
    monkeypatch.setattr(card_publishing, "_shared_blobs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        card_publishing,
        "_verify_final_dataset",
        lambda *args, **kwargs: VerificationReceipt("target", "verified", {}, (), None),
    )

    def run(
        *,
        reference_info: Mapping[str, object] | None = None,
        selected_plan: DatasetPlan = plan,
    ) -> DatasetReceipt:
        events: list[dict[str, object]] = []
        events_by_run.append(events)

        def capture(event: Mapping[str, object]) -> None:
            events.append(dict(event))

        return card_publishing._finalize_plan(
            cast(HubApi, object()),
            selected_plan,
            options=card_publishing._PlanOptions(
                sidecar_root=sidecar_root,
                workdir=tmp_path,
                batch_size=1,
                reference_info=reference_info
                or {
                    "source_version": "EEA-test",
                    "assets": [{"path": "Prob_R11.tif", "sha256": "reference-a"}],
                },
                progress=capture,
                http_client=cast(StreamClient, object()),
                source_cache_root=None,
                max_intersection_errors=None,
            ),
        )

    return SimpleNamespace(
        plan=plan,
        sidecar_root=sidecar_root,
        progress_path=progress_path,
        run=run,
        upload_paths=upload_paths,
        uploaded=uploaded,
        fail_next_path=fail_next_path,
        checkpoint_seen_by_upload=checkpoint_seen_by_upload,
        events_by_run=events_by_run,
        card_rows=card_rows,
        restore_snapshots=restore_snapshots,
        target_revisions=target_revisions,
    )


def _record_partial_finalize_progress(harness) -> bytes:
    with pytest.raises(RuntimeError, match="interrupted upload"):
        harness.run()
    assert harness.progress_path.is_file()
    return harness.progress_path.read_bytes()


def _assert_finalize_inputs_rejected_without_writes(harness, run) -> None:
    saved_progress = harness.progress_path.read_bytes()
    prior_upload_paths = list(harness.upload_paths)
    prior_restore_count = len(harness.restore_snapshots)
    with pytest.raises(RuntimeError, match=r"finalize progress.*inputs.*changed"):
        run()
    assert harness.progress_path.read_bytes() == saved_progress
    assert harness.upload_paths == prior_upload_paths
    assert harness.events_by_run[-1] == []
    assert len(harness.restore_snapshots) == prior_restore_count


def test_finalize_plan_resumes_unchanged_inputs_after_an_interrupted_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _finalize_resume_harness(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="interrupted upload"):
        harness.run()
    progress = json.loads(harness.progress_path.read_text(encoding="utf-8"))
    assert isinstance(progress.get("input_identity"), str)
    assert harness.upload_paths == ["polygons/a.parquet", "polygons/b.parquet"]

    receipt = harness.run()

    assert harness.upload_paths == [
        "polygons/a.parquet",
        "polygons/b.parquet",
        "polygons/b.parquet",
    ]
    assert set(harness.uploaded) == set(harness.plan.geometry_paths)
    assert [expectation.path for expectation in receipt.expectations] == list(
        harness.plan.geometry_paths
    )
    assert harness.card_rows == [2]
    assert len(harness.restore_snapshots) == 1
    assert harness.target_revisions == ["target-head-1", "target-head-2"]
    assert not harness.progress_path.exists()


def test_finalize_plan_rejects_changed_committed_sidecar_before_skipping_or_uploading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _finalize_resume_harness(tmp_path, monkeypatch)
    _record_partial_finalize_progress(harness)
    _write_finalize_sidecar(harness.sidecar_root, harness.plan, "polygons/a.parquet", "R12")

    _assert_finalize_inputs_rejected_without_writes(harness, harness.run)


def test_finalize_plan_rejects_changed_reference_before_skipping_or_uploading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _finalize_resume_harness(tmp_path, monkeypatch)
    _record_partial_finalize_progress(harness)

    def run_with_new_reference():
        return harness.run(
            reference_info={
                "source_version": "EEA-test",
                "assets": [{"path": "Prob_R11.tif", "sha256": "reference-b"}],
            }
        )

    _assert_finalize_inputs_rejected_without_writes(harness, run_with_new_reference)


@pytest.mark.parametrize("change", ["revision", "layout"])
def test_finalize_plan_rejects_changed_source_plan_before_skipping_or_uploading(
    tmp_path: Path,
    monkeypatch,
    change: str,
) -> None:
    harness = _finalize_resume_harness(tmp_path, monkeypatch)
    _record_partial_finalize_progress(harness)
    selected_plan = harness.plan
    if change == "revision":
        selected_plan = DatasetPlan(
            selected_plan.spec,
            "source-revision-next",
            selected_plan.source_files,
            selected_plan.geometry_paths,
            selected_plan.link_paths,
        )
    else:
        selected_plan = DatasetPlan(
            selected_plan.spec,
            selected_plan.source_revision,
            (*selected_plan.source_files, "metadata.json"),
            selected_plan.geometry_paths,
            selected_plan.link_paths,
        )

    def run_with_changed_plan():
        return harness.run(selected_plan=selected_plan)

    _assert_finalize_inputs_rejected_without_writes(harness, run_with_changed_plan)


def test_finalize_plan_rejects_changed_code_identity_before_skipping_or_uploading(
    tmp_path: Path,
    monkeypatch,
) -> None:
    code_identity = ["code-v1"]
    monkeypatch.setattr(
        card_publishing,
        "_output_code_identity",
        lambda: code_identity[0],
        raising=False,
    )
    harness = _finalize_resume_harness(tmp_path, monkeypatch)
    _record_partial_finalize_progress(harness)
    code_identity[0] = "code-v2"

    _assert_finalize_inputs_rejected_without_writes(harness, harness.run)


def test_output_code_identity_changes_when_the_dependency_lock_changes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project_root = tmp_path / "project"
    package_root = project_root / "src" / "osm_polygon_eunis"
    package_root.mkdir(parents=True)
    (package_root / "card_publishing.py").write_text("source = 'same'\n", encoding="utf-8")
    (project_root / "pyproject.toml").write_text("[project]\n", encoding="utf-8")
    lock = project_root / "uv.lock"
    lock.write_text("dependency = 'one'\n", encoding="utf-8")
    monkeypatch.setattr(card_publishing, "__file__", str(package_root / "card_publishing.py"))

    original_identity = card_publishing._output_code_identity()
    lock.write_text("dependency = 'two'\n", encoding="utf-8")

    assert card_publishing._output_code_identity() != original_identity


def test_finalize_plan_persists_identity_before_a_failed_first_upload(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _finalize_resume_harness(tmp_path, monkeypatch)
    harness.fail_next_path[:] = ["polygons/a.parquet"]

    with pytest.raises(RuntimeError, match="interrupted upload"):
        harness.run()

    assert harness.progress_path.is_file()
    progress = json.loads(harness.progress_path.read_text(encoding="utf-8"))
    assert isinstance(progress.get("input_identity"), str)
    assert progress["expectations"] == []
    assert harness.checkpoint_seen_by_upload[0] is not None


def test_planning_manifest_and_shared_blobs_are_deterministic(monkeypatch) -> None:
    entries = {
        "NoeFlandre/osm-polygon-website-tag": (
            SimpleNamespace(path="README.md"),
            SimpleNamespace(path="polygons/a.parquet"),
        ),
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia": (
            SimpleNamespace(path="polygons/a.parquet", blob_id="polygon"),
            SimpleNamespace(path="polygon_document_links/a.parquet", blob_id="link"),
            SimpleNamespace(path="wikipedia/a.parquet", blob_id="wikipedia"),
        ),
        "NoeFlandre/osm-polygon-description-tag": (SimpleNamespace(path="data/a.parquet"),),
    }
    monkeypatch.setattr(release_plan, "capture_revision", lambda api, repo: f"rev:{repo}")
    monkeypatch.setattr(release_plan, "list_repo_files", lambda api, repo, revision: entries[repo])
    monkeypatch.setattr(
        manifest_state, "list_repo_files", lambda api, repo, revision: entries[repo]
    )

    plans = release_plan.plan_datasets(cast(HubApi, object()))

    assert [plan.spec.name for plan in plans] == ["website", "wikidata", "description"]
    assert plans[1].link_paths == ("polygon_document_links/a.parquet",)
    with pytest.raises(ValueError, match="no geometry"):
        release_plan._geometry_paths(
            DatasetSpec("empty", "source", "target", "polygons/*.parquet"), ("README.md",)
        )

    raster = _asset("/Prob_R11.tif")
    vector = _asset("/habitats.gpkg", code=None)
    group = EeaGroup("record", "title", "folder", "service", {}, (raster,), vector)
    manifest = manifest_state._reference_manifest(
        (group,),
        {"record:/Prob_R11.tif": "sha"},
        source_version="EEA-test",
        crs="EPSG:3035",
        threshold=0,
        config={"classification_record": "class"},
    )
    assert manifest == {
        "source_version": "EEA-test",
        "crs": "EPSG:3035",
        "threshold": 0,
        "classification_record": "class",
        "assets": [
            {
                "record_id": "record",
                "path": "/Prob_R11.tif",
                "url": "https://example.test/Prob_R11.tif",
                "size": 1,
                "etag": "etag",
                "code": "R11",
                "name": "steppe",
                "sha256": "sha",
            },
            {
                "record_id": "record",
                "path": "/habitats.gpkg",
                "url": "https://example.test/habitats.gpkg",
                "size": 1,
                "etag": "etag",
                "code": None,
                "name": None,
                "sha256": None,
            },
        ],
    }

    shared = manifest_state._shared_blobs(cast(HubApi, object()), plans[1], {"polygons/a.parquet"})
    assert shared == {
        "polygon_document_links/a.parquet": "link",
        "wikipedia/a.parquet": "wikipedia",
    }


# Scheduling invariant: every plan's geometry job is processed once, in plan order.


def test_finalize_plan_builds_manifest_and_verifies_target(monkeypatch, tmp_path: Path) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )
    expectation = ShardExpectation("polygons/a.parquet", 2, "schema")
    verification = VerificationReceipt(
        "target", "verified", {expectation.path: 2}, ("README.md",), None
    )
    calls = []
    source_readmes = []
    monkeypatch.setattr(card_publishing, "capture_revision", lambda api, repo: "target-base")
    monkeypatch.setattr(
        card_publishing,
        "_download_source_readme",
        lambda *args: "pinned source card",
    )
    monkeypatch.setattr(
        card_publishing,
        "finalize_dataset",
        lambda *args, **kwargs: ((expectation,), "commit-1"),
    )
    monkeypatch.setattr(
        card_publishing,
        "upload_manifest",
        lambda *args, **kwargs: SimpleNamespace(oid="commit-2"),
    )
    monkeypatch.setattr(manifest_state, "verify_dataset", lambda *args, **kwargs: verification)

    class FakeCard:
        def __init__(self, *, source_readme: str | None = None) -> None:
            source_readmes.append(source_readme)

        def write_artifacts(self, directory: Path, **kwargs) -> CardArtifacts:
            del kwargs
            directory.mkdir(parents=True, exist_ok=True)
            readme = directory / "README.md"
            world_map = directory / "world-map.svg"
            readme.write_text("card", encoding="utf-8")
            world_map.write_text("map", encoding="utf-8")
            return CardArtifacts(
                {"README.md": readme, "eunis/world-map.svg": world_map},
                {"README.md": "readme", "eunis/world-map.svg": "map"},
                {"total_rows": 2},
            )

    monkeypatch.setattr(card_publishing, "DatasetCardAccumulator", FakeCard)
    monkeypatch.setattr(card_publishing, "_upload_file", lambda *args, **kwargs: "commit-card")

    def fake_build_manifest(options):
        calls.append(options)
        return {"manifest": True}

    monkeypatch.setattr(card_publishing, "build_manifest", fake_build_manifest)
    monkeypatch.setattr(
        card_publishing,
        "_shared_blobs",
        lambda *args, **kwargs: {"README.md": "blob"},
    )
    sidecar = release_plan._sidecar_path(tmp_path / "sidecars", plan.spec, plan.geometry_paths[0])
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"label sidecar")

    result = card_publishing._finalize_plan(
        cast(HubApi, object()),
        plan,
        options=card_publishing._PlanOptions(
            sidecar_root=tmp_path / "sidecars",
            workdir=tmp_path,
            batch_size=2,
            reference_info={"source_version": "EEA-test"},
            progress=None,
            http_client=cast(StreamClient, object()),
            source_cache_root=None,
            max_intersection_errors=None,
        ),
    )

    assert result.verification is verification
    assert source_readmes == ["pinned source card"]
    assert calls[0].changed_paths == ("polygons/a.parquet", "README.md")
    assert calls[0].added_paths == ("eunis/world-map.svg",)


def test_run_release_coordinates_pooled_processing(monkeypatch, tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 0}),
        encoding="utf-8",
    )
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "rev",
        (),
        (),
        (),
    )
    group = EeaGroup("record", "title", "folder", "service", {}, (), _asset("/x.gpkg", code=None))
    receipt = DatasetReceipt(plan, (), VerificationReceipt("target", "verified", {}, (), None))
    seen: list[tuple[object, int]] = []
    monkeypatch.setattr(release_orchestration, "plan_datasets", lambda api, names=None: (plan,))
    monkeypatch.setattr(release_orchestration, "_duplicate_outputs", lambda *args: None)
    monkeypatch.setattr(manifest_state, "_load_existing_manifest", lambda *args: None)
    monkeypatch.setattr(release_orchestration, "resolve_config_data", lambda config: (group,))
    monkeypatch.setattr(
        release_orchestration,
        "_process_reference_groups",
        lambda options: seen.append((options.http_client, options.limits.workers)),
    )
    monkeypatch.setattr(release_orchestration, "_finalize_plan", lambda *args, **kwargs: receipt)

    result = release_orchestration.run_release(
        cast(HubApi, object()),
        ReleaseOptions(
            reference_config=config,
            workdir=tmp_path / "run",
            limits=BatchLimits(parquet_batch_size=2),
        ),
    )

    assert result.datasets == (receipt,)
    assert len(seen) == 1
    assert seen[0][1] == BatchLimits().workers
