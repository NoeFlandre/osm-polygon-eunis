import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis import (
    card_publishing,
    geometry_workers,
    manifest_state,
    publish,
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
    assert (tmp_path / "source" / "website" / "polygons__test.parquet").is_file()


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
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", "a" * 40)
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
    parents: list[str | None] = []

    def fake_upload(api, repo_id, files, *, parent_commit=None):
        del api, repo_id
        parents.append(parent_commit)
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

    def run(parent_commit: str) -> tuple[DatasetCardAccumulator, tuple]:
        card = DatasetCardAccumulator()
        expectations, _ = shard_processing.finalize_dataset(
            cast(HubApi, object()),
            plan,
            shard_processing.FinalizeOptions(
                sidecar_root=tmp_path / "sidecars",
                local_root=tmp_path / "final",
                batch_size=10,
                parent_commit=parent_commit,
                progress=None,
                card=card,
                source_cache_root=None,
                http_client=cast(StreamClient, object()),
            ),
        )
        return card, expectations

    with pytest.raises(RuntimeError, match="interrupted"):
        run("base")
    card, expectations = run("commit-1")

    assert commits == [["polygons/a.parquet"], ["polygons/b.parquet"], ["polygons/b.parquet"]]
    assert parents == ["base", "commit-1", "commit-1"]
    assert [e.path for e in expectations] == ["polygons/a.parquet", "polygons/b.parquet"]
    assert card.total_rows == 2
    assert all(sidecar.exists() for sidecar in (tmp_path / "sidecars").rglob("*.parquet"))


@pytest.mark.parametrize("changed_input", ["sidecar", "reference", "software", "source_commit"])
def test_finalize_dataset_reprocesses_committed_shards_when_inputs_change(
    tmp_path: Path,
    monkeypatch,
    changed_input: str,
) -> None:
    source_commit = "a" * 40
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", source_commit)
    sources = {
        f"polygons/{name}.parquet": pa.table(
            {"polygon_id": [name], "geometry": ['{"type":"Point","coordinates":[0,0]}']}
        )
        for name in ("a", "b")
    }

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision, client
        destination = directory / path.replace("/", "__")
        pq.write_table(sources[path], destination)
        return destination

    commits = 0
    interrupt = True
    uploaded: list[tuple[str, list[str]]] = []

    def fake_upload(api, repo_id, files, *, parent_commit=None):
        nonlocal commits
        del api, repo_id, parent_commit
        commits += 1
        if interrupt and commits == 2:
            raise RuntimeError("interrupted")
        for path, local in files:
            uploaded.append((path, pq.read_table(local)["eunis_code"].to_pylist()))
        return SimpleNamespace(oid=f"commit-{commits}")

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

    def write_sidecars(code: str, version: str) -> None:
        for path in plan.geometry_paths:
            sidecar = release_plan._sidecar_path(tmp_path / "sidecars", plan.spec, path)
            sidecar.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(
                pa.table(
                    {
                        "eunis_code": [code],
                        "eunis_name": ["steppe"],
                        "eunis_overlap_percentage": [75.0],
                        "eunis_source_version": [version],
                    }
                ),
                sidecar,
            )

    def run(reference_version: str) -> None:
        card = DatasetCardAccumulator()
        shard_processing.finalize_dataset(
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
                reference_info={"source_version": reference_version},
            ),
        )

    write_sidecars("R11", "EEA-v1")
    with pytest.raises(RuntimeError, match="interrupted"):
        run("EEA-v1")
    # Shard "a" is committed with the v1 labels before the interruption.
    assert uploaded == [("polygons/a.parquet", ["R11"])]

    uploaded.clear()
    interrupt = False
    expected_code, reference_version = {
        "sidecar": ("C22", "EEA-v1"),
        "reference": ("R11", "EEA-v2"),
        "software": ("R11", "EEA-v1"),
        "source_commit": ("R11", "EEA-v1"),
    }[changed_input]
    change_inputs = {
        "sidecar": lambda: write_sidecars("C22", "EEA-v2"),
        "reference": lambda: None,
        "software": lambda: monkeypatch.setattr(publish, "version", lambda _name: "next-release"),
        "source_commit": lambda: monkeypatch.setenv("EUNIS_SOURCE_COMMIT", "b" * 40),
    }
    change_inputs[changed_input]()
    run(reference_version)

    assert uploaded == [
        ("polygons/a.parquet", [expected_code]),
        ("polygons/b.parquet", [expected_code]),
    ]


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
