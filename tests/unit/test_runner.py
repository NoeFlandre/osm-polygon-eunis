import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_eunis.runner as runner
from osm_polygon_eunis.cards import CardArtifacts
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.publish import ShardExpectation, VerificationReceipt
from osm_polygon_eunis.runner import DatasetPlan, DatasetReceipt, process_geometry_paths
from osm_polygon_eunis.sources import DatasetSpec


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

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision
        del client
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(downloads[path].read_bytes())
        return destination

    monkeypatch.setattr(runner, "download_to_temp", fake_download)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/test.parquet",),
        ("polygons/test.parquet",),
        (),
    )

    process_geometry_paths(
        object(),
        plan,
        reference=_Reference(EunisResult("R11", "steppe", 25.0, "test")),
        sidecar_root=tmp_path / "sidecars",
        source_root=tmp_path / "source",
        batch_size=1,
    )

    sidecar = tmp_path / "sidecars" / "website" / "polygons__test.parquet.labels.parquet"
    assert pq.read_table(sidecar)["eunis_code"].to_pylist() == ["R11", "R11"]
    assert list((tmp_path / "source").iterdir()) == []


def test_process_geometry_paths_forwards_a_reusable_http_client(
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
    calls: list[object] = []

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision
        calls.append(client)
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        return destination

    monkeypatch.setattr(runner, "download_to_temp", fake_download)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/test.parquet",),
        ("polygons/test.parquet",),
        (),
    )
    reusable_client = object()

    process_geometry_paths(
        object(),
        plan,
        reference=_Reference(EunisResult("R11", "steppe", 25.0, "test")),
        sidecar_root=tmp_path / "sidecars",
        source_root=tmp_path / "source",
        batch_size=1,
        http_client=reusable_client,
    )

    assert calls == [reusable_client]


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

    monkeypatch.setattr(runner, "download_to_temp", fake_download)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/test.parquet",),
        ("polygons/test.parquet",),
        (),
    )

    for _ in range(2):
        process_geometry_paths(
            object(),
            plan,
            reference=_Reference(EunisResult("R11", "steppe", 25.0, "test")),
            sidecar_root=tmp_path / "sidecars",
            source_root=tmp_path / "source",
            batch_size=1,
            retain_source=True,
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


def test_settings_and_reference_metadata_are_validated(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 2}),
        encoding="utf-8",
    )

    assert runner._settings(config)[:3] == ("EEA-test", "EPSG:3035", 2)
    assert (
        runner._sidecar_path(
            tmp_path,
            DatasetSpec("website", "source", "target", "polygons/*.parquet"),
            "polygons/a.parquet",
        ).name
        == "polygons__a.parquet.labels.parquet"
    )
    assert runner._asset_filename(_asset("/folder/Prob_R11_100M.TIF")) == "R11.tif"
    assert runner._asset_filename(_asset("/folder/reference.gpkg", code=None)) == "reference.gpkg"
    assert (
        runner._asset_key(
            EeaGroup("record", "title", "folder", "service", {}, (), None),
            _asset("/asset.tif"),
        )
        == "record:/asset.tif"
    )

    config.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="must be an object"):
        runner._settings(config)
    for field, value in (("source_version", ""), ("crs", None), ("threshold", -1)):
        payload = {"source_version": "ok", "crs": "EPSG:3035", "threshold": 0}
        payload[field] = value
        config.write_text(json.dumps(payload), encoding="utf-8")
        message = "invalid threshold" if field == "threshold" else f"missing {field}"
        with pytest.raises(ValueError, match=message):
            runner._settings(config)


def test_open_reference_group_reuses_client_for_raster_and_vector(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class FakeReference:
        def __init__(self, *args, **kwargs) -> None:
            self.args = args
            self.kwargs = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    downloaded: list[tuple[str, object]] = []

    def fake_download(client, asset, destination):
        downloaded.append((asset.path, client))
        destination.write_bytes(b"asset")
        return f"sha-{asset.path}"

    monkeypatch.setattr(runner, "download_asset", fake_download)
    monkeypatch.setattr(runner, "RasterReference", FakeReference)
    monkeypatch.setattr(runner, "GeoPackageReference", FakeReference)
    client = object()
    checksums: dict[str, str] = {}
    raster_group = EeaGroup(
        "raster-record",
        "raster",
        "folder",
        "service",
        {"R11": "steppe"},
        (_asset("/Prob_R11.tif"),),
        None,
    )
    vector_group = EeaGroup(
        "vector-record",
        "vector",
        "folder",
        "service",
        {"Q11": "bog"},
        (),
        _asset("/habitats.gpkg", code=None),
    )

    with runner.open_reference_group(
        raster_group, tmp_path / "raster", threshold=0, checksums=checksums, client=client
    ) as reference:
        assert isinstance(reference, FakeReference)
    with runner.open_reference_group(
        vector_group, tmp_path / "vector", threshold=1, checksums=checksums, client=client
    ) as reference:
        assert isinstance(reference, FakeReference)

    assert downloaded == [("/Prob_R11.tif", client), ("/habitats.gpkg", client)]
    assert checksums == {
        "raster-record:/Prob_R11.tif": "sha-/Prob_R11.tif",
        "vector-record:/habitats.gpkg": "sha-/habitats.gpkg",
    }

    empty_group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)
    with (
        pytest.raises(ValueError, match="no reference asset"),
        runner.open_reference_group(empty_group, tmp_path / "empty", threshold=0, client=client),
    ):
        pass


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

    def fake_upload(api, repo_id, path, local_path, *, parent_commit=None):
        del api, repo_id, parent_commit
        uploaded.append((path, local_path))
        return SimpleNamespace(oid=f"commit-{len(uploaded)}")

    monkeypatch.setattr(runner, "download_to_temp", fake_download)
    monkeypatch.setattr(runner, "upload_replacement", fake_upload)
    plan = DatasetPlan(
        DatasetSpec(
            "wikidata", "source", "target", "polygons/*.parquet", "polygon_document_links/*.parquet"
        ),
        "source-revision",
        ("polygons/region.parquet", "polygon_document_links/region.parquet"),
        ("polygons/region.parquet",),
        ("polygon_document_links/region.parquet",),
    )
    sidecar = runner._sidecar_path(tmp_path / "sidecars", plan.spec, plan.geometry_paths[0])
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

    expectations, commit = runner.finalize_dataset(
        object(),
        plan,
        sidecar_root=tmp_path / "sidecars",
        local_root=tmp_path / "final",
        batch_size=1,
        parent_commit="base",
        progress=capture_progress,
        http_client=object(),
    )

    assert commit == "commit-2"
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
    assert not sidecar.exists()
    assert list((tmp_path / "final").iterdir()) == []


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
    monkeypatch.setattr(runner, "capture_revision", lambda api, repo: f"rev:{repo}")
    monkeypatch.setattr(runner, "list_repo_files", lambda api, repo, revision: entries[repo])

    plans = runner.plan_datasets(object())

    assert [plan.spec.name for plan in plans] == ["website", "wikidata", "description"]
    assert plans[1].link_paths == ("polygon_document_links/a.parquet",)
    with pytest.raises(ValueError, match="no geometry"):
        runner._geometry_paths(
            DatasetSpec("empty", "source", "target", "polygons/*.parquet"), ("README.md",)
        )

    raster = _asset("/Prob_R11.tif")
    vector = _asset("/habitats.gpkg", code=None)
    group = EeaGroup("record", "title", "folder", "service", {}, (raster,), vector)
    manifest = runner._reference_manifest(
        (group,),
        {"record:/Prob_R11.tif": "sha"},
        source_version="EEA-test",
        crs="EPSG:3035",
        threshold=0,
        config={"classification_record": "class"},
    )
    assets = manifest["assets"]
    assert isinstance(assets, list)
    assert [asset["path"] for asset in assets] == ["/Prob_R11.tif", "/habitats.gpkg"]

    shared = runner._shared_blobs(object(), plans[1], {"polygons/a.parquet"})
    assert shared == {
        "polygon_document_links/a.parquet": "link",
        "wikipedia/a.parquet": "wikipedia",
    }


def test_process_reference_groups_batches_reference_groups_for_all_plans(
    tmp_path: Path,
    monkeypatch,
) -> None:
    group = EeaGroup(
        "record", "title", "folder", "service", {}, (), _asset("/habitats.gpkg", code=None)
    )
    second_group = EeaGroup(
        "record-2", "title", "folder", "service", {}, (), _asset("/other.gpkg", code=None)
    )
    plans = (
        DatasetPlan(
            DatasetSpec("website", "source", "target", "polygons/*.parquet"), "rev", (), ("a",), ()
        ),
        DatasetPlan(
            DatasetSpec("description", "source", "target", "data/*.parquet"), "rev", (), ("b",), ()
        ),
    )
    seen: list[tuple[str, tuple[object, ...], object]] = []

    @contextmanager
    def fake_open(*args, **kwargs):
        del args, kwargs
        reference = object()
        yield reference

    def fake_process(api, plan, **kwargs):
        del api
        seen.append((plan.spec.name, kwargs["references"], kwargs["http_client"]))

    monkeypatch.setattr(runner, "open_reference_group", fake_open)
    monkeypatch.setattr(runner, "_process_geometry_path", fake_process)
    client = object()
    runner._process_reference_groups(
        object(),
        plans,
        (group, second_group),
        sidecar_root=tmp_path / "sidecars",
        source_root=tmp_path / "source",
        workdir=tmp_path,
        threshold=0,
        checksums={},
        batch_size=2,
        progress=None,
        http_client=client,
    )

    assert [name for name, _, _ in seen] == ["website", "description"]
    assert all(len(references) == 2 for _, references, _ in seen)
    assert seen[0][1] is seen[1][1]
    assert all(item[2] is client for item in seen)


def test_process_reference_groups_dispatches_parallel_batches(
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

    def fake_parallel(*args, **kwargs):
        del args
        seen.append(
            (tuple(group.record_id for group in kwargs["groups"]), kwargs["parallelism"])
        )

    monkeypatch.setattr(runner, "_process_reference_batch_parallel", fake_parallel)
    runner._process_reference_groups(
        object(),
        (plan,),
        (group,),
        sidecar_root=tmp_path / "sidecars",
        source_root=tmp_path / "source",
        workdir=tmp_path,
        threshold=0,
        checksums={},
        batch_size=2,
        progress=None,
        http_client=object(),
        parallelism=2,
    )

    assert seen == [(("record",), 2)]


def test_reference_group_batches_bound_rasters_and_coalesce_vectors() -> None:
    raster = _asset("/Prob_R11.tif")
    raster_groups = tuple(
        EeaGroup(str(index), "title", "folder", "service", {}, (raster,), None)
        for index in range(4)
    )
    vector_groups = tuple(
        EeaGroup(
            str(index),
            "title",
            "folder",
            "service",
            {},
            (),
            _asset(f"/{index}.gpkg", code=None),
        )
        for index in range(4, 7)
    )

    batches = runner._reference_group_batches(raster_groups + vector_groups)

    assert [[group.record_id for group in batch] for batch in batches] == [
        ["0", "1"],
        ["2", "3"],
        ["4", "5", "6"],
    ]

    reversed_batches = runner._reference_group_batches(vector_groups[:1] + raster_groups[:1])
    assert [[group.record_id for group in batch] for batch in reversed_batches] == [["4"], ["0"]]


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
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runner, "capture_revision", lambda api, repo: "target-base")
    monkeypatch.setattr(
        runner, "finalize_dataset", lambda *args, **kwargs: ((expectation,), "commit-1")
    )
    monkeypatch.setattr(
        runner, "upload_manifest", lambda *args, **kwargs: SimpleNamespace(oid="commit-2")
    )
    monkeypatch.setattr(runner, "verify_dataset", lambda *args, **kwargs: verification)

    class FakeCard:
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

    monkeypatch.setattr(runner, "DatasetCardAccumulator", FakeCard)
    monkeypatch.setattr(runner, "_upload_file", lambda *args, **kwargs: "commit-card")

    def fake_build_manifest(**kwargs):
        calls.append(kwargs)
        return {"manifest": True}

    monkeypatch.setattr(runner, "build_manifest", fake_build_manifest)
    monkeypatch.setattr(
        runner,
        "_shared_blobs",
        lambda *args, **kwargs: {"README.md": "blob"},
    )

    result = runner._finalize_plan(
        object(),
        plan,
        sidecar_root=tmp_path / "sidecars",
        workdir=tmp_path,
        batch_size=2,
        reference_info={"source_version": "EEA-test"},
        progress=None,
        http_client=object(),
    )

    assert result.verification is verification
    assert calls[0]["changed_paths"] == ("polygons/a.parquet", "README.md")
    assert calls[0]["added_paths"] == ("eunis/world-map.svg",)


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
    monkeypatch.setattr(runner, "plan_datasets", lambda api: (plan,))
    monkeypatch.setattr(runner, "_duplicate_outputs", lambda *args: None)
    monkeypatch.setattr(runner, "_load_existing_manifest", lambda *args: None)
    monkeypatch.setattr(runner, "resolve_config", lambda path: (group,))
    monkeypatch.setattr(
        runner,
        "_process_reference_groups",
        lambda *args, **kwargs: seen.append(
            (kwargs["http_client"], kwargs["parallelism"])
        ),
    )
    monkeypatch.setattr(runner, "_finalize_plan", lambda *args, **kwargs: receipt)

    result = runner.run_release(
        object(), reference_config=config, workdir=tmp_path / "run", batch_size=2
    )

    assert result.datasets == (receipt,)
    assert len(seen) == 1
    assert seen[0][1] == runner._SOURCE_WORKERS


def test_run_release_verifies_matching_manifests_without_processing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 0}),
        encoding="utf-8",
    )
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )
    reference = {"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 0}
    manifest = {
        "manifest_version": 3,
        "source_repo": "source",
        "target_repo": "target",
        "source_revision": "source-revision",
        "source_paths": ["README.md", "polygons/a.parquet"],
        "changed_paths": ["README.md", "polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
        "shared_paths": [],
        "rows_by_path": {"polygons/a.parquet": 2},
        "schema_by_path": {"polygons/a.parquet": "schema"},
        "reference": reference,
        "card": {
            "readme_path": "README.md",
            "map_path": "eunis/world-map.svg",
            "readme_sha256": "readme",
            "map_sha256": "map",
        },
    }
    receipt = DatasetReceipt(
        plan,
        (ShardExpectation("polygons/a.parquet", 2, "schema"),),
        VerificationReceipt("target", "verified", {}, (), manifest),
        no_op=True,
    )
    monkeypatch.setattr(runner, "plan_datasets", lambda api: (plan,))
    monkeypatch.setattr(runner, "_duplicate_outputs", lambda *args: None)
    monkeypatch.setattr(runner, "resolve_config", lambda path: ())
    monkeypatch.setattr(runner, "_reference_manifest", lambda *args, **kwargs: reference)
    monkeypatch.setattr(
        runner,
        "_load_existing_manifest",
        lambda *args, **kwargs: runner._ExistingManifest("target", manifest),
    )
    monkeypatch.setattr(runner, "_verify_no_op_dataset", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(
        runner,
        "_process_reference_groups",
        lambda *args, **kwargs: pytest.fail("matching release must not process source shards"),
    )
    monkeypatch.setattr(
        runner,
        "_finalize_plan",
        lambda *args, **kwargs: pytest.fail("matching release must not upload source shards"),
    )

    result = runner.run_release(
        object(), reference_config=config, workdir=tmp_path / "run", batch_size=2
    )

    assert result.datasets == (receipt,)
    assert result.datasets[0].no_op is True


def test_no_op_manifest_helpers_validate_and_load_pinned_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )
    reference = {"assets": [{"code": "R11", "sha256": "downloaded"}]}
    manifest = {
        "manifest_version": 3,
        "source_repo": "source",
        "target_repo": "target",
        "source_revision": "source-revision",
        "source_paths": ["README.md", "polygons/a.parquet"],
        "changed_paths": ["README.md", "polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
        "rows_by_path": {"polygons/a.parquet": 2},
        "schema_by_path": {"polygons/a.parquet": "schema"},
        "reference": reference,
        "card": {
            "readme_path": "README.md",
            "map_path": "eunis/world-map.svg",
            "readme_sha256": "readme",
            "map_sha256": "map",
        },
    }

    assert runner._reference_identity(reference) == {"assets": [{"code": "R11"}]}
    assert runner._manifest_matches_inputs(plan, manifest, {"assets": [{"code": "R11"}]})
    assert not runner._manifest_matches_inputs(
        plan,
        {**manifest, "source_revision": "different"},
        {"assets": [{"code": "R11"}]},
    )
    assert runner._manifest_expectations(manifest) == (
        ShardExpectation("polygons/a.parquet", 2, "schema"),
    )
    assert runner._manifest_artifacts(manifest) == {
        "README.md": "readme",
        "eunis/world-map.svg": "map",
    }
    assert runner._required_no_op_parts(manifest)[2:] == (
        ("README.md", "polygons/a.parquet"),
        ("eunis/world-map.svg",),
    )
    with pytest.raises(ValueError, match="incomplete"):
        runner._required_manifest_paths({})
    with pytest.raises(ValueError, match="incomplete"):
        runner._required_manifest_paths({"changed_paths": ["ok", 1], "added_paths": []})
    assert runner._manifest_expectations({"rows_by_path": {}, "schema_by_path": {}}) == ()
    assert runner._manifest_artifacts({"card": {}}) is None

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="target-revision")

        def list_repo_tree(self, *_args, **_kwargs):
            return iter((SimpleNamespace(path="eunis/manifest.json"),))

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, path, revision, client
        destination = directory / "manifest.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(manifest), encoding="utf-8")
        return destination

    monkeypatch.setattr(runner, "download_to_temp", fake_download)
    loaded = runner._load_existing_manifest(Api(), plan, tmp_path / "noop", object())
    assert loaded == runner._ExistingManifest("target-revision", manifest)


def test_verify_no_op_dataset_reuses_manifest_expectations(monkeypatch, tmp_path: Path) -> None:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )
    manifest = {
        "rows_by_path": {"polygons/a.parquet": 1},
        "schema_by_path": {"polygons/a.parquet": "schema"},
        "changed_paths": ["polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
        "card": {
            "readme_path": "README.md",
            "map_path": "eunis/world-map.svg",
            "readme_sha256": "readme",
            "map_sha256": "map",
        },
    }
    verification = VerificationReceipt("target", "verified", {}, (), manifest)
    monkeypatch.setattr(runner, "_shared_blobs", lambda *args, **kwargs: {})
    monkeypatch.setattr(runner, "_verify_final_dataset", lambda *args, **kwargs: verification)

    result = runner._verify_no_op_dataset(
        object(),
        plan,
        runner._ExistingManifest("target", manifest),
        workdir=tmp_path,
        client=object(),
    )

    assert result.no_op is True
    assert result.expectations == (ShardExpectation("polygons/a.parquet", 1, "schema"),)
