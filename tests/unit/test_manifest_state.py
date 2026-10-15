"""Release manifest compatibility and verified no-op behavior."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_eunis import manifest_state, release_orchestration
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.domain import EunisTaxonomy
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.geometry import GEOMETRY_POLICY
from osm_polygon_eunis.options import BatchLimits, ReleaseOptions
from osm_polygon_eunis.publish import (
    DatasetVerificationOptions,
    ShardExpectation,
    VerificationError,
    VerificationReceipt,
)
from osm_polygon_eunis.reference_staging import asset_key
from osm_polygon_eunis.release_plan import DatasetPlan, DatasetReceipt, ReleaseReceipt
from osm_polygon_eunis.sources import DatasetSpec


def _plan() -> DatasetPlan:
    return DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )


def _matching_manifest(plan: DatasetPlan) -> dict[str, object]:
    return {
        "manifest_version": manifest_state.MANIFEST_VERSION,
        "software": manifest_state.software_provenance(),
        "source_repo": plan.spec.source_repo,
        "target_repo": plan.spec.output_repo,
        "source_revision": plan.source_revision,
        "source_paths": list(plan.source_files),
        "reference": {"source_version": "EEA-test"},
    }


def test_reference_manifest_ignores_groups_without_assets() -> None:
    group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)

    manifest = manifest_state.reference_manifest(
        (group,), {}, source_version="EEA-test", crs="EPSG:3035", threshold=0, config={}
    )

    assert manifest["assets"] == []


def test_reference_manifest_records_classification_workbook_provenance() -> None:
    taxonomy = EunisTaxonomy(
        {"R11": "steppe"},
        {"R11": 1},
        {},
        source_record_id="classification-record",
        source_path="eunis-2021.xlsx",
        source_url="https://example.test/eunis-2021.xlsx",
        source_size=123,
        source_etag="etag-v1",
        source_sha256="workbook-sha256",
    )
    group = EeaGroup("record", "title", "folder", "service", {}, (), None, taxonomy=taxonomy)

    manifest = manifest_state.reference_manifest(
        (group,), {}, source_version="EEA-test", crs="EPSG:3035", threshold=80, config={}
    )

    assert manifest["taxonomy"] == {
        "record_id": "classification-record",
        "path": "eunis-2021.xlsx",
        "url": "https://example.test/eunis-2021.xlsx",
        "size": 123,
        "etag": "etag-v1",
        "sha256": "workbook-sha256",
        "fingerprint": taxonomy.fingerprint,
    }


def test_reference_manifest_rejects_mixed_classification_taxonomies() -> None:
    first = EunisTaxonomy({"R11": "steppe"}, {"R11": 1}, {})
    second = EunisTaxonomy({"R11": "dry grassland"}, {"R11": 1}, {})
    groups = (
        EeaGroup("first", "first", "folder", "service", {}, (), None, taxonomy=first),
        EeaGroup("second", "second", "folder", "service", {}, (), None, taxonomy=second),
    )

    with pytest.raises(ValueError, match="one classification taxonomy"):
        manifest_state.reference_manifest(
            groups, {}, source_version="EEA-test", crs="EPSG:3035", threshold=80, config={}
        )


@pytest.mark.parametrize(
    "manifest",
    [
        {},
        {"rows_by_path": [], "schema_by_path": {}},
        {"rows_by_path": {"polygons/a.parquet": 1}, "schema_by_path": {}},
        {
            "rows_by_path": {"polygons/a.parquet": "not-an-integer"},
            "schema_by_path": {"polygons/a.parquet": "schema"},
        },
        {"rows_by_path": {1: 1}, "schema_by_path": {1: "schema"}},
    ],
)
def test_manifest_expectations_rejects_incomplete_or_malformed_tables(
    manifest: dict[str, object],
) -> None:
    assert manifest_state._manifest_expectations(manifest) is None


@pytest.mark.parametrize(
    ("rows", "schema"),
    [
        (True, "schema"),
        (False, "schema"),
        (3.9, "schema"),
        (1.0, "schema"),
        ("5", "schema"),
        (None, "schema"),
        (-1, "schema"),
        ([], "schema"),
        ({}, "schema"),
        (1, None),
        (1, True),
        (1, 5),
        (1, 3.9),
        (1, []),
        (1, {}),
    ],
)
def test_manifest_expectations_rejects_invalid_field_types(rows: object, schema: object) -> None:
    manifest: dict[str, object] = {
        "rows_by_path": {"polygons/a.parquet": 0, "polygons/b.parquet": rows},
        "schema_by_path": {"polygons/a.parquet": "schema", "polygons/b.parquet": schema},
        "changed_paths": ["polygons/a.parquet", "polygons/b.parquet"],
        "added_paths": [],
        "card": {
            "readme_path": "README.md",
            "readme_sha256": "readme",
            "map_path": "eunis/world-map.svg",
            "map_sha256": "map",
        },
    }

    assert manifest_state._manifest_expectations(manifest) is None
    with pytest.raises(VerificationError, match="incomplete for no-op verification"):
        manifest_state._required_no_op_parts(manifest)


def test_manifest_expectations_preserves_valid_json_fields_and_path_order() -> None:
    manifest = json.loads(
        '{"rows_by_path": {"z.parquet": 9007199254740993, "a.parquet": 0}, '
        '"schema_by_path": {"z.parquet": "None", "a.parquet": ""}}'
    )

    assert manifest_state._manifest_expectations(manifest) == (
        ShardExpectation("a.parquet", 0, ""),
        ShardExpectation("z.parquet", 9007199254740993, "None"),
    )


def test_existing_manifest_ignores_non_object_json(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()

    class Api:
        def repo_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha="target-revision")

        def list_repo_tree(self, *_args: object, **_kwargs: object) -> tuple[SimpleNamespace, ...]:
            return (SimpleNamespace(path="eunis/manifest.json"),)

    def fake_download(_api, _repo_id, _path, _revision, directory, *, client=None) -> Path:
        del client
        result = directory / "manifest.json"
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text("[]", encoding="utf-8")
        return result

    monkeypatch.setattr(manifest_state, "download_to_temp", fake_download)

    assert (
        manifest_state._load_existing_manifest(
            cast(HubApi, Api()), plan, tmp_path, cast(StreamClient, object())
        )
        is None
    )


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
        "manifest_version": manifest_state.MANIFEST_VERSION,
        "software": manifest_state.software_provenance(),
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
        "geometry_policy": GEOMETRY_POLICY,
        "card": {
            "readme_path": "README.md",
            "map_path": "eunis/world-map.svg",
            "readme_sha256": "readme",
            "map_sha256": "map",
            "geometry_policy": GEOMETRY_POLICY,
        },
    }
    receipt = DatasetReceipt(
        plan,
        (ShardExpectation("polygons/a.parquet", 2, "schema"),),
        VerificationReceipt("target", "verified", {}, (), manifest),
        no_op=True,
    )
    monkeypatch.setattr(release_orchestration, "plan_datasets", lambda api, names=None: (plan,))
    monkeypatch.setattr(
        release_orchestration,
        "_duplicate_outputs",
        lambda *args: pytest.fail("a verified no-op must not duplicate target repos"),
    )
    monkeypatch.setattr(release_orchestration, "resolve_config_data", lambda config: ())
    monkeypatch.setattr(
        release_orchestration,
        "reference_manifest",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        manifest_state,
        "_load_existing_manifest",
        lambda *args, **kwargs: manifest_state.ExistingManifest("target", manifest),
    )
    monkeypatch.setattr(manifest_state, "verify_no_op_dataset", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(
        release_orchestration,
        "process_reference_groups",
        lambda *args, **kwargs: pytest.fail("matching release must not process source shards"),
    )
    monkeypatch.setattr(
        release_orchestration,
        "finalize_plan",
        lambda *args, **kwargs: pytest.fail("matching release must not upload source shards"),
    )

    result = release_orchestration.run_release(
        cast(HubApi, object()),
        ReleaseOptions(
            reference_config=config,
            workdir=tmp_path / "run",
            limits=BatchLimits(parquet_batch_size=2),
        ),
    )

    assert result.datasets == (receipt,)
    assert result.datasets[0].no_op is True


def test_no_op_manifest_helpers_validate_and_load_pinned_state(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", "a" * 40)
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )
    reference = {"assets": [{"code": "R11", "sha256": "downloaded"}]}
    manifest = {
        "manifest_version": manifest_state.MANIFEST_VERSION,
        "software": manifest_state.software_provenance(),
        "source_repo": "source",
        "target_repo": "target",
        "source_revision": "source-revision",
        "source_paths": ["README.md", "polygons/a.parquet"],
        "changed_paths": ["README.md", "polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
        "rows_by_path": {"polygons/a.parquet": 2},
        "schema_by_path": {"polygons/a.parquet": "schema"},
        "reference": reference,
        "geometry_policy": GEOMETRY_POLICY,
        "card": {
            "readme_path": "README.md",
            "map_path": "eunis/world-map.svg",
            "readme_sha256": "readme",
            "map_sha256": "map",
            "geometry_policy": GEOMETRY_POLICY,
        },
    }

    assert manifest_state._reference_identity(reference) == {"assets": [{"code": "R11"}]}
    assert manifest_state._manifest_matches_inputs(plan, manifest, {"assets": [{"code": "R11"}]})
    assert not manifest_state._manifest_matches_inputs(
        plan,
        {**manifest, "manifest_version": 4},
        {"assets": [{"code": "R11"}]},
    )
    assert not manifest_state._manifest_matches_inputs(
        plan,
        {**manifest, "source_revision": "different"},
        {"assets": [{"code": "R11"}]},
    )
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", "b" * 40)
    assert not manifest_state._manifest_matches_inputs(
        plan,
        manifest,
        {"assets": [{"code": "R11"}]},
    )
    assert manifest_state._manifest_expectations(manifest) == (
        ShardExpectation("polygons/a.parquet", 2, "schema"),
    )
    assert manifest_state._manifest_artifacts(manifest) == {
        "README.md": "readme",
        "eunis/world-map.svg": "map",
    }
    assert manifest_state._required_no_op_parts(manifest)[2:] == (
        ("README.md", "polygons/a.parquet"),
        ("eunis/world-map.svg",),
    )
    with pytest.raises(ValueError, match="incomplete"):
        manifest_state._required_manifest_paths({})
    with pytest.raises(ValueError, match="incomplete"):
        manifest_state._required_manifest_paths({"changed_paths": ["ok", 1], "added_paths": []})
    assert manifest_state._manifest_expectations({"rows_by_path": {}, "schema_by_path": {}}) == ()
    assert manifest_state._manifest_artifacts({"card": {}}) is None

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

    monkeypatch.setattr(manifest_state, "download_to_temp", fake_download)
    loaded = manifest_state._load_existing_manifest(
        cast(HubApi, Api()), plan, tmp_path / "noop", cast(StreamClient, object())
    )
    assert loaded == manifest_state.ExistingManifest("target-revision", manifest)


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
    monkeypatch.setattr(manifest_state, "collect_shared_blobs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        manifest_state, "verify_final_dataset", lambda *args, **kwargs: verification
    )

    result = manifest_state.verify_no_op_dataset(
        cast(HubApi, object()),
        plan,
        manifest_state.ExistingManifest("target", manifest),
        workdir=tmp_path,
        client=cast(StreamClient, object()),
    )

    assert result.no_op is True
    assert result.expectations == (ShardExpectation("polygons/a.parquet", 1, "schema"),)


@pytest.mark.parametrize("existing", [None, "mismatch"], ids=["missing-manifest", "changed-inputs"])
def test_no_op_release_falls_back_when_manifest_is_missing_or_stale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    existing: str | None,
) -> None:
    plan = _plan()
    manifest = _matching_manifest(plan)
    if existing == "mismatch":
        manifest["source_revision"] = "older-revision"
    if existing is None:
        item = None
    else:
        item = manifest_state.ExistingManifest("target-revision", manifest)
    monkeypatch.setattr(manifest_state, "load_existing_manifests", lambda *args: (item,))
    monkeypatch.setattr(
        manifest_state,
        "verify_no_op_dataset",
        lambda *args, **kwargs: pytest.fail("incompatible manifests must not be verified as no-op"),
    )

    receipt = manifest_state.try_no_op_release(
        cast(HubApi, object()),
        (plan,),
        {"source_version": "EEA-test"},
        workdir=tmp_path,
        client=cast(StreamClient, object()),
        progress=None,
    )

    assert receipt is None


def test_no_op_release_fails_closed_when_manifest_artifacts_are_incomplete(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    manifest = {
        **_matching_manifest(plan),
        "rows_by_path": {"polygons/a.parquet": 1},
        "schema_by_path": {"polygons/a.parquet": "schema"},
        "changed_paths": ["polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
    }
    existing = manifest_state.ExistingManifest("target-revision", manifest)
    monkeypatch.setattr(manifest_state, "load_existing_manifests", lambda *args: (existing,))

    with pytest.raises(VerificationError, match="incomplete"):
        manifest_state.try_no_op_release(
            cast(HubApi, object()),
            (plan,),
            {"source_version": "EEA-test"},
            workdir=tmp_path,
            client=cast(StreamClient, object()),
            progress=None,
        )


def test_no_op_release_reports_each_verified_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    manifest = _matching_manifest(plan)
    existing = manifest_state.ExistingManifest("target-revision", manifest)
    receipt = DatasetReceipt(
        plan,
        (),
        VerificationReceipt("target", "target-revision", {}, (), manifest),
        no_op=True,
    )
    monkeypatch.setattr(manifest_state, "load_existing_manifests", lambda *args: (existing,))
    monkeypatch.setattr(manifest_state, "_no_op_receipts", lambda *args, **kwargs: (receipt,))
    events: list[Mapping[str, object]] = []

    result = manifest_state.try_no_op_release(
        cast(HubApi, object()),
        (plan,),
        {"source_version": "EEA-test"},
        workdir=tmp_path,
        client=cast(StreamClient, object()),
        progress=events.append,
    )

    assert result == ReleaseReceipt((receipt,), {"source_version": "EEA-test"})
    assert events == [{"event": "dataset_no_op", "dataset": "website"}]


def test_reference_manifest_lists_every_asset_with_checksums_in_record_then_path_order() -> None:
    raster_a = RemoteAsset(
        "z-raster/a.tif",
        "https://example.test/a.tif",
        11,
        "etag-a",
        "R1",
        "A raster",
        "record-a",
        "EEA-test",
    )
    raster_b = RemoteAsset(
        "raster/b.tif",
        "https://example.test/b.tif",
        22,
        None,
        "R2",
        "B raster",
        "record-b",
        "EEA-test",
    )
    vector_b = RemoteAsset(
        "aa-vector/b.gpkg",
        "https://example.test/b.gpkg",
        33,
        "etag-v",
        None,
        "B vector",
        "record-b",
        "EEA-test",
    )
    group_a = EeaGroup("record-a", "A", "folder-a", "service-a", {}, (raster_a,), None)
    group_b = EeaGroup("record-b", "B", "folder-b", "service-b", {}, (raster_b,), vector_b)
    checksums = {
        asset_key(group_a, raster_a): "sha-a",
        asset_key(group_b, raster_b): "sha-b",
        asset_key(group_b, vector_b): "sha-vector",
    }

    manifest = manifest_state.reference_manifest(
        (group_b, group_a),
        checksums,
        source_version="EEA-test",
        crs="EPSG:3035",
        threshold=7,
        config={"classification_record": "clc-2018"},
    )

    assert manifest == {
        "source_version": "EEA-test",
        "crs": "EPSG:3035",
        "threshold": 7,
        "classification_record": "clc-2018",
        "assets": [
            {
                "record_id": "record-a",
                "path": "z-raster/a.tif",
                "url": "https://example.test/a.tif",
                "size": 11,
                "etag": "etag-a",
                "code": "R1",
                "name": "A raster",
                "sha256": "sha-a",
            },
            {
                "record_id": "record-b",
                "path": "aa-vector/b.gpkg",
                "url": "https://example.test/b.gpkg",
                "size": 33,
                "etag": "etag-v",
                "code": None,
                "name": "B vector",
                "sha256": "sha-vector",
            },
            {
                "record_id": "record-b",
                "path": "raster/b.tif",
                "url": "https://example.test/b.tif",
                "size": 22,
                "etag": None,
                "code": "R2",
                "name": "B raster",
                "sha256": "sha-b",
            },
        ],
    }


def test_mapping_field_returns_only_mapping_values() -> None:
    assert manifest_state._mapping_field({"reference": {"source_version": "EEA"}}, "reference") == {
        "source_version": "EEA"
    }
    assert manifest_state._mapping_field({"reference": ["not", "a", "mapping"]}, "reference") == {}
    assert manifest_state._mapping_field({}, "reference") == {}


def test_existing_manifest_reads_the_pinned_target_and_removes_the_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    capture_calls: list[tuple[object, str]] = []
    list_calls: list[tuple[object, str, str]] = []
    download_calls: list[tuple[object, str, str, str, Path, object]] = []
    downloaded: list[Path] = []

    def fake_capture(api_arg: object, repo: str) -> str:
        capture_calls.append((api_arg, repo))
        return "target-revision"

    def fake_list(api_arg: object, repo: str, revision: str) -> tuple[SimpleNamespace, ...]:
        list_calls.append((api_arg, repo, revision))
        return (SimpleNamespace(path="README.md"), SimpleNamespace(path="eunis/manifest.json"))

    def fake_download(
        api_arg: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        download_calls.append((api_arg, repo_id, path, revision, directory, client))
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "manifest.json"
        destination.write_text(json.dumps({"source_revision": "source-revision"}), encoding="utf-8")
        downloaded.append(destination)
        return destination

    monkeypatch.setattr(manifest_state, "capture_revision", fake_capture)
    monkeypatch.setattr(manifest_state, "list_repo_files", fake_list)
    monkeypatch.setattr(manifest_state, "download_to_temp", fake_download)

    existing = manifest_state._load_existing_manifest(api, plan, tmp_path, client)

    assert capture_calls == [(api, "target")]
    assert list_calls == [(api, "target", "target-revision")]
    assert download_calls == [
        (api, "target", "eunis/manifest.json", "target-revision", tmp_path, client)
    ]
    assert existing == manifest_state.ExistingManifest(
        "target-revision", {"source_revision": "source-revision"}
    )
    assert not downloaded[0].exists()


def test_existing_manifest_is_absent_when_the_target_has_no_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(manifest_state, "capture_revision", lambda api, repo: "target-revision")
    monkeypatch.setattr(
        manifest_state,
        "list_repo_files",
        lambda api, repo, revision: (SimpleNamespace(path="README.md"),),
    )
    monkeypatch.setattr(
        manifest_state,
        "download_to_temp",
        lambda *args, **kwargs: pytest.fail("a target without a manifest must not download"),
    )

    assert (
        manifest_state._load_existing_manifest(
            cast(HubApi, object()), _plan(), tmp_path, cast(StreamClient, object())
        )
        is None
    )


@pytest.mark.parametrize(
    ("path", "rows", "schema", "message"),
    [
        (1, 1, "schema", "manifest shard path must be a string"),
        (
            "polygons/a.parquet",
            True,
            "schema",
            "manifest shard rows must be a non-negative integer",
        ),
        ("polygons/a.parquet", 1, None, "manifest shard schema must be a string"),
    ],
)
def test_manifest_expectation_names_the_invalid_field(
    path: object,
    rows: object,
    schema: object,
    message: str,
) -> None:
    with pytest.raises(TypeError) as excinfo:
        manifest_state._manifest_expectation(path, rows, schema)

    assert str(excinfo.value) == message


def test_manifest_artifacts_require_both_a_path_and_a_digest_for_each_card_file() -> None:
    complete = {
        "readme_path": "README.md",
        "readme_sha256": "readme",
        "map_path": "eunis/world-map.svg",
        "map_sha256": "map",
    }

    assert manifest_state._manifest_artifacts({"card": complete}) == {
        "README.md": "readme",
        "eunis/world-map.svg": "map",
    }
    assert manifest_state._manifest_artifacts({"card": {**complete, "readme_sha256": None}}) is None
    assert manifest_state._manifest_artifacts({"card": {**complete, "map_path": None}}) is None


def test_no_op_manifest_errors_use_the_exact_incomplete_manifest_message() -> None:
    with pytest.raises(VerificationError) as excinfo:
        manifest_state._required_string_list(
            {"changed_paths": "polygons/a.parquet"}, "changed_paths"
        )
    assert str(excinfo.value) == "matching EUNIS manifest is incomplete for no-op verification"

    with pytest.raises(VerificationError) as mixed:
        manifest_state._required_string_list({"changed_paths": ["ok", 1]}, "changed_paths")
    assert str(mixed.value) == "matching EUNIS manifest is incomplete for no-op verification"

    with pytest.raises(VerificationError) as missing:
        manifest_state._required_no_op_parts({})
    assert str(missing.value) == "matching EUNIS manifest is incomplete for no-op verification"


def test_verify_no_op_dataset_passes_exact_expectations_to_final_verification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
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
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    verification = VerificationReceipt(
        "target", "verified", {"polygons/a.parquet": 1}, (), manifest
    )
    shared_calls: list[tuple[object, DatasetPlan, set[str]]] = []
    final_calls: list[tuple[object, DatasetPlan, object]] = []

    def fake_shared(api_arg: object, plan_arg: DatasetPlan, changed: set[str]) -> dict[str, str]:
        shared_calls.append((api_arg, plan_arg, changed))
        return {"README.md": "readme-blob"}

    def fake_final(api_arg: object, plan_arg: DatasetPlan, options: object) -> VerificationReceipt:
        final_calls.append((api_arg, plan_arg, options))
        return verification

    monkeypatch.setattr(manifest_state, "collect_shared_blobs", fake_shared)
    monkeypatch.setattr(manifest_state, "verify_final_dataset", fake_final)

    result = manifest_state.verify_no_op_dataset(
        api,
        plan,
        manifest_state.ExistingManifest("target", manifest),
        workdir=tmp_path,
        client=client,
    )

    assert shared_calls == [(api, plan, {"polygons/a.parquet"})]
    assert final_calls == [
        (
            api,
            plan,
            manifest_state.FinalDatasetVerificationOptions(
                expectations=(ShardExpectation("polygons/a.parquet", 1, "schema"),),
                added_paths=("eunis/world-map.svg",),
                expected_shared_blobs={"README.md": "readme-blob"},
                expected_manifest=manifest,
                expected_artifacts={"README.md": "readme", "eunis/world-map.svg": "map"},
                temp_dir=tmp_path / "verify-no-op",
                http_client=client,
            ),
        )
    ]
    assert result == DatasetReceipt(
        plan,
        (ShardExpectation("polygons/a.parquet", 1, "schema"),),
        verification,
        no_op=True,
    )


def test_load_existing_manifests_reads_every_plan_with_the_same_client_and_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _plan()
    second = DatasetPlan(
        DatasetSpec("other", "source-2", "target-2", "polygons/*.parquet"),
        "other-revision",
        ("polygons/b.parquet",),
        ("polygons/b.parquet",),
        (),
    )
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    calls: list[tuple[object, DatasetPlan, Path, object]] = []

    def fake_load(
        api_arg: object,
        plan: DatasetPlan,
        directory: Path,
        client_arg: object,
    ) -> str:
        calls.append((api_arg, plan, directory, client_arg))
        return f"manifest-{plan.spec.name}"

    monkeypatch.setattr(manifest_state, "_load_existing_manifest", fake_load)

    result = manifest_state.load_existing_manifests(api, (first, second), tmp_path, client)

    assert result == ("manifest-website", "manifest-other")
    assert calls == [(api, first, tmp_path, client), (api, second, tmp_path, client)]


def test_compatible_manifests_rejects_fewer_manifests_than_plans() -> None:
    plan = _plan()
    existing = (manifest_state.ExistingManifest("target-revision", _matching_manifest(plan)),)

    assert (
        manifest_state.compatible_manifests((plan,), existing, {"source_version": "EEA-test"})
        is True
    )
    with pytest.raises(ValueError):
        manifest_state.compatible_manifests((plan, plan), existing, {"source_version": "EEA-test"})


def test_no_op_receipts_verify_each_plan_with_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    item = manifest_state.ExistingManifest("target-revision", _matching_manifest(plan))
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    calls: list[tuple[object, DatasetPlan, object, Path, object]] = []
    receipt = DatasetReceipt(
        plan,
        (),
        VerificationReceipt("target", "target-revision", {}, (), {}),
        no_op=True,
    )

    def fake_verify(
        api_arg: object,
        plan_arg: DatasetPlan,
        existing: object,
        *,
        workdir: Path,
        client: object,
    ) -> DatasetReceipt:
        calls.append((api_arg, plan_arg, existing, workdir, client))
        return receipt

    monkeypatch.setattr(manifest_state, "verify_no_op_dataset", fake_verify)

    result = manifest_state._no_op_receipts(
        api,
        (plan,),
        (item,),
        workdir=tmp_path,
        client=client,
    )

    assert result == (receipt,)
    assert calls == [(api, plan, item, tmp_path, client)]


def test_no_op_receipts_reject_mismatched_plan_and_manifest_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    item = manifest_state.ExistingManifest("target-revision", _matching_manifest(plan))
    monkeypatch.setattr(
        manifest_state,
        "verify_no_op_dataset",
        lambda *args, **kwargs: DatasetReceipt(
            plan, (), VerificationReceipt("target", "rev", {}, (), {}), no_op=True
        ),
    )

    with pytest.raises(ValueError):
        manifest_state._no_op_receipts(
            cast(HubApi, object()),
            (plan, plan),
            (item,),
            workdir=tmp_path,
            client=cast(StreamClient, object()),
        )


def test_try_no_op_release_threads_pinned_arguments_through_every_step(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    manifest = _matching_manifest(plan)
    existing = (manifest_state.ExistingManifest("target-revision", manifest),)
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    load_calls: list[tuple[object, tuple[DatasetPlan, ...], Path, object]] = []
    receipt_calls: list[tuple[object, tuple[DatasetPlan, ...], object, Path, object]] = []
    receipts = (
        DatasetReceipt(
            plan,
            (),
            VerificationReceipt("target", "target-revision", {}, (), manifest),
            no_op=True,
        ),
    )

    def fake_load(
        api_arg: object,
        plans: tuple[DatasetPlan, ...],
        directory: Path,
        client_arg: object,
    ) -> tuple[manifest_state.ExistingManifest, ...]:
        load_calls.append((api_arg, plans, directory, client_arg))
        return existing

    def fake_receipts(
        api_arg: object,
        plans: tuple[DatasetPlan, ...],
        items: object,
        *,
        workdir: Path,
        client: object,
    ) -> tuple[DatasetReceipt, ...]:
        receipt_calls.append((api_arg, plans, items, workdir, client))
        return receipts

    monkeypatch.setattr(manifest_state, "load_existing_manifests", fake_load)
    monkeypatch.setattr(manifest_state, "_no_op_receipts", fake_receipts)
    events: list[Mapping[str, object]] = []

    result = manifest_state.try_no_op_release(
        api,
        (plan,),
        {"source_version": "EEA-test"},
        workdir=tmp_path,
        client=client,
        progress=events.append,
    )

    assert load_calls == [(api, (plan,), tmp_path / "noop", client)]
    assert receipt_calls == [(api, (plan,), existing, tmp_path, client)]
    assert events == [{"event": "dataset_no_op", "dataset": "website"}]
    assert result == ReleaseReceipt(receipts, {"source_version": "EEA-test"})


def test_verify_final_dataset_requests_the_pinned_tree_and_expectations(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    plan = _plan()
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    calls: list[tuple[object, str, DatasetVerificationOptions]] = []

    def fake_verify(
        api_arg: object,
        repo: str,
        options: DatasetVerificationOptions,
    ) -> str:
        calls.append((api_arg, repo, options))
        return "verified"

    monkeypatch.setattr(manifest_state, "verify_dataset", fake_verify)
    options = manifest_state.FinalDatasetVerificationOptions(
        expectations=(ShardExpectation("polygons/a.parquet", 1, "schema"),),
        added_paths=("eunis/world-map.svg",),
        expected_shared_blobs={"README.md": "readme"},
        expected_manifest={"source_revision": "source-revision"},
        expected_artifacts={"eunis/world-map.svg": "map"},
        temp_dir=tmp_path,
        http_client=client,
    )

    result = manifest_state.verify_final_dataset(api, plan, options)

    assert result == "verified"
    assert calls == [
        (
            api,
            "target",
            DatasetVerificationOptions(
                expectations=(ShardExpectation("polygons/a.parquet", 1, "schema"),),
                expected_tree_paths=(
                    "README.md",
                    "polygons/a.parquet",
                    "eunis/world-map.svg",
                    "eunis/manifest.json",
                ),
                expected_shared_blobs={"README.md": "readme"},
                expected_manifest={"source_revision": "source-revision"},
                expected_artifacts={"eunis/world-map.svg": "map"},
                temp_dir=tmp_path,
                http_client=client,
            ),
        )
    ]


def test_collect_shared_blobs_maps_only_unchanged_files_that_have_a_blob_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _plan()
    api = cast(HubApi, object())
    calls: list[tuple[object, str, str]] = []
    entries = (
        SimpleNamespace(path="README.md", blob_id="readme-blob"),
        SimpleNamespace(path="polygons/a.parquet", blob_id="changed-blob"),
        SimpleNamespace(path="polygons/b.parquet", blob_id="shared-blob"),
        SimpleNamespace(path="polygons/c.parquet", blob_id=None),
        SimpleNamespace(path="polygons/d.parquet"),
    )

    def fake_list(api_arg: object, repo: str, revision: str) -> tuple[SimpleNamespace, ...]:
        calls.append((api_arg, repo, revision))
        return entries

    monkeypatch.setattr(manifest_state, "list_repo_files", fake_list)

    result = manifest_state.collect_shared_blobs(api, plan, {"polygons/a.parquet"})

    assert calls == [(api, "source", "source-revision")]
    assert result == {"README.md": "readme-blob", "polygons/b.parquet": "shared-blob"}


def test_existing_manifest_is_read_as_utf8_and_cleanup_tolerates_a_removed_download(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    encodings: list[object] = []
    original_read_text = Path.read_text

    def read_then_remove(self: Path, *, encoding: str | None = None) -> str:
        encodings.append(encoding)
        text = original_read_text(self, encoding=encoding)
        self.unlink()
        return text

    def fake_download(
        api: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        del api, repo_id, path, revision, client
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "manifest.json"
        destination.write_text(json.dumps({"source_revision": "source-revision"}), encoding="utf-8")
        return destination

    monkeypatch.setattr(Path, "read_text", read_then_remove)
    monkeypatch.setattr(manifest_state, "capture_revision", lambda api, repo: "target-revision")
    monkeypatch.setattr(
        manifest_state,
        "list_repo_files",
        lambda api, repo, revision: (SimpleNamespace(path="eunis/manifest.json"),),
    )
    monkeypatch.setattr(manifest_state, "download_to_temp", fake_download)

    existing = manifest_state._load_existing_manifest(
        cast(HubApi, object()), _plan(), tmp_path, cast(StreamClient, object())
    )

    assert encodings == ["utf-8"]
    assert existing == manifest_state.ExistingManifest(
        "target-revision", {"source_revision": "source-revision"}
    )
