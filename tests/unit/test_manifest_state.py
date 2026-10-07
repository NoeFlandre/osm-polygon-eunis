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
from osm_polygon_eunis.eea import EeaGroup
from osm_polygon_eunis.geometry import GEOMETRY_POLICY
from osm_polygon_eunis.options import BatchLimits, ReleaseOptions
from osm_polygon_eunis.publish import ShardExpectation, VerificationError, VerificationReceipt
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
        "software": manifest_state._software_provenance(),
        "source_repo": plan.spec.source_repo,
        "target_repo": plan.spec.output_repo,
        "source_revision": plan.source_revision,
        "source_paths": list(plan.source_files),
        "reference": {"source_version": "EEA-test"},
    }


def test_reference_manifest_ignores_groups_without_assets() -> None:
    group = EeaGroup("empty", "empty", "folder", "service", {}, (), None)

    manifest = manifest_state._reference_manifest(
        (group,), {}, source_version="EEA-test", crs="EPSG:3035", threshold=0, config={}
    )

    assert manifest["assets"] == []


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
        "software": manifest_state._software_provenance(),
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
        "_reference_manifest",
        lambda *args, **kwargs: reference,
    )
    monkeypatch.setattr(
        manifest_state,
        "_load_existing_manifest",
        lambda *args, **kwargs: manifest_state._ExistingManifest("target", manifest),
    )
    monkeypatch.setattr(manifest_state, "_verify_no_op_dataset", lambda *args, **kwargs: receipt)
    monkeypatch.setattr(
        release_orchestration,
        "_process_reference_groups",
        lambda *args, **kwargs: pytest.fail("matching release must not process source shards"),
    )
    monkeypatch.setattr(
        release_orchestration,
        "_finalize_plan",
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
        "software": manifest_state._software_provenance(),
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
    assert loaded == manifest_state._ExistingManifest("target-revision", manifest)


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
    monkeypatch.setattr(manifest_state, "_shared_blobs", lambda *args, **kwargs: {})
    monkeypatch.setattr(
        manifest_state, "_verify_final_dataset", lambda *args, **kwargs: verification
    )

    result = manifest_state._verify_no_op_dataset(
        cast(HubApi, object()),
        plan,
        manifest_state._ExistingManifest("target", manifest),
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
        item = manifest_state._ExistingManifest("target-revision", manifest)
    monkeypatch.setattr(manifest_state, "_load_existing_manifests", lambda *args: (item,))
    monkeypatch.setattr(
        manifest_state,
        "_verify_no_op_dataset",
        lambda *args, **kwargs: pytest.fail("incompatible manifests must not be verified as no-op"),
    )

    receipt = manifest_state._try_no_op_release(
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
    existing = manifest_state._ExistingManifest("target-revision", manifest)
    monkeypatch.setattr(manifest_state, "_load_existing_manifests", lambda *args: (existing,))

    with pytest.raises(VerificationError, match="incomplete"):
        manifest_state._try_no_op_release(
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
    existing = manifest_state._ExistingManifest("target-revision", manifest)
    receipt = DatasetReceipt(
        plan,
        (),
        VerificationReceipt("target", "target-revision", {}, (), manifest),
        no_op=True,
    )
    monkeypatch.setattr(manifest_state, "_load_existing_manifests", lambda *args: (existing,))
    monkeypatch.setattr(manifest_state, "_no_op_receipts", lambda *args, **kwargs: (receipt,))
    events: list[Mapping[str, object]] = []

    result = manifest_state._try_no_op_release(
        cast(HubApi, object()),
        (plan,),
        {"source_version": "EEA-test"},
        workdir=tmp_path,
        client=cast(StreamClient, object()),
        progress=events.append,
    )

    assert result == ReleaseReceipt((receipt,), {"source_version": "EEA-test"})
    assert events == [{"event": "dataset_no_op", "dataset": "website"}]
