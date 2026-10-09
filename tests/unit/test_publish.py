from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.utils import httpx

from osm_polygon_eunis import publish
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.publish import (
    DatasetVerificationOptions,
    ManifestBuildOptions,
    ShardExpectation,
    build_manifest,
    duplicate_source,
    parquet_signature,
    upload_manifest,
    upload_replacement,
    verify_dataset,
)


def test_duplicate_source_skips_existing_target() -> None:
    calls: list[tuple[str, str]] = []

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(id="target")

    def duplicate(source: str, target: str, **_kwargs):
        calls.append((source, target))

    assert duplicate_source(cast(HubApi, Api()), "source", "target", duplicate=duplicate) is False
    assert calls == []


def test_duplicate_source_calls_server_side_copy_once_when_missing() -> None:
    calls: list[tuple[str, str]] = []

    class Api:
        def repo_info(self, *_args, **_kwargs):
            from huggingface_hub.utils import RepositoryNotFoundError

            raise RepositoryNotFoundError(
                "missing",
                response=httpx.Response(404, request=httpx.Request("GET", "https://example.test")),
            )

    def duplicate(source: str, target: str, **_kwargs):
        calls.append((source, target))

    assert duplicate_source(cast(HubApi, Api()), "source", "target", duplicate=duplicate) is True
    assert calls == [("source", "target")]


def test_manifest_is_deterministic_and_contains_source_conservation() -> None:
    manifest = build_manifest(
        ManifestBuildOptions(
            source_repo="org/source",
            target_repo="org/target",
            source_revision="abc",
            source_paths=("polygons/z.parquet", "polygons/a.parquet"),
            changed_paths=("polygons/z.parquet",),
            reference_manifest={"version": "EEA-test", "sha256": "ref"},
            rows_by_path={"polygons/z.parquet": 12},
        )
    )

    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    assert list(manifest["source_paths"]) == ["polygons/a.parquet", "polygons/z.parquet"]
    assert manifest["changed_paths"] == ["polygons/z.parquet"]
    assert json.dumps(manifest, sort_keys=True, separators=(",", ":")) == encoded


def test_manifest_records_software_provenance(monkeypatch) -> None:
    source_commit = "0123456789abcdef0123456789abcdef01234567"
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", source_commit)

    manifest = build_manifest(
        ManifestBuildOptions(
            source_repo="org/source",
            target_repo="org/target",
            source_revision="abc",
            source_paths=("polygons/a.parquet",),
            changed_paths=("polygons/a.parquet",),
            reference_manifest={"version": "EEA-test"},
            rows_by_path={"polygons/a.parquet": 1},
        )
    )

    assert manifest["manifest_version"] == 5
    assert manifest["software"] == {
        "name": "osm-polygon-eunis",
        "version": "0.1.0",
        "commit": source_commit,
    }


def test_manifest_rejects_invalid_source_commit(monkeypatch) -> None:
    monkeypatch.setenv("EUNIS_SOURCE_COMMIT", "not-a-commit")

    with pytest.raises(ValueError, match="40- or 64-character hexadecimal SHA"):
        publish._software_provenance()


def test_manifest_uses_git_commit_when_environment_is_empty(monkeypatch) -> None:
    source_commit = "abcdef0123456789abcdef0123456789abcdef01"
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(publish.shutil, "which", lambda _name: "/usr/bin/git")

    def run_git(arguments, **_kwargs):
        if tuple(arguments[1:]) == ("status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout=f"{source_commit}\n")

    monkeypatch.setattr(publish.subprocess, "run", run_git)

    assert publish._software_provenance()["commit"] == source_commit


def test_manifest_rejects_dirty_git_checkout_when_falling_back_to_head(monkeypatch) -> None:
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(publish.shutil, "which", lambda _name: "/usr/bin/git")

    def run_git(arguments, **_kwargs):
        if tuple(arguments[1:]) == ("status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout=" M src/osm_polygon_eunis/publish.py\n")
        pytest.fail("a dirty checkout must be rejected before accepting HEAD")

    monkeypatch.setattr(publish.subprocess, "run", run_git)

    with pytest.raises(ValueError, match="source checkout is dirty"):
        publish._software_provenance()


@pytest.mark.parametrize("git_path", (None, "/usr/bin/git"), ids=("git-not-found", "git-failed"))
def test_manifest_requires_source_commit(monkeypatch, git_path: str | None) -> None:
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(
        publish.shutil,
        "which",
        lambda _name: git_path,
    )
    if git_path is not None:
        monkeypatch.setattr(
            publish.subprocess,
            "run",
            lambda *_args, **_kwargs: SimpleNamespace(returncode=128, stdout=""),
        )

    with pytest.raises(ValueError, match="source commit unavailable"):
        publish._software_provenance()


def test_manifest_records_added_card_artifacts() -> None:
    manifest = build_manifest(
        ManifestBuildOptions(
            source_repo="org/source",
            target_repo="org/target",
            source_revision="abc",
            source_paths=("README.md", "polygons/a.parquet"),
            changed_paths=("README.md", "polygons/a.parquet"),
            added_paths=("eunis/world-map.svg",),
            reference_manifest={"version": "EEA-test"},
            rows_by_path={"polygons/a.parquet": 1},
            schema_by_path={"polygons/a.parquet": "schema"},
            card_manifest={
                "map_path": "eunis/world-map.svg",
                "geometry_policy": {"overlap_kernel_version": 3},
            },
        )
    )

    assert manifest["manifest_version"] == 5
    assert manifest["added_paths"] == ["eunis/world-map.svg"]
    assert manifest["shared_paths"] == []
    assert manifest["schema_by_path"] == {"polygons/a.parquet": "schema"}
    assert manifest["card"] == {
        "map_path": "eunis/world-map.svg",
        "geometry_policy": {"overlap_kernel_version": 3},
    }
    assert manifest["geometry_policy"] == {"overlap_kernel_version": 3}


def test_manifest_geometry_policy_ignores_non_mapping_card_value() -> None:
    assert publish._manifest_geometry_policy({"geometry_policy": ["unexpected"]}) is None


def test_upload_replacement_and_manifest_use_explicit_paths(tmp_path: Path) -> None:
    source = tmp_path / "replacement.parquet"
    source.write_bytes(b"replacement")
    calls: list[dict[str, object]] = []

    class Api:
        def upload_file(self, **kwargs):
            calls.append(kwargs)
            return "commit"

    api = Api()
    upload_replacement(cast(HubApi, api), "org/target", "polygons/france.parquet", source)
    upload_manifest(cast(HubApi, api), "org/target", {"source_revision": "abc"})

    assert calls[0]["path_in_repo"] == "polygons/france.parquet"
    assert calls[1]["path_in_repo"] == "eunis/manifest.json"
    assert calls[1]["path_or_fileobj"] == b'{"source_revision":"abc"}\n'


def test_parquet_signature_reports_rows_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a", "b"]}), path)

    rows, schema = parquet_signature(path)

    assert rows == 2
    expected = hashlib.sha256(
        pa.schema([("polygon_id", pa.string())]).serialize().to_pybytes()
    ).hexdigest()
    assert schema == expected


def test_verify_dataset_rejects_missing_target_paths() -> None:
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="target-sha")

        def list_repo_tree(self, *_args, **_kwargs):
            return iter((SimpleNamespace(type="file", path="README.md", blob_id="readme"),))

    with pytest.raises(ValueError, match="missing target paths"):
        verify_dataset(
            cast(HubApi, Api()),
            "org/target",
            DatasetVerificationOptions(
                expectations=(ShardExpectation("polygons/a.parquet", 1, "schema"),),
                expected_tree_paths=("polygons/a.parquet",),
            ),
        )


class _FakeTargetApi:
    def __init__(self, entries: tuple[SimpleNamespace, ...]) -> None:
        self.entries = entries

    def repo_info(self, *_args, **_kwargs):
        return SimpleNamespace(sha="target-sha")

    def list_repo_tree(self, *_args, **_kwargs):
        return iter(self.entries)


def _verification_kwargs(tmp_path: Path, monkeypatch) -> dict:
    parquet = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a"]}), parquet)
    rows, schema = parquet_signature(parquet)
    manifest = {"source_revision": "abc", "changed_paths": ["polygons/a.parquet"]}
    artifact = b"static-map"

    def fake_download(api, repo_id, path, revision, directory, *, client=None):
        del api, repo_id, revision
        del client
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / path.replace("/", "__")
        if path == "eunis/manifest.json":
            destination.write_text(json.dumps(manifest), encoding="utf-8")
        elif path == "eunis/world-map.svg":
            destination.write_bytes(artifact)
        else:
            destination.write_bytes(parquet.read_bytes())
        return destination

    monkeypatch.setattr("osm_polygon_eunis.publish.download_to_temp", fake_download)
    return {
        "expectations": (ShardExpectation("polygons/a.parquet", rows, schema),),
        "expected_tree_paths": (
            "README.md",
            "polygons/a.parquet",
            "eunis/manifest.json",
            "eunis/world-map.svg",
        ),
        "expected_shared_blobs": {"README.md": "readme"},
        "expected_manifest": manifest,
        "expected_artifacts": {
            "eunis/world-map.svg": hashlib.sha256(artifact).hexdigest(),
        },
        "temp_dir": tmp_path / "verify",
    }


_TARGET_ENTRIES = (
    SimpleNamespace(path="README.md", blob_id="readme"),
    SimpleNamespace(path="polygons/a.parquet", blob_id="new-shard"),
    SimpleNamespace(path="eunis/manifest.json", blob_id="manifest"),
    SimpleNamespace(path="eunis/world-map.svg", blob_id="map"),
)


def test_verify_dataset_checks_remote_parquet_shared_blobs_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    kwargs = _verification_kwargs(tmp_path, monkeypatch)

    receipt = verify_dataset(
        cast(HubApi, _FakeTargetApi(_TARGET_ENTRIES)),
        "org/target",
        DatasetVerificationOptions(**kwargs),
    )

    assert receipt.target_revision == "target-sha"
    assert receipt.rows_by_path == {"polygons/a.parquet": 1}
    assert receipt.manifest == kwargs["expected_manifest"]


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"expected_tree_paths": ("README.md", "polygons/a.parquet", "eunis/manifest.json")},
            "unexpected target paths",
        ),
        ({"expected_shared_blobs": {"README.md": "old-readme"}}, "shared path changed"),
        (
            {"expectations": (ShardExpectation("polygons/a.parquet", 2, "schema"),)},
            "remote Parquet mismatch",
        ),
        (
            {"expected_manifest": {"source_revision": "other"}},
            "remote EUNIS manifest does not match",
        ),
        (
            {"expected_artifacts": {"eunis/world-map.svg": "0" * 64}},
            "remote artifact mismatch",
        ),
    ],
    ids=["extra-path", "shared-blob", "parquet", "manifest", "artifact"],
)
def test_verify_dataset_fails_closed_on_remote_mismatch(
    tmp_path: Path,
    monkeypatch,
    override: dict,
    message: str,
) -> None:
    kwargs = _verification_kwargs(tmp_path, monkeypatch) | override

    with pytest.raises(ValueError, match=message):
        verify_dataset(
            cast(HubApi, _FakeTargetApi(_TARGET_ENTRIES)),
            "org/target",
            DatasetVerificationOptions(**kwargs),
        )


@pytest.mark.parametrize(
    ("changed", "added", "message"),
    [
        (("polygons/missing.parquet",), (), "changed paths are not a subset"),
        ((), ("polygons/a.parquet",), "added paths already exist"),
        (("polygons/a.parquet",), ("polygons/a.parquet",), "cannot be both changed and added"),
    ],
    ids=["changed-not-in-source", "added-in-source", "added-and-changed"],
)
def test_build_manifest_rejects_inconsistent_path_sets(
    changed: tuple[str, ...], added: tuple[str, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        build_manifest(
            ManifestBuildOptions(
                source_repo="org/source",
                target_repo="org/target",
                source_revision="abc",
                source_paths=("polygons/a.parquet",),
                changed_paths=changed,
                added_paths=added,
                reference_manifest={"version": "EEA-test"},
                rows_by_path={},
            )
        )


def test_upload_replacements_groups_several_shards_into_one_commit(tmp_path: Path) -> None:
    from osm_polygon_eunis.publish import upload_replacements

    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    calls: list[dict[str, Any]] = []

    class Api:
        def create_commit(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(oid="grouped")

    result = upload_replacements(
        cast(HubApi, Api()),
        "org/target",
        [("polygons/a.parquet", first), ("polygons/b.parquet", second)],
        parent_commit="base",
    )

    assert result.oid == "grouped"
    assert len(calls) == 1
    assert calls[0]["parent_commit"] == "base"
    assert [op.path_in_repo for op in calls[0]["operations"]] == [
        "polygons/a.parquet",
        "polygons/b.parquet",
    ]
