from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub.utils import httpx

from osm_polygon_eunis import publish
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.publish import (
    DatasetVerificationOptions,
    ManifestBuildOptions,
    ShardExpectation,
    VerificationError,
    VerificationReceipt,
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
        publish.software_provenance()


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

    assert publish.software_provenance()["commit"] == source_commit


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
        publish.software_provenance()


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
        publish.software_provenance()


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


def test_target_exists_queries_the_dataset_repository_read_only() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Api:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            calls.append((args, kwargs))
            return SimpleNamespace(id="org/target")

    assert publish.target_exists(cast(HubApi, Api()), "org/target") is True
    assert calls == [(("org/target",), {"repo_type": "dataset"})]


def test_duplicate_source_copies_only_an_absent_target_with_exact_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = cast(HubApi, object())
    exists_calls: list[tuple[object, str]] = []
    duplicate_calls: list[tuple[str, str, dict[str, object]]] = []

    def fake_exists(api_arg: object, repo: str) -> bool:
        exists_calls.append((api_arg, repo))
        return False

    def duplicate(source: str, target: str, **kwargs: object) -> None:
        duplicate_calls.append((source, target, kwargs))

    monkeypatch.setattr(publish, "target_exists", fake_exists)

    assert (
        publish.duplicate_source(
            api,
            "org/source",
            "org/target",
            token="hf-token",
            duplicate=duplicate,
        )
        is True
    )
    assert exists_calls == [(api, "org/target")]
    assert duplicate_calls == [
        (
            "org/source",
            "org/target",
            {
                "repo_type": "dataset",
                "private": False,
                "token": "hf-token",
                "exist_ok": False,
            },
        )
    ]


def test_upload_replacement_rejects_a_missing_file_by_its_own_path(tmp_path: Path) -> None:
    missing = tmp_path / "absent.parquet"

    class Api:
        def upload_file(self, **_kwargs: object) -> None:
            pytest.fail("a missing shard must not reach the Hub")

    with pytest.raises(FileNotFoundError) as excinfo:
        publish.upload_replacement(
            cast(HubApi, Api()),
            "org/target",
            "polygons/a.parquet",
            missing,
        )

    assert excinfo.value.args == (missing,)


def test_upload_replacement_passes_exact_hub_arguments(tmp_path: Path) -> None:
    local = tmp_path / "replacement.parquet"
    local.write_bytes(b"replacement")
    calls: list[dict[str, object]] = []

    class Api:
        def upload_file(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "commit"

    api = cast(HubApi, Api())
    assert publish.upload_replacement(api, "org/target", "polygons/a.parquet", local) == "commit"
    publish.upload_replacement(
        api,
        "org/target",
        "polygons/b.parquet",
        local,
        parent_commit="base",
        commit_message="Custom message",
    )

    assert calls == [
        {
            "path_or_fileobj": local,
            "path_in_repo": "polygons/a.parquet",
            "repo_id": "org/target",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": None,
            "commit_message": "Add EUNIS labels to polygons/a.parquet",
        },
        {
            "path_or_fileobj": local,
            "path_in_repo": "polygons/b.parquet",
            "repo_id": "org/target",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": "base",
            "commit_message": "Custom message",
        },
    ]


def test_upload_replacements_single_shard_forwards_every_argument(tmp_path: Path) -> None:
    local = tmp_path / "a.parquet"
    local.write_bytes(b"a")
    calls: list[dict[str, object]] = []

    class Api:
        def upload_file(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "single"

        def create_commit(self, **_kwargs: object) -> None:
            pytest.fail("one shard is a plain upload, not a commit")

    result = publish.upload_replacements(
        cast(HubApi, Api()),
        "org/target",
        [("polygons/a.parquet", local)],
        parent_commit="base",
        commit_message="Custom message",
    )

    assert result == "single"
    assert calls == [
        {
            "path_or_fileobj": local,
            "path_in_repo": "polygons/a.parquet",
            "repo_id": "org/target",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": "base",
            "commit_message": "Custom message",
        }
    ]


def test_upload_replacements_commits_several_shards_with_exact_arguments(tmp_path: Path) -> None:
    first, second = tmp_path / "a.parquet", tmp_path / "b.parquet"
    first.write_bytes(b"a")
    second.write_bytes(b"b")
    calls: list[dict[str, Any]] = []

    class Api:
        def create_commit(self, **kwargs: Any) -> SimpleNamespace:
            calls.append(kwargs)
            return SimpleNamespace(oid="grouped")

    result = publish.upload_replacements(
        cast(HubApi, Api()),
        "org/target",
        [("polygons/a.parquet", first), ("polygons/b.parquet", second)],
        parent_commit="base",
    )

    assert result.oid == "grouped"
    assert len(calls) == 1
    operations = calls[0].pop("operations")
    assert [(op.path_in_repo, op.path_or_fileobj) for op in operations] == [
        ("polygons/a.parquet", str(first)),
        ("polygons/b.parquet", str(second)),
    ]
    assert calls == [
        {
            "repo_id": "org/target",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": "base",
            "commit_message": "Add EUNIS labels to 2 shards",
        }
    ]


def test_upload_manifest_uses_the_fixed_path_branch_and_message() -> None:
    calls: list[dict[str, object]] = []

    class Api:
        def upload_file(self, **kwargs: object) -> str:
            calls.append(kwargs)
            return "manifest-commit"

    result = publish.upload_manifest(
        cast(HubApi, Api()),
        "org/target",
        {"b": 1, "a": 2},
        parent_commit="base",
    )

    assert result == "manifest-commit"
    assert calls == [
        {
            "path_or_fileobj": b'{"a":2,"b":1}\n',
            "path_in_repo": "eunis/manifest.json",
            "repo_id": "org/target",
            "repo_type": "dataset",
            "revision": "main",
            "parent_commit": "base",
            "commit_message": "Record EUNIS enrichment manifest",
        }
    ]


def test_manifest_bytes_are_sorted_compact_ascii_json() -> None:
    assert publish._manifest_bytes({"b": 1, "a": "é"}) == b'{"a":"\\u00e9","b":1}\n'


@pytest.mark.parametrize(
    ("environment", "expected"),
    [
        (
            {
                "EUNIS_SOURCE_COMMIT": "a" * 40,
                "GRID5000_SOURCE_REVISION": "b" * 40,
                "GITHUB_SHA": "c" * 40,
            },
            "a" * 40,
        ),
        ({"GRID5000_SOURCE_REVISION": "b" * 40, "GITHUB_SHA": "c" * 40}, "b" * 40),
        ({"GITHUB_SHA": "c" * 40}, "c" * 40),
        ({}, None),
    ],
    ids=["eunis-first", "grid5000-second", "github-third", "none-declared"],
)
def test_environment_source_commit_prefers_declared_variables_in_order(
    monkeypatch: pytest.MonkeyPatch,
    environment: dict[str, str],
    expected: str | None,
) -> None:
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)

    assert publish._environment_source_commit() == expected


def test_git_source_commit_queries_only_the_source_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_commit = "abcdef0123456789abcdef0123456789abcdef01"
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    which_calls: list[str] = []
    run_calls: list[tuple[tuple[str, ...], dict[str, object]]] = []

    def which(name: str) -> str:
        which_calls.append(name)
        return "/usr/bin/git"

    def run_git(arguments: tuple[str, ...], **kwargs: object) -> SimpleNamespace:
        run_calls.append((tuple(arguments), kwargs))
        if tuple(arguments[1:]) == ("status", "--porcelain"):
            return SimpleNamespace(returncode=0, stdout="")
        return SimpleNamespace(returncode=0, stdout=f"{source_commit}\n")

    monkeypatch.setattr(publish.shutil, "which", which)
    monkeypatch.setattr(publish.subprocess, "run", run_git)
    source_root = Path(publish.__file__).resolve().parents[2]

    assert publish.software_provenance()["commit"] == source_commit
    assert which_calls == ["git"]
    expected_options = {
        "cwd": source_root,
        "capture_output": True,
        "check": False,
        "text": True,
    }
    assert run_calls == [
        (("/usr/bin/git", "status", "--porcelain"), expected_options),
        (("/usr/bin/git", "rev-parse", "HEAD"), expected_options),
    ]


def test_git_source_commit_refuses_a_dirty_checkout_with_the_exact_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("EUNIS_SOURCE_COMMIT", "GRID5000_SOURCE_REVISION", "GITHUB_SHA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(publish.shutil, "which", lambda _name: "/usr/bin/git")
    monkeypatch.setattr(
        publish.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=" M src/x.py\n"),
    )

    with pytest.raises(ValueError) as excinfo:
        publish.software_provenance()

    assert str(excinfo.value) == (
        "source checkout is dirty; commit changes or set EUNIS_SOURCE_COMMIT to a full commit SHA"
    )


def test_git_command_output_runs_in_the_requested_directory(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    subprocess.run(  # noqa: S603 - resolved Git executable and fixed arguments
        [git, "init", "-q"], cwd=tmp_path, check=True
    )

    assert publish._git_command_output(git, ("rev-parse", "--show-toplevel"), tmp_path) == str(
        tmp_path.resolve()
    )
    version_output = publish._git_command_output(git, ("--version",), tmp_path)
    assert version_output is not None
    assert version_output.startswith("git version ")


def test_git_command_output_reports_failure_as_none(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git is not installed")
    subprocess.run(  # noqa: S603 - resolved Git executable and fixed arguments
        [git, "init", "-q"], cwd=tmp_path, check=True
    )

    assert publish._git_command_output(git, ("rev-parse", "HEAD"), tmp_path) is None
    assert publish._git_command_output("/nonexistent/git", ("status",), tmp_path) is None


def test_validated_source_commit_accepts_upper_case_and_returns_lower_case() -> None:
    assert publish._validated_source_commit("A" * 40) == "a" * 40
    assert publish._validated_source_commit("F" * 64) == "f" * 64


def test_validated_source_commit_reports_the_exact_failures() -> None:
    with pytest.raises(ValueError) as missing:
        publish._validated_source_commit(None)
    assert str(missing.value) == (
        "source commit unavailable; set EUNIS_SOURCE_COMMIT to a full commit SHA"
    )

    with pytest.raises(ValueError) as invalid:
        publish._validated_source_commit("z" * 40)
    assert str(invalid.value) == "source commit must be a 40- or 64-character hexadecimal SHA"


@pytest.mark.parametrize(
    ("source", "changed", "added", "message"),
    [
        (
            ("polygons/a.parquet",),
            ("polygons/missing.parquet",),
            (),
            "changed paths are not a subset of source paths",
        ),
        (
            ("polygons/a.parquet",),
            (),
            ("polygons/a.parquet",),
            "added paths already exist in source paths",
        ),
        (
            ("polygons/a.parquet",),
            ("polygons/a.parquet",),
            ("polygons/a.parquet",),
            "a path cannot be both changed and added",
        ),
    ],
    ids=["changed-not-in-source", "added-in-source", "added-and-changed"],
)
def test_build_manifest_names_the_exact_inconsistent_path_set(
    source: tuple[str, ...],
    changed: tuple[str, ...],
    added: tuple[str, ...],
    message: str,
) -> None:
    with pytest.raises(ValueError) as excinfo:
        publish.build_manifest(
            ManifestBuildOptions(
                source_repo="org/source",
                target_repo="org/target",
                source_revision="abc",
                source_paths=source,
                changed_paths=changed,
                added_paths=added,
                reference_manifest={"version": "EEA-test"},
                rows_by_path={},
            )
        )

    assert str(excinfo.value) == message


def test_build_manifest_exact_output_with_card_and_schema() -> None:
    manifest = publish.build_manifest(
        ManifestBuildOptions(
            source_repo="org/source",
            target_repo="org/target",
            source_revision="rev",
            source_paths=("polygons/b.parquet", "README.md", "polygons/a.parquet"),
            changed_paths=("polygons/a.parquet", "README.md"),
            added_paths=("eunis/world-map.svg",),
            reference_manifest={"version": "EEA-test"},
            rows_by_path={"polygons/b.parquet": 3, "polygons/a.parquet": 2},
            schema_by_path={"polygons/a.parquet": "schema-a"},
            card_manifest={
                "map_path": "eunis/world-map.svg",
                "geometry_policy": {"overlap_kernel_version": 3},
            },
        )
    )

    assert manifest == {
        "manifest_version": 5,
        "source_repo": "org/source",
        "target_repo": "org/target",
        "source_revision": "rev",
        "source_paths": ["README.md", "polygons/a.parquet", "polygons/b.parquet"],
        "changed_paths": ["README.md", "polygons/a.parquet"],
        "added_paths": ["eunis/world-map.svg"],
        "shared_paths": ["polygons/b.parquet"],
        "rows_by_path": {"polygons/a.parquet": 2, "polygons/b.parquet": 3},
        "schema_by_path": {"polygons/a.parquet": "schema-a"},
        "software": publish.software_provenance(),
        "reference": {"version": "EEA-test"},
        "geometry_policy": {"overlap_kernel_version": 3},
        "card": {
            "map_path": "eunis/world-map.svg",
            "geometry_policy": {"overlap_kernel_version": 3},
        },
    }


def test_build_manifest_without_a_card_records_null_card_fields() -> None:
    manifest = publish.build_manifest(
        ManifestBuildOptions(
            source_repo="org/source",
            target_repo="org/target",
            source_revision="rev",
            source_paths=("polygons/a.parquet",),
            changed_paths=(),
            reference_manifest={"version": "EEA-test"},
            rows_by_path={"polygons/a.parquet": 3},
        )
    )

    assert manifest["shared_paths"] == ["polygons/a.parquet"]
    assert manifest["schema_by_path"] == {}
    assert manifest["geometry_policy"] is None
    assert manifest["card"] is None


def test_remote_files_lists_the_recursive_tree_and_keeps_only_file_paths() -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    class Api:
        def list_repo_tree(self, *args: object, **kwargs: object) -> object:
            calls.append((args, kwargs))
            return iter(
                (
                    SimpleNamespace(path="README.md", blob_id="readme"),
                    SimpleNamespace(path="polygons", tree_id="directory"),
                    SimpleNamespace(blob_id="unnamed"),
                    SimpleNamespace(path=7, blob_id="not-a-path"),
                )
            )

    result = publish._remote_files(cast(HubApi, Api()), "org/target", "rev")

    assert calls == [
        (
            ("org/target",),
            {"path_in_repo": "", "recursive": True, "revision": "rev", "repo_type": "dataset"},
        )
    ]
    assert list(result) == ["README.md"]
    assert result["README.md"].blob_id == "readme"


def test_verify_dataset_pins_every_remote_read_to_the_captured_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a"]}), parquet)
    rows, schema = parquet_signature(parquet)
    manifest = {"source_revision": "abc"}
    artifact = b"static-map"
    info_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    tree_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    download_calls: list[tuple[object, str, str, str, Path, object]] = []

    class Api:
        def repo_info(self, *args: object, **kwargs: object) -> SimpleNamespace:
            info_calls.append((args, kwargs))
            return SimpleNamespace(sha="target-sha")

        def list_repo_tree(self, *args: object, **kwargs: object) -> object:
            tree_calls.append((args, kwargs))
            return iter(_TARGET_ENTRIES)

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
        destination = directory / path.replace("/", "__")
        if path == "eunis/manifest.json":
            destination.write_text(json.dumps(manifest), encoding="utf-8")
        elif path == "eunis/world-map.svg":
            destination.write_bytes(artifact)
        else:
            destination.write_bytes(parquet.read_bytes())
        return destination

    monkeypatch.setattr(publish, "download_to_temp", fake_download)
    api = Api()
    client = cast(StreamClient, object())
    workdir = tmp_path / "verify"

    receipt = publish.verify_dataset(
        cast(HubApi, api),
        "org/target",
        DatasetVerificationOptions(
            expectations=(ShardExpectation("polygons/a.parquet", rows, schema),),
            expected_tree_paths=(
                "README.md",
                "polygons/a.parquet",
                "eunis/manifest.json",
                "eunis/world-map.svg",
            ),
            expected_shared_blobs={"README.md": "readme"},
            expected_manifest=manifest,
            expected_artifacts={"eunis/world-map.svg": hashlib.sha256(artifact).hexdigest()},
            temp_dir=workdir,
            http_client=client,
        ),
    )

    assert info_calls == [(("org/target",), {"repo_type": "dataset"})]
    assert tree_calls == [
        (
            ("org/target",),
            {
                "path_in_repo": "",
                "recursive": True,
                "revision": "target-sha",
                "repo_type": "dataset",
            },
        )
    ]
    assert download_calls == [
        (api, "org/target", "polygons/a.parquet", "target-sha", workdir, client),
        (api, "org/target", "eunis/manifest.json", "target-sha", workdir, client),
        (api, "org/target", "eunis/world-map.svg", "target-sha", workdir, client),
    ]
    assert receipt == VerificationReceipt(
        "org/target",
        "target-sha",
        {"polygons/a.parquet": rows},
        ("README.md",),
        manifest,
    )


def test_verify_dataset_downloads_into_a_private_directory_that_is_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parquet = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a"]}), parquet)
    rows, schema = parquet_signature(parquet)
    manifest = {"source_revision": "abc"}
    directories: list[Path] = []

    class Api:
        def repo_info(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return SimpleNamespace(sha="target-sha")

        def list_repo_tree(self, *_args: object, **_kwargs: object) -> object:
            return iter(
                (
                    SimpleNamespace(path="polygons/a.parquet", blob_id="shard"),
                    SimpleNamespace(path="eunis/manifest.json", blob_id="manifest"),
                )
            )

    def fake_download(
        api_arg: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        del api_arg, repo_id, revision, client
        directories.append(directory)
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / path.replace("/", "__")
        if path == "eunis/manifest.json":
            destination.write_text(json.dumps(manifest), encoding="utf-8")
        else:
            destination.write_bytes(parquet.read_bytes())
        return destination

    monkeypatch.setattr(publish, "download_to_temp", fake_download)

    receipt = publish.verify_dataset(
        cast(HubApi, Api()),
        "org/target",
        DatasetVerificationOptions(
            expectations=(ShardExpectation("polygons/a.parquet", rows, schema),),
            expected_tree_paths=("polygons/a.parquet", "eunis/manifest.json"),
            expected_manifest=manifest,
        ),
    )

    assert receipt.shared_paths == ()
    assert len(directories) == 2
    assert all(directory.name.startswith("osm-polygon-eunis-verify-") for directory in directories)
    assert not any(directory.exists() for directory in directories)


def test_verify_shared_blobs_treats_a_missing_blob_id_as_a_changed_path() -> None:
    with pytest.raises(VerificationError) as excinfo:
        publish._verify_shared_blobs(
            {"README.md": SimpleNamespace(path="README.md")},
            {"README.md": "readme"},
        )
    assert str(excinfo.value) == "shared path changed: README.md"

    assert publish._verify_shared_blobs({}, None) is None


@pytest.mark.parametrize(
    ("rows_delta", "schema"),
    [(1, "schema-from-parquet"), (0, "not-the-schema")],
    ids=["rows-only", "schema-only"],
)
def test_verify_expectations_rejects_either_mismatched_field(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rows_delta: int,
    schema: str,
) -> None:
    parquet = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a"]}), parquet)
    rows, real_schema = parquet_signature(parquet)
    if schema == "schema-from-parquet":
        schema = real_schema
    downloads: list[tuple[object, str, str, str, Path, object]] = []

    def fake_download(
        api_arg: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        downloads.append((api_arg, repo_id, path, revision, directory, client))
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "shard.parquet"
        destination.write_bytes(parquet.read_bytes())
        return destination

    monkeypatch.setattr(publish, "download_to_temp", fake_download)
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    workdir = tmp_path / "verify"

    with pytest.raises(VerificationError) as excinfo:
        publish._verify_expectations(
            api,
            "org/target",
            "rev",
            (ShardExpectation("polygons/a.parquet", rows + rows_delta, schema),),
            workdir,
            client,
        )

    assert str(excinfo.value) == "remote Parquet mismatch: polygons/a.parquet"
    assert downloads == [(api, "org/target", "polygons/a.parquet", "rev", workdir, client)]


def test_verify_manifest_fetches_from_the_pinned_revision_and_names_mismatches(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[tuple[object, str, str, str, Path, object]] = []

    def fake_download(
        api_arg: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        downloads.append((api_arg, repo_id, path, revision, directory, client))
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "manifest.json"
        destination.write_text(json.dumps({"source_revision": "abc"}), encoding="utf-8")
        return destination

    monkeypatch.setattr(publish, "download_to_temp", fake_download)
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    workdir = tmp_path / "verify"

    assert publish._verify_manifest(api, "org/target", "rev", None, workdir, client) is None
    assert downloads == []
    assert publish._verify_manifest(
        api, "org/target", "rev", {"source_revision": "abc"}, workdir, client
    ) == {"source_revision": "abc"}
    with pytest.raises(VerificationError) as excinfo:
        publish._verify_manifest(
            api, "org/target", "rev", {"source_revision": "other"}, workdir, client
        )
    assert str(excinfo.value) == "remote EUNIS manifest does not match expected manifest"
    assert downloads == [
        (api, "org/target", "eunis/manifest.json", "rev", workdir, client),
        (api, "org/target", "eunis/manifest.json", "rev", workdir, client),
    ]


def test_verify_artifacts_fetch_each_artifact_from_the_pinned_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    downloads: list[tuple[object, str, str, str, Path, object]] = []

    def fake_download(
        api_arg: object,
        repo_id: str,
        path: str,
        revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        downloads.append((api_arg, repo_id, path, revision, directory, client))
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / "world-map.svg"
        destination.write_bytes(b"static-map")
        return destination

    monkeypatch.setattr(publish, "download_to_temp", fake_download)
    api = cast(HubApi, object())
    client = cast(StreamClient, object())
    workdir = tmp_path / "verify"

    with pytest.raises(VerificationError) as excinfo:
        publish._verify_artifacts(
            api,
            "org/target",
            "rev",
            {"eunis/world-map.svg": "0" * 64},
            workdir,
            client,
        )

    assert str(excinfo.value) == "remote artifact mismatch: eunis/world-map.svg"
    assert downloads == [(api, "org/target", "eunis/world-map.svg", "rev", workdir, client)]


def test_verify_manifest_reads_the_remote_manifest_as_utf8(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    encodings: list[object] = []
    original_read_text = Path.read_text

    def recording_read_text(self: Path, *, encoding: str | None = None) -> str:
        encodings.append(encoding)
        return original_read_text(self, encoding=encoding)

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
        destination.write_text(json.dumps({"source_revision": "abc"}), encoding="utf-8")
        return destination

    monkeypatch.setattr(Path, "read_text", recording_read_text)
    monkeypatch.setattr(publish, "download_to_temp", fake_download)

    assert publish._verify_manifest(
        cast(HubApi, object()),
        "org/target",
        "rev",
        {"source_revision": "abc"},
        tmp_path,
        cast(StreamClient, object()),
    ) == {"source_revision": "abc"}
    assert encodings == ["utf-8"]
