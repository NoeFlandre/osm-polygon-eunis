from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis.publish import (
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

    assert duplicate_source(Api(), "source", "target", duplicate=duplicate) is False
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

    assert duplicate_source(Api(), "source", "target", duplicate=duplicate) is True
    assert calls == [("source", "target")]


def test_manifest_is_deterministic_and_contains_source_conservation() -> None:
    manifest = build_manifest(
        source_repo="org/source",
        target_repo="org/target",
        source_revision="abc",
        source_paths=("polygons/z.parquet", "polygons/a.parquet"),
        changed_paths=("polygons/z.parquet",),
        reference_manifest={"version": "EEA-test", "sha256": "ref"},
        rows_by_path={"polygons/z.parquet": 12},
    )

    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))

    assert list(manifest["source_paths"]) == ["polygons/a.parquet", "polygons/z.parquet"]
    assert manifest["changed_paths"] == ["polygons/z.parquet"]
    assert json.dumps(manifest, sort_keys=True, separators=(",", ":")) == encoded


def test_manifest_records_added_card_artifacts() -> None:
    manifest = build_manifest(
        source_repo="org/source",
        target_repo="org/target",
        source_revision="abc",
        source_paths=("README.md", "polygons/a.parquet"),
        changed_paths=("README.md", "polygons/a.parquet"),
        added_paths=("eunis/world-map.svg",),
        reference_manifest={"version": "EEA-test"},
        rows_by_path={"polygons/a.parquet": 1},
        schema_by_path={"polygons/a.parquet": "schema"},
        card_manifest={"map_path": "eunis/world-map.svg"},
    )

    assert manifest["manifest_version"] == 3
    assert manifest["added_paths"] == ["eunis/world-map.svg"]
    assert manifest["shared_paths"] == []
    assert manifest["schema_by_path"] == {"polygons/a.parquet": "schema"}
    assert manifest["card"] == {"map_path": "eunis/world-map.svg"}


def test_upload_replacement_and_manifest_use_explicit_paths(tmp_path: Path) -> None:
    source = tmp_path / "replacement.parquet"
    source.write_bytes(b"replacement")
    calls: list[dict[str, object]] = []

    class Api:
        def upload_file(self, **kwargs):
            calls.append(kwargs)
            return "commit"

    api = Api()
    upload_replacement(api, "org/target", "polygons/france.parquet", source)
    upload_manifest(api, "org/target", {"source_revision": "abc"})

    assert calls[0]["path_in_repo"] == "polygons/france.parquet"
    assert calls[1]["path_in_repo"] == "eunis/manifest.json"
    assert calls[1]["path_or_fileobj"] == b'{"source_revision":"abc"}\n'


def test_parquet_signature_reports_rows_and_schema(tmp_path: Path) -> None:
    path = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a", "b"]}), path)

    rows, schema = parquet_signature(path)

    assert rows == 2
    assert isinstance(schema, str) and schema


def test_verify_dataset_rejects_missing_target_paths() -> None:
    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="target-sha")

        def list_repo_tree(self, *_args, **_kwargs):
            return iter((SimpleNamespace(type="file", path="README.md", blob_id="readme"),))

    with pytest.raises(ValueError, match="missing target paths"):
        verify_dataset(
            Api(),
            "org/target",
            expectations=(ShardExpectation("polygons/a.parquet", 1, "schema"),),
            expected_tree_paths=("polygons/a.parquet",),
        )


def test_verify_dataset_checks_remote_parquet_shared_blobs_and_manifest(
    tmp_path: Path,
    monkeypatch,
) -> None:
    parquet = tmp_path / "shard.parquet"
    pq.write_table(pa.table({"polygon_id": ["a"]}), parquet)
    rows, schema = parquet_signature(parquet)
    manifest = {"source_revision": "abc", "changed_paths": ["polygons/a.parquet"]}
    artifact = b"static-map"
    entries = (
        SimpleNamespace(path="README.md", blob_id="readme"),
        SimpleNamespace(path="polygons/a.parquet", blob_id="new-shard"),
        SimpleNamespace(path="eunis/manifest.json", blob_id="manifest"),
        SimpleNamespace(path="eunis/world-map.svg", blob_id="map"),
    )

    class Api:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="target-sha")

        def list_repo_tree(self, *_args, **_kwargs):
            return iter(entries)

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
    receipt = verify_dataset(
        Api(),
        "org/target",
        expectations=(ShardExpectation("polygons/a.parquet", rows, schema),),
        expected_tree_paths=(
            "README.md",
            "polygons/a.parquet",
            "eunis/manifest.json",
            "eunis/world-map.svg",
        ),
        expected_shared_blobs={"README.md": "readme"},
        expected_manifest=manifest,
        expected_artifacts={
            "eunis/world-map.svg": hashlib.sha256(artifact).hexdigest(),
        },
        temp_dir=tmp_path / "verify",
    )

    assert receipt.target_revision == "target-sha"
    assert receipt.rows_by_path == {"polygons/a.parquet": 1}
    assert receipt.manifest == manifest
