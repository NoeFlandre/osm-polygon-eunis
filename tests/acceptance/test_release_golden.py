from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_eunis import card_publishing, release_orchestration, release_plan, shard_processing
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.geometry import GEOMETRY_POLICY
from osm_polygon_eunis.geometry_jobs import _GeometryRunOptions
from osm_polygon_eunis.options import BatchLimits, ReleaseOptions
from osm_polygon_eunis.publish import (
    MANIFEST_VERSION,
    VerificationReceipt,
    _manifest_bytes,
)
from osm_polygon_eunis.sources import DatasetSpec

_PRE_REFACTOR_GOLDEN = {
    "eunis/world-map.svg": "c39b39feffc29e4621ee3aaab527ef75e7dca7e2918044bf0c8bb867f050aba3",
    "polygon_document_links/region.parquet": (
        "3d8c8d95ce1a7f62a59e221f52ccd17dc5ba77f3e4b47a5a110b0d643b9c78c0"
    ),
    "polygons/region.parquet": "e2d3d549a14795d938b498a60b57d614f2e96b6378c1af4fce3a0b44aeae83b4",
}
_PRE_REFACTOR_STABLE_MANIFEST_SHA256 = (
    "e53ebfa8158f17de4a63c188a3fb9080ae7f2dd1359f3f19d5eef339f64a3765"
)
_CURRENT_README_SHA256 = "292fcd6a192a8b2c8c86967ed981c61c809ca70d7bdb95c29dc3d3bdbd57fd90"


class _GoldenHub:
    endpoint = "https://hub.example"
    token = None
    upload_file: Callable[..., Any]
    create_commit: Callable[..., Any]

    def __init__(
        self, source_files: Mapping[str, bytes], source_repo: str, target_repo: str
    ) -> None:
        self.repositories = {
            source_repo: dict(source_files),
            target_repo: {},
        }
        self.source_repo = source_repo
        self.upload_file = self._upload_file_impl
        self.create_commit = self._create_commit_impl

    def repo_info(self, repo_id: str, **_kwargs: object) -> SimpleNamespace:
        revision = "source-revision" if repo_id == self.source_repo else "target-base"
        return SimpleNamespace(sha=revision)

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
    ) -> tuple[SimpleNamespace, ...]:
        del path_in_repo, recursive, revision, repo_type
        return tuple(
            SimpleNamespace(path=path, blob_id=hashlib.sha256(payload).hexdigest())
            for path, payload in sorted(self.repositories[repo_id].items())
        )

    def _create_commit_impl(
        self,
        *,
        operations: list[Any],
        repo_id: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        for operation in operations:
            self._upload_file_impl(
                path_or_fileobj=Path(operation.path_or_fileobj),
                path_in_repo=operation.path_in_repo,
                repo_id=repo_id,
            )
        return SimpleNamespace(oid=f"target-commit-{len(self.repositories[repo_id])}")

    def _upload_file_impl(
        self,
        *,
        path_or_fileobj: object,
        path_in_repo: str,
        repo_id: str,
        **_kwargs: object,
    ) -> SimpleNamespace:
        if isinstance(path_or_fileobj, Path):
            payload = path_or_fileobj.read_bytes()
        elif isinstance(path_or_fileobj, bytes):
            payload = path_or_fileobj
        else:
            raise TypeError("golden Hub accepts only paths or byte payloads")
        self.repositories[repo_id][path_in_repo] = payload
        return SimpleNamespace(oid=f"target-commit-{len(self.repositories[repo_id])}")


def _stable_manifest_sha256(manifest: Mapping[str, Any]) -> str:
    """Hash unchanged fields; policy and provenance intentionally differ."""
    stable = dict(manifest)
    for field in ("manifest_version", "software", "geometry_policy"):
        stable.pop(field, None)
    card = dict(stable["card"])
    for field in ("geometry_policy", "readme_sha256"):
        card.pop(field, None)
    stable["card"] = card
    encoded = json.dumps(stable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def test_release_outputs_match_pre_refactor_golden_snapshot(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Pin stable release outputs captured with the monolithic runner at d134a19."""
    spec = DatasetSpec(
        "wikidata",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis",
        "polygons/*.parquet",
        "polygon_document_links/*.parquet",
    )
    source_tables = {
        "polygons/region.parquet": pa.table(
            {
                "polygon_id": ["a", "b"],
                "geometry": [
                    '{"type":"Polygon","coordinates":[[[0,0],[1,0],[1,1],[0,0]]]}',
                    '{"type":"Polygon","coordinates":[[[1,1],[2,1],[2,2],[1,1]]]}',
                ],
            }
        ),
        "polygon_document_links/region.parquet": pa.table({"polygon_id": ["a", "missing"]}),
    }
    source_files: dict[str, bytes] = {}
    for path, table in source_tables.items():
        local = tmp_path / path.replace("/", "__")
        pq.write_table(table, local)
        source_files[path] = local.read_bytes()
    hub = _GoldenHub(source_files, spec.source_repo, spec.output_repo)

    config = tmp_path / "reference.json"
    config.write_text(
        json.dumps({"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 0}),
        encoding="utf-8",
    )
    workdir = tmp_path / "release"

    def prepare_labels(options: _GeometryRunOptions) -> None:
        for plan in options.plans:
            sidecar = release_plan._sidecar_path(
                options.sidecar_root, plan.spec, plan.geometry_paths[0]
            )
            sidecar.parent.mkdir(parents=True, exist_ok=True)
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

    def download_source(
        _api: HubApi,
        repo_id: str,
        path: str,
        _revision: str,
        directory: Path,
        *,
        client: object = None,
    ) -> Path:
        del client
        destination = directory / path.replace("/", "__")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(hub.repositories[repo_id][path])
        return destination

    def verify_published(
        _api: HubApi,
        plan,
        options,
    ) -> VerificationReceipt:
        remote_files = hub.repositories[plan.spec.output_repo]
        assert remote_files["eunis/manifest.json"] == _manifest_bytes(options.expected_manifest)
        for path, expected_hash in options.expected_artifacts.items():
            assert hashlib.sha256(remote_files[path]).hexdigest() == expected_hash
        rows: dict[str, int] = {}
        for expected in options.expectations:
            table = pq.read_table(pa.BufferReader(remote_files[expected.path]))
            schema_hash = hashlib.sha256(table.schema.serialize().to_pybytes()).hexdigest()
            rows[expected.path] = table.num_rows
            assert rows[expected.path] == expected.rows
            assert schema_hash == expected.schema
        return VerificationReceipt(
            plan.spec.output_repo,
            "target-verified",
            rows,
            tuple(sorted(options.expected_shared_blobs)),
            options.expected_manifest,
        )

    monkeypatch.setattr(release_orchestration, "resolve_config_data", lambda _config: ())
    monkeypatch.setattr(release_orchestration, "_process_reference_groups", prepare_labels)
    monkeypatch.setattr(shard_processing, "download_to_temp", download_source)
    monkeypatch.setattr(card_publishing, "_verify_final_dataset", verify_published)

    receipt = release_orchestration.run_release(
        hub,
        ReleaseOptions(
            reference_config=config,
            workdir=workdir,
            limits=BatchLimits(parquet_batch_size=1),
            datasets=("wikidata",),
        ),
    )

    outputs = hub.repositories[spec.output_repo]
    output_hashes = {
        path: hashlib.sha256(outputs[path]).hexdigest() for path in _PRE_REFACTOR_GOLDEN
    }
    assert output_hashes == _PRE_REFACTOR_GOLDEN
    assert hashlib.sha256(outputs["README.md"]).hexdigest() == _CURRENT_README_SHA256

    manifest = json.loads(outputs["eunis/manifest.json"])
    assert manifest["manifest_version"] == MANIFEST_VERSION == 5
    assert manifest["geometry_policy"] == GEOMETRY_POLICY
    assert manifest["card"]["geometry_policy"] == GEOMETRY_POLICY
    assert _stable_manifest_sha256(manifest) == _PRE_REFACTOR_STABLE_MANIFEST_SHA256
    assert receipt.datasets[0].verification.target_revision == "target-verified"
