from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pytest_bdd import given, scenarios, then, when
from shapely.geometry import box, mapping
from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis import cli, manifest_state, release_orchestration, runner
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.cli import CliDependencies
from osm_polygon_eunis.domain import EunisResult
from osm_polygon_eunis.publish import ShardExpectation, VerificationReceipt
from osm_polygon_eunis.runner import DatasetPlan, DatasetReceipt
from osm_polygon_eunis.sources import DatasetSpec
from osm_polygon_eunis.transform import enrich_parquet_shard

scenarios("features/release_workflow.feature")


@pytest.fixture
def acceptance_state() -> dict[str, object]:
    return {}


class _FakeHub:
    def __init__(self) -> None:
        self.uploads: list[str] = []
        self.current_manifest: dict[str, object] | None = None

    def repo_info(self, repo_id: str, **_kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(sha=f"{repo_id}-revision")

    def upload_file(self, *, path_in_repo: str, **_kwargs: object) -> SimpleNamespace:
        self.uploads.append(path_in_repo)
        return SimpleNamespace(oid=f"upload-{len(self.uploads)}")


class _NoOverlapReference:
    def overlap(self, polygon: BaseGeometry | None) -> EunisResult:
        del polygon
        return EunisResult(None, None, None, None)


def _first_dataset(output: dict[str, object]) -> dict[str, object]:
    datasets = cast(list[object], output["datasets"])
    return cast(dict[str, object], datasets[0])


def _write_config(path: Path, source_version: str = "EEA-test") -> None:
    path.write_text(
        json.dumps({"source_version": source_version, "crs": "EPSG:3035", "threshold": 0}),
        encoding="utf-8",
    )


def _plan() -> DatasetPlan:
    return DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("README.md", "polygons/a.parquet"),
        ("polygons/a.parquet",),
        (),
    )


def _reference_for(config: Path) -> dict[str, object]:
    settings = json.loads(config.read_text(encoding="utf-8"))
    return {
        "source_version": settings["source_version"],
        "crs": settings["crs"],
        "threshold": settings["threshold"],
        "classification_record": settings.get("classification_record"),
        "assets": [],
    }


def _manifest_for(plan: DatasetPlan, reference: dict[str, object]) -> dict[str, object]:
    return {
        "manifest_version": manifest_state.MANIFEST_VERSION,
        "software": manifest_state._software_provenance(),
        "source_repo": plan.spec.source_repo,
        "target_repo": plan.spec.output_repo,
        "source_revision": plan.source_revision,
        "source_paths": list(plan.source_files),
        "reference": reference,
    }


def _receipt(
    plan: DatasetPlan,
    reference: dict[str, object],
    *,
    no_op: bool,
) -> DatasetReceipt:
    manifest = _manifest_for(plan, reference)
    return DatasetReceipt(
        plan,
        (ShardExpectation("polygons/a.parquet", 1, "schema"),),
        VerificationReceipt(
            plan.spec.output_repo,
            "target-revision",
            {"polygons/a.parquet": 1},
            (),
            manifest,
        ),
        no_op=no_op,
    )


def _prepare_fake_release(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    state: dict[str, object],
) -> None:
    plan = _plan()
    config = tmp_path / "reference.json"
    _write_config(config)
    hub = _FakeHub()
    monkeypatch.setenv("HF_TOKEN", "fake-token")
    monkeypatch.setattr(release_orchestration, "plan_datasets", lambda _api, _names=None: (plan,))
    monkeypatch.setattr(release_orchestration, "resolve_config_data", lambda _config: ())
    monkeypatch.setattr(release_orchestration, "_duplicate_outputs", lambda *_args: None)
    monkeypatch.setattr(
        release_orchestration, "_process_reference_groups", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        manifest_state,
        "_load_existing_manifest",
        lambda *_args: (
            manifest_state._ExistingManifest("target-revision", hub.current_manifest)
            if hub.current_manifest is not None
            else None
        ),
    )
    monkeypatch.setattr(
        manifest_state,
        "_verify_no_op_dataset",
        lambda _api, selected_plan, existing, **_kwargs: _receipt(
            selected_plan, cast(dict[str, object], existing.manifest["reference"]), no_op=True
        ),
    )

    def fake_finalize(_api, selected_plan, *, options):
        reference = dict(options.reference_info)
        for path in (
            "polygons/a.parquet",
            "eunis/manifest.json",
            "README.md",
            "eunis/world-map.svg",
        ):
            hub.upload_file(path_in_repo=path)
        hub.current_manifest = _manifest_for(selected_plan, reference)
        return _receipt(selected_plan, reference, no_op=False)

    monkeypatch.setattr(release_orchestration, "_finalize_plan", fake_finalize)
    state.update(
        hub=hub,
        plan=plan,
        config=config,
        workdir=tmp_path / "release-workdir",
        temp_path=tmp_path,
        cli_dependencies=CliDependencies(api_factory=lambda _endpoint: cast(HubApi, hub)),
    )


@given("a single-shard fake Hub release")
def fake_hub_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, acceptance_state) -> None:
    _prepare_fake_release(monkeypatch, tmp_path, acceptance_state)


@given("a fake Hub release already published with the current reference")
def current_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, acceptance_state) -> None:
    _prepare_fake_release(monkeypatch, tmp_path, acceptance_state)
    hub = cast(_FakeHub, acceptance_state["hub"])
    hub.current_manifest = _manifest_for(
        cast(DatasetPlan, acceptance_state["plan"]),
        _reference_for(cast(Path, acceptance_state["config"])),
    )


@given("a fake Hub release published with an older reference")
def older_release(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, acceptance_state) -> None:
    _prepare_fake_release(monkeypatch, tmp_path, acceptance_state)
    hub = cast(_FakeHub, acceptance_state["hub"])
    hub.current_manifest = _manifest_for(
        cast(DatasetPlan, acceptance_state["plan"]), {"source_version": "EEA-old"}
    )


@when("I preview the release through the CLI")
def preview_release(acceptance_state, capsys) -> None:
    plan = cast(DatasetPlan, acceptance_state["plan"])
    report = runner.DryRunReport(
        (runner.DryRunDataset(plan, True, plan.geometry_paths),),
        False,
        1,
    )
    dependencies = replace(
        cast(CliDependencies, acceptance_state["cli_dependencies"]),
        plan_release=lambda *_args, **_kwargs: report,
    )
    config = cast(Path, acceptance_state["config"])
    tmp_path = cast(Path, acceptance_state["temp_path"])
    code = cli.main(
        ["release", "--dry-run", "--reference-config", str(config), "--workdir", str(tmp_path)],
        dependencies=dependencies,
    )
    acceptance_state["preview_code"] = code
    acceptance_state["preview_output"] = json.loads(capsys.readouterr().out)


@when("I publish the release through the CLI")
def publish_release(acceptance_state, capsys) -> None:
    config = cast(Path, acceptance_state["config"])
    workdir = cast(Path, acceptance_state["workdir"])
    code = cli.main(
        ["release", "--reference-config", str(config), "--workdir", str(workdir)],
        dependencies=cast(CliDependencies, acceptance_state["cli_dependencies"]),
    )
    captured = capsys.readouterr()
    acceptance_state["release_code"] = code
    acceptance_state["release_stderr"] = captured.err
    acceptance_state["release_output"] = json.loads(captured.out) if captured.out else {}


@then("the preview lists the shard and the Hub receives no uploads")
def preview_has_no_side_effects(acceptance_state) -> None:
    output = cast(dict[str, object], acceptance_state["preview_output"])
    assert acceptance_state["preview_code"] == cli.EXIT_OK
    assert _first_dataset(output)["shards_to_upload"] == ["polygons/a.parquet"]
    assert cast(_FakeHub, acceptance_state["hub"]).uploads == []


@then("the Hub receives the shard manifest and card")
def published_artifacts_are_recorded(acceptance_state) -> None:
    hub = cast(_FakeHub, acceptance_state["hub"])
    assert hub.uploads == [
        "polygons/a.parquet",
        "eunis/manifest.json",
        "README.md",
        "eunis/world-map.svg",
    ]
    output = cast(dict[str, object], acceptance_state["release_output"])
    assert _first_dataset(output)["no_op"] is False


@then("the CLI reports the verified revision")
def publish_reports_verification(acceptance_state) -> None:
    output = cast(dict[str, object], acceptance_state["release_output"])
    assert acceptance_state["release_code"] == cli.EXIT_OK
    assert _first_dataset(output)["verified_revision"] == "target-revision"


@then("the CLI reports a no-op and the Hub receives no uploads")
def no_op_has_no_side_effects(acceptance_state) -> None:
    output = cast(dict[str, object], acceptance_state["release_output"])
    assert acceptance_state["release_code"] == cli.EXIT_OK
    assert _first_dataset(output)["no_op"] is True
    assert cast(_FakeHub, acceptance_state["hub"]).uploads == []


@then("the CLI reports a newly verified revision")
def reference_change_reports_verification(acceptance_state) -> None:
    output = cast(dict[str, object], acceptance_state["release_output"])
    assert acceptance_state["release_code"] == cli.EXIT_OK
    assert _first_dataset(output)["no_op"] is False
    assert _first_dataset(output)["verified_revision"] == "target-revision"


@given("a polygon shard with no matching EUNIS geometry")
def polygon_without_overlap(tmp_path: Path, acceptance_state) -> None:
    source = tmp_path / "polygon.parquet"
    pq.write_table(
        pa.table(
            {
                "polygon_id": ["outside"],
                "geometry": [json.dumps(mapping(box(2, 48, 2.1, 48.1)))],
            }
        ),
        source,
    )
    acceptance_state.update(source=source, destination=tmp_path / "labels.parquet")


@when("I enrich the shard through the public API")
def enrich_without_overlap(acceptance_state) -> None:
    rows = enrich_parquet_shard(
        cast(Path, acceptance_state["source"]),
        cast(Path, acceptance_state["destination"]),
        reference=_NoOverlapReference(),
        batch_size=1,
    )
    acceptance_state["rows"] = rows


@then("its EUNIS labels are null")
def null_label_result(acceptance_state) -> None:
    output = pq.read_table(cast(Path, acceptance_state["destination"]))
    assert acceptance_state["rows"] == 1
    assert output["eunis_code"].to_pylist() == [None]
    assert output["eunis_name"].to_pylist() == [None]
    assert output["eunis_overlap_percentage"].to_pylist() == [None]


@given("a release command without a Hub token")
def release_without_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    acceptance_state,
) -> None:
    config = tmp_path / "reference.json"
    _write_config(config)
    monkeypatch.delenv("HF_TOKEN", raising=False)
    acceptance_state.update(
        config=config,
        workdir=tmp_path / "release-workdir",
        cli_dependencies=CliDependencies(
            api_factory=lambda *_args: pytest.fail("Hub must not be contacted")
        ),
    )


@then("the command exits with a usage error without contacting the Hub")
def token_failure_is_fail_fast(acceptance_state) -> None:
    assert acceptance_state["release_code"] == cli.EXIT_USAGE
    assert "HF_TOKEN is not set" in cast(str, acceptance_state["release_stderr"])
