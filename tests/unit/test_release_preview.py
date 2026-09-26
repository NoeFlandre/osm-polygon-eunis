"""Read-only release previews, verification and fail-fast validation."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from huggingface_hub.utils import RepositoryNotFoundError

from osm_polygon_eunis import cli, manifest_state, runner
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.runner import DatasetPlan
from osm_polygon_eunis.sources import DatasetSpec


def _missing() -> RepositoryNotFoundError:
    return RepositoryNotFoundError(
        "missing",
        response=httpx.Response(404, request=httpx.Request("GET", "https://example.test")),
    )


class _ReadOnlyApi:
    """Fake Hub API that only answers reads; any other attribute is a write."""

    def __init__(self, existing: set[str]) -> None:
        self.existing = existing
        self.reads: list[str] = []

    def repo_info(self, repo_id: str, **_kwargs: object) -> SimpleNamespace:
        self.reads.append(repo_id)
        if repo_id not in self.existing:
            raise _missing()
        return SimpleNamespace(sha=f"{repo_id}-rev")

    def list_repo_tree(self, *_args: object, **_kwargs: object) -> list[object]:
        return []

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"dry run must not call Hub method {name}")


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"source_version": "EEA-test", "crs": "EPSG:3035", "threshold": 0}),
        encoding="utf-8",
    )
    return config


def _plans() -> tuple[DatasetPlan, ...]:
    return (
        DatasetPlan(
            DatasetSpec("website", "source-a", "target-a", "polygons/*.parquet"),
            "rev-a",
            ("polygons/a.parquet",),
            ("polygons/a.parquet",),
            (),
        ),
        DatasetPlan(
            DatasetSpec("wikidata", "source-b", "target-b", "polygons/*.parquet"),
            "rev-b",
            ("polygons/b.parquet", "links/b.parquet"),
            ("polygons/b.parquet",),
            ("links/b.parquet",),
        ),
    )


def test_plan_release_makes_no_hub_writes(monkeypatch, tmp_path: Path) -> None:
    api = _ReadOnlyApi({"target-a"})
    monkeypatch.setattr(runner, "plan_datasets", lambda _api, _names=None: _plans())
    monkeypatch.setattr(runner, "resolve_config_data", lambda _config: ())
    monkeypatch.setattr(
        runner, "_duplicate_outputs", lambda *args: pytest.fail("dry run must not duplicate")
    )

    report = runner.plan_release(
        cast(HubApi, api), reference_config=_config(tmp_path), workdir=tmp_path / "run"
    )

    assert report.no_op is False
    assert [item.target_exists for item in report.datasets] == [True, False]
    assert [item.would_duplicate for item in report.datasets] == [False, True]
    assert report.datasets[1].shards_to_upload == ("polygons/b.parquet", "links/b.parquet")


def test_plan_release_reports_no_op_without_uploads(monkeypatch, tmp_path: Path) -> None:
    api = _ReadOnlyApi({"target-a", "target-b"})
    monkeypatch.setattr(runner, "plan_datasets", lambda _api, _names=None: _plans())
    monkeypatch.setattr(runner, "resolve_config_data", lambda _config: ())
    monkeypatch.setattr(runner, "_load_existing_manifests", lambda *args: (None, None))
    monkeypatch.setattr(runner, "_compatible_manifests", lambda *args: True)

    report = runner.plan_release(
        cast(HubApi, api), reference_config=_config(tmp_path), workdir=tmp_path / "run"
    )

    assert report.no_op is True
    assert all(item.shards_to_upload == () for item in report.datasets)


def test_existing_manifest_is_absent_for_missing_target(tmp_path: Path) -> None:
    api = _ReadOnlyApi(set())
    plan = _plans()[0]

    assert (
        manifest_state._load_existing_manifest(
            cast(HubApi, api), plan, tmp_path, cast(StreamClient, object())
        )
        is None
    )


def test_validate_reference_config_rejects_missing_and_invalid(tmp_path: Path) -> None:
    with pytest.raises(runner.ConfigError, match="not found"):
        runner.validate_reference_config(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="not valid JSON"):
        runner.validate_reference_config(broken)
    incomplete = tmp_path / "incomplete.json"
    incomplete.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source_version"):
        runner.validate_reference_config(incomplete)
    runner.validate_reference_config(_config(tmp_path))


def _no_network(monkeypatch) -> None:
    monkeypatch.setattr(cli, "_api", lambda endpoint: pytest.fail("no Hub request expected"))
    monkeypatch.setattr(cli, "run_release", lambda *a, **k: pytest.fail("no release expected"))


@pytest.mark.parametrize(
    ("argv", "token", "message"),
    [
        (["--batch-size", "0"], "token", "positive integer"),
        (["--batch-size", "x"], "token", "positive integer"),
        ([], None, "HF_TOKEN"),
        (["--reference-config", "missing.json"], "token", "not found"),
    ],
)
def test_release_fails_fast_before_any_hub_request(
    monkeypatch, capsys, tmp_path: Path, argv: list[str], token: str | None, message: str
) -> None:
    _no_network(monkeypatch)
    if token is None:
        monkeypatch.delenv("HF_TOKEN", raising=False)
    else:
        monkeypatch.setenv("HF_TOKEN", token)
    monkeypatch.chdir(tmp_path)
    base = ["release", "--reference-config", str(_config(tmp_path))]

    try:
        code = cli.main([*base, *argv])
    except SystemExit as raised:  # argparse usage errors
        code = raised.code

    assert code == 2
    err = capsys.readouterr().err
    assert message in err
    assert "Traceback" not in err


def test_release_dry_run_prints_preview_without_token(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.delenv("HF_TOKEN", raising=False)
    plan = _plans()[0]
    report = runner.DryRunReport(
        (runner.DryRunDataset(plan, False, ("polygons/a.parquet",)),), False, 3
    )
    monkeypatch.setattr(cli, "_api", lambda endpoint: object())
    monkeypatch.setattr(cli, "plan_release", lambda *a, **k: report)
    monkeypatch.setattr(cli, "run_release", lambda *a, **k: pytest.fail("dry run must not run"))

    code = cli.main(["release", "--dry-run", "--reference-config", str(_config(tmp_path))])

    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["dry_run"] is True
    assert payload["reference_assets"] == 3
    assert payload["datasets"][0]["would_duplicate"] is True
    assert payload["datasets"][0]["shards_to_upload"] == ["polygons/a.parquet"]


def test_verify_release_fails_when_target_has_no_manifest(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner, "plan_datasets", lambda _api, _names=None: _plans()[:1])
    api = _ReadOnlyApi(set())

    with pytest.raises(ValueError, match="no EUNIS manifest"):
        runner.verify_release(cast(HubApi, api), workdir=tmp_path)


def test_verify_release_pins_manifest_inputs_and_reports(monkeypatch, tmp_path: Path) -> None:
    from osm_polygon_eunis.publish import VerificationReceipt

    plan = _plans()[0]
    manifest = {"source_revision": "published-rev", "source_paths": ["polygons/a.parquet"]}
    existing = manifest_state._ExistingManifest("target-rev", manifest)
    seen: list[DatasetPlan] = []

    def fake_verify(_api, pinned, _existing, **_kwargs):
        seen.append(pinned)
        verification = VerificationReceipt("target-a", "target-rev", {}, (), manifest)
        return runner.DatasetReceipt(pinned, (), verification, no_op=True)

    monkeypatch.setattr(runner, "plan_datasets", lambda _api, names=None: (plan,))
    monkeypatch.setattr(runner, "_load_existing_manifests", lambda *args: (existing,))
    monkeypatch.setattr(runner, "_verify_no_op_dataset", fake_verify)

    receipt = runner.verify_release(
        cast(HubApi, _ReadOnlyApi({"target-a"})), workdir=tmp_path, datasets=["website"]
    )

    assert seen[0].source_revision == "published-rev"
    assert receipt.datasets[0].no_op is False
    assert receipt.datasets[0].verification.target_revision == "target-rev"


def test_verify_release_rejects_manifest_without_source_pins(monkeypatch, tmp_path: Path) -> None:
    existing = manifest_state._ExistingManifest("rev", {"source_revision": 3})
    monkeypatch.setattr(runner, "plan_datasets", lambda _api, names=None: _plans()[:1])
    monkeypatch.setattr(runner, "_load_existing_manifests", lambda *args: (existing,))

    with pytest.raises(ValueError, match="lacks source revision"):
        runner.verify_release(cast(HubApi, _ReadOnlyApi(set())), workdir=tmp_path)


def test_run_release_rejects_non_positive_workers(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        runner.run_release(
            cast(HubApi, object()),
            reference_config=_config(tmp_path),
            workdir=tmp_path,
            batch_size=1,
            workers=0,
        )
