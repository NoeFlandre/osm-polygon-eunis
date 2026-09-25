import argparse
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import osm_polygon_eunis.cli as cli
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.publish import ShardExpectation, VerificationReceipt
from osm_polygon_eunis.runner import (
    DATASET_NAMES,
    DatasetPlan,
    DatasetReceipt,
    ReleaseReceipt,
    plan_datasets,
    selected_dataset_names,
)
from osm_polygon_eunis.sources import DatasetSpec


def _receipt() -> ReleaseReceipt:
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "source-revision",
        ("polygons/a.parquet",),
        ("polygons/a.parquet",),
        (),
    )
    expectation = ShardExpectation("polygons/a.parquet", 1, "schema")
    verification = VerificationReceipt("target", "target-revision", {}, (), None)
    return ReleaseReceipt(
        (DatasetReceipt(plan, (expectation,), verification),),
        {"source_version": "test"},
    )


def test_plan_command_prints_pinned_layout(monkeypatch, capsys) -> None:
    plan = _receipt().datasets[0].plan
    monkeypatch.setattr(cli, "_api", lambda endpoint: SimpleNamespace(endpoint=endpoint))
    monkeypatch.setattr(cli, "plan_datasets", lambda api, names: (plan,))

    assert cli.main(["plan"]) == 0
    assert '"source_revision": "source-revision"' in capsys.readouterr().out


def test_release_command_prints_verified_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_api", lambda endpoint: SimpleNamespace(endpoint=endpoint))
    monkeypatch.setattr(cli, "run_release", lambda *args, **kwargs: _receipt())
    monkeypatch.setenv("HF_TOKEN", "token")
    (tmp_path / "reference.json").write_text(
        '{"source_version": "v", "crs": "EPSG:3035", "threshold": 0}', encoding="utf-8"
    )

    assert (
        cli.main(
            [
                "release",
                "--reference-config",
                str(tmp_path / "reference.json"),
                "--workdir",
                str(tmp_path / "run"),
                "--batch-size",
                "2",
            ]
        )
        == 0
    )
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out


def test_every_option_has_help_and_top_level_help_shows_examples(capsys) -> None:
    parser = cli._parser()
    subparsers = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    for name, subparser in subparsers.choices.items():
        for action in subparser._actions:
            assert action.help, f"{name} {action.option_strings} has no help"
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert "examples:" in out
    assert "HF_TOKEN" in out
    assert "osm-polygon-eunis verify" in out


@pytest.mark.parametrize(
    "argv",
    [
        ["release", "--workers", "0"],
        ["release", "--dataset", "unknown"],
        ["verify", "--dataset", "unknown"],
        ["bogus"],
        [],
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    assert raised.value.code == 2
    assert "usage:" in capsys.readouterr().err


def test_release_passes_dataset_selection_and_workers(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_release(*_args, **kwargs):
        seen.update(kwargs)
        return _receipt()

    config = tmp_path / "reference.json"
    config.write_text('{"source_version": "v", "crs": "EPSG:3035", "threshold": 0}')
    monkeypatch.setenv("HF_TOKEN", "token")
    monkeypatch.setattr(cli, "_api", lambda endpoint: object())
    monkeypatch.setattr(cli, "run_release", fake_release)

    argv = ["release", "--reference-config", str(config), "--dataset", "wikidata"]
    assert cli.main([*argv, "--dataset", "website", "--workers", "3"]) == 0
    assert seen["datasets"] == ["wikidata", "website"]
    assert seen["workers"] == 3


def test_plan_datasets_touches_only_selected_dataset() -> None:
    touched: list[str] = []

    class Api:
        def repo_info(self, repo_id: str, **_kwargs):
            touched.append(repo_id)
            return SimpleNamespace(sha="rev")

        def list_repo_tree(self, repo_id: str, **_kwargs):
            touched.append(repo_id)
            paths = ("polygons/a.parquet", "data/a.parquet", "polygon_document_links/a.parquet")
            return [SimpleNamespace(path=path, blob_id="b") for path in paths]

    plans = plan_datasets(cast(HubApi, Api()), ["website"])

    assert [plan.spec.name for plan in plans] == ["website"]
    assert set(touched) == {"NoeFlandre/osm-polygon-website-tag"}
    assert [p.spec.name for p in plan_datasets(cast(HubApi, Api()), None)] == list(DATASET_NAMES)


def test_selected_dataset_names_keeps_release_order_and_rejects_unknown() -> None:
    assert selected_dataset_names(["description", "website"]) == ("website", "description")
    with pytest.raises(ValueError, match="unknown dataset"):
        selected_dataset_names(["nope"])


def test_verify_command_prints_receipt(monkeypatch, capsys, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_verify(*_args, **kwargs):
        seen.update(kwargs)
        return _receipt()

    monkeypatch.setattr(cli, "_api", lambda endpoint: object())
    monkeypatch.setattr(cli, "verify_release", fake_verify)

    assert cli.main(["verify", "--dataset", "website", "--workdir", str(tmp_path)]) == 0
    assert seen["datasets"] == ["website"]
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out
