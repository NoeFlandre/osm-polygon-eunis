import argparse
import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest

from osm_polygon_eunis import cli
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.cli import CliDependencies
from osm_polygon_eunis.grid5000 import Grid5000Config
from osm_polygon_eunis.options import BatchLimits, ReleaseOptions
from osm_polygon_eunis.publish import ShardExpectation, VerificationError, VerificationReceipt
from osm_polygon_eunis.release_orchestration import ConfigError
from osm_polygon_eunis.release_plan import (
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


def _dependencies(**overrides: object) -> CliDependencies:
    return replace(CliDependencies(), **overrides)


def test_report_start_emits_arguments_only_when_verbose() -> None:
    events: list[Mapping[str, object]] = []
    verbose_args = argparse.Namespace(verbose=True, command="plan", marker="value")
    quiet_args = argparse.Namespace(verbose=False, command="plan", marker="value")

    cli._report_start(verbose_args, events.append)
    cli._report_start(quiet_args, events.append)

    assert events == [{"event": "start", "command": "plan", "arguments": vars(verbose_args)}]


def test_default_reference_config_resolves_outside_the_repository(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    args = cli._parser().parse_args(["release", "--dry-run"])
    cli._resolve_defaults(args)

    assert args.reference_config.is_file()
    assert args.reference_config.name == "eea-2021-reference.json"


def test_workdir_defaults_to_environment_path(monkeypatch, tmp_path: Path) -> None:
    workdir = tmp_path / "persistent-run"
    monkeypatch.setenv("OSM_EUNIS_WORKDIR", str(workdir))

    args = cli._parser().parse_args(["verify"])
    cli._resolve_defaults(args)

    assert args.workdir == workdir


def test_plan_command_prints_pinned_layout(capsys) -> None:
    plan = _receipt().datasets[0].plan
    dependencies = _dependencies(
        api_factory=lambda endpoint: cast(
            HubApi, SimpleNamespace(endpoint=endpoint, token=None, upload_file=None)
        ),
        plan_datasets=lambda _api, _names: (plan,),
    )

    assert cli.main(["plan"], dependencies=dependencies) == 0
    assert '"source_revision": "source-revision"' in capsys.readouterr().out


def test_release_command_prints_verified_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")
    dependencies = _dependencies(
        api_factory=lambda endpoint: cast(
            HubApi, SimpleNamespace(endpoint=endpoint, token=None, upload_file=None)
        ),
        run_release=lambda *_args, **_kwargs: _receipt(),
    )
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
            ],
            dependencies=dependencies,
        )
        == 0
    )
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out


def test_release_writes_full_receipt_atomically(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setenv("HF_TOKEN", "token")
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()),
        run_release=lambda *_args, **_kwargs: _receipt(),
    )
    config = tmp_path / "reference.json"
    config.write_text('{"source_version":"v","crs":"EPSG:3035","threshold":0}')
    receipt_path = tmp_path / "receipts" / "run.json"

    assert (
        cli.main(
            [
                "release",
                "--reference-config",
                str(config),
                "--receipt",
                str(receipt_path),
            ],
            dependencies=dependencies,
        )
        == cli.EXIT_OK
    )

    stdout_payload = json.loads(capsys.readouterr().out)
    receipt_payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert stdout_payload == receipt_payload
    assert receipt_path.read_bytes() == (
        json.dumps(stdout_payload, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    assert receipt_payload["status"] == "verified"
    dataset = receipt_payload["datasets"][0]
    assert dataset["source_revision"] == "source-revision"
    assert dataset["source_file_count"] == 1
    assert len(dataset["manifest_sha256"]) == 64
    assert not receipt_path.with_name(".run.json.tmp").exists()


def test_grid5000_submit_accepts_an_explicit_site_and_defaults_to_a_short_job(
    capsys, tmp_path: Path
) -> None:
    calls: dict[str, object] = {}

    def fake_submit(config, local_root, **kwargs):
        calls["config"] = config
        calls["local_root"] = local_root
        calls.update(kwargs)
        return SimpleNamespace(
            job=None,
            datasets=("website", "wikidata", "description"),
            source_revision="abc123",
            commands=(),
        )

    state = tmp_path / "grid5000-state.json"
    dependencies = _dependencies(
        resolve_source_revision=lambda _root, **_kwargs: "abc123",
        submit_grid5000=fake_submit,
    )

    assert (
        cli.main(
            [
                "grid5000",
                "submit",
                "--site",
                "grenoble",
                "--frontend",
                "grenoble",
                "--cluster",
                "dahu",
                "--persistent-root",
                "/home/nflandre/osm-polygon-eunis",
                "--exclude-site",
                "bordeaux",
                "--state",
                str(state),
                "--dry-run",
            ],
            dependencies=dependencies,
        )
        == 0
    )

    config = cast(Grid5000Config, calls["config"])
    assert config.site == "grenoble"
    assert config.cluster == "dahu"
    assert config.walltime == "1:00:00"
    assert config.excluded_sites == ("bordeaux",)
    assert calls["state_path"] == state
    assert calls["dry_run"] is True
    assert json.loads(capsys.readouterr().out)["source_revision"] == "abc123"


def test_top_level_help_shows_examples(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--help"])
    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert "examples:" in out
    assert "HF_TOKEN" in out
    assert "osm-polygon-eunis verify" in out


@pytest.mark.parametrize(
    ("argv", "options"),
    [
        (["plan"], ["--dataset", "--endpoint"]),
        (
            ["release"],
            ["--workers", "--max-intersection-errors", "--dataset", "--receipt", "--dry-run"],
        ),
        (["verify"], ["--dataset", "--endpoint", "--workdir"]),
        (["analyze-run"], ["--log", "--sidecars", "--slowest-shards"]),
        (["grid5000"], ["submit"]),
        (["grid5000", "submit"], ["--site", "--cluster", "--walltime"]),
    ],
)
def test_subcommand_help_lists_options(argv: list[str], options: list[str], capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main([*argv, "--help"])
    out = capsys.readouterr().out
    assert raised.value.code == 0
    assert out.startswith(f"usage: osm-polygon-eunis {' '.join(argv)}")
    for option in options:
        assert option in out, f"`{' '.join(argv)} --help` does not document {option}"


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["release", "--workers", "0"], "argument --workers: expected a positive integer"),
        (
            ["release", "--max-intersection-errors", "-1"],
            "argument --max-intersection-errors: expected a non-negative integer, got '-1'",
        ),
        (
            ["release", "--max-intersection-errors", "x"],
            "argument --max-intersection-errors: expected a non-negative integer, got 'x'",
        ),
        (["release", "--dataset", "unknown"], "argument --dataset: invalid choice: 'unknown'"),
        (["verify", "--dataset", "unknown"], "argument --dataset: invalid choice: 'unknown'"),
        (["bogus"], "argument command: invalid choice: 'bogus'"),
        ([], "the following arguments are required: command"),
    ],
)
def test_invalid_arguments_exit_with_usage_error(argv: list[str], message: str, capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(argv)
    err = capsys.readouterr().err
    assert raised.value.code == cli.EXIT_USAGE
    assert err.startswith("usage: osm-polygon-eunis")
    assert f"error: {message}" in err


def test_release_passes_dataset_selection_and_workers(monkeypatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_release(_api, options: ReleaseOptions):
        seen["options"] = options
        return _receipt()

    config = tmp_path / "reference.json"
    config.write_text('{"source_version": "v", "crs": "EPSG:3035", "threshold": 0}')
    monkeypatch.setenv("HF_TOKEN", "token")
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()), run_release=fake_release
    )

    argv = ["release", "--reference-config", str(config), "--dataset", "wikidata"]
    assert (
        cli.main([*argv, "--dataset", "website", "--workers", "3"], dependencies=dependencies) == 0
    )
    options = cast(ReleaseOptions, seen["options"])
    assert options.datasets == ["wikidata", "website"]
    assert options.limits == BatchLimits(workers=3)
    assert options.max_intersection_errors is None
    assert cli.main([*argv, "--max-intersection-errors", "0"], dependencies=dependencies) == 0
    assert cast(ReleaseOptions, seen["options"]).max_intersection_errors == 0


class _TreeApi:
    def __init__(self) -> None:
        self.touched: list[str] = []

    def repo_info(self, repo_id: str, **_kwargs):
        self.touched.append(repo_id)
        return SimpleNamespace(sha="rev")

    def list_repo_tree(self, repo_id: str, **_kwargs):
        self.touched.append(repo_id)
        paths = ("polygons/a.parquet", "data/a.parquet", "polygon_document_links/a.parquet")
        return [SimpleNamespace(path=path, blob_id="b") for path in paths]


def test_plan_datasets_touches_only_selected_dataset() -> None:
    api = _TreeApi()

    plans = plan_datasets(cast(HubApi, api), ["website"])

    assert [plan.spec.name for plan in plans] == ["website"]
    assert set(api.touched) == {"NoeFlandre/osm-polygon-website-tag"}


def test_plan_datasets_without_selection_plans_every_dataset() -> None:
    plans = plan_datasets(cast(HubApi, _TreeApi()), None)

    assert [plan.spec.name for plan in plans] == list(DATASET_NAMES)


def test_selected_dataset_names_keeps_release_order_and_rejects_unknown() -> None:
    assert selected_dataset_names(["description", "website"]) == ("website", "description")
    with pytest.raises(ValueError, match="unknown dataset"):
        selected_dataset_names(["nope"])


def test_verify_command_prints_receipt(capsys, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_verify(*_args, **kwargs):
        seen.update(kwargs)
        return _receipt()

    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()), verify_release=fake_verify
    )

    assert (
        cli.main(
            ["verify", "--dataset", "website", "--workdir", str(tmp_path)],
            dependencies=dependencies,
        )
        == 0
    )
    assert seen["datasets"] == ["website"]
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out


def _raise(error: Exception):
    def fail(*_args, **_kwargs):
        raise error

    return fail


def _hub_error() -> Exception:
    from huggingface_hub.errors import HfHubHTTPError
    from huggingface_hub.utils import httpx as hub_httpx

    request = hub_httpx.Request("GET", "https://example.test")
    return HfHubHTTPError("401 unauthorized", response=hub_httpx.Response(401, request=request))


@pytest.mark.parametrize(
    ("error", "code"),
    [
        (RuntimeError("boom"), cli.EXIT_ERROR),
        (ConfigError("bad config"), cli.EXIT_USAGE),
        (httpx.ConnectError("offline"), cli.EXIT_REMOTE),
        (_hub_error(), cli.EXIT_REMOTE),
        (VerificationError("remote Parquet mismatch: x"), cli.EXIT_VERIFICATION),
    ],
)
def test_errors_map_to_documented_exit_codes_without_traceback(
    capsys, error: Exception, code: int
) -> None:
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()), verify_release=_raise(error)
    )

    assert cli.main(["verify"], dependencies=dependencies) == code
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert "Traceback" not in err


def test_debug_reraises_with_traceback() -> None:
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()),
        verify_release=_raise(VerificationError("mismatch")),
    )

    with pytest.raises(VerificationError):
        cli.main(["verify", "--debug"], dependencies=dependencies)


def test_verbose_failure_logs_traceback_at_debug_and_keeps_short_error(caplog, capsys) -> None:
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()),
        verify_release=_raise(RuntimeError("boom")),
    )

    with caplog.at_level(logging.DEBUG, logger=cli.__name__):
        assert cli.main(["verify", "-v"], dependencies=dependencies) == cli.EXIT_ERROR

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" in captured.err
    assert captured.err.rstrip().endswith("error: boom")
    assert any(
        record.levelno == logging.DEBUG
        and record.exc_info is not None
        and "command failed" in record.getMessage()
        for record in caplog.records
    )


def test_version_prints_package_version(capsys) -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["--version"])
    assert raised.value.code == 0
    assert capsys.readouterr().out.startswith("osm-polygon-eunis ")


def _release_with_progress(monkeypatch, tmp_path: Path, *flags: str) -> None:
    def fake_release(_api, options: ReleaseOptions):
        progress = options.progress
        if progress is not None:
            progress({"event": "shards_uploaded", "dataset": "website"})
        return _receipt()

    config = tmp_path / "reference.json"
    config.write_text('{"source_version": "v", "crs": "EPSG:3035", "threshold": 0}')
    monkeypatch.setenv("HF_TOKEN", "token")
    dependencies = _dependencies(
        api_factory=lambda _endpoint: cast(HubApi, object()),
        run_release=fake_release,
    )
    assert (
        cli.main(
            ["release", "--reference-config", str(config), *flags],
            dependencies=dependencies,
        )
        == 0
    )


def test_progress_goes_to_stderr_and_stdout_is_only_the_receipt(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    _release_with_progress(monkeypatch, tmp_path)
    captured = capsys.readouterr()
    assert json.loads(captured.out)["datasets"][0]["dataset"] == "website"
    assert '"shards_uploaded"' in captured.err


def test_quiet_release_prints_only_the_receipt(monkeypatch, capsys, tmp_path: Path) -> None:
    _release_with_progress(monkeypatch, tmp_path, "-q")
    captured = capsys.readouterr()
    assert json.loads(captured.out)["datasets"]
    assert captured.err == ""


def test_verbose_adds_start_record(monkeypatch, capsys, tmp_path: Path) -> None:
    _release_with_progress(monkeypatch, tmp_path, "-v")
    err_lines = capsys.readouterr().err.splitlines()
    assert json.loads(err_lines[0])["event"] == "start"


@pytest.mark.parametrize(
    ("flags", "expected_level"),
    [([], logging.INFO), (["-v"], logging.DEBUG), (["-q"], logging.WARNING)],
)
def test_progress_handler_maps_cli_verbosity_to_logging_level(
    monkeypatch, caplog, capsys, tmp_path: Path, flags: list[str], expected_level: int
) -> None:
    with caplog.at_level(logging.DEBUG, logger=cli.__name__):
        _release_with_progress(monkeypatch, tmp_path, *flags)

    progress_records = [
        record
        for record in caplog.records
        if record.name == cli.__name__ and hasattr(record, "progress_event")
    ]
    if flags == ["-q"]:
        assert progress_records == []
        assert capsys.readouterr().err == ""
        return

    assert any(record.levelno == logging.INFO for record in progress_records)
    if expected_level == logging.DEBUG:
        assert any(
            record.levelno == logging.DEBUG and record.progress_event["event"] == "start"
            for record in progress_records
        )
    assert json.loads(capsys.readouterr().err.splitlines()[-1]) == {
        "dataset": "website",
        "event": "shards_uploaded",
    }


def test_quiet_and_verbose_are_mutually_exclusive() -> None:
    with pytest.raises(SystemExit) as raised:
        cli.main(["release", "-q", "-v"])
    assert raised.value.code == 2
