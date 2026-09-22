from pathlib import Path
from types import SimpleNamespace

import osm_polygon_eunis.cli as cli
from osm_polygon_eunis.publish import ShardExpectation, VerificationReceipt
from osm_polygon_eunis.runner import DatasetPlan, DatasetReceipt, ReleaseReceipt
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
    monkeypatch.setattr(cli, "plan_datasets", lambda api: (plan,))

    assert cli.main(["plan"]) == 0
    assert '"source_revision": "source-revision"' in capsys.readouterr().out


def test_release_command_prints_verified_summary(monkeypatch, capsys, tmp_path: Path) -> None:
    monkeypatch.setattr(cli, "_api", lambda endpoint: SimpleNamespace(endpoint=endpoint))
    monkeypatch.setattr(cli, "run_release", lambda *args, **kwargs: _receipt())

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
                "--execution",
                "local",
            ]
        )
        == 0
    )
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out


def test_release_defaults_to_description_on_grid5000(monkeypatch, tmp_path: Path) -> None:
    calls: dict[str, object] = {}

    monkeypatch.setattr(cli, "_api", lambda endpoint: SimpleNamespace(endpoint=endpoint))

    def fake_release(*args, **kwargs):
        calls.update(kwargs)
        return _receipt()

    monkeypatch.setattr(cli, "run_release", fake_release)

    assert (
        cli.main(
            [
                "release",
                "--reference-config",
                str(tmp_path / "reference.json"),
                "--workdir",
                str(tmp_path / "run"),
            ]
        )
        == 0
    )
    assert calls["dataset_names"] == ("description",)
    assert calls["execution"] == "grid5000"


def test_grid5000_status_calls_only_remote_oarstat(monkeypatch, capsys) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return "job status"

    monkeypatch.setattr(cli, "run_command", fake_runner)

    assert cli.main(["grid5000", "status", "--frontend", "flille", "--job-id", "123456"]) == 0
    assert calls == [("ssh", "flille", "oarstat", "-j", "123456")]
    assert "job status" in capsys.readouterr().out


def test_grid5000_cancel_calls_only_remote_oardel(monkeypatch, capsys) -> None:
    calls: list[tuple[str, ...]] = []

    def fake_runner(command: tuple[str, ...]) -> str:
        calls.append(command)
        return "cancelled"

    monkeypatch.setattr(cli, "run_command", fake_runner)

    assert cli.main(["grid5000", "cancel", "--frontend", "flille", "--job-id", "123456"]) == 0
    assert calls == [("ssh", "flille", "oardel", "123456")]
    assert "cancelled" in capsys.readouterr().out


def test_grid5000_submit_prints_job_dataset_and_source_without_token(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    from osm_polygon_eunis.grid5000 import Grid5000Job, Grid5000Submission

    monkeypatch.setattr(cli, "resolve_source_revision", lambda *args, **kwargs: "abc123")
    monkeypatch.setattr(
        cli,
        "submit_grid5000",
        lambda *args, **kwargs: Grid5000Submission(
            Grid5000Job(
                "123456",
                "2026-09-22T10:00:00+00:00",
                args[0],
                "abc123",
            ),
            "description",
            "abc123",
            (("ssh", "flille", "oarsub"),),
        ),
    )

    assert (
        cli.main(
            [
                "grid5000",
                "submit",
                "--frontend",
                "flille",
                "--persistent-root",
                "/home/u/eunis",
                "--source-root",
                str(tmp_path),
            ]
        )
        == 0
    )
    output = capsys.readouterr().out
    assert '"dataset": "description"' in output
    assert '"job_id": "123456"' in output
    assert '"source_revision": "abc123"' in output
    assert "HF_TOKEN" not in output
