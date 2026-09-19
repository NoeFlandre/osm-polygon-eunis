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
            ]
        )
        == 0
    )
    assert '"verified_revision": "target-revision"' in capsys.readouterr().out
