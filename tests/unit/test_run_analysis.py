"""Offline summaries for geometry worker logs and sidecar checkpoints."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from osm_polygon_eunis import run_analysis
from osm_polygon_eunis.cli import EXIT_OK, EXIT_USAGE, main


def _write_log(path: Path, records: list[object], *, extra: str = "") -> None:
    path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records) + extra,
        encoding="utf-8",
    )


def _write_checkpoint(
    root: Path,
    dataset: str,
    shard_path: str,
    signature: str,
    batches: list[object],
) -> None:
    marker = root / dataset / f"{shard_path.replace('/', '__')}.labels.parquet.done"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(
        json.dumps({"signature": signature, "batches": batches}),
        encoding="utf-8",
    )


def _plan() -> dict[str, object]:
    return {
        "event": "geometry_run_plan",
        "dataset": "website",
        "reference_batch_ids": [0, 2],
        "checkpoint_signatures": {
            "polygons/a.parquet": "signature-a",
            "polygons/b.parquet": "signature-b",
        },
    }


def _batch(path: str, reference_batch: int, seconds: float) -> dict[str, object]:
    return {
        "event": "geometry_batch_done",
        "dataset": "website",
        "path": path,
        "reference_batch": reference_batch,
        "seconds": seconds,
    }


def test_summary_counts_repeats_but_uses_unique_checkpoint_batches(
    tmp_path: Path,
) -> None:
    first_log = tmp_path / "job-1.log"
    retry_log = tmp_path / "job-2.log"
    sidecars = tmp_path / "sidecars"
    _write_log(
        first_log,
        [_plan(), _batch("polygons/a.parquet", 0, 2.0), _batch("polygons/a.parquet", 2, 3.0)],
    )
    _write_log(
        retry_log,
        [
            _plan(),
            _batch("polygons/a.parquet", 0, 1.0),
            _batch("polygons/b.parquet", 0, 4.0),
        ],
    )
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "signature-a", [0, 2])
    _write_checkpoint(sidecars, "website", "polygons/b.parquet", "signature-b", [0])

    report = run_analysis.summarize_run([first_log, retry_log], sidecars)

    dataset = report["datasets"][0]
    assert dataset == {
        "dataset": "website",
        "event_count": 4,
        "unique_event_count": 3,
        "repeated_event_count": 1,
        "worker_seconds": 10.0,
        "throughput_batches_per_worker_hour": 1440.0,
        "checkpoint_batches_observed": 3,
        "progress": {
            "geometry_shards_completed": 1,
            "geometry_shards_total": 2,
            "reference_batches_completed": 3,
            "reference_batches_total": 4,
            "reference_batches_remaining": 1,
            "percent": 75.0,
        },
        "slowest_shards": [
            {"path": "polygons/a.parquet", "event_count": 3, "worker_seconds": 6.0},
            {"path": "polygons/b.parquet", "event_count": 1, "worker_seconds": 4.0},
        ],
    }
    assert report["inputs"] == {
        "log_files": 2,
        "malformed_json_lines": 0,
        "malformed_records": 0,
        "repeated_plans": 1,
        "malformed_checkpoints": 0,
        "stale_checkpoints": 0,
        "missing_metadata_datasets": [],
    }


def test_summary_skips_incomplete_log_records_and_malformed_checkpoints(
    tmp_path: Path,
) -> None:
    log = tmp_path / "job.log"
    sidecars = tmp_path / "sidecars"
    _write_log(
        log,
        [
            _plan(),
            _batch("polygons/a.parquet", 0, 2.0),
            {"event": "geometry_batch_done"},
            {
                "event": "geometry_batch_done",
                "dataset": "website",
                "path": "polygons/b.parquet",
                "reference_batch": 0,
                "seconds": 10**1000,
            },
        ],
        extra="not-json\n",
    )
    malformed = sidecars / "website" / "polygons__a.parquet.labels.parquet.done"
    malformed.parent.mkdir(parents=True)
    malformed.write_text('{"signature":"signature-a","batches":[0', encoding="utf-8")

    report = run_analysis.summarize_run([log], sidecars)

    dataset = report["datasets"][0]
    assert dataset["progress"] == {
        "geometry_shards_completed": 0,
        "geometry_shards_total": 2,
        "reference_batches_completed": 0,
        "reference_batches_total": 4,
        "reference_batches_remaining": 4,
        "percent": 0.0,
    }
    assert report["inputs"] == {
        "log_files": 1,
        "malformed_json_lines": 1,
        "malformed_records": 2,
        "repeated_plans": 0,
        "malformed_checkpoints": 1,
        "stale_checkpoints": 0,
        "missing_metadata_datasets": [],
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"dataset": "unknown"},
        {"path": ""},
        {"reference_batch": True},
        {"reference_batch": -1},
        {"seconds": True},
        {"seconds": "1"},
        {"seconds": -1.0},
        {"seconds": float("inf")},
        {"seconds": 10**1000},
    ],
)
def test_summary_ignores_timing_records_with_invalid_fields(
    tmp_path: Path, changes: dict[str, object]
) -> None:
    log = tmp_path / "invalid-events.log"
    invalid_event = _batch("polygons/a.parquet", 0, 1.0) | changes
    _write_log(log, [_plan(), invalid_event, {"event": "other"}], extra="\n")

    report = run_analysis.summarize_run([log], tmp_path / "missing-sidecars")

    assert report["datasets"][0]["event_count"] == 0
    assert report["inputs"]["malformed_records"] == 1
    assert report["inputs"]["malformed_json_lines"] == 0


@pytest.mark.parametrize(
    ("plan", "missing_datasets"),
    [
        ({**_plan(), "reference_batch_ids": "0,2"}, ["website"]),
        ({**_plan(), "reference_batch_ids": [0, True]}, ["website"]),
        ({**_plan(), "reference_batch_ids": [0, 0]}, ["website"]),
        ({**_plan(), "reference_batch_ids": [-1]}, ["website"]),
        ({**_plan(), "checkpoint_signatures": []}, ["website"]),
        ({**_plan(), "checkpoint_signatures": {"": "signature"}}, ["website"]),
        ({**_plan(), "checkpoint_signatures": {"polygons/a.parquet": ""}}, ["website"]),
        ({**_plan(), "dataset": "unknown"}, []),
    ],
)
def test_summary_marks_incomplete_plan_metadata_as_unknown(
    tmp_path: Path,
    plan: dict[str, object],
    missing_datasets: list[str],
) -> None:
    log = tmp_path / "invalid-plan.log"
    _write_log(log, [plan])

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["inputs"]["malformed_records"] == 1
    assert report["inputs"]["missing_metadata_datasets"] == missing_datasets
    if missing_datasets:
        assert report["datasets"][0]["progress"] is None
    else:
        assert report["datasets"] == []


def test_conflicting_run_plans_disable_checkpoint_progress(tmp_path: Path) -> None:
    log = tmp_path / "conflicting-plans.log"
    _write_log(
        log,
        [_plan(), {**_plan(), "reference_batch_ids": [0]}],
    )

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["datasets"][0]["progress"] is None
    assert report["inputs"]["malformed_records"] == 1
    assert report["inputs"]["missing_metadata_datasets"] == ["website"]


def test_incomplete_later_plan_invalidates_checkpoint_progress(tmp_path: Path) -> None:
    log = tmp_path / "valid-then-incomplete-plan.log"
    _write_log(
        log,
        [_plan(), {"event": "geometry_run_plan", "dataset": "website"}],
    )

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["datasets"][0]["progress"] is None
    assert report["inputs"]["malformed_records"] == 1
    assert report["inputs"]["missing_metadata_datasets"] == ["website"]


def test_summary_excludes_timing_events_outside_run_plan(tmp_path: Path) -> None:
    log = tmp_path / "out-of-plan-events.log"
    _write_log(
        log,
        [
            _plan(),
            _batch("polygons/a.parquet", 0, 2.0),
            _batch("polygons/undeclared.parquet", 0, 90.0),
            _batch("polygons/a.parquet", 1, 60.0),
        ],
    )

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["datasets"][0]["event_count"] == 1
    assert report["datasets"][0]["worker_seconds"] == 2.0
    assert report["datasets"][0]["throughput_batches_per_worker_hour"] == 1800.0
    assert report["datasets"][0]["slowest_shards"] == [
        {"path": "polygons/a.parquet", "event_count": 1, "worker_seconds": 2.0}
    ]
    assert report["inputs"]["malformed_records"] == 2


def test_summary_rejects_timing_events_that_overflow_worker_seconds(tmp_path: Path) -> None:
    log = tmp_path / "overflowing-events.log"
    _write_log(
        log,
        [
            _plan(),
            _batch("polygons/a.parquet", 0, 1e308),
            _batch("polygons/b.parquet", 0, 1e308),
        ],
    )

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["datasets"][0]["event_count"] == 0
    assert report["datasets"][0]["worker_seconds"] == 0.0
    assert report["datasets"][0]["throughput_batches_per_worker_hour"] is None
    assert report["datasets"][0]["slowest_shards"] == []
    assert report["inputs"]["malformed_records"] == 2


def test_summary_marks_checkpoint_batch_ids_outside_plan_as_malformed(
    tmp_path: Path,
) -> None:
    log = tmp_path / "extra-checkpoint-batches.log"
    sidecars = tmp_path / "sidecars"
    _write_log(log, [_plan()])
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "signature-a", [0, 999])

    report = run_analysis.summarize_run([log], sidecars)

    progress = report["datasets"][0]["progress"]
    assert progress is not None
    assert progress["reference_batches_completed"] == 1
    assert report["inputs"]["malformed_checkpoints"] == 1


def test_summary_counts_a_checkpoint_with_multiple_defects_once(tmp_path: Path) -> None:
    log = tmp_path / "multi-defect-checkpoint.log"
    sidecars = tmp_path / "sidecars"
    _write_log(log, [_plan()])
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "signature-a", [0, 0, 999])

    report = run_analysis.summarize_run([log], sidecars)

    assert report["inputs"]["malformed_checkpoints"] == 1
    progress = report["datasets"][0]["progress"]
    assert progress is not None
    assert progress["reference_batches_completed"] == 1


def test_summary_omits_unrepresentable_throughput_rate(tmp_path: Path) -> None:
    log = tmp_path / "tiny-duration.log"
    _write_log(log, [_plan(), _batch("polygons/a.parquet", 0, 1e-320)])

    report = run_analysis.summarize_run([log], tmp_path / "sidecars")

    assert report["datasets"][0]["throughput_batches_per_worker_hour"] is None
    assert "Infinity" not in json.dumps(report, allow_nan=False)


def test_summary_reports_unknown_progress_when_old_logs_lack_plan_metadata(
    tmp_path: Path,
) -> None:
    log = tmp_path / "old-job.log"
    sidecars = tmp_path / "sidecars"
    _write_log(log, [_batch("polygons/a.parquet", 0, 2.0)])
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "old-signature", [0])

    report = run_analysis.summarize_run([log], sidecars)

    dataset = report["datasets"][0]
    assert dataset["progress"] is None
    assert dataset["checkpoint_batches_observed"] == 1
    assert report["inputs"]["missing_metadata_datasets"] == ["website"]


def test_summary_reports_zero_checkpoint_batches_when_directory_is_missing(
    tmp_path: Path,
) -> None:
    log = tmp_path / "old-job.log"
    _write_log(log, [_batch("polygons/a.parquet", 0, 2.0)])

    report = run_analysis.summarize_run([log], tmp_path / "missing-sidecars")

    assert report["datasets"][0]["progress"] is None
    assert report["datasets"][0]["checkpoint_batches_observed"] == 0


@pytest.mark.parametrize(
    ("payload", "malformed", "stale", "completed"),
    [
        ([], 1, 0, 0),
        ({"batches": [0]}, 1, 0, 0),
        ({"signature": "signature-a", "batches": "0"}, 1, 0, 0),
        ({"signature": "old-signature", "batches": [0]}, 0, 1, 0),
        ({"signature": "signature-a", "batches": [0, True, "2"]}, 1, 0, 1),
    ],
)
def test_summary_validates_checkpoint_metadata(
    tmp_path: Path,
    payload: object,
    malformed: int,
    stale: int,
    completed: int,
) -> None:
    log = tmp_path / "job.log"
    sidecars = tmp_path / "sidecars"
    _write_log(log, [_plan()])
    marker = sidecars / "website" / "polygons__a.parquet.labels.parquet.done"
    marker.parent.mkdir(parents=True)
    marker.write_text(json.dumps(payload), encoding="utf-8")

    report = run_analysis.summarize_run([log], sidecars)

    progress = report["datasets"][0]["progress"]
    assert progress is not None
    assert progress["reference_batches_completed"] == completed
    assert report["inputs"]["malformed_checkpoints"] == malformed
    assert report["inputs"]["stale_checkpoints"] == stale


def test_summary_rejects_a_missing_log_file(tmp_path: Path) -> None:
    missing_log = tmp_path / "missing.log"

    with pytest.raises(run_analysis.RunAnalysisError, match="cannot read log"):
        run_analysis.summarize_run([missing_log], tmp_path / "sidecars")


def test_summary_requires_logs_and_a_positive_slowest_limit(tmp_path: Path) -> None:
    log = tmp_path / "job.log"
    _write_log(log, [])

    with pytest.raises(run_analysis.RunAnalysisError, match="at least one --log"):
        run_analysis.summarize_run([], tmp_path / "sidecars")
    with pytest.raises(run_analysis.RunAnalysisError, match="--slowest"):
        run_analysis.summarize_run([log], tmp_path / "sidecars", slowest_limit=0)


def test_cli_analyze_run_prints_offline_json_and_uses_input_exit_code(
    capsys, tmp_path: Path
) -> None:
    log = tmp_path / "job.log"
    sidecars = tmp_path / "sidecars"
    _write_log(log, [_plan(), _batch("polygons/a.parquet", 0, 2.0)])
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "signature-a", [0])

    assert main(["analyze-run", "--log", str(log), "--sidecars", str(sidecars)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)
    assert payload["datasets"][0]["progress"]["reference_batches_remaining"] == 3

    assert main(["analyze-run", "--log", str(tmp_path / "missing.log")]) == EXIT_USAGE
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "Traceback" not in captured.err


def test_cli_analyze_run_defaults_to_sidecars_under_configured_workdir(
    capsys,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    workdir = tmp_path / "persistent-run"
    sidecars = workdir / "sidecars"
    log = tmp_path / "job.log"
    _write_log(log, [_plan()])
    _write_checkpoint(sidecars, "website", "polygons/a.parquet", "signature-a", [0])
    monkeypatch.setenv("OSM_EUNIS_WORKDIR", str(workdir))
    monkeypatch.delenv("EUNIS_SIDECAR_DIR", raising=False)

    assert main(["analyze-run", "--log", str(log)]) == EXIT_OK
    payload = json.loads(capsys.readouterr().out)

    assert payload["datasets"][0]["progress"]["reference_batches_completed"] == 1
