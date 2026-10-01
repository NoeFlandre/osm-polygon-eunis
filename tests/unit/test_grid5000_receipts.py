import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ReceiptOptions:
    exit_status: str = "124"
    attempt: str = "3"
    error_count: str = "2"
    stop_margin: str = "300"
    termination_grace: str = "20"
    stop_state: str | None = "deadline"


def _write_receipt(
    tmp_path: Path,
    options: ReceiptOptions | None = None,
) -> dict[str, Any]:
    options = ReceiptOptions() if options is None else options
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/receipts.py"
    output = tmp_path / "receipt.json"
    stop_marker = tmp_path / "worker-stop-state"
    if options.stop_state is not None:
        stop_marker.write_text(f"{options.stop_state}\n", encoding="utf-8")
    command = [
        sys.executable,
        str(helper),
        "--output",
        str(output),
        "--job-id",
        "482079",
        "--source-commit",
        "abc123",
        "--config-sha256",
        "def456",
        "--config-path",
        "config/eea-2021-reference.json",
        "--site",
        "toulouse",
        "--frontend",
        "ftoulouse",
        "--cluster",
        "montcalm",
        "--queue",
        "default",
        "--cores",
        "16",
        "--workers",
        "16",
        "--walltime",
        "1:00:00",
        "--batch-size",
        "256",
        "--excluded-sites-json",
        '["bordeaux","sophia"]',
        "--attempts",
        options.attempt,
        "--error-count",
        options.error_count,
        "--exit-status",
        options.exit_status,
        "--job-started-epoch",
        "1790830000",
        "--stop-margin-seconds",
        options.stop_margin,
        "--termination-grace-seconds",
        options.termination_grace,
        "--log-path",
        "/home/nflandre/osm-polygon-eunis/logs/job-482079.log",
        "--stop-marker",
        str(stop_marker),
        "--external-signal",
        "none",
    ]
    result = subprocess.run(command, capture_output=True, check=False, text=True)  # noqa: S603

    assert result.returncode == 0, result.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_deadline_receipt_preserves_attempt_error_and_retry_counts(tmp_path: Path) -> None:
    payload = _write_receipt(tmp_path)

    assert payload["status"] == "incomplete"
    grid = payload["grid5000"]
    assert grid["attempts"] == 3
    assert grid["retries"] == 2
    assert grid["errors"] == {"count": 2, "last_exit_status": 124}
    assert grid["stop_reason"] == "graceful_deadline"


def test_child_exit_124_without_deadline_marker_is_a_failed_attempt(tmp_path: Path) -> None:
    payload = _write_receipt(tmp_path, ReceiptOptions(error_count="0", stop_state=None))

    assert payload["status"] == "failed"
    grid = payload["grid5000"]
    assert grid["errors"] == {"count": 1, "last_exit_status": 124}
    assert grid["stop_reason"] == "worker_failure"


def test_invalid_deadline_overrides_do_not_block_failure_receipt(tmp_path: Path) -> None:
    payload = _write_receipt(
        tmp_path,
        ReceiptOptions(
            exit_status="2",
            attempt="0",
            error_count="1",
            stop_margin="invalid",
            termination_grace="NaN",
            stop_state=None,
        ),
    )

    grid = payload["grid5000"]
    assert payload["status"] == "failed"
    assert grid["stop_margin_seconds"] is None
    assert grid["requested_stop_margin_seconds"] == "invalid"
    assert grid["termination_grace_seconds"] is None
    assert grid["requested_termination_grace_seconds"] == "NaN"
    json.dumps(payload, allow_nan=False)
