import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
from scripts.grid5000.worker_deadline import (
    parse_walltime_seconds,
    remaining_runtime_seconds,
    run_command,
)


@pytest.mark.parametrize(
    ("walltime", "expected"),
    [("1:00:00", 3600), ("00:05:07", 307), ("0:59:59", 3599)],
)
def test_parse_walltime_seconds(walltime: str, expected: int) -> None:
    assert parse_walltime_seconds(walltime) == expected


@pytest.mark.parametrize(
    "walltime",
    ["", "1:2:03", "1:60:00", "1:00:60", "-1:00:00", "1:00:01", "120:00:00"],
)
def test_parse_walltime_rejects_malformed_values(walltime: str) -> None:
    with pytest.raises(ValueError, match="walltime"):
        parse_walltime_seconds(walltime)


@pytest.mark.parametrize(
    ("walltime", "stop_margin", "termination_grace"),
    [
        ("1:00:00", "0", "20"),
        ("1:00:00", "49", "20"),
        ("1:00:00", "3600", "20"),
        ("1:00:01", "300", "20"),
    ],
)
def test_worker_rejects_deadline_settings_without_receipt_window(
    walltime: str,
    stop_margin: str,
    termination_grace: str,
) -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(helper),
            "--walltime",
            walltime,
            "--started-at",
            str(time.time()),
            "--stop-margin-seconds",
            stop_margin,
            "--termination-grace-seconds",
            termination_grace,
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 2
    assert "receipt window" in result.stderr or "walltime" in result.stderr


def test_remaining_runtime_accounts_for_elapsed_setup_and_safety_margin() -> None:
    assert (
        remaining_runtime_seconds(
            "1:00:00",
            started_at=1_000,
            now=1_600,
            stop_margin_seconds=300,
        )
        == 2_700
    )


def test_remaining_runtime_never_becomes_negative() -> None:
    assert (
        remaining_runtime_seconds(
            "1:00:00",
            started_at=1_000,
            now=5_000,
            stop_margin_seconds=300,
        )
        == 0
    )


def test_run_command_preserves_child_exit_status() -> None:
    result = run_command(
        [sys.executable, "-c", "raise SystemExit(17)"],
        timeout_seconds=5,
        termination_grace_seconds=1,
    )

    assert result == 17


def test_run_command_terminates_child_process_group_at_deadline(capfd) -> None:
    child = (
        "import os, signal, time; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
        "print(os.getpid(), flush=True); "
        "time.sleep(30)"
    )

    result = run_command(
        [sys.executable, "-c", child],
        timeout_seconds=0.2,
        termination_grace_seconds=0.2,
    )

    output = capfd.readouterr().out.strip()
    assert result == 124
    assert output.isdigit()
    with pytest.raises(ProcessLookupError):
        os.kill(int(output), 0)


def test_child_exit_124_is_not_marked_as_a_deadline(tmp_path: Path) -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    stop_marker = tmp_path / "worker-stop-state"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(helper),
            "--walltime",
            "1:00:00",
            "--started-at",
            str(time.time()),
            "--stop-margin-seconds",
            "300",
            "--termination-grace-seconds",
            "20",
            "--stop-marker",
            str(stop_marker),
            "--",
            sys.executable,
            "-c",
            "raise SystemExit(124)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 124
    assert not stop_marker.exists()


def test_deadline_writes_a_distinguishing_stop_marker(tmp_path: Path) -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    stop_marker = tmp_path / "worker-stop-state"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(helper),
            "--walltime",
            "0:01:00",
            "--started-at",
            str(time.time() - 28),
            "--stop-margin-seconds",
            "31",
            "--termination-grace-seconds",
            "1",
            "--stop-marker",
            str(stop_marker),
            "--",
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 124
    assert stop_marker.read_text(encoding="utf-8").strip() == "deadline"


def test_worker_deadline_returns_shell_status_for_external_sigterm(tmp_path: Path) -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    stop_marker = tmp_path / "worker-stop-state"
    command = [
        sys.executable,
        str(helper),
        "--walltime",
        "1:00:00",
        "--started-at",
        str(time.time()),
        "--stop-margin-seconds",
        "300",
        "--termination-grace-seconds",
        "1",
        "--stop-marker",
        str(stop_marker),
        "--",
        sys.executable,
        "-c",
        "import os, time; print(os.getpid(), flush=True); time.sleep(30)",
    ]
    process = subprocess.Popen(  # noqa: S603
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )
    assert process.stdout is not None
    child_pid_line = process.stdout.readline().strip()
    assert child_pid_line.isdigit()
    process.send_signal(signal.SIGTERM)

    assert process.wait(timeout=5) == 143
    assert stop_marker.read_text(encoding="utf-8").strip() == "signal:15"
    stop_marker.unlink()
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_line), 0)


def test_worker_accepts_minimum_receipt_window() -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    result = subprocess.run(  # noqa: S603
        [
            sys.executable,
            str(helper),
            "--walltime",
            "1:00:00",
            "--started-at",
            str(time.time()),
            "--stop-margin-seconds",
            "50",
            "--termination-grace-seconds",
            "20",
            "--",
            sys.executable,
            "-c",
            "pass",
        ],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
