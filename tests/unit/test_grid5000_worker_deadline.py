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
    [("1:00:00", 3600), ("00:05:07", 307), ("120:00:00", 432000)],
)
def test_parse_walltime_seconds(walltime: str, expected: int) -> None:
    assert parse_walltime_seconds(walltime) == expected


@pytest.mark.parametrize("walltime", ["", "1:2:03", "1:60:00", "1:00:60", "-1:00:00"])
def test_parse_walltime_rejects_malformed_values(walltime: str) -> None:
    with pytest.raises(ValueError, match="walltime"):
        parse_walltime_seconds(walltime)


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


def test_worker_deadline_returns_shell_status_for_external_sigterm() -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/worker_deadline.py"
    command = [
        sys.executable,
        str(helper),
        "--walltime",
        "1:00:00",
        "--started-at",
        str(time.time()),
        "--stop-margin-seconds",
        "0",
        "--termination-grace-seconds",
        "0.2",
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
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_line), 0)
