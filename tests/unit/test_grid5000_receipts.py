import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
RELEASE = REPO_ROOT / "scripts" / "grid5000" / "release.sh"
JOB_ID = "4242"


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
        f"--stop-margin-seconds={options.stop_margin}",
        f"--termination-grace-seconds={options.termination_grace}",
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


@pytest.mark.slow
@pytest.mark.parametrize(
    ("exit_status", "stop_state", "stop_reason"),
    [
        pytest.param("124", "deadline", "graceful_deadline", id="deadline"),
        pytest.param("130", "signal:2", "interrupt", id="interrupt"),
        pytest.param("143", "signal:15", "termination_signal", id="termination"),
    ],
)
def test_guarded_stops_are_incomplete_with_their_own_reason(
    tmp_path: Path,
    exit_status: str,
    stop_state: str,
    stop_reason: str,
) -> None:
    payload = _write_receipt(
        tmp_path, ReceiptOptions(exit_status=exit_status, stop_state=stop_state)
    )

    assert payload["status"] == "incomplete"
    assert payload["grid5000"]["stop_reason"] == stop_reason


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


def test_option_like_invalid_deadline_overrides_do_not_block_receipt(tmp_path: Path) -> None:
    payload = _write_receipt(
        tmp_path,
        ReceiptOptions(
            exit_status="2",
            attempt="0",
            error_count="1",
            stop_margin="--malformed-margin",
            termination_grace="--malformed-grace",
            stop_state=None,
        ),
    )

    grid = payload["grid5000"]
    assert payload["status"] == "failed"
    assert grid["requested_stop_margin_seconds"] == "--malformed-margin"
    assert grid["requested_termination_grace_seconds"] == "--malformed-grace"


def _persistent_root(workspace: Path) -> str:
    """Return a root that passes release.sh's literal /home/ gate inside workspace.

    The gate matches the string prefix, so the root starts with /home/ and then
    climbs out with /home/.. so that the kernel resolves it to the temporary
    workspace. This needs /home to be a real directory on the test machine.
    """

    if Path("/home/..").resolve() != Path("/"):
        pytest.fail("these tests need /home to be a real directory")
    return f"/home/../{workspace.relative_to('/').as_posix()}/persistent"


@pytest.mark.slow
def test_release_failure_handler_passes_raw_deadline_values_as_option_assignments(
    tmp_path: Path,
) -> None:
    # Metadata validation rejects the two option-like deadline values before any
    # work starts, and the EXIT handler then writes the failure receipt. Argparse
    # reads a space-separated value that starts with "--" as an option, so the
    # receipt exists only if the handler passes each value as --flag=value.
    shell = shutil.which("bash")
    assert shell is not None
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "python3").symlink_to(sys.executable)
    persistent_root = _persistent_root(tmp_path)
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "HOME": str(tmp_path / "home"),
        "HF_HOME": str(tmp_path / "hf-home"),
        "TMPDIR": str(tmp_path / "tmp"),
        "OAR_JOB_ID": JOB_ID,
        "GRID5000_PERSISTENT_ROOT": persistent_root,
        "GRID5000_SOURCE_REVISION": "c" * 40,
        "GRID5000_SITE": "test-site",
        "GRID5000_FRONTEND": "frontend.test",
        "GRID5000_CLUSTER": "test-cluster",
        "GRID5000_QUEUE": "test-queue",
        "GRID5000_CORES": "4",
        "GRID5000_WORKERS": "2",
        "GRID5000_BATCH_SIZE": "8",
        "GRID5000_WALLTIME": "0:10:00",
        "GRID5000_STOP_MARGIN_SECONDS": "--malformed-margin",
        "GRID5000_TERMINATION_GRACE_SECONDS": "--malformed-grace",
    }

    # The script and all arguments are local test fixtures from trusted paths.
    result = subprocess.run(  # noqa: S603
        [shell, str(RELEASE)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2, result.stdout + result.stderr
    receipt = json.loads(
        (Path(persistent_root) / "receipts" / f"eunis-{JOB_ID}.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "failed"
    grid = receipt["grid5000"]
    assert grid["site"] == "test-site"
    assert grid["requested_stop_margin_seconds"] == "--malformed-margin"
    assert grid["requested_termination_grace_seconds"] == "--malformed-grace"
