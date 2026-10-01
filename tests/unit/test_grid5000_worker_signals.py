import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path


def test_sigterm_is_forwarded_to_active_helper_before_receipt_exit(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    signal_library = project_root / "scripts/grid5000/worker_signals.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    helper_started = tmp_path / "helper-started"
    helper_signal = tmp_path / "helper-signal"
    receipt_signal = tmp_path / "receipt-signal"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "def stop(signum, _frame):\n"
            "    Path(os.environ['HELPER_SIGNAL']).write_text(str(signum))\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "ready_file = os.environ.get('GRID5000_WORKER_READY_FILE')\n"
            "if ready_file:\n"
            "    Path(ready_file).write_text('ready')\n"
            "Path(os.environ['HELPER_STARTED']).write_text('started')\n"
            "time.sleep(30)\n"
        )
        + ' "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TMPDIR": str(tmp_path),
            "HELPER_STARTED": str(helper_started),
            "HELPER_SIGNAL": str(helper_signal),
        }
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            'source "$1"',
            "external_signal=none",
            'receipt_path="$2"',
            'trap \'printf "%s\\n" "$external_signal" > "$receipt_path"\' EXIT',
            "install_worker_signal_traps",
            "run_deadline_helper ignored-arguments",
        )
    )
    process = subprocess.Popen(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            script,
            "worker-signal-test",
            str(signal_library),
            str(receipt_signal),
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        start_deadline = time.monotonic() + 5
        while not helper_started.exists() and process.poll() is None:
            if time.monotonic() >= start_deadline:
                break
            time.sleep(0.02)
        assert helper_started.exists(), process.communicate(timeout=2)[1]

        process.send_signal(signal.SIGTERM)
        assert process.wait(timeout=5) == 143
        process.communicate(timeout=1)

        assert helper_signal.read_text(encoding="utf-8") == "15"
        assert receipt_signal.read_text(encoding="utf-8").strip() == "TERM"
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=2)


def test_sigterm_during_helper_pid_registration_is_queued(tmp_path: Path) -> None:
    project_root = Path(__file__).resolve().parents[2]
    signal_library = project_root / "scripts/grid5000/worker_signals.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    helper_started = tmp_path / "helper-started"
    helper_signal = tmp_path / "helper-signal"
    receipt_signal = tmp_path / "receipt-signal"
    helper_ready = tmp_path / "helper-ready"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "def stop(signum, _frame):\n"
            "    Path(os.environ['HELPER_SIGNAL']).write_text(str(signum))\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGTERM, stop)\n"
            "ready_file = os.environ.get('GRID5000_WORKER_READY_FILE')\n"
            "if ready_file:\n"
            "    Path(ready_file).write_text('ready')\n"
            "Path(os.environ['HELPER_STARTED']).write_text('started')\n"
            "time.sleep(30)\n"
        )
        + ' "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TMPDIR": str(tmp_path),
            "GRID5000_WORKER_READY_FILE": str(helper_ready),
            "HELPER_STARTED": str(helper_started),
            "HELPER_SIGNAL": str(helper_signal),
        }
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            'source "$1"',
            "external_signal=none",
            "helper_launching=false",
            'pending_worker_signal=""',
            'active_helper_pid=""',
            'receipt_path="$2"',
            'trap \'printf "%s\\n" "$external_signal" > "$receipt_path"\' EXIT',
            "install_worker_signal_traps",
            "helper_launching=true",
            "python3 ignored-arguments &",
            "launched_helper_pid=$!",
            "start_deadline=$(date +%s)",
            f"while [[ ! -e {shlex.quote(str(helper_started))} ]]; do",
            "  if (( $(date +%s) - start_deadline > 5 )); then exit 99; fi",
            "  sleep 0.01",
            "done",
            "handle_worker_signal TERM 143",
            'register_worker_helper "$launched_helper_pid"',
            'wait_for_worker_helper_ready "$GRID5000_WORKER_READY_FILE"',
        )
    )
    process = subprocess.Popen(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            script,
            "worker-signal-race-test",
            str(signal_library),
            str(receipt_signal),
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        assert process.wait(timeout=8) == 143
        assert helper_signal.read_text(encoding="utf-8") == "15"
        assert receipt_signal.read_text(encoding="utf-8").strip() == "TERM"
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=2)
        process.communicate(timeout=2)


def test_sigint_before_helper_installs_handler_is_queued_until_ready(
    tmp_path: Path,
) -> None:
    project_root = Path(__file__).resolve().parents[2]
    signal_library = project_root / "scripts/grid5000/worker_signals.sh"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    helper_started = tmp_path / "helper-started"
    helper_signal = tmp_path / "helper-signal"
    receipt_signal = tmp_path / "receipt-signal"
    fake_python = fake_bin / "python3"
    fake_python.write_text(
        "#!/bin/sh\n"
        f"exec {shlex.quote(sys.executable)} -c "
        + shlex.quote(
            "import os, signal, time\n"
            "from pathlib import Path\n"
            "signal.signal(signal.SIGINT, signal.SIG_IGN)\n"
            "Path(os.environ['HELPER_STARTED']).write_text('started')\n"
            "time.sleep(0.25)\n"
            "def stop(signum, _frame):\n"
            "    Path(os.environ['HELPER_SIGNAL']).write_text(str(signum))\n"
            "    raise SystemExit(0)\n"
            "signal.signal(signal.SIGINT, stop)\n"
            "ready_file = os.environ.get('GRID5000_WORKER_READY_FILE')\n"
            "if ready_file:\n"
            "    Path(ready_file).write_text('ready')\n"
            "time.sleep(30)\n"
        )
        + ' "$@"\n',
        encoding="utf-8",
    )
    fake_python.chmod(0o755)
    environment = os.environ.copy()
    environment.update(
        {
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "TMPDIR": str(tmp_path),
            "HELPER_STARTED": str(helper_started),
            "HELPER_SIGNAL": str(helper_signal),
        }
    )
    script = "\n".join(
        (
            "set -euo pipefail",
            'source "$1"',
            "external_signal=none",
            'receipt_path="$2"',
            'trap \'printf "%s\\n" "$external_signal" > "$receipt_path"\' EXIT',
            "install_worker_signal_traps",
            "run_deadline_helper ignored-arguments",
        )
    )
    process = subprocess.Popen(  # noqa: S603
        [
            "/bin/bash",
            "-c",
            script,
            "worker-signal-startup-race-test",
            str(signal_library),
            str(receipt_signal),
        ],
        cwd=project_root,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )

    try:
        start_deadline = time.monotonic() + 5
        while not helper_started.exists() and process.poll() is None:
            if time.monotonic() >= start_deadline:
                break
            time.sleep(0.02)
        assert helper_started.exists(), process.communicate(timeout=2)[1]

        process.send_signal(signal.SIGINT)
        assert process.wait(timeout=5) == 130
        process.communicate(timeout=1)

        assert helper_signal.read_text(encoding="utf-8") == str(signal.SIGINT)
        assert receipt_signal.read_text(encoding="utf-8").strip() == "INT"
    finally:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=2)
        process.communicate(timeout=2)
