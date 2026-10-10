import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
RELEASE = REPO_ROOT / "scripts" / "grid5000" / "release.sh"
REFERENCE_CONFIG = REPO_ROOT / "config" / "eea-2021-reference.json"
JOB_ID = "4242"
SOURCE_REVISION = "a" * 40
CREDENTIAL = "hf_test_credential_value"
RECORDED_ENVIRONMENT = [
    "PATH",
    "HF_TOKEN",
    "EUNIS_REFERENCE_DIR",
    "EUNIS_SIDECAR_DIR",
    "EUNIS_SOURCE_DIR",
    "EUNIS_SOURCE_COMMIT",
    "UV_CACHE_DIR",
    "UV_PROJECT_ENVIRONMENT",
]

# Stand-ins for the OAR and uv commands. They record each call so tests can
# assert which worker commands ran and what environment they received.
_UV_STUB = """\
import json
import os
import sys
import time
from pathlib import Path

arguments = sys.argv[1:]
calls_file = Path(os.environ["STUB_STATE_DIR"]) / "uv-calls.jsonl"
recorded = json.loads(os.environ["STUB_RECORDED_ENVIRONMENT"])
call = {
    "argv": arguments,
    "env": {name: os.environ.get(name) for name in recorded},
}
with calls_file.open("a", encoding="utf-8") as output:
    output.write(json.dumps(call) + "\\n")
if arguments[:1] != ["run"]:
    raise SystemExit(0)
time.sleep(float(os.environ["STUB_RELEASE_SECONDS"]))
statuses = json.loads(os.environ["STUB_RELEASE_STATUSES"])
release_runs = sum(
    json.loads(line)["argv"][:1] == ["run"]
    for line in calls_file.read_text(encoding="utf-8").splitlines()
)
status = statuses[min(release_runs, len(statuses)) - 1]
if status == 0:
    receipt = Path(arguments[arguments.index("--receipt") + 1])
    receipt.write_text('{"release": {"status": "complete"}}\\n', encoding="utf-8")
raise SystemExit(status)
"""

_OARSTAT_STUB = """\
import json
import os
import sys

start_time = os.environ["STUB_OAR_START_TIME"]
if not start_time.isdigit():
    raise SystemExit(1)
job_id = sys.argv[sys.argv.index("-j") + 1]
print(json.dumps({job_id: {"start_time": int(start_time)}}))
"""

# Answers a remote OAR lookup only when a test provides a start time, so no test
# reaches a real frontend by accident.
_SSH_STUB = """\
import json
import os
import sys
from pathlib import Path

calls_file = Path(os.environ["STUB_STATE_DIR"]) / "ssh-calls.jsonl"
with calls_file.open("a", encoding="utf-8") as output:
    output.write(json.dumps(sys.argv[1:]) + "\\n")
start_time = os.environ["STUB_SSH_START_TIME"]
if not start_time.isdigit():
    raise SystemExit(255)
job_id = sys.argv[sys.argv.index("-j") + 1]
print(json.dumps({job_id: {"start_time": int(start_time)}}))
"""


def _bash() -> str:
    shell = shutil.which("bash")
    assert shell is not None
    return shell


def _persistent_root(workspace: Path) -> str:
    """Return a root that passes release.sh's literal /home/ gate inside workspace.

    The gate matches the string prefix, so the root starts with /home/ and then
    climbs out with /home/.. so that the kernel resolves it to the temporary
    workspace. This needs /home to be a real directory on the test machine.
    """

    if Path("/home/..").resolve() != Path("/"):
        pytest.fail("these tests need /home to be a real directory")
    return f"/home/../{workspace.relative_to('/').as_posix()}/persistent"


def _write_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _install_stubs(bin_dir: Path) -> None:
    bin_dir.mkdir(parents=True, exist_ok=True)
    # python3 resolves to the test interpreter. The scripts use only the standard library.
    (bin_dir / "python3").symlink_to(sys.executable)
    _write_stub(bin_dir / "uv", _UV_STUB)
    _write_stub(bin_dir / "oarstat", _OARSTAT_STUB)
    _write_stub(bin_dir / "ssh", _SSH_STUB)


def _prepare_workspace(workspace: Path, *, token: bool) -> None:
    _install_stubs(workspace / "bin")
    for directory in (
        workspace / "state",
        workspace / "home",
        workspace / "hf-home",
        workspace / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    if token:
        (workspace / "hf-home" / "token").write_text(f"{CREDENTIAL}\n", encoding="utf-8")


def _release_environment(
    workspace: Path,
    overrides: Mapping[str, str] | None = None,
    *,
    token: bool = True,
) -> dict[str, str]:
    _prepare_workspace(workspace, token=token)
    bin_dir = workspace / "bin"
    environment = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "HOME": str(workspace / "home"),
        "HF_HOME": str(workspace / "hf-home"),
        "TMPDIR": str(workspace / "tmp"),
        "OAR_JOB_ID": JOB_ID,
        "GRID5000_PERSISTENT_ROOT": _persistent_root(workspace),
        "GRID5000_SOURCE_REVISION": SOURCE_REVISION,
        "GRID5000_SITE": "test-site",
        "GRID5000_FRONTEND": "frontend.test",
        "GRID5000_CLUSTER": "test-cluster",
        "GRID5000_QUEUE": "test-queue",
        "GRID5000_CORES": "4",
        "GRID5000_WORKERS": "2",
        "GRID5000_BATCH_SIZE": "8",
        "GRID5000_WALLTIME": "0:10:00",
        "GRID5000_RETRY_DELAY": "0",
        "STUB_STATE_DIR": str(workspace / "state"),
        "STUB_RECORDED_ENVIRONMENT": json.dumps(RECORDED_ENVIRONMENT),
        "STUB_OAR_START_TIME": str(int(time.time())),
        "STUB_SSH_START_TIME": "",
        "STUB_RELEASE_STATUSES": "[0]",
        "STUB_RELEASE_SECONDS": "0",
    }
    environment.update(overrides or {})
    return environment


def _run_release(
    workspace: Path,
    overrides: Mapping[str, str] | None = None,
    *,
    token: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [_bash(), str(RELEASE)],
        capture_output=True,
        check=False,
        env=_release_environment(workspace, overrides, token=token),
        text=True,
        timeout=60,
    )


def _tight_deadline_overrides() -> dict[str, str]:
    """Leave about five seconds of worker time before the OAR deadline.

    A 60-second walltime with a 30-second stop margin gives 30 seconds of runtime
    counted from the OAR start. The job is recorded as started 25 seconds ago, so
    about 5 seconds remain. A deadline restarted from the script's own clock would
    allow the full 30 seconds instead.
    """

    return {
        "GRID5000_WALLTIME": "0:01:00",
        "GRID5000_STOP_MARGIN_SECONDS": "30",
        "GRID5000_TERMINATION_GRACE_SECONDS": "0",
        "STUB_OAR_START_TIME": str(int(time.time()) - 25),
    }


def _uv_calls(workspace: Path) -> list[dict[str, Any]]:
    calls_file = workspace / "state" / "uv-calls.jsonl"
    if not calls_file.exists():
        return []
    return [json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()]


def _release_runs(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [call for call in calls if call["argv"][:1] == ["run"]]


def _release_run_started(workspace: Path) -> bool:
    try:
        return bool(_release_runs(_uv_calls(workspace)))
    except json.JSONDecodeError:
        # The uv stub may be part-way through writing its record.
        return False


def _wait_for_release_run(workspace: Path, timeout_seconds: float = 30) -> None:
    deadline = time.monotonic() + timeout_seconds
    while not _release_run_started(workspace):
        if time.monotonic() >= deadline:
            pytest.fail("the release run never started")
        time.sleep(0.05)


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _stop_marker(workspace: Path) -> Path:
    return workspace / "tmp" / f"osm-polygon-eunis-{JOB_ID}" / "worker-stop-state"


def _receipt(workspace: Path) -> dict[str, Any]:
    return _read_json(Path(_persistent_root(workspace)) / "receipts" / f"eunis-{JOB_ID}.json")


def test_release_script_has_valid_bash_syntax() -> None:
    result = subprocess.run(  # noqa: S603
        [_bash(), "-n", str(RELEASE)],
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_release_script_keeps_strict_mode_and_drops_the_retired_script() -> None:
    # Strict mode and the retired script name leave no observable trace in a run,
    # so these two narrow source checks are kept on purpose.
    script = RELEASE.read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert "description-release.sh" not in script


@pytest.mark.parametrize("root", ["/var/lib/eunis", "/homework/eunis", "/srv", "relative/root"])
@pytest.mark.slow
def test_release_rejects_roots_outside_persistent_storage(tmp_path: Path, root: str) -> None:
    result = _run_release(tmp_path, {"GRID5000_PERSISTENT_ROOT": root})

    assert result.returncode == 2
    assert "must be persistent remote storage" in result.stderr
    assert _uv_calls(tmp_path) == []


@pytest.mark.slow
def test_release_requires_an_oar_job_before_doing_any_work(tmp_path: Path) -> None:
    result = _run_release(tmp_path, {"OAR_JOB_ID": ""})

    assert result.returncode == 1
    assert "must run inside an OAR job" in result.stderr
    assert _uv_calls(tmp_path) == []


@pytest.mark.slow
def test_release_refuses_to_start_without_a_hugging_face_token(tmp_path: Path) -> None:
    result = _run_release(tmp_path, token=False)

    # The script redirects stderr into its log after it starts, so check both streams.
    assert result.returncode == 2
    assert "HF_TOKEN or the Hugging Face cache must be available" in result.stdout + result.stderr
    assert _uv_calls(tmp_path) == []


@pytest.mark.parametrize(
    ("token_dir", "overrides"),
    [
        pytest.param("hf-home", {}, id="hf-home-env"),
        pytest.param("home/.cache/huggingface", {"HF_HOME": ""}, id="home-cache"),
    ],
)
@pytest.mark.slow
def test_release_run_gets_cached_token_and_checkpoint_paths(
    tmp_path: Path,
    token_dir: str,
    overrides: dict[str, str],
) -> None:
    cache = tmp_path / token_dir
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "token").write_text(f"{CREDENTIAL}\n", encoding="utf-8")
    root = _persistent_root(tmp_path)
    scratch = tmp_path / "tmp" / f"osm-polygon-eunis-{JOB_ID}"

    result = _run_release(tmp_path, overrides, token=False)

    assert result.returncode == 0, result.stdout + result.stderr
    calls = _uv_calls(tmp_path)
    assert calls[0]["argv"] == ["sync", "--frozen", "--no-dev"]
    assert len(calls) == 2
    release = calls[1]
    assert release["argv"][:5] == ["run", "--frozen", "--no-dev", "osm-polygon-eunis", "release"]
    environment = release["env"]
    assert environment["PATH"].split(os.pathsep)[0] == str(tmp_path / "home" / ".local" / "bin")
    assert environment["HF_TOKEN"] == CREDENTIAL
    assert environment["EUNIS_REFERENCE_DIR"] == f"{root}/cache/reference"
    assert environment["EUNIS_SIDECAR_DIR"] == f"{root}/sidecars/eunis"
    assert environment["EUNIS_SOURCE_DIR"] == f"{scratch}/source"
    assert environment["EUNIS_SOURCE_COMMIT"] == SOURCE_REVISION
    assert environment["UV_CACHE_DIR"] == f"{scratch}/uv-cache"
    assert environment["UV_PROJECT_ENVIRONMENT"] == f"{scratch}/venv"
    argv = release["argv"]
    assert _flag_value(argv, "--workdir") == f"{root}/runs/eunis"
    assert _flag_value(argv, "--workers") == "2"
    assert _flag_value(argv, "--batch-size") == "8"
    assert _flag_value(argv, "--receipt") == f"{root}/receipts/eunis-{JOB_ID}.release.json"
    assert Path(_flag_value(argv, "--reference-config")).resolve() == REFERENCE_CONFIG.resolve()
    grid = _read_json(Path(root) / "receipts" / f"eunis-{JOB_ID}.json")["grid5000"]
    assert grid["job_id"] == JOB_ID
    assert grid["source_commit"] == SOURCE_REVISION
    assert grid["attempts"] == 1
    assert grid["retries"] == 0
    assert grid["errors"]["count"] == 0
    assert grid["stop_margin_seconds"] == 300
    assert grid["termination_grace_seconds"] == 20
    assert grid["config"]["sha256"] == hashlib.sha256(REFERENCE_CONFIG.read_bytes()).hexdigest()
    log_text = (Path(root) / "logs" / f"job-{JOB_ID}.log").read_text(encoding="utf-8")
    assert CREDENTIAL not in log_text
    assert CREDENTIAL not in result.stdout + result.stderr


@pytest.mark.slow
def test_release_retries_a_failed_attempt_after_saving_checkpoints(tmp_path: Path) -> None:
    result = _run_release(tmp_path, {"STUB_RELEASE_STATUSES": "[7, 0]"})

    assert result.returncode == 0, result.stdout + result.stderr
    assert "release attempt 1 failed with rc=7; retrying after checkpoints" in result.stdout
    assert len(_release_runs(_uv_calls(tmp_path))) == 2
    grid = _read_json(Path(_persistent_root(tmp_path)) / "receipts" / f"eunis-{JOB_ID}.json")[
        "grid5000"
    ]
    assert grid["attempts"] == 2
    assert grid["retries"] == 1
    assert grid["errors"]["count"] == 1


@pytest.mark.slow
def test_release_gives_up_after_max_attempts_and_writes_a_failure_receipt(
    tmp_path: Path,
) -> None:
    result = _run_release(
        tmp_path,
        {"STUB_RELEASE_STATUSES": "[3]", "GRID5000_MAX_ATTEMPTS": "2"},
    )

    assert result.returncode == 3
    assert "release attempt 2/2" in result.stdout
    assert len(_release_runs(_uv_calls(tmp_path))) == 2
    receipt = _read_json(Path(_persistent_root(tmp_path)) / "receipts" / f"eunis-{JOB_ID}.json")
    assert receipt["status"] == "failed"
    assert receipt["grid5000"]["stop_reason"] == "worker_failure"
    assert receipt["grid5000"]["attempts"] == 2
    assert receipt["grid5000"]["errors"] == {"count": 2, "last_exit_status": 3}


@pytest.mark.slow
def test_release_defaults_to_sixteen_workers_and_twenty_attempts(tmp_path: Path) -> None:
    # An empty value takes the script's default, as an unset variable does.
    result = _run_release(tmp_path, {"GRID5000_WORKERS": "", "STUB_RELEASE_STATUSES": "[3]"})

    assert result.returncode == 3, result.stdout + result.stderr
    assert "release attempt 20/20" in result.stdout
    runs = _release_runs(_uv_calls(tmp_path))
    assert len(runs) == 20
    assert {_flag_value(run["argv"], "--workers") for run in runs} == {"16"}


@pytest.mark.slow
def test_release_stops_at_the_oar_deadline_before_any_worker_runs(tmp_path: Path) -> None:
    # Started an hour ago, so the 10-minute walltime is already spent.
    result = _run_release(tmp_path, {"STUB_OAR_START_TIME": str(int(time.time()) - 3600)})

    assert result.returncode == 124
    assert _uv_calls(tmp_path) == []
    stop_marker = tmp_path / "tmp" / f"osm-polygon-eunis-{JOB_ID}" / "worker-stop-state"
    assert stop_marker.read_text(encoding="utf-8").strip() == "deadline"
    receipt = _read_json(Path(_persistent_root(tmp_path)) / "receipts" / f"eunis-{JOB_ID}.json")
    assert receipt["status"] == "incomplete"
    assert receipt["grid5000"]["stop_reason"] == "graceful_deadline"
    assert receipt["grid5000"]["attempts"] == 0
    assert receipt["grid5000"]["errors"]["count"] == 0


@pytest.mark.slow
def test_release_run_is_stopped_at_the_oar_deadline(tmp_path: Path) -> None:
    # The stub outlives the five seconds left, but not the 30 seconds a clock
    # restarted at script start would allow.
    result = _run_release(
        tmp_path,
        {**_tight_deadline_overrides(), "STUB_RELEASE_SECONDS": "20"},
    )

    assert result.returncode == 124, result.stdout + result.stderr
    assert "graceful worker deadline reached; saved checkpoints are ready to resume" in (
        result.stdout
    )
    assert [call["argv"][:1] for call in _uv_calls(tmp_path)] == [["sync"], ["run"]]
    assert _stop_marker(tmp_path).read_text(encoding="utf-8").strip() == "deadline"
    receipt = _receipt(tmp_path)
    assert receipt["status"] == "incomplete"
    assert receipt["grid5000"]["stop_reason"] == "graceful_deadline"
    assert receipt["grid5000"]["attempts"] == 1
    assert receipt["grid5000"]["errors"]["count"] == 0


@pytest.mark.slow
def test_retry_delay_is_stopped_at_the_oar_deadline_without_another_attempt(
    tmp_path: Path,
) -> None:
    result = _run_release(
        tmp_path,
        {
            **_tight_deadline_overrides(),
            "STUB_RELEASE_STATUSES": "[7]",
            "GRID5000_RETRY_DELAY": "60",
        },
    )

    assert result.returncode == 124, result.stdout + result.stderr
    assert "release attempt 1 failed with rc=7; retrying after checkpoints" in result.stdout
    assert "graceful worker deadline reached during retry delay" in result.stdout
    assert len(_release_runs(_uv_calls(tmp_path))) == 1
    assert _stop_marker(tmp_path).read_text(encoding="utf-8").strip() == "deadline"
    receipt = _receipt(tmp_path)
    assert receipt["status"] == "incomplete"
    assert receipt["grid5000"]["stop_reason"] == "graceful_deadline"
    assert receipt["grid5000"]["attempts"] == 1
    assert receipt["grid5000"]["errors"]["count"] == 1


@pytest.mark.slow
def test_sigterm_during_a_release_run_stops_the_worker_and_records_the_signal(
    tmp_path: Path,
) -> None:
    process = subprocess.Popen(  # noqa: S603
        [_bash(), str(RELEASE)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_release_environment(tmp_path, {"STUB_RELEASE_SECONDS": "60"}),
        text=True,
    )
    try:
        _wait_for_release_run(tmp_path)
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=60)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()

    assert process.returncode == 143, stdout + stderr
    assert [call["argv"][:1] for call in _uv_calls(tmp_path)] == [["sync"], ["run"]]
    assert _stop_marker(tmp_path).read_text(encoding="utf-8").strip() == "signal:15"
    receipt = _receipt(tmp_path)
    assert receipt["status"] == "incomplete"
    assert receipt["grid5000"]["stop_reason"] == "termination_signal"
    assert receipt["grid5000"]["attempts"] == 1


@pytest.mark.slow
def test_release_writes_a_failure_receipt_when_the_oar_start_time_is_unreadable(
    tmp_path: Path,
) -> None:
    result = _run_release(tmp_path, {"STUB_OAR_START_TIME": "unavailable"})

    assert result.returncode == 2
    assert "could not read OAR start time" in result.stdout + result.stderr
    assert _uv_calls(tmp_path) == []
    receipt = _read_json(Path(_persistent_root(tmp_path)) / "receipts" / f"eunis-{JOB_ID}.json")
    assert receipt["status"] == "failed"
    assert receipt["grid5000"]["attempts"] == 0
    assert receipt["grid5000"]["errors"] == {"count": 1, "last_exit_status": 2}


@pytest.mark.slow
def test_oar_start_time_falls_back_to_the_configured_frontend(tmp_path: Path) -> None:
    result = _run_release(
        tmp_path,
        {"STUB_OAR_START_TIME": "unavailable", "STUB_SSH_START_TIME": str(int(time.time()))},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "grid5000_job_started_epoch_source=frontend-oarstat" in result.stdout + result.stderr
    ssh_calls = (tmp_path / "state" / "ssh-calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ssh_calls) == 1
    argv = json.loads(ssh_calls[0])
    assert argv[argv.index("oarstat") - 1] == "frontend.test"
    assert _flag_value(argv, "-j") == JOB_ID


def test_grid_token_loader_exports_cached_token_without_printing_it(tmp_path: Path) -> None:
    credential_value = "hf_test_credential_value"
    token_file = tmp_path / "token"
    token_file.write_text(f"{credential_value}\n", encoding="utf-8")
    loader = Path(__file__).parents[2] / "scripts" / "grid5000" / "load_hf_token.sh"
    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    environment["EXPECTED_TOKEN"] = credential_value
    shell = shutil.which("bash")
    assert shell is not None

    # The script and all arguments are local test fixtures from trusted paths.
    result = subprocess.run(  # noqa: S603
        [
            shell,
            "-c",
            'source "$1"; eunis_load_hf_token "$2"; test "$HF_TOKEN" = "$EXPECTED_TOKEN"',
            "bash",
            str(loader),
            str(token_file),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert credential_value not in result.stdout
    assert credential_value not in result.stderr
