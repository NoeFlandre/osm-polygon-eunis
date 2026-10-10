import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).parents[2]
RELEASE = REPO_ROOT / "scripts" / "grid5000" / "release.sh"
JOB_ID = "4343"
SOURCE_REVISION = "b" * 40

# Records each uv call with the source commit it saw, and writes the release
# receipt that release.sh later merges into the Grid'5000 receipt.
_UV_STUB = """\
import json
import os
import sys
from pathlib import Path

arguments = sys.argv[1:]
calls_file = Path(os.environ["STUB_STATE_DIR"]) / "uv-calls.jsonl"
call = {
    "argv": arguments,
    "env": {"EUNIS_SOURCE_COMMIT": os.environ.get("EUNIS_SOURCE_COMMIT")},
}
with calls_file.open("a", encoding="utf-8") as output:
    output.write(json.dumps(call) + "\\n")
if arguments[:1] == ["run"]:
    receipt = Path(arguments[arguments.index("--receipt") + 1])
    receipt.write_text('{"release": {"status": "complete"}}\\n', encoding="utf-8")
"""

_OARSTAT_STUB = """\
import json
import os
import sys

job_id = sys.argv[sys.argv.index("-j") + 1]
print(json.dumps({job_id: {"start_time": int(os.environ["STUB_OAR_START_TIME"])}}))
"""


@dataclass(frozen=True)
class _ReleaseRun:
    persistent_root: str
    uv_calls: list[dict[str, Any]]
    receipt: dict[str, Any]


def _bash() -> str:
    shell = shutil.which("bash")
    assert shell is not None
    return shell


def _write_stub(path: Path, body: str) -> None:
    path.write_text(f"#!/usr/bin/env python3\n{body}", encoding="utf-8")
    path.chmod(0o755)


def _persistent_root(workspace: Path) -> str:
    """Return a root that passes release.sh's literal /home/ gate inside workspace.

    The gate matches the string prefix, so the root starts with /home/ and then
    climbs out with /home/.. so that the kernel resolves it to the temporary
    workspace. This needs /home to be a real directory on the test machine.
    """

    if Path("/home/..").resolve() != Path("/"):
        pytest.fail("these tests need /home to be a real directory")
    return f"/home/../{workspace.relative_to('/').as_posix()}/persistent"


def _release_environment(workspace: Path) -> dict[str, str]:
    bin_dir = workspace / "bin"
    bin_dir.mkdir(parents=True)
    # python3 resolves to the test interpreter. The scripts use only the standard library.
    (bin_dir / "python3").symlink_to(sys.executable)
    _write_stub(bin_dir / "uv", _UV_STUB)
    _write_stub(bin_dir / "oarstat", _OARSTAT_STUB)
    for name in ("home", "hf-home", "tmp", "state"):
        (workspace / name).mkdir()
    return {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ.get('PATH', os.defpath)}",
        "HOME": str(workspace / "home"),
        "HF_HOME": str(workspace / "hf-home"),
        "HF_TOKEN": "test-token",
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
        "GRID5000_EXCLUDED_SITES": '["bordeaux","sophia"]',
        "GRID5000_RETRY_DELAY": "0",
        "STUB_STATE_DIR": str(workspace / "state"),
        "STUB_OAR_START_TIME": str(int(time.time())),
    }


@pytest.fixture(scope="module")
def release_run(tmp_path_factory: pytest.TempPathFactory) -> _ReleaseRun:
    workspace = tmp_path_factory.mktemp("release")
    environment = _release_environment(workspace)
    result = subprocess.run(  # noqa: S603
        [_bash(), str(RELEASE)],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    persistent_root = environment["GRID5000_PERSISTENT_ROOT"]
    calls_file = workspace / "state" / "uv-calls.jsonl"
    receipt = Path(persistent_root) / "receipts" / f"eunis-{JOB_ID}.json"
    return _ReleaseRun(
        persistent_root=persistent_root,
        uv_calls=[json.loads(line) for line in calls_file.read_text(encoding="utf-8").splitlines()],
        receipt=json.loads(receipt.read_text(encoding="utf-8")),
    )


def _release_call(run: _ReleaseRun) -> dict[str, Any]:
    calls = [call for call in run.uv_calls if call["argv"][:1] == ["run"]]
    assert len(calls) == 1, run.uv_calls
    return calls[0]


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_production_contract_is_grid5000_all_source_and_site_neutral() -> None:
    root = Path(__file__).parents[2]
    operations = (root / "docs" / "operations.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")

    assert "Grid'5000" in operations
    assert "grid5000 submit" in operations
    assert "usagepolicycheck -t" in operations
    assert "host=1/core=16" in operations
    assert "--site SITE" in operations
    assert "--exclude-site bordeaux" in operations
    assert "--exclude-site sophia" in operations
    assert "--cluster CLUSTER" in operations
    assert "cluster='CLUSTER'" in operations
    assert "-q default" in operations
    assert "website" in operations
    assert "wikidata" in operations
    assert "description" in operations
    assert "--receipt" in operations
    assert "EUNIS_SIDECAR_DIR" in operations
    assert "EUNIS_SOURCE_DIR" in operations
    assert "node-local" in operations
    assert "not backed up" in operations
    assert (
        "uv run osm-polygon-eunis release --batch-size 256 --workdir .eunis-run" not in operations
    )
    assert "Grid'5000-only" in readme
    assert "all three" in readme


@pytest.mark.slow
def test_release_command_writes_the_receipt_the_worker_merges(release_run: _ReleaseRun) -> None:
    release = _release_call(release_run)

    assert _flag_value(release["argv"], "--receipt") == (
        f"{release_run.persistent_root}/receipts/eunis-{JOB_ID}.release.json"
    )
    assert release_run.receipt["release"] == {"status": "complete"}


@pytest.mark.slow
def test_release_command_runs_without_the_retired_dataset_flag(release_run: _ReleaseRun) -> None:
    assert release_run.uv_calls, (
        "release.sh made no uv calls, so the check below would pass vacuously"
    )
    for call in release_run.uv_calls:
        assert not any(argument.startswith("--dataset") for argument in call["argv"]), call


@pytest.mark.slow
def test_release_command_sees_the_source_commit_exported_before_it_starts(
    release_run: _ReleaseRun,
) -> None:
    assert _release_call(release_run)["env"]["EUNIS_SOURCE_COMMIT"] == SOURCE_REVISION
    assert release_run.receipt["grid5000"]["source_commit"] == SOURCE_REVISION


@pytest.mark.slow
def test_receipt_records_the_grid_settings_with_their_types(release_run: _ReleaseRun) -> None:
    grid = release_run.receipt["grid5000"]

    assert grid["site"] == "test-site"
    assert grid["frontend"] == "frontend.test"
    assert grid["cluster"] == "test-cluster"
    assert grid["queue"] == "test-queue"
    assert grid["walltime"] == "0:10:00"
    assert grid["excluded_sites"] == ["bordeaux", "sophia"]
    for field, expected in (("cores", 4), ("workers", 2), ("batch_size", 8)):
        assert grid[field] == expected
        assert isinstance(grid[field], int)
    assert grid["errors"] == {"count": 0}
    assert isinstance(grid["errors"]["count"], int)
