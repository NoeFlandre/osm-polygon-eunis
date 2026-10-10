"""Characterisation of OAR status and usage-policy gating through the public submit path.

These tests drive ``submit_grid5000`` with a fake command runner so they pin the
behaviour that operators depend on: how OAR table states, policy warnings and
the all-site job report decide whether a submission may proceed. They use only
the public ``osm_polygon_eunis.grid5000`` surface, so they stay valid while the
parsing and gate code moves between modules.

The OAR table and policy-warning samples follow the layout already captured in
``test_grid5000.py``. They have not been checked against a live Grid'5000 run
from this environment.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
from collections.abc import Callable
from pathlib import Path

import pytest

from osm_polygon_eunis import grid5000
from osm_polygon_eunis.grid5000 import Grid5000Config, Grid5000Submission, submit_grid5000

_CLEAN_POLICY = "No jobs flagged\n"
_EMPTY_REPORT = '{"active_jobs": [], "errors": []}'
_POLICY_WARNINGS = (
    "This script is testing usage policy conformance between a and b\n"
    "in lyon, on cluster taurus, nflandre crossed day/night boundaries\n"
    "  Job 2071820: 11h\n"
)
_OAR_HEADER = (
    "Job id     Name           User           Submission Date     S Queue\n"
    "---------- -------------- -------------- ------------------- - ----------\n"
)
_STATE_JOB_ID = "123456"
_NEW_JOB_ID = "654321"
_FRONTEND = "fgrenoble"

Runner = Callable[[tuple[str, ...]], str]


def _config() -> Grid5000Config:
    return Grid5000Config(
        frontend=_FRONTEND,
        persistent_root="/home/u/eunis",
        site="grenoble",
        cluster="dahu",
    )


def _oar_table(*states: str) -> str:
    """Return an OAR job table with one row per state letter, in order."""

    rows = "".join(
        f"{482097 + index:<10} osm-polygon-eu nflandre       2026-10-01 23:24:32 {state} default\n"
        for index, state in enumerate(states)
    )
    return _OAR_HEADER + rows


def _fake_runner(
    calls: list[tuple[str, ...]],
    *,
    status: str | subprocess.CalledProcessError = "",
    policy: str = _CLEAN_POLICY,
    all_site_report: str = _EMPTY_REPORT,
) -> Runner:
    def run(command: tuple[str, ...]) -> str:
        calls.append(command)
        if command == grid5000.build_active_eunis_jobs_command(_FRONTEND):
            return all_site_report
        if command[:2] == ("ssh", _FRONTEND):
            argv = shlex.split(command[2])
            if argv[0] == "oarstat":
                if isinstance(status, subprocess.CalledProcessError):
                    raise status
                return status
            if argv[0] == "oarsub":
                return f"Adding job {_NEW_JOB_ID}\n"
            if argv[0] == "usagepolicycheck":
                return policy
        return ""

    return run


def _submit(
    tmp_path: Path,
    runner: Runner,
    *,
    state_job_id: str | None = _STATE_JOB_ID,
):
    state = tmp_path / "job.json"
    if state_job_id is not None:
        state.write_text(json.dumps({"job_id": state_job_id}) + "\n", encoding="utf-8")
    return submit_grid5000(
        _config(),
        tmp_path,
        source_revision="abc123",
        state_path=state,
        runner=runner,
    )


def _oarsub_calls(calls: list[tuple[str, ...]]) -> list[tuple[str, ...]]:
    return [
        call
        for call in calls
        if call[:2] == ("ssh", _FRONTEND) and shlex.split(call[2])[0] == "oarsub"
    ]


@pytest.mark.parametrize("state", ["W", "H", "L", "R", "S", "F", "A", "r"])
def test_oar_table_active_state_blocks_resubmission(tmp_path: Path, state: str) -> None:
    calls: list[tuple[str, ...]] = []

    with pytest.raises(RuntimeError, match=f"Grid'5000 job {_STATE_JOB_ID} is already active"):
        _submit(tmp_path, _fake_runner(calls, status=_oar_table(state)))

    assert _oarsub_calls(calls) == []


@pytest.mark.parametrize("state", ["T", "E", "t", "e"])
def test_oar_table_terminal_state_allows_resubmission(tmp_path: Path, state: str) -> None:
    calls: list[tuple[str, ...]] = []

    submission = _submit(tmp_path, _fake_runner(calls, status=_oar_table(state)))

    assert submission.job is not None
    assert submission.job.job_id == _NEW_JOB_ID
    assert len(_oarsub_calls(calls)) == 1


def test_oar_table_unrecognised_state_letter_fails_closed(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    with pytest.raises(RuntimeError, match="cannot verify Grid'5000 job 123456 status"):
        _submit(tmp_path, _fake_runner(calls, status=_oar_table("Z")))

    assert _oarsub_calls(calls) == []


@pytest.mark.parametrize(
    ("states", "outcome"),
    [
        (("T", "R"), "submitted"),
        (("R", "T"), "active"),
        (("E", "W", "T"), "submitted"),
    ],
)
def test_oar_table_first_row_decides_the_job_state(
    tmp_path: Path, states: tuple[str, ...], outcome: str
) -> None:
    calls: list[tuple[str, ...]] = []
    runner = _fake_runner(calls, status=_oar_table(*states))

    if outcome == "active":
        with pytest.raises(RuntimeError, match="already active"):
            _submit(tmp_path, runner)
        assert _oarsub_calls(calls) == []
    else:
        submission = _submit(tmp_path, runner)
        assert submission.job is not None
        assert submission.job.job_id == _NEW_JOB_ID


@pytest.mark.parametrize(
    ("status", "outcome"),
    [
        ("Job 123456 running", "active"),
        ("ERROR: job not found", "submitted"),
        ("Job 123456 terminated", "submitted"),
    ],
)
def test_oar_text_status_fallback_decides_resubmission(
    tmp_path: Path, status: str, outcome: str
) -> None:
    calls: list[tuple[str, ...]] = []
    runner = _fake_runner(calls, status=status)

    if outcome == "active":
        with pytest.raises(RuntimeError, match="already active"):
            _submit(tmp_path, runner)
    else:
        submission = _submit(tmp_path, runner)
        assert submission.job is not None
        assert submission.job.job_id == _NEW_JOB_ID


def test_oar_status_command_failure_with_missing_job_allows_resubmission(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    failure = subprocess.CalledProcessError(1, ("oarstat",), stderr="ERROR: job not found")

    submission = _submit(tmp_path, _fake_runner(calls, status=failure))

    assert submission.job is not None
    assert submission.job.job_id == _NEW_JOB_ID


@pytest.mark.parametrize(
    "policy_output",
    [
        _POLICY_WARNINGS,
        _CLEAN_POLICY,
        "This script is testing usage policy conformance\n",
    ],
)
def test_policy_output_with_completion_marker_allows_submission(
    tmp_path: Path, policy_output: str
) -> None:
    calls: list[tuple[str, ...]] = []

    submission = _submit(
        tmp_path,
        _fake_runner(calls, policy=policy_output),
        state_job_id=None,
    )

    assert submission.job is not None
    assert submission.job.job_id == _NEW_JOB_ID


@pytest.mark.parametrize(
    "policy_output",
    [
        "",
        "Warning: day/night boundary crossed by an unrelated job\n",
        "testing usage policy conformance\nError: policy database could not be reached\n",
        "testing usage policy conformance\nFATAL: report aborted\n",
    ],
)
def test_policy_output_without_clean_completion_blocks_submission(
    tmp_path: Path, policy_output: str
) -> None:
    calls: list[tuple[str, ...]] = []

    with pytest.raises(RuntimeError, match="refusing Grid'5000 submission"):
        _submit(tmp_path, _fake_runner(calls, policy=policy_output), state_job_id=None)

    assert _oarsub_calls(calls) == []


def test_active_eunis_job_on_any_site_is_reported_with_site_job_and_state(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, ...]] = []
    report = json.dumps(
        {
            "active_jobs": [
                {
                    "site": "lyon",
                    "job_id": "777",
                    "state": "running",
                    "name": "osm-polygon-eunis",
                }
            ],
            "errors": [],
            "excluded_sites": [],
        }
    )

    with pytest.raises(
        RuntimeError,
        match="active EUNIS Grid'5000 job\\(s\\) already exist: lyon:777 \\(running\\)",
    ):
        _submit(
            tmp_path,
            _fake_runner(calls, all_site_report=report),
            state_job_id=None,
        )

    assert _oarsub_calls(calls) == []


def test_remote_policy_and_inventory_scripts_are_pinned() -> None:
    policy = list(grid5000.build_policy_command(("nancy", "lyon")))
    inventory = list(grid5000.build_active_eunis_jobs_command(_FRONTEND, ("nancy", "lyon")))

    assert hashlib.sha256(json.dumps(policy).encode()).hexdigest() == (
        "19e99c3c8b4643608d905d4b6f7415534b5b9d0ce15ec98697ec8e97be52698b"
    )
    assert hashlib.sha256(json.dumps(inventory).encode()).hexdigest() == (
        "0f2afc8eb61ccb17503a3e353d672bc37eb638a58b044b13a778f68433edbf81"
    )


def test_submission_result_type_is_unchanged(tmp_path: Path) -> None:
    calls: list[tuple[str, ...]] = []

    submission = _submit(tmp_path, _fake_runner(calls), state_job_id=None)

    assert isinstance(submission, Grid5000Submission)
    assert submission.source_revision == "abc123"
