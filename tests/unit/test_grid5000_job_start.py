from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from scripts.grid5000.job_start import (
    parse_oar_job_start_epoch,
    read_oar_job_start_epoch,
)


def _command(path: Path, body: str) -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def test_parse_oar_job_start_epoch_uses_the_requested_job() -> None:
    payload = {
        "41": {"start_time": 1_700_000_000, "state": "Running"},
        "42": {"start_time": 1_700_000_123, "state": "Running"},
    }

    assert parse_oar_job_start_epoch(payload, "42") == 1_700_000_123


@pytest.mark.parametrize(
    ("payload", "job_id"),
    [
        ({}, "42"),
        ({"42": {"start_time": 0}}, "42"),
        ({"42": {"start_time": "not-a-time"}}, "42"),
        ({"41": {"start_time": 1_700_000_000}}, "42"),
    ],
)
def test_parse_oar_job_start_epoch_rejects_missing_or_invalid_start(
    payload: object, job_id: str
) -> None:
    with pytest.raises(ValueError, match="start time"):
        parse_oar_job_start_epoch(payload, job_id)


def test_read_oar_job_start_epoch_uses_oarstat_when_available(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oarstat = tmp_path / "oarstat"
    _command(
        oarstat,
        'printf \'%s\\n\' \'{"42":{"start_time":1700000123,"state":"Running"}}\'',
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    assert read_oar_job_start_epoch("42", frontend="toulouse") == 1_700_000_123


def test_read_oar_job_start_epoch_uses_frontend_when_local_oarstat_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _command(tmp_path / "oarstat", "exit 1")
    ssh = tmp_path / "ssh"
    _command(
        ssh,
        'printf \'%s\\n\' \'{"42":{"start_time":1700000456,"state":"Running"}}\'',
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    assert read_oar_job_start_epoch("42", frontend="toulouse") == 1_700_000_456


def test_read_oar_job_start_epoch_uses_frontend_when_local_payload_is_malformed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _command(tmp_path / "oarstat", "printf '%s\\n' '{\"42\":\"invalid-job-entry\"}'")
    _command(
        tmp_path / "ssh",
        'printf \'%s\\n\' \'{"42":{"start_time":1700000678,"state":"Running"}}\'',
    )
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    assert read_oar_job_start_epoch("42", frontend="toulouse") == 1_700_000_678


def test_read_oar_job_start_epoch_fails_when_both_oar_queries_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _command(tmp_path / "oarstat", "exit 1")
    _command(tmp_path / "ssh", "exit 1")
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")

    with pytest.raises(RuntimeError, match="OAR start time"):
        read_oar_job_start_epoch("42", frontend="toulouse")


def test_job_start_cli_prints_only_the_epoch(tmp_path: Path) -> None:
    helper = Path(__file__).resolve().parents[2] / "scripts/grid5000/job_start.py"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    _command(
        fake_bin / "oarstat",
        'printf \'%s\\n\' \'{"42":{"start_time":1700000789,"state":"Running"}}\'',
    )
    env = {**os.environ, "PATH": f"{fake_bin}{os.pathsep}{os.environ['PATH']}"}

    result = subprocess.run(  # noqa: S603
        [sys.executable, str(helper), "--job-id", "42", "--frontend", "toulouse"],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "1700000789"
