"""Parse OAR submission and status output for Grid'5000 jobs.

These functions read text only. They never contact Grid'5000 and never decide
whether a submission may proceed; grid5000.py makes that decision.
"""

from __future__ import annotations

import re
import subprocess
from typing import Final

from .grid5000_commands import Command

_JOB_ID_PATTERN: Final = re.compile(r"(?:\bAdding job\s+|\bOAR_JOB_ID=)(\d+)\b")


def validate_job_id(job_id: str) -> str:
    """Return job_id unchanged when it contains only ASCII digits."""

    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job ID must contain only digits")
    return job_id


def parse_job_id(output: str) -> str:
    """Extract exactly one job ID from OAR's submission output."""

    matches = set(_JOB_ID_PATTERN.findall(output))
    if len(matches) != 1:
        raise ValueError("submission output must contain exactly one OAR job ID")
    return validate_job_id(matches.pop())


def build_status_command(job_id: str) -> Command:
    """Build a status query for one validated OAR job."""

    return ("oarstat", "-j", validate_job_id(job_id))


def status_error_output(error: subprocess.CalledProcessError) -> str:
    """Join the stdout and stderr captured from a failed OAR status command."""

    return "\n".join(str(value) for value in (error.stdout, error.stderr) if value is not None)


_OAR_TABLE_STATE = re.compile(
    r"^\d+\s.*?\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\s+([A-Za-z])\s", re.MULTILINE
)


_OAR_TABLE_STATES = {
    **dict.fromkeys("TE", "terminal"),
    **dict.fromkeys("WHLRSFA", "active"),
}


def _table_job_state(output: str) -> str | None:
    match = _OAR_TABLE_STATE.search(output)
    return None if match is None else _OAR_TABLE_STATES.get(match.group(1).upper())


def job_status_state(output: str) -> str:
    """Classify OAR status output as active, terminal, missing or unknown."""

    table_state = _table_job_state(output)
    if table_state is not None:
        return table_state
    if _job_status_is_active(output):
        return "active"
    if _job_status_is_missing(output):
        return "missing"
    if _job_status_is_terminal(output):
        return "terminal"
    return "unknown"


def _job_status_is_missing(output: str) -> bool:
    normalized = output.casefold()
    return any(
        message in normalized
        for message in ("no job", "unknown job", "job not found", "job does not exist")
    )


def _job_status_is_active(output: str) -> bool:
    return (
        re.search(
            r"\b(waiting|hold|tolaunch|launching|running|suspended|resuming|finishing)\b",
            output,
            re.IGNORECASE,
        )
        is not None
    )


def _job_status_is_terminal(output: str) -> bool:
    return re.search(r"\b(terminated|error)\b", output, re.IGNORECASE) is not None
