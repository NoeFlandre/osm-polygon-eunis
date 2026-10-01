#!/usr/bin/env python3
"""Read the authoritative start time for a running Grid'5000 OAR job."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections.abc import Mapping, Sequence

_JOB_ID = re.compile(r"^[1-9][0-9]*$")
_FRONTEND = re.compile(r"^[A-Za-z0-9][A-Za-z0-9.-]*$")
_QUERY_TIMEOUT_SECONDS = 10


def parse_oar_job_start_epoch(payload: object, job_id: str) -> int:
    """Return this job's positive OAR ``start_time`` from ``oarstat --json``."""

    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid OAR job ID")
    job: object = None
    if isinstance(payload, Mapping):
        job = payload.get(job_id)
        if job is None and payload.get("id") == int(job_id):
            job = payload
        if job is None and isinstance(payload.get("items"), list):
            job = next(
                (
                    item
                    for item in payload["items"]
                    if isinstance(item, Mapping) and item.get("job_id") == int(job_id)
                ),
                None,
            )
    elif isinstance(payload, list):
        job = next(
            (
                item
                for item in payload
                if isinstance(item, Mapping) and item.get("job_id", item.get("id")) == int(job_id)
            ),
            None,
        )
    if job is None:
        raise ValueError("OAR start time is missing for this job")
    if not isinstance(job, Mapping):
        raise TypeError("OAR job entry has an invalid type")
    start_time = job.get("start_time")
    if type(start_time) is not int or start_time <= 0:
        raise ValueError("OAR start time is missing or invalid")
    return start_time


def _query_start_epoch(command: Sequence[str], job_id: str) -> int | None:
    try:
        result = subprocess.run(  # noqa: S603
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=_QUERY_TIMEOUT_SECONDS,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    try:
        return parse_oar_job_start_epoch(json.loads(result.stdout), job_id)
    except (TypeError, ValueError):
        return None


def read_oar_job_start_epoch(job_id: str, *, frontend: str) -> int:
    """Read OAR start time locally, then through the site's frontend."""

    if not _JOB_ID.fullmatch(job_id):
        raise ValueError("invalid OAR job ID")
    if not _FRONTEND.fullmatch(frontend):
        raise ValueError("invalid Grid'5000 frontend")

    local = _query_start_epoch(["oarstat", "-j", job_id, "--json"], job_id)
    if local is not None:
        print("grid5000_job_started_epoch_source=oarstat", file=sys.stderr)
        return local

    remote = _query_start_epoch(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=5",
            frontend,
            "oarstat",
            "-j",
            job_id,
            "--json",
        ],
        job_id,
    )
    if remote is not None:
        print("grid5000_job_started_epoch_source=frontend-oarstat", file=sys.stderr)
        return remote
    raise RuntimeError(f"could not read OAR start time for job {job_id}")


def main(argv: Sequence[str] | None = None) -> int:
    """Print the actual OAR start epoch for one job."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--frontend", required=True)
    arguments = parser.parse_args(argv)
    try:
        print(read_oar_job_start_epoch(arguments.job_id, frontend=arguments.frontend))
    except (RuntimeError, ValueError) as error:
        print(error, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
