#!/usr/bin/env python3
"""Write a failure receipt without trusting optional deadline overrides."""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

_DEADLINE_EXIT_STATUS = 124
_INTERRUPT_EXIT_STATUS = 130
_TERMINATION_EXIT_STATUS = 143


def _optional_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _optional_float(value: str) -> float | None:
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) and result >= 0 else None


def _excluded_sites(value: str) -> list[str]:
    try:
        parsed: Any = json.loads(value)
    except json.JSONDecodeError:
        return []
    if not isinstance(parsed, list) or not all(isinstance(item, str) for item in parsed):
        return []
    return parsed


def _read_stop_state(path: str) -> str | None:
    try:
        return Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None


def _incomplete_reason(
    exit_status: int | None,
    stop_state: str | None,
    external_signal: str,
) -> str | None:
    signal_name = external_signal.casefold()
    if exit_status == _DEADLINE_EXIT_STATUS and stop_state == "deadline":
        return "graceful_deadline"
    if exit_status == _INTERRUPT_EXIT_STATUS and (
        stop_state == "signal:2" or signal_name in {"int", "sigint", "2"}
    ):
        return "interrupt"
    if exit_status == _TERMINATION_EXIT_STATUS and (
        stop_state == "signal:15" or signal_name in {"term", "sigterm", "15"}
    ):
        return "termination_signal"
    return None


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(payload, output, sort_keys=True, indent=2, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        Path(temporary_name).replace(path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            Path(temporary_name).unlink()
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output",
        "job-id",
        "source-commit",
        "config-sha256",
        "config-path",
        "site",
        "frontend",
        "cluster",
        "queue",
        "cores",
        "workers",
        "walltime",
        "batch-size",
        "excluded-sites-json",
        "attempts",
        "error-count",
        "exit-status",
        "job-started-epoch",
        "stop-margin-seconds",
        "termination-grace-seconds",
        "log-path",
        "stop-marker",
        "external-signal",
    ):
        parser.add_argument(f"--{name}", required=True)
    return parser


def main() -> int:
    """Write one atomic failure receipt from worker and job metadata."""

    args = _parser().parse_args()
    exit_status = _optional_int(args.exit_status)
    stop_state = _read_stop_state(args.stop_marker)
    stop_reason = _incomplete_reason(exit_status, stop_state, args.external_signal)
    attempt_count = max(0, _optional_int(args.attempts) or 0)
    error_count = max(0, _optional_int(args.error_count) or 0)
    if exit_status not in {None, 0} and error_count == 0 and stop_reason is None:
        error_count = 1

    payload: dict[str, object] = {
        "status": "incomplete" if stop_reason is not None else "failed",
        "datasets": ["website", "wikidata", "description"],
        "grid5000": {
            "job_id": args.job_id,
            "source_commit": args.source_commit,
            "site": args.site,
            "frontend": args.frontend,
            "cluster": args.cluster,
            "queue": args.queue,
            "cores": _optional_int(args.cores),
            "workers": _optional_int(args.workers),
            "walltime": args.walltime,
            "batch_size": _optional_int(args.batch_size),
            "excluded_sites": _excluded_sites(args.excluded_sites_json),
            "config": {"path": args.config_path, "sha256": args.config_sha256},
            "attempts": attempt_count,
            "retries": max(0, attempt_count - 1),
            "errors": {"count": error_count, "last_exit_status": exit_status},
            "stop_reason": stop_reason or "worker_failure",
            "job_started_epoch": _optional_int(args.job_started_epoch),
            "stop_margin_seconds": _optional_int(args.stop_margin_seconds),
            "requested_stop_margin_seconds": args.stop_margin_seconds,
            "termination_grace_seconds": _optional_float(args.termination_grace_seconds),
            "requested_termination_grace_seconds": args.termination_grace_seconds,
            "log_path": args.log_path,
        },
    }
    _write_json_atomic(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
