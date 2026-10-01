#!/usr/bin/env python3
"""Run one worker command with a graceful deadline before its OAR walltime."""

from __future__ import annotations

import argparse
import math
import os
import re
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from contextlib import suppress
from pathlib import Path

_WALLTIME = re.compile(r"^(\d+):([0-5]\d):([0-5]\d)$")
_POLL_INTERVAL_SECONDS = 0.1
_RECEIPT_WINDOW_SECONDS = 30
_ONE_HOUR_SECONDS = 60 * 60


def parse_walltime_seconds(walltime: str) -> int:
    """Parse a positive OAR walltime no longer than one hour."""

    match = _WALLTIME.fullmatch(walltime)
    if match is None:
        raise ValueError("walltime must use H:MM:SS with minutes and seconds from 00 to 59")
    hours, minutes, seconds = (int(part) for part in match.groups())
    total_seconds = hours * 3600 + minutes * 60 + seconds
    if not 0 < total_seconds <= _ONE_HOUR_SECONDS:
        raise ValueError("walltime must be positive and no longer than one hour")
    return total_seconds


def validate_deadline_settings(
    walltime: str,
    *,
    stop_margin_seconds: int,
    termination_grace_seconds: int,
) -> None:
    """Require shutdown grace and a receipt buffer before the OAR deadline."""

    walltime_seconds = parse_walltime_seconds(walltime)
    if stop_margin_seconds <= 0:
        raise ValueError("stop margin must be positive")
    if termination_grace_seconds < 0:
        raise ValueError("termination grace must be non-negative")
    if stop_margin_seconds < termination_grace_seconds + _RECEIPT_WINDOW_SECONDS:
        raise ValueError(
            "stop margin must preserve termination grace plus a 30-second receipt window"
        )
    if stop_margin_seconds >= walltime_seconds:
        raise ValueError("stop margin must be less than walltime")


def remaining_runtime_seconds(
    walltime: str,
    *,
    started_at: float,
    stop_margin_seconds: int,
    now: float | None = None,
) -> int:
    """Return time left before the internal stop, including prior setup time."""

    if not math.isfinite(started_at):
        raise ValueError("job start time must be finite")
    walltime_seconds = parse_walltime_seconds(walltime)
    if not 0 < stop_margin_seconds < walltime_seconds:
        raise ValueError("stop margin must be positive and less than walltime")
    current_time = time.time() if now is None else now
    if not math.isfinite(current_time):
        raise ValueError("current time must be finite")
    remaining = walltime_seconds - (current_time - started_at)
    remaining -= stop_margin_seconds
    return max(0, math.floor(remaining))


def _process_group_exists(process_group_id: int) -> bool:
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _terminate_process_group(
    process: subprocess.Popen[bytes],
    *,
    termination_signal: int,
    grace_seconds: float,
) -> None:
    with suppress(ProcessLookupError):
        os.killpg(process.pid, termination_signal)

    grace_deadline = time.monotonic() + grace_seconds
    while time.monotonic() < grace_deadline and _process_group_exists(process.pid):
        process.poll()
        time.sleep(min(_POLL_INTERVAL_SECONDS, max(0.0, grace_deadline - time.monotonic())))

    if _process_group_exists(process.pid):
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
    process.wait()


def _shell_status(return_code: int) -> int:
    return 128 + abs(return_code) if return_code < 0 else return_code


def run_command(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
    on_stop: Callable[[str], None] | None = None,
) -> int:
    """Run a command group, returning 124 on deadline or shell signal status."""

    if not command:
        raise ValueError("worker command must not be empty")
    if not math.isfinite(timeout_seconds) or timeout_seconds < 0:
        raise ValueError("timeout must be a finite non-negative number")
    if not math.isfinite(termination_grace_seconds) or termination_grace_seconds < 0:
        raise ValueError("termination grace must be a finite non-negative number")
    if timeout_seconds == 0:
        if on_stop is not None:
            on_stop("deadline")
        return 124

    return _run_with_signal_handlers(
        command,
        timeout_seconds=timeout_seconds,
        termination_grace_seconds=termination_grace_seconds,
        on_stop=on_stop,
    )


def _run_with_signal_handlers(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
    on_stop: Callable[[str], None] | None,
) -> int:
    received_signal: int | None = None

    def handle_signal(signum: int, _frame: object) -> None:
        nonlocal received_signal
        received_signal = signum

    watched_signals = (signal.SIGINT, signal.SIGTERM)
    original_handlers = {signum: signal.getsignal(signum) for signum in watched_signals}
    try:
        for signum in watched_signals:
            signal.signal(signum, handle_signal)
        return _spawn_and_wait(
            command,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
            received_signal=lambda: received_signal,
            on_stop=on_stop,
        )
    finally:
        for signum, handler in original_handlers.items():
            signal.signal(signum, handler)


def _spawn_and_wait(
    command: Sequence[str],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
    received_signal: Callable[[], int | None],
    on_stop: Callable[[str], None] | None,
) -> int:
    process = subprocess.Popen(command, start_new_session=True)  # noqa: S603
    try:
        return _wait_for_command(
            process,
            timeout_seconds=timeout_seconds,
            termination_grace_seconds=termination_grace_seconds,
            received_signal=received_signal,
            on_stop=on_stop,
        )
    except BaseException:
        if process.poll() is None:
            _terminate_process_group(
                process,
                termination_signal=signal.SIGTERM,
                grace_seconds=termination_grace_seconds,
            )
        raise


def _wait_for_command(
    process: subprocess.Popen[bytes],
    *,
    timeout_seconds: float,
    termination_grace_seconds: float,
    received_signal: Callable[[], int | None],
    on_stop: Callable[[str], None] | None,
) -> int:
    deadline = time.monotonic() + timeout_seconds
    while True:
        signum = received_signal()
        if signum is not None:
            _terminate_process_group(
                process,
                termination_signal=signum,
                grace_seconds=termination_grace_seconds,
            )
            if on_stop is not None:
                on_stop(f"signal:{signum}")
            return 128 + signum

        return_code = process.poll()
        if return_code is not None:
            return _shell_status(return_code)

        remaining = deadline - time.monotonic()
        if remaining <= 0:
            _terminate_process_group(
                process,
                termination_signal=signal.SIGTERM,
                grace_seconds=termination_grace_seconds,
            )
            if on_stop is not None:
                on_stop("deadline")
            return 124
        time.sleep(min(_POLL_INTERVAL_SECONDS, remaining))


def _write_stop_marker(path: Path, state: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(f"{state}\n", encoding="utf-8")
    temporary.replace(path)


def _non_negative_integer(value: str) -> int:
    try:
        result = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a non-negative integer") from error
    if result < 0:
        raise argparse.ArgumentTypeError("expected a non-negative integer")
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--walltime", required=True, help="OAR walltime in H:MM:SS form")
    parser.add_argument("--started-at", required=True, type=float, help="OAR job start epoch")
    parser.add_argument(
        "--stop-margin-seconds",
        type=_non_negative_integer,
        default=300,
        help="seconds reserved before OAR walltime for receipts and shutdown (default: 300)",
    )
    parser.add_argument(
        "--termination-grace-seconds",
        type=_non_negative_integer,
        default=20,
        help="seconds between TERM and KILL (default: 20)",
    )
    parser.add_argument("--stop-marker", type=Path, help="write the reason for a guarded stop")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="worker command after --")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Calculate the remaining execution window, then run the worker."""

    parser = _parser()
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command and args.command[0] == "--" else args.command
    if not command:
        parser.error("provide a worker command after --")
    if args.stop_marker is not None:
        try:
            args.stop_marker.unlink(missing_ok=True)
        except OSError as error:
            parser.error(f"cannot clear stop marker: {error}")
    try:
        validate_deadline_settings(
            args.walltime,
            stop_margin_seconds=args.stop_margin_seconds,
            termination_grace_seconds=args.termination_grace_seconds,
        )
        timeout_seconds = remaining_runtime_seconds(
            args.walltime,
            started_at=args.started_at,
            stop_margin_seconds=args.stop_margin_seconds,
        )
    except ValueError as error:
        parser.error(str(error))
    return run_command(
        command,
        timeout_seconds=timeout_seconds,
        termination_grace_seconds=args.termination_grace_seconds,
        on_stop=(
            (lambda state: _write_stop_marker(args.stop_marker, state))
            if args.stop_marker is not None
            else None
        ),
    )


if __name__ == "__main__":
    sys.exit(main())
