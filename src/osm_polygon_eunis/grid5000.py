"""Safe, testable command construction for the Grid'5000 release workflow."""

from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Final

Command = tuple[str, ...]

_JOB_ID_PATTERN: Final = re.compile(r"\bAdding job\s+(\d+)\b")
_WALLTIME_PATTERN: Final = re.compile(r"\d+:[0-5]\d:[0-5]\d")
_PERSISTENT_PREFIXES: Final = ("/home/", "/groups/", "/srv/")


@dataclass(frozen=True, slots=True)
class Grid5000Config:
    """Immutable resource and storage settings for one OAR submission."""

    frontend: str
    persistent_root: str
    site: str = "lille"
    cluster: str = "chuc"
    queue: str = "default"
    job_type: str = "night"
    cores: int = 16
    workers: int = 16
    walltime: str = "12:00:00"
    batch_size: int = 256

    def __post_init__(self) -> None:
        if not self.frontend or any(character.isspace() for character in self.frontend):
            raise ValueError("frontend must be a non-empty host name")
        validate_persistent_root(self.persistent_root)
        if self.cores <= 0:
            raise ValueError("cores must be positive")
        if self.workers <= 0:
            raise ValueError("workers must be positive")
        if self.workers > self.cores:
            raise ValueError("workers must not exceed cores")
        if not _WALLTIME_PATTERN.fullmatch(self.walltime):
            raise ValueError("walltime must use HH:MM:SS")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        for field_name in ("site", "cluster", "queue", "job_type"):
            value = getattr(self, field_name)
            if not value or any(character.isspace() for character in value):
                raise ValueError(f"{field_name} must be a non-empty value")


@dataclass(frozen=True, slots=True)
class Grid5000Job:
    """The non-secret state needed to monitor one submitted job."""

    job_id: str
    submitted_at: str
    config: Grid5000Config
    source_revision: str


def validate_persistent_root(path: str) -> str:
    """Validate a persistent Grid'5000 path and return normalized POSIX text."""

    if not path or "\x00" in path:
        raise ValueError("persistent root must be a non-empty absolute path")
    candidate = PurePosixPath(path)
    normalized = str(candidate)
    if not candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("persistent root must be an absolute path")
    if not normalized.startswith(_PERSISTENT_PREFIXES):
        raise ValueError(
            "persistent root must be on remote persistent storage under /home, /groups, or /srv"
        )
    if normalized in {"/home", "/groups", "/srv"}:
        raise ValueError("persistent root must name a project directory")
    return normalized


def build_policy_command() -> Command:
    """Build the required usage-policy check command."""

    return ("usagepolicycheck", "-t")


def build_oarsub_command(config: Grid5000Config, script: str) -> Command:
    """Build a one-host, CPU-only OAR submission command."""

    if not script or not script.startswith("/"):
        raise ValueError("worker script must be an absolute remote path")
    return (
        "oarsub",
        "-q",
        config.queue,
        "-t",
        config.job_type,
        "-p",
        config.cluster,
        "-l",
        f"host=1/core={config.cores},walltime={config.walltime}",
        "-S",
        script,
    )


def _validate_job_id(job_id: str) -> str:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job ID must contain only digits")
    return job_id


def parse_job_id(output: str) -> str:
    """Extract exactly one job ID from OAR's submission output."""

    matches = _JOB_ID_PATTERN.findall(output)
    if len(matches) != 1:
        raise ValueError("submission output must contain exactly one OAR job ID")
    return _validate_job_id(matches[0])


def build_status_command(job_id: str) -> Command:
    """Build a status query for one validated OAR job."""

    return ("oarstat", "-j", _validate_job_id(job_id))


def build_cancel_command(job_id: str) -> Command:
    """Build a cancellation command for one validated OAR job."""

    return ("oardel", _validate_job_id(job_id))


def build_ssh_command(frontend: str, command: Command) -> Command:
    """Prefix a remote command with SSH without invoking a local shell."""

    if not frontend or any(character.isspace() for character in frontend):
        raise ValueError("frontend must be a non-empty host name")
    if not command:
        raise ValueError("remote command must not be empty")
    return ("ssh", frontend, *command)


def build_rsync_command(local_root: Path, frontend: str, remote_root: str) -> Command:
    """Build a code-only sync command with secrets and run state excluded."""

    remote = validate_persistent_root(remote_root)
    source = Path(local_root).resolve()
    return (
        "rsync",
        "-az",
        "--exclude=.git",
        "--exclude=.venv",
        "--exclude=.eunis-run-final",
        "--exclude=.env",
        "--exclude=.cache",
        "--exclude=.uv-cache",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        f"{source.as_posix().rstrip('/')}/",
        f"{frontend}:{remote.rstrip('/')}/",
    )


def run_command(command: Command, *, cwd: Path | None = None) -> str:
    """Run one argument-array command and return stdout."""

    completed = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout
