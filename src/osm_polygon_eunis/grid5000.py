"""Safe, testable command construction for the Grid'5000 release workflow."""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

Command = tuple[str, ...]
CommandRunner = Callable[[Command], str]

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


@dataclass(frozen=True, slots=True)
class Grid5000Submission:
    """Result of a submission or a no-contact dry run."""

    job: Grid5000Job | None
    dataset: str
    source_revision: str
    commands: tuple[Command, ...]


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
        "--exclude=.eunis-run",
        "--exclude=.eunis-run-final",
        "--exclude=.env",
        "--exclude=.cache",
        "--exclude=.uv-cache",
        "--exclude=data",
        "--exclude=results",
        "--exclude=artifacts",
        "--exclude=__pycache__",
        "--exclude=.pytest_cache",
        f"{source.as_posix().rstrip('/')}/",
        f"{frontend}:{remote.rstrip('/')}/",
    )


def _remote_source_root(config: Grid5000Config) -> str:
    return f"{config.persistent_root.rstrip('/')}/source"


def _remote_worker_script(config: Grid5000Config) -> str:
    return f"{_remote_source_root(config)}/scripts/grid5000/description-release.sh"


def _state_job_id(state_path: Path) -> str:
    try:
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read Grid'5000 job state {state_path}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("job_id"), str):
        raise ValueError(f"Grid'5000 job state has no job ID: {state_path}")
    return _validate_job_id(payload["job_id"])


def _reject_active_state(
    config: Grid5000Config, state_path: Path, runner: CommandRunner
) -> None:
    job_id = _state_job_id(state_path)
    try:
        runner(build_ssh_command(config.frontend, build_status_command(job_id)))
    except subprocess.CalledProcessError:
        return
    raise RuntimeError(f"Grid'5000 job {job_id} is already active")


def _write_job_state(path: Path, job: Grid5000Job, *, dataset: str) -> None:
    payload = {
        "dataset": dataset,
        "frontend": job.config.frontend,
        "job_id": job.job_id,
        "persistent_root": job.config.persistent_root,
        "source_revision": job.source_revision,
        "submitted_at": job.submitted_at,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_source_revision(
    local_root: Path,
    *,
    explicit: str | None = None,
    allow_dirty: bool = False,
    runner: CommandRunner | None = None,
) -> str:
    """Resolve a source commit and refuse uncommitted code by default."""

    command_runner = runner or run_command
    root = str(local_root.resolve())
    status = command_runner(("git", "-C", root, "status", "--porcelain"))
    dirty = bool(status.strip())
    if dirty and not allow_dirty:
        raise RuntimeError("source tree is dirty; commit it or pass --allow-dirty-source")
    revision = explicit or command_runner(("git", "-C", root, "rev-parse", "HEAD")).strip()
    if not revision or any(character.isspace() for character in revision):
        raise ValueError("source revision must be a non-empty token")
    return f"{revision}-dirty" if dirty else revision


def submit_grid5000(
    config: Grid5000Config,
    local_root: Path,
    *,
    source_revision: str,
    runner: CommandRunner | None = None,
    state_path: Path | None = None,
    dry_run: bool = False,
) -> Grid5000Submission:
    """Submit the description worker after policy and duplicate checks."""

    if not source_revision or any(character.isspace() for character in source_revision):
        raise ValueError("source_revision must be a non-empty token")
    command_runner = runner or run_command
    if state_path is not None and state_path.exists():
        _reject_active_state(config, state_path, command_runner)

    source_root = _remote_source_root(config)
    commands = (
        build_ssh_command(config.frontend, build_policy_command()),
        build_ssh_command(config.frontend, ("mkdir", "-p", source_root)),
        build_rsync_command(local_root, config.frontend, source_root),
        build_ssh_command(
            config.frontend,
            build_oarsub_command(config, _remote_worker_script(config)),
        ),
        build_ssh_command(config.frontend, build_policy_command()),
    )
    if dry_run:
        return Grid5000Submission(None, "description", source_revision, commands)

    command_runner(commands[0])
    command_runner(commands[1])
    command_runner(commands[2])
    job_id = parse_job_id(command_runner(commands[3]))
    command_runner(commands[4])
    job = Grid5000Job(
        job_id=job_id,
        submitted_at=datetime.now(UTC).isoformat(),
        config=config,
        source_revision=source_revision,
    )
    if state_path is not None:
        _write_job_state(state_path, job, dataset="description")
    return Grid5000Submission(job, "description", source_revision, commands)


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
