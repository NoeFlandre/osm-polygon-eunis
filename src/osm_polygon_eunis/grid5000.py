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
_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PERSISTENT_PREFIXES: Final = ("/home/", "/groups/", "/srv/")
DEFAULT_DATASETS: Final = ("website", "wikidata", "description")


@dataclass(frozen=True, slots=True)
class Grid5000Config:
    """Immutable resource and storage settings for one OAR submission."""

    frontend: str
    persistent_root: str
    site: str
    cluster: str
    queue: str = "default"
    job_type: str = "night"
    cores: int = 16
    workers: int = 16
    walltime: str = "12:00:00"
    batch_size: int = 256

    def __post_init__(self) -> None:
        _validate_config_values(self)


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
    datasets: tuple[str, ...]
    source_revision: str
    commands: tuple[Command, ...]


def validate_persistent_root(path: str) -> str:
    """Validate a persistent Grid'5000 path and return normalized POSIX text."""

    _validate_path_text(path)
    candidate = PurePosixPath(path)
    normalized = str(candidate)
    _validate_absolute_path(candidate)
    _validate_persistent_prefix(normalized)
    _validate_project_root(normalized)
    return normalized


def _validate_config_values(config: Grid5000Config) -> None:
    _validate_host(config.frontend, "frontend")
    validate_persistent_root(config.persistent_root)
    _validate_positive("cores", config.cores)
    _validate_positive("workers", config.workers)
    if config.workers > config.cores:
        raise ValueError("workers must not exceed cores")
    if not _WALLTIME_PATTERN.fullmatch(config.walltime):
        raise ValueError("walltime must use HH:MM:SS")
    _validate_positive("batch_size", config.batch_size)
    for field_name in ("site", "cluster", "queue", "job_type"):
        _validate_host(getattr(config, field_name), field_name)


def _validate_host(value: str, field_name: str) -> None:
    if not _TOKEN_PATTERN.fullmatch(value):
        suffix = " host name" if field_name == "frontend" else " value"
        raise ValueError(f"{field_name} must be a non-empty{suffix}")


def _validate_positive(field_name: str, value: int) -> None:
    if value <= 0:
        raise ValueError(f"{field_name} must be positive")


def _validate_path_text(path: str) -> None:
    if not path or "\x00" in path:
        raise ValueError("persistent root must be a non-empty absolute path")


def _validate_absolute_path(path: PurePosixPath) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("persistent root must be an absolute path")


def _validate_persistent_prefix(path: str) -> None:
    if not path.startswith(_PERSISTENT_PREFIXES):
        raise ValueError(
            "persistent root must be on remote persistent storage under /home, /groups, or /srv"
        )


def _validate_project_root(path: str) -> None:
    if path in {"/home", "/groups", "/srv"}:
        raise ValueError("persistent root must name a project directory")


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
        f"cluster='{config.cluster}'",
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
        "--exclude=.grid5000-*.json",
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
    return f"{_remote_source_root(config)}/scripts/grid5000/release.sh"


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


def _write_job_state(path: Path, job: Grid5000Job) -> None:
    payload = {
        "cluster": job.config.cluster,
        "datasets": list(DEFAULT_DATASETS),
        "frontend": job.config.frontend,
        "job_id": job.job_id,
        "persistent_root": job.config.persistent_root,
        "site": job.config.site,
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

    command_runner = _command_runner(runner)
    root = str(local_root.resolve())
    dirty = _source_tree_is_dirty(command_runner, root)
    if dirty and not allow_dirty:
        raise RuntimeError("source tree is dirty; commit it or pass --allow-dirty-source")
    revision = _resolve_revision(command_runner, root, explicit)
    return f"{revision}-dirty" if dirty else revision


def _command_runner(runner: CommandRunner | None) -> CommandRunner:
    return run_command if runner is None else runner


def _source_tree_is_dirty(runner: CommandRunner, root: str) -> bool:
    return bool(runner(("git", "-C", root, "status", "--porcelain")).strip())


def _resolve_revision(runner: CommandRunner, root: str, explicit: str | None) -> str:
    revision = explicit or runner(("git", "-C", root, "rev-parse", "HEAD")).strip()
    if not revision or any(character.isspace() for character in revision):
        raise ValueError("source revision must be a non-empty token")
    return revision


def submit_grid5000(
    config: Grid5000Config,
    local_root: Path,
    *,
    source_revision: str,
    runner: CommandRunner | None = None,
    state_path: Path | None = None,
    dry_run: bool = False,
) -> Grid5000Submission:
    """Submit the all-source worker after policy and duplicate checks."""

    _validate_source_revision(source_revision)
    command_runner = _command_runner(runner)
    _reject_existing_state(config, state_path, command_runner)

    commands = _submission_commands(config, local_root)
    if dry_run:
        return Grid5000Submission(None, DEFAULT_DATASETS, source_revision, commands)

    job_id = _run_submission(commands, command_runner)
    job = Grid5000Job(
        job_id=job_id,
        submitted_at=datetime.now(UTC).isoformat(),
        config=config,
        source_revision=source_revision,
    )
    _write_optional_state(state_path, job)
    return Grid5000Submission(job, DEFAULT_DATASETS, source_revision, commands)


def _validate_source_revision(source_revision: str) -> None:
    if not source_revision or any(character.isspace() for character in source_revision):
        raise ValueError("source_revision must be a non-empty token")


def _reject_existing_state(
    config: Grid5000Config, state_path: Path | None, runner: CommandRunner
) -> None:
    if state_path is not None and state_path.exists():
        _reject_active_state(config, state_path, runner)


def _submission_commands(config: Grid5000Config, local_root: Path) -> tuple[Command, ...]:
    source_root = _remote_source_root(config)
    return (
        build_ssh_command(config.frontend, build_policy_command()),
        build_ssh_command(config.frontend, ("mkdir", "-p", source_root)),
        build_rsync_command(local_root, config.frontend, source_root),
        build_ssh_command(
            config.frontend,
            build_oarsub_command(config, _remote_worker_script(config)),
        ),
        build_ssh_command(config.frontend, build_policy_command()),
    )


def _run_submission(commands: tuple[Command, ...], runner: CommandRunner) -> str:
    runner(commands[0])
    runner(commands[1])
    runner(commands[2])
    job_id = parse_job_id(runner(commands[3]))
    runner(commands[4])
    return job_id


def _write_optional_state(path: Path | None, job: Grid5000Job) -> None:
    if path is not None:
        _write_job_state(path, job)


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
