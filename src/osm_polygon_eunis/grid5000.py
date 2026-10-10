"""Safe, testable command construction for the Grid'5000 release workflow."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final

from .fileio import write_json_atomic
from .grid5000_commands import Command, CommandRunner, build_ssh_command, validate_host
from .grid5000_oar import (
    build_status_command,
    job_status_state,
    parse_job_id,
    status_error_output,
    validate_job_id,
)
from .grid5000_policy import (
    build_active_eunis_jobs_command,
    build_policy_command,
    reject_active_eunis_jobs,
    reject_unclean_policy_output,
    validate_excluded_sites,
)
from .options import DEFAULT_BATCH_SIZE

DEFAULT_GRID_CORES = 16
DEFAULT_GRID_WORKERS = 16

_WALLTIME_PATTERN: Final = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")
_PERSISTENT_PREFIXES: Final = ("/home/", "/groups/", "/srv/")
_ONE_HOUR_SECONDS: Final = 60 * 60
_DEFAULT_STOP_MARGIN_SECONDS: Final = 300
DEFAULT_DATASETS: Final = ("website", "wikidata", "description")


@dataclass(frozen=True, slots=True)
class Grid5000Config:
    """Immutable resource and storage settings for one OAR submission."""

    frontend: str
    persistent_root: str
    site: str
    cluster: str
    queue: str = "default"
    job_type: str | None = None
    cores: int = DEFAULT_GRID_CORES
    workers: int = DEFAULT_GRID_WORKERS
    walltime: str = "1:00:00"
    batch_size: int = DEFAULT_BATCH_SIZE
    excluded_sites: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Validate the requested resources and persistent paths."""
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
    validate_host(config.frontend, "frontend")
    validate_persistent_root(config.persistent_root)
    _validate_positive("cores", config.cores)
    _validate_positive("workers", config.workers)
    if config.workers > config.cores:
        raise ValueError("workers must not exceed cores")
    validate_walltime(config.walltime)
    _validate_positive("batch_size", config.batch_size)
    for field_name in ("site", "cluster", "queue"):
        validate_host(getattr(config, field_name), field_name)
    if config.job_type is not None:
        validate_host(config.job_type, "job_type")
    _validate_config_exclusions(config)


def validate_walltime(walltime: str) -> None:
    """Raise ValueError unless walltime is HH:MM:SS, within one hour, above the stop margin."""

    match = _WALLTIME_PATTERN.fullmatch(walltime)
    if match is None:
        raise ValueError("walltime must use HH:MM:SS")
    hours, minutes, seconds = (int(part) for part in match.groups())
    walltime_seconds = hours * 3600 + minutes * 60 + seconds
    if not 0 < walltime_seconds <= _ONE_HOUR_SECONDS:
        raise ValueError("walltime must be positive and no longer than one hour")
    if walltime_seconds <= _DEFAULT_STOP_MARGIN_SECONDS:
        raise ValueError("walltime must exceed the default 300-second stop margin")


def _validate_config_exclusions(config: Grid5000Config) -> None:
    excluded_sites = validate_excluded_sites(config.excluded_sites)
    if config.site in excluded_sites:
        raise ValueError("selected compute site cannot be in excluded_sites")


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


def build_oarsub_command(
    config: Grid5000Config,
    script: str,
    *,
    source_revision: str | None = None,
) -> Command:
    """Build a one-host, CPU-only OAR submission command."""

    if not script or not script.startswith("/"):
        raise ValueError("worker script must be an absolute remote path")
    command = ["oarsub", "-q", config.queue]
    command.extend(("-n", "osm-polygon-eunis"))
    if config.job_type is not None:
        command.extend(("-t", config.job_type))
    worker_command = " ".join(
        (*_submission_environment(config, source_revision), shlex.quote(script))
    )
    command.extend(
        (
            "-p",
            f"cluster='{config.cluster}'",
            "-l",
            f"host=1/core={config.cores},walltime={config.walltime}",
            worker_command,
        )
    )
    return tuple(command)


def _submission_environment(
    config: Grid5000Config,
    source_revision: str | None,
) -> list[str]:
    environment = [
        f"GRID5000_PERSISTENT_ROOT={shlex.quote(config.persistent_root)}",
        f"GRID5000_FRONTEND={shlex.quote(config.frontend)}",
        f"GRID5000_SITE={shlex.quote(config.site)}",
        f"GRID5000_CLUSTER={shlex.quote(config.cluster)}",
        f"GRID5000_QUEUE={shlex.quote(config.queue)}",
        f"GRID5000_CORES={config.cores}",
        f"GRID5000_WORKERS={config.workers}",
        f"GRID5000_WALLTIME={shlex.quote(config.walltime)}",
        f"GRID5000_BATCH_SIZE={config.batch_size}",
        f"GRID5000_EXCLUDED_SITES={shlex.quote(json.dumps(config.excluded_sites))}",
    ]
    if config.job_type is not None:
        environment.append(f"GRID5000_JOB_TYPE={shlex.quote(config.job_type)}")
    if source_revision is not None:
        _validate_source_revision(source_revision)
        environment.append(f"GRID5000_SOURCE_REVISION={shlex.quote(source_revision)}")
    return environment


def build_source_sync_command(
    local_root: Path,
    frontend: str,
    persistent_root: str,
    source_revision: str,
) -> Command:
    """Stream one exact Git commit to a staged, tracked-files-only source tree."""

    remote_root = validate_persistent_root(persistent_root)
    source_root = f"{remote_root.rstrip('/')}/source"
    stage_root = f"{remote_root.rstrip('/')}/.eunis-source-stage.XXXXXX"
    remote_script = f"""set -eu
target={shlex.quote(source_root)}
parent={shlex.quote(remote_root)}
stage="$(mktemp -d {shlex.quote(stage_root)})"
backup="$parent/.eunis-source-backup.$$"
cleanup() {{
  if [ -d "$stage" ]; then rm -rf -- "$stage"; fi
  if [ -e "$backup" ] && [ ! -e "$target" ]; then mv -- "$backup" "$target"; fi
}}
trap cleanup EXIT
if [ -e "$backup" ] || [ -L "$target" ] || {{ [ -e "$target" ] && [ ! -d "$target" ]; }}; then
  echo 'unsafe existing Grid source path' >&2
  exit 2
fi
tar -xf - -C "$stage"
if [ -e "$target" ]; then mv -- "$target" "$backup"; fi
if ! mv -- "$stage" "$target"; then
  if [ -e "$backup" ]; then mv -- "$backup" "$target"; fi
  exit 1
fi
stage=''
if [ -e "$backup" ]; then rm -rf -- "$backup"; fi
trap - EXIT
"""
    remote_command = build_ssh_command(frontend, ("bash", "-c", remote_script))
    archive_command = (
        "git",
        "-C",
        str(Path(local_root).resolve()),
        "archive",
        "--format=tar",
        source_revision,
    )
    pipeline = f"{shlex.join(archive_command)} | {shlex.join(remote_command)}"
    return ("bash", "-o", "pipefail", "-c", pipeline)


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
        raise ValueError(f"Grid'5000 job state has no job ID: {state_path}")  # noqa: TRY004 -- malformed persisted state is a value-validation error
    return validate_job_id(payload["job_id"])


def _reject_active_state(config: Grid5000Config, state_path: Path, runner: CommandRunner) -> None:
    job_id = _state_job_id(state_path)
    output = _read_job_status(config, job_id, runner)
    state = job_status_state(output)
    if state == "active":
        raise RuntimeError(f"Grid'5000 job {job_id} is already active")
    if state not in {"terminal", "missing"}:
        raise RuntimeError(
            f"cannot verify Grid'5000 job {job_id} status through {config.frontend}: "
            "OAR returned no recognized active or terminal state"
        )


def _read_job_status(config: Grid5000Config, job_id: str, runner: CommandRunner) -> str:
    try:
        return runner(build_ssh_command(config.frontend, build_status_command(job_id)))
    except subprocess.CalledProcessError as error:
        output = status_error_output(error)
        if job_status_state(output) == "missing":
            return output
        raise RuntimeError(
            f"cannot verify Grid'5000 job {job_id} status through {config.frontend}: "
            f"{output.strip() or 'status command failed'}"
        ) from error


def _write_job_state(path: Path, job: Grid5000Job) -> None:
    payload = {
        "cluster": job.config.cluster,
        "datasets": list(DEFAULT_DATASETS),
        "excluded_sites": list(job.config.excluded_sites),
        "frontend": job.config.frontend,
        "job_id": job.job_id,
        "batch_size": job.config.batch_size,
        "cores": job.config.cores,
        "job_type": job.config.job_type,
        "persistent_root": job.config.persistent_root,
        "queue": job.config.queue,
        "site": job.config.site,
        "source_revision": job.source_revision,
        "submitted_at": job.submitted_at,
        "walltime": job.config.walltime,
        "workers": job.config.workers,
    }
    write_json_atomic(path, payload)


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
    status = runner(("git", "-C", root, "status", "--porcelain"))
    return any(not line.startswith("?? ") for line in status.splitlines())


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
    state_path = _validate_live_submission_state(state_path, dry_run=dry_run)

    commands = _submission_commands(config, local_root, source_revision)
    if dry_run:
        return Grid5000Submission(None, DEFAULT_DATASETS, source_revision, commands)

    policy_output = command_runner(commands[0])
    reject_unclean_policy_output(policy_output)
    reject_active_eunis_jobs(command_runner(commands[1]), config.excluded_sites)
    _reject_existing_state(config, state_path, command_runner)
    command_runner(commands[2])
    command_runner(commands[3])
    job_id = parse_job_id(command_runner(commands[4]))
    job = Grid5000Job(
        job_id=job_id,
        submitted_at=datetime.now(UTC).isoformat(),
        config=config,
        source_revision=source_revision,
    )
    _write_optional_state(state_path, job)
    command_runner(commands[5])
    return Grid5000Submission(job, DEFAULT_DATASETS, source_revision, commands)


def _validate_source_revision(source_revision: str) -> None:
    if not source_revision or any(character.isspace() for character in source_revision):
        raise ValueError("source_revision must be a non-empty token")


def _validate_live_submission_state(state_path: Path | None, *, dry_run: bool) -> Path | None:
    if not dry_run and state_path is None:
        raise ValueError("state_path is required for a live Grid'5000 submission")
    return state_path


def _reject_existing_state(
    config: Grid5000Config, state_path: Path | None, runner: CommandRunner
) -> None:
    if state_path is not None and state_path.exists():
        _reject_active_state(config, state_path, runner)


def _submission_commands(
    config: Grid5000Config,
    local_root: Path,
    source_revision: str,
) -> tuple[Command, ...]:
    return (
        build_ssh_command(config.frontend, build_policy_command(config.excluded_sites)),
        build_active_eunis_jobs_command(config.frontend, config.excluded_sites),
        build_ssh_command(config.frontend, ("mkdir", "-p", config.persistent_root)),
        build_source_sync_command(
            local_root,
            config.frontend,
            config.persistent_root,
            source_revision,
        ),
        build_ssh_command(
            config.frontend,
            build_oarsub_command(
                config,
                _remote_worker_script(config),
                source_revision=source_revision,
            ),
        ),
        build_ssh_command(config.frontend, build_policy_command(config.excluded_sites)),
    )


def _write_optional_state(path: Path | None, job: Grid5000Job) -> None:
    if path is not None:
        _write_job_state(path, job)


def run_command(command: Command, *, cwd: Path | None = None) -> str:
    """Run one argument-array command and return combined output."""

    # The command is an argv tuple built from validated Grid configuration.
    completed = subprocess.run(  # noqa: S603
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout + completed.stderr
