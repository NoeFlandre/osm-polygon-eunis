"""Safe, testable command construction for the Grid'5000 release workflow."""

from __future__ import annotations

import json
import re
import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Final, TypeGuard

Command = tuple[str, ...]
CommandRunner = Callable[[Command], str]

_JOB_ID_PATTERN: Final = re.compile(r"(?:\bAdding job\s+|\bOAR_JOB_ID=)(\d+)\b")
_WALLTIME_PATTERN: Final = re.compile(r"(\d+):([0-5]\d):([0-5]\d)")
_TOKEN_PATTERN: Final = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_PERSISTENT_PREFIXES: Final = ("/home/", "/groups/", "/srv/")
_ONE_HOUR_SECONDS: Final = 60 * 60
_DEFAULT_STOP_MARGIN_SECONDS: Final = 300
DEFAULT_DATASETS: Final = ("website", "wikidata", "description")
_POLICY_CHECK_SCRIPT: Final = r"""import json
import subprocess
import sys
import urllib.parse
import urllib.request

BASE = "https://api.grid5000.fr/stable"
EXCLUDED_SITE_IDS = set(__EXCLUDED_SITE_IDS__)

def items(url):
    visited = set()
    while url:
        if url in visited:
            raise RuntimeError("Grid5000 API pagination loop")
        visited.add(url)
        request = urllib.request.Request(url, headers={"Accept": "application/json"})
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.load(response)
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("unexpected Grid5000 API collection response")
        yield from payload["items"]
        next_link = next(
            (link.get("href") for link in payload.get("links", [])
             if isinstance(link, dict) and link.get("rel") == "next"),
            None,
        )
        url = urllib.parse.urljoin(BASE + "/", next_link) if next_link else None

sites = list(items(BASE + "/sites"))
site_ids = set()
for site in sites:
    if not isinstance(site, dict):
        raise RuntimeError("unexpected Grid5000 site entry")
    site_id = site.get("uid") or site.get("id")
    if not isinstance(site_id, str) or not site_id:
        raise RuntimeError("Grid5000 site entry has no identifier")
    site_ids.add(site_id)
unknown_exclusions = sorted(EXCLUDED_SITE_IDS - site_ids)
if unknown_exclusions:
    raise RuntimeError("excluded site is absent from the current API inventory: "
                       + ",".join(unknown_exclusions))
included_site_ids = sorted(site_ids - EXCLUDED_SITE_IDS)
if not included_site_ids:
    raise RuntimeError("no Grid5000 sites remain in the policy check")
sys.stderr.write("Grid'5000 policy sites: " + ",".join(included_site_ids) + "\n")
result = subprocess.run(
    ["usagepolicycheck", "-t", "--sites", ",".join(included_site_ids)],
    check=False,
    capture_output=True,
    text=True,
)
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)
sys.exit(result.returncode)
"""
_ACTIVE_EUNIS_JOBS_SCRIPT: Final = r"""import json
import sys
import subprocess
import urllib.parse
import urllib.request

BASE = "https://api.grid5000.fr/stable"
TERMINAL = {"terminated", "error", "killed", "deleted", "finished", "completed"}
MARKERS = ("osm-polygon-eunis", "/scripts/grid5000/release.sh", "grid5000_source_revision")
EXCLUDED_SITE_IDS = set(__EXCLUDED_SITE_IDS__)

def get_json(url):
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    except Exception as error:
        raise RuntimeError(f"Grid5000 API request failed for {url}: {error}") from error

def items(url):
    visited = set()
    while url:
        if url in visited:
            raise RuntimeError("Grid5000 API pagination loop")
        visited.add(url)
        payload = get_json(url)
        if isinstance(payload, list):
            yield from payload
            return
        if not isinstance(payload, dict) or not isinstance(payload.get("items"), list):
            raise RuntimeError("unexpected Grid5000 API collection response")
        yield from payload["items"]
        links = payload.get("links", [])
        next_link = next(
            (link.get("href") for link in links
             if isinstance(link, dict) and link.get("rel") == "next"),
            None,
        )
        url = urllib.parse.urljoin(BASE + "/", next_link) if next_link else None

sites = list(items(BASE + "/sites"))
active_jobs = []
errors = []
site_ids = {
    site.get("uid") or site.get("id")
    for site in sites
    if isinstance(site, dict) and isinstance(site.get("uid") or site.get("id"), str)
}
for site_id in sorted(EXCLUDED_SITE_IDS - site_ids):
    errors.append(
        {"site": site_id, "error": "excluded site is absent from the current API inventory"}
    )
for site in sites:
    if not isinstance(site, dict):
        raise RuntimeError("unexpected Grid5000 site entry")
    site_id = site.get("uid") or site.get("id")
    if not isinstance(site_id, str) or not site_id:
        raise RuntimeError("Grid5000 site entry has no identifier")
    if site_id in EXCLUDED_SITE_IDS:
        continue
    try:
        result = subprocess.run(
            ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=8", site_id,
             "oarstat", "-u", "-J"],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        jobs = json.loads(result.stdout) if result.stdout.strip() else {}
        if not isinstance(jobs, dict):
            raise RuntimeError("oarstat did not return a job object")
        for job_id, job in jobs.items():
            if not isinstance(job, dict):
                raise RuntimeError("oarstat returned a malformed job entry")
            state = str(job.get("state", "")).casefold()
            searchable = " ".join(
                str(job.get(key, ""))
                for key in ("name", "command", "launching_directory", "initial_request")
            ).casefold()
            if state not in TERMINAL and any(marker in searchable for marker in MARKERS):
                active_jobs.append({
                    "site": site_id,
                    "job_id": str(job.get("id", job_id)),
                    "state": state or "unknown",
                    "name": job.get("name"),
                })
    except Exception as error:
        errors.append({"site": site_id, "error": f"{type(error).__name__}: {error}"})
sys.stdout.write(
    json.dumps(
        {
            "active_jobs": active_jobs,
            "errors": errors,
            "excluded_sites": sorted(EXCLUDED_SITE_IDS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    + "\n"
)
"""


@dataclass(frozen=True, slots=True)
class Grid5000Config:
    """Immutable resource and storage settings for one OAR submission."""

    frontend: str
    persistent_root: str
    site: str
    cluster: str
    queue: str = "default"
    job_type: str | None = None
    cores: int = 16
    workers: int = 16
    walltime: str = "1:00:00"
    batch_size: int = 256
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
    _validate_host(config.frontend, "frontend")
    validate_persistent_root(config.persistent_root)
    _validate_positive("cores", config.cores)
    _validate_positive("workers", config.workers)
    if config.workers > config.cores:
        raise ValueError("workers must not exceed cores")
    _validate_walltime(config.walltime)
    _validate_positive("batch_size", config.batch_size)
    for field_name in ("site", "cluster", "queue"):
        _validate_host(getattr(config, field_name), field_name)
    if config.job_type is not None:
        _validate_host(config.job_type, "job_type")
    _validate_config_exclusions(config)


def _validate_walltime(walltime: str) -> None:
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
    excluded_sites = _validate_excluded_sites(config.excluded_sites)
    if config.site in excluded_sites:
        raise ValueError("selected compute site cannot be in excluded_sites")


def _validate_excluded_sites(excluded_sites: tuple[str, ...]) -> tuple[str, ...]:
    if not isinstance(excluded_sites, tuple):
        raise TypeError("excluded_sites must be a tuple of site identifiers")
    for site_id in excluded_sites:
        _validate_host(site_id, "excluded_sites")
    if len(set(excluded_sites)) != len(excluded_sites):
        raise ValueError("excluded_sites must be unique")
    return tuple(sorted(excluded_sites))


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


def build_policy_command(excluded_sites: tuple[str, ...] = ()) -> Command:
    """Build a usage-policy check, explicitly omitting only approved sites."""

    exclusions = _validate_excluded_sites(excluded_sites)
    if not exclusions:
        return ("usagepolicycheck", "-t")
    script = _POLICY_CHECK_SCRIPT.replace("__EXCLUDED_SITE_IDS__", json.dumps(exclusions))
    return ("python3", "-c", script)


def build_active_eunis_jobs_command(
    frontend: str,
    excluded_sites: tuple[str, ...] = (),
) -> Command:
    """Query all API sites over OAR SSH except explicitly excluded sites."""

    exclusions = _validate_excluded_sites(excluded_sites)
    script = _ACTIVE_EUNIS_JOBS_SCRIPT.replace("__EXCLUDED_SITE_IDS__", json.dumps(exclusions))
    return build_ssh_command(frontend, ("python3", "-c", script))


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


def _validate_job_id(job_id: str) -> str:
    if not re.fullmatch(r"[0-9]+", job_id):
        raise ValueError("job ID must contain only digits")
    return job_id


def parse_job_id(output: str) -> str:
    """Extract exactly one job ID from OAR's submission output."""

    matches = set(_JOB_ID_PATTERN.findall(output))
    if len(matches) != 1:
        raise ValueError("submission output must contain exactly one OAR job ID")
    return _validate_job_id(matches.pop())


def build_status_command(job_id: str) -> Command:
    """Build a status query for one validated OAR job."""

    return ("oarstat", "-j", _validate_job_id(job_id))


def build_ssh_command(frontend: str, command: Command) -> Command:
    """Serialize a remote command so SSH's remote shell preserves its arguments."""

    if not frontend or any(character.isspace() for character in frontend):
        raise ValueError("frontend must be a non-empty host name")
    if not command:
        raise ValueError("remote command must not be empty")
    return ("ssh", frontend, shlex.join(command))


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
    return _validate_job_id(payload["job_id"])


def _reject_active_state(config: Grid5000Config, state_path: Path, runner: CommandRunner) -> None:
    job_id = _state_job_id(state_path)
    output = _read_job_status(config, job_id, runner)
    state = _job_status_state(output)
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
        output = _status_error_output(error)
        if _job_status_state(output) == "missing":
            return output
        raise RuntimeError(
            f"cannot verify Grid'5000 job {job_id} status through {config.frontend}: "
            f"{output.strip() or 'status command failed'}"
        ) from error


def _status_error_output(error: subprocess.CalledProcessError) -> str:
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


def _job_status_state(output: str) -> str:
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
    _reject_unclean_policy_output(policy_output)
    _reject_active_eunis_jobs(command_runner(commands[1]), config.excluded_sites)
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


def _reject_active_eunis_jobs(output: str, excluded_sites: tuple[str, ...] = ()) -> None:
    matches, errors = _parse_active_eunis_jobs_report(output, excluded_sites)
    if matches:
        summary = _active_eunis_job_summary(matches)
        raise RuntimeError(f"active EUNIS Grid'5000 job(s) already exist: {summary}")
    if errors:
        sites = _unreachable_eunis_site_summary(errors)
        raise RuntimeError(f"cannot verify active EUNIS jobs on Grid'5000 site(s): {sites}")


def _reject_unclean_policy_output(output: str) -> None:
    """Require a completed policy check; warnings about other jobs do not block.

    The check must either say nothing was flagged or print its conformance
    report header, and must not report an error. Flagged day/night notices for
    unrelated jobs are tolerated; EUNIS jobs are limited to one host and one
    hour, and duplicate-job checks run separately.
    """

    completed = re.search(r"(?m)^\s*No jobs flagged\s*$|testing usage policy conformance", output)
    reported_error = re.search(r"(?im)^\s*(?:error|fatal):", output)
    if completed is None or reported_error is not None:
        raise RuntimeError(
            "usagepolicycheck did not produce a clean usage-policy result; "
            "refusing Grid'5000 submission"
        )


def _parse_active_eunis_jobs_report(
    output: str,
    excluded_sites: tuple[str, ...] = (),
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    try:
        report = json.loads(output)
    except json.JSONDecodeError as error:
        raise RuntimeError("cannot verify active EUNIS jobs across Grid'5000 sites") from error
    if not isinstance(report, dict):
        raise TypeError("cannot verify active EUNIS jobs across Grid'5000 sites")
    _validate_report_exclusion_scope(report, excluded_sites)
    matches = report.get("active_jobs")
    errors = report.get("errors")
    if not _is_eunis_report_entries(matches) or not _is_eunis_report_entries(errors):
        raise TypeError("cannot verify active EUNIS jobs across Grid'5000 sites")
    return matches, errors


def _validate_report_exclusion_scope(
    report: dict[str, object], excluded_sites: tuple[str, ...]
) -> None:
    if report.get("excluded_sites", []) != list(_validate_excluded_sites(excluded_sites)):
        raise RuntimeError(
            "cannot verify active EUNIS jobs: exclusion scope does not match request"
        )


def _is_eunis_report_entries(value: object) -> TypeGuard[list[dict[str, object]]]:
    return isinstance(value, list) and all(isinstance(item, dict) for item in value)


def _active_eunis_job_summary(matches: list[dict[str, object]]) -> str:
    return ", ".join(
        f"{item.get('site', 'unknown')}:{item.get('job_id', 'unknown')}"
        f" ({item.get('state', 'unknown')})"
        for item in matches
    )


def _unreachable_eunis_site_summary(errors: list[dict[str, object]]) -> str:
    return ", ".join(str(item.get("site", "unknown")) for item in errors)


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
