"""Scriptable command-line entry points for planning and publishing."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Final, cast

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from ._protocols import HubApi
from .fileio import write_json_atomic
from .grid5000 import (
    DEFAULT_GRID_CORES,
    DEFAULT_GRID_WORKERS,
    Grid5000Config,
    Grid5000Submission,
    resolve_source_revision,
    submit_grid5000,
    validate_walltime,
)
from .options import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_WORKDIR,
    DEFAULT_WORKERS,
    BatchLimits,
    ReleaseOptions,
    resolve_sidecar_root,
    resolve_workdir,
)
from .publish import VerificationError
from .release_orchestration import (
    ConfigError,
    DryRunReport,
    plan_release,
    run_release,
    validate_reference_config,
    verify_release,
)
from .release_plan import (
    DATASET_NAMES,
    DatasetPlan,
    Progress,
    ReleaseReceipt,
    plan_datasets,
)
from .run_analysis import RunAnalysisError, summarize_run

logger = logging.getLogger(__name__)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _walltime(value: str) -> str:
    try:
        validate_walltime(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return value


def _non_negative_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        ) from error
    if number < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
    return number


_EPILOG = f"""\
examples:
  osm-polygon-eunis plan
  osm-polygon-eunis release --dry-run --workdir {DEFAULT_WORKDIR}
  osm-polygon-eunis release --batch-size {DEFAULT_BATCH_SIZE} --workdir {DEFAULT_WORKDIR}
  osm-polygon-eunis release --dataset wikidata --workers 4
  osm-polygon-eunis verify --dataset website
  osm-polygon-eunis analyze-run --log job.log --sidecars /data/sidecars/eunis
  osm-polygon-eunis release -q > receipt.json
  osm-polygon-eunis grid5000 submit --site SITE --frontend FRONTEND \\
    --cluster CLUSTER --persistent-root /home/USER/osm-polygon-eunis --state /path/on/HDD/job.json

exit status:
  0 success, 1 unexpected error, 2 usage or config error,
  3 Hub/network or authentication error, 4 verification failed

environment:
  HF_TOKEN  Hugging Face token; release needs write access to the targets,
            plan/verify/--dry-run only read (a token helps with rate limits).

See docs/operations.md for the full release procedure.
"""


def _output_flags(*, top_level: bool = False) -> argparse.ArgumentParser:
    """Output flags as a parent parser, accepted before or after the subcommand.

    The top-level parser sets the defaults. Subcommands use SUPPRESS so that they
    do not overwrite a flag that was given before the subcommand name.
    """

    parser = argparse.ArgumentParser(add_help=False)
    default: object = False if top_level else argparse.SUPPRESS
    volume = parser.add_mutually_exclusive_group()
    volume.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        default=default,
        help="silence progress lines on stderr; stdout keeps only the final JSON",
    )
    volume.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=default,
        help="also emit a start record with the resolved arguments on stderr",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=default,
        help="show the full traceback instead of a one-line error",
    )
    return parser


def _add_endpoint(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--endpoint",
        default=None,
        help="Hugging Face Hub endpoint URL (default: the public Hub)",
    )


def _add_dataset(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        action="append",
        choices=DATASET_NAMES,
        default=None,
        metavar="NAME",
        help=f"limit to one dataset; repeatable (choices: {', '.join(DATASET_NAMES)}; "
        "default: all)",
    )


def _add_workdir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="local staging directory for shards, sidecars and cards "
        f"(default: $OSM_EUNIS_WORKDIR or {DEFAULT_WORKDIR})",
    )


def _default_reference_config() -> Path:
    packaged = Path(__file__).with_name("eea-2021-reference.json")
    if packaged.is_file():
        return packaged
    return Path(__file__).resolve().parents[2] / "config" / "eea-2021-reference.json"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osm-polygon-eunis",
        description="Enrich OSM polygon Hugging Face datasets with exact-overlap EUNIS labels, "
        "publish them and verify the published result.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[_output_flags(top_level=True)],
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan",
        help="inspect pinned public source layouts",
        description="Print the pinned source revision and shard layout of each dataset.",
        parents=[_output_flags()],
    )
    _add_dataset(plan)
    _add_endpoint(plan)
    release = subparsers.add_parser(
        "release",
        help="run and publish the datasets",
        description="Label, publish and independently verify the selected datasets. "
        "Requires HF_TOKEN with write access unless --dry-run is given.",
        parents=[_output_flags()],
    )
    release.add_argument(
        "--reference-config",
        type=Path,
        default=None,
        help="EEA reference config JSON (default: the packaged EEA 2021 config)",
    )
    _add_workdir(release)
    release.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet rows per streamed batch (default: {DEFAULT_BATCH_SIZE})",
    )
    release.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"geometry worker processes (default: {DEFAULT_WORKERS})",
    )
    release.add_argument(
        "--max-intersection-errors",
        type=_non_negative_int,
        default=None,
        metavar="N",
        help="fail a dataset before publishing its manifest when more than N overlap "
        "candidates were dropped by GEOS intersection errors (default: no limit)",
    )
    release.add_argument(
        "--receipt",
        type=Path,
        default=None,
        help="write the verified release receipt atomically to this path",
    )
    _add_dataset(release)
    _add_endpoint(release)
    release.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve plans, references and no-op status and print them without Hub writes",
    )
    verify = subparsers.add_parser(
        "verify",
        help="re-verify published targets (read-only)",
        description="Check each published target against its EUNIS manifest: remote tree, "
        "shared blobs, Parquet rows and schemas, and card artifacts. Makes no writes and "
        "exits nonzero on a mismatch.",
        parents=[_output_flags()],
    )
    _add_workdir(verify)
    _add_dataset(verify)
    _add_endpoint(verify)

    analyze = subparsers.add_parser(
        "analyze-run",
        help="summarize local geometry logs and checkpoints",
        description=(
            "Summarize geometry batch timing records and matching .done checkpoints. "
            "This command reads local files only."
        ),
        parents=[_output_flags()],
    )
    analyze.add_argument(
        "--log",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="job log file; repeat for each retry log",
    )
    analyze.add_argument(
        "--sidecars",
        type=Path,
        default=None,
        help="directory containing dataset sidecars "
        f"(default: $EUNIS_SIDECAR_DIR or <workdir>/sidecars, workdir default {DEFAULT_WORKDIR})",
    )
    analyze.add_argument(
        "--slowest-shards",
        type=_positive_int,
        default=5,
        metavar="N",
        help="slowest shards to show for each dataset (default: 5)",
    )

    grid_parser = subparsers.add_parser(
        "grid5000",
        help="prepare and submit the all-source release on Grid'5000",
        description=(
            "Submit one short, resumable all-source release to an explicitly selected site."
        ),
    )
    grid_subparsers = grid_parser.add_subparsers(
        dest="grid_action", required=True, help="Grid'5000 operation"
    )
    grid_submit = grid_subparsers.add_parser(
        "submit",
        help="check policy, sync the source and submit the release worker",
        description="Run usagepolicycheck -t, sync this source tree and submit one OAR job.",
        parents=[_output_flags()],
    )
    grid_submit.add_argument("--site", required=True, help="Grid'5000 site, for example grenoble")
    grid_submit.add_argument(
        "--frontend",
        default=None,
        help="SSH alias for the selected site's frontend (default: $OSM_EUNIS_GRID5000_FRONTEND)",
    )
    grid_submit.add_argument("--cluster", required=True, help="OAR cluster on the selected site")
    grid_submit.add_argument(
        "--persistent-root",
        default=None,
        help="remote persistent project directory under /home, /groups or /srv "
        "(default: $OSM_EUNIS_GRID5000_PERSISTENT_ROOT)",
    )
    grid_submit.add_argument(
        "--exclude-site",
        action="append",
        default=[],
        help="explicitly omit one unreachable site from policy and duplicate-job checks",
    )
    grid_submit.add_argument(
        "--state",
        required=True,
        type=Path,
        help="local controller state file on persistent storage such as the external HDD",
    )
    grid_submit.add_argument("--queue", default="default", help="OAR queue (default: default)")
    grid_submit.add_argument(
        "--job-type", default=None, help="optional OAR job type such as day or night"
    )
    grid_submit.add_argument(
        "--cores",
        type=_positive_int,
        default=DEFAULT_GRID_CORES,
        help=f"cores on one host (default: {DEFAULT_GRID_CORES})",
    )
    grid_submit.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_GRID_WORKERS,
        help=f"geometry workers (default: {DEFAULT_GRID_WORKERS})",
    )
    grid_submit.add_argument(
        "--walltime",
        type=_walltime,
        default="1:00:00",
        help="OAR walltime in HH:MM:SS, above 5 minutes and at most 1 hour (default: 1:00:00)",
    )
    grid_submit.add_argument(
        "--batch-size",
        type=_positive_int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Parquet rows per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    grid_submit.add_argument(
        "--dry-run",
        action="store_true",
        help="show the validated plan without contacting Grid'5000",
    )
    return parser


def _api(endpoint: str | None) -> HubApi:
    token = os.environ.get("HF_TOKEN")
    client = HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)
    return cast(HubApi, client)


@dataclass(frozen=True, slots=True)
class CliDependencies:
    """Operations the CLI calls, injectable at its public ``main`` boundary.

    Supplying a fake Hub or runner here lets callers exercise argument parsing,
    validation, output and exit-code behavior without replacing module globals.
    """

    api_factory: Callable[[str | None], HubApi] = _api
    plan_datasets: Callable[[HubApi, Sequence[str] | None], tuple[DatasetPlan, ...]] = plan_datasets
    plan_release: Callable[..., DryRunReport] = plan_release
    run_release: Callable[[HubApi, ReleaseOptions], ReleaseReceipt] = run_release
    validate_reference_config: Callable[[Path], None] = validate_reference_config
    verify_release: Callable[..., ReleaseReceipt] = verify_release
    resolve_source_revision: Callable[..., str] = resolve_source_revision
    submit_grid5000: Callable[..., Grid5000Submission] = submit_grid5000


EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2
EXIT_REMOTE = 3
EXIT_VERIFICATION = 4


def _version() -> str:
    try:
        return metadata.version("osm-polygon-eunis")
    except metadata.PackageNotFoundError:
        return "unknown"


def _exit_code(error: Exception) -> int:
    """Map a failure to the documented exit status."""

    if isinstance(error, VerificationError):
        return EXIT_VERIFICATION
    if isinstance(error, (ConfigError, RunAnalysisError)):
        return EXIT_USAGE
    if isinstance(error, (HfHubHTTPError, httpx.HTTPError, ConnectionError, TimeoutError)):
        return EXIT_REMOTE
    return EXIT_ERROR


class _ProgressFormatter(logging.Formatter):
    """Render progress records as the existing JSON-lines stderr contract."""

    def format(self, record: logging.LogRecord) -> str:
        event = getattr(record, "progress_event", None)
        if event is not None:
            return json.dumps(event, sort_keys=True, default=str)
        return super().format(record)


class _ProgressHandler(logging.StreamHandler):
    def __init__(self, previous_root_level: int, previous_cli_level: int) -> None:
        super().__init__(sys.stderr)
        self.previous_root_level = previous_root_level
        self.previous_cli_level = previous_cli_level


def _logging_level(args: argparse.Namespace) -> int:
    if args.debug or args.verbose:
        return logging.DEBUG
    if args.quiet:
        return logging.WARNING
    return logging.INFO


def _configure_logging(args: argparse.Namespace) -> None:
    _reset_logging()
    root_logger = logging.getLogger()
    previous_root_level = root_logger.level
    previous_cli_level = logger.level
    level = _logging_level(args)
    root_logger.setLevel(level)
    logger.setLevel(level)
    handler = _ProgressHandler(previous_root_level, previous_cli_level)
    handler.setLevel(level)
    handler.setFormatter(_ProgressFormatter("%(message)s"))
    root_logger.addHandler(handler)


def _reset_logging() -> None:
    root_logger = logging.getLogger()
    previous_root_level: int | None = None
    previous_cli_level: int | None = None
    for handler in tuple(root_logger.handlers):
        if isinstance(handler, _ProgressHandler):
            root_logger.removeHandler(handler)
            handler.close()
            previous_root_level = handler.previous_root_level
            previous_cli_level = handler.previous_cli_level
    if previous_root_level is not None:
        root_logger.setLevel(previous_root_level)
    if previous_cli_level is not None:
        logger.setLevel(previous_cli_level)


def _progress_sink(args: argparse.Namespace) -> Progress | None:
    """Progress JSON lines go to stderr so stdout carries only the final result."""

    if args.quiet:
        return None

    def emit(event: Mapping[str, object]) -> None:
        level = logging.DEBUG if event.get("event") == "start" else logging.INFO
        logger.log(level, "progress", extra={"progress_event": event})

    return emit


def _print_json(payload: object) -> None:
    print(json.dumps(payload, sort_keys=True, indent=2))


def _dry_run_payload(report: DryRunReport) -> dict[str, object]:
    return {
        "dry_run": True,
        "no_op": report.no_op,
        "reference_assets": report.reference_assets,
        "datasets": [
            {
                "dataset": item.plan.spec.name,
                "source_repo": item.plan.spec.source_repo,
                "source_revision": item.plan.source_revision,
                "target_repo": item.plan.spec.output_repo,
                "target_exists": item.target_exists,
                "would_duplicate": item.would_duplicate,
                "shards_to_upload": list(item.shards_to_upload),
            }
            for item in report.datasets
        ],
    }


def _validate_release(
    args: argparse.Namespace,
    validate_config: Callable[[Path], None],
) -> None:
    """Reject bad inputs before any network call."""

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise ConfigError("HF_TOKEN is not set; release needs a Hugging Face write token")
    validate_config(args.reference_config)


def main(
    argv: list[str] | None = None,
    *,
    dependencies: CliDependencies | None = None,
) -> int:
    """Parse and execute one CLI command.

    Args:
        argv: Optional argument vector; defaults to the process command line.
        dependencies: Injectable Hub, runner, and Grid operations. Production
            callers can omit it to use the configured implementations.

    Returns:
        The documented command exit status.
    """
    services = dependencies or CliDependencies()
    args = _parser().parse_args(argv)
    _configure_logging(args)
    try:
        return _run(args, services)
    except Exception as error:
        logger.debug("command failed", exc_info=True)
        if args.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        return _exit_code(error)
    finally:
        _reset_logging()


def _resolve_defaults(args: argparse.Namespace) -> None:
    """Fill environment- and filesystem-dependent defaults after parsing."""
    resolvers: dict[str, Callable[[], Path]] = {
        "workdir": resolve_workdir,
        "sidecars": resolve_sidecar_root,
        "reference_config": _default_reference_config,
    }
    for name, resolve in resolvers.items():
        if getattr(args, name, "unset") is None:
            setattr(args, name, resolve())
    if args.command == "grid5000":
        _resolve_grid5000_environment(args)


def _resolve_grid5000_environment(args: argparse.Namespace) -> None:
    """Fill the Grid'5000 frontend and persistent root from the environment."""
    if args.frontend is None:
        args.frontend = os.environ.get("OSM_EUNIS_GRID5000_FRONTEND") or None
    if args.persistent_root is None:
        args.persistent_root = os.environ.get("OSM_EUNIS_GRID5000_PERSISTENT_ROOT") or None
    missing = []
    if args.frontend is None:
        missing.append("--frontend or OSM_EUNIS_GRID5000_FRONTEND")
    if args.persistent_root is None:
        missing.append("--persistent-root or OSM_EUNIS_GRID5000_PERSISTENT_ROOT")
    if missing:
        raise ConfigError(f"grid5000 submit requires {' and '.join(missing)}")


def _run(args: argparse.Namespace, services: CliDependencies) -> int:
    _resolve_defaults(args)
    progress = _progress_sink(args)
    _report_start(args, progress)
    if args.command == "plan":
        return _run_plan(args, services)
    if args.command == "verify":
        _print_receipt(
            services.verify_release(
                services.api_factory(args.endpoint),
                workdir=args.workdir,
                datasets=args.dataset,
            )
        )
        return EXIT_OK
    if args.command == "analyze-run":
        _print_json(
            summarize_run(
                args.log,
                args.sidecars,
                slowest_limit=args.slowest_shards,
            )
        )
        return EXIT_OK
    if args.command == "grid5000":
        return _run_grid5000_submit(args, services)
    return _run_release(args, progress, services)


# Options that may appear in the verbose start record. Anything else stays out of it.
# Endpoint URLs are left out on purpose: a URL can carry credentials.
_START_RECORD_FIELDS: Final = (
    "command",
    "grid_action",
    "dataset",
    "reference_config",
    "workdir",
    "batch_size",
    "workers",
    "max_intersection_errors",
    "receipt",
    "dry_run",
    "log",
    "sidecars",
    "slowest_shards",
    "site",
    "frontend",
    "cluster",
    "persistent_root",
    "exclude_site",
    "state",
    "queue",
    "job_type",
    "cores",
    "walltime",
)


def _report_start(args: argparse.Namespace, progress: Progress | None) -> None:
    if args.verbose and progress is not None:
        arguments = {
            name: getattr(args, name) for name in _START_RECORD_FIELDS if hasattr(args, name)
        }
        progress({"event": "start", "command": args.command, "arguments": arguments})


def _run_grid5000_submit(args: argparse.Namespace, services: CliDependencies) -> int:
    config = Grid5000Config(
        frontend=args.frontend,
        persistent_root=args.persistent_root,
        site=args.site,
        cluster=args.cluster,
        queue=args.queue,
        job_type=args.job_type,
        cores=args.cores,
        workers=args.workers,
        walltime=args.walltime,
        batch_size=args.batch_size,
        excluded_sites=tuple(args.exclude_site),
    )
    source_revision = services.resolve_source_revision(
        Path.cwd(),
    )
    submission = services.submit_grid5000(
        config,
        Path.cwd(),
        source_revision=source_revision,
        state_path=args.state,
        dry_run=args.dry_run,
    )
    _print_json(
        {
            "dry_run": args.dry_run,
            "job_id": None if submission.job is None else submission.job.job_id,
            "site": config.site,
            "cluster": config.cluster,
            "excluded_sites": list(config.excluded_sites),
            "source_revision": submission.source_revision,
            "datasets": list(submission.datasets),
            "commands": [list(command) for command in submission.commands]
            if args.dry_run
            else None,
        }
    )
    return EXIT_OK


def _run_plan(args: argparse.Namespace, services: CliDependencies) -> int:
    plans = services.plan_datasets(services.api_factory(args.endpoint), args.dataset)
    _print_json(
        [
            {
                "dataset": plan.spec.name,
                "source_repo": plan.spec.source_repo,
                "source_revision": plan.source_revision,
                "source_files": len(plan.source_files),
                "geometry_shards": len(plan.geometry_paths),
                "link_shards": len(plan.link_paths),
            }
            for plan in plans
        ]
    )
    return EXIT_OK


def _run_release(
    args: argparse.Namespace,
    progress: Progress | None,
    services: CliDependencies,
) -> int:
    _validate_release(args, services.validate_reference_config)
    if args.dry_run:
        report = services.plan_release(
            services.api_factory(args.endpoint),
            reference_config=args.reference_config,
            workdir=args.workdir,
            datasets=args.dataset,
        )
        _print_json(_dry_run_payload(report))
        return EXIT_OK
    receipt = services.run_release(
        services.api_factory(args.endpoint),
        ReleaseOptions(
            reference_config=args.reference_config,
            workdir=args.workdir,
            limits=BatchLimits(
                workers=args.workers,
                parquet_batch_size=args.batch_size,
            ),
            token=os.environ.get("HF_TOKEN"),
            progress=progress,
            datasets=args.dataset,
            max_intersection_errors=args.max_intersection_errors,
        ),
    )
    payload = _receipt_payload(receipt)
    if args.receipt is not None:
        _write_receipt_file(args.receipt, payload)
    _print_json(payload)
    return EXIT_OK


def _print_receipt(receipt: ReleaseReceipt) -> None:
    _print_json(_receipt_payload(receipt))


def _receipt_payload(receipt: ReleaseReceipt) -> dict[str, object]:
    datasets: list[dict[str, object]] = []
    for item in receipt.datasets:
        manifest = item.verification.manifest or {}
        card = manifest.get("card")
        card_metadata = card if isinstance(card, Mapping) else {}
        rows_by_path = dict(sorted(item.verification.rows_by_path.items()))
        manifest_json = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
        datasets.append(
            {
                "dataset": item.plan.spec.name,
                "source_repo": item.plan.spec.source_repo,
                "target_repo": item.plan.spec.output_repo,
                "source_revision": item.plan.source_revision,
                "source_file_count": len(item.plan.source_files),
                "geometry_shards": len(item.plan.geometry_paths),
                "link_shards": len(item.plan.link_paths),
                "verified_revision": item.verification.target_revision,
                "changed_shards": len(item.expectations),
                "rows_by_path": rows_by_path,
                "verified_total_rows": sum(rows_by_path.values()),
                "invalid_geometries": card_metadata.get("invalid_geometries"),
                "intersection_errors": card_metadata.get("intersection_errors"),
                "manifest_sha256": hashlib.sha256(manifest_json.encode("utf-8")).hexdigest(),
                "manifest": dict(manifest),
                "no_op": item.no_op,
            }
        )
    return {"status": "verified", "datasets": datasets, "reference": dict(receipt.reference)}


def _write_receipt_file(path: Path, payload: Mapping[str, object]) -> None:
    write_json_atomic(path, payload)
