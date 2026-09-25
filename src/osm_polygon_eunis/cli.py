"""Scriptable command-line entry points for planning and publishing."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path

import httpx
from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError

from .runner import (
    DATASET_NAMES,
    DEFAULT_WORKERS,
    ConfigError,
    DryRunReport,
    Progress,
    ReleaseReceipt,
    VerificationError,
    plan_datasets,
    plan_release,
    run_release,
    validate_reference_config,
    verify_release,
)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


_EPILOG = """\
examples:
  osm-polygon-eunis plan
  osm-polygon-eunis release --dry-run --workdir .eunis-run
  osm-polygon-eunis release --batch-size 256 --workdir .eunis-run
  osm-polygon-eunis release --dataset wikidata --workers 4
  osm-polygon-eunis verify --dataset website
  osm-polygon-eunis release -q > receipt.json

exit status:
  0 success, 1 unexpected error, 2 usage or config error,
  3 Hub/network or authentication error, 4 verification failed

environment:
  HF_TOKEN  Hugging Face token; release needs write access to the targets,
            plan/verify/--dry-run only read (a token helps with rate limits).

See docs/operations.md for the full release procedure.
"""


def _add_common(parser: argparse.ArgumentParser, *, top_level: bool = False) -> None:
    """Output flags, accepted before or after the subcommand."""

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
        default=Path(".eunis-run"),
        help="local staging directory for shards, sidecars and cards (default: .eunis-run)",
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="osm-polygon-eunis",
        description="Enrich OSM polygon Hugging Face datasets with exact-overlap EUNIS labels, "
        "publish them and verify the published result.",
        epilog=_EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {_version()}")
    _add_common(parser, top_level=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan",
        help="inspect pinned public source layouts",
        description="Print the pinned source revision and shard layout of each dataset.",
    )
    _add_dataset(plan)
    _add_endpoint(plan)
    _add_common(plan)
    release = subparsers.add_parser(
        "release",
        help="run and publish the datasets",
        description="Label, publish and independently verify the selected datasets. "
        "Requires HF_TOKEN with write access unless --dry-run is given.",
    )
    release.add_argument(
        "--reference-config",
        type=Path,
        default=Path("config/eea-2021-reference.json"),
        help="EEA reference config JSON (default: config/eea-2021-reference.json)",
    )
    _add_workdir(release)
    release.add_argument(
        "--batch-size",
        type=_positive_int,
        default=256,
        help="Parquet rows per streamed batch (default: 256)",
    )
    release.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"geometry worker processes (default: {DEFAULT_WORKERS})",
    )
    _add_dataset(release)
    _add_endpoint(release)
    _add_common(release)
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
    )
    _add_workdir(verify)
    _add_dataset(verify)
    _add_endpoint(verify)
    _add_common(verify)
    return parser


def _api(endpoint: str | None) -> HfApi:
    token = os.environ.get("HF_TOKEN")
    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


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
    if isinstance(error, ConfigError):
        return EXIT_USAGE
    if isinstance(error, (HfHubHTTPError, httpx.HTTPError, ConnectionError, TimeoutError)):
        return EXIT_REMOTE
    return EXIT_ERROR


def _progress_sink(args: argparse.Namespace) -> Progress | None:
    """Progress JSON lines go to stderr so stdout carries only the final result."""

    if args.quiet:
        return None

    def emit(event: Mapping[str, object]) -> None:
        print(json.dumps(event, sort_keys=True, default=str), file=sys.stderr, flush=True)

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


def _validate_release(args: argparse.Namespace) -> None:
    """Reject bad inputs before any network call."""

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        raise ConfigError("HF_TOKEN is not set; release needs a Hugging Face write token")
    validate_reference_config(args.reference_config)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        return _run(args)
    except Exception as error:
        if args.debug:
            raise
        print(f"error: {error}", file=sys.stderr)
        return _exit_code(error)


def _run(args: argparse.Namespace) -> int:
    progress = _progress_sink(args)
    if args.verbose and progress is not None:
        progress({"event": "start", "command": args.command, "arguments": vars(args)})
    if args.command == "plan":
        return _run_plan(args)
    if args.command == "verify":
        _print_receipt(
            verify_release(_api(args.endpoint), workdir=args.workdir, datasets=args.dataset)
        )
        return EXIT_OK
    return _run_release(args, progress)


def _run_plan(args: argparse.Namespace) -> int:
    plans = plan_datasets(_api(args.endpoint), args.dataset)
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


def _run_release(args: argparse.Namespace, progress: Progress | None) -> int:
    _validate_release(args)
    if args.dry_run:
        report = plan_release(
            _api(args.endpoint),
            reference_config=args.reference_config,
            workdir=args.workdir,
            datasets=args.dataset,
        )
        _print_json(_dry_run_payload(report))
        return EXIT_OK
    receipt = run_release(
        _api(args.endpoint),
        reference_config=args.reference_config,
        workdir=args.workdir,
        batch_size=args.batch_size,
        token=os.environ.get("HF_TOKEN"),
        progress=progress,
        datasets=args.dataset,
        workers=args.workers,
    )
    _print_receipt(receipt)
    return EXIT_OK


def _print_receipt(receipt: ReleaseReceipt) -> None:
    _print_json(
        {
            "datasets": [
                {
                    "dataset": item.plan.spec.name,
                    "target_repo": item.plan.spec.output_repo,
                    "verified_revision": item.verification.target_revision,
                    "changed_shards": len(item.expectations),
                    "no_op": item.no_op,
                }
                for item in receipt.datasets
            ]
        }
    )
