"""Scriptable command-line entry points for planning and publishing."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import HfApi

from .runner import (
    DryRunReport,
    plan_datasets,
    plan_release,
    run_release,
    validate_reference_config,
)


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}") from error
    if number <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return number


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osm-polygon-eunis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="inspect pinned public source layouts")
    plan.add_argument("--endpoint", default=None)
    release = subparsers.add_parser("release", help="run and publish the three datasets")
    release.add_argument(
        "--reference-config",
        type=Path,
        default=Path("config/eea-2021-reference.json"),
    )
    release.add_argument("--workdir", type=Path, default=Path(".eunis-run"))
    release.add_argument("--batch-size", type=_positive_int, default=256)
    release.add_argument("--endpoint", default=None)
    release.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve plans, references and no-op status and print them without Hub writes",
    )
    return parser


def _api(endpoint: str | None) -> HfApi:
    token = os.environ.get("HF_TOKEN")
    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


def _print_progress(event: Mapping[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


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


def _validate_release(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Reject bad inputs before any network call."""

    if not args.dry_run and not os.environ.get("HF_TOKEN"):
        parser.error("HF_TOKEN is not set; release needs a Hugging Face write token")
    try:
        validate_reference_config(args.reference_config)
    except (OSError, ValueError) as error:
        parser.error(str(error))


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.command == "plan":
        plans = plan_datasets(_api(args.endpoint))
        print(
            json.dumps(
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
                ],
                sort_keys=True,
                indent=2,
            )
        )
        return 0
    _validate_release(parser, args)
    if args.dry_run:
        report = plan_release(
            _api(args.endpoint),
            reference_config=args.reference_config,
            workdir=args.workdir,
        )
        print(json.dumps(_dry_run_payload(report), sort_keys=True, indent=2))
        return 0
    receipt = run_release(
        _api(args.endpoint),
        reference_config=args.reference_config,
        workdir=args.workdir,
        batch_size=args.batch_size,
        token=os.environ.get("HF_TOKEN"),
        progress=_print_progress,
    )
    print(
        json.dumps(
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
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0
