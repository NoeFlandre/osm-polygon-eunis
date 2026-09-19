"""Scriptable command-line entry points for planning and publishing."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import HfApi

from .runner import plan_datasets, run_release


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
    release.add_argument("--batch-size", type=int, default=256)
    release.add_argument("--endpoint", default=None)
    return parser


def _api(endpoint: str | None) -> HfApi:
    token = os.environ.get("HF_TOKEN")
    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


def _print_progress(event: Mapping[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
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
                    }
                    for item in receipt.datasets
                ]
            },
            sort_keys=True,
            indent=2,
        )
    )
    return 0
