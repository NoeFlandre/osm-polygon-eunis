"""Scriptable command-line entry points for planning and publishing."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path

from huggingface_hub import HfApi

from .grid5000 import (
    Grid5000Config,
    build_cancel_command,
    build_ssh_command,
    build_status_command,
    resolve_source_revision,
    run_command,
    submit_grid5000,
)
from .runner import plan_datasets, run_release


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="osm-polygon-eunis")
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser("plan", help="inspect pinned public source layouts")
    plan.add_argument("--endpoint", default=None)
    release = subparsers.add_parser("release", help="run and publish selected datasets")
    release.add_argument(
        "--reference-config",
        type=Path,
        default=Path("config/eea-2021-reference.json"),
    )
    release.add_argument("--workdir", type=Path, default=Path(".eunis-run"))
    release.add_argument("--batch-size", type=int, default=256)
    release.add_argument("--endpoint", default=None)
    release.add_argument("--dataset", action="append", dest="datasets")
    release.add_argument(
        "--execution",
        choices=("grid5000", "local"),
        default="grid5000",
    )
    release.add_argument("--receipt", type=Path, default=None)
    grid5000 = subparsers.add_parser(
        "grid5000", help="submit and monitor the description release on Grid'5000"
    )
    grid_commands = grid5000.add_subparsers(dest="grid_command", required=True)
    submit = grid_commands.add_parser("submit", help="submit one description worker")
    submit.add_argument("--frontend", required=True)
    submit.add_argument("--persistent-root", required=True)
    submit.add_argument("--source-root", type=Path, default=Path("."))
    submit.add_argument("--source-revision", default=None)
    submit.add_argument("--allow-dirty-source", action="store_true")
    submit.add_argument("--state-file", type=Path, default=None)
    submit.add_argument("--dry-run", action="store_true")
    submit.add_argument("--cluster", default="chuc")
    submit.add_argument("--queue", default="default")
    submit.add_argument("--job-type", default="night")
    submit.add_argument("--cores", type=int, default=16)
    submit.add_argument("--workers", type=int, default=16)
    submit.add_argument("--walltime", default="12:00:00")
    submit.add_argument("--batch-size", type=int, default=256)
    status = grid_commands.add_parser("status", help="show one remote OAR job")
    status.add_argument("--frontend", required=True)
    status.add_argument("--job-id", required=True)
    cancel = grid_commands.add_parser("cancel", help="cancel one remote OAR job")
    cancel.add_argument("--frontend", required=True)
    cancel.add_argument("--job-id", required=True)
    return parser


def _api(endpoint: str | None) -> HfApi:
    token = os.environ.get("HF_TOKEN")
    return HfApi(endpoint=endpoint, token=token) if endpoint else HfApi(token=token)


def _print_progress(event: Mapping[str, object]) -> None:
    print(json.dumps(event, sort_keys=True), flush=True)


def _grid5000_submit(args: argparse.Namespace) -> int:
    source_root = args.source_root.resolve()
    source_revision = resolve_source_revision(
        source_root,
        explicit=args.source_revision,
        allow_dirty=args.allow_dirty_source,
    )
    config = Grid5000Config(
        frontend=args.frontend,
        persistent_root=args.persistent_root,
        cluster=args.cluster,
        queue=args.queue,
        job_type=args.job_type,
        cores=args.cores,
        workers=args.workers,
        walltime=args.walltime,
        batch_size=args.batch_size,
    )
    submission = submit_grid5000(
        config,
        source_root,
        source_revision=source_revision,
        state_path=args.state_file,
        dry_run=args.dry_run,
    )
    payload: dict[str, object] = {
        "commands": [list(command) for command in submission.commands],
        "dataset": submission.dataset,
        "dry_run": submission.job is None,
        "source_revision": submission.source_revision,
    }
    if submission.job is not None:
        payload.update(
            {
                "job_id": submission.job.job_id,
                "submitted_at": submission.job.submitted_at,
            }
        )
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


def _grid5000_main(args: argparse.Namespace) -> int:
    if args.grid_command == "submit":
        return _grid5000_submit(args)
    command = (
        build_status_command(args.job_id)
        if args.grid_command == "status"
        else build_cancel_command(args.job_id)
    )
    output = run_command(build_ssh_command(args.frontend, command))
    print(output, end="" if output.endswith("\n") else "\n")
    return 0


def _plan_command(args: argparse.Namespace) -> int:
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


def _release_command(args: argparse.Namespace) -> int:
    receipt = run_release(
        _api(args.endpoint),
        reference_config=args.reference_config,
        workdir=args.workdir,
        batch_size=args.batch_size,
        token=os.environ.get("HF_TOKEN"),
        progress=_print_progress,
        dataset_names=tuple(args.datasets or ("description",)),
        execution=args.execution,
        receipt_path=args.receipt,
    )
    print(json.dumps(_release_summary(receipt), sort_keys=True, indent=2))
    return 0


def _release_summary(receipt) -> dict[str, object]:
    return {
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


def _dispatch(args: argparse.Namespace) -> int:
    if args.command == "plan":
        return _plan_command(args)
    if args.command == "grid5000":
        return _grid5000_main(args)
    return _release_command(args)


def main(argv: list[str] | None = None) -> int:
    return _dispatch(_parser().parse_args(argv))
