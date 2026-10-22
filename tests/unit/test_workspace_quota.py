from __future__ import annotations

import multiprocessing
import os
from pathlib import Path
from typing import Any

from osm_polygon_eunis.pool_adapters.workspace_quota import (
    WorkspaceQuota,
    WorkspaceQuotaExceededError,
    workspace_size,
)


def _competing_quota_writer(
    workspace_text: str,
    barrier: Any,
    release: Any,
    results: Any,
) -> None:
    workspace = Path(workspace_text)
    quota = WorkspaceQuota(workspace, 100)
    with quota.lease() as lease:
        temporary = workspace / f".writer-{os.getpid()}.partial"
        barrier.wait(timeout=10)
        try:
            with lease.writer(temporary) as output:
                output.write(b"x" * 60)
                output.flush()
            results.put(("reserved", temporary.name, temporary.stat().st_size))
            release.wait(timeout=10)
        except WorkspaceQuotaExceededError:
            results.put(("denied", temporary.name, 0))


def _crash_with_owned_partial(workspace_text: str, ready: Any) -> None:
    workspace = Path(workspace_text)
    quota = WorkspaceQuota(workspace, 10)
    lease = quota.lease()
    temporary = workspace / ".crashed.partial"
    with lease.writer(temporary) as output:
        output.write(b"x" * 10)
        output.flush()
    ready.set()
    os._exit(0)


def test_distinct_processes_cannot_spend_the_same_workspace_capacity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(2)
    release = context.Event()
    results = context.Queue()
    processes = [
        context.Process(
            target=_competing_quota_writer,
            args=(str(workspace), barrier, release, results),
        )
        for _ in range(2)
    ]
    for process in processes:
        process.start()
    observed = [results.get(timeout=20), results.get(timeout=20)]
    release.set()
    for process in processes:
        process.join(timeout=20)

    assert [process.exitcode for process in processes] == [0, 0]
    assert sorted(row[0] for row in observed) == ["denied", "reserved"]
    assert next(row[2] for row in observed if row[0] == "reserved") == 60
    assert workspace_size(workspace) == 0


def test_crashed_lease_removes_only_its_partial_and_releases_capacity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(
        target=_crash_with_owned_partial,
        args=(str(workspace), ready),
    )
    process.start()
    assert ready.wait(timeout=20)
    process.join(timeout=20)
    assert process.exitcode == 0
    crashed_partial = workspace / ".crashed.partial"
    assert crashed_partial.read_bytes() == b"x" * 10

    with WorkspaceQuota(workspace, 10).lease() as lease:
        temporary = workspace / ".recovered.partial"
        lease.track_temporary_path(temporary)
        with lease.writer(temporary) as output:
            output.write(b"y" * 10)
            output.flush()
        lease.promote_temporary_path(temporary, workspace / "recovered.bin")

    assert not crashed_partial.exists()
    assert (workspace / "recovered.bin").read_bytes() == b"y" * 10
    assert workspace_size(workspace) == 10
