from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any, cast

import pytest

from osm_polygon_eunis.pool_adapters.workspace_quota import (
    WorkspaceQuota,
    WorkspaceQuotaExceededError,
    WorkspaceQuotaLease,
    _owned_path,
    _path_size,
    _paths_overlap,
    _read_lease_state,
    _temporary_path,
    _valid_counter,
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


@pytest.mark.slow
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


@pytest.mark.slow
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


def test_quota_checks_existing_files_and_active_reservations(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    existing = workspace / "existing.bin"
    existing.write_bytes(b"1234")
    quota = WorkspaceQuota(workspace, 8)

    quota.ensure(4)
    with pytest.raises(WorkspaceQuotaExceededError):
        quota.ensure(5)
    with pytest.raises(ValueError, match="cannot be negative"):
        quota.ensure(-1)

    with quota.lease() as lease:
        temporary = workspace / ".reserved.partial"
        lease.track_temporary_path(temporary)
        lease.reserve(4, temporary)
        with pytest.raises(WorkspaceQuotaExceededError):
            quota.ensure(1)
        lease.release(4, temporary)


def test_lease_owns_only_unique_controller_temporary_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.partial"
    occupied = workspace / ".occupied.partial"
    occupied.write_bytes(b"x")

    with WorkspaceQuota(workspace, 32).lease() as owner:
        with pytest.raises(ValueError, match="controller temporary"):
            owner.track_temporary_path(workspace / "ordinary.txt")
        with pytest.raises(ValueError, match="escaped"):
            owner.track_temporary_path(outside)
        with pytest.raises(FileExistsError):
            owner.track_temporary_path(occupied)

        partial = workspace / ".shared.partial"
        owner.track_temporary_path(partial)
        owner.track_temporary_path(partial)
        with (
            WorkspaceQuota(workspace, 32).lease() as other,
            pytest.raises(RuntimeError, match="active owner"),
        ):
            other.track_temporary_path(partial)


def test_writer_reserves_before_buffering_and_releases_unused_capacity(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 6)
    temporary = workspace / ".payload.partial"

    with quota.lease() as lease:
        with pytest.raises(RuntimeError, match="not open"):
            lease.writer(temporary).write(b"x")
        with lease.writer(temporary) as output:
            with pytest.raises(TypeError, match="bytes only"):
                output.write(cast(Any, "text"))
            assert output.write(b"abc") == 3
            assert output.fileno() >= 0
            with pytest.raises(WorkspaceQuotaExceededError):
                lease.reserve(4, temporary)
            output.flush()
            output.flush()
        assert temporary.read_bytes() == b"abc"
        assert workspace_size(workspace) == 3
        lease.discard_temporary_path(temporary)
        assert not temporary.exists()


def test_writer_exception_closes_stream_and_lease_removes_partial(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".failed.partial"

    with WorkspaceQuota(workspace, 8).lease() as lease:
        with pytest.raises(RuntimeError, match="stop"), lease.writer(temporary) as output:
            output.write(b"data")
            raise RuntimeError("stop")
        assert temporary.exists()

    assert not temporary.exists()


def test_promote_and_forget_require_registered_completed_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".complete.partial"
    target = workspace / "complete.bin"
    directory = workspace / "lucas-reference-evidence-test"

    with WorkspaceQuota(workspace, 16).lease() as lease:
        lease.track_temporary_path(temporary)
        lease.reserve(1, temporary)
        with pytest.raises(RuntimeError, match="outstanding"):
            lease.promote_temporary_path(temporary, target)
        lease.release(1, temporary)
        temporary.write_bytes(b"x")
        lease.promote_temporary_path(temporary, target)
        assert target.read_bytes() == b"x"

        duplicate = workspace / ".duplicate.partial"
        lease.track_temporary_path(duplicate)
        duplicate.write_bytes(b"x")
        with pytest.raises(FileExistsError):
            lease.promote_temporary_path(duplicate, target)
        lease.discard_temporary_path(duplicate)

        lease.track_temporary_path(directory)
        directory.mkdir()
        with pytest.raises(RuntimeError, match="still exists"):
            lease.forget_temporary_path(directory)
        directory.rmdir()
        lease.forget_temporary_path(directory)


def test_invalid_persisted_lease_state_fails_closed(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    lease_id = "bad-state"
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    state_path = leases / f"{lease_id}.json"
    state_path.write_text("{", encoding="utf-8")

    with pytest.raises(ValueError, match="state is invalid"):
        quota.ensure(0)

    state_path.write_text(
        json.dumps({"version": 99, "lease_id": lease_id, "paths": {}}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unsupported shape"):
        quota.ensure(0)

    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "lease_id": lease_id,
                "paths": {".unsafe.partial": {"accounted_bytes": True, "reserved_bytes": 0}},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="path counters are invalid"):
        _read_lease_state(state_path, lease_id, workspace)


def test_quota_path_helpers_enforce_temporary_paths_and_counter_shapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert _temporary_path(workspace, workspace / ".one.partial")[1] == ".one.partial"
    assert _temporary_path(workspace, workspace / "lucas-reference-evidence-a")[1]
    assert _paths_overlap("a", "a")
    assert _paths_overlap("a", "a/b.partial")
    assert _paths_overlap("a/b.partial", "a")
    assert not _paths_overlap("a/b.partial", "a/c.partial")
    assert _valid_counter(0)
    assert _valid_counter(3)
    assert not _valid_counter(True)
    assert not _valid_counter(-1)
    assert not _valid_counter("1")

    with pytest.raises(ValueError, match="unsafe temporary path"):
        _owned_path(workspace, "../escape.partial")
    with pytest.raises(ValueError, match="non-temporary"):
        _owned_path(workspace, "ordinary.txt")
    with pytest.raises(ValueError, match="controller temporary"):
        _temporary_path(workspace, workspace / "ordinary.txt")


def test_workspace_quota_rejects_symlink_roots_and_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    symlink = tmp_path / "linked-workspace"
    symlink.symlink_to(workspace, target_is_directory=True)
    with pytest.raises(ValueError, match="workspace must not be a symlink"):
        WorkspaceQuota(symlink, 8)

    linked_file = workspace / ".linked.partial"
    linked_file.symlink_to(tmp_path / "outside")
    with pytest.raises(ValueError, match="must not contain symlinks"):
        workspace_size(workspace)
    with pytest.raises(ValueError, match="must not be a symlink"):
        _owned_path(workspace, ".linked.partial")


def test_quota_rejects_bad_capacity_and_cleans_orphaned_control_files(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    with pytest.raises(ValueError, match="must be positive"):
        WorkspaceQuota(workspace, 0)
    assert workspace_size(tmp_path / "missing-workspace") == 0

    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    orphaned_lock = leases / "orphan.lock"
    orphaned_lock.touch()
    quota.ensure(0)
    assert not orphaned_lock.exists()


def test_quota_rejects_symlink_control_directories_and_escapes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    target = tmp_path / "control-target"
    target.mkdir()
    quota._control_root.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="control directory must not be a symlink"):
        quota.ensure(0)

    quota._control_root.unlink()
    outside = tmp_path / "outside"
    outside.mkdir()
    (workspace / "escape").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escaped"):
        _owned_path(workspace, "escape/.unsafe.partial")


def test_quota_counts_owned_paths_and_removes_temporary_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    with quota.lease() as lease:
        evidence = workspace / "lucas-reference-evidence-cleanup"
        lease.track_temporary_path(evidence)
        evidence.mkdir()
        (evidence / "scratch.partial").write_bytes(b"123")
        assert _path_size(evidence) == 3
        with pytest.raises(ValueError, match="overlap or escape"):
            quota._accounted_usage(
                {
                    "overlapping": {
                        "paths": {
                            "lucas-reference-evidence-cleanup": {
                                "accounted_bytes": 0,
                                "reserved_bytes": 0,
                            },
                            "lucas-reference-evidence-cleanup/scratch.partial": {
                                "accounted_bytes": 0,
                                "reserved_bytes": 0,
                            },
                        }
                    }
                }
            )

    assert not evidence.exists()


def test_quota_rejects_invalid_settlement_and_reserved_path_removal(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".reserved.partial"
    other = workspace / ".other.partial"
    with WorkspaceQuota(workspace, 8).lease() as lease:
        lease.track_temporary_path(temporary)
        with pytest.raises(ValueError, match="not owned"):
            lease.reserve(1, other)
        with pytest.raises(ValueError, match="bounds"):
            lease.reserve_up_to(0, temporary)
        with pytest.raises(ValueError, match="cannot be negative"):
            lease.consume(-1, temporary)
        with pytest.raises(ValueError, match="outstanding"):
            lease.release(1, temporary)

        lease.reserve(2, temporary)
        with pytest.raises(RuntimeError, match="outstanding"):
            lease.discard_temporary_path(temporary)
        with pytest.raises(RuntimeError, match="outstanding"):
            lease.forget_temporary_path(temporary)
        lease.release(2, temporary)


def test_lease_close_is_idempotent_and_failed_initialization_unlocks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    lease = quota.lease()
    descriptor = lease._descriptor
    lease._descriptor = None
    with pytest.raises(RuntimeError, match="no owner lock"):
        lease.close()
    lease._descriptor = descriptor
    lease.close()
    lease.close()

    def fail_state_write(_lease: WorkspaceQuotaLease) -> None:
        raise OSError("synthetic state write failure")

    monkeypatch.setattr(WorkspaceQuotaLease, "_write_state", fail_state_write)
    with pytest.raises(OSError, match="synthetic state write failure"):
        quota.lease()
    monkeypatch.undo()
    with quota.lease():
        pass


def test_quota_writer_accepts_empty_bytes_and_rejects_symlinked_owned_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_directory = workspace / "lucas-reference-evidence-link"
    linked_directory.mkdir()
    (linked_directory / "child.bin").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="must not contain symlinks"):
        _path_size(linked_directory)

    linked_partial = workspace / ".linked.partial"
    linked_partial.symlink_to(outside / "file")
    with WorkspaceQuota(workspace, 8).lease() as lease:
        with pytest.raises(ValueError, match="must not be a symlink"):
            lease.track_temporary_path(linked_partial)
        temporary = workspace / ".empty.partial"
        with lease.writer(temporary) as output:
            assert output.write(b"") == 0
