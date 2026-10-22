from __future__ import annotations

import fcntl
import hashlib
import json
import multiprocessing
import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest

import osm_polygon_eunis.pool_adapters.workspace_quota as quota_module
from osm_polygon_eunis.pool_adapters.workspace_quota import (
    WorkspaceQuota,
    WorkspaceQuotaExceededError,
    WorkspaceQuotaLease,
    _owned_path,
    _path_size,
    _paths_overlap,
    _read_lease_state,
    _temporary_path,
    _try_acquire_lease_lock,
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


def _own_shared_partial(
    workspace_text: str,
    owner_ready: Any,
    competitor_finished: Any,
    results: Any,
) -> None:
    workspace = Path(workspace_text)
    temporary = workspace / ".shared.partial"
    target = workspace / "shared.bin"
    with WorkspaceQuota(workspace, 64).lease() as lease:
        with lease.writer(temporary) as output:
            output.write(b"active-owner-data")
            output.flush()
        owner_ready.set()
        if not competitor_finished.wait(timeout=20):
            results.put(("owner-timeout", b""))
            return
        if temporary.read_bytes() != b"active-owner-data":
            results.put(("owner-bytes-changed", temporary.read_bytes()))
            return
        lease.promote_temporary_path(temporary, target)
        results.put(("owner-completed", target.read_bytes()))


def _reject_shared_partial(
    workspace_text: str,
    owner_ready: Any,
    competitor_finished: Any,
    results: Any,
) -> None:
    workspace = Path(workspace_text)
    temporary = workspace / ".shared.partial"
    try:
        if not owner_ready.wait(timeout=20):
            results.put(("owner-not-ready", b""))
            return
        with (
            WorkspaceQuota(workspace, 64).lease() as lease,
            lease.writer(temporary) as output,
        ):
            output.write(b"competing-data")
    except RuntimeError as error:
        results.put(("rejected", str(error).encode("utf-8")))
    else:
        results.put(("unexpected-success", b""))
    finally:
        competitor_finished.set()


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


@pytest.mark.slow
def test_rejected_process_cannot_remove_another_lease_partial(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    context = multiprocessing.get_context("spawn")
    owner_ready = context.Event()
    competitor_finished = context.Event()
    results = context.Queue()
    owner = context.Process(
        target=_own_shared_partial,
        args=(str(workspace), owner_ready, competitor_finished, results),
    )
    competitor = context.Process(
        target=_reject_shared_partial,
        args=(str(workspace), owner_ready, competitor_finished, results),
    )
    owner.start()
    competitor.start()
    observed = [results.get(timeout=30), results.get(timeout=30)]
    owner.join(timeout=30)
    competitor.join(timeout=30)

    assert owner.exitcode == 0
    assert competitor.exitcode == 0
    assert sorted(row[0] for row in observed) == ["owner-completed", "rejected"]
    owner_result = next(row for row in observed if row[0] == "owner-completed")
    assert owner_result[1] == b"active-owner-data"
    rejection = next(row for row in observed if row[0] == "rejected")
    assert b"already has an active owner" in rejection[1]
    assert (workspace / "shared.bin").read_bytes() == b"active-owner-data"


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


def test_quota_sums_multiple_files_and_existing_partial_bytes(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "first.bin").write_bytes(b"abc")
    (workspace / "second.bin").write_bytes(b"de")
    assert workspace_size(workspace) == 5

    quota = WorkspaceQuota(workspace, 8)
    with quota.lease() as lease:
        temporary = workspace / ".unreserved.partial"
        lease.track_temporary_path(temporary)
        temporary.write_bytes(b"fg")
        quota.ensure(1)
        with pytest.raises(WorkspaceQuotaExceededError):
            quota.ensure(2)


def test_quota_adds_repeated_reservations_for_one_owned_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    first_path = workspace / ".first.partial"
    second_path = workspace / ".second.partial"

    with quota.lease() as first:
        first.track_temporary_path(first_path)
        assert first.reserve_up_to(4, first_path) == 4
        assert first.reserve_up_to(4, first_path) == 4
        with quota.lease() as second:
            second.track_temporary_path(second_path)
            with pytest.raises(WorkspaceQuotaExceededError):
                second.reserve(1, second_path)


def test_reservation_control_roots_are_distinct_for_distinct_workspaces(
    tmp_path: Path,
) -> None:
    first_workspace = tmp_path / "first"
    second_workspace = tmp_path / "second"
    first_workspace.mkdir()
    second_workspace.mkdir()

    first = WorkspaceQuota(first_workspace, 8)
    second = WorkspaceQuota(second_workspace, 8)

    assert first._control_root != second._control_root
    assert first._control_root == WorkspaceQuota(first_workspace, 8)._control_root


def test_default_reservation_accepts_one_byte_and_zero_minimum_is_rejected(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".one-byte.partial"
    with WorkspaceQuota(workspace, 1).lease() as lease:
        lease.track_temporary_path(temporary)
        assert lease.reserve_up_to(1, temporary) == 1
        with pytest.raises(ValueError, match="bounds"):
            lease.reserve_up_to(1, temporary, minimum_bytes=0)
        with pytest.raises(ValueError, match="bounds"):
            lease.reserve_up_to(1, temporary, minimum_bytes=2)


def test_exact_reservation_rejects_a_partial_grant(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".partial-grant.partial"
    quota = WorkspaceQuota(workspace, 2)

    with quota.lease() as lease:
        lease.track_temporary_path(temporary)
        with pytest.raises(WorkspaceQuotaExceededError):
            lease.reserve(3, temporary)
        quota.ensure(2)


def test_relative_temporary_paths_resolve_inside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    assert _temporary_path(workspace, Path(".relative.partial")) == (
        workspace / ".relative.partial",
        ".relative.partial",
    )


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

        occupied_directory = workspace / "lucas-reference-evidence-occupied"
        occupied_directory.mkdir()
        (occupied_directory / "existing.partial").write_bytes(b"x")
        with pytest.raises(FileExistsError):
            owner.track_temporary_path(occupied_directory)


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
            output.write(b"d")
        assert temporary.read_bytes() == b"abcd"
        assert workspace_size(workspace) == 4
        lease.discard_temporary_path(temporary)
        assert not temporary.exists()


def test_writer_rejects_a_buffer_larger_than_remaining_capacity_before_writing(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 4)
    temporary = workspace / ".too-large.partial"

    with quota.lease() as lease, lease.writer(temporary) as output:
        with pytest.raises(WorkspaceQuotaExceededError, match="5-byte write"):
            output.write(b"12345")
        assert temporary.read_bytes() == b""


def test_writer_reports_zero_capacity_when_workspace_is_full(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "existing.bin").write_bytes(b"full")
    temporary = workspace / ".no-capacity.partial"

    with WorkspaceQuota(workspace, 4).lease() as lease, lease.writer(temporary) as output:
        with pytest.raises(WorkspaceQuotaExceededError, match="only 0 bytes available"):
            output.write(b"x")
        assert temporary.read_bytes() == b""


def test_writer_creates_missing_parent_directories_and_rejects_reuse(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / "nested" / "deeper" / ".payload.partial"

    with WorkspaceQuota(workspace, 8).lease() as lease:
        writer = lease.writer(temporary)
        with writer as output:
            output.write(b"data")
        assert temporary.read_bytes() == b"data"
        with pytest.raises(RuntimeError, match="not open"):
            writer.write(b"again")


def test_writer_uses_an_exact_reservation_under_concurrent_pressure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 64 * 1024)
    owner_path = workspace / ".owner.partial"
    competitor_path = workspace / ".competitor.partial"

    with quota.lease() as owner, quota.lease() as competitor:
        owner.track_temporary_path(owner_path)
        competitor.track_temporary_path(competitor_path)
        reserve_up_to = owner.reserve_up_to
        calls = 0

        def reserve_with_competitor(
            requested_bytes: int,
            path: Path,
            *,
            minimum_bytes: int = 1,
        ) -> int:
            nonlocal calls
            calls += 1
            if calls == 2:
                competitor.reserve(1, competitor_path)
            return reserve_up_to(requested_bytes, path, minimum_bytes=minimum_bytes)

        monkeypatch.setattr(owner, "reserve_up_to", reserve_with_competitor)
        with owner.writer(owner_path) as output:
            output.write(b"x" * (64 * 1024 - 1))
            output.write(b"y")
        assert owner_path.read_bytes() == b"x" * (64 * 1024 - 1) + b"y"


def test_writer_accumulates_flushed_bytes_against_the_quota(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 6)
    temporary = workspace / ".payload.partial"

    with quota.lease() as lease:
        with lease.writer(temporary) as output:
            output.write(b"abc")
            output.flush()
            output.write(b"def")
            output.flush()
        assert temporary.read_bytes() == b"abcdef"
        with pytest.raises(WorkspaceQuotaExceededError):
            quota.ensure(1)


def test_writer_accounts_for_multiple_buffered_writes_before_flush(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".payload.partial"

    with WorkspaceQuota(workspace, 8).lease() as lease, lease.writer(temporary) as output:
        output.write(b"abc")
        output.write(b"def")
        output.flush()
        assert lease._state["paths"][str(temporary.relative_to(workspace))] == {
            "accounted_bytes": 6,
            "reserved_bytes": 0,
        }


def test_consumed_quota_survives_partial_file_truncation(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 6)
    temporary = workspace / ".payload.partial"

    with quota.lease() as lease:
        with lease.writer(temporary) as output:
            output.write(b"abc")
            output.flush()
            output.write(b"def")
            output.flush()
        temporary.write_bytes(b"")
        with pytest.raises(WorkspaceQuotaExceededError):
            quota.ensure(3)


def test_zero_byte_settlement_is_valid(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    temporary = workspace / ".zero.partial"
    with WorkspaceQuota(workspace, 8).lease() as lease:
        lease.track_temporary_path(temporary)
        lease.consume(0, temporary)
        lease.release(0, temporary)


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
        json.dumps({"version": 1, "lease_id": "wrong-id", "paths": {}}),
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

    state_path.write_text(
        json.dumps(
            {
                "version": 1,
                "lease_id": lease_id,
                "paths": {".unsafe.partial": []},
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


def test_quota_reclaims_every_stale_lease_state_and_lock(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    for lease_id in ("stale-a", "stale-b"):
        (leases / f"{lease_id}.json").write_text(
            json.dumps({"lease_id": lease_id, "paths": {}, "version": 1}),
            encoding="utf-8",
        )
        (leases / f"{lease_id}.lock").touch()

    assert quota._active_states() == {}
    assert list(leases.iterdir()) == []


@pytest.mark.parametrize("target_suffix", (".json", ".lock"))
def test_stale_lease_cleanup_tolerates_path_disappearing_during_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_suffix: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    state_path = leases / "racing-lease.json"
    lock_path = leases / "racing-lease.lock"
    state_path.write_text(
        json.dumps({"lease_id": "racing-lease", "paths": {}, "version": 1}),
        encoding="utf-8",
    )
    lock_path.touch()
    disappearing_path = state_path if target_suffix == ".json" else lock_path
    original_unlink = Path.unlink
    disappeared = False

    def disappear_before_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal disappeared
        if path == disappearing_path and not disappeared:
            disappeared = True
            original_unlink(path, missing_ok=True)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", disappear_before_unlink)

    assert quota._active_states() == {}
    assert disappeared
    assert not state_path.exists()
    assert not lock_path.exists()


def test_orphan_lock_cleanup_tolerates_path_disappearing_during_unlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    orphan_lock = leases / "orphan.lock"
    orphan_lock.touch()
    original_unlink = Path.unlink
    disappeared = False

    def disappear_before_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal disappeared
        if path == orphan_lock and not disappeared:
            disappeared = True
            original_unlink(path, missing_ok=True)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", disappear_before_unlink)

    assert quota._active_states() == {}
    assert disappeared
    assert not orphan_lock.exists()


@pytest.mark.parametrize("target_name", ("state", "lock"))
def test_lease_close_tolerates_its_state_or_lock_disappearing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target_name: str,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lease = WorkspaceQuota(workspace, 8).lease()
    disappearing_path = lease._state_path if target_name == "state" else lease._lock_path
    original_unlink = Path.unlink
    disappeared = False

    def disappear_before_unlink(path: Path, *, missing_ok: bool = False) -> None:
        nonlocal disappeared
        if path == disappearing_path and not disappeared:
            disappeared = True
            original_unlink(path, missing_ok=True)
        original_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(Path, "unlink", disappear_before_unlink)

    lease.close()
    assert disappeared
    assert not lease._state_path.exists()
    assert not lease._lock_path.exists()


def test_failed_lease_initialization_preserves_original_error_when_lock_vanishes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)

    def fail_after_removing_lock(lease: WorkspaceQuotaLease) -> None:
        lease._lock_path.unlink()
        raise RuntimeError("injected state write failure")

    monkeypatch.setattr(WorkspaceQuotaLease, "_write_state", fail_after_removing_lock)

    with pytest.raises(RuntimeError, match="injected state write failure"):
        WorkspaceQuotaLease(quota)


def test_quota_state_directories_and_lease_files_are_private(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    previous_umask = os.umask(0)
    try:
        assert quota._active_states() == {}
        assert stat.S_IMODE(quota._control_root.stat().st_mode) & ~0o700 == 0
        leases = quota._control_root / "leases"
        assert stat.S_IMODE(leases.stat().st_mode) & ~0o700 == 0

        with quota.lease() as lease:
            for path in (lease._state_path, lease._lock_path):
                assert stat.S_IMODE(path.stat().st_mode) & ~0o600 == 0
    finally:
        os.umask(previous_umask)


def test_quota_creates_missing_control_parent_directories(tmp_path: Path) -> None:
    workspace = tmp_path / "missing" / "parent" / "workspace"
    assert WorkspaceQuota(workspace, 8)._active_states() == {}


def test_quota_control_path_stably_names_the_canonical_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    canonical_workspace = workspace.resolve()
    digest = hashlib.sha256(os.fsencode(canonical_workspace)).hexdigest()[:20]

    assert WorkspaceQuota(workspace, 8)._control_root == (
        canonical_workspace.parent / f".workspace-quota-{digest}"
    )


def test_missing_lease_lock_does_not_leave_a_stale_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    missing_lock = leases / "absent.lock"
    original_exists = Path.exists

    def exists_during_unlink_race(path: Path) -> bool:
        if path == missing_lock:
            return True
        return original_exists(path)

    monkeypatch.setattr(Path, "exists", exists_during_unlink_race)
    missing_lock_descriptor = _try_acquire_lease_lock(missing_lock)
    assert missing_lock_descriptor is not None and missing_lock_descriptor < 0
    monkeypatch.setattr(Path, "exists", original_exists)
    lease_id = "missing-lock"
    state_path = leases / f"{lease_id}.json"
    state_path.write_text(
        json.dumps({"lease_id": lease_id, "paths": {}, "version": 1}),
        encoding="utf-8",
    )

    assert quota._active_states() == {}
    assert not state_path.exists()


def test_active_lease_lock_does_not_block_cleanup_of_later_orphans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    active_id = "active-lease"
    active_state = leases / f"{active_id}.json"
    active_lock = leases / f"{active_id}.lock"
    active_state.write_text(
        json.dumps({"lease_id": active_id, "paths": {}, "version": 1}),
        encoding="utf-8",
    )
    descriptor = os.open(active_lock, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    orphan_lock = leases / "orphan.lock"
    orphan_lock.touch()
    original_glob = Path.glob

    def ordered_lock_glob(path: Path, pattern: str):
        if path == leases and pattern == "*.lock":
            return iter((active_lock, orphan_lock))
        return original_glob(path, pattern)

    monkeypatch.setattr(Path, "glob", ordered_lock_glob)

    try:
        assert set(quota._active_states()) == {active_id}
        assert not orphan_lock.exists()
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        active_state.unlink(missing_ok=True)
        active_lock.unlink(missing_ok=True)


def test_zero_lease_descriptor_is_released(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    quota = WorkspaceQuota(workspace, 8)
    leases = quota._control_root / "leases"
    leases.mkdir(parents=True)
    (leases / "zero-descriptor.json").write_text(
        json.dumps({"lease_id": "zero-descriptor", "paths": {}, "version": 1}),
        encoding="utf-8",
    )
    (leases / "zero-descriptor.lock").touch()
    (leases / "orphan.lock").touch()
    observed: list[tuple[str, int]] = []
    monkeypatch.setattr(quota_module, "_try_acquire_lease_lock", lambda _path: 0)
    monkeypatch.setattr(
        quota_module.fcntl,
        "flock",
        lambda descriptor, _operation: observed.append(("flock", descriptor)),
    )
    monkeypatch.setattr(
        quota_module.os,
        "close",
        lambda descriptor: observed.append(("close", descriptor)),
    )

    assert quota._active_states() == {}
    assert observed == [("flock", 0), ("close", 0), ("flock", 0), ("close", 0)]
    assert not (leases / "zero-descriptor.json").exists()
    assert not (leases / "zero-descriptor.lock").exists()
    assert not (leases / "orphan.lock").exists()


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
        (evidence / "other.partial").write_bytes(b"45")
        assert _path_size(evidence) == 5
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
