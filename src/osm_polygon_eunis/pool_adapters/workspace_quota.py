"""Cross-process byte reservations for temporary pool-controller outputs."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

_STATE_VERSION = 1
_RESERVATION_BLOCK_BYTES = 64 * 1024
_TEMP_DIRECTORY_PREFIX = "lucas-reference-evidence-"


class WorkspaceQuotaExceededError(RuntimeError):
    """A workspace write cannot fit under the configured byte cap."""


class WorkspaceQuota:
    """Atomically account workspace payload and active temporary-file leases."""

    def __init__(self, workspace: Path, maximum_bytes: int) -> None:
        """Bind quota accounting to one canonical workspace root."""

        if maximum_bytes <= 0:
            raise ValueError("workspace quota must be positive")
        if workspace.is_symlink():
            raise ValueError("workspace must not be a symlink")
        self.workspace = workspace.resolve()
        self.maximum_bytes = maximum_bytes
        digest = hashlib.sha256(os.fsencode(self.workspace)).hexdigest()[:20]
        self._control_root = self.workspace.parent / f".workspace-quota-{digest}"

    def lease(self) -> WorkspaceQuotaLease:
        """Create an owned reservation that releases itself after close or a crash."""

        return WorkspaceQuotaLease(self)

    def ensure(self, additional_bytes: int) -> None:
        """Check current usage and live reservations under the workspace lock."""

        if additional_bytes < 0:
            raise ValueError("quota reservation cannot be negative")
        with self._locked():
            states = self._active_states()
            self._require_capacity(states, additional_bytes)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self._control_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._control_root.is_symlink():
            raise ValueError("workspace quota control directory must not be a symlink")
        lock_path = self._control_root / "workspace.lock"
        descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)

    def _active_states(self) -> dict[str, dict[str, Any]]:
        self._control_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        leases = self._control_root / "leases"
        leases.mkdir(mode=0o700, exist_ok=True)
        states: dict[str, dict[str, Any]] = {}
        for state_path in sorted(leases.glob("*.json")):
            lease_id = state_path.stem
            lock_path = leases / f"{lease_id}.lock"
            descriptor = _try_acquire_lease_lock(lock_path)
            if descriptor is not None:
                try:
                    state = _read_lease_state(state_path, lease_id, self.workspace)
                    self._remove_owned_paths(state)
                    state_path.unlink(missing_ok=True)
                finally:
                    if descriptor >= 0:
                        fcntl.flock(descriptor, fcntl.LOCK_UN)
                        os.close(descriptor)
                lock_path.unlink(missing_ok=True)
                continue
            states[lease_id] = _read_lease_state(state_path, lease_id, self.workspace)
        for lock_path in leases.glob("*.lock"):
            state_path = lock_path.with_suffix(".json")
            if state_path.exists():
                continue
            descriptor = _try_acquire_lease_lock(lock_path)
            if descriptor is not None:
                if descriptor >= 0:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)
                lock_path.unlink(missing_ok=True)
        return states

    def _require_capacity(self, states: dict[str, dict[str, Any]], additional_bytes: int) -> None:
        used = self._accounted_usage(states)
        if used + additional_bytes > self.maximum_bytes:
            raise WorkspaceQuotaExceededError(
                f"workspace cap {self.maximum_bytes} bytes would be exceeded by a "
                f"{additional_bytes}-byte reservation"
            )

    def _accounted_usage(self, states: dict[str, dict[str, Any]]) -> int:
        total = workspace_size(self.workspace)
        managed: list[tuple[Path, int, int]] = []
        for state in states.values():
            for relative_path, counters in state["paths"].items():
                path = _owned_path(self.workspace, relative_path)
                actual_bytes = _path_size(path)
                managed.append(
                    (
                        path,
                        actual_bytes,
                        counters["accounted_bytes"] + counters["reserved_bytes"],
                    )
                )
        managed_actual = sum(actual for _, actual, _ in managed)
        if managed_actual > total:
            raise ValueError("workspace quota lease paths overlap or escape the workspace")
        return total - managed_actual + sum(max(actual, tracked) for _, actual, tracked in managed)

    def _remove_owned_paths(self, state: dict[str, Any]) -> None:
        for relative_path in state["paths"]:
            path = _owned_path(self.workspace, relative_path)
            if path.is_symlink():
                raise ValueError("owned temporary path must not be a symlink")
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)


class WorkspaceQuotaLease:
    """One process-owned reservation with crash-cleaned temporary paths."""

    def __init__(self, quota: WorkspaceQuota) -> None:
        """Acquire a process-liveness lock and create an empty reservation state."""

        self.quota = quota
        self.lease_id = uuid.uuid4().hex
        self._lease_root = quota._control_root / "leases"
        self._state_path = self._lease_root / f"{self.lease_id}.json"
        self._lock_path = self._lease_root / f"{self.lease_id}.lock"
        self._closed = False
        self._descriptor: int | None = None
        self._state: dict[str, Any] = {
            "lease_id": self.lease_id,
            "paths": {},
            "version": _STATE_VERSION,
        }
        with quota._locked():
            quota._active_states()
            self._descriptor = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(self._descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self._write_state()
            except Exception:
                os.close(self._descriptor)
                self._descriptor = None
                self._lock_path.unlink(missing_ok=True)
                raise

    def track_temporary_path(self, path: Path) -> None:
        """Register one exclusive, workspace-local partial file or LUCAS temp directory."""

        resolved, relative_path = _temporary_path(self.quota.workspace, path)
        with self.quota._locked():
            states = self.quota._active_states()
            if relative_path in self._state["paths"]:
                return
            for state in states.values():
                if any(_paths_overlap(relative_path, other) for other in state["paths"]):
                    raise RuntimeError("workspace temporary path already has an active owner")
            if resolved.exists() and (not resolved.is_dir() or any(resolved.iterdir())):
                raise FileExistsError(f"workspace temporary path already exists: {resolved}")
            self._state["paths"][relative_path] = {"accounted_bytes": 0, "reserved_bytes": 0}
            self._write_state()

    def reserve(self, size_bytes: int, path: Path) -> None:
        """Reserve exact bytes before a write to the registered temporary path."""

        granted = self.reserve_up_to(size_bytes, path, minimum_bytes=size_bytes)
        if granted != size_bytes:
            raise RuntimeError("exact workspace reservation returned a partial grant")

    def reserve_up_to(self, requested_bytes: int, path: Path, *, minimum_bytes: int = 1) -> int:
        """Reserve up to a block while guaranteeing a next write fits."""

        if requested_bytes <= 0 or minimum_bytes <= 0 or minimum_bytes > requested_bytes:
            raise ValueError("workspace reservation bounds must be positive and ordered")
        relative_path = _relative_owned_path(self.quota.workspace, path)
        with self.quota._locked():
            states = self.quota._active_states()
            tracked = self._tracked_path(relative_path)
            used = self.quota._accounted_usage(states)
            available = self.quota.maximum_bytes - used
            grant = min(requested_bytes, available)
            if grant < minimum_bytes:
                raise WorkspaceQuotaExceededError(
                    f"workspace cap {self.quota.maximum_bytes} bytes has only "
                    f"{max(available, 0)} bytes available for a {minimum_bytes}-byte write"
                )
            tracked["reserved_bytes"] += grant
            self._write_state()
            return grant

    def consume(self, size_bytes: int, path: Path) -> None:
        """Convert flushed reservation bytes into accounted temporary-file bytes."""

        self._settle(size_bytes, path, consume=True)

    def release(self, size_bytes: int, path: Path) -> None:
        """Return unused reservation bytes to the shared workspace budget."""

        self._settle(size_bytes, path, consume=False)

    def _settle(self, size_bytes: int, path: Path, *, consume: bool) -> None:
        if size_bytes < 0:
            raise ValueError("workspace reservation settlement cannot be negative")
        relative_path = _relative_owned_path(self.quota.workspace, path)
        with self.quota._locked():
            self.quota._active_states()
            tracked = self._tracked_path(relative_path)
            reserved = tracked["reserved_bytes"]
            if size_bytes > reserved:
                raise ValueError("workspace reservation settlement exceeds outstanding bytes")
            tracked["reserved_bytes"] -= size_bytes
            if consume:
                tracked["accounted_bytes"] += size_bytes
            self._write_state()

    def promote_temporary_path(self, temporary: Path, target: Path) -> None:
        """Atomically install a completed file and stop tracking its partial name."""

        source_relative = _relative_owned_path(self.quota.workspace, temporary)
        target_resolved = _workspace_path(self.quota.workspace, target)
        with self.quota._locked():
            self.quota._active_states()
            tracked = self._tracked_path(source_relative)
            if tracked["reserved_bytes"]:
                raise RuntimeError("cannot install a file with outstanding quota reservations")
            source = _owned_path(self.quota.workspace, source_relative)
            if target_resolved.exists() or target_resolved.is_symlink():
                raise FileExistsError(target_resolved)
            source.replace(target_resolved)
            del self._state["paths"][source_relative]
            self._write_state()

    def discard_temporary_path(self, path: Path) -> None:
        """Delete one owned partial file and release its quota accounting."""

        relative_path = _relative_owned_path(self.quota.workspace, path)
        with self.quota._locked():
            self.quota._active_states()
            tracked = self._tracked_path(relative_path)
            if tracked["reserved_bytes"]:
                raise RuntimeError("cannot discard a file with outstanding quota reservations")
            self.quota._remove_owned_paths({"paths": {relative_path: tracked}})
            del self._state["paths"][relative_path]
            self._write_state()

    def forget_temporary_path(self, path: Path) -> None:
        """Forget a temporary directory after its owner has removed it."""

        relative_path = _relative_owned_path(self.quota.workspace, path)
        with self.quota._locked():
            self.quota._active_states()
            tracked = self._tracked_path(relative_path)
            if path.exists() or path.is_symlink():
                raise RuntimeError("cannot forget a temporary path while it still exists")
            if tracked["reserved_bytes"]:
                raise RuntimeError("cannot forget a path with outstanding quota reservations")
            del self._state["paths"][relative_path]
            self._write_state()

    def writer(self, path: Path) -> WorkspaceQuotaWriter:
        """Open a buffered binary writer backed by this lease."""

        return WorkspaceQuotaWriter(self, path)

    def close(self) -> None:
        """Remove unpromoted temporary paths and release the process lease."""

        if self._closed:
            return
        descriptor = self._descriptor
        if descriptor is None:
            raise RuntimeError("workspace quota lease has no owner lock")
        try:
            with self.quota._locked():
                self.quota._remove_owned_paths(self._state)
                self._state["paths"].clear()
                self._state_path.unlink(missing_ok=True)
        finally:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
            os.close(descriptor)
            self._descriptor = None
            self._closed = True
            self._lock_path.unlink(missing_ok=True)

    def __enter__(self) -> WorkspaceQuotaLease:
        """Return the open lease for use as a context manager."""

        return self

    def __exit__(self, *_exc: object) -> None:
        """Release reservations and remove unpromoted temporary paths."""

        self.close()

    def _tracked_path(self, relative_path: str) -> dict[str, int]:
        try:
            return self._state["paths"][relative_path]
        except KeyError as error:
            raise ValueError("workspace output path is not owned by this quota lease") from error

    def _write_state(self) -> None:
        temporary = self._state_path.with_name(f".{self.lease_id}.{uuid.uuid4().hex}.tmp")
        raw = json.dumps(self._state, sort_keys=True, separators=(",", ":")).encode("utf-8")
        descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        try:
            with os.fdopen(descriptor, "wb", closefd=False) as output:
                output.write(raw)
                output.flush()
                os.fsync(output.fileno())
            temporary.replace(self._state_path)
        finally:
            os.close(descriptor)
            temporary.unlink(missing_ok=True)


class WorkspaceQuotaWriter:
    """Buffer writes while reserving every byte before it can reach disk."""

    def __init__(self, lease: WorkspaceQuotaLease, path: Path) -> None:
        """Bind a buffered writer to one previously registered temporary path."""

        self.lease = lease
        self.path = path
        self._stream: BinaryIO | None = None
        self._reserved_bytes = 0
        self._pending_bytes = 0

    def __enter__(self) -> WorkspaceQuotaWriter:
        """Register and open the temporary file before its first write."""

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lease.track_temporary_path(self.path)
        self._stream = self.path.open("xb")
        return self

    def write(self, data: bytes) -> int:
        """Reserve before buffering bytes and return the number written."""

        stream = self._require_stream()
        if not isinstance(data, bytes):
            raise TypeError("workspace quota writer accepts bytes only")
        if not data:
            return 0
        if self._reserved_bytes < len(data):
            self.flush()
            grant = self.lease.reserve_up_to(
                max(_RESERVATION_BLOCK_BYTES, len(data)),
                self.path,
                minimum_bytes=len(data),
            )
            self._reserved_bytes = grant
        written = stream.write(data)
        if written != len(data):
            raise OSError("workspace quota writer could not write the complete buffer")
        self._pending_bytes += written
        self._reserved_bytes -= written
        return written

    def flush(self) -> None:
        """Flush buffered bytes and convert their reservation to file usage."""

        stream = self._require_stream()
        stream.flush()
        if self._pending_bytes:
            self.lease.consume(self._pending_bytes, self.path)
        if self._reserved_bytes:
            self.lease.release(self._reserved_bytes, self.path)
        self._pending_bytes = 0
        self._reserved_bytes = 0

    def fileno(self) -> int:
        """Return the underlying file descriptor for fsync."""

        return self._require_stream().fileno()

    def __exit__(self, exc_type: object, *_exc: object) -> None:
        """Flush successful writes and close the underlying file."""

        stream = self._require_stream()
        try:
            if exc_type is None:
                self.flush()
            stream.close()
        finally:
            self._stream = None

    def _require_stream(self) -> BinaryIO:
        if self._stream is None:
            raise RuntimeError("workspace quota writer is not open")
        return self._stream


def workspace_size(workspace: Path) -> int:
    """Return actual regular-file payload bytes and reject symlink escapes."""

    if workspace.is_symlink():
        raise ValueError("workspace must not be a symlink")
    if not workspace.exists():
        return 0
    total = 0
    for path in workspace.rglob("*"):
        if path.is_symlink():
            raise ValueError("workspace must not contain symlinks")
        if path.is_file():
            total += path.stat().st_size
    return total


def _try_acquire_lease_lock(path: Path) -> int | None:
    if not path.exists():
        return -1
    try:
        descriptor = os.open(path, os.O_RDWR)
    except FileNotFoundError:
        return -1
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _read_lease_state(path: Path, lease_id: str, workspace: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"workspace quota lease state is invalid: {path}") from error
    if (
        not isinstance(value, dict)
        or value.get("version") != _STATE_VERSION
        or value.get("lease_id") != lease_id
        or not isinstance(value.get("paths"), dict)
    ):
        raise ValueError(f"workspace quota lease state has an unsupported shape: {path}")
    for relative_path, counters in value["paths"].items():
        if (
            not isinstance(relative_path, str)
            or not isinstance(counters, dict)
            or not _valid_counter(counters.get("reserved_bytes"))
            or not _valid_counter(counters.get("accounted_bytes"))
        ):
            raise ValueError(f"workspace quota lease path counters are invalid: {path}")
        _owned_path(workspace, relative_path)
    return value


def _valid_counter(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _temporary_path(workspace: Path, path: Path) -> tuple[Path, str]:
    resolved = _workspace_path(workspace, path)
    if resolved.name.endswith(".partial"):
        return resolved, resolved.relative_to(workspace).as_posix()
    if resolved.name.startswith(_TEMP_DIRECTORY_PREFIX):
        return resolved, resolved.relative_to(workspace).as_posix()
    raise ValueError("owned workspace path must be a controller temporary file or directory")


def _relative_owned_path(workspace: Path, path: Path) -> str:
    resolved = _workspace_path(workspace, path)
    return resolved.relative_to(workspace).as_posix()


def _workspace_path(workspace: Path, path: Path) -> Path:
    candidate = path if path.is_absolute() else workspace / path
    if candidate.is_symlink():
        raise ValueError("owned temporary path must not be a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("owned temporary path escaped the bounded workspace") from error
    return resolved


def _owned_path(workspace: Path, relative_path: str) -> Path:
    relative = Path(relative_path)
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise ValueError("workspace quota state contains an unsafe temporary path")
    candidate = workspace / relative
    if candidate.is_symlink():
        raise ValueError("owned temporary path must not be a symlink")
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("owned temporary path escaped the bounded workspace") from error
    if not (resolved.name.endswith(".partial") or resolved.name.startswith(_TEMP_DIRECTORY_PREFIX)):
        raise ValueError("workspace quota state names a non-temporary path")
    return resolved


def _paths_overlap(first: str, second: str) -> bool:
    first_path = Path(first)
    second_path = Path(second)
    return (
        first_path == second_path
        or first_path in second_path.parents
        or second_path in first_path.parents
    )


def _path_size(path: Path) -> int:
    if path.is_symlink():
        raise ValueError("owned temporary path must not be a symlink")
    if path.is_file():
        return path.stat().st_size
    if not path.is_dir():
        return 0
    total = 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise ValueError("owned temporary directory must not contain symlinks")
        if child.is_file():
            total += child.stat().st_size
    return total
