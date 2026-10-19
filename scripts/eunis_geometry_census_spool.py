"""Pinned, ordered geometry-row spools for restartable census regions."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import re
import time
import uuid
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

SPOOL_CHUNK_ROWS = 256
SPOOL_SCHEMA_VERSION = 1
SPOOL_ORDER_VERSION = "sha256-source-decision-stream-sentence-key-v1"
_BYTE_MARKER = "__eunis_spool_bytes_v1__"
_HASH_PATTERN = re.compile(r"[0-9a-f]{64}")
_ATTEMPT_PATTERN = re.compile(r"[0-9a-f]{32}")
_CHUNK_METADATA_PATTERN = re.compile(r"chunk-(\d{8})\.json")


@dataclass(frozen=True, slots=True)
class OrderedGeometrySpoolIdentity:
    """Immutable source and algorithm identity for one ordered region stream."""

    run_signature: str
    region_signature: str
    code_sha256: str
    source: str
    stream: str
    shard: str
    label_repository: str
    label_revision: str
    label_path: str
    raw_repository: str
    raw_revision: str
    raw_paths: tuple[str, ...]
    query_sha256: str
    chunk_rows: int = SPOOL_CHUNK_ROWS
    schema_version: int = SPOOL_SCHEMA_VERSION
    order_version: str = SPOOL_ORDER_VERSION

    def __post_init__(self) -> None:
        """Reject malformed or unsupported spool identities."""
        for name in ("run_signature", "region_signature", "code_sha256", "query_sha256"):
            if _HASH_PATTERN.fullmatch(getattr(self, name)) is None:
                raise ValueError(f"spool identity has malformed {name}")
        if self.chunk_rows < 1:
            raise ValueError("spool chunk row count must be positive")
        if self.schema_version != SPOOL_SCHEMA_VERSION:
            raise ValueError("unsupported geometry spool schema version")
        if self.order_version != SPOOL_ORDER_VERSION:
            raise ValueError("unsupported geometry spool order version")
        if (
            not all(
                (
                    self.source,
                    self.stream,
                    self.shard,
                    self.label_repository,
                    self.label_revision,
                    self.label_path,
                    self.raw_repository,
                    self.raw_revision,
                )
            )
            or not self.raw_paths
        ):
            raise ValueError("spool identity is missing a pinned source field")

    def payload(self) -> dict[str, object]:
        """Return the canonical JSON object persisted in spool metadata."""
        return {
            "schema_version": self.schema_version,
            "order_version": self.order_version,
            "chunk_rows": self.chunk_rows,
            "run_signature": self.run_signature,
            "region_signature": self.region_signature,
            "code_sha256": self.code_sha256,
            "source": self.source,
            "stream": self.stream,
            "shard": self.shard,
            "label_repository": self.label_repository,
            "label_revision": self.label_revision,
            "label_path": self.label_path,
            "raw_repository": self.raw_repository,
            "raw_revision": self.raw_revision,
            "raw_paths": list(self.raw_paths),
            "query_sha256": self.query_sha256,
        }

    @property
    def sha256(self) -> str:
        """Return the digest of this complete pinned identity."""
        return hashlib.sha256(_canonical_json(self.payload())).hexdigest()


@dataclass(frozen=True, slots=True)
class _SavedChunk:
    index: int
    data_path: Path
    metadata_path: Path
    metadata: dict[str, object]


@dataclass(slots=True)
class _SpoolBuild:
    source_rows: Iterator[Mapping[str, object]]
    attempt_id: str
    attempt_path: Path
    saved_chunks: tuple[_SavedChunk, ...]
    recovered_rows: int
    source_query_seconds: float
    chunk_write_seconds: float = 0.0
    chunks: list[dict[str, object]] = field(default_factory=list)
    digest: Any = field(default_factory=hashlib.sha256)
    row_count: int = 0
    byte_count: int = 0
    pending: list[bytes] = field(default_factory=list)
    chunk_index: int = 0

    def record(self, line: bytes) -> None:
        self.digest.update(line)
        self.row_count += 1
        self.byte_count += len(line)


@dataclass(slots=True)
class OrderedGeometrySpool:
    """Stream rows into a private spool while preserving query consumption order."""

    root: Path
    identity: OrderedGeometrySpoolIdentity
    mode: str
    build: _SpoolBuild | None
    manifest: dict[str, object] | None
    recovered_rows: int
    validation_seconds: float
    read_seconds: float = 0.0
    _rows_started: bool = False

    def __post_init__(self) -> None:
        """Reject impossible combinations of complete and in-progress state."""
        if (self.build is None) == (self.manifest is None):
            raise ValueError("geometry spool must be either building or complete")

    @property
    def source_query_seconds(self) -> float:
        """Return time spent advancing the pinned geometry query."""
        if self.build is not None:
            return self.build.source_query_seconds
        return _float_field(self._completed_manifest(), "source_query_seconds", default=0.0)

    @property
    def chunk_write_seconds(self) -> float:
        """Return time spent encoding and durably writing spool chunks."""
        if self.build is not None:
            return self.build.chunk_write_seconds
        return _float_field(self._completed_manifest(), "chunk_write_seconds", default=0.0)

    @property
    def row_count(self) -> int:
        """Return the number of geometry rows after the stream is complete."""
        return _int_field(self._completed_manifest(), "row_count")

    @property
    def byte_count(self) -> int:
        """Return the canonical encoded spool byte count after completion."""
        return _int_field(self._completed_manifest(), "byte_count")

    @property
    def content_sha256(self) -> str:
        """Return the digest of all ordered encoded geometry rows."""
        return str(self._completed_manifest()["content_sha256"])

    @property
    def creation_seconds(self) -> float:
        """Return source-query and chunk-write time for the published spool."""
        return _float_field(self._completed_manifest(), "creation_seconds")

    @property
    def reuse_seconds(self) -> float:
        """Return validation and row-read time for the current reuse attempt."""
        return self.validation_seconds + self.read_seconds

    def _completed_manifest(self) -> dict[str, object]:
        if self.manifest is None:
            raise RuntimeError("ordered geometry spool is not complete")
        return self.manifest

    def rows(self) -> Iterator[dict[str, object]]:
        """Yield original-order rows and persist verified chunks as the query streams."""
        if self._rows_started:
            raise RuntimeError("ordered geometry spool rows can only be consumed once")
        self._rows_started = True
        if self.manifest is not None:
            source = iter(_iter_manifest_rows(self.root, self.manifest))
            while True:
                started = time.monotonic()
                try:
                    row = next(source)
                except StopIteration:
                    self.read_seconds += time.monotonic() - started
                    break
                self.read_seconds += time.monotonic() - started
                yield row
            return
        if self.build is None:
            raise RuntimeError("ordered geometry spool build state is incomplete")
        yield from self._stream_rows()

    def _stream_rows(self) -> Iterator[dict[str, object]]:
        if self.build is None:
            raise RuntimeError("geometry spool build state is missing")
        build = self.build
        build.chunks = [
            _manifest_chunk(chunk.metadata, is_final=chunk.metadata["final"] is True)
            for chunk in build.saved_chunks
        ]
        build.chunk_index = len(build.chunks)
        for line, row in self._recovered_records(build):
            build.record(line)
            yield row
        for row, line in self._fresh_records(build):
            build.record(line)
            build.pending.append(line)
            if len(build.pending) == self.identity.chunk_rows:
                self._publish_full_chunk(build)
            yield dict(row)
        self._finish_chunks(build)
        self._publish_complete(build)

    def _recovered_records(self, build: _SpoolBuild) -> Iterator[tuple[bytes, dict[str, object]]]:
        for saved in build.saved_chunks:
            started = time.monotonic()
            saved_rows = _read_verified_saved_rows(saved)
            self.read_seconds += time.monotonic() - started
            yield from saved_rows

    def _fresh_records(self, build: _SpoolBuild) -> Iterator[tuple[Mapping[str, object], bytes]]:
        while True:
            started = time.monotonic()
            try:
                row = next(build.source_rows)
            except StopIteration:
                build.source_query_seconds += time.monotonic() - started
                return
            build.source_query_seconds += time.monotonic() - started
            started = time.monotonic()
            line = _encode_row(row)
            build.chunk_write_seconds += time.monotonic() - started
            yield row, line

    def _publish_full_chunk(self, build: _SpoolBuild) -> None:
        started = time.monotonic()
        _chunk, metadata = _publish_chunk(
            build.attempt_path / "chunks",
            build.chunk_index,
            build.pending,
            final=False,
        )
        build.chunk_write_seconds += time.monotonic() - started
        build.chunks.append(_manifest_chunk(metadata, is_final=False))
        build.chunk_index += 1
        build.pending = []

    def _finish_chunks(self, build: _SpoolBuild) -> None:
        if build.pending:
            started = time.monotonic()
            _chunk, metadata = _publish_chunk(
                build.attempt_path / "chunks",
                build.chunk_index,
                build.pending,
                final=True,
            )
            build.chunk_write_seconds += time.monotonic() - started
            build.chunks.append(_manifest_chunk(metadata, is_final=True))
        if build.chunks:
            build.chunks[-1]["is_final"] = True

    def _publish_complete(self, build: _SpoolBuild) -> None:
        creation_seconds = build.source_query_seconds + build.chunk_write_seconds
        manifest = {
            "schema_version": SPOOL_SCHEMA_VERSION,
            "identity": self.identity.payload(),
            "identity_sha256": self.identity.sha256,
            "attempt_id": build.attempt_id,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "creation_seconds": creation_seconds,
            "source_query_seconds": build.source_query_seconds,
            "chunk_write_seconds": build.chunk_write_seconds,
            "row_count": build.row_count,
            "byte_count": build.byte_count,
            "content_sha256": build.digest.hexdigest(),
            "chunks": build.chunks,
        }
        manifest_path = build.attempt_path / "manifest.json"
        _atomic_write(manifest_path, _pretty_json(manifest))
        _fsync_directory(build.attempt_path.parent)
        complete_path = self.root / "complete.json"
        try:
            os.link(manifest_path, complete_path)
        except FileExistsError:
            winner = _load_complete(self.root, self.identity, started=time.monotonic())
            if (
                winner.content_sha256 != manifest["content_sha256"]
                or winner.row_count != build.row_count
                or winner.byte_count != build.byte_count
            ):
                raise ValueError(
                    "concurrent geometry spool has different ordered content"
                ) from None
            self.manifest = winner.manifest
            self.mode = "reused"
            self.validation_seconds += winner.validation_seconds
        else:
            _fsync_directory(self.root)
            self.manifest = manifest
        self.build = None


def prepare_ordered_geometry_spool(
    rows: Iterable[Mapping[str, object]],
    spool_root: Path,
    identity: OrderedGeometrySpoolIdentity,
) -> OrderedGeometrySpool:
    """Validate saved prefixes, then stream the query through a private spool attempt."""
    started = time.monotonic()
    _make_directory(spool_root)
    identity_root = spool_root / identity.sha256
    _make_directory(identity_root)
    complete_path = identity_root / "complete.json"
    if os.path.lexists(complete_path):
        return _load_complete(identity_root, identity, started=started)

    attempts_path = identity_root / "attempts"
    _make_directory(attempts_path)
    saved_chunks = _find_saved_chunks(identity_root, identity)
    source_rows = iter(rows)
    source_query_seconds = _verify_source_prefix(source_rows, saved_chunks)
    recovered_rows = sum(_int_field(chunk.metadata, "rows") for chunk in saved_chunks)

    attempt_id = uuid.uuid4().hex
    attempt_path = attempts_path / attempt_id
    attempt_path.mkdir()
    chunks_path = attempt_path / "chunks"
    chunks_path.mkdir()
    for saved in saved_chunks:
        _link_saved_chunk(saved, chunks_path)
    build_payload = {
        "attempt_id": attempt_id,
        "identity": identity.payload(),
        "identity_sha256": identity.sha256,
    }
    _atomic_write(attempt_path / "build.json", _pretty_json(build_payload))
    _fsync_directory(attempts_path)
    build = _SpoolBuild(
        source_rows=source_rows,
        attempt_id=attempt_id,
        attempt_path=attempt_path,
        saved_chunks=tuple(saved_chunks),
        recovered_rows=recovered_rows,
        source_query_seconds=source_query_seconds,
    )
    return OrderedGeometrySpool(
        root=identity_root,
        identity=identity,
        mode="recovered" if recovered_rows else "built",
        build=build,
        manifest=None,
        recovered_rows=recovered_rows,
        validation_seconds=time.monotonic() - started,
    )


def _encode_row(row: Mapping[str, object]) -> bytes:
    encoded: dict[str, object] = {}
    for key, value in row.items():
        if not isinstance(key, str):
            raise TypeError("spooled geometry row keys must be strings")
        if key == "geometry" and isinstance(value, (memoryview, bytes, bytearray)):
            encoded[key] = {_BYTE_MARKER: bytes(value).hex()}
        elif key == "geometry" and hasattr(value, "wkb"):
            encoded[key] = {_BYTE_MARKER: bytes(cast(Any, value).wkb).hex()}
        elif value is None or isinstance(value, (str, int, float, bool)):
            encoded[key] = value
        else:
            raise TypeError(f"unsupported geometry spool value for {key!r}")
    return _canonical_json(encoded) + b"\n"


def _decode_row(line: bytes) -> dict[str, object]:
    try:
        value = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("geometry spool row is not valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError("geometry spool row must be a JSON object")
    geometry = value.get("geometry")
    if isinstance(geometry, dict):
        if set(geometry) != {_BYTE_MARKER} or not isinstance(geometry[_BYTE_MARKER], str):
            raise ValueError("geometry spool geometry encoding is unsupported")
        try:
            value["geometry"] = bytes.fromhex(geometry[_BYTE_MARKER])
        except ValueError as error:
            raise ValueError("geometry spool WKB is malformed") from error
    decoded = cast(dict[str, object], value)
    if _encode_row(decoded) != line:
        raise ValueError("geometry spool row is not in canonical encoding")
    return decoded


def _verify_source_prefix(
    rows: Iterator[Mapping[str, object]], saved_chunks: Sequence[_SavedChunk]
) -> float:
    source_query_seconds = 0.0
    for saved in saved_chunks:
        source_query_seconds += _verify_saved_chunk_prefix(rows, saved)
    if saved_chunks and saved_chunks[-1].metadata["final"] is True:
        source_query_seconds += _require_source_end(rows)
    return source_query_seconds


def _verify_saved_chunk_prefix(rows: Iterator[Mapping[str, object]], saved: _SavedChunk) -> float:
    digest = hashlib.sha256()
    source_query_seconds = 0.0
    with saved.data_path.open("rb") as saved_rows:
        for _ in range(_int_field(saved.metadata, "rows")):
            started = time.monotonic()
            try:
                row = next(rows)
            except StopIteration as error:
                source_query_seconds += time.monotonic() - started
                raise ValueError(
                    "pinned geometry query is shorter than the saved spool prefix"
                ) from error
            source_query_seconds += time.monotonic() - started
            saved_line = saved_rows.readline()
            if saved_line != _encode_row(row):
                raise ValueError("pinned geometry query differs from saved spool prefix")
            digest.update(saved_line)
        if saved_rows.read(1):
            raise ValueError("saved spool chunk exceeds its verified row count")
    if digest.hexdigest() != saved.metadata["sha256"]:
        raise ValueError("saved spool chunk checksum changed during recovery")
    return source_query_seconds


def _require_source_end(rows: Iterator[Mapping[str, object]]) -> float:
    started = time.monotonic()
    try:
        next(rows)
    except StopIteration:
        return time.monotonic() - started
    raise ValueError("pinned geometry query extends beyond the saved final spool chunk")


def _read_verified_saved_rows(saved: _SavedChunk) -> list[tuple[bytes, dict[str, object]]]:
    payload = saved.data_path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    lines = payload.splitlines(keepends=True)
    if (
        len(lines) != saved.metadata["rows"]
        or len(payload) != saved.metadata["bytes"]
        or digest != saved.metadata["sha256"]
    ):
        raise ValueError("saved geometry spool chunk checksum mismatch")
    records = []
    for line in lines:
        if not line.endswith(b"\n"):
            raise ValueError("saved geometry spool chunk has a partial line")
        records.append((line, _decode_row(line)))
    return records


def _find_saved_chunks(
    identity_root: Path, identity: OrderedGeometrySpoolIdentity
) -> list[_SavedChunk]:
    attempts_path = identity_root / "attempts"
    if not attempts_path.is_dir() or attempts_path.is_symlink():
        return []
    candidates: list[list[_SavedChunk]] = []
    for attempt_path in sorted(attempts_path.iterdir()):
        chunks = _read_attempt_chunks(attempt_path, identity)
        if chunks:
            candidates.append(chunks)
    if not candidates:
        return []
    return max(
        candidates,
        key=lambda chunks: (
            sum(_int_field(chunk.metadata, "rows") for chunk in chunks),
            len(chunks),
        ),
    )


def _read_attempt_chunks(
    attempt_path: Path, identity: OrderedGeometrySpoolIdentity
) -> list[_SavedChunk]:
    if _ATTEMPT_PATTERN.fullmatch(attempt_path.name) is None:
        if attempt_path.is_symlink() or _has_attempt_state_in_renamed_attempt(attempt_path):
            raise ValueError("geometry spool attempt entry has an invalid name")
        return []
    if attempt_path.is_symlink() or not attempt_path.is_dir():
        raise ValueError("geometry spool attempt entry is not a regular directory")
    chunks_path = attempt_path / "chunks"
    if not os.path.lexists(chunks_path):
        if os.path.lexists(attempt_path / "build.json"):
            raise ValueError("interrupted geometry spool chunk directory is missing")
        return []
    if chunks_path.is_symlink() or not chunks_path.is_dir():
        raise ValueError("interrupted geometry spool chunk directory is unsafe")
    committed_indices = _chunk_metadata_indices(chunks_path)
    if not _attempt_build_matches(
        attempt_path,
        identity,
        committed_indices=committed_indices,
    ):
        return []
    return _read_contiguous_chunks(
        chunks_path,
        identity.chunk_rows,
        committed_indices=committed_indices,
    )


def _has_attempt_state_in_renamed_attempt(attempt_path: Path) -> bool:
    if not attempt_path.is_dir():
        return False
    if os.path.lexists(attempt_path / "build.json"):
        return True
    for child in attempt_path.iterdir():
        if child.is_symlink():
            return True
        if child.is_dir() and _chunk_metadata_indices(child):
            return True
    return False


def _attempt_build_matches(
    attempt_path: Path,
    identity: OrderedGeometrySpoolIdentity,
    *,
    committed_indices: set[int],
) -> bool:
    build_path = attempt_path / "build.json"
    if not _regular_file(build_path):
        if committed_indices:
            raise ValueError("committed geometry spool chunks lack valid build identity")
        return False
    try:
        build = _read_json(build_path)
    except (OSError, TypeError, ValueError) as error:
        if committed_indices:
            raise ValueError(
                "committed geometry spool chunks have corrupt build identity"
            ) from error
        return False
    if (
        build.get("attempt_id") != attempt_path.name
        or build.get("identity") != identity.payload()
        or build.get("identity_sha256") != identity.sha256
    ):
        if committed_indices:
            raise ValueError("committed geometry spool chunks have mismatched build identity")
        return False
    return True


def _chunk_metadata_indices(chunks_path: Path) -> set[int]:
    indices = set()
    for path in chunks_path.iterdir():
        match = _CHUNK_METADATA_PATTERN.fullmatch(path.name)
        if match is not None and os.path.lexists(path):
            indices.add(int(match.group(1)))
    return indices


def _read_contiguous_chunks(
    chunks_path: Path,
    chunk_rows: int,
    *,
    committed_indices: set[int],
) -> list[_SavedChunk]:
    chunks: list[_SavedChunk] = []
    last_committed_index = max(committed_indices, default=-1)
    for index in itertools.count():
        name = _chunk_name(index)
        data_path = chunks_path / name
        metadata_path = chunks_path / _chunk_metadata_name(index)
        if not os.path.lexists(metadata_path):
            if last_committed_index > index:
                raise ValueError("saved geometry spool has a gap before committed chunks")
            break
        if not _regular_file(data_path) or not _regular_file(metadata_path):
            raise ValueError("committed geometry spool chunk files are missing or unsafe")
        metadata = _read_json(metadata_path)
        _validate_chunk_file(data_path, metadata, expected_index=index)
        rows = _int_field(metadata, "rows")
        final = metadata["final"] is True
        if rows < chunk_rows and not final:
            raise ValueError("short saved geometry spool chunk is not marked final")
        chunk = _SavedChunk(index, data_path, metadata_path, metadata)
        chunks.append(chunk)
        if final:
            if last_committed_index > index:
                raise ValueError("saved geometry spool has committed chunks after its final chunk")
            break
    return chunks


def _verify_saved_prefix(saved: _SavedChunk, lines: Sequence[bytes], *, is_final: bool) -> None:
    metadata = saved.metadata
    if metadata["final"] is True and not is_final:
        raise ValueError("pinned geometry query extends beyond the saved final spool chunk")
    if _int_field(metadata, "rows") != len(lines):
        raise ValueError("pinned geometry query differs from saved spool chunk length")
    digest = hashlib.sha256()
    with saved.data_path.open("rb") as source:
        for line in lines:
            saved_line = source.readline()
            if saved_line != line:
                raise ValueError("pinned geometry query differs from saved spool prefix")
            digest.update(saved_line)
        if source.read(1):
            raise ValueError("saved spool chunk exceeds its verified row count")
    if digest.hexdigest() != metadata["sha256"]:
        raise ValueError("saved spool chunk checksum changed during recovery")


def _link_saved_chunk(saved: _SavedChunk, destination: Path) -> None:
    data_target = destination / saved.data_path.name
    metadata_target = destination / saved.metadata_path.name
    os.link(saved.data_path, data_target)
    _fsync_directory(destination)
    os.link(saved.metadata_path, metadata_target)
    _fsync_directory(destination)


def _publish_chunk(
    chunks_path: Path,
    index: int,
    lines: Sequence[bytes],
    *,
    final: bool,
) -> tuple[dict[str, object], dict[str, object]]:
    payload = b"".join(lines)
    digest = hashlib.sha256(payload).hexdigest()
    metadata: dict[str, object] = {
        "index": index,
        "rows": len(lines),
        "bytes": len(payload),
        "sha256": digest,
        "final": final,
    }
    data_path = chunks_path / _chunk_name(index)
    metadata_path = chunks_path / _chunk_metadata_name(index)
    _atomic_write(data_path, payload)
    _atomic_write(metadata_path, _pretty_json(metadata))
    return _manifest_chunk(metadata, is_final=final), metadata


def _manifest_chunk(metadata: Mapping[str, object], *, is_final: bool) -> dict[str, object]:
    return {
        "name": _chunk_name(_int_field(metadata, "index")),
        "rows": _int_field(metadata, "rows"),
        "bytes": _int_field(metadata, "bytes"),
        "sha256": str(metadata["sha256"]),
        "is_final": is_final,
    }


def _load_complete(
    identity_root: Path,
    identity: OrderedGeometrySpoolIdentity,
    *,
    started: float,
) -> OrderedGeometrySpool:
    complete_path = identity_root / "complete.json"
    if not _regular_file(complete_path):
        raise ValueError("completed geometry spool manifest is not a regular file")
    manifest = _read_json(complete_path)
    if (
        manifest.get("identity") != identity.payload()
        or manifest.get("identity_sha256") != identity.sha256
        or manifest.get("schema_version") != SPOOL_SCHEMA_VERSION
    ):
        raise ValueError("completed geometry spool identity does not match pinned inputs")
    attempt_id = manifest.get("attempt_id")
    if not isinstance(attempt_id, str) or _ATTEMPT_PATTERN.fullmatch(attempt_id) is None:
        raise ValueError("completed geometry spool attempt identifier is malformed")
    attempt_path = identity_root / "attempts" / attempt_id
    if (
        attempt_path.is_symlink()
        or not attempt_path.is_dir()
        or (identity_root / "attempts").is_symlink()
    ):
        raise ValueError("completed geometry spool attempt directory is invalid")
    manifest_path = attempt_path / "manifest.json"
    if not _regular_file(manifest_path) or not complete_path.samefile(manifest_path):
        raise ValueError("completed geometry spool publication pointer is invalid")
    build = _read_json(attempt_path / "build.json")
    if build.get("identity") != identity.payload() or build.get("attempt_id") != attempt_id:
        raise ValueError("completed geometry spool build identity is invalid")
    _validate_manifest_chunks(identity_root, manifest)
    return OrderedGeometrySpool(
        root=identity_root,
        identity=identity,
        mode="reused",
        build=None,
        manifest=manifest,
        recovered_rows=0,
        validation_seconds=time.monotonic() - started,
    )


def _validate_manifest_chunks(identity_root: Path, manifest: Mapping[str, object]) -> None:
    attempt_id = str(manifest["attempt_id"])
    chunks_path = identity_root / "attempts" / attempt_id / "chunks"
    if chunks_path.is_symlink() or not chunks_path.is_dir():
        raise ValueError("completed geometry spool chunk directory is missing")
    chunks = manifest.get("chunks")
    if not isinstance(chunks, list):
        raise TypeError("completed geometry spool chunk list is malformed")
    chunk_rows = _int_field(cast(dict[str, object], manifest["identity"]), "chunk_rows")
    _validate_manifest_layout(chunks, chunk_rows)
    total_rows = 0
    total_bytes = 0
    digest = hashlib.sha256()
    for index, chunk in enumerate(cast(list[dict[str, object]], chunks)):
        rows, byte_count, _ = _validate_manifest_chunk(
            chunks_path,
            chunk,
            index=index,
            chunk_rows=chunk_rows,
        )
        data_path = chunks_path / _chunk_name(index)
        _update_digest_from_file(digest, data_path)
        total_rows += rows
        total_bytes += byte_count
    if (
        total_rows != manifest.get("row_count")
        or total_bytes != manifest.get("byte_count")
        or digest.hexdigest() != manifest.get("content_sha256")
    ):
        raise ValueError("completed geometry spool stream checksum mismatch")


def _validate_manifest_layout(chunks: Sequence[object], chunk_rows: int) -> None:
    if chunks and cast(dict[str, object], chunks[-1]).get("is_final") is not True:
        raise ValueError("completed geometry spool final chunk is not marked complete")
    if any(cast(dict[str, object], chunk).get("is_final") is True for chunk in chunks[:-1]):
        raise ValueError("completed geometry spool has an early final chunk")
    for index, raw_chunk in enumerate(chunks):
        if not isinstance(raw_chunk, dict):
            raise TypeError("completed geometry spool chunk entry is malformed")
        chunk = cast(dict[str, object], raw_chunk)
        if chunk.get("name") != _chunk_name(index):
            raise ValueError("completed geometry spool chunk order is malformed")
        rows = chunk.get("rows")
        if type(rows) is not int or rows < 1 or rows > chunk_rows:
            raise ValueError("completed geometry spool chunk row count is invalid")
        if rows < chunk_rows and chunk.get("is_final") is not True:
            raise ValueError("short geometry spool chunk is not the final chunk")


def _validate_manifest_chunk(
    chunks_path: Path,
    chunk: dict[str, object],
    *,
    index: int,
    chunk_rows: int,
) -> tuple[int, int, str]:
    data_path = chunks_path / _chunk_name(index)
    metadata_path = chunks_path / _chunk_metadata_name(index)
    if not _regular_file(data_path) or not _regular_file(metadata_path):
        raise ValueError("completed geometry spool chunk file is missing")
    metadata = _read_json(metadata_path)
    _validate_manifest_chunk_pair(chunk, metadata, expected_index=index, chunk_rows=chunk_rows)
    rows, byte_count, chunk_digest = _hash_jsonl_chunk(data_path)
    if (
        rows != chunk.get("rows")
        or byte_count != chunk.get("bytes")
        or chunk_digest != chunk.get("sha256")
    ):
        raise ValueError("completed geometry spool chunk checksum mismatch")
    return rows, byte_count, chunk_digest


def _validate_manifest_chunk_pair(
    chunk: Mapping[str, object],
    metadata: Mapping[str, object],
    *,
    expected_index: int,
    chunk_rows: int,
) -> None:
    if (
        metadata.get("index") != expected_index
        or metadata.get("rows") != chunk.get("rows")
        or metadata.get("bytes") != chunk.get("bytes")
        or metadata.get("sha256") != chunk.get("sha256")
        or (metadata.get("final") is True and chunk.get("is_final") is not True)
    ):
        raise ValueError("completed geometry spool chunk metadata disagrees with its manifest")
    rows = chunk.get("rows")
    if type(rows) is not int or rows < 1 or rows > chunk_rows:
        raise ValueError("completed geometry spool chunk row count is invalid")
    if rows < chunk_rows and chunk.get("is_final") is not True:
        raise ValueError("short geometry spool chunk is not the final chunk")


def _iter_manifest_rows(
    identity_root: Path, manifest: Mapping[str, object]
) -> Iterator[dict[str, object]]:
    attempt_id = str(manifest["attempt_id"])
    chunks_path = identity_root / "attempts" / attempt_id / "chunks"
    digest = hashlib.sha256()
    row_count = 0
    byte_count = 0
    for index, raw_chunk in enumerate(cast(list[object], manifest["chunks"])):
        chunk = cast(dict[str, object], raw_chunk)
        data_path = chunks_path / _chunk_name(index)
        chunk_digest = hashlib.sha256()
        chunk_rows = 0
        chunk_bytes = 0
        with data_path.open("rb") as source:
            while line := source.readline():
                if not line.endswith(b"\n"):
                    raise ValueError("geometry spool contains an incomplete row")
                decoded = _decode_row(line)
                digest.update(line)
                chunk_digest.update(line)
                row_count += 1
                chunk_rows += 1
                byte_count += len(line)
                chunk_bytes += len(line)
                yield decoded
        if (
            chunk_rows != chunk.get("rows")
            or chunk_bytes != chunk.get("bytes")
            or chunk_digest.hexdigest() != chunk.get("sha256")
        ):
            raise ValueError("geometry spool chunk changed while being read")
    if (
        row_count != manifest.get("row_count")
        or byte_count != manifest.get("byte_count")
        or digest.hexdigest() != manifest.get("content_sha256")
    ):
        raise ValueError("geometry spool content changed while being read")


def _validate_chunk_file(
    path: Path, metadata: Mapping[str, object], *, expected_index: int
) -> None:
    if metadata.get("index") != expected_index:
        raise ValueError("saved geometry spool chunk index is malformed")
    rows = metadata.get("rows")
    byte_count = metadata.get("bytes")
    digest = metadata.get("sha256")
    final = metadata.get("final")
    if (
        not isinstance(rows, int)
        or rows < 1
        or not isinstance(byte_count, int)
        or byte_count < 1
        or not isinstance(digest, str)
        or _HASH_PATTERN.fullmatch(digest) is None
        or not isinstance(final, bool)
    ):
        raise ValueError("saved geometry spool chunk metadata is malformed")
    actual_rows, actual_bytes, actual_digest = _hash_jsonl_chunk(path)
    if (actual_rows, actual_bytes, actual_digest) != (rows, byte_count, digest):
        raise ValueError("saved geometry spool chunk checksum mismatch")


def _hash_jsonl_chunk(path: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    rows = 0
    byte_count = 0
    with path.open("rb") as source:
        while line := source.readline():
            if not line.endswith(b"\n"):
                raise ValueError("saved geometry spool chunk has a partial line")
            _decode_row(line)
            digest.update(line)
            rows += 1
            byte_count += len(line)
    return rows, byte_count, digest.hexdigest()


def _update_digest_from_file(digest: Any, path: Path) -> None:
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)


def _chunk_name(index: int) -> str:
    return f"chunk-{index:08d}.jsonl"


def _chunk_metadata_name(index: int) -> str:
    return f"chunk-{index:08d}.json"


def _regular_file(path: Path) -> bool:
    return not path.is_symlink() and path.is_file()


def _make_directory(path: Path) -> None:
    if path.exists():
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"geometry spool path is not a regular directory: {path}")
        return
    if not path.parent.exists():
        _make_directory(path.parent)
    try:
        path.mkdir()
    except FileExistsError:
        if path.is_symlink() or not path.is_dir():
            raise ValueError(f"geometry spool path is not a regular directory: {path}") from None
    else:
        _fsync_directory(path.parent)


def _read_json(path: Path) -> dict[str, object]:
    if not _regular_file(path):
        raise ValueError(f"geometry spool metadata is not a regular file: {path}")
    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_reject_duplicate_pairs)
    if not isinstance(value, dict):
        raise TypeError("geometry spool metadata must be an object")
    return cast(dict[str, object], value)


def _int_field(value: Mapping[str, object], key: str) -> int:
    field_value = value.get(key)
    if type(field_value) is not int:
        raise ValueError(f"geometry spool field {key!r} is not an integer")
    return field_value


def _float_field(value: Mapping[str, object], key: str, *, default: float | None = None) -> float:
    field_value = value.get(key, default)
    if isinstance(field_value, bool) or not isinstance(field_value, (int, float)):
        raise TypeError(f"geometry spool field {key!r} is not numeric")
    return float(field_value)


def _reject_duplicate_pairs(pairs: Sequence[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON object key: {key}")
        value[key] = item
    return value


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _pretty_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("xb") as output:
        output.write(payload)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)
    _fsync_directory(path.parent)


def _fsync_directory(path: Path) -> None:
    directory_fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
