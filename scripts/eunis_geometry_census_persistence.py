"""Atomic region checkpoints and bounded unique-candidate census index."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import BinaryIO, cast

from .eunis_geometry_census_core import Assignment, CachedAssignment, CensusCounts
from .eunis_geometry_census_sources import (
    H3_RESOLUTION,
    MAX_PER_CELL,
    MINIMUM_H3_CELL_COUNT,
    TARGET,
    _geometry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite census artifact {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _region_signature(
    run_signature: str, stream: str, label_path: str, raw_paths: Sequence[str]
) -> str:
    payload = json.dumps(
        {
            "run": run_signature,
            "stream": stream,
            "label_path": label_path,
            "raw_paths": list(raw_paths),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


_ASSIGNMENT_CACHE_SCHEMA = 1
_CLASSIFICATION_COUNT_FIELDS = (
    "invalid_geometry_rows",
    "outside_native_extent_rows",
    "projection_invalid_rows",
    "native_overlap_errors",
    "ambiguous_native_rows",
    "unassigned_native_rows",
)


def _canonical_json(payload: object) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _assignment_identity(row: Mapping[str, object]) -> tuple[str, str]:
    fields = tuple(
        None if row.get(name) is None else str(row.get(name))
        for name in ("source_pbf", "osm_type", "osm_id", "polygon_id")
    )
    fallback = None
    if not any(value for value in fields):
        fallback_value = row.get("_order") or row.get("sentence_key")
        if fallback_value is None or not str(fallback_value):
            raise ValueError("geometry row lacks a stable polygon identity")
        fallback = str(fallback_value)
    encoded = _canonical_json({"fields": fields, "fallback": fallback})
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _assignment_geometry_digest(geometry: object) -> str:
    if geometry is None:
        tag, payload = b"none", b""
    elif isinstance(geometry, memoryview):
        tag, payload = b"bytes", geometry.tobytes()
    elif isinstance(geometry, (bytes, bytearray)):
        tag, payload = b"bytes", bytes(geometry)
    elif isinstance(geometry, str):
        tag, payload = b"text", geometry.encode("utf-8")
    elif isinstance(geometry, Mapping):
        tag = b"mapping"
        payload = json.dumps(dict(geometry), sort_keys=True, separators=(",", ":")).encode("utf-8")
    elif hasattr(geometry, "wkb"):
        wkb_attribute = "wkb"
        wkb = getattr(geometry, wkb_attribute)
        if not isinstance(wkb, (bytes, bytearray, memoryview)):
            raise TypeError("geometry WKB attribute is not bytes-like")
        tag, payload = b"wkb", bytes(wkb)
    else:
        raise TypeError(f"unsupported geometry cache input: {type(geometry).__name__}")
    return hashlib.sha256(tag + b"\0" + payload).hexdigest()


def _create_assignment_cache_file(path: Path, region_signature: str) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.init-", dir=path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    db = sqlite3.connect(temporary)
    try:
        db.execute("PRAGMA synchronous=FULL")
        with db:
            db.execute(
                "CREATE TABLE cache_metadata ("
                "id INTEGER PRIMARY KEY CHECK (id = 1), "
                "schema_version INTEGER NOT NULL, region_signature TEXT NOT NULL)"
            )
            db.execute(
                "INSERT INTO cache_metadata VALUES (1, ?, ?)",
                (_ASSIGNMENT_CACHE_SCHEMA, region_signature),
            )
            db.execute(
                "CREATE TABLE assignments ("
                "identity_sha256 TEXT PRIMARY KEY, identity_json TEXT NOT NULL, "
                "geometry_sha256 TEXT NOT NULL, payload_json TEXT NOT NULL, "
                "record_sha256 TEXT NOT NULL)"
            )
            db.execute(f"PRAGMA user_version={_ASSIGNMENT_CACHE_SCHEMA}")
        db.close()
        try:
            os.link(temporary, path)
        except FileExistsError as error:
            raise ValueError("assignment cache appeared during initialization") from error
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        temporary.unlink()
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        db.close()
        raise


class AssignmentCache:
    """Persist verified per-polygon native assignments across region retries."""

    def __init__(self, path: Path, region_signature: str, *, commit_interval: int = 256) -> None:
        """Open a region-bound cache and reject existing files with unknown state."""
        if commit_interval < 1:
            raise ValueError("assignment cache commit interval must be positive")
        self.path = path
        self.region_signature = region_signature
        self.commit_interval = commit_interval
        self.hits = 0
        self.misses = 0
        self._pending = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink():
            raise ValueError(f"assignment cache is not a regular file: {path}")
        if path.exists() and not path.is_file():
            raise ValueError(f"assignment cache is not a regular file: {path}")
        if not path.exists():
            _create_assignment_cache_file(path, region_signature)
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.execute("PRAGMA busy_timeout=5000")
            self._validate_database()
        except sqlite3.Error as error:
            self.db.close()
            raise ValueError(f"assignment cache is corrupt or unsupported: {path}") from error
        except BaseException:
            self.db.close()
            raise

    def _validate_database(self) -> None:
        version = self.db.execute("PRAGMA user_version").fetchone()
        metadata = self.db.execute(
            "SELECT schema_version, region_signature FROM cache_metadata WHERE id=1"
        ).fetchone()
        if version != (_ASSIGNMENT_CACHE_SCHEMA,) or metadata != (
            _ASSIGNMENT_CACHE_SCHEMA,
            self.region_signature,
        ):
            raise ValueError("assignment cache schema or region signature mismatch")
        self.db.execute("SELECT 1 FROM assignments LIMIT 1").fetchone()

    def classify(
        self,
        row: Mapping[str, object],
        *,
        geometry: object,
        classify_geometry: Callable[[object], Assignment | None],
        counts: CensusCounts,
    ) -> Assignment | None:
        """Reuse a checked assignment or classify and durably record this polygon."""
        identity_json, identity_sha256 = _assignment_identity(row)
        geometry_sha256 = _assignment_geometry_digest(geometry)
        try:
            saved = self.db.execute(
                "SELECT identity_json, geometry_sha256, payload_json, record_sha256 "
                "FROM assignments WHERE identity_sha256=?",
                (identity_sha256,),
            ).fetchone()
        except sqlite3.Error as error:
            raise ValueError(f"assignment cache read failed: {self.path}") from error
        if saved is not None:
            saved_identity, saved_geometry, payload_json, record_sha256 = saved
            if saved_identity != identity_json or saved_geometry != geometry_sha256:
                raise ValueError("assignment cache polygon identity or geometry changed")
            payload = self._decode_record(
                identity_sha256,
                identity_json,
                geometry_sha256,
                payload_json,
                record_sha256,
            )
            count_deltas = cast(dict[str, int], payload["count_deltas"])
            for name, value in count_deltas.items():
                setattr(counts, name, getattr(counts, name) + value)
            self.hits += 1
            return self._decode_assignment(payload["assignment"])

        before = {name: getattr(counts, name) for name in _CLASSIFICATION_COUNT_FIELDS}
        assignment = classify_geometry(geometry)
        deltas = {
            name: getattr(counts, name) - before[name] for name in _CLASSIFICATION_COUNT_FIELDS
        }
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in deltas.values()
        ):
            raise ValueError("classification counter deltas must be non-negative integers")
        encoded_assignment = self._encode_assignment(assignment)
        self._decode_assignment(encoded_assignment)
        payload = {
            "schema_version": _ASSIGNMENT_CACHE_SCHEMA,
            "region_signature": self.region_signature,
            "identity_sha256": identity_sha256,
            "identity_json": identity_json,
            "geometry_sha256": geometry_sha256,
            "assignment": encoded_assignment,
            "count_deltas": deltas,
        }
        payload_json = _canonical_json(payload)
        record_sha256 = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()
        try:
            self.db.execute(
                "INSERT INTO assignments VALUES (?, ?, ?, ?, ?)",
                (
                    identity_sha256,
                    identity_json,
                    geometry_sha256,
                    payload_json,
                    record_sha256,
                ),
            )
        except sqlite3.Error as error:
            raise ValueError(f"assignment cache write failed: {self.path}") from error
        self._pending += 1
        self.misses += 1
        if self._pending >= self.commit_interval:
            self.db.commit()
            self._pending = 0
        return assignment

    def _decode_record(
        self,
        identity_sha256: str,
        identity_json: str,
        geometry_sha256: str,
        payload_json: str,
        record_sha256: str,
    ) -> dict[str, object]:
        if hashlib.sha256(payload_json.encode("utf-8")).hexdigest() != record_sha256:
            raise ValueError("assignment cache record checksum mismatch")
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError as error:
            raise ValueError("assignment cache record is malformed") from error
        if not isinstance(payload, dict) or _canonical_json(payload) != payload_json:
            raise ValueError("assignment cache record is not canonical JSON")
        if any(
            payload.get(name) != value
            for name, value in (
                ("schema_version", _ASSIGNMENT_CACHE_SCHEMA),
                ("region_signature", self.region_signature),
                ("identity_sha256", identity_sha256),
                ("identity_json", identity_json),
                ("geometry_sha256", geometry_sha256),
            )
        ):
            raise ValueError("assignment cache record metadata mismatch")
        deltas = payload.get("count_deltas")
        if (
            not isinstance(deltas, dict)
            or set(deltas) != set(_CLASSIFICATION_COUNT_FIELDS)
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in deltas.values()
            )
        ):
            raise ValueError("assignment cache count deltas are malformed")
        self._decode_assignment(payload.get("assignment"))
        return payload

    @staticmethod
    def _encode_assignment(assignment: Assignment | None) -> dict[str, object] | None:
        if assignment is None:
            return None
        return {
            "code": assignment.code,
            "name": assignment.name,
            "overlap_percentage": assignment.overlap_percentage,
            "source_version": assignment.source_version,
        }

    @staticmethod
    def _decode_assignment(payload: object) -> CachedAssignment | None:
        if payload is None:
            return None
        if not isinstance(payload, dict) or set(payload) != {
            "code",
            "name",
            "overlap_percentage",
            "source_version",
        }:
            raise ValueError("assignment cache result is malformed")
        for name in ("code", "name", "source_version"):
            if payload[name] is not None and not isinstance(payload[name], str):
                raise ValueError("assignment cache result fields must be strings or null")
        overlap = payload["overlap_percentage"]
        if overlap is not None and (
            isinstance(overlap, bool)
            or not isinstance(overlap, (float, int))
            or not math.isfinite(overlap)
        ):
            raise ValueError("assignment cache overlap percentage is malformed")
        return CachedAssignment(
            code=cast(str | None, payload["code"]),
            name=cast(str | None, payload["name"]),
            overlap_percentage=cast(float | None, overlap),
            source_version=cast(str | None, payload["source_version"]),
        )

    def __enter__(self) -> AssignmentCache:
        """Return this cache for a bounded context-managed region pass."""
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Commit a successful pass and roll back its uncommitted tail on error."""
        try:
            if exc_type is None:
                self.db.commit()
                self._pending = 0
            else:
                self.db.rollback()
        finally:
            self.db.close()


def _load_completed_region(
    checkpoint: Path, output: Path, signature: str
) -> dict[str, object] | None:
    if not checkpoint.exists():
        return None
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise ValueError(f"region checkpoint input mismatch: {checkpoint}")
    if not output.is_file() or _sha256_file(output) != payload.get("output_sha256"):
        raise ValueError(f"region output digest mismatch: {output}")
    payload["output_path"] = str(output)
    return payload


def _replay_stage(
    output: Path, partial: Path, replay_stage: Path
) -> tuple[Path | None, BinaryIO | None]:
    if output.exists():
        return None, None
    if replay_stage.exists():
        return replay_stage, replay_stage.open("ab")
    if partial.exists():
        return replay_stage, replay_stage.open("xb")
    return partial, partial.open("xb")


def _validate_replay_artifacts(paths: Sequence[Path]) -> None:
    for path in paths:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"region replay artifact is not a regular file: {path}")


def _compare_replay_record(
    readers: Sequence[tuple[Path, BinaryIO]],
    remaining_bytes: dict[Path, int],
    replay_stage: Path | None,
    record: bytes,
) -> int:
    if not isinstance(record, bytes):
        raise TypeError("replayed region records must be bytes")
    replayed_stage_bytes = 0
    for path, reader in readers:
        length = min(len(record), remaining_bytes[path])
        saved = reader.read(length)
        if len(saved) != length or saved != record[:length]:
            raise ValueError(f"region replay differs from saved prefix: {path}")
        remaining_bytes[path] -= length
        if path == replay_stage:
            replayed_stage_bytes = length
    return replayed_stage_bytes


def _verify_replay_end(
    output: Path,
    remaining_bytes: Mapping[Path, int],
    byte_count: int,
) -> None:
    if any(remaining_bytes.values()):
        raise ValueError("saved region artifact exceeds replayed output")
    if output.exists() and output.stat().st_size != byte_count:
        raise ValueError(f"uncheckpointed region output is incomplete: {output}")


def _promote_replay_stage(output: Path, stage: Path | None) -> None:
    if stage is None or output.exists():
        return
    stage.replace(output)
    directory_fd = os.open(output.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_replayed_region_output(output: Path, records: Iterable[bytes]) -> tuple[str, int]:
    """Replay a region into a durable output after verifying any saved byte prefix.

    Orphan partials and outputs without checkpoints are never trusted on their own.
    Every existing artifact must be an exact byte prefix of the newly generated
    stream. A partial is left untouched until replay has matched it completely.
    """
    partial = output.with_name(output.name + ".partial")
    replay_stage_path = output.with_name(output.name + ".partial.replay")
    existing_paths = [path for path in (output, partial, replay_stage_path) if path.exists()]
    _validate_replay_artifacts(existing_paths)
    stage, target = _replay_stage(output, partial, replay_stage_path)
    readers: list[tuple[Path, BinaryIO]] = []
    remaining_bytes: dict[Path, int] = {}
    try:
        for path in existing_paths:
            readers.append((path, path.open("rb")))
            remaining_bytes[path] = path.stat().st_size
        digest = hashlib.sha256()
        row_count = 0
        byte_count = 0
        if target is None:
            for record in records:
                _compare_replay_record(readers, remaining_bytes, None, record)
                digest.update(record)
                row_count += 1
                byte_count += len(record)
        else:
            saved_stage = replay_stage_path if replay_stage_path in remaining_bytes else None
            with target:
                for record in records:
                    replayed_stage_bytes = _compare_replay_record(
                        readers, remaining_bytes, saved_stage, record
                    )
                    target.write(record[replayed_stage_bytes:])
                    digest.update(record)
                    row_count += 1
                    byte_count += len(record)
                    if row_count % 256 == 0:
                        target.flush()
                        os.fsync(target.fileno())
                target.flush()
                os.fsync(target.fileno())
        _verify_replay_end(output, remaining_bytes, byte_count)
        _promote_replay_stage(output, stage)
        return digest.hexdigest(), row_count
    finally:
        if target is not None and not target.closed:
            target.close()
        for _, reader in readers:
            reader.close()


def _open_index(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS eligible ("
        "source TEXT NOT NULL, decision TEXT NOT NULL, polygon_key TEXT NOT NULL, "
        "geometry_sha256 TEXT NOT NULL, h3_cell TEXT NOT NULL, "
        "PRIMARY KEY (source, decision, polygon_key))"
    )
    db.execute(
        "CREATE TABLE IF NOT EXISTS indexed_regions ("
        "source TEXT NOT NULL, stream TEXT NOT NULL, shard TEXT NOT NULL, "
        "run_signature TEXT NOT NULL, output_sha256 TEXT NOT NULL, "
        "PRIMARY KEY (source, stream, shard))"
    )
    db.commit()
    return db


def _candidate_polygon_key(row: Mapping[str, object]) -> str:
    key_fields = tuple(
        str(row.get(key) or "") for key in ("source_pbf", "osm_type", "osm_id", "polygon_id")
    )
    return json.dumps(key_fields, separators=(",", ":"))


def _candidate_index_values(row: Mapping[str, object]) -> tuple[str, str, str]:
    h3 = import_module("h3")
    geometry = _geometry(row["geometry"])
    point = geometry.representative_point()
    cell = h3.latlng_to_cell(point.y, point.x, H3_RESOLUTION)
    polygon_key = _candidate_polygon_key(row)
    geometry_digest = hashlib.sha256(geometry.wkb).hexdigest()
    return polygon_key, geometry_digest, cell


def _index_candidate(db: sqlite3.Connection, source: str, row: Mapping[str, object]) -> None:
    polygon_key, geometry_digest, cell = _candidate_index_values(row)
    decision = str(row["decision"])
    existing = db.execute(
        "SELECT geometry_sha256, h3_cell FROM eligible "
        "WHERE source=? AND decision=? AND polygon_key=?",
        (source, decision, polygon_key),
    ).fetchone()
    if existing is not None:
        if existing != (geometry_digest, cell):
            raise ValueError(
                f"duplicate polygon has conflicting geometry: {source}/{decision}/{polygon_key}"
            )
        return
    db.execute(
        "INSERT INTO eligible VALUES (?, ?, ?, ?, ?)",
        (source, decision, polygon_key, geometry_digest, cell),
    )


def _index_region(
    db: sqlite3.Connection,
    source: str,
    output: Path,
    *,
    marker: tuple[str, str, str, str, str] | None = None,
) -> None:
    with db, output.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                _index_candidate(db, source, json.loads(line))
        if marker is not None:
            db.execute("INSERT INTO indexed_regions VALUES (?, ?, ?, ?, ?)", marker)


def _indexed_entries(db: sqlite3.Connection) -> dict[tuple[str, str, str], tuple[str, str]]:
    return {
        (source, decision, polygon_key): (geometry_sha256, h3_cell)
        for source, decision, polygon_key, geometry_sha256, h3_cell in db.execute(
            "SELECT source, decision, polygon_key, geometry_sha256, h3_cell FROM eligible"
        )
    }


def _expected_region_index(
    artifacts: Sequence[tuple[str, str, str, Path, str]],
) -> tuple[
    dict[tuple[str, str, str], tuple[str, str]],
    dict[tuple[str, str, str], set[tuple[str, str, str]]],
]:
    expected: dict[tuple[str, str, str], tuple[str, str]] = {}
    region_keys: dict[tuple[str, str, str], set[tuple[str, str, str]]] = {}
    geometry_cache: dict[tuple[str, str, str], tuple[str, tuple[str, str, str]]]
    geometry_cache = {}
    for source, stream, shard, output, output_sha256 in artifacts:
        if _sha256_file(output) != output_sha256:
            raise ValueError(
                f"checkpoint output digest changed during index verification: {output}"
            )
        region_id = (source, stream, shard)
        keys = region_keys.setdefault(region_id, set())
        with output.open(encoding="utf-8") as source_file:
            for line in source_file:
                if not line.strip():
                    continue
                row = json.loads(line)
                decision = str(row["decision"])
                polygon_key = _candidate_polygon_key(row)
                key = (source, decision, polygon_key)
                geometry_wire = json.dumps(row["geometry"], sort_keys=True, separators=(",", ":"))
                cached = geometry_cache.get(key)
                if cached is not None and cached[0] == geometry_wire:
                    values = cached[1]
                else:
                    values = _candidate_index_values(row)
                    if cached is not None and cached[1] != values:
                        raise ValueError(f"checkpoint outputs conflict for indexed polygon: {key}")
                    geometry_cache[key] = (geometry_wire, values)
                if values[0] != polygon_key:
                    raise ValueError(
                        f"candidate polygon key changed during index verification: {key}"
                    )
                value = (values[1], values[2])
                prior = expected.get(key)
                if prior is not None and prior != value:
                    raise ValueError(f"checkpoint outputs conflict for indexed polygon: {key}")
                expected[key] = value
                keys.add(key)
    return expected, region_keys


def _verify_and_backfill_indexed_regions(  # noqa: C901
    db: sqlite3.Connection,
    artifacts: Sequence[tuple[str, str, str, Path, str]],
    run_signature: str,
) -> None:
    """Validate persisted index coverage once, then mark checkpointed outputs."""
    expected_artifacts = {
        (source, stream, shard): (output, output_sha256)
        for source, stream, shard, output, output_sha256 in artifacts
    }
    existing_markers = {
        (source, stream, shard): (signature, output_sha256)
        for source, stream, shard, signature, output_sha256 in db.execute(
            "SELECT source, stream, shard, run_signature, output_sha256 FROM indexed_regions"
        )
    }
    for region_id, marker in existing_markers.items():
        expected = expected_artifacts.get(region_id)
        if expected is None or marker != (run_signature, expected[1]):
            raise ValueError(
                f"indexed-region marker does not match current checkpoint: {region_id}"
            )
    if len(existing_markers) == len(expected_artifacts) and (
        expected_artifacts or not _indexed_entries(db)
    ):
        return

    expected_entries, region_keys = _expected_region_index(artifacts)
    actual_entries = _indexed_entries(db)
    for key, value in actual_entries.items():
        if key not in expected_entries or expected_entries[key] != value:
            raise ValueError(f"persisted eligible index does not match checkpoint outputs: {key}")

    for source, stream, shard, output, _output_sha256 in artifacts:
        region_id = (source, stream, shard)
        missing = any(key not in actual_entries for key in region_keys[region_id])
        if missing:
            if region_id in existing_markers:
                raise ValueError(f"indexed-region marker has missing index rows: {region_id}")
            _index_region(db, source, output)

    actual_entries = _indexed_entries(db)
    if actual_entries != expected_entries:
        raise ValueError("persisted eligible index is incomplete for checkpoint outputs")
    with db:
        for source, stream, shard, _, output_sha256 in artifacts:
            region_id = (source, stream, shard)
            if region_id in existing_markers:
                continue
            db.execute(
                "INSERT INTO indexed_regions VALUES (?, ?, ?, ?, ?)",
                (source, stream, shard, run_signature, output_sha256),
            )


def _index_region_if_needed(
    db: sqlite3.Connection,
    source: str,
    stream: str,
    shard: str,
    output: Path,
    run_signature: str,
) -> None:
    output_sha256 = _sha256_file(output)
    marker = db.execute(
        "SELECT run_signature, output_sha256 FROM indexed_regions "
        "WHERE source=? AND stream=? AND shard=?",
        (source, stream, shard),
    ).fetchone()
    if marker is not None:
        if marker != (run_signature, output_sha256):
            raise ValueError(f"indexed-region marker input mismatch: {source}/{stream}/{shard}")
        return
    _index_region(
        db,
        source,
        output,
        marker=(source, stream, shard, run_signature, output_sha256),
    )


def _summary(db: sqlite3.Connection, requested_sources: Sequence[str]) -> dict[str, object]:
    rows = db.execute(
        "SELECT source, decision, h3_cell, count(*) FROM eligible "
        "GROUP BY source, decision, h3_cell"
    ).fetchall()
    cells: dict[str, dict[str, dict[str, int]]] = {}
    for source, decision, cell, count in rows:
        cells.setdefault(source, {}).setdefault(decision, {})[cell] = count
    summary: dict[str, object] = {}
    for source in requested_sources:
        by_decision = cells.get(source, {})
        decisions = ("yes", "no", *sorted(set(by_decision) - {"yes", "no"}))
        summary[source] = {
            decision: _decision_summary(by_decision.get(decision, {})) for decision in decisions
        }
    return {
        "h3_resolution": H3_RESOLUTION,
        "max_per_h3_cell_per_decision": MAX_PER_CELL,
        "target_per_decision": TARGET,
        "sources": summary,
    }


def _decision_summary(cell_counts: Mapping[str, int]) -> dict[str, object]:
    capped_capacity = sum(min(count, MAX_PER_CELL) for count in cell_counts.values())
    return {
        "unique_eligible_polygons": sum(cell_counts.values()),
        "h3_cells": len(cell_counts),
        f"capacity_at_{MAX_PER_CELL}_per_cell": capped_capacity,
        "max_eligible_per_cell": max(cell_counts.values(), default=0),
        f"can_reach_{TARGET}_and_{MINIMUM_H3_CELL_COUNT}_cells": (
            capped_capacity >= TARGET and len(cell_counts) >= MINIMUM_H3_CELL_COUNT
        ),
    }
