"""Build a full pinned Parquet key census without loading geometries into memory."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import stat
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, BinaryIO

import pyarrow as pa
import pyarrow.parquet as pq

from osm_polygon_eunis.pool_adapters.controller import (
    RawInventoryCensus,
    _census_from_document,
)
from osm_polygon_eunis.pool_adapters.source_pins import (
    RawSourceTreePin,
    SourceTreeFile,
    polygon_geometry_tree_pins,
)
from osm_polygon_eunis.pool_contract import TextSource, canonical_native_key, native_key_json

_BATCH_ROWS = 4096
_DEFAULT_SCRATCH_CAP = 16 * 1024**3
_SQLITE_PAGE_BYTES = 4096
_MAX_BATCH_ROWS = 65_536
_NATIVE_KEY_COLUMNS = ("source_pbf", "osm_type", "osm_id")


def generate_census_document(
    source_pin: RawSourceTreePin,
    input_root: Path,
    scratch_dir: Path,
    *,
    batch_rows: int = _BATCH_ROWS,
    max_scratch_bytes: int = _DEFAULT_SCRATCH_CAP,
) -> dict[str, Any]:
    """Read pinned source keys and return the exact controller census document."""

    _validate_scan_limits(batch_rows, max_scratch_bytes)
    input_root = input_root.resolve(strict=True)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    if not scratch_dir.is_dir() or scratch_dir.is_symlink():
        raise ValueError("scratch directory must be a real directory")

    with tempfile.TemporaryDirectory(prefix="eunis-raw-census-", dir=scratch_dir) as work:
        database_path = Path(work) / "keys.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _configure_database(connection, max_scratch_bytes)
            connection.execute("CREATE TABLE keys (key_json TEXT PRIMARY KEY) WITHOUT ROWID")
            shard_rows: list[dict[str, Any]] = []
            raw_row_count = 0
            for file_pin in sorted(source_pin.files, key=lambda row: row.path):
                shard = _scan_shard(
                    connection,
                    input_root,
                    file_pin,
                    batch_rows,
                )
                shard_rows.append(shard)
                raw_row_count += shard["row_count"]

            unique_key_count = connection.execute("SELECT count(*) FROM keys").fetchone()[0]
            if unique_key_count != raw_row_count:
                raise ValueError("raw source census contains duplicate native keys")
            keyset_digest = _sorted_keyset_digest(connection)
        finally:
            connection.close()

    document = {
        "format_version": "raw-inventory-keyset-v1",
        "text_source": source_pin.text_source.value,
        "repository": source_pin.repository,
        "revision": source_pin.revision,
        "path_prefix": source_pin.path_prefix,
        "tree_inventory_sha256": source_pin.inventory_sha256,
        "shard_count": len(shard_rows),
        "raw_row_count": raw_row_count,
        "unique_key_count": unique_key_count,
        "unique_keyset_sha256": keyset_digest,
        "shards": shard_rows,
    }
    payload = _canonical_json_bytes(document)
    _census_from_document(
        RawInventoryCensus,
        document,
        hashlib.sha256(payload).hexdigest(),
    ).validate(source_pin)
    return document


def _scan_shard(
    connection: sqlite3.Connection,
    input_root: Path,
    file_pin: SourceTreeFile,
    batch_rows: int,
) -> dict[str, Any]:
    source_path = _resolve_source_path(input_root, file_pin.path)
    ordered_digest = hashlib.sha256()
    with _verified_source_file(source_path, file_pin) as stream:
        stream.seek(0)
        parquet = pq.ParquetFile(stream)
        row_count = _scan_parquet_keys(
            connection,
            parquet,
            batch_rows,
            ordered_digest,
            file_pin.path,
        )

    return {
        "path": file_pin.path,
        "size_bytes": file_pin.size_bytes,
        "blob_id": file_pin.blob_id,
        "lfs_oid": file_pin.lfs_oid,
        "row_count": row_count,
        "ordered_key_sha256": ordered_digest.hexdigest(),
    }


def _resolve_source_path(input_root: Path, source_path: str) -> Path:
    candidate = input_root / source_path
    if candidate.is_symlink():
        raise ValueError(f"pinned source shard is a symlink: {source_path}")
    resolved = candidate.resolve(strict=True)
    try:
        resolved.relative_to(input_root)
    except ValueError as error:
        raise ValueError(f"pinned source path escaped the input root: {source_path}") from error
    if not resolved.is_file():
        raise ValueError(f"pinned source shard is not a regular file: {source_path}")
    return resolved


@contextmanager
def _verified_source_file(path: Path, file_pin: SourceTreeFile) -> Iterator[BinaryIO]:
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"pinned source shard is not a regular file: {file_pin.path}")
        if before.st_size != _expected_local_size(file_pin):
            raise ValueError(f"source shard has the wrong pinned byte size: {file_pin.path}")
        initial_hashes = _source_hashes(stream, before.st_size)
        if not _source_hash_matches(file_pin, before.st_size, *initial_hashes):
            raise ValueError(f"source shard bytes differ from the pinned tree: {file_pin.path}")
        try:
            yield stream
        finally:
            after = os.fstat(stream.fileno())
            if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            ):
                raise ValueError(f"source shard changed while scanning: {file_pin.path}")
            final_hashes = _source_hashes(stream, after.st_size)
            if final_hashes != initial_hashes or not _source_hash_matches(
                file_pin,
                after.st_size,
                *final_hashes,
            ):
                raise ValueError(f"source shard changed while scanning: {file_pin.path}")


def _scan_parquet_keys(
    connection: sqlite3.Connection,
    parquet: pq.ParquetFile,
    batch_rows: int,
    ordered_digest: Any,
    path: str,
) -> int:
    missing = set(_NATIVE_KEY_COLUMNS) - set(parquet.schema_arrow.names)
    if missing:
        raise ValueError(f"source shard lacks native key columns: {sorted(missing)}")
    row_count = 0
    connection.execute("BEGIN")
    with connection:
        for batch in parquet.iter_batches(batch_size=batch_rows, columns=list(_NATIVE_KEY_COLUMNS)):
            row_count = _scan_key_batch(
                connection,
                batch.to_pylist(),
                ordered_digest,
                row_count,
                path,
            )
    if row_count != parquet.metadata.num_rows:
        raise ValueError(f"Parquet row count changed while scanning: {path}")
    return row_count


def _scan_key_batch(
    connection: sqlite3.Connection,
    rows: list[dict[str, Any]],
    ordered_digest: Any,
    row_count: int,
    path: str,
) -> int:
    for row in rows:
        key = canonical_native_key({name: row[name] for name in _NATIVE_KEY_COLUMNS})
        encoded_key = native_key_json(key)
        if row_count:
            ordered_digest.update(b"\n")
        ordered_digest.update(encoded_key.encode("utf-8"))
        try:
            connection.execute("INSERT INTO keys(key_json) VALUES (?)", (encoded_key,))
        except sqlite3.IntegrityError as error:
            raise ValueError(
                f"duplicate native key in source census at {path}#row={row_count}"
            ) from error
        row_count += 1
    return row_count


def _source_hashes(stream: BinaryIO, size_bytes: int) -> tuple[str, str]:
    git_digest = hashlib.sha1(usedforsecurity=False)
    git_digest.update(f"blob {size_bytes}\0".encode("ascii"))
    content_digest = hashlib.sha256()
    stream.seek(0)
    while chunk := stream.read(1024 * 1024):
        git_digest.update(chunk)
        content_digest.update(chunk)
    return git_digest.hexdigest(), content_digest.hexdigest()


def _expected_local_size(file_pin: SourceTreeFile) -> int:
    if file_pin.lfs_oid is not None or file_pin.verified_lfs_content_sha256 is not None:
        if file_pin.lfs_size_bytes is None:
            raise ValueError("pinned Git LFS source file has no content size")
        return file_pin.lfs_size_bytes
    return file_pin.size_bytes


def _source_hash_matches(
    file_pin: SourceTreeFile,
    size_bytes: int,
    git_blob_sha1: str,
    content_sha256: str,
) -> bool:
    if file_pin.verified_lfs_content_sha256 is not None:
        return (
            size_bytes == file_pin.lfs_size_bytes
            and content_sha256 == file_pin.verified_lfs_content_sha256
        )
    if file_pin.lfs_oid is None:
        return size_bytes == file_pin.size_bytes and git_blob_sha1 == file_pin.blob_id
    expected = file_pin.lfs_oid.removeprefix("sha256:")
    return size_bytes == file_pin.lfs_size_bytes and content_sha256 == expected


def _sorted_keyset_digest(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    first = True
    for (key_json,) in connection.execute("SELECT key_json FROM keys ORDER BY key_json"):
        if not first:
            digest.update(b"\n")
        digest.update(key_json.encode("utf-8"))
        first = False
    return digest.hexdigest()


def _configure_database(connection: sqlite3.Connection, max_scratch_bytes: int) -> None:
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA cache_size=-65536")
    connection.execute(f"PRAGMA max_page_count={max_scratch_bytes // _SQLITE_PAGE_BYTES}")


def _validate_scan_limits(batch_rows: int, max_scratch_bytes: int) -> None:
    if batch_rows < 1 or batch_rows > _MAX_BATCH_ROWS:
        raise ValueError("Parquet batch rows must be between 1 and 65536")
    if max_scratch_bytes < _SQLITE_PAGE_BYTES:
        raise ValueError("scratch byte cap must be at least one SQLite page")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _write_new_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", prefix=f".{path.name}.", suffix=".partial", dir=path.parent, delete=False
        ) as output:
            temporary = Path(output.name)
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _single_source_pin(source: TextSource) -> RawSourceTreePin:
    pins = polygon_geometry_tree_pins(source)
    if len(pins) != 1:
        raise ValueError("selected source must have exactly one pinned polygon geometry tree")
    return pins[0]


def main() -> int:
    """Generate and atomically install a verified full-source key census."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, choices=[source.value for source in TextSource])
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--scratch-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-rows", type=int, default=_BATCH_ROWS)
    parser.add_argument("--max-scratch-bytes", type=int, default=_DEFAULT_SCRATCH_CAP)
    args = parser.parse_args()

    try:
        source = TextSource(args.source)
        document = generate_census_document(
            _single_source_pin(source),
            args.input_root,
            args.scratch_dir,
            batch_rows=args.batch_rows,
            max_scratch_bytes=args.max_scratch_bytes,
        )
        payload = _canonical_json_bytes(document)
        _write_new_file(args.output, payload)
        manifest_sha256 = hashlib.sha256(payload).hexdigest()
        print(
            f"source={source.value} shards={document['shard_count']} "
            f"rows={document['raw_row_count']} unique_keys={document['unique_key_count']} "
            f"unique_keyset_sha256={document['unique_keyset_sha256']} "
            f"manifest={args.output} manifest_sha256={manifest_sha256}"
        )
    except (OSError, ValueError, TypeError, KeyError, sqlite3.Error, pa.ArrowException) as error:
        print(f"raw census generation failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
