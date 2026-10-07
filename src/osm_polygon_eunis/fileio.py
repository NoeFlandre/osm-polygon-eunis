"""Small file helpers shared by pipeline and command modules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import IO

import httpx

from ._protocols import Hasher

# Bounded network timeouts for large streamed downloads. ``read`` applies per
# chunk, not to the whole transfer, so multi-GB files still complete while a
# stalled connection fails instead of hanging the job forever.
DOWNLOAD_TIMEOUT = httpx.Timeout(connect=30.0, read=300.0, write=300.0, pool=30.0)


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write sorted, indented JSON through a sibling temporary file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file read in 1 MiB chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_chunks(chunks: Iterable[bytes], output: IO[bytes], digest: Hasher | None = None) -> int:
    """Write chunks to an open binary stream, optionally hashing them; return bytes written."""

    written = 0
    for chunk in chunks:
        output.write(chunk)
        if digest is not None:
            digest.update(chunk)
        written += len(chunk)
    return written
