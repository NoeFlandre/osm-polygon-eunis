"""Small file helpers shared by download, publish and card modules."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from pathlib import Path
from typing import IO, Any


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file read in 1 MiB chunks."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_chunks(chunks: Iterable[bytes], output: IO[bytes], digest: Any = None) -> int:
    """Write chunks to an open binary stream, optionally hashing them; return bytes written."""

    written = 0
    for chunk in chunks:
        output.write(chunk)
        if digest is not None:
            digest.update(chunk)
        written += len(chunk)
    return written
