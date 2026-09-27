from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

from osm_polygon_eunis.fileio import sha256_file, write_chunks


def test_sha256_file_hashes_large_file_incrementally(tmp_path: Path) -> None:
    path = tmp_path / "payload.bin"
    payload = b"polygon-data" * 200_000
    path.write_bytes(payload)

    assert sha256_file(path) == hashlib.sha256(payload).hexdigest()


def test_write_chunks_writes_and_hashes_streamed_bytes() -> None:
    output = BytesIO()
    digest = hashlib.sha256()

    written = write_chunks((b"first", b"", b"second"), output, digest)

    assert written == 11
    assert output.getvalue() == b"firstsecond"
    assert digest.hexdigest() == hashlib.sha256(b"firstsecond").hexdigest()


def test_write_chunks_without_digest_only_counts_bytes() -> None:
    output = BytesIO()

    written = write_chunks((b"one", b"two"), output)

    assert written == 6
    assert output.getvalue() == b"onetwo"
