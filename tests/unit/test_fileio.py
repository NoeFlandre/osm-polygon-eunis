from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import pytest

from osm_polygon_eunis.fileio import sha256_file, write_chunks, write_json_atomic


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


def test_write_json_atomic_creates_parents_and_preserves_serialization(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "receipts" / "run.json"

    write_json_atomic(path, {"z": [True, None], "a": {"z": "é", "a": 1}})

    assert path.read_bytes() == (
        b'{\n  "a": {\n    "a": 1,\n    "z": "\\u00e9"\n  },\n'
        b'  "z": [\n    true,\n    null\n  ]\n}\n'
    )
    assert not path.with_name(".run.json.tmp").exists()


def test_write_json_atomic_preserves_nonfinite_float_serialization(tmp_path: Path) -> None:
    path = tmp_path / "run.json"

    write_json_atomic(path, {"values": [float("nan"), float("inf"), float("-inf")]})

    assert (
        path.read_bytes() == b'{\n  "values": [\n    NaN,\n    Infinity,\n    -Infinity\n  ]\n}\n'
    )


def test_write_json_atomic_replaces_from_sibling_after_utf8_write(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "run.json"
    path.write_bytes(b"original\n")
    temporary = path.with_name(".run.json.tmp")
    expected = b'{\n  "value": "new"\n}\n'
    original_write_text = Path.write_text
    original_replace = Path.replace
    writes = []
    replacements = []

    def record_write(source, text, *, encoding):
        writes.append((source, text, encoding))
        return original_write_text(source, text, encoding=encoding)

    def record_replace(source, target):
        replacements.append((source, target))
        assert source.read_bytes() == expected
        assert path.read_bytes() == b"original\n"
        return original_replace(source, target)

    monkeypatch.setattr(Path, "write_text", record_write)
    monkeypatch.setattr(Path, "replace", record_replace)

    write_json_atomic(path, {"value": "new"})

    assert writes == [(temporary, expected.decode("utf-8"), "utf-8")]
    assert replacements == [(temporary, path)]
    assert path.read_bytes() == expected
    assert not temporary.exists()


@pytest.mark.parametrize("existing_destination", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "write", "replace"])
def test_write_json_atomic_preserves_destination_and_cleans_up_on_failure(
    monkeypatch, tmp_path: Path, existing_destination: bool, failure: str
) -> None:
    path = tmp_path / "run.json"
    if existing_destination:
        path.write_bytes(b"original\n")
    temporary = path.with_name(".run.json.tmp")
    temporary.write_bytes(b"stale temporary\n")
    payload = {"value": object() if failure == "serialization" else "new"}

    def fail_write(source, _text, *, encoding):
        assert source == temporary
        assert encoding == "utf-8"
        source.write_bytes(b"partial")
        raise OSError("write failed")

    def fail_replace(source, target):
        assert source == temporary
        assert target == path
        raise OSError("replace failed")

    if failure == "write":
        monkeypatch.setattr(Path, "write_text", fail_write)
    elif failure == "replace":
        monkeypatch.setattr(Path, "replace", fail_replace)

    error = TypeError if failure == "serialization" else OSError
    message = "not JSON serializable" if failure == "serialization" else f"{failure} failed"
    with pytest.raises(error, match=message):
        write_json_atomic(path, payload)

    if existing_destination:
        assert path.read_bytes() == b"original\n"
    else:
        assert not path.exists()
    assert not temporary.exists()
