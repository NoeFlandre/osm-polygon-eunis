"""Behavioral tests for the restartable ordered geometry spool."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from threading import Barrier
from typing import cast

import pytest
from scripts.eunis_geometry_census_sources import _json_row
from scripts.eunis_geometry_census_spool import (
    OrderedGeometrySpool,
    OrderedGeometrySpoolIdentity,
    prepare_ordered_geometry_spool,
)


def _identity(**overrides: object) -> OrderedGeometrySpoolIdentity:
    values: dict[str, object] = {
        "run_signature": "1" * 64,
        "region_signature": "2" * 64,
        "code_sha256": "3" * 64,
        "source": "website",
        "stream": "website",
        "shard": "england",
        "label_repository": "owner/labels",
        "label_revision": "4" * 40,
        "label_path": "labels/polygons/england-latest.parquet",
        "raw_repository": "owner/raw",
        "raw_revision": "5" * 40,
        "raw_paths": ("polygons/england.parquet",),
        "query_sha256": "6" * 64,
        "chunk_rows": 2,
    }
    values.update(overrides)
    return OrderedGeometrySpoolIdentity(
        run_signature=cast(str, values["run_signature"]),
        region_signature=cast(str, values["region_signature"]),
        code_sha256=cast(str, values["code_sha256"]),
        source=cast(str, values["source"]),
        stream=cast(str, values["stream"]),
        shard=cast(str, values["shard"]),
        label_repository=cast(str, values["label_repository"]),
        label_revision=cast(str, values["label_revision"]),
        label_path=cast(str, values["label_path"]),
        raw_repository=cast(str, values["raw_repository"]),
        raw_revision=cast(str, values["raw_revision"]),
        raw_paths=cast(tuple[str, ...], values["raw_paths"]),
        query_sha256=cast(str, values["query_sha256"]),
        chunk_rows=cast(int, values["chunk_rows"]),
    )


def _rows(count: int = 5) -> list[dict[str, object]]:
    sequence = list(reversed(range(count)))
    return [
        {
            "label_path": "labels/polygons/england-latest.parquet",
            "shard": "england",
            "stream": "website",
            "decision": "yes" if index % 2 else "no",
            "sentence_key": f"polygon-{index}|website|0",
            "polygon_id": str(index),
            "osm_type": "way",
            "osm_id": str(1000 + index),
            "source_pbf": None,
            "text_sha256": hashlib.sha256(f"text-{index}".encode()).hexdigest(),
            "geometry": b"\x01wkb-" + str(index).encode(),
            "_field": "website",
            "_sentence_index": 0,
            "_description_identity": f"description-{index}",
            "_tag_key": "landuse",
            "_sentence_id": f"sentence-{index}",
            "_raw_polygon_url": f"https://raw.example/polygon/{index}",
            "_raw_text_url": f"https://raw.example/text/{index}",
            "_order": f"order-{sequence.index(index)}",
        }
        for index in sequence
    ]


class _ObservedRows:
    def __init__(self) -> None:
        self.iterated = False

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        self.iterated = True
        return iter(())


def _candidate_output_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    return b"".join(
        (json.dumps(_json_row(row), sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )


def _spool_bytes(rows: Iterable[Mapping[str, object]]) -> bytes:
    encoded = []
    for row in rows:
        record = dict(row)
        geometry = record.get("geometry")
        if isinstance(geometry, bytes):
            record["geometry"] = {"__eunis_spool_bytes_v1__": geometry.hex()}
        encoded.append(
            (
                json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"
            ).encode()
        )
    return b"".join(encoded)


def _interrupted_rows(rows: list[dict[str, object]]) -> Iterator[dict[str, object]]:
    yield from rows[:3]
    raise RuntimeError("simulated interrupted query")


def _interrupted_after_two_chunks(
    rows: list[dict[str, object]],
) -> Iterator[dict[str, object]]:
    yield from rows[:5]
    raise RuntimeError("simulated interrupted query")


def test_spool_round_trip_preserves_hidden_query_fields_order_and_output_bytes(
    tmp_path: Path,
) -> None:
    rows = _rows()
    expected_output = _candidate_output_bytes(rows)

    spool = prepare_ordered_geometry_spool(rows, tmp_path / "spool", _identity())
    restored = list(spool.rows())

    assert spool.mode == "built"
    assert restored == rows
    assert [row["_order"] for row in restored] == [row["_order"] for row in rows]
    assert all(row["_field"] == "website" for row in restored)
    assert all(row["_sentence_index"] == 0 for row in restored)
    assert all("_raw_polygon_url" in row and "_raw_text_url" in row for row in restored)
    assert all(isinstance(row["geometry"], bytes) for row in restored)
    assert _candidate_output_bytes(restored) == expected_output
    assert (
        hashlib.sha256(expected_output).hexdigest()
        == hashlib.sha256(_candidate_output_bytes(restored)).hexdigest()
    )
    assert spool.row_count == len(rows)
    assert spool.byte_count == len(_spool_bytes(rows))
    assert spool.content_sha256 == hashlib.sha256(_spool_bytes(rows)).hexdigest()


def test_completed_spool_reuse_does_not_execute_the_geometry_query(tmp_path: Path) -> None:
    rows = _rows()
    identity = _identity()
    first = prepare_ordered_geometry_spool(rows, tmp_path / "spool", identity)
    assert list(first.rows()) == rows

    def forbidden_query() -> Iterator[dict[str, object]]:
        raise AssertionError("a complete pinned spool should bypass the geometry query")
        yield {}

    reused = prepare_ordered_geometry_spool(forbidden_query(), tmp_path / "spool", identity)

    assert first.content_sha256 == reused.content_sha256
    assert reused.mode == "reused"
    assert list(reused.rows()) == rows


def test_interrupted_spool_recovers_only_verified_ordered_prefix(tmp_path: Path) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"

    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    identity_root = spool_root / identity.sha256
    old_attempts = list((identity_root / "attempts").iterdir())
    assert len(old_attempts) == 1
    old_chunk = old_attempts[0] / "chunks" / "chunk-00000000.jsonl"
    saved_chunk = old_chunk.read_bytes()

    recovered = prepare_ordered_geometry_spool(rows, spool_root, identity)

    assert recovered.mode == "recovered"
    assert recovered.recovered_rows == 2
    assert list(recovered.rows()) == rows
    assert old_chunk.read_bytes() == saved_chunk
    assert recovered.row_count == len(rows)
    assert recovered.content_sha256 == hashlib.sha256(_spool_bytes(rows)).hexdigest()


def test_recovery_rejects_changed_prefix_and_preserves_interrupted_attempt(
    tmp_path: Path,
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    identity_root = spool_root / identity.sha256
    attempt_path = next((identity_root / "attempts").iterdir())
    before = {
        path.relative_to(attempt_path): path.read_bytes()
        for path in attempt_path.rglob("*")
        if path.is_file()
    }
    changed = [dict(row) for row in rows]
    changed[0]["sentence_key"] = "different-pinned-row"

    with pytest.raises(ValueError, match="differs from saved spool prefix"):
        prepare_ordered_geometry_spool(changed, spool_root, identity)

    after = {
        path.relative_to(attempt_path): path.read_bytes()
        for path in attempt_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert not (identity_root / "complete.json").exists()


def test_corrupt_committed_interrupted_chunk_fails_before_query_consumption(
    tmp_path: Path,
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    identity_root = spool_root / identity.sha256
    attempt_path = next((identity_root / "attempts").iterdir())
    chunk = attempt_path / "chunks" / "chunk-00000000.jsonl"
    chunk.write_bytes(chunk.read_bytes().replace(b"order-0", b"order-x"))
    corrupted = chunk.read_bytes()
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert chunk.read_bytes() == corrupted


def test_corrupt_interrupted_build_identity_fails_before_query_consumption(
    tmp_path: Path,
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(
            _interrupted_after_two_chunks(rows), spool_root, identity
        )
        list(spool.rows())

    attempt_path = next((spool_root / identity.sha256 / "attempts").iterdir())
    build_path = attempt_path / "build.json"
    build_path.write_text("{", encoding="utf-8")
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="corrupt build identity"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert build_path.read_text(encoding="utf-8") == "{"


def test_missing_chunk_metadata_before_committed_chunk_fails_closed(
    tmp_path: Path,
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(
            _interrupted_after_two_chunks(rows), spool_root, identity
        )
        list(spool.rows())

    attempt_path = next((spool_root / identity.sha256 / "attempts").iterdir())
    chunks_path = attempt_path / "chunks"
    first_metadata = chunks_path / "chunk-00000000.json"
    second_metadata = chunks_path / "chunk-00000001.json"
    first_metadata.unlink()
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="gap before committed chunks"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert second_metadata.is_file()
    assert not first_metadata.exists()


def test_symlinked_attempt_directory_fails_before_query_consumption(tmp_path: Path) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    attempts_path = spool_root / identity.sha256 / "attempts"
    attempt_path = next(attempts_path.iterdir())
    preserved_path = attempt_path.with_name(attempt_path.name + "-preserved")
    attempt_path.rename(preserved_path)
    attempt_path.symlink_to(preserved_path, target_is_directory=True)
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="not a regular directory"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert attempt_path.is_symlink()
    assert (preserved_path / "chunks" / "chunk-00000000.jsonl").is_file()


def test_renamed_attempt_with_committed_chunk_fails_before_query_consumption(
    tmp_path: Path,
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    attempts_path = spool_root / identity.sha256 / "attempts"
    attempt_path = next(attempts_path.iterdir())
    renamed_path = attempt_path.with_name(attempt_path.name + "-preserved")
    attempt_path.rename(renamed_path)
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="invalid name"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert (renamed_path / "chunks" / "chunk-00000000.jsonl").is_file()


@pytest.mark.parametrize("chunk_directory", ["missing", "renamed"])
def test_renamed_attempt_with_missing_or_renamed_chunks_fails_before_query_consumption(
    tmp_path: Path, chunk_directory: str
) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    attempts_path = spool_root / identity.sha256 / "attempts"
    attempt_path = next(attempts_path.iterdir())
    chunks_path = attempt_path / "chunks"
    if chunk_directory == "missing":
        chunks_path.rename(spool_root / "orphaned-chunks")
    else:
        chunks_path.rename(attempt_path / "chunks-preserved")
    renamed_path = attempt_path.with_name(attempt_path.name + "-preserved")
    attempt_path.rename(renamed_path)
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="invalid name"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert (renamed_path / "build.json").is_file()
    if chunk_directory == "missing":
        assert (spool_root / "orphaned-chunks" / "chunk-00000000.jsonl").is_file()
    else:
        assert (renamed_path / "chunks-preserved" / "chunk-00000000.jsonl").is_file()


def test_missing_chunk_directory_with_build_record_fails_closed(tmp_path: Path) -> None:
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"
    with pytest.raises(RuntimeError, match="simulated interrupted query"):
        spool = prepare_ordered_geometry_spool(_interrupted_rows(rows), spool_root, identity)
        list(spool.rows())

    attempts_path = spool_root / identity.sha256 / "attempts"
    attempt_path = next(attempts_path.iterdir())
    chunks_path = attempt_path / "chunks"
    preserved_chunks = attempt_path / "chunks-preserved"
    chunks_path.rename(preserved_chunks)
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="chunk directory is missing"):
        prepare_ordered_geometry_spool(untouched_rows, spool_root, identity)

    assert not untouched_rows.iterated
    assert (preserved_chunks / "chunk-00000000.jsonl").is_file()


def test_dangling_complete_pointer_fails_before_query_consumption(tmp_path: Path) -> None:
    identity = _identity()
    identity_root = tmp_path / "spool" / identity.sha256
    identity_root.mkdir(parents=True)
    (identity_root / "complete.json").symlink_to("missing-manifest.json")
    untouched_rows = _ObservedRows()

    with pytest.raises(ValueError, match="manifest is not a regular file"):
        prepare_ordered_geometry_spool(untouched_rows, tmp_path / "spool", identity)

    assert not untouched_rows.iterated


def test_pin_change_uses_a_distinct_spool_and_keeps_the_first_immutable(
    tmp_path: Path,
) -> None:
    rows = _rows()
    spool_root = tmp_path / "spool"
    first = prepare_ordered_geometry_spool(rows, spool_root, _identity())
    assert list(first.rows()) == rows
    changed_identity = replace(_identity(), raw_revision="7" * 40)

    second = prepare_ordered_geometry_spool(rows, spool_root, changed_identity)

    assert second.mode == "built"
    assert first.root != second.root
    assert list(second.rows()) == rows
    assert (first.root / "complete.json").is_file()
    assert (second.root / "complete.json").is_file()


def test_completed_chunk_corruption_fails_closed_without_repairing_it(tmp_path: Path) -> None:
    rows = _rows()
    identity = _identity()
    spool = prepare_ordered_geometry_spool(rows, tmp_path / "spool", identity)
    assert list(spool.rows()) == rows
    assert spool.manifest is not None
    attempt_id = str(spool.manifest["attempt_id"])
    chunk = spool.root / "attempts" / attempt_id / "chunks" / "chunk-00000000.jsonl"
    saved = chunk.read_bytes()
    lines = saved.splitlines(keepends=True)
    record = json.loads(lines[0])
    record["_order"] = str(record["_order"]) + "corrupt"
    lines[0] = (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()
    chunk.write_bytes(b"".join(lines))
    corrupted = chunk.read_bytes()

    with pytest.raises(ValueError, match="checksum mismatch"):
        prepare_ordered_geometry_spool(rows, tmp_path / "spool", identity)

    assert chunk.read_bytes() == corrupted


def test_concurrent_builders_publish_one_complete_manifest_without_overwrite(
    tmp_path: Path,
) -> None:
    barrier = Barrier(2)
    rows = _rows()
    identity = _identity()
    spool_root = tmp_path / "spool"

    def gated_rows() -> Iterator[dict[str, object]]:
        barrier.wait(timeout=10)
        yield from rows

    def build() -> OrderedGeometrySpool:
        spool = prepare_ordered_geometry_spool(gated_rows(), spool_root, identity)
        assert list(spool.rows()) == rows
        return spool

    with ThreadPoolExecutor(max_workers=2) as executor:
        first, second = executor.submit(build), executor.submit(build)
        spools = [first.result(timeout=20), second.result(timeout=20)]

    assert sorted(spool.mode for spool in spools) == ["built", "reused"]
    attempts = list((spool_root / identity.sha256 / "attempts").iterdir())
    assert len(attempts) == 2
    assert (spool_root / identity.sha256 / "complete.json").is_file()
