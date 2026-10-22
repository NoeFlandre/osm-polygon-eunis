from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from scripts.generate_raw_inventory_census import generate_census_document

from osm_polygon_eunis.pool_adapters.source_pins import RawSourceTreePin, SourceTreeFile
from osm_polygon_eunis.pool_contract import TextSource, canonical_native_key, native_key_json
from osm_polygon_eunis.pool_contract.builder import native_keyset_sha256
from osm_polygon_eunis.pool_contract.models import NativeKey


def _write_shard(path: Path, rows: list[dict[str, object]]) -> SourceTreeFile:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
    payload = path.read_bytes()
    git_blob_id = hashlib.sha1(
        f"blob {len(payload)}\0".encode("ascii") + payload,
        usedforsecurity=False,
    ).hexdigest()
    return SourceTreeFile(
        path=path.relative_to(path.parents[1]).as_posix(),
        size_bytes=len(payload),
        blob_id=git_blob_id,
        lfs_oid=None,
        lfs_size_bytes=None,
    )


def _source_pin(files: list[SourceTreeFile]) -> RawSourceTreePin:
    return RawSourceTreePin(
        text_source=TextSource.WEBSITE,
        repository="example/polygons",
        revision="1" * 40,
        path_prefix="polygons/",
        file_count=len(files),
        total_size_bytes=sum(file.size_bytes for file in files),
        inventory_sha256="a" * 64,
        files=tuple(files),
        purpose="polygon_geometry",
    )


def _row(osm_id: int, source_pbf: str = "region.osm.pbf") -> dict[str, object]:
    return {
        "source_pbf": source_pbf,
        "osm_type": "way",
        "osm_id": osm_id,
        "geometry": b"synthetic-wkb",
    }


def _native_key(row: dict[str, object]) -> NativeKey:
    source_pbf = row["source_pbf"]
    osm_type = row["osm_type"]
    osm_id = row["osm_id"]
    assert isinstance(source_pbf, str)
    assert isinstance(osm_type, str)
    assert isinstance(osm_id, int)
    return canonical_native_key({"source_pbf": source_pbf, "osm_type": osm_type, "osm_id": osm_id})


def _ordered_digest(rows: list[dict[str, object]]) -> str:
    digest = hashlib.sha256()
    for index, row in enumerate(rows):
        if index:
            digest.update(b"\n")
        key = _native_key(row)
        digest.update(native_key_json(key).encode("utf-8"))
    return digest.hexdigest()


def test_census_generator_emits_verified_shard_and_sorted_keyset_digests(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    scratch_dir = tmp_path / "scratch"
    first_rows = [_row(4), _row(2)]
    second_rows = [_row(3), _row(1, "other.osm.pbf")]
    first_pin = _write_shard(input_root / "polygons/a.parquet", first_rows)
    second_pin = _write_shard(input_root / "polygons/b.parquet", second_rows)
    pin = _source_pin([first_pin, second_pin])

    document = generate_census_document(
        pin,
        input_root,
        scratch_dir,
        batch_rows=1,
        max_scratch_bytes=4 * 1024 * 1024,
    )

    keys = [_native_key(row) for row in first_rows + second_rows]
    assert document["format_version"] == "raw-inventory-keyset-v1"
    assert document["shard_count"] == 2
    assert document["raw_row_count"] == document["unique_key_count"] == 4
    assert document["unique_keyset_sha256"] == native_keyset_sha256(keys)
    assert [row["ordered_key_sha256"] for row in document["shards"]] == [
        _ordered_digest(first_rows),
        _ordered_digest(second_rows),
    ]
    assert list(scratch_dir.iterdir()) == []


def test_census_generator_rejects_duplicate_native_keys_across_shards(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    first_pin = _write_shard(input_root / "polygons/a.parquet", [_row(1)])
    second_pin = _write_shard(input_root / "polygons/b.parquet", [_row(1)])
    pin = _source_pin([first_pin, second_pin])

    with pytest.raises(ValueError, match="duplicate native key"):
        generate_census_document(
            pin,
            input_root,
            tmp_path / "scratch",
            max_scratch_bytes=4 * 1024 * 1024,
        )


def test_census_generator_rejects_bytes_outside_the_pinned_source_tree(
    tmp_path: Path,
) -> None:
    input_root = tmp_path / "input"
    pin_file = _write_shard(input_root / "polygons/a.parquet", [_row(1)])
    shard = input_root / "polygons/a.parquet"
    payload = bytearray(shard.read_bytes())
    payload[0] ^= 0xFF
    shard.write_bytes(payload)

    with pytest.raises(ValueError, match="differ from the pinned tree"):
        generate_census_document(
            _source_pin([pin_file]),
            input_root,
            tmp_path / "scratch",
            max_scratch_bytes=4 * 1024 * 1024,
        )
