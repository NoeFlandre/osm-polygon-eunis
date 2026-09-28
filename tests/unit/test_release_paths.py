from __future__ import annotations

from pathlib import Path

from osm_polygon_eunis import release_orchestration


def test_source_cache_uses_grid_node_scratch_and_cleans_only_its_run_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    scratch_root = tmp_path / "node-scratch" / "source"
    scratch_root.mkdir(parents=True)
    sentinel = scratch_root / "user-file"
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setenv("EUNIS_SOURCE_DIR", str(scratch_root))

    with release_orchestration._source_cache(tmp_path / "persistent-workdir") as source_root:
        assert source_root.parent == scratch_root
        (source_root / "shard.parquet").write_bytes(b"temporary")

    assert sentinel.read_text(encoding="utf-8") == "preserve"
    assert list(scratch_root.iterdir()) == [sentinel]


def test_sidecar_root_uses_grid_persistent_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    persistent_root = tmp_path / "grid-home" / "sidecars" / "eunis"
    monkeypatch.setenv("EUNIS_SIDECAR_DIR", str(persistent_root))

    sidecar_root = release_orchestration._sidecar_root(tmp_path / "workdir")
    checkpoint = sidecar_root / "website" / "polygons" / "shard.parquet.labels.parquet.done"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint", encoding="utf-8")

    assert sidecar_root == persistent_root
    assert checkpoint.read_text(encoding="utf-8") == "checkpoint"
