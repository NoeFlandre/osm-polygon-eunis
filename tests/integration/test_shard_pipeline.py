import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from pyproj import Transformer
from shapely.geometry import box, mapping
from shapely.ops import transform

from osm_polygon_eunis import geometry_chunks, geometry_jobs, reference_cache, run_analysis
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.options import BatchLimits
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference
from osm_polygon_eunis.release_plan import DatasetPlan
from osm_polygon_eunis.sources import DatasetSpec
from osm_polygon_eunis.transform import enrich_parquet_shard


def test_real_geometry_to_synthetic_raster_shard(tmp_path: Path, single_pixel_raster) -> None:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "output.parquet"
    polygon = box(2.0, 48.0, 2.05, 48.05)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
    projected = transform(project, polygon)
    raster_path = single_pixel_raster(tmp_path / "Prob_R11_1000m.tif", projected)

    pq.write_table(
        pa.table(
            {
                "polygon_id": ["france-test"],
                "geometry": [json.dumps(mapping(polygon))],
            }
        ),
        source,
    )
    reference = RasterReference(
        (RasterLayer("R11", "Pannonian steppe", raster_path, "EEA-test"),),
    )

    enrich_parquet_shard(source, destination, reference=reference, batch_size=1)

    output = pq.read_table(destination)
    assert output["polygon_id"].to_pylist() == ["france-test"]
    assert output["eunis_code"].to_pylist() == ["R11"]
    assert output["eunis_overlap_percentage"].to_pylist() == [pytest.approx(4.822971, abs=1e-4)]


def _spawned_geometry_fixture(
    tmp_path: Path,
    monkeypatch,
    single_pixel_raster,
) -> tuple[
    Path,
    Path,
    pa.Table,
    Path,
    geometry_chunks.GeometryRunOptions,
    list[Mapping[str, object]],
]:
    source_root = tmp_path / "source"
    source_directory = source_root / "website"
    source_directory.mkdir(parents=True)
    sidecar_root = tmp_path / "sidecars"
    polygons = (box(2.0, 48.0, 2.05, 48.05), box(3.0, 49.0, 3.05, 49.05))
    source = pa.table(
        {
            "polygon_id": ["france-test-a", "france-test-b"],
            "geometry": [json.dumps(mapping(polygon)) for polygon in polygons],
        }
    )
    for name in ("a", "b"):
        pq.write_table(source, source_directory / f"polygons__{name}.parquet")

    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
    reference_inputs = (
        ("R11", "steppe", "record-r11", polygons[0]),
        ("R12", "moorland", "record-r12", polygons[1]),
    )
    raster_paths: dict[str, Path] = {}
    groups: list[EeaGroup] = []
    for code, name, record_id, polygon in reference_inputs:
        raster_path = single_pixel_raster(
            tmp_path / f"Prob_{code}_1000m.tif", transform(project, polygon)
        )
        asset_path = f"/Prob_{code}_1000m.tif"
        raster_paths[asset_path] = raster_path
        groups.append(
            EeaGroup(
                record_id,
                f"raster-{code.lower()}",
                "folder",
                "service",
                {code: name},
                (
                    RemoteAsset(
                        asset_path,
                        f"https://example.test/{code.lower()}.tif",
                        raster_path.stat().st_size,
                        f"etag-{code.lower()}",
                        code,
                        name,
                        record_id,
                        "EEA-test",
                    ),
                ),
                None,
            )
        )

    def fake_download(_client, asset, destination):
        destination.write_bytes(raster_paths[asset.path].read_bytes())
        return f"sha-{asset.code.lower()}"

    monkeypatch.setattr(reference_cache, "download_asset", fake_download)
    run_directory = tmp_path / "run"
    run_directory.mkdir()
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/a.parquet", "polygons/b.parquet"),
        ("polygons/a.parquet", "polygons/b.parquet"),
        (),
    )
    progress: list[Mapping[str, object]] = []
    options = geometry_chunks.GeometryRunOptions(
        api=cast(HubApi, SimpleNamespace(endpoint="https://huggingface.co", token=None)),
        plans=(plan,),
        groups=tuple(groups),
        sidecar_root=sidecar_root,
        source_root=source_root,
        workdir=run_directory,
        threshold=0,
        checksums={},
        limits=BatchLimits(workers=2, parquet_batch_size=1, raster_groups_per_batch=1),
        progress=progress.append,
        http_client=cast(StreamClient, object()),
    )
    return source_root, source_directory, source, sidecar_root, options, progress


def _json_events(log_text: str) -> list[dict[str, object]]:
    return [json.loads(line) for line in log_text.splitlines() if line.startswith("{")]


def _assert_timing_batches(
    records: list[dict[str, object]],
    expected_batches: Mapping[str, list[int]],
) -> None:
    timing_records = [record for record in records if record["event"] == "geometry_batch_done"]
    actual_batches = {
        path: [record["reference_batch"] for record in timing_records if record["path"] == path]
        for path in expected_batches
    }
    assert actual_batches == expected_batches


def _assert_checkpoint_state(
    sidecar_root: Path,
    name: str,
    signature: str,
) -> None:
    labels = pq.read_table(sidecar_root / "website" / f"polygons__{name}.parquet.labels.parquet")
    assert labels["eunis_code"].to_pylist() == ["R11", "R12"]
    marker = sidecar_root / "website" / f"polygons__{name}.parquet.labels.parquet.done"
    assert marker.read_bytes() == f'{{"signature": "{signature}", "batches": [0, 1]}}'.encode()


def _assert_run_analysis(
    log_text: str,
    log_path: Path,
    sidecar_root: Path,
    event_count: int,
) -> None:
    log_path.write_text(log_text, encoding="utf-8")
    summary = run_analysis.summarize_run((log_path,), sidecar_root)
    assert summary["datasets"][0]["event_count"] == event_count
    assert summary["datasets"][0]["checkpoint_batches_observed"] == 4
    assert summary["datasets"][0]["progress"] == {
        "geometry_shards_completed": 2,
        "geometry_shards_total": 2,
        "reference_batches_completed": 4,
        "reference_batches_total": 4,
        "reference_batches_remaining": 0,
        "percent": 100.0,
    }
    assert summary["inputs"] == {
        "log_files": 1,
        "malformed_json_lines": 0,
        "malformed_records": 0,
        "repeated_plans": 0,
        "malformed_checkpoints": 0,
        "stale_checkpoints": 0,
        "missing_metadata_datasets": [],
    }


@pytest.mark.slow
def test_parallel_reference_batch_processes_cached_geometry_shards(
    tmp_path: Path,
    monkeypatch,
    single_pixel_raster,
    capfd,
) -> None:
    source_root, source_directory, source, sidecar_root, options, progress = (
        _spawned_geometry_fixture(tmp_path, monkeypatch, single_pixel_raster)
    )
    geometry_jobs.process_reference_groups(options)
    first_log = capfd.readouterr().err
    first_records = _json_events(first_log)

    expected_signatures = {
        "polygons/a.parquet": "50a7639a50ba905c3aeccb50e9ae945eea28f6e0c6551231a9ebbbde023c1339",
        "polygons/b.parquet": "a65c631602fda5e61eaab16bc6288db0497530b08aa9128049a21ae28c8903dd",
    }
    assert [record for record in first_records if record["event"] == "geometry_run_plan"] == [
        {
            "event": "geometry_run_plan",
            "dataset": "website",
            "reference_batch_ids": [0, 1],
            "checkpoint_signatures": expected_signatures,
        }
    ]
    _assert_timing_batches(first_records, {path: [0, 1] for path in expected_signatures})
    assert progress == [
        {"event": "sidecar_updated", "dataset": "website", "path": path}
        for path in expected_signatures
    ]
    for name, path in (("a", "polygons/a.parquet"), ("b", "polygons/b.parquet")):
        _assert_checkpoint_state(sidecar_root, name, expected_signatures[path])
    assert not list((source_root / "website").glob("*.parquet"))
    _assert_run_analysis(first_log, tmp_path / "first-run.jsonl", sidecar_root, 4)

    capfd.readouterr()
    for name, path in (("a", "polygons/a.parquet"), ("b", "polygons/b.parquet")):
        marker = sidecar_root / "website" / f"polygons__{name}.parquet.labels.parquet.done"
        marker.write_text(
            json.dumps({"signature": expected_signatures[path], "batches": [0]}),
            encoding="utf-8",
        )
        pq.write_table(source, source_directory / f"polygons__{name}.parquet")
    progress.clear()

    geometry_jobs.process_reference_groups(options)
    resume_log = capfd.readouterr().err
    resume_records = _json_events(resume_log)
    _assert_timing_batches(resume_records, {path: [1] for path in expected_signatures})
    assert progress == [
        {"event": "sidecar_updated", "dataset": "website", "path": path}
        for path in expected_signatures
    ]
    for name, path in (("a", "polygons/a.parquet"), ("b", "polygons/b.parquet")):
        _assert_checkpoint_state(sidecar_root, name, expected_signatures[path])
    _assert_run_analysis(resume_log, tmp_path / "resume-run.jsonl", sidecar_root, 2)
