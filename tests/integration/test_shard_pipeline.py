import json
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform

import osm_polygon_eunis.runner as runner
from osm_polygon_eunis.eea import EeaGroup, RemoteAsset
from osm_polygon_eunis.reference import RasterLayer, RasterReference
from osm_polygon_eunis.runner import DatasetPlan
from osm_polygon_eunis.sources import DatasetSpec
from osm_polygon_eunis.transform import enrich_parquet_shard


def test_real_geometry_to_synthetic_raster_shard(tmp_path: Path) -> None:
    source = tmp_path / "source.parquet"
    destination = tmp_path / "output.parquet"
    polygon = box(2.0, 48.0, 2.05, 48.05)
    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
    projected = transform(project, polygon)
    min_x, _min_y, _max_x, max_y = projected.bounds
    pixel_size = 1_000.0
    origin_x = min_x - pixel_size
    origin_y = max_y + pixel_size
    width = 4
    height = 4
    raster_path = tmp_path / "Prob_R11_1000m.tif"

    values = np.zeros((height, width), dtype="uint8")
    center_x, center_y = projected.centroid.coords[0]
    col = int((center_x - origin_x) // pixel_size)
    row = int((origin_y - center_y) // pixel_size)
    values[row, col] = 1
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(origin_x, origin_y, pixel_size, pixel_size),
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)

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
    assert output["eunis_overlap_percentage"].to_pylist()[0] > 0.0


def test_parallel_reference_batch_processes_cached_geometry_shards(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source_root = tmp_path / "source"
    sidecar_root = tmp_path / "sidecars"
    source_root.joinpath("website").mkdir(parents=True)
    polygon = box(2.0, 48.0, 2.05, 48.05)
    source = pa.table(
        {
            "polygon_id": ["france-test"],
            "geometry": [json.dumps(mapping(polygon))],
        }
    )
    for name in ("a", "b"):
        pq.write_table(source, source_root / "website" / f"polygons__{name}.parquet")

    project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
    projected = transform(project, polygon)
    min_x, _min_y, _max_x, max_y = projected.bounds
    raster_path = tmp_path / "Prob_R11_1000m.tif"
    values = np.zeros((4, 4), dtype="uint8")
    center_x, center_y = projected.centroid.coords[0]
    pixel_size = 1_000.0
    origin_x = min_x - pixel_size
    origin_y = max_y + pixel_size
    col = int((center_x - origin_x) // pixel_size)
    row = int((origin_y - center_y) // pixel_size)
    values[row, col] = 1
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=4,
        height=4,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(origin_x, origin_y, pixel_size, pixel_size),
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)

    group = EeaGroup(
        "record",
        "raster",
        "folder",
        "service",
        {"R11": "Pannonian steppe"},
        (
            RemoteAsset(
                "/Prob_R11_1000m.tif",
                "https://example.test/raster.tif",
                raster_path.stat().st_size,
                "etag",
                "R11",
                "Pannonian steppe",
                "record",
                "EEA-test",
            ),
        ),
        None,
    )

    def fake_download(_client, _asset, destination):
        destination.write_bytes(raster_path.read_bytes())
        return "sha"

    monkeypatch.setattr(runner, "download_asset", fake_download)
    (tmp_path / "run").mkdir()
    plan = DatasetPlan(
        DatasetSpec("website", "source", "target", "polygons/*.parquet"),
        "revision",
        ("polygons/a.parquet", "polygons/b.parquet"),
        ("polygons/a.parquet", "polygons/b.parquet"),
        (),
    )
    progress: list[Mapping[str, object]] = []

    runner._process_reference_groups(
        SimpleNamespace(endpoint="https://huggingface.co", token=None),
        (plan,),
        (group,),
        sidecar_root=sidecar_root,
        source_root=source_root,
        workdir=tmp_path / "run",
        threshold=0,
        checksums={},
        batch_size=1,
        progress=progress.append,
        http_client=object(),
        parallelism=2,
    )

    for name in ("a", "b"):
        labels = pq.read_table(
            sidecar_root / "website" / f"polygons__{name}.parquet.labels.parquet"
        )
        assert labels["eunis_code"].to_pylist() == ["R11"]
    assert not list((source_root / "website").glob("*.parquet"))
    assert len(progress) == 2
