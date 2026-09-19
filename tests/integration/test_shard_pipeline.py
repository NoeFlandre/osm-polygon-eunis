import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rasterio
from pyproj import Transformer
from rasterio.transform import from_origin
from shapely.geometry import box, mapping
from shapely.ops import transform

from osm_polygon_eunis.reference import RasterLayer, RasterReference
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
