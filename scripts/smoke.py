"""Run one deterministic real-geometry enrichment outside pytest."""

from __future__ import annotations

import json
import math
import tempfile
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


def main() -> int:
    with tempfile.TemporaryDirectory(prefix="osm-polygon-eunis-smoke-") as raw:
        root = Path(raw)
        source = root / "source.parquet"
        output = root / "output.parquet"
        polygon = box(2.0, 48.0, 2.05, 48.05)
        project = Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True).transform
        projected = transform(project, polygon)
        min_x, _min_y, _max_x, max_y = projected.bounds
        # Duplicates tests/conftest.py:write_single_pixel_raster on purpose: the smoke
        # script must run without importing the test tree.
        raster = root / "Prob_R11_1000m.tif"
        values = np.zeros((4, 4), dtype="uint8")
        center_x, center_y = projected.centroid.coords[0]
        col = int((center_x - (min_x - 1000.0)) // 1000.0)
        row = int(((max_y + 1000.0) - center_y) // 1000.0)
        values[row, col] = 1
        with rasterio.open(
            raster,
            "w",
            driver="GTiff",
            width=4,
            height=4,
            count=1,
            dtype="uint8",
            crs="EPSG:3035",
            transform=from_origin(min_x - 1000.0, max_y + 1000.0, 1000.0, 1000.0),
            nodata=0,
        ) as dataset:
            dataset.write(values, 1)
        pq.write_table(
            pa.table({"polygon_id": ["smoke"], "geometry": [json.dumps(mapping(polygon))]}),
            source,
        )
        reference = RasterReference((RasterLayer("R11", "Pannonian steppe", raster, "EEA-smoke"),))
        enrich_parquet_shard(source, output, reference=reference, batch_size=1)
        result = pq.read_table(output)
        assert result["eunis_code"].to_pylist() == ["R11"]
        (percentage,) = result["eunis_overlap_percentage"].to_pylist()
        assert math.isclose(percentage, 4.822971, abs_tol=1e-4), percentage
    print("smoke: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
