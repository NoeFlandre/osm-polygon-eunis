from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest
import rasterio
from hypothesis import HealthCheck, settings
from rasterio.transform import from_origin
from shapely.geometry.base import BaseGeometry

settings.register_profile(
    "deterministic",
    settings(
        deadline=None,
        derandomize=True,
        max_examples=100,
        suppress_health_check=[HealthCheck.too_slow],
    ),
)
settings.load_profile("deterministic")


def write_single_pixel_raster(
    path: Path,
    polygon_3035: BaseGeometry,
    pixel_size: float = 1000.0,
    size: int = 4,
) -> Path:
    """Write a uint8 EPSG:3035 GeoTIFF with only the pixel under the centroid set.

    ``scripts/smoke.py`` keeps its own copy because it must run without the test tree.
    """

    min_x, _min_y, _max_x, max_y = polygon_3035.bounds
    origin_x = min_x - pixel_size
    origin_y = max_y + pixel_size
    values = np.zeros((size, size), dtype="uint8")
    center_x, center_y = polygon_3035.centroid.coords[0]
    col = int((center_x - origin_x) // pixel_size)
    row = int((origin_y - center_y) // pixel_size)
    values[row, col] = 1
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=size,
        height=size,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(origin_x, origin_y, pixel_size, pixel_size),
        nodata=0,
    ) as dataset:
        dataset.write(values, 1)
    return path


@pytest.fixture
def single_pixel_raster() -> Callable[..., Path]:
    return write_single_pixel_raster
