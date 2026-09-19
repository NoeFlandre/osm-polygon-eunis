from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis.reference import (
    RasterLayer,
    RasterReference,
    load_label_table,
    parse_layer_code,
)


def _write_raster(path: Path, values: list[list[int]]) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 20, 10, 10),
        nodata=0,
    ) as dataset:
        dataset.write(np.asarray(values, dtype="uint8"), 1)
    return path


def test_parse_layer_code_and_label_table(tmp_path: Path) -> None:
    labels = tmp_path / "labels.json"
    labels.write_text('{"R11": "Pannonian and Pontic sandy steppe"}', encoding="utf-8")

    assert parse_layer_code("Prob_R11_100m.tif") == "R11"
    assert load_label_table(labels) == {"R11": "Pannonian and Pontic sandy steppe"}


def test_parse_layer_code_rejects_non_reference_files() -> None:
    with pytest.raises(ValueError, match="reference layer code"):
        parse_layer_code("README.md")


def test_actual_cell_intersection_drives_percentage(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    reference = RasterReference(
        (RasterLayer("R11", "steppe", raster, "EEA-test"),),
    )

    result = reference.overlap(box(1, 11, 19, 19))

    assert result.code == "R11"
    assert result.overlap_percentage == 50.0


def test_bbox_touch_without_actual_intersection_is_null(tmp_path: Path) -> None:
    raster = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    reference = RasterReference(
        (RasterLayer("R11", "steppe", raster, "EEA-test"),),
    )

    result = reference.overlap(box(10, 10, 20, 20))

    assert result.is_empty


def test_largest_actual_overlap_wins_across_raster_layers(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    second = _write_raster(tmp_path / "Prob_R12_100m.tif", [[0, 1], [0, 0]])
    reference = RasterReference(
        (
            RasterLayer("R11", "first", first, "EEA-test"),
            RasterLayer("R12", "second", second, "EEA-test"),
        ),
    )

    result = reference.overlap(box(1, 11, 18, 19))

    assert result.code == "R11"
    assert result.overlap_percentage == pytest.approx(9 * 8 / (17 * 8) * 100)


def test_reference_rejects_mixed_source_versions(tmp_path: Path) -> None:
    first = _write_raster(tmp_path / "Prob_R11_100m.tif", [[1, 0], [0, 0]])
    second = _write_raster(tmp_path / "Prob_R12_100m.tif", [[0, 1], [0, 0]])

    with pytest.raises(ValueError, match="source version"):
        RasterReference(
            (
                RasterLayer("R11", "first", first, "one"),
                RasterLayer("R12", "second", second, "two"),
            ),
        )
