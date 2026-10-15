"""Regression coverage for reference-mask and taxonomy provenance."""

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis.domain import EunisResult, EunisTaxonomy
from osm_polygon_eunis.eea import parse_classification_taxonomy_rows
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference


def _write_masked_raster(
    path: Path,
    *,
    internal_mask: bool,
) -> Path:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=1,
        count=1,
        dtype="uint8",
        crs="EPSG:3035",
        transform=from_origin(0, 1, 1, 1),
    ) as dataset:
        dataset.write(np.array([[1, 1]], dtype="uint8"), 1)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=internal_mask), rasterio.open(path, "r+") as dataset:
        dataset.write_mask(np.array([[255, 0]], dtype="uint8"))
    return path


def test_external_mask_sidecar_is_rejected(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "Prob_R11_100m.tif",
        internal_mask=False,
    )

    assert Path(f"{raster}.msk").is_file()
    with (
        pytest.raises(ValueError, match="external raster sidecar"),
        RasterReference((RasterLayer("R11", "steppe", raster, "fixture"),)),
    ):
        pass


def test_internal_mask_remains_usable(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "Prob_R11_100m.tif",
        internal_mask=True,
    )

    assert not Path(f"{raster}.msk").exists()
    with RasterReference((RasterLayer("R11", "steppe", raster, "fixture"),)) as reference:
        result = reference.overlap(box(0, 0, 2, 1))

    assert result == EunisResult(None, None, 50.0, "fixture")


def test_empty_native_mask_flags_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no native validity mask"):
        RasterReference._validate_native_mask(
            cast(rasterio.DatasetReader, SimpleNamespace(mask_flag_enums=[()])),
            tmp_path / "unmasked.tif",
        )


def test_taxonomy_rejects_parent_codes_without_labels() -> None:
    with pytest.raises(ValueError, match="missing label"):
        EunisTaxonomy(
            labels={"Q111": "shared label", "Q112": "shared label"},
            levels={"Q11": 2, "Q111": 3, "Q112": 3},
            parents={"Q111": "Q11", "Q112": "Q11"},
        )


def test_classification_rejects_code_rows_without_names() -> None:
    rows = (
        ("Level", "Code", "Name"),
        (2, "Q11", "parent habitat"),
        (3, "Q111", ""),
    )

    with pytest.raises(ValueError, match="missing name"):
        parse_classification_taxonomy_rows(rows)
