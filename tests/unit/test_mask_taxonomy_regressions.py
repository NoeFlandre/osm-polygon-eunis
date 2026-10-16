"""Regression coverage for reference-mask and taxonomy provenance."""

import struct
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
import rasterio
from rasterio.enums import MaskFlags
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis.domain import EunisResult, EunisTaxonomy
from osm_polygon_eunis.eea import parse_classification_taxonomy_rows
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference


def _write_masked_raster(
    path: Path,
    *,
    internal_mask: bool,
    compression: str | None = "deflate",
) -> Path:
    create_options: dict[str, str] = {}
    if compression is not None:
        create_options["compress"] = compression
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
        **create_options,
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


def test_internal_mask_with_relative_path_parent_components_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "nested").mkdir()
    _write_masked_raster(
        tmp_path / "nested" / "Prob_R11_100m.tif",
        internal_mask=True,
    )
    monkeypatch.chdir(tmp_path)
    relative_raster = Path("nested/../nested/Prob_R11_100m.tif")

    with rasterio.open(relative_raster) as dataset:
        assert dataset.name == str(relative_raster)
        assert dataset.files == [str(relative_raster)]

    with RasterReference((RasterLayer("R11", "steppe", relative_raster, "fixture"),)) as reference:
        result = reference.overlap(box(0, 0, 2, 1))

    assert result == EunisResult(None, None, 50.0, "fixture")


def test_external_mask_sidecar_with_relative_path_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "nested").mkdir()
    _write_masked_raster(
        tmp_path / "nested" / "Prob_R11_100m.tif",
        internal_mask=False,
    )
    monkeypatch.chdir(tmp_path)
    relative_raster = Path("nested/../nested/Prob_R11_100m.tif")

    with (
        pytest.raises(ValueError, match="external raster sidecar"),
        RasterReference((RasterLayer("R11", "steppe", relative_raster, "fixture"),)),
    ):
        pass


def _compression_dataset(
    driver: str,
    compression: object,
    *,
    path: Path | None = None,
) -> rasterio.DatasetReader:
    attributes: dict[str, object] = {
        "mask_flag_enums": [[MaskFlags.per_dataset]],
        "driver": driver,
        "compression": compression,
    }
    if path is not None:
        attributes["name"] = str(path)
        attributes["files"] = [str(path)]
    return cast(
        rasterio.DatasetReader,
        SimpleNamespace(**attributes),
    )


def _append_lossy_tiff_directory(path: Path, *, subifd: bool) -> None:
    content = bytearray(path.read_bytes())
    endian = "<" if content[:2] == b"II" else ">"
    assert struct.unpack(f"{endian}H", content[2:4])[0] == 42
    first_ifd = struct.unpack(f"{endian}I", content[4:8])[0]

    def align(offset: int) -> int:
        return (offset + 3) // 4 * 4

    compression_entry = (
        struct.pack(f"{endian}HHI", 259, 3, 1) + struct.pack(f"{endian}H", 7) + b"\0\0"
    )
    lossy_ifd = struct.pack(f"{endian}H", 1) + compression_entry + struct.pack(f"{endian}I", 0)
    if subifd:
        wrapper_offset = align(len(content))
        child_offset = wrapper_offset + 2 + 12 + 4
        subifd_entry = struct.pack(f"{endian}HHI", 330, 4, 1) + struct.pack(
            f"{endian}I", child_offset
        )
        wrapper_ifd = (
            struct.pack(f"{endian}H", 1) + subifd_entry + struct.pack(f"{endian}I", first_ifd)
        )
        content.extend(b"\0" * (wrapper_offset - len(content)))
        content.extend(wrapper_ifd)
        content.extend(lossy_ifd)
        content[4:8] = struct.pack(f"{endian}I", wrapper_offset)
        path.write_bytes(content)
        return

    entry_count = struct.unpack(f"{endian}H", content[first_ifd : first_ifd + 2])[0]
    next_pointer = first_ifd + 2 + entry_count * 12
    child_offset = align(len(content))
    content.extend(b"\0" * (child_offset - len(content)))
    content.extend(lossy_ifd)
    content[next_pointer : next_pointer + 4] = struct.pack(f"{endian}I", child_offset)
    path.write_bytes(content)


@pytest.mark.parametrize(
    ("subifd", "source_compression", "reported_compression"),
    ((False, None, None), (True, "deflate", "deflate")),
    ids=("next-ifd-without-rasterio-metadata", "subifd-with-rasterio-metadata"),
)
def test_lossy_tiff_overview_is_rejected(
    tmp_path: Path,
    subifd: bool,
    source_compression: str | None,
    reported_compression: object,
) -> None:
    raster = _write_masked_raster(
        tmp_path / "overview.tif",
        internal_mask=True,
        compression=source_compression,
    )
    _append_lossy_tiff_directory(raster, subifd=subifd)
    dataset = _compression_dataset("GTiff", reported_compression, path=raster)

    with pytest.raises(ValueError, match="unverified compression"):
        RasterReference._validate_native_mask(dataset, raster)


def test_uncompressed_internal_mask_is_accepted_after_tiff_tag_verification(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "Prob_R11_100m.tif",
        internal_mask=True,
        compression=None,
    )

    with rasterio.open(raster) as dataset:
        assert dataset.compression is None

    with RasterReference((RasterLayer("R11", "steppe", raster, "fixture"),)) as reference:
        result = reference.overlap(box(0, 0, 2, 1))

    assert result == EunisResult(None, None, 50.0, "fixture")


def test_internal_mask_remains_usable_through_lazy_opening(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "Prob_R11_100m.tif",
        internal_mask=True,
    )

    result = RasterReference((RasterLayer("R11", "steppe", raster, "fixture"),)).overlap(
        box(0, 0, 2, 1)
    )

    assert result == EunisResult(None, None, 50.0, "fixture")


def test_lossy_internal_mask_raster_is_rejected(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "Prob_R11_100m.tif",
        internal_mask=True,
        compression="jpeg",
    )

    with (
        pytest.raises(ValueError, match="unverified compression: jpeg"),
        RasterReference((RasterLayer("R11", "steppe", raster, "fixture"),)),
    ):
        pass


def test_unknown_compression_is_rejected(tmp_path: Path) -> None:
    dataset = _compression_dataset("GTiff", "unverified")

    with pytest.raises(ValueError, match="unverified compression"):
        RasterReference._validate_native_mask(dataset, tmp_path / "unknown.tif")


def test_missing_rasterio_metadata_does_not_hide_compression_tag(tmp_path: Path) -> None:
    raster = _write_masked_raster(
        tmp_path / "compressed.tif",
        internal_mask=True,
        compression="deflate",
    )

    with pytest.raises(ValueError, match="no verifiable compression metadata"):
        RasterReference._validate_native_mask(
            _compression_dataset("GTiff", None),
            raster,
        )


def test_missing_compression_metadata_without_tiff_is_rejected(tmp_path: Path) -> None:
    raster = tmp_path / "not-a-tiff.tif"
    raster.write_bytes(b"not a TIFF")

    with pytest.raises(ValueError, match="no verifiable compression metadata"):
        RasterReference._validate_native_mask(
            _compression_dataset("GTiff", None),
            raster,
        )


def test_non_geotiff_reference_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported raster driver"):
        RasterReference._validate_native_mask(
            _compression_dataset("PNG", "none"),
            tmp_path / "not-geotiff.tif",
        )


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
