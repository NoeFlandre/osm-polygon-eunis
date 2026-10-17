"""Native-mask validation for local EEA rasters."""

from pathlib import Path

import rasterio
from rasterio.enums import MaskFlags

from .tiff_compression import validate_lossless_compression


def validate_native_mask(dataset: rasterio.DatasetReader, path: Path) -> None:
    """Require a lossless GeoTIFF with an in-file mask and no sidecars."""
    _validated_mask_flags(dataset, path)
    validate_lossless_compression(dataset, path)
    _reject_external_sidecars(dataset, path)


def _validated_mask_flags(dataset: rasterio.DatasetReader, path: Path) -> tuple[MaskFlags, ...]:
    try:
        flags = dataset.mask_flag_enums[0]
    except (IndexError, AttributeError, TypeError) as error:
        raise ValueError(f"reference raster {path} has no native validity mask") from error
    if not flags or MaskFlags.all_valid in flags:
        raise ValueError(f"reference raster {path} has no native validity mask")
    return flags


def _reject_external_sidecars(dataset: rasterio.DatasetReader, path: Path) -> None:
    try:
        source_path = Path(dataset.name).resolve()
        files = dataset.files
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(f"reference raster {path} has no verifiable native file list") from error
    for filename in files:
        if Path(filename).resolve() != source_path:
            raise ValueError(f"reference raster {path} uses an external raster sidecar: {filename}")
