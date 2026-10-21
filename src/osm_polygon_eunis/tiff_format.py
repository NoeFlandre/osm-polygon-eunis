"""Private TIFF header, directory layout, and bounded offset definitions."""

import struct
from typing import NamedTuple

TIFF_COMPRESSION_TAG = 259
TIFF_SUB_IFDS_TAG = 330
_TIFF_CLASSIC_MAGIC = 42
_TIFF_BIG_MAGIC = 43
_TIFF_CLASSIC_HEADER_SIZE = 8
TIFF_CLASSIC_OFFSET_SIZE = 4
TIFF_BIG_HEADER_SIZE = 16
TIFF_BIG_OFFSET_SIZE = 8
TIFF_SHORT_TYPE = 3
TIFF_LONG_TYPE = 4
TIFF_IFD_TYPE = 13
TIFF_LONG8_TYPE = 16
TIFF_IFD8_TYPE = 18
TIFF_MAX_DIRECTORY_ENTRIES = 4096
TIFF_MAX_DIRECTORIES = 4096
TIFF_MAX_TOTAL_DIRECTORY_ENTRIES = 65536


class TiffDirectoryLayout(NamedTuple):
    """Offsets and decoding formats for one TIFF image file directory."""

    directory_offset: int
    header_size: int
    count_size: int
    count_format: str
    entry_size: int
    entry_format: str
    value_offset: int
    pointer_size: int


class TiffFileLayout(NamedTuple):
    """Directory layout paired with byte order and file-size bounds."""

    directory: TiffDirectoryLayout
    endian: str
    file_size: int


def tiff_endian(header: bytes) -> str | None:
    """Return the TIFF struct prefix (`<` or `>`), or None for an invalid header."""
    if len(header) < _TIFF_CLASSIC_HEADER_SIZE:
        return None
    if header[:2] == b"II":
        return "<"
    if header[:2] == b"MM":
        return ">"
    return None


def tiff_directory_layout(header: bytes, endian: str) -> TiffDirectoryLayout | None:
    """Decode the first-directory layout from a TIFF header."""
    magic = struct.unpack(f"{endian}H", header[2:4])[0]
    if magic == _TIFF_CLASSIC_MAGIC:
        return _classic_tiff_directory_layout(header, endian)
    if magic == _TIFF_BIG_MAGIC:
        return _big_tiff_directory_layout(header, endian)
    return None


def _classic_tiff_directory_layout(header: bytes, endian: str) -> TiffDirectoryLayout:
    return TiffDirectoryLayout(
        struct.unpack(f"{endian}I", header[4:8])[0],
        _TIFF_CLASSIC_HEADER_SIZE,
        2,
        "H",
        12,
        f"{endian}HHI",
        8,
        4,
    )


def _big_tiff_directory_layout(header: bytes, endian: str) -> TiffDirectoryLayout | None:
    if len(header) < TIFF_BIG_HEADER_SIZE:
        return None
    offset_size, reserved = struct.unpack(f"{endian}HH", header[4:8])
    if offset_size != TIFF_BIG_OFFSET_SIZE or reserved != 0:
        return None
    return TiffDirectoryLayout(
        struct.unpack(f"{endian}Q", header[8:16])[0],
        TIFF_BIG_HEADER_SIZE,
        8,
        "Q",
        20,
        f"{endian}HHQ",
        12,
        8,
    )


def valid_tiff_directory_offset(offset: int, tiff: TiffFileLayout) -> bool:
    """Return whether the offset lies inside the TIFF file bounds."""
    return tiff.directory.header_size <= offset < tiff.file_size
