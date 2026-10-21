"""Decode bounded TIFF SubIFD offset arrays."""

import struct
from typing import BinaryIO

from .tiff_format import (
    TIFF_BIG_OFFSET_SIZE,
    TIFF_CLASSIC_OFFSET_SIZE,
    TIFF_IFD8_TYPE,
    TIFF_IFD_TYPE,
    TIFF_LONG8_TYPE,
    TIFF_LONG_TYPE,
    TIFF_MAX_DIRECTORY_ENTRIES,
    TiffDirectoryLayout,
    TiffFileLayout,
    valid_tiff_directory_offset,
)


def read_tiff_sub_ifd_offsets(
    source: BinaryIO,
    value_field: bytes,
    field_type: int,
    value_count: int,
    tiff: TiffFileLayout,
) -> tuple[int, ...] | None:
    """Decode and bounds-check a TIFF entry's SubIFD offsets."""
    pointer_size = _sub_ifd_pointer_size(field_type, tiff.directory)
    if pointer_size is None or value_count > TIFF_MAX_DIRECTORY_ENTRIES:
        return None
    values = _read_tiff_sub_ifd_values(
        source,
        value_field,
        pointer_size,
        value_count,
        tiff,
    )
    if values is None:
        return None
    return _decode_tiff_sub_ifd_offsets(values, pointer_size, value_count, tiff)


def _sub_ifd_pointer_size(field_type: int, layout: TiffDirectoryLayout) -> int | None:
    pointer_size_by_type = {
        TIFF_LONG_TYPE: TIFF_CLASSIC_OFFSET_SIZE,
        TIFF_IFD_TYPE: TIFF_CLASSIC_OFFSET_SIZE,
        TIFF_LONG8_TYPE: TIFF_BIG_OFFSET_SIZE,
        TIFF_IFD8_TYPE: TIFF_BIG_OFFSET_SIZE,
    }
    pointer_size = pointer_size_by_type.get(field_type)
    if pointer_size is None:
        return None
    if pointer_size == TIFF_BIG_OFFSET_SIZE and layout.pointer_size != TIFF_BIG_OFFSET_SIZE:
        return None
    return pointer_size


def _read_tiff_sub_ifd_values(
    source: BinaryIO,
    value_field: bytes,
    pointer_size: int,
    value_count: int,
    tiff: TiffFileLayout,
) -> bytes | None:
    layout, endian, file_size = tiff
    value_size = pointer_size * value_count
    if value_size <= layout.pointer_size:
        return value_field[:value_size]
    return _read_external_tiff_sub_ifd_values(
        source,
        value_field,
        value_size,
        layout,
        endian,
        file_size,
    )


def _read_external_tiff_sub_ifd_values(
    source: BinaryIO,
    value_field: bytes,
    value_size: int,
    layout: TiffDirectoryLayout,
    endian: str,
    file_size: int,
) -> bytes | None:
    format_code = "I" if layout.pointer_size == TIFF_CLASSIC_OFFSET_SIZE else "Q"
    payload_offset = struct.unpack(f"{endian}{format_code}", value_field)[0]
    if not _valid_tiff_payload_range(payload_offset, value_size, layout, file_size):
        return None
    source.seek(payload_offset)
    values = source.read(value_size)
    if len(values) != value_size:
        return None
    return values


def _valid_tiff_payload_range(
    offset: int,
    size: int,
    layout: TiffDirectoryLayout,
    file_size: int,
) -> bool:
    return offset >= layout.header_size and offset + size <= file_size


def _decode_tiff_sub_ifd_offsets(
    values: bytes,
    pointer_size: int,
    value_count: int,
    tiff: TiffFileLayout,
) -> tuple[int, ...] | None:
    format_code = "I" if pointer_size == TIFF_CLASSIC_OFFSET_SIZE else "Q"
    offsets = struct.unpack(f"{tiff.endian}{value_count}{format_code}", values)
    if not _valid_tiff_sub_ifd_offsets(offsets, tiff):
        return None
    return offsets


def _valid_tiff_sub_ifd_offsets(offsets: tuple[int, ...], tiff: TiffFileLayout) -> bool:
    return all(valid_tiff_directory_offset(offset, tiff) for offset in offsets)
