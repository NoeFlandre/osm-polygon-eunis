"""Validate compression across every reachable TIFF directory."""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

import rasterio

from .tiff_format import (
    TIFF_BIG_HEADER_SIZE,
    TIFF_CLASSIC_OFFSET_SIZE,
    TIFF_COMPRESSION_TAG,
    TIFF_MAX_DIRECTORIES,
    TIFF_MAX_DIRECTORY_ENTRIES,
    TIFF_MAX_TOTAL_DIRECTORY_ENTRIES,
    TIFF_SHORT_TYPE,
    TIFF_SUB_IFDS_TAG,
    TiffDirectoryLayout,
    TiffFileLayout,
    tiff_directory_layout,
    tiff_endian,
    valid_tiff_directory_offset,
)
from .tiff_subifd import read_tiff_sub_ifd_offsets

_LOSSLESS_COMPRESSION_NAMES = frozenset(
    {
        "ccittfax3",
        "ccittfax4",
        "ccittrle",
        "deflate",
        "lzma",
        "lzw",
        "none",
        "packbits",
        "zstd",
    }
)
_TIFF_LOSSLESS_COMPRESSION_CODES = frozenset({1, 2, 3, 4, 5, 8, 32773, 32946, 34925, 50000})
_TIFF_COMPRESSION_CODES_BY_NAME: dict[str | None, frozenset[int]] = {
    None: frozenset({1}),
    "ccittfax3": frozenset({3}),
    "ccittfax4": frozenset({4}),
    "ccittrle": frozenset({2}),
    "deflate": frozenset({8, 32946}),
    "lzma": frozenset({34925}),
    "lzw": frozenset({5}),
    "none": frozenset({1}),
    "packbits": frozenset({32773}),
    "zstd": frozenset({50000}),
}


@dataclass
class _TiffDirectoryWalk:
    pending: list[int]
    queued: set[int]
    visited: set[int]
    compression_codes: list[int]
    total_entries: int = 0


def validate_lossless_compression(dataset: rasterio.DatasetReader, path: Path) -> None:
    """Require lossless encoding in every reachable TIFF IFD and SubIFD."""
    compression_name = _reported_compression_name(dataset, path)
    compression_codes = _tiff_compression_codes(path)
    _validate_tiff_compression_codes(compression_codes, compression_name, path)


def _validate_tiff_compression_codes(
    compression_codes: tuple[int, ...] | None,
    compression_name: str | None,
    path: Path,
) -> None:
    if compression_codes is None:
        raise ValueError(f"reference raster {path} has no verifiable compression metadata")
    _require_lossless_compression_codes(compression_codes, path)
    accepted_codes = _TIFF_COMPRESSION_CODES_BY_NAME[compression_name]
    if compression_codes[0] not in accepted_codes:
        raise ValueError(f"reference raster {path} has no verifiable compression metadata")


def _require_lossless_compression_codes(codes: tuple[int, ...], path: Path) -> None:
    if not all(code in _TIFF_LOSSLESS_COMPRESSION_CODES for code in codes):
        raise ValueError(f"reference raster {path} uses unverified compression in a TIFF directory")


def _reported_compression_name(dataset: rasterio.DatasetReader, path: Path) -> str | None:
    try:
        driver = dataset.driver
        compression = dataset.compression
    except (AttributeError, OSError, RuntimeError, TypeError, ValueError) as error:
        raise ValueError(
            f"reference raster {path} has no verifiable compression metadata"
        ) from error
    if driver != "GTiff":
        raise ValueError(f"reference raster {path} uses unsupported raster driver: {driver}")
    return _normalize_compression_name(compression, path)


def _normalize_compression_name(compression: object, path: Path) -> str | None:
    if compression is None:
        return None
    name = getattr(compression, "name", compression)
    compression_name = name.casefold() if isinstance(name, str) else "<non-string>"
    if compression_name not in _LOSSLESS_COMPRESSION_NAMES:
        raise ValueError(f"reference raster {path} uses unverified compression: {compression_name}")
    return compression_name


def _tiff_compression_codes(path: Path) -> tuple[int, ...] | None:
    try:
        with path.open("rb") as source:
            header = source.read(TIFF_BIG_HEADER_SIZE)
            endian = tiff_endian(header)
            if endian is None:
                return None
            layout = tiff_directory_layout(header, endian)
            if layout is None:
                return None
            source.seek(0, 2)
            file_size = source.tell()
            return _read_tiff_compression_codes(source, TiffFileLayout(layout, endian, file_size))
    except (OSError, OverflowError, struct.error):
        return None


def _read_tiff_compression_codes(
    source: BinaryIO,
    tiff: TiffFileLayout,
) -> tuple[int, ...] | None:
    first_offset = tiff.directory.directory_offset
    if not valid_tiff_directory_offset(first_offset, tiff):
        return None
    walk = _TiffDirectoryWalk(
        pending=[first_offset],
        queued={first_offset},
        visited=set(),
        compression_codes=[],
    )
    while walk.pending:
        if not _process_tiff_directory(source, tiff, walk):
            return None
    return tuple(walk.compression_codes) if walk.compression_codes else None


def _process_tiff_directory(
    source: BinaryIO,
    tiff: TiffFileLayout,
    walk: _TiffDirectoryWalk,
) -> bool:
    directory_offset = walk.pending.pop()
    walk.queued.remove(directory_offset)
    if not _can_visit_tiff_directory(directory_offset, walk.visited):
        return False
    walk.visited.add(directory_offset)
    directory = _read_tiff_directory(source, directory_offset, tiff)
    if directory is None:
        return False
    compression_code, next_offset, sub_ifd_offsets, entry_count = directory
    next_total_entries = walk.total_entries + entry_count
    if next_total_entries > TIFF_MAX_TOTAL_DIRECTORY_ENTRIES:
        return False
    if not _enqueue_tiff_directory_links(
        sub_ifd_offsets,
        next_offset,
        walk.pending,
        walk.queued,
        walk.visited,
    ):
        return False
    walk.compression_codes.append(compression_code)
    walk.total_entries = next_total_entries
    return True


def _can_visit_tiff_directory(directory_offset: int, visited: set[int]) -> bool:
    return directory_offset not in visited and len(visited) < TIFF_MAX_DIRECTORIES


def _enqueue_tiff_directory_links(
    sub_ifd_offsets: tuple[int, ...],
    next_offset: int,
    pending: list[int],
    queued: set[int],
    visited: set[int],
) -> bool:
    for linked_offset in (*sub_ifd_offsets, next_offset):
        if linked_offset == 0:
            continue
        if _duplicate_tiff_directory_link(linked_offset, queued, visited):
            return False
        queued.add(linked_offset)
        pending.append(linked_offset)
    return True


def _duplicate_tiff_directory_link(
    linked_offset: int,
    queued: set[int],
    visited: set[int],
) -> bool:
    return linked_offset in visited or linked_offset in queued


def _read_tiff_directory(
    source: BinaryIO,
    directory_offset: int,
    tiff: TiffFileLayout,
) -> tuple[int, int, tuple[int, ...], int] | None:
    directory_info = _tiff_directory_info(source, directory_offset, tiff)
    if directory_info is None:
        return None
    entry_count, entries_offset, next_offset_position = directory_info
    parsed_entries = _read_tiff_directory_entries(
        source,
        entries_offset,
        entry_count,
        tiff,
    )
    if parsed_entries is None:
        return None
    next_offset = _read_tiff_next_ifd(source, next_offset_position, tiff)
    if next_offset is None:
        return None
    compression_code, sub_ifd_offsets = parsed_entries
    return compression_code, next_offset, sub_ifd_offsets, entry_count


def _read_tiff_directory_entries(
    source: BinaryIO,
    entries_offset: int,
    entry_count: int,
    tiff: TiffFileLayout,
) -> tuple[int, tuple[int, ...]] | None:
    compression_code = 1
    compression_found = False
    sub_ifd_offsets: list[int] = []
    for entry_index in range(entry_count):
        entry_info = _read_tiff_directory_entry(
            source,
            entries_offset + entry_index * tiff.directory.entry_size,
            tiff,
            compression_found,
        )
        if entry_info is None:
            return None
        tag, entry_compression, entry_sub_ifds = entry_info
        if tag == TIFF_COMPRESSION_TAG:
            compression_code = entry_compression
            compression_found = True
        sub_ifd_offsets.extend(entry_sub_ifds)
    return compression_code, tuple(sub_ifd_offsets)


def _tiff_directory_info(
    source: BinaryIO,
    directory_offset: int,
    tiff: TiffFileLayout,
) -> tuple[int, int, int] | None:
    layout, endian, file_size = tiff
    if not valid_tiff_directory_offset(directory_offset, tiff):
        return None
    source.seek(directory_offset)
    count_bytes = source.read(layout.count_size)
    if len(count_bytes) != layout.count_size:
        return None
    entry_count = struct.unpack(f"{endian}{layout.count_format}", count_bytes)[0]
    entries_offset = directory_offset + layout.count_size
    next_offset_position = entries_offset + entry_count * layout.entry_size
    if _tiff_entries_exceed_file(entry_count, next_offset_position, layout, file_size):
        return None
    return entry_count, entries_offset, next_offset_position


def _tiff_entries_exceed_file(
    entry_count: int,
    next_offset_position: int,
    layout: TiffDirectoryLayout,
    file_size: int,
) -> bool:
    return (
        entry_count > TIFF_MAX_DIRECTORY_ENTRIES
        or next_offset_position + layout.pointer_size > file_size
    )


def _read_tiff_directory_entry(
    source: BinaryIO,
    entry_offset: int,
    tiff: TiffFileLayout,
    compression_found: bool,
) -> tuple[int, int, tuple[int, ...]] | None:
    layout = tiff.directory
    source.seek(entry_offset)
    entry = source.read(layout.entry_size)
    if len(entry) != layout.entry_size:
        return None
    tag, field_type, value_count = struct.unpack(
        layout.entry_format,
        entry[: layout.value_offset],
    )
    value_field = entry[layout.value_offset : layout.value_offset + layout.pointer_size]
    if tag == TIFF_COMPRESSION_TAG:
        return _read_tiff_compression_entry(
            value_field,
            field_type,
            value_count,
            compression_found,
            tiff.endian,
        )
    if tag == TIFF_SUB_IFDS_TAG:
        return _read_tiff_sub_ifd_entry(
            source,
            value_field,
            field_type,
            value_count,
            tiff,
        )
    return tag, 1, ()


def _read_tiff_compression_entry(
    value_field: bytes,
    field_type: int,
    value_count: int,
    compression_found: bool,
    endian: str,
) -> tuple[int, int, tuple[int, ...]] | None:
    if compression_found or field_type != TIFF_SHORT_TYPE or value_count != 1:
        return None
    compression_code = struct.unpack(f"{endian}H", value_field[:2])[0]
    return TIFF_COMPRESSION_TAG, compression_code, ()


def _read_tiff_sub_ifd_entry(
    source: BinaryIO,
    value_field: bytes,
    field_type: int,
    value_count: int,
    tiff: TiffFileLayout,
) -> tuple[int, int, tuple[int, ...]] | None:
    offsets = read_tiff_sub_ifd_offsets(
        source,
        value_field,
        field_type,
        value_count,
        tiff,
    )
    if offsets is None:
        return None
    return TIFF_SUB_IFDS_TAG, 1, offsets


def _read_tiff_next_ifd(
    source: BinaryIO,
    offset_position: int,
    tiff: TiffFileLayout,
) -> int | None:
    layout, endian, file_size = tiff
    source.seek(offset_position)
    offset_bytes = source.read(layout.pointer_size)
    if len(offset_bytes) != layout.pointer_size:
        return None
    format_code = "I" if layout.pointer_size == TIFF_CLASSIC_OFFSET_SIZE else "Q"
    offset = struct.unpack(f"{endian}{format_code}", offset_bytes)[0]
    if offset and not layout.header_size <= offset < file_size:
        return None
    return offset
