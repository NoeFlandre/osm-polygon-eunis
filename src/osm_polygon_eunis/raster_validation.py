"""Native-mask and lossless-compression validation for local EEA rasters."""

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, NamedTuple

import rasterio
from rasterio.enums import MaskFlags

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
_TIFF_COMPRESSION_TAG = 259
_TIFF_SUB_IFDS_TAG = 330
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
_TIFF_CLASSIC_MAGIC = 42
_TIFF_BIG_MAGIC = 43
_TIFF_CLASSIC_HEADER_SIZE = 8
_TIFF_CLASSIC_OFFSET_SIZE = 4
_TIFF_BIG_HEADER_SIZE = 16
_TIFF_BIG_OFFSET_SIZE = 8
_TIFF_SHORT_TYPE = 3
_TIFF_LONG_TYPE = 4
_TIFF_IFD_TYPE = 13
_TIFF_LONG8_TYPE = 16
_TIFF_IFD8_TYPE = 18
_TIFF_MAX_DIRECTORY_ENTRIES = 4096
_TIFF_MAX_DIRECTORIES = 4096
_TIFF_MAX_TOTAL_DIRECTORY_ENTRIES = 65536


class _TiffDirectoryLayout(NamedTuple):
    directory_offset: int
    header_size: int
    count_size: int
    count_format: str
    entry_size: int
    entry_format: str
    value_offset: int
    pointer_size: int


class _TiffFileLayout(NamedTuple):
    directory: _TiffDirectoryLayout
    endian: str
    file_size: int


@dataclass
class _TiffDirectoryWalk:
    pending: list[int]
    queued: set[int]
    visited: set[int]
    compression_codes: list[int]
    total_entries: int = 0


def validate_native_mask(dataset: rasterio.DatasetReader, path: Path) -> None:
    """Require a lossless GeoTIFF with an in-file mask and no sidecars."""
    _validated_mask_flags(dataset, path)
    _validate_lossless_compression(dataset, path)
    _reject_external_sidecars(dataset, path)


def _validate_lossless_compression(dataset: rasterio.DatasetReader, path: Path) -> None:
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
            header = source.read(_TIFF_BIG_HEADER_SIZE)
            endian = _tiff_endian(header)
            if endian is None:
                return None
            layout = _tiff_directory_layout(header, endian)
            if layout is None:
                return None
            source.seek(0, 2)
            file_size = source.tell()
            return _read_tiff_compression_codes(source, _TiffFileLayout(layout, endian, file_size))
    except (OSError, OverflowError, struct.error):
        return None


def _tiff_endian(header: bytes) -> str | None:
    if len(header) < _TIFF_CLASSIC_HEADER_SIZE:
        return None
    if header[:2] == b"II":
        return "<"
    if header[:2] == b"MM":
        return ">"
    return None


def _tiff_directory_layout(header: bytes, endian: str) -> _TiffDirectoryLayout | None:
    magic = struct.unpack(f"{endian}H", header[2:4])[0]
    if magic == _TIFF_CLASSIC_MAGIC:
        return _classic_tiff_directory_layout(header, endian)
    if magic == _TIFF_BIG_MAGIC:
        return _big_tiff_directory_layout(header, endian)
    return None


def _classic_tiff_directory_layout(header: bytes, endian: str) -> _TiffDirectoryLayout:
    return _TiffDirectoryLayout(
        struct.unpack(f"{endian}I", header[4:8])[0],
        _TIFF_CLASSIC_HEADER_SIZE,
        2,
        "H",
        12,
        f"{endian}HHI",
        8,
        4,
    )


def _big_tiff_directory_layout(header: bytes, endian: str) -> _TiffDirectoryLayout | None:
    if len(header) < _TIFF_BIG_HEADER_SIZE:
        return None
    offset_size, reserved = struct.unpack(f"{endian}HH", header[4:8])
    if offset_size != _TIFF_BIG_OFFSET_SIZE or reserved != 0:
        return None
    return _TiffDirectoryLayout(
        struct.unpack(f"{endian}Q", header[8:16])[0],
        _TIFF_BIG_HEADER_SIZE,
        8,
        "Q",
        20,
        f"{endian}HHQ",
        12,
        8,
    )


def _read_tiff_compression_codes(
    source: BinaryIO,
    tiff: _TiffFileLayout,
) -> tuple[int, ...] | None:
    first_offset = tiff.directory.directory_offset
    if not _valid_tiff_directory_offset(first_offset, tiff):
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


def _valid_tiff_directory_offset(offset: int, tiff: _TiffFileLayout) -> bool:
    return tiff.directory.header_size <= offset < tiff.file_size


def _process_tiff_directory(
    source: BinaryIO,
    tiff: _TiffFileLayout,
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
    if next_total_entries > _TIFF_MAX_TOTAL_DIRECTORY_ENTRIES:
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
    return directory_offset not in visited and len(visited) < _TIFF_MAX_DIRECTORIES


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
    tiff: _TiffFileLayout,
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
    tiff: _TiffFileLayout,
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
        if tag == _TIFF_COMPRESSION_TAG:
            compression_code = entry_compression
            compression_found = True
        sub_ifd_offsets.extend(entry_sub_ifds)
    return compression_code, tuple(sub_ifd_offsets)


def _tiff_directory_info(
    source: BinaryIO,
    directory_offset: int,
    tiff: _TiffFileLayout,
) -> tuple[int, int, int] | None:
    layout, endian, file_size = tiff
    if not _valid_tiff_directory_offset(directory_offset, tiff):
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
    layout: _TiffDirectoryLayout,
    file_size: int,
) -> bool:
    return (
        entry_count > _TIFF_MAX_DIRECTORY_ENTRIES
        or next_offset_position + layout.pointer_size > file_size
    )


def _read_tiff_directory_entry(
    source: BinaryIO,
    entry_offset: int,
    tiff: _TiffFileLayout,
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
    if tag == _TIFF_COMPRESSION_TAG:
        return _read_tiff_compression_entry(
            value_field,
            field_type,
            value_count,
            compression_found,
            tiff.endian,
        )
    if tag == _TIFF_SUB_IFDS_TAG:
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
    if compression_found or field_type != _TIFF_SHORT_TYPE or value_count != 1:
        return None
    compression_code = struct.unpack(f"{endian}H", value_field[:2])[0]
    return _TIFF_COMPRESSION_TAG, compression_code, ()


def _read_tiff_sub_ifd_entry(
    source: BinaryIO,
    value_field: bytes,
    field_type: int,
    value_count: int,
    tiff: _TiffFileLayout,
) -> tuple[int, int, tuple[int, ...]] | None:
    offsets = _read_tiff_sub_ifd_offsets(
        source,
        value_field,
        field_type,
        value_count,
        tiff,
    )
    if offsets is None:
        return None
    return _TIFF_SUB_IFDS_TAG, 1, offsets


def _read_tiff_next_ifd(
    source: BinaryIO,
    offset_position: int,
    tiff: _TiffFileLayout,
) -> int | None:
    layout, endian, file_size = tiff
    source.seek(offset_position)
    offset_bytes = source.read(layout.pointer_size)
    if len(offset_bytes) != layout.pointer_size:
        return None
    format_code = "I" if layout.pointer_size == _TIFF_CLASSIC_OFFSET_SIZE else "Q"
    offset = struct.unpack(f"{endian}{format_code}", offset_bytes)[0]
    if offset and not layout.header_size <= offset < file_size:
        return None
    return offset


def _read_tiff_sub_ifd_offsets(
    source: BinaryIO,
    value_field: bytes,
    field_type: int,
    value_count: int,
    tiff: _TiffFileLayout,
) -> tuple[int, ...] | None:
    pointer_size = _sub_ifd_pointer_size(field_type, tiff.directory)
    if pointer_size is None or value_count > _TIFF_MAX_DIRECTORY_ENTRIES:
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


def _sub_ifd_pointer_size(field_type: int, layout: _TiffDirectoryLayout) -> int | None:
    pointer_size_by_type = {
        _TIFF_LONG_TYPE: _TIFF_CLASSIC_OFFSET_SIZE,
        _TIFF_IFD_TYPE: _TIFF_CLASSIC_OFFSET_SIZE,
        _TIFF_LONG8_TYPE: _TIFF_BIG_OFFSET_SIZE,
        _TIFF_IFD8_TYPE: _TIFF_BIG_OFFSET_SIZE,
    }
    pointer_size = pointer_size_by_type.get(field_type)
    if pointer_size is None:
        return None
    if pointer_size == _TIFF_BIG_OFFSET_SIZE and layout.pointer_size != _TIFF_BIG_OFFSET_SIZE:
        return None
    return pointer_size


def _read_tiff_sub_ifd_values(
    source: BinaryIO,
    value_field: bytes,
    pointer_size: int,
    value_count: int,
    tiff: _TiffFileLayout,
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
    layout: _TiffDirectoryLayout,
    endian: str,
    file_size: int,
) -> bytes | None:
    format_code = "I" if layout.pointer_size == _TIFF_CLASSIC_OFFSET_SIZE else "Q"
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
    layout: _TiffDirectoryLayout,
    file_size: int,
) -> bool:
    return offset >= layout.header_size and offset + size <= file_size


def _decode_tiff_sub_ifd_offsets(
    values: bytes,
    pointer_size: int,
    value_count: int,
    tiff: _TiffFileLayout,
) -> tuple[int, ...] | None:
    format_code = "I" if pointer_size == _TIFF_CLASSIC_OFFSET_SIZE else "Q"
    offsets = struct.unpack(f"{tiff.endian}{value_count}{format_code}", values)
    if not _valid_tiff_sub_ifd_offsets(offsets, tiff):
        return None
    return offsets


def _valid_tiff_sub_ifd_offsets(offsets: tuple[int, ...], tiff: _TiffFileLayout) -> bool:
    return all(_valid_tiff_directory_offset(offset, tiff) for offset in offsets)


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
