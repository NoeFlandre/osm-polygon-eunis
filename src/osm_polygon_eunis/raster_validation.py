"""Native-mask and lossless-compression validation for local EEA rasters."""

import struct
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
_TIFF_COMPRESSION_CODES_BY_NAME = {
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


def validate_native_mask(dataset: rasterio.DatasetReader, path: Path) -> None:
    """Require a lossless GeoTIFF with an in-file mask and no sidecars."""
    _validated_mask_flags(dataset, path)
    _validate_lossless_compression(dataset, path)
    _reject_external_sidecars(dataset, path)


def _validate_lossless_compression(dataset: rasterio.DatasetReader, path: Path) -> None:
    compression_name = _reported_compression_name(dataset, path)
    compression_codes = _tiff_compression_codes(path)
    if compression_codes is None:
        raise ValueError(f"reference raster {path} has no verifiable compression metadata")
    if any(code not in _TIFF_LOSSLESS_COMPRESSION_CODES for code in compression_codes):
        raise ValueError(f"reference raster {path} uses unverified compression in a TIFF directory")
    if compression_name is None:
        if compression_codes[0] != 1:
            raise ValueError(f"reference raster {path} has no verifiable compression metadata")
    elif compression_codes[0] not in _TIFF_COMPRESSION_CODES_BY_NAME[compression_name]:
        raise ValueError(f"reference raster {path} has no verifiable compression metadata")


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
    if magic != _TIFF_BIG_MAGIC or len(header) < _TIFF_BIG_HEADER_SIZE:
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
    layout = tiff.directory
    file_size = tiff.file_size
    if layout.directory_offset < layout.header_size or layout.directory_offset >= file_size:
        return None
    pending = [layout.directory_offset]
    queued = {layout.directory_offset}
    visited: set[int] = set()
    compression_codes: list[int] = []
    total_entries = 0
    while pending:
        directory_offset = pending.pop()
        queued.remove(directory_offset)
        if directory_offset in visited or len(visited) >= _TIFF_MAX_DIRECTORIES:
            return None
        visited.add(directory_offset)
        directory = _read_tiff_directory(
            source,
            directory_offset,
            tiff,
        )
        if directory is None:
            return None
        compression_code, next_offset, sub_ifd_offsets, entry_count = directory
        total_entries += entry_count
        if total_entries > _TIFF_MAX_TOTAL_DIRECTORY_ENTRIES:
            return None
        compression_codes.append(compression_code)
        linked_offsets = (*sub_ifd_offsets, next_offset)
        for linked_offset in linked_offsets:
            if linked_offset == 0:
                continue
            if linked_offset in visited or linked_offset in queued:
                return None
            queued.add(linked_offset)
            pending.append(linked_offset)
    return tuple(compression_codes) if compression_codes else None


def _read_tiff_directory(
    source: BinaryIO,
    directory_offset: int,
    tiff: _TiffFileLayout,
) -> tuple[int, int, tuple[int, ...], int] | None:
    layout = tiff.directory
    directory_info = _tiff_directory_info(source, directory_offset, tiff)
    if directory_info is None:
        return None
    entry_count, entries_offset, next_offset_position = directory_info
    compression_code = 1
    compression_found = False
    sub_ifd_offsets: list[int] = []
    for entry_index in range(entry_count):
        entry_info = _read_tiff_directory_entry(
            source,
            entries_offset + entry_index * layout.entry_size,
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
    next_offset = _read_tiff_next_ifd(source, next_offset_position, tiff)
    if next_offset is None:
        return None
    return compression_code, next_offset, tuple(sub_ifd_offsets), entry_count


def _tiff_directory_info(
    source: BinaryIO,
    directory_offset: int,
    tiff: _TiffFileLayout,
) -> tuple[int, int, int] | None:
    layout, endian, file_size = tiff
    count_offset = directory_offset + layout.count_size
    if directory_offset < layout.header_size or count_offset > file_size:
        return None
    source.seek(directory_offset)
    count_bytes = source.read(layout.count_size)
    if len(count_bytes) != layout.count_size:
        return None
    entry_count = struct.unpack(f"{endian}{layout.count_format}", count_bytes)[0]
    entries_offset = count_offset
    next_offset_position = entries_offset + entry_count * layout.entry_size
    if (
        entry_count > _TIFF_MAX_DIRECTORY_ENTRIES
        or next_offset_position + layout.pointer_size > file_size
    ):
        return None
    return entry_count, entries_offset, next_offset_position


def _read_tiff_directory_entry(
    source: BinaryIO,
    entry_offset: int,
    tiff: _TiffFileLayout,
    compression_found: bool,
) -> tuple[int, int, tuple[int, ...]] | None:
    layout = tiff.directory
    endian = tiff.endian
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
        if compression_found or field_type != _TIFF_SHORT_TYPE or value_count != 1:
            return None
        compression_code = struct.unpack(f"{endian}H", value_field[:2])[0]
        return tag, compression_code, ()
    if tag == _TIFF_SUB_IFDS_TAG:
        offsets = _read_tiff_sub_ifd_offsets(
            source,
            value_field,
            field_type,
            value_count,
            tiff,
        )
        if offsets is None:
            return None
        return tag, 1, offsets
    return tag, 1, ()


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
    layout, endian, file_size = tiff
    pointer_size_by_type = {
        _TIFF_LONG_TYPE: _TIFF_CLASSIC_OFFSET_SIZE,
        _TIFF_IFD_TYPE: _TIFF_CLASSIC_OFFSET_SIZE,
        _TIFF_LONG8_TYPE: _TIFF_BIG_OFFSET_SIZE,
        _TIFF_IFD8_TYPE: _TIFF_BIG_OFFSET_SIZE,
    }
    pointer_size = pointer_size_by_type.get(field_type)
    if (
        pointer_size is None
        or (pointer_size == _TIFF_BIG_OFFSET_SIZE and layout.pointer_size != _TIFF_BIG_OFFSET_SIZE)
        or value_count > _TIFF_MAX_DIRECTORY_ENTRIES
    ):
        return None
    value_size = pointer_size * value_count
    if value_size <= layout.pointer_size:
        values = value_field[:value_size]
    else:
        format_code = "I" if layout.pointer_size == _TIFF_CLASSIC_OFFSET_SIZE else "Q"
        payload_offset = struct.unpack(f"{endian}{format_code}", value_field)[0]
        if payload_offset < layout.header_size or payload_offset + value_size > file_size:
            return None
        source.seek(payload_offset)
        values = source.read(value_size)
        if len(values) != value_size:
            return None
    if value_count == 0:
        return ()
    format_code = "I" if pointer_size == _TIFF_CLASSIC_OFFSET_SIZE else "Q"
    offsets = struct.unpack(f"{endian}{value_count}{format_code}", values)
    if any(offset < layout.header_size or offset >= file_size for offset in offsets):
        return None
    return offsets


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
