"""Bounded, body-free joins for EUNIS source text and relevance tables.

The inputs mirror the pinned Hugging Face Arrow schemas. Text bodies are read
only while computing an exact UTF-8 SHA-256; the join spool and output records
contain hashes and provenance, never sentence bodies.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from itertools import chain
from pathlib import Path
from typing import IO, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from .pool_adapter import (
    EUNIS_POOL_SOURCE_LAYOUTS,
    EUNIS_WEBSITE_FIELD_TO_SENTENCE_COLUMN,
    EunisPoolSourceLayout,
    SourceTreeFile,
    SourceTreeGroupPin,
    validate_source_tree_group,
)
from .pool_contract import (
    BodyStatus,
    PolygonInventoryRow,
    PoolBatch,
    RelevanceAssessment,
    RelevanceStatus,
    SourceDocument,
    StreamingLimits,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
    canonical_native_key,
    native_key_json,
)
from .pool_contract.models import NativeKey
from .pool_contract.streaming import DEFAULT_MAX_BATCH_BYTES, DEFAULT_MAX_BATCH_ROWS, MAX_ROW_BYTES

_SHA256_LENGTH = 64
_JSON = json.dumps
_DESCRIPTION_MAPPING_RULE = "description-identity-source-text-sha256-tag-language-index-v1"
_WEBSITE_MAPPING_RULE = "polygon-id-field-index-text-sha256-v1"
_WIKI_MAPPING_RULE = "corpus-sentence-id-text-sha256-document-link-v1"
_MISSING_LINK_PREFIX = "no-source-polygon-link"
_BATCH_ROW_OVERHEAD = 2
_OSM_KEY_FIELD_COUNT = 3


@dataclass(frozen=True, slots=True)
class PinnedSourceRow:
    """One row from a locally verified Parquet shard, with stable provenance."""

    path_in_repo: str
    row_ordinal: int
    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Reject incomplete source row locators."""

        if not self.path_in_repo.strip() or self.row_ordinal < 0:
            raise ValueError("pinned source rows need a repository path and nonnegative ordinal")

    @property
    def locator(self) -> str:
        """Return an immutable table path and row locator."""

        return f"{self.path_in_repo}#row={self.row_ordinal}"


@dataclass(frozen=True, slots=True)
class JoinedTextRecord:
    """One source document, all its polygon links, and one relevance result."""

    document: SourceDocument
    links: tuple[TextPolygonLink, ...]
    relevance: RelevanceAssessment


@dataclass(frozen=True, slots=True)
class JoinAllocationBounds:
    """Hard per-record bounds applied before output links and evidence are materialized."""

    max_record_rows: int = DEFAULT_MAX_BATCH_ROWS
    max_record_bytes: int = DEFAULT_MAX_BATCH_BYTES
    max_output_row_bytes: int = MAX_ROW_BYTES

    def __post_init__(self) -> None:
        """Reject nonpositive or internally inconsistent allocation limits."""

        if min(self.max_record_rows, self.max_record_bytes, self.max_output_row_bytes) <= 0:
            raise ValueError("join allocation bounds must be positive")
        if self.max_output_row_bytes > self.max_record_bytes:
            raise ValueError("maximum output row bytes cannot exceed maximum record bytes")

    @classmethod
    def from_streaming_limits(cls, limits: StreamingLimits) -> JoinAllocationBounds:
        """Match the join producer bounds to the selected shared export limits."""

        return cls(limits.max_batch_rows, limits.max_batch_bytes, limits.max_row_bytes)


_BOUNDED_JOIN_STREAM_TOKEN = object()


@dataclass(frozen=True, slots=True, init=False)
class BoundedJoinedTextRecords:
    """Single-pass joined records whose producer enforced explicit allocation bounds."""

    records: Iterable[JoinedTextRecord]
    allocation_bounds: JoinAllocationBounds
    _token: object = dataclass_field(repr=False, compare=False)

    def __init__(
        self,
        records: Iterable[JoinedTextRecord],
        allocation_bounds: JoinAllocationBounds,
        *,
        _token: object,
    ) -> None:
        """Accept only the private token held by a bounded join producer."""

        if _token is not _BOUNDED_JOIN_STREAM_TOKEN:
            raise TypeError("bounded join streams must be created by the bounded join producer")
        object.__setattr__(self, "records", records)
        object.__setattr__(self, "allocation_bounds", allocation_bounds)
        object.__setattr__(self, "_token", _token)

    def __iter__(self) -> Iterator[JoinedTextRecord]:
        """Return the underlying single-pass record iterator."""

        return iter(self.records)

    @classmethod
    def combine(cls, *streams: BoundedJoinedTextRecords) -> BoundedJoinedTextRecords:
        """Chain compatible verified streams without discarding their bound proof."""

        if not streams:
            raise ValueError("at least one bounded join stream is required")
        bounds = streams[0].allocation_bounds
        if any(stream.allocation_bounds != bounds for stream in streams[1:]):
            raise ValueError("bounded join streams must use identical allocation bounds")
        return cls(chain(*streams), bounds, _token=_BOUNDED_JOIN_STREAM_TOKEN)


class _BoundedTextBuilder:
    """Spill UTF-8 output after 64 KiB and reject growth before writing past its cap."""

    def __init__(self, max_bytes: int, description: str) -> None:
        self._max_bytes = max_bytes
        self._description = description
        self._byte_count = 0
        self._stack = ExitStack()
        self._file: IO[str] | None = None

    def __enter__(self) -> _BoundedTextBuilder:
        self._file = self._stack.enter_context(
            tempfile.SpooledTemporaryFile(
                max_size=min(self._max_bytes, 64 * 1024),
                mode="w+t",
                encoding="utf-8",
                newline="",
            )
        )
        return self

    def __exit__(self, *_exc: object) -> None:
        self._stack.close()

    def append(self, fragment: str) -> None:
        fragment_bytes = len(fragment.encode("utf-8"))
        if self._byte_count + fragment_bytes > self._max_bytes:
            raise ValueError(self._description)
        self._require_open_file().write(fragment)
        self._byte_count += fragment_bytes

    def append_repeated(self, fragment: str, count: int) -> None:
        """Append repeated text only after proving its total size fits the cap."""
        if count <= 0 or not fragment:
            return
        fragment_bytes = len(fragment.encode("utf-8"))
        if self._byte_count + fragment_bytes * count > self._max_bytes:
            raise ValueError(self._description)
        repeats_per_chunk = max(1, (64 * 1024) // fragment_bytes)
        while count:
            repeats = min(count, repeats_per_chunk)
            self.append(fragment * repeats)
            count -= repeats

    def value(self) -> str:
        file = self._require_open_file()
        file.flush()
        file.seek(0)
        return file.read()

    def _require_open_file(self) -> IO[str]:
        if self._file is None:
            raise RuntimeError("bounded text builder is not active")
        return self._file


@dataclass(frozen=True, slots=True)
class _RelevanceRowContract:
    role: str
    required_fields: tuple[str, ...]
    key_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _DescriptionRowMetadata:
    identity: str | None
    tag_key: str | None
    language: str | None
    source_text_sha256: str
    polygon_key: NativeKey | None


@dataclass(frozen=True, slots=True)
class _WebsiteRowMetadata:
    polygon_id: str | None
    polygon_key: NativeKey | None


@dataclass(frozen=True, slots=True)
class _WikiSentenceDocument:
    document_id: str
    native_key: NativeKey
    content_sha256: str | None
    body_status: BodyStatus
    scope: str
    join_key: str | None
    document_id_from_source: str | None
    project: str | None
    source_locator: str
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class _RelevanceResolution:
    status: RelevanceStatus
    raw_value: str | None
    reason: str | None


@dataclass(frozen=True, slots=True)
class _StagedTextDocument:
    document: SourceDocument
    relevance_evidence_sha256: str | None
    label_scope: str
    label_join_key: str | None
    direct_link_key: str | None
    wiki_document_id: str | None
    wiki_project: str | None
    source_locator: str
    corpus: str | None
    missing_reason: str | None


@dataclass(frozen=True, slots=True)
class _LinkRecordContext:
    document: SourceDocument
    source_revision: str
    relevance: RelevanceAssessment
    allocation_bounds: JoinAllocationBounds


@dataclass(frozen=True, slots=True)
class _JoinEmissionContext:
    text_source: TextSource
    layout: EunisPoolSourceLayout
    mapping_rule: str
    wiki_corpus: str | None
    release_id: str
    relevance_repository: str
    relevance_revision: str
    allocation_bounds: JoinAllocationBounds


@dataclass(frozen=True, slots=True)
class _OrphanLabelContext:
    text_source: TextSource
    layout: EunisPoolSourceLayout
    release_id: str
    mapping_rule: str
    wiki_corpus: str | None


@dataclass(frozen=True, slots=True)
class _OrphanLabelIdentity:
    scope: str
    label_id: str
    locator: str
    join_key_sha256: str

    def fields(self, text_source: TextSource) -> dict[str, str]:
        return {
            "text_source": text_source.value,
            "scope": self.scope,
            "label_id": self.label_id,
            "label_row": self.locator,
            "join_key_sha256": self.join_key_sha256,
        }


@dataclass(frozen=True, slots=True)
class _WikiLinkRow:
    document_id: str
    project: str
    native_key: NativeKey
    evidence: str
    complete_key: bool


@dataclass(frozen=True, slots=True)
class _PinnedJoinRequest:
    text_source: TextSource
    text_table: EunisPinnedTable
    relevance_table: EunisPinnedTable
    inventory_rows: Iterable[PolygonInventoryRow]
    wiki_link_rows: Iterable[PinnedSourceRow]
    wiki_corpus: str | None
    website_field_to_column: Mapping[str, str] | None
    layout: EunisPoolSourceLayout
    batch_size: int
    staging_directory: Path | None
    allocation_bounds: JoinAllocationBounds


@dataclass(slots=True)
class _JoinBatchAccumulator:
    starting_index: int
    max_rows: int
    max_bytes: int
    batch_index: int = dataclass_field(init=False)
    documents: list[SourceDocument] = dataclass_field(default_factory=list)
    links: list[TextPolygonLink] = dataclass_field(default_factory=list)
    relevance: list[RelevanceAssessment] = dataclass_field(default_factory=list)
    row_count: int = 0
    byte_count: int = 0

    def __post_init__(self) -> None:
        self.batch_index = self.starting_index

    def append(self, record: JoinedTextRecord) -> PoolBatch | None:
        record_rows = _joined_record_row_count(record)
        record_size = _record_size(record)
        _validate_record_batch_bounds(record_rows, record_size, self.max_rows, self.max_bytes)
        emitted = self._flush_before_append(record_rows, record_size)
        self.documents.append(record.document)
        self.links.extend(record.links)
        self.relevance.append(record.relevance)
        self.row_count += record_rows
        self.byte_count += record_size
        return emitted

    def flush(self) -> PoolBatch | None:
        if not self.documents:
            return None
        batch = PoolBatch(
            batch_index=self.batch_index,
            documents=tuple(self.documents),
            text_polygon_links=tuple(self.links),
            relevance_assessments=tuple(self.relevance),
        )
        self.batch_index += 1
        self.documents.clear()
        self.links.clear()
        self.relevance.clear()
        self.row_count = 0
        self.byte_count = 0
        return batch

    def _would_exceed(self, record_rows: int, record_size: int) -> bool:
        return (
            self.row_count + record_rows > self.max_rows
            or self.byte_count + record_size > self.max_bytes
        )

    def _flush_before_append(self, record_rows: int, record_size: int) -> PoolBatch | None:
        if not self.documents or not self._would_exceed(record_rows, record_size):
            return None
        batch = self.flush()
        if batch is None:
            raise RuntimeError("a nonempty downstream batch failed to flush")
        return batch


def _validate_record_batch_bounds(
    record_rows: int,
    record_size: int,
    max_rows: int,
    max_bytes: int,
) -> None:
    exceeds_row_limit = record_rows > max_rows
    exceeds_byte_limit = record_size > max_bytes
    if exceeds_row_limit or exceeds_byte_limit:
        raise ValueError("one joined text record exceeds the configured batch bounds")


@dataclass(frozen=True, slots=True)
class EunisPinnedParquetFile:
    """One local Parquet file tied to its pinned Hub tree metadata."""

    tree_file: SourceTreeFile
    local_path: Path


@dataclass(frozen=True, slots=True)
class EunisPinnedTable:
    """A full, metadata-pinned group of Parquet files with bounded row reading."""

    pin: SourceTreeGroupPin
    files: tuple[EunisPinnedParquetFile, ...]

    def __post_init__(self) -> None:
        """Check this table's complete file inventory against its tree pin."""

        validate_source_tree_group(self.pin, (item.tree_file for item in self.files))

    def iter_rows(
        self,
        required_columns: Sequence[str],
        *,
        batch_size: int = 4096,
        expected_types: Mapping[str, str] | None = None,
    ) -> Iterator[PinnedSourceRow]:
        """Read only named columns after verifying every local file against its pin."""

        _validate_pinned_row_request(required_columns, batch_size)
        for item in sorted(self.files, key=lambda value: value.tree_file.path):
            yield from _iter_pinned_file_rows(item, required_columns, batch_size, expected_types)


def _validate_pinned_row_request(required_columns: Sequence[str], batch_size: int) -> None:
    if batch_size <= 0:
        raise ValueError("Parquet row batch size must be positive")
    if not required_columns or len(set(required_columns)) != len(required_columns):
        raise ValueError("pinned table reads need a unique, nonempty column list")


def _iter_pinned_file_rows(
    item: EunisPinnedParquetFile,
    required_columns: Sequence[str],
    batch_size: int,
    expected_types: Mapping[str, str] | None,
) -> Iterator[PinnedSourceRow]:
    _verify_local_tree_file(item)
    parquet = pq.ParquetFile(item.local_path)
    missing = sorted(set(required_columns) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"{item.tree_file.path} lacks pinned source columns {missing}")
    _validate_arrow_types(parquet.schema_arrow, expected_types or {}, item.tree_file.path)
    row_ordinal = 0
    for batch in parquet.iter_batches(
        batch_size=batch_size,
        columns=list(required_columns),
        use_threads=False,
    ):
        for row in batch.to_pylist():
            yield PinnedSourceRow(item.tree_file.path, row_ordinal, row)
            row_ordinal += 1


def missing_wiki_join_tree_groups(
    layout: EunisPoolSourceLayout | None = None,
) -> tuple[str, ...]:
    """List Wiki metadata groups that block a verified document/link join."""

    selected = layout or EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI]
    if selected.text_source is not TextSource.WIKI:
        raise ValueError("Wiki tree requirements apply only to the Wiki source")
    known = _source_tree_prefixes(selected)
    return tuple(path for path in _required_wiki_tree_prefixes() if path not in known)


def _source_tree_prefixes(layout: EunisPoolSourceLayout) -> set[str]:
    return {
        layout.raw_polygon_tree_pin.path_prefix,
        *(pin.path_prefix for pin in layout.additional_tree_pins),
    }


def _required_wiki_tree_prefixes() -> tuple[str, ...]:
    return (
        "polygon_document_links/",
        "wikipedia/documents/",
        "wikivoyage/documents/",
    )


def iter_eunis_join_records(  # noqa: PLR0913
    text_source: TextSource,
    text_rows: Iterable[PinnedSourceRow],
    relevance_rows: Iterable[PinnedSourceRow],
    inventory_rows: Iterable[PolygonInventoryRow],
    *,
    wiki_link_rows: Iterable[PinnedSourceRow] = (),
    wiki_corpus: str | None = None,
    website_field_to_column: Mapping[str, str] | None = None,
    layout: EunisPoolSourceLayout | None = None,
    staging_directory: Path | None = None,
    allocation_bounds: JoinAllocationBounds | None = None,
) -> BoundedJoinedTextRecords:
    """Join a pinned source table to relevance labels without filtering rows.

    ``text_rows`` are Description sentence rows, Website polygon rows, or one
    Wiki sentence table. ``relevance_rows`` are the matching pinned labels
    table. Wiki joins additionally consume the pinned polygon-document links.
    The inputs are streamed into a disk-backed index; no text body is persisted.
    """

    bounds = allocation_bounds or JoinAllocationBounds()
    records = _iter_eunis_join_records(
        text_source,
        text_rows,
        relevance_rows,
        inventory_rows,
        wiki_link_rows=wiki_link_rows,
        wiki_corpus=wiki_corpus,
        website_field_to_column=website_field_to_column,
        layout=layout,
        staging_directory=staging_directory,
        allocation_bounds=bounds,
    )
    return BoundedJoinedTextRecords(records, bounds, _token=_BOUNDED_JOIN_STREAM_TOKEN)


def _iter_eunis_join_records(  # noqa: PLR0913
    text_source: TextSource,
    text_rows: Iterable[PinnedSourceRow],
    relevance_rows: Iterable[PinnedSourceRow],
    inventory_rows: Iterable[PolygonInventoryRow],
    *,
    wiki_link_rows: Iterable[PinnedSourceRow],
    wiki_corpus: str | None,
    website_field_to_column: Mapping[str, str] | None,
    layout: EunisPoolSourceLayout | None,
    staging_directory: Path | None,
    allocation_bounds: JoinAllocationBounds,
) -> Iterator[JoinedTextRecord]:
    selected_layout = layout or EUNIS_POOL_SOURCE_LAYOUTS[text_source]
    _validate_join_layout(selected_layout, text_source)
    _validate_join_source_options(text_source, wiki_corpus, website_field_to_column)

    mapper = {
        TextSource.DESCRIPTION: _stage_description_rows,
        TextSource.WEBSITE: _stage_website_rows,
        TextSource.WIKI: _stage_wiki_rows,
    }[text_source]
    mapping_rule = {
        TextSource.DESCRIPTION: _DESCRIPTION_MAPPING_RULE,
        TextSource.WEBSITE: _WEBSITE_MAPPING_RULE,
        TextSource.WIKI: f"{_WIKI_MAPPING_RULE}:{wiki_corpus}",
    }[text_source]
    with tempfile.TemporaryDirectory(dir=staging_directory, prefix="eunis-text-join-") as work:
        connection = sqlite3.connect(Path(work) / "text-join.sqlite")
        try:
            connection.execute("PRAGMA temp_store=FILE")
            _initialize_join_index(connection)
            _stage_inventory(connection, inventory_rows)
            _stage_relevance(
                connection,
                relevance_rows,
                text_source,
                wiki_corpus,
                allocation_bounds.max_output_row_bytes,
            )
            if text_source is TextSource.WIKI:
                _stage_wiki_links(
                    connection,
                    wiki_link_rows,
                    str(wiki_corpus),
                    allocation_bounds.max_output_row_bytes,
                )
            mapper(
                connection,
                text_rows,
                selected_layout,
                mapping_rule,
                wiki_corpus=wiki_corpus,
                website_field_to_column=website_field_to_column,
            )
            yield from _emit_join_records(
                connection,
                text_source,
                selected_layout,
                mapping_rule,
                wiki_corpus,
                allocation_bounds,
            )
        finally:
            connection.close()


def iter_pinned_eunis_join_records(  # noqa: PLR0913
    text_source: TextSource,
    text_table: EunisPinnedTable,
    relevance_table: EunisPinnedTable,
    inventory_rows: Iterable[PolygonInventoryRow],
    *,
    wiki_link_table: EunisPinnedTable | None = None,
    wiki_document_tables: Sequence[EunisPinnedTable] = (),
    wiki_corpus: str | None = None,
    website_field_to_column: Mapping[str, str] | None = None,
    layout: EunisPoolSourceLayout | None = None,
    batch_size: int = 4096,
    staging_directory: Path | None = None,
    allocation_bounds: JoinAllocationBounds | None = None,
) -> BoundedJoinedTextRecords:
    """Validate source group pins and schemas, then run the bounded join."""

    selected_layout = layout or EUNIS_POOL_SOURCE_LAYOUTS[text_source]
    _validate_join_layout(selected_layout, text_source)
    _validate_join_source_options(text_source, wiki_corpus, website_field_to_column)
    wiki_link_rows = _pinned_wiki_link_rows(
        selected_layout,
        text_source,
        wiki_link_table,
        wiki_document_tables,
        wiki_corpus,
        batch_size,
    )
    return _run_pinned_join(
        _PinnedJoinRequest(
            text_source,
            text_table,
            relevance_table,
            inventory_rows,
            wiki_link_rows,
            wiki_corpus,
            website_field_to_column,
            selected_layout,
            batch_size,
            staging_directory,
            allocation_bounds or JoinAllocationBounds(),
        )
    )


def _validate_join_layout(layout: EunisPoolSourceLayout, text_source: TextSource) -> None:
    if layout.text_source is not text_source:
        raise ValueError("text-source layout does not match the requested join")


def _validate_join_source_options(
    text_source: TextSource,
    wiki_corpus: str | None,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    _validate_wiki_corpus_option(text_source, wiki_corpus)
    _validate_non_website_mapping_option(text_source, website_field_to_column)
    _validate_website_mapping_option(text_source, website_field_to_column)


def _validate_wiki_corpus_option(text_source: TextSource, wiki_corpus: str | None) -> None:
    if text_source is TextSource.WIKI and wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki joins require corpus='wikipedia' or corpus='wikivoyage'")


def _validate_non_website_mapping_option(
    text_source: TextSource,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    if text_source is not TextSource.WEBSITE and website_field_to_column is not None:
        raise ValueError("Website field mapping can only be supplied for Website joins")


def _validate_website_mapping_option(
    text_source: TextSource,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    if text_source is TextSource.WEBSITE and not _website_field_mapping_is_pinned(
        website_field_to_column
    ):
        raise ValueError("Website field mapping must match the pinned source label vocabulary")


def _website_field_mapping_is_pinned(
    website_field_to_column: Mapping[str, str] | None,
) -> bool:
    return website_field_to_column is None or dict(website_field_to_column) == dict(
        EUNIS_WEBSITE_FIELD_TO_SENTENCE_COLUMN
    )


def _pinned_wiki_link_rows(
    layout: EunisPoolSourceLayout,
    text_source: TextSource,
    link_table: EunisPinnedTable | None,
    document_tables: Sequence[EunisPinnedTable],
    wiki_corpus: str | None,
    batch_size: int,
) -> Iterable[PinnedSourceRow]:
    if text_source is not TextSource.WIKI:
        return ()
    missing = missing_wiki_join_tree_groups(layout)
    if missing:
        missing_paths = ", ".join(missing)
        raise ValueError("Wiki document/link tree metadata is not pinned for: " + missing_paths)
    if link_table is None:
        raise ValueError("Wiki joins require the pinned polygon_document_links table")
    _require_expected_pin(layout, link_table, "wiki_document_links", "polygon_document_links/")
    _require_wiki_document_tables(layout, document_tables, wiki_corpus)
    return _iter_pinned_wiki_link_rows(link_table, batch_size)


def _require_wiki_document_tables(
    layout: EunisPoolSourceLayout,
    tables: Sequence[EunisPinnedTable],
    wiki_corpus: str | None,
) -> None:
    prefix = f"{wiki_corpus}/documents/"
    if not tables or not any(table.pin.path_prefix == prefix for table in tables):
        raise ValueError(f"Wiki document pin for {prefix} is absent")
    for table in tables:
        _require_expected_pin(layout, table, "wiki_documents", table.pin.path_prefix)


def _iter_pinned_wiki_link_rows(
    table: EunisPinnedTable,
    batch_size: int,
) -> Iterable[PinnedSourceRow]:
    columns = (
        "document_id",
        "project",
        "polygon_id",
        "source_pbf",
        "osm_type",
        "osm_id",
        "link_sources",
    )
    types = {
        "document_id": "string",
        "project": "string",
        "polygon_id": "string",
        "source_pbf": "string",
        "osm_type": "string",
        "osm_id": "int64",
        "link_sources": "string",
    }
    return table.iter_rows(columns, batch_size=batch_size, expected_types=types)


def _run_pinned_join(request: _PinnedJoinRequest) -> BoundedJoinedTextRecords:
    text_source = request.text_source
    text_table = request.text_table
    relevance_table = request.relevance_table
    text_role, text_prefix, text_columns, text_types = _text_table_contract(
        text_source, request.wiki_corpus
    )
    relevance_role, relevance_prefix, relevance_columns, relevance_types = (
        _relevance_table_contract(text_source, request.wiki_corpus)
    )
    _require_expected_pin(request.layout, text_table, text_role, text_prefix)
    _require_expected_pin(request.layout, relevance_table, relevance_role, relevance_prefix)
    return iter_eunis_join_records(
        text_source,
        text_table.iter_rows(
            text_columns,
            batch_size=request.batch_size,
            expected_types=text_types,
        ),
        relevance_table.iter_rows(
            relevance_columns,
            batch_size=request.batch_size,
            expected_types=relevance_types,
        ),
        request.inventory_rows,
        wiki_link_rows=request.wiki_link_rows,
        wiki_corpus=request.wiki_corpus,
        website_field_to_column=request.website_field_to_column,
        layout=request.layout,
        staging_directory=request.staging_directory,
        allocation_bounds=request.allocation_bounds,
    )


def _require_expected_pin(
    layout: EunisPoolSourceLayout,
    table: EunisPinnedTable,
    role: str,
    path_prefix: str,
) -> None:
    expected = _expected_layout_pins(layout, role, path_prefix)
    if len(expected) != 1:
        raise ValueError(f"source layout has no unique verified tree pin for {role}:{path_prefix}")
    if table.pin != expected[0]:
        raise ValueError(f"input table pin differs from source layout for {role}:{path_prefix}")


def _expected_layout_pins(
    layout: EunisPoolSourceLayout,
    role: str,
    path_prefix: str,
) -> list[SourceTreeGroupPin]:
    return [
        pin
        for pin in (layout.raw_polygon_tree_pin, *layout.additional_tree_pins)
        if pin.role == role and pin.path_prefix == path_prefix
    ]


def _text_table_contract(
    source: TextSource,
    wiki_corpus: str | None,
) -> tuple[str, str, tuple[str, ...], dict[str, str]]:
    if source is TextSource.DESCRIPTION:
        return (
            "raw_text",
            "language-v1/data/",
            (
                "description_identity",
                "source_pbf",
                "osm_type",
                "osm_id",
                "tag_key",
                "original_text",
                "text_sha256",
                "language_code",
                "sentences",
                "split_status",
                "split_reason",
            ),
            {
                "description_identity": "string",
                "source_pbf": "string",
                "osm_type": "string",
                "osm_id": "int64",
                "tag_key": "string",
                "original_text": "string",
                "text_sha256": "string",
                "language_code": "string",
                "sentences": "list<string>",
                "split_status": "string",
                "split_reason": "string",
            },
        )
    if source is TextSource.WEBSITE:
        return (
            "raw_geometry",
            "polygons/",
            (
                "polygon_id",
                "source_pbf",
                "osm_type",
                "osm_id",
                "website_sentences",
                "contact_website_sentences",
                "website_sentence_status",
                "contact_website_sentence_status",
                "website_text",
                "contact_website_text",
                "website_text_status",
                "contact_website_text_status",
            ),
            {
                "polygon_id": "string",
                "source_pbf": "string",
                "osm_type": "string",
                "osm_id": "int64",
                "website_sentences": "list<string>",
                "contact_website_sentences": "list<string>",
                "website_sentence_status": "string",
                "contact_website_sentence_status": "string",
                "website_text": "large_string",
                "contact_website_text": "large_string",
                "website_text_status": "string",
                "contact_website_text_status": "string",
            },
        )
    if wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki joins require corpus='wikipedia' or corpus='wikivoyage'")
    prefix = f"{wiki_corpus}/sentences/"
    return (
        f"{wiki_corpus}_text",
        prefix,
        (
            "sentence_id",
            "document_id",
            "project",
            "language",
            "page_id",
            "revision_id",
            "text",
            "content_hash",
            "segmentation_status",
        ),
        {
            "sentence_id": "string",
            "document_id": "string",
            "project": "string",
            "language": "string",
            "page_id": "int64",
            "revision_id": "int64",
            "text": "string",
            "content_hash": "string",
            "segmentation_status": "string",
        },
    )


def _relevance_table_contract(
    source: TextSource,
    wiki_corpus: str | None,
) -> tuple[str, str, tuple[str, ...], dict[str, str]]:
    if source is TextSource.DESCRIPTION:
        return (
            "relevance_labels",
            "labels/language-v1/data/",
            (
                "label_id",
                "description_identity",
                "tag_key",
                "sentence_index",
                "text_sha256",
                "decision",
                "language",
            ),
            {
                "label_id": "string",
                "description_identity": "string",
                "tag_key": "string",
                "sentence_index": "int32",
                "text_sha256": "string",
                "decision": "string",
                "language": "string",
            },
        )
    if source is TextSource.WEBSITE:
        return (
            "relevance_labels",
            "labels/polygons/",
            (
                "label_id",
                "polygon_id",
                "field",
                "sentence_index",
                "text_sha256",
                "decision",
            ),
            {
                "label_id": "string",
                "polygon_id": "string",
                "field": "string",
                "sentence_index": "int32",
                "text_sha256": "string",
                "decision": "string",
            },
        )
    if wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki joins require corpus='wikipedia' or corpus='wikivoyage'")
    return (
        f"{wiki_corpus}_relevance_labels",
        f"labels/{wiki_corpus}/sentences/",
        ("label_id", "sentence_id", "text_sha256", "decision"),
        {
            "label_id": "string",
            "sentence_id": "string",
            "text_sha256": "string",
            "decision": "string",
        },
    )


def _validate_arrow_types(schema: Any, requirements: Mapping[str, str], path: str) -> None:
    for name, expected in requirements.items():
        _validate_arrow_field_type(schema, name, expected, path)


def _validate_arrow_field_type(schema: Any, name: str, expected: str, path: str) -> None:
    if name not in schema.names:
        raise ValueError(f"{path} lacks pinned schema field {name}")
    actual = schema.field(name).type
    valid = _arrow_type_matches(actual, expected)
    if valid is None:
        raise ValueError(f"unknown schema type contract {expected!r} for {name}")
    if not valid:
        raise TypeError(f"{path} field {name} has {actual}, expected {expected}")


def _arrow_type_matches(actual: Any, expected: str) -> bool | None:
    return {
        "string": pa.types.is_string(actual),
        "int32": pa.types.is_int32(actual),
        "int64": pa.types.is_int64(actual),
        "list<string>": pa.types.is_list(actual) and pa.types.is_string(actual.value_type),
        "large_string": pa.types.is_large_string(actual),
    }.get(expected)


def iter_eunis_join_batches(
    records: Iterable[JoinedTextRecord],
    *,
    starting_batch_index: int,
    max_batch_rows: int,
    max_batch_bytes: int,
) -> Iterator[PoolBatch]:
    """Pack joined document records into consecutive bounded core batches."""

    _validate_join_batch_limits(starting_batch_index, max_batch_rows, max_batch_bytes)
    batches = _JoinBatchAccumulator(starting_batch_index, max_batch_rows, max_batch_bytes)
    for record in records:
        batch = batches.append(record)
        if batch is not None:
            yield batch
    batch = batches.flush()
    if batch is not None:
        yield batch


def _validate_join_batch_limits(
    starting_batch_index: int,
    max_batch_rows: int,
    max_batch_bytes: int,
) -> None:
    if starting_batch_index < 0:
        raise ValueError("batch index and limits must be positive")
    if max_batch_rows <= 0 or max_batch_bytes <= 0:
        raise ValueError("batch index and limits must be positive")


def _joined_record_row_count(record: JoinedTextRecord) -> int:
    return _BATCH_ROW_OVERHEAD + sum(
        1 + len(link.candidate_inventory_row_ids) for link in record.links
    )


def _initialize_join_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE inventory (
            native_key_json TEXT PRIMARY KEY,
            inventory_row_id TEXT NOT NULL,
            polygon_snapshot_id TEXT
        );
        CREATE TABLE relevance_rows (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            scope TEXT NOT NULL,
            join_key TEXT,
            row_json TEXT NOT NULL,
            consumed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX relevance_lookup ON relevance_rows(scope, join_key);
        CREATE TABLE wiki_links (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL,
            project TEXT NOT NULL,
            native_key_json TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            complete_key INTEGER NOT NULL,
            consumed INTEGER NOT NULL DEFAULT 0
        );
        CREATE INDEX wiki_link_lookup ON wiki_links(project, document_id);
        CREATE TABLE text_docs (
            sequence INTEGER PRIMARY KEY AUTOINCREMENT,
            document_id TEXT NOT NULL UNIQUE,
            native_key_json TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            content_sha256 TEXT,
            relevance_evidence_sha256 TEXT,
            body_status TEXT NOT NULL,
            label_scope TEXT NOT NULL,
            label_join_key TEXT,
            direct_link_key TEXT,
            wiki_document_id TEXT,
            wiki_project TEXT,
            source_locator TEXT NOT NULL,
            corpus TEXT,
            missing_reason TEXT
        );
        CREATE INDEX text_label_lookup ON text_docs(label_scope, label_join_key);
        """
    )


def _stage_inventory(
    connection: sqlite3.Connection,
    inventory_rows: Iterable[PolygonInventoryRow],
) -> None:
    for row in inventory_rows:
        try:
            connection.execute(
                "INSERT INTO inventory VALUES (?, ?, ?)",
                (
                    native_key_json(row.native_key),
                    row.inventory_row_id,
                    row.polygon_snapshot_id,
                ),
            )
        except sqlite3.IntegrityError as error:
            raise ValueError("raw inventory contains a duplicate native key") from error
    connection.commit()


def _stage_relevance(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    text_source: TextSource,
    wiki_corpus: str | None,
    max_output_row_bytes: int,
) -> None:
    scope = wiki_corpus or text_source.value
    contract = _relevance_row_contract(text_source)
    for source_row in rows:
        value = source_row.values
        _require_fields(source_row, contract.role, contract.required_fields)
        key_fields = tuple(value.get(field) for field in contract.key_fields)
        join_key = _join_key(key_fields)
        # Persist only the relevance schema's metadata fields. In particular,
        # never copy a sentence or description body into the SQLite spool.
        metadata = {
            key: value.get(key)
            for key in (
                "label_id",
                "description_identity",
                "tag_key",
                "sentence_index",
                "text_sha256",
                "decision",
                "language",
                "polygon_id",
                "field",
                "sentence_id",
                "parse_mode",
                "failure_reason",
                "generation_id",
                "config_fingerprint",
                "input_revision",
            )
        }
        row_json = _canonical_json(metadata | {"_locator": source_row.locator})
        _require_utf8_size(
            row_json,
            max_output_row_bytes,
            "relevance row metadata exceeds the maximum output row size",
        )
        connection.execute(
            "INSERT INTO relevance_rows(scope, join_key, row_json) VALUES (?, ?, ?)",
            (scope, join_key, row_json),
        )
    connection.commit()


def _relevance_row_contract(text_source: TextSource) -> _RelevanceRowContract:
    if text_source is TextSource.DESCRIPTION:
        return _RelevanceRowContract(
            "Description relevance",
            (
                "label_id",
                "description_identity",
                "tag_key",
                "sentence_index",
                "text_sha256",
                "decision",
                "language",
            ),
            (
                "description_identity",
                "tag_key",
                "language",
                "sentence_index",
                "text_sha256",
            ),
        )
    if text_source is TextSource.WEBSITE:
        return _RelevanceRowContract(
            "Website relevance",
            (
                "label_id",
                "polygon_id",
                "field",
                "sentence_index",
                "text_sha256",
                "decision",
            ),
            ("polygon_id", "field", "sentence_index", "text_sha256"),
        )
    return _RelevanceRowContract(
        "Wiki relevance",
        ("label_id", "sentence_id", "text_sha256", "decision"),
        ("sentence_id", "text_sha256"),
    )


def _stage_wiki_links(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    wiki_corpus: str,
    max_output_row_bytes: int,
) -> None:
    for source_row in rows:
        link = _wiki_link_row(source_row)
        _require_utf8_size(
            link.evidence,
            max_output_row_bytes,
            "Wiki link evidence exceeds the maximum output row size",
        )
        if link.project != wiki_corpus:
            continue
        connection.execute(
            "INSERT INTO wiki_links(document_id, project, native_key_json, evidence_json, "
            "complete_key) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                link.document_id,
                link.project,
                native_key_json(link.native_key),
                link.evidence,
                int(link.complete_key),
            ),
        )
    connection.commit()


def _wiki_link_row(source_row: PinnedSourceRow) -> _WikiLinkRow:
    _require_fields(
        source_row,
        "Wiki polygon-document link",
        (
            "document_id",
            "project",
            "polygon_id",
            "source_pbf",
            "osm_type",
            "osm_id",
            "link_sources",
        ),
    )
    value = source_row.values
    document_id = _as_nonempty_string(value.get("document_id"))
    project = _as_nonempty_string(value.get("project"))
    if project not in {"wikipedia", "wikivoyage"}:
        raise ValueError(f"Wiki link row has an invalid project value at {source_row.locator}")
    selected_document_id = document_id or f"missing-document:{source_row.locator}"
    native_key, complete_key = _wiki_link_native_key(value, source_row)
    return _WikiLinkRow(
        selected_document_id,
        cast(str, project),
        native_key,
        _canonical_json(_wiki_link_evidence(source_row, value, complete_key)),
        complete_key,
    )


def _wiki_link_native_key(
    value: Mapping[str, Any],
    source_row: PinnedSourceRow,
) -> tuple[NativeKey, bool]:
    key_values = {
        field: value[field]
        for field in ("source_pbf", "osm_type", "osm_id")
        if _native_scalar(value.get(field))
    }
    return (
        canonical_native_key(key_values or {"incomplete_link_row": source_row.locator}),
        len(key_values) == _OSM_KEY_FIELD_COUNT,
    )


def _wiki_link_evidence(
    source_row: PinnedSourceRow,
    value: Mapping[str, Any],
    complete_key: bool,
) -> dict[str, Any]:
    return {
        "table": source_row.path_in_repo,
        "row": source_row.row_ordinal,
        "polygon_id": value.get("polygon_id"),
        "link_sources": _safe_metadata_value(value.get("link_sources")),
        "key_complete": complete_key,
    }


def _stage_description_rows(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    layout: EunisPoolSourceLayout,
    _mapping_rule: str,
    *,
    wiki_corpus: str | None,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    del wiki_corpus, website_field_to_column
    for source_row in rows:
        _stage_description_row(connection, source_row, layout)


def _stage_description_row(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
) -> None:
    _require_fields(
        source_row,
        "Description text",
        (
            "description_identity",
            "source_pbf",
            "osm_type",
            "osm_id",
            "tag_key",
            "original_text",
            "text_sha256",
            "language_code",
            "sentences",
        ),
    )
    metadata = _description_row_metadata(source_row)
    sentences = _description_sentence_array(source_row.values.get("sentences"))
    if not sentences:
        _insert_empty_description(connection, source_row, layout, metadata)
        return
    _insert_description_sentences(connection, source_row, layout, metadata, sentences)


def _description_row_metadata(source_row: PinnedSourceRow) -> _DescriptionRowMetadata:
    value = source_row.values
    identity = _as_nonempty_string(value.get("description_identity"))
    tag_key = _as_nonempty_string(value.get("tag_key"))
    language = _as_nonempty_string(value.get("language_code"))
    source_hash = _validated_description_source_hash(source_row)
    _validate_description_identity(source_row, identity, tag_key, source_hash)
    polygon_key = _native_key_from_fields(value, ("source_pbf", "osm_type", "osm_id"))
    return _DescriptionRowMetadata(identity, tag_key, language, source_hash, polygon_key)


def _validated_description_source_hash(source_row: PinnedSourceRow) -> str:
    source_hash = source_row.values.get("text_sha256")
    if not isinstance(source_hash, str) or not _is_sha256(source_hash):
        raise ValueError(f"Description row {source_row.locator} has no valid source text hash")
    original_text, observed_hash = _text_and_sha256(source_row.values.get("original_text"))
    if original_text is None or observed_hash != source_hash:
        raise ValueError(
            f"Description text hash does not match its pinned row at {source_row.locator}"
        )
    return source_hash


def _validate_description_identity(
    source_row: PinnedSourceRow,
    identity: str | None,
    tag_key: str | None,
    source_hash: str,
) -> None:
    if identity is None or tag_key is None:
        return
    native_identity = _description_native_identity(source_row)
    if (
        native_identity is not None
        and _description_identity(*native_identity, tag_key, source_hash) != identity
    ):
        raise ValueError(
            f"Description identity does not match its pinned row at {source_row.locator}"
        )


def _description_native_identity(source_row: PinnedSourceRow) -> tuple[str, int] | None:
    value = source_row.values
    osm_type = _as_nonempty_string(value.get("osm_type"))
    osm_id = value.get("osm_id")
    if osm_type is None or not isinstance(osm_id, int) or isinstance(osm_id, bool):
        return None
    return osm_type, osm_id


def _description_sentence_array(value: Any) -> Sequence[Any]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise TypeError("Description sentences must be an Arrow list of strings")
    return value


def _insert_empty_description(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _DescriptionRowMetadata,
) -> None:
    _store_staged_text_document(
        connection,
        _empty_description_record(source_row, layout, metadata),
    )


def _empty_description_record(
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _DescriptionRowMetadata,
) -> _StagedTextDocument:
    value = source_row.values
    split_status = value.get("split_status")
    split_reason = value.get("split_reason")
    sentence_state = _null_empty_state(value.get("sentences"))
    identity_fields = {
        "description_identity": metadata.identity or "",
        "tag_key": metadata.tag_key or "",
        "language_code": metadata.language or "",
        "source_text_sha256": metadata.source_text_sha256,
        "split_status": _string_or_missing(split_status),
        "split_reason": _string_or_missing(split_reason),
        "sentence_state": sentence_state,
        "source_row": source_row.locator,
    }
    return _StagedTextDocument(
        SourceDocument(
            document_id=_document_id(TextSource.DESCRIPTION, layout.raw_revision, identity_fields),
            native_key=_document_native_key(identity_fields, source_row.locator),
            source_revision=layout.raw_revision,
            content_sha256=metadata.source_text_sha256,
            body_status=BodyStatus.PRIVATE,
        ),
        None,
        TextSource.DESCRIPTION.value,
        None,
        _native_key_json_or_none(metadata.polygon_key),
        None,
        None,
        source_row.locator,
        None,
        _empty_description_reason(sentence_state, split_status, split_reason),
    )


def _null_empty_state(value: Any) -> str:
    return "null" if value is None else "empty"


def _string_or_missing(value: Any) -> str:
    return str(value or "missing")


def _empty_description_reason(sentence_state: str, split_status: Any, split_reason: Any) -> str:
    return (
        "pinned Description row has no sentence output; "
        f"sentence array={sentence_state}, "
        f"split status={split_status or 'missing'}, "
        f"split reason={split_reason or 'missing'}; no sentence-level "
        "relevance assessment applies"
    )


def _insert_description_sentences(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _DescriptionRowMetadata,
    sentences: Sequence[Any],
) -> None:
    for sentence_index, sentence in enumerate(sentences):
        _store_staged_text_document(
            connection,
            _description_sentence_record(
                source_row,
                layout,
                metadata,
                sentence_index,
                sentence,
            ),
        )


def _description_sentence_record(
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _DescriptionRowMetadata,
    sentence_index: int,
    sentence: Any,
) -> _StagedTextDocument:
    text, digest = _text_and_sha256(sentence)
    join_key = _join_key(
        (
            metadata.identity,
            metadata.tag_key,
            metadata.language,
            sentence_index,
            metadata.source_text_sha256,
        )
    )
    identity_fields = {
        "description_identity": metadata.identity or "",
        "tag_key": metadata.tag_key or "",
        "language_code": metadata.language or "",
        "sentence_index": sentence_index,
        "text_sha256": digest or "",
        "source_row": source_row.locator,
    }
    return _StagedTextDocument(
        SourceDocument(
            document_id=_document_id(TextSource.DESCRIPTION, layout.raw_revision, identity_fields),
            native_key=_document_native_key(identity_fields, source_row.locator),
            source_revision=layout.raw_revision,
            content_sha256=digest,
            body_status=_private_or_missing(text),
        ),
        metadata.source_text_sha256,
        TextSource.DESCRIPTION.value,
        join_key,
        _native_key_json_or_none(metadata.polygon_key),
        None,
        None,
        source_row.locator,
        None,
        None,
    )


def _private_or_missing(text: Any) -> BodyStatus:
    return BodyStatus.PRIVATE if text is not None else BodyStatus.MISSING


def _native_key_json_or_none(native_key: NativeKey | None) -> str | None:
    return native_key_json(native_key) if native_key else None


def _stage_website_rows(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    layout: EunisPoolSourceLayout,
    _mapping_rule: str,
    *,
    wiki_corpus: str | None,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    del wiki_corpus
    fields = _validated_website_fields(website_field_to_column)
    for source_row in rows:
        _require_fields(source_row, "Website polygon", _WEBSITE_ROW_FIELDS)
        metadata = _website_row_metadata(source_row)
        for label_field, column in fields:
            _stage_website_field(
                connection,
                source_row,
                layout,
                metadata,
                label_field,
                column,
            )


_WEBSITE_ROW_FIELDS = (
    "polygon_id",
    "source_pbf",
    "osm_type",
    "osm_id",
    "website_sentences",
    "contact_website_sentences",
    "website_sentence_status",
    "contact_website_sentence_status",
    "website_text",
    "contact_website_text",
    "website_text_status",
    "contact_website_text_status",
)


def _validated_website_fields(
    field_to_column: Mapping[str, str] | None,
) -> tuple[tuple[str, str], ...]:
    selected = dict(field_to_column or EUNIS_WEBSITE_FIELD_TO_SENTENCE_COLUMN)
    if selected != dict(EUNIS_WEBSITE_FIELD_TO_SENTENCE_COLUMN):
        raise ValueError("Website field mapping must match the pinned source label vocabulary")
    return tuple(sorted(selected.items()))


def _website_row_metadata(source_row: PinnedSourceRow) -> _WebsiteRowMetadata:
    value = source_row.values
    return _WebsiteRowMetadata(
        _as_nonempty_string(value.get("polygon_id")),
        _native_key_from_fields(value, ("source_pbf", "osm_type", "osm_id")),
    )


def _stage_website_field(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _WebsiteRowMetadata,
    label_field: str,
    column: str,
) -> None:
    sentences = _website_sentence_array(source_row, column)
    if not sentences:
        _insert_empty_website_field(connection, source_row, layout, metadata, label_field, column)
        return
    _insert_website_sentences(
        connection,
        source_row,
        layout,
        metadata,
        label_field,
        sentences,
    )


def _website_sentence_array(source_row: PinnedSourceRow, column: str) -> Sequence[Any]:
    sentences = source_row.values.get(column)
    if sentences is None:
        return ()
    if not isinstance(sentences, (list, tuple)):
        raise TypeError(f"Website {column} must be an Arrow list of strings")
    return sentences


def _insert_empty_website_field(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _WebsiteRowMetadata,
    label_field: str,
    column: str,
) -> None:
    _store_staged_text_document(
        connection,
        _empty_website_record(source_row, layout, metadata, label_field, column),
    )


def _empty_website_record(
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _WebsiteRowMetadata,
    label_field: str,
    column: str,
) -> _StagedTextDocument:
    value = source_row.values
    field_prefix = column.removesuffix("_sentences")
    body_text = value.get(f"{field_prefix}_text")
    body_hash = _text_and_sha256(body_text)[1]
    text_status = value.get(f"{field_prefix}_text_status")
    sentence_status = value.get(f"{field_prefix}_sentence_status")
    sentence_state = _null_empty_state(value.get(column))
    identity_fields = {
        "polygon_id": metadata.polygon_id or "",
        "field": label_field,
        "source_text_sha256": body_hash or "",
        "text_status": str(text_status or "missing"),
        "sentence_status": str(sentence_status or "missing"),
        "sentence_state": sentence_state,
        "source_row": source_row.locator,
    }
    return _StagedTextDocument(
        SourceDocument(
            document_id=_document_id(TextSource.WEBSITE, layout.raw_revision, identity_fields),
            native_key=_document_native_key(identity_fields, source_row.locator),
            source_revision=layout.raw_revision,
            content_sha256=body_hash,
            body_status=_private_or_missing(body_text),
        ),
        None,
        TextSource.WEBSITE.value,
        None,
        _native_key_json_or_none(metadata.polygon_key),
        None,
        None,
        source_row.locator,
        None,
        _empty_website_reason(sentence_state, text_status, sentence_status),
    )


def _empty_website_reason(
    sentence_state: str,
    text_status: Any,
    sentence_status: Any,
) -> str:
    return (
        "Website text field has no sentence output; "
        f"sentence array={sentence_state}, "
        f"text status={text_status or 'missing'}, "
        f"sentence status={sentence_status or 'missing'}"
    )


def _insert_website_sentences(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _WebsiteRowMetadata,
    label_field: str,
    sentences: Sequence[Any],
) -> None:
    for sentence_index, sentence in enumerate(sentences):
        _store_staged_text_document(
            connection,
            _website_sentence_record(
                source_row,
                layout,
                metadata,
                label_field,
                sentence_index,
                sentence,
            ),
        )


def _website_sentence_record(
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    metadata: _WebsiteRowMetadata,
    label_field: str,
    sentence_index: int,
    sentence: Any,
) -> _StagedTextDocument:
    text, digest = _text_and_sha256(sentence)
    join_key = _join_key((metadata.polygon_id, label_field, sentence_index, digest))
    identity_fields = {
        "polygon_id": metadata.polygon_id or "",
        "field": label_field,
        "sentence_index": sentence_index,
        "text_sha256": digest or "",
        "source_row": source_row.locator,
    }
    return _StagedTextDocument(
        SourceDocument(
            document_id=_document_id(TextSource.WEBSITE, layout.raw_revision, identity_fields),
            native_key=_document_native_key(identity_fields, source_row.locator),
            source_revision=layout.raw_revision,
            content_sha256=digest,
            body_status=_private_or_missing(text),
        ),
        digest,
        TextSource.WEBSITE.value,
        join_key,
        _native_key_json_or_none(metadata.polygon_key),
        None,
        None,
        source_row.locator,
        None,
        None,
    )


def _stage_wiki_rows(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    layout: EunisPoolSourceLayout,
    _mapping_rule: str,
    *,
    wiki_corpus: str | None,
    website_field_to_column: Mapping[str, str] | None,
) -> None:
    del website_field_to_column
    scope = _validated_wiki_corpus(wiki_corpus)
    for source_row in rows:
        _stage_wiki_row(connection, source_row, layout, scope)


def _validated_wiki_corpus(wiki_corpus: str | None) -> str:
    if wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki sentence rows need a pinned Wikipedia or Wikivoyage table")
    return cast(str, wiki_corpus)


_WIKI_ROW_FIELDS = (
    "sentence_id",
    "document_id",
    "project",
    "language",
    "page_id",
    "revision_id",
    "text",
    "content_hash",
    "segmentation_status",
)


def _stage_wiki_row(
    connection: sqlite3.Connection,
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    scope: str,
) -> None:
    _require_fields(source_row, "Wiki sentence", _WIKI_ROW_FIELDS)
    document = _wiki_sentence_document(source_row, layout, scope)
    _insert_text_doc(
        connection,
        document_id=document.document_id,
        native_key=document.native_key,
        source_revision=layout.raw_revision,
        content_sha256=document.content_sha256,
        relevance_evidence_sha256=document.content_sha256,
        body_status=document.body_status,
        label_scope=document.scope,
        label_join_key=document.join_key,
        direct_link_key=None,
        wiki_document_id=document.document_id_from_source,
        wiki_project=document.project,
        source_locator=document.source_locator,
        corpus=document.scope,
        missing_reason=document.missing_reason,
    )


def _wiki_sentence_document(
    source_row: PinnedSourceRow,
    layout: EunisPoolSourceLayout,
    scope: str,
) -> _WikiSentenceDocument:
    value = source_row.values
    text, digest = _text_and_sha256(value.get("text"))
    _validate_wiki_content_hash(source_row, text, digest)
    sentence_id = _as_nonempty_string(value.get("sentence_id"))
    document_id_raw = _as_nonempty_string(value.get("document_id"))
    project = _as_nonempty_string(value.get("project"))
    identity_fields = _wiki_sentence_identity_fields(
        source_row,
        scope,
        digest,
        sentence_id,
        document_id_raw,
        project,
    )
    return _WikiSentenceDocument(
        _document_id(TextSource.WIKI, layout.raw_revision, identity_fields),
        _document_native_key(identity_fields, source_row.locator),
        digest,
        _private_or_missing(text),
        scope,
        _join_key((sentence_id, digest)),
        document_id_raw,
        project,
        source_row.locator,
        _wiki_sentence_missing_reason(source_row, text),
    )


def _wiki_sentence_identity_fields(
    source_row: PinnedSourceRow,
    scope: str,
    digest: str | None,
    sentence_id: str | None,
    document_id: str | None,
    project: str | None,
) -> dict[str, Any]:
    value = source_row.values
    return {
        "corpus": scope,
        "sentence_id": _string_or_empty(sentence_id),
        "document_id": _string_or_empty(document_id),
        "project": _string_or_empty(project),
        "language": _string_or_empty(_as_nonempty_string(value.get("language"))),
        "page_id": _native_scalar_or_empty(value.get("page_id")),
        "revision_id": _native_scalar_or_empty(value.get("revision_id")),
        "text_sha256": _string_or_empty(digest),
        "source_row": source_row.locator,
    }


def _string_or_empty(value: str | None) -> str:
    return value or ""


def _native_scalar_or_empty(value: Any) -> str | int:
    if not _native_scalar(value):
        return ""
    return value or ""


def _validate_wiki_content_hash(
    source_row: PinnedSourceRow,
    text: str | None,
    digest: str | None,
) -> None:
    source_hash = source_row.values.get("content_hash")
    if text is not None and (not _is_sha256(source_hash) or source_hash != digest):
        raise ValueError(f"Wiki sentence content hash differs at {source_row.locator}")


def _wiki_sentence_missing_reason(
    source_row: PinnedSourceRow,
    text: str | None,
) -> str | None:
    if text is not None:
        return None
    segmentation_status = source_row.values.get("segmentation_status")
    return (
        "Wiki sentence row has no text body; "
        f"segmentation status={segmentation_status or 'missing'}"
    )


def _insert_text_doc(  # noqa: PLR0913
    connection: sqlite3.Connection,
    *,
    document_id: str,
    native_key: NativeKey,
    source_revision: str,
    content_sha256: str | None,
    body_status: BodyStatus,
    label_scope: str,
    label_join_key: str | None,
    direct_link_key: str | None,
    wiki_document_id: str | None,
    wiki_project: str | None,
    source_locator: str,
    corpus: str | None,
    relevance_evidence_sha256: str | None = None,
    missing_reason: str | None = None,
) -> None:
    connection.execute(
        "INSERT INTO text_docs(document_id, native_key_json, source_revision, content_sha256, "
        "relevance_evidence_sha256, "
        "body_status, label_scope, label_join_key, direct_link_key, wiki_document_id, "
        "wiki_project, source_locator, corpus, missing_reason) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            document_id,
            native_key_json(native_key),
            source_revision,
            content_sha256,
            relevance_evidence_sha256,
            body_status.value,
            label_scope,
            label_join_key,
            direct_link_key,
            wiki_document_id,
            wiki_project,
            source_locator,
            corpus,
            missing_reason,
        ),
    )


def _store_staged_text_document(
    connection: sqlite3.Connection,
    staged: _StagedTextDocument,
) -> None:
    document = staged.document
    _insert_text_doc(
        connection,
        document_id=document.document_id,
        native_key=document.native_key,
        source_revision=document.source_revision,
        content_sha256=document.content_sha256,
        relevance_evidence_sha256=staged.relevance_evidence_sha256,
        body_status=document.body_status,
        label_scope=staged.label_scope,
        label_join_key=staged.label_join_key,
        direct_link_key=staged.direct_link_key,
        wiki_document_id=staged.wiki_document_id,
        wiki_project=staged.wiki_project,
        source_locator=staged.source_locator,
        corpus=staged.corpus,
        missing_reason=staged.missing_reason,
    )


def _emit_join_records(
    connection: sqlite3.Connection,
    text_source: TextSource,
    layout: EunisPoolSourceLayout,
    mapping_rule: str,
    wiki_corpus: str | None,
    allocation_bounds: JoinAllocationBounds,
) -> Iterator[JoinedTextRecord]:
    relevance_repository = layout.relevance_repository
    relevance_revision = layout.relevance_revision
    release_id = f"{relevance_repository}@{relevance_revision}"
    _stage_duplicate_document_relevance_keys(connection)
    context = _JoinEmissionContext(
        text_source,
        layout,
        mapping_rule,
        wiki_corpus,
        release_id,
        relevance_repository,
        relevance_revision,
        allocation_bounds,
    )
    for record in _emit_staged_text_documents(connection, context):
        _validate_joined_record_limits(record, allocation_bounds)
        yield record
    for record in _emit_remaining_join_records(connection, context):
        _validate_joined_record_limits(record, allocation_bounds)
        yield record


def _stage_duplicate_document_relevance_keys(connection: sqlite3.Connection) -> None:
    connection.execute(
        "CREATE TEMP TABLE duplicate_text_keys ("
        "label_scope TEXT NOT NULL, label_join_key TEXT NOT NULL, "
        "PRIMARY KEY(label_scope, label_join_key)) WITHOUT ROWID"
    )
    connection.execute(
        "INSERT INTO duplicate_text_keys(label_scope, label_join_key) "
        "SELECT label_scope, label_join_key FROM text_docs "
        "WHERE label_join_key IS NOT NULL "
        "GROUP BY label_scope, label_join_key HAVING COUNT(*) > 1"
    )
    connection.commit()


def _emit_staged_text_documents(
    connection: sqlite3.Connection,
    context: _JoinEmissionContext,
) -> Iterator[JoinedTextRecord]:
    query = (
        "SELECT document_id, native_key_json, source_revision, content_sha256, "
        "relevance_evidence_sha256, body_status, label_scope, label_join_key, "
        "direct_link_key, wiki_document_id, wiki_project, source_locator, corpus, missing_reason "
        "FROM text_docs ORDER BY sequence"
    )
    for row in connection.execute(query):
        yield _joined_record_from_staged_row(
            connection,
            _decode_staged_text_document(row),
            context,
        )


def _decode_staged_text_document(row: Sequence[Any]) -> _StagedTextDocument:
    (
        document_id,
        native_key_payload,
        source_revision,
        content_sha256,
        relevance_evidence_sha256,
        body_status,
        label_scope,
        label_join_key,
        direct_link_key,
        wiki_document_id,
        wiki_project,
        source_locator,
        corpus,
        missing_reason,
    ) = row
    return _StagedTextDocument(
        SourceDocument(
            document_id=str(document_id),
            native_key=_native_key_from_json(str(native_key_payload)),
            source_revision=str(source_revision),
            content_sha256=_truthy_string_or_none(content_sha256),
            body_status=BodyStatus(str(body_status)),
        ),
        _truthy_string_or_none(relevance_evidence_sha256),
        str(label_scope),
        _present_string_or_none(label_join_key),
        _truthy_string_or_none(direct_link_key),
        _truthy_string_or_none(wiki_document_id),
        _truthy_string_or_none(wiki_project),
        str(source_locator),
        _truthy_string_or_none(corpus),
        _truthy_string_or_none(missing_reason),
    )


def _truthy_string_or_none(value: Any) -> str | None:
    return str(value) if value else None


def _present_string_or_none(value: Any) -> str | None:
    return str(value) if value is not None else None


def _joined_record_from_staged_row(
    connection: sqlite3.Connection,
    staged: _StagedTextDocument,
    context: _JoinEmissionContext,
) -> JoinedTextRecord:
    relevance = _resolve_relevance(
        connection,
        document_id=staged.document.document_id,
        join_key=staged.label_join_key,
        scope=staged.label_scope,
        duplicate_doc_key=(
            staged.label_join_key is not None
            and _has_duplicate_document_relevance_key(
                connection,
                staged.label_scope,
                staged.label_join_key,
            )
        ),
        source_repository=context.relevance_repository,
        source_revision=context.relevance_revision,
        release_id=context.release_id,
        mapping_rule=context.mapping_rule,
        evidence_sha256=staged.relevance_evidence_sha256,
        source_locator=staged.source_locator,
        missing_reason=staged.missing_reason,
        allocation_bounds=context.allocation_bounds,
    )
    links = _resolve_links(
        connection,
        context.text_source,
        staged.document,
        direct_link_key=staged.direct_link_key,
        wiki_document_id=staged.wiki_document_id,
        wiki_project=staged.wiki_project,
        source_revision=context.layout.raw_revision,
        source_locator=staged.source_locator,
        corpus=staged.corpus or context.wiki_corpus,
        relevance=relevance,
        allocation_bounds=context.allocation_bounds,
    )
    return JoinedTextRecord(staged.document, links, relevance)


def _has_duplicate_document_relevance_key(
    connection: sqlite3.Connection,
    scope: str,
    join_key: str,
) -> bool:
    return (
        connection.execute(
            "SELECT 1 FROM duplicate_text_keys WHERE label_scope=? AND label_join_key=?",
            (scope, join_key),
        ).fetchone()
        is not None
    )


def _emit_remaining_join_records(
    connection: sqlite3.Connection,
    context: _JoinEmissionContext,
) -> Iterator[JoinedTextRecord]:
    yield from _emit_orphan_label_records(
        connection,
        context.text_source,
        context.layout,
        context.release_id,
        context.mapping_rule,
        context.wiki_corpus,
    )
    if context.text_source is TextSource.WIKI:
        yield from _emit_orphan_wiki_link_records(
            connection, context.layout, context.release_id, context.mapping_rule
        )
    yield from _emit_unrepresented_inventory_records(
        connection,
        context.text_source,
        context.layout,
        context.release_id,
        context.mapping_rule,
        context.wiki_corpus,
    )


def _resolve_relevance(  # noqa: PLR0913
    connection: sqlite3.Connection,
    *,
    document_id: str,
    join_key: str | None,
    scope: str,
    duplicate_doc_key: bool,
    source_repository: str,
    source_revision: str,
    release_id: str,
    mapping_rule: str,
    evidence_sha256: str | None,
    source_locator: str,
    allocation_bounds: JoinAllocationBounds,
    missing_reason: str | None = None,
) -> RelevanceAssessment:
    match_count, first_decision = _matching_relevance_rows(connection, scope, join_key)
    evidence = _relevance_evidence(
        connection,
        source_locator,
        scope,
        join_key,
        allocation_bounds.max_output_row_bytes,
    )
    resolution = _relevance_cardinality(match_count, duplicate_doc_key, missing_reason)
    if resolution is None:
        resolution = _single_relevance_decision(first_decision)
    return RelevanceAssessment(
        document_id=document_id,
        release_id=release_id,
        status=resolution.status,
        source_repository=source_repository,
        source_revision=source_revision,
        mapping_rule_version=mapping_rule,
        evidence=evidence,
        raw_value=resolution.raw_value,
        evidence_sha256=evidence_sha256,
        reason=resolution.reason,
    )


def _matching_relevance_rows(
    connection: sqlite3.Connection,
    scope: str,
    join_key: str | None,
) -> tuple[int, Any]:
    if join_key is None:
        return 0, None
    count = int(
        connection.execute(
            "SELECT COUNT(*) FROM relevance_rows WHERE scope=? AND join_key=?",
            (scope, join_key),
        ).fetchone()[0]
    )
    if count == 0:
        return 0, None
    first = connection.execute(
        "SELECT row_json FROM relevance_rows "
        "WHERE scope=? AND join_key=? ORDER BY sequence LIMIT 1",
        (scope, join_key),
    ).fetchone()
    connection.execute(
        "UPDATE relevance_rows SET consumed=1 WHERE scope=? AND join_key=?",
        (scope, join_key),
    )
    return count, json.loads(str(first[0])).get("decision")


def _relevance_evidence(
    connection: sqlite3.Connection,
    source_locator: str,
    scope: str,
    join_key: str | None,
    max_output_row_bytes: int,
) -> str:
    digest = hashlib.sha256((join_key or "missing").encode()).hexdigest()
    with _BoundedTextBuilder(
        max_output_row_bytes,
        "relevance evidence exceeds the maximum output row size",
    ) as output:
        output.append('{"join_key_sha256":')
        output.append(_canonical_json(digest))
        output.append(',"label_rows":[')
        if join_key is not None:
            for index, (payload,) in enumerate(
                connection.execute(
                    "SELECT row_json FROM relevance_rows "
                    "WHERE scope=? AND join_key=? ORDER BY sequence",
                    (scope, join_key),
                )
            ):
                if index:
                    output.append(",")
                row = json.loads(str(payload))
                output.append(
                    _canonical_json(
                        {"label_id": row.get("label_id"), "locator": row.get("_locator")}
                    )
                )
        output.append('],"text_row":')
        output.append(_canonical_json(source_locator))
        output.append("}")
        return output.value()


def _relevance_cardinality(
    match_count: int,
    duplicate_doc_key: bool,
    missing_reason: str | None,
) -> _RelevanceResolution | None:
    if missing_reason is not None:
        return _RelevanceResolution(RelevanceStatus.MISSING, None, missing_reason)
    if duplicate_doc_key:
        return _RelevanceResolution(
            RelevanceStatus.UNKNOWN,
            None,
            "multiple source text rows share one relevance key",
        )
    if match_count > 1:
        return _RelevanceResolution(
            RelevanceStatus.UNKNOWN,
            None,
            "multiple relevance rows match the source text key",
        )
    if match_count == 0:
        return _RelevanceResolution(
            RelevanceStatus.MISSING,
            None,
            "no relevance row matches the source-native key and text hash",
        )
    return None


def _single_relevance_decision(raw: Any) -> _RelevanceResolution:
    normalized = raw.strip().casefold() if isinstance(raw, str) else ""
    if normalized == "yes":
        return _RelevanceResolution(RelevanceStatus.YES, str(raw), None)
    if normalized == "no":
        return _RelevanceResolution(RelevanceStatus.NO, str(raw), None)
    return _RelevanceResolution(
        RelevanceStatus.UNKNOWN,
        None,
        _invalid_relevance_decision_reason(raw),
    )


def _invalid_relevance_decision_reason(raw: Any) -> str:
    if _empty_relevance_decision(raw):
        return "relevance row has an empty decision"
    return "relevance decision is outside the pinned YES/NO vocabulary"


def _empty_relevance_decision(raw: Any) -> bool:
    if raw is None:
        return True
    return isinstance(raw, str) and not raw.strip()


def _resolve_links(  # noqa: PLR0913
    connection: sqlite3.Connection,
    text_source: TextSource,
    document: SourceDocument,
    *,
    direct_link_key: str | None,
    wiki_document_id: str | None,
    wiki_project: str | None,
    source_revision: str,
    source_locator: str,
    corpus: str | None,
    relevance: RelevanceAssessment,
    allocation_bounds: JoinAllocationBounds,
) -> tuple[TextPolygonLink, ...]:
    record_context = _LinkRecordContext(
        document,
        source_revision,
        relevance,
        allocation_bounds,
    )
    if text_source is TextSource.WIKI:
        return _resolve_wiki_links(
            connection,
            record_context,
            wiki_document_id,
            wiki_project,
            corpus,
        )
    return _resolve_direct_link(
        connection,
        record_context,
        direct_link_key,
        source_locator,
    )


def _resolve_direct_link(
    connection: sqlite3.Connection,
    context: _LinkRecordContext,
    direct_link_key: str | None,
    source_locator: str,
) -> tuple[TextPolygonLink, ...]:
    document = context.document
    source_revision = context.source_revision
    allocation_bounds = context.allocation_bounds
    links: list[TextPolygonLink] = []
    size, rows = _record_base_size(document, context.relevance, allocation_bounds)
    if direct_link_key is None:
        link = _missing_link(document, source_revision, "source polygon key is incomplete")
    else:
        link = _resolve_one_link(
            connection,
            document.document_id,
            _native_key_from_json(direct_link_key),
            source_revision,
            f"direct source-native OSM key from {source_locator}",
            True,
        )
    _append_bounded_link(links, link, size, rows, allocation_bounds)
    return tuple(links)


def _resolve_wiki_links(
    connection: sqlite3.Connection,
    context: _LinkRecordContext,
    wiki_document_id: str | None,
    wiki_project: str | None,
    corpus: str | None,
) -> tuple[TextPolygonLink, ...]:
    document = context.document
    source_revision = context.source_revision
    allocation_bounds = context.allocation_bounds
    links: list[TextPolygonLink] = []
    size, rows = _record_base_size(document, context.relevance, allocation_bounds)
    if not wiki_document_id or not wiki_project:
        link = _missing_link(
            document,
            source_revision,
            "Wiki sentence has no document/project key",
        )
        _append_bounded_link(links, link, size, rows, allocation_bounds)
        return tuple(links)
    match_count = int(
        connection.execute(
            "SELECT COUNT(*) FROM wiki_links WHERE document_id=? AND project=?",
            (wiki_document_id, wiki_project),
        ).fetchone()[0]
    )
    if match_count == 0:
        link = _missing_link(
            document,
            source_revision,
            f"no polygon_document_links row for {corpus}/{wiki_document_id}/{wiki_project}",
        )
        _append_bounded_link(links, link, size, rows, allocation_bounds)
        return tuple(links)
    connection.execute(
        "UPDATE wiki_links SET consumed=1 WHERE document_id=? AND project=?",
        (wiki_document_id, wiki_project),
    )
    groups = connection.execute(
        "SELECT native_key_json, MIN(complete_key), COUNT(*) FROM wiki_links "
        "WHERE document_id=? AND project=? GROUP BY native_key_json "
        "ORDER BY native_key_json",
        (wiki_document_id, wiki_project),
    )
    for native_key_payload, complete_value, evidence_count in groups:
        evidence_payload = _combined_wiki_link_evidence(
            connection,
            wiki_document_id,
            wiki_project,
            str(native_key_payload),
            int(evidence_count),
            allocation_bounds.max_output_row_bytes,
        )
        link = _resolve_one_link(
            connection,
            document.document_id,
            _native_key_from_json(str(native_key_payload)),
            source_revision,
            f"pinned polygon_document_links evidence {evidence_payload}",
            bool(complete_value),
        )
        size, rows = _append_bounded_link(
            links,
            link,
            size,
            rows,
            allocation_bounds,
        )
    return tuple(links)


def _combined_wiki_link_evidence(
    connection: sqlite3.Connection,
    document_id: str,
    project: str,
    native_key_payload: str,
    evidence_count: int,
    max_output_row_bytes: int,
) -> str:
    with _BoundedTextBuilder(
        max_output_row_bytes,
        "combined Wiki link evidence exceeds the maximum output row size",
    ) as output:
        output.append_repeated("[", evidence_count - 1)
        for index, (evidence_payload,) in enumerate(
            connection.execute(
                "SELECT evidence_json FROM wiki_links "
                "WHERE document_id=? AND project=? AND native_key_json=? ORDER BY sequence",
                (document_id, project, native_key_payload),
            )
        ):
            if index:
                output.append(",")
            output.append(str(evidence_payload))
            if index:
                output.append("]")
        return output.value()


def _record_base_size(
    document: SourceDocument,
    relevance: RelevanceAssessment,
    bounds: JoinAllocationBounds,
) -> tuple[int, int]:
    empty_record = JoinedTextRecord(document, (), relevance)
    size = _record_size(empty_record)
    rows = _BATCH_ROW_OVERHEAD
    _require_output_row_size(
        _source_document_row_size(document),
        bounds,
        "source document exceeds the maximum output row size",
    )
    _require_output_row_size(
        _relevance_assessment_row_size(relevance),
        bounds,
        "relevance assessment exceeds the maximum output row size",
    )
    if rows > bounds.max_record_rows:
        raise ValueError("joined record exceeds the maximum output row count")
    if size > bounds.max_record_bytes:
        raise ValueError("joined record metadata exceeds the maximum record size")
    return size, rows


def _append_bounded_link(
    links: list[TextPolygonLink],
    link: TextPolygonLink,
    current_size: int,
    current_rows: int,
    bounds: JoinAllocationBounds,
) -> tuple[int, int]:
    link_size = _text_polygon_link_row_size(link)
    _require_output_row_size(
        link_size,
        bounds,
        "joined link exceeds the maximum output row size",
    )
    next_size = current_size + link_size + int(bool(links))
    next_rows = current_rows + 1 + len(link.candidate_inventory_row_ids)
    if next_rows > bounds.max_record_rows:
        raise ValueError("joined record exceeds the maximum output row count")
    if next_size > bounds.max_record_bytes:
        raise ValueError("joined record exceeds the maximum record size")
    links.append(link)
    return next_size, next_rows


def _validate_joined_record_limits(
    record: JoinedTextRecord,
    bounds: JoinAllocationBounds,
) -> None:
    if _joined_record_row_count(record) > bounds.max_record_rows:
        raise ValueError("joined record exceeds the maximum output row count")
    if _record_size(record) > bounds.max_record_bytes:
        raise ValueError("joined record exceeds the maximum record size")
    _require_output_row_size(
        _source_document_row_size(record.document),
        bounds,
        "source document exceeds the maximum output row size",
    )
    _require_output_row_size(
        _relevance_assessment_row_size(record.relevance),
        bounds,
        "relevance assessment exceeds the maximum output row size",
    )
    for link in record.links:
        _require_output_row_size(
            _text_polygon_link_row_size(link),
            bounds,
            "joined link exceeds the maximum output row size",
        )


def _require_output_row_size(
    row_size: int,
    bounds: JoinAllocationBounds,
    message: str,
) -> None:
    if row_size > bounds.max_output_row_bytes:
        raise ValueError(message)


def _source_document_row_size(document: SourceDocument) -> int:
    return len(
        _canonical_json(
            {
                "document_id": document.document_id,
                "native_key_json": native_key_json(document.native_key),
                "source_revision": document.source_revision,
                "content_sha256": document.content_sha256,
                "body_status": document.body_status.value,
            }
        ).encode("utf-8")
    )


def _relevance_assessment_row_size(relevance: RelevanceAssessment) -> int:
    return len(
        _canonical_json(
            {
                "document_id": relevance.document_id,
                "release_id": relevance.release_id,
                "status": relevance.status.value,
                "source_repository": relevance.source_repository,
                "source_revision": relevance.source_revision,
                "mapping_rule_version": relevance.mapping_rule_version,
                "evidence": relevance.evidence,
                "raw_value": relevance.raw_value,
                "evidence_sha256": relevance.evidence_sha256,
                "reason": relevance.reason,
            }
        ).encode("utf-8")
    )


def _text_polygon_link_row_size(link: TextPolygonLink) -> int:
    return len(
        _canonical_json(
            {
                "document_id": link.document_id,
                "inventory_row_id": link.inventory_row_id,
                "native_link_key_json": native_key_json(link.native_link_key),
                "source_revision": link.source_revision,
                "status": link.status.value,
                "evidence": link.evidence,
                "candidate_inventory_row_ids_json": _canonical_json(
                    sorted(link.candidate_inventory_row_ids)
                ),
            }
        ).encode("utf-8")
    )


def _resolve_one_link(
    connection: sqlite3.Connection,
    document_id: str,
    native_key: NativeKey,
    source_revision: str,
    evidence: str,
    complete_key: bool,
) -> TextPolygonLink:
    match = connection.execute(
        "SELECT inventory_row_id, polygon_snapshot_id FROM inventory WHERE native_key_json=?",
        (native_key_json(native_key),),
    ).fetchone()
    if not complete_key:
        status = TextLinkStatus.UNMATCHED
        inventory_row_id = None
        candidates = ()
        evidence = f"{evidence}; incomplete source-native key"
    elif match is None:
        status = TextLinkStatus.UNMATCHED
        inventory_row_id = None
        candidates = ()
    elif match[1] is None:
        status = TextLinkStatus.GEOMETRY_MISSING
        inventory_row_id = None
        candidates = (str(match[0]),)
    else:
        status = TextLinkStatus.LINKED
        inventory_row_id = str(match[0])
        candidates = ()
    return TextPolygonLink(
        document_id=document_id,
        inventory_row_id=inventory_row_id,
        native_link_key=native_key,
        source_revision=source_revision,
        status=status,
        evidence=evidence,
        candidate_inventory_row_ids=candidates,
    )


def _missing_link(
    document: SourceDocument,
    source_revision: str,
    reason: str,
) -> TextPolygonLink:
    """Record absence with an explicit sentinel, never a fabricated OSM key."""

    return TextPolygonLink(
        document_id=document.document_id,
        inventory_row_id=None,
        native_link_key=canonical_native_key(
            {
                "source_link_status": _MISSING_LINK_PREFIX,
                "source_document_id": document.document_id,
            }
        ),
        source_revision=source_revision,
        status=TextLinkStatus.UNMATCHED,
        evidence=reason,
    )


def _emit_orphan_label_records(
    connection: sqlite3.Connection,
    text_source: TextSource,
    layout: EunisPoolSourceLayout,
    release_id: str,
    mapping_rule: str,
    wiki_corpus: str | None,
) -> Iterator[JoinedTextRecord]:
    context = _OrphanLabelContext(text_source, layout, release_id, mapping_rule, wiki_corpus)
    for row in connection.execute(
        "SELECT sequence, scope, join_key, row_json FROM relevance_rows "
        "WHERE consumed=0 ORDER BY sequence"
    ):
        yield _orphan_label_record(row, context)


def _orphan_label_record(
    row: Sequence[Any],
    context: _OrphanLabelContext,
) -> JoinedTextRecord:
    sequence, scope, join_key, payload_json = row
    payload = json.loads(payload_json)
    locator = _orphan_label_locator(payload)
    label_id = _orphan_label_id(payload, sequence)
    digest = _orphan_label_join_digest(join_key)
    identity = _orphan_label_identity(scope, label_id, locator, digest)
    document_id = _document_id(
        context.text_source,
        context.layout.raw_revision,
        identity.fields(context.text_source),
    )
    document = _orphan_label_document(context, identity, document_id, payload)
    relevance = _orphan_label_assessment(context, identity, document, payload)
    link = _missing_link(
        document,
        context.layout.raw_revision,
        "relevance row has no matching source text or polygon link",
    )
    return JoinedTextRecord(document, (link,), relevance)


def _orphan_label_locator(payload: Mapping[str, Any]) -> str:
    return str(payload.get("_locator") or "unknown-label-row")


def _orphan_label_id(payload: Mapping[str, Any], sequence: int) -> str:
    return _as_nonempty_string(payload.get("label_id")) or str(sequence)


def _orphan_label_join_digest(join_key: Any) -> str:
    return hashlib.sha256(str(join_key or "").encode()).hexdigest()


def _orphan_label_identity(
    scope: Any,
    label_id: str,
    locator: str,
    digest: str,
) -> _OrphanLabelIdentity:
    return _OrphanLabelIdentity(str(scope), label_id, locator, digest)


def _orphan_label_document(
    context: _OrphanLabelContext,
    identity: _OrphanLabelIdentity,
    document_id: str,
    payload: Mapping[str, Any],
) -> SourceDocument:
    native_key = _document_native_key(
        {"label_id": identity.label_id, "label_row": identity.locator},
        identity.locator,
    )
    return SourceDocument(
        document_id=document_id,
        native_key=native_key,
        source_revision=context.layout.raw_revision,
        content_sha256=_metadata_sha_or_none(payload.get("text_sha256")),
        body_status=BodyStatus.MISSING,
    )


def _orphan_label_assessment(
    context: _OrphanLabelContext,
    identity: _OrphanLabelIdentity,
    document: SourceDocument,
    payload: Mapping[str, Any],
) -> RelevanceAssessment:
    status, raw_value, reason = _orphan_label_decision(payload.get("decision"))
    return RelevanceAssessment(
        document_id=document.document_id,
        release_id=context.release_id,
        status=status,
        source_repository=context.layout.relevance_repository,
        source_revision=context.layout.relevance_revision,
        mapping_rule_version=context.mapping_rule,
        evidence=_canonical_json(
            {
                "orphan_label_row": identity.locator,
                "label_id": identity.label_id,
                "join_key_sha256": identity.join_key_sha256,
                "corpus": context.wiki_corpus,
            }
        ),
        raw_value=raw_value,
        evidence_sha256=document.content_sha256,
        reason=reason,
    )


def _orphan_label_decision(raw: Any) -> tuple[RelevanceStatus, str | None, str]:
    normalized = raw.strip().casefold() if isinstance(raw, str) else ""
    if normalized in {"yes", "no"}:
        return (
            RelevanceStatus.MISSING,
            str(raw),
            "orphan relevance row preserves its source decision but has no matching "
            "source text row",
        )
    return (
        RelevanceStatus.UNKNOWN,
        None,
        "orphan relevance row has no matching source text and no valid YES/NO decision",
    )


def _metadata_sha_or_none(value: Any) -> str | None:
    return str(value) if _is_sha256(value) else None


def _emit_orphan_wiki_link_records(
    connection: sqlite3.Connection,
    layout: EunisPoolSourceLayout,
    release_id: str,
    mapping_rule: str,
) -> Iterator[JoinedTextRecord]:
    """Retain link-table rows whose document has no sentence-table row."""

    for (
        sequence,
        document_id_raw,
        project,
        key_payload,
        evidence_payload,
        complete,
    ) in connection.execute(
        "SELECT sequence, document_id, project, native_key_json, evidence_json, complete_key "
        "FROM wiki_links WHERE consumed=0 ORDER BY sequence"
    ):
        evidence = json.loads(evidence_payload)
        table = evidence.get("table", "polygon_document_links")
        row = evidence.get("row", sequence)
        locator = f"{table}#row={row}"
        document_id = _document_id(
            TextSource.WIKI,
            layout.raw_revision,
            {
                "orphan_wiki_link_row": locator,
                "document_id": document_id_raw,
                "project": project,
            },
        )
        document = SourceDocument(
            document_id=document_id,
            native_key=canonical_native_key(
                {
                    "corpus": "wiki",
                    "document_id": str(document_id_raw),
                    "project": str(project),
                    "source_link_row": locator,
                }
            ),
            source_revision=layout.raw_revision,
            content_sha256=None,
            body_status=BodyStatus.MISSING,
        )
        link = _resolve_one_link(
            connection,
            document_id,
            _native_key_from_json(str(key_payload)),
            layout.raw_revision,
            f"pinned polygon_document_links row without a sentence record: {locator}; "
            f"link_sources={_canonical_json(evidence.get('link_sources'))}",
            bool(complete),
        )
        relevance = RelevanceAssessment(
            document_id=document_id,
            release_id=release_id,
            status=RelevanceStatus.MISSING,
            source_repository=layout.relevance_repository,
            source_revision=layout.relevance_revision,
            mapping_rule_version=mapping_rule,
            evidence=_canonical_json(
                {
                    "document_id": document_id_raw,
                    "project": project,
                    "link_row": locator,
                }
            ),
            reason="Wiki polygon-document link has no matching sentence row",
        )
        yield JoinedTextRecord(document, (link,), relevance)


def _emit_unrepresented_inventory_records(
    connection: sqlite3.Connection,
    text_source: TextSource,
    layout: EunisPoolSourceLayout,
    release_id: str,
    mapping_rule: str,
    wiki_corpus: str | None,
) -> Iterator[JoinedTextRecord]:
    """Add one explicit missing-text record for every raw polygon without a text link."""

    record_scope = wiki_corpus or text_source.value
    if text_source is TextSource.WIKI:
        matching_rows = (
            "SELECT native_key_json, inventory_row_id FROM inventory i "
            "WHERE NOT EXISTS (SELECT 1 FROM wiki_links w "
            "WHERE w.native_key_json=i.native_key_json) ORDER BY inventory_row_id"
        )
    else:
        matching_rows = (
            "SELECT native_key_json, inventory_row_id FROM inventory i "
            "WHERE NOT EXISTS (SELECT 1 FROM text_docs d "
            "WHERE d.direct_link_key=i.native_key_json) ORDER BY inventory_row_id"
        )
    for native_key_payload, inventory_row_id in connection.execute(matching_rows):
        document_id = _document_id(
            text_source,
            layout.raw_revision,
            {
                "missing_text_for_inventory_row": str(inventory_row_id),
                "native_key_json": str(native_key_payload),
                "corpus": record_scope,
            },
        )
        document = SourceDocument(
            document_id=document_id,
            native_key=canonical_native_key(
                {
                    "missing_text_for_inventory_row": str(inventory_row_id),
                    "native_key_json": str(native_key_payload),
                    "corpus": record_scope,
                }
            ),
            source_revision=layout.raw_revision,
            content_sha256=None,
            body_status=BodyStatus.MISSING,
        )
        link = _missing_link(
            document,
            layout.raw_revision,
            "no source text/document link row matched this raw polygon",
        )
        relevance = RelevanceAssessment(
            document_id=document_id,
            release_id=release_id,
            status=RelevanceStatus.MISSING,
            source_repository=layout.relevance_repository,
            source_revision=layout.relevance_revision,
            mapping_rule_version=mapping_rule,
            evidence=_canonical_json(
                {
                    "inventory_row_id": str(inventory_row_id),
                    "polygon_native_key": json.loads(str(native_key_payload)),
                    "corpus": record_scope,
                }
            ),
            reason=(
                "no downstream text or relevance assessment exists for this raw polygon"
                + (f" in {wiki_corpus}" if wiki_corpus else "")
            ),
        )
        yield JoinedTextRecord(document, (link,), relevance)


def _native_key_from_fields(
    values: Mapping[str, Any],
    fields: Sequence[str],
) -> NativeKey | None:
    if any(not _native_scalar(values.get(field)) for field in fields):
        return None
    return canonical_native_key({field: values[field] for field in fields})


def _document_native_key(values: Mapping[str, Any], locator: str) -> NativeKey:
    filtered = {str(key): value for key, value in values.items() if _native_scalar(value)}
    if not filtered:
        filtered = {"source_row_locator": locator}
    return canonical_native_key(filtered)


def _document_id(text_source: TextSource, revision: str, identity: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(
        _canonical_json({"revision": revision, **identity}).encode()
    ).hexdigest()
    return f"{text_source.value}:{digest}"


def _description_identity(
    osm_type: str,
    osm_id: int,
    tag_key: str,
    text_sha256: str,
) -> str:
    """Reproduce the pinned Description row identity formula exactly."""

    material = _canonical_json([osm_type, osm_id, tag_key, text_sha256]).encode("utf-8")
    return hashlib.sha256(material).hexdigest()


def _text_and_sha256(value: Any) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, str):
        raise TypeError("pinned text sentence values must be strings or null")
    return value, hashlib.sha256(value.encode("utf-8")).hexdigest()


def _join_key(values: Sequence[Any]) -> str | None:
    if any(value is None or isinstance(value, bool) for value in values):
        return None
    return _canonical_json(list(values))


def _require_fields(row: PinnedSourceRow, table_name: str, columns: Sequence[str]) -> None:
    missing = sorted(set(columns) - set(row.values))
    if missing:
        raise ValueError(f"{table_name} row from {row.path_in_repo} lacks columns {missing}")


def _native_scalar(value: Any) -> bool:
    return (
        isinstance(value, (str, int))
        and not isinstance(value, bool)
        and bool(value.strip() if isinstance(value, str) else True)
    )


def _as_nonempty_string(value: Any) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(character in "0123456789abcdef" for character in value)
    )


def _safe_metadata_value(value: Any) -> Any:
    if _is_json_scalar(value):
        return value
    if isinstance(value, (list, tuple)):
        return _safe_metadata_sequence(value)
    if isinstance(value, Mapping):
        return _safe_metadata_mapping(value)
    return str(value)


def _is_json_scalar(value: Any) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def _safe_metadata_sequence(values: Sequence[Any]) -> list[Any]:
    return [_safe_metadata_value(value) for value in values]


def _safe_metadata_mapping(values: Mapping[Any, Any]) -> dict[str, Any]:
    return {str(key): _safe_metadata_value(value) for key, value in values.items()}


def _canonical_json(value: Any) -> str:
    return _JSON(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _require_utf8_size(payload: str, max_bytes: int, message: str) -> None:
    if len(payload.encode("utf-8")) > max_bytes:
        raise ValueError(message)


def _native_key_from_json(value: str) -> NativeKey:
    payload = json.loads(value)
    return tuple((str(key), item) for key, item in payload)


def _record_size(record: JoinedTextRecord) -> int:
    payload = {
        "document": {
            "document_id": record.document.document_id,
            "native_key": record.document.native_key,
            "source_revision": record.document.source_revision,
            "content_sha256": record.document.content_sha256,
            "body_status": record.document.body_status.value,
        },
        "links": [
            {
                "document_id": row.document_id,
                "inventory_row_id": row.inventory_row_id,
                "native_link_key": row.native_link_key,
                "source_revision": row.source_revision,
                "status": row.status.value,
                "evidence": row.evidence,
                "candidate_inventory_row_ids": row.candidate_inventory_row_ids,
            }
            for row in record.links
        ],
        "relevance": {
            "document_id": record.relevance.document_id,
            "release_id": record.relevance.release_id,
            "status": record.relevance.status.value,
            "source_repository": record.relevance.source_repository,
            "source_revision": record.relevance.source_revision,
            "mapping_rule_version": record.relevance.mapping_rule_version,
            "evidence": record.relevance.evidence,
            "raw_value": record.relevance.raw_value,
            "evidence_sha256": record.relevance.evidence_sha256,
            "reason": record.relevance.reason,
        },
    }
    return len(_canonical_json(payload).encode("utf-8")) + _BATCH_ROW_OVERHEAD


def _verify_local_tree_file(item: EunisPinnedParquetFile) -> None:
    size = _verified_local_tree_file_size(item)
    sha256_digest, git_blob_digest = _local_tree_file_digests(item.local_path, size)
    _validate_local_tree_file_digest(item.tree_file, size, sha256_digest, git_blob_digest)


def _verified_local_tree_file_size(item: EunisPinnedParquetFile) -> int:
    if not item.local_path.is_file():
        raise FileNotFoundError(f"pinned Parquet file is absent: {item.tree_file.path}")
    size = item.local_path.stat().st_size
    if size != item.tree_file.size_bytes:
        raise ValueError(f"pinned Parquet size differs for {item.tree_file.path}")
    return size


def _local_tree_file_digests(path: Path, size: int) -> tuple[str, str]:
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha1.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
            sha1.update(chunk)
    return sha256.hexdigest(), sha1.hexdigest()


def _validate_local_tree_file_digest(
    tree_file: SourceTreeFile,
    size: int,
    sha256_digest: str,
    git_blob_digest: str,
) -> None:
    if tree_file.lfs_oid is not None:
        if sha256_digest != tree_file.lfs_oid or size != tree_file.lfs_size:
            raise ValueError(f"pinned Parquet LFS hash differs for {tree_file.path}")
        return
    if git_blob_digest != tree_file.blob_id:
        raise ValueError(f"pinned Parquet Git blob hash differs for {tree_file.path}")
