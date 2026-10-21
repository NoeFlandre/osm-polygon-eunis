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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from .pool_adapter import (
    EUNIS_POOL_SOURCE_LAYOUTS,
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
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
    canonical_native_key,
    native_key_json,
)
from .pool_contract.models import NativeKey

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

        if batch_size <= 0:
            raise ValueError("Parquet row batch size must be positive")
        if not required_columns or len(set(required_columns)) != len(required_columns):
            raise ValueError("pinned table reads need a unique, nonempty column list")
        for item in sorted(self.files, key=lambda value: value.tree_file.path):
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
    known = {
        selected.raw_polygon_tree_pin.path_prefix,
        *(pin.path_prefix for pin in selected.additional_tree_pins),
    }
    # These groups are present in the pinned source revision but absent from the
    # sanitized source-tree handoff. Do not synthesize pins for them.
    required = (
        "polygon_document_links/",
        "wikipedia/documents/",
        "wikivoyage/documents/",
    )
    return tuple(path for path in required if path not in known)


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
) -> Iterator[JoinedTextRecord]:
    """Join a pinned source table to relevance labels without filtering rows.

    ``text_rows`` are Description sentence rows, Website polygon rows, or one
    Wiki sentence table. ``relevance_rows`` are the matching pinned labels
    table. Wiki joins additionally consume the pinned polygon-document links.
    The inputs are streamed into a disk-backed index; no text body is persisted.
    """

    selected_layout = layout or EUNIS_POOL_SOURCE_LAYOUTS[text_source]
    if selected_layout.text_source is not text_source:
        raise ValueError("text-source layout does not match the requested join")
    if text_source is TextSource.WIKI and wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki joins require corpus='wikipedia' or corpus='wikivoyage'")
    if text_source is not TextSource.WEBSITE and website_field_to_column is not None:
        raise ValueError("Website field mapping can only be supplied for Website joins")
    if text_source is TextSource.WEBSITE and not website_field_to_column:
        raise ValueError("Website joins require a pinned field-to-sentence-column mapping")

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
            _initialize_join_index(connection)
            _stage_inventory(connection, inventory_rows)
            _stage_relevance(connection, relevance_rows, text_source, wiki_corpus)
            if text_source is TextSource.WIKI:
                _stage_wiki_links(connection, wiki_link_rows, str(wiki_corpus))
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
) -> Iterator[JoinedTextRecord]:
    """Validate source group pins and schemas, then run the bounded join."""

    selected_layout = layout or EUNIS_POOL_SOURCE_LAYOUTS[text_source]
    if selected_layout.text_source is not text_source:
        raise ValueError("text-source layout does not match the requested join")
    text_role, text_prefix, text_columns, text_types = _text_table_contract(
        text_source, wiki_corpus
    )
    relevance_role, relevance_prefix, relevance_columns, relevance_types = (
        _relevance_table_contract(text_source, wiki_corpus)
    )
    _require_expected_pin(selected_layout, text_table, text_role, text_prefix)
    _require_expected_pin(selected_layout, relevance_table, relevance_role, relevance_prefix)

    wiki_link_rows: Iterable[PinnedSourceRow] = ()
    if text_source is TextSource.WIKI:
        missing = missing_wiki_join_tree_groups(selected_layout)
        if missing:
            raise ValueError(
                "Wiki document/link tree metadata is not pinned for: " + ", ".join(missing)
            )
        if wiki_link_table is None:
            raise ValueError("Wiki joins require the pinned polygon_document_links table")
        _require_expected_pin(
            selected_layout,
            wiki_link_table,
            "wiki_document_links",
            "polygon_document_links/",
        )
        expected_document_prefix = (
            "wikipedia/documents/" if wiki_corpus == "wikipedia" else "wikivoyage/documents/"
        )
        if not wiki_document_tables:
            raise ValueError("Wiki joins require the pinned document metadata table")
        if not any(
            table.pin.path_prefix == expected_document_prefix for table in wiki_document_tables
        ):
            raise ValueError(f"Wiki document pin for {expected_document_prefix} is absent")
        for table in wiki_document_tables:
            _require_expected_pin(
                selected_layout,
                table,
                "wiki_documents",
                table.pin.path_prefix,
            )
        wiki_link_rows = wiki_link_table.iter_rows(
            (
                "document_id",
                "project",
                "polygon_id",
                "source_pbf",
                "osm_type",
                "osm_id",
                "link_sources",
            ),
            batch_size=batch_size,
            expected_types={
                "document_id": "string",
                "project": "string",
                "polygon_id": "string",
                "source_pbf": "string",
                "osm_type": "string",
                "osm_id": "int64",
                "link_sources": "string",
            },
        )

    return iter_eunis_join_records(
        text_source,
        text_table.iter_rows(
            text_columns,
            batch_size=batch_size,
            expected_types=text_types,
        ),
        relevance_table.iter_rows(
            relevance_columns,
            batch_size=batch_size,
            expected_types=relevance_types,
        ),
        inventory_rows,
        wiki_link_rows=wiki_link_rows,
        wiki_corpus=wiki_corpus,
        website_field_to_column=website_field_to_column,
        layout=selected_layout,
        staging_directory=staging_directory,
    )


def _require_expected_pin(
    layout: EunisPoolSourceLayout,
    table: EunisPinnedTable,
    role: str,
    path_prefix: str,
) -> None:
    expected = [
        pin
        for pin in (layout.raw_polygon_tree_pin, *layout.additional_tree_pins)
        if pin.role == role and pin.path_prefix == path_prefix
    ]
    if len(expected) != 1:
        raise ValueError(f"source layout has no unique verified tree pin for {role}:{path_prefix}")
    if table.pin != expected[0]:
        raise ValueError(f"input table pin differs from source layout for {role}:{path_prefix}")


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
        if name not in schema.names:
            raise ValueError(f"{path} lacks pinned schema field {name}")
        actual = schema.field(name).type
        valid = {
            "string": pa.types.is_string(actual),
            "int32": pa.types.is_int32(actual),
            "int64": pa.types.is_int64(actual),
            "list<string>": pa.types.is_list(actual) and pa.types.is_string(actual.value_type),
            "large_string": pa.types.is_large_string(actual),
        }.get(expected)
        if valid is None:
            raise ValueError(f"unknown schema type contract {expected!r} for {name}")
        if not valid:
            raise TypeError(f"{path} field {name} has {actual}, expected {expected}")


def iter_eunis_join_batches(
    records: Iterable[JoinedTextRecord],
    *,
    starting_batch_index: int,
    max_batch_rows: int,
    max_batch_bytes: int,
) -> Iterator[PoolBatch]:
    """Pack joined document records into consecutive bounded core batches."""

    if starting_batch_index < 0 or max_batch_rows <= 0 or max_batch_bytes <= 0:
        raise ValueError("batch index and limits must be positive")
    index = starting_batch_index
    documents: list[SourceDocument] = []
    links: list[TextPolygonLink] = []
    relevance: list[RelevanceAssessment] = []
    rows = 0
    size = 0

    def flush() -> PoolBatch | None:
        nonlocal index, rows, size
        if not documents:
            return None
        batch = PoolBatch(
            batch_index=index,
            documents=tuple(documents),
            text_polygon_links=tuple(links),
            relevance_assessments=tuple(relevance),
        )
        index += 1
        documents.clear()
        links.clear()
        relevance.clear()
        rows = 0
        size = 0
        return batch

    for record in records:
        record_rows = 2 + sum(1 + len(link.candidate_inventory_row_ids) for link in record.links)
        record_size = _record_size(record)
        if record_rows > max_batch_rows or record_size > max_batch_bytes:
            raise ValueError("one joined text record exceeds the configured batch bounds")
        if documents and (
            rows + record_rows > max_batch_rows or size + record_size > max_batch_bytes
        ):
            batch = flush()
            if batch is None:
                raise RuntimeError("a nonempty downstream batch failed to flush")
            yield batch
        documents.append(record.document)
        links.extend(record.links)
        relevance.append(record.relevance)
        rows += record_rows
        size += record_size
    batch = flush()
    if batch is not None:
        yield batch


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
) -> None:
    scope = wiki_corpus or text_source.value
    for source_row in rows:
        value = source_row.values
        if text_source is TextSource.DESCRIPTION:
            _require_fields(
                source_row,
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
            )
            key_fields = (
                value.get("description_identity"),
                value.get("tag_key"),
                value.get("language"),
                value.get("sentence_index"),
                value.get("text_sha256"),
            )
        elif text_source is TextSource.WEBSITE:
            _require_fields(
                source_row,
                "Website relevance",
                (
                    "label_id",
                    "polygon_id",
                    "field",
                    "sentence_index",
                    "text_sha256",
                    "decision",
                ),
            )
            key_fields = (
                value.get("polygon_id"),
                value.get("field"),
                value.get("sentence_index"),
                value.get("text_sha256"),
            )
        else:
            _require_fields(
                source_row,
                "Wiki relevance",
                (
                    "label_id",
                    "sentence_id",
                    "text_sha256",
                    "decision",
                ),
            )
            key_fields = (value.get("sentence_id"), value.get("text_sha256"))
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
        connection.execute(
            "INSERT INTO relevance_rows(scope, join_key, row_json) VALUES (?, ?, ?)",
            (scope, join_key, _canonical_json(metadata | {"_locator": source_row.locator})),
        )
    connection.commit()


def _stage_wiki_links(
    connection: sqlite3.Connection,
    rows: Iterable[PinnedSourceRow],
    wiki_corpus: str,
) -> None:
    for source_row in rows:
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
        if document_id is None:
            # Retain malformed identifiers under a source-row-specific ID.
            document_id = f"missing-document:{source_row.locator}"
        if project != wiki_corpus:
            continue
        key_values = {
            field: value[field]
            for field in ("source_pbf", "osm_type", "osm_id")
            if _native_scalar(value.get(field))
        }
        is_complete = len(key_values) == _OSM_KEY_FIELD_COUNT
        if not key_values:
            key_values = {"incomplete_link_row": source_row.locator}
        key = canonical_native_key(key_values)
        link_sources = value.get("link_sources")
        evidence = {
            "table": source_row.path_in_repo,
            "row": source_row.row_ordinal,
            "polygon_id": value.get("polygon_id"),
            "link_sources": _safe_metadata_value(link_sources),
            "key_complete": is_complete,
        }
        connection.execute(
            "INSERT INTO wiki_links(document_id, project, native_key_json, evidence_json, "
            "complete_key) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                document_id,
                project,
                native_key_json(key),
                _canonical_json(evidence),
                int(is_complete),
            ),
        )
    connection.commit()


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
        value = source_row.values
        identity = _as_nonempty_string(value.get("description_identity"))
        tag_key = _as_nonempty_string(value.get("tag_key"))
        language = _as_nonempty_string(value.get("language_code"))
        source_text_sha256 = value.get("text_sha256")
        if not isinstance(source_text_sha256, str) or not _is_sha256(source_text_sha256):
            raise ValueError(f"Description row {source_row.locator} has no valid source text hash")
        original_text, observed_source_hash = _text_and_sha256(value.get("original_text"))
        if original_text is None or observed_source_hash != source_text_sha256:
            raise ValueError(
                f"Description text hash does not match its pinned row at {source_row.locator}"
            )
        osm_type = _as_nonempty_string(value.get("osm_type"))
        osm_id = value.get("osm_id")
        if (
            identity is not None
            and tag_key is not None
            and osm_type is not None
            and isinstance(osm_id, int)
            and not isinstance(osm_id, bool)
            and _description_identity(osm_type, osm_id, tag_key, source_text_sha256) != identity
        ):
            raise ValueError(
                f"Description identity does not match its pinned row at {source_row.locator}"
            )
        polygon_key = _native_key_from_fields(value, ("source_pbf", "osm_type", "osm_id"))
        sentences = value.get("sentences")
        if sentences is None:
            sentences = ()
        if not isinstance(sentences, (list, tuple)):
            raise TypeError("Description sentences must be an Arrow list of strings")
        if not sentences:
            split_status = value.get("split_status")
            split_reason = value.get("split_reason")
            sentence_state = "null" if value.get("sentences") is None else "empty"
            identity_fields = {
                "description_identity": identity or "",
                "tag_key": tag_key or "",
                "language_code": language or "",
                "source_text_sha256": source_text_sha256,
                "split_status": str(split_status or "missing"),
                "split_reason": str(split_reason or "missing"),
                "sentence_state": sentence_state,
                "source_row": source_row.locator,
            }
            _insert_text_doc(
                connection,
                document_id=_document_id(
                    TextSource.DESCRIPTION,
                    layout.raw_revision,
                    identity_fields,
                ),
                native_key=_document_native_key(identity_fields, source_row.locator),
                source_revision=layout.raw_revision,
                content_sha256=str(source_text_sha256),
                body_status=BodyStatus.PRIVATE,
                label_scope=TextSource.DESCRIPTION.value,
                label_join_key=None,
                direct_link_key=native_key_json(polygon_key) if polygon_key else None,
                wiki_document_id=None,
                wiki_project=None,
                source_locator=source_row.locator,
                corpus=None,
                missing_reason=(
                    "pinned Description row has no sentence output; "
                    f"sentence array={sentence_state}, "
                    f"split status={split_status or 'missing'}, "
                    f"split reason={split_reason or 'missing'}; no sentence-level "
                    "relevance assessment applies"
                ),
            )
            continue
        for sentence_index, sentence in enumerate(sentences):
            text, digest = _text_and_sha256(sentence)
            join_key = _join_key((identity, tag_key, language, sentence_index, source_text_sha256))
            identity_fields = {
                "description_identity": identity or "",
                "tag_key": tag_key or "",
                "language_code": language or "",
                "sentence_index": sentence_index,
                "text_sha256": digest or "",
                "source_row": source_row.locator,
            }
            native_key = _document_native_key(identity_fields, source_row.locator)
            document_id = _document_id(TextSource.DESCRIPTION, layout.raw_revision, identity_fields)
            _insert_text_doc(
                connection,
                document_id=document_id,
                native_key=native_key,
                source_revision=layout.raw_revision,
                content_sha256=digest,
                relevance_evidence_sha256=source_text_sha256,
                body_status=BodyStatus.PRIVATE if text is not None else BodyStatus.MISSING,
                label_scope=TextSource.DESCRIPTION.value,
                label_join_key=join_key,
                direct_link_key=native_key_json(polygon_key) if polygon_key else None,
                wiki_document_id=None,
                wiki_project=None,
                source_locator=source_row.locator,
                corpus=None,
            )


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
    if not website_field_to_column:
        raise ValueError("Website joins require a pinned field-to-sentence-column mapping")
    if len(set(website_field_to_column.values())) != len(website_field_to_column):
        raise ValueError("Website sentence columns must map to exactly one label field each")
    if set(website_field_to_column.values()) != {
        "website_sentences",
        "contact_website_sentences",
    }:
        raise ValueError("Website field map must cover both pinned sentence columns exactly once")
    for source_row in rows:
        _require_fields(
            source_row,
            "Website polygon",
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
        )
        value = source_row.values
        polygon_id = _as_nonempty_string(value.get("polygon_id"))
        polygon_key = _native_key_from_fields(value, ("source_pbf", "osm_type", "osm_id"))
        for label_field, column in sorted(website_field_to_column.items()):
            sentences = value.get(column)
            if sentences is None:
                sentences = ()
            if not isinstance(sentences, (list, tuple)):
                raise TypeError(f"Website {column} must be an Arrow list of strings")
            if not sentences:
                field_prefix = column.removesuffix("_sentences")
                body_text = value.get(f"{field_prefix}_text")
                body_status = (
                    BodyStatus.PRIVATE if isinstance(body_text, str) else BodyStatus.MISSING
                )
                body_hash = _text_and_sha256(body_text)[1]
                text_status = value.get(f"{field_prefix}_text_status")
                sentence_status = value.get(f"{field_prefix}_sentence_status")
                sentence_state = "null" if value.get(column) is None else "empty"
                identity_fields = {
                    "polygon_id": polygon_id or "",
                    "field": label_field,
                    "source_text_sha256": body_hash or "",
                    "text_status": str(text_status or "missing"),
                    "sentence_status": str(sentence_status or "missing"),
                    "sentence_state": sentence_state,
                    "source_row": source_row.locator,
                }
                _insert_text_doc(
                    connection,
                    document_id=_document_id(
                        TextSource.WEBSITE,
                        layout.raw_revision,
                        identity_fields,
                    ),
                    native_key=_document_native_key(identity_fields, source_row.locator),
                    source_revision=layout.raw_revision,
                    content_sha256=body_hash,
                    body_status=body_status,
                    label_scope=TextSource.WEBSITE.value,
                    label_join_key=None,
                    direct_link_key=native_key_json(polygon_key) if polygon_key else None,
                    wiki_document_id=None,
                    wiki_project=None,
                    source_locator=source_row.locator,
                    corpus=None,
                    missing_reason=(
                        "Website text field has no sentence output; "
                        f"sentence array={sentence_state}, "
                        f"text status={text_status or 'missing'}, "
                        f"sentence status={sentence_status or 'missing'}"
                    ),
                )
                continue
            for sentence_index, sentence in enumerate(sentences):
                text, digest = _text_and_sha256(sentence)
                join_key = _join_key((polygon_id, label_field, sentence_index, digest))
                identity_fields = {
                    "polygon_id": polygon_id or "",
                    "field": label_field,
                    "sentence_index": sentence_index,
                    "text_sha256": digest or "",
                    "source_row": source_row.locator,
                }
                native_key = _document_native_key(identity_fields, source_row.locator)
                document_id = _document_id(TextSource.WEBSITE, layout.raw_revision, identity_fields)
                _insert_text_doc(
                    connection,
                    document_id=document_id,
                    native_key=native_key,
                    source_revision=layout.raw_revision,
                    content_sha256=digest,
                    relevance_evidence_sha256=digest,
                    body_status=BodyStatus.PRIVATE if text is not None else BodyStatus.MISSING,
                    label_scope=TextSource.WEBSITE.value,
                    label_join_key=join_key,
                    direct_link_key=native_key_json(polygon_key) if polygon_key else None,
                    wiki_document_id=None,
                    wiki_project=None,
                    source_locator=source_row.locator,
                    corpus=None,
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
    if wiki_corpus not in {"wikipedia", "wikivoyage"}:
        raise ValueError("Wiki sentence rows need a pinned Wikipedia or Wikivoyage table")
    scope = str(wiki_corpus)
    for source_row in rows:
        _require_fields(
            source_row,
            "Wiki sentence",
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
        )
        value = source_row.values
        sentence_id = _as_nonempty_string(value.get("sentence_id"))
        document_id_raw = _as_nonempty_string(value.get("document_id"))
        project = _as_nonempty_string(value.get("project"))
        text, digest = _text_and_sha256(value.get("text"))
        source_hash = value.get("content_hash")
        segmentation_status = value.get("segmentation_status")
        if text is not None and (not _is_sha256(source_hash) or source_hash != digest):
            raise ValueError(f"Wiki sentence content hash differs at {source_row.locator}")
        identity_fields = {
            "corpus": scope,
            "sentence_id": sentence_id or "",
            "document_id": document_id_raw or "",
            "project": project or "",
            "language": _as_nonempty_string(value.get("language")) or "",
            "page_id": _native_scalar(value.get("page_id")) or "",
            "revision_id": _native_scalar(value.get("revision_id")) or "",
            "text_sha256": digest or "",
            "source_row": source_row.locator,
        }
        native_key = _document_native_key(identity_fields, source_row.locator)
        output_document_id = _document_id(TextSource.WIKI, layout.raw_revision, identity_fields)
        join_key = _join_key((sentence_id, digest))
        _insert_text_doc(
            connection,
            document_id=output_document_id,
            native_key=native_key,
            source_revision=layout.raw_revision,
            content_sha256=digest,
            relevance_evidence_sha256=digest,
            body_status=BodyStatus.PRIVATE if text is not None else BodyStatus.MISSING,
            label_scope=scope,
            label_join_key=join_key,
            direct_link_key=None,
            wiki_document_id=document_id_raw,
            wiki_project=project,
            source_locator=source_row.locator,
            corpus=scope,
            missing_reason=(
                "Wiki sentence row has no text body; "
                f"segmentation status={segmentation_status or 'missing'}"
                if text is None
                else None
            ),
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


def _emit_join_records(
    connection: sqlite3.Connection,
    text_source: TextSource,
    layout: EunisPoolSourceLayout,
    mapping_rule: str,
    wiki_corpus: str | None,
) -> Iterator[JoinedTextRecord]:
    relevance_repository = layout.relevance_repository
    relevance_revision = layout.relevance_revision
    release_id = f"{relevance_repository}@{relevance_revision}"
    duplicate_docs = {
        (str(scope), str(join_key))
        for scope, join_key, count in connection.execute(
            "SELECT label_scope, label_join_key, COUNT(*) FROM text_docs "
            "WHERE label_join_key IS NOT NULL "
            "GROUP BY label_scope, label_join_key HAVING COUNT(*) > 1"
        )
    }
    for row in connection.execute(
        "SELECT document_id, native_key_json, source_revision, content_sha256, "
        "relevance_evidence_sha256, body_status, "
        "label_scope, label_join_key, direct_link_key, wiki_document_id, wiki_project, "
        "source_locator, corpus, missing_reason FROM text_docs ORDER BY sequence"
    ):
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
        document = SourceDocument(
            document_id=str(document_id),
            native_key=_native_key_from_json(str(native_key_payload)),
            source_revision=str(source_revision),
            content_sha256=str(content_sha256) if content_sha256 else None,
            body_status=BodyStatus(str(body_status)),
        )
        relevance = _resolve_relevance(
            connection,
            document_id=str(document_id),
            join_key=str(label_join_key) if label_join_key is not None else None,
            scope=str(label_scope),
            duplicate_doc_key=(str(label_scope), str(label_join_key)) in duplicate_docs
            if label_join_key is not None
            else False,
            source_repository=relevance_repository,
            source_revision=relevance_revision,
            release_id=release_id,
            mapping_rule=mapping_rule,
            evidence_sha256=(str(relevance_evidence_sha256) if relevance_evidence_sha256 else None),
            source_locator=str(source_locator),
            missing_reason=str(missing_reason) if missing_reason else None,
        )
        links = _resolve_links(
            connection,
            text_source,
            document,
            direct_link_key=str(direct_link_key) if direct_link_key else None,
            wiki_document_id=str(wiki_document_id) if wiki_document_id else None,
            wiki_project=str(wiki_project) if wiki_project else None,
            source_revision=layout.raw_revision,
            source_locator=str(source_locator),
            corpus=str(corpus) if corpus else wiki_corpus,
        )
        yield JoinedTextRecord(document, links, relevance)

    yield from _emit_orphan_label_records(
        connection,
        text_source,
        layout,
        release_id,
        mapping_rule,
        wiki_corpus,
    )
    if text_source is TextSource.WIKI:
        yield from _emit_orphan_wiki_link_records(connection, layout, release_id, mapping_rule)
    yield from _emit_unrepresented_inventory_records(
        connection,
        text_source,
        layout,
        release_id,
        mapping_rule,
        wiki_corpus,
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
    missing_reason: str | None = None,
) -> RelevanceAssessment:
    matches = []
    if join_key is not None:
        matches = list(
            connection.execute(
                "SELECT sequence, row_json FROM relevance_rows "
                "WHERE scope=? AND join_key=? ORDER BY sequence",
                (scope, join_key),
            )
        )
        if matches:
            connection.executemany(
                "UPDATE relevance_rows SET consumed=1 WHERE sequence=?",
                ((int(sequence),) for sequence, _payload in matches),
            )
    evidence = {
        "text_row": source_locator,
        "join_key_sha256": hashlib.sha256((join_key or "missing").encode()).hexdigest(),
        "label_rows": [
            {
                "label_id": json.loads(payload).get("label_id"),
                "locator": json.loads(payload).get("_locator"),
            }
            for _sequence, payload in matches
        ],
    }
    status: RelevanceStatus
    reason = None
    raw_value = None
    if missing_reason is not None:
        status = RelevanceStatus.MISSING
        reason = missing_reason
    elif duplicate_doc_key:
        status = RelevanceStatus.UNKNOWN
        reason = "multiple source text rows share one relevance key"
    elif len(matches) > 1:
        status = RelevanceStatus.UNKNOWN
        reason = "multiple relevance rows match the source text key"
    elif not matches:
        status = RelevanceStatus.MISSING
        reason = "no relevance row matches the source-native key and text hash"
    else:
        payload = json.loads(matches[0][1])
        raw = payload.get("decision")
        normalized = raw.strip().casefold() if isinstance(raw, str) else ""
        if normalized == "yes":
            status = RelevanceStatus.YES
            raw_value = str(raw)
        elif normalized == "no":
            status = RelevanceStatus.NO
            raw_value = str(raw)
        else:
            status = RelevanceStatus.UNKNOWN
            reason = (
                "relevance row has an empty decision"
                if raw is None or (isinstance(raw, str) and not raw.strip())
                else "relevance decision is outside the pinned YES/NO vocabulary"
            )
    return RelevanceAssessment(
        document_id=document_id,
        release_id=release_id,
        status=status,
        source_repository=source_repository,
        source_revision=source_revision,
        mapping_rule_version=mapping_rule,
        evidence=_canonical_json(evidence),
        raw_value=raw_value,
        evidence_sha256=evidence_sha256,
        reason=reason,
    )


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
) -> tuple[TextPolygonLink, ...]:
    if text_source is not TextSource.WIKI:
        if direct_link_key is None:
            return (_missing_link(document, source_revision, "source polygon key is incomplete"),)
        key = _native_key_from_json(direct_link_key)
        return (
            _resolve_one_link(
                connection,
                document.document_id,
                key,
                source_revision,
                f"direct source-native OSM key from {source_locator}",
                True,
            ),
        )

    if not wiki_document_id or not wiki_project:
        return (
            _missing_link(
                document,
                source_revision,
                "Wiki sentence has no document/project key",
            ),
        )
    rows = list(
        connection.execute(
            "SELECT sequence, native_key_json, evidence_json, complete_key FROM wiki_links "
            "WHERE document_id=? AND project=? ORDER BY sequence",
            (wiki_document_id, wiki_project),
        )
    )
    # The source may attach one document to multiple polygons. Those are
    # separate explicit links; they are not collapsed into an arbitrary pick.
    if not rows:
        return (
            _missing_link(
                document,
                source_revision,
                f"no polygon_document_links row for {corpus}/{wiki_document_id}/{wiki_project}",
            ),
        )
    combined: dict[str, tuple[str, bool]] = {}
    connection.executemany(
        "UPDATE wiki_links SET consumed=1 WHERE sequence=?",
        ((int(sequence),) for sequence, _key, _evidence, _complete in rows),
    )
    for _sequence, native_key_payload, evidence_payload, complete in rows:
        prior = combined.get(str(native_key_payload))
        evidence_json = str(evidence_payload)
        is_complete = bool(complete)
        if prior is not None:
            evidence_json = _canonical_json([json.loads(prior[0]), json.loads(evidence_json)])
            is_complete = bool(prior[1]) and is_complete
        combined[str(native_key_payload)] = (evidence_json, is_complete)
    resolved = []
    for native_key_payload, (evidence_payload, complete) in sorted(combined.items()):
        key = _native_key_from_json(native_key_payload)
        resolved.append(
            _resolve_one_link(
                connection,
                document.document_id,
                key,
                source_revision,
                f"pinned polygon_document_links evidence {evidence_payload}",
                complete,
            )
        )
    return tuple(resolved)


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
    for sequence, scope, join_key, payload_json in connection.execute(
        "SELECT sequence, scope, join_key, row_json FROM relevance_rows "
        "WHERE consumed=0 ORDER BY sequence"
    ):
        payload = json.loads(payload_json)
        locator = str(payload.get("_locator") or "unknown-label-row")
        label_id = _as_nonempty_string(payload.get("label_id")) or str(sequence)
        identity = {
            "text_source": text_source.value,
            "scope": str(scope),
            "label_id": label_id,
            "label_row": locator,
            "join_key_sha256": hashlib.sha256(str(join_key or "").encode()).hexdigest(),
        }
        document_id = _document_id(text_source, layout.raw_revision, identity)
        native_key = _document_native_key({"label_id": label_id, "label_row": locator}, locator)
        hash_value = payload.get("text_sha256")
        content_sha256 = str(hash_value) if _is_sha256(hash_value) else None
        document = SourceDocument(
            document_id=document_id,
            native_key=native_key,
            source_revision=layout.raw_revision,
            content_sha256=content_sha256,
            body_status=BodyStatus.MISSING,
        )
        raw = payload.get("decision")
        normalized = raw.strip().casefold() if isinstance(raw, str) else ""
        if normalized in {"yes", "no"}:
            status = RelevanceStatus.MISSING
            raw_value = str(raw)
            reason = (
                "orphan relevance row preserves its source decision but has no matching "
                "source text row"
            )
        else:
            status = RelevanceStatus.UNKNOWN
            raw_value = None
            reason = "orphan relevance row has no matching source text and no valid YES/NO decision"
        relevance = RelevanceAssessment(
            document_id=document_id,
            release_id=release_id,
            status=status,
            source_repository=layout.relevance_repository,
            source_revision=layout.relevance_revision,
            mapping_rule_version=mapping_rule,
            evidence=_canonical_json(
                {
                    "orphan_label_row": locator,
                    "label_id": label_id,
                    "join_key_sha256": identity["join_key_sha256"],
                    "corpus": wiki_corpus,
                }
            ),
            raw_value=raw_value,
            evidence_sha256=content_sha256,
            reason=reason,
        )
        link = _missing_link(
            document,
            layout.raw_revision,
            "relevance row has no matching source text or polygon link",
        )
        yield JoinedTextRecord(document, (link,), relevance)


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
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_metadata_value(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _safe_metadata_value(item) for key, item in value.items()}
    return str(value)


def _canonical_json(value: Any) -> str:
    return _JSON(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


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
    path = item.local_path
    if not path.is_file():
        raise FileNotFoundError(f"pinned Parquet file is absent: {item.tree_file.path}")
    size = path.stat().st_size
    if size != item.tree_file.size_bytes:
        raise ValueError(f"pinned Parquet size differs for {item.tree_file.path}")
    sha256 = hashlib.sha256()
    sha1 = hashlib.sha1(usedforsecurity=False)
    sha1.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha256.update(chunk)
            sha1.update(chunk)
    if item.tree_file.lfs_oid is not None:
        if sha256.hexdigest() != item.tree_file.lfs_oid or size != item.tree_file.lfs_size:
            raise ValueError(f"pinned Parquet LFS hash differs for {item.tree_file.path}")
    elif sha1.hexdigest() != item.tree_file.blob_id:
        raise ValueError(f"pinned Parquet Git blob hash differs for {item.tree_file.path}")
