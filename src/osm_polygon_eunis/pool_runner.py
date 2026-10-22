"""Bounded, resumable raw-first staging and streaming export for EUNIS pools.

This module writes per-shard JSONL checkpoints and passes bounded batches to
the shared core's SQLite-backed exporter. It does not materialize full pools
or Parquet bytes in memory.
"""

from __future__ import annotations

import base64
import hashlib
import heapq
import json
import os
import re
import tempfile
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import asdict, dataclass, replace
from dataclasses import field as dataclass_field
from fnmatch import fnmatchcase
from pathlib import Path
from typing import IO, Any, cast
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq
from pyproj.exceptions import ProjError
from rasterio.errors import RasterioError
from shapely import to_wkb
from shapely.errors import GEOSException
from shapely.geometry.base import BaseGeometry

from .eea import EEA_2021_CLASSIFICATION_LINEAGE_PIN
from .eunis_asset_manifest import (
    EEA_2021_MAP_RELEASE,
    EEA_2021_NATIVE_ASSET_INVENTORY_SHA256,
    EEA_2021_NATIVE_MANIFEST_SHA256,
    EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256,
)
from .geometry import is_usable, parse_geometry, to_equal_area
from .pool_adapter import (
    EUNIS_AREA_MEASUREMENT_ID,
    EUNIS_ASSIGNMENT_RULE_VERSION,
    EUNIS_POOL_LEVELS,
    EUNIS_SUPPORT_AREA_POLICY,
    EUNIS_THRESHOLD,
    EunisLabelAttempt,
    EunisPoolSourceLayout,
    EunisTaxonomy,
    SourceTreeFile,
    hierarchy_support_groups,
    resolve_eunis_labels,
    validate_source_tree_group,
)
from .pool_contract import (
    AmbiguityBasis,
    AreaMeasurementPin,
    ArtifactManifestPin,
    AssignmentKind,
    AttemptStatus,
    FileArtifactSink,
    GeometryStatus,
    LabelCandidate,
    LabelOutcome,
    LabelSupportEvidence,
    OutcomeStatus,
    PolygonAttemptReceipt,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBatch,
    PoolBuildInput,
    PoolKey,
    PoolStreamSpec,
    RawInventoryEvidence,
    ReferencePin,
    ReferenceSystem,
    SourceSnapshot,
    StreamingExportReceipt,
    StreamingLimits,
    SupportSemantics,
    canonical_native_key,
    export_pool_batches,
    inventory_row_identifier,
    native_key_json,
    polygon_snapshot_identifier,
)
from .pool_contract.models import NativeKey
from .raster_reference import RasterReference
from .text_join import (
    BoundedJoinedTextRecords,
    EunisPinnedTable,
    JoinAllocationBounds,
    JoinedTextRecord,
    iter_eunis_join_batches,
    iter_pinned_eunis_join_records,
)

_KEY_FIELDS = ("source_pbf", "osm_type", "osm_id")
_GEOMETRY_FIELD = "geometry"
_RAW_CHECKPOINT_VERSION = 1
_ATTEMPT_CHECKPOINT_VERSION = 4
_EEA_REFERENCE_ID_NAMESPACE = "eea-eunis-2021-probability-map-l3-code"
_ENUMERATION_RULE = (
    "enumerate every geometry row from every pinned polygon shard before any text, "
    "document-link, YES/NO, or relevance join; native key is (source_pbf, osm_type, osm_id)"
)
_CHUNK_KEYS = 50_000
_MERGE_FAN_IN = 64
_SHA256_HEX_PATTERN = re.compile(r"[0-9a-f]{64}")
_CHECKPOINT_URI_FRAGMENT_PARTS = 2
_GEOMETRY_ERRORS = (
    ArithmeticError,
    AttributeError,
    GEOSException,
    OSError,
    ProjError,
    TypeError,
    ValueError,
)
_REFERENCE_ERRORS = (
    ArithmeticError,
    GEOSException,
    OSError,
    RasterioError,
    RuntimeError,
    TypeError,
    ValueError,
)


@dataclass(frozen=True, slots=True)
class PinnedPolygonShard:
    """One local Parquet file verified against its immutable repository tree pin."""

    path_in_repo: str
    revision: str
    sha256: str
    size_bytes: int
    blob_id: str
    lfs_oid: str | None
    lfs_size: int
    local_path: Path


@dataclass(frozen=True, slots=True)
class FrozenPolygonSource:
    """Complete raw polygon-file inventory for one immutable Hub revision."""

    layout: EunisPoolSourceLayout
    source_snapshot: SourceSnapshot
    manifest_uri: str
    manifest_sha256: str
    manifest_paths: tuple[str, ...]
    polygon_shards: tuple[PinnedPolygonShard, ...]

    def __post_init__(self) -> None:
        """Validate that all raw geometry shards come from one frozen manifest."""

        _validate_frozen_polygon_source(self)


@dataclass(frozen=True, slots=True)
class RawShardReceipt:
    """Verified, resumable enumeration artifact for one source shard."""

    path_in_repo: str
    source_sha256: str
    source_size_bytes: int
    row_count: int
    checkpoint_uri: str
    checkpoint_sha256: str


@dataclass(frozen=True, slots=True)
class RawInventoryScan:
    """Source-wide raw inventory receipt and its per-shard row checkpoints."""

    source: FrozenPolygonSource
    code_pin: PoolAdapterCodePin
    source_fingerprint: str
    evidence: RawInventoryEvidence
    shards: tuple[RawShardReceipt, ...]


@dataclass(frozen=True, slots=True)
class AttemptShardReceipt:
    """Completed label-attempt artifact for one raw polygon shard."""

    path_in_repo: str
    checkpoint_uri: str
    checkpoint_sha256: str
    row_count: int
    attempt_counts: tuple[tuple[str, int], ...]
    outcome_counts: tuple[tuple[str, int], ...]
    candidate_counts: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class EunisAttemptRun:
    """Resumable label results for all raw shards after source census succeeds."""

    raw_scan: RawInventoryScan
    reference_pin: ReferencePin
    taxonomy: EunisTaxonomy
    code_pin: PoolAdapterCodePin
    signature: str
    report_uri: str
    report_sha256: str
    shards: tuple[AttemptShardReceipt, ...]


@dataclass(frozen=True, slots=True)
class EunisPoolExport:
    """Streaming pool receipt plus durable adapter evidence locations."""

    receipt: StreamingExportReceipt
    raw_inventory_report_uri: str
    label_attempt_report_uri: str
    attempt_checkpoint_uris: tuple[tuple[str, str], ...]
    downstream_join_included: bool


@dataclass(slots=True)
class _PoolBatchAccumulator:
    """Build consecutive batches bounded by rows and the number of polygons."""

    limits: StreamingLimits
    max_polygons_per_batch: int
    batch_index: int = 0
    inventory: list[PolygonInventoryRow] = dataclass_field(default_factory=list)
    polygons: list[PolygonSnapshot] = dataclass_field(default_factory=list)
    candidates: list[LabelCandidate] = dataclass_field(default_factory=list)
    outcomes: list[LabelOutcome] = dataclass_field(default_factory=list)
    attempts: list[PolygonAttemptReceipt] = dataclass_field(default_factory=list)
    support_evidence: list[LabelSupportEvidence] = dataclass_field(default_factory=list)
    current_rows: int = 0
    current_polygons: int = 0

    def add(
        self,
        inventory: PolygonInventoryRow,
        polygon: PolygonSnapshot | None,
        candidates: tuple[LabelCandidate, ...],
        outcomes: tuple[LabelOutcome, ...],
        attempt: PolygonAttemptReceipt,
        support_evidence: tuple[LabelSupportEvidence, ...],
    ) -> PoolBatch | None:
        """Append one raw polygon attempt, flushing a full batch first."""

        row_count = _pool_attempt_row_count(polygon, candidates, outcomes, support_evidence)
        if row_count > self.limits.max_batch_rows:
            raise ValueError(
                f"one polygon attempt has {row_count} rows, exceeding the configured "
                f"max_batch_rows={self.limits.max_batch_rows}"
            )
        previous = None
        if self.inventory and _pool_batch_is_full(self, row_count):
            previous = self.flush()
        self.inventory.append(inventory)
        if polygon is not None:
            self.polygons.append(polygon)
        self.candidates.extend(candidates)
        self.outcomes.extend(outcomes)
        self.attempts.append(attempt)
        self.support_evidence.extend(support_evidence)
        self.current_rows += row_count
        self.current_polygons += 1
        return previous

    def flush(self) -> PoolBatch | None:
        """Return the buffered batch and release its row references."""

        if not self.inventory:
            return None
        batch = PoolBatch(
            batch_index=self.batch_index,
            inventory=tuple(self.inventory),
            polygons=tuple(self.polygons),
            label_candidates=tuple(self.candidates),
            label_outcomes=tuple(self.outcomes),
            attempt_receipts=tuple(self.attempts),
            support_evidence=tuple(self.support_evidence),
        )
        self.batch_index += 1
        self.inventory.clear()
        self.polygons.clear()
        self.candidates.clear()
        self.outcomes.clear()
        self.attempts.clear()
        self.support_evidence.clear()
        self.current_rows = 0
        self.current_polygons = 0
        return batch


def _pool_attempt_row_count(
    polygon: PolygonSnapshot | None,
    candidates: tuple[LabelCandidate, ...],
    outcomes: tuple[LabelOutcome, ...],
    support_evidence: tuple[LabelSupportEvidence, ...],
) -> int:
    return 2 + int(polygon is not None) + len(candidates) + len(outcomes) + len(support_evidence)


def _pool_batch_is_full(accumulator: _PoolBatchAccumulator, row_count: int) -> bool:
    return (
        accumulator.current_rows + row_count > accumulator.limits.max_batch_rows
        or accumulator.current_polygons >= accumulator.max_polygons_per_batch
    )


@dataclass(frozen=True, slots=True)
class PoolAdapterCodePin:
    """Immutable adapter and shared-core revisions used for attempt results."""

    adapter_revision: str
    builder_revision: str

    def __post_init__(self) -> None:
        """Require both revisions before source rows can be processed."""

        for name, value in (
            ("adapter", self.adapter_revision),
            ("shared-core builder", self.builder_revision),
        ):
            if len(value) not in {40, 64} or any(
                character not in "0123456789abcdef" for character in value
            ):
                raise ValueError(f"{name} revision must be a full lowercase Git SHA")


@dataclass(frozen=True, slots=True)
class AttemptRunContext:
    """Pinned reference and paths shared while producing bounded shards."""

    raw_scan: RawInventoryScan
    reference: RasterReference
    reference_pin: ReferencePin
    taxonomy: EunisTaxonomy
    code_pin: PoolAdapterCodePin
    signature: str
    output_root: Path


@dataclass(frozen=True, slots=True)
class AttemptRowContext:
    """Resources needed to transform one raw row into contract records."""

    source_snapshot: SourceSnapshot
    reference: RasterReference
    reference_pin: ReferencePin
    taxonomy: EunisTaxonomy
    output_path: Path
    raw_checkpoint_uri: str


@dataclass(frozen=True, slots=True)
class _PreparedAttemptGeometry:
    projected: BaseGeometry | None
    exclusion_reason: str | None
    polygon: PolygonSnapshot | None
    projected_wkb: bytes | None


@dataclass(frozen=True, slots=True)
class _AttemptCheckpointData:
    inventory: PolygonInventoryRow
    geometry: _PreparedAttemptGeometry
    candidates: tuple[LabelCandidate, ...]
    outcomes: tuple[LabelOutcome, ...]
    receipt: PolygonAttemptReceipt
    support_evidence: tuple[LabelSupportEvidence, ...]
    score_diagnostics: Mapping[str, object] | None


@dataclass(frozen=True, slots=True)
class _AttemptSummaryData:
    raw_scan: RawInventoryScan
    reference_pin: ReferencePin
    taxonomy: EunisTaxonomy
    code_pin: PoolAdapterCodePin
    signature: str
    receipts: list[AttemptShardReceipt]
    total_rows: int
    attempt_counts: Counter[str]
    outcome_counts: Counter[str]
    candidate_counts: Counter[str]


class DuplicateNativeKeyError(ValueError):
    """Raised when a raw source contains duplicate native composite keys."""


def scan_raw_inventory(
    source: FrozenPolygonSource,
    checkpoint_root: Path,
    code_pin: PoolAdapterCodePin,
    *,
    batch_size: int = 4096,
    key_chunk_size: int = _CHUNK_KEYS,
) -> RawInventoryScan:
    """Enumerate every pinned raw polygon row and emit the core scan evidence.

    Source file hashes are verified before reading. Per-shard JSONL artifacts
    retain every native key, raw geometry payload, and source-row locator. The
    canonical sorted-key digest is computed with external sorting, so peak key
    memory is bounded by ``key_chunk_size`` rather than source row count.
    """

    _validate_raw_scan_limits(batch_size, key_chunk_size)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    raw_root = checkpoint_root / "raw-inventory"
    raw_root.mkdir(parents=True, exist_ok=True)
    fingerprint = _source_fingerprint(source, code_pin)
    receipts, key_count, keyset_sha256 = _scan_raw_shards(
        source,
        fingerprint,
        raw_root,
        checkpoint_root,
        batch_size,
        key_chunk_size,
    )

    row_count = sum(receipt.row_count for receipt in receipts)
    _validate_raw_scan_count(key_count, row_count)
    report = {
        "format_version": _RAW_CHECKPOINT_VERSION,
        "enumeration_rule": _ENUMERATION_RULE,
        "source_snapshot": asdict(source.source_snapshot),
        "source_layout": asdict(source.layout),
        "frozen_manifest_uri": source.manifest_uri,
        "frozen_manifest_sha256": source.manifest_sha256,
        "source_fingerprint": fingerprint,
        "adapter_revision": code_pin.adapter_revision,
        "builder_revision": code_pin.builder_revision,
        "polygon_shard_count": len(receipts),
        "raw_row_count": row_count,
        "unique_native_key_count": key_count,
        "native_keyset_sha256": keyset_sha256,
        "polygon_shards": [asdict(item) for item in receipts],
    }
    report_path = checkpoint_root / "raw-inventory-scan.json"
    report_bytes = _canonical_json_bytes(report)
    _write_atomic(report_path, report_bytes)
    evidence = RawInventoryEvidence(
        enumeration_rule=_ENUMERATION_RULE,
        scan_report_uri=report_path.resolve().as_uri(),
        scan_report_sha256=hashlib.sha256(report_bytes).hexdigest(),
        expected_key_count=key_count,
        expected_keyset_sha256=keyset_sha256,
    )
    return RawInventoryScan(source, code_pin, fingerprint, evidence, tuple(receipts))


def _validate_raw_scan_limits(batch_size: int, key_chunk_size: int) -> None:
    if batch_size <= 0 or key_chunk_size <= 0:
        raise ValueError("scan batch and key sort chunk sizes must be positive")


def _validate_raw_scan_count(key_count: int, row_count: int) -> None:
    if key_count != row_count:
        raise RuntimeError("external key digest count differs from enumerated raw polygon rows")


def _scan_raw_shards(
    source: FrozenPolygonSource,
    fingerprint: str,
    raw_root: Path,
    checkpoint_root: Path,
    batch_size: int,
    key_chunk_size: int,
) -> tuple[list[RawShardReceipt], int, str]:
    receipts: list[RawShardReceipt] = []
    with tempfile.TemporaryDirectory(prefix=".native-key-sort-", dir=checkpoint_root) as temp_dir:
        sorter = _ExternalKeyDigest(Path(temp_dir), key_chunk_size)
        for shard in sorted(source.polygon_shards, key=lambda item: item.path_in_repo):
            receipt = _scan_or_resume_shard(source, shard, fingerprint, raw_root, batch_size)
            receipts.append(receipt)
            for row in _read_jsonl(_local_file_uri_path(receipt.checkpoint_uri)):
                sorter.add(str(row["native_key_json"]))
        key_count, keyset_sha256 = sorter.finish()
    return receipts, key_count, keyset_sha256


def run_eunis_label_attempts(
    raw_scan: RawInventoryScan,
    reference: RasterReference,
    reference_pin: ReferencePin,
    taxonomy: EunisTaxonomy,
    checkpoint_root: Path,
) -> EunisAttemptRun:
    """Attempt L1/L2/L3 for every raw key and checkpoint each shard atomically."""

    _validate_eunis_reference_pin(reference_pin)
    _validate_reference_release(reference, reference_pin)
    _validate_classification_asset_inventory(reference, reference_pin)
    code_pin = raw_scan.code_pin
    _validate_attempt_taxonomy_inputs(raw_scan, reference_pin, taxonomy, code_pin, reference)
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    attempts_root = checkpoint_root / "eunis-label-attempts"
    attempts_root.mkdir(parents=True, exist_ok=True)
    signature = _attempt_signature(
        raw_scan,
        reference_pin,
        taxonomy,
        reference.layer_codes,
        code_pin,
    )
    context = AttemptRunContext(
        raw_scan,
        reference,
        reference_pin,
        taxonomy,
        code_pin,
        signature,
        attempts_root,
    )
    receipts = _run_attempt_shards(context, raw_scan.shards)

    summary = _attempt_summary(
        raw_scan,
        reference_pin,
        taxonomy,
        code_pin,
        signature,
        receipts,
    )
    report_path = checkpoint_root / "eunis-label-attempts-report.json"
    report_bytes = _canonical_json_bytes(summary)
    _write_atomic(report_path, report_bytes)
    return EunisAttemptRun(
        raw_scan,
        reference_pin,
        taxonomy,
        code_pin,
        signature,
        report_path.resolve().as_uri(),
        hashlib.sha256(report_bytes).hexdigest(),
        tuple(receipts),
    )


def _validate_reference_release(
    reference: RasterReference,
    reference_pin: ReferencePin,
) -> None:
    if reference.source_version != reference_pin.release_id:
        raise ValueError("raster layer source version does not match the pinned reference release")


def _validate_attempt_taxonomy_inputs(
    raw_scan: RawInventoryScan,
    reference_pin: ReferencePin,
    taxonomy: EunisTaxonomy,
    code_pin: PoolAdapterCodePin,
    reference: RasterReference,
) -> None:
    taxonomy.require_production_lineage(EEA_2021_CLASSIFICATION_LINEAGE_PIN)
    taxonomy.validate_direct_codes(reference.layer_codes)
    if raw_scan.source_fingerprint != _source_fingerprint(raw_scan.source, code_pin):
        raise ValueError("raw inventory source fingerprint no longer matches its pins")
    if (
        reference_pin.taxonomy_id != taxonomy.taxonomy_id
        or reference_pin.taxonomy_revision != taxonomy.pinned_revision
    ):
        raise ValueError("reference pin must match the hash-pinned EUNIS taxonomy mapping")
    _require_eunis_v12_artifact_and_area_pins(
        raw_scan.source.source_snapshot,
        reference_pin,
        purpose="label attempts",
    )


def _run_attempt_shards(
    context: AttemptRunContext,
    raw_shards: tuple[RawShardReceipt, ...],
) -> list[AttemptShardReceipt]:
    attempts_root = context.output_root
    receipts = []
    for raw_shard in raw_shards:
        output_path = attempts_root / f"{_safe_token(raw_shard.path_in_repo)}.jsonl"
        sidecar_path = output_path.with_suffix(".done.json")
        cached = _valid_attempt_checkpoint(
            output_path,
            sidecar_path,
            context.signature,
            raw_shard,
        )
        if cached is not None:
            receipts.append(cached)
            continue
        receipts.append(_write_attempt_shard(context, raw_shard))
    return receipts


def export_eunis_pool(  # noqa: PLR0913
    attempts: EunisAttemptRun,
    sink: FileArtifactSink,
    *,
    limits: StreamingLimits | None = None,
    staging_directory: Path | None = None,
    contract_version: str = "1.2.0",
    pool_version: str = "2.0.0",
    downstream_records: Iterable[JoinedTextRecord] | None = None,
) -> EunisPoolExport:
    """Finalize one EUNIS source pool from bounded per-shard attempt records.

    The adapter copies the small raw-census receipt and each attempt checkpoint
    through the streaming sink, then emits bounded ``PoolBatch`` instances.
    It never collects the full pool or Parquet payloads in memory. It can append
    downstream document/link/relevance records when supplied. This low-level
    path remains useful for fixtures and recovery; use
    :func:`export_eunis_full_pool` for a full pool with pinned joins. It never
    performs evaluation sampling.
    """

    _require_eunis_contract_v12_provenance(attempts, contract_version)
    selected_limits = limits or StreamingLimits()
    raw_report_bytes, attempt_report, checkpoint_paths = _preflight_export_evidence(attempts)
    artifact_root = _pool_artifact_root(
        text_source=attempts.raw_scan.source.layout.text_source.value,
        snapshot_id=attempts.raw_scan.source.source_snapshot.snapshot_id,
        contract_version=contract_version,
        pool_version=pool_version,
    )
    raw_report_uri, portable_attempt_uri, checkpoint_uris, checkpoint_uri_by_path = (
        _publish_adapter_evidence(
            attempts,
            sink,
            artifact_root,
            raw_report_bytes,
            attempt_report,
            checkpoint_paths,
        )
    )
    spec = _eunis_pool_stream_spec(
        attempts,
        raw_inventory_evidence=replace(
            attempts.raw_scan.evidence,
            scan_report_uri=raw_report_uri,
        ),
        contract_version=contract_version,
        pool_version=pool_version,
    )
    receipt = export_pool_batches(
        spec,
        _iter_eunis_pool_batches(
            attempts,
            checkpoint_uri_by_path,
            selected_limits,
            downstream_records=downstream_records,
        ),
        sink,
        limits=selected_limits,
        staging_directory=staging_directory,
    )
    return EunisPoolExport(
        receipt=receipt,
        raw_inventory_report_uri=raw_report_uri,
        label_attempt_report_uri=portable_attempt_uri,
        attempt_checkpoint_uris=tuple(checkpoint_uris),
        downstream_join_included=downstream_records is not None,
    )


def _preflight_export_evidence(
    attempts: EunisAttemptRun,
) -> tuple[bytes, dict[str, Any], dict[str, Path]]:
    raw_report_path = _verified_report_path(
        attempts.raw_scan.evidence.scan_report_uri,
        attempts.raw_scan.evidence.scan_report_sha256,
        "raw inventory scan report",
    )
    raw_report_bytes = raw_report_path.read_bytes()
    raw_report = _json_object(raw_report_bytes, "raw inventory scan report")
    _validate_raw_scan_report(attempts.raw_scan, raw_report)

    attempt_report_path = _verified_report_path(
        attempts.report_uri,
        attempts.report_sha256,
        "EUNIS label-attempt report",
    )
    attempt_report_bytes = attempt_report_path.read_bytes()
    attempt_report = _json_object(attempt_report_bytes, "EUNIS label-attempt report")
    _validate_attempt_report(attempts, attempt_report)

    checkpoint_paths = {
        shard.path_in_repo: _verified_report_path(
            shard.checkpoint_uri,
            shard.checkpoint_sha256,
            f"label-attempt checkpoint for {shard.path_in_repo}",
        )
        for shard in attempts.shards
    }
    return raw_report_bytes, attempt_report, checkpoint_paths


def _publish_adapter_evidence(
    attempts: EunisAttemptRun,
    sink: FileArtifactSink,
    artifact_root: str,
    raw_report_bytes: bytes,
    attempt_report: dict[str, Any],
    checkpoint_paths: dict[str, Path],
) -> tuple[str, str, list[tuple[str, str]], dict[str, str]]:
    evidence_root = f"{artifact_root}/adapter-evidence"
    raw_report_uri = sink.put(
        f"{evidence_root}/raw-inventory-scan.json",
        raw_report_bytes,
    )
    _require_resolvable_uri(raw_report_uri, "stored raw inventory report")

    checkpoint_uris = []
    checkpoint_uri_by_path: dict[str, str] = {}
    for shard in attempts.shards:
        uri = sink.put_file(
            f"{evidence_root}/label-attempts/{_safe_token(shard.path_in_repo)}.jsonl",
            checkpoint_paths[shard.path_in_repo],
        )
        _require_resolvable_uri(uri, f"stored label-attempt checkpoint {shard.path_in_repo}")
        if urlparse(uri).fragment:
            raise ValueError("streaming sink file URIs cannot already contain a fragment")
        checkpoint_uris.append((shard.path_in_repo, uri))
        checkpoint_uri_by_path[shard.path_in_repo] = uri

    portable_attempt_report = dict(attempt_report)
    portable_attempt_report["raw_inventory_scan_report_uri"] = raw_report_uri
    portable_attempt_report["shards"] = [
        {
            **record,
            "checkpoint_uri": checkpoint_uri_by_path[str(record["path_in_repo"])],
        }
        for record in attempt_report["shards"]
    ]
    portable_attempt_bytes = _canonical_json_bytes(portable_attempt_report)
    portable_attempt_uri = sink.put(
        f"{evidence_root}/label-attempts-report.json",
        portable_attempt_bytes,
    )
    _require_resolvable_uri(portable_attempt_uri, "stored EUNIS label-attempt report")

    return raw_report_uri, portable_attempt_uri, checkpoint_uris, checkpoint_uri_by_path


def iter_eunis_downstream_records(  # noqa: PLR0913
    attempts: EunisAttemptRun,
    text_table: EunisPinnedTable,
    relevance_table: EunisPinnedTable,
    *,
    wiki_link_table: EunisPinnedTable | None = None,
    wiki_document_tables: tuple[EunisPinnedTable, ...] = (),
    wiki_corpus: str | None = None,
    website_field_to_column: dict[str, str] | None = None,
    batch_size: int = 4096,
    staging_directory: Path | None = None,
    allocation_bounds: JoinAllocationBounds | None = None,
) -> BoundedJoinedTextRecords:
    """Join pinned downstream text and YES/NO rows against the full inventory."""

    return iter_pinned_eunis_join_records(
        attempts.raw_scan.source.layout.text_source,
        text_table,
        relevance_table,
        iter_eunis_attempt_inventory_rows(attempts),
        wiki_link_table=wiki_link_table,
        wiki_document_tables=wiki_document_tables,
        wiki_corpus=wiki_corpus,
        website_field_to_column=website_field_to_column,
        layout=attempts.raw_scan.source.layout,
        batch_size=batch_size,
        staging_directory=staging_directory,
        allocation_bounds=allocation_bounds,
    )


def iter_eunis_wiki_downstream_records(  # noqa: PLR0913
    attempts: EunisAttemptRun,
    wikipedia_text_table: EunisPinnedTable,
    wikipedia_relevance_table: EunisPinnedTable,
    wikivoyage_text_table: EunisPinnedTable,
    wikivoyage_relevance_table: EunisPinnedTable,
    wiki_link_table: EunisPinnedTable,
    *,
    wikipedia_document_tables: tuple[EunisPinnedTable, ...] = (),
    wikivoyage_document_tables: tuple[EunisPinnedTable, ...] = (),
    batch_size: int = 4096,
    staging_directory: Path | None = None,
    allocation_bounds: JoinAllocationBounds | None = None,
) -> BoundedJoinedTextRecords:
    """Join both pinned Wiki corpora against the same full raw inventory."""

    bounds = allocation_bounds or JoinAllocationBounds()
    wikipedia = iter_eunis_downstream_records(
        attempts,
        wikipedia_text_table,
        wikipedia_relevance_table,
        wiki_link_table=wiki_link_table,
        wiki_document_tables=wikipedia_document_tables,
        wiki_corpus="wikipedia",
        batch_size=batch_size,
        staging_directory=staging_directory,
        allocation_bounds=bounds,
    )
    wikivoyage = iter_eunis_downstream_records(
        attempts,
        wikivoyage_text_table,
        wikivoyage_relevance_table,
        wiki_link_table=wiki_link_table,
        wiki_document_tables=wikivoyage_document_tables,
        wiki_corpus="wikivoyage",
        batch_size=batch_size,
        staging_directory=staging_directory,
        allocation_bounds=bounds,
    )
    return BoundedJoinedTextRecords.combine(wikipedia, wikivoyage)


def iter_eunis_attempt_inventory_rows(
    attempts: EunisAttemptRun,
) -> Iterator[PolygonInventoryRow]:
    """Read each raw inventory row from verified per-shard attempt receipts."""

    total = 0
    for shard in attempts.shards:
        path = _verified_report_path(
            shard.checkpoint_uri,
            shard.checkpoint_sha256,
            f"EUNIS attempt checkpoint for {shard.path_in_repo}",
        )
        shard_count = 0
        for record in _read_jsonl(path):
            yield _inventory_from_json(record["inventory"])
            shard_count += 1
            total += 1
        if shard_count != shard.row_count:
            raise ValueError(f"attempt inventory changed for {shard.path_in_repo}")
    if total != attempts.raw_scan.evidence.expected_key_count:
        raise ValueError("attempt inventory rows differ from the raw source key count")


def export_eunis_full_pool(  # noqa: PLR0913
    attempts: EunisAttemptRun,
    downstream_records: BoundedJoinedTextRecords,
    sink: FileArtifactSink,
    *,
    limits: StreamingLimits | None = None,
    staging_directory: Path | None = None,
    contract_version: str = "1.2.0",
    pool_version: str = "2.0.0",
) -> EunisPoolExport:
    """Finalize a source-wide label pool together with its verified text joins."""

    attempts.taxonomy.require_production_lineage(EEA_2021_CLASSIFICATION_LINEAGE_PIN)
    if not isinstance(downstream_records, BoundedJoinedTextRecords):
        raise TypeError("full EUNIS export requires a bounds-checked pinned join stream")
    selected_limits = limits or StreamingLimits()
    expected_bounds = JoinAllocationBounds.from_streaming_limits(selected_limits)
    if downstream_records.allocation_bounds != expected_bounds:
        raise ValueError("join allocation bounds must match the selected streaming export limits")
    return export_eunis_pool(
        attempts,
        sink,
        limits=limits,
        staging_directory=staging_directory,
        contract_version=contract_version,
        pool_version=pool_version,
        downstream_records=downstream_records,
    )


def _require_eunis_contract_v12_provenance(
    attempts: EunisAttemptRun,
    contract_version: str,
) -> None:
    """Reject incomplete v1.2 evidence before copying any export artifacts."""

    if contract_version != "1.2.0":
        raise ValueError("EUNIS pool exports require contract version 1.2.0")
    evidence = attempts.raw_scan.evidence
    if evidence.expected_key_count <= 0:
        raise ValueError("contract 1.2 EUNIS export needs a verified non-empty raw inventory")
    _require_sha256(evidence.expected_keyset_sha256, "raw inventory key-set SHA-256")
    if not _has_uri(evidence.scan_report_uri):
        raise ValueError("contract 1.2 EUNIS export needs a pinned raw inventory scan report URI")
    _require_sha256(evidence.scan_report_sha256, "raw inventory scan report SHA-256")
    _require_eunis_v12_artifact_and_area_pins(
        attempts.raw_scan.source.source_snapshot,
        attempts.reference_pin,
        purpose="export",
    )


def _require_eunis_v12_artifact_and_area_pins(
    source_snapshot: SourceSnapshot,
    reference_pin: ReferencePin,
    *,
    purpose: str,
) -> None:
    """Require real source/reference artifact lists and the area method pin."""

    _require_artifact_manifest_pin(
        source_snapshot.source_artifact_manifest,
        "source_snapshot.source_artifact_manifest",
        purpose,
    )
    _require_artifact_manifest_pin(
        reference_pin.reference_artifact_manifest,
        "reference_pin.reference_artifact_manifest",
        purpose,
    )
    _require_eunis_area_pin(reference_pin.area_measurement, purpose)


def _require_artifact_manifest_pin(
    pin: ArtifactManifestPin | None,
    name: str,
    purpose: str,
) -> None:
    if pin is None:
        raise ValueError(f"contract 1.2 EUNIS {purpose} are missing {name}")
    if pin.format_version != "artifact-list-jsonl-v1":
        raise ValueError(f"contract 1.2 EUNIS {name} has an unsupported format")
    if not _has_uri(pin.manifest_uri):
        raise ValueError(f"contract 1.2 EUNIS {name} needs a manifest URI")
    if pin.expected_artifact_count <= 0:
        raise ValueError(f"contract 1.2 EUNIS {name} must include at least one artifact")
    _require_sha256(pin.manifest_sha256, f"contract 1.2 EUNIS {name} digest")


def _require_eunis_area_pin(area_pin: AreaMeasurementPin | None, purpose: str) -> None:
    if area_pin is None:
        raise ValueError(f"contract 1.2 EUNIS {purpose} are missing reference_pin.area_measurement")
    if any(
        spec.method_id != EUNIS_AREA_MEASUREMENT_ID or spec.output_unit != "m2"
        for spec in (area_pin.numerator, area_pin.denominator)
    ):
        raise ValueError("contract 1.2 EUNIS area measurement pin does not match the adapter")


def iter_attempt_checkpoint(uri: str) -> Iterator[dict[str, Any]]:
    """Stream one local JSONL label checkpoint for a future core batch loader."""

    if not uri.startswith("file://"):
        raise ValueError("source adapter checkpoint reader supports local file URIs only")
    yield from _read_jsonl(_local_file_uri_path(uri))


def read_projected_geometry_wkb(uri: str) -> bytes:
    """Recover one projected WKB geometry through its pinned checkpoint URI."""

    row = _read_checkpoint_fragment(uri, "projected_geometry_wkb")
    payload = row.get("projected_geometry_wkb_base64")
    if not isinstance(payload, str):
        raise TypeError("checkpoint row has no projected WKB geometry")
    return base64.b64decode(payload, validate=True)


def read_raw_geometry(uri: str) -> Any:
    """Recover one raw geometry payload through its raw-first checkpoint URI."""

    row = _read_checkpoint_fragment(uri, "geometry_payload")
    return _decode_geometry_payload(row["geometry_payload"])


def _eunis_pool_stream_spec(
    attempts: EunisAttemptRun,
    *,
    raw_inventory_evidence: RawInventoryEvidence,
    contract_version: str,
    pool_version: str,
) -> PoolStreamSpec:
    metadata_input = PoolBuildInput(
        pool_key=PoolKey(
            text_source=attempts.raw_scan.source.layout.text_source,
            reference_system=ReferenceSystem.EUNIS,
        ),
        source_snapshot=attempts.raw_scan.source.source_snapshot,
        raw_inventory_evidence=raw_inventory_evidence,
        reference_pin=attempts.reference_pin,
        adapter_id="osm-polygon-eunis/eunis-sourcewide-pool-adapter",
        adapter_revision=attempts.code_pin.adapter_revision,
        builder_revision=attempts.code_pin.builder_revision,
        inventory=(),
        polygons=(),
        levels=EUNIS_POOL_LEVELS,
        label_candidates=(),
        label_outcomes=(),
    )
    return PoolStreamSpec.from_build_input(
        metadata_input,
        contract_version=contract_version,
        pool_version=pool_version,
    )


def _validate_raw_scan_report(
    raw_scan: RawInventoryScan,
    report: dict[str, Any],
) -> None:
    evidence = raw_scan.evidence
    expected = {
        "enumeration_rule": evidence.enumeration_rule,
        "source_snapshot": asdict(raw_scan.source.source_snapshot),
        "source_fingerprint": raw_scan.source_fingerprint,
        "adapter_revision": raw_scan.code_pin.adapter_revision,
        "builder_revision": raw_scan.code_pin.builder_revision,
        "raw_row_count": evidence.expected_key_count,
        "unique_native_key_count": evidence.expected_key_count,
        "native_keyset_sha256": evidence.expected_keyset_sha256,
    }
    _require_report_fields(
        report,
        expected,
        "raw inventory report does not match its source evidence receipt",
    )
    observed = _index_report_shards(
        report,
        "polygon_shards",
        {shard.path_in_repo for shard in raw_scan.shards},
        "raw inventory report must include its polygon shard receipts",
        "raw inventory report shard set differs from the source census",
    )
    for shard in raw_scan.shards:
        _validate_raw_shard_receipt(observed[shard.path_in_repo], shard)


def _require_report_fields(
    report: dict[str, Any],
    expected: Mapping[str, Any],
    error_message: str,
) -> None:
    if any(report.get(key) != value for key, value in expected.items()):
        raise ValueError(error_message)


def _index_report_shards(
    report: dict[str, Any],
    field_name: str,
    expected_paths: set[str],
    type_error: str,
    set_error: str,
) -> dict[str, dict[str, Any]]:
    report_shards = report.get(field_name)
    if not isinstance(report_shards, list):
        raise TypeError(type_error)
    observed = {str(row.get("path_in_repo")): row for row in report_shards if isinstance(row, dict)}
    if set(observed) != expected_paths:
        raise ValueError(set_error)
    return observed


def _validate_raw_shard_receipt(
    row: dict[str, Any],
    shard: RawShardReceipt,
) -> None:
    if not _raw_shard_receipt_matches(row, shard):
        raise ValueError(f"raw inventory report receipt changed for {shard.path_in_repo}")


def _raw_shard_receipt_matches(row: dict[str, Any], shard: RawShardReceipt) -> bool:
    expected = {
        "source_sha256": shard.source_sha256,
        "source_size_bytes": shard.source_size_bytes,
        "row_count": shard.row_count,
        "checkpoint_sha256": shard.checkpoint_sha256,
    }
    return all(row.get(field) == value for field, value in expected.items())


def _validate_attempt_report(
    attempts: EunisAttemptRun,
    report: dict[str, Any],
) -> None:
    evidence = attempts.raw_scan.evidence
    expected = {
        "raw_inventory_scan_report_sha256": evidence.scan_report_sha256,
        "raw_inventory_key_count": evidence.expected_key_count,
        "raw_inventory_keyset_sha256": evidence.expected_keyset_sha256,
        "reference_pin": asdict(attempts.reference_pin),
        "taxonomy_id": attempts.taxonomy.taxonomy_id,
        "taxonomy_revision": attempts.taxonomy.taxonomy_revision,
        "taxonomy_sha256": attempts.taxonomy.taxonomy_sha256,
        **_taxonomy_lineage_report_fields(attempts.taxonomy),
        "adapter_revision": attempts.code_pin.adapter_revision,
        "builder_revision": attempts.code_pin.builder_revision,
        "signature": attempts.signature,
        "total_attempted_polygons": evidence.expected_key_count,
    }
    _require_report_fields(
        report,
        expected,
        "label-attempt report does not match its source or reference pins",
    )
    observed = _index_report_shards(
        report,
        "shards",
        {shard.path_in_repo for shard in attempts.shards},
        "label-attempt report must include its shard receipts",
        "label-attempt report shard set differs from its checkpoints",
    )
    for shard in attempts.shards:
        _validate_attempt_shard_receipt(observed[shard.path_in_repo], shard)
    if sum(shard.row_count for shard in attempts.shards) != evidence.expected_key_count:
        raise ValueError("label-attempt checkpoint rows differ from the raw inventory count")


def _validate_attempt_shard_receipt(
    row: dict[str, Any],
    shard: AttemptShardReceipt,
) -> None:
    error = f"label-attempt report receipt changed for {shard.path_in_repo}"
    if row.get("row_count") != shard.row_count:
        raise ValueError(error)
    _require_attempt_counts(row, "attempt_counts", shard.attempt_counts, error)
    if row.get("checkpoint_sha256") != shard.checkpoint_sha256:
        raise ValueError(error)
    _require_attempt_counts(row, "outcome_counts", shard.outcome_counts, error)
    _require_attempt_counts(row, "candidate_counts", shard.candidate_counts, error)


def _require_attempt_counts(
    row: dict[str, Any],
    field_name: str,
    expected: tuple[tuple[str, int], ...],
    error_message: str,
) -> None:
    actual = tuple((str(key), int(value)) for key, value in row[field_name])
    if actual != expected:
        raise ValueError(error_message)


def _iter_eunis_pool_batches(
    attempts: EunisAttemptRun,
    checkpoint_uris: dict[str, str],
    limits: StreamingLimits,
    *,
    downstream_records: Iterable[JoinedTextRecord] | None = None,
) -> Iterator[PoolBatch]:
    accumulator = _PoolBatchAccumulator(limits, _polygons_per_batch(attempts, limits))
    total_rows = sum(shard.row_count for shard in attempts.shards)
    _validate_attempt_inventory_count(attempts, total_rows)
    yield from _iter_attempt_pool_batches(attempts, checkpoint_uris, accumulator)
    final_batch = accumulator.flush()
    if final_batch is not None:
        yield final_batch
    if downstream_records is not None:
        yield from iter_eunis_join_batches(
            downstream_records,
            starting_batch_index=accumulator.batch_index,
            max_batch_rows=limits.max_batch_rows,
            max_batch_bytes=limits.max_batch_bytes,
        )


def _iter_attempt_pool_batches(
    attempts: EunisAttemptRun,
    checkpoint_uris: dict[str, str],
    accumulator: _PoolBatchAccumulator,
) -> Iterator[PoolBatch]:
    for shard in attempts.shards:
        durable_uri = checkpoint_uris.get(shard.path_in_repo)
        if durable_uri is None:
            raise ValueError(f"no durable attempt checkpoint for {shard.path_in_repo}")
        for row_number, record in _verified_attempt_shard_rows(shard):
            inventory, polygon, candidates, outcomes, attempt, support_evidence = _pool_batch_row(
                record,
                durable_uri,
                row_number,
            )
            completed = accumulator.add(
                inventory,
                polygon,
                candidates,
                outcomes,
                attempt,
                support_evidence,
            )
            if completed is not None:
                yield completed


def _validate_attempt_inventory_count(attempts: EunisAttemptRun, total_rows: int) -> None:
    if total_rows != attempts.raw_scan.evidence.expected_key_count:
        raise ValueError("streamed EUNIS attempts do not account for every raw source key")


def _polygons_per_batch(attempts: EunisAttemptRun, limits: StreamingLimits) -> int:
    direct_code_count = len(attempts.taxonomy.paths_by_direct_code)
    max_rows_per_polygon = 6 + 6 * direct_code_count
    if limits.max_batch_bytes < StreamingLimits().max_batch_bytes:
        return 1
    return max(1, limits.max_batch_rows // max_rows_per_polygon)


def _verified_attempt_shard_rows(
    shard: AttemptShardReceipt,
) -> Iterator[tuple[int, dict[str, Any]]]:
    source_path = _local_file_uri_path(shard.checkpoint_uri)
    row_count = 0
    for row_number, record in enumerate(_read_jsonl(source_path)):
        yield row_number, record
        row_count += 1
    if row_count != shard.row_count:
        raise ValueError(
            f"attempt checkpoint {shard.path_in_repo} has {row_count} rows, "
            f"expected {shard.row_count}"
        )


def _pool_batch_row(
    record: dict[str, Any],
    checkpoint_uri: str,
    row_number: int,
) -> tuple[
    PolygonInventoryRow,
    PolygonSnapshot | None,
    tuple[LabelCandidate, ...],
    tuple[LabelOutcome, ...],
    PolygonAttemptReceipt,
    tuple[LabelSupportEvidence, ...],
]:
    inventory = _inventory_from_json(record["inventory"])
    geometry_uri = f"{checkpoint_uri}#row={row_number}/projected_geometry_wkb"
    polygon = _polygon_from_json(record["polygon"], geometry_uri)
    candidates = tuple(_candidate_from_json(item) for item in record["candidates"])
    outcomes = tuple(_outcome_from_json(item) for item in record["outcomes"])
    attempt = _attempt_receipt_from_json(record["attempt_receipt"])
    support_evidence = tuple(
        _support_evidence_from_json(item) for item in record["support_evidence"]
    )
    _verify_projected_geometry(record, polygon)
    return inventory, polygon, candidates, outcomes, attempt, support_evidence


def _inventory_from_json(value: dict[str, Any]) -> PolygonInventoryRow:
    return PolygonInventoryRow(
        inventory_row_id=str(value["inventory_row_id"]),
        native_key=_native_key_from_json(str(value["native_key_json"])),
        source_row_locator=str(value["source_row_locator"]),
        polygon_snapshot_id=(
            str(value["polygon_snapshot_id"]) if value["polygon_snapshot_id"] else None
        ),
        exclusion_reason=(str(value["exclusion_reason"]) if value["exclusion_reason"] else None),
    )


def _polygon_from_json(
    value: dict[str, Any] | None,
    durable_geometry_uri: str,
) -> PolygonSnapshot | None:
    if value is None:
        return None
    return PolygonSnapshot(
        polygon_snapshot_id=str(value["polygon_snapshot_id"]),
        inventory_row_id=str(value["inventory_row_id"]),
        native_key=_native_key_from_json(str(value["native_key_json"])),
        geometry_uri=durable_geometry_uri,
        geometry_sha256=str(value["geometry_sha256"]),
        geometry_format=str(value["geometry_format"]),
        crs=str(value["crs"]),
        coordinate_transform=str(value["coordinate_transform"]),
        geometry_status=GeometryStatus(str(value["geometry_status"])),
        original_area=(
            float(value["original_area"]) if value["original_area"] is not None else None
        ),
        area_unit=str(value["area_unit"]),
    )


def _candidate_from_json(value: dict[str, Any]) -> LabelCandidate:
    return LabelCandidate(
        inventory_row_id=str(value["inventory_row_id"]),
        polygon_snapshot_id=str(value["polygon_snapshot_id"]),
        level=str(value["level"]),
        code=str(value["code"]),
        numerator=float(value["numerator"]),
        denominator=float(value["denominator"]),
        ratio=float(value["ratio"]),
        threshold=float(value["threshold"]),
        assignment_rule_version=str(value["assignment_rule_version"]),
        assignment_kind=AssignmentKind(str(value["assignment_kind"])),
        support_semantics=SupportSemantics(str(value["support_semantics"])),
        support_unit=str(value["support_unit"]),
        source_revision=str(value["source_revision"]),
        taxonomy_revision=str(value["taxonomy_revision"]),
    )


def _outcome_from_json(value: dict[str, Any]) -> LabelOutcome:
    return LabelOutcome(
        inventory_row_id=str(value["inventory_row_id"]),
        polygon_snapshot_id=(
            str(value["polygon_snapshot_id"]) if value["polygon_snapshot_id"] else None
        ),
        level=str(value["level"]),
        status=OutcomeStatus(str(value["status"])),
        assigned_code=str(value["assigned_code"]) if value["assigned_code"] else None,
        reason=str(value["reason"]) if value["reason"] else None,
        ambiguity_basis=(
            AmbiguityBasis(str(value["ambiguity_basis"])) if value["ambiguity_basis"] else None
        ),
    )


def _attempt_receipt_from_json(value: dict[str, Any]) -> PolygonAttemptReceipt:
    return PolygonAttemptReceipt(
        inventory_row_id=str(value["inventory_row_id"]),
        status=AttemptStatus(str(value["status"])),
        reference_id_namespace=str(value["reference_id_namespace"]),
        attempted_reference_ids=tuple(str(item) for item in value["attempted_reference_ids"]),
        missing_reference_ids=tuple(str(item) for item in value["missing_reference_ids"]),
        reason=str(value["reason"]) if value["reason"] else None,
    )


def _support_evidence_from_json(value: dict[str, Any]) -> LabelSupportEvidence:
    return LabelSupportEvidence(
        inventory_row_id=str(value["inventory_row_id"]),
        level=str(value["level"]),
        code=str(value["code"]),
        reference_id_namespace=str(value["reference_id_namespace"]),
        reference_ids=tuple(str(item) for item in value["reference_ids"]),
    )


def _verify_projected_geometry(
    record: dict[str, Any],
    polygon: PolygonSnapshot | None,
) -> None:
    payload = record.get("projected_geometry_wkb_base64")
    if polygon is None:
        if payload is not None:
            raise ValueError("attempt has projected WKB without a polygon snapshot")
        return
    if not isinstance(payload, str):
        raise TypeError("usable polygon attempt has no projected WKB")
    geometry = base64.b64decode(payload, validate=True)
    if hashlib.sha256(geometry).hexdigest() != polygon.geometry_sha256:
        raise ValueError("projected WKB hash differs from the polygon snapshot pin")


def _read_checkpoint_fragment(uri: str, expected_field: str) -> dict[str, Any]:
    path, fragment = _checkpoint_uri_parts(uri)
    row_number = _checkpoint_row_number(fragment, expected_field)
    for index, row in enumerate(_read_jsonl(path)):
        if index == row_number:
            return row
    raise IndexError(f"checkpoint row {row_number} does not exist")


def _checkpoint_uri_parts(uri: str) -> tuple[Path, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("source adapter checkpoint reader supports local file URIs only")
    return Path(unquote(parsed.path)), parsed.fragment


def _checkpoint_row_number(fragment_value: str, expected_field: str) -> int:
    fragment = fragment_value.split("/", maxsplit=1)
    if len(fragment) != _CHECKPOINT_URI_FRAGMENT_PARTS or fragment[1] != expected_field:
        raise ValueError(f"checkpoint URI must point to {expected_field}")
    row_prefix, separator, raw_index = fragment[0].partition("=")
    if row_prefix != "row" or not separator:
        raise ValueError("checkpoint URI fragment needs a row number")
    return int(raw_index)


def _scan_or_resume_shard(
    source: FrozenPolygonSource,
    shard: PinnedPolygonShard,
    source_fingerprint: str,
    raw_root: Path,
    batch_size: int,
) -> RawShardReceipt:
    token = _safe_token(shard.path_in_repo)
    checkpoint_path = raw_root / f"{token}.jsonl"
    sidecar_path = raw_root / f"{token}.done.json"
    cached = _valid_raw_checkpoint(checkpoint_path, sidecar_path, shard, source_fingerprint)
    if cached is not None:
        return cached
    _verify_local_shard(shard)
    row_count = 0
    temporary_path = checkpoint_path.with_name(f".{checkpoint_path.name}.partial")
    try:
        with temporary_path.open("wb") as output:
            parquet = pq.ParquetFile(shard.local_path)
            schema_names = set(parquet.schema_arrow.names)
            _ensure_required_raw_columns(shard.path_in_repo, schema_names)
            for batch in parquet.iter_batches(
                batch_size=batch_size,
                columns=[*_KEY_FIELDS, _GEOMETRY_FIELD],
                use_threads=False,
            ):
                for row in batch.to_pylist():
                    native_key = _native_key_from_row(row, shard.path_in_repo, row_count)
                    record = {
                        "native_key_json": native_key_json(native_key),
                        "inventory_row_id": inventory_row_identifier(
                            source.source_snapshot.snapshot_id,
                            native_key,
                        ),
                        "source_row_locator": f"{shard.path_in_repo}#row={row_count}",
                        "source_shard_path": shard.path_in_repo,
                        "source_row_ordinal": row_count,
                        "geometry_payload": _encode_geometry_payload(row[_GEOMETRY_FIELD]),
                    }
                    output.write(_canonical_json_bytes(record))
                    row_count += 1
            output.flush()
            os.fsync(output.fileno())
        temporary_path.replace(checkpoint_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise
    checkpoint_sha256 = _sha256_file(checkpoint_path)
    receipt = RawShardReceipt(
        shard.path_in_repo,
        shard.sha256,
        shard.size_bytes,
        row_count,
        checkpoint_path.resolve().as_uri(),
        checkpoint_sha256,
    )
    sidecar = {
        "format_version": _RAW_CHECKPOINT_VERSION,
        "source_fingerprint": source_fingerprint,
        "path_in_repo": shard.path_in_repo,
        "source_sha256": shard.sha256,
        "source_size_bytes": shard.size_bytes,
        "row_count": row_count,
        "checkpoint_uri": receipt.checkpoint_uri,
        "checkpoint_sha256": receipt.checkpoint_sha256,
    }
    _write_atomic(sidecar_path, _canonical_json_bytes(sidecar))
    return receipt


def _write_attempt_shard(
    context: AttemptRunContext,
    raw_shard: RawShardReceipt,
) -> AttemptShardReceipt:
    output_path = context.output_root / f"{_safe_token(raw_shard.path_in_repo)}.jsonl"
    sidecar_path = output_path.with_suffix(".done.json")
    raw_path = _local_file_uri_path(raw_shard.checkpoint_uri)
    temporary_path = output_path.with_name(f".{output_path.name}.partial")
    attempt_counts: Counter[str] = Counter()
    outcome_counts: Counter[str] = Counter()
    candidate_counts: Counter[str] = Counter()
    row_count = 0
    try:
        with temporary_path.open("wb") as output:
            for raw_row in _read_jsonl(raw_path):
                record = _attempt_raw_row(
                    raw_row,
                    AttemptRowContext(
                        context.raw_scan.source.source_snapshot,
                        context.reference,
                        context.reference_pin,
                        context.taxonomy,
                        output_path,
                        raw_shard.checkpoint_uri,
                    ),
                    row_count,
                )
                output.write(_canonical_json_bytes(record))
                row_count += 1
                attempt_counts[record["attempt_receipt"]["status"]] += 1
                for item in record["outcomes"]:
                    outcome_counts[f"{item['level']}:{item['status']}"] += 1
                for item in record["candidates"]:
                    candidate_counts[item["level"]] += 1
            output.flush()
            os.fsync(output.fileno())
        _require_attempt_row_count(row_count, raw_shard)
        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    receipt = AttemptShardReceipt(
        path_in_repo=raw_shard.path_in_repo,
        checkpoint_uri=output_path.resolve().as_uri(),
        checkpoint_sha256=_sha256_file(output_path),
        row_count=row_count,
        attempt_counts=tuple(sorted(attempt_counts.items())),
        outcome_counts=tuple(sorted(outcome_counts.items())),
        candidate_counts=tuple(sorted(candidate_counts.items())),
    )
    _write_atomic(
        sidecar_path,
        _canonical_json_bytes(
            {
                "format_version": _ATTEMPT_CHECKPOINT_VERSION,
                "signature": context.signature,
                "raw_checkpoint_sha256": raw_shard.checkpoint_sha256,
                "path_in_repo": raw_shard.path_in_repo,
                "checkpoint_uri": receipt.checkpoint_uri,
                "checkpoint_sha256": receipt.checkpoint_sha256,
                "row_count": row_count,
                "attempt_counts": receipt.attempt_counts,
                "outcome_counts": receipt.outcome_counts,
                "candidate_counts": receipt.candidate_counts,
            }
        ),
    )
    return receipt


def _attempt_raw_row(
    raw_row: dict[str, Any],
    context: AttemptRowContext,
    row_number: int,
) -> dict[str, Any]:
    key = _native_key_from_json(str(raw_row["native_key_json"]))
    inventory_id = str(raw_row["inventory_row_id"])
    locator = str(raw_row["source_row_locator"])
    geometry = _prepare_attempt_geometry(
        _decode_geometry_payload(raw_row["geometry_payload"]),
        context,
        key,
        inventory_id,
        row_number,
    )

    inventory = PolygonInventoryRow(
        inventory_row_id=inventory_id,
        native_key=key,
        source_row_locator=locator,
        polygon_snapshot_id=(
            geometry.polygon.polygon_snapshot_id if geometry.polygon is not None else None
        ),
        exclusion_reason=geometry.exclusion_reason,
    )
    support_groups = hierarchy_support_groups(context.taxonomy)
    support_areas, score_diagnostics, query_error = _attempt_support_areas(
        context.reference,
        geometry.projected,
        geometry.polygon,
        support_groups,
    )
    polygon_id = geometry.polygon.polygon_snapshot_id if geometry.polygon else None
    polygon_area = geometry.polygon.original_area if geometry.polygon else None
    candidates, outcomes = resolve_eunis_labels(
        EunisLabelAttempt(
            inventory_row_id=inventory_id,
            polygon_snapshot_id=polygon_id,
            original_area=polygon_area,
            support_areas=support_areas,
            taxonomy=context.taxonomy,
            source_revision=context.reference_pin.source_revision,
            score_diagnostics=score_diagnostics,
            query_error=query_error,
            exclusion_reason=geometry.exclusion_reason,
        )
    )
    attempt_receipt = _attempt_receipt(
        context,
        inventory_id,
        geometry.polygon,
        geometry.exclusion_reason,
        query_error,
    )
    support_evidence = _support_evidence_rows(candidates, support_groups)
    return _attempt_checkpoint_record(
        raw_row,
        context,
        _AttemptCheckpointData(
            inventory,
            geometry,
            candidates,
            outcomes,
            attempt_receipt,
            support_evidence,
            score_diagnostics,
        ),
    )


def _prepare_attempt_geometry(
    raw_geometry: Any,
    context: AttemptRowContext,
    key: NativeKey,
    inventory_id: str,
    row_number: int,
) -> _PreparedAttemptGeometry:
    projected, reason = _project_raw_geometry(raw_geometry)
    polygon, payload = _snapshot_projected_geometry(
        projected,
        context,
        key,
        inventory_id,
        row_number,
    )
    return _PreparedAttemptGeometry(projected, reason, polygon, payload)


def _project_raw_geometry(raw_geometry: Any) -> tuple[BaseGeometry | None, str | None]:
    if raw_geometry is None:
        return None, "missing_geometry"
    try:
        return _parse_project_geometry(raw_geometry)
    except _GEOMETRY_ERRORS as error:
        return None, f"geometry_transform_error:{type(error).__name__}"


def _parse_project_geometry(
    raw_geometry: Any,
) -> tuple[BaseGeometry | None, str | None]:
    parsed = parse_geometry(raw_geometry)
    if parsed is None:
        return None, "invalid_or_empty_polygon_geometry"
    projected = to_equal_area(parsed)
    reason = None if is_usable(projected) else "geometry_transform_failed_or_antimeridian_rejected"
    return projected, reason


def _snapshot_projected_geometry(
    projected: BaseGeometry | None,
    context: AttemptRowContext,
    key: NativeKey,
    inventory_id: str,
    row_number: int,
) -> tuple[PolygonSnapshot | None, bytes | None]:
    if projected is None:
        return None, None
    projected_wkb = to_wkb(projected, byte_order=1, output_dimension=2)
    geometry_sha256 = hashlib.sha256(projected_wkb).hexdigest()
    snapshot_id = polygon_snapshot_identifier(
        context.source_snapshot.snapshot_id,
        key,
        geometry_sha256,
    )
    geometry_uri = (
        f"{context.output_path.resolve().as_uri()}#row={row_number}/projected_geometry_wkb"
    )
    polygon = PolygonSnapshot(
        polygon_snapshot_id=snapshot_id,
        inventory_row_id=inventory_id,
        native_key=key,
        geometry_uri=geometry_uri,
        geometry_sha256=geometry_sha256,
        geometry_format="WKB",
        crs="EPSG:3035",
        coordinate_transform=(
            "EPSG:4326 to EPSG:3035; WGS84 edges segmentized at 0.01 degrees; "
            "antimeridian spans above 180 degrees rejected; geometry policy v4"
        ),
        geometry_status=GeometryStatus.USABLE,
        original_area=float(projected.area),
        area_unit="m2",
    )
    return polygon, projected_wkb


def _attempt_support_areas(
    reference: RasterReference,
    projected: BaseGeometry | None,
    polygon: PolygonSnapshot | None,
    support_groups: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[Mapping[tuple[str, str], float] | None, Mapping[str, object] | None, str | None]:
    if polygon is None or projected is None:
        return None, None, None
    try:
        result = reference.classification_areas(projected, support_groups)
        return result.areas_m2, result.diagnostics.as_dict(), None
    except _REFERENCE_ERRORS as error:
        return None, None, f"reference_query_error:{type(error).__name__}"


def _attempt_receipt(
    context: AttemptRowContext,
    inventory_id: str,
    polygon: PolygonSnapshot | None,
    geometry_reason: str | None,
    query_error: str | None,
) -> PolygonAttemptReceipt:
    if geometry_reason is not None:
        status = AttemptStatus.EXCLUDED
    elif query_error is not None:
        status = AttemptStatus.FAILED
    else:
        status = AttemptStatus.COMPLETED
    return PolygonAttemptReceipt(
        inventory_row_id=inventory_id,
        status=status,
        reference_id_namespace=_EEA_REFERENCE_ID_NAMESPACE,
        attempted_reference_ids=(
            tuple(sorted(context.reference.layer_codes)) if polygon is not None else ()
        ),
        reason=geometry_reason or query_error,
    )


def _support_evidence_rows(
    candidates: tuple[LabelCandidate, ...],
    support_groups: Mapping[tuple[str, str], tuple[str, ...]],
) -> tuple[LabelSupportEvidence, ...]:
    return tuple(
        LabelSupportEvidence(
            inventory_row_id=candidate.inventory_row_id,
            level=candidate.level,
            code=candidate.code,
            reference_id_namespace=_EEA_REFERENCE_ID_NAMESPACE,
            reference_ids=support_groups[(candidate.level, candidate.code)],
        )
        for candidate in candidates
    )


def _attempt_checkpoint_record(
    raw_row: dict[str, Any],
    context: AttemptRowContext,
    data: _AttemptCheckpointData,
) -> dict[str, Any]:
    return {
        "inventory": _inventory_json(data.inventory),
        "polygon": _polygon_json(data.geometry.polygon),
        "projected_geometry_wkb_base64": _projected_geometry_wkb_base64(
            data.geometry.projected_wkb
        ),
        "raw_geometry_checkpoint_uri": (
            f"{context.raw_checkpoint_uri}#row={raw_row['source_row_ordinal']}/geometry_payload"
        ),
        "candidates": [_candidate_json(candidate) for candidate in data.candidates],
        "outcomes": [_outcome_json(outcome) for outcome in data.outcomes],
        "attempt_receipt": _attempt_receipt_json(data.receipt),
        "support_evidence": [_support_evidence_json(item) for item in data.support_evidence],
        "score_diagnostics": (
            dict(data.score_diagnostics) if data.score_diagnostics is not None else None
        ),
    }


def _projected_geometry_wkb_base64(payload: bytes | None) -> str | None:
    return base64.b64encode(payload).decode("ascii") if payload else None


def _attempt_summary(
    raw_scan: RawInventoryScan,
    reference_pin: ReferencePin,
    taxonomy: EunisTaxonomy,
    code_pin: PoolAdapterCodePin,
    signature: str,
    receipts: list[AttemptShardReceipt],
) -> dict[str, Any]:
    total_rows = sum(receipt.row_count for receipt in receipts)
    attempts, outcomes, candidates = _aggregate_attempt_counters(receipts)
    _validate_supported_outcome_count(total_rows, outcomes)
    return _attempt_summary_payload(
        _AttemptSummaryData(
            raw_scan,
            reference_pin,
            taxonomy,
            code_pin,
            signature,
            receipts,
            total_rows,
            attempts,
            outcomes,
            candidates,
        )
    )


def _aggregate_attempt_counters(
    receipts: list[AttemptShardReceipt],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    attempts: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    candidates: Counter[str] = Counter()
    for receipt in receipts:
        attempts.update(dict(receipt.attempt_counts))
        outcomes.update(dict(receipt.outcome_counts))
        candidates.update(dict(receipt.candidate_counts))

    return attempts, outcomes, candidates


def _validate_supported_outcome_count(total_rows: int, outcomes: Counter[str]) -> None:
    expected_level_rows = total_rows * sum(level.supported for level in EUNIS_POOL_LEVELS)
    if sum(outcomes.values()) != expected_level_rows:
        raise RuntimeError("label-attempt report does not account for every supported level row")


def _attempt_summary_payload(data: _AttemptSummaryData) -> dict[str, Any]:
    return {
        "format_version": _ATTEMPT_CHECKPOINT_VERSION,
        "raw_inventory_scan_report_uri": data.raw_scan.evidence.scan_report_uri,
        "raw_inventory_scan_report_sha256": data.raw_scan.evidence.scan_report_sha256,
        "raw_inventory_key_count": data.raw_scan.evidence.expected_key_count,
        "raw_inventory_keyset_sha256": data.raw_scan.evidence.expected_keyset_sha256,
        "reference_pin": asdict(data.reference_pin),
        "taxonomy_id": data.taxonomy.taxonomy_id,
        "taxonomy_revision": data.taxonomy.taxonomy_revision,
        "taxonomy_sha256": data.taxonomy.taxonomy_sha256,
        **_taxonomy_lineage_report_fields(data.taxonomy),
        "assignment_rule_version": EUNIS_ASSIGNMENT_RULE_VERSION,
        "adapter_revision": data.code_pin.adapter_revision,
        "builder_revision": data.code_pin.builder_revision,
        "signature": data.signature,
        "total_attempted_polygons": data.total_rows,
        "attempt_counts_by_status": dict(sorted(data.attempt_counts.items())),
        "supported_levels": [level.name for level in EUNIS_POOL_LEVELS if level.supported],
        "outcome_counts_by_level_status": dict(sorted(data.outcome_counts.items())),
        "candidate_counts_by_level": dict(sorted(data.candidate_counts.items())),
        "shards": [asdict(receipt) for receipt in data.receipts],
    }


def _verified_report_path(uri: str, expected_sha256: str, description: str) -> Path:
    path = _local_file_uri_path(uri)
    if not path.is_file():
        raise FileNotFoundError(f"{description} is not available at {uri}")
    if _sha256_file(path) != expected_sha256:
        raise ValueError(f"{description} checksum differs from its receipt")
    return path


def _json_object(payload: bytes, description: str) -> dict[str, Any]:
    try:
        value = json.loads(payload)
    except json.JSONDecodeError as error:
        raise ValueError(f"{description} is not valid JSON") from error
    if not isinstance(value, dict):
        raise TypeError(f"{description} must be a JSON object")
    return value


def _pool_artifact_root(
    *,
    text_source: str,
    snapshot_id: str,
    contract_version: str,
    pool_version: str,
) -> str:
    return f"pools/{contract_version}/{text_source}/eunis/{snapshot_id}/{pool_version}"


def _require_resolvable_uri(uri: str, description: str) -> None:
    if not _has_uri(uri):
        raise ValueError(f"{description} did not return a resolvable URI")


class _ExternalKeyDigest:
    """Disk-backed exact digest matching the shared core's canonical key hash."""

    def __init__(self, directory: Path, chunk_size: int) -> None:
        self._directory = directory
        self._chunk_size = chunk_size
        self._keys: list[str] = []
        self._chunks: list[Path] = []

    def add(self, serialized_key: str) -> None:
        self._keys.append(serialized_key)
        if len(self._keys) >= self._chunk_size:
            self._flush()

    def finish(self) -> tuple[int, str]:
        if self._keys or not self._chunks:
            self._flush()
        chunks = _reduce_sorted_key_chunks(self._directory, self._chunks)
        return _digest_sorted_key_chunks(chunks)

    def _flush(self) -> None:
        if not self._keys and self._chunks:
            return
        self._keys.sort()
        target = self._directory / f"chunk-{len(self._chunks):08d}.txt"
        with target.open("w", encoding="utf-8", newline="\n") as output:
            for key in self._keys:
                output.write(key)
                output.write("\n")
        self._chunks.append(target)
        self._keys.clear()


def _reduce_sorted_key_chunks(directory: Path, chunks: list[Path]) -> list[Path]:
    merge_index = 0
    while len(chunks) > _MERGE_FAN_IN:
        merged = []
        for offset in range(0, len(chunks), _MERGE_FAN_IN):
            group = chunks[offset : offset + _MERGE_FAN_IN]
            target = directory / f"merge-{merge_index:06d}.txt"
            _merge_sorted_key_files(group, target)
            merged.append(target)
            merge_index += 1
        chunks = merged
    return chunks


def _digest_sorted_key_chunks(chunks: list[Path]) -> tuple[int, str]:
    streams: list[IO[str]] = []
    try:
        streams = [path.open(encoding="utf-8") for path in chunks]
        digest = hashlib.sha256()
        count = 0
        previous = None
        for current_line in heapq.merge(*(stream for stream in streams)):
            current_key = current_line.removesuffix("\n")
            _update_native_key_digest(digest, current_key, previous, count)
            previous = current_key
            count += 1
        return count, digest.hexdigest()
    finally:
        for stream in streams:
            stream.close()


def _update_native_key_digest(
    digest: Any,
    current_key: str,
    previous: str | None,
    count: int,
) -> None:
    if previous == current_key:
        raise DuplicateNativeKeyError(f"duplicate raw source-native polygon key {current_key}")
    if count:
        digest.update(b"\n")
    digest.update(current_key.encode("utf-8"))


def _merge_sorted_key_files(paths: list[Path], output_path: Path) -> None:
    streams: list[IO[str]] = []
    try:
        streams = [path.open(encoding="utf-8") for path in paths]
        with output_path.open("w", encoding="utf-8", newline="\n") as output:
            for key in heapq.merge(*(stream for stream in streams)):
                output.write(key)
    finally:
        for stream in streams:
            stream.close()


def _valid_raw_checkpoint(
    checkpoint_path: Path,
    sidecar_path: Path,
    shard: PinnedPolygonShard,
    source_fingerprint: str,
) -> RawShardReceipt | None:
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not _raw_sidecar_matches_source(sidecar, shard, source_fingerprint):
            return None
        if not _checkpoint_matches_sidecar(checkpoint_path, sidecar):
            return None
        return _raw_receipt_from_sidecar(sidecar, shard, checkpoint_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _raw_sidecar_matches_source(
    sidecar: dict[str, Any],
    shard: PinnedPolygonShard,
    source_fingerprint: str,
) -> bool:
    return (
        sidecar.get("format_version") == _RAW_CHECKPOINT_VERSION
        and sidecar.get("source_fingerprint") == source_fingerprint
        and sidecar.get("path_in_repo") == shard.path_in_repo
        and sidecar.get("source_sha256") == shard.sha256
        and sidecar.get("source_size_bytes") == shard.size_bytes
    )


def _checkpoint_matches_sidecar(checkpoint_path: Path, sidecar: dict[str, Any]) -> bool:
    return checkpoint_path.is_file() and _sha256_file(checkpoint_path) == sidecar.get(
        "checkpoint_sha256"
    )


def _raw_receipt_from_sidecar(
    sidecar: dict[str, Any],
    shard: PinnedPolygonShard,
    checkpoint_path: Path,
) -> RawShardReceipt:
    return RawShardReceipt(
        path_in_repo=shard.path_in_repo,
        source_sha256=shard.sha256,
        source_size_bytes=shard.size_bytes,
        row_count=int(sidecar["row_count"]),
        checkpoint_uri=checkpoint_path.resolve().as_uri(),
        checkpoint_sha256=str(sidecar["checkpoint_sha256"]),
    )


def _valid_attempt_checkpoint(
    checkpoint_path: Path,
    sidecar_path: Path,
    signature: str,
    raw_shard: RawShardReceipt,
) -> AttemptShardReceipt | None:
    try:
        sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        if not _attempt_sidecar_matches_raw(sidecar, signature, raw_shard):
            return None
        if not _checkpoint_matches_sidecar(checkpoint_path, sidecar):
            return None
        return _attempt_receipt_from_sidecar(sidecar, raw_shard, checkpoint_path)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def _attempt_sidecar_matches_raw(
    sidecar: dict[str, Any],
    signature: str,
    raw_shard: RawShardReceipt,
) -> bool:
    return (
        sidecar.get("format_version") == _ATTEMPT_CHECKPOINT_VERSION
        and sidecar.get("signature") == signature
        and sidecar.get("raw_checkpoint_sha256") == raw_shard.checkpoint_sha256
        and sidecar.get("path_in_repo") == raw_shard.path_in_repo
    )


def _attempt_receipt_from_sidecar(
    sidecar: dict[str, Any],
    raw_shard: RawShardReceipt,
    checkpoint_path: Path,
) -> AttemptShardReceipt:
    return AttemptShardReceipt(
        path_in_repo=raw_shard.path_in_repo,
        checkpoint_uri=checkpoint_path.resolve().as_uri(),
        checkpoint_sha256=str(sidecar["checkpoint_sha256"]),
        row_count=int(sidecar["row_count"]),
        attempt_counts=_count_pairs(sidecar["attempt_counts"]),
        outcome_counts=_count_pairs(sidecar["outcome_counts"]),
        candidate_counts=_count_pairs(sidecar["candidate_counts"]),
    )


def _count_pairs(values: Iterable[tuple[Any, Any]]) -> tuple[tuple[str, int], ...]:
    return tuple((str(key), int(value)) for key, value in values)


def _source_fingerprint(
    source: FrozenPolygonSource,
    code_pin: PoolAdapterCodePin,
) -> str:
    body = {
        "source_snapshot": asdict(source.source_snapshot),
        "layout": asdict(source.layout),
        "manifest_uri": source.manifest_uri,
        "manifest_sha256": source.manifest_sha256,
        "adapter_revision": code_pin.adapter_revision,
        "builder_revision": code_pin.builder_revision,
        "manifest_paths": sorted(source.manifest_paths),
        "shards": [
            {
                "path": shard.path_in_repo,
                "revision": shard.revision,
                "sha256": shard.sha256,
                "size_bytes": shard.size_bytes,
            }
            for shard in sorted(source.polygon_shards, key=lambda item: item.path_in_repo)
        ],
        "raw_checkpoint_version": _RAW_CHECKPOINT_VERSION,
        "enumeration_rule": _ENUMERATION_RULE,
    }
    return hashlib.sha256(_canonical_json_bytes(body)).hexdigest()


def _attempt_signature(
    raw_scan: RawInventoryScan,
    reference_pin: ReferencePin,
    taxonomy: EunisTaxonomy,
    direct_codes: tuple[str, ...],
    code_pin: PoolAdapterCodePin,
) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            {
                "raw_scan_sha256": raw_scan.evidence.scan_report_sha256,
                "reference_pin": asdict(reference_pin),
                "taxonomy_id": taxonomy.taxonomy_id,
                "taxonomy_revision": taxonomy.taxonomy_revision,
                "taxonomy_sha256": taxonomy.taxonomy_sha256,
                **_taxonomy_lineage_report_fields(taxonomy),
                "direct_codes": sorted(direct_codes),
                "assignment_rule_version": EUNIS_ASSIGNMENT_RULE_VERSION,
                "adapter_revision": code_pin.adapter_revision,
                "builder_revision": code_pin.builder_revision,
                "attempt_checkpoint_version": _ATTEMPT_CHECKPOINT_VERSION,
            }
        )
    ).hexdigest()


def _taxonomy_lineage_report_fields(taxonomy: EunisTaxonomy) -> dict[str, Any]:
    proof = taxonomy.classification_lineage
    return {
        "classification_lineage_pin": asdict(proof.lineage_pin) if proof is not None else None,
        "classification_taxonomy_mapping_sha256": (
            proof.taxonomy_mapping_sha256 if proof is not None else None
        ),
        "classification_paths_sha256": taxonomy.classification_paths_sha256,
        "classification_lineage_sha256": taxonomy.classification_lineage_sha256,
    }


def _validate_eunis_reference_pin(reference_pin: ReferencePin) -> None:
    _validate_eunis_support_policy(reference_pin)
    _validate_eunis_provenance(reference_pin)
    if reference_pin.release_id != EEA_2021_MAP_RELEASE:
        return
    _validate_official_eunis_revision(reference_pin.source_revision)


def _validate_official_eunis_revision(source_revision: str) -> None:
    revision_pins = _reference_revision_fields(source_revision)
    _require_revision_pin(
        revision_pins,
        "native-manifest-sha256",
        EEA_2021_NATIVE_MANIFEST_SHA256,
        "production EUNIS reference pin has the wrong native asset manifest",
    )
    _require_revision_pin(
        revision_pins,
        "asset-inventory-sha256",
        EEA_2021_NATIVE_ASSET_INVENTORY_SHA256,
        "production EUNIS reference pin has the wrong complete asset inventory",
    )
    if EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256 is None:
        raise ValueError(
            "production EUNIS label attempts require verified per-asset "
            "source-output semantics evidence"
        )
    _require_revision_pin(
        revision_pins,
        "source-output-semantics-evidence-sha256",
        EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256,
        "production EUNIS reference pin has the wrong source-output evidence",
    )


def _require_revision_pin(
    revision_pins: Mapping[str, str],
    field: str,
    expected: str,
    message: str,
) -> None:
    if revision_pins.get(field) != expected:
        raise ValueError(message)


def _validate_classification_asset_inventory(
    reference: object,
    reference_pin: ReferencePin,
) -> None:
    if not isinstance(reference, RasterReference):
        if reference_pin.release_id == EEA_2021_MAP_RELEASE:
            raise ValueError(
                "production EUNIS classification requires a RasterReference bound to its "
                "opened, pinned raster assets"
            )
        return
    revision_pins = _reference_revision_fields(reference_pin.source_revision)
    reference.validate_classification_inventory(
        expected_native_manifest_sha256=revision_pins.get("native-manifest-sha256"),
        expected_asset_inventory_sha256=revision_pins.get("asset-inventory-sha256"),
        expected_output_semantics_sha256=(
            revision_pins.get("source-output-semantics-evidence-sha256")
            if revision_pins.get("source-output-semantics-evidence-sha256") != "unverified"
            else None
        ),
    )


def _reference_revision_fields(source_revision: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for part in source_revision.split(";"):
        name, separator, value = part.partition(":")
        if separator and name:
            fields[name] = value
    return fields


def _validate_eunis_support_policy(reference_pin: ReferencePin) -> None:
    requirements = (
        (
            reference_pin.threshold,
            EUNIS_THRESHOLD,
            "EUNIS full-pool labels require the exact 0.8 area threshold",
            False,
        ),
        (
            reference_pin.support_union_policy,
            EUNIS_SUPPORT_AREA_POLICY,
            "EUNIS full-pool labels require the pinned highest-source-score area policy",
            False,
        ),
        (
            reference_pin.support_semantics,
            SupportSemantics.MODELED_SCORE_DERIVED_CLASSIFICATION,
            "EUNIS full-pool labels must declare a modeled score-derived classification",
            True,
        ),
        (
            reference_pin.support_unit,
            "m2",
            "EUNIS full-pool support and polygon denominator must use m2",
            False,
        ),
    )
    for actual, expected, message, require_identity in requirements:
        if not _eunis_policy_values_match(actual, expected, require_identity):
            raise ValueError(message)


def _eunis_policy_values_match(actual: Any, expected: Any, require_identity: bool) -> bool:
    return actual is expected if require_identity else actual == expected


def _validate_eunis_provenance(reference_pin: ReferencePin) -> None:
    _validate_reference_provenance_fields(reference_pin)
    _validate_optional_eunis_area_pin(reference_pin.area_measurement)


def _validate_reference_provenance_fields(reference_pin: ReferencePin) -> None:
    for field_name, value in (
        ("reference source repository", reference_pin.source_repository),
        ("reference source revision", reference_pin.source_revision),
        ("reference release", reference_pin.release_id),
        ("taxonomy id", reference_pin.taxonomy_id),
        ("taxonomy revision", reference_pin.taxonomy_revision),
    ):
        if not value.strip():
            raise ValueError(f"{field_name} must be pinned")


def _validate_optional_eunis_area_pin(area_pin: AreaMeasurementPin | None) -> None:
    if area_pin is None:
        return
    if any(
        measure.method_id != EUNIS_AREA_MEASUREMENT_ID or measure.output_unit != "m2"
        for measure in (area_pin.numerator, area_pin.denominator)
    ):
        raise ValueError("EUNIS area measurement pin does not match the pinned EPSG:3035 method")


def _native_key_from_row(row: dict[str, Any], shard_path: str, row_ordinal: int) -> NativeKey:
    source_pbf = row.get("source_pbf")
    osm_type = row.get("osm_type")
    osm_id = row.get("osm_id")
    if (
        not _is_nonempty_string(source_pbf)
        or not _is_nonempty_string(osm_type)
        or type(osm_id) is not int
    ):
        raise ValueError(
            f"raw source row {shard_path}#row={row_ordinal} lacks the complete "
            "(source_pbf, osm_type, osm_id) native key"
        )
    return canonical_native_key(
        {
            "source_pbf": cast(str, source_pbf),
            "osm_type": cast(str, osm_type),
            "osm_id": cast(int, osm_id),
        }
    )


def _is_nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _native_key_from_json(serialized: str) -> NativeKey:
    values = json.loads(serialized)
    if not isinstance(values, list):
        raise TypeError("checkpoint native key must be a JSON pair list")
    return tuple((str(item[0]), item[1]) for item in values)


def _encode_geometry_payload(value: Any) -> dict[str, Any]:
    if value is None:
        return {"kind": "missing"}
    if isinstance(value, (bytes, bytearray, memoryview)):
        return {"kind": "base64", "value": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, str):
        return {"kind": "text", "value": value}
    if isinstance(value, (dict, list, int, float, bool)):
        return {"kind": "json", "value": value}
    return {"kind": "unsupported", "value": type(value).__name__}


def _decode_geometry_payload(payload: dict[str, Any]) -> Any:
    match payload.get("kind"):
        case "missing" | "unsupported":
            return None
        case "base64":
            return base64.b64decode(str(payload["value"]), validate=True)
        case "text":
            return str(payload["value"])
        case "json":
            return payload["value"]
        case _:
            raise ValueError("raw geometry checkpoint has an unknown payload encoding")


def _inventory_json(row: PolygonInventoryRow) -> dict[str, Any]:
    return {
        "inventory_row_id": row.inventory_row_id,
        "native_key_json": native_key_json(row.native_key),
        "source_row_locator": row.source_row_locator,
        "polygon_snapshot_id": row.polygon_snapshot_id,
        "exclusion_reason": row.exclusion_reason,
    }


def _polygon_json(row: PolygonSnapshot | None) -> dict[str, Any] | None:
    if row is None:
        return None
    result = asdict(row)
    result["native_key_json"] = native_key_json(row.native_key)
    result.pop("native_key")
    result["geometry_status"] = row.geometry_status.value
    return result


def _candidate_json(row: LabelCandidate) -> dict[str, Any]:
    result = asdict(row)
    result["assignment_kind"] = row.assignment_kind.value
    result["support_semantics"] = row.support_semantics.value
    return result


def _attempt_receipt_json(row: PolygonAttemptReceipt) -> dict[str, Any]:
    result = asdict(row)
    result["status"] = row.status.value
    return result


def _support_evidence_json(row: LabelSupportEvidence) -> dict[str, Any]:
    return asdict(row)


def _outcome_json(row: LabelOutcome) -> dict[str, Any]:
    result = asdict(row)
    result["status"] = row.status.value
    result["ambiguity_basis"] = row.ambiguity_basis.value if row.ambiguity_basis else None
    return result


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid checkpoint JSON at {path}:{line_number}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"checkpoint row must be an object at {path}:{line_number}")
                yield record


def _verify_local_shard(shard: PinnedPolygonShard) -> None:
    _verify_local_shard_file(shard)
    _verify_local_shard_tree_identity(shard)


def _verify_local_shard_file(shard: PinnedPolygonShard) -> None:
    if not shard.local_path.is_file():
        raise FileNotFoundError(f"pinned raw shard is not staged: {shard.path_in_repo}")
    if shard.local_path.stat().st_size != shard.size_bytes:
        raise ValueError(f"staged raw shard size differs from its manifest: {shard.path_in_repo}")
    if _sha256_file(shard.local_path) != shard.sha256:
        raise ValueError(
            f"staged raw shard checksum differs from its manifest: {shard.path_in_repo}"
        )


def _verify_local_shard_tree_identity(shard: PinnedPolygonShard) -> None:
    if shard.lfs_oid is not None:
        if shard.sha256 != shard.lfs_oid or shard.size_bytes != shard.lfs_size:
            raise ValueError(
                f"staged raw LFS shard differs from its tree pin: {shard.path_in_repo}"
            )
    elif _git_blob_sha1(shard.local_path) != shard.blob_id:
        raise ValueError(
            f"staged raw shard Git blob id differs from its tree pin: {shard.path_in_repo}"
        )


def _local_file_uri_path(uri: str) -> Path:
    parsed = urlparse(uri)
    if parsed.scheme != "file":
        raise ValueError("raw inventory checkpoints must have file URIs")
    return Path(unquote(parsed.path))


def _safe_token(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return f"{payload}\n".encode()


def _write_atomic(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.partial")
    with temporary.open("wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    """Return the Git blob SHA-1 for a file without loading it into memory."""

    size = path.stat().st_size
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {size}\0".encode())
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _require_sha256(value: str | None, description: str) -> None:
    if not isinstance(value, str) or _SHA256_HEX_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{description} must be lowercase hexadecimal")


def _matches_glob(path: str, pattern: str) -> bool:
    path_parts = Path(path).parts
    pattern_parts = Path(pattern).parts
    if len(path_parts) != len(pattern_parts):
        return False
    return all(
        fnmatchcase(path_part, pattern_part)
        for path_part, pattern_part in zip(path_parts, pattern_parts, strict=True)
    )


def _has_uri(value: str | None) -> bool:
    if not isinstance(value, str):
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.path))


def _validate_frozen_polygon_source(source: FrozenPolygonSource) -> None:
    _validate_frozen_source_identity(source)
    _validate_manifest_paths(source)
    _validate_shard_pins(source)


def _validate_frozen_source_identity(source: FrozenPolygonSource) -> None:
    _validate_source_repository_revision(source)
    _validate_source_manifest_provenance(source)


def _validate_source_repository_revision(source: FrozenPolygonSource) -> None:
    _require_source_repository(source)
    revision = source.source_snapshot.revision
    _require_full_source_revision(revision)
    _require_pinned_source_revision(revision, source.layout.raw_revision)


def _require_source_repository(source: FrozenPolygonSource) -> None:
    if source.source_snapshot.repository != source.layout.repository:
        raise ValueError("source snapshot repository does not match its EUNIS layout")


def _require_full_source_revision(revision: str) -> None:
    if len(revision) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in revision
    ):
        raise ValueError("raw polygon source revision must be a full lowercase Git SHA")


def _require_pinned_source_revision(revision: str, pinned_revision: str) -> None:
    if revision != pinned_revision:
        raise ValueError("raw polygon source revision does not match the pinned EUNIS layout")


def _validate_source_manifest_provenance(source: FrozenPolygonSource) -> None:
    _require_sha256(source.source_snapshot.pbf_sha256, "source PBF or PBF-manifest SHA-256")
    if not _has_uri(source.source_snapshot.pbf_uri):
        raise ValueError("source PBF or PBF-manifest URI must be resolvable")
    if not _has_uri(source.manifest_uri):
        raise ValueError("frozen source manifest needs a resolvable URI")
    _require_sha256(source.manifest_sha256, "frozen source manifest SHA-256")


def _validate_manifest_paths(source: FrozenPolygonSource) -> None:
    _validate_manifest_path_inventory(source.manifest_paths)
    expected = _manifest_polygon_paths(source)
    actual = {shard.path_in_repo for shard in source.polygon_shards}
    if not expected or actual != expected:
        raise ValueError(
            "pinned polygon shards must exactly equal the frozen manifest paths matching "
            f"{source.layout.polygon_glob!r}: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )
    _validate_unique_polygon_shards(source.polygon_shards, actual)


def _validate_manifest_path_inventory(paths: tuple[str, ...]) -> None:
    if not paths:
        raise ValueError("frozen source manifest must list repository files")
    if len(set(paths)) != len(paths):
        raise ValueError("frozen source manifest contains duplicate paths")


def _manifest_polygon_paths(source: FrozenPolygonSource) -> set[str]:
    return {
        path for path in source.manifest_paths if _matches_glob(path, source.layout.polygon_glob)
    }


def _validate_unique_polygon_shards(
    shards: tuple[PinnedPolygonShard, ...],
    actual_paths: set[str],
) -> None:
    if len(actual_paths) != len(shards):
        raise ValueError("frozen source contains duplicate polygon shard pins")


def _validate_shard_pins(source: FrozenPolygonSource) -> None:
    pin = source.layout.raw_polygon_tree_pin
    if pin.repository != source.layout.repository or pin.revision != source.layout.raw_revision:
        raise ValueError("raw polygon tree pin does not match its EUNIS source layout")
    pinned_files = [_source_tree_file_for_shard(source, shard) for shard in source.polygon_shards]
    validate_source_tree_group(pin, pinned_files)


def _source_tree_file_for_shard(
    source: FrozenPolygonSource,
    shard: PinnedPolygonShard,
) -> SourceTreeFile:
    if shard.revision != source.source_snapshot.revision:
        raise ValueError(f"polygon shard {shard.path_in_repo} has another revision")
    _require_sha256(shard.sha256, f"polygon shard {shard.path_in_repo} SHA-256")
    if shard.size_bytes < 0:
        raise ValueError("polygon shard size cannot be negative")
    return SourceTreeFile(
        path=shard.path_in_repo,
        size_bytes=shard.size_bytes,
        blob_id=shard.blob_id,
        lfs_oid=shard.lfs_oid,
        lfs_size=shard.lfs_size,
    )


def _ensure_required_raw_columns(shard_path: str, schema_names: set[str]) -> None:
    missing = sorted({*_KEY_FIELDS, _GEOMETRY_FIELD} - schema_names)
    if missing:
        raise ValueError(f"{shard_path} missing raw columns {missing}")


def _require_attempt_row_count(row_count: int, shard: RawShardReceipt) -> None:
    if row_count != shard.row_count:
        raise RuntimeError(
            f"attempted {row_count} rows from {shard.path_in_repo}, expected {shard.row_count}"
        )
