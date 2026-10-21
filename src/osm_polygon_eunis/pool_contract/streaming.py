"""Bounded, restart-safe export for source-wide polygon-label pools."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass, fields, is_dataclass
from enum import Enum
from itertools import islice
from pathlib import Path
from typing import Protocol, TypeGuard, cast
from urllib.parse import quote, unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq

from .builder import (
    build_full_pool,
    inventory_row_identifier,
    native_key_json,
    polygon_snapshot_identifier,
)
from .export import (
    _TABLE_FIELDS,
    _area_measurement_dict,
    _artifact_manifest_dict,
    _canonical_json_bytes,
    _canonical_json_string_array,
    table_schemas,
)
from .models import (
    AmbiguityBasis,
    AssignmentKind,
    AttemptStatus,
    BodyStatus,
    EligibilityStatus,
    EntityLink,
    GateReport,
    GeometryStatus,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    LabelSupportEvidence,
    OutcomeStatus,
    PolygonAttemptReceipt,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBuildInput,
    PoolBundle,
    PoolEligibility,
    PoolKey,
    PoolValidationReport,
    RawInventoryEvidence,
    ReferencePin,
    ReferenceSystem,
    RelevanceAssessment,
    RelevanceStatus,
    SourceDocument,
    SourceSnapshot,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
)
from .validation import _valid_uri, validate_full_pool

DEFAULT_MAX_BATCH_ROWS = 2_048
DEFAULT_MAX_BATCH_BYTES = 32 * 1024 * 1024
MAX_BATCH_ROWS = 10_000
MAX_BATCH_BYTES = 64 * 1024 * 1024
MAX_ROW_BYTES = 1024 * 1024
MAX_SHARD_ROWS = 8_192
MAX_SHARD_BYTES = 64 * 1024 * 1024
_STORAGE_LAYOUT_VERSION = "sharded-parquet-v1"
_ARTIFACT_MANIFEST_FORMAT = "artifact-list-jsonl-v1"
_GATE_PARQUET_BATCH_SIZE = 2_048
_READINESS_EXAMPLE_LIMIT = 3
_KEY_SHA_EMPTY = hashlib.sha256(b"").hexdigest()
_SQLITE_CACHE_KIB = 4_096
_SHA256_LENGTH = 64
_KEY_COMPONENT_WIDTH = 2
_POOL_ID_PARTS = 3
_FIRST_SHARD_INDEX = 0
_MIN_AMBIGUOUS_COUNT = 2
_MAX_DIAGNOSTICS = 100
_POOL_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class PoolStreamSpec:
    """Pool-wide immutable pins supplied before any rows are streamed."""

    pool_key: PoolKey
    source_snapshot: SourceSnapshot
    raw_inventory_evidence: RawInventoryEvidence
    reference_pin: ReferencePin
    adapter_id: str
    adapter_revision: str
    builder_revision: str
    levels: tuple[LabelLevel, ...]
    contract_version: str = "1.2.0"
    pool_version: str = "1.0.0"

    @classmethod
    def from_build_input(
        cls,
        source: PoolBuildInput,
        *,
        contract_version: str = "1.2.0",
        pool_version: str = "1.0.0",
    ) -> PoolStreamSpec:
        """Create stream metadata from an existing adapter build input."""

        return cls(
            pool_key=source.pool_key,
            source_snapshot=source.source_snapshot,
            raw_inventory_evidence=source.raw_inventory_evidence,
            reference_pin=source.reference_pin,
            adapter_id=source.adapter_id,
            adapter_revision=source.adapter_revision,
            builder_revision=source.builder_revision,
            levels=source.levels,
            contract_version=contract_version,
            pool_version=pool_version,
        )


@dataclass(frozen=True, slots=True)
class PoolBatch:
    """One bounded group of inventory, geometry, labels, and lineage rows."""

    batch_index: int
    inventory: tuple[PolygonInventoryRow, ...] = ()
    polygons: tuple[PolygonSnapshot, ...] = ()
    label_candidates: tuple[LabelCandidate, ...] = ()
    label_outcomes: tuple[LabelOutcome, ...] = ()
    documents: tuple[SourceDocument, ...] = ()
    text_polygon_links: tuple[TextPolygonLink, ...] = ()
    entity_links: tuple[EntityLink, ...] = ()
    eligibility: tuple[PoolEligibility, ...] = ()
    attempt_receipts: tuple[PolygonAttemptReceipt, ...] = ()
    support_evidence: tuple[LabelSupportEvidence, ...] = ()
    relevance_assessments: tuple[RelevanceAssessment, ...] = ()

    @property
    def row_count(self) -> int:
        """Count all staged contract rows in this batch."""

        contract_rows = sum(
            len(rows)
            for rows in (
                self.inventory,
                self.polygons,
                self.label_candidates,
                self.label_outcomes,
                self.documents,
                self.text_polygon_links,
                self.entity_links,
                self.eligibility,
                self.attempt_receipts,
                self.support_evidence,
                self.relevance_assessments,
            )
        )
        link_candidates = sum(
            len(link.candidate_inventory_row_ids) for link in self.text_polygon_links
        )
        return contract_rows + link_candidates


@dataclass(frozen=True, slots=True)
class StreamingLimits:
    """Hard bounds for staged rows, JSON records, diagnostics, and shards."""

    max_batch_rows: int = DEFAULT_MAX_BATCH_ROWS
    max_batch_bytes: int = DEFAULT_MAX_BATCH_BYTES
    max_row_bytes: int = MAX_ROW_BYTES
    max_shard_rows: int = MAX_SHARD_ROWS
    max_shard_bytes: int = MAX_SHARD_BYTES
    max_diagnostics: int = _MAX_DIAGNOSTICS


@dataclass(frozen=True, slots=True)
class StreamingLevelAccounting:
    """Bounded per-level counts with exclusion detail stored in a sidecar."""

    level: str
    supported: bool
    unsupported_reason: str | None
    total_source_keys: int
    processed_decisions: int
    qualifying_candidates: int
    assigned: int
    ambiguous: int
    unassigned: int
    excluded: int
    exclusion_reason_group_count: int


@dataclass(frozen=True, slots=True)
class _ManifestArtifacts:
    """Small set of immutable file receipts added to the completion manifest."""

    shard_manifest_path: str
    shard_manifest_sha256: str
    inventory_keyset_sha256: str
    exclusion_reason_path: str
    exclusion_reason_sha256: str
    exclusion_reason_group_count: int


@dataclass(frozen=True, slots=True)
class StreamingExportReceipt:
    """Completion marker and inventory receipt for a streamed pool."""

    pool_id: str
    manifest_uri: str
    manifest_sha256: str
    shard_manifest_uri: str
    shard_manifest_sha256: str
    shard_count: int
    total_source_keys: int
    inventory_keyset_sha256: str
    level_accounting: tuple[StreamingLevelAccounting, ...]


class FileArtifactSink(Protocol):
    """Persist immutable artifacts, copying shard files without buffering them."""

    def put(self, relative_path: str, content: bytes) -> str:
        """Persist a small artifact such as the final completion manifest."""

    def put_file(self, relative_path: str, source_path: Path) -> str:
        """Atomically persist a file and permit only identical retries."""


class PoolArtifactReader(Protocol):
    """Read pinned artifact snapshots, inspect Parquet, and read bounded rows."""

    def read_bytes(self, uri: str) -> bytes:
        """Read a small completion manifest from its URI."""

    def iter_lines(self, uri: str) -> Iterable[bytes]:
        """Stream lines from a JSONL manifest or sidecar without materializing it."""

    def iter_verified_lines(self, uri: str, expected_sha256: str) -> Iterable[bytes]:
        """Yield JSONL lines from the exact snapshot whose bytes match the pin."""

    def sha256(self, uri: str) -> str:
        """Return the content SHA-256, streaming large artifacts when needed."""

    def parquet_metadata(self, uri: str) -> ParquetArtifactMetadata:
        """Read one Parquet footer without materializing row-group contents."""

    def iter_parquet_rows(
        self,
        uri: str,
        columns: tuple[str, ...],
        *,
        batch_size: int,
    ) -> Iterable[dict[str, object]]:
        """Yield selected Parquet columns in batches no larger than ``batch_size``."""

    def iter_verified_parquet_rows(
        self,
        uri: str,
        expected_sha256: str,
        columns: tuple[str, ...],
        *,
        batch_size: int,
    ) -> Iterable[dict[str, object]]:
        """Yield bounded rows from the exact Parquet snapshot whose bytes match the pin."""

    def resolve_uri(self, manifest_uri: str, relative_path: str) -> str:
        """Resolve a sink-relative artifact path beside the given pool manifest."""


@dataclass(frozen=True, slots=True)
class ParquetArtifactMetadata:
    """Footer metadata needed to check one shard's schema and row count."""

    row_count: int
    schema: pa.Schema


@dataclass(frozen=True, slots=True)
class FilesystemStreamingSink:
    """Atomic local sink for pool shards, manifests, and retry tests."""

    root: Path

    def put(self, relative_path: str, content: bytes) -> str:
        """Write small manifest bytes through the immutable file path."""

        self.root.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=self.root, delete=False) as handle:
            handle.write(content)
            source_path = Path(handle.name)
        try:
            return self.put_file(relative_path, source_path)
        finally:
            source_path.unlink(missing_ok=True)

    def put_file(self, relative_path: str, source_path: Path) -> str:
        """Copy through a same-directory temporary file before atomic linking."""

        target = self.root / relative_path
        target.resolve().relative_to(self.root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary_name = tempfile.mkstemp(
            dir=target.parent,
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "wb") as destination, source_path.open("rb") as source:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
                destination.flush()
                os.fsync(destination.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError as existing:
                if _sha256_file(target) != _sha256_file(temporary):
                    raise FileExistsError(
                        f"immutable pool artifact differs at {relative_path}"
                    ) from existing
            return target.resolve().as_uri()
        finally:
            temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class FilesystemArtifactReader:
    """Filesystem reader used by streamed gate verification and tests."""

    def read_bytes(self, uri: str) -> bytes:
        """Read a small manifest from a local file URI."""

        return self._path(uri).read_bytes()

    def iter_lines(self, uri: str) -> Iterable[bytes]:
        """Yield one line at a time from a local JSONL artifact."""

        with self._path(uri).open("rb") as source:
            yield from source

    def iter_verified_lines(self, uri: str, expected_sha256: str) -> Iterable[bytes]:
        """Verify a disk snapshot before yielding its JSONL records."""

        with (
            self._verified_snapshot(uri, expected_sha256) as snapshot_path,
            snapshot_path.open("rb") as snapshot,
        ):
            yield from snapshot

    def sha256(self, uri: str) -> str:
        """Hash a local artifact in bounded chunks."""

        return _sha256_file(self._path(uri))

    def parquet_metadata(self, uri: str) -> ParquetArtifactMetadata:
        """Inspect the footer of one local Parquet shard."""

        parquet_file = pq.ParquetFile(self._path(uri))
        return ParquetArtifactMetadata(
            row_count=parquet_file.metadata.num_rows,
            schema=parquet_file.schema_arrow,
        )

    def iter_parquet_rows(
        self,
        uri: str,
        columns: tuple[str, ...],
        *,
        batch_size: int,
    ) -> Iterable[dict[str, object]]:
        """Yield selected rows through bounded Arrow record batches."""

        if batch_size <= 0:
            raise ValueError("Parquet batch size must be positive")
        parquet_file = pq.ParquetFile(self._path(uri))
        for batch in parquet_file.iter_batches(
            batch_size=batch_size,
            columns=list(columns),
            use_threads=False,
        ):
            yield from batch.to_pylist()

    def iter_verified_parquet_rows(
        self,
        uri: str,
        expected_sha256: str,
        columns: tuple[str, ...],
        *,
        batch_size: int,
    ) -> Iterable[dict[str, object]]:
        """Verify a disk snapshot, then scan that same copy in bounded batches."""

        if batch_size <= 0:
            raise ValueError("Parquet batch size must be positive")
        with self._verified_snapshot(uri, expected_sha256) as snapshot_path:
            parquet_file = pq.ParquetFile(snapshot_path)
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=list(columns),
                use_threads=False,
            ):
                yield from batch.to_pylist()

    def resolve_uri(self, manifest_uri: str, relative_path: str) -> str:
        """Resolve a pool-relative path under the local artifact root."""

        parsed = urlparse(manifest_uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("filesystem artifact reader only accepts file URIs")
        path = unquote(parsed.path)
        marker = "/pools/"
        if marker not in path:
            raise ValueError("pool manifest URI does not contain the artifact root")
        root = path.split(marker, 1)[0]
        artifact_path = f"{root}/{relative_path}"
        return parsed._replace(path=quote(artifact_path, safe="/")).geturl()

    @contextmanager
    def _verified_snapshot(self, uri: str, expected_sha256: str) -> Iterator[Path]:
        """Copy and hash one artifact, then expose only that verified temporary copy."""

        if not _valid_sha256(expected_sha256):
            raise ValueError("artifact snapshot pin is not a SHA-256 digest")
        source_path = self._path(uri)
        descriptor, temporary_name = tempfile.mkstemp(
            prefix="pool-gate-snapshot-",
            suffix=source_path.suffix,
        )
        snapshot_path = Path(temporary_name)
        digest = hashlib.sha256()
        try:
            with os.fdopen(descriptor, "wb") as snapshot, source_path.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    snapshot.write(chunk)
            if digest.hexdigest() != expected_sha256:
                raise ValueError("artifact snapshot SHA-256 does not match its pin")
            yield snapshot_path
        finally:
            snapshot_path.unlink(missing_ok=True)

    @staticmethod
    def _path(uri: str) -> Path:
        parsed = urlparse(uri)
        if parsed.scheme != "file" or parsed.netloc not in ("", "localhost"):
            raise ValueError("filesystem artifact reader only accepts file URIs")
        return Path(unquote(parsed.path))


def validate_nine_streamed_pool_gate(
    receipts: Iterable[StreamingExportReceipt],
    reader: PoolArtifactReader,
) -> GateReport:
    """Verify nine streamed completion manifests and their immutable artifacts."""

    expected = {
        PoolKey(text_source, reference_system)
        for text_source in TextSource
        for reference_system in ReferenceSystem
    }
    seen: dict[PoolKey, int] = defaultdict(int)
    invalid: list[tuple[str, tuple[str, ...]]] = []
    versions: set[tuple[str, str]] = set()
    for receipt_count, receipt in enumerate(receipts):
        if receipt_count == len(expected):
            invalid.append(("nine-pool-set", ("the streamed gate accepts exactly nine receipts",)))
            break
        issues: list[str] = []
        pool_name = receipt.pool_id or "unknown-streamed-pool"
        pool_key: PoolKey | None = None
        try:
            manifest, pool_key, actual_manifest_sha = _read_streamed_manifest(receipt, reader)
            if actual_manifest_sha != receipt.manifest_sha256:
                issues.append("completion manifest SHA-256 does not match receipt")
            pool_name = pool_key.name
            seen[pool_key] += 1
            version = (manifest.get("contract_version"), manifest.get("pool_version"))
            if all(_present(value) for value in version):
                versions.add(cast(tuple[str, str], version))
            issues.extend(_stream_manifest_issues(receipt, manifest, reader, pool_key))
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as error:
            issues.append(
                f"streamed receipt could not be verified: {type(error).__name__}: {error}"
            )
            parsed_key = _pool_key_from_id(receipt.pool_id)
            if parsed_key is not None and pool_key is None:
                pool_key = parsed_key
                pool_name = pool_key.name
                seen[pool_key] += 1
        if issues:
            invalid.append((pool_name, tuple(issues)))

    missing = tuple(sorted(key.name for key in expected - seen.keys()))
    duplicates = tuple(sorted(key.name for key, count in seen.items() if count > 1))
    if len(versions) > 1:
        invalid.append(
            ("nine-pool-version-set", ("pool versions must match across all nine pools",))
        )
    return GateReport(missing, duplicates, tuple(invalid))


def validate_nine_streamed_evaluation_ready_gate(
    receipts: Iterable[StreamingExportReceipt],
    reader: PoolArtifactReader,
) -> GateReport:
    """Require nine structurally valid pools and readable, complete evidence."""

    receipt_rows = tuple(islice(receipts, len(TextSource) * len(ReferenceSystem) + 1))
    archive_report = validate_nine_streamed_pool_gate(receipt_rows, reader)
    if archive_report.invalid_pools or archive_report.duplicate_pools or not receipt_rows:
        return archive_report

    invalid: list[tuple[str, tuple[str, ...]]] = []
    for receipt in receipt_rows:
        pool_name = receipt.pool_id.split("/")[1] if "/" in receipt.pool_id else receipt.pool_id
        issues: list[str] = []
        try:
            manifest, pool_key, actual_manifest_sha = _read_streamed_manifest(receipt, reader)
            pool_name = pool_key.name
            if actual_manifest_sha != receipt.manifest_sha256:
                issues.append(
                    "evaluation readiness blocked: completion manifest changed since "
                    "structural validation"
                )
                invalid.append((pool_name, tuple(issues)))
                continue
            source = _manifest_mapping(manifest, "source_snapshot", issues)
            reference = _manifest_mapping(manifest, "reference_pin", issues)
            issues.extend(
                _evaluation_provenance_issues(
                    source,
                    reference,
                    reader,
                )
            )
            issues.extend(_evaluation_artifact_issues(receipt, reader))
        except (OSError, TypeError, ValueError, KeyError, AttributeError) as error:
            issues.append(
                f"evaluation evidence could not be verified: {type(error).__name__}: {error}"
            )
        if issues:
            invalid.append((pool_name, tuple(issues)))

    return GateReport(
        archive_report.missing_pools,
        archive_report.duplicate_pools,
        tuple(invalid),
    )


def _evaluation_provenance_issues(
    source: dict[str, object],
    reference: dict[str, object],
    reader: PoolArtifactReader,
) -> list[str]:
    issues: list[str] = []
    pbf_uri = source.get("pbf_uri")
    pbf_sha256 = source.get("pbf_sha256")
    pbf_present = pbf_uri is not None or pbf_sha256 is not None
    if pbf_present:
        if (
            isinstance(pbf_uri, str)
            and _valid_uri(pbf_uri)
            and isinstance(pbf_sha256, str)
            and _valid_sha256(pbf_sha256)
        ):
            _verify_pinned_artifact(pbf_uri, pbf_sha256, "raw PBF", reader, issues)
        else:
            issues.append("evaluation readiness blocked: raw PBF artifact pin is incomplete")

    source_tree = source.get("source_artifact_manifest")
    if not isinstance(source_tree, dict):
        issues.append("evaluation readiness blocked: raw source artifact manifest pin is missing")
    else:
        source_tree_issues, pbf_listed = _verify_artifact_manifest(
            source_tree,
            "raw source",
            reader,
            required_uri=str(pbf_uri) if pbf_present and isinstance(pbf_uri, str) else None,
            required_sha256=(
                str(pbf_sha256)
                if pbf_present and isinstance(pbf_sha256, str) and _valid_sha256(pbf_sha256)
                else None
            ),
        )
        issues.extend(source_tree_issues)
        if pbf_present and isinstance(pbf_uri, str) and not pbf_listed:
            issues.append("evaluation readiness blocked: raw source manifest omits the pinned PBF")

    reference_tree = reference.get("reference_artifact_manifest")
    if not isinstance(reference_tree, dict):
        issues.append("evaluation readiness blocked: reference artifact manifest pin is missing")
    else:
        reference_issues, _ = _verify_artifact_manifest(reference_tree, "reference", reader)
        issues.extend(reference_issues)

    issues.extend(_area_measurement_issues(reference))
    return issues


def _area_measurement_issues(reference: dict[str, object]) -> list[str]:
    method = reference.get("area_measurement")
    if not isinstance(method, dict) or any(
        not _present(method.get(field)) for field in ("implementation", "implementation_version")
    ):
        return ["evaluation readiness blocked: area measurement method pin is incomplete"]
    support_unit = reference.get("support_unit")
    issues: list[str] = []
    for area_name in ("numerator", "denominator"):
        spec = method.get(area_name)
        if not isinstance(spec, dict):
            issues.append(
                f"evaluation readiness blocked: area measurement {area_name} method is missing"
            )
            continue
        issues.extend(_area_measure_spec_issues(area_name, spec, support_unit))
    return issues


def _area_measure_spec_issues(
    area_name: str,
    spec: dict[str, object],
    support_unit: object,
) -> list[str]:
    issues = [
        f"evaluation readiness blocked: area measurement {area_name} lacks {field}"
        for field in ("method_id", "boundary_convention", "densification_method", "output_unit")
        if not _present(spec.get(field))
    ]
    coordinate_system = spec.get("coordinate_system")
    ellipsoid = spec.get("ellipsoid")
    if coordinate_system is not None and not _present(coordinate_system):
        issues.append(
            f"evaluation readiness blocked: area measurement {area_name} has an invalid "
            "coordinate system"
        )
    if ellipsoid is not None and not _present(ellipsoid):
        issues.append(
            f"evaluation readiness blocked: area measurement {area_name} has an invalid ellipsoid"
        )
    if coordinate_system is None and ellipsoid is None:
        issues.append(
            f"evaluation readiness blocked: area measurement {area_name} needs a coordinate "
            "system or ellipsoid"
        )
    if spec.get("output_unit") != support_unit:
        issues.append(
            f"evaluation readiness blocked: area measurement {area_name} output unit differs "
            "from the support unit"
        )
    for field in ("densification_parameters_json", "error_settings_json"):
        value = spec.get(field)
        if not isinstance(value, str) or not _canonical_json_object(value):
            issues.append(
                f"evaluation readiness blocked: area measurement {area_name} {field} "
                "is not a non-empty canonical JSON object"
            )
    return issues


def _verify_artifact_manifest(
    pin: dict[str, object],
    label: str,
    reader: PoolArtifactReader,
    *,
    required_uri: str | None = None,
    required_sha256: str | None = None,
) -> tuple[list[str], bool]:
    issues, manifest_uri, manifest_sha256, expected_count = _artifact_manifest_pin_fields(
        pin, label
    )
    if manifest_uri is None or manifest_sha256 is None or expected_count is None:
        return issues, False
    row_issues, line_count, found_required_uri = _verify_artifact_manifest_rows(
        manifest_uri,
        manifest_sha256,
        label,
        reader,
        required_uri,
        required_sha256,
    )
    issues.extend(row_issues)
    if line_count != expected_count:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: {label} artifact manifest lists {line_count} "
            f"artifacts, expected {expected_count}",
        )
    if line_count == 0:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: {label} artifact manifest is empty",
        )
    return issues, found_required_uri


def _artifact_manifest_pin_fields(
    pin: dict[str, object],
    label: str,
) -> tuple[list[str], str | None, str | None, int | None]:
    issues: list[str] = []
    uri = pin.get("manifest_uri")
    digest = pin.get("manifest_sha256")
    count = pin.get("expected_artifact_count")
    if pin.get("format_version") != _ARTIFACT_MANIFEST_FORMAT:
        issues.append(
            f"evaluation readiness blocked: {label} artifact manifest format is unsupported"
        )
    if (
        not isinstance(uri, str)
        or not _valid_uri(uri)
        or not isinstance(digest, str)
        or not _valid_sha256(digest)
        or not _is_positive_count(count)
    ):
        issues.append(f"evaluation readiness blocked: {label} artifact manifest pin is invalid")
        return issues, None, None, None
    return issues, uri, digest, cast(int, count)


def _verify_artifact_manifest_rows(
    manifest_uri: str,
    expected_sha256: str,
    label: str,
    reader: PoolArtifactReader,
    required_uri: str | None,
    required_sha256: str | None,
) -> tuple[list[str], int, bool]:
    issues: list[str] = []
    previous_uri: str | None = None
    line_count = 0
    found_required_uri = False
    try:
        for line in reader.iter_verified_lines(manifest_uri, expected_sha256):
            line_count += 1
            row_issues, artifact_uri, artifact_sha256 = _artifact_manifest_row(
                line,
                label,
                previous_uri,
                required_uri,
                required_sha256,
            )
            for issue in row_issues:
                _append_bounded_issue(issues, issue)
            if artifact_uri is None or artifact_sha256 is None:
                continue
            previous_uri = artifact_uri
            if artifact_uri == required_uri:
                found_required_uri = True
            _verify_pinned_artifact(
                artifact_uri,
                artifact_sha256,
                f"{label} artifact",
                reader,
                issues,
            )
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: {label} artifact manifest could not be read: "
            f"{type(error).__name__}",
        )
    return issues, line_count, found_required_uri


def _artifact_manifest_row(
    line: bytes,
    label: str,
    previous_uri: str | None,
    required_uri: str | None,
    required_sha256: str | None,
) -> tuple[list[str], str | None, str | None]:
    issues: list[str] = []
    row = json.loads(line)
    if not isinstance(row, dict):
        return [f"{label} artifact manifest contains a non-object row"], None, None
    if set(row) != {"artifact_uri", "sha256"}:
        _append_bounded_issue(issues, f"{label} artifact manifest row has an invalid schema")
    uri = row.get("artifact_uri")
    digest = row.get("sha256")
    if (
        not isinstance(uri, str)
        or not _valid_uri(uri)
        or not isinstance(digest, str)
        or not _valid_sha256(digest)
    ):
        _append_bounded_issue(issues, f"{label} artifact manifest contains an invalid artifact pin")
        return issues, None, None
    if _canonical_json_bytes(row) != line:
        _append_bounded_issue(issues, f"{label} artifact manifest rows are not canonical JSONL")
    if previous_uri is not None and uri <= previous_uri:
        _append_bounded_issue(issues, f"{label} artifact manifest is not uniquely sorted")
    if uri == required_uri and required_sha256 is not None and digest != required_sha256:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: required {label} artifact hash differs "
            "from its pool pin",
        )
    return issues, uri, digest


def _verify_pinned_artifact(
    uri: str,
    expected_sha256: str,
    label: str,
    reader: PoolArtifactReader,
    issues: list[str],
) -> None:
    try:
        actual_sha256 = reader.sha256(uri)
    except (OSError, TypeError, ValueError) as error:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: {label} could not be read: {type(error).__name__}",
        )
        return
    if actual_sha256 != expected_sha256:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: {label} SHA-256 does not match its pin",
        )


def _evaluation_artifact_issues(
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
) -> list[str]:
    issues: list[str] = []
    counts = {"failed": 0, "partial": 0, "missing": 0, "invalid": 0}
    examples = {key: [] for key in counts}
    try:
        for table, row in _iter_evaluation_rows(receipt, reader):
            if table == "polygon_attempts":
                _scan_attempt_readiness_row(row, counts, examples, reader, issues)
            elif table == "polygon_snapshots":
                _scan_polygon_readiness_row(row, reader, issues)
            else:
                _scan_support_readiness_row(row, reader, issues)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: reference evidence rows could not be scanned: "
            f"{type(error).__name__}",
        )
    _append_readiness_summaries(issues, counts, examples)
    return issues


def _iter_evaluation_rows(
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
) -> Iterator[tuple[str, dict[str, object]]]:
    columns = {
        "polygon_attempts": (
            "inventory_row_id",
            "status",
            "attempted_reference_ids_json",
            "missing_reference_ids_json",
            "measurement_sidecar_uri",
            "measurement_sidecar_sha256",
        ),
        "polygon_snapshots": ("polygon_snapshot_id", "geometry_uri", "geometry_sha256"),
        "label_support_evidence": (
            "inventory_row_id",
            "level",
            "code",
            "reference_id_namespace",
            "reference_ids_json",
            "measurement_sidecar_uri",
            "measurement_sidecar_sha256",
        ),
    }
    for line in reader.iter_verified_lines(
        receipt.shard_manifest_uri,
        receipt.shard_manifest_sha256,
    ):
        record = json.loads(line)
        if not isinstance(record, dict):
            raise TypeError("shard manifest row is not an object")
        table = record.get("table")
        if not isinstance(table, str) or table not in _TABLE_FIELDS:
            raise ValueError("shard manifest row has an unknown table")
        if table not in columns:
            continue
        relative_path = record.get("path")
        expected_sha256 = record.get("sha256")
        if not isinstance(relative_path, str) or not isinstance(expected_sha256, str):
            raise TypeError("evaluation shard path or hash is not text")
        if not _valid_sha256(expected_sha256):
            raise ValueError("evaluation shard path or hash is invalid")
        shard_uri = reader.resolve_uri(receipt.manifest_uri, relative_path)
        for row in reader.iter_verified_parquet_rows(
            shard_uri,
            expected_sha256,
            columns[table],
            batch_size=_GATE_PARQUET_BATCH_SIZE,
        ):
            yield table, row


def _scan_attempt_readiness_row(
    row: dict[str, object],
    counts: dict[str, int],
    examples: dict[str, list[str]],
    reader: PoolArtifactReader,
    issues: list[str],
) -> None:
    row_id = str(row.get("inventory_row_id") or "unknown")
    status = row.get("status")
    if status == AttemptStatus.FAILED.value:
        _record_readiness_problem(counts, examples, "failed", row_id)
    elif status == AttemptStatus.PARTIAL.value:
        _record_readiness_problem(counts, examples, "partial", row_id)
    elif status not in {item.value for item in AttemptStatus}:
        _record_readiness_problem(counts, examples, "invalid", row_id)
    attempted_ids = _parse_json_id_array(row.get("attempted_reference_ids_json"))
    missing_ids = _parse_json_id_array(row.get("missing_reference_ids_json"))
    if attempted_ids is None or missing_ids is None:
        _record_readiness_problem(counts, examples, "invalid", row_id)
    elif missing_ids:
        _record_readiness_problem(counts, examples, "missing", row_id)
    _verify_optional_sidecar(
        row.get("measurement_sidecar_uri"),
        row.get("measurement_sidecar_sha256"),
        f"polygon attempt {row_id}",
        reader,
        issues,
    )


def _scan_polygon_readiness_row(
    row: dict[str, object],
    reader: PoolArtifactReader,
    issues: list[str],
) -> None:
    geometry_uri = row.get("geometry_uri")
    geometry_sha256 = row.get("geometry_sha256")
    if (
        not isinstance(geometry_uri, str)
        or not _valid_uri(geometry_uri)
        or not isinstance(geometry_sha256, str)
        or not _valid_sha256(geometry_sha256)
    ):
        _append_bounded_issue(
            issues,
            "evaluation readiness blocked: polygon geometry pin is incomplete",
        )
        return
    _verify_pinned_artifact(
        geometry_uri,
        geometry_sha256,
        "polygon geometry",
        reader,
        issues,
    )


def _scan_support_readiness_row(
    row: dict[str, object],
    reader: PoolArtifactReader,
    issues: list[str],
) -> None:
    evidence_key = f"{row.get('inventory_row_id')}/{row.get('level')}/{row.get('code')}"
    reference_ids = _parse_json_id_array(row.get("reference_ids_json"))
    has_sidecar = row.get("measurement_sidecar_uri") is not None
    if (
        reference_ids is None
        or bool(reference_ids) == has_sidecar
        or not _present(row.get("reference_id_namespace"))
    ):
        _append_bounded_issue(
            issues,
            f"evaluation readiness blocked: malformed reference support record {evidence_key}",
        )
    _verify_optional_sidecar(
        row.get("measurement_sidecar_uri"),
        row.get("measurement_sidecar_sha256"),
        f"label support evidence {evidence_key}",
        reader,
        issues,
    )


def _record_readiness_problem(
    counts: dict[str, int],
    examples: dict[str, list[str]],
    category: str,
    row_id: str,
) -> None:
    counts[category] += 1
    if len(examples[category]) < _READINESS_EXAMPLE_LIMIT:
        examples[category].append(row_id)


def _append_readiness_summaries(
    issues: list[str],
    counts: dict[str, int],
    examples: dict[str, list[str]],
) -> None:
    messages = {
        "failed": "FAILED polygon attempts",
        "partial": "PARTIAL polygon attempts",
        "missing": "attempts list missing reference IDs",
        "invalid": "attempt rows have unknown status or malformed reference IDs",
    }
    for category, label in messages.items():
        count = counts[category]
        if count:
            issues.append(
                f"evaluation readiness blocked: {count} {label} "
                f"(examples: {', '.join(examples[category])})"
            )


def _verify_optional_sidecar(
    uri: object,
    sha256: object,
    label: str,
    reader: PoolArtifactReader,
    issues: list[str],
) -> None:
    if uri is None:
        return
    if (
        not isinstance(uri, str)
        or not _valid_uri(uri)
        or not isinstance(sha256, str)
        or not _valid_sha256(sha256)
    ):
        _append_bounded_issue(
            issues, f"evaluation readiness blocked: {label} sidecar pin is incomplete"
        )
        return
    _verify_pinned_artifact(uri, sha256, label, reader, issues)


def _canonical_json_object(value: str) -> bool:
    try:
        parsed = json.loads(value)
        canonical = json.dumps(
            parsed,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(parsed, dict) and bool(parsed) and canonical == value


def _is_positive_count(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _append_bounded_issue(issues: list[str], message: str) -> None:
    if len(issues) < _MAX_DIAGNOSTICS:
        issues.append(message)
    elif len(issues) == _MAX_DIAGNOSTICS:
        issues.append("additional evaluation-readiness diagnostics omitted")


def _read_streamed_manifest(
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
) -> tuple[dict[str, object], PoolKey, str]:
    manifest_bytes = reader.read_bytes(receipt.manifest_uri)
    actual_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if not isinstance(manifest, dict):
        raise TypeError("completion manifest must be a JSON object")
    pool_data = manifest.get("pool_key")
    if not isinstance(pool_data, dict):
        raise TypeError("completion manifest has no pool key")
    pool_key = PoolKey(
        TextSource(pool_data["text_source"]),
        ReferenceSystem(pool_data["reference_system"]),
    )
    return manifest, pool_key, actual_sha


def _pool_key_from_id(pool_id: object) -> PoolKey | None:
    if not isinstance(pool_id, str):
        return None
    parts = pool_id.split("/")
    if len(parts) != _POOL_ID_PARTS:
        return None
    try:
        text_source, reference_system = parts[1].split("-", 1)
        return PoolKey(TextSource(text_source), ReferenceSystem(reference_system))
    except (ValueError, IndexError):
        return None


def _stream_manifest_issues(
    receipt: StreamingExportReceipt,
    manifest: dict[str, object],
    reader: PoolArtifactReader,
    pool_key: PoolKey,
) -> list[str]:
    issues: list[str] = []
    source = _manifest_mapping(manifest, "source_snapshot", issues)
    raw_evidence = _manifest_mapping(manifest, "raw_inventory_evidence", issues)
    reference = _manifest_mapping(manifest, "reference_pin", issues)
    implementation = _manifest_mapping(manifest, "implementation_pins", issues)
    issues.extend(_manifest_identity_issues(receipt, manifest, pool_key, source, implementation))
    issues.extend(_manifest_reference_issues(pool_key, reference))
    issues.extend(_manifest_inventory_issues(receipt, manifest, raw_evidence, reader))
    total_keys = manifest.get("total_source_keys")
    source_id = source.get("snapshot_id")
    issues.extend(_stream_accounting_issues(manifest, receipt, total_keys))
    issues.extend(_stream_artifact_issues(manifest, receipt, reader, pool_key, source_id))
    return issues


def _manifest_mapping(
    manifest: dict[str, object],
    field_name: str,
    issues: list[str],
) -> dict[str, object]:
    value = manifest.get(field_name)
    if isinstance(value, dict):
        return value
    issues.append(f"completion manifest is missing {field_name}")
    return {}


def _manifest_identity_issues(
    receipt: StreamingExportReceipt,
    manifest: dict[str, object],
    pool_key: PoolKey,
    source: dict[str, object],
    implementation: dict[str, object],
) -> list[str]:
    issues: list[str] = []
    contract_version = manifest.get("contract_version")
    pool_version = manifest.get("pool_version")
    source_id = source.get("snapshot_id")
    if manifest.get("contract") != "noeflandre.polygon-label-pool":
        issues.append("completion manifest contract identifier is invalid")
    if manifest.get("pool_complete") is not True:
        issues.append("completion manifest does not mark the pool complete")
    if not all(_present(value) for value in (contract_version, pool_version)):
        issues.append("completion manifest contract and pool versions are missing")
    expected_pool_id = f"{pool_version}/{pool_key.name}/{source_id}"
    if manifest.get("pool_id") != expected_pool_id or receipt.pool_id != expected_pool_id:
        issues.append("pool identity differs between receipt and completion manifest")
    if manifest.get("storage_layout_version") != _STORAGE_LAYOUT_VERSION:
        issues.append("streaming storage layout version is unsupported")
    expected_suffix = (
        f"/pools/{contract_version}/{pool_key.text_source.value}/"
        f"{pool_key.reference_system.value}/{source_id}/{pool_version}/layouts/"
        f"{_STORAGE_LAYOUT_VERSION}/manifest.json"
    )
    if not unquote(urlparse(receipt.manifest_uri).path).endswith(expected_suffix):
        issues.append("completion manifest URI does not match its pinned pool identity")
    required = (
        source_id,
        source.get("repository"),
        source.get("revision"),
        implementation.get("adapter_id"),
        implementation.get("adapter_revision"),
        implementation.get("builder_revision"),
    )
    if any(not _present(value) for value in required):
        issues.append("completion manifest has incomplete source or code provenance")
    issues.extend(_source_artifact_identity_issues(source))
    return issues


def _source_artifact_identity_issues(source: dict[str, object]) -> list[str]:
    issues: list[str] = []
    pbf_uri = source.get("pbf_uri")
    pbf_sha256 = source.get("pbf_sha256")
    pbf_present = pbf_uri is not None or pbf_sha256 is not None
    if pbf_present:
        if (
            not isinstance(pbf_uri, str)
            or not _valid_uri(pbf_uri)
            or not isinstance(pbf_sha256, str)
            or not _valid_sha256(pbf_sha256)
        ):
            issues.append("completion manifest has an invalid PBF artifact pin")
    elif not isinstance(source.get("source_artifact_manifest"), dict):
        issues.append("completion manifest lacks a PBF pin and source artifact tree pin")
    return issues


def _manifest_reference_issues(
    pool_key: PoolKey,
    reference: dict[str, object],
) -> list[str]:
    required = (
        reference.get("release_id"),
        reference.get("source_repository"),
        reference.get("source_revision"),
        reference.get("taxonomy_id"),
        reference.get("taxonomy_revision"),
        reference.get("support_unit"),
    )
    if any(not _present(value) for value in required):
        return ["completion manifest has incomplete reference provenance"]
    expected_semantics = {
        ReferenceSystem.WORLDCOVER: "mapped_cover",
        ReferenceSystem.LUCAS: "surveyed_footprint",
        ReferenceSystem.EUNIS: "modeled_habitat_support",
    }[pool_key.reference_system]
    if (
        reference.get("threshold") != _POOL_THRESHOLD
        or reference.get("support_union_policy") != "union_by_class"
        or reference.get("support_semantics") != expected_semantics
    ):
        return ["completion manifest reference rule does not match the shared contract"]
    return []


def _manifest_inventory_issues(
    receipt: StreamingExportReceipt,
    manifest: dict[str, object],
    raw_evidence: dict[str, object],
    reader: PoolArtifactReader,
) -> list[str]:
    issues: list[str] = []
    total_keys = manifest.get("total_source_keys")
    inventory_receipt = manifest.get("source_inventory_receipt")
    expected_count = raw_evidence.get("expected_key_count")
    expected_keyset = raw_evidence.get("expected_keyset_sha256")
    observed_count = (
        inventory_receipt.get("processed_key_count")
        if isinstance(inventory_receipt, dict)
        else None
    )
    observed_keyset = (
        inventory_receipt.get("observed_keyset_sha256")
        if isinstance(inventory_receipt, dict)
        else None
    )
    valid_total = (
        isinstance(total_keys, int)
        and not isinstance(total_keys, bool)
        and total_keys >= 0
        and isinstance(expected_count, int)
        and not isinstance(expected_count, bool)
        and expected_count >= 0
        and isinstance(observed_count, int)
        and not isinstance(observed_count, bool)
        and observed_count >= 0
        and total_keys == expected_count
        and total_keys == observed_count
    )
    valid_digest = _valid_sha256(expected_keyset) and expected_keyset == observed_keyset
    if (
        not valid_total
        or not valid_digest
        or not isinstance(inventory_receipt, dict)
        or inventory_receipt.get("matches_source_evidence") is not True
    ):
        issues.append("raw inventory receipt is incomplete or differs from source evidence")
    if (
        not _is_nonnegative_count(receipt.total_source_keys)
        or not _is_nonnegative_count(receipt.shard_count)
        or receipt.total_source_keys != total_keys
        or receipt.inventory_keyset_sha256 != observed_keyset
    ):
        issues.append("raw inventory receipt differs between object and manifest")
    if not all(
        _present(value)
        for value in (
            raw_evidence.get("enumeration_rule"),
            raw_evidence.get("scan_report_uri"),
        )
    ) or not _valid_sha256(raw_evidence.get("scan_report_sha256")):
        issues.append("raw inventory scan-report pin is incomplete")
        return issues
    try:
        actual_scan_hash = reader.sha256(str(raw_evidence["scan_report_uri"]))
    except (OSError, TypeError, ValueError) as error:
        issues.append(f"raw inventory scan report could not be verified: {type(error).__name__}")
    else:
        if actual_scan_hash != raw_evidence.get("scan_report_sha256"):
            issues.append("raw inventory scan report does not match its pinned hash")
    return issues


def _stream_accounting_issues(
    manifest: dict[str, object],
    receipt: StreamingExportReceipt,
    total_keys: object,
) -> list[str]:
    issues: list[str] = []
    levels = manifest.get("levels")
    accounting = manifest.get("level_accounting")
    if not isinstance(levels, list) or not isinstance(accounting, list):
        return ["completion manifest has no level accounting"]
    level_names = [row.get("name") for row in levels if isinstance(row, dict)]
    accounting_names = [row.get("level") for row in accounting if isinstance(row, dict)]
    if (
        not level_names
        or len(level_names) != len(set(level_names))
        or len(accounting_names) != len(set(accounting_names))
        or set(level_names) != set(accounting_names)
    ):
        issues.append("completion manifest level declarations and accounting differ")
    declared_levels = {row.get("name"): row for row in levels if isinstance(row, dict)}
    for item in accounting:
        issues.extend(
            _stream_level_accounting_row_issues(
                item, total_keys, cast(dict[object, object], declared_levels)
            )
        )
    receipt_levels = tuple(
        {
            "level": row.level,
            "supported": row.supported,
            "unsupported_reason": row.unsupported_reason,
            "total_source_keys": row.total_source_keys,
            "processed_decisions": row.processed_decisions,
            "qualifying_candidates": row.qualifying_candidates,
            "assigned": row.assigned,
            "ambiguous": row.ambiguous,
            "unassigned": row.unassigned,
            "excluded": row.excluded,
            "exclusion_reason_group_count": row.exclusion_reason_group_count,
        }
        for row in receipt.level_accounting
    )
    if tuple(accounting) != receipt_levels:
        issues.append("per-level accounting differs between receipt and completion manifest")
    return issues


def _stream_level_accounting_row_issues(
    item: object,
    total_keys: object,
    declared_levels: dict[object, object],
) -> list[str]:
    if not isinstance(item, dict):
        return ["completion manifest has invalid level accounting"]
    issues: list[str] = []
    name = item.get("level")
    supported = item.get("supported")
    if not isinstance(supported, bool):
        return [f"level {name} has no Boolean supported state"]
    declared = declared_levels.get(name)
    if (
        not isinstance(declared, dict)
        or declared.get("supported") is not supported
        or declared.get("unsupported_reason") != item.get("unsupported_reason")
    ):
        issues.append(f"level {name} accounting differs from its level declaration")
    counts = [
        item.get("total_source_keys"),
        item.get("processed_decisions"),
        item.get("qualifying_candidates"),
        item.get("assigned"),
        item.get("ambiguous"),
        item.get("unassigned"),
        item.get("excluded"),
        item.get("exclusion_reason_group_count"),
    ]
    if any(not _is_nonnegative_count(value) for value in counts):
        issues.append(f"level {name} has invalid accounting counts")
        return issues
    (
        total,
        processed,
        qualifying,
        assigned,
        ambiguous,
        unassigned,
        excluded,
        _,
    ) = cast(tuple[int, int, int, int, int, int, int, int], tuple(counts))
    if total != total_keys or processed != (total if supported else 0):
        issues.append(f"level {name} does not account for every source key")
    if assigned + ambiguous + unassigned + excluded != processed:
        issues.append(f"level {name} outcome counts do not match processed decisions")
    if qualifying < assigned:
        issues.append(f"level {name} qualifying candidate count is inconsistent")
    if not supported and not _present(item.get("unsupported_reason")):
        issues.append(f"unsupported level {name} needs a reason")
    return issues


def _stream_artifact_issues(
    manifest: dict[str, object],
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
    pool_key: PoolKey,
    source_id: object,
) -> list[str]:
    storage = manifest.get("storage_layout")
    if not isinstance(storage, dict):
        return ["completion manifest has no streaming storage receipt"]
    shard_manifest = storage.get("shard_manifest")
    exclusions = storage.get("exclusions")
    tables = storage.get("tables")
    if not all(isinstance(value, dict) for value in (shard_manifest, exclusions, tables)):
        return ["completion manifest has incomplete storage metadata"]
    shard_manifest = cast(dict[str, object], shard_manifest)
    exclusions = cast(dict[str, object], exclusions)
    tables = cast(dict[str, object], tables)
    expected_root = _stream_layout_root(manifest, pool_key, source_id)
    issues = _shard_manifest_reference_issues(shard_manifest, receipt, reader, expected_root)
    issues.extend(_exclusion_sidecar_issues(exclusions, receipt, reader, expected_root))
    shard_counts, row_counts, inventory_issues = _verify_shard_inventory(
        receipt,
        reader,
        expected_root,
        table_schemas(str(manifest.get("contract_version", ""))),
    )
    issues.extend(inventory_issues)
    issues.extend(
        _table_inventory_issues(
            tables,
            shard_counts,
            row_counts,
            manifest,
            receipt,
            shard_manifest,
        )
    )
    return issues


def _stream_layout_root(
    manifest: dict[str, object],
    pool_key: PoolKey,
    source_id: object,
) -> str:
    return (
        f"pools/{manifest.get('contract_version')}/{pool_key.text_source.value}/"
        f"{pool_key.reference_system.value}/{source_id}/{manifest.get('pool_version')}/"
        f"layouts/{_STORAGE_LAYOUT_VERSION}"
    )


def _shard_manifest_reference_issues(
    shard_manifest: dict[str, object],
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
    expected_root: str,
) -> list[str]:
    issues: list[str] = []
    if shard_manifest.get("sha256") != receipt.shard_manifest_sha256:
        issues.append("shard manifest differs between receipt and completion manifest")
    if shard_manifest.get("path") != f"{expected_root}/shards.jsonl":
        issues.append("shard manifest path differs from pool layout")
    try:
        expected_shard_manifest_uri = reader.resolve_uri(
            receipt.manifest_uri, str(shard_manifest.get("path"))
        )
        if expected_shard_manifest_uri != receipt.shard_manifest_uri:
            issues.append("shard manifest URI differs from its pool-relative path")
        if reader.sha256(receipt.shard_manifest_uri) != receipt.shard_manifest_sha256:
            issues.append("shard manifest content hash does not match receipt")
    except (OSError, TypeError, ValueError) as error:
        issues.append(f"shard manifest could not be verified: {type(error).__name__}")
    return issues


def _exclusion_sidecar_issues(
    exclusions: dict[str, object],
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
    expected_root: str,
) -> list[str]:
    issues: list[str] = []
    try:
        exclusion_uri = reader.resolve_uri(receipt.manifest_uri, str(exclusions.get("path")))
        if reader.sha256(exclusion_uri) != exclusions.get("sha256"):
            issues.append("exclusion sidecar content hash does not match manifest")
        if exclusions.get("path") != f"{expected_root}/exclusions.jsonl":
            issues.append("exclusion sidecar path differs from pool layout")
        exclusion_lines = sum(1 for _ in reader.iter_lines(exclusion_uri))
        if exclusion_lines != exclusions.get("group_count"):
            issues.append("exclusion sidecar record count differs from manifest")
    except (OSError, TypeError, ValueError) as error:
        issues.append(f"exclusion sidecar could not be verified: {type(error).__name__}")
    return issues


def _verify_shard_inventory(
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
    expected_root: str,
    expected_schemas: dict[str, pa.Schema],
) -> tuple[dict[str, int], dict[str, int], list[str]]:
    issues: list[str] = []
    shard_counts = dict.fromkeys(_TABLE_FIELDS, 0)
    row_counts = dict.fromkeys(_TABLE_FIELDS, 0)
    table_order = {name: index for index, name in enumerate(sorted(_TABLE_FIELDS))}
    previous_rank = -1
    previous_index = -1
    try:
        for line in reader.iter_lines(receipt.shard_manifest_uri):
            table, index, row_count = _verify_shard_entry(
                line, receipt, reader, expected_root, table_order, expected_schemas
            )
            rank = table_order[table]
            _validate_shard_order(rank, index, previous_rank, previous_index)
            previous_rank, previous_index = rank, index
            shard_counts[table] += 1
            row_counts[table] += row_count
    except (OSError, TypeError, ValueError) as error:
        issues.append(f"shard inventory could not be verified: {type(error).__name__}: {error}")
    return shard_counts, row_counts, issues


def _validate_shard_order(rank: int, index: int, previous_rank: int, previous_index: int) -> None:
    if rank < previous_rank or (rank == previous_rank and index != previous_index + 1):
        raise ValueError("shard manifest ordering or numbering is not deterministic")
    if rank > previous_rank and index != _FIRST_SHARD_INDEX:
        raise ValueError("first shard for a table must have index zero")


def _verify_shard_entry(
    line: bytes,
    receipt: StreamingExportReceipt,
    reader: PoolArtifactReader,
    expected_root: str,
    table_order: dict[str, int],
    expected_schemas: dict[str, pa.Schema],
) -> tuple[str, int, int]:
    record = json.loads(line)
    if not isinstance(record, dict):
        raise TypeError("shard manifest row is not an object")
    table = record.get("table")
    if not isinstance(table, str) or table not in table_order:
        raise ValueError("shard manifest contains an unknown table")
    index = record.get("shard_index")
    if not isinstance(index, int) or isinstance(index, bool) or index < 0:
        raise ValueError("shard index must be a non-negative integer")
    relative_path = f"{expected_root}/shards/{table}/part-{index:08d}.parquet"
    digest = record.get("sha256")
    if record.get("path") != relative_path or not _valid_sha256(digest):
        raise ValueError("shard path or hash is invalid")
    row_count = record.get("row_count")
    if not isinstance(row_count, int) or isinstance(row_count, bool) or row_count <= 0:
        raise ValueError("shard row count must be positive")
    uri = reader.resolve_uri(receipt.manifest_uri, relative_path)
    if reader.sha256(uri) != digest:
        raise ValueError(f"shard content hash mismatch at {relative_path}")
    metadata = reader.parquet_metadata(uri)
    if metadata.row_count != row_count:
        raise ValueError(f"Parquet footer row count differs from shard manifest at {relative_path}")
    if not metadata.schema.equals(expected_schemas[table], check_metadata=True):
        raise ValueError(f"Parquet footer schema differs from contract at {relative_path}")
    return table, index, row_count


def _table_inventory_issues(
    tables: dict[str, object],
    shard_counts: dict[str, int],
    row_counts: dict[str, int],
    manifest: dict[str, object],
    receipt: StreamingExportReceipt,
    shard_manifest: dict[str, object],
) -> list[str]:
    issues: list[str] = []
    total_shards = sum(shard_counts.values())
    manifest_shard_count = shard_manifest.get("shard_count")
    if (
        not _is_nonnegative_count(manifest_shard_count)
        or total_shards != receipt.shard_count
        or total_shards != manifest_shard_count
    ):
        issues.append("shard count differs between receipt and shard inventory")
    required_tables = set(_TABLE_FIELDS)
    if set(tables) != required_tables:
        issues.append("completion manifest table inventory differs from contract schema")
    for table in _TABLE_FIELDS:
        info = tables.get(table)
        if not isinstance(info, dict):
            issues.append(f"completion manifest has no count for table {table}")
            continue
        manifest_rows = info.get("row_count")
        manifest_shards = info.get("shard_count")
        if not _is_nonnegative_count(manifest_rows) or not _is_nonnegative_count(manifest_shards):
            issues.append(f"completion manifest has invalid counts for table {table}")
            continue
        if manifest_rows != row_counts[table] or manifest_shards != shard_counts[table]:
            issues.append(f"shard inventory differs from table counts for {table}")
    inventory_table = tables.get("polygon_inventory")
    if isinstance(inventory_table, dict) and inventory_table.get("row_count") != manifest.get(
        "total_source_keys"
    ):
        issues.append("exported polygon inventory count differs from completion manifest")
    attempt_table = tables.get("polygon_attempts")
    if isinstance(attempt_table, dict) and attempt_table.get("row_count") != manifest.get(
        "total_source_keys"
    ):
        issues.append("exported attempt receipt count differs from raw inventory")
    support_evidence_table = tables.get("label_support_evidence")
    candidates_table = tables.get("label_candidates")
    if (
        isinstance(support_evidence_table, dict)
        and isinstance(candidates_table, dict)
        and support_evidence_table.get("row_count") != candidates_table.get("row_count")
    ):
        issues.append("support evidence rows differ from candidate measurement rows")
    return issues


def _is_nonnegative_count(value: object) -> TypeGuard[int]:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


class PoolStreamError(ValueError):
    """Raised when streaming validation or immutable output checks fail."""

    def __init__(self, issues: Iterable[str], max_diagnostics: int = _MAX_DIAGNOSTICS) -> None:
        """Attach all streaming validation failures to the raised error."""

        if max_diagnostics <= 0:
            raise ValueError("max_diagnostics must be positive")
        self.report = PoolValidationReport(tuple(_bounded_items(issues, max_diagnostics)))
        super().__init__("; ".join(self.report.issues))


@dataclass(frozen=True, slots=True)
class _TableSpec:
    """Logical-table identity fields used for disk-backed uniqueness and ordering."""

    name: str
    sort_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _ShardWriteContext:
    """Shared writer dependencies for bounded deterministic shard creation."""

    sink: FileArtifactSink
    connection: sqlite3.Connection
    work_dir: Path
    root: str
    schemas: dict[str, pa.Schema]


@dataclass(slots=True)
class _BatchByteCounter:
    value: int = 0


@dataclass(frozen=True, slots=True)
class _OutcomeStatusCheck:
    key: tuple[str, str]
    status: str
    assigned_code: str | None
    reason: str | None
    ambiguity_basis: str | None
    qualifier_count: int
    candidate_count: int
    assigned_match_count: int


_TABLE_REGISTRY = tuple(
    _TableSpec(name, fields)
    for name, fields in (
        ("polygon_inventory", ("inventory_row_id",)),
        ("polygon_snapshots", ("polygon_snapshot_id",)),
        ("label_candidates", ("inventory_row_id", "level", "code")),
        ("label_outcomes", ("inventory_row_id", "level")),
        ("polygon_attempts", ("inventory_row_id",)),
        ("label_support_evidence", ("inventory_row_id", "level", "code")),
        ("source_documents", ("document_id",)),
        ("document_relevance", ("document_id", "release_id")),
        ("text_polygon_links", ("document_id", "native_link_key_json")),
        (
            "entity_links",
            (
                "link_version",
                "left_source_snapshot_id",
                "left_native_key_json",
                "right_source_snapshot_id",
                "right_native_key_json",
            ),
        ),
        ("pool_eligibility", ("inventory_row_id", "cohort_policy_version")),
    )
)
_TABLE_SPEC_BY_NAME = {table.name: table for table in _TABLE_REGISTRY}


def export_pool_batches(
    spec: PoolStreamSpec,
    batches: Iterable[PoolBatch],
    sink: FileArtifactSink,
    *,
    limits: StreamingLimits | None = None,
    staging_directory: Path | None = None,
) -> StreamingExportReceipt:
    """Validate, shard, and finalize a complete pool using bounded batches."""

    limits = limits or StreamingLimits()
    _validate_limits(limits)
    _validate_spec(spec, limits.max_diagnostics)
    root = (
        f"pools/{spec.contract_version}/{spec.pool_key.text_source.value}/"
        f"{spec.pool_key.reference_system.value}/{spec.source_snapshot.snapshot_id}/"
        f"{spec.pool_version}"
    )
    storage_root = f"{root}/layouts/{_STORAGE_LAYOUT_VERSION}"
    with tempfile.TemporaryDirectory(dir=staging_directory, prefix="pool-stream-") as work:
        work_dir = Path(work)
        database = work_dir / "pool-index.sqlite"
        connection = _open_index(database)
        try:
            _initialize_index(connection)
            for expected_batch_index, batch in enumerate(batches):
                if batch.batch_index != expected_batch_index:
                    raise PoolStreamError(
                        (f"expected batch index {expected_batch_index}, got {batch.batch_index}",),
                        limits.max_diagnostics,
                    )
                if batch.row_count > limits.max_batch_rows:
                    raise PoolStreamError(
                        (
                            f"batch {batch.batch_index} has {batch.row_count} rows, "
                            f"exceeding max_batch_rows={limits.max_batch_rows}",
                        ),
                        limits.max_diagnostics,
                    )
                batch_bytes = _estimated_batch_bytes(batch)
                if batch_bytes > limits.max_batch_bytes:
                    raise PoolStreamError(
                        (
                            f"batch {batch.batch_index} estimated row bytes {batch_bytes} "
                            f"exceed max_batch_bytes={limits.max_batch_bytes}",
                        ),
                        limits.max_diagnostics,
                    )
                _export_batch(spec, batch, connection, limits)
            return _finalize_pool(spec, sink, connection, work_dir, storage_root, limits)
        finally:
            connection.close()


def _validate_limits(limits: StreamingLimits) -> None:
    values = {
        "max_batch_rows": limits.max_batch_rows,
        "max_batch_bytes": limits.max_batch_bytes,
        "max_row_bytes": limits.max_row_bytes,
        "max_shard_rows": limits.max_shard_rows,
        "max_shard_bytes": limits.max_shard_bytes,
        "max_diagnostics": limits.max_diagnostics,
    }
    invalid = [name for name, value in values.items() if value <= 0]
    if invalid:
        raise ValueError(f"streaming limits must be positive: {', '.join(invalid)}")
    maximums = {
        "max_batch_rows": MAX_BATCH_ROWS,
        "max_batch_bytes": MAX_BATCH_BYTES,
        "max_row_bytes": MAX_ROW_BYTES,
        "max_shard_rows": MAX_SHARD_ROWS,
        "max_shard_bytes": MAX_SHARD_BYTES,
        "max_diagnostics": _MAX_DIAGNOSTICS,
    }
    excessive = [name for name, value in values.items() if value > maximums[name]]
    if excessive:
        raise ValueError(f"streaming limits exceed hard bounds: {', '.join(excessive)}")
    if limits.max_row_bytes > limits.max_shard_bytes:
        raise ValueError("max_row_bytes cannot exceed max_shard_bytes")


def _estimated_batch_bytes(batch: PoolBatch) -> int:
    return sum(
        _estimated_json_size(row)
        for rows in (
            batch.inventory,
            batch.polygons,
            batch.label_candidates,
            batch.label_outcomes,
            batch.attempt_receipts,
            batch.support_evidence,
            batch.documents,
            batch.relevance_assessments,
            batch.text_polygon_links,
            batch.entity_links,
            batch.eligibility,
        )
        for row in rows
    )


def _estimated_json_size(value: object) -> int:
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, str):
        size = 2 + (6 * len(value))
    elif value is None:
        size = 4
    elif isinstance(value, bool):
        size = 4 if value else 5
    elif isinstance(value, int):
        size = len(str(value))
    elif isinstance(value, float):
        size = len(repr(value))
    elif is_dataclass(value) and not isinstance(value, type):
        data_fields = fields(value)
        size = (
            2
            + sum(
                _estimated_json_size(field.name)
                + 1
                + _estimated_json_size(getattr(value, field.name))
                for field in data_fields
            )
            + max(0, len(data_fields) - 1)
        )
    elif isinstance(value, (tuple, list)):
        size = 2 + sum(_estimated_json_size(item) for item in value) + max(0, len(value) - 1)
    elif isinstance(value, dict):
        size = (
            2
            + sum(
                _estimated_json_size(key) + 1 + _estimated_json_size(item)
                for key, item in value.items()
            )
            + max(0, len(value) - 1)
        )
    else:
        raise TypeError(f"unsupported value in pool batch size estimate: {type(value).__name__}")
    return size


def _validate_spec(spec: PoolStreamSpec, max_diagnostics: int) -> None:
    evidence = spec.raw_inventory_evidence
    issues = []
    if evidence.expected_key_count < 0:
        issues.append("raw inventory expected key count cannot be negative")
    if len(evidence.expected_keyset_sha256) != _SHA256_LENGTH or any(
        char not in "0123456789abcdef" for char in evidence.expected_keyset_sha256
    ):
        issues.append("raw inventory canonical key-set SHA-256 is invalid")
    if issues:
        raise PoolStreamError(issues, max_diagnostics)
    dummy_evidence = RawInventoryEvidence(
        enumeration_rule=evidence.enumeration_rule,
        scan_report_uri=evidence.scan_report_uri,
        scan_report_sha256=evidence.scan_report_sha256,
        expected_key_count=0,
        expected_keyset_sha256=_KEY_SHA_EMPTY,
    )
    dummy = _empty_bundle(spec, raw_inventory_evidence=dummy_evidence)
    report = validate_full_pool(dummy)
    if not report.ok:
        raise PoolStreamError(report.issues, max_diagnostics)


def _empty_bundle(
    spec: PoolStreamSpec,
    *,
    raw_inventory_evidence: RawInventoryEvidence | None = None,
) -> PoolBundle:
    evidence = raw_inventory_evidence or RawInventoryEvidence(
        enumeration_rule=spec.raw_inventory_evidence.enumeration_rule,
        scan_report_uri=spec.raw_inventory_evidence.scan_report_uri,
        scan_report_sha256=spec.raw_inventory_evidence.scan_report_sha256,
        expected_key_count=0,
        expected_keyset_sha256=_KEY_SHA_EMPTY,
    )
    source = PoolBuildInput(
        pool_key=spec.pool_key,
        source_snapshot=spec.source_snapshot,
        raw_inventory_evidence=evidence,
        reference_pin=spec.reference_pin,
        adapter_id=spec.adapter_id,
        adapter_revision=spec.adapter_revision,
        builder_revision=spec.builder_revision,
        inventory=(),
        polygons=(),
        levels=spec.levels,
        label_candidates=(),
        label_outcomes=(),
    )
    return build_full_pool(
        source,
        contract_version=spec.contract_version,
        pool_version=spec.pool_version,
    )


def _open_index(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute(f"PRAGMA cache_size=-{_SQLITE_CACHE_KIB}")
    connection.execute("PRAGMA temp_store=FILE")
    connection.execute("PRAGMA mmap_size=0")
    connection.execute("PRAGMA journal_mode=DELETE")
    connection.execute("PRAGMA foreign_keys=ON")
    return connection


def _initialize_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        CREATE TABLE inventory (
            inventory_row_id TEXT PRIMARY KEY,
            native_key_json TEXT NOT NULL UNIQUE,
            source_row_locator TEXT NOT NULL,
            polygon_snapshot_id TEXT UNIQUE,
            exclusion_reason TEXT
        );
        CREATE TABLE polygons (
            polygon_snapshot_id TEXT PRIMARY KEY,
            inventory_row_id TEXT NOT NULL UNIQUE,
            native_key_json TEXT NOT NULL,
            geometry_uri TEXT NOT NULL,
            geometry_sha256 TEXT NOT NULL,
            geometry_format TEXT NOT NULL,
            crs TEXT NOT NULL,
            coordinate_transform TEXT NOT NULL,
            geometry_status TEXT NOT NULL,
            original_area REAL,
            area_unit TEXT NOT NULL
        );
        CREATE TABLE candidates (
            inventory_row_id TEXT NOT NULL,
            level TEXT NOT NULL,
            code TEXT NOT NULL,
            polygon_snapshot_id TEXT NOT NULL,
            label_release TEXT NOT NULL,
            numerator REAL NOT NULL,
            denominator REAL NOT NULL,
            ratio REAL NOT NULL,
            threshold REAL NOT NULL,
            assignment_rule_version TEXT NOT NULL,
            assignment_kind TEXT NOT NULL,
            support_semantics TEXT NOT NULL,
            support_unit TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            taxonomy_revision TEXT NOT NULL,
            PRIMARY KEY (inventory_row_id, level, code)
        );
        CREATE TABLE outcomes (
            inventory_row_id TEXT NOT NULL,
            level TEXT NOT NULL,
            polygon_snapshot_id TEXT,
            status TEXT NOT NULL,
            assigned_code TEXT,
            reason TEXT,
            ambiguity_basis TEXT,
            PRIMARY KEY (inventory_row_id, level)
        );
        CREATE TABLE polygon_attempts (
            inventory_row_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            reference_id_namespace TEXT NOT NULL,
            attempted_reference_ids_json TEXT NOT NULL,
            missing_reference_ids_json TEXT NOT NULL,
            measurement_sidecar_uri TEXT,
            measurement_sidecar_sha256 TEXT,
            measurement_record_count INTEGER,
            reason TEXT
        );
        CREATE TABLE support_evidence (
            inventory_row_id TEXT NOT NULL,
            level TEXT NOT NULL,
            code TEXT NOT NULL,
            reference_id_namespace TEXT NOT NULL,
            reference_ids_json TEXT NOT NULL,
            measurement_sidecar_uri TEXT,
            measurement_sidecar_sha256 TEXT,
            measurement_record_count INTEGER,
            PRIMARY KEY (inventory_row_id, level, code)
        );
        CREATE TABLE source_documents (
            document_id TEXT PRIMARY KEY,
            native_key_json TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            content_sha256 TEXT,
            body_status TEXT NOT NULL
        );
        CREATE TABLE document_relevance (
            document_id TEXT NOT NULL,
            release_id TEXT NOT NULL,
            status TEXT NOT NULL,
            source_repository TEXT NOT NULL,
            source_revision TEXT NOT NULL,
            mapping_rule_version TEXT NOT NULL,
            evidence TEXT NOT NULL,
            raw_value TEXT,
            evidence_sha256 TEXT,
            reason TEXT,
            PRIMARY KEY (document_id, release_id)
        );
        CREATE TABLE text_links (
            link_id INTEGER PRIMARY KEY,
            document_id TEXT NOT NULL,
            native_link_key_json TEXT NOT NULL,
            inventory_row_id TEXT,
            source_revision TEXT NOT NULL,
            status TEXT NOT NULL,
            evidence TEXT NOT NULL,
            UNIQUE (document_id, native_link_key_json)
        );
        CREATE TABLE link_candidates (
            link_id INTEGER NOT NULL,
            inventory_row_id TEXT NOT NULL,
            PRIMARY KEY (link_id, inventory_row_id),
            FOREIGN KEY (link_id) REFERENCES text_links(link_id)
        );
        CREATE TABLE entity_links (
            link_version TEXT NOT NULL,
            left_ref TEXT NOT NULL,
            right_ref TEXT NOT NULL,
            evidence TEXT NOT NULL,
            evidence_sha256 TEXT,
            PRIMARY KEY (link_version, left_ref, right_ref)
        );
        CREATE TABLE eligibility (
            inventory_row_id TEXT NOT NULL,
            cohort_policy_version TEXT NOT NULL,
            status TEXT NOT NULL,
            reason TEXT,
            replaced_by_inventory_row_id TEXT,
            evidence TEXT NOT NULL,
            PRIMARY KEY (inventory_row_id, cohort_policy_version)
        );
        CREATE TABLE spooled_rows (
            table_name TEXT NOT NULL,
            sort_key TEXT NOT NULL,
            row_json TEXT NOT NULL,
            row_bytes INTEGER NOT NULL,
            PRIMARY KEY (table_name, sort_key),
            FOREIGN KEY (table_name) REFERENCES table_counts(table_name)
        ) WITHOUT ROWID;
        CREATE TABLE shards (
            table_name TEXT NOT NULL,
            shard_index INTEGER NOT NULL,
            path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            PRIMARY KEY (table_name, shard_index)
        );
        CREATE TABLE table_counts (
            table_name TEXT PRIMARY KEY,
            row_count INTEGER NOT NULL,
            shard_count INTEGER NOT NULL
        );
        """
    )
    connection.executemany(
        "INSERT INTO table_counts(table_name, row_count, shard_count) VALUES (?, 0, 0)",
        ((name,) for name in _TABLE_FIELDS),
    )


def _export_batch(
    spec: PoolStreamSpec,
    batch: PoolBatch,
    connection: sqlite3.Connection,
    limits: StreamingLimits,
) -> None:
    rows_by_table = _batch_table_rows(spec, batch)
    byte_counter = _BatchByteCounter()
    try:
        with connection:
            _index_batch(spec, connection, batch)
            for table_name in _TABLE_FIELDS:
                table_spec = _TABLE_SPEC_BY_NAME[table_name]
                rows_written = 0
                connection.executemany(
                    "INSERT INTO spooled_rows VALUES (?, ?, ?, ?)",
                    _serialized_table_rows(
                        table_spec,
                        rows_by_table[table_name],
                        limits,
                        byte_counter,
                    ),
                )
                rows_written = _row_count_for_batch_table(batch, table_name)
                connection.execute(
                    "UPDATE table_counts SET row_count = row_count + ? WHERE table_name = ?",
                    (rows_written, table_name),
                )
    except sqlite3.IntegrityError as error:
        raise PoolStreamError(
            (f"global row uniqueness or key constraint failed: {error}",),
            limits.max_diagnostics,
        ) from error


def _batch_table_rows(
    spec: PoolStreamSpec,
    batch: PoolBatch,
) -> dict[str, Iterator[dict[str, object]]]:
    return {
        "polygon_inventory": (
            {
                "inventory_row_id": row.inventory_row_id,
                "native_key_json": native_key_json(row.native_key),
                "source_row_locator": row.source_row_locator,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "exclusion_reason": row.exclusion_reason,
            }
            for row in batch.inventory
        ),
        "polygon_snapshots": (
            {
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "inventory_row_id": row.inventory_row_id,
                "native_key_json": native_key_json(row.native_key),
                "geometry_uri": row.geometry_uri,
                "geometry_sha256": row.geometry_sha256,
                "geometry_format": row.geometry_format,
                "crs": row.crs,
                "coordinate_transform": row.coordinate_transform,
                "geometry_status": row.geometry_status.value,
                "original_area": row.original_area,
                "area_unit": row.area_unit,
            }
            for row in batch.polygons
        ),
        "label_candidates": (
            {
                "inventory_row_id": row.inventory_row_id,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "label_release": spec.reference_pin.release_id,
                "level": row.level,
                "code": row.code,
                "numerator": row.numerator,
                "denominator": row.denominator,
                "ratio": row.ratio,
                "threshold": row.threshold,
                "assignment_rule_version": row.assignment_rule_version,
                "assignment_kind": row.assignment_kind.value,
                "support_semantics": row.support_semantics.value,
                "support_unit": row.support_unit,
                "source_revision": row.source_revision,
                "taxonomy_revision": row.taxonomy_revision,
            }
            for row in batch.label_candidates
        ),
        "label_outcomes": (
            {
                "inventory_row_id": row.inventory_row_id,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "level": row.level,
                "status": row.status.value,
                "assigned_code": row.assigned_code,
                "reason": row.reason,
                "ambiguity_basis": row.ambiguity_basis.value if row.ambiguity_basis else None,
            }
            for row in batch.label_outcomes
        ),
        "polygon_attempts": (
            {
                "inventory_row_id": row.inventory_row_id,
                "status": row.status.value,
                "reference_id_namespace": row.reference_id_namespace,
                "attempted_reference_ids_json": _canonical_json_string_array(
                    row.attempted_reference_ids
                ),
                "missing_reference_ids_json": _canonical_json_string_array(
                    row.missing_reference_ids
                ),
                "measurement_sidecar_uri": row.measurement_sidecar_uri,
                "measurement_sidecar_sha256": row.measurement_sidecar_sha256,
                "measurement_record_count": row.measurement_record_count,
                "reason": row.reason,
            }
            for row in batch.attempt_receipts
        ),
        "label_support_evidence": (
            {
                "inventory_row_id": row.inventory_row_id,
                "level": row.level,
                "code": row.code,
                "reference_id_namespace": row.reference_id_namespace,
                "reference_ids_json": _canonical_json_string_array(row.reference_ids),
                "measurement_sidecar_uri": row.measurement_sidecar_uri,
                "measurement_sidecar_sha256": row.measurement_sidecar_sha256,
                "measurement_record_count": row.measurement_record_count,
            }
            for row in batch.support_evidence
        ),
        "source_documents": (
            {
                "document_id": row.document_id,
                "native_key_json": native_key_json(row.native_key),
                "source_revision": row.source_revision,
                "content_sha256": row.content_sha256,
                "body_status": row.body_status.value,
            }
            for row in batch.documents
        ),
        "document_relevance": (
            {
                "document_id": row.document_id,
                "release_id": row.release_id,
                "status": row.status.value,
                "source_repository": row.source_repository,
                "source_revision": row.source_revision,
                "mapping_rule_version": row.mapping_rule_version,
                "evidence": row.evidence,
                "raw_value": row.raw_value,
                "evidence_sha256": row.evidence_sha256,
                "reason": row.reason,
            }
            for row in batch.relevance_assessments
        ),
        "text_polygon_links": (
            {
                "document_id": row.document_id,
                "inventory_row_id": row.inventory_row_id,
                "native_link_key_json": native_key_json(row.native_link_key),
                "source_revision": row.source_revision,
                "status": row.status.value,
                "evidence": row.evidence,
                "candidate_inventory_row_ids_json": _canonical_json_string_array(
                    row.candidate_inventory_row_ids
                ),
            }
            for row in batch.text_polygon_links
        ),
        "entity_links": (
            {
                "link_version": row.link_version,
                "left_source_snapshot_id": row.left.source_snapshot_id,
                "left_native_key_json": native_key_json(row.left.native_key),
                "right_source_snapshot_id": row.right.source_snapshot_id,
                "right_native_key_json": native_key_json(row.right.native_key),
                "evidence": row.evidence,
                "evidence_sha256": row.evidence_sha256,
            }
            for row in batch.entity_links
        ),
        "pool_eligibility": (
            {
                "inventory_row_id": row.inventory_row_id,
                "cohort_policy_version": row.cohort_policy_version,
                "status": row.status.value,
                "evidence": row.evidence,
                "reason": row.reason,
                "replaced_by_inventory_row_id": row.replaced_by_inventory_row_id,
            }
            for row in batch.eligibility
        ),
    }


def _row_count_for_batch_table(batch: PoolBatch, table_name: str) -> int:
    rows = {
        "polygon_inventory": batch.inventory,
        "polygon_snapshots": batch.polygons,
        "label_candidates": batch.label_candidates,
        "label_outcomes": batch.label_outcomes,
        "polygon_attempts": batch.attempt_receipts,
        "label_support_evidence": batch.support_evidence,
        "source_documents": batch.documents,
        "document_relevance": batch.relevance_assessments,
        "text_polygon_links": batch.text_polygon_links,
        "entity_links": batch.entity_links,
        "pool_eligibility": batch.eligibility,
    }
    return len(rows[table_name])


def _serialized_table_rows(
    table_spec: _TableSpec,
    rows: Iterable[dict[str, object]],
    limits: StreamingLimits,
    byte_counter: _BatchByteCounter,
) -> Iterator[tuple[str, str, str, int]]:
    for row in rows:
        payload = _canonical_json_bytes(row).rstrip(b"\n")
        if len(payload) > limits.max_row_bytes:
            raise PoolStreamError(
                (f"{table_spec.name} row exceeds max_row_bytes={limits.max_row_bytes}",),
                limits.max_diagnostics,
            )
        byte_counter.value += len(payload)
        if byte_counter.value > limits.max_batch_bytes:
            raise PoolStreamError(
                (f"batch exceeds max_batch_bytes={limits.max_batch_bytes}",),
                limits.max_diagnostics,
            )
        yield (
            table_spec.name,
            _table_row_key(table_spec, row),
            payload.decode("utf-8"),
            len(payload),
        )


def _table_row_key(table_spec: _TableSpec, row: dict[str, object]) -> str:
    try:
        values = [row[field] for field in table_spec.sort_fields]
    except KeyError as error:
        raise PoolStreamError(
            (f"{table_spec.name} row is missing key field {error.args[0]}",)
        ) from error
    return _canonical_json_bytes(values).decode("utf-8").rstrip("\n")


def _bounded_items(items: Iterable[str], limit: int) -> Iterator[str]:
    iterator = iter(items)
    for _ in range(limit):
        try:
            yield next(iterator)
        except StopIteration:
            return
    try:
        next(iterator)
    except StopIteration:
        return
    yield f"additional diagnostics omitted after the first {limit} issues"


def _index_batch(
    spec: PoolStreamSpec,
    connection: sqlite3.Connection,
    batch: PoolBatch,
) -> None:
    for row in batch.inventory:
        connection.execute(
            "INSERT INTO inventory VALUES (?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                native_key_json(row.native_key),
                row.source_row_locator,
                row.polygon_snapshot_id,
                row.exclusion_reason,
            ),
        )
    for row in batch.polygons:
        connection.execute(
            "INSERT INTO polygons VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.polygon_snapshot_id,
                row.inventory_row_id,
                native_key_json(row.native_key),
                row.geometry_uri,
                row.geometry_sha256,
                row.geometry_format,
                row.crs,
                row.coordinate_transform,
                row.geometry_status.value,
                row.original_area,
                row.area_unit,
            ),
        )
    for row in batch.label_candidates:
        connection.execute(
            "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                row.level,
                row.code,
                row.polygon_snapshot_id,
                spec.reference_pin.release_id,
                row.numerator,
                row.denominator,
                row.ratio,
                row.threshold,
                row.assignment_rule_version,
                row.assignment_kind.value,
                row.support_semantics.value,
                row.support_unit,
                row.source_revision,
                row.taxonomy_revision,
            ),
        )
    for row in batch.label_outcomes:
        connection.execute(
            "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                row.level,
                row.polygon_snapshot_id,
                row.status.value,
                row.assigned_code,
                row.reason,
                row.ambiguity_basis.value if row.ambiguity_basis else None,
            ),
        )
    _index_evidence_rows(batch, connection)
    for row in batch.documents:
        connection.execute(
            "INSERT INTO source_documents VALUES (?, ?, ?, ?, ?)",
            (
                row.document_id,
                native_key_json(row.native_key),
                row.source_revision,
                row.content_sha256,
                row.body_status.value,
            ),
        )
    for row in batch.text_polygon_links:
        cursor = connection.execute(
            "INSERT INTO text_links "
            "(document_id, native_link_key_json, inventory_row_id, source_revision, "
            "status, evidence) VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.document_id,
                native_key_json(row.native_link_key),
                row.inventory_row_id,
                row.source_revision,
                row.status.value,
                row.evidence,
            ),
        )
        connection.executemany(
            "INSERT INTO link_candidates VALUES (?, ?)",
            ((cursor.lastrowid, item) for item in row.candidate_inventory_row_ids),
        )
    for row in batch.entity_links:
        left = _entity_ref_key(row.left.source_snapshot_id, row.left.native_key)
        right = _entity_ref_key(row.right.source_snapshot_id, row.right.native_key)
        connection.execute(
            "INSERT INTO entity_links VALUES (?, ?, ?, ?, ?)",
            (
                row.link_version,
                min(left, right),
                max(left, right),
                row.evidence,
                row.evidence_sha256,
            ),
        )
    for row in batch.eligibility:
        connection.execute(
            "INSERT INTO eligibility VALUES (?, ?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                row.cohort_policy_version,
                row.status.value,
                row.reason,
                row.replaced_by_inventory_row_id,
                row.evidence,
            ),
        )


def _index_evidence_rows(batch: PoolBatch, connection: sqlite3.Connection) -> None:
    for row in batch.attempt_receipts:
        connection.execute(
            "INSERT INTO polygon_attempts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                row.status.value,
                row.reference_id_namespace,
                _canonical_json_string_array(row.attempted_reference_ids),
                _canonical_json_string_array(row.missing_reference_ids),
                row.measurement_sidecar_uri,
                row.measurement_sidecar_sha256,
                row.measurement_record_count,
                row.reason,
            ),
        )
    for row in batch.support_evidence:
        connection.execute(
            "INSERT INTO support_evidence VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.inventory_row_id,
                row.level,
                row.code,
                row.reference_id_namespace,
                _canonical_json_string_array(row.reference_ids),
                row.measurement_sidecar_uri,
                row.measurement_sidecar_sha256,
                row.measurement_record_count,
            ),
        )
    for row in batch.relevance_assessments:
        connection.execute(
            "INSERT INTO document_relevance VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                row.document_id,
                row.release_id,
                row.status.value,
                row.source_repository,
                row.source_revision,
                row.mapping_rule_version,
                row.evidence,
                row.raw_value,
                row.evidence_sha256,
                row.reason,
            ),
        )


def _finalize_pool(
    spec: PoolStreamSpec,
    sink: FileArtifactSink,
    connection: sqlite3.Connection,
    work_dir: Path,
    root: str,
    limits: StreamingLimits,
) -> StreamingExportReceipt:
    issues = list(_validate_global_index(spec, connection, limits.max_diagnostics))
    total_keys = connection.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
    keyset_hash = _inventory_keyset_sha256(connection)
    evidence = spec.raw_inventory_evidence
    if total_keys != evidence.expected_key_count:
        issues.append("raw inventory key count differs from its independent scan evidence")
    if keyset_hash != evidence.expected_keyset_sha256:
        issues.append("raw inventory key set differs from its independent scan evidence")
    if issues:
        raise PoolStreamError(issues, limits.max_diagnostics)
    accounting = _level_accounting(spec, connection, total_keys)
    _write_shards(spec, sink, connection, work_dir, root, limits)
    shards = _write_shard_manifest(connection, work_dir)
    shard_manifest_path = f"{root}/shards.jsonl"
    shard_manifest_hash = _sha256_file(shards)
    shard_manifest_uri = sink.put_file(shard_manifest_path, shards)
    exclusion_path = work_dir / "exclusions.jsonl"
    exclusion_group_count = _write_exclusion_sidecar(connection, exclusion_path)
    exclusion_relative_path = f"{root}/exclusions.jsonl"
    exclusion_hash = _sha256_file(exclusion_path)
    sink.put_file(exclusion_relative_path, exclusion_path)
    manifest_bytes = _stream_manifest_bytes(
        spec,
        accounting,
        total_keys,
        connection,
        _ManifestArtifacts(
            shard_manifest_path=shard_manifest_path,
            shard_manifest_sha256=shard_manifest_hash,
            inventory_keyset_sha256=keyset_hash,
            exclusion_reason_path=exclusion_relative_path,
            exclusion_reason_sha256=exclusion_hash,
            exclusion_reason_group_count=exclusion_group_count,
        ),
    )
    manifest_path = f"{root}/manifest.json"
    manifest_uri = sink.put(manifest_path, manifest_bytes)
    return StreamingExportReceipt(
        pool_id=f"{spec.pool_version}/{spec.pool_key.name}/{spec.source_snapshot.snapshot_id}",
        manifest_uri=manifest_uri,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        shard_manifest_uri=shard_manifest_uri,
        shard_manifest_sha256=shard_manifest_hash,
        shard_count=_read_shards_count(connection),
        total_source_keys=total_keys,
        inventory_keyset_sha256=keyset_hash,
        level_accounting=accounting,
    )


def _validate_global_index(
    spec: PoolStreamSpec,
    connection: sqlite3.Connection,
    max_diagnostics: int,
) -> tuple[str, ...]:
    return tuple(_bounded_items(_iter_global_issues(spec, connection), max_diagnostics))


def _iter_global_issues(spec: PoolStreamSpec, connection: sqlite3.Connection) -> Iterator[str]:
    total_keys = connection.execute("SELECT COUNT(*) FROM inventory").fetchone()[0]
    yield from _iter_inventory_issues(spec, connection)
    yield from _iter_polygon_issues(spec, connection)
    yield from _iter_candidate_issues(spec, connection)
    yield from _iter_outcome_issues(spec, connection, total_keys)
    yield from _iter_attempt_receipt_issues(connection, total_keys)
    yield from _iter_support_evidence_issues(connection)
    yield from _iter_document_issues(connection)
    yield from _iter_document_relevance_issues(connection)
    yield from _iter_text_link_issues(connection)
    yield from _iter_entity_link_issues(connection)
    yield from _iter_eligibility_issues(connection)


def _iter_inventory_issues(spec: PoolStreamSpec, connection: sqlite3.Connection) -> Iterator[str]:
    for row_id, key_json, locator, polygon_id, exclusion_reason in connection.execute(
        "SELECT inventory_row_id, native_key_json, source_row_locator, "
        "polygon_snapshot_id, exclusion_reason FROM inventory ORDER BY inventory_row_id"
    ):
        key = _parse_native_key(key_json)
        if key is None:
            yield f"inventory row {row_id} has an invalid source-native composite key"
            continue
        if row_id != inventory_row_identifier(spec.source_snapshot.snapshot_id, key):
            yield f"inventory row id does not encode its source pin and native key: {row_id}"
        if not _present(locator):
            yield f"inventory row {row_id} has no source-row locator"
        if polygon_id is None and not _present(exclusion_reason):
            yield f"inventory row {row_id} has no geometry snapshot or exclusion"
        if polygon_id is not None and exclusion_reason is not None:
            yield f"inventory row {row_id} has both snapshot and exclusion"


def _iter_polygon_issues(spec: PoolStreamSpec, connection: sqlite3.Connection) -> Iterator[str]:
    yield from _iter_polygon_link_issues(connection)
    for row in connection.execute(
        "SELECT p.polygon_snapshot_id, p.native_key_json, p.geometry_uri, "
        "p.geometry_sha256, p.geometry_format, p.crs, p.coordinate_transform, "
        "p.geometry_status, p.original_area, p.area_unit, "
        "i.native_key_json AS inventory_key, i.polygon_snapshot_id AS inventory_polygon "
        "FROM polygons p LEFT JOIN inventory i ON i.inventory_row_id=p.inventory_row_id "
        "ORDER BY p.polygon_snapshot_id"
    ):
        yield from _polygon_record_issues(spec, row)


def _iter_polygon_link_issues(connection: sqlite3.Connection) -> Iterator[str]:
    polygon_mismatch = connection.execute(
        "SELECT COUNT(*) FROM inventory i LEFT JOIN polygons p "
        "ON p.inventory_row_id = i.inventory_row_id "
        "WHERE (i.polygon_snapshot_id IS NULL AND p.polygon_snapshot_id IS NOT NULL) "
        "OR (i.polygon_snapshot_id IS NOT NULL AND p.polygon_snapshot_id IS NULL) "
        "OR (i.polygon_snapshot_id IS NOT NULL AND i.polygon_snapshot_id != p.polygon_snapshot_id)"
    ).fetchone()[0]
    if polygon_mismatch:
        yield "inventory-to-polygon snapshot links are incomplete or inconsistent"
    polygon_orphans = connection.execute(
        "SELECT COUNT(*) FROM polygons p LEFT JOIN inventory i "
        "ON i.inventory_row_id = p.inventory_row_id WHERE i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if polygon_orphans:
        yield "polygon snapshot references an unknown inventory row"


def _polygon_record_issues(spec: PoolStreamSpec, row: tuple[object, ...]) -> Iterator[str]:
    (
        polygon_id,
        key_json,
        uri,
        geometry_hash,
        geometry_format,
        crs,
        transform,
        status,
        area,
        area_unit,
        inventory_key_json,
        inventory_polygon_id,
    ) = row
    key = _parse_native_key(key_json)
    if key is None:
        yield f"polygon snapshot {polygon_id} has an invalid source-native key"
        return
    if inventory_key_json is None:
        return
    if key_json != inventory_key_json or polygon_id != inventory_polygon_id:
        yield f"polygon snapshot {polygon_id} does not match its inventory identity"
    if not isinstance(geometry_hash, str):
        yield f"polygon snapshot {polygon_id} geometry artifact is not pinned"
        return
    expected_id = polygon_snapshot_identifier(spec.source_snapshot.snapshot_id, key, geometry_hash)
    if polygon_id != expected_id:
        yield "polygon snapshot id does not encode key and geometry hash"
    if not _valid_resolvable_uri(uri) or not _valid_sha256(geometry_hash):
        yield f"polygon snapshot {polygon_id} geometry artifact is not pinned"
    if not all(_present(value) for value in (geometry_format, crs, transform, area_unit)):
        yield f"polygon snapshot {polygon_id} lacks geometry metadata"
    yield from _polygon_area_issues(polygon_id, status, area)


def _polygon_area_issues(polygon_id: object, status: object, area: object) -> Iterator[str]:
    if status not in {value.value for value in GeometryStatus}:
        yield f"polygon snapshot {polygon_id} has an unknown geometry status"
    elif status == GeometryStatus.USABLE.value:
        numbers = _finite_numbers((area,))
        if numbers is None or numbers[0] <= 0:
            yield f"usable polygon snapshot {polygon_id} needs positive full area"
    elif area is not None:
        yield f"unusable polygon snapshot {polygon_id} cannot claim full area"


def _iter_candidate_issues(spec: PoolStreamSpec, connection: sqlite3.Connection) -> Iterator[str]:
    for row in connection.execute(
        "SELECT c.inventory_row_id, c.level, c.code, c.polygon_snapshot_id, "
        "c.label_release, c.numerator, c.denominator, c.ratio, c.threshold, "
        "c.assignment_rule_version, c.assignment_kind, c.support_semantics, "
        "c.support_unit, c.source_revision, c.taxonomy_revision, "
        "i.polygon_snapshot_id AS inventory_polygon, p.geometry_status, p.original_area, "
        "p.area_unit AS polygon_area_unit "
        "FROM candidates c LEFT JOIN inventory i ON i.inventory_row_id=c.inventory_row_id "
        "LEFT JOIN polygons p ON p.polygon_snapshot_id=c.polygon_snapshot_id "
        "ORDER BY c.inventory_row_id, c.level, c.code"
    ):
        yield from _candidate_row_issues(spec, row)


def _candidate_row_issues(spec: PoolStreamSpec, row: tuple[object, ...]) -> Iterator[str]:
    (
        _inventory_id,
        level,
        code,
        polygon_id,
        _release,
        numerator,
        denominator,
        ratio,
        threshold,
        _rule_version,
        _assignment_kind,
        _support_semantics,
        _support_unit,
        _source_revision,
        _taxonomy_revision,
        inventory_polygon_id,
        geometry_status,
        original_area,
        polygon_area_unit,
    ) = row
    if inventory_polygon_id is None or geometry_status is None:
        yield f"label candidate {code} references an unknown polygon"
        return
    if inventory_polygon_id != polygon_id:
        yield f"label candidate {code} does not match its inventory geometry"
    if geometry_status != GeometryStatus.USABLE.value or original_area is None:
        yield f"label candidate {code} uses unusable polygon geometry"
        return
    supported_levels = {item.name for item in spec.levels if item.supported}
    if level not in supported_levels:
        yield f"label candidate {code} uses an undeclared or unsupported level"
    yield from _candidate_measurement_issues(
        (code, numerator, denominator, ratio, threshold, original_area)
    )
    if row[12] != polygon_area_unit:
        yield f"label candidate {code} does not use polygon support units"
    yield from _candidate_pin_issues(spec, row)


def _candidate_measurement_issues(values: tuple[object, ...]) -> Iterator[str]:
    code, numerator, denominator, ratio, threshold, original_area = values
    numbers = _finite_numbers((numerator, denominator, ratio, threshold, original_area))
    if numbers is None:
        yield f"label candidate {code} has a non-finite measurement"
        return
    numerator_value, denominator_value, ratio_value, _threshold_value, area_value = numbers
    if denominator_value <= 0 or numerator_value < 0:
        yield f"label candidate {code} has an invalid numerator or denominator"
    if numerator_value > denominator_value:
        yield f"label candidate {code} support exceeds full polygon area"
    if not math.isclose(denominator_value, area_value, rel_tol=1e-12, abs_tol=1e-12):
        yield f"label candidate {code} denominator is not full original polygon area"
    expected_ratio = numerator_value / denominator_value if denominator_value > 0 else math.nan
    if not math.isclose(ratio_value, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        yield f"label candidate {code} ratio does not match numerator/denominator"


def _candidate_pin_issues(
    spec: PoolStreamSpec,
    row: tuple[object, ...],
) -> Iterator[str]:
    (
        _inventory_id,
        _level,
        code,
        _polygon_id,
        release,
        _numerator,
        _denominator,
        _ratio,
        threshold,
        rule_version,
        assignment_kind,
        support_semantics,
        support_unit,
        source_revision,
        taxonomy_revision,
        *_geometry,
    ) = row
    reference = spec.reference_pin
    if threshold != reference.threshold:
        yield f"label candidate {code} does not use the pinned threshold"
    if support_semantics != reference.support_semantics.value:
        yield f"label candidate {code} changes reference support semantics"
    if support_unit != reference.support_unit:
        yield f"label candidate {code} does not use the pinned support unit"
    if source_revision != reference.source_revision:
        yield f"label candidate {code} has a mismatched source revision"
    if taxonomy_revision != reference.taxonomy_revision:
        yield f"label candidate {code} has a mismatched taxonomy revision"
    if not _present(rule_version) or not _present(code):
        yield f"label candidate {code} lacks code or assignment-rule version"
    if assignment_kind not in {value.value for value in AssignmentKind}:
        yield f"label candidate {code} has an unknown assignment kind"
    if release != reference.release_id:
        yield f"label candidate {code} has a mismatched label release"


def _iter_outcome_issues(
    spec: PoolStreamSpec,
    connection: sqlite3.Connection,
    total_keys: int,
) -> Iterator[str]:
    levels = {level.name: level.supported for level in spec.levels}
    for level in spec.levels:
        outcome_count = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE level=?", (level.name,)
        ).fetchone()[0]
        expected_count = total_keys if level.supported else 0
        if outcome_count != expected_count:
            yield (
                f"level {level.name} decision coverage mismatch: "
                f"found {outcome_count}, expected {expected_count}"
            )
        if level.supported:
            missing_count = connection.execute(
                "SELECT COUNT(*) FROM inventory i LEFT JOIN outcomes o "
                "ON o.inventory_row_id=i.inventory_row_id AND o.level=? "
                "WHERE o.inventory_row_id IS NULL",
                (level.name,),
            ).fetchone()[0]
            if missing_count:
                yield f"level {level.name} decision coverage mismatch: {missing_count} missing"
    for row in connection.execute(
        "SELECT o.inventory_row_id, o.level, o.polygon_snapshot_id, o.status, "
        "o.assigned_code, o.reason, o.ambiguity_basis, "
        "i.polygon_snapshot_id AS inventory_polygon, p.geometry_status, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.inventory_row_id=o.inventory_row_id "
        "AND c.level=o.level AND c.ratio >= c.threshold) AS qualifier_count, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.inventory_row_id=o.inventory_row_id "
        "AND c.level=o.level) AS candidate_count, "
        "(SELECT COUNT(*) FROM candidates c WHERE c.inventory_row_id=o.inventory_row_id "
        "AND c.level=o.level AND c.ratio >= c.threshold AND c.code=o.assigned_code) "
        "AS assigned_match_count, i.inventory_row_id AS matched_inventory_id "
        "FROM outcomes o LEFT JOIN inventory i ON i.inventory_row_id=o.inventory_row_id "
        "LEFT JOIN polygons p ON p.inventory_row_id=i.inventory_row_id "
        "ORDER BY o.inventory_row_id, o.level"
    ):
        yield from _outcome_row_issues(row, levels)


def _outcome_row_issues(
    row: tuple[object, ...],
    levels: dict[str, bool],
) -> Iterator[str]:
    (
        inventory_id,
        level_name,
        polygon_id,
        status,
        assigned_code,
        reason,
        ambiguity_basis,
        inventory_polygon_id,
        geometry_status,
        qualifier_count,
        candidate_count,
        assigned_match_count,
        matched_inventory_id,
    ) = row
    key = inventory_id, level_name
    if matched_inventory_id is None:
        yield f"label outcome {key} references unknown inventory"
        return
    if levels.get(level_name) is not True:
        yield f"label outcome {key} uses an undeclared or unsupported level"
    if polygon_id != inventory_polygon_id:
        yield f"label outcome {key} has a mismatched polygon snapshot"
    if (geometry_status is None or geometry_status != GeometryStatus.USABLE.value) and (
        status != OutcomeStatus.EXCLUDED.value
    ):
        yield f"unusable geometry must have an excluded outcome for {key}"
    yield from _outcome_status_issues(
        _OutcomeStatusCheck(
            key=(cast(str, inventory_id), cast(str, level_name)),
            status=cast(str, status),
            assigned_code=cast(str | None, assigned_code),
            reason=cast(str | None, reason),
            ambiguity_basis=cast(str | None, ambiguity_basis),
            qualifier_count=cast(int, qualifier_count),
            candidate_count=cast(int, candidate_count),
            assigned_match_count=cast(int, assigned_match_count),
        )
    )


def _outcome_status_issues(outcome: _OutcomeStatusCheck) -> Iterator[str]:
    if outcome.status == OutcomeStatus.ASSIGNED.value:
        yield from _assigned_outcome_issues(outcome)
    elif outcome.status == OutcomeStatus.AMBIGUOUS.value:
        yield from _ambiguous_outcome_issues(outcome)
    elif outcome.status == OutcomeStatus.UNASSIGNED.value:
        yield from _unassigned_outcome_issues(outcome)
    elif outcome.status == OutcomeStatus.EXCLUDED.value:
        yield from _excluded_outcome_issues(outcome)
    else:
        yield f"label outcome {outcome.key} has an unknown status"


def _assigned_outcome_issues(outcome: _OutcomeStatusCheck) -> Iterator[str]:
    if outcome.qualifier_count != 1 or outcome.assigned_match_count != 1:
        yield f"assigned outcome {outcome.key} must select its only qualifying code"
    if outcome.reason is not None or outcome.ambiguity_basis is not None:
        yield f"assigned outcome {outcome.key} cannot have an exclusion or ambiguity reason"


def _unassigned_outcome_issues(outcome: _OutcomeStatusCheck) -> Iterator[str]:
    if outcome.qualifier_count:
        yield f"unassigned outcome {outcome.key} has one or more qualifying codes"
    if outcome.assigned_code is not None or not _present(outcome.reason):
        yield f"unassigned outcome {outcome.key} must have a reason and no assigned code"
    if outcome.ambiguity_basis is not None:
        yield f"unassigned outcome {outcome.key} cannot have ambiguity basis"


def _excluded_outcome_issues(outcome: _OutcomeStatusCheck) -> Iterator[str]:
    if outcome.assigned_code is not None or not _present(outcome.reason):
        yield f"excluded outcome {outcome.key} must have a reason and no assigned code"
    if outcome.ambiguity_basis is not None or outcome.candidate_count:
        yield f"excluded outcome {outcome.key} cannot contain assignment candidates"


def _ambiguous_outcome_issues(outcome: _OutcomeStatusCheck) -> Iterator[str]:
    valid_bases = {value.value for value in AmbiguityBasis}
    if outcome.assigned_code is not None or not _present(outcome.reason):
        yield f"ambiguous outcome {outcome.key} must remain unassigned with a reason"
    if outcome.ambiguity_basis not in valid_bases:
        yield f"ambiguous outcome {outcome.key} must state its ambiguity basis"
    elif outcome.ambiguity_basis == AmbiguityBasis.MULTIPLE_QUALIFIERS.value:
        if outcome.qualifier_count < _MIN_AMBIGUOUS_COUNT:
            yield f"ambiguous outcome {outcome.key} needs at least two qualifying codes"
    elif (
        outcome.ambiguity_basis == AmbiguityBasis.AMBIGUOUS_EVIDENCE.value
        and outcome.candidate_count == 0
    ):
        yield f"ambiguous-evidence outcome {outcome.key} needs inspectable candidate evidence"
    if (
        outcome.qualifier_count == 1
        and outcome.ambiguity_basis != AmbiguityBasis.AMBIGUOUS_EVIDENCE.value
    ):
        yield f"single qualifying code at {outcome.key} requires an evidence-ambiguity basis"
    if (
        outcome.qualifier_count > 1
        and outcome.ambiguity_basis != AmbiguityBasis.MULTIPLE_QUALIFIERS.value
    ):
        yield f"competing qualifying codes at {outcome.key} require multiple-qualifier ambiguity"


def _iter_attempt_receipt_issues(
    connection: sqlite3.Connection,
    total_keys: int,
) -> Iterator[str]:
    attempt_count = connection.execute("SELECT COUNT(*) FROM polygon_attempts").fetchone()[0]
    if attempt_count != total_keys:
        yield "polygon attempt receipt count does not match raw inventory"
    unknown_inventory = connection.execute(
        "SELECT COUNT(*) FROM polygon_attempts a LEFT JOIN inventory i "
        "ON i.inventory_row_id=a.inventory_row_id WHERE i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if unknown_inventory:
        yield "polygon attempt receipt references unknown inventory"
    for (
        row_id,
        status,
        namespace,
        attempted_json,
        missing_json,
        uri,
        digest,
        count,
        reason,
    ) in connection.execute(
        "SELECT inventory_row_id, status, reference_id_namespace, "
        "attempted_reference_ids_json, missing_reference_ids_json, "
        "measurement_sidecar_uri, measurement_sidecar_sha256, measurement_record_count, reason "
        "FROM polygon_attempts ORDER BY inventory_row_id"
    ):
        if status not in {item.value for item in AttemptStatus}:
            yield f"polygon attempt receipt for {row_id} has an unknown status"
        if not _present(namespace):
            yield f"polygon attempt receipt for {row_id} needs a reference ID namespace"
        yield from _json_id_array_issues(attempted_json, f"attempted references for {row_id}")
        yield from _json_id_array_issues(missing_json, f"missing references for {row_id}")
        if status in {
            AttemptStatus.PARTIAL.value,
            AttemptStatus.FAILED.value,
            AttemptStatus.EXCLUDED.value,
        } and not _present(reason):
            yield f"non-complete polygon attempt {row_id} needs a reason"
        yield from _sidecar_issues(uri, digest, count, f"polygon attempt {row_id}")


def _iter_support_evidence_issues(connection: sqlite3.Connection) -> Iterator[str]:
    missing = connection.execute(
        "SELECT COUNT(*) FROM candidates c LEFT JOIN support_evidence e "
        "ON e.inventory_row_id=c.inventory_row_id AND e.level=c.level AND e.code=c.code "
        "WHERE e.inventory_row_id IS NULL"
    ).fetchone()[0]
    extra = connection.execute(
        "SELECT COUNT(*) FROM support_evidence e LEFT JOIN candidates c "
        "ON c.inventory_row_id=e.inventory_row_id AND c.level=e.level AND c.code=e.code "
        "WHERE c.inventory_row_id IS NULL"
    ).fetchone()[0]
    if missing or extra:
        yield "candidate support evidence coverage mismatch"
    for row_id, level, code, namespace, ids_json, uri, digest, count in connection.execute(
        "SELECT e.inventory_row_id, e.level, e.code, e.reference_id_namespace, "
        "e.reference_ids_json, e.measurement_sidecar_uri, e.measurement_sidecar_sha256, "
        "e.measurement_record_count FROM support_evidence e "
        "ORDER BY e.inventory_row_id, e.level, e.code"
    ):
        key = f"({row_id}, {level}, {code})"
        if not _present(namespace):
            yield f"label support evidence {key} needs a reference ID namespace"
        parsed_ids = _parse_json_id_array(ids_json)
        if parsed_ids is None:
            yield f"label support evidence {key} has invalid reference IDs"
            parsed_ids = ()
        has_ids = bool(parsed_ids)
        has_sidecar = uri is not None
        if has_ids == has_sidecar:
            yield (
                f"label support evidence {key} needs either reference IDs or a measurement sidecar"
            )
        yield from _sidecar_issues(uri, digest, count, f"label support evidence {key}")


def _iter_document_relevance_issues(connection: sqlite3.Connection) -> Iterator[str]:
    unknown_documents = connection.execute(
        "SELECT COUNT(*) FROM document_relevance r LEFT JOIN source_documents d "
        "ON d.document_id=r.document_id WHERE d.document_id IS NULL"
    ).fetchone()[0]
    if unknown_documents:
        yield "relevance assessment references unknown source document"
    for row in connection.execute(
        "SELECT document_id, release_id, status, source_repository, source_revision, "
        "mapping_rule_version, evidence, raw_value, evidence_sha256, reason "
        "FROM document_relevance ORDER BY document_id, release_id"
    ):
        (
            document_id,
            release_id,
            status,
            repository,
            revision,
            rule,
            evidence,
            raw,
            digest,
            reason,
        ) = row
        key = f"({document_id}, {release_id})"
        if status not in {item.value for item in RelevanceStatus}:
            yield f"relevance assessment {key} has an unknown status"
            continue
        if not all(
            _present(value)
            for value in (document_id, release_id, repository, revision, rule, evidence)
        ):
            yield f"relevance assessment {key} lacks provenance"
        if status in {RelevanceStatus.YES.value, RelevanceStatus.NO.value} and not _present(raw):
            yield f"yes/no relevance assessment {key} must preserve its source value"
        if status == RelevanceStatus.UNKNOWN.value and not _present(reason):
            yield f"unknown relevance assessment {key} needs a reason"
        if status == RelevanceStatus.MISSING.value and (raw is not None or not _present(reason)):
            yield f"missing relevance assessment {key} needs an absence reason only"
        if digest is not None and not _valid_sha256(digest):
            yield f"relevance assessment {key} has an invalid evidence SHA-256"


def _json_id_array_issues(value: object, name: str) -> Iterator[str]:
    parsed = _parse_json_id_array(value)
    if parsed is None:
        yield f"{name} are not a canonical unique string array"


def _parse_json_id_array(value: object) -> tuple[str, ...] | None:
    if not isinstance(value, str):
        return None
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError:
        return None
    if not isinstance(decoded, list) or any(
        not isinstance(item, str) or not _present(item) for item in decoded
    ):
        return None
    values = tuple(decoded)
    if len(values) != len(set(values)) or tuple(sorted(values)) != values:
        return None
    if _canonical_json_string_array(values) != value:
        return None
    return values


def _sidecar_issues(
    uri: object,
    digest: object,
    count: object,
    name: str,
) -> Iterator[str]:
    present = (uri is not None, digest is not None, count is not None)
    if any(present) and not all(present):
        yield f"{name} sidecar fields must be supplied together"
    elif all(present):
        if not _valid_resolvable_uri(uri):
            yield f"{name} sidecar URI is invalid"
        if not _valid_sha256(digest):
            yield f"{name} sidecar SHA-256 is invalid"
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            yield f"{name} sidecar record count must be positive"


def _iter_document_issues(connection: sqlite3.Connection) -> Iterator[str]:
    for document_id, key_json, revision, content_hash, body_status in connection.execute(
        "SELECT document_id, native_key_json, source_revision, content_sha256, body_status "
        "FROM source_documents ORDER BY document_id"
    ):
        if not _present(document_id) or _parse_native_key(key_json) is None:
            yield f"source document {document_id} needs a native composite key"
        if not _present(revision):
            yield f"source document {document_id} has no source revision"
        if body_status not in {value.value for value in BodyStatus}:
            yield f"source document {document_id} has an unknown body status"
        elif body_status == BodyStatus.AVAILABLE.value and not _valid_sha256(content_hash):
            yield f"available source document {document_id} needs a content hash"
        if content_hash is not None and not _valid_sha256(content_hash):
            yield f"source document {document_id} has an invalid content hash"


def _iter_text_link_issues(connection: sqlite3.Connection) -> Iterator[str]:
    yield from _iter_text_link_reference_issues(connection)
    for row in connection.execute(
        "SELECT t.link_id, t.document_id, t.native_link_key_json, t.inventory_row_id, "
        "t.source_revision, t.status, t.evidence, COUNT(c.inventory_row_id) "
        "FROM text_links t LEFT JOIN link_candidates c ON c.link_id=t.link_id "
        "GROUP BY t.link_id ORDER BY t.link_id"
    ):
        yield from _text_link_record_issues(row)


def _iter_text_link_reference_issues(connection: sqlite3.Connection) -> Iterator[str]:
    unknown_documents = connection.execute(
        "SELECT COUNT(*) FROM text_links t LEFT JOIN source_documents d "
        "ON d.document_id=t.document_id WHERE d.document_id IS NULL"
    ).fetchone()[0]
    if unknown_documents:
        yield "text link references an unknown source document"
    unknown_inventory = connection.execute(
        "SELECT COUNT(*) FROM text_links t LEFT JOIN inventory i "
        "ON i.inventory_row_id=t.inventory_row_id WHERE t.inventory_row_id IS NOT NULL "
        "AND i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if unknown_inventory:
        yield "text link references an unknown inventory row"
    unknown_candidates = connection.execute(
        "SELECT COUNT(*) FROM link_candidates c LEFT JOIN inventory i "
        "ON i.inventory_row_id=c.inventory_row_id WHERE i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if unknown_candidates:
        yield "text link candidate references an unknown inventory row"


def _text_link_record_issues(row: tuple[object, ...]) -> Iterator[str]:
    link_id, document_id, key_json, inventory_id, revision, status, evidence, count = row
    if not _present(document_id) or _parse_native_key(key_json) is None:
        yield f"text link {link_id} needs a source-native composite key"
    if not _present(revision) or not _present(evidence):
        yield f"text link {link_id} must retain revision and link evidence"
    if status not in {value.value for value in TextLinkStatus}:
        yield f"text link {link_id} has an unknown status"
        return
    if not isinstance(count, int):
        yield f"text link {link_id} has an invalid candidate count"
        return
    if status == TextLinkStatus.LINKED.value and (inventory_id is None or count):
        yield f"linked document {document_id} needs one selected polygon"
    if status != TextLinkStatus.LINKED.value and inventory_id is not None:
        yield f"unresolved text link {document_id} cannot select a polygon"
    if status == TextLinkStatus.AMBIGUOUS.value and count < _MIN_AMBIGUOUS_COUNT:
        yield f"ambiguous text link {document_id} needs two polygon candidates"
    if status == TextLinkStatus.UNMATCHED.value and count:
        yield f"unmatched text link {document_id} cannot contain candidates"


def _iter_entity_link_issues(connection: sqlite3.Connection) -> Iterator[str]:
    for version, left_ref, right_ref, evidence, evidence_hash in connection.execute(
        "SELECT link_version, left_ref, right_ref, evidence, evidence_sha256 "
        "FROM entity_links ORDER BY link_version, left_ref, right_ref"
    ):
        left = _parse_entity_ref(left_ref)
        right = _parse_entity_ref(right_ref)
        if not _present(version) or not _present(evidence):
            yield "entity links require a version and evidence"
        if left is None or right is None:
            yield "entity-link endpoints need source snapshots and full native keys"
            continue
        if left == right:
            yield "entity link cannot link a polygon to itself"
        if evidence_hash is not None and not _valid_sha256(evidence_hash):
            yield "entity-link evidence SHA-256 is invalid"


def _iter_eligibility_issues(connection: sqlite3.Connection) -> Iterator[str]:
    yield from _iter_eligibility_reference_issues(connection)
    for row in connection.execute(
        "SELECT inventory_row_id, cohort_policy_version, status, reason, "
        "replaced_by_inventory_row_id, evidence FROM eligibility "
        "ORDER BY inventory_row_id, cohort_policy_version"
    ):
        yield from _eligibility_record_issues(row)


def _iter_eligibility_reference_issues(connection: sqlite3.Connection) -> Iterator[str]:
    unknown_inventory = connection.execute(
        "SELECT COUNT(*) FROM eligibility e LEFT JOIN inventory i "
        "ON i.inventory_row_id=e.inventory_row_id WHERE i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if unknown_inventory:
        yield "eligibility record references unknown inventory"
    unknown_replacements = connection.execute(
        "SELECT COUNT(*) FROM eligibility e LEFT JOIN inventory i "
        "ON i.inventory_row_id=e.replaced_by_inventory_row_id "
        "WHERE e.replaced_by_inventory_row_id IS NOT NULL AND i.inventory_row_id IS NULL"
    ).fetchone()[0]
    if unknown_replacements:
        yield "replacement target for an eligibility record is not in raw inventory"


def _eligibility_record_issues(row: tuple[object, ...]) -> Iterator[str]:
    row_id, policy, status, reason, replacement, evidence = row
    key = row_id, policy
    if not _present(policy) or not _present(evidence):
        yield f"eligibility record {key} needs policy version and evidence"
    if status == EligibilityStatus.ELIGIBLE.value:
        if reason is not None or replacement is not None:
            yield f"eligible record {key} cannot have exclusion or replacement fields"
    elif status == EligibilityStatus.EXCLUDED.value:
        if not _present(reason) or replacement is not None:
            yield f"excluded record {key} needs a reason and no replacement"
    elif status == EligibilityStatus.REPLACED.value:
        if not _present(reason) or not replacement:
            yield f"replaced record {key} needs reason and replacement lineage"
        elif replacement == row_id:
            yield f"replacement record {key} cannot replace itself"
    else:
        yield f"eligibility record {key} has an unknown status"


def _level_accounting(
    spec: PoolStreamSpec,
    connection: sqlite3.Connection,
    total_keys: int,
) -> tuple[StreamingLevelAccounting, ...]:
    counts: list[StreamingLevelAccounting] = []
    for level in sorted(spec.levels, key=lambda item: item.name):
        candidate_count = connection.execute(
            "SELECT COUNT(*) FROM candidates WHERE level=? AND ratio >= threshold",
            (level.name,),
        ).fetchone()[0]
        status_counts = dict(
            connection.execute(
                "SELECT status, COUNT(*) FROM outcomes WHERE level=? GROUP BY status",
                (level.name,),
            )
        )
        exclusion_group_count = connection.execute(
            "SELECT COUNT(*) FROM (SELECT reason FROM outcomes "
            "WHERE level=? AND status='excluded' GROUP BY reason)",
            (level.name,),
        ).fetchone()[0]
        processed = connection.execute(
            "SELECT COUNT(*) FROM outcomes WHERE level=?", (level.name,)
        ).fetchone()[0]
        counts.append(
            StreamingLevelAccounting(
                level=level.name,
                supported=level.supported,
                unsupported_reason=level.unsupported_reason,
                total_source_keys=total_keys,
                processed_decisions=processed,
                qualifying_candidates=candidate_count,
                assigned=status_counts.get("assigned", 0),
                ambiguous=status_counts.get("ambiguous", 0),
                unassigned=status_counts.get("unassigned", 0),
                excluded=status_counts.get("excluded", 0),
                exclusion_reason_group_count=exclusion_group_count,
            )
        )
    return tuple(counts)


def _write_shards(
    spec: PoolStreamSpec,
    sink: FileArtifactSink,
    connection: sqlite3.Connection,
    work_dir: Path,
    root: str,
    limits: StreamingLimits,
) -> None:
    schemas = table_schemas(spec.contract_version)
    context = _ShardWriteContext(sink, connection, work_dir, root, schemas)
    for table_spec in _TABLE_REGISTRY:
        cursor = connection.execute(
            "SELECT row_json, row_bytes FROM spooled_rows "
            "WHERE table_name=? ORDER BY sort_key, row_json",
            (table_spec.name,),
        )
        shard_rows: list[dict[str, object]] = []
        shard_bytes = 0
        shard_index = 0
        for row_json, row_bytes in cursor:
            if shard_rows and (
                len(shard_rows) >= limits.max_shard_rows
                or shard_bytes + row_bytes > limits.max_shard_bytes
            ):
                _write_one_shard(
                    table_spec.name,
                    shard_index,
                    shard_rows,
                    context,
                )
                shard_index += 1
                shard_rows = []
                shard_bytes = 0
            shard_rows.append(json.loads(row_json))
            shard_bytes += row_bytes
        if shard_rows:
            _write_one_shard(
                table_spec.name,
                shard_index,
                shard_rows,
                context,
            )


def _write_one_shard(
    table_name: str,
    shard_index: int,
    rows: list[dict[str, object]],
    context: _ShardWriteContext,
) -> None:
    table = pa.Table.from_pylist(rows, schema=context.schemas[table_name])
    local_path = context.work_dir / f"{table_name}-{shard_index:08d}.parquet"
    pq.write_table(
        table,
        local_path,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
        row_group_size=65_536,
    )
    relative_path = f"{context.root}/shards/{table_name}/part-{shard_index:08d}.parquet"
    digest = _sha256_file(local_path)
    context.sink.put_file(relative_path, local_path)
    with context.connection:
        context.connection.execute(
            "INSERT INTO shards VALUES (?, ?, ?, ?, ?)",
            (table_name, shard_index, relative_path, digest, len(rows)),
        )
        context.connection.execute(
            "UPDATE table_counts SET shard_count = shard_count + 1 WHERE table_name = ?",
            (table_name,),
        )
    local_path.unlink(missing_ok=True)


def _write_shard_manifest(
    connection: sqlite3.Connection,
    work_dir: Path,
) -> Path:
    path = work_dir / "shards.jsonl"
    with path.open("wb") as output:
        cursor = connection.execute(
            "SELECT table_name, shard_index, path, sha256, row_count "
            "FROM shards ORDER BY table_name, shard_index"
        )
        for table, shard_index, artifact_path, digest, row_count in cursor:
            record = {
                "table": table,
                "shard_index": shard_index,
                "path": artifact_path,
                "sha256": digest,
                "row_count": row_count,
            }
            output.write(_canonical_json_bytes(record))
    return path


def _write_exclusion_sidecar(connection: sqlite3.Connection, path: Path) -> int:
    group_count = 0
    with path.open("wb") as output:
        cursor = connection.execute(
            "SELECT level, reason, COUNT(*) FROM outcomes "
            "WHERE status='excluded' GROUP BY level, reason ORDER BY level, reason"
        )
        for level, reason, count in cursor:
            output.write(_canonical_json_bytes({"level": level, "reason": reason, "count": count}))
            group_count += 1
    return group_count


def _stream_manifest_bytes(
    spec: PoolStreamSpec,
    accounting: tuple[StreamingLevelAccounting, ...],
    total_keys: int,
    connection: sqlite3.Connection,
    artifacts: _ManifestArtifacts,
) -> bytes:
    counts = {
        table: {"row_count": row_count, "shard_count": shard_count}
        for table, row_count, shard_count in connection.execute(
            "SELECT table_name, row_count, shard_count FROM table_counts ORDER BY table_name"
        )
    }
    levels = tuple(sorted(spec.levels, key=lambda item: item.name))
    return _canonical_json_bytes(
        {
            "contract": "noeflandre.polygon-label-pool",
            "contract_version": spec.contract_version,
            "pool_version": spec.pool_version,
            "pool_id": (
                f"{spec.pool_version}/{spec.pool_key.name}/{spec.source_snapshot.snapshot_id}"
            ),
            "storage_layout_version": _STORAGE_LAYOUT_VERSION,
            "implementation_pins": {
                "adapter_id": spec.adapter_id,
                "adapter_revision": spec.adapter_revision,
                "builder_revision": spec.builder_revision,
            },
            "pool_key": {
                "text_source": spec.pool_key.text_source.value,
                "reference_system": spec.pool_key.reference_system.value,
            },
            "source_snapshot": {
                "snapshot_id": spec.source_snapshot.snapshot_id,
                "repository": spec.source_snapshot.repository,
                "revision": spec.source_snapshot.revision,
                "pbf_uri": spec.source_snapshot.pbf_uri,
                "pbf_sha256": spec.source_snapshot.pbf_sha256,
                "source_artifact_manifest": _artifact_manifest_dict(
                    spec.source_snapshot.source_artifact_manifest
                ),
            },
            "raw_inventory_evidence": {
                "enumeration_rule": spec.raw_inventory_evidence.enumeration_rule,
                "scan_report_uri": spec.raw_inventory_evidence.scan_report_uri,
                "scan_report_sha256": spec.raw_inventory_evidence.scan_report_sha256,
                "expected_key_count": spec.raw_inventory_evidence.expected_key_count,
                "expected_keyset_sha256": spec.raw_inventory_evidence.expected_keyset_sha256,
            },
            "reference_pin": {
                "release_id": spec.reference_pin.release_id,
                "source_repository": spec.reference_pin.source_repository,
                "source_revision": spec.reference_pin.source_revision,
                "taxonomy_id": spec.reference_pin.taxonomy_id,
                "taxonomy_revision": spec.reference_pin.taxonomy_revision,
                "support_semantics": spec.reference_pin.support_semantics.value,
                "support_unit": spec.reference_pin.support_unit,
                "threshold": spec.reference_pin.threshold,
                "support_union_policy": spec.reference_pin.support_union_policy,
                "reference_artifact_manifest": _artifact_manifest_dict(
                    spec.reference_pin.reference_artifact_manifest
                ),
                "area_measurement": _area_measurement_dict(spec.reference_pin.area_measurement),
            },
            "total_source_keys": total_keys,
            "source_inventory_receipt": {
                "processed_key_count": total_keys,
                "observed_keyset_sha256": artifacts.inventory_keyset_sha256,
                "matches_source_evidence": True,
            },
            "levels": [
                {
                    "name": level.name,
                    "supported": level.supported,
                    "unsupported_reason": level.unsupported_reason,
                }
                for level in levels
            ],
            "level_accounting": [
                {
                    "level": item.level,
                    "supported": item.supported,
                    "unsupported_reason": item.unsupported_reason,
                    "total_source_keys": item.total_source_keys,
                    "processed_decisions": item.processed_decisions,
                    "qualifying_candidates": item.qualifying_candidates,
                    "assigned": item.assigned,
                    "ambiguous": item.ambiguous,
                    "unassigned": item.unassigned,
                    "excluded": item.excluded,
                    "exclusion_reason_group_count": item.exclusion_reason_group_count,
                }
                for item in accounting
            ],
            "storage_layout": {
                "shard_manifest": {
                    "path": artifacts.shard_manifest_path,
                    "sha256": artifacts.shard_manifest_sha256,
                    "shard_count": _read_shards_count(connection),
                },
                "exclusions": {
                    "path": artifacts.exclusion_reason_path,
                    "sha256": artifacts.exclusion_reason_sha256,
                    "group_count": artifacts.exclusion_reason_group_count,
                },
                "tables": counts,
            },
            "pool_complete": True,
        }
    )


def _inventory_keyset_sha256(connection: sqlite3.Connection) -> str:
    digest = hashlib.sha256()
    cursor = connection.execute("SELECT native_key_json FROM inventory ORDER BY native_key_json")
    first = True
    for (key_json,) in cursor:
        if not first:
            digest.update(b"\n")
        digest.update(key_json.encode("utf-8"))
        first = False
    return digest.hexdigest()


def _read_shards_count(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT COUNT(*) FROM shards").fetchone()[0]


def _entity_ref_key(snapshot_id: str, key: tuple[tuple[str, str | int], ...]) -> str:
    return json.dumps(
        (snapshot_id, native_key_json(key)),
        ensure_ascii=False,
        separators=(",", ":"),
    )


def _parse_native_key(value: object) -> tuple[tuple[str, str | int], ...] | None:
    if not isinstance(value, str):
        return None
    try:
        raw = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(raw, list) or any(
        not isinstance(pair, list) or len(pair) != _KEY_COMPONENT_WIDTH for pair in raw
    ):
        return None
    key = tuple((pair[0], pair[1]) for pair in raw)
    if not _valid_native_key(key) or native_key_json(key) != value:
        return None
    return key


def _parse_entity_ref(value: object) -> tuple[str, tuple[tuple[str, str | int], ...]] | None:
    raw = None
    if isinstance(value, str):
        with suppress(json.JSONDecodeError):
            raw = json.loads(value)
    if isinstance(raw, list) and len(raw) == _KEY_COMPONENT_WIDTH:
        snapshot_id, key_json = raw
        if isinstance(snapshot_id, str) and isinstance(key_json, str) and _present(snapshot_id):
            key = _parse_native_key(key_json)
            if key is not None and _entity_ref_key(snapshot_id, key) == value:
                return snapshot_id, key
    return None


def _present(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value: object) -> TypeGuard[int | float]:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _valid_native_key(key: tuple[tuple[str, str | int], ...]) -> bool:
    if not key:
        return False
    names = [name for name, _ in key]
    return (
        all(isinstance(name, str) and name.strip() for name in names)
        and names == sorted(names)
        and len(names) == len(set(names))
        and all(isinstance(value, (str, int)) and not isinstance(value, bool) for _, value in key)
    )


def _finite_numbers(values: tuple[object, ...]) -> tuple[float, ...] | None:
    if not all(_finite_number(value) for value in values):
        return None
    numeric_values = cast(tuple[int | float, ...], values)
    numbers = tuple(float(value) for value in numeric_values)
    return numbers if all(math.isfinite(value) for value in numbers) else None


def _valid_resolvable_uri(value: object) -> bool:
    return isinstance(value, str) and _valid_uri(value)


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in "0123456789abcdef" for char in value)
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
