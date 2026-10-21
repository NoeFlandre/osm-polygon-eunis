"""Deterministic Parquet export through a small immutable artifact-sink API."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import pyarrow as pa
import pyarrow.parquet as pq

from .builder import native_key_json
from .models import PoolBundle, PoolValidationReport
from .validation import validate_full_pool

_CONTRACT_VERSION = "1.0.0"
_TABLE_FIELDS: dict[str, tuple[pa.Field, ...]] = {
    "polygon_inventory": (
        pa.field("inventory_row_id", pa.string(), nullable=False),
        pa.field("native_key_json", pa.string(), nullable=False),
        pa.field("source_row_locator", pa.string(), nullable=False),
        pa.field("polygon_snapshot_id", pa.string()),
        pa.field("exclusion_reason", pa.string()),
    ),
    "polygon_snapshots": (
        pa.field("polygon_snapshot_id", pa.string(), nullable=False),
        pa.field("inventory_row_id", pa.string(), nullable=False),
        pa.field("native_key_json", pa.string(), nullable=False),
        pa.field("geometry_uri", pa.string(), nullable=False),
        pa.field("geometry_sha256", pa.string(), nullable=False),
        pa.field("geometry_format", pa.string(), nullable=False),
        pa.field("crs", pa.string(), nullable=False),
        pa.field("coordinate_transform", pa.string(), nullable=False),
        pa.field("geometry_status", pa.string(), nullable=False),
        pa.field("original_area", pa.float64()),
        pa.field("area_unit", pa.string(), nullable=False),
    ),
    "label_candidates": (
        pa.field("inventory_row_id", pa.string(), nullable=False),
        pa.field("polygon_snapshot_id", pa.string(), nullable=False),
        pa.field("label_release", pa.string(), nullable=False),
        pa.field("level", pa.string(), nullable=False),
        pa.field("code", pa.string(), nullable=False),
        pa.field("numerator", pa.float64(), nullable=False),
        pa.field("denominator", pa.float64(), nullable=False),
        pa.field("ratio", pa.float64(), nullable=False),
        pa.field("threshold", pa.float64(), nullable=False),
        pa.field("assignment_rule_version", pa.string(), nullable=False),
        pa.field("assignment_kind", pa.string(), nullable=False),
        pa.field("support_semantics", pa.string(), nullable=False),
        pa.field("support_unit", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("taxonomy_revision", pa.string(), nullable=False),
    ),
    "label_outcomes": (
        pa.field("inventory_row_id", pa.string(), nullable=False),
        pa.field("polygon_snapshot_id", pa.string()),
        pa.field("level", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("assigned_code", pa.string()),
        pa.field("reason", pa.string()),
        pa.field("ambiguity_basis", pa.string()),
    ),
    "source_documents": (
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("native_key_json", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("content_sha256", pa.string()),
        pa.field("body_status", pa.string(), nullable=False),
    ),
    "text_polygon_links": (
        pa.field("document_id", pa.string(), nullable=False),
        pa.field("inventory_row_id", pa.string()),
        pa.field("native_link_key_json", pa.string(), nullable=False),
        pa.field("source_revision", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("evidence", pa.string(), nullable=False),
        pa.field("candidate_inventory_row_ids_json", pa.string(), nullable=False),
    ),
    "entity_links": (
        pa.field("link_version", pa.string(), nullable=False),
        pa.field("left_source_snapshot_id", pa.string(), nullable=False),
        pa.field("left_native_key_json", pa.string(), nullable=False),
        pa.field("right_source_snapshot_id", pa.string(), nullable=False),
        pa.field("right_native_key_json", pa.string(), nullable=False),
        pa.field("evidence", pa.string(), nullable=False),
        pa.field("evidence_sha256", pa.string()),
    ),
    "pool_eligibility": (
        pa.field("inventory_row_id", pa.string(), nullable=False),
        pa.field("cohort_policy_version", pa.string(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("evidence", pa.string(), nullable=False),
        pa.field("reason", pa.string()),
        pa.field("replaced_by_inventory_row_id", pa.string()),
    ),
}


class PoolArtifactSink(Protocol):
    """Write immutable relative artifacts and return their resolvable URI."""

    def put(self, relative_path: str, content: bytes) -> str:
        """Persist bytes at a stable path, rejecting conflicting rewrites."""


@dataclass(frozen=True, slots=True)
class FilesystemSink:
    """Local implementation useful for tests, previews, and controlled staging."""

    root: Path

    def put(self, relative_path: str, content: bytes) -> str:
        """Write bytes once, allowing an identical retry of immutable content."""

        target = self.root / relative_path
        target.resolve().relative_to(self.root.resolve())
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if target.read_bytes() != content:
                raise FileExistsError(f"immutable pool artifact differs at {relative_path}")
            return target.as_uri()
        temporary = target.with_name(f".{target.name}.tmp")
        temporary.write_bytes(content)
        temporary.replace(target)
        return target.as_uri()


@dataclass(frozen=True, slots=True)
class ExportReceipt:
    """Content hashes and locations for a validated full-pool artifact."""

    pool_id: str
    manifest_uri: str
    manifest_sha256: str
    table_uris: tuple[tuple[str, str], ...]
    table_sha256: tuple[tuple[str, str], ...]


class InvalidPoolError(ValueError):
    """Raised when export is attempted before full-pool validation passes."""

    def __init__(self, report: PoolValidationReport) -> None:
        """Attach all validation issues to the failed export request."""

        self.report = report
        super().__init__("; ".join(report.issues))


def export_full_pool(bundle: PoolBundle, sink: PoolArtifactSink) -> ExportReceipt:
    """Validate, encode, hash, then export every versioned logical pool table."""

    report = validate_full_pool(bundle)
    if not report.ok:
        raise InvalidPoolError(report)
    table_paths = _table_paths(bundle)
    tables = _tables(bundle)
    encoded = {name: _parquet_bytes(table) for name, table in tables.items()}
    table_hashes = {name: _sha256(payload) for name, payload in encoded.items()}
    uris = {name: sink.put(table_paths[name], encoded[name]) for name in sorted(encoded)}
    manifest = _manifest_dict(bundle, table_paths, table_hashes)
    manifest_bytes = _canonical_json_bytes(manifest)
    manifest_path = f"{_pool_root(bundle)}/manifest.json"
    manifest_uri = sink.put(manifest_path, manifest_bytes)
    return ExportReceipt(
        pool_id=bundle.manifest.pool_id,
        manifest_uri=manifest_uri,
        manifest_sha256=_sha256(manifest_bytes),
        table_uris=tuple(sorted(uris.items())),
        table_sha256=tuple(sorted(table_hashes.items())),
    )


def table_schemas() -> dict[str, pa.Schema]:
    """Return fresh explicit Arrow schemas for all pool tables."""

    return {
        name: pa.schema(
            fields,
            metadata={b"contract_version": _CONTRACT_VERSION.encode(), b"table": name.encode()},
        )
        for name, fields in _TABLE_FIELDS.items()
    }


def _tables(bundle: PoolBundle) -> dict[str, pa.Table]:
    schemas = table_schemas()
    rows = _table_rows(bundle)
    return {name: pa.Table.from_pylist(rows[name], schema=schemas[name]) for name in schemas}


def _table_rows(bundle: PoolBundle) -> dict[str, list[dict[str, object]]]:
    manifest = bundle.manifest
    return {
        "polygon_inventory": [
            {
                "inventory_row_id": row.inventory_row_id,
                "native_key_json": native_key_json(row.native_key),
                "source_row_locator": row.source_row_locator,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "exclusion_reason": row.exclusion_reason,
            }
            for row in bundle.inventory
        ],
        "polygon_snapshots": [
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
            for row in bundle.polygons
        ],
        "label_candidates": [
            {
                "inventory_row_id": row.inventory_row_id,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "label_release": manifest.reference_pin.release_id,
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
            for row in bundle.label_candidates
        ],
        "label_outcomes": [
            {
                "inventory_row_id": row.inventory_row_id,
                "polygon_snapshot_id": row.polygon_snapshot_id,
                "level": row.level,
                "status": row.status.value,
                "assigned_code": row.assigned_code,
                "reason": row.reason,
                "ambiguity_basis": row.ambiguity_basis.value if row.ambiguity_basis else None,
            }
            for row in bundle.label_outcomes
        ],
        "source_documents": [
            {
                "document_id": row.document_id,
                "native_key_json": native_key_json(row.native_key),
                "source_revision": row.source_revision,
                "content_sha256": row.content_sha256,
                "body_status": row.body_status.value,
            }
            for row in bundle.documents
        ],
        "text_polygon_links": [
            {
                "document_id": row.document_id,
                "inventory_row_id": row.inventory_row_id,
                "native_link_key_json": native_key_json(row.native_link_key),
                "source_revision": row.source_revision,
                "status": row.status.value,
                "evidence": row.evidence,
                "candidate_inventory_row_ids_json": json.dumps(
                    sorted(row.candidate_inventory_row_ids), separators=(",", ":")
                ),
            }
            for row in bundle.text_polygon_links
        ],
        "entity_links": [
            {
                "link_version": row.link_version,
                "left_source_snapshot_id": row.left.source_snapshot_id,
                "left_native_key_json": native_key_json(row.left.native_key),
                "right_source_snapshot_id": row.right.source_snapshot_id,
                "right_native_key_json": native_key_json(row.right.native_key),
                "evidence": row.evidence,
                "evidence_sha256": row.evidence_sha256,
            }
            for row in bundle.entity_links
        ],
        "pool_eligibility": [
            {
                "inventory_row_id": row.inventory_row_id,
                "cohort_policy_version": row.cohort_policy_version,
                "status": row.status.value,
                "evidence": row.evidence,
                "reason": row.reason,
                "replaced_by_inventory_row_id": row.replaced_by_inventory_row_id,
            }
            for row in bundle.eligibility
        ],
    }


def _manifest_dict(
    bundle: PoolBundle,
    table_paths: dict[str, str],
    table_hashes: dict[str, str],
) -> dict[str, object]:
    manifest = bundle.manifest
    return {
        "contract": "noeflandre.polygon-label-pool",
        "contract_version": manifest.contract_version,
        "pool_version": manifest.pool_version,
        "pool_id": manifest.pool_id,
        "implementation_pins": {
            "adapter_id": manifest.adapter_id,
            "adapter_revision": manifest.adapter_revision,
            "builder_revision": manifest.builder_revision,
        },
        "pool_key": {
            "text_source": manifest.pool_key.text_source.value,
            "reference_system": manifest.pool_key.reference_system.value,
        },
        "source_snapshot": {
            "snapshot_id": manifest.source_snapshot.snapshot_id,
            "repository": manifest.source_snapshot.repository,
            "revision": manifest.source_snapshot.revision,
            "pbf_uri": manifest.source_snapshot.pbf_uri,
            "pbf_sha256": manifest.source_snapshot.pbf_sha256,
        },
        "reference_pin": {
            "release_id": manifest.reference_pin.release_id,
            "source_repository": manifest.reference_pin.source_repository,
            "source_revision": manifest.reference_pin.source_revision,
            "taxonomy_id": manifest.reference_pin.taxonomy_id,
            "taxonomy_revision": manifest.reference_pin.taxonomy_revision,
            "support_semantics": manifest.reference_pin.support_semantics.value,
            "support_unit": manifest.reference_pin.support_unit,
            "threshold": manifest.reference_pin.threshold,
            "support_union_policy": manifest.reference_pin.support_union_policy,
        },
        "total_source_keys": manifest.total_source_keys,
        "levels": [
            {
                "name": level.name,
                "supported": level.supported,
                "unsupported_reason": level.unsupported_reason,
            }
            for level in bundle.levels
        ],
        "level_accounting": [
            {
                "level": count.level,
                "supported": count.supported,
                "unsupported_reason": count.unsupported_reason,
                "total_source_keys": count.total_source_keys,
                "processed_decisions": count.processed_decisions,
                "qualifying_candidates": count.qualifying_candidates,
                "assigned": count.assigned,
                "ambiguous": count.ambiguous,
                "unassigned": count.unassigned,
                "excluded": count.excluded,
                "exclusions_by_reason": dict(count.exclusions_by_reason),
            }
            for count in manifest.level_accounting
        ],
        "pool_complete": True,
        "files": {
            name: {"path": table_paths[name], "sha256": table_hashes[name]}
            for name in sorted(table_paths)
        },
    }


def _parquet_bytes(table: pa.Table) -> bytes:
    stream = pa.BufferOutputStream()
    pq.write_table(
        table,
        stream,
        compression="zstd",
        compression_level=9,
        use_dictionary=False,
        write_statistics=True,
        version="2.6",
        data_page_version="1.0",
        row_group_size=65_536,
    )
    return stream.getvalue().to_pybytes()


def _pool_root(bundle: PoolBundle) -> str:
    manifest = bundle.manifest
    return (
        f"pools/{manifest.contract_version}/{manifest.pool_key.text_source.value}/"
        f"{manifest.pool_key.reference_system.value}/{manifest.source_snapshot.snapshot_id}/"
        f"{manifest.pool_version}"
    )


def _table_paths(bundle: PoolBundle) -> dict[str, str]:
    root = _pool_root(bundle)
    return {name: f"{root}/{name}.parquet" for name in _TABLE_FIELDS}


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
