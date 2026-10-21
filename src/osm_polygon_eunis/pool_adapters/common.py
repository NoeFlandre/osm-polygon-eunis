"""Shared decision handling for the source-specific pool adapters."""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass

from shapely.geometry.base import BaseGeometry

from osm_polygon_eunis.pool_contract import (
    AmbiguityBasis,
    AssignmentKind,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    OutcomeStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    RawInventoryEvidence,
    ReferencePin,
)
from osm_polygon_eunis.pool_contract.models import GeometryStatus

ASSIGNMENT_RULE_VERSION = "unique-class-at-or-above-0.8/1.0"
POOL_THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class SupportArea:
    """One within-class union area measured against a polygon's full area."""

    level: str
    code: str
    numerator: float
    assignment_kind: AssignmentKind
    evidence_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class SupportAttempt:
    """The reference measurement result for one usable source polygon."""

    supports: tuple[SupportArea, ...]
    exclusion_reason: str | None = None
    attempted_reference_ids: tuple[str, ...] = ()
    attempted_tile_ids: tuple[str, ...] = ()
    missing_tile_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AttemptEvidenceRecord:
    """Auditable source-query receipt emitted for every raw source polygon."""

    inventory_row_id: str
    source_polygon_id: str
    attempted_reference_ids: tuple[str, ...]
    attempted_tile_ids: tuple[str, ...]
    missing_tile_ids: tuple[str, ...]
    support_evidence: tuple[tuple[str, str, tuple[str, ...]], ...]
    status: str
    reason: str | None


@dataclass(frozen=True, slots=True)
class PoolPolygonInput:
    """One raw source polygon plus its immutable identity and geometry snapshot."""

    source_polygon_id: str
    inventory: PolygonInventoryRow
    snapshot: PolygonSnapshot | None
    geometry: BaseGeometry | None


@dataclass(frozen=True, slots=True)
class PoolPolygonLabelRecords:
    """All support candidates and supported-level decisions for one raw polygon."""

    inventory_row_id: str
    label_candidates: tuple[LabelCandidate, ...]
    label_outcomes: tuple[LabelOutcome, ...]
    attempt_evidence: AttemptEvidenceRecord


def build_raw_inventory_evidence(
    *,
    enumeration_rule: str,
    scan_report_uri: str,
    scan_report: bytes,
    expected_key_count: int,
    expected_keyset_sha256: str,
) -> RawInventoryEvidence:
    """Pin a pre-join scan report and independently computed raw-key totals."""

    if not enumeration_rule.strip():
        raise ValueError("raw inventory enumeration rule is required")
    if not scan_report_uri.strip():
        raise ValueError("raw inventory scan-report URI is required")
    if not scan_report:
        raise ValueError("raw inventory scan report must not be empty")
    if expected_key_count < 0:
        raise ValueError("raw inventory expected key count cannot be negative")
    if not re.fullmatch(r"[0-9a-f]{64}", expected_keyset_sha256):
        raise ValueError("raw inventory key-set digest must be lowercase SHA-256")
    return RawInventoryEvidence(
        enumeration_rule=enumeration_rule,
        scan_report_uri=scan_report_uri,
        scan_report_sha256=hashlib.sha256(scan_report).hexdigest(),
        expected_key_count=expected_key_count,
        expected_keyset_sha256=expected_keyset_sha256,
    )


def label_polygon(
    polygon: PoolPolygonInput,
    attempt: SupportAttempt,
    levels: Sequence[LabelLevel],
    reference_pin: ReferencePin,
    *,
    assignment_rule_version: str = ASSIGNMENT_RULE_VERSION,
) -> PoolPolygonLabelRecords:
    """Create candidates and one decision for each supported level."""

    supported_levels = _validate_label_request(
        polygon, levels, reference_pin, assignment_rule_version
    )
    snapshot = polygon.snapshot
    exclusion_reason = _unusable_reason(polygon, snapshot) or attempt.exclusion_reason
    if exclusion_reason is not None:
        return _excluded_records(polygon, attempt, supported_levels, exclusion_reason)

    if snapshot is None or snapshot.original_area is None:
        raise ValueError("usable pool polygon needs a geometry snapshot and full area")
    if snapshot.area_unit != reference_pin.support_unit:
        raise ValueError("reference and polygon area units do not match")

    candidates_by_level = _candidates_by_level(
        polygon,
        attempt,
        supported_levels,
        reference_pin,
        assignment_rule_version,
    )
    candidates = tuple(
        candidate
        for level_candidates in candidates_by_level.values()
        for candidate in level_candidates
    )
    outcomes = tuple(
        _outcome(polygon.inventory, level.name, candidates_by_level[level.name])
        for level in supported_levels
    )
    evidence = _attempt_evidence(polygon, attempt, candidates, "completed", None)
    return PoolPolygonLabelRecords(
        polygon.inventory.inventory_row_id, candidates, outcomes, evidence
    )


def _validate_label_request(
    polygon: PoolPolygonInput,
    levels: Sequence[LabelLevel],
    reference_pin: ReferencePin,
    assignment_rule_version: str,
) -> tuple[LabelLevel, ...]:
    """Validate shared threshold, source snapshot, and supported-level pins."""

    if reference_pin.threshold != POOL_THRESHOLD:
        raise ValueError("polygon-label pools require the pinned 0.8 threshold")
    if not assignment_rule_version.strip():
        raise ValueError("assignment-rule version is required")
    if polygon.inventory.polygon_snapshot_id != (
        polygon.snapshot.polygon_snapshot_id if polygon.snapshot is not None else None
    ):
        raise ValueError("pool polygon snapshot does not match its raw inventory row")

    supported_levels = tuple(level for level in levels if level.supported)
    if not supported_levels:
        raise ValueError("at least one supported level is required")
    level_names = {level.name for level in supported_levels}
    if len(level_names) != len(supported_levels):
        raise ValueError("supported level names must be unique")

    return supported_levels


def _excluded_records(
    polygon: PoolPolygonInput,
    attempt: SupportAttempt,
    levels: Sequence[LabelLevel],
    reason: str,
) -> PoolPolygonLabelRecords:
    """Record the same explicit source failure at every supported level."""

    evidence = _attempt_evidence(polygon, attempt, (), "excluded", reason)
    outcomes = tuple(
        LabelOutcome(
            inventory_row_id=polygon.inventory.inventory_row_id,
            polygon_snapshot_id=polygon.inventory.polygon_snapshot_id,
            level=level.name,
            status=OutcomeStatus.EXCLUDED,
            reason=reason,
        )
        for level in levels
    )
    return PoolPolygonLabelRecords(polygon.inventory.inventory_row_id, (), outcomes, evidence)


def _candidates_by_level(
    polygon: PoolPolygonInput,
    attempt: SupportAttempt,
    levels: Sequence[LabelLevel],
    reference_pin: ReferencePin,
    assignment_rule_version: str,
) -> dict[str, list[LabelCandidate]]:
    """Validate unioned class areas and build per-level candidate records."""

    snapshot = polygon.snapshot
    if snapshot is None or snapshot.original_area is None:
        raise ValueError("usable pool polygon needs a geometry snapshot and full area")
    level_names = {level.name for level in levels}
    by_level_code: dict[tuple[str, str], SupportArea] = {}
    for support in attempt.supports:
        if support.level not in level_names:
            raise ValueError(f"support uses undeclared or unsupported level {support.level!r}")
        if not support.code.strip():
            raise ValueError("support class code must not be empty")
        key = support.level, support.code
        if key in by_level_code:
            raise ValueError(f"support must be unioned before labeling: duplicate {key}")
        _validate_area(support.numerator, snapshot.original_area, key)
        if support.numerator > 0.0:
            by_level_code[key] = support

    candidates_by_level: dict[str, list[LabelCandidate]] = {name: [] for name in level_names}
    for _key, support in sorted(by_level_code.items()):
        candidate = _candidate(
            polygon.inventory,
            snapshot,
            support,
            reference_pin,
            assignment_rule_version,
        )
        candidates_by_level[candidate.level].append(candidate)
    return candidates_by_level


def _attempt_evidence(
    polygon: PoolPolygonInput,
    attempt: SupportAttempt,
    candidates: Sequence[LabelCandidate],
    status: str,
    reason: str | None,
) -> AttemptEvidenceRecord:
    support_evidence = tuple(
        sorted(
            (
                candidate.level,
                candidate.code,
                support.evidence_ids,
            )
            for support in attempt.supports
            for candidate in candidates
            if (candidate.level, candidate.code) == (support.level, support.code)
        )
    )
    return AttemptEvidenceRecord(
        inventory_row_id=polygon.inventory.inventory_row_id,
        source_polygon_id=polygon.source_polygon_id,
        attempted_reference_ids=attempt.attempted_reference_ids,
        attempted_tile_ids=attempt.attempted_tile_ids,
        missing_tile_ids=attempt.missing_tile_ids,
        support_evidence=support_evidence,
        status=status,
        reason=reason,
    )


def _unusable_reason(
    polygon: PoolPolygonInput,
    snapshot: PolygonSnapshot | None,
) -> str | None:
    if snapshot is None:
        return polygon.inventory.exclusion_reason or "geometry_snapshot_missing"
    if snapshot.geometry_status is not GeometryStatus.USABLE:
        return polygon.inventory.exclusion_reason or snapshot.geometry_status.value
    if polygon.geometry is None or polygon.geometry.is_empty or not polygon.geometry.is_valid:
        return "geometry_unusable"
    if (
        snapshot.original_area is None
        or not math.isfinite(snapshot.original_area)
        or snapshot.original_area <= 0
    ):
        return polygon.inventory.exclusion_reason or "geometry_area_unusable"
    return None


def _validate_area(numerator: float, denominator: float, key: tuple[str, str]) -> None:
    if not math.isfinite(numerator) or numerator < 0.0:
        raise ValueError(f"support area for {key} must be finite and non-negative")
    if numerator > denominator:
        raise ValueError(f"support area for {key} exceeds full polygon area")


def _candidate(
    inventory: PolygonInventoryRow,
    snapshot: PolygonSnapshot,
    support: SupportArea,
    reference_pin: ReferencePin,
    assignment_rule_version: str,
) -> LabelCandidate:
    denominator = snapshot.original_area
    if denominator is None:
        raise ValueError("usable pool polygon needs a full area")
    return LabelCandidate(
        inventory_row_id=inventory.inventory_row_id,
        polygon_snapshot_id=snapshot.polygon_snapshot_id,
        level=support.level,
        code=support.code,
        numerator=support.numerator,
        denominator=denominator,
        ratio=support.numerator / denominator,
        threshold=POOL_THRESHOLD,
        assignment_rule_version=assignment_rule_version,
        assignment_kind=support.assignment_kind,
        support_semantics=reference_pin.support_semantics,
        support_unit=reference_pin.support_unit,
        source_revision=reference_pin.source_revision,
        taxonomy_revision=reference_pin.taxonomy_revision,
    )


def _outcome(
    inventory: PolygonInventoryRow,
    level: str,
    candidates: Sequence[LabelCandidate],
) -> LabelOutcome:
    qualifiers = [candidate for candidate in candidates if candidate.ratio >= POOL_THRESHOLD]
    if len(qualifiers) == 1:
        return LabelOutcome(
            inventory_row_id=inventory.inventory_row_id,
            polygon_snapshot_id=inventory.polygon_snapshot_id,
            level=level,
            status=OutcomeStatus.ASSIGNED,
            assigned_code=qualifiers[0].code,
        )
    if len(qualifiers) > 1:
        return LabelOutcome(
            inventory_row_id=inventory.inventory_row_id,
            polygon_snapshot_id=inventory.polygon_snapshot_id,
            level=level,
            status=OutcomeStatus.AMBIGUOUS,
            reason="multiple_classes_meet_threshold",
            ambiguity_basis=AmbiguityBasis.MULTIPLE_QUALIFIERS,
        )
    reason = "no_reference_coverage" if not candidates else "no_class_meets_threshold"
    return LabelOutcome(
        inventory_row_id=inventory.inventory_row_id,
        polygon_snapshot_id=inventory.polygon_snapshot_id,
        level=level,
        status=OutcomeStatus.UNASSIGNED,
        reason=reason,
    )
