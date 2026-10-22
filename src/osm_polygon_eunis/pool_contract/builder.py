"""Pool assembly helpers; source adapters supply inventories and measurements."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable, Mapping

from .models import (
    EntityLink,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    LabelSupportEvidence,
    LevelAccounting,
    NativeKey,
    PolygonAttemptReceipt,
    PolygonInventoryRow,
    PoolBuildInput,
    PoolBundle,
    PoolEligibility,
    PoolManifest,
    RelevanceAssessment,
    TextPolygonLink,
)


def canonical_native_key(values: Mapping[str, str | int]) -> NativeKey:
    """Return an immutable, deterministically ordered native composite key."""

    if not values:
        raise ValueError("a source-native composite key must contain at least one field")
    for name, value in values.items():
        if not name.strip():
            raise ValueError("source-native key field names must not be empty")
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise TypeError("source-native key values must be strings or integers")
    return tuple(sorted(values.items()))


def native_key_json(key: NativeKey) -> str:
    """Serialize a typed native key without erasing integer/string distinctions."""

    return json.dumps(key, ensure_ascii=False, separators=(",", ":"))


def inventory_row_identifier(source_snapshot_id: str, key: NativeKey) -> str:
    """Create a stable row identifier from the frozen source pin and full key."""

    payload = f"{source_snapshot_id}\n{native_key_json(key)}".encode()
    return hashlib.sha256(payload).hexdigest()


def polygon_snapshot_identifier(
    source_snapshot_id: str,
    key: NativeKey,
    geometry_sha256: str,
) -> str:
    """Create a stable geometry-snapshot identifier from key and artifact hash."""

    payload = f"{source_snapshot_id}\n{native_key_json(key)}\n{geometry_sha256}".encode()
    return hashlib.sha256(payload).hexdigest()


def native_keyset_sha256(keys: Iterable[NativeKey]) -> str:
    """Hash the sorted canonical native keys from a complete source scan."""

    payload = "\n".join(sorted(native_key_json(key) for key in keys)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_full_pool(
    source: PoolBuildInput,
    *,
    contract_version: str = "1.2.0",
    pool_version: str = "1.0.0",
) -> PoolBundle:
    """Assemble and canonically order pool tables without filtering inventory.

    Source adapters own source parsing and reference-support measurement. This
    function records their complete raw inventory, candidates, terminal
    per-level decisions, document links, and optional downstream eligibility.
    It deliberately does not derive labels or remove rows lacking documents.
    """

    inventory_rows = tuple(sorted(source.inventory, key=lambda row: row.inventory_row_id))
    polygon_rows = tuple(sorted(source.polygons, key=lambda row: row.polygon_snapshot_id))
    level_rows = tuple(sorted(source.levels, key=lambda level: level.name))
    candidate_rows = tuple(sorted(source.label_candidates, key=_candidate_sort_key))
    outcome_rows = tuple(sorted(source.label_outcomes, key=_outcome_sort_key))
    document_rows = tuple(sorted(source.documents, key=lambda row: row.document_id))
    link_rows = tuple(sorted(source.text_polygon_links, key=_text_link_sort_key))
    entity_rows = tuple(sorted(source.entity_links, key=_entity_link_sort_key))
    eligibility_rows = tuple(sorted(source.eligibility, key=_eligibility_sort_key))
    attempt_rows = tuple(sorted(source.attempt_receipts, key=_attempt_sort_key))
    support_evidence_rows = tuple(sorted(source.support_evidence, key=_support_evidence_sort_key))
    relevance_rows = tuple(sorted(source.relevance_assessments, key=_relevance_sort_key))
    accounting = _build_level_accounting(inventory_rows, level_rows, candidate_rows, outcome_rows)
    manifest = PoolManifest(
        contract_version=contract_version,
        pool_version=pool_version,
        pool_key=source.pool_key,
        source_snapshot=source.source_snapshot,
        raw_inventory_evidence=source.raw_inventory_evidence,
        reference_pin=source.reference_pin,
        adapter_id=source.adapter_id,
        adapter_revision=source.adapter_revision,
        builder_revision=source.builder_revision,
        total_source_keys=len(inventory_rows),
        level_accounting=accounting,
    )
    return PoolBundle(
        manifest=manifest,
        levels=level_rows,
        inventory=inventory_rows,
        polygons=polygon_rows,
        label_candidates=candidate_rows,
        label_outcomes=outcome_rows,
        documents=document_rows,
        text_polygon_links=link_rows,
        entity_links=entity_rows,
        eligibility=eligibility_rows,
        attempt_receipts=attempt_rows,
        support_evidence=support_evidence_rows,
        relevance_assessments=relevance_rows,
    )


def _build_level_accounting(
    inventory: tuple[PolygonInventoryRow, ...],
    levels: tuple[LabelLevel, ...],
    candidates: tuple[LabelCandidate, ...],
    outcomes: tuple[LabelOutcome, ...],
) -> tuple[LevelAccounting, ...]:
    accounting: list[LevelAccounting] = []
    for level in levels:
        level_candidates = [candidate for candidate in candidates if candidate.level == level.name]
        level_outcomes = [outcome for outcome in outcomes if outcome.level == level.name]
        excluded_reasons = Counter(
            outcome.reason
            for outcome in level_outcomes
            if outcome.status.value == "excluded" and outcome.reason is not None
        )
        accounting.append(
            LevelAccounting(
                level=level.name,
                supported=level.supported,
                unsupported_reason=level.unsupported_reason,
                total_source_keys=len(inventory),
                processed_decisions=len(level_outcomes),
                qualifying_candidates=sum(
                    candidate.ratio >= candidate.threshold for candidate in level_candidates
                ),
                assigned=sum(outcome.status.value == "assigned" for outcome in level_outcomes),
                ambiguous=sum(outcome.status.value == "ambiguous" for outcome in level_outcomes),
                unassigned=sum(outcome.status.value == "unassigned" for outcome in level_outcomes),
                excluded=sum(outcome.status.value == "excluded" for outcome in level_outcomes),
                exclusions_by_reason=tuple(sorted(excluded_reasons.items())),
            )
        )
    return tuple(accounting)


def _candidate_sort_key(candidate: LabelCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.inventory_row_id,
        candidate.level,
        candidate.code,
        candidate.assignment_kind.value,
    )


def _outcome_sort_key(outcome: LabelOutcome) -> tuple[str, str]:
    return outcome.inventory_row_id, outcome.level


def _text_link_sort_key(link: TextPolygonLink) -> tuple[str, str, str]:
    return link.inventory_row_id or "", link.document_id, native_key_json(link.native_link_key)


def _entity_link_sort_key(link: EntityLink) -> tuple[str, str, str]:
    left_key = native_key_json(link.left.native_key)
    right_key = native_key_json(link.right.native_key)
    return link.link_version, left_key, right_key


def _eligibility_sort_key(
    eligibility: PoolEligibility,
) -> tuple[str, str, str]:
    return (
        eligibility.inventory_row_id,
        eligibility.cohort_policy_version,
        eligibility.status.value,
    )


def _attempt_sort_key(
    receipt: PolygonAttemptReceipt,
) -> tuple[str]:
    return (receipt.inventory_row_id,)


def _support_evidence_sort_key(
    evidence: LabelSupportEvidence,
) -> tuple[str, str, str]:
    return evidence.inventory_row_id, evidence.level, evidence.code


def _relevance_sort_key(
    assessment: RelevanceAssessment,
) -> tuple[str, str]:
    return assessment.document_id, assessment.release_id
