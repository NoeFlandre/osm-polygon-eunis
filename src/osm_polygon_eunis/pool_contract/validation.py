"""Completeness, provenance, and ambiguity checks for polygon-label pools."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable
from urllib.parse import urlparse

from .builder import (
    build_full_pool,
    inventory_row_identifier,
    native_key_json,
    polygon_snapshot_identifier,
)
from .models import (
    AmbiguityBasis,
    BodyStatus,
    EligibilityStatus,
    EntityLink,
    EvaluationCase,
    EvaluationPayload,
    EvaluationTrace,
    GateReport,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    OutcomeStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBuildInput,
    PoolBundle,
    PoolEligibility,
    PoolKey,
    PoolValidationReport,
    ReferencePin,
    ReferenceSystem,
    SourceDocument,
    SourceSnapshot,
    SplitAssignment,
    SupportSemantics,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTRACT_THRESHOLD = 0.8
_SUPPORT_UNION_POLICY = "union_by_class"
_MIN_AMBIGUOUS_COUNT = 2
_SUPPORT_SEMANTICS = {
    ReferenceSystem.WORLDCOVER: SupportSemantics.MAPPED_COVER,
    ReferenceSystem.LUCAS: SupportSemantics.SURVEYED_FOOTPRINT,
    ReferenceSystem.EUNIS: SupportSemantics.MODELED_HABITAT_SUPPORT,
}


def validate_full_pool(bundle: PoolBundle) -> PoolValidationReport:
    """Check every source key, per-level decision, link, and manifest count."""

    issues: list[str] = []
    issues.extend(_validate_pins(bundle))
    issues.extend(_validate_levels(bundle.levels))
    inventory_by_id, polygon_by_id = _validate_inventory(bundle, issues)
    candidates_by_key = _validate_candidates(bundle, inventory_by_id, polygon_by_id, issues)
    _validate_outcomes(bundle, inventory_by_id, polygon_by_id, candidates_by_key, issues)
    _validate_documents(bundle, inventory_by_id, issues)
    _validate_entity_links(bundle.entity_links, issues)
    _validate_eligibility(bundle, inventory_by_id, issues)
    _validate_accounting(bundle, issues)
    return PoolValidationReport(tuple(issues))


def validate_nine_pool_gate(bundles: Iterable[PoolBundle]) -> GateReport:
    """Require one complete pool for every text-source by reference-system pair."""

    expected = {
        PoolKey(text_source, reference_system)
        for text_source in TextSource
        for reference_system in ReferenceSystem
    }
    grouped: dict[PoolKey, list[PoolBundle]] = defaultdict(list)
    for bundle in bundles:
        grouped[bundle.manifest.pool_key].append(bundle)
    missing = tuple(sorted(key.name for key in expected - grouped.keys()))
    duplicates = tuple(sorted(key.name for key, values in grouped.items() if len(values) > 1))
    invalid = []
    versions: set[tuple[str, str]] = set()
    for key in sorted(expected & grouped.keys(), key=lambda item: item.name):
        for bundle in grouped[key]:
            report = validate_full_pool(bundle)
            versions.add((bundle.manifest.contract_version, bundle.manifest.pool_version))
            if not report.ok:
                invalid.append((key.name, report.issues))
    if len(versions) > 1:
        invalid.append(
            ("nine-pool-version-set", ("pool versions must match across all nine pools",))
        )
    return GateReport(missing, duplicates, tuple(invalid))


def validate_evaluation_trace(trace: EvaluationTrace) -> PoolValidationReport:
    """Validate exact payload hashes and settings for one separate eval trace."""

    issues: list[str] = []
    if not trace.trace_id.strip() or not trace.case_id.strip() or not trace.model_id.strip():
        issues.append("evaluation trace identifiers and model id must not be empty")
    if not _canonical_settings(trace.settings_json):
        issues.append("evaluation settings_json must be a canonical JSON object")
    _validate_payload(trace.prompt, "prompt", issues)
    _validate_payload(trace.raw_response, "raw_response", issues)
    if not _valid_uri(trace.reference_geometry_uri):
        issues.append("evaluation reference geometry URI must not be empty")
    if not _valid_sha256(trace.reference_geometry_sha256):
        issues.append("evaluation reference geometry SHA-256 is invalid")
    return PoolValidationReport(tuple(issues))


def validate_evaluation_case(case: EvaluationCase) -> PoolValidationReport:
    """Check case identity and sampling lineage independently of dataset splits."""

    issues = []
    if any(
        not _present(value)
        for value in (
            case.case_id,
            case.pool_id,
            case.inventory_row_id,
            case.target_level,
            case.sampling_policy_version,
        )
    ):
        issues.append("evaluation case ids, target level, and sampling policy must be present")
    if case.selection_order < 0:
        issues.append("evaluation case selection order cannot be negative")
    if len(case.source_document_ids) != len(set(case.source_document_ids)):
        issues.append("evaluation case cannot repeat source document ids")
    return PoolValidationReport(tuple(issues))


def validate_split_assignment(assignment: SplitAssignment) -> PoolValidationReport:
    """Check split provenance without inferring or changing split membership."""

    issues = []
    if not _present(assignment.entity.source_snapshot_id):
        issues.append("split assignment needs a source snapshot pin")
    if not _valid_native_key(assignment.entity.native_key):
        issues.append("split assignment needs a full composite polygon key")
    if not _present(assignment.policy_version) or not _present(assignment.evidence):
        issues.append("split assignment needs policy version and membership evidence")
    return PoolValidationReport(tuple(issues))


def _validate_pins(bundle: PoolBundle) -> list[str]:
    issues = []
    manifest = bundle.manifest
    source = manifest.source_snapshot
    reference = manifest.reference_pin
    if not _present(manifest.contract_version) or not _present(manifest.pool_version):
        issues.append("manifest contract and pool versions must not be empty")
    if not _present(manifest.pool_key.name):
        issues.append("pool key must not be empty")
    _validate_source_pin(source, issues)
    required_reference_fields = (
        reference.release_id,
        reference.source_repository,
        reference.source_revision,
        reference.taxonomy_id,
        reference.taxonomy_revision,
        reference.support_semantics,
        reference.support_unit,
    )
    if any(not _present(value) for value in required_reference_fields):
        issues.append("reference release, source, taxonomy, and support pins must be present")
    code_revisions = (manifest.adapter_id, manifest.adapter_revision, manifest.builder_revision)
    if any(not _present(value) for value in code_revisions):
        issues.append("adapter and shared-builder code revisions must be pinned")
    _validate_support_contract(bundle, issues)
    return issues


def _validate_source_pin(source: SourceSnapshot, issues: list[str]) -> None:
    if not _present(source.snapshot_id) or not _present(source.repository):
        issues.append("source snapshot id and repository must not be empty")
    if not _present(source.revision) or not _valid_uri(source.pbf_uri):
        issues.append("source revision and PBF URI must not be empty")
    if not _valid_sha256(source.pbf_sha256):
        issues.append("source PBF SHA-256 is invalid")


def _validate_support_contract(bundle: PoolBundle, issues: list[str]) -> None:
    reference = bundle.manifest.reference_pin
    if reference.threshold != _CONTRACT_THRESHOLD:
        issues.append("reference threshold must be exactly 0.8")
    if reference.support_union_policy != _SUPPORT_UNION_POLICY:
        issues.append("reference support must declare within-class union_by_class")
    expected_semantics = _SUPPORT_SEMANTICS[bundle.manifest.pool_key.reference_system]
    if reference.support_semantics is not expected_semantics:
        issues.append(
            f"{bundle.manifest.pool_key.reference_system.value} support semantics must be "
            f"{expected_semantics.value}"
        )


def _validate_levels(levels: tuple[LabelLevel, ...]) -> list[str]:
    issues = []
    names = [level.name for level in levels]
    if not levels:
        issues.append("at least one taxonomy level must be declared")
    if len(names) != len(set(names)):
        issues.append("taxonomy level names must be unique")
    for level in levels:
        if not _present(level.name):
            issues.append("taxonomy level names must not be empty")
        if level.supported and level.unsupported_reason is not None:
            issues.append(f"supported level {level.name} cannot have an unsupported reason")
        if not level.supported and not _present(level.unsupported_reason):
            issues.append(f"unsupported level {level.name} requires an explicit reason")
    return issues


def _validate_inventory(
    bundle: PoolBundle,
    issues: list[str],
) -> tuple[dict[str, PolygonInventoryRow], dict[str, PolygonSnapshot]]:
    source_id = bundle.manifest.source_snapshot.snapshot_id
    inventory_by_id = _validate_inventory_rows(bundle, source_id, issues)
    if len(bundle.inventory) != bundle.manifest.total_source_keys:
        issues.append("manifest total_source_keys does not equal raw inventory row count")
    polygon_by_id = _validate_polygons(bundle.polygons, inventory_by_id, source_id, issues)
    _validate_inventory_geometry_links(bundle.inventory, bundle.polygons, polygon_by_id, issues)
    return inventory_by_id, polygon_by_id


def _validate_inventory_rows(
    bundle: PoolBundle,
    source_id: str,
    issues: list[str],
) -> dict[str, PolygonInventoryRow]:
    inventory_by_id: dict[str, PolygonInventoryRow] = {}
    native_keys: set[str] = set()
    for row in bundle.inventory:
        if row.inventory_row_id in inventory_by_id:
            issues.append(f"duplicate inventory row id {row.inventory_row_id}")
        inventory_by_id[row.inventory_row_id] = row
        key_json = native_key_json(row.native_key)
        if key_json in native_keys:
            issues.append(f"duplicate source-native polygon key {key_json}")
        native_keys.add(key_json)
        _validate_inventory_row(row, source_id, key_json, issues)
    return inventory_by_id


def _validate_inventory_row(
    row: PolygonInventoryRow,
    source_id: str,
    key_json: str,
    issues: list[str],
) -> None:
    if row.inventory_row_id != inventory_row_identifier(source_id, row.native_key):
        issues.append(f"inventory row id does not encode source pin and native key: {key_json}")
    if not _valid_native_key(row.native_key):
        issues.append(f"invalid source-native composite key {key_json}")
    if not _present(row.source_row_locator):
        issues.append(f"inventory row {row.inventory_row_id} has no source-row locator")
    if row.polygon_snapshot_id is None and not _present(row.exclusion_reason):
        issues.append(f"inventory row {row.inventory_row_id} has no geometry snapshot or exclusion")
    if row.polygon_snapshot_id is not None and row.exclusion_reason is not None:
        issues.append(f"inventory row {row.inventory_row_id} has both snapshot and exclusion")


def _validate_inventory_geometry_links(
    inventory: tuple[PolygonInventoryRow, ...],
    polygons: tuple[PolygonSnapshot, ...],
    polygon_by_id: dict[str, PolygonSnapshot],
    issues: list[str],
) -> None:
    snapshot_rows = {polygon.inventory_row_id for polygon in polygons}
    for row in inventory:
        if row.polygon_snapshot_id is not None and row.polygon_snapshot_id not in polygon_by_id:
            issues.append(
                f"inventory row {row.inventory_row_id} references a missing polygon snapshot"
            )
        if row.polygon_snapshot_id is None and row.inventory_row_id in snapshot_rows:
            issues.append(f"excluded inventory row {row.inventory_row_id} has a polygon snapshot")


def _validate_polygons(
    polygons: tuple[PolygonSnapshot, ...],
    inventory_by_id: dict[str, PolygonInventoryRow],
    source_id: str,
    issues: list[str],
) -> dict[str, PolygonSnapshot]:
    polygon_by_id: dict[str, PolygonSnapshot] = {}
    for polygon in polygons:
        if polygon.polygon_snapshot_id in polygon_by_id:
            issues.append(f"duplicate polygon snapshot id {polygon.polygon_snapshot_id}")
        polygon_by_id[polygon.polygon_snapshot_id] = polygon
        _validate_polygon_identity(polygon, inventory_by_id, source_id, issues)
        _validate_polygon_geometry(polygon, issues)
    return polygon_by_id


def _validate_polygon_identity(
    polygon: PolygonSnapshot,
    inventory_by_id: dict[str, PolygonInventoryRow],
    source_id: str,
    issues: list[str],
) -> None:
    inventory = inventory_by_id.get(polygon.inventory_row_id)
    if inventory is None:
        issues.append(f"polygon snapshot {polygon.polygon_snapshot_id} has no inventory row")
        return
    if inventory.native_key != polygon.native_key:
        issues.append(f"polygon snapshot {polygon.polygon_snapshot_id} has a mismatched native key")
    if inventory.polygon_snapshot_id != polygon.polygon_snapshot_id:
        issues.append(
            f"inventory row {inventory.inventory_row_id} has a mismatched polygon snapshot id"
        )
    expected_id = polygon_snapshot_identifier(
        source_id, polygon.native_key, polygon.geometry_sha256
    )
    if polygon.polygon_snapshot_id != expected_id:
        issues.append("polygon snapshot id does not encode key and geometry hash")
    if not _valid_native_key(polygon.native_key):
        issues.append(f"polygon snapshot {polygon.polygon_snapshot_id} has an invalid native key")


def _validate_polygon_geometry(polygon: PolygonSnapshot, issues: list[str]) -> None:
    polygon_id = polygon.polygon_snapshot_id
    if not _valid_uri(polygon.geometry_uri) or not _valid_sha256(polygon.geometry_sha256):
        issues.append(f"polygon snapshot {polygon_id} geometry artifact is not pinned")
    metadata = (
        polygon.geometry_format,
        polygon.crs,
        polygon.coordinate_transform,
        polygon.area_unit,
    )
    if any(not _present(value) for value in metadata):
        issues.append(f"polygon snapshot {polygon_id} lacks geometry metadata")
    if polygon.geometry_status.value == "usable":
        if polygon.original_area is None or not _finite_positive(polygon.original_area):
            issues.append(f"usable polygon snapshot {polygon_id} needs positive full area")
    elif polygon.original_area is not None:
        issues.append(f"unusable polygon snapshot {polygon_id} cannot claim full area")


def _validate_candidates(
    bundle: PoolBundle,
    inventory_by_id: dict[str, PolygonInventoryRow],
    polygon_by_id: dict[str, PolygonSnapshot],
    issues: list[str],
) -> dict[tuple[str, str], list[LabelCandidate]]:
    levels = {level.name: level for level in bundle.levels}
    grouped: dict[tuple[str, str], list[LabelCandidate]] = defaultdict(list)
    candidate_keys: set[tuple[str, str, str]] = set()
    reference = bundle.manifest.reference_pin
    for candidate in bundle.label_candidates:
        inventory = inventory_by_id.get(candidate.inventory_row_id)
        polygon = polygon_by_id.get(candidate.polygon_snapshot_id)
        if inventory is None or polygon is None:
            issues.append(f"label candidate {candidate.code} references an unknown polygon")
            continue
        if inventory.polygon_snapshot_id != candidate.polygon_snapshot_id:
            issues.append(f"label candidate {candidate.code} does not match its inventory geometry")
        level = levels.get(candidate.level)
        if level is None or not level.supported:
            issues.append(
                f"label candidate {candidate.code} uses an undeclared or unsupported level"
            )
        key = candidate.inventory_row_id, candidate.level, candidate.code
        if key in candidate_keys:
            issues.append(f"duplicate label candidate {key}")
        candidate_keys.add(key)
        grouped[(candidate.inventory_row_id, candidate.level)].append(candidate)
        _validate_candidate_values(candidate, polygon, reference, issues)
    return grouped


def _validate_candidate_values(
    candidate: LabelCandidate,
    polygon: PolygonSnapshot,
    reference: ReferencePin,
    issues: list[str],
) -> None:
    if polygon.geometry_status.value != "usable" or polygon.original_area is None:
        issues.append(f"label candidate {candidate.code} uses unusable polygon geometry")
        return
    _validate_candidate_measurement(candidate, polygon, issues)
    _validate_candidate_provenance(candidate, polygon, reference, issues)


def _validate_candidate_measurement(
    candidate: LabelCandidate,
    polygon: PolygonSnapshot,
    issues: list[str],
) -> None:
    original_area = polygon.original_area
    if original_area is None:
        issues.append(f"label candidate {candidate.code} needs a full polygon area")
        return
    if not all(
        math.isfinite(value)
        for value in (
            candidate.numerator,
            candidate.denominator,
            candidate.ratio,
            candidate.threshold,
        )
    ):
        issues.append(f"label candidate {candidate.code} has a non-finite measurement")
        return
    if candidate.denominator <= 0 or candidate.numerator < 0:
        issues.append(f"label candidate {candidate.code} has an invalid numerator or denominator")
    if candidate.numerator > candidate.denominator:
        issues.append(f"label candidate {candidate.code} support exceeds full polygon area")
    if not math.isclose(candidate.denominator, original_area, rel_tol=1e-12, abs_tol=1e-12):
        issues.append(
            f"label candidate {candidate.code} denominator is not full original polygon area"
        )
    expected_ratio = (
        candidate.numerator / candidate.denominator if candidate.denominator > 0 else math.nan
    )
    if not math.isclose(candidate.ratio, expected_ratio, rel_tol=1e-12, abs_tol=1e-12):
        issues.append(
            f"label candidate {candidate.code} ratio does not match numerator/denominator"
        )


def _validate_candidate_provenance(
    candidate: LabelCandidate,
    polygon: PolygonSnapshot,
    reference: ReferencePin,
    issues: list[str],
) -> None:
    if candidate.threshold != reference.threshold:
        issues.append(f"label candidate {candidate.code} does not use the pinned threshold")
    if candidate.support_semantics != reference.support_semantics:
        issues.append(f"label candidate {candidate.code} changes reference support semantics")
    if (
        candidate.support_unit != reference.support_unit
        or candidate.support_unit != polygon.area_unit
    ):
        issues.append(f"label candidate {candidate.code} does not use polygon support units")
    if candidate.source_revision != reference.source_revision:
        issues.append(f"label candidate {candidate.code} has a mismatched source revision")
    if candidate.taxonomy_revision != reference.taxonomy_revision:
        issues.append(f"label candidate {candidate.code} has a mismatched taxonomy revision")
    if not _present(candidate.assignment_rule_version) or not _present(candidate.code):
        issues.append(f"label candidate {candidate.code} lacks code or assignment-rule version")


def _validate_outcomes(
    bundle: PoolBundle,
    inventory_by_id: dict[str, PolygonInventoryRow],
    polygon_by_id: dict[str, PolygonSnapshot],
    candidates_by_key: dict[tuple[str, str], list[LabelCandidate]],
    issues: list[str],
) -> None:
    levels = {level.name: level for level in bundle.levels}
    outcomes_by_key = {}
    for outcome in bundle.label_outcomes:
        key = outcome.inventory_row_id, outcome.level
        if key in outcomes_by_key:
            issues.append(f"duplicate label outcome {key}")
        outcomes_by_key[key] = outcome
        inventory = inventory_by_id.get(outcome.inventory_row_id)
        if inventory is None:
            issues.append(f"label outcome {key} references unknown inventory")
            continue
        if outcome.polygon_snapshot_id != inventory.polygon_snapshot_id:
            issues.append(f"label outcome {key} has a mismatched polygon snapshot")
        level = levels.get(outcome.level)
        if level is None or not level.supported:
            issues.append(f"label outcome {key} uses an undeclared or unsupported level")
            continue
        polygon = polygon_by_id.get(inventory.polygon_snapshot_id or "")
        candidates = candidates_by_key.get(key, [])
        qualifiers = [
            candidate for candidate in candidates if candidate.ratio >= candidate.threshold
        ]
        _validate_one_outcome(outcome, polygon, qualifiers, candidates, key, issues)
    _validate_outcome_coverage(bundle, levels, outcomes_by_key, issues)


def _validate_one_outcome(
    outcome: LabelOutcome,
    polygon: PolygonSnapshot | None,
    qualifiers: list[LabelCandidate],
    candidates: list[LabelCandidate],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if (polygon is None or polygon.geometry_status.value != "usable") and (
        outcome.status is not OutcomeStatus.EXCLUDED
    ):
        issues.append(f"unusable geometry must have an excluded outcome for {key}")
    if outcome.status is OutcomeStatus.ASSIGNED:
        _validate_assigned_outcome(outcome, qualifiers, key, issues)
    elif outcome.status is OutcomeStatus.AMBIGUOUS:
        _validate_ambiguous_outcome(outcome, qualifiers, candidates, key, issues)
    elif outcome.status is OutcomeStatus.UNASSIGNED:
        _validate_unassigned_outcome(outcome, qualifiers, key, issues)
    elif outcome.status is OutcomeStatus.EXCLUDED:
        _validate_excluded_outcome(outcome, candidates, key, issues)


def _validate_assigned_outcome(
    outcome: LabelOutcome,
    qualifiers: list[LabelCandidate],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if len(qualifiers) != 1 or qualifiers[0].code != outcome.assigned_code:
        issues.append(f"assigned outcome {key} must select its only qualifying code")
    if outcome.reason is not None or outcome.ambiguity_basis is not None:
        issues.append(f"assigned outcome {key} cannot have an exclusion or ambiguity reason")


def _validate_ambiguous_outcome(
    outcome: LabelOutcome,
    qualifiers: list[LabelCandidate],
    candidates: list[LabelCandidate],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if outcome.assigned_code is not None or not _present(outcome.reason):
        issues.append(f"ambiguous outcome {key} must remain unassigned with a reason")
    if outcome.ambiguity_basis is None:
        issues.append(f"ambiguous outcome {key} must state its ambiguity basis")
    if outcome.ambiguity_basis is AmbiguityBasis.MULTIPLE_QUALIFIERS:
        if len(qualifiers) < _MIN_AMBIGUOUS_COUNT:
            issues.append(f"ambiguous outcome {key} needs at least two qualifying codes")
    elif outcome.ambiguity_basis is AmbiguityBasis.AMBIGUOUS_EVIDENCE and not candidates:
        issues.append(f"ambiguous-evidence outcome {key} needs inspectable candidate evidence")
    if len(qualifiers) == 1 and outcome.ambiguity_basis is not AmbiguityBasis.AMBIGUOUS_EVIDENCE:
        issues.append(f"single qualifying code at {key} requires an evidence-ambiguity basis")
    if len(qualifiers) > 1 and outcome.ambiguity_basis is not AmbiguityBasis.MULTIPLE_QUALIFIERS:
        issues.append(f"competing qualifying codes at {key} require multiple-qualifier ambiguity")


def _validate_unassigned_outcome(
    outcome: LabelOutcome,
    qualifiers: list[LabelCandidate],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if qualifiers:
        issues.append(f"unassigned outcome {key} has one or more qualifying codes")
    if outcome.assigned_code is not None or not _present(outcome.reason):
        issues.append(f"unassigned outcome {key} must have a reason and no assigned code")
    if outcome.ambiguity_basis is not None:
        issues.append(f"unassigned outcome {key} cannot have ambiguity basis")


def _validate_excluded_outcome(
    outcome: LabelOutcome,
    candidates: list[LabelCandidate],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if outcome.assigned_code is not None or not _present(outcome.reason):
        issues.append(f"excluded outcome {key} must have a reason and no assigned code")
    if outcome.ambiguity_basis is not None or candidates:
        issues.append(f"excluded outcome {key} cannot contain assignment candidates")


def _validate_outcome_coverage(
    bundle: PoolBundle,
    levels: dict[str, LabelLevel],
    outcomes_by_key: dict[tuple[str, str], LabelOutcome],
    issues: list[str],
) -> None:
    inventory_ids = {row.inventory_row_id for row in bundle.inventory}
    for level in levels.values():
        rows = {row_id for row_id, name in outcomes_by_key if name == level.name}
        if level.supported and rows != inventory_ids:
            missing = sorted(inventory_ids - rows)
            extra = sorted(rows - inventory_ids)
            issues.append(
                f"level {level.name} decision coverage mismatch: missing={missing}, extra={extra}"
            )
        if not level.supported and rows:
            issues.append(f"unsupported level {level.name} must not have per-polygon outcomes")


def _validate_documents(
    bundle: PoolBundle,
    inventory_by_id: dict[str, PolygonInventoryRow],
    issues: list[str],
) -> None:
    docs = _validate_source_documents(bundle, issues)
    _validate_text_links(bundle, docs, inventory_by_id, issues)


def _validate_source_documents(
    bundle: PoolBundle,
    issues: list[str],
) -> dict[str, SourceDocument]:
    docs = {document.document_id: document for document in bundle.documents}
    if len(docs) != len(bundle.documents):
        issues.append("source document ids must be unique")
    for document in bundle.documents:
        if not _present(document.document_id) or not _valid_native_key(document.native_key):
            issues.append(f"source document {document.document_id} needs a native composite key")
        if not _present(document.source_revision):
            issues.append(f"source document {document.document_id} has no source revision")
        if document.body_status is BodyStatus.AVAILABLE and not _valid_sha256(
            document.content_sha256
        ):
            issues.append(f"available source document {document.document_id} needs a content hash")
        if document.content_sha256 is not None and not _valid_sha256(document.content_sha256):
            issues.append(f"source document {document.document_id} has an invalid content hash")
    return docs


def _validate_text_links(
    bundle: PoolBundle,
    docs: dict[str, SourceDocument],
    inventory_by_id: dict[str, PolygonInventoryRow],
    issues: list[str],
) -> None:
    link_keys = set()
    for link in bundle.text_polygon_links:
        key = link.document_id, native_key_json(link.native_link_key)
        if key in link_keys:
            issues.append(f"duplicate source-native text link decision {key}")
        link_keys.add(key)
        if link.document_id not in docs:
            issues.append(f"text link references unknown document {link.document_id}")
        if not _present(link.source_revision) or not _present(link.evidence):
            issues.append(f"text link {key} must retain revision and link evidence")
        _validate_text_link_target(link, inventory_by_id, issues)


def _validate_text_link_target(
    link: TextPolygonLink,
    inventory_by_id: dict[str, PolygonInventoryRow],
    issues: list[str],
) -> None:
    candidates = link.candidate_inventory_row_ids
    if len(candidates) != len(set(candidates)):
        issues.append(f"text link {link.document_id} repeats a candidate polygon")
    if any(candidate not in inventory_by_id for candidate in candidates):
        issues.append(f"text link {link.document_id} references unknown candidate inventory rows")
    if link.status is TextLinkStatus.LINKED:
        if link.inventory_row_id not in inventory_by_id or candidates:
            issues.append(f"linked document {link.document_id} needs one selected polygon")
    elif link.inventory_row_id is not None:
        issues.append(f"unresolved text link {link.document_id} cannot select a polygon")
    if link.status is TextLinkStatus.AMBIGUOUS and len(candidates) < _MIN_AMBIGUOUS_COUNT:
        issues.append(f"ambiguous text link {link.document_id} must retain at least two candidates")
    if link.status is TextLinkStatus.UNMATCHED and candidates:
        issues.append(f"unmatched text link {link.document_id} cannot contain polygon candidates")


def _validate_entity_links(links: tuple[EntityLink, ...], issues: list[str]) -> None:
    keys = set()
    for link in links:
        if not _present(link.link_version) or not _present(link.evidence):
            issues.append("entity links require a version and evidence")
        left = link.left.source_snapshot_id, native_key_json(link.left.native_key)
        right = link.right.source_snapshot_id, native_key_json(link.right.native_key)
        if left == right:
            issues.append("entity link cannot link a polygon to itself")
        key = link.link_version, min(left, right), max(left, right)
        if key in keys:
            issues.append(f"duplicate versioned entity link {key}")
        keys.add(key)
        if not _present(link.left.source_snapshot_id) or not _valid_native_key(
            link.left.native_key
        ):
            issues.append("entity-link left endpoint needs a source snapshot and full native key")
        if not _present(link.right.source_snapshot_id) or not _valid_native_key(
            link.right.native_key
        ):
            issues.append("entity-link right endpoint needs a source snapshot and full native key")
        if link.evidence_sha256 is not None and not _valid_sha256(link.evidence_sha256):
            issues.append("entity-link evidence SHA-256 is invalid")


def _validate_eligibility(
    bundle: PoolBundle,
    inventory_by_id: dict[str, PolygonInventoryRow],
    issues: list[str],
) -> None:
    seen: set[tuple[str, str]] = set()
    for record in bundle.eligibility:
        key = record.inventory_row_id, record.cohort_policy_version
        if key in seen:
            issues.append(f"duplicate downstream eligibility record {key}")
        seen.add(key)
        _validate_eligibility_record(record, inventory_by_id, key, issues)


def _validate_eligibility_record(
    record: PoolEligibility,
    inventory_by_id: dict[str, PolygonInventoryRow],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    if record.inventory_row_id not in inventory_by_id:
        issues.append(f"eligibility record {key} references unknown inventory")
    if not _present(record.cohort_policy_version) or not _present(record.evidence):
        issues.append(f"eligibility record {key} needs policy version and evidence")
    if record.status is EligibilityStatus.ELIGIBLE:
        if record.reason is not None or record.replaced_by_inventory_row_id is not None:
            issues.append(f"eligible record {key} cannot have exclusion or replacement fields")
    elif record.status is EligibilityStatus.EXCLUDED:
        if not _present(record.reason) or record.replaced_by_inventory_row_id is not None:
            issues.append(f"excluded record {key} needs a reason and no replacement")
    elif record.status is EligibilityStatus.REPLACED:
        _validate_replacement(record, inventory_by_id, key, issues)


def _validate_replacement(
    record: PoolEligibility,
    inventory_by_id: dict[str, PolygonInventoryRow],
    key: tuple[str, str],
    issues: list[str],
) -> None:
    replacement = record.replaced_by_inventory_row_id
    if not _present(record.reason) or not replacement:
        issues.append(f"replaced record {key} needs reason and replacement lineage")
    elif replacement not in inventory_by_id:
        issues.append(f"replacement target for {key} is not in raw inventory")
    elif replacement == record.inventory_row_id:
        issues.append(f"replacement record {key} cannot replace itself")


def _validate_accounting(bundle: PoolBundle, issues: list[str]) -> None:
    expected = build_full_pool(
        PoolBuildInput(
            pool_key=bundle.manifest.pool_key,
            source_snapshot=bundle.manifest.source_snapshot,
            reference_pin=bundle.manifest.reference_pin,
            adapter_id=bundle.manifest.adapter_id,
            adapter_revision=bundle.manifest.adapter_revision,
            builder_revision=bundle.manifest.builder_revision,
            inventory=bundle.inventory,
            polygons=bundle.polygons,
            levels=bundle.levels,
            label_candidates=bundle.label_candidates,
            label_outcomes=bundle.label_outcomes,
            documents=bundle.documents,
            text_polygon_links=bundle.text_polygon_links,
            entity_links=bundle.entity_links,
            eligibility=bundle.eligibility,
        ),
        contract_version=bundle.manifest.contract_version,
        pool_version=bundle.manifest.pool_version,
    )
    if bundle.manifest.total_source_keys != len(bundle.inventory):
        issues.append("manifest source-key count does not equal raw polygon inventory")
    if bundle.manifest.level_accounting != expected.manifest.level_accounting:
        issues.append("manifest per-level accounting does not match table rows")
    for level, counts in zip(bundle.levels, bundle.manifest.level_accounting, strict=False):
        if (
            counts.supported != level.supported
            or counts.unsupported_reason != level.unsupported_reason
        ):
            issues.append(f"manifest support declaration does not match level {level.name}")
        if level.supported and counts.processed_decisions != counts.total_source_keys:
            issues.append(f"supported level {level.name} does not account for every source key")
        outcome_sum = counts.assigned + counts.ambiguous + counts.unassigned + counts.excluded
        if level.supported and outcome_sum != counts.processed_decisions:
            issues.append(f"level {level.name} outcome counts do not sum to processed decisions")


def _validate_payload(payload: EvaluationPayload, name: str, issues: list[str]) -> None:
    if payload.status is BodyStatus.AVAILABLE:
        if payload.value is None:
            issues.append(f"available evaluation {name} needs its exact body")
        elif not _valid_sha256(payload.sha256) or _sha256(payload.value) != payload.sha256:
            issues.append(f"available evaluation {name} hash does not match exact body")
    elif payload.value is not None:
        issues.append(f"{payload.status.value} evaluation {name} cannot expose a body")
    if payload.sha256 is not None and not _valid_sha256(payload.sha256):
        issues.append(f"evaluation {name} SHA-256 is invalid")


def _canonical_settings(value: str) -> bool:
    try:
        settings = json.loads(value)
    except json.JSONDecodeError:
        return False
    return (
        isinstance(settings, dict)
        and json.dumps(settings, ensure_ascii=False, sort_keys=True, separators=(",", ":")) == value
    )


def _valid_native_key(key: tuple[tuple[str, str | int], ...]) -> bool:
    if not key or tuple(sorted(key)) != key:
        return False
    names = [name for name, _ in key]
    return (
        len(names) == len(set(names))
        and all(_present(name) for name in names)
        and all(isinstance(value, (str, int)) and not isinstance(value, bool) for _, value in key)
    )


def _present(value: str | None) -> bool:
    return value is not None and bool(value.strip())


def _finite_positive(value: float) -> bool:
    return math.isfinite(value) and value > 0


def _valid_sha256(value: str | None) -> bool:
    return value is not None and bool(_SHA256_RE.fullmatch(value))


def _valid_uri(value: str) -> bool:
    if not _present(value):
        return False
    parsed = urlparse(value)
    return bool(parsed.scheme and (parsed.netloc or parsed.path))


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()
