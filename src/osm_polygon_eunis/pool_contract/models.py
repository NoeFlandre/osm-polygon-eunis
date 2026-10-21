"""Typed, source-independent records for versioned polygon-label pools."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

type NativeKey = tuple[tuple[str, str | int], ...]


class TextSource(StrEnum):
    """Text corpus crossed with a reference-label system."""

    DESCRIPTION = "description"
    WEBSITE = "website"
    WIKI = "wiki"


class ReferenceSystem(StrEnum):
    """Reference source used to assign polygon labels."""

    WORLDCOVER = "worldcover"
    LUCAS = "lucas"
    EUNIS = "eunis"


class SupportSemantics(StrEnum):
    """Meaning of the reference geometry support numerator."""

    MAPPED_COVER = "mapped_cover"
    SURVEYED_FOOTPRINT = "surveyed_footprint"
    MODELED_HABITAT_SUPPORT = "modeled_habitat_support"


class GeometryStatus(StrEnum):
    """Usability of the captured source geometry."""

    USABLE = "usable"
    INVALID = "invalid"
    EMPTY = "empty"
    TRANSFORM_FAILED = "transform_failed"


class AssignmentKind(StrEnum):
    """Whether support comes directly from a class or a hierarchy rollup."""

    DIRECT = "direct"
    ANCESTOR_DERIVED = "ancestor_derived"


class OutcomeStatus(StrEnum):
    """One terminal reference-label decision for a polygon at one level."""

    ASSIGNED = "assigned"
    AMBIGUOUS = "ambiguous"
    UNASSIGNED = "unassigned"
    EXCLUDED = "excluded"


class AmbiguityBasis(StrEnum):
    """Why a per-level label outcome remains unassigned as ambiguous."""

    MULTIPLE_QUALIFIERS = "multiple_qualifiers"
    AMBIGUOUS_EVIDENCE = "ambiguous_evidence"


class TextLinkStatus(StrEnum):
    """Resolution state for source-native text-to-polygon links."""

    LINKED = "linked"
    AMBIGUOUS = "ambiguous"
    UNMATCHED = "unmatched"
    KEY_MISMATCH = "key_mismatch"
    TEXT_HASH_MISMATCH = "text_hash_mismatch"
    GEOMETRY_MISSING = "geometry_missing"


class BodyStatus(StrEnum):
    """Availability state for source text or evaluation payloads."""

    AVAILABLE = "available"
    MISSING = "missing"
    PRIVATE = "private"
    NOT_RETAINED = "not_retained"


class EligibilityStatus(StrEnum):
    """Downstream candidate-cohort status, separate from pool labels."""

    ELIGIBLE = "eligible"
    EXCLUDED = "excluded"
    REPLACED = "replaced"


class EvaluationArm(StrEnum):
    """Evaluation sampling arm; these are not dataset split names."""

    YES = "yes"
    NO = "no"


class DataSplit(StrEnum):
    """Train/validation/test assignment kept outside full label pools."""

    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


@dataclass(frozen=True, slots=True)
class PoolKey:
    """Identify one text-source by reference-system pool."""

    text_source: TextSource
    reference_system: ReferenceSystem

    @property
    def name(self) -> str:
        """Return the stable lowercase pool key."""

        return f"{self.text_source.value}-{self.reference_system.value}"


@dataclass(frozen=True, slots=True)
class SourceSnapshot:
    """Pin the complete frozen source inventory and its native key space."""

    snapshot_id: str
    repository: str
    revision: str
    pbf_uri: str
    pbf_sha256: str


@dataclass(frozen=True, slots=True)
class RawInventoryEvidence:
    """Independent receipt from enumerating native polygon keys before joins."""

    enumeration_rule: str
    scan_report_uri: str
    scan_report_sha256: str
    expected_key_count: int
    expected_keyset_sha256: str


@dataclass(frozen=True, slots=True)
class ReferencePin:
    """Pin label release, taxonomy, semantics, and shared assignment rule."""

    release_id: str
    source_repository: str
    source_revision: str
    taxonomy_id: str
    taxonomy_revision: str
    support_semantics: SupportSemantics
    support_unit: str
    threshold: float
    support_union_policy: str


@dataclass(frozen=True, slots=True)
class LabelLevel:
    """Declare a directly supported or explicitly unsupported taxonomy level."""

    name: str
    supported: bool
    unsupported_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PolygonInventoryRow:
    """One key from the raw frozen polygon source, before text filtering."""

    inventory_row_id: str
    native_key: NativeKey
    source_row_locator: str
    polygon_snapshot_id: str | None
    exclusion_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PolygonSnapshot:
    """Captured native-key geometry and its recoverable artifact location."""

    polygon_snapshot_id: str
    inventory_row_id: str
    native_key: NativeKey
    geometry_uri: str
    geometry_sha256: str
    geometry_format: str
    crs: str
    coordinate_transform: str
    geometry_status: GeometryStatus
    original_area: float | None
    area_unit: str


@dataclass(frozen=True, slots=True)
class LabelCandidate:
    """One long-form class-support measurement for a polygon and label level."""

    inventory_row_id: str
    polygon_snapshot_id: str
    level: str
    code: str
    numerator: float
    denominator: float
    ratio: float
    threshold: float
    assignment_rule_version: str
    assignment_kind: AssignmentKind
    support_semantics: SupportSemantics
    support_unit: str
    source_revision: str
    taxonomy_revision: str


@dataclass(frozen=True, slots=True)
class LabelOutcome:
    """Resolved, ambiguous, unassigned, or excluded decision per polygon level."""

    inventory_row_id: str
    polygon_snapshot_id: str | None
    level: str
    status: OutcomeStatus
    assigned_code: str | None = None
    reason: str | None = None
    ambiguity_basis: AmbiguityBasis | None = None


@dataclass(frozen=True, slots=True)
class SourceDocument:
    """Source-native document metadata without exposing a private text body."""

    document_id: str
    native_key: NativeKey
    source_revision: str
    content_sha256: str | None
    body_status: BodyStatus


@dataclass(frozen=True, slots=True)
class TextPolygonLink:
    """Explicit source-native link resolution from a document to inventory rows."""

    document_id: str
    inventory_row_id: str | None
    native_link_key: NativeKey
    source_revision: str
    status: TextLinkStatus
    evidence: str
    candidate_inventory_row_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityReference:
    """Fully namespaced polygon identity used by cross-release entity links."""

    source_snapshot_id: str
    native_key: NativeKey


@dataclass(frozen=True, slots=True)
class EntityLink:
    """Versioned cross-release polygon link with inspectable evidence."""

    link_version: str
    left: EntityReference
    right: EntityReference
    evidence: str
    evidence_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class PoolEligibility:
    """Optional downstream cohort eligibility and replacement lineage."""

    inventory_row_id: str
    cohort_policy_version: str
    status: EligibilityStatus
    evidence: str
    reason: str | None = None
    replaced_by_inventory_row_id: str | None = None


@dataclass(frozen=True, slots=True)
class PoolBuildInput:
    """All caller-supplied tables for one source-independent pool build."""

    pool_key: PoolKey
    source_snapshot: SourceSnapshot
    raw_inventory_evidence: RawInventoryEvidence
    reference_pin: ReferencePin
    adapter_id: str
    adapter_revision: str
    builder_revision: str
    inventory: tuple[PolygonInventoryRow, ...]
    polygons: tuple[PolygonSnapshot, ...]
    levels: tuple[LabelLevel, ...]
    label_candidates: tuple[LabelCandidate, ...]
    label_outcomes: tuple[LabelOutcome, ...]
    documents: tuple[SourceDocument, ...] = ()
    text_polygon_links: tuple[TextPolygonLink, ...] = ()
    entity_links: tuple[EntityLink, ...] = ()
    eligibility: tuple[PoolEligibility, ...] = ()


@dataclass(frozen=True, slots=True)
class LevelAccounting:
    """Manifest counts for all source keys at one declared label level."""

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
    exclusions_by_reason: tuple[tuple[str, int], ...]


@dataclass(frozen=True, slots=True)
class PoolManifest:
    """Immutable provenance pins and inventory accounting for one pool."""

    contract_version: str
    pool_version: str
    pool_key: PoolKey
    source_snapshot: SourceSnapshot
    raw_inventory_evidence: RawInventoryEvidence
    reference_pin: ReferencePin
    adapter_id: str
    adapter_revision: str
    builder_revision: str
    total_source_keys: int
    level_accounting: tuple[LevelAccounting, ...]

    @property
    def pool_id(self) -> str:
        """Return the stable logical pool identifier."""

        return f"{self.pool_version}/{self.pool_key.name}/{self.source_snapshot.snapshot_id}"


@dataclass(frozen=True, slots=True)
class PoolBundle:
    """Separate logical pool tables plus their immutable source manifest."""

    manifest: PoolManifest
    levels: tuple[LabelLevel, ...]
    inventory: tuple[PolygonInventoryRow, ...]
    polygons: tuple[PolygonSnapshot, ...]
    label_candidates: tuple[LabelCandidate, ...]
    label_outcomes: tuple[LabelOutcome, ...]
    documents: tuple[SourceDocument, ...]
    text_polygon_links: tuple[TextPolygonLink, ...]
    entity_links: tuple[EntityLink, ...]
    eligibility: tuple[PoolEligibility, ...]


@dataclass(frozen=True, slots=True)
class PoolValidationReport:
    """Collect all contract issues without hiding independent failures."""

    issues: tuple[str, ...]

    @property
    def ok(self) -> bool:
        """Whether every invariant passed."""

        return not self.issues


@dataclass(frozen=True, slots=True)
class GateReport:
    """Validation result for the required nine source-reference pools."""

    missing_pools: tuple[str, ...]
    duplicate_pools: tuple[str, ...]
    invalid_pools: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def ok(self) -> bool:
        """Whether all nine distinct pools are complete and valid."""

        return not (self.missing_pools or self.duplicate_pools or self.invalid_pools)


@dataclass(frozen=True, slots=True)
class EvaluationPayload:
    """Exact evaluation text or an explicit missing/private retention state."""

    status: BodyStatus
    value: str | None
    sha256: str | None


@dataclass(frozen=True, slots=True)
class EvaluationCase:
    """Evaluation sampling record kept downstream from full label pools."""

    case_id: str
    pool_id: str
    inventory_row_id: str
    target_level: str
    arm: EvaluationArm
    sampling_policy_version: str
    selection_order: int
    source_document_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class EvaluationTrace:
    """Inspectable prompt, response, settings, and geometry for one case."""

    trace_id: str
    case_id: str
    model_id: str
    settings_json: str
    prompt: EvaluationPayload
    raw_response: EvaluationPayload
    reference_geometry_uri: str
    reference_geometry_sha256: str


@dataclass(frozen=True, slots=True)
class SplitAssignment:
    """Versioned dataset split evidence, separate from evaluation arms."""

    entity: EntityReference
    split: DataSplit
    policy_version: str
    evidence: str
