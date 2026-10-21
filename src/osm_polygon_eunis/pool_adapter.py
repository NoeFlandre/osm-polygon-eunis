"""EUNIS-specific full-pool assignments, independent of text eligibility."""

from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite

from .eea import (
    EEA_2021_TAXONOMY_CACHE_SHA256 as EEA_2021_TAXONOMY_SHA256,
)
from .eea import (
    ClassificationTaxonomy,
)
from .pool_contract import (
    AmbiguityBasis,
    AssignmentKind,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    OutcomeStatus,
    PolygonInventoryRow,
    ReferencePin,
    SupportSemantics,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
)
from .pool_contract.models import NativeKey

EUNIS_POOL_LEVELS = (
    LabelLevel("L1", True),
    LabelLevel("L2", True),
    LabelLevel("L3", True),
    LabelLevel("L4", False, "EEA probability maps directly support L3 only"),
    LabelLevel("L5", False, "EEA probability maps directly support L3 only"),
    LabelLevel("L6", False, "EEA probability maps directly support L3 only"),
    LabelLevel("L7", False, "EEA probability maps directly support L3 only"),
    LabelLevel("L8", False, "EEA probability maps directly support L3 only"),
)
EUNIS_ASSIGNMENT_RULE_VERSION = "eea-positive-cell-union-80pct-v1"
EUNIS_THRESHOLD = 0.8
EUNIS_SUPPORT_SEMANTICS = SupportSemantics.MODELED_HABITAT_SUPPORT
_EUNIS_DIRECT_MAP_LEVEL = 3
EEA_2021_NATIVE_MANIFEST_SHA256 = "4a91e553b2e0271e7ca935e542dd092fe45288c5f9bb4a09f40f6b74da4eae22"
EEA_2021_ASSET_COUNT = 136
EEA_2021_ASSET_BYTES = 5_007_209_626
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True, slots=True)
class EunisTaxonomyPath:
    """Explicit L1/L2/L3 ancestry for one directly mapped EUNIS code."""

    level_1: str
    level_2: str
    level_3: str


@dataclass(frozen=True, slots=True)
class EunisTaxonomy:
    """Hash-pinned, explicit hierarchy paths for direct EEA map codes.

    The paths are deliberately supplied by the caller. EUNIS code prefixes are
    not a safe substitute for the published taxonomy hierarchy.
    """

    taxonomy_id: str
    taxonomy_revision: str
    taxonomy_sha256: str
    paths_by_direct_code: Mapping[str, EunisTaxonomyPath]

    def __post_init__(self) -> None:
        """Validate the explicit mapping and its immutable taxonomy pin."""

        if not self.taxonomy_id.strip() or not self.taxonomy_revision.strip():
            raise ValueError("EUNIS taxonomy id and revision must be present")
        if not _SHA256_RE.fullmatch(self.taxonomy_sha256):
            raise ValueError("EUNIS taxonomy SHA-256 must be lowercase hexadecimal")
        if not self.paths_by_direct_code:
            raise ValueError("EUNIS taxonomy mapping must contain direct map codes")
        for direct_code, path in self.paths_by_direct_code.items():
            if not direct_code.strip() or path.level_3 != direct_code:
                raise ValueError("each direct map code must map to itself at EUNIS L3")
            if any(not value.strip() for value in (path.level_1, path.level_2, path.level_3)):
                raise ValueError(f"taxonomy path for {direct_code} contains an empty code")

    @property
    def pinned_revision(self) -> str:
        """Return the taxonomy revision with its immutable source checksum."""

        return f"{self.taxonomy_revision}@sha256:{self.taxonomy_sha256}"

    def support_groups(self) -> dict[tuple[str, str], tuple[str, ...]]:
        """Return direct map-code members grouped by each supported class."""

        members: dict[tuple[str, str], set[str]] = defaultdict(set)
        for direct_code, path in self.paths_by_direct_code.items():
            members[("L1", path.level_1)].add(direct_code)
            members[("L2", path.level_2)].add(direct_code)
            members[("L3", path.level_3)].add(direct_code)
        return {key: tuple(sorted(codes)) for key, codes in sorted(members.items())}

    def validate_direct_codes(self, direct_codes: Iterable[str]) -> None:
        """Require a one-to-one taxonomy path for every pinned direct map code."""

        mapped = set(self.paths_by_direct_code)
        found = set(direct_codes)
        missing = sorted(found - mapped)
        unused = sorted(mapped - found)
        if missing or unused:
            raise ValueError(
                "taxonomy/map direct-code mismatch: "
                f"without taxonomy={missing}, without map={unused}"
            )


@dataclass(frozen=True, slots=True)
class EunisPoolSourceLayout:
    """Pinned raw and relevance-source paths for one text-source pool."""

    text_source: TextSource
    repository: str
    polygon_glob: str
    raw_revision: str
    relevance_repository: str
    relevance_revision: str
    relevance_path_globs: tuple[tuple[str, str], ...]
    document_link_glob: str | None = None
    document_glob: str | None = None
    sentence_glob: str | None = None


EUNIS_POOL_SOURCE_LAYOUTS: Mapping[TextSource, EunisPoolSourceLayout] = {
    TextSource.DESCRIPTION: EunisPoolSourceLayout(
        text_source=TextSource.DESCRIPTION,
        repository="NoeFlandre/osm-polygon-description-tag",
        polygon_glob="data/*.parquet",
        raw_revision="b4706eb315c66a8a57135289e7612dca8e03caf8",
        relevance_repository="NoeFlandre/osm-polygon-description-tag-landuse",
        relevance_revision="ce98b328f3c6543800a3d931877b51eb3f0cf768",
        relevance_path_globs=(("description", "labels/language-v1/data/*.parquet"),),
    ),
    TextSource.WEBSITE: EunisPoolSourceLayout(
        text_source=TextSource.WEBSITE,
        repository="NoeFlandre/osm-polygon-website-tag",
        polygon_glob="polygons/*.parquet",
        raw_revision="4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
        relevance_repository="NoeFlandre/osm-polygon-website-tag-landuse",
        relevance_revision="e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
        relevance_path_globs=(("website", "labels/polygons/*.parquet"),),
    ),
    TextSource.WIKI: EunisPoolSourceLayout(
        text_source=TextSource.WIKI,
        repository="NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        polygon_glob="polygons/*.parquet",
        raw_revision="d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
        relevance_repository="NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
        relevance_revision="49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
        relevance_path_globs=(
            ("wikipedia", "labels/wikipedia/sentences/*.parquet"),
            ("wikivoyage", "labels/wikivoyage/sentences/*.parquet"),
        ),
        document_link_glob="polygon_document_links/*.parquet",
        document_glob="wikipedia/documents/*.parquet",
        sentence_glob="wikivoyage/sentences/*.parquet",
    ),
}


def hierarchy_support_groups(
    taxonomy: EunisTaxonomy,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Expose map groups for direct L3 plus independently rolled-up L1/L2."""

    return taxonomy.support_groups()


def make_eunis_taxonomy_from_classification(
    classification: ClassificationTaxonomy,
    direct_codes: Iterable[str],
    *,
    taxonomy_id: str,
    taxonomy_revision: str,
    taxonomy_sha256: str,
) -> EunisTaxonomy:
    """Build explicit map-code paths from a verified classification hierarchy."""

    codes = tuple(sorted(set(direct_codes)))
    if not codes:
        raise ValueError("EEA direct map codes must be present")
    paths: dict[str, EunisTaxonomyPath] = {}
    for code in codes:
        if classification.levels.get(code) != _EUNIS_DIRECT_MAP_LEVEL:
            raise ValueError(f"EEA direct map code {code} is not a classification L3 code")
        by_level = {
            classification.levels[ancestor]: ancestor
            for ancestor in classification.ancestors_including_self(code)
            if ancestor in classification.levels and classification.levels[ancestor] in {1, 2, 3}
        }
        if set(by_level) != {1, 2, 3} or by_level[3] != code:
            raise ValueError(f"EEA direct map code {code} lacks one L1-L2-L3 parent path")
        paths[code] = EunisTaxonomyPath(by_level[1], by_level[2], by_level[3])
    return EunisTaxonomy(
        taxonomy_id=taxonomy_id,
        taxonomy_revision=taxonomy_revision,
        taxonomy_sha256=taxonomy_sha256,
        paths_by_direct_code=paths,
    )


def make_eunis_2021_reference_pin(taxonomy: EunisTaxonomy) -> ReferencePin:
    """Build the checked EEA v1 2021 pin from verified manifest digests."""

    if taxonomy.taxonomy_sha256 != EEA_2021_TAXONOMY_SHA256:
        raise ValueError("taxonomy digest does not match the verified EEA 2021 taxonomy input")
    return ReferencePin(
        release_id="EEA EUNIS habitat probability maps v1 2021",
        source_repository="https://biodiversity.europa.eu/resources/search-habitat",
        source_revision=(
            f"native-manifest-sha256:{EEA_2021_NATIVE_MANIFEST_SHA256};"
            f"asset-count:{EEA_2021_ASSET_COUNT};asset-bytes:{EEA_2021_ASSET_BYTES};"
            "crs:EPSG:3035;positive-map-value-threshold:0"
        ),
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.pinned_revision,
        support_semantics=EUNIS_SUPPORT_SEMANTICS,
        support_unit="m2",
        threshold=EUNIS_THRESHOLD,
        support_union_policy="union_by_class",
    )


@dataclass(frozen=True, slots=True)
class EunisLabelAttempt:
    """One polygon's complete support evidence or terminal exclusion reason."""

    inventory_row_id: str
    polygon_snapshot_id: str | None
    original_area: float | None
    support_areas: Mapping[tuple[str, str], float] | None
    taxonomy: EunisTaxonomy
    source_revision: str
    query_error: str | None = None
    exclusion_reason: str | None = None
    threshold: float = EUNIS_THRESHOLD


def resolve_eunis_labels(
    attempt: EunisLabelAttempt,
) -> tuple[tuple[LabelCandidate, ...], tuple[LabelOutcome, ...]]:
    """Create one independent outcome per supported level, preserving attempts.

    All positive per-class support measurements are retained as long-form
    candidates, including those below threshold. The denominator is always the
    full projected polygon area. A level is assigned only when exactly one
    class has support at least 80 percent of that full area.
    """

    excluded = _validated_exclusion(attempt)
    if excluded is not None:
        return (), _excluded_outcomes(attempt, excluded)
    if attempt.original_area is None or attempt.support_areas is None:
        raise ValueError("validated label attempts need a positive denominator and support map")

    denominator = float(attempt.original_area)
    taxonomy_by_class = _taxonomy_code_paths(attempt.taxonomy)
    candidates = []
    by_level: dict[str, list[LabelCandidate]] = {"L1": [], "L2": [], "L3": []}
    for (level, code), raw_numerator in sorted(attempt.support_areas.items()):
        if level not in by_level:
            raise ValueError(f"unsupported EUNIS support level {level!r}")
        numerator = float(raw_numerator)
        if numerator < 0.0 or numerator > denominator:
            raise ValueError(f"support for {level}/{code} falls outside full polygon area")
        if numerator == 0.0:
            continue
        candidate = _make_candidate(
            attempt,
            level,
            code,
            numerator,
            denominator,
            taxonomy_by_class,
        )
        candidates.append(candidate)
        by_level[level].append(candidate)

    outcomes = tuple(
        _resolve_level(attempt, level, by_level[level.name])
        for level in EUNIS_POOL_LEVELS
        if level.supported
    )
    return tuple(candidates), outcomes


def _validated_exclusion(attempt: EunisLabelAttempt) -> str | None:
    if attempt.threshold != EUNIS_THRESHOLD:
        raise ValueError("EUNIS pool assignment uses the fixed 0.8 support threshold")
    if not attempt.inventory_row_id.strip() or not attempt.source_revision.strip():
        raise ValueError("inventory row id and EEA source revision must be present")
    if attempt.query_error is not None and attempt.exclusion_reason is not None:
        raise ValueError("a label attempt cannot have both a query error and geometry exclusion")
    excluded = attempt.query_error or attempt.exclusion_reason
    if excluded is not None:
        return excluded
    if attempt.support_areas is None:
        raise ValueError("a usable polygon label attempt needs all class support areas")
    if any(
        not isinstance(value, (int, float)) or not isfinite(value)
        for value in attempt.support_areas.values()
    ):
        raise TypeError("class support areas must be finite numeric values")
    if (
        attempt.polygon_snapshot_id is None
        or attempt.original_area is None
        or not isfinite(attempt.original_area)
        or attempt.original_area <= 0.0
    ):
        return "usable_geometry_missing_full_area"
    return None


def _make_candidate(
    attempt: EunisLabelAttempt,
    level: str,
    code: str,
    numerator: float,
    denominator: float,
    taxonomy_by_class: Mapping[tuple[str, str], set[str]],
) -> LabelCandidate:
    if (level, code) not in taxonomy_by_class:
        raise ValueError(f"support class {level}/{code} is absent from pinned taxonomy")
    return LabelCandidate(
        inventory_row_id=attempt.inventory_row_id,
        polygon_snapshot_id=attempt.polygon_snapshot_id or "",
        level=level,
        code=code,
        numerator=numerator,
        denominator=denominator,
        ratio=numerator / denominator,
        threshold=attempt.threshold,
        assignment_rule_version=EUNIS_ASSIGNMENT_RULE_VERSION,
        assignment_kind=(
            AssignmentKind.DIRECT if level == "L3" else AssignmentKind.ANCESTOR_DERIVED
        ),
        support_semantics=EUNIS_SUPPORT_SEMANTICS,
        support_unit="m2",
        source_revision=attempt.source_revision,
        taxonomy_revision=attempt.taxonomy.pinned_revision,
    )


def _excluded_outcomes(attempt: EunisLabelAttempt, reason: str) -> tuple[LabelOutcome, ...]:
    return tuple(
        LabelOutcome(
            inventory_row_id=attempt.inventory_row_id,
            polygon_snapshot_id=attempt.polygon_snapshot_id,
            level=level.name,
            status=OutcomeStatus.EXCLUDED,
            reason=reason,
        )
        for level in EUNIS_POOL_LEVELS
        if level.supported
    )


def _resolve_level(
    attempt: EunisLabelAttempt,
    level: LabelLevel,
    candidates: list[LabelCandidate],
) -> LabelOutcome:
    qualifiers = [candidate for candidate in candidates if candidate.ratio >= attempt.threshold]
    if len(qualifiers) == 1:
        return LabelOutcome(
            inventory_row_id=attempt.inventory_row_id,
            polygon_snapshot_id=attempt.polygon_snapshot_id,
            level=level.name,
            status=OutcomeStatus.ASSIGNED,
            assigned_code=qualifiers[0].code,
        )
    if len(qualifiers) > 1:
        return LabelOutcome(
            inventory_row_id=attempt.inventory_row_id,
            polygon_snapshot_id=attempt.polygon_snapshot_id,
            level=level.name,
            status=OutcomeStatus.AMBIGUOUS,
            reason="multiple_classes_reach_80_percent_of_full_polygon_area",
            ambiguity_basis=AmbiguityBasis.MULTIPLE_QUALIFIERS,
        )
    reason = (
        "no_class_reaches_80_percent_of_full_polygon_area"
        if candidates
        else "no_positive_reference_support"
    )
    return LabelOutcome(
        inventory_row_id=attempt.inventory_row_id,
        polygon_snapshot_id=attempt.polygon_snapshot_id,
        level=level.name,
        status=OutcomeStatus.UNASSIGNED,
        reason=reason,
    )


@dataclass(frozen=True, slots=True)
class PolygonDocumentEvidence:
    """A downstream source-native document-to-polygon key and its evidence."""

    document_id: str
    polygon_native_key: NativeKey
    evidence: str


def resolve_polygon_document_links(
    evidence_rows: Iterable[PolygonDocumentEvidence],
    inventory: Iterable[PolygonInventoryRow],
    *,
    source_revision: str,
) -> tuple[TextPolygonLink, ...]:
    """Resolve text links after raw inventory creation without filtering rows."""

    if not source_revision.strip():
        raise ValueError("text-link source revision must be present")
    inventory_by_key: dict[NativeKey, list[PolygonInventoryRow]] = defaultdict(list)
    for row in inventory:
        inventory_by_key[row.native_key].append(row)

    links = []
    seen_document_keys: set[tuple[str, NativeKey]] = set()
    for item in evidence_rows:
        if not item.document_id.strip() or not item.evidence.strip():
            raise ValueError("text-link evidence needs a document id and evidence")
        document_key = (item.document_id, item.polygon_native_key)
        if document_key in seen_document_keys:
            raise ValueError(f"duplicate source link evidence for {document_key}")
        seen_document_keys.add(document_key)
        matches = inventory_by_key.get(item.polygon_native_key, [])
        if len(matches) == 1:
            row = matches[0]
            if row.polygon_snapshot_id is None:
                status = TextLinkStatus.GEOMETRY_MISSING
                inventory_row_id = None
                candidates = (row.inventory_row_id,)
            else:
                status = TextLinkStatus.LINKED
                inventory_row_id = row.inventory_row_id
                candidates = ()
        elif len(matches) > 1:
            status = TextLinkStatus.AMBIGUOUS
            inventory_row_id = None
            candidates = tuple(sorted(row.inventory_row_id for row in matches))
        else:
            status = TextLinkStatus.UNMATCHED
            inventory_row_id = None
            candidates = ()
        links.append(
            TextPolygonLink(
                document_id=item.document_id,
                inventory_row_id=inventory_row_id,
                native_link_key=item.polygon_native_key,
                source_revision=source_revision,
                status=status,
                evidence=item.evidence,
                candidate_inventory_row_ids=candidates,
            )
        )
    return tuple(sorted(links, key=lambda link: (link.document_id, link.inventory_row_id or "")))


def _taxonomy_code_paths(
    taxonomy: EunisTaxonomy,
) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for direct_code, path in taxonomy.paths_by_direct_code.items():
        result[("L1", path.level_1)].add(direct_code)
        result[("L2", path.level_2)].add(direct_code)
        result[("L3", path.level_3)].add(direct_code)
    return result
