"""EUNIS-specific full-pool assignments, independent of text eligibility."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import cast

from .eea import (
    EEA_2021_CLASSIFICATION_LINEAGE_PIN,
    ClassificationTaxonomy,
    ClassificationTaxonomyLineagePin,
    VerifiedClassificationTaxonomy,
)
from .eunis_asset_manifest import (
    EEA_2021_ASSET_BYTES,
    EEA_2021_ASSET_COUNT,
    EEA_2021_MAP_RELEASE,
    EEA_2021_NATIVE_ASSET_INVENTORY_SHA256,
    EEA_2021_NATIVE_MANIFEST_SHA256,
    EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256,
)
from .geometry import OVERLAP_KERNEL_VERSION, WGS84_MAX_SEGMENT_LENGTH_DEGREES
from .pool_contract import (
    AmbiguityBasis,
    AreaMeasurementPin,
    AreaMeasureSpec,
    ArtifactManifestPin,
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
EUNIS_ASSIGNMENT_RULE_VERSION = "eea-highest-source-score-derived-80pct-full-area-v1"
EUNIS_THRESHOLD = 0.8
EUNIS_SUPPORT_SEMANTICS = SupportSemantics.MODELED_SCORE_DERIVED_CLASSIFICATION
EUNIS_SUPPORT_AREA_POLICY = "highest_source_score_per_cell_then_unique_class_area_per_level:v1"
EUNIS_AREA_MEASUREMENT_ID = "eunis-epsg3035-densified-source-boundary-v1"
_EUNIS_DIRECT_MAP_LEVEL = 3
EUNIS_WEBSITE_FIELD_TO_SENTENCE_COLUMN = MappingProxyType(
    {
        "website": "website_sentences",
        "contact_website": "contact_website_sentences",
    }
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GIT_SHA1_LENGTH = 40
_SHA256_LENGTH = 64


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
    classification_lineage: VerifiedClassificationTaxonomy | None = None

    def __post_init__(self) -> None:
        """Validate the explicit mapping and its immutable taxonomy pin."""

        _validate_taxonomy_identity(self.taxonomy_id, self.taxonomy_revision, self.taxonomy_sha256)
        if not self.paths_by_direct_code:
            raise ValueError("EUNIS taxonomy mapping must contain direct map codes")
        for direct_code, path in self.paths_by_direct_code.items():
            _validate_taxonomy_path(direct_code, path)
        frozen_paths = MappingProxyType(dict(self.paths_by_direct_code))
        object.__setattr__(self, "paths_by_direct_code", frozen_paths)
        if self.classification_lineage is not None:
            self._validate_classification_mapping()

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

    @property
    def classification_paths_sha256(self) -> str:
        """Digest the exact emitted L1/L2/L3 paths used by this adapter."""

        payload = json.dumps(
            {
                code: {
                    "L1": path.level_1,
                    "L2": path.level_2,
                    "L3": path.level_3,
                }
                for code, path in sorted(self.paths_by_direct_code.items())
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def classification_lineage_sha256(self) -> str | None:
        """Digest the workbook proof together with the exact derived map paths."""

        proof = self.classification_lineage
        if proof is None:
            return None
        payload = json.dumps(
            {
                "cache_sha256": proof.lineage_pin.cache_sha256,
                "classification_record_id": proof.lineage_pin.classification_record_id,
                "workbook_sha256": proof.lineage_pin.workbook_sha256,
                "taxonomy_mapping_sha256": proof.taxonomy_mapping_sha256,
                "classification_paths_sha256": self.classification_paths_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @property
    def production_eligible(self) -> bool:
        """Whether fixed-pin workbook proof covers this exact EUNIS map path."""

        try:
            self.require_production_lineage()
        except ValueError:
            return False
        return True

    def require_production_lineage(
        self,
        expected: ClassificationTaxonomyLineagePin | None = None,
    ) -> None:
        """Fail closed unless validated workbook evidence proves this exact path map."""

        proof = self.classification_lineage
        if proof is None:
            raise ValueError("production EUNIS taxonomy requires verified workbook lineage")
        expected_pin = expected or EEA_2021_CLASSIFICATION_LINEAGE_PIN
        if proof.lineage_pin != expected_pin:
            raise ValueError("classification workbook/cache lineage does not match the EEA pin")
        if self.taxonomy_sha256 != expected_pin.cache_sha256:
            raise ValueError("EUNIS taxonomy digest does not match the verified EEA cache pin")
        self._validate_classification_mapping()

    def _validate_classification_mapping(self) -> None:
        proof = self.classification_lineage
        if proof is None:
            return
        expected_paths = {
            code: _classification_map_path(proof.taxonomy, code)
            for code in self.paths_by_direct_code
        }
        if expected_paths != dict(self.paths_by_direct_code):
            raise ValueError(
                "EUNIS L1/L2/L3 paths differ from the verified classification hierarchy"
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
    raw_polygon_tree_pin: SourceTreeGroupPin
    additional_tree_pins: tuple[SourceTreeGroupPin, ...] = ()
    document_link_glob: str | None = None
    document_glob: str | None = None
    wikivoyage_document_glob: str | None = None
    sentence_glob: str | None = None


@dataclass(frozen=True, slots=True)
class SourceTreeFile:
    """One file entry from an immutable Hugging Face repository tree."""

    path: str
    size_bytes: int
    blob_id: str
    lfs_oid: str | None
    lfs_size: int

    def __post_init__(self) -> None:
        """Reject incomplete or malformed immutable tree entries."""

        if not self.path.strip() or self.size_bytes < 0 or self.lfs_size < 0:
            raise ValueError("source tree file path and sizes must be valid")
        _validate_tree_blob_id(self.blob_id)
        _validate_tree_lfs_oid(self.lfs_oid)


def _lowercase_hex(value: str) -> bool:
    return all(character in "0123456789abcdef" for character in value)


def _is_sha256(value: str) -> bool:
    return _SHA256_RE.fullmatch(value) is not None


def _validate_taxonomy_identity(taxonomy_id: str, revision: str, digest: str) -> None:
    if not taxonomy_id.strip() or not revision.strip():
        raise ValueError("EUNIS taxonomy id and revision must be present")
    if not _is_sha256(digest):
        raise ValueError("EUNIS taxonomy SHA-256 must be lowercase hexadecimal")


def _validate_taxonomy_path(direct_code: str, path: EunisTaxonomyPath) -> None:
    if not direct_code.strip() or path.level_3 != direct_code:
        raise ValueError("each direct map code must map to itself at EUNIS L3")
    if any(not value.strip() for value in (path.level_1, path.level_2, path.level_3)):
        raise ValueError(f"taxonomy path for {direct_code} contains an empty code")


def _validate_tree_blob_id(blob_id: str) -> None:
    if len(blob_id) != _GIT_SHA1_LENGTH or not _lowercase_hex(blob_id):
        raise ValueError("source tree blob id must be a full lowercase Git SHA-1")


def _validate_tree_lfs_oid(lfs_oid: str | None) -> None:
    if lfs_oid is not None and not _is_sha256(lfs_oid):
        raise ValueError("source tree LFS oid must be a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class SourceTreeGroupPin:
    """Expected immutable inventory for one path group in a pinned repository tree."""

    repository: str
    revision: str
    role: str
    path_prefix: str
    file_count: int
    total_size_bytes: int
    inventory_sha256: str

    def __post_init__(self) -> None:
        """Validate group identity and inventory digest fields."""

        _validate_tree_group_identity(self.repository, self.role, self.path_prefix)
        _validate_tree_group_revision(self.revision)
        _validate_tree_group_totals(self.file_count, self.total_size_bytes)
        _validate_tree_group_hash(self.inventory_sha256)


def _validate_tree_group_identity(repository: str, role: str, path_prefix: str) -> None:
    if not repository.strip() or not role.strip() or not path_prefix:
        raise ValueError("source tree pin requires repository, role, and path prefix")


def _validate_tree_group_revision(revision: str) -> None:
    if len(revision) not in {_GIT_SHA1_LENGTH, _SHA256_LENGTH} or not _lowercase_hex(revision):
        raise ValueError("source tree revision must be a full lowercase Git SHA")


def _validate_tree_group_totals(file_count: int, total_size_bytes: int) -> None:
    if file_count <= 0 or total_size_bytes < 0:
        raise ValueError("source tree file count and byte total must be valid")


def _validate_tree_group_hash(inventory_sha256: str) -> None:
    if not _is_sha256(inventory_sha256):
        raise ValueError("source tree inventory SHA-256 must be lowercase hexadecimal")


def source_tree_inventory_sha256(files: Iterable[SourceTreeFile]) -> str:
    """Hash source tree metadata using the handoff's canonical JSON encoding."""

    ordered = sorted(files, key=lambda row: row.path)
    if len({row.path for row in ordered}) != len(ordered):
        raise ValueError("source tree inventory contains duplicate paths")
    entries = [
        {
            "path": row.path,
            "size": row.size_bytes,
            "blob_id": row.blob_id,
            "lfs_oid": row.lfs_oid,
            "lfs_size": row.lfs_size,
        }
        for row in ordered
    ]
    canonical = json.dumps(entries, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_source_tree_group(
    pin: SourceTreeGroupPin,
    files: Iterable[SourceTreeFile],
) -> None:
    """Require exact file paths, count, size, and metadata digest for one tree group."""

    entries = tuple(files)
    _validate_tree_prefix(entries, pin.path_prefix)
    _validate_tree_group_count_and_size(pin, entries)
    _validate_tree_group_digest(pin, entries)


def _validate_tree_prefix(entries: tuple[SourceTreeFile, ...], prefix: str) -> None:
    if any(not row.path.startswith(prefix) for row in entries):
        raise ValueError(f"source tree entries fall outside pinned prefix {prefix!r}")


def _validate_tree_group_count_and_size(
    pin: SourceTreeGroupPin,
    entries: tuple[SourceTreeFile, ...],
) -> None:
    if len(entries) != pin.file_count:
        raise ValueError(
            f"source tree file count differs for {pin.repository}:{pin.path_prefix}: "
            f"{len(entries)} != {pin.file_count}"
        )
    total_bytes = sum(row.size_bytes for row in entries)
    if total_bytes != pin.total_size_bytes:
        raise ValueError(
            f"source tree byte total differs for {pin.repository}:{pin.path_prefix}: "
            f"{total_bytes} != {pin.total_size_bytes}"
        )


def _validate_tree_group_digest(
    pin: SourceTreeGroupPin,
    entries: tuple[SourceTreeFile, ...],
) -> None:
    digest = source_tree_inventory_sha256(entries)
    if digest != pin.inventory_sha256:
        raise ValueError(
            f"source tree inventory digest differs for {pin.repository}:{pin.path_prefix}"
        )


EUNIS_POOL_SOURCE_LAYOUTS: Mapping[TextSource, EunisPoolSourceLayout] = {
    TextSource.DESCRIPTION: EunisPoolSourceLayout(
        text_source=TextSource.DESCRIPTION,
        repository="NoeFlandre/osm-polygon-description-tag",
        polygon_glob="data/*.parquet",
        raw_revision="b4706eb315c66a8a57135289e7612dca8e03caf8",
        relevance_repository="NoeFlandre/osm-polygon-description-tag-landuse",
        relevance_revision="ce98b328f3c6543800a3d931877b51eb3f0cf768",
        relevance_path_globs=(("description", "labels/language-v1/data/*.parquet"),),
        raw_polygon_tree_pin=SourceTreeGroupPin(
            "NoeFlandre/osm-polygon-description-tag",
            "b4706eb315c66a8a57135289e7612dca8e03caf8",
            "raw_geometry",
            "data/",
            386,
            746_727_545,
            "cfedbc2235eecda19e52527dc45efee043af5909ac483976db8b844dd4a0f599",
        ),
        additional_tree_pins=(
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-description-tag",
                "b4706eb315c66a8a57135289e7612dca8e03caf8",
                "raw_text",
                "language-v1/data/",
                386,
                103_449_770,
                "d76a3a42708307d1932c55c24499f39c2fd48c5fec712d15f0b65b0cb6c60f53",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-description-tag-landuse",
                "ce98b328f3c6543800a3d931877b51eb3f0cf768",
                "relevance_labels",
                "labels/language-v1/data/",
                386,
                100_444_539,
                "79ef71e658e6ac666a2a23ccdd3746220b525e0bb9ef6294ab8ceda9767a7768",
            ),
        ),
    ),
    TextSource.WEBSITE: EunisPoolSourceLayout(
        text_source=TextSource.WEBSITE,
        repository="NoeFlandre/osm-polygon-website-tag",
        polygon_glob="polygons/*.parquet",
        raw_revision="4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
        relevance_repository="NoeFlandre/osm-polygon-website-tag-landuse",
        relevance_revision="e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
        relevance_path_globs=(("website", "labels/polygons/*.parquet"),),
        raw_polygon_tree_pin=SourceTreeGroupPin(
            "NoeFlandre/osm-polygon-website-tag",
            "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
            "raw_geometry",
            "polygons/",
            386,
            3_943_775_497,
            "5ef66bc56532914428d800a1671bd1fd1b33cf54966617197e6e1c548788ca9d",
        ),
        additional_tree_pins=(
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-website-tag-landuse",
                "e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
                "relevance_labels",
                "labels/polygons/",
                386,
                3_299_123_482,
                "0a2c112b0d4198fddaf0c20153f0f1fc8f0f661c265446f741da787fa867ee96",
            ),
        ),
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
        raw_polygon_tree_pin=SourceTreeGroupPin(
            "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
            "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
            "raw_geometry",
            "polygons/",
            386,
            2_203_317_876,
            "1628f43bf1d42041d0e41b18156abbb1f52420fd7f138027ad4b96fcfa0c9dd6",
        ),
        additional_tree_pins=(
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wikipedia_text",
                "wikipedia/sentences/",
                386,
                18_075_023_818,
                "79f54e06708b5755f3cf09c59890c6504669095c3529b690016c09aef04df6c8",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wikivoyage_text",
                "wikivoyage/sentences/",
                375,
                373_112_864,
                "c407eb5b70cada37fde70a2b92882ddad301e954f9c2a28713443663ab9fefa7",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
                "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
                "wikipedia_relevance_labels",
                "labels/wikipedia/sentences/",
                386,
                8_191_284_820,
                "310214740d4c0d10b4228583992c0e0842df680ef68996f8cc019f9cdcb4268f",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
                "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
                "wikivoyage_relevance_labels",
                "labels/wikivoyage/sentences/",
                375,
                189_959_172,
                "3cdd0e4f0fa8b6588e578ff077467a3e4516e676d7a4abd30cb622477cf8245d",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wiki_document_links",
                "polygon_document_links/",
                386,
                112_581_825,
                "4d3bcf32094d8c7a42d72f6a52da61b7b6940b9feae564fe10622716f0b48cde",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wiki_documents",
                "wikipedia/documents/",
                386,
                8_739_038_060,
                "780c9f7c187b08693e78c6aa37d2df005064fdc8d3d8472b6941b4f1c5f91e34",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wiki_documents",
                "wikivoyage/documents/",
                375,
                80_078_157,
                "875520ce84370f18c1551c9f78cc3fdaba1f48932cb809e2df9a245988722d8d",
            ),
            SourceTreeGroupPin(
                "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
                "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
                "wiki_language_split_links",
                "language_splits/polygon_document_links_by_language/",
                356,
                154_309_530,
                "7c1eb4caae9eb84a89db678bc7e2cbccfd5450ac808e89330c439c9a67b20c6a",
            ),
        ),
        document_link_glob="polygon_document_links/*.parquet",
        document_glob="wikipedia/documents/*.parquet",
        wikivoyage_document_glob="wikivoyage/documents/*.parquet",
        sentence_glob="wikivoyage/sentences/*.parquet",
    ),
}


def hierarchy_support_groups(
    taxonomy: EunisTaxonomy,
) -> dict[tuple[str, str], tuple[str, ...]]:
    """Expose map groups for direct L3 plus independently rolled-up L1/L2."""

    return taxonomy.support_groups()


def make_eunis_taxonomy_from_classification(
    classification: ClassificationTaxonomy | VerifiedClassificationTaxonomy,
    direct_codes: Iterable[str],
    *,
    taxonomy_id: str,
    taxonomy_revision: str,
    taxonomy_sha256: str,
) -> EunisTaxonomy:
    """Build explicit map paths; only byte-verified input is production eligible."""

    if isinstance(classification, VerifiedClassificationTaxonomy):
        lineage = classification
        hierarchy = classification.taxonomy
    else:
        lineage = None
        hierarchy = classification
    codes = tuple(sorted(set(direct_codes)))
    if not codes:
        raise ValueError("EEA direct map codes must be present")
    paths = {code: _classification_map_path(hierarchy, code) for code in codes}
    return EunisTaxonomy(
        taxonomy_id=taxonomy_id,
        taxonomy_revision=taxonomy_revision,
        taxonomy_sha256=taxonomy_sha256,
        paths_by_direct_code=paths,
        classification_lineage=lineage,
    )


def _classification_map_path(
    classification: ClassificationTaxonomy,
    code: str,
) -> EunisTaxonomyPath:
    if not _is_direct_map_code(classification, code):
        raise ValueError(f"EEA direct map code {code} is not a classification L3 code")
    by_level = _classification_ancestor_levels(classification, code)
    _validate_l123_map_path(code, by_level)
    return EunisTaxonomyPath(by_level[1], by_level[2], by_level[3])


def _is_direct_map_code(classification: ClassificationTaxonomy, code: str) -> bool:
    return classification.levels.get(code) == _EUNIS_DIRECT_MAP_LEVEL


def _classification_ancestor_levels(
    classification: ClassificationTaxonomy,
    code: str,
) -> dict[int, str]:
    ancestors_by_level: dict[int, str] = {}
    for ancestor in classification.ancestors_including_self(code):
        level = _supported_classification_ancestor_level(classification, ancestor)
        if level is not None:
            ancestors_by_level[level] = ancestor
    return ancestors_by_level


def _supported_classification_ancestor_level(
    classification: ClassificationTaxonomy,
    ancestor: str,
) -> int | None:
    level = classification.levels.get(ancestor)
    if level in {1, 2, 3}:
        return level
    return None


def _validate_l123_map_path(code: str, ancestors_by_level: Mapping[int, str]) -> None:
    if set(ancestors_by_level) != {1, 2, 3}:
        raise ValueError(f"EEA direct map code {code} lacks one L1-L2-L3 parent path")
    if ancestors_by_level[3] != code:
        raise ValueError(f"EEA direct map code {code} lacks one L1-L2-L3 parent path")


def make_eunis_2021_reference_pin(
    taxonomy: EunisTaxonomy,
    *,
    reference_artifact_manifest: ArtifactManifestPin | None = None,
) -> ReferencePin:
    """Build the checked EEA v1 2021 pin from verified manifest digests."""

    if not taxonomy.production_eligible:
        taxonomy.require_production_lineage()
    output_semantics_digest = EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256 or "unverified"
    return ReferencePin(
        release_id=EEA_2021_MAP_RELEASE,
        source_repository="https://biodiversity.europa.eu/resources/search-habitat",
        source_revision=(
            f"native-manifest-sha256:{EEA_2021_NATIVE_MANIFEST_SHA256};"
            f"asset-inventory-sha256:{EEA_2021_NATIVE_ASSET_INVENTORY_SHA256};"
            f"source-output-semantics-evidence-sha256:{output_semantics_digest};"
            f"asset-count:{EEA_2021_ASSET_COUNT};asset-bytes:{EEA_2021_ASSET_BYTES};"
            "crs:EPSG:3035;score-range:0..1;identity-scale-offset;"
            "rule:highest-source-score-derived-classification-v1"
        ),
        taxonomy_id=taxonomy.taxonomy_id,
        taxonomy_revision=taxonomy.pinned_revision,
        support_semantics=EUNIS_SUPPORT_SEMANTICS,
        support_unit="m2",
        threshold=EUNIS_THRESHOLD,
        support_union_policy=EUNIS_SUPPORT_AREA_POLICY,
        reference_artifact_manifest=reference_artifact_manifest,
        area_measurement=eunis_area_measurement_pin(),
    )


def eunis_area_measurement_pin() -> AreaMeasurementPin:
    """Describe the EPSG:3035 area estimate and disclose its uncertified error."""

    parameters = json.dumps(
        {
            "wgs84_max_segment_length_degrees": WGS84_MAX_SEGMENT_LENGTH_DEGREES,
            "projected_crs": "EPSG:3035",
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    errors = json.dumps(
        {
            "total_error_bound": "not_certified",
            "unresolved_query_policy": "fail_closed",
            "excluded_error_sources": [],
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    common = {
        "method_id": EUNIS_AREA_MEASUREMENT_ID,
        "coordinate_system": "EPSG:3035",
        "ellipsoid": "WGS84",
        "densification_method": (
            "segmentize straight WGS84 source edges before EPSG:3035 projection"
        ),
        "densification_parameters_json": parameters,
        "error_settings_json": errors,
        "output_unit": "m2",
    }
    numerator = AreaMeasureSpec(
        **common,
        boundary_convention=(
            "projected class winner cells intersect the unchanged projected source polygon"
        ),
    )
    denominator = AreaMeasureSpec(
        **common,
        boundary_convention=(
            "full projected source polygon area after the pinned WGS84 edge densification"
        ),
    )
    return AreaMeasurementPin(
        implementation="osm-polygon-eunis weighted raster cells and Shapely intersections",
        implementation_version=(
            f"geometry-policy-v{OVERLAP_KERNEL_VERSION};"
            f"maximum-segment-degrees:{WGS84_MAX_SEGMENT_LENGTH_DEGREES}"
        ),
        numerator=numerator,
        denominator=denominator,
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
    score_diagnostics: Mapping[str, object] | None = None
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
    denominator = cast(float, attempt.original_area)
    candidates, by_level = _positive_support_candidates(attempt, denominator)

    outcomes = tuple(
        _resolve_level(attempt, level, by_level[level.name])
        for level in EUNIS_POOL_LEVELS
        if level.supported
    )
    return tuple(candidates), outcomes


def _positive_support_candidates(
    attempt: EunisLabelAttempt,
    denominator: float,
) -> tuple[list[LabelCandidate], dict[str, list[LabelCandidate]]]:
    support_areas = cast(Mapping[tuple[str, str], float], attempt.support_areas)
    taxonomy_by_class = _taxonomy_code_paths(attempt.taxonomy)
    candidates: list[LabelCandidate] = []
    by_level: dict[str, list[LabelCandidate]] = {"L1": [], "L2": [], "L3": []}
    for (level, code), raw_numerator in sorted(support_areas.items()):
        candidate = _positive_support_candidate(
            attempt,
            level,
            code,
            raw_numerator,
            denominator,
            taxonomy_by_class,
        )
        if candidate is None:
            continue
        candidates.append(candidate)
        by_level[level].append(candidate)
    return candidates, by_level


def _positive_support_candidate(
    attempt: EunisLabelAttempt,
    level: str,
    code: str,
    raw_numerator: float,
    denominator: float,
    taxonomy_by_class: Mapping[tuple[str, str], set[str]],
) -> LabelCandidate | None:
    if level not in {"L1", "L2", "L3"}:
        raise ValueError(f"unsupported EUNIS support level {level!r}")
    numerator = float(raw_numerator)
    if numerator < 0.0 or numerator > denominator:
        raise ValueError(f"support for {level}/{code} falls outside full polygon area")
    if numerator == 0.0:
        return None
    return _make_candidate(attempt, level, code, numerator, denominator, taxonomy_by_class)


def _validated_exclusion(attempt: EunisLabelAttempt) -> str | None:
    excluded = _terminal_attempt_exclusion(attempt)
    if excluded is not None:
        return excluded
    _validate_attempt_support_map(attempt.support_areas)
    return _unusable_geometry_reason(attempt)


def _terminal_attempt_exclusion(attempt: EunisLabelAttempt) -> str | None:
    _validate_attempt_threshold(attempt)
    _validate_attempt_identity(attempt)
    _validate_terminal_error_exclusivity(attempt)
    return _attempt_exclusion_reason(attempt)


def _validate_attempt_threshold(attempt: EunisLabelAttempt) -> None:
    if attempt.threshold != EUNIS_THRESHOLD:
        raise ValueError("EUNIS pool assignment uses the fixed 0.8 support threshold")


def _validate_attempt_identity(attempt: EunisLabelAttempt) -> None:
    if not attempt.inventory_row_id.strip() or not attempt.source_revision.strip():
        raise ValueError("inventory row id and EEA source revision must be present")


def _validate_terminal_error_exclusivity(attempt: EunisLabelAttempt) -> None:
    if attempt.query_error is not None and attempt.exclusion_reason is not None:
        raise ValueError("a label attempt cannot have both a query error and geometry exclusion")


def _attempt_exclusion_reason(attempt: EunisLabelAttempt) -> str | None:
    return attempt.query_error or attempt.exclusion_reason


def _validate_attempt_support_map(
    support_areas: Mapping[tuple[str, str], float] | None,
) -> None:
    if support_areas is None:
        raise ValueError("a usable polygon label attempt needs all class support areas")
    if any(
        not isinstance(value, (int, float)) or not isfinite(value)
        for value in support_areas.values()
    ):
        raise TypeError("class support areas must be finite numeric values")


def _unusable_geometry_reason(attempt: EunisLabelAttempt) -> str | None:
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
    return LabelOutcome(
        inventory_row_id=attempt.inventory_row_id,
        polygon_snapshot_id=attempt.polygon_snapshot_id,
        level=level.name,
        status=OutcomeStatus.UNASSIGNED,
        reason=_unassigned_level_reason(candidates, attempt.score_diagnostics, level.name),
    )


def _unassigned_level_reason(
    candidates: list[LabelCandidate],
    diagnostics: Mapping[str, object] | None,
    level: str,
) -> str:
    uncertainty = _has_score_uncertainty(diagnostics, level)
    if candidates:
        return (
            "no_class_reaches_80_percent_with_unknown_or_tied_source_scores"
            if uncertainty
            else "no_class_reaches_80_percent_of_full_polygon_area"
        )
    return (
        "unknown_or_tied_source_scores_prevent_unique_classification"
        if uncertainty
        else "no_positive_source_score"
    )


def _has_score_uncertainty(diagnostics: Mapping[str, object] | None, level: str) -> bool:
    if diagnostics is None:
        return False
    ambiguous = diagnostics.get("ambiguous_area_m2_by_level", {})
    level_ambiguous = ambiguous.get(level, 0.0) if isinstance(ambiguous, Mapping) else 0.0
    return _is_positive_area(diagnostics.get("unknown_area_m2", 0.0)) or _is_positive_area(
        level_ambiguous
    )


def _is_positive_area(value: object) -> bool:
    return isinstance(value, (int, float)) and value > 0.0


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

    seen_document_keys: set[tuple[str, NativeKey]] = set()
    links = [
        _document_link(item, inventory_by_key, source_revision, seen_document_keys)
        for item in evidence_rows
    ]
    return tuple(sorted(links, key=lambda link: (link.document_id, link.inventory_row_id or "")))


def _document_link(
    item: PolygonDocumentEvidence,
    inventory_by_key: Mapping[NativeKey, list[PolygonInventoryRow]],
    source_revision: str,
    seen_document_keys: set[tuple[str, NativeKey]],
) -> TextPolygonLink:
    if not item.document_id.strip() or not item.evidence.strip():
        raise ValueError("text-link evidence needs a document id and evidence")
    document_key = (item.document_id, item.polygon_native_key)
    if document_key in seen_document_keys:
        raise ValueError(f"duplicate source link evidence for {document_key}")
    seen_document_keys.add(document_key)
    status, inventory_row_id, candidates = _link_match_result(
        inventory_by_key.get(item.polygon_native_key, [])
    )
    return TextPolygonLink(
        document_id=item.document_id,
        inventory_row_id=inventory_row_id,
        native_link_key=item.polygon_native_key,
        source_revision=source_revision,
        status=status,
        evidence=item.evidence,
        candidate_inventory_row_ids=candidates,
    )


def _link_match_result(
    matches: list[PolygonInventoryRow],
) -> tuple[TextLinkStatus, str | None, tuple[str, ...]]:
    if len(matches) == 1:
        return _single_inventory_link(matches[0])
    if len(matches) > 1:
        candidates = tuple(sorted(row.inventory_row_id for row in matches))
        return TextLinkStatus.AMBIGUOUS, None, candidates
    return TextLinkStatus.UNMATCHED, None, ()


def _single_inventory_link(
    row: PolygonInventoryRow,
) -> tuple[TextLinkStatus, str | None, tuple[str, ...]]:
    if row.polygon_snapshot_id is None:
        return TextLinkStatus.GEOMETRY_MISSING, None, (row.inventory_row_id,)
    return TextLinkStatus.LINKED, row.inventory_row_id, ()


def _taxonomy_code_paths(
    taxonomy: EunisTaxonomy,
) -> dict[tuple[str, str], set[str]]:
    result: dict[tuple[str, str], set[str]] = defaultdict(set)
    for direct_code, path in taxonomy.paths_by_direct_code.items():
        result[("L1", path.level_1)].add(direct_code)
        result[("L2", path.level_2)].add(direct_code)
        result[("L3", path.level_3)].add(direct_code)
    return result
