"""Independent contract tests for full polygon-label pools."""

from __future__ import annotations

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import osm_polygon_eunis.pool_contract.streaming as streaming_module
from osm_polygon_eunis.pool_contract import (
    AmbiguityBasis,
    AreaMeasurementPin,
    AreaMeasureSpec,
    ArtifactManifestPin,
    AssignmentKind,
    AttemptStatus,
    BodyStatus,
    DataSplit,
    EligibilityStatus,
    EntityLink,
    EntityReference,
    EvaluationArm,
    EvaluationCase,
    EvaluationPayload,
    EvaluationTrace,
    ExportReceipt,
    FilesystemArtifactReader,
    FilesystemSink,
    FilesystemStreamingSink,
    GeometryStatus,
    InvalidPoolError,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    LabelSupportEvidence,
    OutcomeStatus,
    PolygonAttemptReceipt,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBatch,
    PoolBuildInput,
    PoolEligibility,
    PoolKey,
    PoolStreamError,
    PoolStreamSpec,
    RawInventoryEvidence,
    ReferencePin,
    ReferenceSystem,
    RelevanceAssessment,
    RelevanceStatus,
    SourceDocument,
    SourceSnapshot,
    SplitAssignment,
    StreamingLimits,
    SupportSemantics,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
    build_full_pool,
    canonical_native_key,
    export_full_pool,
    export_pool_batches,
    inventory_row_identifier,
    native_keyset_sha256,
    polygon_snapshot_identifier,
    validate_evaluation_case,
    validate_evaluation_trace,
    validate_full_pool,
    validate_nine_pool_gate,
    validate_nine_streamed_evaluation_ready_gate,
    validate_nine_streamed_pool_gate,
    validate_split_assignment,
)

_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _support_semantics(reference_system: ReferenceSystem) -> SupportSemantics:
    return {
        ReferenceSystem.WORLDCOVER: SupportSemantics.MAPPED_COVER,
        ReferenceSystem.LUCAS: SupportSemantics.SURVEYED_FOOTPRINT,
        ReferenceSystem.EUNIS: SupportSemantics.MODELED_HABITAT_SUPPORT,
    }[reference_system]


def _pool_input(
    text_source: TextSource = TextSource.WEBSITE,
    reference_system: ReferenceSystem = ReferenceSystem.EUNIS,
    *,
    ambiguity: bool = False,
) -> PoolBuildInput:
    source = SourceSnapshot(
        snapshot_id=f"raw-{text_source.value}",
        repository="NoeFlandre/source-polygons",
        revision="source-revision-1",
        pbf_uri="https://artifacts.example/source.osm.pbf",
        pbf_sha256=_SHA_A,
    )
    pin = ReferencePin(
        release_id=f"{reference_system.value}-release-1",
        source_repository="NoeFlandre/reference-labels",
        source_revision="reference-revision-1",
        taxonomy_id=f"{reference_system.value}-taxonomy",
        taxonomy_revision="taxonomy-revision-1",
        support_semantics=_support_semantics(reference_system),
        support_unit="m2",
        threshold=0.8,
        support_union_policy="union_by_class",
    )
    raw_keys = (
        canonical_native_key({"source_pbf": "source.osm.pbf", "osm_type": "way", "osm_id": 10}),
        canonical_native_key({"source_pbf": "source.osm.pbf", "osm_type": "way", "osm_id": 11}),
        canonical_native_key(
            {"source_pbf": "source.osm.pbf", "osm_type": "relation", "osm_id": 12}
        ),
    )
    inventory_ids = tuple(inventory_row_identifier(source.snapshot_id, key) for key in raw_keys)
    inventory_evidence = RawInventoryEvidence(
        enumeration_rule="all raw polygon rows before joining text or relevance labels",
        scan_report_uri="https://artifacts.example/raw-inventory-scan.json",
        scan_report_sha256=_SHA_A,
        expected_key_count=len(raw_keys),
        expected_keyset_sha256=native_keyset_sha256(raw_keys),
    )
    polygon_ids = (
        polygon_snapshot_identifier(source.snapshot_id, raw_keys[0], _SHA_B),
        None,
        polygon_snapshot_identifier(source.snapshot_id, raw_keys[2], "c" * 64),
    )
    inventory = tuple(
        PolygonInventoryRow(
            inventory_row_id=inventory_ids[index],
            native_key=raw_keys[index],
            source_row_locator=f"polygons/region-{index}.parquet#row=0",
            polygon_snapshot_id=polygon_ids[index],
            exclusion_reason="geometry_artifact_missing" if index == 1 else None,
        )
        for index in range(3)
    )
    polygons = (
        PolygonSnapshot(
            polygon_snapshot_id=polygon_ids[0],
            inventory_row_id=inventory_ids[0],
            native_key=raw_keys[0],
            geometry_uri="https://artifacts.example/geometries/10.wkb",
            geometry_sha256=_SHA_B,
            geometry_format="WKB",
            crs="EPSG:3035",
            coordinate_transform="EPSG:4326 -> EPSG:3035",
            geometry_status=GeometryStatus.USABLE,
            original_area=100.0,
            area_unit="m2",
        ),
        PolygonSnapshot(
            polygon_snapshot_id=polygon_ids[2],
            inventory_row_id=inventory_ids[2],
            native_key=raw_keys[2],
            geometry_uri="https://artifacts.example/geometries/12.wkb",
            geometry_sha256="c" * 64,
            geometry_format="WKB",
            crs="EPSG:3035",
            coordinate_transform="EPSG:4326 -> EPSG:3035",
            geometry_status=GeometryStatus.INVALID,
            original_area=None,
            area_unit="m2",
        ),
    )
    levels = (
        LabelLevel("L1", supported=True),
        LabelLevel("L2", supported=True),
        LabelLevel("L4", supported=False, unsupported_reason="no finer reference map evidence"),
    )
    candidates = [
        LabelCandidate(
            inventory_row_id=inventory_ids[0],
            polygon_snapshot_id=polygon_ids[0],
            level="L1",
            code="A",
            numerator=80.0,
            denominator=100.0,
            ratio=0.8,
            threshold=0.8,
            assignment_rule_version="full-area-80-v1",
            assignment_kind=AssignmentKind.ANCESTOR_DERIVED,
            support_semantics=pin.support_semantics,
            support_unit="m2",
            source_revision=pin.source_revision,
            taxonomy_revision=pin.taxonomy_revision,
        ),
        LabelCandidate(
            inventory_row_id=inventory_ids[0],
            polygon_snapshot_id=polygon_ids[0],
            level="L2",
            code="A1",
            numerator=80.0,
            denominator=100.0,
            ratio=0.8,
            threshold=0.8,
            assignment_rule_version="full-area-80-v1",
            assignment_kind=AssignmentKind.DIRECT,
            support_semantics=pin.support_semantics,
            support_unit="m2",
            source_revision=pin.source_revision,
            taxonomy_revision=pin.taxonomy_revision,
        ),
    ]
    if ambiguity:
        candidates.append(replace(candidates[0], code="B", numerator=90.0, ratio=0.9))
    outcomes = []
    for index, inventory_id in enumerate(inventory_ids):
        for level in ("L1", "L2"):
            if index == 0 and level == "L1" and ambiguity:
                outcomes.append(
                    LabelOutcome(
                        inventory_row_id=inventory_id,
                        polygon_snapshot_id=polygon_ids[index],
                        level=level,
                        status=OutcomeStatus.AMBIGUOUS,
                        reason="two classes meet the threshold",
                        ambiguity_basis=AmbiguityBasis.MULTIPLE_QUALIFIERS,
                    )
                )
            elif index == 0:
                outcomes.append(
                    LabelOutcome(
                        inventory_row_id=inventory_id,
                        polygon_snapshot_id=polygon_ids[index],
                        level=level,
                        status=OutcomeStatus.ASSIGNED,
                        assigned_code="A" if level == "L1" else "A1",
                    )
                )
            else:
                outcomes.append(
                    LabelOutcome(
                        inventory_row_id=inventory_id,
                        polygon_snapshot_id=polygon_ids[index],
                        level=level,
                        status=OutcomeStatus.EXCLUDED,
                        reason="geometry_artifact_missing" if index == 1 else "invalid_geometry",
                    )
                )
    documents = (
        SourceDocument(
            document_id="document-private",
            native_key=canonical_native_key(
                {"polygon_id": "poly-10", "field": "website", "sentence_index": 0}
            ),
            source_revision="text-revision-1",
            content_sha256=None,
            body_status=BodyStatus.PRIVATE,
        ),
        SourceDocument(
            document_id="document-missing",
            native_key=canonical_native_key(
                {"polygon_id": "poly-11", "field": "website", "sentence_index": 1}
            ),
            source_revision="text-revision-1",
            content_sha256=None,
            body_status=BodyStatus.MISSING,
        ),
    )
    links = (
        TextPolygonLink(
            document_id="document-private",
            inventory_row_id=inventory_ids[0],
            native_link_key=canonical_native_key(
                {"polygon_id": "poly-10", "field": "website", "sentence_index": 0}
            ),
            source_revision="text-revision-1",
            status=TextLinkStatus.LINKED,
            evidence="exact source-native polygon_id crosswalk",
        ),
        TextPolygonLink(
            document_id="document-missing",
            inventory_row_id=None,
            native_link_key=canonical_native_key(
                {"polygon_id": "poly-11", "field": "website", "sentence_index": 1}
            ),
            source_revision="text-revision-1",
            status=TextLinkStatus.AMBIGUOUS,
            evidence="two candidate geometry rows",
            candidate_inventory_row_ids=(inventory_ids[0], inventory_ids[1]),
        ),
    )
    entity_links = (
        EntityLink(
            link_version="entity-links-v1",
            left=EntityReference(source.snapshot_id, raw_keys[0]),
            right=EntityReference(
                "raw-second-snapshot",
                canonical_native_key(
                    {"source_pbf": "next.osm.pbf", "osm_type": "way", "osm_id": 210}
                ),
            ),
            evidence="versioned source crosswalk row 7",
        ),
    )
    eligibility = (
        PoolEligibility(
            inventory_row_id=inventory_ids[0],
            cohort_policy_version="eval-candidates-v1",
            status=EligibilityStatus.ELIGIBLE,
            evidence="source document and context available",
        ),
        PoolEligibility(
            inventory_row_id=inventory_ids[1],
            cohort_policy_version="eval-candidates-v1",
            status=EligibilityStatus.EXCLUDED,
            evidence="candidate-stage check",
            reason="no evaluation context",
        ),
    )
    attempt_receipts = tuple(
        PolygonAttemptReceipt(
            inventory_row_id=row.inventory_row_id,
            status=(
                AttemptStatus.COMPLETED
                if row.inventory_row_id == inventory_ids[0]
                else AttemptStatus.EXCLUDED
            ),
            reference_id_namespace="fixture/reference-feature-id",
            attempted_reference_ids=(
                ("feature-A", "feature-A1") if row.inventory_row_id == inventory_ids[0] else ()
            ),
            reason=(
                None
                if row.inventory_row_id == inventory_ids[0]
                else "geometry_artifact_missing"
                if row.inventory_row_id == inventory_ids[1]
                else "invalid_geometry"
            ),
        )
        for row in inventory
    )
    support_evidence = tuple(
        LabelSupportEvidence(
            inventory_row_id=row.inventory_row_id,
            level=row.level,
            code=row.code,
            reference_id_namespace="fixture/reference-feature-id",
            reference_ids=(f"feature-{row.code}",),
        )
        for row in candidates
    )
    relevance_assessments = (
        RelevanceAssessment(
            document_id="document-private",
            release_id="relevance-release-1",
            status=RelevanceStatus.YES,
            source_repository="NoeFlandre/website-relevance",
            source_revision="relevance-revision-1",
            mapping_rule_version="relevance-map-v1",
            evidence="native relevance record yes",
            raw_value="yes",
        ),
        RelevanceAssessment(
            document_id="document-missing",
            release_id="relevance-release-1",
            status=RelevanceStatus.MISSING,
            source_repository="NoeFlandre/website-relevance",
            source_revision="relevance-revision-1",
            mapping_rule_version="relevance-map-v1",
            evidence="no relevance row for source document",
            reason="source relevance record absent",
        ),
    )
    return PoolBuildInput(
        pool_key=PoolKey(text_source, reference_system),
        source_snapshot=source,
        raw_inventory_evidence=inventory_evidence,
        reference_pin=pin,
        adapter_id=f"{reference_system.value}-adapter",
        adapter_revision="adapter-revision-1",
        builder_revision="pool-contract-commit-1",
        inventory=inventory,
        polygons=polygons,
        levels=levels,
        label_candidates=tuple(candidates),
        label_outcomes=tuple(outcomes),
        documents=documents,
        text_polygon_links=links,
        entity_links=entity_links,
        eligibility=eligibility,
        attempt_receipts=attempt_receipts,
        support_evidence=support_evidence,
        relevance_assessments=relevance_assessments,
    )


def _build(input_data: PoolBuildInput):
    return build_full_pool(input_data)


def _with_unicode_reference_ids(source: PoolBuildInput) -> PoolBuildInput:
    return replace(
        source,
        attempt_receipts=(
            replace(
                source.attempt_receipts[0],
                attempted_reference_ids=("feature-Å/東京", "feature-A1"),
                missing_reference_ids=("missing-ñ",),
            ),
            *source.attempt_receipts[1:],
        ),
        support_evidence=(
            replace(source.support_evidence[0], reference_ids=("reference-é/東京",)),
            *source.support_evidence[1:],
        ),
    )


def _file_uri_path(uri: str) -> Path:
    return Path(unquote(urlparse(uri).path))


def test_full_pool_keeps_raw_keys_without_text_or_usable_geometry():
    bundle = _build(_pool_input())

    report = validate_full_pool(bundle)

    assert report.ok, report.issues
    assert len(bundle.inventory) == 3
    assert len(bundle.documents) == 2
    assert len(bundle.label_outcomes) == 6
    assert bundle.manifest.total_source_keys == 3
    supported = [row for row in bundle.manifest.level_accounting if row.supported]
    assert [row.processed_decisions for row in supported] == [3, 3]
    assert [row.excluded for row in supported] == [2, 2]
    assert bundle.manifest.level_accounting[-1].processed_decisions == 0


def test_full_pool_allows_non_pbf_snapshot_with_source_tree_pin():
    source = _pool_input()
    source_snapshot = replace(
        source.source_snapshot,
        pbf_uri=None,
        pbf_sha256=None,
        source_artifact_manifest=ArtifactManifestPin(
            format_version="artifact-list-jsonl-v1",
            manifest_uri="https://artifacts.example/source-files.jsonl",
            manifest_sha256=_SHA_A,
            expected_artifact_count=2,
        ),
    )

    bundle = build_full_pool(replace(source, source_snapshot=source_snapshot))

    assert validate_full_pool(bundle).ok


def test_assignments_are_independent_at_each_level_and_80_percent_is_inclusive():
    bundle = _build(_pool_input())
    assignments = {
        outcome.level: outcome.assigned_code
        for outcome in bundle.label_outcomes
        if outcome.status is OutcomeStatus.ASSIGNED
    }
    candidates = {candidate.level: candidate for candidate in bundle.label_candidates}

    assert assignments == {"L1": "A", "L2": "A1"}
    assert candidates["L1"].ratio == 0.8
    assert candidates["L1"].assignment_kind is AssignmentKind.ANCESTOR_DERIVED
    assert candidates["L2"].assignment_kind is AssignmentKind.DIRECT
    assert candidates["L1"].denominator == 100.0


def test_no_data_does_not_shrink_the_full_original_area_denominator():
    input_data = _pool_input()
    first = input_data.label_candidates[0]
    candidate_with_covered_area_as_denominator = replace(
        first,
        denominator=80.0,
        ratio=1.0,
    )
    incorrect = replace(
        input_data,
        label_candidates=(
            candidate_with_covered_area_as_denominator,
            *input_data.label_candidates[1:],
        ),
    )

    report = validate_full_pool(build_full_pool(incorrect))

    assert not report.ok
    assert any("denominator is not full original polygon area" in issue for issue in report.issues)


def test_support_below_threshold_requires_an_unassigned_decision():
    input_data = _pool_input()
    first = replace(input_data.label_candidates[0], numerator=79.0, ratio=0.79)
    outcomes = tuple(
        replace(
            outcome,
            status=OutcomeStatus.UNASSIGNED,
            assigned_code=None,
            reason="below_threshold",
        )
        if outcome.level == "L1" and outcome.status is OutcomeStatus.ASSIGNED
        else outcome
        for outcome in input_data.label_outcomes
    )
    below_threshold = replace(
        input_data,
        label_candidates=(first, *input_data.label_candidates[1:]),
        label_outcomes=outcomes,
    )

    report = validate_full_pool(build_full_pool(below_threshold))

    assert report.ok, report.issues


def test_competing_qualifying_codes_are_ambiguous_not_assigned():
    bundle = _build(_pool_input(ambiguity=True))
    report = validate_full_pool(bundle)

    outcome = next(
        row
        for row in bundle.label_outcomes
        if row.level == "L1" and row.status is OutcomeStatus.AMBIGUOUS
    )
    assert report.ok, report.issues
    assert outcome.status is OutcomeStatus.AMBIGUOUS
    assert outcome.assigned_code is None
    assert outcome.ambiguity_basis is AmbiguityBasis.MULTIPLE_QUALIFIERS


def test_manifest_accounting_must_cover_every_supported_level():
    bundle = _build(_pool_input())
    incomplete = replace(bundle, label_outcomes=bundle.label_outcomes[:-1])

    report = validate_full_pool(incomplete)

    assert not report.ok
    assert any("decision coverage mismatch" in issue for issue in report.issues)
    assert any("per-level accounting" in issue for issue in report.issues)


def test_raw_inventory_evidence_must_match_count_and_full_composite_key_set():
    input_data = _pool_input()
    wrong_count = replace(
        input_data,
        raw_inventory_evidence=replace(
            input_data.raw_inventory_evidence,
            expected_key_count=len(input_data.inventory) - 1,
        ),
    )
    wrong_keyset = replace(
        input_data,
        raw_inventory_evidence=replace(
            input_data.raw_inventory_evidence,
            expected_keyset_sha256="d" * 64,
        ),
    )

    count_report = validate_full_pool(build_full_pool(wrong_count))
    keyset_report = validate_full_pool(build_full_pool(wrong_keyset))

    assert any("key count differs" in issue for issue in count_report.issues)
    assert any("key set differs" in issue for issue in keyset_report.issues)


def test_export_is_deterministic_and_manifest_is_the_complete_marker(tmp_path):
    bundle = _build(_pool_input())
    first = export_full_pool(bundle, FilesystemSink(tmp_path / "first"))
    second = export_full_pool(bundle, FilesystemSink(tmp_path / "second"))

    assert first.table_sha256 == second.table_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(_file_uri_path(first.manifest_uri).read_text())
    assert manifest["pool_complete"] is True
    assert manifest["total_source_keys"] == 3
    assert manifest["raw_inventory_evidence"]["expected_key_count"] == 3
    assert len(manifest["files"]) == 11
    table_uri = dict(first.table_uris)["label_candidates"]
    table = pq.read_table(_file_uri_path(table_uri))
    assert table.num_rows == 2
    assert table.schema.metadata[b"contract_version"] == b"1.2.0"
    assert table.column("denominator").to_pylist() == [100.0, 100.0]


def test_export_schema_metadata_uses_the_manifest_contract_version(tmp_path):
    bundle = build_full_pool(_pool_input(), contract_version="1.2.0")
    receipt = export_full_pool(bundle, FilesystemSink(tmp_path))

    table_path = _file_uri_path(dict(receipt.table_uris)["label_candidates"])

    assert pq.read_schema(table_path).metadata[b"contract_version"] == b"1.2.0"


def test_export_refuses_incomplete_pool(tmp_path):
    bundle = _build(_pool_input())
    incomplete = replace(bundle, label_outcomes=bundle.label_outcomes[:-1])

    with pytest.raises(InvalidPoolError):
        export_full_pool(incomplete, FilesystemSink(tmp_path))


def test_nine_pool_gate_requires_all_three_by_three_combinations():
    bundles = [
        _build(_pool_input(text_source, reference_system))
        for text_source in TextSource
        for reference_system in ReferenceSystem
    ]

    assert validate_nine_pool_gate(bundles).ok
    gate = validate_nine_pool_gate(bundles[:-1])
    assert not gate.ok
    assert gate.missing_pools


def _stream_fixture_pool(
    tmp_path: Path,
    text_source: TextSource = TextSource.DESCRIPTION,
    reference_system: ReferenceSystem = ReferenceSystem.WORLDCOVER,
    *,
    source: PoolBuildInput | None = None,
):
    source = source or _pool_input(text_source, reference_system)
    scan_bytes = f"fixture scan {text_source.value}/{reference_system.value}\n".encode()
    scan_path = tmp_path / "scan-reports" / f"{text_source.value}-{reference_system.value}.json"
    scan_path.parent.mkdir(parents=True, exist_ok=True)
    scan_path.write_bytes(scan_bytes)
    evidence = replace(
        source.raw_inventory_evidence,
        scan_report_uri=scan_path.as_uri(),
        scan_report_sha256=hashlib.sha256(scan_bytes).hexdigest(),
    )
    evidence_root = (
        tmp_path / "evaluation-evidence" / f"{text_source.value}-{reference_system.value}"
    )
    raw_artifact_root = evidence_root / "raw-tree"
    pbf_path = raw_artifact_root / "source.osm.pbf"
    pbf_path.parent.mkdir(parents=True, exist_ok=True)
    pbf_bytes = f"fixture PBF {text_source.value}/{reference_system.value}".encode()
    pbf_path.write_bytes(pbf_bytes)
    geometry_rows = []
    geometry_id_map = {}
    geometry_artifacts = {}
    for index, polygon in enumerate(source.polygons):
        geometry_bytes = f"fixture geometry {index}".encode()
        geometry_path = raw_artifact_root / "geometries" / f"polygon-{index:04d}.wkb"
        geometry_path.parent.mkdir(parents=True, exist_ok=True)
        geometry_path.write_bytes(geometry_bytes)
        geometry_sha256 = hashlib.sha256(geometry_bytes).hexdigest()
        polygon_snapshot_id = polygon_snapshot_identifier(
            source.source_snapshot.snapshot_id,
            polygon.native_key,
            geometry_sha256,
        )
        geometry_id_map[polygon.polygon_snapshot_id] = polygon_snapshot_id
        geometry_artifacts[f"geometries/polygon-{index:04d}.wkb"] = geometry_bytes
        geometry_rows.append(
            replace(
                polygon,
                polygon_snapshot_id=polygon_snapshot_id,
                geometry_uri=geometry_path.as_uri(),
                geometry_sha256=geometry_sha256,
            )
        )
    source_tree_pin, _ = _write_artifact_manifest(
        raw_artifact_root,
        "raw-artifacts.jsonl",
        {
            "source.osm.pbf": pbf_bytes,
            "polygons-region-0.parquet": b"fixture raw polygon shard",
            **geometry_artifacts,
        },
    )
    reference_tree_pin, _ = _write_artifact_manifest(
        evidence_root / "reference-tree",
        "reference-artifacts.jsonl",
        {
            "reference-tile-00.tif": b"fixture reference tile 00",
            "reference-tile-01.tif": b"fixture reference tile 01",
        },
    )
    source = replace(
        source,
        raw_inventory_evidence=evidence,
        inventory=tuple(
            replace(
                row,
                polygon_snapshot_id=geometry_id_map.get(
                    row.polygon_snapshot_id,
                    row.polygon_snapshot_id,
                ),
            )
            for row in source.inventory
        ),
        polygons=tuple(geometry_rows),
        label_candidates=tuple(
            replace(
                row,
                polygon_snapshot_id=geometry_id_map[row.polygon_snapshot_id],
            )
            for row in source.label_candidates
        ),
        label_outcomes=tuple(
            replace(
                row,
                polygon_snapshot_id=(
                    geometry_id_map.get(row.polygon_snapshot_id)
                    if row.polygon_snapshot_id is not None
                    else None
                ),
            )
            for row in source.label_outcomes
        ),
        source_snapshot=replace(
            source.source_snapshot,
            pbf_uri=pbf_path.as_uri(),
            pbf_sha256=hashlib.sha256(pbf_bytes).hexdigest(),
            source_artifact_manifest=source_tree_pin,
        ),
        reference_pin=replace(
            source.reference_pin,
            reference_artifact_manifest=reference_tree_pin,
            area_measurement=AreaMeasurementPin(
                implementation="fixture.geometry.measure",
                implementation_version="1",
                numerator=_fixture_area_measure(),
                denominator=_fixture_area_measure(),
            ),
        ),
    )
    sink = FilesystemStreamingSink(
        tmp_path / "streamed" / f"{text_source.value}-{reference_system.value}"
    )
    return export_pool_batches(
        PoolStreamSpec.from_build_input(source),
        _single_batch(source),
        sink,
    )


def _fixture_area_measure() -> AreaMeasureSpec:
    return AreaMeasureSpec(
        method_id="fixture-area-v1",
        coordinate_system="EPSG:3035",
        ellipsoid="GRS80",
        boundary_convention="straight segments in EPSG:3035",
        densification_method="none",
        densification_parameters_json='{"applied":false,"reason":"projected fixture geometry"}',
        error_settings_json='{"quantification":"not_estimated","reason":"test fixture"}',
        output_unit="m2",
    )


def _write_artifact_manifest(
    root: Path,
    manifest_filename: str,
    artifacts: dict[str, bytes],
):
    artifact_rows = []
    artifact_paths = []
    for artifact_name, content in artifacts.items():
        path = root / artifact_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        artifact_paths.append(path)
        artifact_rows.append(
            {
                "artifact_uri": path.as_uri(),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
        )
    manifest_path = root / manifest_filename
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in sorted(artifact_rows, key=lambda row: row["artifact_uri"])
    )
    manifest_path.write_bytes(manifest_bytes)
    return (
        ArtifactManifestPin(
            format_version="artifact-list-jsonl-v1",
            manifest_uri=manifest_path.as_uri(),
            manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
            expected_artifact_count=len(artifact_rows),
        ),
        tuple(artifact_paths),
    )


def _rewrite_stream_manifest(receipt, update):
    manifest_path = _file_uri_path(receipt.manifest_uri)
    manifest = json.loads(manifest_path.read_text())
    update(manifest)
    manifest_bytes = (
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode()
    manifest_path.write_bytes(manifest_bytes)
    return replace(receipt, manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest())


def _completed_unassigned_fixture(source: PoolBuildInput) -> PoolBuildInput:
    row = source.inventory[1]
    polygon_id = polygon_snapshot_identifier(
        source.source_snapshot.snapshot_id,
        row.native_key,
        "d" * 64,
    )
    polygon = PolygonSnapshot(
        polygon_snapshot_id=polygon_id,
        inventory_row_id=row.inventory_row_id,
        native_key=row.native_key,
        geometry_uri="https://artifacts.example/geometries/no-reference-overlap.wkb",
        geometry_sha256="d" * 64,
        geometry_format="WKB",
        crs="EPSG:3035",
        coordinate_transform="EPSG:4326 -> EPSG:3035",
        geometry_status=GeometryStatus.USABLE,
        original_area=100.0,
        area_unit="m2",
    )
    inventory = tuple(
        replace(
            item,
            polygon_snapshot_id=polygon_id,
            exclusion_reason=None,
        )
        if item.inventory_row_id == row.inventory_row_id
        else item
        for item in source.inventory
    )
    outcomes = tuple(
        replace(
            item,
            polygon_snapshot_id=polygon_id,
            status=OutcomeStatus.UNASSIGNED,
            assigned_code=None,
            reason="completed reference search found no qualifying class",
            ambiguity_basis=None,
        )
        if item.inventory_row_id == row.inventory_row_id
        else item
        for item in source.label_outcomes
    )
    attempts = tuple(
        replace(
            item,
            status=AttemptStatus.COMPLETED,
            attempted_reference_ids=(),
            missing_reference_ids=(),
            reason=None,
        )
        if item.inventory_row_id == row.inventory_row_id
        else item
        for item in source.attempt_receipts
    )
    return replace(
        source,
        inventory=inventory,
        polygons=(*source.polygons, polygon),
        label_outcomes=outcomes,
        attempt_receipts=attempts,
    )


def _stream_nine_fixture_pools(
    tmp_path: Path,
    *,
    include_completed_unassigned: bool = False,
):
    receipts = []
    for text_source in TextSource:
        for reference_system in ReferenceSystem:
            source = _pool_input(text_source, reference_system)
            if (
                include_completed_unassigned
                and text_source is TextSource.DESCRIPTION
                and reference_system is ReferenceSystem.WORLDCOVER
            ):
                source = _completed_unassigned_fixture(source)
            receipts.append(
                _stream_fixture_pool(
                    tmp_path,
                    text_source,
                    reference_system,
                    source=source,
                )
            )
    return tuple(receipts)


def test_streamed_nine_pool_gate_verifies_manifests_shards_and_inventory(tmp_path: Path):
    receipts = _stream_nine_fixture_pools(tmp_path, include_completed_unassigned=True)

    gate = validate_nine_streamed_pool_gate(receipts, FilesystemArtifactReader())
    evaluation_gate = validate_nine_streamed_evaluation_ready_gate(
        receipts,
        FilesystemArtifactReader(),
    )

    assert gate.ok, gate.invalid_pools
    assert evaluation_gate.ok, evaluation_gate.invalid_pools
    forged = replace(receipts[0], total_source_keys=receipts[0].total_source_keys + 1)
    rejected = validate_nine_streamed_pool_gate((forged, *receipts[1:]), FilesystemArtifactReader())
    assert not rejected.ok
    assert any(
        "raw inventory receipt differs" in issue
        for _pool, issues in rejected.invalid_pools
        for issue in issues
    )
    duplicate = validate_nine_streamed_pool_gate(
        (receipts[0], *receipts), FilesystemArtifactReader()
    )
    assert "description-worldcover" in duplicate.duplicate_pools
    incomplete = validate_nine_streamed_pool_gate(receipts[:-1], FilesystemArtifactReader())
    assert "wiki-eunis" in incomplete.missing_pools


def test_streamed_evaluation_gate_blocks_failed_partial_and_missing_reference_receipts(
    tmp_path: Path,
):
    source = _pool_input(TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER)
    attempts = (
        replace(
            source.attempt_receipts[0],
            status=AttemptStatus.FAILED,
            reason="tile_read_failed",
        ),
        replace(
            source.attempt_receipts[1],
            status=AttemptStatus.PARTIAL,
            reason="one required shard was unreadable",
        ),
        replace(
            source.attempt_receipts[2],
            status=AttemptStatus.COMPLETED,
            missing_reference_ids=("required-reference-tile-7",),
            reason=None,
        ),
    )
    receipt = _stream_fixture_pool(
        tmp_path,
        TextSource.DESCRIPTION,
        ReferenceSystem.WORLDCOVER,
        source=replace(source, attempt_receipts=attempts),
    )
    reader = FilesystemArtifactReader()

    structural = validate_nine_streamed_pool_gate((receipt,), reader)
    readiness = validate_nine_streamed_evaluation_ready_gate((receipt,), reader)
    issues = " ".join(issue for _, rows in readiness.invalid_pools for issue in rows)

    assert not structural.invalid_pools
    assert not readiness.ok
    assert "FAILED polygon attempts" in issues
    assert "PARTIAL polygon attempts" in issues
    assert "missing reference IDs" in issues


def test_streamed_evaluation_gate_bounds_receipt_materialization(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    consumed = 0

    def receipts():
        nonlocal consumed
        for _ in range(100):
            consumed += 1
            yield receipt

    readiness = validate_nine_streamed_evaluation_ready_gate(
        receipts(),
        FilesystemArtifactReader(),
    )

    assert not readiness.ok
    assert consumed == 10


def test_streamed_evaluation_gate_blocks_unread_pinned_reference_artifacts(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    reference_pin = manifest["reference_pin"]["reference_artifact_manifest"]
    first_artifact = json.loads(
        _file_uri_path(reference_pin["manifest_uri"]).read_text().splitlines()[0]
    )
    Path(unquote(urlparse(first_artifact["artifact_uri"]).path)).unlink()

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, rows in readiness.invalid_pools for issue in rows)

    assert not readiness.ok
    assert "reference artifact could not be read" in issues


def test_streamed_evaluation_gate_requires_artifact_lists_and_area_method(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)

    def remove_or_corrupt_readiness_pins(manifest):
        manifest["source_snapshot"]["source_artifact_manifest"]["format_version"] = "unknown-v0"
        manifest["reference_pin"]["reference_artifact_manifest"] = None
        manifest["reference_pin"]["area_measurement"]["numerator"][
            "densification_parameters_json"
        ] = "{}"

    receipt = _rewrite_stream_manifest(receipt, remove_or_corrupt_readiness_pins)
    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, rows in readiness.invalid_pools for issue in rows)

    assert "raw source artifact manifest format is unsupported" in issues
    assert "reference artifact manifest pin is missing" in issues
    assert (
        "area measurement numerator densification_parameters_json is not a non-empty "
        "canonical JSON object" in issues
    )

    def remove_readiness_pins(manifest):
        manifest["source_snapshot"]["source_artifact_manifest"] = None
        manifest["reference_pin"]["area_measurement"] = None

    receipt = _rewrite_stream_manifest(receipt, remove_readiness_pins)
    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, rows in readiness.invalid_pools for issue in rows)

    assert "raw source artifact manifest pin is missing" in issues
    assert "area measurement method pin is incomplete" in issues


def test_streamed_evaluation_gate_requires_explicit_area_geometry_settings(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)

    def remove_area_method_evidence(manifest):
        method = manifest["reference_pin"]["area_measurement"]
        method["numerator"]["coordinate_system"] = None
        method["numerator"]["ellipsoid"] = None
        method["numerator"]["boundary_convention"] = ""
        method["numerator"]["error_settings_json"] = "[]"
        method["denominator"]["output_unit"] = "ha"

    receipt = _rewrite_stream_manifest(receipt, remove_area_method_evidence)
    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, rows in readiness.invalid_pools for issue in rows)

    assert "numerator needs a coordinate system or ellipsoid" in issues
    assert "numerator lacks boundary_convention" in issues
    assert "numerator error_settings_json is not a non-empty canonical JSON object" in issues
    assert "denominator output unit differs from the support unit" in issues


def test_streamed_evaluation_gate_requires_pinned_pbf_in_source_manifest(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    source = manifest["source_snapshot"]
    pin = source["source_artifact_manifest"]
    manifest_path = _file_uri_path(pin["manifest_uri"])
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    rows = [row for row in rows if row["artifact_uri"] != source["pbf_uri"]]
    manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    manifest_path.write_bytes(manifest_bytes)
    pin["manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    pin["expected_artifact_count"] = len(rows)
    receipt = _rewrite_stream_manifest(receipt, lambda updated: updated.update(manifest))

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, entries in readiness.invalid_pools for issue in entries)

    assert "raw source manifest omits the pinned PBF" in issues


def test_streamed_evaluation_gate_accepts_a_pinned_non_pbf_source_tree(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    source = manifest["source_snapshot"]
    pin = source["source_artifact_manifest"]
    artifact_manifest_path = _file_uri_path(pin["manifest_uri"])
    rows = [json.loads(line) for line in artifact_manifest_path.read_text().splitlines()]
    rows = [row for row in rows if row["artifact_uri"] != source["pbf_uri"]]
    artifact_manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    artifact_manifest_path.write_bytes(artifact_manifest_bytes)
    pin["manifest_sha256"] = hashlib.sha256(artifact_manifest_bytes).hexdigest()
    pin["expected_artifact_count"] = len(rows)
    source["pbf_uri"] = None
    source["pbf_sha256"] = None
    receipt = _rewrite_stream_manifest(receipt, lambda updated: updated.update(manifest))

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )

    assert not readiness.invalid_pools
    assert readiness.missing_pools


def test_streamed_evaluation_gate_checks_artifact_manifest_hash_order_and_count(
    tmp_path: Path,
):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    pin = manifest["reference_pin"]["reference_artifact_manifest"]
    manifest_path = _file_uri_path(pin["manifest_uri"])
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    rows.reverse()
    rows[0]["sha256"] = "0" * 64
    artifact_manifest_bytes = b"".join(
        (json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode()
        for row in rows
    )
    manifest_path.write_bytes(artifact_manifest_bytes)
    pin["manifest_sha256"] = hashlib.sha256(artifact_manifest_bytes).hexdigest()
    pin["expected_artifact_count"] = len(rows) + 1
    receipt = _rewrite_stream_manifest(receipt, lambda updated: updated.update(manifest))

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, entries in readiness.invalid_pools for issue in entries)

    assert "artifact manifest is not uniquely sorted" in issues
    assert "reference artifact SHA-256 does not match its pin" in issues
    assert "artifact manifest lists 2 artifacts, expected 3" in issues


def test_streamed_evaluation_gate_hashes_measurement_sidecars(tmp_path: Path):
    source = _pool_input(TextSource.DESCRIPTION, ReferenceSystem.WORLDCOVER)
    sidecar = tmp_path / "measurements.jsonl"
    sidecar_bytes = b'{"feature_id":"tile-7","intersection_area":80.0}\n'
    sidecar.write_bytes(sidecar_bytes)
    evidence = replace(
        source.support_evidence[0],
        reference_ids=(),
        measurement_sidecar_uri=sidecar.as_uri(),
        measurement_sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
        measurement_record_count=1,
    )
    source = replace(source, support_evidence=(evidence, *source.support_evidence[1:]))
    receipt = _stream_fixture_pool(
        tmp_path,
        TextSource.DESCRIPTION,
        ReferenceSystem.WORLDCOVER,
        source=source,
    )
    reader = FilesystemArtifactReader()

    valid = validate_nine_streamed_evaluation_ready_gate((receipt,), reader)
    sidecar.unlink()
    missing = validate_nine_streamed_evaluation_ready_gate((receipt,), reader)
    issues = " ".join(issue for _, entries in missing.invalid_pools for issue in entries)

    assert not valid.invalid_pools
    assert "label support evidence" in issues
    assert "could not be read" in issues


def test_streamed_evaluation_gate_rejects_malformed_artifact_manifest_rows(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    pin = manifest["reference_pin"]["reference_artifact_manifest"]
    manifest_path = _file_uri_path(pin["manifest_uri"])
    valid_row = json.loads(manifest_path.read_text().splitlines()[0])
    noncanonical_row = json.dumps(
        {"sha256": valid_row["sha256"], "artifact_uri": valid_row["artifact_uri"]},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    artifact_manifest_bytes = (
        f'[]\n{{"artifact_path":"missing-key","sha256":"not-a-digest"}}\n{noncanonical_row}\n'
    ).encode()
    manifest_path.write_bytes(artifact_manifest_bytes)
    pin["manifest_sha256"] = hashlib.sha256(artifact_manifest_bytes).hexdigest()
    pin["expected_artifact_count"] = 3
    receipt = _rewrite_stream_manifest(receipt, lambda updated: updated.update(manifest))

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, entries in readiness.invalid_pools for issue in entries)

    assert "contains a non-object row" in issues
    assert "row has an invalid schema" in issues
    assert "contains an invalid artifact pin" in issues
    assert "rows are not canonical JSONL" in issues


def test_streamed_evaluation_gate_rejects_invalid_and_empty_artifact_manifests(
    tmp_path: Path,
):
    receipt = _stream_fixture_pool(tmp_path)

    def set_invalid_count(manifest):
        manifest["reference_pin"]["reference_artifact_manifest"]["expected_artifact_count"] = 0

    receipt = _rewrite_stream_manifest(receipt, set_invalid_count)
    invalid_pin = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    invalid_pin_issues = " ".join(
        issue for _, entries in invalid_pin.invalid_pools for issue in entries
    )

    assert "reference artifact manifest pin is invalid" in invalid_pin_issues

    def set_empty_manifest(manifest):
        pin = manifest["reference_pin"]["reference_artifact_manifest"]
        _file_uri_path(pin["manifest_uri"]).write_bytes(b"")
        pin["manifest_sha256"] = hashlib.sha256(b"").hexdigest()
        pin["expected_artifact_count"] = 1

    receipt = _rewrite_stream_manifest(receipt, set_empty_manifest)
    empty_manifest = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    empty_issues = " ".join(
        issue for _, entries in empty_manifest.invalid_pools for issue in entries
    )

    assert "artifact manifest is empty" in empty_issues
    assert "lists 0 artifacts, expected 1" in empty_issues


def test_streamed_evaluation_gate_blocks_unread_artifact_manifest(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    pin_uri = manifest["reference_pin"]["reference_artifact_manifest"]["manifest_uri"]
    _file_uri_path(pin_uri).unlink()

    readiness = validate_nine_streamed_evaluation_ready_gate(
        (receipt,),
        FilesystemArtifactReader(),
    )
    issues = " ".join(issue for _, entries in readiness.invalid_pools for issue in entries)

    assert "reference artifact manifest could not be read" in issues


def test_streamed_gate_rejects_tampered_shard(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    shard_manifest = _file_uri_path(receipt.shard_manifest_uri)
    first_shard = json.loads(shard_manifest.read_text().splitlines()[0])
    shard_path = _file_uri_path(
        FilesystemArtifactReader().resolve_uri(receipt.manifest_uri, first_shard["path"])
    )
    shard_path.write_bytes(shard_path.read_bytes() + b"tamper")

    gate = validate_nine_streamed_pool_gate((receipt,), FilesystemArtifactReader())

    assert not gate.ok
    assert any(
        "shard content hash mismatch" in issue
        for pool, issues in gate.invalid_pools
        if pool == "description-worldcover"
        for issue in issues
    )
    assert manifest["pool_complete"] is True


def _rewrite_streamed_receipt(
    receipt: streaming_module.StreamingExportReceipt,
    manifest: dict,
    shard_rows: list[dict],
) -> streaming_module.StreamingExportReceipt:
    shard_manifest_bytes = b"".join(
        streaming_module._canonical_json_bytes(row) for row in shard_rows
    )
    shard_manifest_path = _file_uri_path(receipt.shard_manifest_uri)
    shard_manifest_path.write_bytes(shard_manifest_bytes)
    shard_manifest_sha256 = hashlib.sha256(shard_manifest_bytes).hexdigest()
    manifest["storage_layout"]["shard_manifest"]["sha256"] = shard_manifest_sha256
    manifest_bytes = streaming_module._canonical_json_bytes(manifest)
    _file_uri_path(receipt.manifest_uri).write_bytes(manifest_bytes)
    return replace(
        receipt,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        shard_manifest_sha256=shard_manifest_sha256,
    )


def test_streamed_gate_checks_parquet_footer_row_count_against_manifests(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    shard_rows = [
        json.loads(line)
        for line in _file_uri_path(receipt.shard_manifest_uri).read_text().splitlines()
    ]
    document_shard = next(row for row in shard_rows if row["table"] == "document_relevance")
    document_shard["row_count"] += 1
    manifest["storage_layout"]["tables"]["document_relevance"]["row_count"] += 1
    forged_receipt = _rewrite_streamed_receipt(receipt, manifest, shard_rows)

    gate = validate_nine_streamed_pool_gate((forged_receipt,), FilesystemArtifactReader())

    assert not gate.ok
    assert any(
        "Parquet footer row count differs from shard manifest" in issue
        for pool, issues in gate.invalid_pools
        if pool == "description-worldcover"
        for issue in issues
    )


def test_streamed_gate_checks_parquet_footer_schema_against_contract(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    manifest = json.loads(_file_uri_path(receipt.manifest_uri).read_text())
    shard_rows = [
        json.loads(line)
        for line in _file_uri_path(receipt.shard_manifest_uri).read_text().splitlines()
    ]
    document_shard = next(row for row in shard_rows if row["table"] == "document_relevance")
    shard_path = _file_uri_path(
        FilesystemArtifactReader().resolve_uri(receipt.manifest_uri, document_shard["path"])
    )
    table = pq.read_table(shard_path)
    original_field = table.schema.field(0)
    altered_schema = table.schema.set(
        0,
        pa.field(
            "unexpected_document_column",
            original_field.type,
            nullable=original_field.nullable,
        ),
    )
    pq.write_table(pa.Table.from_arrays(table.columns, schema=altered_schema), shard_path)
    document_shard["sha256"] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    forged_receipt = _rewrite_streamed_receipt(receipt, manifest, shard_rows)

    gate = validate_nine_streamed_pool_gate((forged_receipt,), FilesystemArtifactReader())

    assert not gate.ok
    assert any(
        "Parquet footer schema differs from contract" in issue
        for pool, issues in gate.invalid_pools
        if pool == "description-worldcover"
        for issue in issues
    )


def test_streamed_gate_reports_malformed_receipt_pool_id_after_read_failure(tmp_path: Path):
    receipt = _stream_fixture_pool(tmp_path)
    broken_uri = _file_uri_path(receipt.manifest_uri).with_suffix(".missing").as_uri()
    malformed = replace(
        receipt,
        pool_id="1.0.0/invalid/raw",
        manifest_uri=broken_uri,
    )

    gate = validate_nine_streamed_pool_gate((malformed,), FilesystemArtifactReader())

    assert not gate.ok
    assert any(pool == "1.0.0/invalid/raw" for pool, _issues in gate.invalid_pools)
    assert len(gate.missing_pools) == 9


def test_filesystem_sink_concurrent_writers_never_overwrite_immutable_artifact(tmp_path: Path):
    sink = FilesystemSink(tmp_path)
    barrier = threading.Barrier(2)

    def write(content: bytes):
        barrier.wait()
        try:
            return sink.put("immutable/artifact.bin", content), None
        except FileExistsError as error:
            return None, error

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(write, (b"left", b"right")))

    successes = [uri for uri, error in results if uri is not None and error is None]
    failures = [error for _uri, error in results if error is not None]
    assert len(successes) == 1
    assert len(failures) == 1
    assert _file_uri_path(successes[0]).read_bytes() in {b"left", b"right"}
    assert [path.name for path in (tmp_path / "immutable").iterdir()] == ["artifact.bin"]


def test_relevance_is_typed_provenance_and_never_filters_the_raw_pool(tmp_path: Path):
    source = _pool_input()
    assessments = (
        source.relevance_assessments[0],
        RelevanceAssessment(
            document_id="document-private",
            release_id="relevance-release-no",
            status=RelevanceStatus.NO,
            source_repository="NoeFlandre/website-relevance",
            source_revision="relevance-revision-2",
            mapping_rule_version="relevance-map-v2",
            evidence="native relevance record no",
            raw_value="no",
        ),
        RelevanceAssessment(
            document_id="document-missing",
            release_id="relevance-release-unknown",
            status=RelevanceStatus.UNKNOWN,
            source_repository="NoeFlandre/website-relevance",
            source_revision="relevance-revision-2",
            mapping_rule_version="relevance-map-v2",
            evidence="native relevance record could not be mapped",
            raw_value="ambiguous",
            reason="source value has no verified yes/no mapping",
        ),
        source.relevance_assessments[1],
    )
    populated = replace(source, relevance_assessments=assessments)
    bundle = _build(populated)
    no_relevance_bundle = _build(replace(source, relevance_assessments=()))

    assert validate_full_pool(bundle).ok
    assert len(bundle.inventory) == len(source.inventory) == 3
    assert bundle.label_outcomes == no_relevance_bundle.label_outcomes
    assert bundle.manifest.total_source_keys == no_relevance_bundle.manifest.total_source_keys
    assert not isinstance(RelevanceStatus.YES, EvaluationArm)

    receipt = export_full_pool(bundle, FilesystemSink(tmp_path / "relevance"))
    table = pq.read_table(_file_uri_path(dict(receipt.table_uris)["document_relevance"]))
    assert set(table.column("status").to_pylist()) == {"yes", "no", "unknown", "missing"}
    assert set(table.column("source_revision").to_pylist()) == {
        "relevance-revision-1",
        "relevance-revision-2",
    }


def test_measurement_evidence_accepts_reference_ids_or_hashed_sidecars(tmp_path: Path):
    source = _pool_input()
    evidence = replace(
        source.support_evidence[0],
        reference_ids=(),
        measurement_sidecar_uri="https://artifacts.example/measurements/row-1.jsonl",
        measurement_sidecar_sha256=_SHA_B,
        measurement_record_count=2,
    )
    receipt = replace(
        source.attempt_receipts[0],
        attempted_reference_ids=(),
        measurement_sidecar_uri="https://artifacts.example/attempts/row-1.jsonl",
        measurement_sidecar_sha256=_SHA_A,
        measurement_record_count=3,
    )
    source = replace(
        source,
        support_evidence=(evidence, *source.support_evidence[1:]),
        attempt_receipts=(receipt, *source.attempt_receipts[1:]),
    )

    bundle = _build(source)
    assert validate_full_pool(bundle).ok
    exported = export_full_pool(bundle, FilesystemSink(tmp_path / "evidence"))
    support_table = pq.read_table(
        _file_uri_path(dict(exported.table_uris)["label_support_evidence"])
    )
    attempt_table = pq.read_table(_file_uri_path(dict(exported.table_uris)["polygon_attempts"]))
    assert _SHA_B in support_table.column("measurement_sidecar_sha256").to_pylist()
    assert _SHA_A in attempt_table.column("measurement_sidecar_sha256").to_pylist()


def test_source_native_keys_preserve_component_names_and_value_types():
    integer_key = canonical_native_key({"source_pbf": "x.pbf", "osm_id": 17})
    string_key = canonical_native_key({"source_pbf": "x.pbf", "osm_id": "17"})

    assert integer_key != string_key
    assert json.loads(json.dumps(integer_key)) != json.loads(json.dumps(string_key))
    assert native_keyset_sha256((integer_key, string_key)) == native_keyset_sha256(
        (string_key, integer_key)
    )


def test_evaluation_cases_and_traces_are_separate_and_inspectable():
    case = EvaluationCase(
        case_id="case-1",
        pool_id="1.0.0/website-eunis/raw-website",
        inventory_row_id="polygon-1",
        target_level="L2",
        arm=EvaluationArm.YES,
        sampling_policy_version="h3-spread-v1",
        selection_order=0,
        source_document_ids=("document-private",),
    )
    prompt = "classify this polygon"
    response = "A1"
    trace = EvaluationTrace(
        trace_id="trace-1",
        case_id=case.case_id,
        model_id="model-pinned-revision",
        settings_json='{"temperature":0,"top_p":1}',
        prompt=EvaluationPayload(
            status=BodyStatus.AVAILABLE,
            value=prompt,
            sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        ),
        raw_response=EvaluationPayload(
            status=BodyStatus.AVAILABLE,
            value=response,
            sha256=hashlib.sha256(response.encode()).hexdigest(),
        ),
        reference_geometry_uri="https://artifacts.example/reference.wkb",
        reference_geometry_sha256=_SHA_B,
    )

    assert validate_evaluation_case(case).ok
    assert validate_evaluation_trace(trace).ok
    private_trace = replace(
        trace,
        prompt=EvaluationPayload(status=BodyStatus.PRIVATE, value=None, sha256=None),
    )
    assert validate_evaluation_trace(private_trace).ok
    invalid_trace = replace(trace, settings_json='{"top_p":1,"temperature":0}')
    assert not validate_evaluation_trace(invalid_trace).ok


def test_split_assignment_requires_version_and_full_entity_identity():
    key = canonical_native_key({"source_pbf": "source.osm.pbf", "osm_type": "way", "osm_id": 10})
    assignment = SplitAssignment(
        entity=EntityReference("raw-source", key),
        split=DataSplit.TRAIN,
        policy_version="worldcover-legacy-verified-v1",
        evidence="verified full split artifact sha256",
    )

    assert validate_split_assignment(assignment).ok
    assert not validate_split_assignment(replace(assignment, policy_version="")).ok


def _stream_batches(source: PoolBuildInput) -> tuple[PoolBatch, PoolBatch]:
    first_inventory_id = source.inventory[0].inventory_row_id
    first_ids = {first_inventory_id}
    return (
        PoolBatch(
            batch_index=0,
            inventory=tuple(row for row in source.inventory if row.inventory_row_id in first_ids),
            polygons=tuple(row for row in source.polygons if row.inventory_row_id in first_ids),
            label_candidates=tuple(
                row for row in source.label_candidates if row.inventory_row_id in first_ids
            ),
            label_outcomes=tuple(
                row for row in source.label_outcomes if row.inventory_row_id in first_ids
            ),
            attempt_receipts=tuple(
                row for row in source.attempt_receipts if row.inventory_row_id in first_ids
            ),
            support_evidence=tuple(
                row for row in source.support_evidence if row.inventory_row_id in first_ids
            ),
            documents=(),
            text_polygon_links=source.text_polygon_links,
            entity_links=source.entity_links,
            eligibility=source.eligibility,
        ),
        PoolBatch(
            batch_index=1,
            inventory=tuple(
                row for row in source.inventory if row.inventory_row_id not in first_ids
            ),
            polygons=tuple(row for row in source.polygons if row.inventory_row_id not in first_ids),
            label_candidates=tuple(
                row for row in source.label_candidates if row.inventory_row_id not in first_ids
            ),
            label_outcomes=tuple(
                row for row in source.label_outcomes if row.inventory_row_id not in first_ids
            ),
            attempt_receipts=tuple(
                row for row in source.attempt_receipts if row.inventory_row_id not in first_ids
            ),
            support_evidence=tuple(
                row for row in source.support_evidence if row.inventory_row_id not in first_ids
            ),
            documents=source.documents,
            relevance_assessments=source.relevance_assessments,
        ),
    )


def _single_batch(source: PoolBuildInput) -> tuple[PoolBatch, ...]:
    return (
        PoolBatch(
            batch_index=0,
            inventory=source.inventory,
            polygons=source.polygons,
            label_candidates=source.label_candidates,
            label_outcomes=source.label_outcomes,
            attempt_receipts=source.attempt_receipts,
            support_evidence=source.support_evidence,
            documents=source.documents,
            relevance_assessments=source.relevance_assessments,
            text_polygon_links=source.text_polygon_links,
            entity_links=source.entity_links,
            eligibility=source.eligibility,
        ),
    )


def _tablewise_batches(source: PoolBuildInput) -> tuple[PoolBatch, ...]:
    return (
        PoolBatch(batch_index=0, inventory=source.inventory),
        PoolBatch(batch_index=1, polygons=source.polygons),
        PoolBatch(batch_index=2, label_candidates=source.label_candidates),
        PoolBatch(batch_index=3, label_outcomes=source.label_outcomes),
        PoolBatch(batch_index=4, attempt_receipts=source.attempt_receipts),
        PoolBatch(batch_index=5, support_evidence=source.support_evidence),
        PoolBatch(batch_index=6, text_polygon_links=source.text_polygon_links),
        PoolBatch(batch_index=7, documents=source.documents),
        PoolBatch(batch_index=8, relevance_assessments=source.relevance_assessments),
        PoolBatch(batch_index=9, entity_links=source.entity_links),
        PoolBatch(batch_index=10, eligibility=source.eligibility),
    )


def _canonical_table_rows(table: pa.Table) -> list[str]:
    return sorted(json.dumps(row, sort_keys=True) for row in table.to_pylist())


def _assert_unicode_reference_ids_are_literal(
    receipt: ExportReceipt,
    source: PoolBuildInput,
) -> None:
    attempts = pq.read_table(
        _file_uri_path(dict(receipt.table_uris)["polygon_attempts"])
    ).to_pylist()
    first_attempt = next(
        row for row in attempts if row["inventory_row_id"] == source.inventory[0].inventory_row_id
    )
    assert "feature-Å/東京" in first_attempt["attempted_reference_ids_json"]
    assert "missing-ñ" in first_attempt["missing_reference_ids_json"]
    evidence = pq.read_table(
        _file_uri_path(dict(receipt.table_uris)["label_support_evidence"])
    ).to_pylist()
    first_evidence = next(
        row
        for row in evidence
        if row["inventory_row_id"] == source.inventory[0].inventory_row_id
        and row["level"] == source.support_evidence[0].level
        and row["code"] == source.support_evidence[0].code
    )
    assert "reference-é/東京" in first_evidence["reference_ids_json"]


def test_stream_export_matches_fixture_tables_and_reports_completeness(tmp_path: Path) -> None:
    source = _with_unicode_reference_ids(_pool_input())
    bundle = _build(source)
    legacy = export_full_pool(bundle, FilesystemSink(tmp_path / "legacy"))
    stream_sink = FilesystemStreamingSink(tmp_path / "stream")
    spec = PoolStreamSpec.from_build_input(source)

    streamed = export_pool_batches(
        spec,
        _stream_batches(source),
        stream_sink,
        limits=StreamingLimits(max_batch_rows=32, max_shard_rows=2, max_diagnostics=1),
    )

    assert streamed.pool_id == bundle.manifest.pool_id
    assert streamed.total_source_keys == source.raw_inventory_evidence.expected_key_count
    assert streamed.inventory_keyset_sha256 == source.raw_inventory_evidence.expected_keyset_sha256
    assert streamed.shard_count == 15
    manifest = json.loads(_file_uri_path(streamed.manifest_uri).read_text())
    assert manifest["pool_complete"] is True
    assert manifest["source_inventory_receipt"] == {
        "processed_key_count": source.raw_inventory_evidence.expected_key_count,
        "observed_keyset_sha256": source.raw_inventory_evidence.expected_keyset_sha256,
        "matches_source_evidence": True,
    }
    assert manifest["storage_layout_version"] == "sharded-parquet-v1"
    shard_rows = [
        json.loads(line)
        for line in _file_uri_path(streamed.shard_manifest_uri).read_text().splitlines()
    ]
    assert len(shard_rows) == streamed.shard_count
    legacy_accounting = {item.level: item for item in bundle.manifest.level_accounting}
    assert [item.level for item in streamed.level_accounting] == sorted(legacy_accounting)
    for item in streamed.level_accounting:
        legacy_item = legacy_accounting[item.level]
        assert item.total_source_keys == legacy_item.total_source_keys
        assert item.processed_decisions == legacy_item.processed_decisions
        assert item.qualifying_candidates == legacy_item.qualifying_candidates
        assert item.assigned == legacy_item.assigned
        assert item.ambiguous == legacy_item.ambiguous
        assert item.unassigned == legacy_item.unassigned
        assert item.excluded == legacy_item.excluded
    exclusion_artifact = manifest["storage_layout"]["exclusions"]
    exclusion_path = _file_uri_path(streamed.manifest_uri).parent / "exclusions.jsonl"
    exclusion_rows = [json.loads(line) for line in exclusion_path.read_text().splitlines()]
    assert len(exclusion_rows) == exclusion_artifact["group_count"]
    assert hashlib.sha256(exclusion_path.read_bytes()).hexdigest() == exclusion_artifact["sha256"]
    assert {item["reason"] for item in exclusion_rows} == {
        "geometry_artifact_missing",
        "invalid_geometry",
    }
    assert sum(item["count"] for item in exclusion_rows) == sum(
        item.excluded for item in streamed.level_accounting
    )
    assert exclusion_path.exists()

    shards_by_table = {}
    for shard in shard_rows:
        artifact_path = stream_sink.root / shard["path"]
        shards_by_table.setdefault(shard["table"], []).append(artifact_path)
    for table_name, legacy_uri in legacy.table_uris:
        old_table = pq.read_table(_file_uri_path(legacy_uri))
        new_table = pa.concat_tables([pq.read_table(path) for path in shards_by_table[table_name]])
        assert new_table.schema == old_table.schema
        assert _canonical_table_rows(new_table) == _canonical_table_rows(old_table)
    _assert_unicode_reference_ids_are_literal(legacy, source)

    rerun = export_pool_batches(
        spec,
        _tablewise_batches(source),
        FilesystemStreamingSink(tmp_path / "stream-rebatched"),
        limits=StreamingLimits(max_batch_rows=32, max_shard_rows=2, max_diagnostics=1),
    )
    assert rerun.manifest_sha256 == streamed.manifest_sha256
    assert rerun.shard_manifest_sha256 == streamed.shard_manifest_sha256
    rerun_shards = [
        json.loads(line)
        for line in _file_uri_path(rerun.shard_manifest_uri).read_text().splitlines()
    ]
    assert rerun_shards == shard_rows


def test_stream_enforces_encoded_batch_byte_bound_before_completion(tmp_path: Path) -> None:
    source = _pool_input()
    with pytest.raises(PoolStreamError, match="max_batch_bytes"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            _single_batch(source),
            FilesystemStreamingSink(tmp_path / "stream"),
            limits=StreamingLimits(max_batch_bytes=128),
        )
    assert not list((tmp_path / "stream").rglob("manifest.json"))


def test_stream_batch_row_limit_counts_normalized_link_candidates(tmp_path: Path) -> None:
    source = _pool_input()
    batch = _single_batch(source)[0]

    with pytest.raises(PoolStreamError, match="max_batch_rows"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            (batch,),
            FilesystemStreamingSink(tmp_path / "stream"),
            limits=StreamingLimits(max_batch_rows=batch.row_count - 1),
        )


def test_stream_revalidates_reference_measurements_globally(tmp_path: Path) -> None:
    source = _pool_input()
    candidate = replace(source.label_candidates[0], numerator=79.0)
    invalid = replace(source, label_candidates=(candidate, *source.label_candidates[1:]))
    with pytest.raises(PoolStreamError, match="ratio does not match"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            _single_batch(invalid),
            FilesystemStreamingSink(tmp_path / "stream"),
            limits=StreamingLimits(max_batch_rows=32),
        )
    assert not list((tmp_path / "stream").rglob("manifest.json"))


def test_streaming_and_fixture_validation_reject_polygon_area_unit_mismatch(
    tmp_path: Path,
) -> None:
    source = _pool_input()
    invalid = replace(
        source,
        polygons=(replace(source.polygons[0], area_unit="ha"), source.polygons[1]),
    )

    full_report = validate_full_pool(_build(invalid))
    assert any("does not use polygon support units" in issue for issue in full_report.issues)
    with pytest.raises(PoolStreamError, match="does not use polygon support units"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            _single_batch(invalid),
            FilesystemStreamingSink(tmp_path / "unit-mismatch"),
        )
    assert not list((tmp_path / "unit-mismatch").rglob("manifest.json"))


def test_stream_duplicate_native_key_across_batches_never_completes(tmp_path: Path) -> None:
    source = _pool_input()
    batches = _stream_batches(source)
    duplicate = replace(
        batches[1],
        batch_index=2,
        inventory=(batches[0].inventory[0],),
        polygons=batches[0].polygons,
        label_candidates=batches[0].label_candidates,
        label_outcomes=batches[0].label_outcomes,
    )
    sink_root = tmp_path / "stream"
    with pytest.raises(PoolStreamError, match="uniqueness or key constraint"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            (batches[0], batches[1], duplicate),
            FilesystemStreamingSink(sink_root),
            limits=StreamingLimits(max_batch_rows=32),
        )
    manifest_path = sink_root / "pools/1.2.0/website/eunis/raw-website/1.0.0/manifest.json"
    assert not manifest_path.exists()


def test_stream_inventory_receipt_rejects_missing_polygon_keys(tmp_path: Path) -> None:
    source = _pool_input()
    with pytest.raises(PoolStreamError, match="independent scan evidence"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            _stream_batches(source)[:1],
            FilesystemStreamingSink(tmp_path / "stream"),
            limits=StreamingLimits(max_batch_rows=32),
        )
    assert not list((tmp_path / "stream").rglob("manifest.json"))


def test_stream_requires_one_decision_per_supported_level(tmp_path: Path) -> None:
    source = _pool_input()
    incomplete = replace(_stream_batches(source)[0], label_outcomes=())
    with pytest.raises(PoolStreamError, match="decision coverage mismatch"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            (incomplete,),
            FilesystemStreamingSink(tmp_path / "stream"),
            limits=StreamingLimits(max_batch_rows=32),
        )


def test_stream_interrupted_copy_has_no_manifest_and_retries_identically(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _pool_input()
    batches = _stream_batches(source)
    sink_root = tmp_path / "stream"
    sink = FilesystemStreamingSink(sink_root)
    original_copy = streaming_module.shutil.copyfileobj

    def interrupted_copy(source_file, destination_file, **_kwargs):
        destination_file.write(source_file.read(64))
        raise OSError("interrupted shard copy")

    monkeypatch.setattr(streaming_module.shutil, "copyfileobj", interrupted_copy)
    with pytest.raises(OSError, match="interrupted shard copy"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            batches,
            sink,
            limits=StreamingLimits(max_batch_rows=32),
        )
    assert not list(sink_root.rglob("manifest.json"))
    assert not list(sink_root.rglob("*.tmp"))

    monkeypatch.setattr(streaming_module.shutil, "copyfileobj", original_copy)
    first = export_pool_batches(
        PoolStreamSpec.from_build_input(source),
        batches,
        sink,
        limits=StreamingLimits(max_batch_rows=32),
    )
    second = export_pool_batches(
        PoolStreamSpec.from_build_input(source),
        batches,
        sink,
        limits=StreamingLimits(max_batch_rows=32),
    )
    assert second == first
    assert _file_uri_path(first.manifest_uri).exists()


def test_stream_row_size_limit_and_bounded_diagnostics(tmp_path: Path) -> None:
    source = _pool_input()
    sink_root = tmp_path / "stream"
    with pytest.raises(PoolStreamError, match="max_row_bytes"):
        export_pool_batches(
            PoolStreamSpec.from_build_input(source),
            _single_batch(source),
            FilesystemStreamingSink(sink_root),
            limits=StreamingLimits(max_row_bytes=16, max_shard_bytes=16),
        )
    assert not list(sink_root.rglob("manifest.json"))

    error = PoolStreamError((f"issue-{index}" for index in range(20)), max_diagnostics=2)
    assert error.report.issues == (
        "issue-0",
        "issue-1",
        "additional diagnostics omitted after the first 2 issues",
    )


def test_filesystem_stream_sink_rejects_conflicting_retry(tmp_path: Path) -> None:
    sink = FilesystemStreamingSink(tmp_path / "immutable")
    staged = tmp_path / "staged.bin"
    staged.write_bytes(b"first version")
    sink.put_file("layouts/sharded-parquet-v1/shards/test.parquet", staged)
    staged.write_bytes(b"conflicting version")

    with pytest.raises(FileExistsError, match="immutable pool artifact differs"):
        sink.put_file("layouts/sharded-parquet-v1/shards/test.parquet", staged)
