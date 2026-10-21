"""Independent contract tests for full polygon-label pools."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from urllib.parse import unquote, urlparse

import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis.pool_contract import (
    AmbiguityBasis,
    AssignmentKind,
    BodyStatus,
    DataSplit,
    EligibilityStatus,
    EntityLink,
    EntityReference,
    EvaluationArm,
    EvaluationCase,
    EvaluationPayload,
    EvaluationTrace,
    FilesystemSink,
    GeometryStatus,
    InvalidPoolError,
    LabelCandidate,
    LabelLevel,
    LabelOutcome,
    OutcomeStatus,
    PolygonInventoryRow,
    PolygonSnapshot,
    PoolBuildInput,
    PoolEligibility,
    PoolKey,
    ReferencePin,
    ReferenceSystem,
    SourceDocument,
    SourceSnapshot,
    SplitAssignment,
    SupportSemantics,
    TextLinkStatus,
    TextPolygonLink,
    TextSource,
    build_full_pool,
    canonical_native_key,
    export_full_pool,
    inventory_row_identifier,
    polygon_snapshot_identifier,
    validate_evaluation_case,
    validate_evaluation_trace,
    validate_full_pool,
    validate_nine_pool_gate,
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
    return PoolBuildInput(
        pool_key=PoolKey(text_source, reference_system),
        source_snapshot=source,
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
    )


def _build(input_data: PoolBuildInput):
    return build_full_pool(input_data)


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


def test_export_is_deterministic_and_manifest_is_the_complete_marker(tmp_path):
    bundle = _build(_pool_input())
    first = export_full_pool(bundle, FilesystemSink(tmp_path / "first"))
    second = export_full_pool(bundle, FilesystemSink(tmp_path / "second"))

    assert first.table_sha256 == second.table_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    manifest = json.loads(_file_uri_path(first.manifest_uri).read_text())
    assert manifest["pool_complete"] is True
    assert manifest["total_source_keys"] == 3
    assert len(manifest["files"]) == 8
    table_uri = dict(first.table_uris)["label_candidates"]
    table = pq.read_table(_file_uri_path(table_uri))
    assert table.num_rows == 2
    assert table.schema.metadata[b"contract_version"] == b"1.0.0"
    assert table.column("denominator").to_pylist() == [100.0, 100.0]


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


def test_source_native_keys_preserve_component_names_and_value_types():
    integer_key = canonical_native_key({"source_pbf": "x.pbf", "osm_id": 17})
    string_key = canonical_native_key({"source_pbf": "x.pbf", "osm_id": "17"})

    assert integer_key != string_key
    assert json.loads(json.dumps(integer_key)) != json.loads(json.dumps(string_key))


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
