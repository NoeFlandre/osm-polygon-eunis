from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis import eea, pool_adapter
from osm_polygon_eunis.eea import (
    ClassificationTaxonomy,
    ClassificationTaxonomyCacheLineage,
    ClassificationTaxonomyLineagePin,
    parse_classification_taxonomy_rows,
    verify_classification_taxonomy_lineage,
)
from osm_polygon_eunis.pool_adapter import (
    EEA_2021_ASSET_BYTES,
    EEA_2021_ASSET_COUNT,
    EEA_2021_NATIVE_MANIFEST_SHA256,
    EUNIS_POOL_LEVELS,
    EUNIS_POOL_SOURCE_LAYOUTS,
    EunisLabelAttempt,
    EunisTaxonomy,
    EunisTaxonomyPath,
    PolygonDocumentEvidence,
    SourceTreeFile,
    SourceTreeGroupPin,
    make_eunis_2021_reference_pin,
    make_eunis_taxonomy_from_classification,
    resolve_eunis_labels,
    resolve_polygon_document_links,
    source_tree_inventory_sha256,
    validate_source_tree_group,
)
from osm_polygon_eunis.pool_contract import (
    AssignmentKind,
    OutcomeStatus,
    PolygonInventoryRow,
    TextLinkStatus,
    TextSource,
)
from osm_polygon_eunis.pool_contract.builder import canonical_native_key
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference

_TAXONOMY = EunisTaxonomy(
    taxonomy_id="fixture-eunis",
    taxonomy_revision="taxonomy-fixture-v1",
    taxonomy_sha256="a" * 64,
    paths_by_direct_code={
        "C1": EunisTaxonomyPath("A", "A1", "C1"),
        "C2": EunisTaxonomyPath("A", "A2", "C2"),
        "C3": EunisTaxonomyPath("B", "B1", "C3"),
    },
)


def _attempt(
    areas: dict[tuple[str, str], float],
    *,
    area: float = 100.0,
    reason: str | None = None,
    diagnostics: dict[str, object] | None = None,
) -> EunisLabelAttempt:
    return EunisLabelAttempt(
        inventory_row_id="row-1",
        polygon_snapshot_id="polygon-1",
        original_area=area,
        support_areas=areas,
        taxonomy=_TAXONOMY,
        source_revision="eea-manifest-v1",
        score_diagnostics=diagnostics,
        exclusion_reason=reason,
    )


def _write_raster(
    path: Path,
    values: list[list[float | int]],
    *,
    scale: float = 1.0,
    offset: float = 0.0,
) -> None:
    array = np.asarray(values, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs="EPSG:3035",
        transform=from_origin(0, 10, 10, 10),
        nodata=-9999.0,
    ) as dataset:
        dataset.write(array, 1)
        dataset.scales = (scale,)
        dataset.offsets = (offset,)


def test_eea_2021_reference_pin_records_verified_manifest_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    classification = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "C", "Root C"],
            [2, "C1", "Branch C1"],
            [3, "C11", "Class C11"],
        ]
    )
    cache_payload = b"fixture taxonomy cache bytes"
    workbook_payload = b"fixture workbook bytes"
    lineage_pin = ClassificationTaxonomyLineagePin(
        hashlib.sha256(cache_payload).hexdigest(),
        "fixture-classification-record",
        hashlib.sha256(workbook_payload).hexdigest(),
    )
    monkeypatch.setattr(
        eea,
        "parse_classification_taxonomy_workbook",
        lambda _payload, *, supplemental_labels=None: classification,
    )
    verified = verify_classification_taxonomy_lineage(
        classification,
        cache_payload,
        workbook_payload,
        ClassificationTaxonomyCacheLineage(
            lineage_pin.classification_record_id,
            lineage_pin.workbook_sha256,
        ),
        lineage_pin,
    )
    monkeypatch.setattr(pool_adapter, "EEA_2021_CLASSIFICATION_LINEAGE_PIN", lineage_pin)
    taxonomy = make_eunis_taxonomy_from_classification(
        verified,
        ("C11",),
        taxonomy_id="EUNIS 2021",
        taxonomy_revision="taxonomy-2021",
        taxonomy_sha256=lineage_pin.cache_sha256,
    )

    pin = make_eunis_2021_reference_pin(taxonomy)

    assert taxonomy.production_eligible
    assert pin.release_id == "EEA EUNIS habitat probability maps v1 2021"
    assert f"{EEA_2021_NATIVE_MANIFEST_SHA256}" in pin.source_revision
    assert f"asset-count:{EEA_2021_ASSET_COUNT}" in pin.source_revision
    assert f"asset-bytes:{EEA_2021_ASSET_BYTES}" in pin.source_revision
    assert pin.taxonomy_revision == taxonomy.pinned_revision


def test_eea_2021_reference_pin_rejects_an_unverified_taxonomy_digest() -> None:
    taxonomy = EunisTaxonomy(
        taxonomy_id="EUNIS 2021",
        taxonomy_revision="taxonomy-2021",
        taxonomy_sha256="a" * 64,
        paths_by_direct_code={"C1": EunisTaxonomyPath("A", "A1", "C1")},
    )

    with pytest.raises(ValueError, match="requires verified workbook lineage"):
        make_eunis_2021_reference_pin(taxonomy)


def test_taxonomy_groups_explicit_paths_without_guessing_code_prefixes() -> None:
    groups = _TAXONOMY.support_groups()

    assert groups[("L1", "A")] == ("C1", "C2")
    assert groups[("L2", "A1")] == ("C1",)
    assert groups[("L3", "C1")] == ("C1",)


def test_eunis_taxonomy_paths_come_from_classification_parent_edges() -> None:
    classification = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "X", "Root X"],
            [2, "X1", "Branch X1"],
            [3, "X11", "Class X11"],
            [3, "X12", "Class X12"],
        ]
    )

    taxonomy = make_eunis_taxonomy_from_classification(
        classification,
        ("X12", "X11"),
        taxonomy_id="fixture-eunis",
        taxonomy_revision="classification-record:fixture;workbook-sha256:" + "a" * 64,
        taxonomy_sha256="b" * 64,
    )

    assert taxonomy.paths_by_direct_code == {
        "X11": EunisTaxonomyPath("X", "X1", "X11"),
        "X12": EunisTaxonomyPath("X", "X1", "X12"),
    }
    assert taxonomy.support_groups()[("L2", "X1")] == ("X11", "X12")


def test_eunis_taxonomy_requires_a_complete_l1_l2_l3_path() -> None:
    classification = ClassificationTaxonomy(
        labels={"X": "Root X", "X11": "Class X11"},
        levels={"X": 1, "X11": 3},
        parents={"X11": "X"},
    )

    with pytest.raises(ValueError, match="lacks one L1-L2-L3 parent path"):
        make_eunis_taxonomy_from_classification(
            classification,
            ("X11",),
            taxonomy_id="fixture-eunis",
            taxonomy_revision="classification-fixture-v1",
            taxonomy_sha256="c" * 64,
        )

    with pytest.raises(ValueError, match="direct map codes must be present"):
        make_eunis_taxonomy_from_classification(
            classification,
            (),
            taxonomy_id="fixture-eunis",
            taxonomy_revision="classification-fixture-v1",
            taxonomy_sha256="c" * 64,
        )
    with pytest.raises(ValueError, match="is not a classification L3 code"):
        make_eunis_taxonomy_from_classification(
            classification,
            ("X",),
            taxonomy_id="fixture-eunis",
            taxonomy_revision="classification-fixture-v1",
            taxonomy_sha256="c" * 64,
        )


def test_taxonomy_and_source_tree_pins_reject_malformed_identity_fields() -> None:
    with pytest.raises(ValueError, match="id and revision must be present"):
        EunisTaxonomy(" ", "taxonomy-v1", "a" * 64, {"C1": EunisTaxonomyPath("A", "A1", "C1")})
    with pytest.raises(ValueError, match="SHA-256 must be lowercase hexadecimal"):
        EunisTaxonomy(
            "taxonomy", "taxonomy-v1", "G" * 64, {"C1": EunisTaxonomyPath("A", "A1", "C1")}
        )
    with pytest.raises(ValueError, match="mapping must contain direct map codes"):
        EunisTaxonomy("taxonomy", "taxonomy-v1", "a" * 64, {})
    with pytest.raises(ValueError, match="map code must map to itself"):
        EunisTaxonomy(
            "taxonomy", "taxonomy-v1", "a" * 64, {"C1": EunisTaxonomyPath("A", "A1", "C2")}
        )
    with pytest.raises(ValueError, match="contains an empty code"):
        EunisTaxonomy(
            "taxonomy", "taxonomy-v1", "a" * 64, {"C1": EunisTaxonomyPath(" ", "A1", "C1")}
        )

    with pytest.raises(ValueError, match="path and sizes must be valid"):
        SourceTreeFile(" ", 1, "a" * 40, None, 1)
    with pytest.raises(ValueError, match="Git SHA-1"):
        SourceTreeFile("data/a.parquet", 1, "G" * 40, None, 1)
    with pytest.raises(ValueError, match="LFS oid must be a lowercase SHA-256"):
        SourceTreeFile("data/a.parquet", 1, "a" * 40, "g" * 64, 1)


def test_source_tree_group_validator_checks_prefix_count_size_digest_and_duplicates() -> None:
    file = SourceTreeFile("data/a.parquet", 4, "a" * 40, None, 4)
    pin = SourceTreeGroupPin(
        repository="repo",
        revision="b" * 40,
        role="text",
        path_prefix="data/",
        file_count=1,
        total_size_bytes=4,
        inventory_sha256=source_tree_inventory_sha256((file,)),
    )

    validate_source_tree_group(pin, (file,))
    with pytest.raises(ValueError, match="fall outside pinned prefix"):
        validate_source_tree_group(replace(pin, path_prefix="labels/"), (file,))
    with pytest.raises(ValueError, match="file count differs"):
        validate_source_tree_group(replace(pin, file_count=2), (file,))
    with pytest.raises(ValueError, match="byte total differs"):
        validate_source_tree_group(replace(pin, total_size_bytes=5), (file,))
    with pytest.raises(ValueError, match="inventory digest differs"):
        validate_source_tree_group(replace(pin, inventory_sha256="c" * 64), (file,))
    with pytest.raises(ValueError, match="duplicate paths"):
        source_tree_inventory_sha256((file, file))
    with pytest.raises(ValueError, match="repository, role, and path prefix"):
        SourceTreeGroupPin("", "b" * 40, "text", "data/", 1, 4, "c" * 64)
    with pytest.raises(ValueError, match="full lowercase Git SHA"):
        SourceTreeGroupPin("repo", "G" * 40, "text", "data/", 1, 4, "c" * 64)
    with pytest.raises(ValueError, match="file count and byte total"):
        SourceTreeGroupPin("repo", "b" * 40, "text", "data/", 0, 4, "c" * 64)
    with pytest.raises(ValueError, match="inventory SHA-256 must be lowercase hexadecimal"):
        SourceTreeGroupPin("repo", "b" * 40, "text", "data/", 1, 4, "G" * 64)


def test_pool_sources_cover_the_three_raw_text_sources_with_wiki_artifacts_downstream() -> None:
    assert set(EUNIS_POOL_SOURCE_LAYOUTS) == {
        TextSource.DESCRIPTION,
        TextSource.WEBSITE,
        TextSource.WIKI,
    }
    wiki = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI]
    assert wiki.repository == "NoeFlandre/osm-polygon-wikidata-and-wikipedia"
    assert wiki.document_link_glob == "polygon_document_links/*.parquet"
    assert wiki.document_glob == "wikipedia/documents/*.parquet"
    assert wiki.sentence_glob == "wikivoyage/sentences/*.parquet"


def test_pool_source_layouts_pin_raw_and_relevance_snapshots_separately() -> None:
    expected = {
        TextSource.DESCRIPTION: (
            "NoeFlandre/osm-polygon-description-tag",
            "b4706eb315c66a8a57135289e7612dca8e03caf8",
            "data/*.parquet",
            "NoeFlandre/osm-polygon-description-tag-landuse",
            "ce98b328f3c6543800a3d931877b51eb3f0cf768",
            (("description", "labels/language-v1/data/*.parquet"),),
        ),
        TextSource.WEBSITE: (
            "NoeFlandre/osm-polygon-website-tag",
            "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
            "polygons/*.parquet",
            "NoeFlandre/osm-polygon-website-tag-landuse",
            "e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
            (("website", "labels/polygons/*.parquet"),),
        ),
        TextSource.WIKI: (
            "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
            "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
            "polygons/*.parquet",
            "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
            "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
            (
                ("wikipedia", "labels/wikipedia/sentences/*.parquet"),
                ("wikivoyage", "labels/wikivoyage/sentences/*.parquet"),
            ),
        ),
    }

    for source, pin in expected.items():
        layout = EUNIS_POOL_SOURCE_LAYOUTS[source]
        assert (
            layout.repository,
            layout.raw_revision,
            layout.polygon_glob,
            layout.relevance_repository,
            layout.relevance_revision,
            layout.relevance_path_globs,
        ) == pin


def test_threshold_is_inclusive_candidates_below_threshold_are_retained() -> None:
    candidates, outcomes = resolve_eunis_labels(
        _attempt(
            {
                ("L1", "A"): 80.0,
                ("L2", "A1"): 79.99,
                ("L3", "C1"): 80.0,
            }
        )
    )

    assert [(item.level, item.code) for item in candidates] == [
        ("L1", "A"),
        ("L2", "A1"),
        ("L3", "C1"),
    ]
    assert [item.ratio for item in candidates] == pytest.approx([0.8, 0.7999, 0.8])
    assert [(item.level, item.status, item.assigned_code) for item in outcomes] == [
        ("L1", OutcomeStatus.ASSIGNED, "A"),
        ("L2", OutcomeStatus.UNASSIGNED, None),
        ("L3", OutcomeStatus.ASSIGNED, "C1"),
    ]
    assert candidates[0].assignment_kind is AssignmentKind.ANCESTOR_DERIVED
    assert candidates[-1].assignment_kind is AssignmentKind.DIRECT
    assert all(item.denominator == 100.0 for item in candidates)


def test_each_level_is_resolved_independently_and_multi_class_is_ambiguous() -> None:
    _candidates, outcomes = resolve_eunis_labels(
        _attempt(
            {
                ("L1", "A"): 85.0,
                ("L1", "B"): 80.0,
                ("L2", "A1"): 85.0,
                ("L2", "B1"): 2.0,
                ("L3", "C1"): 79.99,
                ("L3", "C2"): 0.01,
            }
        )
    )

    by_level = {item.level: item for item in outcomes}
    assert by_level["L1"].status is OutcomeStatus.AMBIGUOUS
    assert by_level["L2"].status is OutcomeStatus.ASSIGNED
    assert by_level["L2"].assigned_code == "A1"
    assert by_level["L3"].status is OutcomeStatus.UNASSIGNED
    assert {item.name for item in EUNIS_POOL_LEVELS if not item.supported} == {
        "L4",
        "L5",
        "L6",
        "L7",
        "L8",
    }
    assert {item.level for item in outcomes} == {"L1", "L2", "L3"}


def test_no_data_remains_unassigned_with_full_polygon_as_denominator() -> None:
    candidates, outcomes = resolve_eunis_labels(_attempt({}))

    assert candidates == ()
    assert [item.status for item in outcomes] == [OutcomeStatus.UNASSIGNED] * 3
    assert {item.reason for item in outcomes} == {"no_positive_source_score"}


@pytest.mark.parametrize(
    ("areas", "diagnostics", "expected_reason"),
    [
        ({}, None, "no_positive_source_score"),
        (
            {},
            {"unknown_area_m2": 1.0},
            "unknown_or_tied_source_scores_prevent_unique_classification",
        ),
        (
            {("L3", "C1"): 79.0},
            None,
            "no_class_reaches_80_percent_of_full_polygon_area",
        ),
        (
            {("L3", "C1"): 79.0},
            {"unknown_area_m2": 1.0},
            "no_class_reaches_80_percent_with_unknown_or_tied_source_scores",
        ),
        (
            {("L3", "C1"): 79.0},
            {"unknown_area_m2": 0.0, "ambiguous_area_m2_by_level": {"L3": 1.0}},
            "no_class_reaches_80_percent_with_unknown_or_tied_source_scores",
        ),
    ],
)
def test_unassigned_reasons_distinguish_area_shortfall_from_source_uncertainty(
    areas: dict[tuple[str, str], float],
    diagnostics: dict[str, object] | None,
    expected_reason: str,
) -> None:
    _candidates, outcomes = resolve_eunis_labels(_attempt(areas, diagnostics=diagnostics))

    level_three = next(outcome for outcome in outcomes if outcome.level == "L3")
    assert level_three.reason == expected_reason


def test_label_attempts_retain_the_full_area_rule_and_reject_invalid_support() -> None:
    zero_support = _attempt({("L3", "C1"): 0.0})
    candidates, outcomes = resolve_eunis_labels(zero_support)
    assert candidates == ()
    assert {outcome.reason for outcome in outcomes} == {"no_positive_source_score"}

    with pytest.raises(ValueError, match="unsupported EUNIS support level"):
        resolve_eunis_labels(_attempt({("L4", "C1"): 90.0}))
    with pytest.raises(ValueError, match="outside full polygon area"):
        resolve_eunis_labels(_attempt({("L3", "C1"): 101.0}))
    with pytest.raises(ValueError, match="absent from pinned taxonomy"):
        resolve_eunis_labels(_attempt({("L3", "UNKNOWN"): 90.0}))
    with pytest.raises(TypeError, match="finite numeric values"):
        resolve_eunis_labels(_attempt({("L3", "C1"): float("nan")}))
    with pytest.raises(ValueError, match=r"fixed 0\.8 support threshold"):
        resolve_eunis_labels(replace(_attempt({("L3", "C1"): 90.0}), threshold=0.79))
    with pytest.raises(ValueError, match="both a query error and geometry exclusion"):
        resolve_eunis_labels(
            replace(
                _attempt({("L3", "C1"): 90.0}),
                query_error="raster failed",
                exclusion_reason="bad geometry",
            )
        )
    with pytest.raises(ValueError, match="inventory row id and EEA source revision"):
        resolve_eunis_labels(replace(_attempt({("L3", "C1"): 90.0}), inventory_row_id=" "))
    with pytest.raises(ValueError, match="usable polygon label attempt needs all class support"):
        resolve_eunis_labels(replace(_attempt({}), support_areas=None))
    missing_geometry = replace(
        _attempt({("L3", "C1"): 0.0}),
        polygon_snapshot_id=None,
        original_area=0.0,
    )
    _candidates, excluded = resolve_eunis_labels(missing_geometry)
    assert {outcome.reason for outcome in excluded} == {"usable_geometry_missing_full_area"}


def test_excluded_geometry_and_failed_reference_are_recorded_for_all_levels() -> None:
    geometry_candidates, geometry_outcomes = resolve_eunis_labels(
        EunisLabelAttempt(
            inventory_row_id="row-1",
            polygon_snapshot_id=None,
            original_area=None,
            support_areas=None,
            taxonomy=_TAXONOMY,
            source_revision="eea-manifest-v1",
            exclusion_reason="invalid_or_empty_polygon_geometry",
        )
    )
    query_candidates, query_outcomes = resolve_eunis_labels(
        EunisLabelAttempt(
            inventory_row_id="row-2",
            polygon_snapshot_id="polygon-2",
            original_area=100.0,
            support_areas=None,
            taxonomy=_TAXONOMY,
            source_revision="eea-manifest-v1",
            query_error="reference_query_error:RasterioIOError",
        )
    )

    assert geometry_candidates == query_candidates == ()
    assert [item.status for item in geometry_outcomes] == [OutcomeStatus.EXCLUDED] * 3
    assert {item.reason for item in geometry_outcomes} == {"invalid_or_empty_polygon_geometry"}
    assert {item.reason for item in query_outcomes} == {"reference_query_error:RasterioIOError"}


def test_taxonomy_map_codes_must_match_complete_direct_raster_inventory() -> None:
    with pytest.raises(ValueError, match="direct-code mismatch"):
        _TAXONOMY.validate_direct_codes(("C1", "C2", "new-direct-layer"))


def test_score_classification_keeps_ties_unknown_and_nodata_in_denominator(
    tmp_path: Path,
) -> None:
    first = tmp_path / "Prob_C1_10m.tif"
    second = tmp_path / "Prob_C2_10m.tif"
    _write_raster(first, [[0.9, -9999]])
    _write_raster(second, [[0.9, 0.0]])
    reference = RasterReference(
        (
            RasterLayer("C1", "child one", first, "EEA fixture"),
            RasterLayer("C2", "child two", second, "EEA fixture"),
        )
    )
    groups = {
        ("L1", "A"): ("C1", "C2"),
        ("L2", "A1"): ("C1",),
        ("L2", "A2"): ("C2",),
        ("L3", "C1"): ("C1",),
        ("L3", "C2"): ("C2",),
    }
    polygon = box(0, 0, 20, 10)

    with reference:
        result = reference.classification_areas(polygon, groups)
        areas = result.areas_m2
        legacy = reference.overlap(polygon)

    assert areas[("L1", "A")] == 100.0
    assert areas[("L2", "A1")] == 0.0
    assert areas[("L3", "C1")] == 0.0
    assert areas[("L3", "C2")] == 0.0
    assert areas[("L1", "A")] / polygon.area == 0.5
    assert result.diagnostics.unknown_area_m2 == 100.0
    assert result.diagnostics.ambiguous_area_m2_by_level["L2"] == 100.0
    assert legacy.overlap_percentage == 50.0


def test_highest_score_ties_resolve_only_at_a_shared_ancestor(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[0.9, 0.9]])
    _write_raster(paths["C2"], [[0.9, 0.1]])
    _write_raster(paths["C3"], [[0.1, 0.9]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )
    same_ancestor_groups = {
        ("L1", "A"): ("C1", "C2"),
        ("L1", "B"): ("C3",),
        ("L2", "A1"): ("C1", "C2"),
        ("L2", "B1"): ("C3",),
        ("L3", "C1"): ("C1",),
        ("L3", "C2"): ("C2",),
        ("L3", "C3"): ("C3",),
    }

    with reference:
        shared = reference.classification_areas(box(0, 0, 10, 10), same_ancestor_groups)
        cross_parent = reference.classification_areas(box(10, 0, 20, 10), same_ancestor_groups)

    assert shared.areas_m2[("L1", "A")] == 100.0
    assert shared.areas_m2[("L2", "A1")] == 100.0
    assert shared.areas_m2[("L3", "C1")] == 0.0
    assert shared.areas_m2[("L3", "C2")] == 0.0
    assert shared.diagnostics.ambiguous_area_m2_by_level == {
        "L1": 0.0,
        "L2": 0.0,
        "L3": 100.0,
    }
    assert all(area == 0.0 for area in cross_parent.areas_m2.values())
    assert cross_parent.diagnostics.ambiguous_area_m2_by_level == {
        "L1": 100.0,
        "L2": 100.0,
        "L3": 100.0,
    }


def test_exact_80_percent_accepts_known_class_with_remaining_unknown_area(
    tmp_path: Path,
) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[0.9, 0.9, 0.9, 0.9]])
    _write_raster(paths["C2"], [[0.1, 0.1, 0.1, 0.1]])
    _write_raster(paths["C3"], [[0.2, 0.2, 0.2, 0.2]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )
    polygon = box(0, 0, 50, 10)

    with reference:
        result = reference.classification_areas(polygon, _TAXONOMY.support_groups())

    candidates, outcomes = resolve_eunis_labels(
        _attempt(
            dict(result.areas_m2),
            area=polygon.area,
            diagnostics=result.diagnostics.as_dict(),
        )
    )
    assert result.areas_m2[("L3", "C1")] == 400.0
    assert result.diagnostics.unknown_area_m2 == 100.0
    assert next(item for item in candidates if item.level == "L3").ratio == 0.8
    assert next(item for item in outcomes if item.level == "L3").assigned_code == "C1"


def test_unknown_competing_score_prevents_classification_and_keeps_full_denominator(
    tmp_path: Path,
) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[0.9, 0.9, 0.9, 0.9]])
    _write_raster(paths["C2"], [[-9999.0, 0.1, 0.1, 0.1]])
    _write_raster(paths["C3"], [[0.2, 0.2, 0.2, 0.2]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )
    polygon = box(0, 0, 40, 10)

    with reference:
        result = reference.classification_areas(polygon, _TAXONOMY.support_groups())

    candidates, outcomes = resolve_eunis_labels(
        _attempt(
            dict(result.areas_m2),
            area=polygon.area,
            diagnostics=result.diagnostics.as_dict(),
        )
    )
    assert result.areas_m2[("L3", "C1")] == 300.0
    assert result.diagnostics.unknown_area_m2 == 100.0
    assert not any(
        item.level == "L3" and item.code == "C1" and item.ratio >= 0.8 for item in candidates
    )
    assert next(item for item in outcomes if item.level == "L3").status is OutcomeStatus.UNASSIGNED
    assert next(item for item in outcomes if item.level == "L3").reason == (
        "no_class_reaches_80_percent_with_unknown_or_tied_source_scores"
    )


def test_fractional_cell_boundary_uses_intersection_area(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[0.9, 0.1]])
    _write_raster(paths["C2"], [[0.1, 0.9]])
    _write_raster(paths["C3"], [[0.1, 0.1]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )
    polygon = box(0, 0, 15, 10)

    with reference:
        result = reference.classification_areas(polygon, _TAXONOMY.support_groups())

    assert polygon.area == 150.0
    assert result.areas_m2[("L3", "C1")] == 100.0
    assert result.areas_m2[("L3", "C2")] == 50.0


@pytest.mark.parametrize(("scale", "offset"), [(0.0001, 0.0), (1.0, 0.01)])
def test_undocumented_scale_or_offset_fails_closed(
    tmp_path: Path,
    scale: float,
    offset: float,
) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[0.9]], scale=scale, offset=offset)
    _write_raster(paths["C2"], [[0.1]])
    _write_raster(paths["C3"], [[0.2]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )
    with reference, pytest.raises(ValueError, match="undocumented scale or offset"):
        reference.classification_areas(box(0, 0, 10, 10), _TAXONOMY.support_groups())


def test_scores_outside_documented_range_fail_closed(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    _write_raster(paths["C1"], [[1.1]])
    _write_raster(paths["C2"], [[0.1]])
    _write_raster(paths["C3"], [[0.2]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )

    with reference, pytest.raises(ValueError, match="outside the documented 0-1 range"):
        reference.classification_areas(box(0, 0, 10, 10), _TAXONOMY.support_groups())


@pytest.mark.parametrize(
    ("band_count", "dtype", "expected_error"),
    [(2, "float32", "exactly one band"), (1, "complex64", "real numeric band")],
)
def test_score_decoding_rejects_multiband_and_complex_rasters(
    tmp_path: Path,
    band_count: int,
    dtype: str,
    expected_error: str,
) -> None:
    path = tmp_path / "Prob_C1_10m.tif"
    values = np.ones((band_count, 1, 1), dtype=dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=band_count,
        dtype=dtype,
        crs="EPSG:3035",
        transform=from_origin(0, 10, 10, 10),
    ) as dataset:
        dataset.write(values)
    reference = RasterReference((RasterLayer("C1", "class one", path, "EEA fixture"),))
    groups = {
        ("L1", "A"): ("C1",),
        ("L2", "A1"): ("C1",),
        ("L3", "C1"): ("C1",),
    }

    with reference, pytest.raises(ValueError, match=expected_error):
        reference.classification_areas(box(0, 0, 10, 10), groups)


def test_score_decoding_rejects_duplicate_direct_codes(tmp_path: Path) -> None:
    path = tmp_path / "Prob_C1_10m.tif"
    _write_raster(path, [[0.9]])
    reference = RasterReference(
        (
            RasterLayer("C1", "class one", path, "EEA fixture"),
            RasterLayer("C1", "class one duplicate", path, "EEA fixture"),
        )
    )
    groups = {
        ("L1", "A"): ("C1",),
        ("L2", "A1"): ("C1",),
        ("L3", "C1"): ("C1",),
    }

    with reference, pytest.raises(ValueError, match="raster codes must be unique"):
        reference.classification_areas(box(0, 0, 10, 10), groups)


def test_support_query_fails_closed_when_a_class_uses_an_unpinned_layer(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "Prob_C1_10m.tif"
    _write_raster(raster, [[1.0, 0.0]])
    reference = RasterReference((RasterLayer("C1", "class one", raster, "EEA fixture"),))

    with reference, pytest.raises(ValueError, match="missing raster codes"):
        reference.support_areas(box(0, 0, 20, 10), {("L1", "A"): ("MISSING",)})


def test_classification_requires_every_competing_raster(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2")}
    _write_raster(paths["C1"], [[0.9]])
    _write_raster(paths["C2"], [[0.1]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, path, "EEA fixture") for code, path in paths.items())
    )

    with reference, pytest.raises(ValueError, match=r"missing raster codes: \['C3'\]"):
        reference.classification_areas(box(0, 0, 10, 10), _TAXONOMY.support_groups())


def test_classification_rejects_unmapped_codes_and_incomplete_hierarchy(
    tmp_path: Path,
) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2")}
    for path in paths.values():
        _write_raster(path, [[0.9]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, path, "EEA fixture") for code, path in paths.items())
    )
    polygon = box(0, 0, 10, 10)

    with pytest.raises(ValueError, match="omits competing raster codes"):
        reference.classification_areas(
            polygon,
            {
                ("L1", "A"): ("C1",),
                ("L2", "A1"): ("C1",),
                ("L3", "C1"): ("C1",),
            },
        )
    with pytest.raises(ValueError, match="needs explicit L1, L2, and L3 ancestry"):
        reference.classification_areas(
            polygon,
            {("L1", "A"): ("C1", "C2"), ("L2", "A1"): ("C1", "C2")},
        )


def test_classification_rejects_conflicting_or_missing_level_ancestry(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2")}
    for path in paths.values():
        _write_raster(path, [[0.9]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, path, "EEA fixture") for code, path in paths.items())
    )
    polygon = box(0, 0, 10, 10)

    with pytest.raises(ValueError, match="conflicting L1 ancestors"):
        reference.classification_areas(
            polygon,
            {
                ("L1", "A"): ("C1",),
                ("L1", "B"): ("C1",),
                ("L2", "A1"): ("C1", "C2"),
                ("L3", "C1"): ("C1",),
                ("L3", "C2"): ("C2",),
            },
        )
    with pytest.raises(ValueError, match="does not map every competing raster code at L1"):
        reference.classification_areas(
            polygon,
            {
                ("L1", "A"): ("C1",),
                ("L2", "A1"): ("C1", "C2"),
                ("L3", "C1"): ("C1",),
                ("L3", "C2"): ("C2",),
            },
        )
    with pytest.raises(ValueError, match="must be its own L3 class"):
        reference.classification_areas(
            polygon,
            {
                ("L1", "A"): ("C1", "C2"),
                ("L2", "A1"): ("C1", "C2"),
                ("L3", "OTHER"): ("C1",),
                ("L3", "C2"): ("C2",),
            },
        )


def test_zero_source_scores_do_not_create_a_habitat_label(tmp_path: Path) -> None:
    paths = {code: tmp_path / f"Prob_{code}_10m.tif" for code in ("C1", "C2", "C3")}
    for path in paths.values():
        _write_raster(path, [[0.0]])
    reference = RasterReference(
        tuple(RasterLayer(code, code, paths[code], "EEA fixture") for code in paths)
    )

    with reference:
        result = reference.classification_areas(box(0, 0, 10, 10), _TAXONOMY.support_groups())

    assert not any(result.areas_m2.values())
    assert result.diagnostics.zero_max_score_area_m2 == 100.0
    assert result.diagnostics.unknown_area_m2 == 0.0
    _, outcomes = resolve_eunis_labels(
        _attempt(dict(result.areas_m2), diagnostics=result.diagnostics.as_dict())
    )
    assert {outcome.reason for outcome in outcomes} == {"no_positive_source_score"}


def test_support_query_rejects_empty_geometry_and_malformed_class_groups() -> None:
    reference = RasterReference(())
    polygon = box(0, 0, 10, 10)

    with pytest.raises(ValueError, match="usable projected polygon"):
        reference.support_areas(box(0, 0, 0, 0), {("L1", "A"): ("C1",)})
    with pytest.raises(ValueError, match="positive full polygon area"):
        reference.support_areas(polygon.boundary, {("L1", "A"): ("C1",)})
    with pytest.raises(ValueError, match="at least one support class"):
        reference.support_areas(polygon, {})
    with pytest.raises(ValueError, match=r"non-empty \(level, code\) pairs"):
        reference.support_areas(
            polygon,
            cast(dict[tuple[str, str], tuple[str, ...]], {"L1": ("C1",)}),
        )
    with pytest.raises(ValueError, match="needs non-empty raster codes"):
        reference.support_areas(polygon, {("L1", "A"): ()})


def test_missing_document_links_stay_downstream_of_the_full_inventory() -> None:
    first_key = canonical_native_key({"source_pbf": "a.osm.pbf", "osm_type": "way", "osm_id": 1})
    second_key = canonical_native_key(
        {"source_pbf": "b.osm.pbf", "osm_type": "relation", "osm_id": 2}
    )
    rows = (
        PolygonInventoryRow("row-1", first_key, "polygons/a.parquet#row=0", "polygon-1"),
        PolygonInventoryRow(
            "row-2", second_key, "polygons/a.parquet#row=1", None, "missing_geometry"
        ),
    )
    links = resolve_polygon_document_links(
        (
            PolygonDocumentEvidence("doc-linked", first_key, "source link table row 10"),
            PolygonDocumentEvidence("doc-geometry-missing", second_key, "source link table row 11"),
            PolygonDocumentEvidence(
                "doc-unmatched",
                canonical_native_key(
                    {"source_pbf": "outside.osm.pbf", "osm_type": "way", "osm_id": 9}
                ),
                "source link table row 12",
            ),
        ),
        rows,
        source_revision="wiki-revision-1",
    )

    assert len(rows) == 2
    assert {item.status for item in links} == {
        TextLinkStatus.LINKED,
        TextLinkStatus.GEOMETRY_MISSING,
        TextLinkStatus.UNMATCHED,
    }
    by_document = {item.document_id: item for item in links}
    assert by_document["doc-linked"].inventory_row_id == "row-1"
    assert by_document["doc-linked"].candidate_inventory_row_ids == ()
    assert by_document["doc-geometry-missing"].inventory_row_id is None
    assert by_document["doc-geometry-missing"].candidate_inventory_row_ids == ("row-2",)
    assert by_document["doc-unmatched"].inventory_row_id is None


def test_document_links_preserve_ambiguous_and_reject_invalid_evidence() -> None:
    linked_key = canonical_native_key(
        {"source_pbf": "region.osm.pbf", "osm_type": "way", "osm_id": 1}
    )
    ambiguous_key = canonical_native_key(
        {"source_pbf": "other-region.osm.pbf", "osm_type": "relation", "osm_id": 3}
    )
    inventory = (
        PolygonInventoryRow("inv-linked", linked_key, "raw/0", "polygon-1"),
        PolygonInventoryRow("inv-a", ambiguous_key, "raw/1", "polygon-3a"),
        PolygonInventoryRow("inv-b", ambiguous_key, "raw/2", "polygon-3b"),
    )
    evidence = (
        PolygonDocumentEvidence("doc-linked", linked_key, "direct key"),
        PolygonDocumentEvidence("doc-ambiguous", ambiguous_key, "shared source link"),
    )

    links = resolve_polygon_document_links(evidence, inventory, source_revision="pinned-v1")

    by_document = {link.document_id: link for link in links}
    assert by_document["doc-linked"].status is TextLinkStatus.LINKED
    assert by_document["doc-ambiguous"].status is TextLinkStatus.AMBIGUOUS
    assert by_document["doc-ambiguous"].candidate_inventory_row_ids == ("inv-a", "inv-b")

    with pytest.raises(ValueError, match="revision must be present"):
        resolve_polygon_document_links(evidence, inventory, source_revision=" ")
    with pytest.raises(ValueError, match="document id and evidence"):
        resolve_polygon_document_links(
            (PolygonDocumentEvidence("", linked_key, "evidence"),),
            inventory,
            source_revision="pinned-v1",
        )
    with pytest.raises(ValueError, match="duplicate source link evidence"):
        resolve_polygon_document_links(
            (evidence[0], evidence[0]),
            inventory,
            source_revision="pinned-v1",
        )
