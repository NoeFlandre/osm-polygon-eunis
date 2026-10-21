from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from osm_polygon_eunis.pool_adapter import (
    EEA_2021_ASSET_BYTES,
    EEA_2021_ASSET_COUNT,
    EEA_2021_NATIVE_MANIFEST_SHA256,
    EEA_2021_TAXONOMY_SHA256,
    EUNIS_POOL_LEVELS,
    EUNIS_POOL_SOURCE_LAYOUTS,
    EunisLabelAttempt,
    EunisTaxonomy,
    EunisTaxonomyPath,
    PolygonDocumentEvidence,
    make_eunis_2021_reference_pin,
    resolve_eunis_labels,
    resolve_polygon_document_links,
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
) -> EunisLabelAttempt:
    return EunisLabelAttempt(
        inventory_row_id="row-1",
        polygon_snapshot_id="polygon-1",
        original_area=area,
        support_areas=areas,
        taxonomy=_TAXONOMY,
        source_revision="eea-manifest-v1",
        exclusion_reason=reason,
    )


def _write_raster(path: Path, values: list[list[int]]) -> None:
    array = np.asarray(values, dtype=np.int16)
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
        nodata=-9999,
    ) as dataset:
        dataset.write(array, 1)


def test_eea_2021_reference_pin_records_verified_manifest_metrics() -> None:
    taxonomy = EunisTaxonomy(
        taxonomy_id="EUNIS 2021",
        taxonomy_revision="taxonomy-2021",
        taxonomy_sha256=EEA_2021_TAXONOMY_SHA256,
        paths_by_direct_code={"C1": EunisTaxonomyPath("A", "A1", "C1")},
    )

    pin = make_eunis_2021_reference_pin(taxonomy)

    assert pin.release_id == "EEA EUNIS habitat probability maps v1 2021"
    assert f"{EEA_2021_NATIVE_MANIFEST_SHA256}" in pin.source_revision
    assert f"asset-count:{EEA_2021_ASSET_COUNT}" in pin.source_revision
    assert f"asset-bytes:{EEA_2021_ASSET_BYTES}" in pin.source_revision
    assert pin.taxonomy_revision == taxonomy.pinned_revision


def test_taxonomy_groups_explicit_paths_without_guessing_code_prefixes() -> None:
    groups = _TAXONOMY.support_groups()

    assert groups[("L1", "A")] == ("C1", "C2")
    assert groups[("L2", "A1")] == ("C1",)
    assert groups[("L3", "C1")] == ("C1",)


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
    assert {item.reason for item in outcomes} == {"no_positive_reference_support"}


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


def test_support_query_unions_same_class_layers_and_keeps_nodata_in_denominator(
    tmp_path: Path,
) -> None:
    first = tmp_path / "Prob_C1_10m.tif"
    second = tmp_path / "Prob_C2_10m.tif"
    _write_raster(first, [[1, -9999]])
    _write_raster(second, [[1, 0]])
    reference = RasterReference(
        (
            RasterLayer("C1", "child one", first, "EEA fixture"),
            RasterLayer("C2", "child two", second, "EEA fixture"),
        )
    )
    groups = {
        ("L1", "A"): ("C1", "C2"),
        ("L2", "A1"): ("C1",),
        ("L3", "C1"): ("C1",),
        ("L3", "C2"): ("C2",),
    }
    polygon = box(0, 0, 20, 10)

    with reference:
        areas = reference.support_areas(polygon, groups)
        legacy = reference.overlap(polygon)

    assert areas[("L1", "A")] == 100.0
    assert areas[("L2", "A1")] == 100.0
    assert areas[("L3", "C1")] == 100.0
    assert areas[("L3", "C2")] == 100.0
    assert areas[("L1", "A")] / polygon.area == 0.5
    assert legacy.overlap_percentage == 50.0


def test_support_query_fails_closed_when_a_class_uses_an_unpinned_layer(
    tmp_path: Path,
) -> None:
    raster = tmp_path / "Prob_C1_10m.tif"
    _write_raster(raster, [[1, 0]])
    reference = RasterReference((RasterLayer("C1", "class one", raster, "EEA fixture"),))

    with reference, pytest.raises(ValueError, match="missing raster codes"):
        reference.support_areas(box(0, 0, 20, 10), {("L1", "A"): ("MISSING",)})


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
