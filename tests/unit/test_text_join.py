from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from osm_polygon_eunis.pool_adapter import (
    EUNIS_POOL_SOURCE_LAYOUTS,
    SourceTreeFile,
    SourceTreeGroupPin,
    source_tree_inventory_sha256,
)
from osm_polygon_eunis.pool_contract import (
    BodyStatus,
    PolygonInventoryRow,
    RelevanceStatus,
    TextLinkStatus,
    TextSource,
    canonical_native_key,
)
from osm_polygon_eunis.text_join import (
    EunisPinnedParquetFile,
    EunisPinnedTable,
    PinnedSourceRow,
    iter_eunis_join_batches,
    iter_eunis_join_records,
    missing_wiki_join_tree_groups,
)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _description_identity(osm_type: str, osm_id: int, tag_key: str, text_sha256: str) -> str:
    payload = json.dumps(
        [osm_type, osm_id, tag_key, text_sha256],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _inventory(
    source_pbf: str,
    osm_type: str,
    osm_id: int,
    inventory_row_id: str,
    polygon_snapshot_id: str | None,
) -> PolygonInventoryRow:
    return PolygonInventoryRow(
        inventory_row_id=inventory_row_id,
        native_key=canonical_native_key(
            {"source_pbf": source_pbf, "osm_type": osm_type, "osm_id": osm_id}
        ),
        source_row_locator=f"polygons/{source_pbf}#row=0",
        polygon_snapshot_id=polygon_snapshot_id,
    )


def _pinned_row(path: str, ordinal: int, values: dict) -> PinnedSourceRow:
    if path.startswith("language-v1/data/") and "description_identity" in values:
        values = {
            "split_status": "split" if values.get("sentences") else "empty",
            "split_reason": "" if values.get("sentences") else "fixture-no-output",
            **values,
        }
    if "/sentences/" in path and "sentence_id" in values:
        values = {
            "segmentation_status": "success" if values.get("text") else "empty",
            **values,
        }
    return PinnedSourceRow(path, ordinal, values)


def test_description_join_uses_stable_identity_original_hash_and_preserves_gaps() -> None:
    source_pbf = "region-latest.osm.pbf"
    full_text = "The old stone mill. This one is closed."
    original_hash = _sha(full_text)
    identity = _description_identity("way", 17, "description", original_hash)
    second_text = "Un texte."
    second_hash = _sha(second_text)
    second_identity = _description_identity("relation", 29, "description:fr", second_hash)
    sentence0 = "The old stone mill."
    sentence1 = "This one is closed."
    raw = [
        _pinned_row(
            "language-v1/data/region.parquet",
            0,
            {
                "description_identity": identity,
                "source_pbf": source_pbf,
                "osm_type": "way",
                "osm_id": 17,
                "tag_key": "description",
                "original_text": full_text,
                "text_sha256": original_hash,
                "language_code": "eng",
                "sentences": [sentence0, sentence1],
            },
        ),
        _pinned_row(
            "language-v1/data/other.parquet",
            0,
            {
                "description_identity": second_identity,
                "source_pbf": "not-in-inventory.osm.pbf",
                "osm_type": "relation",
                "osm_id": 29,
                "tag_key": "description:fr",
                "original_text": second_text,
                "text_sha256": second_hash,
                "language_code": "fra",
                "sentences": [second_text],
            },
        ),
    ]
    labels = [
        _pinned_row(
            "labels/language-v1/data/region.parquet",
            index,
            {
                "label_id": label_id,
                "description_identity": identity,
                "tag_key": "description",
                "sentence_index": 0 if index == 0 else 1,
                "text_sha256": original_hash,
                "decision": decision,
                "language": "eng",
            },
        )
        for index, (label_id, decision) in enumerate(
            (("yes-0", "YES"), ("dup-1a", "yes"), ("dup-1b", "no"))
        )
    ]
    labels.append(
        _pinned_row(
            "labels/language-v1/data/orphan.parquet",
            0,
            {
                "label_id": "orphan-no",
                "description_identity": "a" * 64,
                "tag_key": "description",
                "sentence_index": 4,
                "text_sha256": _sha("not present"),
                "decision": "NO",
                "language": "eng",
            },
        )
    )
    inventory = [_inventory(source_pbf, "way", 17, "inv-17", "poly-17")]

    records = list(
        iter_eunis_join_records(
            TextSource.DESCRIPTION,
            raw,
            labels,
            inventory,
        )
    )

    assert len(records) == 4
    matched = records[:2]
    assert [record.relevance.status for record in matched] == [
        RelevanceStatus.YES,
        RelevanceStatus.UNKNOWN,
    ]
    assert matched[0].relevance.raw_value == "YES"
    assert matched[0].document.content_sha256 == _sha(sentence0)
    assert matched[0].relevance.evidence_sha256 == original_hash
    assert matched[1].relevance.reason == "multiple relevance rows match the source text key"
    duplicate_evidence = json.loads(matched[1].relevance.evidence)
    assert {row["label_id"] for row in duplicate_evidence["label_rows"]} == {
        "dup-1a",
        "dup-1b",
    }
    assert matched[1].relevance.evidence_sha256 == original_hash
    assert matched[0].links[0].status is TextLinkStatus.LINKED
    assert matched[0].links[0].inventory_row_id == "inv-17"
    assert records[2].relevance.status is RelevanceStatus.MISSING
    assert records[2].links[0].status is TextLinkStatus.UNMATCHED
    orphan = records[3]
    assert orphan.document.body_status is BodyStatus.MISSING
    assert orphan.relevance.status is RelevanceStatus.MISSING
    assert orphan.relevance.raw_value == "NO"
    assert orphan.relevance.reason == (
        "orphan relevance row preserves its source decision but has no matching source text row"
    )
    assert orphan.links[0].status is TextLinkStatus.UNMATCHED

    serialized = json.dumps([asdict(record) for record in records], default=str)
    for body in (sentence0, sentence1, "Un texte.", full_text):
        assert body not in serialized


def test_description_rejects_source_hash_and_identity_drift() -> None:
    text = "A description"
    text_hash = _sha(text)
    identity = _description_identity("way", 7, "description", text_hash)
    row = _pinned_row(
        "language-v1/data/region.parquet",
        0,
        {
            "description_identity": identity,
            "source_pbf": "region.osm.pbf",
            "osm_type": "way",
            "osm_id": 7,
            "tag_key": "description",
            "original_text": text,
            "text_sha256": "0" * 64,
            "language_code": "eng",
            "sentences": [text],
        },
    )
    with pytest.raises(ValueError, match="text hash does not match"):
        list(iter_eunis_join_records(TextSource.DESCRIPTION, [row], [], []))


def test_website_join_requires_explicit_field_mapping_and_keeps_unmatched_labels() -> None:
    raw = [
        _pinned_row(
            "polygons/region.parquet",
            0,
            {
                "polygon_id": "region:way:42",
                "source_pbf": "region-latest.osm.pbf",
                "osm_type": "way",
                "osm_id": 42,
                "website_sentences": ["Visit the old town."],
                "contact_website_sentences": ["Call the guide."],
                "website_sentence_status": "success",
                "contact_website_sentence_status": "success",
                "website_text": "Visit the old town.",
                "contact_website_text": "Call the guide.",
                "website_text_status": "ok",
                "contact_website_text_status": "ok",
            },
        )
    ]
    labels = [
        _pinned_row(
            "labels/polygons/region.parquet",
            0,
            {
                "label_id": "website-yes",
                "polygon_id": "region:way:42",
                "field": "website",
                "sentence_index": 0,
                "text_sha256": _sha("Visit the old town."),
                "decision": "yes",
            },
        ),
        _pinned_row(
            "labels/polygons/orphan.parquet",
            0,
            {
                "label_id": "unmapped-field",
                "polygon_id": "region:way:99",
                "field": "unmapped-field",
                "sentence_index": 0,
                "text_sha256": _sha("missing raw sentence"),
                "decision": "perhaps",
            },
        ),
    ]
    inventory = [_inventory("region-latest.osm.pbf", "way", 42, "inv-42", None)]

    with pytest.raises(ValueError, match="field-to-sentence-column mapping"):
        list(iter_eunis_join_records(TextSource.WEBSITE, raw, labels, inventory))
    with pytest.raises(ValueError, match="cover both pinned sentence columns"):
        list(
            iter_eunis_join_records(
                TextSource.WEBSITE,
                raw,
                labels,
                inventory,
                website_field_to_column={"website": "website_sentences"},
            )
        )

    records = list(
        iter_eunis_join_records(
            TextSource.WEBSITE,
            raw,
            labels,
            inventory,
            website_field_to_column={
                "website": "website_sentences",
                "contact:website": "contact_website_sentences",
            },
        )
    )
    assert len(records) == 3
    website_record = next(
        record
        for record in records
        if record.document.content_sha256 == _sha("Visit the old town.")
    )
    contact_record = next(
        record for record in records if record.document.content_sha256 == _sha("Call the guide.")
    )
    assert website_record.relevance.status is RelevanceStatus.YES
    assert website_record.document.content_sha256 == _sha("Visit the old town.")
    assert website_record.relevance.evidence_sha256 == _sha("Visit the old town.")
    assert website_record.links[0].status is TextLinkStatus.GEOMETRY_MISSING
    assert website_record.links[0].candidate_inventory_row_ids == ("inv-42",)
    assert contact_record.relevance.status is RelevanceStatus.MISSING
    assert contact_record.links[0].status is TextLinkStatus.GEOMETRY_MISSING
    assert records[2].relevance.status is RelevanceStatus.UNKNOWN
    assert records[2].document.body_status is BodyStatus.MISSING
    assert records[2].links[0].status is TextLinkStatus.UNMATCHED


def test_wiki_join_uses_sentence_hash_and_explicit_document_links() -> None:
    text = "The museum opens on Tuesday."
    raw = [
        _pinned_row(
            "wikipedia/sentences/region.parquet",
            0,
            {
                "sentence_id": "sentence-1",
                "document_id": "document-1",
                "project": "wikipedia",
                "language": "en",
                "page_id": 501,
                "revision_id": 9,
                "text": text,
                "content_hash": _sha(text),
            },
        ),
        _pinned_row(
            "wikipedia/sentences/region.parquet",
            1,
            {
                "sentence_id": "sentence-2",
                "document_id": "document-no-link",
                "project": "wikipedia",
                "language": "en",
                "page_id": 502,
                "revision_id": 2,
                "text": "A second sentence.",
                "content_hash": _sha("A second sentence."),
            },
        ),
    ]
    labels = [
        _pinned_row(
            "labels/wikipedia/sentences/region.parquet",
            0,
            {
                "label_id": "label-1",
                "sentence_id": "sentence-1",
                "text_sha256": _sha(text),
                "decision": "YES",
            },
        ),
        _pinned_row(
            "labels/wikipedia/sentences/orphan.parquet",
            0,
            {
                "label_id": "label-orphan",
                "sentence_id": "not-in-sentences",
                "text_sha256": _sha("not here"),
                "decision": "NO",
            },
        ),
    ]
    links = [
        _pinned_row(
            "polygon_document_links/region.parquet",
            0,
            {
                "document_id": "document-1",
                "project": "wikipedia",
                "polygon_id": "region:way:17",
                "source_pbf": "region-latest.osm.pbf",
                "osm_type": "way",
                "osm_id": 17,
                "link_sources": '["wikidata","wikipedia"]',
            },
        ),
        _pinned_row(
            "polygon_document_links/region.parquet",
            1,
            {
                "document_id": "document-no-link",
                "project": "wikipedia",
                "polygon_id": "other-region:relation:88",
                "source_pbf": "other-region.osm.pbf",
                "osm_type": "relation",
                "osm_id": 88,
                "link_sources": '["wikipedia"]',
            },
        ),
        _pinned_row(
            "polygon_document_links/region.parquet",
            2,
            {
                "document_id": "document-without-sentences",
                "project": "wikipedia",
                "polygon_id": "third-region:way:94",
                "source_pbf": "third-region.osm.pbf",
                "osm_type": "way",
                "osm_id": 94,
                "link_sources": '["wikipedia"]',
            },
        ),
    ]
    inventory = [
        _inventory("region-latest.osm.pbf", "way", 17, "inv-17", "poly-17"),
        _inventory("other-region.osm.pbf", "relation", 88, "inv-88", None),
        _inventory("third-region.osm.pbf", "way", 94, "inv-94", "poly-94"),
    ]

    records = list(
        iter_eunis_join_records(
            TextSource.WIKI,
            raw,
            labels,
            inventory,
            wiki_link_rows=links,
            wiki_corpus="wikipedia",
        )
    )

    assert len(records) == 4
    assert records[0].relevance.status is RelevanceStatus.YES
    assert records[0].relevance.evidence_sha256 == _sha(text)
    assert records[0].links[0].status is TextLinkStatus.LINKED
    assert records[1].relevance.status is RelevanceStatus.MISSING
    assert records[1].links[0].status is TextLinkStatus.GEOMETRY_MISSING
    orphan = records[2]
    assert orphan.relevance.status is RelevanceStatus.MISSING
    assert orphan.relevance.raw_value == "NO"
    assert orphan.document.body_status is BodyStatus.MISSING
    assert orphan.links[0].status is TextLinkStatus.UNMATCHED
    orphan_link = records[3]
    assert orphan_link.relevance.status is RelevanceStatus.MISSING
    assert orphan_link.relevance.reason == "Wiki polygon-document link has no matching sentence row"
    assert orphan_link.links[0].status is TextLinkStatus.LINKED
    assert orphan_link.links[0].inventory_row_id == "inv-94"


def test_empty_description_sentence_output_retains_an_explicit_missing_reason() -> None:
    text = "Text with no sentence output"
    text_hash = _sha(text)
    row = _pinned_row(
        "language-v1/data/region.parquet",
        0,
        {
            "description_identity": _description_identity("way", 8, "description", text_hash),
            "source_pbf": "region.osm.pbf",
            "osm_type": "way",
            "osm_id": 8,
            "tag_key": "description",
            "original_text": text,
            "text_sha256": text_hash,
            "language_code": "eng",
            "sentences": [],
        },
    )

    [record] = list(iter_eunis_join_records(TextSource.DESCRIPTION, [row], [], []))

    assert record.document.body_status is BodyStatus.PRIVATE
    assert record.relevance.status is RelevanceStatus.MISSING
    assert record.relevance.reason == (
        "pinned Description row has no sentence output; sentence array=empty, "
        "split status=empty, split reason=fixture-no-output; no sentence-level "
        "relevance assessment applies"
    )


def test_raw_polygon_without_any_text_row_gets_an_explicit_missing_join_record() -> None:
    polygon = _inventory("region.osm.pbf", "way", 51, "inv-51", "poly-51")

    [record] = list(
        iter_eunis_join_records(
            TextSource.DESCRIPTION,
            [],
            [],
            [polygon],
        )
    )

    assert record.document.body_status is BodyStatus.MISSING
    assert record.relevance.status is RelevanceStatus.MISSING
    assert record.relevance.reason == (
        "no downstream text or relevance assessment exists for this raw polygon"
    )
    assert record.links[0].status is TextLinkStatus.UNMATCHED
    assert "raw polygon" in record.links[0].evidence


def test_wiki_join_rejects_stored_content_hash_mismatch() -> None:
    row = _pinned_row(
        "wikivoyage/sentences/region.parquet",
        0,
        {
            "sentence_id": "sentence-1",
            "document_id": "document-1",
            "project": "en.wikivoyage.org",
            "language": "en",
            "page_id": 1,
            "revision_id": 2,
            "text": "A sentence.",
            "content_hash": "0" * 64,
        },
    )
    with pytest.raises(ValueError, match="content hash differs"):
        list(
            iter_eunis_join_records(
                TextSource.WIKI,
                [row],
                [],
                [],
                wiki_corpus="wikivoyage",
            )
        )


def test_wiki_corpora_are_joined_separately_without_cross_scope_orphans() -> None:
    first_text = "A Wikipedia sentence."
    second_text = "A Wikivoyage sentence."
    inventory = [
        _inventory("wikipedia.osm.pbf", "way", 10, "wiki-10", "poly-10"),
        _inventory("wikivoyage.osm.pbf", "way", 20, "voyage-20", "poly-20"),
        _inventory("no-text.osm.pbf", "relation", 30, "no-text-30", "poly-30"),
    ]
    links = [
        _pinned_row(
            "polygon_document_links/all.parquet",
            0,
            {
                "document_id": "wikipedia-document",
                "project": "wikipedia",
                "polygon_id": "wikipedia:way:10",
                "source_pbf": "wikipedia.osm.pbf",
                "osm_type": "way",
                "osm_id": 10,
                "link_sources": '["wikipedia"]',
            },
        ),
        _pinned_row(
            "polygon_document_links/all.parquet",
            1,
            {
                "document_id": "wikivoyage-document",
                "project": "wikivoyage",
                "polygon_id": "wikivoyage:way:20",
                "source_pbf": "wikivoyage.osm.pbf",
                "osm_type": "way",
                "osm_id": 20,
                "link_sources": '["wikivoyage"]',
            },
        ),
    ]

    wikipedia_records = list(
        iter_eunis_join_records(
            TextSource.WIKI,
            [
                _pinned_row(
                    "wikipedia/sentences/one.parquet",
                    0,
                    {
                        "sentence_id": "wiki-sentence",
                        "document_id": "wikipedia-document",
                        "project": "wikipedia",
                        "language": "en",
                        "page_id": 10,
                        "revision_id": 1,
                        "text": first_text,
                        "content_hash": _sha(first_text),
                    },
                )
            ],
            [],
            inventory,
            wiki_link_rows=links,
            wiki_corpus="wikipedia",
        )
    )
    wikivoyage_records = list(
        iter_eunis_join_records(
            TextSource.WIKI,
            [
                _pinned_row(
                    "wikivoyage/sentences/one.parquet",
                    0,
                    {
                        "sentence_id": "voyage-sentence",
                        "document_id": "wikivoyage-document",
                        "project": "wikivoyage",
                        "language": "en",
                        "page_id": 20,
                        "revision_id": 1,
                        "text": second_text,
                        "content_hash": _sha(second_text),
                    },
                )
            ],
            [],
            inventory,
            wiki_link_rows=links,
            wiki_corpus="wikivoyage",
        )
    )

    assert len(wikipedia_records) == len(wikivoyage_records) == 3
    assert wikipedia_records[0].links[0].inventory_row_id == "wiki-10"
    assert wikivoyage_records[0].links[0].inventory_row_id == "voyage-20"
    wikipedia_no_text = next(
        record
        for record in wikipedia_records
        if json.loads(record.relevance.evidence).get("inventory_row_id") == "no-text-30"
    )
    wikivoyage_no_text = next(
        record
        for record in wikivoyage_records
        if json.loads(record.relevance.evidence).get("inventory_row_id") == "no-text-30"
    )
    assert wikipedia_no_text.relevance.status is RelevanceStatus.MISSING
    assert wikivoyage_no_text.relevance.status is RelevanceStatus.MISSING
    assert wikipedia_no_text.document.document_id != wikivoyage_no_text.document.document_id
    joined_document_ids = {
        record.document.document_id for record in (*wikipedia_records, *wikivoyage_records)
    }
    assert len(joined_document_ids) == 6


def test_wiki_document_and_link_tree_gaps_are_exact_and_explicit() -> None:
    assert missing_wiki_join_tree_groups() == (
        "polygon_document_links/",
        "wikipedia/documents/",
        "wikivoyage/documents/",
    )
    assert EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI].wikivoyage_document_glob == (
        "wikivoyage/documents/*.parquet"
    )


def test_downstream_records_are_batched_with_consecutive_indices() -> None:
    # Orphan label rows are complete downstream records even when there is no
    # matching sentence body or polygon link.
    orphan = list(
        iter_eunis_join_records(
            TextSource.DESCRIPTION,
            [],
            [
                _pinned_row(
                    "labels/language-v1/data/a.parquet",
                    0,
                    {
                        "label_id": "orphan-1",
                        "description_identity": "c" * 64,
                        "tag_key": "description",
                        "sentence_index": 0,
                        "text_sha256": _sha("missing"),
                        "decision": "NO",
                        "language": "eng",
                    },
                ),
                _pinned_row(
                    "labels/language-v1/data/b.parquet",
                    0,
                    {
                        "label_id": "orphan-2",
                        "description_identity": "d" * 64,
                        "tag_key": "description",
                        "sentence_index": 0,
                        "text_sha256": _sha("also missing"),
                        "decision": "YES",
                        "language": "eng",
                    },
                ),
            ],
            [],
        )
    )
    batches = list(
        iter_eunis_join_batches(
            orphan,
            starting_batch_index=7,
            max_batch_rows=3,
            max_batch_bytes=10_000,
        )
    )
    assert [batch.batch_index for batch in batches] == [7, 8]
    assert [batch.row_count for batch in batches] == [3, 3]


def test_pinned_table_verifies_tree_metadata_and_local_git_blob(tmp_path) -> None:
    path = tmp_path / "fixture.parquet"
    pq.write_table(pa.table({"value": pa.array(["one", "two"], type=pa.string())}), path)
    data = path.read_bytes()
    blob_id = hashlib.sha1(
        f"blob {len(data)}\0".encode() + data,
        usedforsecurity=False,
    ).hexdigest()
    tree_file = SourceTreeFile(
        path="language-v1/data/fixture.parquet",
        size_bytes=len(data),
        blob_id=blob_id,
        lfs_oid=None,
        lfs_size=len(data),
    )
    base = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.DESCRIPTION].additional_tree_pins[0]
    pin = SourceTreeGroupPin(
        repository=base.repository,
        revision=base.revision,
        role=base.role,
        path_prefix=base.path_prefix,
        file_count=1,
        total_size_bytes=len(data),
        inventory_sha256=source_tree_inventory_sha256((tree_file,)),
    )
    table = EunisPinnedTable(
        pin=pin,
        files=(EunisPinnedParquetFile(tree_file, path),),
    )

    rows = list(table.iter_rows(("value",), expected_types={"value": "string"}))

    assert [row.values["value"] for row in rows] == ["one", "two"]
    tampered = replace(tree_file, blob_id="0" * 40)
    bad_table = EunisPinnedTable(
        pin=replace(pin, inventory_sha256=source_tree_inventory_sha256((tampered,))),
        files=(EunisPinnedParquetFile(tampered, path),),
    )
    with pytest.raises(ValueError, match="Git blob hash differs"):
        list(bad_table.iter_rows(("value",)))
