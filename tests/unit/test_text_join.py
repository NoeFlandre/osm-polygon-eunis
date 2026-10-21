from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace
from typing import Any, cast

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
    JoinAllocationBounds,
    PinnedSourceRow,
    iter_eunis_join_batches,
    iter_eunis_join_records,
    iter_pinned_eunis_join_records,
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


def _pinned_table(  # noqa: PLR0913
    tmp_path,
    *,
    repository: str,
    revision: str,
    role: str,
    path_prefix: str,
    path_in_repo: str,
    schema: pa.Schema,
    rows: list[dict],
) -> EunisPinnedTable:
    local_path = tmp_path / role / path_in_repo.rsplit("/", 1)[-1]
    local_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), local_path)
    content = local_path.read_bytes()
    tree_file = SourceTreeFile(
        path=path_in_repo,
        size_bytes=len(content),
        blob_id=hashlib.sha1(
            f"blob {len(content)}\0".encode() + content,
            usedforsecurity=False,
        ).hexdigest(),
        lfs_oid=None,
        lfs_size=len(content),
    )
    pin = SourceTreeGroupPin(
        repository=repository,
        revision=revision,
        role=role,
        path_prefix=path_prefix,
        file_count=1,
        total_size_bytes=len(content),
        inventory_sha256=source_tree_inventory_sha256((tree_file,)),
    )
    return EunisPinnedTable(pin, (EunisPinnedParquetFile(tree_file, local_path),))


def _layout_with_fixture_pins(base, *pins, raw_polygon_tree_pin=None):
    replacements = {(pin.role, pin.path_prefix) for pin in pins}
    additional = tuple(
        pin for pin in base.additional_tree_pins if (pin.role, pin.path_prefix) not in replacements
    ) + tuple(pins)
    return replace(
        base,
        raw_polygon_tree_pin=raw_polygon_tree_pin or base.raw_polygon_tree_pin,
        additional_tree_pins=additional,
    )


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


def test_duplicate_relevance_evidence_spills_then_rejects_the_explicit_row_cap() -> None:
    full_text = "A short source description."
    original_hash = _sha(full_text)
    identity = _description_identity("way", 17, "description", original_hash)
    source = _pinned_row(
        "language-v1/data/region.parquet",
        0,
        {
            "description_identity": identity,
            "source_pbf": "region.osm.pbf",
            "osm_type": "way",
            "osm_id": 17,
            "tag_key": "description",
            "original_text": full_text,
            "text_sha256": original_hash,
            "language_code": "eng",
            "sentences": [full_text],
        },
    )
    labels = [
        _pinned_row(
            "labels/language-v1/data/region.parquet",
            index,
            {
                "label_id": f"label-{index}",
                "description_identity": identity,
                "tag_key": "description",
                "sentence_index": 0,
                "text_sha256": original_hash,
                "decision": "YES",
                "language": "eng",
            },
        )
        for index in range(20)
    ]
    bounds = JoinAllocationBounds(
        max_record_rows=32,
        max_record_bytes=8_192,
        max_output_row_bytes=512,
    )

    with pytest.raises(ValueError, match="relevance evidence exceeds the maximum output row size"):
        list(
            iter_eunis_join_records(
                TextSource.DESCRIPTION,
                [source],
                labels,
                [_inventory("region.osm.pbf", "way", 17, "inv-17", "poly-17")],
                allocation_bounds=bounds,
            )
        )


def test_wiki_link_groups_keep_nested_evidence_order_and_enforce_record_rows() -> None:
    text = "A sentence about one document."
    sentence = _pinned_row(
        "wikipedia/sentences/region.parquet",
        0,
        {
            "sentence_id": "sentence-1",
            "document_id": "document-1",
            "project": "wikipedia",
            "language": "en",
            "page_id": 1,
            "revision_id": 1,
            "text": text,
            "content_hash": _sha(text),
        },
    )

    def link_row(ordinal: int, source_pbf: str, marker: str) -> PinnedSourceRow:
        return _pinned_row(
            "polygon_document_links/region.parquet",
            ordinal,
            {
                "document_id": "document-1",
                "project": "wikipedia",
                "polygon_id": f"{source_pbf}:way:{ordinal}",
                "source_pbf": source_pbf,
                "osm_type": "way",
                "osm_id": 10 if source_pbf.startswith("a-") else 20,
                "link_sources": marker,
            },
        )

    links = [
        link_row(0, "a-region.osm.pbf", "first"),
        link_row(1, "z-region.osm.pbf", "other-key"),
        link_row(2, "a-region.osm.pbf", "second"),
        link_row(3, "a-region.osm.pbf", "third"),
    ]
    inventory = [
        _inventory("a-region.osm.pbf", "way", 10, "inv-a", "poly-a"),
        _inventory("z-region.osm.pbf", "way", 20, "inv-z", "poly-z"),
    ]
    records = list(
        iter_eunis_join_records(
            TextSource.WIKI,
            [sentence],
            [],
            inventory,
            wiki_link_rows=links,
            wiki_corpus="wikipedia",
            allocation_bounds=JoinAllocationBounds(max_record_rows=4),
        )
    )

    [record] = records
    assert [link.inventory_row_id for link in record.links] == ["inv-a", "inv-z"]
    nested = json.loads(record.links[0].evidence.split(" evidence ", 1)[1])
    assert [[row["row"] for row in nested[0]], nested[1]["row"]] == [[0, 2], 3]

    too_small = iter_eunis_join_records(
        TextSource.WIKI,
        [sentence],
        [],
        inventory,
        wiki_link_rows=links,
        wiki_corpus="wikipedia",
        allocation_bounds=JoinAllocationBounds(max_record_rows=3),
    )
    with pytest.raises(ValueError, match="joined record exceeds the maximum output row count"):
        next(iter(too_small))


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

    row = _pinned_row(
        "language-v1/data/region.parquet",
        1,
        {
            "description_identity": "f" * 64,
            "source_pbf": "region.osm.pbf",
            "osm_type": "way",
            "osm_id": 7,
            "tag_key": "description",
            "original_text": text,
            "text_sha256": text_hash,
            "language_code": "eng",
            "sentences": [text],
        },
    )
    with pytest.raises(ValueError, match="identity does not match"):
        list(iter_eunis_join_records(TextSource.DESCRIPTION, [row], [], []))


def test_description_missing_native_identity_keeps_text_and_relevance() -> None:
    text = "The source key is incomplete."
    text_hash = _sha(text)
    identity = "e" * 64
    raw = _pinned_row(
        "language-v1/data/region.parquet",
        0,
        {
            "description_identity": identity,
            "source_pbf": "region.osm.pbf",
            "osm_type": None,
            "osm_id": 7,
            "tag_key": "description",
            "original_text": text,
            "text_sha256": text_hash,
            "language_code": "eng",
            "sentences": [text],
        },
    )
    label = _pinned_row(
        "labels/language-v1/data/region.parquet",
        0,
        {
            "label_id": "label-with-incomplete-key",
            "description_identity": identity,
            "tag_key": "description",
            "sentence_index": 0,
            "text_sha256": text_hash,
            "decision": "YES",
            "language": "eng",
        },
    )

    [record] = list(iter_eunis_join_records(TextSource.DESCRIPTION, [raw], [label], []))

    assert record.document.content_sha256 == _sha(text)
    assert record.relevance.status is RelevanceStatus.YES
    assert record.links[0].status is TextLinkStatus.UNMATCHED


def test_invalid_website_decisions_keep_empty_and_unknown_reasons() -> None:
    sentences = ["No decision", "Empty decision", "Whitespace decision", "Other decision"]
    raw = _pinned_row(
        "polygons/region.parquet",
        0,
        {
            "polygon_id": "region:way:55",
            "source_pbf": "region.osm.pbf",
            "osm_type": "way",
            "osm_id": 55,
            "website_sentences": sentences,
            "contact_website_sentences": [],
            "website_sentence_status": "success",
            "contact_website_sentence_status": "empty",
            "website_text": " ".join(sentences),
            "contact_website_text": None,
            "website_text_status": "ok",
            "contact_website_text_status": "missing",
        },
    )
    labels = [
        _pinned_row(
            "labels/polygons/region.parquet",
            index,
            {
                "label_id": f"label-{index}",
                "polygon_id": "region:way:55",
                "field": "website",
                "sentence_index": index,
                "text_sha256": _sha(sentence),
                "decision": decision,
            },
        )
        for index, (sentence, decision) in enumerate(
            zip(sentences, (None, "", "  ", "perhaps"), strict=True)
        )
    ]

    records = list(iter_eunis_join_records(TextSource.WEBSITE, [raw], labels, []))
    relevance_by_text = {record.document.content_sha256: record.relevance for record in records}

    assert relevance_by_text[_sha(sentences[0])].reason == "relevance row has an empty decision"
    assert relevance_by_text[_sha(sentences[1])].reason == "relevance row has an empty decision"
    assert relevance_by_text[_sha(sentences[2])].reason == "relevance row has an empty decision"
    assert relevance_by_text[_sha(sentences[3])].reason == (
        "relevance decision is outside the pinned YES/NO vocabulary"
    )


def test_website_join_uses_pinned_field_mapping_and_keeps_unmatched_labels() -> None:
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
            "labels/polygons/region.parquet",
            1,
            {
                "label_id": "contact-website-no",
                "polygon_id": "region:way:42",
                "field": "contact_website",
                "sentence_index": 0,
                "text_sha256": _sha("Call the guide."),
                "decision": "no",
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

    with pytest.raises(ValueError, match="match the pinned source label vocabulary"):
        list(
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

    records = list(
        iter_eunis_join_records(
            TextSource.WEBSITE,
            raw,
            labels,
            inventory,
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
    assert contact_record.relevance.status is RelevanceStatus.NO
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
                "link_sources": {"imported": ("wikidata", b"source"), 3: True},
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
    assert json.loads(records[0].links[0].evidence.split(" evidence ", 1)[1])["link_sources"] == {
        "3": True,
        "imported": ["wikidata", "b'source'"],
    }
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


def test_website_rows_without_a_complete_native_key_keep_an_unmatched_link() -> None:
    text = "An otherwise linkable sentence."
    raw_row = _pinned_row(
        "polygons/region.parquet",
        0,
        {
            "polygon_id": "region:way:74",
            "source_pbf": None,
            "osm_type": "way",
            "osm_id": 74,
            "website_sentences": [text],
            "contact_website_sentences": [],
            "website_sentence_status": "success",
            "contact_website_sentence_status": "empty",
            "website_text": text,
            "contact_website_text": None,
            "website_text_status": "ok",
            "contact_website_text_status": "missing",
        },
    )

    records = list(
        iter_eunis_join_records(
            TextSource.WEBSITE,
            [raw_row],
            [],
            [],
            website_field_to_column={
                "website": "website_sentences",
                "contact_website": "contact_website_sentences",
            },
        )
    )
    linked_text = next(record for record in records if record.document.content_sha256 == _sha(text))

    assert linked_text.links[0].status is TextLinkStatus.UNMATCHED
    assert "source polygon key is incomplete" in linked_text.links[0].evidence


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
    assert missing_wiki_join_tree_groups() == ()
    wiki_pins = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI].additional_tree_pins
    assert {
        (pin.role, pin.path_prefix)
        for pin in wiki_pins
        if pin.role in {"wiki_document_links", "wiki_documents"}
    } == {
        ("wiki_document_links", "polygon_document_links/"),
        ("wiki_documents", "wikipedia/documents/"),
        ("wiki_documents", "wikivoyage/documents/"),
    }
    assert (
        "wiki_language_split_links",
        "language_splits/polygon_document_links_by_language/",
    ) in {(pin.role, pin.path_prefix) for pin in wiki_pins}
    assert EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI].wikivoyage_document_glob == (
        "wikivoyage/documents/*.parquet"
    )
    with pytest.raises(ValueError, match="only to the Wiki source"):
        missing_wiki_join_tree_groups(EUNIS_POOL_SOURCE_LAYOUTS[TextSource.DESCRIPTION])


def test_join_preconditions_reject_source_and_corpus_mismatches() -> None:
    description_layout = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.DESCRIPTION]
    with pytest.raises(ValueError, match="layout does not match"):
        list(
            iter_eunis_join_records(
                TextSource.DESCRIPTION,
                [],
                [],
                [],
                layout=replace(description_layout, text_source=TextSource.WEBSITE),
            )
        )
    with pytest.raises(ValueError, match="Website field mapping can only"):
        list(
            iter_eunis_join_records(
                TextSource.DESCRIPTION,
                [],
                [],
                [],
                website_field_to_column={"website": "website_sentences"},
            )
        )
    with pytest.raises(ValueError, match="corpus='wikipedia' or corpus='wikivoyage'"):
        iter_pinned_eunis_join_records(
            TextSource.WIKI,
            cast(Any, None),
            cast(Any, None),
            [],
            wiki_corpus=None,
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
    with pytest.raises(ValueError, match="batch index and limits must be positive"):
        list(
            iter_eunis_join_batches(
                orphan,
                starting_batch_index=-1,
                max_batch_rows=3,
                max_batch_bytes=10_000,
            )
        )
    with pytest.raises(ValueError, match="one joined text record exceeds"):
        list(
            iter_eunis_join_batches(
                orphan,
                starting_batch_index=0,
                max_batch_rows=2,
                max_batch_bytes=10_000,
            )
        )


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
    path.unlink()
    with pytest.raises(FileNotFoundError, match="is absent"):
        list(table.iter_rows(("value",)))
    path.write_bytes(data + b"tamper")
    with pytest.raises(ValueError, match="size differs"):
        list(table.iter_rows(("value",)))
    path.write_bytes(data)
    tampered = replace(tree_file, blob_id="0" * 40)
    bad_table = EunisPinnedTable(
        pin=replace(pin, inventory_sha256=source_tree_inventory_sha256((tampered,))),
        files=(EunisPinnedParquetFile(tampered, path),),
    )
    with pytest.raises(ValueError, match="Git blob hash differs"):
        list(bad_table.iter_rows(("value",)))

    lfs_file = replace(tree_file, lfs_oid="0" * 64)
    lfs_pin = replace(pin, inventory_sha256=source_tree_inventory_sha256((lfs_file,)))
    lfs_table = EunisPinnedTable(lfs_pin, (EunisPinnedParquetFile(lfs_file, path),))
    with pytest.raises(ValueError, match="LFS hash differs"):
        list(lfs_table.iter_rows(("value",)))


def test_description_pinned_join_checks_arrow_contract_and_keeps_body_private(tmp_path) -> None:
    base = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.DESCRIPTION]
    text = "The old stone mill."
    source_text = "The old stone mill. It closed in 1980."
    source_hash = _sha(source_text)
    identity = _description_identity("way", 71, "description", source_hash)
    text_table = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="raw_text",
        path_prefix="language-v1/data/",
        path_in_repo="language-v1/data/region.parquet",
        schema=pa.schema(
            [
                pa.field("description_identity", pa.string()),
                pa.field("source_pbf", pa.string()),
                pa.field("osm_type", pa.string()),
                pa.field("osm_id", pa.int64()),
                pa.field("tag_key", pa.string()),
                pa.field("original_text", pa.string()),
                pa.field("text_sha256", pa.string()),
                pa.field("language_code", pa.string()),
                pa.field("sentences", pa.list_(pa.string())),
                pa.field("split_status", pa.string()),
                pa.field("split_reason", pa.string()),
            ]
        ),
        rows=[
            {
                "description_identity": identity,
                "source_pbf": "region.osm.pbf",
                "osm_type": "way",
                "osm_id": 71,
                "tag_key": "description",
                "original_text": source_text,
                "text_sha256": source_hash,
                "language_code": "eng",
                "sentences": [text],
                "split_status": "split",
                "split_reason": "",
            }
        ],
    )
    relevance_table = _pinned_table(
        tmp_path,
        repository=base.relevance_repository,
        revision=base.relevance_revision,
        role="relevance_labels",
        path_prefix="labels/language-v1/data/",
        path_in_repo="labels/language-v1/data/region.parquet",
        schema=pa.schema(
            [
                pa.field("label_id", pa.string()),
                pa.field("description_identity", pa.string()),
                pa.field("tag_key", pa.string()),
                pa.field("sentence_index", pa.int32()),
                pa.field("text_sha256", pa.string()),
                pa.field("decision", pa.string()),
                pa.field("language", pa.string()),
            ]
        ),
        rows=[
            {
                "label_id": "label-71",
                "description_identity": identity,
                "tag_key": "description",
                "sentence_index": 0,
                "text_sha256": source_hash,
                "decision": "YES",
                "language": "eng",
            }
        ],
    )
    layout = _layout_with_fixture_pins(base, text_table.pin, relevance_table.pin)
    inventory = [_inventory("region.osm.pbf", "way", 71, "inv-71", "poly-71")]

    missing_text_pin = replace(
        layout,
        additional_tree_pins=tuple(
            pin for pin in layout.additional_tree_pins if pin.role != "raw_text"
        ),
    )
    with pytest.raises(ValueError, match="no unique verified tree pin"):
        iter_pinned_eunis_join_records(
            TextSource.DESCRIPTION,
            text_table,
            relevance_table,
            inventory,
            layout=missing_text_pin,
        )
    mismatched_text_pin = replace(
        layout,
        additional_tree_pins=tuple(
            replace(pin, total_size_bytes=pin.total_size_bytes + 1)
            if pin.role == "raw_text"
            else pin
            for pin in layout.additional_tree_pins
        ),
    )
    with pytest.raises(ValueError, match="input table pin differs"):
        iter_pinned_eunis_join_records(
            TextSource.DESCRIPTION,
            text_table,
            relevance_table,
            inventory,
            layout=mismatched_text_pin,
        )

    [record] = list(
        iter_pinned_eunis_join_records(
            TextSource.DESCRIPTION,
            text_table,
            relevance_table,
            inventory,
            layout=layout,
            batch_size=1,
            staging_directory=tmp_path,
        )
    )

    assert record.document.content_sha256 == _sha(text)
    assert record.document.body_status is BodyStatus.PRIVATE
    assert record.relevance.status is RelevanceStatus.YES
    assert record.links[0].status is TextLinkStatus.LINKED
    assert text not in repr(record)


def test_website_pinned_join_requires_large_string_text_schema_and_explicit_mapping(
    tmp_path,
) -> None:
    base = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WEBSITE]
    website_text = "https://example.org/old-town"
    text_table = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="raw_geometry",
        path_prefix="polygons/",
        path_in_repo="polygons/region.parquet",
        schema=pa.schema(
            [
                pa.field("polygon_id", pa.string()),
                pa.field("source_pbf", pa.string()),
                pa.field("osm_type", pa.string()),
                pa.field("osm_id", pa.int64()),
                pa.field("website_sentences", pa.list_(pa.string())),
                pa.field("contact_website_sentences", pa.list_(pa.string())),
                pa.field("website_sentence_status", pa.string()),
                pa.field("contact_website_sentence_status", pa.string()),
                pa.field("website_text", pa.large_string()),
                pa.field("contact_website_text", pa.large_string()),
                pa.field("website_text_status", pa.string()),
                pa.field("contact_website_text_status", pa.string()),
            ]
        ),
        rows=[
            {
                "polygon_id": "region:way:72",
                "source_pbf": "region.osm.pbf",
                "osm_type": "way",
                "osm_id": 72,
                "website_sentences": [website_text],
                "contact_website_sentences": [],
                "website_sentence_status": "success",
                "contact_website_sentence_status": "empty",
                "website_text": website_text,
                "contact_website_text": None,
                "website_text_status": "ok",
                "contact_website_text_status": "missing",
            }
        ],
    )
    relevance_table = _pinned_table(
        tmp_path,
        repository=base.relevance_repository,
        revision=base.relevance_revision,
        role="relevance_labels",
        path_prefix="labels/polygons/",
        path_in_repo="labels/polygons/region.parquet",
        schema=pa.schema(
            [
                pa.field("label_id", pa.string()),
                pa.field("polygon_id", pa.string()),
                pa.field("field", pa.string()),
                pa.field("sentence_index", pa.int32()),
                pa.field("text_sha256", pa.string()),
                pa.field("decision", pa.string()),
            ]
        ),
        rows=[
            {
                "label_id": "website-label-72",
                "polygon_id": "region:way:72",
                "field": "website",
                "sentence_index": 0,
                "text_sha256": _sha(website_text),
                "decision": "NO",
            }
        ],
    )
    layout = _layout_with_fixture_pins(
        base,
        relevance_table.pin,
        raw_polygon_tree_pin=text_table.pin,
    )
    inventory = [_inventory("region.osm.pbf", "way", 72, "inv-72", "poly-72")]

    records = list(
        iter_pinned_eunis_join_records(
            TextSource.WEBSITE,
            text_table,
            relevance_table,
            inventory,
            website_field_to_column={
                "website": "website_sentences",
                "contact_website": "contact_website_sentences",
            },
            layout=layout,
            batch_size=1,
            staging_directory=tmp_path,
        )
    )

    assert len(records) == 2
    website_record = next(
        record for record in records if record.document.content_sha256 == _sha(website_text)
    )
    contact_record = next(
        record for record in records if record.document.body_status is BodyStatus.MISSING
    )
    assert website_record.relevance.status is RelevanceStatus.NO
    assert website_record.document.content_sha256 == _sha(website_text)
    assert contact_record.relevance.status is RelevanceStatus.MISSING
    assert contact_record.relevance.reason is not None
    assert "sentence status=empty" in contact_record.relevance.reason


def test_wiki_pinned_join_fails_on_missing_metadata_then_joins_fixture_pins(tmp_path) -> None:
    base = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.WIKI]
    wiki_text = "A sentence about the museum."
    wiki_text_table = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="wikipedia_text",
        path_prefix="wikipedia/sentences/",
        path_in_repo="wikipedia/sentences/region.parquet",
        schema=pa.schema(
            [
                pa.field("sentence_id", pa.string()),
                pa.field("document_id", pa.string()),
                pa.field("project", pa.string()),
                pa.field("language", pa.string()),
                pa.field("page_id", pa.int64()),
                pa.field("revision_id", pa.int64()),
                pa.field("text", pa.string()),
                pa.field("content_hash", pa.string()),
                pa.field("segmentation_status", pa.string()),
            ]
        ),
        rows=[
            {
                "sentence_id": "sentence-73",
                "document_id": "document-73",
                "project": "wikipedia",
                "language": "en",
                "page_id": 73,
                "revision_id": 3,
                "text": wiki_text,
                "content_hash": _sha(wiki_text),
                "segmentation_status": "success",
            }
        ],
    )
    relevance_table = _pinned_table(
        tmp_path,
        repository=base.relevance_repository,
        revision=base.relevance_revision,
        role="wikipedia_relevance_labels",
        path_prefix="labels/wikipedia/sentences/",
        path_in_repo="labels/wikipedia/sentences/region.parquet",
        schema=pa.schema(
            [
                pa.field("label_id", pa.string()),
                pa.field("sentence_id", pa.string()),
                pa.field("text_sha256", pa.string()),
                pa.field("decision", pa.string()),
            ]
        ),
        rows=[
            {
                "label_id": "wiki-label-73",
                "sentence_id": "sentence-73",
                "text_sha256": _sha(wiki_text),
                "decision": "YES",
            }
        ],
    )
    inventory = [_inventory("wiki.osm.pbf", "way", 73, "inv-73", "poly-73")]

    link_table = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="wiki_document_links",
        path_prefix="polygon_document_links/",
        path_in_repo="polygon_document_links/region.parquet",
        schema=pa.schema(
            [
                pa.field("document_id", pa.string()),
                pa.field("project", pa.string()),
                pa.field("polygon_id", pa.string()),
                pa.field("source_pbf", pa.string()),
                pa.field("osm_type", pa.string()),
                pa.field("osm_id", pa.int64()),
                pa.field("link_sources", pa.string()),
            ]
        ),
        rows=[
            {
                "document_id": "document-73",
                "project": "wikipedia",
                "polygon_id": "wiki:way:73",
                "source_pbf": "wiki.osm.pbf",
                "osm_type": "way",
                "osm_id": 73,
                "link_sources": '["wikipedia"]',
            }
        ],
    )
    wikipedia_documents = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="wiki_documents",
        path_prefix="wikipedia/documents/",
        path_in_repo="wikipedia/documents/metadata.parquet",
        schema=pa.schema([pa.field("document_id", pa.string())]),
        rows=[{"document_id": "document-73"}],
    )
    wikivoyage_documents = _pinned_table(
        tmp_path,
        repository=base.repository,
        revision=base.raw_revision,
        role="wiki_documents",
        path_prefix="wikivoyage/documents/",
        path_in_repo="wikivoyage/documents/metadata.parquet",
        schema=pa.schema([pa.field("document_id", pa.string())]),
        rows=[{"document_id": "other-document"}],
    )
    layout = _layout_with_fixture_pins(
        base,
        wiki_text_table.pin,
        relevance_table.pin,
        link_table.pin,
        wikipedia_documents.pin,
        wikivoyage_documents.pin,
    )
    with pytest.raises(ValueError, match="require the pinned polygon_document_links"):
        iter_pinned_eunis_join_records(
            TextSource.WIKI,
            wiki_text_table,
            relevance_table,
            inventory,
            wiki_corpus="wikipedia",
            layout=layout,
        )
    with pytest.raises(ValueError, match="document pin for wikipedia/documents/ is absent"):
        iter_pinned_eunis_join_records(
            TextSource.WIKI,
            wiki_text_table,
            relevance_table,
            inventory,
            wiki_link_table=link_table,
            wiki_corpus="wikipedia",
            layout=layout,
        )
    with pytest.raises(ValueError, match="document pin for wikipedia/documents/ is absent"):
        iter_pinned_eunis_join_records(
            TextSource.WIKI,
            wiki_text_table,
            relevance_table,
            inventory,
            wiki_link_table=link_table,
            wiki_document_tables=(wikivoyage_documents,),
            wiki_corpus="wikipedia",
            layout=layout,
        )
    [record] = list(
        iter_pinned_eunis_join_records(
            TextSource.WIKI,
            wiki_text_table,
            relevance_table,
            inventory,
            wiki_link_table=link_table,
            wiki_document_tables=(wikipedia_documents, wikivoyage_documents),
            wiki_corpus="wikipedia",
            layout=layout,
            batch_size=1,
            staging_directory=tmp_path,
        )
    )

    assert record.relevance.status is RelevanceStatus.YES
    assert record.document.content_sha256 == _sha(wiki_text)
    assert record.links[0].status is TextLinkStatus.LINKED
    assert record.links[0].inventory_row_id == "inv-73"


def test_pinned_table_rejects_duplicate_missing_and_wrong_typed_columns(tmp_path) -> None:
    path = tmp_path / "small.parquet"
    pq.write_table(pa.table({"value": pa.array([1], type=pa.int64())}), path)
    content = path.read_bytes()
    tree_file = SourceTreeFile(
        path="language-v1/data/small.parquet",
        size_bytes=len(content),
        blob_id=hashlib.sha1(
            f"blob {len(content)}\0".encode() + content,
            usedforsecurity=False,
        ).hexdigest(),
        lfs_oid=None,
        lfs_size=len(content),
    )
    base_pin = EUNIS_POOL_SOURCE_LAYOUTS[TextSource.DESCRIPTION].additional_tree_pins[0]
    pin = SourceTreeGroupPin(
        repository=base_pin.repository,
        revision=base_pin.revision,
        role=base_pin.role,
        path_prefix=base_pin.path_prefix,
        file_count=1,
        total_size_bytes=len(content),
        inventory_sha256=source_tree_inventory_sha256((tree_file,)),
    )
    table = EunisPinnedTable(pin, (EunisPinnedParquetFile(tree_file, path),))

    with pytest.raises(ValueError, match="positive"):
        list(table.iter_rows(("value",), batch_size=0))
    with pytest.raises(ValueError, match="unique, nonempty"):
        list(table.iter_rows(("value", "value")))
    with pytest.raises(ValueError, match="lacks pinned source columns"):
        list(table.iter_rows(("missing",)))
    with pytest.raises(ValueError, match="lacks pinned schema field"):
        list(table.iter_rows(("value",), expected_types={"other": "string"}))
    with pytest.raises(TypeError, match="expected string"):
        list(table.iter_rows(("value",), expected_types={"value": "string"}))
    with pytest.raises(ValueError, match="unknown schema type contract"):
        list(table.iter_rows(("value",), expected_types={"value": "decimal"}))


def test_pinned_source_row_rejects_incomplete_locators() -> None:
    with pytest.raises(ValueError, match="repository path"):
        PinnedSourceRow("  ", 0, {})
    with pytest.raises(ValueError, match="nonnegative ordinal"):
        PinnedSourceRow("data/a.parquet", -1, {})
