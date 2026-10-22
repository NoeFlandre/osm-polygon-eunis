# EUNIS source-wide pool adapter

The EUNIS adapter has two stages. The first scans every polygon row from a
complete, immutable source manifest and writes a per-shard raw checkpoint. Only
after that source-wide key census succeeds does the second stage query EEA
support maps and write one label attempt for every raw polygon. Text documents,
document links, and YES/NO relevance labels are downstream inputs; they do not
select the rows in either stage.

## Frozen source layouts

| Pool source | Raw polygon snapshot | Downstream relevance snapshot | Downstream text evidence |
| --- | --- | --- | --- |
| Description | `NoeFlandre/osm-polygon-description-tag` at `b4706eb315c66a8a57135289e7612dca8e03caf8`, `data/*.parquet` (386 files; 746,727,545 bytes) | `NoeFlandre/osm-polygon-description-tag-landuse` at `ce98b328f3c6543800a3d931877b51eb3f0cf768`, `labels/language-v1/data/*.parquet` (386 files; 100,444,539 bytes) | Derived `language-v1/data/*.parquet` (386 files; 103,449,770 bytes); joins back to raw `data/` by `(source_pbf, osm_type, osm_id)`. |
| Website | `NoeFlandre/osm-polygon-website-tag` at `4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd`, `polygons/*.parquet` (386 files; 3,943,775,497 bytes) | `NoeFlandre/osm-polygon-website-tag-landuse` at `e6524c711ee0033c3cd5990e5fdbbe4b89992dfb`, `labels/polygons/*.parquet` (386 files; 3,299,123,482 bytes) | Website fields remain attached to raw rows. |
| Wiki | `NoeFlandre/osm-polygon-wikidata-and-wikipedia` at `d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801`, `polygons/*.parquet` (386 files; 2,203,317,876 bytes) | `NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` at `49307aeb05b6dcc8a5d393f77f59a1e6d13a967a`, Wikipedia labels (386 files; 8,191,284,820 bytes) and Wikivoyage labels (375 files; 189,959,172 bytes) | Raw `wikipedia/sentences/*.parquet` (386 files; 18,075,023,818 bytes) and `wikivoyage/sentences/*.parquet` (375 files; 373,112,864 bytes). |

The checked-in `config/eunis-source-provenance-20261021.json` records the frozen
tree entries, Git blob IDs, byte sizes, and canonical inventory hashes for these
raw, text, relevance, and Wiki document/link groups. Its verified v2 metadata
handoff SHA-256 is
`d1c4bf9e60468a85b17c535be17ae4efeba7284b718d60235e19a39a9e65a072`. The
handoff's per-group digests cover its complete file-entry objects; the checked-in
manifest also records normalized inventory hashes used by the source-tree pin
contract. The pinned Wiki groups are `polygon_document_links/` (386 files,
112,581,825 bytes), `wikipedia/documents/` (386 files, 8,739,038,060 bytes),
and `wikivoyage/documents/` (375 files, 80,078,157 bytes). The separate
`language_splits/polygon_document_links_by_language/` group (356 files,
154,309,530 bytes) is pinned separately and is not unioned with the primary
polygon-document links. These pins confirm the frozen file inventories; they do
not provide Parquet row counts, native-key counts, or proof that all rows have
been scanned. The label snapshots are separate inputs and never select the raw
polygons to enumerate.

The Wiki repository is explicitly mapped to `TextSource.WIKI`; the older
`DatasetSpec("wikidata", ...)` remains unchanged for legacy consumers. The raw
identity is the complete `(source_pbf, osm_type, osm_id)` key. Its source-row
locator and original geometry payload are retained in the raw checkpoint.

The runner accepts only a frozen repository file inventory whose polygon paths
exactly equal all paths matching the source layout and the pinned tree-group
count, total size, and inventory digest. Each staged non-LFS shard is checked
against its Git blob ID, byte count, and SHA-256 before scanning. The source
snapshot must also pin the PBF manifest URI and digest. The scan report records
the source manifest URI/hash, all shard hashes, row counts, the canonical
composite-key-set hash, and the adapter/shared-core revisions. It uses a
disk-backed merge sort to calculate the digest in bounded memory. A duplicate
composite key fails the scan rather than being dropped or merged.

Per-shard checkpoints are content-hashed and resumable. A resume is accepted
only when the complete source manifest, source revision, shard hash, and code
revisions match. The raw inventory evidence matches the shared core's
`native_keyset_sha256()` encoding exactly.

## EUNIS decisions

`RasterReference.support_areas()` returns the exact polygon-intersection area
for each `(level, class)` after unioning positive, non-NoData raster cells
within that class. Separate classes may overlap. Each candidate keeps its
measured numerator, the full projected polygon area denominator, ratio,
threshold, map revision, taxonomy revision, and direct/ancestor-derived kind.
NoData and areas outside map coverage have zero support and remain in the full
area denominator.

The adapter resolves L1, L2, and L3 independently. L3 is direct map support;
L1 and L2 are unions of their explicit descendant L3 map codes. Exactly one
class at a level must reach 0.8 of the full polygon area. Positive candidates
below 0.8 remain in the record. Multiple qualifying classes are ambiguous;
no positive support and positive support below threshold are distinct
unassigned reasons. Invalid geometry and reference-query failures produce
excluded outcomes, with their reasons, at every supported level. L4-L8 are
declared unsupported and receive no per-polygon outcomes.

`make_eunis_2021_reference_pin()` records the supplied verified EEA v1 2021
reference manifest SHA-256, 136 assets/5,007,209,626 bytes, EPSG:3035, and the
positive-value map cutoff of zero. The runner checks that raster source version,
positive-value cutoff, layer codes, taxonomy revision, and code revisions match
the pins before processing. Geometry and raw payloads remain recoverable
through row-addressed checkpoint URIs.

`parse_classification_taxonomy_rows()` parses explicit `Level`, `Code`, and
`Name` rows. It links a code only to the longest proper code prefix present at
a strictly lower level, following the hierarchy rule reviewed in PR #159. The
parser is tested with repeated headers, multiple sheets, same-level prefixes,
and conflicting or incomplete rows. `verify_classification_taxonomy_lineage()`
checks the actual cache and workbook bytes against the pinned cache digest,
classification record, workbook digest, and workbook-derived level/parent
mapping, then returns immutable, token-gated lineage evidence. The older
`validate_classification_taxonomy_lineage()` keeps its `ClassificationTaxonomy`
return type for compatibility; callers preparing production work must use the
new verifier and pass its result to `make_eunis_taxonomy_from_classification()`.
The caller must decode the supplied `ClassificationTaxonomy` from the cache
payload; the verifier independently checks both payload hashes and compares
that decoded hierarchy with a fresh workbook parse.
That builder creates explicit paths for direct L3 map codes and requires an L1
and L2 ancestor for every code. `EunisTaxonomy.production_eligible` reports
whether fixed-pin byte evidence covers the exact emitted path mapping, while
`run_eunis_label_attempts()` and `make_eunis_2021_reference_pin()` fail closed
without it. The production runner and full-pool exporter record and verify the
lineage pin plus digests of both the classification mapping and derived paths.
This uses the hierarchy only; no external habitat crosswalk is needed for
L3-to-L2-to-L1 rollups. The separately transferred original taxonomy cache
`classification-taxonomy-original-20261021.json` is 284,146 bytes and matches SHA-256
`223aa3fb89ac48cb5095e443433412c085e66fc54602e2c29253537455a2424f`. Its
classification record ID, declared workbook digest, labels, levels, and parent
structure were checked: 3,950 labels, 3,949 level entries, and 3,942 parent
links. The cache records EEA classification
`bfe4c237-e378-4a83-ab21-b3807f96c2e2` and workbook SHA-256
`1427d22db8d77d1053ab1683aa60713bdec3d810605dd6b7bc16683b4ca14665`.
Original workbook bytes, its direct URL, and external crosswalk values remain
unavailable, so the workbook digest has not been freshly rehashed and the
cache-to-workbook comparison has not run. No production-eligible taxonomy was
created during this recovery; production remains gated until the original
workbook bytes pass the pinned check.

The source-specific text adapter joins downstream text and relevance rows only
after the complete raw inventory and geometry attempts. It validates pinned
Parquet trees and projected Arrow field types before reading rows, uses a
disk-backed SQLite index, and emits bounded batches of body-free
`SourceDocument`, `TextPolygonLink`, and `RelevanceAssessment` rows. Text bodies
are read only to verify or calculate UTF-8 SHA-256 values; the staging database
and pool artifacts contain hashes, IDs, and provenance, never sentence bodies.
Duplicate document keys are indexed in SQLite. Relevance-label and Wiki-link
evidence is read in stable row order into a spooled builder, and the join
rejects a record before returning it if its evidence, link rows, or serialized
size exceeds the declared allocation bounds. It preserves all evidence and
does not truncate oversized groups. `export_eunis_full_pool()` requires a
bounds-carrying join stream whose per-record limits match the selected
`StreamingLimits`, and checks lineage while it consumes the stream before
writing the completion manifest.

Join keys follow the pinned source conventions. Description uses
`description_identity`, tag key, language, sentence index, and the source
description's `text_sha256`; its identity formula matches the source's
`[osm_type, osm_id, tag_key, text_sha256]` SHA-256. Website uses
`polygon_id`, the pinned label-field mapping `website -> website_sentences` and
`contact_website -> contact_website_sentences`, `sentence_index`, and sentence
`text_sha256`. The adapter applies this mapping directly and rejects a supplied
mapping that disagrees with it. Wiki
uses `sentence_id` and `text_sha256`, verifies the stored sentence
`content_hash`, and resolves each `(document_id, project)` link through the
full `(source_pbf, osm_type, osm_id)` key.

Rows without sentences or relevance matches remain represented with typed
MISSING/UNKNOWN states and source-specific split, sentence, or segmentation
reasons. Orphan relevance rows, Wiki link rows without sentence rows, and raw
polygons without any downstream text record are also retained explicitly.
Website text and sentence status columns remain attached to empty-output
records. Direct, linked, unmatched, and geometry-missing link states do not
remove raw inventory rows. The adapter and contract do not perform YES/NO
balancing, H3 sampling, or evaluation draws.

The v2 metadata handoff includes exact file trees for the Wiki link and document
groups. A join consumes the `polygon_document_links/` rows and checks that the
requested corpus document tree is pinned; it does not read document bodies.
The `language_splits/polygon_document_links_by_language/` tree is retained as a
separate pin and is never unioned into primary polygon links. The handoff's
representative Parquet footer schemas are Afghanistan samples, not a scan of all
386 or 375 files; source-wide schema and row integrity remain to be checked
during the bounded production run.

## Staging artifacts and integration boundary

`scan_raw_inventory()` emits `RawInventoryEvidence` and resumable raw shard
checkpoints. `run_eunis_label_attempts()` emits per-shard attempts, counts, and
a pinned run report. `export_eunis_pool()` verifies those receipts, copies the
scan report and attempt shards through the core file-artifact sink, constructs a
`PoolStreamSpec` with `PoolStreamSpec.from_build_input()`, and yields bounded
`PoolBatch` rows to `export_pool_batches()`. The core performs disk-backed
global validation and writes the completion manifest last. Polygon geometry
URIs point to the durable attempt-shard copy in the sink.

`iter_eunis_downstream_records()` validates each pinned text, relevance, and
link table before joining it to the attempt inventory. For Wiki, use
`iter_eunis_wiki_downstream_records()` to include both Wikipedia and Wikivoyage
sentence/label groups in the same streamed export. Pass those records to
`export_eunis_full_pool()`; it appends them after the raw polygon attempts and
still performs no evaluation sampling. If export limits are customized, pass
matching `JoinAllocationBounds.from_streaming_limits(limits)` when building the
downstream stream. Export is lazy: it persists the raw scan report and label
attempt artifacts before consuming downstream records, so a later oversized
record can leave earlier evidence or attempt artifacts in the sink. A failed
export does not write a completion manifest; only a finalized manifest marks a
complete pool.

This uses PR #165's streaming API at contract 1.1.0, including one attempt
receipt per raw key, support evidence for every label candidate, disk-backed
validation, and a bounded nine-pool gate. The filesystem sink and deterministic
source fixtures exercise the downstream join and export path locally. No
source-wide production export or Hub publication was run; raw snapshot key
counts and key-set hashes are still unknown. The raw-first census and EUNIS
label attempts do not use relevance to select keys.

## Verification limits

The implementation tests use deterministic local fixtures and do not download
the 5 GB EEA raster set or source shards. The Description, Website, and Wiki
raw, text, relevance, and required Wiki link/document groups have verified
metadata pins, but unique raw row counts and key-set hashes are still unknown.
The raw data blobs have not been staged or enumerated, so content and row-level
integrity remain unverified. Production attempts and exports remain gated until
the authoritative source manifests are verified, the taxonomy lineage is
checked against the original workbook bytes, and all frozen raw shards pass the
source-wide scan.
