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
| Description | `NoeFlandre/osm-polygon-description-tag` at `b4706eb315c66a8a57135289e7612dca8e03caf8`, `data/*.parquet` (386 shards; 746,727,545 bytes) | `NoeFlandre/osm-polygon-description-tag-landuse` at `ce98b328f3c6543800a3d931877b51eb3f0cf768`, `labels/language-v1/data/*.parquet` | Description fields remain attached to raw rows. |
| Website | `NoeFlandre/osm-polygon-website-tag` at `4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd`, `polygons/*.parquet` (386 shards; 3,943,775,497 bytes) | `NoeFlandre/osm-polygon-website-tag-landuse` at `e6524c711ee0033c3cd5990e5fdbbe4b89992dfb`, `labels/polygons/*.parquet` | Website fields remain attached to raw rows. |
| Wiki | `NoeFlandre/osm-polygon-wikidata-and-wikipedia` at `d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801`, `polygons/*.parquet` (386 shards; 2,203,317,876 bytes) | `NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` at `49307aeb05b6dcc8a5d393f77f59a1e6d13a967a`, `labels/wikipedia/sentences/*.parquet` and `labels/wikivoyage/sentences/*.parquet` | `wikipedia/documents/*.parquet`, `polygon_document_links/*.parquet`, and `wikivoyage/sentences/*.parquet`. |

The [frozen source config at PR #159's reviewed commit](https://github.com/NoeFlandre/osm-polygon-eunis/blob/d96e0651f4d3ac97de4d73f6a6fee1e1294a7fe3/scripts/eunis_geometry_census_sources.py)
records these repository IDs, revisions, and path globs. This is configuration
provenance. It does not verify the raw Hub manifests or shard contents at those
revisions. The label snapshots are separate inputs and never select the raw
polygons to enumerate.

The Wiki repository is explicitly mapped to `TextSource.WIKI`; the older
`DatasetSpec("wikidata", ...)` remains unchanged for legacy consumers. The raw
identity is the complete `(source_pbf, osm_type, osm_id)` key. Its source-row
locator and original geometry payload are retained in the raw checkpoint.

The runner accepts only a frozen repository file inventory whose polygon paths
exactly equal all paths matching the source layout. Each local polygon shard is
checked against the pinned revision, byte count, and SHA-256 before scanning.
The source snapshot must also pin the PBF manifest URI and digest. The scan
report records the source manifest URI/hash, all shard hashes, row counts, the
canonical composite-key-set hash, and the adapter/shared-core revisions. It
uses a disk-backed merge sort to calculate the digest in bounded memory. A
duplicate composite key fails the scan rather than being dropped or merged.

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
and conflicting or incomplete rows. `validate_classification_taxonomy_lineage()`
checks a taxonomy cache against the pinned cache digest, classification record,
workbook digest, and workbook-derived level/parent mapping. After validation,
`make_eunis_taxonomy_from_classification()` creates explicit paths for direct
L3 map codes and requires an L1 and L2 ancestor for every code. This uses the
hierarchy only; no external habitat crosswalk is needed for L3-to-L2-to-L1
rollups. The verified metadata pins are taxonomy cache SHA-256
`223aa3fb89ac48cb5095e443433412c085e66fc54602e2c29253537455a2424f`, EEA
classification record `bfe4c237-e378-4a83-ab21-b3807f96c2e2`, and workbook
SHA-256 `1427d22db8d77d1053ab1683aa60713bdec3d810605dd6b7bc16683b4ca14665`.
The cache does not retain the direct workbook URL or crosswalk values; the
lineage validator requires the workbook bytes and checks their digest before
accepting the cached hierarchy.

Text link resolution is a separate helper. It retains linked, ambiguous,
unmatched, and geometry-missing links without removing source inventory rows.
The shared contract v1.1 also has typed `RelevanceAssessment` records for YES,
NO, UNKNOWN, and MISSING. The EUNIS exporter does not yet emit documents, links,
or relevance rows: the source-specific text schemas and revision-pinned
manifests still need verification before implementing those joins. This keeps
missing-link and relevance states available without guessing at label meaning.
Neither the helper nor the contract performs YES/NO balancing, H3 sampling, or
evaluation draws.

## Staging artifacts and integration boundary

`scan_raw_inventory()` emits `RawInventoryEvidence` and resumable raw shard
checkpoints. `run_eunis_label_attempts()` emits per-shard attempts, counts, and
a pinned run report. `export_eunis_pool()` verifies those receipts, copies the
scan report and attempt shards through the core file-artifact sink, constructs a
`PoolStreamSpec` with `PoolStreamSpec.from_build_input()`, and yields bounded
`PoolBatch` rows to `export_pool_batches()`. The core performs disk-backed
global validation and writes the completion manifest last. Polygon geometry
URIs point to the durable attempt-shard copy in the sink.

This uses PR #165's streaming API at contract 1.1.0, including one attempt
receipt per raw key, support evidence for every label candidate, disk-backed
validation, and a bounded nine-pool gate. The filesystem sink and deterministic
source fixtures exercise this path locally. No source-wide production export
or Hub publication was run. Text documents and YES/NO relevance remain
downstream until their pinned manifests and schemas are verified; the existing
raw-first census and EUNIS label attempts do not use relevance to select keys.

## Verification limits

The implementation tests use deterministic local fixtures and do not download
the 5 GB EEA raster set or source shards. Pinned raw Hub trees contain 386
polygon shards each; their metadata totals are 746,727,545 bytes for Description,
3,943,775,497 bytes for Website, and 2,203,317,876 bytes for Wiki. These tree
totals do not provide unique raw row counts or key-set hashes. The source
manifest and individual shard hashes still need verification at the pinned
revisions. The taxonomy lineage validator is implemented but the cached
hierarchy and workbook bytes have not yet been transferred into this workspace
for a production check. Production attempts and exports remain gated until raw
manifests, the taxonomy lineage, and downstream text/relevance manifests and
schemas are verified.
