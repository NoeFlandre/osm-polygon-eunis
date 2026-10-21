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
| Description | `NoeFlandre/osm-polygon-description-tag` at `b4706eb315c66a8a57135289e7612dca8e03caf8`, `data/*.parquet` | `NoeFlandre/osm-polygon-description-tag-landuse` at `ce98b328f3c6543800a3d931877b51eb3f0cf768`, `labels/language-v1/data/*.parquet` | Description fields remain attached to raw rows. |
| Website | `NoeFlandre/osm-polygon-website-tag` at `4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd`, `polygons/*.parquet` | `NoeFlandre/osm-polygon-website-tag-landuse` at `e6524c711ee0033c3cd5990e5fdbbe4b89992dfb`, `labels/polygons/*.parquet` | Website fields remain attached to raw rows. |
| Wiki | `NoeFlandre/osm-polygon-wikidata-and-wikipedia` at `d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801`, `polygons/*.parquet` | `NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse` at `49307aeb05b6dcc8a5d393f77f59a1e6d13a967a`, `labels/wikipedia/sentences/*.parquet` and `labels/wikivoyage/sentences/*.parquet` | `wikipedia/documents/*.parquet`, `polygon_document_links/*.parquet`, and `wikivoyage/sentences/*.parquet`. |

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
and conflicting or incomplete rows. The source-wide runner still requires an
explicit map-code-to-L1/L2/L3 mapping. The parser does not select or fetch a
workbook, because the workbook URL, crosswalk rows, and artifact checksum have
not been verified against the classification record. The old classification
URL is not used as a workbook URL.

Text link resolution is a separate helper. It retains linked, ambiguous,
unmatched, and geometry-missing links without removing source inventory rows.
It does not perform relevance selection, YES/NO balancing, H3 sampling, or
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

This uses PR #165's streaming API added at `26a6ce9`; this branch is stacked on
that open PR. The filesystem sink and deterministic source fixtures exercise
the path locally. No source-wide production export or Hub publication was run.
Text documents and YES/NO relevance remain downstream; the shared contract has
no typed YES/NO relevance record yet, so this adapter does not encode that label
as pool eligibility or a reference outcome.

## Verification limits

The implementation tests use deterministic local fixtures and do not download
the 5 GB EEA raster set or source shards. Hub tree metadata reports 386 polygon
shards per raw source, but the revision-specific raw manifests and shard bytes
have not been validated in this work. Unique raw row counts and key-set hashes
remain unknown. The EEA workbook URL, crosswalk rows, and artifact checksum also
remain unverified. The parser exposes the reviewed prefix hierarchy rule, while
the runner still accepts only an explicit taxonomy mapping whose digest matches
the EEA taxonomy pin. Production attempts and exports remain gated until source
manifest pins, the workbook artifact, and the complete code-to-path mapping are
verified.
