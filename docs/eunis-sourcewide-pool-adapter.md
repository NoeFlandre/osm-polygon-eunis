# EUNIS source-wide pool adapter

The EUNIS adapter has two stages. The first scans every polygon row from a
complete, immutable source manifest and writes a per-shard raw checkpoint. Only
after that source-wide key census succeeds does the second stage query EEA
support maps and write one label attempt for every raw polygon. Text documents,
document links, and YES/NO relevance labels are downstream inputs; they do not
select the rows in either stage.

## Frozen source layouts

| Pool source | Repository | Raw polygon shards | Downstream text evidence |
| --- | --- | --- | --- |
| Description | `NoeFlandre/osm-polygon-description-tag` | `data/*.parquet` | Description fields remain attached to raw rows. |
| Website | `NoeFlandre/osm-polygon-website-tag` | `polygons/*.parquet` | Website fields remain attached to raw rows. |
| Wiki | `NoeFlandre/osm-polygon-wikidata-and-wikipedia` | `polygons/*.parquet` | `wikipedia/documents/*.parquet`, `polygon_document_links/*.parquet`, and `wikivoyage/sentences/*.parquet`. |

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

Text link resolution is a separate helper. It retains linked, ambiguous,
unmatched, and geometry-missing links without removing source inventory rows.
It does not perform relevance selection, YES/NO balancing, H3 sampling, or
evaluation draws.

## Staging artifacts and integration boundary

`scan_raw_inventory()` emits `RawInventoryEvidence` and resumable raw shard
checkpoints. `run_eunis_label_attempts()` emits per-shard attempts, counts, and
a pinned run report. `iter_attempt_checkpoint()` streams rows for a later
bounded core loader. The checkpoints are adapter staging artifacts; they are
not nine completed pools and are not a public dataset. The shared-core builder
and exporter on the base branch still materialize complete pools and Parquet
bytes. Production finalization therefore depends on the shared core owner's
bounded batching change. This adapter does not duplicate that serialization or
validation work.

## Verification limits

The implementation tests use deterministic local fixtures and do not download
the 5 GB EEA raster set or source shards. Current Hub tree metadata reports 386
polygon shards per source, but anonymous access to source manifests and shard
content returned HTTP 401. The source revisions are not yet verified as the
required frozen snapshots, and raw row counts/key-set hashes remain unknown.
The local EEA parser exposes code/name labels but no hierarchy mapping. The
adapter therefore requires an explicit L3-to-L1/L2/L3 path mapping and never
guesses ancestry from code prefixes. The native taxonomy artifact's hierarchy
mapping must be provided and verified against the pinned taxonomy digest before
any production attempt.
