# Polygon-label pool contract

The versioned contract is implemented in
`src/osm_polygon_eunis/pool_contract/`.
It provides a source-independent record model, full-pool validator, nine-pool
gate, and deterministic Parquet exporter. Source adapters provide frozen source
inventories, native reference support measurements, and document-link evidence.
The shared code does not import EUNIS matching logic or compute overlays.

## Pool identity and source coverage

The current additive contract version is
`noeflandre.polygon-label-pool/1.2.0`. A pool is one of `description`,
`website`, or `wiki` crossed with `worldcover`, `lucas`, or `eunis`. Each pool
pins its text-source snapshot tree, source repository revision, reference
release, and taxonomy revision. A PBF pin is present when that source snapshot
uses a PBF; other frozen multi-file inventories use their exact artifact
manifest without inventing a PBF.

Adapters must enumerate every distinct native polygon key in the frozen raw
source before joining text, relevance labels, evaluation arms, or token-budget
filters. Each raw key becomes a `PolygonInventoryRow`. Its geometry snapshot may
be absent only when the row records an explicit geometry exclusion. A supported
reference level must then contain exactly one `LabelOutcome` per inventory key.
Rows without YES/NO documents, without text, or without evaluation context still
remain in the pool and receive their reference-label decision.

Each raw inventory key also has one `PolygonAttemptReceipt`, including failed
and geometry-excluded attempts. Every positive or below-threshold label
candidate has one `LabelSupportEvidence` row with source reference IDs or a
resolvable measurement-sidecar URI, SHA-256, and record count. These records
make the adapter's overlay inputs inspectable without changing raw-polygon
accounting.

Evaluation readiness needs more provenance than structural archive validity.
Contract 1.2 adds optional `ArtifactManifestPin` records for the raw source and
reference artifact sets, plus an `AreaMeasurementPin` on the reference pin.
Each artifact manifest is canonical JSONL, sorted by `artifact_uri`; every row
contains exactly `artifact_uri` and `sha256`. The pin stores the manifest URI,
its SHA-256, format version, and expected artifact count. All listed artifacts
are required. When a PBF pin is present, the raw source manifest must list it.
Non-PBF snapshots pin their actual source files directly. The area-method
pin records the implementation and version plus separate `AreaMeasureSpec`
records for numerator and denominator. Each record includes a method ID, a
coordinate system and/or ellipsoid, boundary convention, densification method
and canonical settings, error settings, and output unit. Both outputs must use
the reference support unit. The shared contract does not select or approve a
scientific area method; adapters pin their source-specific choice and settings.

The EUNIS PR #159 relevance census starts from relevance-labeled sentence
rows. It includes only polygons reachable through those rows. Build the EUNIS
inventory by enumerating raw PBF polygon composite keys, then record
source-native text links. Keep the published largest-intersection EUNIS labels
separate from the new full-pool 80% assignment rule.

## Logical tables

| Table | Contract contents |
| --- | --- |
| `polygon_inventory` | Every frozen source-native composite key and source-row locator, including rows with no geometry snapshot. |
| `polygon_snapshots` | Recoverable geometry URI and SHA-256, native format, CRS, transform, geometry status, and full original polygon area. |
| `label_candidates` | Long-form polygon × release × level × code support, including below-threshold candidates, threshold, numerator, denominator, ratio, assignment-rule version, support semantics, and direct or ancestor-derived provenance. |
| `label_outcomes` | One assigned, ambiguous, unassigned, or excluded decision per raw polygon and supported level. |
| `polygon_attempts` | Exactly one typed reference-query receipt per raw polygon, with attempted/missing reference IDs, terminal attempt state, reason, and optional hashed measurement sidecar. |
| `label_support_evidence` | One evidence row per label candidate, retaining reference feature/tile IDs directly or a hashed measurement sidecar. |
| `source_documents` | Source-native document key, revision, content hash, and explicit available/missing/private/not-retained body state. The table has no body field. |
| `document_relevance` | Typed YES/NO/UNKNOWN/MISSING relevance assessments by document and release, with source pins, mapping-rule version, native value, and evidence. Relevance is a downstream text property and never filters raw polygons or labels. |
| `text_polygon_links` | Native link key, link evidence, revision, and linked, ambiguous, unmatched, key-mismatch, text-hash-mismatch, or geometry-missing status. Unresolved links can retain candidate polygon rows. |
| `entity_links` | Versioned cross-snapshot polygon links, with both full source-native keys and evidence. No join uses a bare OSM ID or text hash. |
| `pool_eligibility` | Optional downstream candidate/cohort eligibility, exclusion reasons, and replacement lineage. It does not filter the full pool. |

Train/validation/test membership and evaluation cases/traces are separate typed
records and are not pool tables. Evaluation YES/NO arms are sampling strata,
not data splits. The trace schema records exact prompt and response payload
states and hashes, model, canonical settings JSON, and reference-geometry URI
and hash. A body may be explicitly marked missing, private, or not retained.
This schema does not authorize public release of private text.

## Reference assignment rule

Each candidate numerator is the within-class union support intersecting the
polygon. The denominator is the full original polygon area; uncovered and
NoData area remains in that denominator. The adapter declares the source support
meaning and same-class union method. Meanings are pinned as mapped cover for
WorldCover, surveyed footprint for LUCAS, and modeled habitat support for EUNIS.

The shared rule assigns a code when its support ratio is at least `0.8`,
inclusive. Exactly one qualifying code at a level is assigned. Multiple
qualifiers or explicitly ambiguous evidence produce an unassigned ambiguous
outcome. A level is resolved independently: assigning an ancestor does not
prune a qualifying descendant or vice versa. Every candidate records its
numerator, full-area denominator, ratio, threshold, support unit, source and
taxonomy pins, assignment-rule version, and direct/ancestor-derived kind.

The validator checks denominator equality against the polygon's original area,
unit equality between polygon area and candidate support, ratio arithmetic,
threshold, pin consistency, outcome cardinality, and accounting. It cannot
recalculate geometric unions. The source adapter must compute and declare each
per-class union before it constructs `LabelCandidate` rows. It must also record
the contributing reference IDs or a hashed measurement sidecar in
`LabelSupportEvidence`. Recompute scalar EUNIS deepest-winner census results
into independent level decisions. EUNIS direct map support covers L3, and
ancestor unions cover L1–L2. Declare levels L4–L8 unsupported until finer map
evidence exists.

## Build and export API

Source-specific adapters stay separate from this package. For fixtures and
small pools, an adapter can construct one complete `PoolBuildInput` and call
`build_full_pool`. Its required fields are `pool_key`, `source_snapshot`,
`raw_inventory_evidence`, `reference_pin`, `adapter_id`, `adapter_revision`,
`builder_revision`, `inventory`, `polygons`, `levels`, `label_candidates`,
`label_outcomes`, and `attempt_receipts`. Each candidate requires one matching
`support_evidence` row. Optional text-related tables are `documents`,
`relevance_assessments`, and `text_polygon_links`. `entity_links` and
`eligibility` retain separate lineage. Build the input inventory from the full
raw polygon snapshot before text or evaluation filtering.

`raw_inventory_evidence` is also required. It pins the unfiltered enumeration
rule and scan-report URI/hash, expected raw key count, and SHA-256 of the
canonical sorted composite-key set. The validator compares both values to the
inventory table. The adapter must create this receipt directly from its raw
source scan, before joining text or relevance labels. The digest input is each
key serialized by `native_key_json`, sorted by that serialized form, joined by
literal LF with no trailing LF, then encoded as UTF-8 before SHA-256.

`RelevanceAssessment` uses `RelevanceStatus.YES` or `.NO` only when the native
value and mapping provenance support that result. Use `.UNKNOWN` for a present
but unresolved value and `.MISSING` with an absence reason when a source record
has no relevance value. Do not coerce either state into an evaluation arm.

The public record types include `PoolKey`, `SourceSnapshot`,
`RawInventoryEvidence`, `ReferencePin`, `PolygonInventoryRow`,
`PolygonSnapshot`, `LabelLevel`, `LabelCandidate`, `LabelOutcome`,
`PolygonAttemptReceipt`, `LabelSupportEvidence`, `SourceDocument`,
`RelevanceAssessment`, `TextPolygonLink`, `EntityLink`,
`PoolEligibility`, `PoolManifest`, and `PoolBundle`. `EvaluationCase`,
`EvaluationTrace`, and `SplitAssignment` are separate downstream types. The
complete API is re-exported from `osm_polygon_eunis.pool_contract`.

The in-memory API constructs `PoolBuildInput` with all records, then calls:

```python
from osm_polygon_eunis.pool_contract import (
    FilesystemSink,
    PoolBuildInput,
    build_full_pool,
    export_full_pool,
    validate_full_pool,
)

bundle = build_full_pool(PoolBuildInput(...))
report = validate_full_pool(bundle)
if not report.ok:
    raise ValueError(report.issues)
receipt = export_full_pool(bundle, FilesystemSink(output_directory))
```

`build_full_pool` sorts records deterministically and computes manifest counts;
it does not select, drop, or label polygons. `validate_full_pool` checks the
whole inventory and decisions. `export_full_pool` validates again and writes
explicit Arrow schemas as Parquet tables plus a canonical JSON manifest.
`PoolArtifactSink` is the in-memory exporter interface.

For full source snapshots, use the disk-backed streaming API. Create a
`PoolStreamSpec` from the pool-wide metadata and independent raw-inventory
receipt, then yield bounded `PoolBatch` objects to `export_pool_batches`:

```python
from osm_polygon_eunis.pool_contract import (
    FilesystemStreamingSink,
    PoolBatch,
    PoolStreamSpec,
    StreamingLimits,
    export_pool_batches,
)

spec = PoolStreamSpec.from_build_input(metadata_input)
receipt = export_pool_batches(
    spec,
    batches,
    FilesystemStreamingSink(output_directory),
    limits=StreamingLimits(),
)
```

`metadata_input` carries the pins, levels, and independent raw-source count and
key-set digest; its row tables can be empty. Each batch has a zero-based,
consecutive `batch_index` and any combination of inventory, geometry,
candidate, outcome, document, link, entity-link, or eligibility rows. Rows may
be split across batches. Attempt receipts, support evidence, and relevance
assessments can also be split across batches. Yield one batch at a time and
keep it within the configured limits. The default maximum is 2,048 contract
rows and 32 MiB estimated input per batch. Hard ceilings are 10,000 rows and
64 MiB.
Each stored row is capped at 1 MiB, and each Parquet shard at 8,192 rows and
64 MiB of uncompressed row JSON. Validation returns at most 100 issue details
plus one truncation marker.

The stream is spooled into a local SQLite index with a 4 MiB page cache and
file-backed temporary data. Global uniqueness, raw inventory count and
key-set digest, geometry links, document links, cross-batch references, and
one outcome per supported level, one attempt receipt per raw key, one support
evidence row per candidate, and relevance-document references are checked from
that index before shard writing. The exporter revalidates candidate
measurements, area units, provenance, ambiguity, and exclusions. It stores
staged rows in scratch SQLite files, then reads them sequentially to create
canonical shards.

Logical table rows are sorted by stable contract keys and split into fixed
shards, so the same inputs produce the same files regardless of batch
boundaries. The stream writes under a versioned layout directory. It writes
`manifest.json` last as the completion marker; an interrupted write can leave
immutable shards without a completion manifest. An identical retry reuses those
files, while a conflicting retry fails. The existing unsharded exporter and
its output paths remain unchanged. Its filesystem sink uses atomic create-only
writes and rejects conflicting concurrent writers. A cloud adapter implements
`FileArtifactSink` with atomic immutable `put_file` behavior. The filesystem
sink supports controlled staging and tests. Streaming does not publish to
Hugging Face or select credentials.

The output layout is:

```text
pools/<contract-version>/<text-source>/<reference-system>/
  <source-snapshot-id>/<pool-version>/
    layouts/sharded-parquet-v1/
      shards/<logical-table>/part-00000000.parquet
      shards.jsonl
      exclusions.jsonl
      manifest.json
```

The streaming manifest lists the total source-key count and observed key-set
digest, per-level processed decisions and outcomes, qualifying candidate
counts, the exclusion-reason sidecar, source and reference pins, adapter and
shared-builder revisions, and the hash and row count of every shard.
`validate_nine_streamed_pool_gate(receipts, reader)` verifies structural archive
completeness: manifests, scan receipts, shard inventories, shard hashes,
per-level accounting, and all nine distinct combinations without
reconstructing pools in RAM. It remains useful for validating preserved
archives even when they contain failed or partial attempts.

Before evaluation sampling, use
`validate_nine_streamed_evaluation_ready_gate(receipts, reader)`. It first
requires the structural gate to pass, then reads attempt, geometry, and support
evidence rows in bounded Parquet batches. `FAILED` and `PARTIAL` attempts,
malformed or non-empty `missing_reference_ids`, unreadable geometry or
measurement sidecars, missing source/reference artifact manifests, unavailable
listed shards, missing area-method provenance, inconsistent area units, or
PBF/hash mismatches block readiness. A `COMPLETED` attempt with no matching
reference IDs remains valid:
natural lack of reference coverage is a scientific coverage caveat, not an
operational read failure. An explicit `EXCLUDED` outcome with its reason also
remains accounted. Adapters must record operational read errors as `FAILED` or
`PARTIAL`, and list missing required reference IDs, rather than encoding them
as natural no-coverage.

A cloud `PoolArtifactReader` streams JSONL and artifact hashes, reads Parquet
footer metadata without loading row groups, and implements
`iter_parquet_rows(uri, columns, batch_size=2048)` as a bounded selected-column
row iterator. For strict readiness, it must also implement
`iter_verified_lines(uri, expected_sha256)` and
`iter_verified_parquet_rows(uri, expected_sha256, columns, batch_size=2048)` so
the bytes hashed are the same snapshot later parsed or scanned. A cloud reader
must use an immutable object version or copy the object to a bounded temporary
snapshot; it must not hash one read and reopen a mutable URI for the decision.
The filesystem reader copies one artifact at a time to temporary storage,
checks its pin, then scans that same copy, keeping row memory bounded. The gate
checks each Parquet footer's schema and row count before reading rows.
`FilesystemArtifactReader` supports controlled staging and tests. Hash and
manifest checks bind files to the supplied pins; they do not authenticate who
produced a manifest or independently prove that its file list contains every
object in the pinned source release. Adapters must build these lists from the
frozen source tree or reference catalog, and keep per-polygon required-reference
IDs complete. The gate also does not verify the scientific correctness of area
measurements. Evaluation sampling must wait for the strict readiness gate to
pass and for source-manifest provenance review.

These checks bind shard contents to the supplied manifests and receipts. They
do not authenticate who produced those records or verify the underlying
scientific measurements. A trusted signature or external trust anchor remains
outside this contract.

## Splits and migration

The pool contract does not invent split memberships. Any split assignment must
carry a versioned policy and evidence, and it remains separate from geographic
cells used for later evaluation sampling. Preserve the historical WorldCover
H3 resolution-5 / seed-20180101 split only if the exact assignment artifact is
verified. No common split evidence is established here for the other eight
pools. Existing label assignments are not presumed to meet this contract until
revalidated. Migrate by writing versioned new artifacts while retaining old
artifacts for recovery.
