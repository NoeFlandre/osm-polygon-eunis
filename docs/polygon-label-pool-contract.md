# Polygon-label pool contract

The versioned contract is implemented in
`src/osm_polygon_eunis/pool_contract/`.
It provides a source-independent record model, full-pool validator, nine-pool
gate, and deterministic Parquet exporter. Source adapters provide frozen source
inventories, native reference support measurements, and document-link evidence.
The shared code does not import EUNIS matching logic or compute overlays.

## Pool identity and source coverage

The first contract version is `noeflandre.polygon-label-pool/1.0.0`. A pool is
one of `description`, `website`, or `wiki` crossed with `worldcover`, `lucas`,
or `eunis`. Each pool pins its text-source snapshot, PBF artifact, source
repository revision, reference release, and taxonomy revision.

Adapters must enumerate every distinct native polygon key in the frozen raw
source before joining text, relevance labels, evaluation arms, or token-budget
filters. Each raw key becomes a `PolygonInventoryRow`. Its geometry snapshot may
be absent only when the row records an explicit geometry exclusion. A supported
reference level must then contain exactly one `LabelOutcome` per inventory key.
Rows without YES/NO documents, without text, or without evaluation context still
remain in the pool and receive their reference-label decision.

The EUNIS PR #159 relevance census is candidate evidence, not a source-wide
inventory: its source query starts from relevance-labeled sentence rows. It
cannot be supplied as the EUNIS `inventory` input. Enumerate raw PBF polygon
composite keys first, then record source-native text links. Keep the EUNIS
published largest-intersection label product distinct from the new full-pool
80% assignment rule.

## Logical tables

| Table | Contract contents |
| --- | --- |
| `polygon_inventory` | Every frozen source-native composite key and source-row locator, including rows with no geometry snapshot. |
| `polygon_snapshots` | Recoverable geometry URI and SHA-256, native format, CRS, transform, geometry status, and full original polygon area. |
| `label_candidates` | Long-form polygon × release × level × code support, including below-threshold candidates, threshold, numerator, denominator, ratio, assignment-rule version, support semantics, and direct or ancestor-derived provenance. |
| `label_outcomes` | One assigned, ambiguous, unassigned, or excluded decision per raw polygon and supported level. |
| `source_documents` | Source-native document key, revision, content hash, and explicit available/missing/private/not-retained body state. The table has no body field. |
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
ratio arithmetic, threshold, pin consistency, outcome cardinality, and
accounting. It cannot independently recalculate geometric unions; the source
adapter must compute and declare the per-class union before constructing
`LabelCandidate` rows. Existing scalar EUNIS deepest-winner census results must
be recomputed into independent level decisions. EUNIS direct map support is
available at L3, with ancestor unions at L1–L2; levels L4–L8 should be declared
unsupported until finer map evidence exists.

## Build and export API

Source-specific adapters stay separate from this package. There is no
core-owned loader protocol: an adapter constructs one complete `PoolBuildInput`
and calls `build_full_pool`. The required input fields are `pool_key`,
`source_snapshot`, `raw_inventory_evidence`, `reference_pin`, `adapter_id`,
`adapter_revision`, `builder_revision`, `inventory`, `polygons`, `levels`,
`label_candidates`, and `label_outcomes`. The optional tables are `documents`,
`text_polygon_links`, `entity_links`, and `eligibility`. The input inventory
must come from the full raw polygon snapshot before any text or evaluation
filtering.

`raw_inventory_evidence` is also required. It pins the unfiltered enumeration
rule and scan-report URI/hash, expected raw key count, and SHA-256 of the
canonical sorted composite-key set. The validator compares both values to the
inventory table. The adapter must create this receipt directly from its raw
source scan, before joining text or relevance labels. The digest input is each
key serialized by `native_key_json`, sorted by that serialized form, joined by
literal LF with no trailing LF, then encoded as UTF-8 before SHA-256.

The public record types include `PoolKey`, `SourceSnapshot`,
`RawInventoryEvidence`, `ReferencePin`, `PolygonInventoryRow`,
`PolygonSnapshot`, `LabelLevel`, `LabelCandidate`, `LabelOutcome`,
`SourceDocument`, `TextPolygonLink`, `EntityLink`,
`PoolEligibility`, `PoolManifest`, and `PoolBundle`. `EvaluationCase`,
`EvaluationTrace`, and `SplitAssignment` are separate downstream types. The
complete API is re-exported from `osm_polygon_eunis.pool_contract`.

Construct `PoolBuildInput` with all records, then call:

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
whole inventory, including processed decisions, candidates, assignments,
ambiguous outcomes, and exclusions at every supported level. Export validates
again and writes explicit Arrow schemas as Parquet tables plus a canonical JSON
manifest containing table hashes. `PoolArtifactSink` is the adapter point for a
future cloud sink. Export does not publish to Hugging Face or choose credentials.
The v1 builder and exporter currently materialize the supplied pool and encoded
table payloads in memory. Add a batch or streaming path before running them on
full-source snapshots that exceed worker memory.

The output layout is:

```text
pools/<contract-version>/<text-source>/<reference-system>/
  <source-snapshot-id>/<pool-version>/
    manifest.json
    polygon_inventory.parquet
    polygon_snapshots.parquet
    label_candidates.parquet
    label_outcomes.parquet
    source_documents.parquet
    text_polygon_links.parquet
    entity_links.parquet
    pool_eligibility.parquet
```

The manifest lists all source keys, per-level processed decisions and outcomes,
qualifying candidate counts, exclusions by reason, source and reference pins,
adapter and shared-builder revisions, and every table hash. The nine-pool gate requires each of the nine distinct
combinations and matching contract/pool versions; evaluation sampling must wait
for that gate to pass.

## Splits and migration

The pool contract does not invent split memberships. Any split assignment must
carry a versioned policy and evidence, and it remains separate from geographic
cells used for later evaluation sampling. Preserve the historical WorldCover
H3 resolution-5 / seed-20180101 split only if the exact assignment artifact is
verified. No common split evidence is established here for the other eight
pools. Existing label assignments are not presumed to meet this contract until
revalidated. Migrate by writing versioned new artifacts while retaining old
artifacts for recovery.
