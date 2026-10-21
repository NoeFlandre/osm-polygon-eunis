# Source wide WorldCover and LUCAS adapters

These adapters emit label measurements for each raw polygon row before
document matching, relevance labels, or evaluation selection. Their one row at
a time iterators avoid holding the whole raw inventory in the adapter. A single
polygon may still intersect many LUCAS rows. The shared `PoolBuildInput`,
builder, and Parquet exporter also materialize complete pool tables in memory,
so production waits for bounded union and shared-core batching.

## Implementation homes and shared core

`osm-worldcover` remains the canonical home of ESA raster tile access and
`class_coverage`. Its source wide iterator is
`osm_worldcover.adapters.worldcover.pool_attempts.iter_sourcewide_coverage`.
It reads the unfiltered polygon table, attempts every row, and emits coverage,
tile IDs, missing tiles, and extraction failures. Existing
`prepare_polygons`, `label_polygons`, text joins, and published Hugging Face
files retain their old behavior and are legacy output evidence only.
The integration iterator `osm_polygon_eunis.pool_adapters.iter_worldcover_source_attempts`
joins those records one to one in raw input order and rejects a count or ID
mismatch.

No LUCAS adapter or source project was present in the mounted repositories.
The manifest driven LUCAS reader and the contract bridge therefore live under
`osm_polygon_eunis.pool_adapters` for now. The LUCAS reference GPKG and codebook
remain external inputs; the reader requires their exact manifest pins and does
not fetch either one.

Both adapters call the existing `osm_polygon_eunis.pool_contract` models. The
source scanner supplies a pre-join count and `native_keyset_sha256` digest to
`build_raw_inventory_evidence`; the adapter never derives the expected raw
inventory receipt from a later filtered table. Tests pass their records to the existing
`build_full_pool` and `validate_full_pool`. The shared core is not copied or
modified in this branch. Keep the three text source joins downstream and retain
unmatched or ambiguous link records in the core's link table.

## Support rules

WorldCover adapts the canonical extractor's class fractions on each polygon's
full planar EPSG:4326 area. NoData and unavailable tiles stay in the denominator
as uncovered area. Class `0` is never a label. The present raster semantics use
degree squared area, so changing to physical area would require a separate,
reviewed reference-system decision.

LUCAS support is measured in square metres in EPSG:3035. At every supported
level, the adapter groups raw survey footprints by that level's source code,
unions each class's footprints independently, intersects the union with the
full original polygon, then applies the shared exact `0.8` threshold. It does
not choose a fine class and roll that result upward. Multiple qualifying
classes produce an ambiguous outcome; no support produces an unassigned
outcome; unusable geometry or a failed reference query produces an excluded
outcome. The reader records the source footprint IDs returned by its RTree
query, while each polygon snapshot retains its source geometry pin.

## Pin and persistence requirements

No authoritative LUCAS file and taxonomy manifest was available in the mounted
workspace. Do not use the incorrect handoff hash, approximate file size, or
unverified taxonomy counts as a source pin. Production requires the checked in
or independently verified manifest's exact byte count, SHA-256, release URI,
layer/table/column names, and codebook URI, revision, and SHA-256. The synthetic
GeoPackage fixture is for adapter behavior only and is not a production pin.

Adapter attempt receipts currently include attempted IDs, missing tile IDs,
per class evidence IDs, and a terminal status/reason. The shared core does not
yet define an exported attempt receipt table or store LUCAS footprint IDs and
the measurement sidecar hash alongside each label candidate. Add that to the
shared core before treating an exported pool as fully auditable; do not fork
the core schema in this repository.
