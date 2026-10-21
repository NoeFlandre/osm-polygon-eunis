# Source wide WorldCover and LUCAS adapters

These adapters emit label measurements for each raw polygon row before
document matching, relevance labels, or evaluation selection. Their iterators
avoid holding the whole raw inventory in the adapter. A single polygon may
still intersect many LUCAS rows; production waits for bounded per-polygon
reference aggregation and verified source pins.

## Implementation homes and shared core

`osm-worldcover` remains the canonical home of ESA raster tile access and the
legacy degree-based `class_coverage`. Its source-wide iterator is
`osm_worldcover.adapters.worldcover.pool_attempts.iter_sourcewide_coverage`.
It accepts the unfiltered polygon table as bounded DataFrame row groups,
attempts every row, and emits coverage, physical area, tile IDs, missing tiles,
and extraction failures. Its pool
coverage uses `geodesic_class_coverage`; existing
`prepare_polygons`, `label_polygons`, text joins, and published Hugging Face
files retain their old behavior and degree-based outputs are legacy evidence
only. The new physical-area values are not relabeled legacy results.
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
inventory receipt from a later filtered table. The source-specific streaming
runner constructs bounded `PoolBatch` objects and calls PR165's
`PoolStreamSpec.from_build_input` and `export_pool_batches`.
The one-pass `iter_worldcover_rows`, `iter_worldcover_source_rows`, and
`iter_lucas_rows` feeds pair each raw polygon with its decision for
`export_labeled_polygons`.
Document and text-link rows can arrive in bounded supplemental batches, so
unmatched links remain represented. Shared core code is reused from PR165 and
is not copied or modified in this branch. Keep the three text source joins
downstream.

## Support rules

WorldCover pool support uses WGS84 ellipsoidal geodesic area in square metres.
The denominator is the full source polygon's geodesic area. Raster interior
cells are weighted by their ellipsoidal row area, and polygon intersections
with boundary cells are measured directly. NoData and unavailable tiles stay
uncovered in the denominator. Class `0` is never a label. The pinned method is
`pyproj.Geod(ellps=WGS84); boundary-cell intersections geodesic; v1`; old
degree-based fractions and published outputs remain legacy evidence.

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
The generic Eurostat [LUCAS 2022 C3 classification](https://ec.europa.eu/eurostat/documents/205002/13686460/C3-LUCAS-2022.pdf)
and the distinct JRC [LUCAS Copernicus 2022 dataset](https://data.jrc.ec.europa.eu/dataset/e3fe3cd0-44db-470e-8769-172a8b9e8874)
describe different legends. Neither source proves which legend is in the
unmounted footprint GPKG, so its layer and codebook must be resolved together.

Adapter attempt receipts currently include attempted IDs, missing tile IDs,
per-class evidence IDs, and a terminal status or reason. PR165's stream
exporter does not accept or persist these per-polygon receipts, the geodesic
measurement-method pin, or a hash for an attempt-evidence sidecar. The adapter
can stream candidate and outcome tables, but the completion manifest cannot
yet prove the full attempt trail. The shared core owner should add a bounded
attempt-evidence artifact and manifest reference before production export; do
not fork the core schema in this repository.
