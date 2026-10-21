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
per-class contributing tile IDs, and extraction failures. The class evidence
lists contain only tiles with positive support for that class; missing tiles
are represented in the attempt receipt and remain uncovered denominator area.
Its pool
coverage uses `geodesic_class_coverage`; existing
`prepare_polygons`, `label_polygons`, text joins, and published Hugging Face
files retain their old behavior and degree-based outputs are legacy evidence
only. The new physical-area values are not relabeled legacy results.
The integration iterator `osm_polygon_eunis.pool_adapters.iter_worldcover_source_attempts`
joins those records one to one in raw input order and rejects a count or ID
mismatch.

The canonical LUCAS source eligibility and taxonomy implementation is
`NoeFlandre/osm-polygon-lucas` at commit
`36dfbcb8a5b35b3cff84d6405cf98ebd4f8df04f`. Its explicit C3 mapping and source
manifest are reused as pins in the EUNIS pool bridge at
`osm_polygon_eunis.pool_adapters`; this branch does not edit or copy the shared
pool core. The bridge verifies caller-provided GPKG and codebook bytes and never
downloads either input.

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

`pool_adapters.data/raw-source-tree-pins.json` is a metadata-only snapshot of
the frozen source file trees. The adapter loader verifies each tree's file
count, byte total, and canonical inventory SHA-256 before exposing its pins.
The inventory contains 386 Website `polygons/` files; 386 Description `data/`
and 386 `language-v1/data/` files; 386 Wiki `polygons/` files; and the two Wiki
sentence trees (386 Wikipedia and 375 Wikivoyage files). Sentence trees are
document inputs downstream of the raw polygon scan. Description's second tree
is preserved as a separately pinned geometry candidate group because its
overlap with `data/` has not been resolved. The metadata handoff has no
Parquet row counts or unique native-key totals, so these tree pins do not prove
raw polygon inventory completeness. A production scan must enumerate both
Description groups and independently establish unique native keys before the
shared gate can pass.

| Text source | Frozen repository revision | Geometry tree groups |
| --- | --- | --- |
| Website | `NoeFlandre/osm-polygon-website-tag@4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd` | `polygons/` |
| Description | `NoeFlandre/osm-polygon-description-tag@b4706eb315c66a8a57135289e7612dca8e03caf8` | `data/`, `language-v1/data/` |
| Wiki | `NoeFlandre/osm-polygon-wikidata-and-wikipedia@d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801` | `polygons/` |

## Support rules

WorldCover pool support uses WGS84 ellipsoidal geodesic area in square metres.
The denominator is the full source polygon's geodesic area. Raster interior
cells are weighted by their ellipsoidal row area, and polygon intersections
with boundary cells are measured directly. NoData and unavailable tiles stay
uncovered in the denominator. Class `0` is never a label. The pinned method is
`pyproj.Geod(ellps=WGS84); boundary-cell intersections geodesic; v1`; old
degree-based fractions and published outputs remain legacy evidence.

LUCAS support is measured in square metres in EPSG:3035. The reference pin
identifies the official 2022 Copernicus asset
`l2022_survey_cop_radpoly_attr.gpkg`, layer
`l2022_cop_radpoly_radqs_allatr_wcol`, size `1121144832` bytes, ETag
`42d35000-61bd0bb0d0770`, and SHA-256
`4d240c5710866f9549712685859a5051522b223f3e1fe580dcc9e398b68b037f`. The
reader applies `survey_year=2022`, `point_copernicus=1`,
`surveycprncando=1`, `survey_obs_type=1` (including the source's exact text to
code crosswalk), and non-null `surveycprnlc`; it retains only valid, positive
area Polygon and MultiPolygon geometries. It does not filter by survey date or
buffer survey points.

The official C3 taxonomy PDF is pinned by SHA-256
`1b77c28c7ad70d21c8ea6a26e39029d3e60e7a0170cb60cc9ee520de19153aaf`; the
explicit code map is pinned to the source file in the LUCAS implementation
commit above, SHA-256
`6afc7f26c977e39844301993f45c85008c17c73f3d4c8f6fc2b5bbd5a4ab9fcc`. It maps
8 main classes, 29 second-level classes, and 74 observed codes by exact lookup;
it never infers parents from prefixes. At every supported level, the adapter
groups eligible support footprints by class, unions each class independently,
intersects the union with the full original polygon, then applies the shared
exact `0.8` threshold. It does not choose a fine class and roll that result
upward. Multiple qualifying classes produce an ambiguous outcome; no support
produces an unassigned outcome; unusable geometry or a failed reference query
produces an excluded outcome. The reader records all RTree-hit footprint IDs
and filter counts, while candidate evidence records the eligible IDs that
contributed to each level and class.

## Pin and persistence requirements

The authoritative source manifest and checksum receipt are pinned to the
LUCAS source repository commit above. The GPKG itself is not mounted in this
workspace, so no production bytes were read or pool run performed here. At
runtime the reader fails closed unless the supplied file's exact byte count and
full SHA-256 match the manifest, and the supplied taxonomy PDF matches its
hash. The synthetic GeoPackage fixture is for adapter behavior only and is not
a production reference. The generic Eurostat [LUCAS 2022 C3 classification](https://ec.europa.eu/eurostat/documents/205002/13686460/C3-LUCAS-2022.pdf)
and the distinct JRC [LUCAS Copernicus 2022 dataset](https://data.jrc.ec.europa.eu/dataset/e3fe3cd0-44db-470e-8769-172a8b9e8874)
describe different legends; only the pinned LUCAS source manifest and explicit
codebook mapping are used by this adapter.

The PR165 stream exporter persists one attempt receipt for every raw polygon
and one evidence row for every label candidate. Receipts include attempted
reference IDs, missing tile IDs, and terminal status or reason. Candidate
evidence names only the source IDs that contributed support. The shared schema
pins reference releases and taxonomies but does not yet pin the area
measurement method alongside candidate support evidence. The adapter holds the
WorldCover geodesic method in source code and computes LUCAS areas in EPSG:3035,
but production exports need an explicit shared-core field for those method
pins before their manifests can prove the exact numerator and denominator
procedure.

The current shared `SourceSnapshot` schema only represents a PBF URI and
SHA-256. These frozen sources are multi-file Parquet trees, so production
metadata cannot yet accurately pin their file manifests through that model.
The adapter-side tree pins preserve the exact paths and commits meanwhile;
the shared schema needs a source-tree manifest field before a production pool
can prove which frozen files were scanned. These are API gaps for the
shared-core owner, not duplicated fields added by this adapter branch.
