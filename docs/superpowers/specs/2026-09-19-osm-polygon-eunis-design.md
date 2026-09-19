# OSM Polygon EUNIS Enrichment

## Goal

Create three public Hugging Face datasets by enriching the existing datasets with
EUNIS habitat labels assigned by spatial overlap:

- `NoeFlandre/osm-polygon-website-tag-eunis`
- `NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis`
- `NoeFlandre/osm-polygon-description-tag-eunis`

The source datasets remain the source of truth for all existing columns and
records. The new datasets add four nullable columns to the polygon-bearing
tables:

- `eunis_code`
- `eunis_name`
- `eunis_overlap_percentage`
- `eunis_source_version`

The code will live in the public GitHub repository `NoeFlandre/osm-polygon-eunis`.

## Source and matching contract

The reference source is the official EEA 2021 EUNIS modelled probability-map
release. The run records the exact EEA asset URLs, release metadata, checksums,
coordinate reference system, and source version in a machine-readable manifest.
The EEA layer is Europe-focused; polygons with no intersecting reference area
receive null EUNIS fields.

The most detailed EUNIS code present in the selected reference data is used. A
polygon may intersect several EUNIS reference shapes or cells. The winning label
is the one with the greatest actual intersection area. The overlap percentage is:

```text
100 * area(input_polygon ∩ winning_eunis_geometry) / area(input_polygon)
```

Bounding boxes and spatial indexes are allowed only to prune candidates. They
must never determine the label or the percentage. Empty, invalid, or unusable
geometries produce null EUNIS fields and a diagnostic record. Equal-area ties
are resolved deterministically by ascending EUNIS code.

## Dataset-preservation contract

The pipeline preserves all source configurations and non-target files.

Geometry-bearing tables are enriched as follows:

- website-tag: `polygons/*`
- description-tag: `data/*`
- wikidata-and-wikipedia: `polygons/*` and `polygon_document_links/*`

The wikidata-and-wikipedia document, section, sentence, Wikidata-fact, and
Wikivoyage tables remain unchanged because they are shared by many polygons and
do not have a one-to-one polygon label. `polygon_document_links` receives the
label looked up by its `polygon_id`.

## Data flow and storage strategy

1. Capture the current source `main` commit for each input and record it before
   processing.
2. Duplicate each source dataset repository server-side into its `-eunis`
   target repository, preserving public metadata and unchanged files.
3. Resolve and validate the pinned EEA reference assets once.
4. Process one region/shard at a time. Read only the required columns, parse
   geometry in bounded batches, perform candidate pruning, calculate exact
   geometry intersections, and write one replacement Parquet shard.
5. Upload each changed shard immediately, verify its schema and row count, then
   remove the temporary local artifact.
6. For wikidata-and-wikipedia, derive the polygon label map while processing a
   region's `polygons` shard and use it to enrich the matching
   `polygon_document_links` shard without building a global polygon cache.
7. Verify the final remote revisions, inventories, schemas, row counts, source
   references, and representative labels independently of upload success.

No complete source dataset or complete output dataset is materialized on the
local disk. Worker count, batch size, temporary bytes, and memory use are
bounded by configuration and reported by the CLI.

## Module boundaries

- `sources`: Hub revisions, file inventories, shard pairing, and remote reads.
- `eunis`: EEA asset resolution, reference indexing, CRS handling, and reference
  geometry access.
- `matching`: pure overlap calculation, winner selection, tie handling, and
  nullable result construction.
- `transform`: Arrow/Parquet batch transformation and schema preservation.
- `publish`: server-side duplication, single-shard commits, verification, and
  cleanup.
- `cli`: scriptable `plan`, `run`, `verify`, and `clean` commands.

I/O, network, filesystem, and publishing code stay at the edges. Geometry
matching and output-field decisions remain pure functions with small stable
interfaces.

## Quality and acceptance gates

Development follows strict RED -> GREEN -> REFACTOR TDD. The test suite includes:

- unit tests for geometry overlap, null/no-match behavior, ties, CRS, and schema
  preservation;
- Hypothesis properties for area bounds, permutation invariance, and deterministic
  tie resolution;
- integration tests using tiny deterministic Parquet shards and a synthetic
  EUNIS reference layer;
- executable Gherkin acceptance scenarios for end-to-end shard enrichment,
  unchanged configurations, and remote-verification receipts;
- automatic dependency-boundary and import-cycle checks;
- coverage, CRAP, complexity, mutation, and smoke checks.

The final deterministic QA order is baseline -> Ruff -> ty -> tests -> property
tests -> acceptance tests -> architecture checks -> CRAP -> mutation tests ->
smoke test -> diff review. Remote publication and Dataset Viewer verification
are reported separately from local checks.

## External project artifacts

The GitHub repository is public and linked to the GeoReSeT project. Four issues
are created and tracked: one shared pipeline issue and one issue for each output
dataset. The three public datasets are added to the public Hugging Face
collection `OSM Polygon EUNIS`.

## Decisions and known limitations

- Server-side duplication and shard-at-a-time updates minimize local storage and
  preserve unchanged source data.
- The EEA reference release is European in coverage; null labels outside its
  coverage are intentional, not inferred.
- Reference-map resolution and modelled probabilities are part of the label
  provenance. They are not silently mixed with newer or lower-resolution data.
- Exact polygon/reference intersection is the correctness rule; spatial indexes
  and batch parallelism are performance mechanisms only.

