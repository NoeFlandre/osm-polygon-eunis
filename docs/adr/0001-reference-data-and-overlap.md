# ADR 0001: EEA reference and exact overlap

## Decision

Use the official EEA 2021 EUNIS habitat probability-map release. Its declared
coordinate system is EPSG:3035. The positive raster cells become reference
cells. The matcher calculates the actual geometry intersections. It chooses the
largest area. A polygon outside the European reference coverage receives null
fields.

The EEA raster families that are published as GeoPackages are tile pyramids.
They are not feature layers. The adapter reads only the highest-resolution tiles
that intersect the polygon bbox. It decodes the positive pixels into EPSG:3035
cell geometry. Then it uses the same exact intersection matcher.

The raster-cell collections carry their disjoint-component invariant. This lets
the matcher sum the exact cell intersections that Shapely calculates in a
vectorized way. Arbitrary geometry collections still use the general overlay
path. A bounded decoded-tile cache is a performance optimization only.

The names come from the EEA 2021 classification workbook in the catalog record
`bfe4c237-e378-4a83-ab21-b3807f96c2e2`. The saltmarsh service has one null name
(`MA223`). The configuration pins its official EUNIS name as a small
supplemental label. The name is not inferred from geometry.

Use bounding boxes and spatial indexes only to select candidate windows. They
cannot select the label. They cannot calculate the percentage.

The pipeline stages the reference groups once. It opens them in bounded batches:

- It opens at most two raster groups together.
- Adjacent vector groups share a batch.

Workers keep only a bounded source micro-batch while they apply every reference
batch. Then they delete those source shards. Compact sidecars make the
interrupted batches resumable.

Each worker process reuses the opened reference handles and their decoded-tile
caches across geometry tasks. This has three results:

- The pipeline does not mirror the input inventory.
- The pipeline does not download the source again for each reference group.
- A worker does not read the reference again and again.

## Consequences

The result is reproducible when you record the source version, the asset URLs,
the ETags, and the checksums. The modelled raster resolution is part of the
meaning of the result. Do not mix it with another EEA release.
