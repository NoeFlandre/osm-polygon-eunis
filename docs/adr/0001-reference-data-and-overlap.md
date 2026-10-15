# ADR 0001: EEA reference and exact overlap

## Decision

Use the official EEA 2021 EUNIS habitat reference. Its declared coordinate
system is EPSG:3035. For each exact EUNIS code, union its reference geometries
before calculating the exact intersection with the OSM polygon. Divide each
per-code intersection by the full OSM polygon area. Assign a code and name only
when exactly one code reaches the inclusive 80% threshold. Keep hierarchy codes
distinct; if more than one qualifies, withhold the code and name and retain the
largest per-code union coverage. Below 80%, withhold the code and name but keep
the measured percentage. A polygon with no valid positive-area intersection
receives null overlap fields.

Vector GeoPackages are queried as indexed feature layers. Raster-tile
GeoPackages are refused as ground truth until their source values and validity
masks are verified. Keep the bounded Q11 source evidence and recovery record in
the separately scoped [PR #157](https://github.com/NoeFlandre/osm-polygon-eunis/pull/157).
Raster labels require a validated native lossless raster and its native
validity mask. Do not infer labels from an unchecked derived tile package.

Native raster-cell collections carry their disjoint-component invariant. The
matcher sums exact cell intersections that Shapely calculates in a vectorized
way. Arbitrary geometry collections still use the general overlay path.

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
