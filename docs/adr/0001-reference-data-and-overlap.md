# ADR 0001: EEA reference and exact overlap

## Decision

Use the official EEA 2021 EUNIS habitat probability-map release. Its declared
coordinate system is EPSG:3035. Positive raster cells become reference cells;
the matcher calculates actual geometry intersections and chooses the largest
area. A polygon outside the European reference coverage receives null fields.

The EEA raster families published as GeoPackages are tile pyramids rather than
feature layers. The adapter reads only highest-resolution tiles intersecting a
polygon bbox, decodes positive pixels into EPSG:3035 cell geometry, and then
uses the same exact intersection matcher. Raster-cell collections explicitly
carry their disjoint-component invariant so the matcher can sum Shapely's
vectorized exact cell intersections; arbitrary geometry collections still use
the general overlay path. A bounded decoded-tile cache is used only as a
performance optimization.

Names come from the EEA 2021 classification workbook in catalog record
`bfe4c237-e378-4a83-ab21-b3807f96c2e2`. The saltmarsh service has one null
name (`MA223`); its official EUNIS name is pinned in the configuration as a
small supplemental label rather than inferred from geometry.

Bounding boxes and spatial indexes are permitted only to select candidate
windows. They cannot select the label or calculate the percentage.

Reference groups are staged once but opened in bounded batches: at most two
raster groups are opened together, while adjacent vector groups share a batch.
Workers retain only a bounded source micro-batch while applying every reference
batch, then delete those source shards. Compact sidecars make interrupted
batches resumable. This avoids mirroring the input inventory while also avoiding
one source download per reference group.

## Consequences

The result is reproducible when the source version, asset URLs, ETags, and
checksums are recorded. The modelled raster resolution is part of the result's
meaning and must not be silently mixed with another EEA release.
