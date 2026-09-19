# ADR 0001: EEA reference and exact overlap

## Decision

Use the official EEA 2021 EUNIS habitat probability-map release. Its declared
coordinate system is EPSG:3035. Positive raster cells become reference cells;
the matcher calculates actual geometry intersections and chooses the largest
area. A polygon outside the European reference coverage receives null fields.

Bounding boxes and spatial indexes are permitted only to select candidate
windows. They cannot select the label or calculate the percentage.

## Consequences

The result is reproducible when the source version, asset URLs, ETags, and
checksums are recorded. The modelled raster resolution is part of the result's
meaning and must not be silently mixed with another EEA release.
