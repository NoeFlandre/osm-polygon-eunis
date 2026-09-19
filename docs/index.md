# OSM Polygon EUNIS

The project adds provenance-aware EUNIS labels to OSM polygon datasets while
preserving source rows, columns, configurations, and shared tables.

The correctness rule is exact polygon/reference geometry intersection in
EPSG:3035. Spatial indexes and bounding boxes are performance filters only.

See the [operations runbook](operations.md) for commands and release gates.
