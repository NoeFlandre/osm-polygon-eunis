# OSM Polygon EUNIS

The project adds EUNIS labels, with provenance, to the OSM polygon datasets. It
keeps the source rows, columns, configurations, and shared tables.

The correctness rule is the exact intersection of the polygon and the reference
geometry in EPSG:3035. The spatial indexes and bounding boxes are performance
filters only.

For the commands and the release gates, see the
[operations runbook](operations.md). For the project terms, see the
[glossary](glossary.md).
