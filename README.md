# OSM Polygon EUNIS

This project enriches three public Hugging Face datasets with EUNIS habitat
labels selected by the largest actual spatial intersection between each OSM
polygon and the official EEA EUNIS habitat probability-map reference.

Inputs:

- `NoeFlandre/osm-polygon-website-tag`
- `NoeFlandre/osm-polygon-wikidata-and-wikipedia`
- `NoeFlandre/osm-polygon-description-tag`

Outputs:

- `NoeFlandre/osm-polygon-website-tag-eunis`
- `NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis`
- `NoeFlandre/osm-polygon-description-tag-eunis`

The polygon-bearing tables receive four nullable columns:

- `eunis_code`
- `eunis_name`
- `eunis_overlap_percentage`
- `eunis_source_version`

The percentage is `100 * area(polygon ∩ EUNIS reference geometry) /
area(polygon)` after transforming both geometries to EPSG:3035. Bounding boxes
may prune candidates, but never determine a label. Empty, invalid, or
out-of-reference polygons receive null EUNIS fields. Equal-area ties use the
ascending EUNIS code.

The pipeline reads one Parquet shard at a time, uses bounded Arrow batches,
streams temporary files, and deletes each shard after verification. It does not
mirror a complete input or output dataset on the local disk.

## Local development

Use a task-scoped uv cache when working on the mounted data volume:

```bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv sync --group dev
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ruff check .
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ty check src tests scripts
```

The complete release procedure is documented in `docs/operations.md`.
