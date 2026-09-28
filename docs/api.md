# Python API

The modules below expose stable helpers for Python callers. The production
release uses the same functions through the CLI.

## Resolve EEA reference data

`osm_polygon_eunis.eea.resolve_config(path)` reads the project's EEA reference
JSON, retrieves classification labels and resolves the configured raster or
GeoPackage assets. It returns an ordered tuple of `EeaGroup` values. It performs
bounded metadata requests and raises `ValueError` for invalid config or
metadata and `httpx.HTTPError` when a request still fails after retries.

Use `resolve_config_data(document)` when the JSON has already been parsed.

## Inspect a pinned Hub revision

`osm_polygon_eunis.sources.list_parquet_files(api, repo_id, revision)` returns
the Parquet paths at one immutable Hugging Face dataset commit, sorted by path.
Passing a commit SHA makes inventory repeatable if the repository's default
branch moves later.

## Validate polygon areas

`osm_polygon_eunis.geometry.safe_area(geometry)` returns the positive area of a
valid, non-empty geometry, or `0.0` when the value is absent, empty, invalid, or
has no positive area. The units are those of the geometry's CRS.

## Enrich a local shard

`osm_polygon_eunis.transform.enrich_parquet_shard(source, destination, *,
reference, batch_size, geometry_column="geometry")` streams a Parquet shard in
bounded batches, preserves the source rows and fields, appends the EUNIS result
columns, and returns the number of rows written. `batch_size` must be positive
and `geometry_column` must exist. The function writes `destination`, so callers
should provide a staging path when they need atomic replacement.
