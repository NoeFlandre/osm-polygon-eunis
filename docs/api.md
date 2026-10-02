# Python API

The modules below give stable helpers to Python callers. The production release
uses the same functions through the CLI.

## Resolve EEA reference data

`osm_polygon_eunis.eea.resolve_config(path)` reads the EEA reference JSON of the
project. It gets the classification labels. It resolves the configured raster or
GeoPackage assets. It returns an ordered tuple of `EeaGroup` values.

The function makes a bounded number of metadata requests. It raises
`ValueError` for an invalid config or invalid metadata. It raises
`httpx.HTTPError` when a request fails after the retries.

If you already parsed the JSON, use `resolve_config_data(document)`.

## Inspect a pinned Hub revision

`osm_polygon_eunis.sources.list_parquet_files(api, repo_id, revision)` returns
the Parquet paths at one immutable Hugging Face dataset commit. The paths are
sorted by path.

Pass a commit SHA. Then the inventory stays the same if the default branch of
the repository moves later.

## Validate polygon areas

`osm_polygon_eunis.geometry.safe_area(geometry)` returns the positive area of a
valid, non-empty geometry. It returns `0.0` when the value is absent, empty, or
invalid. It also returns `0.0` when the geometry has no positive area. The
units are the units of the CRS of the geometry.

## Enrich a local shard

`osm_polygon_eunis.transform.enrich_parquet_shard(source, destination, *,
reference, batch_size, geometry_column="geometry")` streams a Parquet shard in
bounded batches. It keeps the source rows and fields. It adds the EUNIS result
columns. It returns the number of rows that it wrote.

`batch_size` must be positive. `geometry_column` must exist. The function writes
`destination`. If you need an atomic replacement, give a staging path.
