# Bounded WorldCover and LUCAS pool controller

`osm_polygon_eunis.pool_adapters.controller` executes one pinned source shard at
a time. It stores source rows and geometry once, then runs each selected map
against that immutable checkpoint. A shard checkpoint is not a complete pool
manifest. It cannot start evaluation or sampling.

The controller accepts only the three raw text sources and the WorldCover and
LUCAS maps. EUNIS source semantics do not enter its selectors or preflight.
Adapters retain the inclusive 80% full-polygon-area rule and their map-specific
support-union policy through their immutable `MapPin`.

## Inputs

`--inventory-manifest` must be a JSON `raw-inventory-keyset-v1` census whose
bytes are pinned by `--inventory-sha256`. It must enumerate every file in the
checked-in raw source tree and reconcile its shard count, byte sizes, row count,
unique-key count, and full keyset digest. Each shard entry contains its exact
repository path, Git blob ID or LFS object ID, row count, and ordered native-key
digest. The controller verifies the selected source shard bytes while staging
them and independently checks its ordered keys while building the raw
checkpoint.

The census is not the source tree metadata file. It must come from a completed
and reviewed raw-key audit. The checked-in source-tree pins alone intentionally
do not provide raw row or key counts.

The built-in `osm_polygon_eunis.pool_adapters.workers:build_workers` factory
receives `(args, census, source_pin)` and returns
`(map_workers, map_pins, raw_row_factory)`. It reads only native keys and
geometry from the staged shard in bounded Parquet batches. Each raw row keeps
its canonical typed key, derived inventory ID, source row locator, and bounded
JSON-ready geometry payload. Each selected map emits one attempt for every raw
row in the same order. Sidecars are written before their attempt row is
checkpointed.

`--worker-config` and `--worker-config-sha256` pin the local reference paths
and WorldCover catalog. The configuration is at most 1 MiB and has this
versioned shape. Include only the map sections selected for an invocation.

```json
{
  "format_version": "bounded-pool-worker-config-v1",
  "worldcover": {
    "tile_catalog_path": "config/worldcover-afghanistan-first-shard-tile-catalog.json",
    "tile_catalog_sha256": "eb66d2ad10897ba869e0bfc48831151f592c36dfee7d84ce399c6dd866e22b88",
    "tile_root": "/data/worldcover-afghanistan-v200"
  },
  "lucas": {
    "gpkg_path": "/data/l2022_survey_cop_radpoly_attr.gpkg",
    "taxonomy_path": "/data/lucas-2022-codebook.pdf"
  }
}
```

The WorldCover tile catalog is pinned to Website shard
`polygons/afghanistan-latest.parquet` at source commit
`4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd`. The shard content SHA-256 is
`796965a9013b2ed590d1797a77a1dfa86dcda60e75fceb8d05d460d37e70c1b2`. It
records 42 usable geometries, the exact tile-request histogram, six published
tile hashes, and a 600,000,000-byte aggregate cap. Before classification, the
worker re-reads the raw checkpoint, rechecks the full request histogram, and
verifies each requested tile's bytes and raster profile. It copies the verified
bytes into a content-scoped workspace snapshot under the shared byte cap before
the producer opens them. The reference tile directory is caller supplied and
remains unchanged.

LUCAS opens the caller-supplied 2022 GeoPackage read-only after checking its
1,121,144,832-byte size and pinned SHA-256, then verifies the codebook pin.
Temporary attempted/contributing-ID sidecars reserve capacity against the
shared workspace cap before writing.

For WorldCover, the worker must use producer revision
`18c75517c74be22ab2f90edce2b65d3600461048`, the canonical v4 area method, a
source tile catalog, and verified tile bytes. The controller rejects another
WorldCover producer revision or area-method identity. For LUCAS, the worker must use the
verified 2022 GeoPackage and C3 taxonomy inputs. The controller signature binds
these identities through each map's `MapPin`.

## Plan and launch

Use one or more `--map` options and exactly one `--shard` for each
`run-shard` invocation. The plan command reads and validates the complete census
but does not inspect polygon geometry or run an adapter:

```bash
uv run python -m osm_polygon_eunis.pool_adapters.controller plan \
  --source website \
  --shard polygons/afghanistan-latest.parquet \
  --map worldcover --map lucas \
  --input-root /data/pinned-source \
  --inventory-manifest /data/audits/website-keyset.json \
  --inventory-sha256 CENSUS_SHA256 \
  --workspace /data/pool-work \
  --max-workspace-bytes 100000000000
```

After reviewing the plan and verifying all reference paths, hashes, and the
complete census, run the same bounded unit with the built-in factory and
hash-pinned worker configuration:

```bash
uv run python -m osm_polygon_eunis.pool_adapters.controller run-shard \
  --source website \
  --shard polygons/afghanistan-latest.parquet \
  --map worldcover --map lucas \
  --input-root /data/pinned-source \
  --inventory-manifest /data/audits/website-keyset.json \
  --inventory-sha256 CENSUS_SHA256 \
  --workspace /data/pool-work \
  --max-workspace-bytes 100000000000 \
  --max-shards-per-invocation 1 \
  --worker-factory osm_polygon_eunis.pool_adapters.workers:build_workers \
  --worker-config /data/pool-worker-config.json \
  --worker-config-sha256 WORKER_CONFIG_SHA256
```

The run command is enabled only after a reviewed complete source census is
available and the input root, local map paths, config bytes, and digests are
verified. The catalog and tile hashes above are ready for this first shard, but
the complete `raw-inventory-keyset-v1` census must still be supplied. Do not
replace it with a one-shard census or start evaluation from a partial source
inventory. Repeating an exact invocation reuses hash-verified checkpoints; a
changed source, census, adapter, reference, taxonomy, area method, or resource
limit gets a separate checkpoint signature. The workspace cap covers staged
input, raw rows, attempts, receipts, sink artifacts, verified WorldCover tile
snapshots, and LUCAS temporary sidecars. The catalog also bounds the WorldCover
reference entry count and aggregate source-tile bytes.

## Output status

Each selected shard produces a shared `raw/` checkpoint and one map-specific
`attempts.jsonl` plus receipt. Receipts explicitly state
`pool_complete: false` and `evaluation_started: false`. A full-pool exporter
must still reconcile the complete source keyset and all required shards before
exporting a pool or starting downstream evaluation.
