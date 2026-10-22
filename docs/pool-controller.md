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

The `--worker-factory` is an installed `module.path:callable`. The callable
receives `(args, census, source_pin)` and returns
`(map_workers, map_pins, raw_row_factory)`. The raw-row factory reads the staged
shard in bounded batches and yields one row per census key with the canonical
typed `native_key_json`, derived `inventory_row_id`, `source_row_locator`, and
JSON-ready geometry payload. Each map worker calls the corresponding canonical
adapter and yields one complete attempt/evidence record per raw row in the
same order. Measurement sidecars must be persisted before their attempt row is
checkpointed.

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
  --shard polygons/REGION.parquet \
  --map worldcover --map lucas \
  --input-root /data/pinned-source \
  --inventory-manifest /data/audits/website-keyset.json \
  --inventory-sha256 CENSUS_SHA256 \
  --workspace /data/pool-work \
  --max-workspace-bytes 100000000000
```

After reviewing the plan and installing a site-specific factory that supplies
the verified map references, run the same bounded unit with:

```bash
uv run python -m osm_polygon_eunis.pool_adapters.controller run-shard \
  --source website \
  --shard polygons/REGION.parquet \
  --map worldcover --map lucas \
  --input-root /data/pinned-source \
  --inventory-manifest /data/audits/website-keyset.json \
  --inventory-sha256 CENSUS_SHA256 \
  --workspace /data/pool-work \
  --max-workspace-bytes 100000000000 \
  --max-shards-per-invocation 1 \
  --worker-factory site_pool_workers:build_workers
```

Replace the example shard, census digest, data roots, quota, and factory with
verified site inputs. Do not use this example as a production invocation until
the complete census and worker factory are available. Repeating the exact
invocation reuses hash-verified checkpoints; a changed source, census, adapter,
reference, taxonomy, area method, or resource-limit pin gets a separate
checkpoint signature. The workspace cap covers staged input, raw rows,
attempts, receipts, and sidecars supplied through the controller sink.
Worker-managed reference caches must also use a pinned size budget; the
controller does not manage the cache lifecycle of an external worker factory.

## Output status

Each selected shard produces a shared `raw/` checkpoint and one map-specific
`attempts.jsonl` plus receipt. Receipts explicitly state
`pool_complete: false` and `evaluation_started: false`. A full-pool exporter
must still reconcile the complete source keyset and all required shards before
exporting a pool or starting downstream evaluation.
