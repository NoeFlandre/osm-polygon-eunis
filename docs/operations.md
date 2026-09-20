# Operations

Use a temporary directory on the HDD with enough room for one source shard,
one replacement shard, and the resolved EEA reference assets. Set `UV_CACHE_DIR`
outside the dataset root. Production commands emit JSON-line progress records
and verify row counts and schemas after every upload.

Local checks use a task-scoped cache on the temporary volume:

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis plan
```

The release command keeps four-column label sidecars, stages all EEA assets once,
and processes each source shard through the bounded EEA reference batches before
deleting it. Raster groups are capped at two per batch; adjacent vector groups
share one batch. Each worker retains at most 128 source shards at a time, which
keeps HDD usage bounded without redownloading a shard for each reference pass.
It requires a valid `HF_TOKEN` with write access to the target repositories.
Each release uses eight bounded worker processes over disjoint source shards and
shared read-only reference files:

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis release --batch-size 256 --workdir .eunis-run
```

If all three targets already contain a matching manifest for the pinned source
revisions and EEA asset identities, the command performs a verified no-op: it
checks the remote tree, shared blob identities, Parquet rows and schemas, and
card artifact hashes without uploading or rebuilding shards.

Before handoff, run the deterministic gates in this order:

```bash
uv run ruff check src tests scripts
uv run ty check src tests scripts
uv run pytest --cov --cov-report=json --cov-report=term-missing
uv run python scripts/check_architecture.py
uv run python scripts/check_crap.py
uv run python scripts/smoke.py
uv run mutmut run
```

The release order is:

1. Capture the current source revisions and plan inventory.
2. Resolve and checksum the official EEA reference assets.
3. Duplicate each source dataset server-side.
4. Stream bounded source micro-batches through each EUNIS reference batch,
   deleting each source shard after its last pass; for Wikidata, process the
   matching link shard and collect bounded label counts and map bins at the same
   time.
5. Upload and independently verify each dataset card, static SVG map, Parquet
   schema, row count, manifest, and remote tree.
6. Add the verified datasets to the `OSM Polygon EUNIS` collection.

The shared Wikidata/Wikipedia document, section, sentence, Wikivoyage, and
Wikidata-fact tables are intentionally unchanged. Only `polygons` and
`polygon_document_links` receive EUNIS fields.
