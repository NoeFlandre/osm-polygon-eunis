# OSM Polygon EUNIS

## Purpose and release status

This project enriches three public Hugging Face datasets with EUNIS habitat
labels selected by the largest actual spatial intersection between each OSM
polygon and the official EEA EUNIS habitat probability-map reference.

The three output repositories are the intended release targets. Treat a dataset
as published only after its current `eunis/manifest.json`, Hub revision, schema,
row counts, and artifact hashes have been independently verified.

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

The pipeline reads bounded Parquet micro-batches, uses bounded Arrow batches,
streams temporary files, and deletes source shards after their final reference
pass. It does not mirror a complete input or output dataset on the local disk.

The EEA resolver pins the catalog records, public asset metadata, the official
2021 classification workbook, and downloaded SHA-256 checksums in each target's
`eunis/manifest.json`. EEA GeoPackages are read as highest-resolution tiled
rasters; only tiles intersecting the polygon bbox are decoded, and the final
label still uses actual cell/polygon intersection area.

Each output dataset card also contains a deterministic static world map at
`eunis/world-map.svg` and a compact percentage table for every EUNIS label,
including polygons that received no label. The map is built while the final
Parquet shards stream through the pipeline, using bounded 2-degree bins rather
than retaining source geometries.

## Install and run

Install the command from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --from git+https://github.com/NoeFlandre/osm-polygon-eunis osm-polygon-eunis
```

Plan and preview without writing to Hugging Face:

```bash
osm-polygon-eunis plan
osm-polygon-eunis release --dry-run --workdir /path/on/persistent-storage/eunis-run
```

Publish requires `HF_TOKEN` with write access to all three target repositories.
Keep the token in the environment; do not put it in command arguments or files:

```bash
export HF_TOKEN=your_write_token
osm-polygon-eunis release --batch-size 256 --workers 8 \
  --workdir /path/on/persistent-storage/eunis-run
```

Verify an existing release without writing:

```bash
osm-polygon-eunis verify --workdir /path/on/persistent-storage/eunis-verify
```

Run `osm-polygon-eunis --help` or `osm-polygon-eunis <command> --help` for all
options. The default EEA reference config is bundled with the installed command
and comes from [`config/eea-2021-reference.json`](config/eea-2021-reference.json)
in a source checkout. Set `OSM_EUNIS_WORKDIR` to choose the default staging
directory for `release` and `verify`. See the
[operations guide](docs/operations.md) for Grid'5000 execution, receipts,
resuming checkpoints, and the final no-op check.

## Production execution

The current production release is **Grid'5000-only** and processes all three
sources in one resumable job: `website`, `wikidata`, and `description`. The Mac
performs tests and the submission/monitoring commands; it does not compute
Parquet or raster enrichment. The controller accepts any Grid'5000
site/frontend/cluster explicitly, requests one CPU host, and never submits
duplicate jobs across sites. The worker uses persistent sidecars and node-local
scratch for large files. Follow [the operations runbook](docs/operations.md)
for the policy check, submission, receipt, verification, and no-op rerun.

## Local development

```bash
uv sync --locked --group dev
uv run osm-polygon-eunis --help
make quality
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, task targets, and
change expectations. `make quality` is the same deterministic gate CI runs.
The matching kernel mutation baseline runs in its own path-filtered and weekly
workflow; see the [mutation testing guide](docs/mutation-testing.md).

The complete release procedure is documented in `docs/operations.md`.

## Data sources and terms

The source code in this repository is licensed under Apache-2.0; see
[`LICENSE`](LICENSE). That code license does not change the terms for source or
derived datasets. OpenStreetMap data is available under the
[Open Database License](https://www.openstreetmap.org/copyright), with its
attribution and share-alike requirements. EEA datasets can carry item-specific
reuse terms; check the notice attached to the exact reference asset and the
[EEA legal notice](https://www.eea.europa.eu/en/legal-notice). The generated
manifest records the source revisions and EEA asset metadata used for a run.

## Citation

Use [`CITATION.cff`](CITATION.cff) to cite this software. The hosted
documentation and shared Hugging Face collection links will be recorded here
after their live publication and membership are verified.

## Docker

Build and run the CLI as a non-root user:

```bash
docker build -t osm-polygon-eunis .
docker run --rm osm-polygon-eunis --help
docker run --rm osm-polygon-eunis plan --help
```

Mount persistent storage at `/work` and pass `--workdir /work/eunis-run` for a
release. Pass `HF_TOKEN` with `--env-file` or `-e`; the image does not contain
credentials. The default EEA config is included and can be replaced with
`--reference-config`.
