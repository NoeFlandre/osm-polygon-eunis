# OSM Polygon EUNIS

## Purpose and release status

This project adds EUNIS habitat labels to three public Hugging Face datasets. It
selects the label with the largest actual spatial intersection between each OSM
polygon and the official EEA EUNIS habitat probability-map reference.

The three output repositories are the intended release targets. Treat a dataset
as published only after you verify these items independently:

- the current `eunis/manifest.json`
- the Hub revision
- the schema
- the row counts
- the artifact hashes

Inputs:

- `NoeFlandre/osm-polygon-website-tag`
- `NoeFlandre/osm-polygon-wikidata-and-wikipedia`
- `NoeFlandre/osm-polygon-description-tag`

Outputs:

- `NoeFlandre/osm-polygon-website-tag-eunis`
- `NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis`
- `NoeFlandre/osm-polygon-description-tag-eunis`

The tables that contain polygons receive four nullable columns:

- `eunis_code`
- `eunis_name`
- `eunis_overlap_percentage`
- `eunis_source_version`

The percentage is `100 * area(polygon ∩ EUNIS reference geometry) /
area(polygon)`. The pipeline first transforms both geometries to EPSG:3035.
Bounding boxes can remove candidates. They never decide a label. Empty, invalid,
or out-of-reference polygons receive null EUNIS fields. For equal-area ties, the
pipeline uses the ascending EUNIS code.

The pipeline reads bounded Parquet micro-batches. It uses bounded Arrow batches.
It streams temporary files. It deletes the source shards after their final
reference pass. It does not mirror a complete input dataset or output dataset on
the local disk.

The EEA resolver pins these items in the `eunis/manifest.json` of each target:

- the catalog records
- the public asset metadata
- the official 2021 classification workbook
- the SHA-256 checksums of the downloads

The pipeline reads the EEA GeoPackages as tiled rasters of the highest
resolution. It decodes only the tiles that intersect the polygon bbox. The final
label still uses the actual intersection area of the cell and the polygon.

Each output dataset card also contains two items:

- a deterministic static world map at `eunis/world-map.svg`
- a compact percentage table for every EUNIS label, with the polygons that
  received no label

The pipeline builds the map while the final Parquet shards stream through it. It
uses bounded 2-degree bins. It does not keep the source geometries.

## Install and run

Install the command from GitHub with [uv](https://docs.astral.sh/uv/):

```bash
uv tool install --from git+https://github.com/NoeFlandre/osm-polygon-eunis osm-polygon-eunis
```

Plan and preview. These commands do not write to Hugging Face:

```bash
osm-polygon-eunis plan
osm-polygon-eunis release --dry-run --workdir /path/on/persistent-storage/eunis-run
```

To publish, you need `HF_TOKEN` with write access to all three target
repositories. Keep the token in the environment. Do not put it in command
arguments or files:

```bash
export HF_TOKEN=your_write_token
osm-polygon-eunis release --batch-size 256 --workers 8 \
  --workdir /path/on/persistent-storage/eunis-run
```

To verify an existing release without writes, do this:

```bash
osm-polygon-eunis verify --workdir /path/on/persistent-storage/eunis-verify
```

Run `osm-polygon-eunis --help` or `osm-polygon-eunis <command> --help` to see
all options.

The installed command includes the default EEA reference config. In a source
checkout, it comes from
[`config/eea-2021-reference.json`](config/eea-2021-reference.json). Set
`OSM_EUNIS_WORKDIR` to choose the default staging directory for `release` and
`verify`. `EUNIS_SIDECAR_DIR`, `EUNIS_SOURCE_DIR` and `EUNIS_REFERENCE_DIR`
override the sidecar, source and reference-cache directories; the
[operations guide](docs/operations.md) describes them.

The [operations guide](docs/operations.md) describes Grid'5000 execution,
receipts, resuming checkpoints, and the final no-op check.

## Production execution

The current production release is Grid'5000-only. It processes all three
sources in one resumable job: `website`, `wikidata`, and `description`.

The Mac runs tests and the submission and monitoring commands. It does not
compute the Parquet or raster enrichment.

The controller accepts any Grid'5000 site, frontend, or cluster if you give it
explicitly. It requests one CPU host. It never submits duplicate jobs across
sites.

The worker keeps the validated EEA references and sidecars on persistent Grid
storage. It uses node-local scratch for the transient source files and build
files.

Follow [the operations runbook](docs/operations.md) for the policy check, the
submission, the receipt, the verification, and the no-op rerun.

After the all-site duplicate check and the policy check pass, submit the
resumable job from the controller:

```bash
osm-polygon-eunis grid5000 submit \
  --site SITE \
  --frontend FRONTEND \
  --cluster CLUSTER \
  --persistent-root /home/USER/osm-polygon-eunis \
  --state /path/on/external-HDD/eunis-grid5000-state.json
```

## Local development

```bash
uv sync --locked --group dev
uv run osm-polygon-eunis --help
make quality
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the development setup, the task
targets, and the change expectations. `make quality` is the same deterministic
gate that CI runs.

The mutation baselines for five core modules run in their own workflow. The workflow has
a path filter and runs every week. See the
[mutation testing guide](docs/mutation-testing.md).

`docs/operations.md` has the complete release procedure. The
[glossary](docs/glossary.md) defines the project terms.

## Data sources and terms

The source code in this repository has the Apache-2.0 license. See
[`LICENSE`](LICENSE). This code license does not change the terms for the source
datasets or the derived datasets.

OpenStreetMap data is available under the
[Open Database License](https://www.openstreetmap.org/copyright). This license
has attribution requirements and share-alike requirements.

EEA datasets can have reuse terms for each item. Check the notice of the exact
reference asset and the
[EEA legal notice](https://www.eea.europa.eu/en/legal-notice).

The generated manifest records the source revisions and the EEA asset metadata
that a run used. It also records the package version and the source commit that
produced the labels. The Grid'5000 runner supplies its submitted commit SHA to
this field. For releases from an installed artifact, set `EUNIS_SOURCE_COMMIT`
to its full Git SHA.

## Citation

Use [`CITATION.cff`](CITATION.cff) to cite this software. After the live
publication and the membership are verified, this file will record the links to
the hosted documentation and the shared Hugging Face collection.

## Docker

Build and run the CLI as a non-root user:

```bash
docker build -t osm-polygon-eunis .
docker run --rm osm-polygon-eunis --help
docker run --rm osm-polygon-eunis plan --help
```

For a release, mount persistent storage at `/work`. Pass
`--workdir /work/eunis-run`. Pass `HF_TOKEN` with `--env-file` or `-e`. The image
does not contain credentials. The image includes the default EEA config. To
replace it, use `--reference-config`.
