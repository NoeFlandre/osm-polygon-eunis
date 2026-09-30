# Operations

## Grid'5000 execution

Run the production release on one explicitly selected Grid'5000 site. Before
every submission, the controller runs `usagepolicycheck -t`, then queries the
site inventory via the Grid'5000 API, then queries each site's OAR job list over
SSH from the selected frontend using the authenticated account. It fails closed
if either check is incomplete, if any EUNIS job is active, or if any site cannot
be queried. If a site's exclusion has been approved, pass `--exclude-site SITE`;
the controller dynamically checks every site in the current Grid'5000 API
inventory except those explicit exclusions. It refuses an exclusion that is
absent from the API inventory or is also the compute site. The policy scope and
exclusions are recorded in the local job state and run receipt.

The controller syncs only source code and submits all three sources together:
`website`, `wikidata`, and `description`. The site, frontend, and cluster must
be explicit. The one-host request uses the `default` queue, 16 CPU cores and a
one-hour walltime; after a short job ends, rerun from the signature-checked
checkpoints rather than extending a reservation beyond policy.

Keep the controller state file on the external HDD. The worker stores resumable
run data at `GRID5000_PERSISTENT_ROOT/runs/eunis`, sidecars at
`GRID5000_PERSISTENT_ROOT/sidecars/eunis`, logs under `logs`, and receipts under
`receipts`. Source staging and the UV cache use node-local scratch; that scratch
is not backed up, and source bytes and the UV cache are recreated by a resumed
job. Validated EEA assets are cached at
`GRID5000_PERSISTENT_ROOT/cache/reference`, so later short jobs can reuse them
after checking metadata identity, size and recorded SHA-256.

Example after the all-site checks pass (replace the site-specific values and
keep the state file on persistent local storage):

```bash
uv run osm-polygon-eunis grid5000 submit \
  --site SITE \
  --frontend FRONTEND \
  --cluster CLUSTER \
  --persistent-root /home/USER/osm-polygon-eunis \
  --exclude-site bordeaux \
  --state /path/on/external-HDD/eunis-grid5000-state.json
```

The controller runs `usagepolicycheck -t --sites` over the current API site list
minus explicit exclusions before syncing and again after submission, and refuses
a job found in the same all-site inventory. It streams only
the exact Git commit into a clean remote source directory; untracked files,
local caches, secrets, and run data are not part of that archive. Its OAR
request is `oarsub -q default -p "cluster='CLUSTER'" -l
host=1/core=16,walltime=1:00:00`; it does not pass a dataset selector, so the
worker runs the default all-source release. The exact source commit is passed
into the worker environment and written into the receipt. The release CLI
supports `--receipt PATH`; the worker adds the job ID, config hash, attempt and
retry counts, error count and log path before atomically saving the final
receipt under `receipts/`.
`EUNIS_SOURCE_DIR` and `UV_CACHE_DIR` point to node-local scratch, while
`EUNIS_REFERENCE_DIR`, `EUNIS_SIDECAR_DIR` and the release work directory point
to persistent storage. The staged reference cache is reused only after its
size, metadata identity and recorded SHA-256 are checked.
Do not submit again until the prior job is verified terminal and no active
EUNIS job exists on any site.

Use a temporary directory on the HDD with enough room for one source shard,
one replacement shard, and the resolved EEA reference assets. Set `UV_CACHE_DIR`
outside the dataset root. Production commands emit JSON-line progress records
on stderr (silence them with `-q`, add a start record with `-v`) and print only
the final JSON result on stdout. They verify row counts and schemas after every upload.

Local checks use a task-scoped cache on the temporary volume:

```bash
UV_PROJECT_ENVIRONMENT="$TMPDIR/osm-polygon-eunis-venv" \
UV_CACHE_DIR="$TMPDIR/osm-polygon-eunis-uv" \
uv run osm-polygon-eunis plan
```

The release command keeps four-column label sidecars, stages all EEA assets once,
and processes each source shard through the bounded EEA reference batches before
deleting it. Raster groups are capped at two per batch; adjacent vector groups
share one batch. Each worker retains at most 128 source shards at a time, which
keeps HDD usage bounded without redownloading a shard for each reference pass.
Each worker process also reuses its opened reference handles and decoded-tile
caches across its geometry tasks.
It requires a valid `HF_TOKEN` with write access to the target repositories.
Each release uses eight bounded worker processes over disjoint source shards and
shared read-only reference files:

Dataset-scale release runs only through the Grid'5000 controller above. Local
`release --dry-run` is available for previews; do not run the enrichment pipeline
on the Mac.

Hypothesis tests use the `deterministic` profile by default (100 derandomized
examples). CI selects `ci` (300 derandomized examples); local property fuzzing
can use `HYPOTHESIS_PROFILE=dev` for randomized examples. Set
`HYPOTHESIS_DATABASE_DIR` to a persistent directory outside the checkout when
you want Hypothesis to keep examples between runs; the database is disabled by
default.

```bash
HYPOTHESIS_PROFILE=dev \
HYPOTHESIS_DATABASE_DIR=/path/on/persistent-storage/hypothesis \
uv run pytest -m property
```

Preview a release without any Hub writes (no token required):

```bash
uv run osm-polygon-eunis release --dry-run --workdir .eunis-run
```

The dry run resolves the plans, the reference config and the no-op status, then
prints, per dataset, the target repo, whether it would be duplicated, and the
shards that would be uploaded. Before any network call, `release` checks that
`HF_TOKEN` is set (except with `--dry-run`), that `--batch-size` is positive,
and that the reference config exists and parses; a failure prints one line and
exits with status 2.

If all three targets already contain a matching manifest for the pinned source
revisions and EEA asset identities, the command performs a verified no-op: it
checks the remote tree, shared blob identities, Parquet rows and schemas, and
card artifact hashes without uploading or rebuilding shards.

## Command reference

`osm-polygon-eunis --help` and `osm-polygon-eunis <command> --help` show the same
information with examples.

| Command | Option | Default | Meaning |
|---|---|---|---|
| `plan` | `--dataset NAME` | all | limit to `website`, `wikidata` or `description`; repeatable |
| `plan` | `--endpoint URL` | public Hub | Hub endpoint |
| `release` | `--reference-config PATH` | bundled EEA 2021 config | EEA reference config |
| `release`, `verify` | `--workdir PATH` | `OSM_EUNIS_WORKDIR` or `.eunis-run` | local staging directory |
| `release` | `--batch-size N` | 256 | Parquet rows per streamed batch (must be > 0) |
| `release` | `--workers N` | 8 | geometry worker processes (must be > 0) |
| `release` | `--max-intersection-errors N` | no limit | fail a dataset before its manifest is published when more than N overlap candidates were dropped by GEOS intersection errors (`card.intersection_errors`) |
| `release` | `--dataset NAME` | all | re-run or resume only the named datasets; repeatable |
| `release` | `--dry-run` | off | preview without Hub writes |
| `release` | `--endpoint URL` | public Hub | Hub endpoint |
| `verify` | `--dataset NAME` | all | limit verification; repeatable |
| `verify` | `--endpoint URL` | public Hub | Hub endpoint |

The default reference configuration is bundled with the installed wheel. In a
source checkout it resolves to `config/eea-2021-reference.json`. Set
`OSM_EUNIS_WORKDIR` to choose the default local staging directory for both
`release` and `verify`; an explicit `--workdir` takes precedence.

The Python API accepts one `ReleaseOptions` value containing `BatchLimits`.
Defaults are 8 geometry workers, 256 Parquet rows per batch, at most 128
retained source shards per worker, 2 raster groups per batch, and 4 geometry
tasks per worker. The CLI exposes worker and Parquet batch limits as
`--workers` and `--batch-size`; callers that need the other memory bounds can
set them through `BatchLimits` directly.

`verify` is read-only: it loads each target's `eunis/manifest.json` and checks
the remote tree, shared blob identities, Parquet rows and schemas, and card
artifact hashes against it, pinned to the source revision the manifest records.
It exits nonzero if a target has no manifest or does not match it.

Global flags, accepted before or after the command: `--version`, `-q/--quiet`,
`-v/--verbose` and `--debug` (show the full traceback instead of a one-line
`error: ...` message on stderr).

| Exit status | Meaning |
|---|---|
| 0 | success (including a verified no-op) |
| 1 | unexpected error |
| 2 | usage or config error (bad option, missing `HF_TOKEN`, invalid reference config) |
| 3 | Hub/network or authentication error |
| 4 | verification failed (published target does not match its expectation or manifest) |

Before handoff, run the deterministic gates through the same task used by
`.github/workflows/qa.yml`:

```bash
make quality
```

The `Makefile` lists each individual gate. The CRAP check uses statement and
branch coverage; its threshold of `6.0` means a fully covered function must
have cyclomatic complexity below 6. A source module omitted from the selected
coverage JSON fails the gate. Mutation tests run separately in the
path-filtered and weekly workflow; see `docs/mutation-testing.md`.

CI also runs, in parallel jobs of the same workflow: `uv build` plus an
isolated smoke install of the wheel, `uv run mkdocs build --strict`, and
`pip-audit --strict` over `uv export --locked --all-groups`. The aggregate
`qa-ok` job succeeds only when every other QA job succeeded; make it the single
required status check in branch protection. CodeQL (Python) runs in its own
workflow, `.github/workflows/codeql.yml`. QA runs once per PR push (`push` is
limited to `main`), and every third-party action is pinned by commit SHA.
The separate synthetic performance workflow reports raster tile-cache misses
and peak process RSS; candidate RSS growth must remain within the configured
raster tile-cache byte budget.

The release order is:

1. Capture the current source revisions and plan inventory.
2. Resolve and checksum the official EEA reference assets.
3. Detect a verified no-op; only otherwise duplicate each source dataset
   server-side.
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
