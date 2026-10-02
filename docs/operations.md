# Operations

## Grid'5000 execution

Run the production release on one Grid'5000 site. Select the site explicitly.

Before each submission, the controller does these steps in order:

1. It runs `usagepolicycheck -t`.
2. It queries the site inventory through the Grid'5000 API.
3. It queries the OAR job list of each site over SSH. It uses the selected
   frontend and the authenticated account.

The controller fails closed in these cases:

- A check is incomplete.
- Any EUNIS job is active.
- It cannot query a site.

If the exclusion of a site is approved, pass `--exclude-site SITE`. The
controller then checks every site in the current Grid'5000 API inventory except
the explicit exclusions. It refuses an exclusion that is not in the API
inventory. It also refuses an exclusion that is the compute site. The local job
state and the run receipt record the policy scope and the exclusions.

The controller syncs only the source code. It submits all three sources
together: `website`, `wikidata`, and `description`. You must give the site, the
frontend, and the cluster explicitly.

The request is for one host. It uses the `default` queue, 16 CPU cores, and a
walltime of one hour. When a short job ends, rerun from the checkpoints that have
a checked signature. Do not extend a reservation beyond the policy.

### Walltime and stop margin

The controller and the worker enforce a walltime that is above the default stop
margin of five minutes. The walltime must not be longer than one hour.

The worker counts the setup time against this limit. By default, it stops the
release work five minutes before the hard deadline of OAR.

The worker forwards INT and TERM to the active deadline helper. The helper then
stops its child process group and writes the incomplete receipt quickly.

The stop margin must leave these two periods:

- the configured TERM-to-KILL grace
- at least 30 seconds to write a receipt

The defaults are a margin of 300 seconds and a grace of 20 seconds. The helper
sends TERM to the worker process group. It waits for the configured grace. Then
it sends KILL if necessary.

The helper records `incomplete` only when it confirms a deadline exit or a
signal exit. The worker counts ordinary child exit codes, including 124, as
worker failures. You can retry them.

The receipts record these items:

- the exact source commit
- the run config
- the job ID
- the error counts and retry counts
- the stop reason

A later job resumes from the valid checkpoints. The retry delays use the same
deadline guard.

### Storage

Keep the controller state file on the external HDD.

The worker stores its data in these places:

- Resumable run data: `GRID5000_PERSISTENT_ROOT/runs/eunis`
- Sidecars: `GRID5000_PERSISTENT_ROOT/sidecars/eunis`
- Logs: under `logs`
- Receipts: under `receipts`

The source staging and the UV cache use node-local scratch. Nobody backs up this
scratch. A resumed job recreates the source bytes and the UV cache.

The worker caches the validated EEA assets at
`GRID5000_PERSISTENT_ROOT/cache/reference`. Later short jobs can reuse them
after they check the metadata identity, the size, and the recorded SHA-256.

### Submit a job

This example is for use after the all-site checks pass. Replace the
site-specific values. Keep the state file on persistent local storage:

```bash
uv run osm-polygon-eunis grid5000 submit \
  --site SITE \
  --frontend FRONTEND \
  --cluster CLUSTER \
  --persistent-root /home/USER/osm-polygon-eunis \
  --exclude-site bordeaux \
  --exclude-site sophia \
  --state /path/on/external-HDD/eunis-grid5000-state.json
```

The controller runs `usagepolicycheck -t --sites` over the current API site list
minus the explicit exclusions. It runs the check before the sync and again after
the submission. It refuses a job that it finds in the same all-site inventory.

The controller streams only the exact Git commit into a clean remote source
directory. The archive does not contain untracked files, local caches, secrets,
or run data.

Its OAR request is `oarsub -q default -p "cluster='CLUSTER'" -l
host=1/core=16,walltime=1:00:00`. The request does not pass a dataset selector.
Thus the worker runs the default release of all sources.

The controller passes the exact source commit into the worker environment. The
worker writes it into the receipt. The release CLI supports `--receipt PATH`.
The worker adds the job ID, the config hash, the attempt counts and retry
counts, the error count, and the log path. Then it saves the final receipt
under `receipts/` in one atomic operation.

`EUNIS_SOURCE_DIR` and `UV_CACHE_DIR` point to node-local scratch.
`EUNIS_REFERENCE_DIR`, `EUNIS_SIDECAR_DIR`, and the release work directory point
to persistent storage. The worker reuses the staged reference cache only after
it checks the size, the metadata identity, and the recorded SHA-256.

**Warning:** Do not submit again until the prior job is verified as terminal and
no active EUNIS job exists on any site.

### Local storage and QA environment

Use a temporary directory on the HDD. It must have enough room for these items:

- one source shard
- one replacement shard
- the resolved EEA reference assets

Set `UV_CACHE_DIR` outside the dataset root.

The production commands write JSON-line progress records to stderr. Use `-q` to
silence them. Use `-v` to add a start record. The commands print only the final
JSON result to stdout. They verify the row counts and the schemas after each
upload.

Keep the local QA environment and the package cache on the external HDD:

```bash
QA_ROOT=/path/on/external-HDD/osm-polygon-eunis-qa
UV_PROJECT_ENVIRONMENT="$QA_ROOT/venv" \
UV_CACHE_DIR="$QA_ROOT/uv-cache" \
uv run osm-polygon-eunis plan
```

### Release memory bounds

The release command keeps label sidecars of four columns. It stages all EEA
assets once. It processes each source shard through the bounded EEA reference
batches. Then it deletes the shard.

The pipeline caps the raster groups at two for each batch. Adjacent vector
groups share one batch.

Each worker keeps at most 128 source shards at a time. This keeps the HDD usage
bounded. It also prevents a new download of a shard for each reference pass.

Each worker process also reuses its opened reference handles and its decoded-tile
caches across its geometry tasks.

The release command needs a valid `HF_TOKEN` with write access to the target
repositories. Each release uses eight bounded worker processes. They work on
disjoint source shards and shared read-only reference files.

The release of dataset scale runs only through the Grid'5000 controller above.
The local command `release --dry-run` is available for previews.

**Warning:** Do not run the enrichment pipeline on the Mac.

### Hypothesis tests

The Hypothesis tests use the `deterministic` profile by default (100
derandomized examples). CI selects `ci` (300 derandomized examples). For local
property fuzzing, use `HYPOTHESIS_PROFILE=dev` for randomized examples.

By default, the Hypothesis database is off. To keep the examples between runs,
set `HYPOTHESIS_DATABASE_DIR` to a persistent directory outside the checkout.

```bash
HYPOTHESIS_PROFILE=dev \
HYPOTHESIS_DATABASE_DIR=/path/on/persistent-storage/hypothesis \
uv run pytest -m property
```

### Dry run

To preview a release without Hub writes, do this. You do not need a token:

```bash
uv run osm-polygon-eunis release --dry-run --workdir .eunis-run
```

The dry run resolves the plans, the reference config, and the no-op status. Then
it prints the following for each dataset:

- the target repo
- if the command would duplicate it
- the shards that the command would upload

Before any network call, `release` checks these items:

- `HF_TOKEN` is set (except with `--dry-run`).
- `--batch-size` is positive.
- The reference config exists and parses.

If a check fails, the command prints one line and exits with status 2.

### Verified no-op

All three targets can already contain a matching manifest for the pinned source
revisions and EEA asset identities. In this case, the command does a verified
no-op. It checks the remote tree, the shared blob identities, the Parquet rows
and schemas, and the card artifact hashes. It does not upload or rebuild
shards.

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
| `release` | `--max-intersection-errors N` | no limit | fail a dataset before its manifest is published when GEOS intersection errors dropped more than N overlap candidates (`card.intersection_errors`) |
| `release` | `--dataset NAME` | all | re-run or resume only the named datasets; repeatable |
| `release` | `--dry-run` | off | preview without Hub writes |
| `release` | `--endpoint URL` | public Hub | Hub endpoint |
| `verify` | `--dataset NAME` | all | limit verification; repeatable |
| `verify` | `--endpoint URL` | public Hub | Hub endpoint |

The installed wheel includes the default reference configuration. In a source
checkout, it resolves to `config/eea-2021-reference.json`. Set
`OSM_EUNIS_WORKDIR` to choose the default local staging directory for `release`
and `verify`. An explicit `--workdir` has priority.

The Python API accepts one `ReleaseOptions` value that contains `BatchLimits`.
The defaults are:

- 8 geometry workers
- 256 Parquet rows for each batch
- at most 128 retained source shards for each worker
- 2 raster groups for each batch
- 4 geometry tasks for each worker

The CLI shows the limits for workers and Parquet batches as `--workers` and
`--batch-size`. To set the other memory bounds, set them through `BatchLimits`
directly.

`verify` is read-only. It loads the `eunis/manifest.json` of each target. It
checks the remote tree, the shared blob identities, the Parquet rows and
schemas, and the card artifact hashes against the manifest. It pins the check to
the source revision that the manifest records. It exits with a nonzero status if
a target has no manifest or does not match it.

The global flags are `--version`, `-q/--quiet`, `-v/--verbose`, and `--debug`.
You can put them before or after the command. `--debug` shows the full traceback.
Without it, the command shows a one-line `error: ...` message on stderr.

| Exit status | Meaning |
|---|---|
| 0 | success (including a verified no-op) |
| 1 | unexpected error |
| 2 | usage or config error (bad option, missing `HF_TOKEN`, invalid reference config) |
| 3 | Hub/network or authentication error |
| 4 | verification failed (published target does not match its expectation or manifest) |

## Quality gates

Before handoff, run the deterministic gates. Use the same task that
`.github/workflows/qa.yml` uses:

```bash
make quality
```

The `Makefile` lists each individual gate.

The CRAP check uses statement coverage and branch coverage. Its threshold is
`6.0`. This means a fully covered function must have a cyclomatic complexity
below 6. The gate fails if the selected coverage JSON omits a source module.

Mutation tests run separately in the workflow that has a path filter and runs
every week. See `docs/mutation-testing.md`.

CI also runs these items in parallel jobs of the same workflow:

- `uv build` and an isolated smoke install of the wheel
- `uv run mkdocs build --strict`
- `pip-audit --strict` over `uv export --locked --all-groups`

The aggregate `qa-ok` job succeeds only when every other QA job succeeded. Make
it the single required status check in branch protection.

CodeQL (Python) runs in its own workflow, `.github/workflows/codeql.yml`. QA runs
one time for each PR push (`push` is limited to `main`). A commit SHA pins every
third-party action.

The separate synthetic performance workflow reports the raster tile-cache misses
and the peak process RSS. The candidate RSS growth must stay within the
configured raster tile-cache byte budget.

## Release order

1. Capture the current source revisions and the plan inventory.
2. Resolve the official EEA reference assets. Calculate their checksums.
3. Detect a verified no-op. If there is none, duplicate each source dataset
   server-side.
4. Stream bounded source micro-batches through each EUNIS reference batch.
   Delete each source shard after its last pass. For Wikidata, process the
   matching link shard. At the same time, collect the bounded label counts and
   the map bins.
5. Upload each dataset card, static SVG map, Parquet schema, row count,
   manifest, and remote tree. Verify each one independently.
6. Add the verified datasets to the `OSM Polygon EUNIS` collection.

The release intentionally does not change the shared Wikidata/Wikipedia
document, section, sentence, Wikivoyage, and Wikidata-fact tables. Only
`polygons` and `polygon_document_links` receive EUNIS fields.
