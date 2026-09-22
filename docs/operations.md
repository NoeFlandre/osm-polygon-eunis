# Operations

Production enrichment runs on one reserved Grid'5000 CPU host. The Mac is only
the controller: it runs checks, synchronizes code, submits one job, and
monitors or cancels that job. It does not download or compute Parquet or raster
data.

The standard job processes all three sources:

- `website`
- `wikidata`
- `description`

The controller is site-neutral. Give it one available site, frontend, and
cluster for each run; it does not assume Lille and does not submit duplicate
reservations across sites. OAR decides whether that one request is available.
The generated filter is the documented SQL predicate `cluster='CLUSTER'`.
See the official [usage policy](https://www.grid5000.fr/w/Grid5000:UsagePolicy),
[Getting Started](https://grid5000.fr/w/Getting_Started), and
[OAR syntax](https://grid5000.fr/w/OAR_Syntax_simplification) pages.

## Storage and credentials

Create a project directory on remote persistent storage, for example
`/home/$USER/osm-polygon-eunis`. `/home`, `/groups`, and `/srv` are accepted;
Mac paths, relative paths, and node-local `/tmp` paths are rejected for the
persistent root. Grid'5000 storage is site-local and not backed up, so copy
the final receipt, logs, and released metadata outside Grid'5000 after the run.

Keep `HF_TOKEN` only in the reserved-node environment. It is never copied by
rsync, placed in a command argument, written to local job state, or put in the
receipt. The worker keeps source shards, EEA rasters, the virtual environment,
and the uv cache on node-local scratch. Persistent storage contains only code,
compact sidecars, logs, receipts, and resumability metadata.

## Local checks and submission

Use a task-scoped uv environment for local checks:

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv sync --group dev

UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run pytest
```

Commit the source tree before submission. First build the commands without
contacting Grid'5000. Replace the example values with a site and cluster that
are available to your account:

```bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis grid5000 submit \
  --site SITE \
  --frontend FRONTEND \
  --cluster CLUSTER \
  --persistent-root /home/$USER/osm-polygon-eunis \
  --source-root . \
  --dry-run
```

The dry run prints the policy checks, code-only rsync, and one `oarsub`
request. A normal submission uses the same explicit profile and saves one
numeric job ID:

```bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis grid5000 submit \
  --site SITE \
  --frontend FRONTEND \
  --cluster CLUSTER \
  --persistent-root /home/$USER/osm-polygon-eunis \
  --source-root . \
  --state-file .grid5000-eunis-job.json
```

The controller runs `usagepolicycheck -t` before and after the submission. It
rejects a dirty source tree by default, records the source revision, and
blocks a second submission while the saved OAR job is still visible. It asks
for `host=1/core=16` with 16 bounded workers by default; override resources
only when the selected site requires it. The default OAR queue and job type are
`-q default` and `-t night`.

Monitor or cancel only the exact job ID:

```bash
uv run osm-polygon-eunis grid5000 status --frontend FRONTEND --job-id JOB_ID
uv run osm-polygon-eunis grid5000 cancel --frontend FRONTEND --job-id JOB_ID
```

The controller never carries `HF_TOKEN` in its commands. A new submission
after a terminal job should use the retained sidecars and a deliberate new
state-file name.

## Reserved-node worker

OAR runs `scripts/grid5000/release.sh` from the synchronized source tree. It
requires `OAR_JOB_ID`, `HF_TOKEN`, and a persistent root under `/home`,
`/groups`, or `/srv`. It sets `EUNIS_SOURCE_DIR` and `EUNIS_REFERENCE_DIR`
under node-local scratch, `EUNIS_SIDECAR_DIR` under persistent storage, and
`UV_CACHE_DIR` under scratch. It invokes the release CLI without a dataset
filter, so all three sources are selected:

```text
uv run --frozen --no-dev osm-polygon-eunis release \
  --execution grid5000 --reference-config ... --workdir ... --receipt ...
```

The runner checkpoints labels using source checksums and the overlap-kernel
version. A fully checkpointed shard is not downloaded again. The worker writes
a token-free receipt with every source revision, target revision, changed
shards, no-op values, reference identity, and OAR job ID. A failure trap leaves
a small failure receipt and persistent log. Copy these artifacts outside
Grid'5000 before cleaning remote storage.

## Verification gates

Before submission, run the deterministic local gates:

```bash
uv run ruff check src tests scripts
uv run ty check src tests scripts
uv run pytest --cov --cov-report=json --cov-report=term-missing
uv run python scripts/check_architecture.py
uv run python scripts/check_crap.py
uv run python scripts/smoke.py
uv run mutmut run
```

Remote completion is separate from local QA. Inspect the worker log and
receipt, then verify all three target datasets: rows, schemas, manifests,
cards, maps, and remote trees. Only after the first job is terminal, rerun the
same release and require an exact verified no-op for all three targets. Cancel
any remaining job and copy important artifacts off Grid'5000 before cleanup.
