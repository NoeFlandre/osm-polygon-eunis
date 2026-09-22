# Operations

The current production scope is the `description` source only. The heavy
release must run on Grid'5000; the Mac is used only for tests, command
construction, source synchronization, submission, and monitoring.

Grid'5000 is reserved for approved research or education. The controller runs
`usagepolicycheck -t` on the frontend before and after every submission. It
uses one CPU host in Lille's `chuc` cluster: 16 cores, 16 bounded source
workers, the `default` queue (`-q default`), `night` scheduling (`-t night`),
and the `chuc` property (`-p chuc`) with a 12-hour walltime. No GPU is
requested. Hardware pages describe inventory, not live availability;
OAR decides whether the reservation can be admitted.

## Storage and credentials

Create a project directory on remote persistent storage, for example
`/home/$USER/osm-polygon-eunis`. Grid'5000 storage is not backed up, so copy
the final receipt, logs, and any released metadata outside Grid'5000 after the
run. The controller refuses Mac paths, relative paths, `/tmp`, and other
ephemeral roots.

Keep `HF_TOKEN` only in the reserved-node environment. It is never copied by
rsync, placed in a command argument, written to the local job state, or put in
the receipt. The worker keeps large source shards, EEA rasters, and the uv
cache on node-local scratch. Compact checkpoints remain in persistent storage:
`EUNIS_SIDECAR_DIR` and the persistent description work directory.

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

Commit the source tree before submission. A clean source revision is recorded
in the local job state and the remote receipt. First build the commands without
contacting Grid'5000:

```bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis grid5000 submit \
  --frontend flille \
  --persistent-root /home/$USER/osm-polygon-eunis \
  --source-root . \
  --dry-run
```

Submit one description worker and keep the returned job ID:

```bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv \
uv run osm-polygon-eunis grid5000 submit \
  --frontend flille \
  --persistent-root /home/$USER/osm-polygon-eunis \
  --source-root . \
  --state-file .grid5000-description-job.json
```

The Mac-side command only checks policy, creates the remote source directory,
syncs code/configuration with secrets and run data excluded, submits
`host=1/core=16` to `chuc`, and checks policy again. It does not download
Parquet or rasters and does not invoke the release runner locally.

Monitor or cancel only the exact numeric job ID:

```bash
uv run osm-polygon-eunis grid5000 status --frontend flille --job-id JOB_ID
uv run osm-polygon-eunis grid5000 cancel --frontend flille --job-id JOB_ID
```

The controller rejects a second submission while the saved job is still
visible to OAR. After a terminal job, keep the persistent sidecars and use a
new state-file name for a deliberate retry. The worker's description command
is resumable: it reuses valid checkpoints keyed by source checksum and kernel
version, and a completed shard is not downloaded again.

## Reserved-node worker

OAR runs `scripts/grid5000/description-release.sh` from the synchronized
source tree. It requires `OAR_JOB_ID`, `HF_TOKEN`, and a persistent root under
`/home`, `/groups`, or `/srv`. It sets `EUNIS_SOURCE_DIR` and
`EUNIS_REFERENCE_DIR` under node-local scratch, `EUNIS_SIDECAR_DIR` under the
persistent root, and `UV_CACHE_DIR` under scratch. It runs only:

```text
uv run --frozen --no-dev osm-polygon-eunis release \
  --dataset description --execution grid5000
```

The runner writes a token-free JSON receipt with the source revision, target
repository and revision, reference identity, no-op value, and OAR job ID. A
failure trap leaves a small failure receipt and the persistent log in place.
Copy these artifacts outside Grid'5000 before cleaning any remote storage.

## Verification gates

Before submission, run the deterministic local gates in this order:

```bash
uv run ruff check src tests scripts
uv run ty check src tests scripts
uv run pytest --cov --cov-report=json --cov-report=term-missing
uv run python scripts/check_architecture.py
uv run python scripts/check_crap.py
uv run python scripts/smoke.py
uv run mutmut run
```

The remote completion check is separate: inspect the worker log and receipt,
verify the description target's rows, schemas, manifest, card, map, and remote
tree, then submit the same release again only after the first OAR job is
terminal. The second run must be a verified no-op. Website and Wikidata are
outside this production run and remain untouched.
