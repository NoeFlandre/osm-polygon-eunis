# Grid'5000 Description Release Design

## Goal

Run the EUNIS enrichment for `NoeFlandre/osm-polygon-description-tag` only on a
reserved Grid'5000 compute node. The Mac may submit, monitor, and cancel the
job, but must never perform the heavy geometry, raster, Parquet, or publication
work.

## Scope

This change covers the description dataset and its target
`NoeFlandre/osm-polygon-description-tag-eunis`. Website and Wikidata datasets
are excluded from the Grid'5000 command and from its acceptance tests.

The existing exact-overlap kernel and resumable sidecars remain the source of
truth. A new first-class dataset selector replaces the current one-off
monkeypatch used to run description alone.

## Architecture

### Execution boundary

`osm-polygon-eunis grid5000 submit` is a lightweight Mac-side controller. It:

1. validates the configured Grid'5000 site, cluster, queue, paths, and source
   revision;
2. runs `usagepolicycheck -t` on the selected frontend;
3. synchronizes source code only, excluding credentials, caches, and run data;
4. submits one OAR job for one CPU host and records its immutable job manifest;
5. runs the post-submission policy check and returns the exact job ID.

The reserved-node job runs the production release with `OAR_JOB_ID` present.
The release command refuses production execution without that marker, so a
heavy release cannot accidentally run on the Mac. The job performs all EEA and
Hugging Face downloads, geometry work, uploads, verification, and receipt
creation on Grid'5000.

### Modules and boundaries

- `grid5000.py`: immutable configuration, path validation, policy commands,
  OAR command construction, job manifest serialization, and injectable command
  execution. It does not run geometry or import the runner.
- `runner.py`: accepts an explicit dataset selection and an explicit
  Grid'5000 execution guard; it continues to own checkpoints and publication.
- `cli.py`: exposes `release --dataset description --execution grid5000` for
  the node and `grid5000 submit|status|cancel` for the Mac controller.
- `scripts/grid5000/description-release.sh`: small OAR entrypoint that sets
  scratch/cache paths and invokes the pinned project environment.
- `docs/operations.md`: documents setup, policy checks, storage, submission,
  monitoring, cancellation, resume, and final verification.

## Resources and storage

The default profile targets Lille `chuc`: one host, 16 CPU cores, the default
queue, and night scheduling, with all values configurable. No GPU is requested.
The job uses node-local scratch for EEA rasters, source downloads, and uv
cache. Persistent Grid'5000 storage holds only code, compact sidecars,
manifests, logs, and receipts. The source revision, reference identity,
kernel version, batch size, worker count, OAR job ID, and output revision are
recorded in the receipt.

The persistent path must be explicitly supplied and must not be `/tmp`. The
controller rejects paths containing credentials or a Mac-local path. The job
requires `HF_TOKEN` to already be available on Grid'5000 and never copies or
prints it.

## Policy and resilience

- The controller runs `usagepolicycheck -t` before and after submission.
- The default job type is `night`; the user can select another policy-approved
  type explicitly.
- The job is one bounded reservation, never duplicate-submitted by the same
  run manifest, and has a finite walltime.
- Checkpoints are written atomically and keyed by source/reference checksums,
  threshold, and overlap-kernel version.
- Restarting the same persistent workdir resumes completed batches and does
  not redownload fully checkpointed shards.
- `status` reports OAR state and the last receipt without running computation.
- `cancel` requires an exact job ID and is the only command that requests job
  termination.

## Testing and acceptance

TDD covers configuration validation, command construction, policy invocation,
dataset filtering, the Grid'5000 execution guard, receipt persistence, and the
OAR entrypoint contract. No test contacts OAR or Hugging Face.

Acceptance requires:

1. all existing tests plus the new Grid'5000 tests pass;
2. Ruff, ty, architecture, CRAP, coverage, mutation, and smoke gates pass;
3. a dry-run prints the exact OAR command without submission;
4. a real Grid'5000 job runs only the description source, completes remote
   verification, leaves no active job, and writes a receipt;
5. an exact rerun on Grid'5000 reports `no_op=true` and leaves the target
   revision unchanged.

The real job and remote publication are separate from local code verification
and require the user's Grid'5000 account, site access, persistent storage, and
Hugging Face write token.
