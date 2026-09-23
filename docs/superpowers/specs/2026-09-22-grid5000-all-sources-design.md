# Grid'5000 All-Source Release Design

## Goal

Run the complete EUNIS release for the website, Wikidata, and description
sources on one reserved Grid'5000 CPU host. The Mac may validate code,
synchronize it, submit one job, and monitor it; it must not perform the heavy
enrichment locally.

The same controller must work at any Grid'5000 site. Site, frontend, and
cluster are explicit inputs with no Lille-specific default. One run targets one
chosen site and one reservation; it does not submit duplicate jobs across
sites.

## Scope

The default production worker processes all three source datasets in the
runner's deterministic order. The existing repeatable `--dataset` option still
supports a deliberate subset for development or recovery, but the standard
Grid'5000 worker does not omit a source.

The generic worker owns the complete release lifecycle: source planning,
reference downloads, resumable sidecars, output publication, receipt writing,
and final verification. It uses node-local scratch for large transient files
and keeps compact sidecars, logs, receipts, and source metadata under the
user-selected persistent Grid'5000 root.

## Controller and scheduler

`grid5000 submit` validates the source revision, persistent path, resource
values, and explicit site profile. It runs `usagepolicycheck -t`, creates the
remote source directory, synchronizes code without secrets or local run data,
submits one OAR job, and runs the policy check again. The resource predicate is
the documented SQL form `cluster='CLUSTER'`, so the command is not tied to a
Lille alias.

The saved state records the site, frontend, cluster, all selected datasets,
source revision, and numeric job ID without credentials. An existing visible
job blocks a second submission. Status and cancellation operate only on the
explicit numeric job ID.

## Worker

The entrypoint is `scripts/grid5000/release.sh`. It requires an OAR job ID,
either `HF_TOKEN` or the standard Hugging Face token cache, and a persistent
root under `/home`, `/groups`, or `/srv`. It uses
the job's node-local temporary directory for source shards, reference assets,
the virtual environment, and uv cache. It invokes:

```text
uv run --frozen --no-dev osm-polygon-eunis release \
  --execution grid5000 --reference-config ... --workdir ... --receipt ...
```

With no dataset flags, the CLI selects all three sources. A failure trap leaves
a small token-free receipt and log.

## Testing and acceptance

Tests must cover all-source defaults, subset selection, generic non-Lille site
profiles, safe SQL property construction, state/receipt dataset inventories,
credential exclusion, the worker contract, and the OAR execution guard. The
existing Ruff, ty, architecture, coverage, CRAP, smoke, and mutation gates
remain required.

No real Grid'5000 job or public publication is part of this code change. A
dry-run proves the generated commands without contacting a frontend. A real run
is accepted separately only after the remote receipt, all three target
datasets, and an exact no-op rerun are independently verified.
