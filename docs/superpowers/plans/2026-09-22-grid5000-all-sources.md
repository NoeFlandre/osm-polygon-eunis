# Grid'5000 All-Source Release Implementation Plan

**Goal:** Make the standard release run all three EUNIS source datasets on one
explicitly selected Grid'5000 site, with no Lille-only defaults and no heavy
local computation.

**Approach:** Keep the runner independent of SSH/OAR. Generalize the existing
policy-safe controller and worker, then update the acceptance contract. Use
small red/green test steps and preserve the existing release/checkpoint logic.

## Task 1: Define the all-source and site-neutral contract

**Files:** `tests/unit/test_grid5000.py`, `tests/unit/test_cli.py`,
`tests/integration/test_grid5000_entrypoint.py`,
`tests/acceptance/test_grid5000_contract.py`

- Replace description-only assertions with all-source assertions.
- Use a non-Lille example site/frontend/cluster in controller tests.
- Require explicit `site` and `cluster` on the submit command/configuration.
- Assert the OAR predicate uses `cluster='...'` and the generated worker path is
  `scripts/grid5000/release.sh`.
- Assert CLI release defaults to `dataset_names=None`, which means all sources,
  while explicit repeated `--dataset` remains available.
- Run the focused tests and record the expected failures before implementation.

## Task 2: Generalize the command/configuration layer

**Files:** `src/osm_polygon_eunis/grid5000.py`, `src/osm_polygon_eunis/cli.py`,
`.gitignore`

- Add an explicit all-source dataset inventory to the Grid'5000 submission
  payload and state file.
- Remove Lille/`chuc` defaults; validate and preserve explicit site metadata.
- Build the documented SQL cluster predicate safely as one argument.
- Use the generic worker path and generic state-file exclusion.
- Make `grid5000 submit` accept required `--site`, `--frontend`,
  `--persistent-root`, and `--cluster` values.
- Keep dry-run, duplicate-job rejection, policy checks, source revision checks,
  and token-free state behavior unchanged.

## Task 3: Generalize the reserved-node worker

**Files:** `scripts/grid5000/release.sh`,
`tests/integration/test_grid5000_entrypoint.py`

- Replace the description-specific script with one generic release entrypoint.
- Use generic persistent run, sidecar, log, and receipt locations.
- Invoke the release CLI without a dataset restriction so all three sources are
  selected deterministically.
- Keep node-local scratch, frozen dependencies, OAR guard, token requirement,
  and failure receipt behavior.

## Task 4: Update operator documentation and current specs

**Files:** `README.md`, `docs/operations.md`, current design/plan docs

- Document one explicit site/frontend/cluster per run and state that the
  controller supports every site without submitting duplicate reservations.
- Document all three datasets, generic paths, storage/credential boundaries,
  monitoring, cancellation, receipt verification, and exact no-op rerun.
- Remove contradictory description-only and Lille-only instructions.
- Keep the official policy and OAR syntax references linked.

## Task 5: Verify and finish cleanly

- Run focused tests after each implementation step.
- Run the complete suite with coverage, Ruff, ty, architecture, smoke, CRAP,
  and mutation checks.
- Run shell syntax validation and a no-contact dry run using a non-Lille
  example profile.
- Review the diff for secrets, hard-coded site assumptions, duplicate-submit
  paths, and accidental local execution.
- Commit the implementation and leave the branch/worktree clean. Do not submit
  a real Grid'5000 job or publish datasets from this change.
