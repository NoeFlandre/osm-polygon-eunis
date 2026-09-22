# Grid'5000 Description Release Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the description-only EUNIS release submit and execute on one policy-checked Grid'5000 CPU node, while preventing heavy production work from running on the Mac.

**Architecture:** Keep the exact-overlap runner independent of SSH and OAR. Add a standard-library `grid5000` module for immutable scheduler configuration, safe command construction, and injected command execution. The Mac-side CLI only synchronizes code, checks policy, submits, monitors, or cancels; the OAR worker invokes the existing runner on the reserved node with persistent sidecars and node-local scratch.

**Tech Stack:** Python 3.12, uv, subprocess/SSH/rsync/OAR, JSON receipts, pytest, Ruff, ty, architecture checks, coverage, mutation, and smoke tests.

---

## File map

- Create `src/osm_polygon_eunis/grid5000.py` for pure configuration, validation, OAR/SSH/rsync command builders, job-ID parsing, and injected command execution.
- Modify `src/osm_polygon_eunis/runner.py` to filter plans by explicit dataset names, route the source cache through `EUNIS_SOURCE_DIR`, and require `OAR_JOB_ID` for Grid'5000 execution.
- Modify `src/osm_polygon_eunis/cli.py` to add explicit execution and dataset/receipt options plus `grid5000 submit`, `grid5000 status`, and `grid5000 cancel`.
- Create `scripts/grid5000/description-release.sh` as the reserved-node entrypoint.
- Modify `docs/operations.md` and `README.md` to remove the production-local release path and document the Grid'5000-only workflow.
- Create `tests/unit/test_grid5000.py`, extend `tests/unit/test_runner.py` and `tests/unit/test_cli.py`, and create `tests/integration/test_grid5000_entrypoint.py`.

### Task 1: Add explicit description selection and the production execution guard

**Files:**
- Modify: `src/osm_polygon_eunis/runner.py:1296-1348,1955-1970`
- Modify: `src/osm_polygon_eunis/cli.py:13-93`
- Test: `tests/unit/test_runner.py`
- Test: `tests/unit/test_cli.py`

- [ ] **Step 1: Write the failing runner tests**

Add tests with the existing `DatasetPlan` fixtures. Keep selection pure so the
test does not need to fake Hub downloads or publication:

```python
def test_select_dataset_plans_keeps_requested_order():
    plans = (website_plan, description_plan)
    assert runner._select_dataset_plans(plans, ('description',)) == (description_plan,)


def test_grid5000_execution_requires_oar_job(monkeypatch, tmp_path):
    monkeypatch.delenv('OAR_JOB_ID', raising=False)
    with pytest.raises(RuntimeError, match='OAR_JOB_ID'):
        runner.run_release(
            object(), reference_config=config, workdir=tmp_path,
            batch_size=2, dataset_names=('description',), execution='grid5000',
        )
```

Also test that `EUNIS_SOURCE_DIR` is used for the temporary source cache and that an unknown or empty dataset selection is rejected.

- [ ] **Step 2: Run the focused tests and verify RED**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest tests/unit/test_runner.py tests/unit/test_cli.py -q
```

Expected: FAIL because `run_release` does not accept `dataset_names` or `execution` and the CLI has no new options.

- [ ] **Step 3: Implement the minimal selector and guard**

Add keyword-only parameters `dataset_names: tuple[str, ...] | None = None`, `execution: str = 'local'`, and `receipt_path: Path | None = None` to `run_release`. Filter `plan_datasets(api)` with `dataset_spec(name)` while preserving the declared order. Reject unknown names and an empty selection. For `execution == 'grid5000'`, raise unless `OAR_JOB_ID` is non-empty. Keep explicit `execution == 'local'` for development and synthetic tests.

Change `_source_cache` to use `Path(os.environ['EUNIS_SOURCE_DIR'])` when set, otherwise `workdir / 'source'`. Write a compact receipt atomically when `receipt_path` is supplied. It must contain execution, selected dataset names, source revisions, target repositories, OAR job ID when present, target revisions, no-op values, and reference identity. Never serialize tokens or an environment dump.

Add CLI options `--dataset` (repeatable), `--execution` with default `grid5000`, and `--receipt`. Keep local behavior reachable only with explicit `--execution local`.

- [ ] **Step 4: Run the focused tests and verify GREEN**

Run the same pytest command. Expected: all focused tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_eunis/runner.py src/osm_polygon_eunis/cli.py tests/unit/test_runner.py tests/unit/test_cli.py
git commit -m 'feat: select description dataset for production release'
```

### Task 2: Build the policy-safe Grid'5000 command layer

**Files:**
- Create: `src/osm_polygon_eunis/grid5000.py`
- Test: `tests/unit/test_grid5000.py`

- [ ] **Step 1: Write failing pure command tests**

Define the wished-for immutable API and test it without SSH:

```python
def test_default_profile_requests_one_lille_cpu_host():
    config = Grid5000Config(frontend='flille', persistent_root='/home/u/eunis')
    assert build_oarsub_command(config, '/home/u/eunis/source/scripts/grid5000/description-release.sh') == (
        'oarsub', '-q', 'default', '-t', 'night', '-p', 'chuc',
        '-l', 'host=1/core=16,walltime=12:00:00', '-S',
        '/home/u/eunis/source/scripts/grid5000/description-release.sh',
    )


def test_persistent_root_rejects_ephemeral_or_mac_paths():
    for path in ('/tmp/eunis', '/private/tmp/eunis', '/Volumes/Seagate/eunis'):
        with pytest.raises(ValueError):
            validate_persistent_root(path)


def test_job_id_parser_accepts_oar_output_and_rejects_ambiguous_text():
    assert parse_job_id('[AO] Adding job 123456\n') == '123456'
    with pytest.raises(ValueError):
        parse_job_id('submission failed')
```

Cover policy, status, cancel, SSH, rsync exclusions, shell-quoted worker environment, duplicate job-state detection, and receipt serialization.

- [ ] **Step 2: Run tests and verify RED**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest tests/unit/test_grid5000.py -q
```

Expected: import/function failures because the module does not exist.

- [ ] **Step 3: Implement the minimal standard-library module**

Implement:

```python
@dataclass(frozen=True, slots=True)
class Grid5000Config:
    frontend: str
    persistent_root: str
    site: str = 'lille'
    cluster: str = 'chuc'
    queue: str = 'default'
    job_type: str = 'night'
    cores: int = 16
    workers: int = 16
    walltime: str = '12:00:00'
    batch_size: int = 256

@dataclass(frozen=True, slots=True)
class Grid5000Job:
    job_id: str
    submitted_at: str
    config: Grid5000Config
    source_revision: str

def build_policy_command() -> tuple[str, ...]: ...
def build_oarsub_command(config: Grid5000Config, script: str) -> tuple[str, ...]: ...
def build_status_command(job_id: str) -> tuple[str, ...]: ...
def build_cancel_command(job_id: str) -> tuple[str, ...]: ...
def build_ssh_command(frontend: str, command: tuple[str, ...]) -> tuple[str, ...]: ...
def build_rsync_command(local_root: Path, frontend: str, remote_root: str) -> tuple[str, ...]: ...
def validate_persistent_root(path: str) -> str: ...
def parse_job_id(output: str) -> str: ...
```

Use `subprocess.run` only in `run_command`, with argument arrays and captured text. Never use `shell=True`. Require positive cores/workers, finite walltime, one host, a non-empty frontend, and a remote HF_TOKEN that is not copied or printed. Keep the module free of runner/Hub imports so architecture checks remain clean.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same focused command and then `uv run ruff check src/osm_polygon_eunis/grid5000.py tests/unit/test_grid5000.py`.

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_eunis/grid5000.py tests/unit/test_grid5000.py
git commit -m 'feat: add policy-safe Grid5000 command layer'
```

### Task 3: Add the Mac controller and reserved-node entrypoint

**Files:**
- Modify: `src/osm_polygon_eunis/cli.py`
- Create: `scripts/grid5000/description-release.sh`
- Test: `tests/unit/test_cli.py`
- Test: `tests/integration/test_grid5000_entrypoint.py`

- [ ] **Step 1: Write failing CLI/controller tests**

Use a fake command runner that records argument arrays and returns synthetic `oarsub` output. Assert `grid5000 submit` executes, in order:

1. remote `usagepolicycheck -t`;
2. `rsync` with `.git`, `.venv`, `.eunis-run-final`, `.env`, and cache excludes;
3. remote `oarsub` for one `chuc` host;
4. remote post-submit `usagepolicycheck -t`.

Assert the returned JSON contains exactly one job ID, `dataset='description'`, the source commit, and no token. Assert `status` calls only `oarstat`; assert `cancel` calls only `oardel` for a numeric job ID.

The entrypoint test reads the script and asserts it contains `set -euo pipefail`, checks `OAR_JOB_ID`, sets `EUNIS_SOURCE_DIR` to node-local scratch, sets `EUNIS_REFERENCE_DIR` and `UV_CACHE_DIR` under scratch, sets `EUNIS_SIDECAR_DIR` under the persistent root, and invokes:

```text
uv run --frozen --no-dev osm-polygon-eunis release \
  --dataset description --execution grid5000 --receipt ...
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest tests/unit/test_cli.py tests/integration/test_grid5000_entrypoint.py -q
```

Expected: failures because the subcommands and entrypoint do not exist.

- [ ] **Step 3: Implement the controller and script**

Add a `grid5000` subparser with `submit`, `status`, and `cancel`. Submit requires `--frontend`, `--persistent-root`, and an explicit source revision or a clean current Git HEAD; it refuses a dirty source tree unless `--allow-dirty-source` is passed. It writes a small local job-state file only when requested and keeps all data/checkpoints remote.

The submit flow uses an injected runner in tests and `subprocess.run` in real use. It performs policy preflight, creates the remote source directory, rsyncs code/config only, submits, parses the exact job ID, performs the policy post-check, and prints JSON. Existing state with a non-terminal OAR job causes a duplicate-submission error.

The worker script must fail outside OAR, require `HF_TOKEN`, require an absolute persistent root outside `/tmp`, create a per-dataset persistent workdir, and execute the pinned uv environment. It must write stdout/stderr to the persistent run log and leave the receipt on both success and failure.

- [ ] **Step 4: Run focused tests and verify GREEN**

Run the same pytest command plus `uv run ruff check src tests scripts`.

- [ ] **Step 5: Commit**

```bash
git add src/osm_polygon_eunis/cli.py scripts/grid5000/description-release.sh tests/unit/test_cli.py tests/integration/test_grid5000_entrypoint.py
git commit -m 'feat: submit description release to Grid5000'
```

### Task 4: Document the operational contract and remove unsafe instructions

**Files:**
- Modify: `docs/operations.md`
- Modify: `README.md`
- Test: `tests/acceptance/test_grid5000_contract.py`

- [ ] **Step 1: Write failing documentation/acceptance assertions**

Add assertions that production requires Grid'5000, selects description only, mentions `usagepolicycheck -t`, uses one CPU host, keeps sidecars on persistent storage and large files on node-local scratch, and contains no production-local release command.

- [ ] **Step 2: Run the focused test and verify RED**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest tests/acceptance/test_grid5000_contract.py -q
```

- [ ] **Step 3: Update the docs**

Document Mac setup, SSH/rsync requirements, remote HF_TOKEN handling, the default Lille `chuc` profile, OAR night scheduling, policy checks before/after submission, `oarstat` monitoring, exact cancellation, and resume from persistent sidecars. State that Grid'5000 storage is not backed up and the remote receipt must be copied outside Grid'5000 after completion. Keep local commands limited to unit tests, dry-run construction, and controller actions.

- [ ] **Step 4: Run the acceptance test and commit**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest tests/acceptance/test_grid5000_contract.py -q
git add README.md docs/operations.md tests/acceptance/test_grid5000_contract.py
git commit -m 'docs: make description release Grid5000-only'
```

### Task 5: Full verification and dry-run handoff

- [ ] **Step 1: Run the complete local quality gates**

```bash
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync ruff check src tests scripts
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync ty check src tests scripts
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync pytest --cov --cov-report=term-missing
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync python scripts/check_architecture.py
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync python scripts/check_crap.py
UV_PROJECT_ENVIRONMENT=/private/tmp/osm-polygon-eunis-grid5000-venv \
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-grid5000-cache \
uv run --no-sync python scripts/smoke.py
```

- [ ] **Step 2: Run mutation tests for the new pure command layer**

Run the existing matching mutation suite and ensure the new tests do not reduce the prior clean result. Do not mutate subprocess/network calls.

- [ ] **Step 3: Run the no-submit dry run**

Use a fake runner or `grid5000 submit --dry-run` and verify the printed OAR command requests `host=1/core=16`, `-p chuc`, `-q default`, `-t night`, and no GPU. Verify the Mac performs no Parquet/raster work.

- [ ] **Step 4: Review the branch**

```bash
git status --short --branch
git diff main...HEAD --stat
git log --oneline --decorate -8
```

Report exact verification results and the remaining external prerequisite: Grid'5000 credentials/site storage and a remote job run. Do not claim the remote dataset is complete until the job receipt, remote verification, and no-op rerun are independently observed.
