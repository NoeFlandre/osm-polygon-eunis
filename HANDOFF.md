# HANDOFF: issue #108, osm-polygon-eunis (assessment only)

**Status: ASSESSMENT ONLY. Issue #108 stays open.** No production code and no gate configuration was changed. No test was written. The mutation config was tried, then reverted, and is NOT committed.

## 1. Branches

| Branch | Commit | What it is |
|---|---|---|
| `claude/atomic-write-mutation-coverage-go783s` | `35ce838a444e787bc91432586916b9be6a205ef9` (unchanged) | Base. No commits added. Left as it was. |
| `handoff/second-master/issue-108-notes-35ce838` | Orphan branch; this file is at its root. Full SHA: see the commit link returned with this handoff. | Notes only. No code. Trial configuration, baseline, results, survivor classification, first-batch proposal. |

Nothing is pushed to `main`. No PR is open.

## 2. Restore from GitHub (no attachments needed)

```
git clone --branch handoff/second-master/issue-108-notes-35ce838 https://github.com/NoeFlandre/osm-polygon-eunis notes
git clone --branch claude/atomic-write-mutation-coverage-go783s https://github.com/NoeFlandre/osm-polygon-eunis code
git -C code rev-parse HEAD          # must print 35ce838a444e787bc91432586916b9be6a205ef9
git -C code apply --check ../notes/trial/trial-config.patch   # confirms the trial patch applies to the base
```

The trial patch is NOT applied anywhere. Apply it only in a scratch clone.

## 3. What was tried, exactly

File: `trial/trial-config.patch`. Full explanation: `trial/also-copy-17-files.md`.

- `source_paths`: add `manifest_state.py` and `publish.py`.
- `pytest_add_cli_args_test_selection`: add `tests/unit/test_manifest_state.py` and `tests/unit/test_publish.py`.
- `also_copy`: add 17 files: `card_publishing.py`, `cards.py`, `eea.py`, `geometry_checkpoints.py`, `geometry_chunks.py`, `geometry_jobs.py`, `geometry_workers.py`, `geopackage_reference.py`, `geopackage_sql.py`, `geopackage_tiles.py`, `options.py`, `raster_geometry.py`, `raster_reference.py`, `reference_cache.py`, `reference_staging.py`, `release_orchestration.py`, `shard_processing.py`.

Result: parts 1 and 2 alone FAILED. Stats collection raised `ModuleNotFoundError: osm_polygon_eunis.eea` (`trial/attempt1-failed-collection.log`). Part 3 is required.

## 4. Baseline (before any change)

- `tests/unit/test_manifest_state.py` + `tests/unit/test_publish.py`: 54 passed. Re-run in the handoff step: `54 passed in 5.97s`. The summary line shows no skips.
- Same, with `HF_HUB_OFFLINE=1`: 54 passed, 0.83 s (trial report).
- `tests/unit/test_check_mutation.py`: 14 passed (trial report; not re-run in the handoff step).
- External calls: none found. Local fake API classes and monkeypatched `download_to_temp` were used. Covers `huggingface_hub` only, not raw `httpx`.

## 5. Trial results

Full table: `trial/results-summary.md`. Per-mutant status: `trial/results-manifest_state.txt` (443 lines) and `trial/results-publish.txt` (501 lines).

| Module | Mutants | Killed | Survived | No tests | Kill rate | Wall |
|---|---|---|---|---|---|---|
| `manifest_state.py` | 443 | 284 | 129 | 30 | 64.1% | 55 s |
| `publish.py` | 501 | 300 | 201 | 0 | 59.9% | 48 s |

- No timeouts. The 900 s cap was not reached.
- Full configuration estimate (2176 mutants, all modules): about 3-4 min at the same rate. The five existing modules were not timed.
- Gate on this export: exit 1, expected. It fails on unapproved survivors, no-tests, missing minimum scores, and an inventory mismatch (944 vs 2176). Output in `trial/gate-output.txt` and `trial/gate-errors.txt`.
- No `ALLOWED_MUTANTS` entry and no `MODULE_MINIMUM_SCORES` entry was added. Neither should be, from this trial.

## 6. Survivors, by cause

Full table and samples: `analysis/classification.md`.

| Module | A: behaviour gap | B: equivalent-looking | C: fake ignores args | D: selection only | Total |
|---|---|---|---|---|---|
| `manifest_state.py` | 23 | 23 | 54 | 29 | 129 |
| `publish.py` | 107 | 7 | 87 | 0 | 201 |

- A: a real gap. A selected test runs the line but does not check it, or no test runs the branch.
- B: equivalent-looking. Needs written evidence. Not an exception.
- C: the callee is faked by a test and the fake ignores its arguments.
- D: only an unselected test file runs the line.

## 7. The 30 no-tests

Analysis: `analysis/no-tests.md`. Cause: **incomplete selection** (not missing tests, not a collection failure). `test_release_orchestration.py` reaches both `_shared_blobs` and `_verify_final_dataset` for real, and it is not selected. Selecting it alone will not kill the 22 `_verify_final_dataset` mutants, because `verify_dataset` is faked there.

## 8. Proposed first batch (NOT written)

Analysis: `analysis/first-batch.md`.

- 6 test functions (about 8 cases), about 80 lines, in `tests/unit/test_manifest_state.py`.
- Targets 23 group-A mutants in `manifest_state.py`. Predicted, not run.
- Runtime: about 10 s mutmut for touched functions; about 1 min for the module; under 0.1 s extra pytest.
- Needs owner approval before writing.

## 9. Corrections to the trial report

The analysis checked the trial report against the logs. These were wrong or incomplete:
- "Function-level imports" in `also_copy`: none exist. All 17 are module-level transitive imports.
- The no-tests cause also includes `_verify_final_dataset`, not only `_shared_blobs` and `_verify_no_op_dataset`.
- "Survived" in mutmut 3 includes lines that no test runs, because mutmut maps tests to functions.
- `test_publish.py:62-79` compares a value with itself, so it cannot catch non-determinism.

## 10. Not run, and not reviewed

- Full gate with all modules (`just` / CI recipe). Not run.
- No mutation for the new survivors was re-run after any test change. No test change exists.
- The survivor classification comes from one analyst (Haiku). It was spot-checked against the code at the cited lines, but it has NOT had an independent review. Treat the counts as provisional.
- The model is self-reported as `claude-haiku-5-5`. Not verified from outside.

## 11. What is not in this handoff, on purpose

Excluded: `mutants/` (6.2 MB; mutmut's generated copies), `.venv/`, caches, mutmut's cache file, and raw progress logs (about 70-80 KB each, mostly progress lines). The per-mutant results and the survivor lists are kept, and they hold the verdicts. No secrets, credentials, private paths, emails or session IDs are included. A leftover scan found none.

## 12. Files on this branch

- `ledger.md`: working ledger (a snapshot from before the final steps; covers both issues).
- `trial/`: trial patch, 17-file explanation, baseline, results summary, per-mutant results, survivor lists and diffs, no-tests list, gate output, failed first attempt, dry-count and closure scripts.
- `analysis/`: classification, no-tests analysis, first-batch proposal, analysis scripts.

## 13. Open decisions for the owner

1. Approve writing the first batch (section 8).
2. Whether the group-C fakes (args not asserted) should be changed in the same batch or a later one.
3. Whether to apply the trial config on a real branch, and when. It needs tests first (see section 8 and `trial/also-copy-17-files.md`).
