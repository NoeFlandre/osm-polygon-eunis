# Trial results (issue #108)

Settings: `mutmut run --max-children 2`. Test selection: the two new test files. Timeout cap: 900 s per module. No cap was reached, and there were no timeouts.

| Module | Mutants | Killed | Survived | No tests | Kill rate | Wall time |
|---|---|---|---|---|---|---|
| `manifest_state.py` | 443 | 284 | 129 | 30 | 64.1% | 55 s (about 15-20 s of it is the stats step) |
| `publish.py` | 501 | 300 | 201 | 0 | 59.9% | 48 s (stats cache reused) |

Rates come from the progress counter, so they are approximate. There are no per-line timestamps in the logs.

Full per-mutant status: `results-manifest_state.txt` (443 lines) and `results-publish.txt` (501 lines). Each line is `<mutant>: <status>`.

## Gate on the trial export (expected to fail)

`scripts/check_mutation.py` was run on the trial export. It exits 1 with 340 errors:
- 330 "unapproved survivor" (the survivors are not in `ALLOWED_MUTANTS`; none was added).
- 30 "no_tests" reported for `manifest_state`.
- "no minimum mutation score" for both new modules (no `MODULE_MINIMUM_SCORES` entry; none was added).
- 5 "score below baseline". The five existing modules show 0%, because their mutants were not checked in this run.
- 1 "unexpected status: not checked".
- 1 inventory mismatch: 944 mutants measured against 2176 in the full inventory.

`gate-output.txt` shows the measured summary: overall 26.84% (584 of 2176), `manifest_state` 64.11%, `publish` 59.88%, others 0% (not run).

The gate reads module names from the mutant keys, so no workflow edit is needed.

## Cost

- Both modules: 944 mutants, about 1.5 min of mutation time and about 20 s of stats, at `max-children 2` on this machine.
- Full configuration (2176 mutants, all modules): estimated 3-4 min at the same rate. The five existing modules were not timed.
- Mutation runtime is not the cost. The cost is test work: 330 survivors and 30 no-tests. See `analysis/`.
