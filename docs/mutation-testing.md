# Mutation testing

The overlap selection and geometry kernels are mutation-tested in a separate
GitHub Actions workflow. Mutmut targets `matching.py` and `geometry.py`, with
their focused unit suites. The workflow runs on source and test changes, weekly,
and on manual dispatch. Keeping it separate from deterministic QA prevents the
slower mutation run from extending every unrelated Python matrix job.

The first measured baseline was 91.935% (114 of 124 mutants killed). The CI
gate keeps a 91.9% minimum and requires a complete inventory, no mutants without
tests, and no skipped, suspicious, timed out, or interrupted mutants. Each
survivor must have an exact entry and explanation in `scripts/check_mutation.py`.
The current
allowlisted comparison is equivalent because its caller handles unequal
percentages before comparing the values.

The workflow saves the exported statistics and full mutmut result list as a
14-day artifact, including when the gate fails. Seven equivalent survivors are
documented in `scripts/check_mutation.py`: the tie comparison is guarded by the
unequal-area check; valid polygons have positive area; `make_valid` preserves
valid polygons and turns collapsed polygons into non-areal output; and pyproj
resolves lowercase EPSG identifiers to the same CRS. Run the workflow locally
with:

```bash
uv run mutmut run --max-children 2
uv run mutmut export-cicd-stats
uv run mutmut results --all true > mutation-results.txt
uv run python scripts/check_mutation.py
```
