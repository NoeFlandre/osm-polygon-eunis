# Mutation testing

A separate GitHub Actions workflow runs mutation tests on the overlap selection
kernel and the geometry kernel. Mutmut targets `matching.py` and `geometry.py`.
It uses their focused unit suites.

The workflow runs on source changes and test changes. It also runs every week
and on manual dispatch. The workflow is separate from the deterministic QA. This
prevents the slow mutation run from making every unrelated Python matrix job
longer.

The first measured baseline was 91.935% (114 of 124 mutants killed). The CI gate
has these rules:

- The minimum is 91.9%.
- The inventory must be complete.
- No mutant can be without tests.
- No mutant can be skipped, suspicious, timed out, or interrupted.
- Each survivor must have an exact entry and an explanation in
  `scripts/check_mutation.py`.

The workflow saves the exported statistics and the full mutmut result list as an
artifact for 14 days. It saves them also when the gate fails.

`scripts/check_mutation.py` documents seven equivalent survivors:

- The tie comparison is guarded by the unequal-area check. The caller handles
  unequal percentages before it compares the values.
- Valid polygons have positive area.
- `make_valid` keeps valid polygons the same. It changes collapsed polygons to
  non-areal output.
- pyproj resolves lowercase EPSG identifiers to the same CRS.

To run the workflow locally, use these commands:

```bash
uv run mutmut run --max-children 2
uv run mutmut export-cicd-stats
uv run mutmut results --all true > mutation-results.txt
uv run python scripts/check_mutation.py
```
