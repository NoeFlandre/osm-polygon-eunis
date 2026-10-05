# Mutation testing

A separate GitHub Actions workflow runs mutation tests on five modules. Mutmut
targets these modules:

| Module | Why it is mutated |
| --- | --- |
| `matching.py` | The overlap selection kernel. |
| `geometry.py` | The geometry parsing and repair kernel. |
| `grid_overlap.py` | The exact polygon and cell overlap areas. |
| `transform.py` | The batch and label sidecar alignment. |
| `release_plan.py` | The pinned source layout and the sidecar paths. |

Each module uses its focused unit suites. The brute-force grid cross-check in
`tests/unit/test_grid_overlap.py` is marked `slow`, but it runs in under one
second. Mutmut therefore selects the `slow` tests too.

The workflow runs on source changes and test changes. It also runs every week
and on manual dispatch. The workflow is separate from the deterministic QA. This
prevents the slow mutation run from making every unrelated Python matrix job
longer.

The CI gate has these rules:

- Each module has its own minimum score in `MODULE_MINIMUM_SCORES` in
  `scripts/check_mutation.py`. Each module ratchets alone.
- The inventory must be complete.
- No mutant can be without tests.
- No mutant can be skipped, suspicious, timed out, or interrupted.
- Each survivor must have an exact entry and an explanation in
  `scripts/check_mutation.py`.

To add a module, add it to `source_paths` in `pyproject.toml`. Add its tests to
`pytest_add_cli_args_test_selection`. Add its imports to `also_copy`. Kill all
survivors with exact-value tests. Then add the measured score to
`MODULE_MINIMUM_SCORES`.

The workflow saves the exported statistics and the full mutmut result list as an
artifact for 14 days. It saves them also when the gate fails.

`scripts/check_mutation.py` documents the equivalent survivors. An equivalent
mutant changes the code but not the behavior. These are the groups:

- The tie comparison is guarded by the unequal-area check. The caller handles
  unequal percentages before it compares the values.
- Valid polygons have positive area.
- `make_valid` keeps valid polygons the same. It changes collapsed polygons to
  non-areal output.
- pyproj resolves lowercase EPSG identifiers to the same CRS.
- Columns of one Arrow table have the same length, so a strict `zip` never
  fails.
- Parquet compression names are case-insensitive.
- Default NumPy and rasterio arguments equal the explicit arguments. This
  covers `dtype`, `fill`, `copy` and the `STABLE` sort kind.
- Grouping cells by finer keys, preparing a geometry, and clipping a polygon to
  a cell that it does not touch do not change any cell area.

To run the workflow locally, use these commands:

```bash
uv run mutmut run --max-children 2
uv run mutmut export-cicd-stats
uv run mutmut results --all true > mutation-results.txt
uv run python scripts/check_mutation.py
```

Delete the `mutants/` directory after you change the tests. Mutmut can reuse
old coverage data from that directory.
