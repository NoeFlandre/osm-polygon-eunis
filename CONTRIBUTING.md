# Contributing

## Set up the development environment

Use Python 3.12 or 3.13 and install the locked development dependencies:

```bash
uv sync --locked --group dev
uvx pre-commit install
```

The pre-commit hooks check YAML, final newlines, large files, Ruff linting and
formatting, and `ty`. To run them over the checkout:

```bash
uvx pre-commit run --all-files
```

The full quality target also runs tests with the CI Hypothesis profile,
branch-aware CRAP checks, architecture checks, Vulture, the release smoke test,
and strict documentation validation:

```bash
make quality
```

For a single gate, use `make lint`, `make test`, `make architecture`, or another
target listed by `make help`. To keep generated coverage data on a persistent
volume, set `COVERAGE_FILE` and `COVERAGE_JSON` before running `make quality`.

## Change expectations

- Add a failing behavior test before fixing a defect, then keep the regression
  test in the suite.
- Keep large source data, downloaded references, and release outputs outside
  the checkout. The Grid'5000 runbook is in `docs/operations.md`.
- Record source revisions and checksums in release receipts. A successful upload
  is not verified until the remote manifest, files, schemas, row counts, and
  hashes are checked.
- Do not close an issue until its acceptance evidence is available.
- Keep code under Apache-2.0 and follow the separate source-data terms in the
  README.

## Performance changes

Run the synthetic performance suite described in `docs/operations.md` on the
designated Grid'5000 benchmark node. Include the benchmark JSON and the source
revision in the change evidence. Mutation tests run separately through the
workflow described in `docs/mutation-testing.md`.
