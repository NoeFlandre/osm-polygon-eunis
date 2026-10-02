# Contributing

## Set up the development environment

Use Python 3.12 or 3.13. Install the locked development dependencies:

```bash
uv sync --locked --group dev
uvx pre-commit install
```

The pre-commit hooks check YAML, final newlines, and large files. They also run
Ruff linting, Ruff formatting, and `ty`. To run the hooks on the checkout:

```bash
uvx pre-commit run --all-files
```

The default `pytest` command skips the computational tests that have the mark
`slow`. This keeps the run time under five seconds.

`make test` and the full quality target run the fast tests and the slow tests.
They use the CI Hypothesis profile. The quality target also runs these checks:

- the branch-aware CRAP checks
- the architecture checks
- Vulture
- the release smoke test
- the strict documentation validation

```bash
make quality
```

To run a single gate, use `make lint`, `make test`, `make architecture`, or
another target. Use `make help` to see all targets. To keep the generated
coverage data on a persistent volume, set `COVERAGE_FILE` and `COVERAGE_JSON`
before you run `make quality`.

## Change expectations

- Before you fix a defect, add a behavior test that fails. Then keep the
  regression test in the suite.
- Keep the large source data, the downloaded references, and the release outputs
  outside the checkout. The Grid'5000 runbook is in `docs/operations.md`.
- Record the source revisions and checksums in the release receipts. An upload
  is not verified until you check the remote manifest, files, schemas, row
  counts, and hashes.
- Do not close an issue until the acceptance evidence is available.
- Keep the code under Apache-2.0. Follow the separate source-data terms in the
  README.
- Write the documentation in ASD-STE100 (Simplified Technical English). Use the
  terms in the [glossary](docs/glossary.md).

## Performance changes

Run the synthetic performance suite on the designated Grid'5000 benchmark node.
`docs/operations.md` describes the suite. Put the benchmark JSON and the source
revision in the change evidence.

Mutation tests run separately in the workflow that `docs/mutation-testing.md`
describes.
