# Proposed first batch (NOT written; needs owner approval)

Scope: at most 6 new test functions (about 8 cases) in one file, `tests/unit/test_manifest_state.py`, about 80 lines. Goal: kill the group-A survivors in `manifest_state.py`. No production code changes. No gate change. No `ALLOWED_MUTANTS` entry.

| Test | What it checks (exact value) | Mutants it targets |
|---|---|---|
| T1 | Empty group, config `{"classification_record": "class"}`, exact dict returned by `_reference_manifest` | `_reference_manifest` #27-37 (11) |
| T2 | Groups from `r1:/z.tif`, `r1:/b.tif`, `r2:/a.tif` (RemoteAsset, `eea.py:74`), exact order of result | `_reference_manifest` #45, #48 (2) |
| T3 | `_mapping_field({"reference": ["x"]}, "reference") == {}` | `_mapping_field` #4 (1) |
| T4 | Parametrized mixed path and digest types; `_manifest_artifacts` is `None` | `_manifest_artifacts` #19 (1) |
| T5 | Extend `test_verify_no_op_dataset_reuses_manifest_expectations` (`:334`): assert `result.plan is plan` and `result.verification is verification` | `_verify_no_op_dataset` #37, #39 (2) |
| T6 | Parametrized exact `str()` of messages from `_required_string_list` (2 inputs) and `_required_no_op_parts({})` | `_required_string_list` #5, #6, #9, #10, #13, #14 (6) |

Expected kills: 23 group-A mutants in `manifest_state.py`. These are predictions. They were not run.

Runtime estimates:
- mutmut re-run for touched functions only: about 10 s. There are 76 mutants in the touched functions.
- mutmut re-run for the whole module: about 1 min (trial: 55 s for 443 mutants at `max-children 2`).
- pytest for the new tests: under 0.1 s extra. Baseline for the two files is 54 tests in about 1-6 s.

Not in this batch:
- The 30 `no-tests` (needs a selection change, and a fake that records arguments).
- Group B (equivalent-looking). Each needs written equivalence evidence. These are not exceptions.
- Group C (54 in manifest_state, 87 in publish). Needs fakes that assert their arguments. This is a test change, not a gate change.
- `publish.py` (107 group-A). Later batches, after this one is reviewed.

Decision needed: approve writing this batch, and the later batches in the order above.
