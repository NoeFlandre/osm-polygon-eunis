# Survivor classification (issue #108 trial, 330 survivors + 30 no-tests)

Each survivor has ONE primary cause:
- **A. Behaviour gap.** The change alters behaviour. A selected test runs the line but does not check it, or no selected test runs the branch.
- **B. Equivalent-looking.** The change cannot alter behaviour in this repo. Reason given per group. This still needs written evidence. It is not an exception.
- **C. Artefact.** The mutant sits in a call whose callee a selected test replaces with a fake that ignores its arguments.
- **D. Selection.** The line is run only by a test file that is not in the selection.

mutmut 3 maps tests to functions, not to lines. So "survived" also includes lines that no test runs. That is why a group can be A without any test checking the value.

| Module | A behaviour gap | B equivalent-looking | C fake ignores args | D selection only | Total |
|---|---|---|---|---|---|
| `manifest_state.py` | 23 | 23 | 54 | 29 | 129 |
| `publish.py` | 107 | 7 | 87 | 0 | 201 |

## manifest_state.py

- **A (23):** `_reference_manifest` 13 (outer keys #27-37; sort key #45, #48); `_verify_no_op_dataset` 2 (#37, #39); `_required_string_list` 4 and `_required_no_op_parts` 2 (messages); `_mapping_field` 1; `_manifest_artifacts` 1.
  Sample: `manifest_state.py:76` the `"source_version"` key is changed; `tests/unit/test_manifest_state.py:45-53` asserts only `assets == []`.
- **B (23):**
  - `_manifest_expectation` 9: TypeErrors are swallowed at `manifest_state.py:172`, so messages are never seen.
  - `_compatible_manifests` 3 and `_no_op_receipts` 4: `zip(strict=...)` at `manifest_state.py:296`; the lengths are always equal, since the existing manifests are built one per plan (`:278`).
  - `_load_existing_manifest` 4: encoding and `missing_ok` variants.
  - `_manifest_expectations` 2: `key=str` on keys that are always str.
  - `cast()` 2: runtime no-op.
- **C (54):** `_verify_no_op_dataset` 23 (fake `_verify_no_op_dataset` takes `*args`); `_load_existing_manifest` 11; `_try_no_op_release` 8; `_load_existing_manifests` 8; `_no_op_receipts` 4.
  Sample: `manifest_state.py:255` calls `_shared_blobs(None, ...)`, which is faked at `tests/unit/test_manifest_state.py:355` with `*args`.
- **D (29):** `_reference_manifest` asset loop, asset dict keys, and the sort lambda (`manifest_state.py:64, 71, 80`). Only `tests/unit/test_release_orchestration.py:516` runs them, and it checks an exact dict at `:555`.

## publish.py

- **A (107):** `upload_replacements` 27 (14 single-file branch never run; 13 kwargs unasserted); `upload_replacement` 18; `upload_manifest` 17; `build_manifest` 10; `verify_dataset` 8; `_git_source_commit` 5 (messages); `_remote_files` 4; `_manifest_bytes` 4; `_environment_source_commit` 4; `_validated_source_commit` 4; `_validate_manifest_paths` 3; `_verify_expectations` 1; `_verify_manifest` 1; `_verify_shared_blobs` 1.
  Sample: `publish.py:281` the regex is lowercase-only; `tests/unit/test_publish.py:106` uses only `"not-a-commit"`.
  Also in A: `sort_keys` removed or changed (manifest bytes); env-var name case; `and` for `or` in `_remote_files` and `_verify_expectations`; uppercase hex dropped from the SHA regex; `is not None` in `_verify_dataset`.
- **B (7):**
  - `verify_dataset` 4: `#23-25` change only the temp-dir prefix name; `#30` is `rows_by_path = None`, which is overwritten at `publish.py:363` before any read. No test can kill it.
  - `_verify_manifest` 2: encoding variants.
  - `_software_provenance` 1: `version("OSM-POLYGON-EUNIS")` (`publish.py:221`). `importlib.metadata` normalises names, and `test_publish.py:82` still passes.
- **C (87):** `_remote_files` 14; `verify_dataset` 15; `_git_source_commit` 13; `duplicate_source` 13; `_git_command_output` 11; `target_exists` 6; `_verify_expectations` 5; `_verify_manifest` 5; `_verify_artifacts` 5.
  Sample: `publish.py:329` `list_repo_tree` args are ignored by `test_publish.py:257` (`*_args`); `publish.py:264` subprocess kwargs are ignored by `run_git` at `test_publish.py:119`.
- **D (0).** Predicted, not run: about 15 A mutants would be killed by `tests/acceptance/test_release_golden.py`, which is not selected (`build_manifest` keys 10, `upload_manifest` repo_id 2, `upload_replacements` repo_id 2, `_manifest_bytes` separators 1).

## Surprising survivors, checked

- `verify_dataset` #26 (`directory = None`), #27 (`Path(None)`), #22 (`temporary = None`), #21 (inverted check): **A.** The branch at `publish.py:355-357` never runs. All three `verify_dataset` calls in `test_publish.py` are at `:240` (no temp dir; raises at `_validate_tree` first), `:313` and `:356` (temp dir set). No existing test takes the default path. A test with the default temp dir and a fake download that checks the directory should kill #22, #26 and #27. #21 dies only if a test asserts where downloads land.
- `verify_dataset` #30: **B** (see above).
