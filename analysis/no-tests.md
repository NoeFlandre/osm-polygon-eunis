# The 30 no-tests in manifest_state.py

Decision: **(b) incomplete selection.** Not (a) missing tests, and not (c) collection or setup failure.

- `_shared_blobs` (8):
  - A real call from `tests/unit/test_release_orchestration.py:584`. No fake.
  - The only selected caller fakes it at `tests/unit/test_manifest_state.py:355`.
- `_verify_final_dataset` (22):
  - The real path is `tests/unit/test_release_orchestration.py:657` -> `card_publishing.py:80` -> `manifest_state.py:351`. It runs for real there. Only `verify_dataset` is faked (`test_release_orchestration.py:624`).
  - The selected test fakes it at `tests/unit/test_manifest_state.py:357`.
- Collection or setup problem: none found. `test_release_orchestration.py` has no skip or slow marks at `:516` or `:594`. Only parametrize at `:169` and `:374`.

Caveat: adding `tests/unit/test_release_orchestration.py` to the selection will not kill the 22 `_verify_final_dataset` mutants by itself. The `verify_dataset` fake is `lambda *args, **kwargs`, so mutants in the arguments become group C. Killing them needs a fake that records and asserts its arguments.

## Correction to the trial report

The trial report said the no-tests came from tests faking `_verify_no_op_dataset` and `_shared_blobs`. That is incomplete. `_verify_final_dataset` is faked too, at `test_manifest_state.py:357`.
