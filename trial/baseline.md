# Baseline (repo config, before any change), branch at 35ce838

| What | Result | Time |
|---|---|---|
| `tests/unit/test_manifest_state.py` + `tests/unit/test_publish.py` | 54 passed | about 6 s wall (trial report; 5.97 s re-run by the handoff step) |
| Same, with `HF_HUB_OFFLINE=1` | 54 passed | 0.83 s (trial report) |
| `tests/unit/test_check_mutation.py` | 14 passed | 0.09 s (trial report; not re-run in the handoff step) |

External calls: none found in these tests. They use local fake API classes and monkeypatch `download_to_temp`. The offline run passed. The check covers `huggingface_hub` only, not raw `httpx`.

Mutant inventory (dry count, no tests run):
- `manifest_state.py`: 22 top-level functions, 2 classes, 443 mutants.
- `publish.py`: 23 top-level functions, 5 classes, 501 mutants.
- Name format: `osm_polygon_eunis.<module>.x_<function>__mutmut_<n>`. Filter `osm_polygon_eunis.<module>.*` works.
