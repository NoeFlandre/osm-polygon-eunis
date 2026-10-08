# Trial configuration (issue #108): exact changes

The exact patch is `trial-config.patch` in this folder. It applies to `pyproject.toml` at commit `35ce838`, and it was NOT applied to the branch. It has three parts:

1. `source_paths` gains `src/osm_polygon_eunis/manifest_state.py` and `src/osm_polygon_eunis/publish.py`.
2. `pytest_add_cli_args_test_selection` gains `tests/unit/test_manifest_state.py` and `tests/unit/test_publish.py`.
3. `also_copy` gains 17 files (listed below).

`publish.py` is now in both `source_paths` and `also_copy`. The second entry is redundant, and it is harmless. Mutmut uses the mutated copy.

## Why the 17 files are needed

Trial result: with parts 1 and 2 alone, the stats run failed with `ModuleNotFoundError: osm_polygon_eunis.eea` (see `attempt1-failed-collection.log`). Mutmut copies only `source_paths` and `also_copy` into `mutants/`.

A module-level import check found no function-level-only import in the closure. Each file is needed for one of two reasons.

Needed by `manifest_state.py`, `publish.py`, or their unit tests (9):
- `eea.py`, `options.py`, `reference_staging.py`, `geopackage_reference.py`, `geopackage_sql.py`, `geopackage_tiles.py`, `raster_reference.py`, `raster_geometry.py`, `reference_cache.py`

Needed only because `tests/unit/test_manifest_state.py` imports `release_orchestration` at module level (8):
- `card_publishing.py`, `cards.py`, `geometry_checkpoints.py`, `geometry_chunks.py`, `geometry_jobs.py`, `geometry_workers.py`, `release_orchestration.py`, `shard_processing.py`

The closure of module-level imports has 28 modules. That matches the trial list.

`tests/conftest.py` is not in `also_copy`, but mutmut copies all of `tests/`, so it is present. Its autouse fixture sets `EUNIS_SOURCE_COMMIT` for every test.

## Other test files that reach these modules (not selected in the trial)

- `tests/unit/test_release_orchestration.py`: real calls to `_reference_manifest`, `_shared_blobs`, and `card_publishing._finalize_plan` (which runs the real `_verify_final_dataset`). Its closure is already in `also_copy`.
- `tests/acceptance/test_release_golden.py`, `tests/acceptance/test_dataset_enrichment.py`: real `build_manifest` through the release path. Acceptance tests are outside the trial selection.
- `tests/unit/test_release_preview.py`, `tests/acceptance/test_release_workflow.py`: import `cli` at module level. Adding them would also need `cli.py`, `grid5000.py` and `run_analysis.py` in `also_copy`.
- `tests/unit/test_cli.py`: imports publish types only. Indirect reach, not verified.
