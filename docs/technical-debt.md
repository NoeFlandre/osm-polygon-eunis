# Known technical debt

- The EEA source is a modelled probability raster, so the raster resolution and
  threshold affect the semantic precision of a label. The manifest records both.
- A future official vector habitat layer can replace the raster adapter behind
  the same reference interface; the pure matcher and output contract should not
  change.
- Full publication depends on valid GitHub and Hugging Face write credentials;
  public read access is not sufficient evidence of publication authority.
- Undecodable geometries are now counted: each dataset manifest's `card`
  section records `invalid_geometries` (rows whose geometry value is present
  but cannot be decoded or repaired), separate from rows without geometry.
  One error path is still uncounted (tracked in #28):
  `matching._intersection_area` drops a candidate when the exact GEOS
  intersection raises. Counting it needs a per-row counter carried through the
  label sidecar across reference passes and worker processes (the sidecar
  schema and the `OverlapReference.overlap` protocol both change), so it is
  left for a dedicated change. No release-failing threshold is enforced yet.
  GeoPackage tile discovery already fails closed on a corrupt or truncated
  database; only a missing (optional) tile table yields no layers.
- Mutation testing (`[tool.mutmut]` in `pyproject.toml`) only mutates
  `matching.py`, the pure winner-selection core. The fail-closed paths in
  `reference.py`, `transform.py` and `publish.py` are covered by unit and
  acceptance tests but not by mutation testing; grow `source_paths` one module
  at a time (next: `transform.py`) and keep the gate green at each step.
- `ty` is a pre-release (`<0.1`); its checks can tighten between patch releases,
  so upgrades land through the lock file (Dependabot) and are reviewed like code.
- CI runs the lowest supported Python (3.12, pinned by `.python-version`).
  Dependabot bumps GitHub Actions and the uv lock weekly.
