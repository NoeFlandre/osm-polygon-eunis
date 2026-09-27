# Known technical debt

- The EEA source is a modelled probability raster, so the raster resolution and
  threshold affect the semantic precision of a label. The manifest records both.
- A future official vector habitat layer can replace the raster adapter behind
  the same reference interface; the pure matcher and output contract should not
  change.
- There is no configurable limit for invalid geometries. The count is recorded
  in the manifest, but a release currently cannot fail based on that count.
- `ty` is a pre-release (`<0.1`); its checks can tighten between patch releases,
  so upgrades land through the lock file (Dependabot) and are reviewed like code.
- EEA reference assets can carry item-specific reuse terms. Verify the exact
  asset notice and record it with the release evidence before redistributing
  derived data.
