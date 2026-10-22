# Known technical debt

- EEA maps contain modelled suitability/probability scores, not observations of
  actual habitat distribution. The source-wide label is a highest-source-score
  derived classification. Scores are not verified as calibrated across habitat
  models, and the exact raster scale, offset, band, and NoData metadata remain
  to be checked against the pinned assets.
- A future official vector habitat layer can replace the raster adapter behind
  the same reference interface. The pure matcher and the output contract must
  not change.
- There is no configurable limit for invalid geometries. The manifest records
  the count. A release cannot currently fail because of this count.
- `ty` is a pre-release (`<0.1`). Its checks can become stricter between patch
  releases. Upgrades come through the lock file (Dependabot). Review them like
  code.
- EEA reference assets can have reuse terms for each item. Before you
  redistribute derived data, verify the notice of the exact asset. Record the
  notice with the release evidence.
