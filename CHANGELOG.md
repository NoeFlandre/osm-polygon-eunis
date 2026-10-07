# Changelog

This file contains the release notes for this project. The version numbers are in
`pyproject.toml`. This file does not repeat them.

## Unreleased

- Add points to the WGS84 polygon edges before projection. Reject polygons that
  cross the antimeridian.
- Keep only areal geometry. Record the invalid and collapsed inputs in the
  manifest.
- Record the geometry policy in the dataset cards and manifests.
- Restore the Grid'5000 sidecar checkpoints that have a checked signature. Set
  a limit on the worker reference caches.
- Add a non-root Docker image and a CI build check.
- Resume finalization only when the reference, sidecars, software version and
  source revision are unchanged. A changed input now re-publishes every shard.
