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
- Repair invalid EUNIS reference geometries with `make_valid`. Log and count
  the reference geometries that remain unusable as intersection errors. Bump
  the overlap kernel version, which invalidates resumable sidecars.
- Resume finalization only when the reference, sidecars, software version,
  software source commit and source revision are unchanged. A changed input now
  re-publishes every shard. Progress files written before this change carry no
  source commit, so the first run after upgrading re-publishes every shard.
  A run without a resolvable source commit fails instead of resuming.
