# Changelog

Release notes for this project are maintained here. Version numbers are read
from `pyproject.toml` and are not repeated in this file.

## Unreleased

- Densify WGS84 polygon edges before projection and reject antimeridian-spanning
  polygons.
- Retain only areal geometry, recording invalid and collapsed inputs in the
  manifest.
- Record the geometry policy in dataset cards and manifests.
- Restore signature-checked Grid'5000 sidecar checkpoints and bound worker
  reference caches.
- Add a non-root Docker image and CI build check.
