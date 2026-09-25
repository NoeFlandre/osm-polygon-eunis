# Known technical debt

- The EEA source is a modelled probability raster, so the raster resolution and
  threshold affect the semantic precision of a label. The manifest records both.
- A future official vector habitat layer can replace the raster adapter behind
  the same reference interface; the pure matcher and output contract should not
  change.
- Full publication depends on valid GitHub and Hugging Face write credentials;
  public read access is not sufficient evidence of publication authority.
- Two error paths still return a "no data" value without being counted
  (tracked in #28): `matching._intersection_area` drops a candidate when the
  exact GEOS intersection raises, and `geometry.parse_geometry` turns an
  undecodable geometry into `None`, which the output cannot tell apart from a
  row without geometry. The planned fix adds `intersection_errors` and
  `invalid_geometries` counters to each shard manifest with a release-failing
  threshold. GeoPackage tile discovery already fails closed on a corrupt or
  truncated database; only a missing (optional) tile table yields no layers.
