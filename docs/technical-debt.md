# Known technical debt

- The EEA source is a modelled probability raster, so the raster resolution and
  threshold affect the semantic precision of a label. The manifest records both.
- A future official vector habitat layer can replace the raster adapter behind
  the same reference interface; the pure matcher and output contract should not
  change.
- Full publication depends on valid GitHub and Hugging Face write credentials;
  public read access is not sufficient evidence of publication authority.
