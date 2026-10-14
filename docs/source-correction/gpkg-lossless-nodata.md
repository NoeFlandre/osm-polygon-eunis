# Lossless NoData handling for EEA GeoPackage rasters

## Finding

The current GeoPackage path reads the compressed tile's first band with
`masked=True`, counts values above the configured threshold, and applies an
alpha-band check when the tile exposes four bands. The Q11 cells below have no
native habitat presence. The native FileGDB mask marks them NoData, while GDAL's
GeoPackage raster view reports no NoData value and an opaque alpha band. Its
decoded values therefore pass the current threshold. The raw tile is reported
as lossy JPEG; this audit did not save or hash the complete tile blob.

The regression fixture is
[available in the repository](https://github.com/NoeFlandre/osm-polygon-eunis/blob/b9ad56823352d1628b5dfed92c48ab9ef9da1b90/tests/fixtures/gpkg-nodata-q11-window.json).
It preserves the native metadata-file SHA-256, GDAL and Rasterio versions, grid,
cell coordinates, values, masks, and the exact bounded GeoPackage prefix digest.
The GeoPackage digest covers only its first 8 MiB. The two raster fixtures total
330,437 bytes. A separate source-audit task independently verified the Q11
GeoPackage whole-file SHA-256
`a7eab477c2447b7493074b4cc5fab5bc9e118dd27fed227d5ab49dea0a991201` against the
downloaded 1,196,740,608-byte file. This workspace did not perform that
verification.

The EEA catalog records for Q, MA, and U state CC BY 4.0 and identify the EEA as
copyright holder. The fixture carries the attribution:
"European Environment Agency (EEA), EUNIS habitat probability maps v1 2021,
CC BY 4.0." See the [Q record](https://sdi.eea.europa.eu/catalogue/datahub/api/records/ae2fdada-93d6-4cf1-a11c-6ebccf25d286?language=eng),
[MA record](https://sdi.eea.europa.eu/catalogue/datahub/api/records/7a2d78ec-8a31-4e7b-91df-45db6e64e842?language=eng),
[U record](https://sdi.eea.europa.eu/catalogue/datahub/api/records/a6c48c2d-114f-406e-ba99-c9085a5f5aee?language=eng),
and the [EEA legal notice](https://www.eea.europa.eu/en/legal-notice).

For the full 256 by 256 Q11 tile at zoom 8, column 101, row 138, the native
values and masks yield zero positive cells. The current GeoPackage rule yields
36,524. It marks all 33,630 native NoData cells positive, plus 2,894 native
valid zeros. There are no false negatives in this tile. The adjacent 3 by 4
Basilica window accounts for seven of those false positives. These counts cover
one tile only, not the full Q11 raster or published polygons.

The lossy values range across the byte scale for native NoData cells: 31,063
decode to 255, 258 to 254, and 2,309 to values from 1 through 253. A 254 cutoff
would still leave 2,309 NoData cells looking like presence. The native mask
identifies all 33,630 invalid cells directly.

## Distribution inventory

The pinned EEA config resolves three GeoPackages through
`GeoPackageReference`, which uses the same tile decoder:

| Family | GeoPackage | Layers | Count |
| --- | --- | --- | ---: |
| Wetlands | `Wetlands_Q_prob_100m.gpkg` | Q11, Q12, Q21, Q22, Q23, Q24, Q25, Q31, Q41, Q42, Q43, Q44, Q45, Q46, Q51, Q52, Q53, Q54 | 18 |
| Saltmarshes | `Saltmarshes_MA_prob_100m.gpkg` | MA221, MA222, MA223, MA224, MA225, MA232, MA241, MA251, MA252, MA253 | 10 |
| Sparse vegetation | `Sparsely_U_prob_100m.gpkg` | U21, U22, U23, U24, U25, U26, U27, U28, U29, U32, U33, U34, U36, U37, U38, U3A, U3B, U3D, U62 | 19 |

This inventories 47 layers using the GeoPackage tile decoder. Only Q11 has a
native-versus-GeoPackage pixel comparison so far. The other 46 layers are in
scope for the audit, not declared defective.

The three packages have full sizes 1,196,740,608 bytes for Q, 60,710,912 bytes
for MA, and 260,083,712 bytes for U. The fixture records first-8-MiB SHA-256
digests: Q `4f1ef141...e047a80`, MA `6fed7d77...b49ec14f`, and U
`e3eb4bd8...15f47892`. These range hashes are not whole-package checksums.

## Regression assertions

- Use the native raster mask to decide validity. All 33,630 native masked pixels
  stay invalid, including pixels that the GeoPackage decodes as 254 or 255.
- Apply the configured presence threshold only to valid native values. Preserve
  a valid 254 as presence at threshold 0; do not use a 254 or 255 cutoff to
  guess validity.
- The fixed reader returns zero positive cells for this Q11 tile and records
  the exact source revision, raster grid, mask source, and package checksums.
- If a layer lacks a pinned native mask or lossless raster, fail closed and
  report it as unresolved.

## Correction plan

1. Make a pinned native, lossless source the value and validity authority for
   each layer. Read its native NoData value or explicit mask. Keep the existing
   presence threshold for valid source pixels only.
2. Do not infer NoData from decoded JPEG intensity. In particular, do not add a
   cutoff at 254 or 255. JPEG can alter both NoData and valid edge values.
3. If an official native value raster or mask is unavailable, fail closed for
   that layer. Do not produce a substitute label from lossy pixels.
4. Extend the regression fixture tests to prove that a masked native 255 stays
   invalid when the lossy tile decodes it as 254 or 255, and that valid source
   values are evaluated with the normal threshold. Keep alpha and native mask
   behavior explicit.
5. Before recomputing labels, compare native values and masks with the decoded
   GeoPackage for all 47 layers. Report per-layer tile counts, mask mismatches,
   valid-value mismatches, and affected polygon rows and winner labels. The
   current measured bound is 36,524 false positive cells in one Q11 tile; the
   full-layer and polygon-row impact remains unmeasured.
6. Recompute into new output paths under a new source revision and manifest.
   Preserve old published results. Require exact source hashes and reviewed
   impact counts before publishing.

## Verification boundary

The native Q11 metadata file `a00000004.gdbtable` has SHA-256
`a05b6afa9a8691f6449bb04d6c748b186d2dce91cb22f8d14fc39939df0b6809` and records
`NoDataValue=255`. GDAL 3.10.3 reads the native grid as EPSG:3035, 100 m pixels,
origin `(2580000, 5445000)`. The remote Q11 GeoPackage subdataset has four Byte
bands and reports no NoData value. The first 8 MiB of the Q GeoPackage has SHA-256
`4f1ef141446590fa2bc95a459ec928a712583da16d1dc0d7dc53b7aede047a80`; this is a
range digest, not a whole-file digest. The reported Basilica cells are native
NoData, with GeoPackage values 254 and 255 and dataset alpha 255.

No source data or published output was recomputed. This work is separate from
the 1,000-row sampling change. The earlier independent Luna review covered the
Wikidata consumer only. The Haiku review attempt failed during streaming. Neither
fact establishes full quality approval for this source correction.
