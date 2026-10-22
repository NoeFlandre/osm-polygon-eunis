# EUNIS original-source verification, 2026-10-22

This report records the public-source evidence checked for PR #167. It does
not change the reference source or authorize production labels.

## Taxonomy workbook and cache

The original EEA 2021_1 workbook was downloaded from the official public share
on 2026-10-22. The file was 1,439,940 bytes and its SHA-256 was
`1427d22db8d77d1053ab1683aa60713bdec3d810605dd6b7bc16683b4ca14665`, matching
the workbook pin in `src/osm_polygon_eunis/eea.py`. A fresh parse using
`parse_classification_taxonomy_workbook()` and the pinned supplemental `MA223`
label produced 3,950 labels, 3,949 level entries, and 3,942 parent links. The
level counts were 7, 42, 270, 964, 1,419, 910, 314, and 23 for levels 1-8.
Every parsed level code had a label.

The public checkout does not contain the separately transferred cache payload
`classification-taxonomy-original-20261021.json`. It is absent from the PR
branch at `868cc836b654a58f78b2383008e82cf2bc21cfff` and public `main` at
`eddad228021152494b40c7b95e9eb07e8529e6f6`. The repository records its reported
size (284,146 bytes), digest
`223aa3fb89ac48cb5095e443433412c085e66fc54602e2c29253537455a2424f`, and the
same taxonomy counts, but those summary fields are not the cache bytes. Thus the
workbook parse matches the recorded counts only. The cache digest and exact
code-to-label, level, and parent equality could not be verified from public
repository artifacts. No cache payload was reconstructed or read from another
task's private storage.

The workbook is the original classification file, not one of the later
categorical map products:

- EEA classification record: <https://sdi.eea.europa.eu/catalogue/srv/api/records/bfe4c237-e378-4a83-ab21-b3807f96c2e2?language=eng>
- Direct workbook: <https://sdi.eea.europa.eu/datashare/public.php/dav/files/sptXqwkQr5g7Bp5/eea_t_eunis-hab-t_p_2004-on-going_v01_r00/EUNIS%20terrestrial%20habitat%20classification%202021_1%20including%20crosswalks.xlsx>

The workbook's crosswalk sheets were not extracted in this check. The EUNIS
label-map production gate must continue to require the actual pinned cache
payload and full lineage comparison.

## Map products and coverage

The checked-in native inventory has 136 entries and 5,007,209,626 bytes. Its
filenames use four top-level code prefixes: N=17, R=47, S=34, and T=38. The
inventory records names, sizes, and SHA-256 values, but it has no per-asset
source-output-semantics evidence. The raster bytes themselves were not fetched
or inspected in this source review.

The EEA probability-map metadata identifies separate product families. It
describes probability maps as refinements of habitat-suitability maps that
include existing land-cover information. The checked-in `catalog_records`
contains seven record IDs (N, R, S, T, Q, MA, U), while its file inventory
contains only N, R, S, and T. The available record-level facts are:

| Family | Checked-in assets | EEA probability-map record | Public metadata |
| --- | ---: | --- | --- |
| N, coastal | 17 | [99498d2c](https://sdi.eea.europa.eu/catalogue/srv/api/records/99498d2c-7350-4655-b914-92c5b9e016c5?language=all) | 100 m GeoTIFF. The metadata describes modeled habitat probability and values from 0-1 as suitability for the habitat. |
| R, grassland | 47 | [0c4270c4](https://sdi.eea.europa.eu/catalogue/srv/api/records/0c4270c4-fd7e-4099-ac2f-5af2079ddbd8?language=all) | 100 m GeoTIFF. The record identifies the probability-map family and source dataset. |
| S, heathland, scrub, tundra | 34 | [443cdba1](https://sdi.eea.europa.eu/catalogue/srv/api/records/443cdba1-1d4b-4cae-91d5-72fa99a8a758?language=all) | 100 m GeoTIFF. The record identifies the probability-map family and source dataset. |
| T, forest | 38 | [f425a73e](https://sdi.eea.europa.eu/catalogue/srv/api/records/f425a73e-dc6c-40be-93d6-9d234d4bbe1b?language=all) | GeoTIFF probability-map metadata identifies a 1 km product folder and a 0-1 suitability-score interpretation. This conflicts with the pinned `Prob_T..._100m.tif` filenames. |
| Q, wetlands | 0 | [ae2fdada](https://sdi.eea.europa.eu/catalogue/srv/api/records/ae2fdada-93d6-4cf1-a11c-6ebccf25d286?language=all) | 100 m grid metadata lists GDB and GeoPackage formats. No Q assets are in the 136-file inventory. |
| MA, littoral biogenic | 0 | [7a2d78ec](https://sdi.eea.europa.eu/catalogue/srv/api/records/7a2d78ec-8a31-4e7b-91df-45db6e64e842?language=all) | Probability-map record is cataloged separately from the 1 km suitability GeoPackage. No MA assets are in the 136-file inventory. |
| U, inland sparse vegetation | 0 | [a6c48c2d](https://sdi.eea.europa.eu/catalogue/srv/api/records/a6c48c2d-114f-406e-ba99-c9085a5f5aee?language=all) | 100 m GeoTIFF probability-map metadata. No U assets are in the 136-file inventory. |

The probability-map overview does not list V. Separately, the EEA 2021
habitat-suitability series lists eight groups: MA, N, Q, R, S, T, U, and V.
That series says it covers more than 100 raster files at 1 km. Its component
records classify the Q, MA, and U suitability products as 1 km grids distributed
as GeoPackage and geodatabase files:

- Series scope: <https://sdi.eea.europa.eu/catalogue/srv/api/records/adbb2781-2d4d-4a6c-8ce1-875bae0f6703>
- Q wetlands suitability: <https://sdi.eea.europa.eu/catalogue/srv/api/records/917348e8-42ea-4df9-8fe8-4d47bcd8d241>
- MA salt-marsh suitability: <https://sdi.eea.europa.eu/catalogue/srv/api/records/5b3e4da9-4c14-498c-b20e-bc514470eab5>
- U sparse-vegetation suitability: <https://sdi.eea.europa.eu/catalogue/srv/api/records/96b7f7df-3f79-4001-b489-1d918c813a62>
- V vegetated man-made suitability: <https://sdi.eea.europa.eu/catalogue/srv/api/records/05cbf86d-0168-47f5-9384-18c84b14f59c>

These suitability GeoPackages are a distinct output family from probability
maps. EEA metadata says suitability maps model where the environment is
suitable; probability maps additionally incorporate existing land cover and
narrow that modeled extent. The Q, MA, and U suitability GeoPackages therefore
cannot be treated as byte-equivalent replacements for absent probability
assets. The series also identifies V coverage that is not present in the
checked-in probability-record list or 136-file inventory.

## What remains unverified

Public EEA catalog records and product-folder links were identified, and the
workbook bytes were recovered. The full probability-map asset bytes, the
separate Q/MA/U and V suitability GeoPackages, any V probability-map product,
and source-side per-file manifests were not retrieved in this cloud review.
The current public metadata
does not establish the exact source asset corresponding to every pinned TIFF,
TIFF scale/offset tags, NoData or model-domain masks, or a complete per-asset
score-output family mapping. It also does not establish that independently
modeled scores are calibrated across EUNIS classes. The 1 km T catalog record
versus 100 m pinned filenames needs reconciliation.

The public evidence supports only the described proxy interpretations. It does
not certify a cross-habitat probability comparison or an observed-habitat
coverage guarantee. No source family was substituted, no probability
calibration was invented, and no production labeling or evaluation was run.
