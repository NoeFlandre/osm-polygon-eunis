# Glossary

This page defines the project terms. Each term has one meaning in all the
documents.

| Term | Meaning |
|---|---|
| Bounding box (bbox) | The smallest rectangle that contains a geometry. It selects candidates only. It never selects a label. |
| Candidate | A reference cell or window that can overlap a polygon. |
| Checkpoint | A saved state of a run. A later job uses it to resume. |
| Dataset card | The description page of a Hub dataset. It contains the label table and the map. |
| EEA | The European Environment Agency. |
| EPSG:3035 | The equal-area coordinate system that the project uses to measure areas. |
| EUNIS | The European Nature Information System habitat classification. |
| Exact overlap | The actual area of the intersection of a polygon and the reference geometry. |
| Grid'5000 | The French test-bed for large computing jobs. The production release runs there. |
| Hub | The Hugging Face Hub. |
| Manifest | The file `eunis/manifest.json`. It records the source revisions, the reference assets, and the checksums. |
| Micro-batch | A small group of Parquet rows. It has a fixed size limit. |
| No-op | A release run that finds all targets correct. It uploads nothing. |
| OAR | The job scheduler of Grid'5000. |
| Receipt | A file that records the result of one job. |
| Reference | The EEA 2021 EUNIS habitat probability map. |
| Shard | One Parquet file of a dataset. |
| Sidecar | A compact file that stores the labels of a finished batch. |
| Source | An input dataset. The sources are `website`, `wikidata`, and `description`. |
| Target | An output dataset on the Hub. |
| Verify | Compare a published target with its manifest. Do not write to the target. |
| Walltime | The maximum run time of an OAR job. |
| Worker | A process that computes the labels for the shards. |
