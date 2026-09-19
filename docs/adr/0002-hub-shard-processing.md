# ADR 0002: Server-side duplication and shard processing

## Decision

Duplicate each source dataset on the Hugging Face Hub, then replace only
geometry-bearing Parquet shards one at a time. Downloaded input and output
shards are temporary and are removed after schema and row-count verification.

## Consequences

Unchanged files remain remote and local storage stays bounded. A manifest records
the source commit, changed paths, reference assets, and verification receipts so
an interrupted run can resume without reprocessing verified shards.
