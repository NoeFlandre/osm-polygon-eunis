# ADR 0002: Server-side duplication and shard processing

## Decision

Duplicate each source dataset on the Hugging Face Hub. Then replace only the
Parquet shards that contain geometry. Replace them one at a time. The downloaded
input shards and output shards are temporary. Remove them after the schema and
row-count verification.

## Consequences

The unchanged files stay remote. The local storage stays bounded.

A manifest records the source commit, the changed paths, the reference assets,
and the verification receipts. An interrupted run can resume with this manifest.
It does not process the verified shards again.
