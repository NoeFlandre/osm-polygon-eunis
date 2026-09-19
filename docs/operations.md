# Operations

Use a temporary directory on the HDD with enough room for one source shard,
one replacement shard, and one reference group. Set `UV_CACHE_DIR` outside the
dataset root. Production commands emit JSON-line progress records and verify
row counts and schemas after every upload.

The release order is:

1. Capture the current source revisions and plan inventory.
2. Resolve and checksum the official EEA reference assets.
3. Duplicate each source dataset server-side.
4. Process geometry shards and, for Wikidata, the matching link shard.
5. Verify each remote tree independently.
6. Add the verified datasets to the `OSM Polygon EUNIS` collection.

The shared Wikidata/Wikipedia document, section, sentence, Wikivoyage, and
Wikidata-fact tables are intentionally unchanged. Only `polygons` and
`polygon_document_links` receive EUNIS fields.
