"""Pinned Hugging Face source inventory and text-separated DuckDB queries."""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from collections.abc import Iterator, Mapping, Sequence
from typing import Any

LABEL_PINS = {
    "website": (
        "NoeFlandre/osm-polygon-website-tag-landuse",
        "e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
    ),
    "description": (
        "NoeFlandre/osm-polygon-description-tag-landuse",
        "ce98b328f3c6543800a3d931877b51eb3f0cf768",
    ),
    "wikipedia": (
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
        "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
    ),
    "wikivoyage": (
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia-landuse",
        "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
    ),
}
RAW_PINS = {
    "website": (
        "NoeFlandre/osm-polygon-website-tag",
        "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
    ),
    "description": (
        "NoeFlandre/osm-polygon-description-tag",
        "b4706eb315c66a8a57135289e7612dca8e03caf8",
    ),
    "wikipedia": (
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
    ),
    "wikivoyage": (
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
    ),
}
LABEL_PREFIX = {
    "website": "labels/polygons",
    "description": "labels/language-v1/data",
    "wikipedia": "labels/wikipedia/sentences",
    "wikivoyage": "labels/wikivoyage/sentences",
}
SOURCE_VERSION = "EEA EUNIS habitat probability maps v1 2021"
BATCH_SIZE = 256
H3_RESOLUTION = 3
MAX_PER_CELL = 20
TARGET = 500
MINIMUM_H3_CELL_COUNT = 25
EXPECTED_NATIVE_ASSET_COUNT = 136


def _url(repository: str, revision: str, path: str) -> str:
    encoded_path = urllib.parse.quote(path, safe="/")
    return f"https://huggingface.co/datasets/{repository}/resolve/{revision}/{encoded_path}"


def remap_pinned_parquet_urls(
    label_url: str,
    raw_urls: Sequence[str],
    staged_inputs: Mapping[str, str],
    *,
    require_staged: bool = False,
) -> tuple[str, tuple[str, ...]]:
    """Substitute verified local paths for exact pinned Parquet URLs."""

    def resolve(url: str) -> str:
        local_path = staged_inputs.get(url)
        if local_path is None:
            if require_staged:
                raise ValueError(f"required pinned Parquet input is not staged: {url}")
            return url
        return local_path

    return resolve(label_url), tuple(resolve(url) for url in raw_urls)


def _label_paths(stream: str) -> list[str]:
    repository, revision = LABEL_PINS[stream]
    prefix = urllib.parse.quote(LABEL_PREFIX[stream], safe="/")
    url = (
        f"https://huggingface.co/api/datasets/{repository}/tree/{revision}/{prefix}"
        "?recursive=true&expand=false"
    )
    paths: set[str] = set()
    visited: set[str] = set()
    while url:
        if url in visited:
            raise RuntimeError("Hugging Face tree pagination loop")
        visited.add(url)
        parsed_url = urllib.parse.urlsplit(url)
        if parsed_url.scheme != "https" or parsed_url.hostname != "huggingface.co":
            raise ValueError(f"unexpected pinned label URL: {url}")
        with urllib.request.urlopen(url, timeout=45) as response:
            payload = json.load(response)
            next_url = _next_page(response.headers.get("Link", ""), payload)
        entries = payload if isinstance(payload, list) else payload.get("items")
        if not isinstance(entries, list):
            raise TypeError(f"unexpected pinned label tree for {stream}")
        for entry in entries:
            if isinstance(entry, dict) and entry.get("type") == "file":
                path = entry.get("path")
                if isinstance(path, str) and path.endswith("-latest.parquet"):
                    paths.add(path)
        url = next_url
    if not paths:
        raise ValueError(f"no pinned label shards found for {stream}")
    return sorted(paths)


def _next_page(link_header: str, payload: object) -> str | None:
    match = re.search(r"<([^>]+)>\s*;\s*rel=\"?next\"?", link_header)
    if match:
        return match.group(1)
    if isinstance(payload, dict):
        links = payload.get("links", [])
        for link in links if isinstance(links, list) else []:
            if isinstance(link, dict) and link.get("rel") == "next":
                href = link.get("href")
                if isinstance(href, str):
                    return urllib.parse.urljoin("https://huggingface.co/", href)
    return None


def _raw_paths(stream: str, filename: str) -> tuple[str, ...]:
    if stream == "website":
        return (f"polygons/{filename}",)
    if stream == "description":
        return (f"language-v1/data/{filename}", f"data/{filename}")
    project = stream
    return (f"{project}/sentences/{filename}", f"polygons/{filename}")


def geometry_sql(stream: str) -> str:
    """Return a projected geometry-only join; this SQL intentionally selects no text."""
    if stream == "website":
        return """SELECT ? AS label_path, ? AS shard, ? AS stream,
                   l.decision,
                   concat(
                     cast(l.polygon_id AS VARCHAR), '|', l.field, '|',
                     cast(l.sentence_index AS VARCHAR)
                   ) AS sentence_key,
                   cast(l.polygon_id AS VARCHAR) AS polygon_id,
                   cast(p.osm_type AS VARCHAR) AS osm_type, cast(p.osm_id AS VARCHAR) AS osm_id,
                   NULL::VARCHAR AS source_pbf, l.text_sha256, p.geometry,
                   cast(l.field AS VARCHAR) AS _field,
                   cast(l.sentence_index AS INTEGER) AS _sentence_index,
                   sha256(concat('42|', ?,'|', l.decision, '|', ?,'|', sentence_key)) AS _order
            FROM read_parquet(?) l JOIN read_parquet(?) p
              ON cast(l.polygon_id AS VARCHAR) = cast(p.polygon_id AS VARCHAR)
            WHERE l.decision IN ('yes', 'no') AND p.geometry IS NOT NULL
            ORDER BY _order"""
    if stream == "description":
        return """SELECT ? AS label_path, ? AS shard, ? AS stream,
                   l.decision,
                   concat(
                     cast(l.description_identity AS VARCHAR), '|',
                     cast(l.tag_key AS VARCHAR), '|',
                     cast(l.sentence_index AS VARCHAR)
                   ) AS sentence_key,
                   NULL::VARCHAR AS polygon_id,
                   cast(g.osm_type AS VARCHAR) AS osm_type, cast(g.osm_id AS VARCHAR) AS osm_id,
                   cast(t.source_pbf AS VARCHAR) AS source_pbf, l.text_sha256, g.geometry,
                   cast(l.description_identity AS VARCHAR) AS _description_identity,
                   cast(l.tag_key AS VARCHAR) AS _tag_key,
                   cast(l.sentence_index AS INTEGER) AS _sentence_index,
                   sha256(concat('42|', ?,'|', l.decision, '|', ?,'|', sentence_key)) AS _order
            FROM read_parquet(?) l
            JOIN read_parquet(?) t
              ON l.description_identity = t.description_identity AND l.tag_key = t.tag_key
            JOIN read_parquet(?) g
              ON t.source_pbf = g.source_pbf AND t.osm_type = g.osm_type AND t.osm_id = g.osm_id
            WHERE l.decision IN ('yes', 'no') AND g.geometry IS NOT NULL
            ORDER BY _order"""
    if stream not in {"wikipedia", "wikivoyage"}:
        raise ValueError(f"unsupported census stream: {stream!r}")
    return """WITH labels AS (
                SELECT * FROM read_parquet(?) WHERE decision IN ('yes', 'no')
            ), matched AS (
                SELECT l.decision, cast(l.sentence_id AS VARCHAR) AS sentence_key,
                       cast(p.polygon_id AS VARCHAR) AS polygon_id,
                       cast(p.osm_type AS VARCHAR) AS osm_type, cast(p.osm_id AS VARCHAR) AS osm_id,
                       NULL::VARCHAR AS source_pbf, l.text_sha256, p.geometry,
                       cast(l.sentence_id AS VARCHAR) AS _sentence_id,
                       s.project, s.content_hash,
                       count(p.polygon_id) OVER (PARTITION BY l.sentence_id) AS polygon_matches
                FROM labels l
                JOIN read_parquet(?) s ON l.sentence_id = s.sentence_id
                JOIN read_parquet(?) p ON s.wikidata = p.wikidata
            )
            SELECT ? AS label_path, ? AS shard, ? AS stream, decision, sentence_key,
                   polygon_id, osm_type, osm_id, source_pbf, text_sha256, geometry,
                   _sentence_id,
                   sha256(concat('42|', ?,'|', decision, '|', ?,'|', sentence_key)) AS _order
            FROM matched
            WHERE polygon_matches = 1 AND project = ? AND text_sha256 = content_hash
              AND geometry IS NOT NULL
            ORDER BY _order"""


def _geometry_parameters(
    stream: str,
    source: str,
    label_path: str,
    shard: str,
    label_url: str,
    raw_urls: Sequence[str],
) -> list[object]:
    if stream not in {"website", "description", "wikipedia", "wikivoyage"}:
        raise ValueError(f"unsupported census stream: {stream!r}")
    if stream == "website":
        return [label_path, shard, stream, source, stream, label_url, *raw_urls]
    if stream == "description":
        return [label_path, shard, stream, source, stream, label_url, *raw_urls]
    return [
        label_url,
        raw_urls[0],
        raw_urls[1],
        label_path,
        shard,
        stream,
        source,
        stream,
        stream,
    ]


def _value_rows(
    rows: Sequence[Mapping[str, object]], keys: Sequence[str]
) -> tuple[str, list[object]]:
    placeholders = ",".join("(" + ",".join("?" for _ in keys) + ")" for _ in rows)
    values = [row[key] for row in rows for key in keys]
    return placeholders, values


def _text_query(stream: str, rows: Sequence[Mapping[str, object]]) -> tuple[str, list[object]]:
    if stream == "website":
        keys = ("sentence_key", "polygon_id", "_field", "_sentence_index")
        sql = """WITH selected AS (
                   SELECT * FROM (VALUES {})
                   AS v(sentence_key, polygon_id, field, sentence_index)
                 )
                 SELECT selected.sentence_key,
                   CASE selected.field
                     WHEN 'website' THEN list_extract(
                       p.website_sentences, selected.sentence_index + 1
                     )
                     WHEN 'contact_website' THEN list_extract(
                       p.contact_website_sentences, selected.sentence_index + 1
                     )
                   END AS raw_sentence
                 FROM selected JOIN read_parquet(?) p
                   ON cast(selected.polygon_id AS VARCHAR) = cast(p.polygon_id AS VARCHAR)"""
        path_key = "_raw_polygon_url"
    elif stream == "description":
        keys = ("sentence_key", "_description_identity", "_tag_key", "_sentence_index")
        sql = """WITH selected AS (
                   SELECT * FROM (VALUES {})
                   AS v(sentence_key, description_identity, tag_key, sentence_index)
                 )
                 SELECT selected.sentence_key,
                   list_extract(t.sentences, selected.sentence_index + 1) AS raw_sentence
                 FROM selected JOIN read_parquet(?) t
                   ON selected.description_identity = t.description_identity
                   AND selected.tag_key = t.tag_key"""
        path_key = "_raw_text_url"
    elif stream in {"wikipedia", "wikivoyage"}:
        keys = ("sentence_key", "_sentence_id", "text_sha256")
        sql = """WITH selected AS (
                   SELECT * FROM (VALUES {}) AS v(sentence_key, sentence_id, expected_hash)
                 )
                 SELECT selected.sentence_key, s.text AS raw_sentence
                 FROM selected JOIN read_parquet(?) s
                   ON selected.sentence_id = s.sentence_id
                 WHERE s.project = ? AND selected.expected_hash = s.content_hash"""
        path_key = "_raw_text_url"
    else:
        raise ValueError(f"unsupported census stream: {stream!r}")
    values_sql, values = _value_rows(rows, keys)
    sql = sql.format(values_sql)
    path_url = str(rows[0][path_key])
    parameters = [*values, path_url]
    if stream in {"wikipedia", "wikivoyage"}:
        parameters.append(stream)
    return sql, parameters


def _geometry_rows(db: Any, sql: str, params: Sequence[object]) -> Iterator[dict[str, object]]:
    result = db.execute(sql, params)
    names = [column[0] for column in result.description]
    while batch := result.fetchmany(BATCH_SIZE):
        for values in batch:
            yield dict(zip(names, values, strict=True))


def _geometry(value: object) -> Any:
    import shapely  # noqa: PLC0415
    from shapely.geometry import shape  # noqa: PLC0415

    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, memoryview):
        value = value.tobytes()
    if isinstance(value, (bytes, bytearray)):
        return shapely.from_wkb(bytes(value))
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and value.get("encoding") == "wkb-hex":
        return shapely.from_wkb(bytes.fromhex(str(value["value"])))
    if isinstance(value, dict):
        return shape(value)
    raise TypeError(f"unsupported geometry value {type(value).__name__}")


def _json_row(row: Mapping[str, object]) -> dict[str, object]:
    import shapely  # noqa: PLC0415

    output: dict[str, object] = {}
    for key, value in row.items():
        if key.startswith("_"):
            continue
        if key == "geometry":
            geometry = value
            if isinstance(geometry, memoryview):
                geometry = geometry.tobytes()
            if isinstance(geometry, (bytes, bytearray)):
                output[key] = {"encoding": "wkb-hex", "value": bytes(geometry).hex()}
            elif hasattr(geometry, "wkb"):
                output[key] = {
                    "encoding": "wkb-hex",
                    "value": shapely.to_wkb(geometry).hex(),
                }
            else:
                output[key] = value
        else:
            output[key] = value
    return output
