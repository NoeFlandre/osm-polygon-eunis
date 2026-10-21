"""Verified metadata pins for frozen source repository file trees."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from osm_polygon_eunis.pool_contract import TextSource

_HANDOFF_PATH = "data/raw-source-tree-pins.json"
_HANDOFF_SHA256 = "eef1267cca7cbf455f09d4df9d41cff592cf841b313ea0d9c71262d24ad5f107"
_EXPECTED_TREE_COUNT = 2305
_EXPECTED_SOURCE_PINS = {
    "website": (
        "NoeFlandre/osm-polygon-website-tag",
        "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
    ),
    "description": (
        "NoeFlandre/osm-polygon-description-tag",
        "b4706eb315c66a8a57135289e7612dca8e03caf8",
    ),
    "wiki": (
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "d6fc058ccf4643bc5fe5bfd5c1c0bc403d3da801",
    ),
}
_EXPECTED_TREE_DIGESTS = {
    ("website", "polygons/"): "5ef66bc56532914428d800a1671bd1fd1b33cf54966617197e6e1c548788ca9d",
    ("description", "data/"): "cfedbc2235eecda19e52527dc45efee043af5909ac483976db8b844dd4a0f599",
    ("description", "language-v1/data/"): (
        "d76a3a42708307d1932c55c24499f39c2fd48c5fec712d15f0b65b0cb6c60f53"
    ),
    ("wiki", "polygons/"): "1628f43bf1d42041d0e41b18156abbb1f52420fd7f138027ad4b96fcfa0c9dd6",
    ("wiki", "wikipedia/sentences/"): (
        "79f54e06708b5755f3cf09c59890c6504669095c3529b690016c09aef04df6c8"
    ),
    ("wiki", "wikivoyage/sentences/"): (
        "c407eb5b70cada37fde70a2b92882ddad301e954f9c2a28713443663ab9fefa7"
    ),
}


@dataclass(frozen=True, slots=True)
class SourceTreeFile:
    """One immutable repository path and its pinned object metadata."""

    path: str
    size_bytes: int
    blob_id: str
    lfs_oid: str | None
    lfs_size_bytes: int | None


@dataclass(frozen=True, slots=True)
class RawSourceTreePin:
    """One pinned raw-source tree; object bytes are not included here."""

    text_source: TextSource
    repository: str
    revision: str
    path_prefix: str
    file_count: int
    total_size_bytes: int
    inventory_sha256: str
    files: tuple[SourceTreeFile, ...]
    purpose: str


@lru_cache(maxsize=1)
def raw_source_tree_pins() -> tuple[RawSourceTreePin, ...]:
    """Load and verify the metadata-only raw source tree inventory."""

    resource = files("osm_polygon_eunis.pool_adapters").joinpath(_HANDOFF_PATH)
    handoff = json.loads(resource.read_text(encoding="utf-8"))
    if handoff.get("handoff_sha256") != _HANDOFF_SHA256:
        raise ValueError("source tree handoff does not match its verified file SHA-256")
    if handoff.get("source_key_counts") is not None:
        raise ValueError("metadata handoff cannot claim unverified raw-key counts")
    trees = _load_source_trees(handoff["raw_source_file_trees"])
    _validate_source_tree_set(trees)
    return trees


def _load_source_trees(raw_sources: list[dict[str, Any]]) -> tuple[RawSourceTreePin, ...]:
    trees = []
    for source in raw_sources:
        name = source["name"]
        if (source["repo"], source["resolved_sha"]) != _EXPECTED_SOURCE_PINS.get(name):
            raise ValueError(f"source repository or revision differs for {name!r}")
        trees.extend(_load_source_group(name, source, group) for group in source["groups"])
    return tuple(trees)


def _load_source_group(
    name: str,
    source: dict[str, Any],
    group: dict[str, Any],
) -> RawSourceTreePin:
    path_prefix = group["prefix"]
    inventory_sha256 = group["inventory_sha256"]
    if _EXPECTED_TREE_DIGESTS.get((name, path_prefix)) != inventory_sha256:
        raise ValueError(f"source tree {path_prefix!r} differs from its verified pin")
    files_in_group = tuple(
        SourceTreeFile(
            path=row["path"],
            size_bytes=row["size"],
            blob_id=row["blob_id"],
            lfs_oid=row["lfs_oid"],
            lfs_size_bytes=row["lfs_size"],
        )
        for row in group["files"]
    )
    pin = RawSourceTreePin(
        text_source=TextSource(name),
        repository=source["repo"],
        revision=source["resolved_sha"],
        path_prefix=path_prefix,
        file_count=group["file_count"],
        total_size_bytes=group["total_size_bytes"],
        inventory_sha256=inventory_sha256,
        files=files_in_group,
        purpose=_tree_purpose(path_prefix),
    )
    _validate_tree_pin(pin, group["files"])
    return pin


def _tree_purpose(path_prefix: str) -> str:
    if path_prefix in {"data/", "language-v1/data/", "polygons/"}:
        return "polygon_geometry"
    if path_prefix in {"wikipedia/sentences/", "wikivoyage/sentences/"}:
        return "downstream_text_documents"
    raise ValueError(f"unclassified pinned source tree {path_prefix!r}")


def _validate_source_tree_set(trees: tuple[RawSourceTreePin, ...]) -> None:
    if sum(pin.file_count for pin in trees) != _EXPECTED_TREE_COUNT:
        raise ValueError("metadata handoff raw file count differs from its pinned total")
    if {(pin.text_source.value, pin.path_prefix) for pin in trees} != set(_EXPECTED_TREE_DIGESTS):
        raise ValueError("source tree handoff is missing a verified file group")


def polygon_geometry_tree_pins(text_source: TextSource) -> tuple[RawSourceTreePin, ...]:
    """Return every geometry tree pinned for a text source."""

    return tuple(
        pin
        for pin in raw_source_tree_pins()
        if pin.text_source is text_source and pin.purpose == "polygon_geometry"
    )


def _validate_tree_pin(pin: RawSourceTreePin, raw_files: list[dict[str, Any]]) -> None:
    if len(pin.files) != pin.file_count:
        raise ValueError(f"source tree {pin.path_prefix!r} file count does not match metadata")
    if sum(row.size_bytes for row in pin.files) != pin.total_size_bytes:
        raise ValueError(f"source tree {pin.path_prefix!r} byte total does not match metadata")
    paths = tuple(row.path for row in pin.files)
    if len(set(paths)) != len(paths) or any(not path.startswith(pin.path_prefix) for path in paths):
        raise ValueError(f"source tree {pin.path_prefix!r} has duplicate or out-of-prefix paths")
    canonical = json.dumps(
        raw_files,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != pin.inventory_sha256:
        raise ValueError(f"source tree {pin.path_prefix!r} inventory digest does not match")
