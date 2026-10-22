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
_SOURCE_JOIN_CONTRACT_PATH = "data/source-join-contract.json"
_SOURCE_JOIN_HANDOFF_SHA256 = "d1c4bf9e60468a85b17c535be17ae4efeba7284b718d60235e19a39a9e65a072"
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
_EXPECTED_RELEVANCE_REVISIONS = {
    "website": "e6524c711ee0033c3cd5990e5fdbbe4b89992dfb",
    "description": "ce98b328f3c6543800a3d931877b51eb3f0cf768",
    "wiki": "49307aeb05b6dcc8a5d393f77f59a1e6d13a967a",
}
_EXPECTED_AUXILIARY_TREE_PINS = {
    ("website", "labels/polygons/"): (
        386,
        3299123482,
        "0a2c112b0d4198fddaf0c20153f0f1fc8f0f661c265446f741da787fa867ee96",
    ),
    ("description", "labels/language-v1/data/"): (
        386,
        100444539,
        "79ef71e658e6ac666a2a23ccdd3746220b525e0bb9ef6294ab8ceda9767a7768",
    ),
    ("description", "language-v1/data/"): (
        386,
        103449770,
        "d76a3a42708307d1932c55c24499f39c2fd48c5fec712d15f0b65b0cb6c60f53",
    ),
    ("wiki", "polygon_document_links/"): (
        386,
        112581825,
        "0098e51b187d0d916479126d7eddff69bca6b51281778e4c1e951952794fdf12",
    ),
    ("wiki", "wikipedia/documents/"): (
        386,
        8739038060,
        "9382b4ba192618b612cd2aed46a91d222f182c68ea68d681070bf945543c3050",
    ),
    ("wiki", "wikivoyage/documents/"): (
        375,
        80078157,
        "cbb38750e68e7228784ed4b8736fafa24988cea8054e889298c66a25d5ed0022",
    ),
    ("wiki", "language_splits/polygon_document_links_by_language/"): (
        356,
        154309530,
        "1e3b94d1e723fd68c5b980128e1e408ac4201783288b60900a212a73e584464a",
    ),
    ("wiki", "labels/wikipedia/sentences/"): (
        386,
        8191284820,
        "310214740d4c0d10b4228583992c0e0842df680ef68996f8cc019f9cdcb4268f",
    ),
    ("wiki", "labels/wikivoyage/sentences/"): (
        375,
        189959172,
        "3cdd0e4f0fa8b6588e578ff077467a3e4516e676d7a4abd30cb622477cf8245d",
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
    verified_lfs_content_sha256: str | None = None


# Hugging Face's pinned Website shard has a Git LFS pointer whose metadata was
# omitted by the original tree inventory. The public pointer at the pinned
# commit hashes to this Git blob and names this content digest and size. Keep
# the supplemental resolution separate from the metadata inventory digest.
_VERIFIED_LFS_CONTENT_PINS = {
    (
        "NoeFlandre/osm-polygon-website-tag",
        "4cbddf95ce614e68d5ada8b94ca21bf9e48d39dd",
        "polygons/afghanistan-latest.parquet",
    ): (
        "702eb4afb4c63502f7f598141c7f97f120a57f02",
        "796965a9013b2ed590d1797a77a1dfa86dcda60e75fceb8d05d460d37e70c1b2",
        145293,
    ),
}


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


@lru_cache(maxsize=1)
def source_join_contract() -> dict[str, Any]:
    """Load downstream join pins and require provenance to match the verified handoff."""

    resource = files("osm_polygon_eunis.pool_adapters").joinpath(_SOURCE_JOIN_CONTRACT_PATH)
    contract = json.loads(resource.read_text(encoding="utf-8"))
    _validate_source_join_contract(contract)
    return contract


def _validate_source_join_contract(contract: dict[str, Any]) -> None:
    """Check metadata provenance, raw geometry pins, and every auxiliary tree pin."""

    handoff = contract.get("metadata_handoff", {})
    if handoff.get("sha256") != _SOURCE_JOIN_HANDOFF_SHA256:
        raise ValueError("source join contract does not match its verified metadata handoff")
    sources = contract.get("sources", {})
    if set(sources) != {"website", "description", "wiki"}:
        raise ValueError("source join contract must pin all three polygon text sources")
    pinned_trees = {(pin.text_source.value, pin.path_prefix): pin for pin in raw_source_tree_pins()}
    for name, source in sources.items():
        _validate_source_join_entry(name, source, pinned_trees)
    if _observed_auxiliary_tree_paths(sources) != set(_EXPECTED_AUXILIARY_TREE_PINS):
        raise ValueError("source join contract is missing a frozen label, link, or document tree")
    if contract.get("inventory_status", {}).get("unique_raw_polygon_keys") != "not_computed":
        raise ValueError("source join contract cannot claim unverified raw polygon key counts")


def _validate_source_join_entry(
    name: str,
    source: dict[str, Any],
    pinned_trees: dict[tuple[str, str], RawSourceTreePin],
) -> None:
    if (source.get("raw_repository"), source.get("raw_revision")) != _EXPECTED_SOURCE_PINS[name]:
        raise ValueError(f"source join contract revision differs for {name!r}")
    geometry = source.get("raw_geometry_tree", {})
    pin = pinned_trees.get((name, geometry.get("path")))
    observed = (
        geometry.get("file_count"),
        geometry.get("total_size_bytes"),
        geometry.get("inventory_sha256"),
    )
    if pin is None or (pin.file_count, pin.total_size_bytes, pin.inventory_sha256) != observed:
        raise ValueError(f"source join contract raw geometry tree differs for {name!r}")
    if source.get("relevance_revision") != _EXPECTED_RELEVANCE_REVISIONS[name]:
        raise ValueError(f"source join contract relevance revision differs for {name!r}")
    _validate_auxiliary_tree_pins(name, source)


def _validate_auxiliary_tree_pins(name: str, source: dict[str, Any]) -> None:
    for summary in _auxiliary_tree_summaries(source):
        expected = _EXPECTED_AUXILIARY_TREE_PINS.get((name, summary.get("path")))
        observed = (
            summary.get("file_count"),
            summary.get("total_size_bytes"),
            summary.get("inventory_sha256"),
        )
        if expected != observed:
            raise ValueError(
                f"source join contract auxiliary tree differs for {name!r}: {summary.get('path')!r}"
            )


def _auxiliary_tree_summaries(source: dict[str, Any]) -> tuple[dict[str, Any], ...]:
    """Return non-geometry metadata tree summaries in a stable order."""

    summaries = []
    for field in (
        "derived_text_tree",
        "relevance_tree",
        "link_tree",
        "alternate_language_link_tree",
    ):
        tree = source.get(field)
        if tree is not None:
            summaries.append(tree)
    for field in ("document_trees", "relevance_trees"):
        summaries.extend(source.get(field, ()))
    return tuple(summaries)


def _observed_auxiliary_tree_paths(sources: dict[str, Any]) -> set[tuple[str, str]]:
    return {
        (name, summary["path"])
        for name, source in sources.items()
        for summary in _auxiliary_tree_summaries(source)
    }


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
        _source_tree_file(name, source["resolved_sha"], row) for row in group["files"]
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


def _source_tree_file(name: str, revision: str, row: dict[str, Any]) -> SourceTreeFile:
    identity = (f"NoeFlandre/osm-polygon-{name}-tag", revision, row["path"])
    if name == "wiki":
        identity = ("NoeFlandre/osm-polygon-wikidata-and-wikipedia", revision, row["path"])
    verified = _VERIFIED_LFS_CONTENT_PINS.get(identity)
    if verified is not None:
        pointer_blob, content_sha256, content_size = verified
        if (
            row["blob_id"] != pointer_blob
            or row["lfs_oid"] is not None
            or row["size"] != content_size
            or row["lfs_size"] != content_size
        ):
            raise ValueError("verified Git LFS content pin no longer matches its public tree row")
    else:
        content_sha256 = None
    return SourceTreeFile(
        path=row["path"],
        size_bytes=row["size"],
        blob_id=row["blob_id"],
        lfs_oid=row["lfs_oid"],
        lfs_size_bytes=row["lfs_size"],
        verified_lfs_content_sha256=content_sha256,
    )


def _tree_purpose(path_prefix: str) -> str:
    if path_prefix in {"data/", "polygons/"}:
        return "polygon_geometry"
    if path_prefix == "language-v1/data/":
        return "derived_text_layer"
    if path_prefix in {"wikipedia/sentences/", "wikivoyage/sentences/"}:
        return "downstream_text_documents"
    raise ValueError(f"unclassified pinned source tree {path_prefix!r}")


def _validate_source_tree_set(trees: tuple[RawSourceTreePin, ...]) -> None:
    if sum(pin.file_count for pin in trees) != _EXPECTED_TREE_COUNT:
        raise ValueError("metadata handoff raw file count differs from its pinned total")
    if {(pin.text_source.value, pin.path_prefix) for pin in trees} != set(_EXPECTED_TREE_DIGESTS):
        raise ValueError("source tree handoff is missing a verified file group")


def polygon_geometry_tree_pins(text_source: TextSource) -> tuple[RawSourceTreePin, ...]:
    """Return only frozen raw polygon geometry trees for a text source."""

    return tuple(
        pin
        for pin in raw_source_tree_pins()
        if pin.text_source is text_source and pin.purpose == "polygon_geometry"
    )


def derived_text_layer_tree_pins(text_source: TextSource) -> tuple[RawSourceTreePin, ...]:
    """Return separately pinned text layers that join back to raw polygon keys."""

    return tuple(
        pin
        for pin in raw_source_tree_pins()
        if pin.text_source is text_source and pin.purpose == "derived_text_layer"
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
