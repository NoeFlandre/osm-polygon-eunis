from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

from osm_polygon_eunis.pool_adapters.source_staging import SourceIdentity, staged_source_path


def _identity(
    *,
    text_source: str = "website",
    repository: str = "owner/website",
    revision: str = "a" * 40,
    path_prefix: str = "polygons/",
    tree_inventory_sha256: str = "b" * 64,
) -> SourceIdentity:
    return SourceIdentity(
        text_source,
        repository,
        revision,
        path_prefix,
        tree_inventory_sha256,
    )


def test_staging_path_is_deterministic_and_namespaces_shared_relative_paths(
    tmp_path: Path,
) -> None:
    root = tmp_path / "inputs"
    source_path = "polygons/shared-shard.parquet"
    identity = _identity()
    first = staged_source_path(root, identity, source_path)
    second = staged_source_path(root, identity, source_path)
    other_source = staged_source_path(
        root,
        replace(identity, text_source="wiki", repository="owner/wiki"),
        source_path,
    )

    assert first == second
    assert first != other_source
    assert first.parent == other_source.parent == root
    assert first.name.endswith("-shared-shard.parquet")


def test_every_pin_component_changes_the_staging_path(tmp_path: Path) -> None:
    identity = _identity()
    source_path = "polygons/shared-shard.parquet"
    expected = staged_source_path(tmp_path, identity, source_path)

    for field in (
        "text_source",
        "repository",
        "revision",
        "path_prefix",
        "tree_inventory_sha256",
    ):
        value = getattr(identity, field)
        changed = replace(identity, **{field: f"changed-{value}"})
        assert staged_source_path(tmp_path, changed, source_path) != expected
    assert staged_source_path(tmp_path, identity, "polygons/other.parquet") != expected


def test_staging_path_uses_stable_unicode_safe_identity_encoding(tmp_path: Path) -> None:
    identity = _identity(revision="révision")
    source_path = "polygons/shared-shard.parquet"
    encoded_identity = (
        '["website","owner/website","révision","polygons/","'
        + "b" * 64
        + '","polygons/shared-shard.parquet"]'
    ).encode("utf-8")
    expected_digest = hashlib.sha256(encoded_identity).hexdigest()

    assert staged_source_path(tmp_path, identity, source_path) == tmp_path / (
        f"{expected_digest}-shared-shard.parquet"
    )
