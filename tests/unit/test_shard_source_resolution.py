"""Characterise shard source resolution before the cross-module helper renames.

These pin the branches of the shard source helpers that the rest of the suite
does not reach: a cached geometry is reused without a download, and a shard is
refused when its completed label sidecar is missing.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from osm_polygon_eunis import shard_processing
from osm_polygon_eunis._protocols import HubApi, StreamClient
from osm_polygon_eunis.release_plan import DatasetPlan
from osm_polygon_eunis.sources import dataset_spec


def test_final_source_reuses_a_cached_geometry_without_downloading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = DatasetPlan(dataset_spec("wikidata"), "revision", (), (), ())
    cache_root = tmp_path / "cache"
    cached = cache_root / "wikidata" / "polygons__a__b.parquet"
    cached.parent.mkdir(parents=True)
    cached.write_bytes(b"cached geometry")

    def unexpected_download(*_args: object, **_kwargs: object) -> Path:
        raise AssertionError("a cached geometry must not be downloaded")

    monkeypatch.setattr(shard_processing, "download_to_temp", unexpected_download)

    source = shard_processing._final_source(
        cast(HubApi, object()),
        plan,
        "polygons/a/b.parquet",
        source_cache_root=cache_root,
        local_root=tmp_path / "local",
        http_client=cast(StreamClient, object()),
    )

    assert source == (cached, True)


def test_enrich_geometry_shard_requires_the_completed_label_sidecar(tmp_path: Path) -> None:
    plan = DatasetPlan(dataset_spec("wikidata"), "revision", (), (), ())
    context = SimpleNamespace(
        plan=plan,
        options=SimpleNamespace(sidecar_root=tmp_path / "sidecars"),
    )

    with pytest.raises(FileNotFoundError) as error:
        shard_processing._enrich_geometry_shard(
            cast(Any, context),
            "polygons/a/b.parquet",
            tmp_path / "source.parquet",
        )

    sidecar = tmp_path / "sidecars" / "wikidata" / "polygons__a__b.parquet.labels.parquet"
    assert str(error.value) == f"missing completed label sidecar: {sidecar}"
