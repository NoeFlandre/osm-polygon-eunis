from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.release_plan import (
    DATASET_NAMES,
    DatasetPlan,
    _cached_geometry_path,
    _geometry_paths,
    _sidecar_path,
    plan_datasets,
    selected_dataset_names,
)
from osm_polygon_eunis.sources import dataset_spec


class _InventoryApi:
    def __init__(self) -> None:
        self.touched: list[str] = []
        self.listed_revisions: list[object] = []

    def repo_info(self, repo_id: str, **_kwargs: object) -> SimpleNamespace:
        self.touched.append(repo_id)
        return SimpleNamespace(sha=f"revision-{repo_id.rsplit('-', 1)[-1]}")

    def list_repo_tree(self, repo_id: str, **kwargs: object) -> list[SimpleNamespace]:
        self.touched.append(repo_id)
        self.listed_revisions.append(kwargs.get("revision"))
        return [
            SimpleNamespace(path="README.md", blob_id="readme"),
            SimpleNamespace(path="polygons/z.parquet", blob_id="z"),
            SimpleNamespace(path="polygons/a.parquet", blob_id="a"),
            SimpleNamespace(path="polygon_document_links/a.parquet", blob_id="link"),
        ]


def test_selected_dataset_names_uses_declared_release_order() -> None:
    assert selected_dataset_names(["description", "website"]) == ("website", "description")
    assert selected_dataset_names(None) == DATASET_NAMES


def test_selected_dataset_names_rejects_unknown_names() -> None:
    with pytest.raises(ValueError, match="unknown dataset"):
        selected_dataset_names(["unknown"])


def test_plan_datasets_pins_revision_and_sorts_shards() -> None:
    api = _InventoryApi()

    plans = plan_datasets(cast(HubApi, api), ["website"])

    assert len(plans) == 1
    plan = plans[0]
    assert plan.spec.name == "website"
    assert plan.source_revision == "revision-tag"
    assert plan.geometry_paths == ("polygons/a.parquet", "polygons/z.parquet")
    assert plan.link_paths == ()
    assert api.touched == [plan.spec.source_repo, plan.spec.source_repo]


def test_plan_datasets_requires_region_links_for_wikidata() -> None:
    class MissingLinksApi(_InventoryApi):
        def list_repo_tree(self, repo_id: str, **_kwargs: object) -> list[SimpleNamespace]:
            self.touched.append(repo_id)
            return [SimpleNamespace(path="polygons/a.parquet", blob_id="a")]

    with pytest.raises(ValueError, match="unmatched region shards"):
        plan_datasets(cast(HubApi, MissingLinksApi()), ["wikidata"])


def test_plan_datasets_lists_the_pinned_revision_and_keeps_every_source_file() -> None:
    api = _InventoryApi()

    (plan,) = plan_datasets(cast(HubApi, api), ["website"])

    assert api.listed_revisions == [plan.source_revision]
    assert plan.source_files == (
        "README.md",
        "polygon_document_links/a.parquet",
        "polygons/a.parquet",
        "polygons/z.parquet",
    )


def test_unknown_dataset_names_are_reported_sorted_and_comma_separated() -> None:
    with pytest.raises(ValueError) as error:
        selected_dataset_names(["zeta", "alpha", "website"])

    assert str(error.value) == "unknown dataset(s): alpha, zeta"


def test_plan_requires_geometry_shards_with_an_exact_message() -> None:
    spec = dataset_spec("website")

    with pytest.raises(ValueError) as error:
        _geometry_paths(spec, ("README.md",))

    assert str(error.value) == "source has no geometry shards for website"


def test_sidecar_and_cached_geometry_paths_flatten_the_repository_path() -> None:
    spec = dataset_spec("wikidata")
    plan = DatasetPlan(spec, "revision", (), (), ())

    assert _sidecar_path(Path("/work"), spec, "polygons/a/b.parquet") == Path(
        "/work/wikidata/polygons__a__b.parquet.labels.parquet"
    )
    assert _cached_geometry_path(Path("/work"), plan, "polygons/a/b.parquet") == Path(
        "/work/wikidata/polygons__a__b.parquet"
    )
