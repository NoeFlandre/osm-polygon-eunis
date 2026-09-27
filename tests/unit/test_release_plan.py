from __future__ import annotations

from types import SimpleNamespace
from typing import cast

import pytest

from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.release_plan import (
    DATASET_NAMES,
    plan_datasets,
    selected_dataset_names,
)


class _InventoryApi:
    def __init__(self) -> None:
        self.touched: list[str] = []

    def repo_info(self, repo_id: str, **_kwargs: object) -> SimpleNamespace:
        self.touched.append(repo_id)
        return SimpleNamespace(sha=f"revision-{repo_id.rsplit('-', 1)[-1]}")

    def list_repo_tree(self, repo_id: str, **_kwargs: object) -> list[SimpleNamespace]:
        self.touched.append(repo_id)
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
