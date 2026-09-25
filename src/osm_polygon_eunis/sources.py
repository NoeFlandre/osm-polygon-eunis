"""Hugging Face source inventories and storage-bounded shard downloads."""

from __future__ import annotations

import fnmatch
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, Self

import httpx
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import build_hf_headers

from .fileio import write_chunks


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Source/output IDs and the only paths allowed to change."""

    name: str
    source_repo: str
    output_repo: str
    geometry_glob: str
    link_glob: str | None = None


_SPECS: Mapping[str, DatasetSpec] = {
    "website": DatasetSpec(
        "website",
        "NoeFlandre/osm-polygon-website-tag",
        "NoeFlandre/osm-polygon-website-tag-eunis",
        "polygons/*.parquet",
    ),
    "description": DatasetSpec(
        "description",
        "NoeFlandre/osm-polygon-description-tag",
        "NoeFlandre/osm-polygon-description-tag-eunis",
        "data/*.parquet",
    ),
    "wikidata": DatasetSpec(
        "wikidata",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia",
        "NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis",
        "polygons/*.parquet",
        "polygon_document_links/*.parquet",
    ),
}


class _InventoryApi(Protocol):
    def repo_info(
        self,
        repo_id: str,
        *,
        revision: str | None = None,
        repo_type: str | None = None,
    ): ...

    def list_repo_tree(
        self,
        repo_id: str,
        path_in_repo: str | None = None,
        *,
        recursive: bool = False,
        revision: str | None = None,
        repo_type: str | None = None,
    ) -> Iterable[Any]: ...


class _StreamResponse(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, *args: object) -> bool | None: ...

    def raise_for_status(self) -> Any: ...

    def iter_bytes(self, *, chunk_size: int) -> Iterable[bytes]: ...


class _StreamClient(Protocol):
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str],
        follow_redirects: bool,
        timeout: None,
    ) -> _StreamResponse: ...


def dataset_spec(name: str) -> DatasetSpec:
    """Return a known source layout by its short name."""

    try:
        return _SPECS[name]
    except KeyError as error:
        raise ValueError(f"unknown source {name!r}") from error


def capture_revision(api: _InventoryApi, repo_id: str) -> str:
    """Capture the immutable Hub commit used for a run."""

    info = api.repo_info(repo_id, repo_type="dataset")
    revision = getattr(info, "sha", None)
    if not isinstance(revision, str) or not revision:
        raise ValueError(f"Hub did not return a commit revision for {repo_id}")
    return revision


def list_parquet_files(api: _InventoryApi, repo_id: str, revision: str) -> tuple[str, ...]:
    """List remote Parquet files in deterministic order."""

    entries = api.list_repo_tree(
        repo_id,
        path_in_repo="",
        recursive=True,
        revision=revision,
        repo_type="dataset",
    )
    paths = {
        path
        for entry in entries
        if not hasattr(entry, "tree_id")
        and isinstance((path := getattr(entry, "path", None)), str)
        and path.endswith(".parquet")
    }
    return tuple(sorted(paths))


def list_repo_files(api: _InventoryApi, repo_id: str, revision: str) -> tuple[Any, ...]:
    """List file entries in a pinned Hub revision, excluding directory entries."""

    entries = api.list_repo_tree(
        repo_id,
        path_in_repo="",
        recursive=True,
        revision=revision,
        repo_type="dataset",
    )
    return tuple(
        sorted(
            (
                entry
                for entry in entries
                if not hasattr(entry, "tree_id") and isinstance(getattr(entry, "path", None), str)
            ),
            key=lambda entry: entry.path,
        )
    )


def _index_by_filename(paths: Iterable[str], kind: str) -> dict[str, str]:
    indexed: dict[str, str] = {}
    for path in paths:
        filename = Path(path).name
        if filename in indexed:
            raise ValueError(f"duplicate {kind} shard filename {filename!r}")
        indexed[filename] = path
    return indexed


def pair_region_paths(
    polygon_paths: Iterable[str],
    link_paths: Iterable[str],
) -> tuple[tuple[str, str], ...]:
    """Pair Wikidata polygon/link shards by their exact region filename."""

    polygons = _index_by_filename(polygon_paths, "polygon")
    links = _index_by_filename(link_paths, "link")
    if set(polygons) != set(links):
        missing_links = sorted(set(polygons) - set(links))
        missing_polygons = sorted(set(links) - set(polygons))
        raise ValueError(
            f"unmatched region shards: missing links={missing_links}, "
            f"missing polygons={missing_polygons}",
        )
    return tuple((polygons[name], links[name]) for name in sorted(polygons))


def flatten_repo_path(path: str) -> str:
    """Turn a repository path into a flat local filename."""

    return path.replace("/", "__")


def download_to_temp(
    api: HfApi,
    repo_id: str,
    path: str,
    revision: str,
    directory: Path,
    *,
    client: _StreamClient | None = None,
) -> Path:
    """Stream one Hub file to a run-local path and verify Content-Length."""

    directory.mkdir(parents=True, exist_ok=True)
    destination = directory / flatten_repo_path(path)
    url = hf_hub_url(
        repo_id,
        path,
        revision=revision,
        repo_type="dataset",
        endpoint=api.endpoint,
    )
    headers = build_hf_headers(token=api.token)
    if client is None:
        with httpx.stream(
            "GET",
            url,
            headers=headers,
            follow_redirects=True,
            timeout=None,
        ) as response:
            written, expected = _write_response(response, destination)
    else:
        written, expected = _download_with_client(client, url, headers, destination)
    if expected is not None and written != int(expected):
        destination.unlink(missing_ok=True)
        raise ValueError(
            f"downloaded byte count {written} does not match Content-Length {expected}",
        )
    return destination


def _download_with_client(
    client: _StreamClient,
    url: str,
    headers: Mapping[str, str],
    destination: Path,
) -> tuple[int, str | None]:
    with client.stream(
        "GET",
        url,
        headers=headers,
        follow_redirects=True,
        timeout=None,
    ) as response:
        return _write_response(response, destination)


def _write_response(response: Any, destination: Path) -> tuple[int, str | None]:
    response.raise_for_status()
    with destination.open("wb") as output:
        written = write_chunks(response.iter_bytes(chunk_size=8 * 1024 * 1024), output)
    return written, response.headers.get("content-length")


def matches_layout(path: str, pattern: str) -> bool:
    """Check a remote path against a declared one-level shard layout."""

    return fnmatch.fnmatchcase(path, pattern)
