from pathlib import Path
from types import SimpleNamespace
from typing import cast

import httpx
import pytest
from huggingface_hub import HfApi

from osm_polygon_eunis import sources
from osm_polygon_eunis._protocols import HubApi
from osm_polygon_eunis.fileio import DOWNLOAD_TIMEOUT
from osm_polygon_eunis.sources import (
    capture_revision,
    dataset_spec,
    download_to_temp,
    list_parquet_files,
    list_repo_files,
    pair_region_paths,
)


def test_source_layouts_match_the_public_repositories() -> None:
    assert dataset_spec("website").geometry_glob == "polygons/*.parquet"
    assert dataset_spec("description").geometry_glob == "data/*.parquet"
    assert dataset_spec("wikidata").geometry_glob == "polygons/*.parquet"
    assert dataset_spec("wikidata").link_glob == "polygon_document_links/*.parquet"


def test_unknown_source_name_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown source"):
        dataset_spec("unknown")


def test_capture_revision_and_list_parquet_files() -> None:
    class FakeApi:
        def repo_info(self, *_args, **_kwargs):
            return SimpleNamespace(sha="source-sha")

        def list_repo_tree(self, *_args, **_kwargs):
            return iter(
                (
                    SimpleNamespace(type="file", path="polygons/z.parquet"),
                    SimpleNamespace(path="polygons/y.parquet", blob_id="y"),
                    SimpleNamespace(type="file", path="README.md"),
                    SimpleNamespace(type="file", path="polygons/a.parquet"),
                )
            )

    api = FakeApi()

    assert capture_revision(api, "org/source") == "source-sha"
    assert list_parquet_files(api, "org/source", "source-sha") == (
        "polygons/a.parquet",
        "polygons/y.parquet",
        "polygons/z.parquet",
    )
    assert [entry.path for entry in list_repo_files(api, "org/source", "source-sha")] == [
        "README.md",
        "polygons/a.parquet",
        "polygons/y.parquet",
        "polygons/z.parquet",
    ]


def test_region_paths_pair_by_filename() -> None:
    pairs = pair_region_paths(
        ["polygons/france-latest.parquet", "polygons/spain-latest.parquet"],
        [
            "polygon_document_links/spain-latest.parquet",
            "polygon_document_links/france-latest.parquet",
        ],
    )

    assert pairs == (
        (
            "polygons/france-latest.parquet",
            "polygon_document_links/france-latest.parquet",
        ),
        (
            "polygons/spain-latest.parquet",
            "polygon_document_links/spain-latest.parquet",
        ),
    )


def test_unmatched_link_path_fails_closed() -> None:
    with pytest.raises(ValueError, match="unmatched"):
        pair_region_paths(
            ["polygons/france-latest.parquet"],
            ["polygon_document_links/spain-latest.parquet"],
        )


def test_download_to_temp_streams_and_verifies_length(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        @property
        def headers(self):
            return {"content-length": "6"}

        def iter_bytes(self, chunk_size: int):
            assert chunk_size > 0
            yield b"abc"
            yield b"def"

    monkeypatch.setattr(sources, "download_client", lambda: _ScriptedClient(FakeResponse()))
    local_path = download_to_temp(
        cast(HubApi, HfApi(token="hf-test")),
        "org/source",
        "polygons/france-latest.parquet",
        "source-sha",
        tmp_path,
    )

    assert local_path.read_bytes() == b"abcdef"


def test_download_to_temp_rejects_wrong_length(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        @property
        def headers(self):
            return {"content-length": "7"}

        # Signature mirrors the StreamResponse protocol.
        def iter_bytes(self, chunk_size: int):  # noqa: ARG002
            yield b"abc"

    monkeypatch.setattr(sources, "download_client", lambda: _ScriptedClient(FakeResponse()))

    with pytest.raises(ValueError, match="byte count"):
        download_to_temp(
            cast(HubApi, HfApi(token="hf-test")),
            "org/source",
            "polygons/france-latest.parquet",
            "source-sha",
            tmp_path,
        )


def test_download_to_temp_reuses_supplied_http_client(tmp_path: Path, monkeypatch) -> None:
    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def raise_for_status(self) -> None:
            return None

        @property
        def headers(self):
            return {"content-length": "3"}

        def iter_bytes(self, chunk_size: int):
            assert chunk_size == 8 * 1024 * 1024
            yield b"abc"

    class FakeClient:
        def __init__(self) -> None:
            self.calls = 0

        def stream(self, *_args, **_kwargs):
            self.calls += 1
            return FakeResponse()

    def unexpected_module_stream(*_args, **_kwargs):
        raise AssertionError("the supplied client must be used")

    monkeypatch.setattr(httpx, "stream", unexpected_module_stream)
    client = FakeClient()

    local_path = download_to_temp(
        cast(HubApi, HfApi(token="hf-test")),
        "org/source",
        "polygons/france-latest.parquet",
        "source-sha",
        tmp_path,
        client=client,
    )

    assert client.calls == 1
    assert local_path.read_bytes() == b"abc"


class _ScriptedResponse:
    """Response whose body yields ``chunks`` and then optionally raises ``error``."""

    def __init__(self, chunks: list[bytes], content_length: str, error: Exception | None = None):
        self._chunks = chunks
        self._content_length = content_length
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def raise_for_status(self) -> None:
        return None

    @property
    def headers(self):
        return {"content-length": self._content_length}

    def iter_bytes(self, chunk_size: int):  # noqa: ARG002
        yield from self._chunks
        if self._error is not None:
            raise self._error


class _ScriptedClient:
    def __init__(self, response: object) -> None:
        self.response = response
        self.timeouts: list[object] = []

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def stream(self, *_args, timeout=None, **_kwargs):
        self.timeouts.append(timeout)
        return self.response


def _download(tmp_path: Path, client=None) -> Path:
    return download_to_temp(
        cast(HubApi, HfApi(token="hf-test")),
        "org/source",
        "polygons/france-latest.parquet",
        "source-sha",
        tmp_path,
        client=client,
    )


def test_shared_client_download_applies_bounded_timeout(tmp_path: Path) -> None:
    client = _ScriptedClient(_ScriptedResponse([b"abc"], content_length="3"))

    _download(tmp_path, client=client)

    assert client.timeouts == [DOWNLOAD_TIMEOUT]


def test_download_leaves_only_final_file_on_success(tmp_path: Path) -> None:
    client = _ScriptedClient(_ScriptedResponse([b"abc"], content_length="3"))

    local_path = _download(tmp_path, client=client)

    assert sorted(entry.name for entry in tmp_path.iterdir()) == [local_path.name]


@pytest.mark.parametrize("use_shared_client", [False, True])
def test_download_removes_partial_file_when_stream_breaks(
    tmp_path: Path, monkeypatch, use_shared_client: bool
) -> None:
    response = _ScriptedResponse(
        [b"abc"],
        content_length="6",
        error=httpx.ReadError("connection reset"),
    )
    if use_shared_client:
        client = _ScriptedClient(response)
    else:
        monkeypatch.setattr(sources, "download_client", lambda: _ScriptedClient(response))
        client = None

    with pytest.raises(httpx.ReadError):
        _download(tmp_path, client=client)

    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("use_shared_client", [False, True])
def test_download_removes_partial_file_on_length_mismatch(
    tmp_path: Path, monkeypatch, use_shared_client: bool
) -> None:
    response = _ScriptedResponse([b"abc"], content_length="7")
    if use_shared_client:
        client = _ScriptedClient(response)
    else:
        monkeypatch.setattr(sources, "download_client", lambda: _ScriptedClient(response))
        client = None

    with pytest.raises(ValueError, match="byte count"):
        _download(tmp_path, client=client)

    assert list(tmp_path.iterdir()) == []


def test_module_download_applies_bounded_timeout(tmp_path: Path, monkeypatch) -> None:
    client = _ScriptedClient(_ScriptedResponse([b"abc"], content_length="3"))
    monkeypatch.setattr(sources, "download_client", lambda: client)

    _download(tmp_path)

    assert client.timeouts == [DOWNLOAD_TIMEOUT]
