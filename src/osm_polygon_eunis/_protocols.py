"""Structural types for the Hub API and HTTP clients shared across modules."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, BinaryIO, Protocol


class HttpResponse(Protocol):
    @property
    def content(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...

    def raise_for_status(self) -> Any: ...


class RequestClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Any = None,
    ) -> HttpResponse: ...


class HttpClient(RequestClient, Protocol):
    def get(self, url: str, *, params: Any = None) -> HttpResponse: ...


class StreamResponse(Protocol):
    @property
    def headers(self) -> Mapping[str, str]: ...

    def raise_for_status(self) -> Any: ...

    def iter_bytes(self, *, chunk_size: int) -> Iterable[bytes]: ...


class StreamClient(Protocol):
    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] = ...,
        follow_redirects: bool = ...,
        timeout: Any = ...,
    ) -> AbstractContextManager[StreamResponse]: ...


class InventoryApi(Protocol):
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


class HubApi(InventoryApi, Protocol):
    endpoint: str
    token: str | bool | None

    def upload_file(
        self,
        *,
        path_or_fileobj: str | Path | bytes | BinaryIO,
        path_in_repo: str,
        repo_id: str,
        repo_type: str | None = None,
        revision: str | None = None,
        commit_message: str | None = None,
        parent_commit: str | None = None,
    ) -> Any: ...
