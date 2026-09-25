"""Small, streaming adapters for the official EEA EUNIS reference assets."""

from __future__ import annotations

import hashlib
import io
import json
import re
import xml.etree.ElementTree as ET
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote, unquote, urlsplit, urlunsplit
from zipfile import BadZipFile, ZipFile

import httpx

from .fileio import write_chunks
from .reference import parse_layer_code

_USER_AGENT = "osm-polygon-eunis/" + ".".join(version("osm-polygon-eunis").split(".")[:2])

_SHARE_TOKEN = re.compile(
    r"name=[\"']sharingToken[\"']\s+value=[\"']([^\"']+)[\"']",
    re.IGNORECASE,
)
_DEFAULT_CATALOG_API = "https://sdi.eea.europa.eu/catalogue/datahub/api/records"
_DEFAULT_CLASSIFICATION_RECORD = "bfe4c237-e378-4a83-ab21-b3807f96c2e2"
_WEBDAV_DEPTH = "1"
_XLSX_MAIN_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


class _HttpResponse(Protocol):
    @property
    def content(self) -> bytes: ...

    @property
    def text(self) -> str: ...

    def json(self) -> Any: ...

    def raise_for_status(self) -> Any: ...


class _RequestClient(Protocol):
    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Any = None,
    ) -> _HttpResponse: ...


class _HttpClient(_RequestClient, Protocol):
    def get(self, url: str, *, params: Any = None) -> _HttpResponse: ...


@dataclass(frozen=True, slots=True)
class WebDavEntry:
    """One public WebDAV file or directory discovered from a PROPFIND."""

    path: str
    url: str
    size: int | None
    etag: str | None
    is_collection: bool


@dataclass(frozen=True, slots=True)
class RemoteAsset:
    """A verified-by-metadata remote reference asset."""

    path: str
    url: str
    size: int
    etag: str | None
    code: str | None
    name: str | None
    record_id: str
    source_version: str


@dataclass(frozen=True, slots=True)
class EeaGroup:
    """One EEA catalog record and its raster or GeoPackage assets."""

    record_id: str
    title: str
    folder_url: str
    service_url: str
    labels: Mapping[str, str]
    raster_assets: tuple[RemoteAsset, ...]
    vector_asset: RemoteAsset | None


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _child_text(element: ET.Element, name: str) -> str | None:
    for child in element.iter():
        if _local_name(child.tag) == name and child.text:
            return child.text.strip()
    return None


def parse_webdav_entries(payload: bytes) -> tuple[WebDavEntry, ...]:
    """Parse a public Nextcloud WebDAV directory listing."""

    root = ET.fromstring(payload)
    return tuple(
        _webdav_entry(response)
        for response in root.iter()
        if _local_name(response.tag) == "response"
    )


def _webdav_entry(response: ET.Element) -> WebDavEntry:
    path = _child_text(response, "href")
    if not path:
        raise ValueError("WebDAV response is missing href")
    size = _webdav_size(response)
    etag = _webdav_etag(response)
    is_collection = _is_collection(response)
    parts = urlsplit(path)
    url = path if parts.scheme else urlunsplit(("https", "sdi.eea.europa.eu", path, "", ""))
    return WebDavEntry(path, url, size, etag, is_collection)


def _webdav_size(response: ET.Element) -> int | None:
    size_text = _child_text(response, "getcontentlength")
    return int(size_text) if size_text else None


def _webdav_etag(response: ET.Element) -> str | None:
    etag = _child_text(response, "getetag")
    return etag.strip('"') if etag is not None else None


def _is_collection(response: ET.Element) -> bool:
    return any(_local_name(child.tag) == "collection" for child in response.iter())


def extract_share_token(html: str) -> str:
    """Extract the current public-share token from an EEA datastore page."""

    match = _SHARE_TOKEN.search(html)
    if match is None:
        raise ValueError("EEA datastore page has no public share token")
    return match.group(1)


def _text_value(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def parse_arcgis_labels(
    payload: Mapping[str, object],
    fallback_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Read EUNIS code/name attributes, falling back to the official table."""

    features = payload.get("features")
    if not isinstance(features, list):
        raise ValueError("EEA ImageServer response has no features list")
    labels: dict[str, str] = {}
    for feature in features:
        label = _arcgis_feature_label(feature, fallback_labels)
        if label is None:
            continue
        code, name = label
        previous = labels.setdefault(code, name)
        if previous != name:
            raise ValueError(f"EEA ImageServer has conflicting names for {code}")
    return labels


def _arcgis_feature_label(
    feature: object,
    fallback_labels: Mapping[str, str] | None,
) -> tuple[str, str] | None:
    attributes = _arcgis_attributes(feature)
    code = _arcgis_code(attributes)
    if code is None:
        return None
    name = _arcgis_name(attributes, code, fallback_labels)
    if not name:
        raise ValueError(f"EEA ImageServer feature has an incomplete label for {code}")
    return code, name


def _arcgis_attributes(feature: object) -> Mapping[str, object]:
    if not isinstance(feature, Mapping):
        raise ValueError("EEA ImageServer feature is not an object")
    attributes = feature.get("attributes")
    if not isinstance(attributes, Mapping):
        raise ValueError("EEA ImageServer feature has no attributes")
    return attributes


def _arcgis_code(attributes: Mapping[str, object]) -> str | None:
    code = attributes.get("Prob")
    if code is None:
        return None
    if not isinstance(code, str) or not code.strip():
        raise ValueError("EEA ImageServer feature has an incomplete label")
    return code.strip()


def _arcgis_name(
    attributes: Mapping[str, object],
    code: str,
    fallback_labels: Mapping[str, str] | None,
) -> str:
    for field in ("Name_", "Name_1", "Name1", "Name"):
        candidate = _text_value(attributes.get(field))
        if candidate and candidate.casefold() != f"{code}_prob".casefold():
            return candidate
    return _text_value(fallback_labels.get(code)) if fallback_labels else ""


def _normalise_header(value: object) -> str:
    return re.sub(r"[^a-z0-9]", "", _text_value(value).casefold())


def parse_classification_rows(rows: Iterable[Iterable[object]]) -> dict[str, str]:
    """Extract code/name pairs from the repeated headers of EEA worksheets."""

    labels: dict[str, str] = {}
    code_index: int | None = None
    name_index: int | None = None
    for raw_row in rows:
        row = tuple(raw_row)
        indexes = _classification_indexes(row)
        if indexes is not None:
            code_index, name_index = indexes
            continue
        label = _classification_label(row, code_index, name_index)
        if label is None:
            continue
        _merge_label(labels, *label, source="classification")
    if not labels:
        raise ValueError("EEA classification workbook has no code/name rows")
    return labels


def _merge_label(labels: dict[str, str], code: str, name: str, source: str) -> None:
    previous = labels.setdefault(code, name)
    if previous != name:
        raise ValueError(f"EEA {source} has conflicting names for {code}")


def _classification_indexes(row: tuple[object, ...]) -> tuple[int, int] | None:
    headers = {_normalise_header(value): index for index, value in enumerate(row)}
    code_index = _find_header_index(headers, {"code", "code2018"})
    name_index = _find_header_index(headers, {"name", "name2018"})
    return (code_index, name_index) if code_index is not None and name_index is not None else None


def _find_header_index(headers: Mapping[str, int], names: set[str]) -> int | None:
    return next((index for key, index in headers.items() if key in names), None)


def _classification_label(
    row: tuple[object, ...],
    code_index: int | None,
    name_index: int | None,
) -> tuple[str, str] | None:
    if code_index is None or name_index is None:
        return None
    return _classification_values(row, code_index, name_index)


def _classification_values(
    row: tuple[object, ...],
    code_index: int,
    name_index: int,
) -> tuple[str, str] | None:
    if max(code_index, name_index) >= len(row):
        return None
    code = _text_value(row[code_index])
    name = _text_value(row[name_index])
    return (code, name) if code and name else None


def _shared_strings(archive: ZipFile) -> tuple[str, ...]:
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return ()
    return tuple(_shared_string(item) for item in root.iter() if _local_name(item.tag) == "si")


def _shared_string(item: ET.Element) -> str:
    return "".join(element.text or "" for element in item.iter() if _local_name(element.tag) == "t")


def _column_index(reference: str) -> int:
    letters = re.match(r"[A-Z]+", reference)
    if letters is None:
        raise ValueError(f"invalid XLSX cell reference {reference!r}")
    index = 0
    for letter in letters.group(0):
        index = index * 26 + ord(letter) - ord("A") + 1
    return index - 1


def _xlsx_cell_value(cell: ET.Element, shared_strings: tuple[str, ...]) -> str:
    value = cell.find(f"{{{_XLSX_MAIN_NS}}}v")
    raw = "" if value is None or value.text is None else value.text
    cell_type = cell.attrib.get("t")
    if cell_type == "s":
        return _shared_string_value(shared_strings, raw)
    if cell_type == "inlineStr":
        return _inline_string(cell)
    return raw


def _shared_string_value(shared_strings: tuple[str, ...], raw: str) -> str:
    try:
        return shared_strings[int(raw)]
    except (IndexError, ValueError) as error:
        raise ValueError("XLSX cell references an invalid shared string") from error


def _inline_string(cell: ET.Element) -> str:
    return "".join(element.text or "" for element in cell.iter() if _local_name(element.tag) == "t")


def _xlsx_rows(archive: ZipFile, shared_strings: tuple[str, ...]) -> Iterable[tuple[str, ...]]:
    sheet_paths = sorted(
        path
        for path in archive.namelist()
        if path.startswith("xl/worksheets/sheet") and path.endswith(".xml")
    )
    for path in sheet_paths:
        root = ET.fromstring(archive.read(path))
        yield from _sheet_rows(root, shared_strings)


def _sheet_rows(
    root: ET.Element,
    shared_strings: tuple[str, ...],
) -> Iterable[tuple[str, ...]]:
    for row_element in root.iter():
        if _local_name(row_element.tag) != "row":
            continue
        cells = _row_cells(row_element, shared_strings)
        if cells:
            width = max(cells) + 1
            yield tuple(cells.get(index, "") for index in range(width))


def _row_cells(row_element: ET.Element, shared_strings: tuple[str, ...]) -> dict[int, str]:
    return {
        _column_index(cell.attrib["r"]): _xlsx_cell_value(cell, shared_strings)
        for cell in row_element
        if _local_name(cell.tag) == "c" and "r" in cell.attrib
    }


def parse_classification_workbook(payload: bytes) -> dict[str, str]:
    """Parse the official EEA XLSX classification without writing it to disk."""

    try:
        with ZipFile(io.BytesIO(payload)) as archive:
            rows = _xlsx_rows(archive, _shared_strings(archive))
            return parse_classification_rows(rows)
    except (BadZipFile, KeyError) as error:
        raise ValueError("EEA classification asset is not a valid XLSX workbook") from error


def raster_assets_from_entries(
    entries: Iterable[WebDavEntry],
    labels: Mapping[str, str],
    *,
    record_id: str,
    source_version: str,
) -> tuple[RemoteAsset, ...]:
    """Turn discovered GeoTIFF entries into labeled remote assets."""

    assets = [
        asset
        for entry in entries
        if (asset := _raster_asset(entry, labels, record_id, source_version)) is not None
    ]
    return tuple(sorted(assets, key=lambda asset: asset.path))


def _raster_asset(
    entry: WebDavEntry,
    labels: Mapping[str, str],
    record_id: str,
    source_version: str,
) -> RemoteAsset | None:
    if entry.is_collection or not entry.path.lower().endswith((".tif", ".tiff")):
        return None
    code = parse_layer_code(Path(entry.path).name)
    if code not in labels:
        raise ValueError(f"EEA service has no name for raster code {code}")
    if entry.size is None:
        raise ValueError(f"EEA raster has no byte size: {entry.path}")
    return RemoteAsset(
        path=entry.path,
        url=entry.url,
        size=entry.size,
        etag=entry.etag,
        code=code,
        name=labels[code],
        record_id=record_id,
        source_version=source_version,
    )


def _online_resources(value: object) -> Iterable[tuple[str | None, str]]:
    if isinstance(value, Mapping):
        yield from _online_mapping_resources(value)
        return
    if isinstance(value, list):
        yield from _online_list_resources(value)


def _online_mapping_resources(value: Mapping[object, object]) -> Iterable[tuple[str | None, str]]:
    resource = _online_resource(value)
    if resource is not None:
        yield resource
    for child in value.values():
        yield from _online_resources(child)


def _online_list_resources(value: list[object]) -> Iterable[tuple[str | None, str]]:
    for child in value:
        yield from _online_resources(child)


def _online_resource(value: Mapping[object, object]) -> tuple[str | None, str] | None:
    resource = value.get("cit:CI_OnlineResource")
    if not isinstance(resource, Mapping):
        return None
    linkage = resource.get("cit:linkage")
    protocol = resource.get("cit:protocol")
    link = _catalog_value(linkage)
    proto = _catalog_value(protocol)
    return (proto, link) if link else None


def _catalog_value(value: object) -> str | None:
    if not isinstance(value, Mapping):
        return value if isinstance(value, str) and value else None
    return _catalog_mapping_value(value)


def _catalog_mapping_value(value: Mapping[object, object]) -> str | None:
    for key in ("gco:CharacterString", "gcx:Anchor", "#text"):
        nested = _catalog_value(value.get(key))
        if nested:
            return nested
    return None


def _all_text(value: object) -> Iterable[str]:
    if isinstance(value, Mapping):
        yield from _mapping_text(value)
        return
    if isinstance(value, list):
        yield from _list_text(value)
        return
    if isinstance(value, str):
        yield value


def _mapping_text(value: Mapping[object, object]) -> Iterable[str]:
    for child in value.values():
        yield from _all_text(child)


def _list_text(value: list[object]) -> Iterable[str]:
    for child in value:
        yield from _all_text(child)


def catalog_links(record: Mapping[str, object]) -> tuple[str, str, str]:
    """Extract the title, public folder, and ArcGIS service from catalog JSON."""

    title = _catalog_title(record, "probability maps")
    folder = _resource_url(record, "EEA:FOLDERPATH")
    service = _resource_url(record, "ESRI:REST")
    return _validated_catalog_links(title, folder, service, record)


def _validated_catalog_links(
    title: str | None,
    folder: str | None,
    service: str | None,
    record: Mapping[str, object],
) -> tuple[str, str, str]:
    if not isinstance(title, str) or not folder or not service:
        raise ValueError("EEA catalog record lacks title, EPSG:3035, folder, or service")
    if not _has_equal_area_crs(record):
        raise ValueError("EEA catalog record lacks title, EPSG:3035, folder, or service")
    return title, folder, service


def _has_equal_area_crs(record: Mapping[str, object]) -> bool:
    return any(text.strip() == "EPSG:3035" for text in _all_text(record))


def _catalog_title(record: Mapping[str, object], phrase: str) -> str | None:
    return next((text for text in _all_text(record) if phrase in text.lower()), None)


def _resource_url(record: Mapping[str, object], protocol: str) -> str | None:
    return next(
        (
            url
            for candidate_protocol, url in _online_resources(record)
            if candidate_protocol == protocol
        ),
        None,
    )


def _webdav_folder_url(folder_url: str, token: str) -> str:
    parsed = urlsplit(folder_url)
    marker = "/public/"
    if marker not in parsed.path:
        raise ValueError(f"unsupported EEA folder URL {folder_url}")
    relative = parsed.path.split(marker, 1)[1].strip("/")
    encoded = quote(relative, safe="/")
    return (
        f"{parsed.scheme}://{parsed.netloc}/datashare/public.php/dav/files/"
        f"{quote(token, safe='')}/{encoded}/"
    )


def _discover_entries(
    client: _RequestClient,
    root_url: str,
    *,
    max_depth: int = 3,
    file_suffixes: tuple[str, ...] = (".tif", ".tiff", ".gpkg"),
) -> tuple[WebDavEntry, ...]:
    queue: list[tuple[str, int]] = [(root_url, 0)]
    seen: set[str] = set()
    files: list[WebDavEntry] = []
    while queue:
        url, depth = queue.pop(0)
        if not _should_visit(url, depth, seen, max_depth):
            continue
        seen.add(url)
        response = client.request("PROPFIND", url, headers={"Depth": _WEBDAV_DEPTH})
        response.raise_for_status()
        for entry in parse_webdav_entries(response.content):
            _collect_entry(entry, url, depth, max_depth, file_suffixes, queue, files)
    return tuple(sorted(files, key=lambda entry: entry.path))


def _should_visit(url: str, depth: int, seen: set[str], max_depth: int) -> bool:
    return url not in seen and depth <= max_depth


def _collect_entry(
    entry: WebDavEntry,
    parent_url: str,
    depth: int,
    max_depth: int,
    file_suffixes: tuple[str, ...],
    queue: list[tuple[str, int]],
    files: list[WebDavEntry],
) -> None:
    if entry.url.rstrip("/") == parent_url.rstrip("/"):
        return
    if entry.is_collection:
        if depth < max_depth:
            queue.append((entry.url.rstrip("/") + "/", depth + 1))
        return
    if entry.path.lower().endswith(file_suffixes):
        files.append(entry)


def _fetch_arcgis_labels(
    client: _HttpClient,
    service_url: str,
    fallback_labels: Mapping[str, str] | None = None,
) -> dict[str, str]:
    labels = dict(fallback_labels or {})
    offset = 0
    while True:
        page = _fetch_arcgis_page(client, service_url, offset)
        page_labels = parse_arcgis_labels(page, fallback_labels)
        _merge_labels(labels, page_labels)
        if not page.get("exceededTransferLimit") or not page_labels:
            return labels
        offset += len(page_labels)


def _fetch_arcgis_page(client: _HttpClient, service_url: str, offset: int) -> Mapping[str, object]:
    response = client.get(
        f"{service_url.rstrip('/')}/query",
        params={
            "where": "1=1",
            "outFields": "*",
            "returnGeometry": "false",
            "resultRecordCount": 1000,
            "resultOffset": offset,
            "f": "json",
        },
    )
    response.raise_for_status()
    page = response.json()
    if not isinstance(page, Mapping):
        raise ValueError("EEA ImageServer response is not an object")
    return page


def _merge_labels(labels: dict[str, str], additions: Mapping[str, str]) -> None:
    for code, name in additions.items():
        _merge_label(labels, code, name, source="ImageServer")


def _classification_catalog_links(record: Mapping[str, object]) -> tuple[str, str]:
    title = _catalog_title(record, "eunis terrestrial habitat classification")
    folder = _resource_url(record, "EEA:FOLDERPATH")
    if not isinstance(title, str) or not folder:
        raise ValueError("EEA classification record lacks title or public folder")
    return title, folder


def _select_classification_entry(entries: Iterable[WebDavEntry]) -> WebDavEntry:
    entries = tuple(entries)
    preferred = tuple(
        entry for entry in entries if "including crosswalks" in unquote(entry.path).lower()
    )
    candidates = preferred or entries
    if len(candidates) != 1:
        raise ValueError("EEA classification folder does not have one unambiguous XLSX")
    return candidates[0]


def resolve_classification_labels(
    client: _HttpClient,
    record_id: str = _DEFAULT_CLASSIFICATION_RECORD,
    *,
    catalog_api: str = _DEFAULT_CATALOG_API,
) -> dict[str, str]:
    """Fetch and parse the small authoritative 2021 EUNIS table in memory."""

    response = client.get(f"{catalog_api.rstrip('/')}/{record_id}?language=eng")
    response.raise_for_status()
    _title, folder_url = _classification_catalog_links(response.json())
    share_page = client.get(folder_url)
    share_page.raise_for_status()
    token = extract_share_token(share_page.text)
    entries = _discover_entries(
        client,
        _webdav_folder_url(folder_url, token),
        max_depth=1,
        file_suffixes=(".xlsx",),
    )
    entry = _select_classification_entry(entries)
    if entry.size is None:
        raise ValueError("EEA classification workbook has no byte size")
    workbook = client.get(entry.url)
    workbook.raise_for_status()
    if len(workbook.content) != entry.size:
        raise ValueError("EEA classification workbook byte count differs from metadata")
    return parse_classification_workbook(workbook.content)


def resolve_group(
    client: _HttpClient,
    record_id: str,
    *,
    source_version: str,
    fallback_labels: Mapping[str, str] | None = None,
    catalog_api: str = _DEFAULT_CATALOG_API,
) -> EeaGroup:
    """Resolve one official catalog record without downloading its data."""

    response = client.get(f"{catalog_api.rstrip('/')}/{record_id}?language=eng")
    response.raise_for_status()
    title, folder_url, service_url = catalog_links(response.json())
    share_page = client.get(folder_url)
    share_page.raise_for_status()
    token = extract_share_token(share_page.text)
    entries = _discover_entries(client, _webdav_folder_url(folder_url, token))
    labels = _fetch_arcgis_labels(client, service_url, fallback_labels)
    raster_assets = raster_assets_from_entries(
        entries,
        labels,
        record_id=record_id,
        source_version=source_version,
    )
    vector_asset = _vector_asset(entries, record_id, source_version)
    if not raster_assets and vector_asset is None:
        raise ValueError(f"EEA record has no GeoTIFF or GeoPackage assets: {record_id}")
    return EeaGroup(record_id, title, folder_url, service_url, labels, raster_assets, vector_asset)


def _vector_asset(
    entries: Iterable[WebDavEntry],
    record_id: str,
    source_version: str,
) -> RemoteAsset | None:
    vector_entries = _vector_entries(entries)
    if len(vector_entries) > 1:
        raise ValueError(f"EEA record has multiple GeoPackages: {record_id}")
    if not vector_entries:
        return None
    return _build_vector_asset(vector_entries[0], record_id, source_version)


def _vector_entries(entries: Iterable[WebDavEntry]) -> tuple[WebDavEntry, ...]:
    return tuple(entry for entry in entries if entry.path.lower().endswith(".gpkg"))


def _build_vector_asset(
    entry: WebDavEntry,
    record_id: str,
    source_version: str,
) -> RemoteAsset:
    if entry.size is None:
        raise ValueError(f"EEA GeoPackage has no byte size: {entry.path}")
    return RemoteAsset(
        path=entry.path,
        url=entry.url,
        size=entry.size,
        etag=entry.etag,
        code=None,
        name=None,
        record_id=record_id,
        source_version=source_version,
    )


def resolve_config(config_path: Path) -> tuple[EeaGroup, ...]:
    """Resolve all configured EEA records using one bounded HTTP client."""

    config = json.loads(config_path.read_text(encoding="utf-8"))
    settings = _config_settings(config)
    with httpx.Client(
        headers={"Accept": "application/json", "User-Agent": _USER_AGENT},
        follow_redirects=True,
        timeout=60,
    ) as client:
        labels = resolve_classification_labels(client, settings.classification_record)
        labels.update(settings.supplemental_labels)
        return _resolve_groups(client, settings, labels)


@dataclass(frozen=True, slots=True)
class _ConfigSettings:
    source_version: str
    record_ids: tuple[str, ...]
    classification_record: str
    supplemental_labels: dict[str, str]


def _config_settings(config: object) -> _ConfigSettings:
    config = _config_mapping(config)
    source_version = _required_config_string(config.get("source_version"), "source_version")
    record_ids = _record_ids(config.get("catalog_records"))
    classification_record = _required_config_string(
        config.get("classification_record", _DEFAULT_CLASSIFICATION_RECORD),
        "classification_record",
    )
    supplemental_labels = config.get("supplemental_labels", {})
    if not isinstance(supplemental_labels, Mapping):
        raise ValueError("EEA config has invalid supplemental_labels")
    labels = _supplemental_labels(supplemental_labels)
    return _ConfigSettings(source_version, record_ids, classification_record, labels)


def _config_mapping(config: object) -> Mapping[object, object]:
    if not isinstance(config, Mapping):
        raise ValueError("EEA config must be a JSON object")
    return config


def _required_config_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"EEA config has an invalid {field}")
    return value


def _record_ids(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(record_id, str) for record_id in value):
        raise ValueError("EEA config is missing catalog_records")
    return tuple(value)


def _supplemental_labels(value: Mapping[object, object]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for code, name in value.items():
        if not isinstance(code, str) or not isinstance(name, str) or not name.strip():
            raise ValueError("EEA config has an invalid supplemental label")
        labels[code] = name.strip()
    return labels


def _resolve_groups(
    client: _HttpClient,
    settings: _ConfigSettings,
    labels: Mapping[str, str],
) -> tuple[EeaGroup, ...]:
    return tuple(
        resolve_group(
            client,
            record_id,
            source_version=settings.source_version,
            fallback_labels=labels,
        )
        for record_id in settings.record_ids
    )


def download_asset(client: httpx.Client, asset: RemoteAsset, destination: Path) -> str:
    """Stream one EEA asset and return its SHA-256 after size validation."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with client.stream("GET", asset.url) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            written = write_chunks(response.iter_bytes(chunk_size=8 * 1024 * 1024), output, digest)
    if written != asset.size:
        destination.unlink(missing_ok=True)
        raise ValueError(f"EEA asset byte count {written} does not match metadata {asset.size}")
    return digest.hexdigest()
