from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import cast

import httpx
import pytest

import osm_polygon_eunis.eea as eea
from osm_polygon_eunis.eea import (
    RemoteAsset,
    WebDavEntry,
    _classification_catalog_links,
    _discover_entries,
    _fetch_arcgis_labels,
    _select_classification_entry,
    _vector_asset,
    _webdav_folder_url,
    catalog_links,
    download_asset,
    extract_share_token,
    parse_arcgis_labels,
    parse_classification_rows,
    parse_classification_workbook,
    parse_webdav_entries,
    raster_assets_from_entries,
    resolve_classification_labels,
    resolve_group,
)


def test_parse_arcgis_labels_uses_official_probability_fields() -> None:
    payload = {
        "features": [
            {"attributes": {"Prob": "R11", "Name_": "Pannonian and Pontic sandy steppe"}},
            {"attributes": {"Prob": "R12", "Name_": "Rocky grassland"}},
        ]
    }

    assert parse_arcgis_labels(payload) == {
        "R11": "Pannonian and Pontic sandy steppe",
        "R12": "Rocky grassland",
    }


def test_parse_arcgis_labels_rejects_missing_names() -> None:
    with pytest.raises(ValueError, match="label"):
        parse_arcgis_labels({"features": [{"attributes": {"Prob": "R11", "Name_": ""}}]})


def test_parse_arcgis_labels_uses_classification_fallback_for_code_only_service() -> None:
    payload = {"features": [{"attributes": {"Prob": "N11", "Name": "N11_Prob"}}]}

    assert parse_arcgis_labels(payload, {"N11": "Atlantic sand beach"}) == {
        "N11": "Atlantic sand beach"
    }


def test_parse_arcgis_labels_accepts_service_aliases_and_skips_non_habitat_rows() -> None:
    payload = {
        "features": [
            {"attributes": {"Prob": "MA221", "Name_1": "Atlantic saltmarsh driftlines"}},
            {"attributes": {"Prob": "U3A", "Name1": "Temperate inland cliff"}},
            {"attributes": {"Prob": None, "Name": "footprint"}},
        ]
    }

    assert parse_arcgis_labels(payload) == {
        "MA221": "Atlantic saltmarsh driftlines",
        "U3A": "Temperate inland cliff",
    }


def test_parse_classification_rows_accepts_multiple_sheet_headers() -> None:
    rows = [
        ["Level", "Code ", "Name "],
        ["1", "N", "Coastal habitats"],
        ["Level", "Code 2018", "Name 2018"],
        ["1", "V", "Vegetated man-made habitats"],
    ]

    assert parse_classification_rows(rows) == {
        "N": "Coastal habitats",
        "V": "Vegetated man-made habitats",
    }


def test_parse_classification_workbook_reads_shared_strings() -> None:
    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<si><t>Code </t></si><si><t>Name </t></si>"
            "<si><t>R11</t></si><si><t>Pannonian steppe</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c>'
            '<c r="B2" t="s"><v>3</v></c></row></sheetData></worksheet>',
        )

    assert parse_classification_workbook(workbook.getvalue()) == {"R11": "Pannonian steppe"}


def test_classification_entry_selection_decodes_webdav_path() -> None:
    entries = (
        WebDavEntry("/other.xlsx", "https://example.test/other.xlsx", 1, None, False),
        WebDavEntry(
            "/EUNIS%20including%20crosswalks.xlsx",
            "https://example.test/preferred.xlsx",
            1,
            None,
            False,
        ),
        WebDavEntry("/crosswalks.xlsx", "https://example.test/crosswalks.xlsx", 1, None, False),
    )

    assert _select_classification_entry(entries).url.endswith("preferred.xlsx")


def test_catalog_links_extracts_nested_online_resources() -> None:
    record = {
        "title": "Probability maps",
        "crs": "EPSG:3035",
        "resources": [
            {
                "cit:CI_OnlineResource": {
                    "cit:linkage": {"gco:CharacterString": "https://example.test/folder"},
                    "cit:protocol": {"gco:CharacterString": "EEA:FOLDERPATH"},
                }
            },
            {
                "cit:CI_OnlineResource": {
                    "cit:linkage": {"gco:CharacterString": "https://example.test/service"},
                    "cit:protocol": {"gco:CharacterString": "ESRI:REST"},
                }
            },
        ],
    }

    assert catalog_links(record) == (
        "Probability maps",
        "https://example.test/folder",
        "https://example.test/service",
    )
    with pytest.raises(ValueError, match="catalog record"):
        catalog_links({})


def test_webdav_discovery_walks_nested_directories() -> None:
    root_listing = b"""<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response><d:href>/root/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
      <d:response><d:href>/root/nested/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
      <d:response><d:href>/root/top.tif</d:href><d:propstat><d:prop><d:getcontentlength>1</d:getcontentlength></d:prop></d:propstat></d:response>
    </d:multistatus>"""
    nested_listing = b"""<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response><d:href>/root/nested/</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
      <d:response><d:href>/root/nested/deep.tif</d:href><d:propstat><d:prop><d:getcontentlength>2</d:getcontentlength></d:prop></d:propstat></d:response>
    </d:multistatus>"""

    class Response:
        def __init__(self, content: bytes) -> None:
            self.content = content

        def raise_for_status(self) -> None:
            return None

    class Client:
        def request(self, method, url, **kwargs):
            del method, kwargs
            return Response(root_listing if url.endswith("/root/") else nested_listing)

    entries = _discover_entries(Client(), "https://sdi.eea.europa.eu/root/", max_depth=1)

    assert len(entries) == 2
    assert {entry.path for entry in entries} == {
        "/root/top.tif",
        "/root/nested/deep.tif",
    }


def test_webdav_folder_url_requires_public_path() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        _webdav_folder_url("https://example.test/private/folder", "token")


class _FakeResponse:
    def __init__(self, *, payload=None, text="", content=b"") -> None:
        self._payload = payload
        self.text = text
        self.content = content

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        return None


class _FakeHttpClient:
    def __init__(self, responses) -> None:
        self.responses = responses

    def get(self, url, **kwargs):
        del kwargs
        return self.responses[url]

    def request(self, method, url, **kwargs):
        del method, kwargs
        return self.responses[url]


def _catalog_record(folder: str, service: str) -> dict[str, object]:
    return {
        "title": "EEA probability maps",
        "crs": "EPSG:3035",
        "resources": [
            {
                "cit:CI_OnlineResource": {
                    "cit:linkage": {"gco:CharacterString": folder},
                    "cit:protocol": {"gco:CharacterString": "EEA:FOLDERPATH"},
                }
            },
            {
                "cit:CI_OnlineResource": {
                    "cit:linkage": {"gco:CharacterString": service},
                    "cit:protocol": {"gco:CharacterString": "ESRI:REST"},
                }
            },
        ],
    }


def _webdav_listing(filename: str, size: int = 1) -> bytes:
    return _webdav_listing_at(filename, size=size, root="/root/")


def _webdav_listing_at(filename: str, *, size: int, root: str) -> bytes:
    root = "/" + root.strip("/") + "/"
    return f"""<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response><d:href>{root}</d:href><d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat></d:response>
      <d:response><d:href>{root}{filename}</d:href><d:propstat><d:prop><d:getcontentlength>{size}</d:getcontentlength></d:prop></d:propstat></d:response>
    </d:multistatus>""".encode()


def test_resolve_group_uses_fallback_labels_for_code_only_service() -> None:
    folder = "https://sdi.eea.europa.eu/webdav/public/group"
    service = "https://example.test/service"
    catalog_url = "https://catalog.test/records/record?language=eng"
    dav_url = "https://sdi.eea.europa.eu/datashare/public.php/dav/files/token/group/"
    responses = {
        catalog_url: _FakeResponse(payload=_catalog_record(folder, service)),
        folder: _FakeResponse(text='<input name="sharingToken" value="token">'),
        dav_url: _FakeResponse(
            content=_webdav_listing_at(
                "Prob_N11_100m.tif",
                size=1,
                root="/datashare/public.php/dav/files/token/group/",
            )
        ),
        f"{service}/query": _FakeResponse(
            payload={"features": [{"attributes": {"Prob": "N11", "Name": "N11_Prob"}}]}
        ),
    }
    client = _FakeHttpClient(responses)

    group = resolve_group(
        client,
        "record",
        source_version="EEA-test",
        fallback_labels={"N11": "coastal"},
        catalog_api="https://catalog.test/records",
    )

    assert group.raster_assets[0].name == "coastal"


def test_resolve_classification_labels_reads_one_workbook() -> None:
    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<si><t>Code </t></si><si><t>Name </t></si><si><t>R11</t></si>"
            "<si><t>Steppe</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>2</v></c>'
            '<c r="B2" t="s"><v>3</v></c></row></sheetData></worksheet>',
        )
    payload = workbook.getvalue()
    folder = "https://sdi.eea.europa.eu/webdav/public/classification"
    catalog_url = "https://catalog.test/records/classification?language=eng"
    dav_url = "https://sdi.eea.europa.eu/datashare/public.php/dav/files/token/classification/"
    responses = {
        catalog_url: _FakeResponse(
            payload={
                "title": "EUNIS terrestrial habitat classification review",
                "resources": [
                    {
                        "cit:CI_OnlineResource": {
                            "cit:linkage": {"gco:CharacterString": folder},
                            "cit:protocol": {"gco:CharacterString": "EEA:FOLDERPATH"},
                        }
                    }
                ],
            }
        ),
        folder: _FakeResponse(text='<input name="sharingToken" value="token">'),
        dav_url: _FakeResponse(
            content=_webdav_listing_at(
                "EUNIS%20including%20crosswalks.xlsx",
                size=len(payload),
                root="/datashare/public.php/dav/files/token/classification/",
            )
        ),
        (
            "https://sdi.eea.europa.eu/datashare/public.php/dav/files/token/"
            "classification/EUNIS%20including%20crosswalks.xlsx"
        ): _FakeResponse(content=payload),
    }
    client = _FakeHttpClient(responses)

    labels = resolve_classification_labels(
        client,
        "classification",
        catalog_api="https://catalog.test/records",
    )

    assert labels == {"R11": "Steppe"}


def test_parse_webdav_entries_extracts_size_etag_and_collection() -> None:
    xml = """<?xml version="1.0"?>
    <d:multistatus xmlns:d="DAV:">
      <d:response>
        <d:href>/dav/root/</d:href>
        <d:propstat><d:prop><d:resourcetype><d:collection/></d:resourcetype></d:prop></d:propstat>
      </d:response>
      <d:response>
        <d:href>/dav/root/Prob_R11_100m.tif</d:href>
        <d:propstat><d:prop>
          <d:getcontentlength>123</d:getcontentlength>
          <d:getetag>&quot;etag-11&quot;</d:getetag>
        </d:prop></d:propstat>
      </d:response>
    </d:multistatus>"""

    entries = parse_webdav_entries(xml.encode())

    assert entries[0].is_collection
    assert entries[1].path.endswith("Prob_R11_100m.tif")
    assert entries[1].size == 123
    assert entries[1].etag == "etag-11"


def test_extract_share_token_requires_a_public_share_token() -> None:
    assert extract_share_token('<input name="sharingToken" value="token-123">') == "token-123"
    with pytest.raises(ValueError, match="share token"):
        extract_share_token("<html>no token</html>")


def test_raster_assets_require_a_label_for_every_tiff() -> None:
    entries = json.loads(
        json.dumps(
            [
                {
                    "path": "/dav/Prob_R11_100m.tif",
                    "url": "https://example.test/dav/Prob_R11_100m.tif",
                    "size": 10,
                    "etag": "etag",
                    "is_collection": False,
                }
            ]
        )
    )

    from osm_polygon_eunis.eea import WebDavEntry

    webdav_entries = tuple(WebDavEntry(**entry) for entry in entries)
    assets = raster_assets_from_entries(
        webdav_entries,
        {"R11": "Pannonian and Pontic sandy steppe"},
        record_id="record",
        source_version="EEA-test",
    )

    assert assets[0].code == "R11"
    assert assets[0].name == "Pannonian and Pontic sandy steppe"
    assert assets[0].size == 10


def test_eea_parsers_fail_closed_on_malformed_payloads() -> None:
    with pytest.raises(ValueError, match="features list"):
        parse_arcgis_labels({})
    with pytest.raises(ValueError, match="attributes"):
        parse_arcgis_labels({"features": [{}]})
    with pytest.raises(ValueError, match="incomplete label"):
        parse_arcgis_labels({"features": [{"attributes": {"Prob": 11}}]})
    with pytest.raises(ValueError, match="conflicting names"):
        parse_arcgis_labels(
            {
                "features": [
                    {"attributes": {"Prob": "R11", "Name": "one"}},
                    {"attributes": {"Prob": "R11", "Name": "two"}},
                ]
            }
        )
    with pytest.raises(ValueError, match="code/name rows"):
        parse_classification_rows([("not", "a", "header")])
    with pytest.raises(ValueError, match="conflicting names"):
        parse_classification_rows(
            [("Code", "Name"), ("R11", "one"), ("Code", "Name"), ("R11", "two")]
        )


def test_classification_workbook_supports_inline_cells_and_rejects_bad_xlsx() -> None:
    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="inlineStr"><is><t>Code</t></is></c>'
            '<c r="B1" t="inlineStr"><is><t>Name</t></is></c></row>'
            '<row r="2"><c r="A2"><v>R11</v></c>'
            '<c r="B2"><v>Steppe</v></c></row></sheetData></worksheet>',
        )

    assert parse_classification_workbook(workbook.getvalue()) == {"R11": "Steppe"}
    with pytest.raises(ValueError, match="valid XLSX"):
        parse_classification_workbook(b"not-a-zip")


def test_webdav_and_raster_asset_metadata_errors() -> None:
    with pytest.raises(ValueError, match="missing href"):
        parse_webdav_entries(
            b'<d:multistatus xmlns:d="DAV:"><d:response><d:propstat/></d:response></d:multistatus>'
        )
    with pytest.raises(ValueError, match="reference layer code"):
        raster_assets_from_entries(
            (WebDavEntry("/bad.tif", "https://example.test/bad.tif", 1, None, False),),
            {},
            record_id="record",
            source_version="EEA-test",
        )
    entry = WebDavEntry("/Prob_R11_100m.tif", "https://example.test/r11", None, None, False)
    with pytest.raises(ValueError, match="byte size"):
        raster_assets_from_entries(
            (entry,), {"R11": "steppe"}, record_id="record", source_version="EEA-test"
        )


def test_catalog_and_classification_selection_reject_ambiguous_records() -> None:
    record = _catalog_record("https://example.test/folder", "https://example.test/service")
    record["crs"] = "EPSG:4326"
    with pytest.raises(ValueError, match="catalog record"):
        catalog_links(record)
    with pytest.raises(ValueError, match="unambiguous"):
        _select_classification_entry(())
    with pytest.raises(ValueError, match="unambiguous"):
        _select_classification_entry(
            (
                WebDavEntry("/one.xlsx", "https://example.test/one", 1, None, False),
                WebDavEntry("/two.xlsx", "https://example.test/two", 1, None, False),
            )
        )
    with pytest.raises(ValueError, match="title or public folder"):
        _classification_catalog_links({})


def test_arcgis_pagination_and_non_mapping_pages() -> None:
    class PagingClient:
        def __init__(self) -> None:
            self.offsets: list[int] = []

        def get(self, url, **kwargs):
            del url
            self.offsets.append(kwargs["params"]["resultOffset"])
            if len(self.offsets) == 1:
                return _FakeResponse(
                    payload={
                        "features": [{"attributes": {"Prob": "R11", "Name": "steppe"}}],
                        "exceededTransferLimit": True,
                    }
                )
            return _FakeResponse(
                payload={"features": [{"attributes": {"Prob": "R12", "Name": "bog"}}]}
            )

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("the ArcGIS client should use GET")

    client = PagingClient()
    assert _fetch_arcgis_labels(client, "https://example.test/service") == {
        "R11": "steppe",
        "R12": "bog",
    }
    assert client.offsets == [0, 1]

    class BadClient:
        def get(self, url, **kwargs):
            del url, kwargs
            return _FakeResponse(payload=[])

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("the ArcGIS client should use GET")

    with pytest.raises(ValueError, match="not an object"):
        _fetch_arcgis_labels(BadClient(), "https://example.test/service")


def test_vector_asset_and_group_resolution_fail_closed(monkeypatch) -> None:
    first = WebDavEntry("/one.gpkg", "https://example.test/one", 1, None, False)
    second = WebDavEntry("/two.gpkg", "https://example.test/two", 1, None, False)
    with pytest.raises(ValueError, match="multiple GeoPackages"):
        _vector_asset((first, second), "record", "EEA-test")
    with pytest.raises(ValueError, match="byte size"):
        _vector_asset(
            (WebDavEntry("/one.gpkg", "https://example.test/one", None, None, False),),
            "record",
            "EEA-test",
        )

    monkeypatch.setattr(
        eea,
        "catalog_links",
        lambda record: (
            "title",
            "https://sdi.eea.europa.eu/webdav/public/folder",
            "service",
        ),
    )
    monkeypatch.setattr(eea, "extract_share_token", lambda html: "token")
    monkeypatch.setattr(eea, "_discover_entries", lambda *args, **kwargs: ())
    monkeypatch.setattr(eea, "_fetch_arcgis_labels", lambda *args, **kwargs: {})

    class EmptyClient:
        def get(self, url, **kwargs):
            del url, kwargs
            return _FakeResponse(payload={})

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("WebDAV discovery should be stubbed")

    with pytest.raises(ValueError, match="no GeoTIFF"):
        resolve_group(EmptyClient(), "record", source_version="EEA-test")


def test_config_resolution_and_asset_download(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "reference.json"
    config_path.write_text(
        json.dumps(
            {
                "source_version": "EEA-test",
                "catalog_records": ["one"],
                "classification_record": "classification",
                "supplemental_labels": {"MA223": "Saltmarsh"},
            }
        ),
        encoding="utf-8",
    )
    settings = eea._config_settings(json.loads(config_path.read_text(encoding="utf-8")))
    assert settings.record_ids == ("one",)
    assert settings.supplemental_labels == {"MA223": "Saltmarsh"}
    for invalid in (
        None,
        {"source_version": "x"},
        {"source_version": "x", "catalog_records": "one"},
    ):
        with pytest.raises(ValueError):
            eea._config_settings(invalid)
    with pytest.raises(ValueError, match="supplemental"):
        eea._config_settings(
            {"source_version": "x", "catalog_records": [], "supplemental_labels": {"R11": ""}}
        )

    class FakeClient:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

    resolved_group = object()
    monkeypatch.setattr(eea.httpx, "Client", lambda **kwargs: FakeClient())
    monkeypatch.setattr(
        eea,
        "resolve_classification_labels",
        lambda client, record: {"R11": "steppe"},
    )
    monkeypatch.setattr(eea, "_resolve_groups", lambda client, current, labels: (resolved_group,))
    assert eea.resolve_config(config_path) == (resolved_group,)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, *, chunk_size: int):
            assert chunk_size > 0
            yield b"abc"

    class StreamClient:
        def stream(self, method, url):
            assert method == "GET"
            assert url.endswith("asset.bin")
            return Response()

    asset = RemoteAsset(
        "/asset.bin",
        "https://example.test/asset.bin",
        3,
        None,
        None,
        None,
        "r",
        "v",
    )
    destination = tmp_path / "nested" / "asset.bin"
    assert download_asset(cast(httpx.Client, StreamClient()), asset, destination)
    with pytest.raises(ValueError, match="byte count"):
        download_asset(
            cast(httpx.Client, StreamClient()),
            RemoteAsset(
                "/asset.bin",
                "https://example.test/asset.bin",
                4,
                None,
                None,
                None,
                "r",
                "v",
            ),
            tmp_path / "bad.bin",
        )
