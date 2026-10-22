from __future__ import annotations

import hashlib
import io
import json
import logging
import zipfile
from pathlib import Path
from typing import cast

import httpx
import pytest

from osm_polygon_eunis import eea
from osm_polygon_eunis._protocols import HttpClient
from osm_polygon_eunis.eea import (
    ClassificationTaxonomy,
    ClassificationTaxonomyCacheLineage,
    ClassificationTaxonomyLineagePin,
    RemoteAsset,
    WebDavEntry,
    _classification_folder,
    _discover_entries,
    _fetch_arcgis_labels,
    _get_with_retry,
    _select_classification_entry,
    _vector_asset,
    _webdav_folder_url,
    catalog_links,
    download_asset,
    extract_share_token,
    parse_arcgis_labels,
    parse_classification_rows,
    parse_classification_taxonomy_rows,
    parse_classification_taxonomy_workbook,
    parse_classification_workbook,
    parse_webdav_entries,
    raster_assets_from_entries,
    resolve_classification_labels,
    resolve_group,
    validate_classification_taxonomy_lineage,
    verify_classification_taxonomy_lineage,
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


def test_classification_taxonomy_uses_longest_proper_prefix_at_lower_level() -> None:
    taxonomy = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "A", "Root A"],
            [5, "AB", "Higher-level prefix"],
            [2, "ABC", "Lower-level prefix"],
            [4, "ABCD", "Child of ABC"],
            [2, "X", "Root X"],
            [3, "XY", "Same-level prefix"],
            [3, "XYZ", "Child of X"],
            ["Level", "Code 2018", "Name 2018"],
            [1, "Q", "Root Q"],
            [2, "Q1", "Child Q1"],
            [3, "Q11", "Child Q11"],
        ],
        supplemental_labels={"MA223": "Saltmarsh"},
    )

    assert taxonomy.labels["MA223"] == "Saltmarsh"
    assert "MA223" not in taxonomy.levels
    assert taxonomy.parents["ABCD"] == "ABC"
    assert taxonomy.parents["XYZ"] == "X"
    assert taxonomy.parents["Q11"] == "Q1"
    assert taxonomy.ancestors_including_self("Q11") == ("Q11", "Q1", "Q")
    assert taxonomy.root_for("MA223") == "MA223"


def test_classification_taxonomy_rejects_invalid_levels_and_conflicting_rows() -> None:
    with pytest.raises(ValueError, match="invalid Level"):
        parse_classification_taxonomy_rows(
            [["Level", "Code", "Name"], ["not-a-level", "Q1", "Child"]]
        )
    with pytest.raises(ValueError, match="conflicting name or Level"):
        parse_classification_taxonomy_rows(
            [
                ["Level", "Code", "Name"],
                [1, "Q1", "First"],
                [2, "Q1", "Different"],
            ]
        )
    with pytest.raises(ValueError, match="missing name"):
        parse_classification_taxonomy_rows(
            [["Level", "Code", "Name"], [1, "Q1", "Root"], [2, "Q11", ""]]
        )
    with pytest.raises(ValueError, match="has no Level/Code/Name rows"):
        parse_classification_taxonomy_rows([["Level", "Code", "Name"], [1, "", "No code"]])


@pytest.mark.parametrize("invalid_level", [True, 0, -1, "1"])
def test_classification_taxonomy_rejects_nonpositive_or_noninteger_levels(
    invalid_level: object,
) -> None:
    with pytest.raises(ValueError, match="invalid Level"):
        ClassificationTaxonomy(
            labels={"A": "Root"},
            levels={"A": cast(int, invalid_level)},
            parents={},
        )


@pytest.mark.parametrize(
    ("levels", "parents", "error"),
    [
        ({"A11": 3}, {"A11": "A"}, "parent without a Level"),
        ({"A11": 3, "B": 1}, {"A11": "B"}, "invalid prefix parent"),
        ({"A": 2, "A11": 2}, {"A11": "A"}, "not at a lower Level"),
    ],
)
def test_classification_taxonomy_rejects_invalid_parent_edges(
    levels: dict[str, int],
    parents: dict[str, str],
    error: str,
) -> None:
    labels = {code: f"Label {code}" for code in set(levels) | set(parents.values())}
    with pytest.raises(ValueError, match=error):
        ClassificationTaxonomy(labels=labels, levels=levels, parents=parents)


def test_classification_taxonomy_requires_labels_for_every_parent() -> None:
    with pytest.raises(ValueError, match="missing label for A"):
        ClassificationTaxonomy(
            labels={"A11": "Class A11"},
            levels={"A": 1, "A11": 3},
            parents={"A11": "A"},
        )


def test_cached_classification_hierarchy_must_match_pinned_workbook_lineage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_taxonomy = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "A", "Root A"],
            [2, "A1", "Branch A1"],
            [3, "A11", "Class A11"],
        ]
    )
    workbook_payload = b"pinned workbook fixture"
    cache_payload = b"pinned cache fixture"
    workbook_sha256 = hashlib.sha256(workbook_payload).hexdigest()
    cache_sha256 = hashlib.sha256(cache_payload).hexdigest()
    monkeypatch.setattr(
        eea,
        "parse_classification_taxonomy_workbook",
        lambda _payload, *, supplemental_labels=None: workbook_taxonomy,
    )

    verified = validate_classification_taxonomy_lineage(
        workbook_taxonomy,
        cache_payload,
        workbook_payload,
        ClassificationTaxonomyCacheLineage("fixture-record", workbook_sha256),
        ClassificationTaxonomyLineagePin(cache_sha256, "fixture-record", workbook_sha256),
    )

    assert verified == workbook_taxonomy


def test_lineage_verifier_returns_token_gated_immutable_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    taxonomy = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "A", "Root A"],
            [2, "A1", "Branch A1"],
            [3, "A11", "Class A11"],
        ]
    )
    cache_payload = b"cache bytes"
    workbook_payload = b"workbook bytes"
    pin = ClassificationTaxonomyLineagePin(
        hashlib.sha256(cache_payload).hexdigest(),
        "fixture-record",
        hashlib.sha256(workbook_payload).hexdigest(),
    )
    monkeypatch.setattr(
        eea,
        "parse_classification_taxonomy_workbook",
        lambda _payload, *, supplemental_labels=None: taxonomy,
    )

    verified = verify_classification_taxonomy_lineage(
        taxonomy,
        cache_payload,
        workbook_payload,
        ClassificationTaxonomyCacheLineage("fixture-record", pin.workbook_sha256),
        pin,
    )

    assert verified.taxonomy is taxonomy
    assert verified.lineage_pin == pin
    assert len(verified.taxonomy_mapping_sha256) == 64
    with pytest.raises(TypeError, match="created by the lineage verifier"):
        eea.VerifiedClassificationTaxonomy(taxonomy, pin, "0" * 64, _token=object())
    with pytest.raises(TypeError):
        cast(dict[str, int], taxonomy.levels)["A11"] = 2


def test_cached_classification_hierarchy_rejects_changed_parent_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workbook_taxonomy = parse_classification_taxonomy_rows(
        [
            ["Level", "Code", "Name"],
            [1, "A", "Root A"],
            [2, "A1", "Branch A1"],
            [3, "A11", "Class A11"],
        ]
    )
    changed_cache = ClassificationTaxonomy(
        labels=workbook_taxonomy.labels,
        levels=workbook_taxonomy.levels,
        parents={"A11": "A"},
    )
    workbook_payload = b"workbook"
    cache_payload = b"cache"
    workbook_sha256 = hashlib.sha256(workbook_payload).hexdigest()
    monkeypatch.setattr(
        eea,
        "parse_classification_taxonomy_workbook",
        lambda _payload, *, supplemental_labels=None: workbook_taxonomy,
    )

    with pytest.raises(ValueError, match="parent hierarchy differs"):
        validate_classification_taxonomy_lineage(
            changed_cache,
            cache_payload,
            workbook_payload,
            ClassificationTaxonomyCacheLineage("fixture-record", workbook_sha256),
            ClassificationTaxonomyLineagePin(
                hashlib.sha256(cache_payload).hexdigest(),
                "fixture-record",
                workbook_sha256,
            ),
        )


def test_classification_taxonomy_headers_reset_between_workbook_sheets() -> None:
    workbook = io.BytesIO()
    with zipfile.ZipFile(workbook, "w") as archive:
        archive.writestr(
            "xl/sharedStrings.xml",
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            "<si><t>Level</t></si><si><t>Code</t></si><si><t>Name</t></si>"
            "<si><t>1</t></si><si><t>N</t></si><si><t>Coastal habitats</t></si>"
            "<si><t>Name 2021</t></si><si><t>R11</t></si><si><t>Other note</t></si></sst>",
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="s"><v>0</v></c>'
            '<c r="B1" t="s"><v>1</v></c><c r="C1" t="s"><v>2</v></c></row>'
            '<row r="2"><c r="A2" t="s"><v>3</v></c>'
            '<c r="B2" t="s"><v>4</v></c><c r="C2" t="s"><v>5</v></c>'
            "</row></sheetData></worksheet>",
        )
        archive.writestr(
            "xl/worksheets/sheet2.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData><row r="1"><c r="A1" t="s"><v>6</v></c>'
            '<c r="B1" t="s"><v>7</v></c><c r="C1" t="s"><v>8</v></c>'
            "</row></sheetData></worksheet>",
        )

    taxonomy = parse_classification_taxonomy_workbook(workbook.getvalue())

    assert taxonomy.labels == {"N": "Coastal habitats"}
    assert taxonomy.levels == {"N": 1}


def test_classification_taxonomy_workbook_rejects_non_xlsx() -> None:
    with pytest.raises(ValueError, match="valid XLSX"):
        parse_classification_taxonomy_workbook(b"not-a-zip")


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


def test_classification_links_accept_iso_json_character_strings() -> None:
    record = {
        "title": "EUNIS terrestrial habitat classification review",
        "resources": [
            {
                "cit:CI_OnlineResource": {
                    "cit:linkage": {
                        "gco:CharacterString": {"#text": "https://example.test/folder"}
                    },
                    "cit:protocol": {"gco:CharacterString": {"#text": "EEA:FOLDERPATH"}},
                }
            }
        ],
    }

    assert _classification_folder(record) == "https://example.test/folder"


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
    def __init__(self, *, payload=None, text="", content=b"", status_code=200) -> None:
        self._payload = payload
        self.text = text
        self.content = content
        self.status_code = status_code

    def json(self):
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "status",
                request=httpx.Request("GET", "https://example.test"),
                response=httpx.Response(self.status_code),
            )


class _FakeHttpClient:
    def __init__(self, responses) -> None:
        self.responses = responses

    def get(self, url, **kwargs):
        del kwargs
        return self.responses[url]

    def request(self, method, url, **kwargs):
        del method, kwargs
        return self.responses[url]


def test_metadata_get_retries_timeout_and_server_error(monkeypatch) -> None:
    monkeypatch.setattr(eea.time, "sleep", lambda delay: None)

    class FlakyClient:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, **kwargs):
            del url, kwargs
            self.calls += 1
            if self.calls == 1:
                raise httpx.ReadTimeout("temporary timeout")
            if self.calls == 2:
                return _FakeResponse(status_code=503)
            return _FakeResponse(payload={"ok": True})

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("the retry test should use GET")

    client = FlakyClient()
    response = _get_with_retry(client, "https://example.test/metadata")

    assert response.json() == {"ok": True}
    assert client.calls == 3


def test_metadata_get_retry_is_logged(caplog, monkeypatch) -> None:
    monkeypatch.setattr(eea.time, "sleep", lambda delay: None)

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, **kwargs):
            del url, kwargs
            self.calls += 1
            if self.calls == 1:
                return _FakeResponse(status_code=503)
            return _FakeResponse(payload={"ok": True})

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("the retry test should use GET")

    with caplog.at_level(logging.WARNING, logger=eea.__name__):
        assert _get_with_retry(Client(), "https://example.test/metadata").json() == {"ok": True}

    assert any(
        record.levelno == logging.WARNING
        and "retry" in record.getMessage().lower()
        and "metadata" in record.getMessage().lower()
        for record in caplog.records
    )


def test_metadata_get_sleeps_with_exponential_backoff(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(eea.time, "sleep", sleeps.append)

    class Client:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, url, **kwargs):
            del url, kwargs
            self.calls += 1
            return _FakeResponse(status_code=503)

    client = Client()
    with pytest.raises(httpx.HTTPStatusError):
        _get_with_retry(cast(HttpClient, client), "https://example.test/metadata")

    assert client.calls == eea._HTTP_RETRY_ATTEMPTS
    assert sleeps == [0.5, 1.0]


def test_metadata_get_does_not_retry_client_errors(monkeypatch) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(eea.time, "sleep", sleeps.append)

    class Client:
        calls = 0

        def get(self, url, **kwargs):
            del url, kwargs
            self.calls += 1
            return _FakeResponse(status_code=404)

    client = Client()
    with pytest.raises(httpx.HTTPStatusError):
        _get_with_retry(cast(HttpClient, client), "https://example.test/metadata")
    assert client.calls == 1
    assert sleeps == []


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
        _classification_folder({})


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


def test_group_uses_complete_official_fallback_without_arcgis_fetch(monkeypatch) -> None:
    entry = WebDavEntry(
        "/Prob_N11_100m.tif", "https://example.test/Prob_N11_100m.tif", 1, None, False
    )
    monkeypatch.setattr(
        eea,
        "catalog_links",
        lambda record: (
            "title",
            "https://sdi.eea.europa.eu/webdav/public/folder",
            "https://example.test/service",
        ),
    )
    monkeypatch.setattr(eea, "extract_share_token", lambda html: "token")
    monkeypatch.setattr(eea, "_discover_entries", lambda *args, **kwargs: (entry,))
    monkeypatch.setattr(
        eea,
        "_fetch_arcgis_labels",
        lambda *args, **kwargs: pytest.fail("complete official labels should avoid ArcGIS"),
    )

    class Client:
        def get(self, url, **kwargs):
            del url, kwargs
            return _FakeResponse(payload={})

        def request(self, method, url, **kwargs):
            del method, url, kwargs
            raise AssertionError("WebDAV discovery should be stubbed")

    group = resolve_group(
        Client(),
        "record",
        source_version="EEA-test",
        fallback_labels={"N11": "Atlantic sand beach"},
    )

    assert group.labels == {"N11": "Atlantic sand beach"}
    assert [asset.code for asset in group.raster_assets] == ["N11"]


def test_download_asset_retries_an_interrupted_stream(monkeypatch, tmp_path: Path, caplog) -> None:
    monkeypatch.setattr(eea.time, "sleep", lambda delay: None)

    class Response:
        def __init__(self, attempt: int) -> None:
            self.attempt = attempt

        def __enter__(self):
            return self

        def __exit__(self, *_args) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_bytes(self, *, chunk_size: int):
            del chunk_size
            yield b"ab"
            if self.attempt == 1:
                raise httpx.RemoteProtocolError("interrupted stream")
            yield b"c"

    class FlakyStreamClient:
        def __init__(self) -> None:
            self.attempts = 0

        def stream(self, method, url):
            assert method == "GET"
            assert url.endswith("asset.bin")
            self.attempts += 1
            return Response(self.attempts)

    destination = tmp_path / "asset.bin"
    asset = RemoteAsset(
        "/asset.bin",
        "https://example.test/asset.bin",
        3,
        None,
        None,
        None,
        "record",
        "EEA-test",
    )

    client = FlakyStreamClient()
    with caplog.at_level(logging.WARNING, logger=eea.__name__):
        download_asset(cast(httpx.Client, client), asset, destination)

    assert client.attempts == 2
    assert destination.read_bytes() == b"abc"
    assert any("retry" in record.getMessage().lower() for record in caplog.records)


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
    sleeps: list[float] = []
    monkeypatch.setattr(eea.time, "sleep", sleeps.append)
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
            {
                "source_version": "x",
                "catalog_records": [],
                "classification_record": "classification",
                "supplemental_labels": {"R11": ""},
            }
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

    calls: list[str] = []

    class StreamClient:
        def stream(self, method, url):
            assert method == "GET"
            assert url.endswith("asset.bin")
            calls.append(method)
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
    calls.clear()
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
    assert calls == ["GET"] * eea._HTTP_RETRY_ATTEMPTS
    assert sleeps == [0.5, 1.0]


def test_webdav_listing_rejects_xml_entity_declarations() -> None:
    import defusedxml

    payload = (
        b'<?xml version="1.0"?><!DOCTYPE d [<!ENTITY a "aaaa">]>'
        b'<d:multistatus xmlns:d="DAV:">&a;</d:multistatus>'
    )
    with pytest.raises(defusedxml.DefusedXmlException):
        eea.parse_webdav_entries(payload)


def test_wrong_payload_shape_is_both_value_and_type_error() -> None:
    with pytest.raises(TypeError, match="no features list"):
        eea.parse_arcgis_labels({"features": None})
    with pytest.raises(ValueError, match="no features list"):
        eea.parse_arcgis_labels({})
