"""Tests for exact native download scope selection and manifest validation."""

from __future__ import annotations

import pytest

from osm_polygon_eunis.native_asset_scope import (
    require_exact_asset_scope,
    select_assets_for_codes,
)


def test_select_assets_for_authorized_codes_excludes_unrequested_inventory_rows() -> None:
    inventory = (
        {"code": "T1G", "path": "/T1G.tif"},
        {"code": "N17", "path": "/N17.tif"},
        {"code": "R1J", "path": "/R1J.tif"},
        {"code": "S74", "path": "/S74.tif"},
    )

    selected = select_assets_for_codes(inventory, ("S74", "R1J", "T1G"))

    assert tuple(asset["code"] for asset in selected) == ("R1J", "S74", "T1G")
    require_exact_asset_scope(selected, ("T1G", "R1J", "S74"))


def test_final_manifest_rejects_an_asset_outside_authorized_scope() -> None:
    manifest_assets = tuple({"code": code} for code in ("N17", "T1G", "R1J", "S74"))

    with pytest.raises(ValueError, match=r"missing=\[\]; unexpected=\['N17'\]"):
        require_exact_asset_scope(manifest_assets, ("T1G", "R1J", "S74"))


def test_final_manifest_rejects_missing_authorized_asset() -> None:
    with pytest.raises(ValueError, match=r"missing=\['S74'\]; unexpected=\[\]"):
        require_exact_asset_scope(({"code": "T1G"}, {"code": "R1J"}), ("T1G", "R1J", "S74"))


def test_final_manifest_rejects_duplicate_asset_code() -> None:
    with pytest.raises(ValueError, match="manifest asset codes are not unique: T1G"):
        require_exact_asset_scope(({"code": "T1G"}, {"code": "T1G"}), ("T1G",))


def test_inventory_rejects_duplicate_codes_even_outside_requested_subset() -> None:
    with pytest.raises(ValueError, match="source asset codes are not unique: N17"):
        select_assets_for_codes(({"code": "N17"}, {"code": "N17"}), ("T1G",))


def test_inventory_rejects_missing_requested_code() -> None:
    with pytest.raises(ValueError, match=r"requested native asset codes are missing: \['S74'\]"):
        select_assets_for_codes(({"code": "T1G"},), ("T1G", "S74"))


def test_scope_rejects_malformed_codes() -> None:
    with pytest.raises(ValueError, match="requested asset codes must be non-empty strings"):
        select_assets_for_codes((), ("",))
    with pytest.raises(ValueError, match="native asset record is missing"):
        require_exact_asset_scope(({"code": None},), ())
    with pytest.raises(ValueError, match="expected asset codes are not unique"):
        require_exact_asset_scope((), ("T1G", "T1G"))
