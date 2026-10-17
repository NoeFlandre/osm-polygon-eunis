"""Exact code-set selection and validation for native asset manifests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

__all__ = [
    "require_exact_asset_scope",
    "select_assets_for_codes",
]

NativeAssetRecord = Mapping[str, object]


def select_assets_for_codes(
    assets: Iterable[NativeAssetRecord],
    requested_codes: Iterable[str],
) -> tuple[NativeAssetRecord, ...]:
    """Select every requested code once from a source inventory.

    Unrequested source inventory rows are intentionally excluded. Duplicate
    source codes and missing requested codes fail closed.
    """
    requested = _unique_codes(requested_codes, "requested")
    indexed: dict[str, NativeAssetRecord] = {}
    for asset in assets:
        code = _asset_code(asset)
        if code in indexed:
            raise ValueError(f"source asset codes are not unique: {code}")
        indexed[code] = asset
    missing = sorted(requested - indexed.keys())
    if missing:
        raise ValueError(f"requested native asset codes are missing: {missing}")
    return tuple(indexed[code] for code in sorted(requested))


def require_exact_asset_scope(
    assets: Iterable[NativeAssetRecord],
    expected_codes: Iterable[str],
) -> None:
    """Reject missing, duplicate, or unauthorized codes in a final manifest."""
    expected = _unique_codes(expected_codes, "expected")
    actual: set[str] = set()
    for asset in assets:
        code = _asset_code(asset)
        if code in actual:
            raise ValueError(f"manifest asset codes are not unique: {code}")
        actual.add(code)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        raise ValueError(
            f"asset manifest code set mismatch: missing={missing}; unexpected={unexpected}"
        )


def _unique_codes(codes: Iterable[str], label: str) -> set[str]:
    values = tuple(codes)
    if any(not isinstance(code, str) or not code for code in values):
        raise ValueError(f"{label} asset codes must be non-empty strings")
    unique = set(values)
    if len(unique) != len(values):
        raise ValueError(f"{label} asset codes are not unique")
    return unique


def _asset_code(asset: NativeAssetRecord) -> str:
    code = asset.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError("native asset record is missing a non-empty string code")
    return code
