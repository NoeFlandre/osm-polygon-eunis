from __future__ import annotations

from typing import Any

from scripts.compare_benchmark_results import compare_results


def _results(value: float) -> dict[str, Any]:
    return {
        "metrics": {
            "raster_random_seconds": value,
            "raster_spatial_seconds": value,
            "mask_geometry_ms": value,
            "sidecar_update_seconds": value,
            "geopackage_overlap_seconds": value,
        },
        "invariants": {
            "raster_order_independent": True,
            "mask_geometry_matches_geojson": True,
            "sidecar_rows_match_input": True,
            "sidecar_sha256": "a" * 64,
            "geopackage_result_count": 1000,
        },
    }


def test_compare_results_accepts_measurements_within_the_regression_budget() -> None:
    assert compare_results(_results(10.0), _results(11.4)) == []


def test_compare_results_accepts_a_sha256_sidecar_invariant() -> None:
    assert compare_results(_results(10.0), _results(10.0)) == []


def test_compare_results_rejects_a_malformed_sidecar_sha256() -> None:
    candidate = _results(10.0)
    candidate["invariants"]["sidecar_sha256"] = "not-a-sha256"  # type: ignore[index]

    errors = compare_results(_results(10.0), candidate)

    assert any("sidecar_sha256" in error for error in errors)


def test_compare_results_reports_slowdowns_and_failed_output_invariants() -> None:
    candidate = _results(12.0)
    candidate["invariants"]["raster_order_independent"] = False  # type: ignore[index]

    errors = compare_results(_results(10.0), candidate)

    assert any("raster_random_seconds is 20.0% slower" in error for error in errors)
    assert any("raster_order_independent" in error for error in errors)


def test_compare_results_rejects_incomplete_measurements() -> None:
    candidate = _results(10.0)
    candidate["metrics"].pop("mask_geometry_ms")  # type: ignore[union-attr]

    errors = compare_results(_results(10.0), candidate)

    assert errors == ["candidate has no positive measurement for mask_geometry_ms"]
