from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from benchmarks.test_synthetic_performance import _update_sidecar_compatibly
from scripts.compare_benchmark_results import compare_results


def _results(value: float) -> dict[str, Any]:
    return {
        "metrics": {
            "raster_random_seconds": value,
            "raster_spatial_seconds": value,
            "mask_geometry_ms": value,
            "sidecar_update_seconds": value,
            "geopackage_overlap_seconds": value,
            "raster_random_cache_miss_rate": 0.1,
            "raster_spatial_cache_miss_rate": 0.1,
            "sidecar_cache_miss_rate": 0.1,
        },
        "invariants": {
            "raster_order_independent": True,
            "mask_geometry_matches_geojson": True,
            "sidecar_rows_match_input": True,
            "sidecar_sha256": "a" * 64,
            "geopackage_result_count": 1000,
        },
    }


def test_sidecar_benchmark_call_supports_legacy_public_signature() -> None:
    seen: list[tuple[object, ...]] = []

    def update(source, destination, *, reference, batch_size):
        seen.append((source, destination, reference, batch_size))
        return 12

    module = SimpleNamespace(update_label_sidecar=update)
    reference = object()

    result = _update_sidecar_compatibly(module, Path("source"), Path("output"), reference, 128)

    assert result == 12
    assert seen == [(Path("source"), Path("output"), reference, 128)]


def test_sidecar_benchmark_call_supports_options_public_signature() -> None:
    def update(_source, _destination, options):
        return options

    module = SimpleNamespace(
        SidecarUpdateOptions=lambda **values: values,
        update_label_sidecar=update,
    )
    reference = object()

    result = _update_sidecar_compatibly(module, Path("source"), Path("output"), reference, 128)

    assert result == {"reference": reference, "batch_size": 128}


def test_compare_results_accepts_measurements_within_the_regression_budget() -> None:
    assert compare_results(_results(10.0), _results(11.4)) == []


def test_compare_results_accepts_a_sha256_sidecar_invariant() -> None:
    assert compare_results(_results(10.0), _results(10.0)) == []


def test_compare_results_reports_tile_cache_miss_rates(capsys) -> None:
    candidate = _results(10.0)
    candidate["metrics"].update(
        {
            "raster_random_cache_miss_rate": 0.478,
            "raster_spatial_cache_miss_rate": 0.312,
            "sidecar_cache_miss_rate": 0.25,
        }
    )

    assert compare_results(_results(10.0), candidate) == []

    output = capsys.readouterr().out
    assert "raster_random_cache_miss_rate: candidate=47.8%" in output
    assert "raster_spatial_cache_miss_rate: candidate=31.2%" in output
    assert "sidecar_cache_miss_rate: candidate=25.0%" in output


def test_compare_results_rejects_invalid_tile_cache_miss_rates() -> None:
    candidate = _results(10.0)
    candidate["metrics"]["sidecar_cache_miss_rate"] = 1.1  # type: ignore[index]

    errors = compare_results(_results(10.0), candidate)

    assert errors == ["candidate has no valid measurement for sidecar_cache_miss_rate"]


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
