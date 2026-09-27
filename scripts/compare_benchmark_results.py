"""Compare synthetic performance measurements from base and candidate commits."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

_TIMING_METRICS = (
    "raster_random_seconds",
    "raster_spatial_seconds",
    "mask_geometry_ms",
    "sidecar_update_seconds",
    "geopackage_overlap_seconds",
)
_CACHE_MISS_METRICS = (
    "raster_random_cache_miss_rate",
    "raster_spatial_cache_miss_rate",
    "sidecar_cache_miss_rate",
)
_MAX_CACHE_MISS_RATE = 0.30


def compare_results(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    max_slowdown: float = 0.15,
) -> list[str]:
    """Return missing, invalid, invariant, or over-budget benchmark results."""

    errors: list[str] = []
    baseline_metrics = baseline.get("metrics")
    candidate_metrics = candidate.get("metrics")
    if not isinstance(baseline_metrics, dict) or not isinstance(candidate_metrics, dict):
        return ["benchmark results must contain a metrics object"]
    candidate_invariants = candidate.get("invariants")
    if not isinstance(candidate_invariants, dict):
        return ["candidate benchmark results must contain an invariants object"]
    errors.extend(_candidate_invariant_errors(candidate_invariants))
    errors.extend(_timing_errors(baseline_metrics, candidate_metrics, max_slowdown))
    errors.extend(_report_cache_miss_rates(candidate_metrics))
    return errors


def _report_cache_miss_rates(metrics: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name in _CACHE_MISS_METRICS:
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):
            errors.append(f"candidate has no valid measurement for {name}")
            continue
        print(f"{name}: candidate={value:.1%}")
        if value > _MAX_CACHE_MISS_RATE:
            errors.append(f"{name} is {value:.1%}, above the {_MAX_CACHE_MISS_RATE:.1%} limit")
    return errors


def _candidate_invariant_errors(invariants: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for name, valid in invariants.items():
        if name == "sidecar_sha256":
            if not isinstance(valid, str) or re.fullmatch(r"[0-9a-f]{64}", valid) is None:
                errors.append(f"candidate invariant {name!r} is not a SHA-256 digest: {valid!r}")
            continue
        if name == "geopackage_result_count":
            if isinstance(valid, bool) or not isinstance(valid, int) or valid <= 0:
                errors.append(f"candidate invariant {name!r} failed: {valid!r}")
            continue
        if valid is not True:
            errors.append(f"candidate invariant {name!r} failed: {valid!r}")
    return errors


def _timing_errors(
    baseline_metrics: dict[str, Any],
    candidate_metrics: dict[str, Any],
    max_slowdown: float,
) -> list[str]:
    errors: list[str] = []
    for name in _TIMING_METRICS:
        base_value = baseline_metrics.get(name)
        candidate_value = candidate_metrics.get(name)
        if not isinstance(base_value, (float, int)) or base_value <= 0:
            errors.append(f"baseline has no positive measurement for {name}")
            continue
        if not isinstance(candidate_value, (float, int)) or candidate_value <= 0:
            errors.append(f"candidate has no positive measurement for {name}")
            continue
        slowdown = (candidate_value / base_value) - 1
        print(
            f"{name}: baseline={base_value:.6f}, "
            f"candidate={candidate_value:.6f}, change={slowdown:+.1%}"
        )
        if slowdown > max_slowdown:
            errors.append(f"{name} is {slowdown:.1%} slower, above the {max_slowdown:.1%} limit")
    return errors


def _read_results(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return payload


def main(argv: list[str] | None = None) -> int:
    """Fail when candidate performance regresses more than the configured limit."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--max-slowdown", type=float, default=0.15)
    args = parser.parse_args(argv)
    try:
        errors = compare_results(
            _read_results(args.baseline),
            _read_results(args.candidate),
            max_slowdown=args.max_slowdown,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"cannot read benchmark results: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("performance comparison passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
