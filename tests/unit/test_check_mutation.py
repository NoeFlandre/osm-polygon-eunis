import tomllib
from pathlib import Path

from scripts.check_mutation import ALLOWED_MUTANTS, evaluate


def _rows(killed: int, survivors: list[str]) -> str:
    rows = [f"osm_polygon_eunis.matching.mutant_{index}: killed" for index in range(killed)]
    rows.extend(f"{name}: survived" for name in survivors)
    return "\n".join(rows)


def _stats(killed: int, survived: int, total: int, *, no_tests: int = 0) -> dict[str, int]:
    return {
        "killed": killed,
        "survived": survived,
        "total": total,
        "no_tests": no_tests,
        "skipped": 0,
        "suspicious": 0,
        "timeout": 0,
        "segfault": 0,
        "check_was_interrupted_by_user": 0,
    }


def test_accepts_threshold_and_only_documented_equivalent_mutant() -> None:
    stats = _stats(123, 1, 124)
    results = _rows(123, ["osm_polygon_eunis.matching.x__outranks__mutmut_5"])

    assert evaluate(stats, results) == []


def test_accepts_current_geometry_and_matching_equivalent_mutants() -> None:
    stats = _stats(203, 7, 210)
    results = _rows(203, list(ALLOWED_MUTANTS))

    assert evaluate(stats, results) == []


def test_rejects_score_below_measured_baseline() -> None:
    survivors = [f"osm_polygon_eunis.matching.mutant_{index}" for index in range(11)]
    stats = _stats(113, 11, 124)
    results = _rows(
        113,
        survivors,
    )

    errors = evaluate(stats, results)

    assert any("91.9%" in error for error in errors)
    assert any("unapproved survivor" in error for error in errors)


def test_rejects_mutants_without_tests_or_inconsistent_inventory() -> None:
    stats = _stats(4, 0, 6, no_tests=1)
    results = _rows(4, []) + "\nosm_polygon_eunis.mutant_5: no tests"

    errors = evaluate(stats, results)

    assert any("no_tests" in error for error in errors)
    assert any("inventory" in error for error in errors)


def test_rejects_unlisted_survivors() -> None:
    stats = _stats(123, 1, 124)
    results = _rows(123, ["osm_polygon_eunis.matching.x__percentage__mutmut_5"])

    assert any("unapproved survivor" in error for error in evaluate(stats, results))


def test_rejects_missing_result_rows() -> None:
    stats = _stats(2, 0, 2)

    assert any("inventory" in error for error in evaluate(stats, ""))


def test_rejects_interrupted_run() -> None:
    stats = _stats(2, 0, 2)
    stats["check_was_interrupted_by_user"] = 1

    errors = evaluate(stats, _rows(2, []))

    assert any("interrupted" in error for error in errors)


def test_mutation_scope_includes_geometry_and_its_unit_tests() -> None:
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]

    assert "src/osm_polygon_eunis/geometry.py" in mutation["source_paths"]
    assert "tests/unit/test_geometry.py" in mutation["pytest_add_cli_args_test_selection"]
    assert (
        "tests/property/test_geometry_properties.py"
        in mutation["pytest_add_cli_args_test_selection"]
    )
