import tomllib
from pathlib import Path

from scripts.check_mutation import (
    ALLOWED_MUTANTS,
    MODULE_MINIMUM_SCORES,
    evaluate,
    module_of,
    module_scores,
)


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
    stats = _stats(124, 1, 125)
    results = _rows(124, ["osm_polygon_eunis.matching.x__outranks__mutmut_5"])

    assert evaluate(stats, results) == []


def test_accepts_every_allowlisted_mutant_when_module_scores_hold() -> None:
    killed = 10_000
    stats = _stats(killed, len(ALLOWED_MUTANTS), killed + len(ALLOWED_MUTANTS))
    rows = []
    for module in MODULE_MINIMUM_SCORES:
        rows.extend(f"osm_polygon_eunis.{module}.mutant_{index}: killed" for index in range(2_000))
    results = "\n".join(rows[:killed] + [f"{name}: survived" for name in ALLOWED_MUTANTS])

    assert evaluate(stats, results) == []


def test_rejects_a_module_below_its_own_baseline() -> None:
    survivors = [f"osm_polygon_eunis.matching.mutant_{index}" for index in range(11)]
    stats = _stats(113, 11, 124)
    results = _rows(113, survivors)

    errors = evaluate(stats, results)

    assert any("matching" in error and "baseline" in error for error in errors)
    assert any("unapproved survivor" in error for error in errors)


def test_modules_ratchet_independently() -> None:
    rows = [f"osm_polygon_eunis.geometry.m_{index}: killed" for index in range(100)]
    rows += [f"osm_polygon_eunis.transform.m_{index}: killed" for index in range(98)]
    rows += [
        "osm_polygon_eunis.transform.m_98: survived",
        "osm_polygon_eunis.transform.m_99: killed",
    ]
    stats = _stats(199, 1, 200)

    errors = evaluate(stats, "\n".join(rows))

    assert not any("geometry" in error and "baseline" in error for error in errors)
    assert not any("transform" in error and "baseline" in error for error in errors)
    assert any("unapproved survivor: osm_polygon_eunis.transform.m_98" in e for e in errors)


def test_rejects_modules_without_a_configured_baseline() -> None:
    stats = _stats(1, 0, 1)

    errors = evaluate(stats, "osm_polygon_eunis.unlisted.x: killed")

    assert errors == ["no minimum mutation score is configured for module 'unlisted'"]


def test_module_helpers_split_mutant_names() -> None:
    assert module_of("osm_polygon_eunis.grid_overlap.x__burn__mutmut_1") == "grid_overlap"
    assert module_of("osm_polygon_eunis.mutant_5") == ""
    assert module_scores(
        {"osm_polygon_eunis.a.x": "killed", "osm_polygon_eunis.a.y": "survived"}
    ) == {"a": 50.0}


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


def test_mutation_scope_lists_the_ratcheted_modules_and_their_tests() -> None:
    root = Path(__file__).resolve().parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    mutation = config["tool"]["mutmut"]
    mutated = {Path(path).stem for path in mutation["source_paths"]}

    assert mutated == set(MODULE_MINIMUM_SCORES)
    assert {
        "tests/unit/test_grid_overlap.py",
        "tests/unit/test_transform_contract.py",
        "tests/unit/test_release_plan.py",
    } <= set(mutation["pytest_add_cli_args_test_selection"])
