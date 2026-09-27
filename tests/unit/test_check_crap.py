from __future__ import annotations

import json
from pathlib import Path

from scripts import check_crap


def test_coverage_fraction_counts_executed_and_missing_branches() -> None:
    fraction = check_crap._coverage_fraction(
        {
            "executed_lines": [10, 11],
            "missing_lines": [12],
            "executed_branches": [[10, 0]],
            "missing_branches": [[10, 1], [11, 0]],
        },
        10,
        12,
    )

    assert fraction == 0.5


def test_main_fails_if_any_source_module_has_no_coverage_entry(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "src" / "osm_polygon_eunis" / "unmeasured.py"
    source.parent.mkdir(parents=True)
    source.write_text("def unmeasured():\n    return True\n", encoding="utf-8")
    (tmp_path / "coverage.json").write_text(json.dumps({"files": {}}), encoding="utf-8")
    monkeypatch.setattr(check_crap, "ROOT", tmp_path)

    assert check_crap.main([]) == 1
    assert "src/osm_polygon_eunis/unmeasured.py" in capsys.readouterr().err


def test_main_accepts_coverage_json_outside_the_repository_root(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    source = tmp_path / "src" / "osm_polygon_eunis" / "unmeasured.py"
    source.parent.mkdir(parents=True)
    source.write_text("def unmeasured():\n    return True\n", encoding="utf-8")
    coverage_path = tmp_path / "scratch" / "coverage.json"
    coverage_path.parent.mkdir()
    coverage_path.write_text(json.dumps({"files": {}}), encoding="utf-8")
    monkeypatch.setattr(check_crap, "ROOT", tmp_path)

    assert check_crap.main(["--coverage-json", str(coverage_path)]) == 1
    assert "src/osm_polygon_eunis/unmeasured.py" in capsys.readouterr().err
