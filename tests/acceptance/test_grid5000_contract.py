from pathlib import Path


def test_production_contract_is_grid5000_description_only() -> None:
    root = Path(__file__).parents[2]
    operations = (root / "docs" / "operations.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    worker = (
        root / "scripts" / "grid5000" / "description-release.sh"
    ).read_text(encoding="utf-8")

    assert "Grid'5000" in operations
    assert "grid5000 submit" in operations
    assert "usagepolicycheck -t" in operations
    assert "host=1/core=16" in operations
    assert "-p chuc" in operations
    assert "-t night" in operations
    assert "--dataset description" in operations
    assert "--execution grid5000" in operations
    assert "EUNIS_SIDECAR_DIR" in operations
    assert "EUNIS_SOURCE_DIR" in operations
    assert "node-local" in operations
    assert "not backed up" in operations
    assert (
        "uv run osm-polygon-eunis release --batch-size 256 --workdir .eunis-run"
        not in operations
    )
    assert "Grid'5000-only" in readme
    assert "--dataset description" in worker
    assert "--execution grid5000" in worker
