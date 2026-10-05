import tomllib
from pathlib import Path


def test_reference_modules_stay_under_four_hundred_lines() -> None:
    package = Path(__file__).parents[2] / "src" / "osm_polygon_eunis"
    module_names = (
        "reference_staging.py",
        "reference_cache.py",
        "raster_reference.py",
        "raster_geometry.py",
        "geopackage_reference.py",
        "geopackage_tiles.py",
        "geopackage_sql.py",
    )

    for module_name in module_names:
        source = (package / module_name).read_text(encoding="utf-8")
        assert len(source.splitlines()) < 400, module_name


def test_reference_staging_module_uses_unambiguous_name() -> None:
    package = Path(__file__).parents[2] / "src" / "osm_polygon_eunis"
    assert (package / "reference_staging.py").is_file()
    assert not (package / "references.py").exists()

    root = Path(__file__).parents[2]
    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    per_file_ignores = config["tool"]["ruff"]["lint"]["per-file-ignores"]
    assert "src/osm_polygon_eunis/references.py" not in per_file_ignores
