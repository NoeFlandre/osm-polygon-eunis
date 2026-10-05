from pathlib import Path
from runpy import run_path


def test_reference_reader_split_is_declared_in_architecture_layers() -> None:
    root = Path(__file__).parents[2]
    architecture = run_path(str(root / "scripts" / "check_architecture.py"))
    layers = architecture["LAYERS"]
    positions = {layer: index for index, layer in enumerate(layers)}
    required = {
        "geopackage_sql",
        "geopackage_tiles",
        "raster_reference",
        "geopackage_reference",
        "reference_staging",
    }
    dependencies = (
        ("geopackage_sql", "geopackage_tiles"),
        ("matching", "raster_reference"),
        ("geopackage_tiles", "geopackage_reference"),
        ("geopackage_sql", "geopackage_reference"),
        ("_protocols", "reference_staging"),
        ("eea", "reference_staging"),
        ("fileio", "reference_staging"),
        ("geopackage_reference", "reference_staging"),
        ("options", "reference_staging"),
        ("raster_reference", "reference_staging"),
        ("reference_cache", "reference_staging"),
        ("transform", "reference_staging"),
    )

    assert required <= positions.keys()
    assert all(positions[dependency] < positions[consumer] for dependency, consumer in dependencies)
