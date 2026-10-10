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


def test_private_names_imported_from_package_modules_are_detected(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    architecture = run_path(str(root / "scripts" / "check_architecture.py"))
    source = tmp_path / "consumer.py"
    source.write_text(
        "from .producer import _helper, public\n"
        "from osm_polygon_eunis.producer import _Other\n"
        "from . import _module\n"
        "from .producer import __version__, Public as _Alias\n"
        "from defusedxml import fromstring as _fromstring\n",
        encoding="utf-8",
    )

    assert architecture["_private_imports"](source) == {
        (".producer", "_helper"),
        ("osm_polygon_eunis.producer", "_Other"),
        (".", "_module"),
    }


def test_private_attributes_of_package_modules_are_detected(tmp_path: Path) -> None:
    root = Path(__file__).parents[2]
    architecture = run_path(str(root / "scripts" / "check_architecture.py"))
    source = tmp_path / "consumer.py"
    source.write_text(
        "import os\n"
        "from . import release_plan\n"
        "from osm_polygon_eunis import geometry_chunks as chunks\n"
        "release_plan._sample_helper(1)\n"
        "release_plan.public_helper(1)\n"
        "release_plan.__name__\n"
        "chunks._SampleChunk\n"
        "os._exit(0)\n"
        "unknown._hidden\n",
        encoding="utf-8",
    )

    assert architecture["_private_attributes"](source) == {
        ("release_plan", "_sample_helper"),
        ("geometry_chunks", "_SampleChunk"),
    }


def test_package_modules_use_no_private_names_across_modules() -> None:
    root = Path(__file__).parents[2]
    architecture = run_path(str(root / "scripts" / "check_architecture.py"))
    offenders = {
        module: sorted(
            architecture["_private_imports"](path) | architecture["_private_attributes"](path)
        )
        for module, path in architecture["MODULES"].items()
        if architecture["_private_imports"](path) or architecture["_private_attributes"](path)
    }

    assert offenders == {}
