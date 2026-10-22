from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

import osm_polygon_eunis.eunis_asset_manifest as asset_manifest
from osm_polygon_eunis.eunis_asset_manifest import (
    EunisRasterAssetManifest,
    EunisRasterAssetPin,
    SourceOutputSemanticsEvidence,
    load_eunis_2021_asset_manifest,
)
from osm_polygon_eunis.pool_adapter import EunisTaxonomy, EunisTaxonomyPath
from osm_polygon_eunis.raster_reference import RasterLayer, RasterReference

_TAXONOMY = EunisTaxonomy(
    taxonomy_id="fixture-eunis",
    taxonomy_revision="fixture-v1",
    taxonomy_sha256="a" * 64,
    paths_by_direct_code={
        "C1": EunisTaxonomyPath("A", "A1", "C1"),
        "C2": EunisTaxonomyPath("A", "A2", "C2"),
        "C3": EunisTaxonomyPath("B", "B1", "C3"),
    },
)


def _write_raster(path, value: float) -> None:
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=1,
        width=1,
        count=1,
        dtype="float32",
        crs="EPSG:3035",
        transform=from_origin(0, 10, 10, 10),
    ) as dataset:
        dataset.write(np.array([[value]], dtype=np.float32), 1)


def _pin(code: str, path, *, family: str | None = "synthetic-suitability-score"):
    evidence = (
        SourceOutputSemanticsEvidence(
            family,
            "test-fixture://eunis/output-semantics",
            "b" * 64,
        )
        if family is not None
        else None
    )
    return EunisRasterAssetPin(
        code=code,
        filename=f"Prob_{code}_100m.tif",
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        output_evidence=evidence,
    )


def _manifest(*assets: EunisRasterAssetPin) -> EunisRasterAssetManifest:
    return EunisRasterAssetManifest("a" * 64, tuple(assets))


def _layer(code: str, path, *, filename: str | None = None) -> RasterLayer:
    return RasterLayer(
        code,
        code,
        path,
        "EEA fixture",
        filename or f"Prob_{code}_100m.tif",
    )


def _classes() -> dict[tuple[str, str], tuple[str, ...]]:
    return _TAXONOMY.support_groups()


def test_score_classification_rejects_rasters_without_a_pinned_source_inventory(
    tmp_path,
) -> None:
    layers = []
    for code, value in (("C1", 0.9), ("C2", 0.2), ("C3", 0.1)):
        path = tmp_path / f"Prob_{code}_10m.tif"
        _write_raster(path, value)
        layers.append(RasterLayer(code, code, path, "EEA fixture"))

    reference = RasterReference(tuple(layers))

    with reference, pytest.raises(ValueError, match="pinned complete raster asset inventory"):
        reference.classification_areas(box(0, 0, 10, 10), _classes())


def test_complete_synthetic_inventory_classifies_all_competitors(tmp_path) -> None:
    layers = []
    pins = []
    for code, value in (("C1", 0.9), ("C2", 0.2), ("C3", 0.1)):
        path = tmp_path / f"{code}.tif"
        _write_raster(path, value)
        layers.append(_layer(code, path))
        pins.append(_pin(code, path))

    reference = RasterReference(tuple(layers), asset_manifest=_manifest(*pins))
    manifest = _manifest(*pins)
    assert manifest.output_semantics_evidence_sha256 is not None
    manifest.validate_layers(
        layers,
        expected_native_manifest_sha256=manifest.native_manifest_sha256,
        expected_asset_inventory_sha256=manifest.asset_inventory_sha256,
        expected_output_semantics_sha256=manifest.output_semantics_evidence_sha256,
    )

    with reference:
        result = reference.classification_areas(box(0, 0, 10, 10), _classes())
        reference.validate_classification_inventory(
            expected_native_manifest_sha256=manifest.native_manifest_sha256,
            expected_asset_inventory_sha256=manifest.asset_inventory_sha256,
            expected_output_semantics_sha256=manifest.output_semantics_evidence_sha256,
        )
        with pytest.raises(ValueError, match="pinned native manifest"):
            reference.validate_classification_inventory(expected_native_manifest_sha256="f" * 64)

    assert result.areas_m2[("L3", "C1")] == 100.0
    assert result.areas_m2[("L3", "C2")] == 0.0
    assert result.areas_m2[("L3", "C3")] == 0.0


def test_missing_competitor_fails_against_the_complete_pinned_inventory(tmp_path) -> None:
    paths = {}
    for code, value in (("C1", 0.9), ("C2", 0.2), ("C3", 0.1)):
        path = tmp_path / f"{code}.tif"
        _write_raster(path, value)
        paths[code] = path
    reference = RasterReference(
        (_layer("C1", paths["C1"]), _layer("C2", paths["C2"])),
        asset_manifest=_manifest(*(_pin(code, path) for code, path in paths.items())),
    )

    with reference, pytest.raises(ValueError, match=r"missing=\['C3'\]"):
        reference.classification_areas(box(0, 0, 10, 10), _classes())


def test_substituted_raster_bytes_fail_before_score_classification(tmp_path) -> None:
    expected = tmp_path / "expected-C1.tif"
    actual = tmp_path / "substituted-C1.tif"
    other = tmp_path / "C2.tif"
    third = tmp_path / "C3.tif"
    for path, value in ((expected, 0.9), (actual, 0.1), (other, 0.2), (third, 0.3)):
        _write_raster(path, value)
    manifest = _manifest(_pin("C1", expected), _pin("C2", other), _pin("C3", third))
    reference = RasterReference(
        (_layer("C1", actual), _layer("C2", other), _layer("C3", third)),
        asset_manifest=manifest,
    )

    with reference, pytest.raises(ValueError, match="bytes differ from its pinned asset SHA-256"):
        reference.classification_areas(box(0, 0, 10, 10), _classes())


def test_missing_and_mixed_source_output_semantics_fail_closed(tmp_path) -> None:
    paths = {}
    for code, value in (("C1", 0.9), ("C2", 0.2), ("C3", 0.1)):
        path = tmp_path / f"{code}.tif"
        _write_raster(path, value)
        paths[code] = path
    layers = tuple(_layer(code, path) for code, path in paths.items())
    missing = _manifest(
        _pin("C1", paths["C1"], family=None),
        _pin("C2", paths["C2"]),
        _pin("C3", paths["C3"]),
    )
    mixed = _manifest(
        _pin("C1", paths["C1"], family="raw-score"),
        _pin("C2", paths["C2"], family="logistic-probability"),
        _pin("C3", paths["C3"], family="cloglog-probability"),
    )

    for manifest, expected in (
        (missing, "source-output semantics evidence is missing"),
        (mixed, "mixes source-output semantics families"),
    ):
        reference = RasterReference(layers, asset_manifest=manifest)
        with reference, pytest.raises(ValueError, match=expected):
            reference.classification_areas(box(0, 0, 10, 10), _classes())


def test_checked_in_eea_inventory_is_complete_but_lacks_output_semantics_evidence() -> None:
    manifest = load_eunis_2021_asset_manifest()

    assert len(manifest.assets) == 136
    assert manifest.asset_inventory_sha256 == (
        "d8e0a7c739a5959781018030b70dd8462375e6f12fc834fabf2b382724bb5c87"
    )
    assert manifest.output_semantics_evidence_sha256 is None


def test_official_map_release_uses_the_pinned_default_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _manifest(_dummy_pin("C1", "Prob_C1_100m.tif"))
    monkeypatch.setattr(asset_manifest, "load_eunis_2021_asset_manifest", lambda: manifest)

    with pytest.raises(ValueError, match=r"missing=\['C1'\]"):
        asset_manifest.validate_classification_inventory(
            (), asset_manifest.EEA_2021_MAP_RELEASE, None
        )


@pytest.mark.parametrize(
    ("filename", "size_bytes", "sha256", "message"),
    [
        ("Prob_C2_100m.tif", 1, "a" * 64, "filename does not match"),
        ("Prob_C1_100m.tif", 0, "a" * 64, "size must be positive"),
        ("Prob_C1_100m.tif", 1, "invalid", "needs a SHA-256"),
    ],
)
def test_asset_pins_reject_invalid_identity_values(
    filename: str,
    size_bytes: int,
    sha256: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        EunisRasterAssetPin("C1", filename, size_bytes, sha256)


def test_asset_manifests_reject_invalid_hash_empty_and_duplicate_codes() -> None:
    with pytest.raises(ValueError, match="native-manifest SHA-256"):
        EunisRasterAssetManifest("invalid", (_dummy_pin("C1", "Prob_C1_100m.tif"),))
    with pytest.raises(ValueError, match="unique direct codes"):
        EunisRasterAssetManifest("a" * 64, ())
    duplicate_code_assets = (
        _dummy_pin("C1", "Prob_C1_100m.tif"),
        _dummy_pin("C1", "Prob_C1_200m.tif"),
    )
    with pytest.raises(ValueError, match="unique direct codes"):
        EunisRasterAssetManifest("a" * 64, duplicate_code_assets)


def _dummy_pin(code: str, filename: str) -> EunisRasterAssetPin:
    return EunisRasterAssetPin(code, filename, 1, "a" * 64)


@pytest.mark.parametrize(
    ("pin_name", "message"),
    [
        ("native", "pinned native manifest"),
        ("inventory", "inventory digest differs"),
        ("semantics", "output-semantics evidence differs"),
    ],
)
def test_reference_manifest_rejects_each_mismatched_reference_pin(
    tmp_path,
    pin_name: str,
    message: str,
) -> None:
    path = tmp_path / "Prob_C1_100m.tif"
    _write_raster(path, 0.8)
    manifest = _manifest(_pin("C1", path))
    expected = {
        "native": {"expected_native_manifest_sha256": "f" * 64},
        "inventory": {"expected_asset_inventory_sha256": "f" * 64},
        "semantics": {"expected_output_semantics_sha256": "f" * 64},
    }[pin_name]
    with pytest.raises(ValueError, match=message):
        manifest.validate_layers((_layer("C1", path),), **expected)


def _write_source_config(tmp_path: Path, update) -> Path:
    config_path = Path(__file__).parents[2] / "config/eunis-source-provenance-20261021.json"
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    update(payload["reference"])
    output = tmp_path / "source-config.json"
    output.write_text(json.dumps(payload), encoding="utf-8")
    return output


@pytest.mark.parametrize(
    ("update", "error", "message"),
    [
        (
            lambda reference: reference.update(native_manifest_sha256="f" * 64),
            ValueError,
            "native manifest SHA-256",
        ),
        (
            lambda reference: reference.update(native_assets="missing"),
            TypeError,
            "no native asset list",
        ),
        (
            lambda reference: reference.update(native_assets=reference["native_assets"][:-1]),
            ValueError,
            "asset count",
        ),
        (
            lambda reference: reference["native_assets"][0].update(
                size_bytes=reference["native_assets"][0]["size_bytes"] + 1
            ),
            ValueError,
            "asset byte total",
        ),
        (
            lambda reference: reference["native_assets"][0].update(sha256="f" * 64),
            ValueError,
            "asset inventory differs",
        ),
        (
            lambda reference: reference["native_assets"][0].update(
                source_output_semantics="synthetic-score-family",
                source_output_semantics_evidence_uri="test-fixture://eunis/output",
                source_output_semantics_evidence_sha256="b" * 64,
            ),
            ValueError,
            "asset inventory differs",
        ),
        (
            lambda reference: reference["native_assets"][0].update(
                source_output_semantics="synthetic-score-family"
            ),
            ValueError,
            "incomplete source-output evidence",
        ),
        (
            lambda reference: reference["native_assets"][0].pop("filename"),
            TypeError,
            "missing code, filename, size, or SHA-256",
        ),
        (
            lambda reference: reference["native_assets"].__setitem__(0, "invalid"),
            TypeError,
            "asset entry must be an object",
        ),
    ],
)
def test_checked_in_inventory_loader_rejects_unpinned_changes(
    tmp_path: Path,
    update,
    error: type[Exception],
    message: str,
) -> None:
    config_path = _write_source_config(tmp_path, update)
    with pytest.raises(error, match=message):
        load_eunis_2021_asset_manifest(config_path)


def test_checked_in_inventory_loader_rejects_non_object_root(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text("[]", encoding="utf-8")
    with pytest.raises(TypeError, match="no reference inventory"):
        load_eunis_2021_asset_manifest(config_path)


def test_checked_in_inventory_loader_rejects_non_object_reference(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.json"
    config_path.write_text(json.dumps({"reference": []}), encoding="utf-8")
    with pytest.raises(TypeError, match="no reference inventory"):
        load_eunis_2021_asset_manifest(config_path)


def test_checked_in_inventory_loader_uses_package_resource_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_is_file = Path.is_file

    def no_source_tree_file(path: Path) -> bool:
        if path.name == "eunis-source-provenance-20261021.json":
            return False
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", no_source_tree_file)
    source_config = (
        Path(__file__).parents[2] / "config/eunis-source-provenance-20261021.json"
    ).read_text(encoding="utf-8")

    class PackageResource:
        def joinpath(self, _name: str) -> PackageResource:
            return self

        def read_text(self, *, encoding: str) -> str:
            return source_config.encode(encoding).decode(encoding)

    monkeypatch.setattr(asset_manifest.resources, "files", lambda _package: PackageResource())
    assert len(load_eunis_2021_asset_manifest().assets) == 136
