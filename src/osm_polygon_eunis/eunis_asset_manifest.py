"""Pinned identity and output semantics for EUNIS raster source assets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Protocol, cast

from .fileio import sha256_file

EEA_2021_MAP_RELEASE = "EEA EUNIS habitat probability maps v1 2021"
EEA_2021_NATIVE_MANIFEST_SHA256 = "4a91e553b2e0271e7ca935e542dd092fe45288c5f9bb4a09f40f6b74da4eae22"
EEA_2021_NATIVE_ASSET_INVENTORY_SHA256 = (
    "d8e0a7c739a5959781018030b70dd8462375e6f12fc834fabf2b382724bb5c87"
)
EEA_2021_ASSET_COUNT = 136
EEA_2021_ASSET_BYTES = 5_007_209_626
EEA_2021_OUTPUT_SEMANTICS_EVIDENCE_SHA256: str | None = None
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_LAYER_FILENAME = re.compile(r"^Prob_(?P<code>[A-Z][A-Z0-9.]+)_\d+m\.tif$")


@dataclass(frozen=True, slots=True)
class SourceOutputSemanticsEvidence:
    """Source documentation identifying one raster's score-output family."""

    family: str
    source_uri: str
    source_sha256: str

    def __post_init__(self) -> None:
        """Reject incomplete source evidence."""

        if not self.family.strip() or not self.source_uri.strip():
            raise ValueError("source-output semantics evidence needs a family and source URI")
        if not _SHA256.fullmatch(self.source_sha256):
            raise ValueError("source-output semantics evidence needs a SHA-256 digest")


@dataclass(frozen=True, slots=True)
class EunisRasterAssetPin:
    """Expected identity and source-output evidence for one EUNIS raster."""

    code: str
    filename: str
    size_bytes: int
    sha256: str
    output_evidence: SourceOutputSemanticsEvidence | None = None

    def __post_init__(self) -> None:
        """Reject invalid or unpinned raster identities."""

        match = _LAYER_FILENAME.fullmatch(self.filename)
        if match is None or match.group("code") != self.code:
            raise ValueError("EUNIS raster inventory filename does not match its direct code")
        if self.size_bytes <= 0:
            raise ValueError("EUNIS raster inventory size must be positive")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("EUNIS raster inventory needs a SHA-256 digest")


class RasterLayerIdentity(Protocol):
    """Identity fields that a raster reference must bind to one pinned asset."""

    @property
    def code(self) -> str:
        """Return the direct habitat code assigned to the raster."""

    @property
    def path(self) -> Path:
        """Return the raster file path used by the reference."""

    @property
    def asset_filename(self) -> str | None:
        """Return the source asset filename, if retained by the layer."""


@dataclass(frozen=True, slots=True)
class EunisRasterAssetManifest:
    """Complete immutable inventory of the direct EUNIS raster competitors."""

    native_manifest_sha256: str
    assets: tuple[EunisRasterAssetPin, ...]

    def __post_init__(self) -> None:
        """Reject invalid or duplicate inventory identities."""

        _validate_native_manifest_sha256(self.native_manifest_sha256)
        _validate_unique_asset_codes(self.assets)

    @property
    def asset_inventory_sha256(self) -> str:
        """Return a canonical digest for identities and per-asset output evidence."""

        return _canonical_asset_inventory_sha256(self.assets)

    @property
    def output_semantics_evidence_sha256(self) -> str | None:
        """Return the evidence-list digest only when every asset has source evidence."""

        if any(asset.output_evidence is None for asset in self.assets):
            return None
        evidence = [
            {
                "code": asset.code,
                "family": asset.output_evidence.family,
                "source_uri": asset.output_evidence.source_uri,
                "source_sha256": asset.output_evidence.source_sha256,
            }
            for asset in sorted(self.assets, key=lambda item: item.code)
            if asset.output_evidence is not None
        ]
        payload = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def validate_layers(
        self,
        layers: Iterable[RasterLayerIdentity],
        *,
        expected_native_manifest_sha256: str | None = None,
        expected_asset_inventory_sha256: str | None = None,
        expected_output_semantics_sha256: str | None = None,
    ) -> None:
        """Require exact inventory, byte identity, and one documented output family."""

        _validate_manifest_pins(
            self,
            expected_native_manifest_sha256,
            expected_asset_inventory_sha256,
            expected_output_semantics_sha256,
        )
        layers_by_code = _index_raster_layers(layers)
        assets_by_code = {asset.code: asset for asset in self.assets}
        _validate_exact_codes(assets_by_code, layers_by_code)
        _validate_asset_filenames(assets_by_code, layers_by_code)
        _validate_output_semantics(self.assets)
        _verify_asset_bytes(assets_by_code, layers_by_code)

    def validate_reference_pins(
        self,
        *,
        expected_native_manifest_sha256: str | None = None,
        expected_asset_inventory_sha256: str | None = None,
        expected_output_semantics_sha256: str | None = None,
    ) -> None:
        """Check an already byte-verified inventory against its reference pin."""

        _validate_manifest_pins(
            self,
            expected_native_manifest_sha256,
            expected_asset_inventory_sha256,
            expected_output_semantics_sha256,
        )


@dataclass(frozen=True, slots=True)
class RasterAssetReferencePins:
    """Optional release identities expected by a classification caller."""

    native_manifest_sha256: str | None = None
    asset_inventory_sha256: str | None = None
    output_semantics_evidence_sha256: str | None = None


def _validate_manifest_pins(
    manifest: EunisRasterAssetManifest,
    expected_native_manifest_sha256: str | None,
    expected_asset_inventory_sha256: str | None,
    expected_output_semantics_sha256: str | None,
) -> None:
    _validate_optional_pin(
        manifest.native_manifest_sha256,
        expected_native_manifest_sha256,
        "EUNIS raster inventory does not match the pinned native manifest",
    )
    _validate_optional_pin(
        manifest.asset_inventory_sha256,
        expected_asset_inventory_sha256,
        "EUNIS raster inventory digest differs from the reference pin",
    )
    _validate_optional_pin(
        manifest.output_semantics_evidence_sha256,
        expected_output_semantics_sha256,
        "EUNIS raster output-semantics evidence differs from the reference pin",
    )


def _validate_optional_pin(actual: str | None, expected: str | None, message: str) -> None:
    if expected is not None and actual != expected:
        raise ValueError(message)


def _validate_native_manifest_sha256(value: str) -> None:
    if not _SHA256.fullmatch(value):
        raise ValueError("EUNIS raster manifest needs a native-manifest SHA-256 digest")


def _validate_unique_asset_codes(assets: tuple[EunisRasterAssetPin, ...]) -> None:
    codes = tuple(asset.code for asset in assets)
    if not assets or len(set(codes)) != len(codes):
        raise ValueError("EUNIS raster manifest must contain unique direct codes")


def _index_raster_layers(
    layers: Iterable[RasterLayerIdentity],
) -> dict[str, RasterLayerIdentity]:
    layer_items = tuple(layers)
    layers_by_code = {layer.code: layer for layer in layer_items}
    if len(layers_by_code) != len(layer_items):
        raise ValueError("classification raster codes must be unique")
    return layers_by_code


def _validate_exact_codes(
    assets_by_code: Mapping[str, EunisRasterAssetPin],
    layers_by_code: Mapping[str, RasterLayerIdentity],
) -> None:
    missing = sorted(assets_by_code.keys() - layers_by_code.keys())
    extra = sorted(layers_by_code.keys() - assets_by_code.keys())
    if missing or extra:
        raise ValueError(
            "opened EUNIS raster codes differ from the complete pinned inventory: "
            f"missing={missing}, extra={extra}"
        )


def _validate_asset_filenames(
    assets_by_code: Mapping[str, EunisRasterAssetPin],
    layers_by_code: Mapping[str, RasterLayerIdentity],
) -> None:
    for code, asset in assets_by_code.items():
        if layers_by_code[code].asset_filename != asset.filename:
            raise ValueError(f"EUNIS raster {code} filename differs from its pinned asset")


def _validate_output_semantics(assets: Iterable[EunisRasterAssetPin]) -> None:
    asset_items = tuple(assets)
    _require_output_evidence(asset_items)
    families = _output_evidence_families(asset_items)
    if len(families) != 1:
        raise ValueError(
            f"EUNIS raster inventory mixes source-output semantics families: {sorted(families)}"
        )


def _require_output_evidence(assets: tuple[EunisRasterAssetPin, ...]) -> None:
    missing = tuple(asset.code for asset in assets if asset.output_evidence is None)
    if missing:
        raise ValueError(
            f"source-output semantics evidence is missing for EUNIS raster codes: {sorted(missing)}"
        )


def _output_evidence_families(assets: tuple[EunisRasterAssetPin, ...]) -> set[str]:
    return {cast(SourceOutputSemanticsEvidence, asset.output_evidence).family for asset in assets}


def _verify_asset_bytes(
    assets_by_code: Mapping[str, EunisRasterAssetPin],
    layers_by_code: Mapping[str, RasterLayerIdentity],
) -> None:
    for code, asset in assets_by_code.items():
        layer = layers_by_code[code]
        try:
            size_bytes = layer.path.stat().st_size
        except OSError as error:
            raise ValueError(f"EUNIS raster {code} cannot be read from its pinned path") from error
        if size_bytes != asset.size_bytes:
            raise ValueError(f"EUNIS raster {code} byte size differs from its pinned asset")
        if sha256_file(layer.path) != asset.sha256:
            raise ValueError(f"EUNIS raster {code} bytes differ from its pinned asset SHA-256")


def load_eunis_2021_asset_manifest(path: Path | None = None) -> EunisRasterAssetManifest:
    """Load and verify the checked-in 2021 EUNIS filename, size, and hash inventory."""

    reference = _reference_inventory(_read_manifest_payload(path))
    assets = _manifest_assets(reference)
    manifest = EunisRasterAssetManifest(EEA_2021_NATIVE_MANIFEST_SHA256, assets)
    _validate_eunis_asset_pins(manifest)
    return manifest


def _read_manifest_payload(path: Path | None) -> object:
    if path is not None:
        return json.loads(path.read_text(encoding="utf-8"))
    return _read_default_manifest_payload()


def _read_default_manifest_payload() -> object:
    source_tree_path = (
        Path(__file__).resolve().parents[2] / "config/eunis-source-provenance-20261021.json"
    )
    if source_tree_path.is_file():
        return json.loads(source_tree_path.read_text(encoding="utf-8"))
    resource = resources.files("osm_polygon_eunis").joinpath(
        "eunis-source-provenance-20261021.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _reference_inventory(payload: object) -> Mapping[str, object]:
    reference = payload.get("reference") if isinstance(payload, Mapping) else None
    if not isinstance(reference, Mapping):
        raise TypeError("EUNIS provenance file has no reference inventory")
    if reference.get("native_manifest_sha256") != EEA_2021_NATIVE_MANIFEST_SHA256:
        raise ValueError("EUNIS native manifest SHA-256 differs from the code pin")
    return reference


def _manifest_assets(reference: Mapping[str, object]) -> tuple[EunisRasterAssetPin, ...]:
    raw_assets = reference.get("native_assets")
    if not isinstance(raw_assets, list):
        raise TypeError("EUNIS provenance file has no native asset list")
    return tuple(_asset_from_mapping(item) for item in raw_assets)


def _validate_eunis_asset_pins(manifest: EunisRasterAssetManifest) -> None:
    assets = manifest.assets
    if len(assets) != EEA_2021_ASSET_COUNT:
        raise ValueError("EUNIS native asset count differs from the code pin")
    if sum(asset.size_bytes for asset in assets) != EEA_2021_ASSET_BYTES:
        raise ValueError("EUNIS native asset byte total differs from the code pin")
    if manifest.asset_inventory_sha256 != EEA_2021_NATIVE_ASSET_INVENTORY_SHA256:
        raise ValueError("EUNIS native asset inventory differs from the code pin")


def validate_classification_inventory(
    layers: Iterable[RasterLayerIdentity],
    source_version: str | None,
    manifest: EunisRasterAssetManifest | None,
    *,
    reference_pins: RasterAssetReferencePins | None = None,
    verify_asset_files: bool = True,
) -> EunisRasterAssetManifest:
    """Bind configured raster layers to a complete pinned inventory."""

    if manifest is None:
        manifest = _manifest_for_source_version(source_version)
    if manifest is None:
        raise ValueError("score classification requires a pinned complete raster asset inventory")
    pins = reference_pins if reference_pins is not None else RasterAssetReferencePins()
    manifest.validate_reference_pins(
        expected_native_manifest_sha256=pins.native_manifest_sha256,
        expected_asset_inventory_sha256=pins.asset_inventory_sha256,
        expected_output_semantics_sha256=pins.output_semantics_evidence_sha256,
    )
    if verify_asset_files:
        manifest.validate_layers(layers)
    return manifest


def _manifest_for_source_version(
    source_version: str | None,
) -> EunisRasterAssetManifest | None:
    if source_version == EEA_2021_MAP_RELEASE:
        return load_eunis_2021_asset_manifest()
    return None


def _asset_from_mapping(value: object) -> EunisRasterAssetPin:
    if not isinstance(value, Mapping):
        raise TypeError("EUNIS native asset entry must be an object")
    code = _required_string(value, "code")
    filename = _required_string(value, "filename")
    size_bytes = value.get("size_bytes")
    if type(size_bytes) is not int:
        raise TypeError("EUNIS native asset entry is missing code, filename, size, or SHA-256")
    sha256 = _required_string(value, "sha256")
    evidence = _output_evidence_from_mapping(value, code)
    return EunisRasterAssetPin(code, filename, size_bytes, sha256, evidence)


def _required_string(value: Mapping[str, object], field_name: str) -> str:
    if not isinstance(value.get(field_name), str):
        raise TypeError("EUNIS native asset entry is missing code, filename, size, or SHA-256")
    return cast(str, value[field_name])


def _output_evidence_from_mapping(
    value: Mapping[str, object],
    code: str,
) -> SourceOutputSemanticsEvidence | None:
    evidence_values = (
        value.get("source_output_semantics"),
        value.get("source_output_semantics_evidence_uri"),
        value.get("source_output_semantics_evidence_sha256"),
    )
    if all(item is None for item in evidence_values):
        return None
    if not all(isinstance(item, str) for item in evidence_values):
        raise ValueError(f"EUNIS raster {code} has incomplete source-output evidence")
    return SourceOutputSemanticsEvidence(
        cast(str, evidence_values[0]),
        cast(str, evidence_values[1]),
        cast(str, evidence_values[2]),
    )


def _canonical_asset_inventory_sha256(assets: Iterable[EunisRasterAssetPin]) -> str:
    rows = [
        {
            "code": asset.code,
            "filename": asset.filename,
            "size_bytes": asset.size_bytes,
            "sha256": asset.sha256,
            "output_semantics": (
                asset.output_evidence.family if asset.output_evidence is not None else None
            ),
            "output_semantics_evidence_uri": (
                asset.output_evidence.source_uri if asset.output_evidence is not None else None
            ),
            "output_semantics_evidence_sha256": (
                asset.output_evidence.source_sha256 if asset.output_evidence is not None else None
            ),
        }
        for asset in sorted(assets, key=lambda item: item.code)
    ]
    payload = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
