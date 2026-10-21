"""Pinned LUCAS footprint support for complete raw-polygon label pools.

This module never downloads a reference. The checked-in source manifest pins
the official release and taxonomy; production queries still require the caller
to provide the bytes so their size and SHA-256 can be verified.
"""

from __future__ import annotations

import hashlib
import json
import math
import numbers
import re
import sqlite3
import struct
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import loads as load_wkb

from osm_polygon_eunis.geopackage_sql import sql_identifier
from osm_polygon_eunis.pool_contract import (
    AssignmentKind,
    LabelLevel,
    ReferencePin,
    SupportSemantics,
)

from .common import (
    MeasurementSidecar,
    PoolPolygonInput,
    PoolPolygonLabelRecords,
    SupportArea,
    SupportAttempt,
    label_polygon,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPKG_HEADER_SIZE = 8
_LEGACY_GPKG_ROW_SIZE = 7
_EPSG_3035 = 3035
LUCAS_REFERENCE_ID_NAMESPACE = "lucas-footprint-id"
LUCAS_2022_GPKG_URI = (
    "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/LUCAS/"
    "LUCAS_2022_Copernicus/l2022_survey_cop_radpoly_attr.gpkg"
)
LUCAS_2022_GPKG_SHA256 = "4d240c5710866f9549712685859a5051522b223f3e1fe580dcc9e398b68b037f"
LUCAS_2022_GPKG_SIZE_BYTES = 1_121_144_832
LUCAS_2022_CODEBOOK_URI = (
    "https://ec.europa.eu/eurostat/documents/205002/13686460/C3-LUCAS-2022.pdf"
)
LUCAS_2022_CODEBOOK_SHA256 = "1b77c28c7ad70d21c8ea6a26e39029d3e60e7a0170cb60cc9ee520de19153aaf"
LUCAS_TAXONOMY_SOURCE_COMMIT = "36dfbcb8a5b35b3cff84d6405cf98ebd4f8df04f"
LUCAS_TAXONOMY_SOURCE_SHA256 = "6afc7f26c977e39844301993f45c85008c17c73f3d4c8f6fc2b5bbd5a4ab9fcc"
LUCAS_TAXONOMY_REVISION = (
    f"official-C3-2022; source={LUCAS_TAXONOMY_SOURCE_COMMIT};"
    f"source-sha256={LUCAS_TAXONOMY_SOURCE_SHA256}"
)
LUCAS_2022_LAYER = "l2022_cop_radpoly_radqs_allatr_wcol"

_LUCAS_C3_ROWS = (
    ("A00", "A10", ("A11", "A12", "A13")),
    ("A00", "A20", ("A21", "A22")),
    ("A00", "A30", ("A30",)),
    ("B00", "B10", ("B11", "B12", "B13", "B14", "B15", "B16", "B17", "B18", "B19")),
    ("B00", "B20", ("B21", "B22", "B23")),
    ("B00", "B30", ("B31", "B32", "B33", "B34", "B35", "B36", "B37")),
    ("B00", "B40", ("B41", "B42", "B43", "B44", "B45")),
    ("B00", "B50", ("B51", "B52", "B53", "B54", "B55")),
    ("B00", "B70", ("B71", "B72", "B73", "B74", "B75", "B76", "B77")),
    ("B00", "B80", ("B81", "B82", "B83", "B84")),
    ("C00", "C10", ("C10",)),
    ("C00", "C20", ("C21", "C22", "C23")),
    ("C00", "C30", ("C31", "C32", "C33")),
    ("D00", "D10", ("D10",)),
    ("D00", "D20", ("D20",)),
    ("E00", "E10", ("E10",)),
    ("E00", "E20", ("E20",)),
    ("E00", "E30", ("E30",)),
    ("F00", "F10", ("F10",)),
    ("F00", "F20", ("F20",)),
    ("F00", "F30", ("F30",)),
    ("F00", "F40", ("F40",)),
    ("G00", "G10", ("G11", "G12")),
    ("G00", "G20", ("G21", "G22")),
    ("G00", "G30", ("G30",)),
    ("G00", "G40", ("G40",)),
    ("G00", "G50", ("G50",)),
    ("H00", "H10", ("H11", "H12")),
    ("H00", "H20", ("H21", "H22", "H23")),
)

_OBSERVATION_TYPE_CODES = {
    "Field Survey, point visible <= 100 m": 1,
    "Field Survey, point visible > 100 m": 2,
    "Photo-interpretation in the field": 3,
}

LUCAS_LEVELS = (
    LabelLevel("L1", supported=True),
    LabelLevel("L2", supported=True),
    LabelLevel("L3", supported=True),
)


@dataclass(frozen=True, slots=True)
class LucasCode:
    """One codebook path from a main class through its most detailed class."""

    l1_code: str
    l2_code: str
    l3_code: str


@dataclass(frozen=True, slots=True)
class LucasTaxonomy:
    """A checked-in or otherwise verified LUCAS codebook snapshot."""

    taxonomy_id: str
    revision: str
    sha256: str
    codes: tuple[LucasCode, ...]

    def __post_init__(self) -> None:
        """Reject invalid codebook pins and inconsistent hierarchy paths."""

        if not self.taxonomy_id.strip() or not self.revision.strip():
            raise ValueError("LUCAS taxonomy identity and revision are required")
        if not _SHA256_RE.fullmatch(self.sha256):
            raise ValueError("LUCAS taxonomy SHA-256 must be a lowercase 64-digit digest")
        if not self.codes:
            raise ValueError("LUCAS taxonomy must contain code paths")
        if any(not code.l1_code or not code.l2_code or not code.l3_code for code in self.codes):
            raise ValueError("LUCAS code paths must contain L1, L2, and L3 codes")
        if len({(code.l1_code, code.l2_code, code.l3_code) for code in self.codes}) != len(
            self.codes
        ):
            raise ValueError("LUCAS taxonomy code paths must be unique")
        parents: dict[str, str] = {}
        for code in self.codes:
            previous = parents.setdefault(code.l2_code, code.l1_code)
            if previous != code.l1_code:
                raise ValueError(f"LUCAS L2 code {code.l2_code!r} has multiple L1 parents")
        l3_parents: dict[str, tuple[str, str]] = {}
        for code in self.codes:
            parent = (code.l1_code, code.l2_code)
            previous = l3_parents.setdefault(code.l3_code, parent)
            if previous != parent:
                raise ValueError(f"LUCAS L3 code {code.l3_code!r} has multiple parents")

    @classmethod
    def from_records(
        cls,
        records: Iterable[Mapping[str, str]],
        *,
        taxonomy_id: str,
        revision: str,
        sha256: str,
    ) -> LucasTaxonomy:
        """Construct a taxonomy from manifest-selected codebook columns."""

        return cls(
            taxonomy_id=taxonomy_id,
            revision=revision,
            sha256=sha256,
            codes=tuple(
                LucasCode(
                    l1_code=row["L1"].strip(),
                    l2_code=row["L2"].strip(),
                    l3_code=row["L3"].strip(),
                )
                for row in records
            ),
        )

    @classmethod
    def lucas_2022_c3(cls) -> LucasTaxonomy:
        """Return the explicit 2022 C3 map pinned to the official codebook."""
        return cls(
            taxonomy_id="lucas-2022-c3",
            revision=LUCAS_TAXONOMY_REVISION,
            sha256=LUCAS_2022_CODEBOOK_SHA256,
            codes=tuple(
                LucasCode(main, second, observed)
                for main, second, observed_codes in _LUCAS_C3_ROWS
                for observed in observed_codes
            ),
        )

    def validate_path(self, l1_code: str | None, l2_code: str | None, l3_code: str | None) -> None:
        """Require each supplied observed code to follow the pinned hierarchy."""

        if l1_code is None:
            if l2_code is not None or l3_code is not None:
                raise ValueError("LUCAS child code is present without its L1 parent")
            return
        matching_l1 = [row for row in self.codes if row.l1_code == l1_code]
        if not matching_l1:
            raise ValueError(f"unknown LUCAS L1 code {l1_code!r}")
        if l2_code is None:
            if l3_code is not None:
                raise ValueError("LUCAS L3 code is present without its L2 parent")
            return
        matching_l2 = [row for row in matching_l1 if row.l2_code == l2_code]
        if not matching_l2:
            raise ValueError(f"unknown LUCAS L2 code path {l1_code!r}/{l2_code!r}")
        if l3_code is not None and not any(row.l3_code == l3_code for row in matching_l2):
            raise ValueError(f"unknown LUCAS L3 code path {l1_code!r}/{l2_code!r}/{l3_code!r}")

    def path_for_observed_code(self, observed_code: str) -> LucasCode | None:
        """Resolve one exact observed C3 code without prefix inference."""
        return next((row for row in self.codes if row.l3_code == observed_code), None)


@dataclass(frozen=True, slots=True)
class LucasLayerManifest:
    """Exact table, geometry, ID, code, and RTree names for one GPKG layer."""

    table: str
    geometry_column: str
    primary_key: str
    rtree_table: str
    support_class_column: str
    survey_year_column: str = "survey_year"
    point_copernicus_column: str = "point_copernicus"
    survey_can_do_column: str = "surveycprncando"
    survey_obs_type_column: str = "survey_obs_type"
    geometry_type: str = "MULTISURFACE"
    srs_id: int = _EPSG_3035

    def __post_init__(self) -> None:
        """Require the reference layer's pinned equal-area CRS."""

        if self.srs_id != _EPSG_3035:
            raise ValueError("LUCAS survey footprints must be in EPSG:3035")
        for name in (
            self.table,
            self.geometry_column,
            self.primary_key,
            self.rtree_table,
            self.support_class_column,
            self.survey_year_column,
            self.point_copernicus_column,
            self.survey_can_do_column,
            self.survey_obs_type_column,
            self.geometry_type,
        ):
            if not name or "\x00" in name:
                raise ValueError("LUCAS layer manifest identifiers must be non-empty")


@dataclass(frozen=True, slots=True)
class LucasReferenceManifest:
    """Authoritative pins required before a LUCAS reference can be opened."""

    release_id: str
    source_uri: str
    source_revision: str
    file_size_bytes: int
    file_sha256: str
    taxonomy_id: str
    taxonomy_uri: str
    taxonomy_revision: str
    taxonomy_sha256: str
    layer: LucasLayerManifest

    def __post_init__(self) -> None:
        """Require complete release, file, and codebook pins."""

        if any(
            not value.strip()
            for value in (
                self.release_id,
                self.source_uri,
                self.source_revision,
                self.taxonomy_id,
                self.taxonomy_uri,
                self.taxonomy_revision,
            )
        ):
            raise ValueError("LUCAS reference manifest pins must be present")
        if self.file_size_bytes <= 0:
            raise ValueError("LUCAS reference size must be positive")
        if not _SHA256_RE.fullmatch(self.file_sha256) or not _SHA256_RE.fullmatch(
            self.taxonomy_sha256
        ):
            raise ValueError("LUCAS reference and codebook require full SHA-256 digests")


def lucas_2022_copernicus_manifest() -> LucasReferenceManifest:
    """Return source pins from the verified LUCAS 2022 source manifest."""
    return LucasReferenceManifest(
        release_id="lucas-2022-copernicus-radial-polygon-support",
        source_uri=LUCAS_2022_GPKG_URI,
        source_revision=(
            "modified=2024-06-26T20:19:25Z;etag=42d35000-61bd0bb0d0770;"
            f"sha256={LUCAS_2022_GPKG_SHA256}"
        ),
        file_size_bytes=LUCAS_2022_GPKG_SIZE_BYTES,
        file_sha256=LUCAS_2022_GPKG_SHA256,
        taxonomy_id="lucas-2022-c3",
        taxonomy_uri=LUCAS_2022_CODEBOOK_URI,
        taxonomy_revision=LUCAS_TAXONOMY_REVISION,
        taxonomy_sha256=LUCAS_2022_CODEBOOK_SHA256,
        layer=LucasLayerManifest(
            table=LUCAS_2022_LAYER,
            geometry_column="geom",
            primary_key="fid",
            rtree_table=f"rtree_{LUCAS_2022_LAYER}_geom",
            support_class_column="surveycprnlc",
        ),
    )


def lucas_2022_reference_pin(
    manifest: LucasReferenceManifest | None = None,
) -> ReferencePin:
    """Build the shared-core reference pin from the verified source manifest."""
    source = manifest or lucas_2022_copernicus_manifest()
    return ReferencePin(
        release_id=source.release_id,
        source_repository="European Commission JRC LUCAS Copernicus",
        source_revision=source.source_revision,
        taxonomy_id=source.taxonomy_id,
        taxonomy_revision=source.taxonomy_revision,
        support_semantics=SupportSemantics.SURVEYED_FOOTPRINT,
        support_unit="m^2",
        threshold=0.8,
        support_union_policy="independent_union_per_level_and_class:v1",
    )


@dataclass(frozen=True, slots=True)
class LucasFootprint:
    """One observed survey-footprint geometry and its source-native codes."""

    footprint_id: str
    l1_code: str | None
    l2_code: str | None
    l3_code: str | None
    geometry: BaseGeometry


@dataclass(frozen=True, slots=True)
class LucasFeatureQuery:
    """Reference features returned by a spatial-index query and its receipt."""

    footprints: tuple[LucasFootprint, ...]
    attempted_feature_ids: tuple[str, ...]
    exclusion_reason: str | None = None
    partial_reason: str | None = None
    attempt_reason: str | None = None
    preaggregated_supports: tuple[SupportArea, ...] | None = None
    attempted_feature_sidecar: MeasurementSidecar | None = None


@dataclass(frozen=True, slots=True)
class LucasResourceLimits:
    """Fail-closed ceilings for one source-polygon reference query."""

    max_candidate_features: int = 100_000
    max_geometry_blob_bytes: int = 16 * 1024 * 1024
    max_footprint_coordinates: int = 100_000
    max_total_input_coordinates: int = 2_000_000
    max_union_coordinates: int = 2_000_000
    max_overlay_operand_coordinates: int = 500_000

    def __post_init__(self) -> None:
        """Reject zero or negative ceilings before any reference is queried."""
        if any(
            value <= 0
            for value in (
                self.max_candidate_features,
                self.max_geometry_blob_bytes,
                self.max_footprint_coordinates,
                self.max_total_input_coordinates,
                self.max_union_coordinates,
                self.max_overlay_operand_coordinates,
            )
        ):
            raise ValueError("LUCAS resource limits must be positive")


class LucasResourceLimitError(RuntimeError):
    """A reference query hit an explicit geometry or scan resource ceiling."""


DEFAULT_LUCAS_RESOURCE_LIMITS = LucasResourceLimits()


class LucasGeoPackageReader:
    """Read a manifest-pinned EPSG:3035 footprint layer without downloading it."""

    def __init__(
        self,
        path: Path,
        taxonomy_path: Path,
        manifest: LucasReferenceManifest,
        taxonomy: LucasTaxonomy,
        *,
        fetch_size: int = 512,
        resource_limits: LucasResourceLimits = DEFAULT_LUCAS_RESOURCE_LIMITS,
    ) -> None:
        """Prepare a read-only reader without downloading reference data."""

        if fetch_size <= 0:
            raise ValueError("GeoPackage query fetch size must be positive")
        if taxonomy.taxonomy_id != manifest.taxonomy_id:
            raise ValueError("LUCAS taxonomy ID does not match the source manifest")
        if taxonomy.revision != manifest.taxonomy_revision:
            raise ValueError("LUCAS taxonomy revision does not match the source manifest")
        if taxonomy.sha256 != manifest.taxonomy_sha256:
            raise ValueError("LUCAS taxonomy hash does not match the source manifest")
        self.path = path
        self.taxonomy_path = taxonomy_path
        self.manifest = manifest
        self.taxonomy = taxonomy
        self.fetch_size = fetch_size
        self.resource_limits = resource_limits
        self._connection: sqlite3.Connection | None = None
        self._sidecar_directory: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> LucasGeoPackageReader:
        """Verify supplied file pins and open the manifest-selected layer."""

        if self._connection is not None:
            raise RuntimeError("LUCAS GeoPackage reader is already open")
        verify_lucas_file(self.path, self.manifest)
        verify_lucas_taxonomy_file(self.taxonomy_path, self.manifest)
        self._connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        sidecar_directory = tempfile.TemporaryDirectory(prefix="lucas-reference-evidence-")
        self._sidecar_directory = sidecar_directory
        try:
            self._validate_schema()
        except Exception:
            self._connection.close()
            self._connection = None
            sidecar_directory.cleanup()
            self._sidecar_directory = None
            raise
        return self

    @property
    def is_open(self) -> bool:
        """Whether the manifest-selected reference layer is ready for queries."""

        return self._connection is not None

    def __exit__(self, *_exc: object) -> None:
        """Close the read-only SQLite connection."""

        if self._connection is not None:
            self._connection.close()
            self._connection = None
        if self._sidecar_directory is not None:
            self._sidecar_directory.cleanup()
            self._sidecar_directory = None

    def query(self, geometry: BaseGeometry) -> LucasFeatureQuery:
        """Query all RTree candidates, retaining IDs even for zero-area touches."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("LUCAS GeoPackage reader must be opened before querying")
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("LUCAS query geometry must be valid and non-empty")
        if shapely.get_num_coordinates(geometry) > self.resource_limits.max_total_input_coordinates:
            return LucasFeatureQuery(
                (), (), partial_reason="lucas_source_polygon_coordinate_limit_exceeded"
            )
        sidecar_directory = self._sidecar_directory
        if sidecar_directory is None:
            raise RuntimeError("LUCAS reader sidecar workspace is not open")
        try:
            cursor = _query_rtree_candidates(
                connection,
                self.manifest.layer,
                geometry.bounds,
                max_geometry_blob_bytes=self.resource_limits.max_geometry_blob_bytes,
            )
        except sqlite3.DatabaseError:
            return LucasFeatureQuery((), (), "lucas_reference_query_failed:DatabaseError")
        scan = _scan_rtree_candidates(
            cursor,
            _LucasScanContext(
                self.fetch_size,
                geometry,
                self.taxonomy,
                self.manifest.layer,
                Path(sidecar_directory.name),
                self.resource_limits,
            ),
        )
        filter_summary = _reason_summary("filtered", scan.filter_counts)
        partial_summary = _reason_summary("partial", scan.partial_counts)
        attempt_reason = (
            ";".join(
                summary for summary in (filter_summary, partial_summary) if summary is not None
            )
            or None
        )
        if scan.scan_failed:
            return LucasFeatureQuery(
                (),
                (),
                "lucas_reference_query_failed:DatabaseError",
                attempt_reason=attempt_reason,
                attempted_feature_sidecar=scan.attempted_sidecar,
            )
        if scan.resource_failure is not None:
            return LucasFeatureQuery(
                (),
                (),
                partial_reason=scan.resource_failure,
                attempt_reason=attempt_reason,
                attempted_feature_sidecar=scan.attempted_sidecar,
            )
        return LucasFeatureQuery(
            (),
            (),
            partial_reason=partial_summary,
            attempt_reason=attempt_reason,
            preaggregated_supports=scan.supports,
            attempted_feature_sidecar=scan.attempted_sidecar,
        )

    def _validate_schema(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("LUCAS GeoPackage connection is not open")
        layer = self.manifest.layer
        contents = connection.execute(
            "SELECT data_type, srs_id FROM gpkg_contents WHERE table_name=?",
            (layer.table,),
        ).fetchone()
        if contents != ("features", layer.srs_id):
            raise ValueError("LUCAS GeoPackage contents do not match the layer manifest")
        metadata = connection.execute(
            "SELECT geometry_type_name, srs_id, z, m FROM gpkg_geometry_columns "
            "WHERE table_name=? AND column_name=?",
            (layer.table, layer.geometry_column),
        ).fetchone()
        if metadata != (layer.geometry_type, layer.srs_id, 0, 0):
            raise ValueError("LUCAS GeoPackage geometry metadata does not match its manifest")
        spatial_ref = connection.execute(
            "SELECT organization, organization_coordsys_id FROM gpkg_spatial_ref_sys "
            "WHERE srs_id=?",
            (layer.srs_id,),
        ).fetchone()
        if spatial_ref != ("EPSG", layer.srs_id):
            raise ValueError("LUCAS GeoPackage coordinate reference does not match EPSG:3035")
        column_rows = connection.execute(
            f"PRAGMA table_info({sql_identifier(layer.table)})"
        ).fetchall()
        columns = {row[1] for row in column_rows}
        required = {
            layer.primary_key,
            layer.geometry_column,
            layer.support_class_column,
            layer.survey_year_column,
            layer.point_copernicus_column,
            layer.survey_can_do_column,
            layer.survey_obs_type_column,
        }
        if not required.issubset(columns):
            raise ValueError("LUCAS GeoPackage columns do not match its manifest")
        if not any(row[1] == layer.primary_key and row[5] > 0 for row in column_rows):
            raise ValueError("LUCAS GeoPackage primary key does not match its manifest")
        rtree_exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (layer.rtree_table,)
        ).fetchone()
        if rtree_exists is None:
            raise ValueError("LUCAS GeoPackage reference layer has no declared RTree")
        rtree_columns = {
            row[1]
            for row in connection.execute(f"PRAGMA table_info({sql_identifier(layer.rtree_table)})")
        }
        if not {"id", "minx", "maxx", "miny", "maxy"}.issubset(rtree_columns):
            raise ValueError("LUCAS GeoPackage RTree columns do not match its manifest")


def verify_lucas_file(path: Path, manifest: LucasReferenceManifest) -> None:
    """Verify exact size and streaming SHA-256 before using the GPKG."""

    if not path.is_file():
        raise FileNotFoundError(f"pinned LUCAS reference is not present: {path}")
    if path.stat().st_size != manifest.file_size_bytes:
        raise ValueError("LUCAS reference byte size does not match authoritative manifest")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != manifest.file_sha256:
        raise ValueError("LUCAS reference SHA-256 does not match authoritative manifest")


def verify_lucas_taxonomy_file(path: Path, manifest: LucasReferenceManifest) -> None:
    """Verify the exact codebook bytes supplied to the taxonomy parser."""

    if not path.is_file():
        raise FileNotFoundError(f"pinned LUCAS taxonomy is not present: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != manifest.taxonomy_sha256:
        raise ValueError("LUCAS taxonomy SHA-256 does not match authoritative manifest")


def _eligibility_reasons(
    survey_year: object,
    point_copernicus: object,
    survey_can_do: object,
    survey_obs_type: object,
    support_class: object,
) -> tuple[str, ...]:
    reasons = []
    if not _integer_equals(survey_year, 2022):
        reasons.append("survey_year")
    if not _integer_equals(point_copernicus, 1):
        reasons.append("point_copernicus")
    if not _integer_equals(survey_can_do, 1):
        reasons.append("surveycprncando")
    observation_code = (
        _OBSERVATION_TYPE_CODES.get(survey_obs_type)
        if isinstance(survey_obs_type, str)
        else survey_obs_type
    )
    if not _integer_equals(observation_code, 1):
        reasons.append("survey_obs_type")
    if support_class is None:
        reasons.append("surveycprnlc")
    return tuple(reasons)


def _integer_equals(value: object, expected: int) -> bool:
    return (
        isinstance(value, numbers.Integral)
        and not isinstance(value, bool)
        and int(value) == expected
    )


def _reason_summary(prefix: str, counts: Mapping[str, int]) -> str | None:
    if not counts:
        return None
    values = ",".join(f"{name}={count}" for name, count in sorted(counts.items()))
    return f"{prefix}:{values}"


def _query_rtree_candidates(
    connection: sqlite3.Connection,
    layer: LucasLayerManifest,
    bounds: tuple[float, float, float, float],
    *,
    max_geometry_blob_bytes: int,
) -> sqlite3.Cursor:
    """Open a bounded query over the manifest-selected RTree."""
    table = sql_identifier(layer.table)
    geometry = sql_identifier(layer.geometry_column)
    primary_key = sql_identifier(layer.primary_key)
    rtree = sql_identifier(layer.rtree_table)
    columns = tuple(
        sql_identifier(name)
        for name in (
            layer.support_class_column,
            layer.survey_year_column,
            layer.point_copernicus_column,
            layer.survey_can_do_column,
            layer.survey_obs_type_column,
        )
    )
    query = (
        f"SELECT t.{primary_key}, t.{columns[0]}, "  # noqa: S608 -- pinned manifest identifiers
        f"CASE WHEN length(t.{geometry}) <= ? THEN t.{geometry} ELSE NULL END, "
        f"length(t.{geometry}), t.{columns[1]}, "
        f"t.{columns[2]}, t.{columns[3]}, t.{columns[4]} "
        f"FROM {table} AS t JOIN {rtree} AS r ON r.id=t.{primary_key} "
        "WHERE r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"
    )
    min_x, min_y, max_x, max_y = bounds
    return connection.execute(query, (max_geometry_blob_bytes, min_x, max_x, min_y, max_y))


@dataclass(frozen=True, slots=True)
class _LucasScanContext:
    fetch_size: int
    geometry: BaseGeometry
    taxonomy: LucasTaxonomy
    layer: LucasLayerManifest
    sidecar_directory: Path
    resource_limits: LucasResourceLimits


@dataclass(frozen=True, slots=True)
class _LucasScanResult:
    attempted_sidecar: MeasurementSidecar | None
    supports: tuple[SupportArea, ...]
    filter_counts: dict[str, int]
    partial_counts: dict[str, int]
    scan_failed: bool
    resource_failure: str | None


class _LucasScanState:
    """Consume bounded candidate rows while streaming receipts and support IDs."""

    def __init__(self, context: _LucasScanContext) -> None:
        self.context = context
        self.attempted_writer = _JsonlReferenceIdWriter(context.sidecar_directory, "attempted")
        self.supports = _LucasSupportAccumulator(
            context.geometry,
            context.taxonomy,
            sidecar_directory=context.sidecar_directory,
            resource_limits=context.resource_limits,
        )
        self.filter_counts: defaultdict[str, int] = defaultdict(int)
        self.partial_counts: defaultdict[str, int] = defaultdict(int)
        self.scanned_features = 0
        self.resource_failure: str | None = None

    def next_fetch_size(self) -> int:
        remaining = self.context.resource_limits.max_candidate_features - self.scanned_features
        return min(self.context.fetch_size, remaining + 1)

    def process(self, row: tuple[object, ...]) -> bool:
        """Process one row, returning false if the candidate scan must stop."""
        limits = self.context.resource_limits
        if self.scanned_features >= limits.max_candidate_features:
            self.attempted_writer.write_id(str(row[0]))
            self.resource_failure = "lucas_candidate_feature_limit_exceeded"
            return False
        parsed = _parse_reference_row(
            row,
            self.context.geometry,
            self.context.taxonomy,
            self.context.layer,
            resource_limits=limits,
        )
        footprint_id, footprint, filtered_reason, partial_reason = parsed
        self.attempted_writer.write_id(footprint_id)
        self.scanned_features += 1
        if filtered_reason is not None:
            self.filter_counts[filtered_reason] += 1
        if partial_reason is not None:
            self.partial_counts[partial_reason] += 1
        if footprint is not None:
            try:
                self.supports.add(footprint)
            except LucasResourceLimitError as error:
                self.resource_failure = f"lucas_resource_limit:{error}"
                return False
            except (ValueError, shapely.errors.GEOSException) as error:
                self.resource_failure = f"lucas_measurement_incomplete:{type(error).__name__}"
                return False
        return True

    def finish(self, *, scan_failed: bool) -> _LucasScanResult:
        if scan_failed or self.resource_failure is not None or self.partial_counts:
            self.supports.discard()
            supports: tuple[SupportArea, ...] = ()
        else:
            try:
                supports = self.supports.finish()
            except (LucasResourceLimitError, ValueError, shapely.errors.GEOSException) as error:
                reason = (
                    "lucas_resource_limit"
                    if isinstance(error, LucasResourceLimitError)
                    else "lucas_measurement_incomplete"
                )
                self.resource_failure = f"{reason}:{type(error).__name__}"
                self.supports.discard()
                supports = ()
        return _LucasScanResult(
            attempted_sidecar=self.attempted_writer.finish(),
            supports=supports,
            filter_counts=dict(self.filter_counts),
            partial_counts=dict(self.partial_counts),
            scan_failed=scan_failed,
            resource_failure=self.resource_failure,
        )


def _scan_rtree_candidates(cursor: sqlite3.Cursor, context: _LucasScanContext) -> _LucasScanResult:
    state = _LucasScanState(context)
    scan_failed = False
    while True:
        try:
            rows = cursor.fetchmany(state.next_fetch_size())
        except sqlite3.DatabaseError:
            scan_failed = True
            break
        if not rows:
            break
        if any(not state.process(row) for row in rows):
            break
    return state.finish(scan_failed=scan_failed)


def _parse_reference_row(
    row: tuple[object, ...],
    geometry: BaseGeometry,
    taxonomy: LucasTaxonomy,
    layer: LucasLayerManifest,
    *,
    resource_limits: LucasResourceLimits = DEFAULT_LUCAS_RESOURCE_LIMITS,
) -> tuple[str, LucasFootprint | None, str | None, str | None]:
    """Apply source filters, exact C3 lookup, and geometry checks to one row."""
    if len(row) == _LEGACY_GPKG_ROW_SIZE:
        raw_id, raw_code, raw_blob, year, copernicus, can_do, observation_type = row
        raw_blob_bytes = len(raw_blob) if isinstance(raw_blob, (bytes, memoryview)) else None
    else:
        raw_id, raw_code, raw_blob, raw_blob_bytes, year, copernicus, can_do, observation_type = row
    feature_id = str(raw_id)
    rejected = _eligibility_reasons(year, copernicus, can_do, observation_type, raw_code)
    if rejected:
        return feature_id, None, "+".join(rejected), None
    if not isinstance(raw_code, str):
        return feature_id, None, None, "support_class_type_invalid"
    code_path = taxonomy.path_for_observed_code(raw_code)
    if code_path is None:
        return feature_id, None, "support_class_not_in_c3", None
    partial_reason = None
    if (
        isinstance(raw_blob_bytes, numbers.Integral)
        and int(raw_blob_bytes) > resource_limits.max_geometry_blob_bytes
    ):
        return feature_id, None, None, "reference_geometry_blob_limit_exceeded"
    try:
        footprint_geometry = _decode_gpkg_geometry(raw_blob, layer.srs_id)
    except (TypeError, ValueError, struct.error, shapely.errors.GEOSException):
        partial_reason = "reference_geometry_unusable"
        footprint_geometry = None
    if footprint_geometry is not None and not _is_positive_polygon(footprint_geometry):
        partial_reason = "reference_geometry_unusable"
    if (
        footprint_geometry is not None
        and shapely.get_num_coordinates(footprint_geometry)
        > resource_limits.max_footprint_coordinates
    ):
        partial_reason = "reference_geometry_coordinate_limit_exceeded"
    if partial_reason is not None:
        return feature_id, None, None, partial_reason
    footprint = (
        LucasFootprint(
            footprint_id=feature_id,
            l1_code=code_path.l1_code,
            l2_code=code_path.l2_code,
            l3_code=code_path.l3_code,
            geometry=footprint_geometry,
        )
        if footprint_geometry is not None and footprint_geometry.intersects(geometry)
        else None
    )
    return (
        feature_id,
        footprint,
        None,
        None,
    )


def _is_positive_polygon(geometry: BaseGeometry) -> bool:
    return (
        isinstance(geometry, Polygon | MultiPolygon)
        and not geometry.is_empty
        and geometry.is_valid
        and geometry.area > 0.0
    )


def measure_lucas_supports(
    polygon: PoolPolygonInput,
    query: LucasFeatureQuery,
    taxonomy: LucasTaxonomy,
    reference_pin: ReferencePin,
) -> PoolPolygonLabelRecords:
    """Union footprint supports independently at L1, L2, and L3 before thresholding."""

    if reference_pin.support_unit != "m^2":
        raise ValueError("LUCAS survey support must use square metres in EPSG:3035")
    if polygon.geometry is None or polygon.snapshot is None:
        attempt = SupportAttempt((), "geometry_unusable")
    elif query.exclusion_reason is not None:
        attempt = SupportAttempt(
            (),
            query.exclusion_reason,
            attempted_reference_ids=query.attempted_feature_ids,
            attempt_reason=query.attempt_reason,
            attempted_reference_sidecar=query.attempted_feature_sidecar,
        )
    elif query.partial_reason is not None:
        attempt = SupportAttempt(
            (),
            query.partial_reason,
            attempted_reference_ids=query.attempted_feature_ids,
            partial_reason=query.partial_reason,
            attempt_reason=query.attempt_reason,
            attempted_reference_sidecar=query.attempted_feature_sidecar,
        )
    elif (
        polygon.snapshot.original_area is None
        or not math.isfinite(polygon.snapshot.original_area)
        or polygon.snapshot.original_area <= 0.0
    ):
        attempt = SupportAttempt(
            (),
            "geometry_area_unusable",
            attempted_reference_ids=query.attempted_feature_ids,
            attempt_reason=query.attempt_reason,
            attempted_reference_sidecar=query.attempted_feature_sidecar,
        )
    else:
        original = polygon.geometry
        if polygon.snapshot.crs != "EPSG:3035" or polygon.snapshot.area_unit != "m^2":
            raise ValueError("LUCAS polygon snapshots must use EPSG:3035 square metres")
        if polygon.snapshot.original_area is not None and not math.isclose(
            float(shapely.area(original)),
            polygon.snapshot.original_area,
            rel_tol=1e-9,
            abs_tol=1e-9,
        ):
            raise ValueError("LUCAS snapshot area must be the full original polygon area")
        if not original.is_valid or original.is_empty:
            attempt = SupportAttempt(
                (),
                "geometry_unusable",
                attempted_reference_ids=query.attempted_feature_ids,
                attempt_reason=query.attempt_reason,
                attempted_reference_sidecar=query.attempted_feature_sidecar,
            )
        else:
            supports = query.preaggregated_supports
            if supports is None:
                supports = _union_by_level(original, query.footprints, taxonomy)
            attempt = SupportAttempt(
                supports,
                attempted_reference_ids=query.attempted_feature_ids,
                attempt_reason=query.attempt_reason,
                attempted_reference_sidecar=query.attempted_feature_sidecar,
            )
    return label_polygon(
        polygon,
        attempt,
        LUCAS_LEVELS,
        reference_pin,
        reference_id_namespace=LUCAS_REFERENCE_ID_NAMESPACE,
    )


def _union_by_level(
    original: BaseGeometry,
    footprints: Sequence[LucasFootprint],
    taxonomy: LucasTaxonomy,
) -> tuple[SupportArea, ...]:
    accumulator = _LucasSupportAccumulator(original, taxonomy)
    for footprint in footprints:
        accumulator.add(footprint)
    return accumulator.finish()


class _SupportUnionForest:
    """Merge incoming support geometries with logarithmic retained state."""

    def __init__(self) -> None:
        self._levels: list[BaseGeometry | None] = []

    @property
    def retained_geometry_count(self) -> int:
        return sum(geometry is not None for geometry in self._levels)

    @property
    def retained_coordinate_count(self) -> int:
        return sum(
            int(shapely.get_num_coordinates(geometry))
            for geometry in self._levels
            if geometry is not None
        )

    def add(
        self,
        geometry: BaseGeometry,
        *,
        max_union_coordinates: int | None = None,
        max_overlay_operand_coordinates: int | None = None,
    ) -> None:
        level = 0
        while level < len(self._levels) and self._levels[level] is not None:
            previous = self._levels[level]
            if previous is None:
                raise RuntimeError("support union forest has inconsistent levels")
            operand_coordinates = int(shapely.get_num_coordinates(previous)) + int(
                shapely.get_num_coordinates(geometry)
            )
            if (
                max_overlay_operand_coordinates is not None
                and operand_coordinates > max_overlay_operand_coordinates
            ):
                raise LucasResourceLimitError("union overlay operand coordinate limit exceeded")
            geometry = unary_union((previous, geometry))
            output_coordinates = int(shapely.get_num_coordinates(geometry))
            if max_union_coordinates is not None and output_coordinates > max_union_coordinates:
                raise LucasResourceLimitError("union output coordinate limit exceeded")
            self._levels[level] = None
            level += 1
        if (
            max_union_coordinates is not None
            and int(shapely.get_num_coordinates(geometry)) > max_union_coordinates
        ):
            raise LucasResourceLimitError("union output coordinate limit exceeded")
        if level == len(self._levels):
            self._levels.append(geometry)
        else:
            self._levels[level] = geometry

    def finish(
        self,
        *,
        max_union_coordinates: int | None = None,
        max_overlay_operand_coordinates: int | None = None,
    ) -> BaseGeometry:
        geometries = [geometry for geometry in self._levels if geometry is not None]
        if not geometries:
            raise ValueError("cannot finish an empty support union")
        self._levels.clear()
        geometry = geometries[0]
        for next_geometry in geometries[1:]:
            operand_coordinates = int(shapely.get_num_coordinates(geometry)) + int(
                shapely.get_num_coordinates(next_geometry)
            )
            if (
                max_overlay_operand_coordinates is not None
                and operand_coordinates > max_overlay_operand_coordinates
            ):
                raise LucasResourceLimitError(
                    "final union overlay operand coordinate limit exceeded"
                )
            geometry = unary_union((geometry, next_geometry))
            if (
                max_union_coordinates is not None
                and int(shapely.get_num_coordinates(geometry)) > max_union_coordinates
            ):
                raise LucasResourceLimitError("final union output coordinate limit exceeded")
        if (
            max_union_coordinates is not None
            and int(shapely.get_num_coordinates(geometry)) > max_union_coordinates
        ):
            raise LucasResourceLimitError("final union output coordinate limit exceeded")
        return geometry


class _JsonlReferenceIdWriter:
    """Write one JSON string per line while maintaining a streaming digest."""

    def __init__(self, directory: Path, prefix: str) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(  # noqa: SIM115 -- kept open until finish/discard
            mode="wb", prefix=f"{prefix}-", suffix=".jsonl", dir=directory, delete=False
        )
        self.path = Path(handle.name)
        self._handle = handle
        self._digest = hashlib.sha256()
        self.record_count = 0

    def write_id(self, reference_id: str) -> None:
        line = (json.dumps(reference_id, ensure_ascii=False, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        self._handle.write(line)
        self._digest.update(line)
        self.record_count += 1

    def finish(self) -> MeasurementSidecar | None:
        self._handle.flush()
        self._handle.close()
        if self.record_count == 0:
            self.path.unlink(missing_ok=True)
            return None
        return MeasurementSidecar(
            sha256=self._digest.hexdigest(),
            record_count=self.record_count,
            path=self.path,
        )

    def discard(self) -> None:
        if not self._handle.closed:
            self._handle.close()
        self.path.unlink(missing_ok=True)


class _LucasSupportAccumulator:
    """Union each hierarchy class as footprint rows arrive, retaining no row geometries."""

    def __init__(
        self,
        original: BaseGeometry,
        taxonomy: LucasTaxonomy,
        *,
        sidecar_directory: Path | None = None,
        resource_limits: LucasResourceLimits | None = None,
    ) -> None:
        self._original = original
        self._taxonomy = taxonomy
        self._sidecar_directory = sidecar_directory
        self._resource_limits = resource_limits
        self._unions: dict[tuple[str, str], _SupportUnionForest] = {}
        self._evidence_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
        self._evidence_writers: dict[tuple[str, str], _JsonlReferenceIdWriter] = {}
        self._total_input_coordinates = 0
        self._retained_union_coordinates = 0
        self.peak_retained_union_geometry_count = 0

    def add(self, footprint: LucasFootprint) -> None:
        self._taxonomy.validate_path(footprint.l1_code, footprint.l2_code, footprint.l3_code)
        footprint_coordinates = int(shapely.get_num_coordinates(footprint.geometry))
        self._check_footprint_limits(footprint_coordinates)
        clipped = footprint.geometry.intersection(self._original)
        if clipped.is_empty or float(shapely.area(clipped)) <= 0.0:
            return
        for level, code in (
            ("L1", footprint.l1_code),
            ("L2", footprint.l2_code),
            ("L3", footprint.l3_code),
        ):
            if code is None:
                continue
            self._add_class_support((level, code), footprint, footprint_coordinates)
        retained = sum(forest.retained_geometry_count for forest in self._unions.values())
        self.peak_retained_union_geometry_count = max(
            self.peak_retained_union_geometry_count, retained
        )

    def _check_footprint_limits(self, footprint_coordinates: int) -> None:
        limits = self._resource_limits
        if limits is None:
            return
        if footprint_coordinates > limits.max_footprint_coordinates:
            raise LucasResourceLimitError("footprint coordinate limit exceeded")
        if (
            footprint_coordinates + int(shapely.get_num_coordinates(self._original))
            > limits.max_overlay_operand_coordinates
        ):
            raise LucasResourceLimitError("footprint clip operand coordinate limit exceeded")
        self._total_input_coordinates += footprint_coordinates
        if self._total_input_coordinates > limits.max_total_input_coordinates:
            raise LucasResourceLimitError("total input coordinate limit exceeded")

    def _add_class_support(
        self, key: tuple[str, str], footprint: LucasFootprint, footprint_coordinates: int
    ) -> None:
        limits = self._resource_limits
        forest = self._unions.setdefault(key, _SupportUnionForest())
        if limits is not None and (
            self._retained_union_coordinates + footprint_coordinates > limits.max_union_coordinates
        ):
            raise LucasResourceLimitError("retained union coordinate limit exceeded")
        previous_coordinates = forest.retained_coordinate_count
        forest.add(
            footprint.geometry,
            max_union_coordinates=(limits.max_union_coordinates if limits else None),
            max_overlay_operand_coordinates=(
                limits.max_overlay_operand_coordinates if limits else None
            ),
        )
        self._retained_union_coordinates += forest.retained_coordinate_count - previous_coordinates
        if limits is not None and self._retained_union_coordinates > limits.max_union_coordinates:
            raise LucasResourceLimitError("retained union coordinate limit exceeded")
        self._write_support_id(key, footprint.footprint_id)

    def _write_support_id(self, key: tuple[str, str], reference_id: str) -> None:
        if self._sidecar_directory is None:
            self._evidence_ids[key].append(reference_id)
            return
        writer = self._evidence_writers.get(key)
        if writer is None:
            level_code = hashlib.sha256(f"{key[0]}:{key[1]}".encode()).hexdigest()
            writer = _JsonlReferenceIdWriter(self._sidecar_directory, f"support-{level_code}")
            self._evidence_writers[key] = writer
        writer.write_id(reference_id)

    def finish(self) -> tuple[SupportArea, ...]:
        kinds = {
            "L1": AssignmentKind.ANCESTOR_DERIVED,
            "L2": AssignmentKind.ANCESTOR_DERIVED,
            "L3": AssignmentKind.DIRECT,
        }
        supports = []
        for level, code in sorted(self._unions):
            limits = self._resource_limits
            forest = self._unions[(level, code)]
            previous_coordinates = forest.retained_coordinate_count
            unioned = forest.finish(
                max_union_coordinates=(limits.max_union_coordinates if limits else None),
                max_overlay_operand_coordinates=(
                    limits.max_overlay_operand_coordinates if limits else None
                ),
            )
            unioned_coordinates = int(shapely.get_num_coordinates(unioned))
            retained_after_finish = self._retained_union_coordinates - previous_coordinates
            if (
                limits is not None
                and retained_after_finish + unioned_coordinates > limits.max_union_coordinates
            ):
                raise LucasResourceLimitError("final retained union coordinate limit exceeded")
            self._retained_union_coordinates = max(0, retained_after_finish + unioned_coordinates)
            if limits is not None and (
                int(shapely.get_num_coordinates(unioned))
                + int(shapely.get_num_coordinates(self._original))
                > limits.max_overlay_operand_coordinates
            ):
                raise LucasResourceLimitError(
                    "final polygon clip operand coordinate limit exceeded"
                )
            intersection = unioned.intersection(self._original)
            area = float(shapely.area(intersection))
            if not math.isfinite(area) or area < 0.0:
                raise ValueError("LUCAS intersection area must be finite and non-negative")
            if area > 0.0:
                writer = self._evidence_writers.pop((level, code), None)
                sidecar = writer.finish() if writer is not None else None
                supports.append(
                    SupportArea(
                        level,
                        code,
                        area,
                        kinds[level],
                        tuple(self._evidence_ids.get((level, code), ())),
                        sidecar,
                    )
                )
            else:
                writer = self._evidence_writers.pop((level, code), None)
                if writer is not None:
                    writer.discard()
            self._retained_union_coordinates = max(
                0, self._retained_union_coordinates - unioned_coordinates
            )
            del unioned, intersection
        return tuple(supports)

    def discard(self) -> None:
        """Close and remove all temporary contributing-ID files after a failed scan."""
        for writer in self._evidence_writers.values():
            writer.discard()
        self._evidence_writers.clear()


def _decode_gpkg_geometry(blob: object, expected_srs_id: int) -> BaseGeometry:
    if not isinstance(blob, (bytes, memoryview)):
        raise TypeError("LUCAS geometry field is not a GeoPackage binary blob")
    data = bytes(blob)
    if len(data) < _GPKG_HEADER_SIZE or data[:2] != b"GP":
        raise ValueError("invalid GeoPackage geometry header")
    flags = data[3]
    byte_order = "<" if flags & 1 else ">"
    srs_id = struct.unpack_from(f"{byte_order}i", data, 4)[0]
    if srs_id != expected_srs_id:
        raise ValueError("LUCAS footprint geometry CRS does not match EPSG:3035 manifest")
    envelope_type = (flags >> 1) & 0b111
    try:
        envelope_size = {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[envelope_type]
    except KeyError as error:
        raise ValueError("unsupported GeoPackage envelope type") from error
    return load_wkb(data[_GPKG_HEADER_SIZE + envelope_size :])
