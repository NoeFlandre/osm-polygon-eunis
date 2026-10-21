"""Pinned LUCAS footprint support for complete raw-polygon label pools.

This module never downloads a reference. Production use requires the caller to
provide a verified file and codebook manifest; handoff hashes and guessed layer
names are deliberately not defaults.
"""

from __future__ import annotations

import hashlib
import math
import re
import sqlite3
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import shapely
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from shapely.wkb import loads as load_wkb

from osm_polygon_eunis.geopackage_sql import sql_identifier
from osm_polygon_eunis.pool_contract import AssignmentKind, LabelLevel, ReferencePin

from .common import (
    PoolPolygonInput,
    PoolPolygonLabelRecords,
    SupportArea,
    SupportAttempt,
    label_polygon,
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_GPKG_HEADER_SIZE = 8
_EPSG_3035 = 3035
_UNION_BATCH_SIZE = 128

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


@dataclass(frozen=True, slots=True)
class LucasLayerManifest:
    """Exact table, geometry, ID, code, and RTree names for one GPKG layer."""

    table: str
    geometry_column: str
    primary_key: str
    rtree_table: str
    l1_column: str
    l2_column: str
    l3_column: str
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
            self.l1_column,
            self.l2_column,
            self.l3_column,
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
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> LucasGeoPackageReader:
        """Verify supplied file pins and open the manifest-selected layer."""

        if self._connection is not None:
            raise RuntimeError("LUCAS GeoPackage reader is already open")
        verify_lucas_file(self.path, self.manifest)
        verify_lucas_taxonomy_file(self.taxonomy_path, self.manifest)
        self._connection = sqlite3.connect(f"{self.path.resolve().as_uri()}?mode=ro", uri=True)
        try:
            self._validate_schema()
        except Exception:
            self._connection.close()
            self._connection = None
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

    def query(self, geometry: BaseGeometry) -> LucasFeatureQuery:
        """Query all RTree candidates, retaining IDs even for zero-area touches."""

        connection = self._connection
        if connection is None:
            raise RuntimeError("LUCAS GeoPackage reader must be opened before querying")
        if geometry.is_empty or not geometry.is_valid:
            raise ValueError("LUCAS query geometry must be valid and non-empty")
        layer = self.manifest.layer
        table = sql_identifier(layer.table)
        geom = sql_identifier(layer.geometry_column)
        pk = sql_identifier(layer.primary_key)
        rtree = sql_identifier(layer.rtree_table)
        l1 = sql_identifier(layer.l1_column)
        l2 = sql_identifier(layer.l2_column)
        l3 = sql_identifier(layer.l3_column)
        query = (
            f"SELECT t.{pk}, t.{l1}, t.{l2}, t.{l3}, t.{geom} "  # noqa: S608
            f"FROM {table} AS t JOIN {rtree} AS r ON r.id=t.{pk} "
            "WHERE r.maxx >= ? AND r.minx <= ? AND r.maxy >= ? AND r.miny <= ?"
        )
        min_x, min_y, max_x, max_y = geometry.bounds
        cursor = connection.execute(query, (min_x, max_x, min_y, max_y))
        attempted_ids: list[str] = []
        footprints: list[LucasFootprint] = []
        exclusion_reasons: set[str] = set()
        try:
            while rows := cursor.fetchmany(self.fetch_size):
                for raw_id, raw_l1, raw_l2, raw_l3, raw_blob in rows:
                    footprint_id = str(raw_id)
                    attempted_ids.append(footprint_id)
                    codes = tuple(_optional_code(value) for value in (raw_l1, raw_l2, raw_l3))
                    try:
                        self.taxonomy.validate_path(*codes)
                        footprint_geometry = _decode_gpkg_geometry(raw_blob, layer.srs_id)
                    except (TypeError, ValueError, struct.error, shapely.errors.GEOSException):
                        exclusion_reasons.add("lucas_reference_row_unusable")
                        continue
                    if not footprint_geometry.is_empty and footprint_geometry.is_valid:
                        if footprint_geometry.intersects(geometry):
                            footprints.append(
                                LucasFootprint(
                                    footprint_id=footprint_id,
                                    l1_code=codes[0],
                                    l2_code=codes[1],
                                    l3_code=codes[2],
                                    geometry=footprint_geometry,
                                )
                            )
                    else:
                        exclusion_reasons.add("lucas_reference_geometry_unusable")
        except sqlite3.DatabaseError:
            exclusion_reasons.add("lucas_reference_query_failed:DatabaseError")
        reason = ";".join(sorted(exclusion_reasons)) if exclusion_reasons else None
        return LucasFeatureQuery(tuple(footprints), tuple(attempted_ids), reason)

    def _validate_schema(self) -> None:
        connection = self._connection
        if connection is None:
            raise RuntimeError("LUCAS GeoPackage connection is not open")
        layer = self.manifest.layer
        metadata = connection.execute(
            "SELECT srs_id FROM gpkg_geometry_columns WHERE table_name=? AND column_name=?",
            (layer.table, layer.geometry_column),
        ).fetchone()
        if metadata != (layer.srs_id,):
            raise ValueError("LUCAS GeoPackage layer CRS does not match its manifest")
        column_rows = connection.execute(
            f"PRAGMA table_info({sql_identifier(layer.table)})"
        ).fetchall()
        columns = {row[1] for row in column_rows}
        required = {
            layer.primary_key,
            layer.geometry_column,
            layer.l1_column,
            layer.l2_column,
            layer.l3_column,
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
            )
        else:
            attempt = SupportAttempt(
                _union_by_level(original, query.footprints, taxonomy),
                attempted_reference_ids=query.attempted_feature_ids,
            )
    return label_polygon(polygon, attempt, LUCAS_LEVELS, reference_pin)


def _union_by_level(
    original: BaseGeometry,
    footprints: Sequence[LucasFootprint],
    taxonomy: LucasTaxonomy,
) -> tuple[SupportArea, ...]:
    grouped: dict[tuple[str, str], list[BaseGeometry]] = defaultdict(list)
    evidence_ids: dict[tuple[str, str], list[str]] = defaultdict(list)
    kinds = {"L1": AssignmentKind.ANCESTOR_DERIVED, "L2": AssignmentKind.ANCESTOR_DERIVED}
    kinds["L3"] = AssignmentKind.DIRECT
    for footprint in footprints:
        taxonomy.validate_path(footprint.l1_code, footprint.l2_code, footprint.l3_code)
        for level, code in (
            ("L1", footprint.l1_code),
            ("L2", footprint.l2_code),
            ("L3", footprint.l3_code),
        ):
            if code is not None:
                grouped[(level, code)].append(footprint.geometry)
                evidence_ids[(level, code)].append(footprint.footprint_id)

    supports = []
    for (level, code), geometries in sorted(grouped.items()):
        unioned = _bounded_union(geometries)
        intersection = unioned.intersection(original)
        if not intersection.is_empty:
            area = float(shapely.area(intersection))
            if not math.isfinite(area) or area < 0.0:
                raise ValueError("LUCAS intersection area must be finite and non-negative")
            if area > 0.0:
                supports.append(
                    SupportArea(
                        level,
                        code,
                        area,
                        kinds[level],
                        tuple(evidence_ids[(level, code)]),
                    )
                )
    return tuple(supports)


def _bounded_union(geometries: Sequence[BaseGeometry]) -> BaseGeometry:
    """Union geometry in small batches while never summing overlapping area."""

    if len(geometries) == 1:
        return geometries[0]
    pending = list(geometries)
    while len(pending) > 1:
        next_level: list[BaseGeometry] = []
        for start in range(0, len(pending), _UNION_BATCH_SIZE):
            batch = pending[start : start + _UNION_BATCH_SIZE]
            next_level.append(batch[0] if len(batch) == 1 else unary_union(batch))
        pending = next_level
    return pending[0]


def _optional_code(value: object) -> str | None:
    if value is None:
        return None
    code = str(value).strip()
    if not code:
        return None
    return code


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
