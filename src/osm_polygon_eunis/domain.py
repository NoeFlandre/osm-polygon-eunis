"""Small immutable domain values shared by the pure processing layers."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from shapely.geometry.base import BaseGeometry

EPSG_LAEA_EUROPE: Final = 3035
WGS84_EPSG_CODE: Final = 4326
EPSG_LAEA_EUROPE_CRS: Final = f"EPSG:{EPSG_LAEA_EUROPE}"
WGS84_CRS: Final = f"EPSG:{WGS84_EPSG_CODE}"

EUNIS_FIELDS: Final[tuple[str, ...]] = (
    "eunis_code",
    "eunis_name",
    "eunis_overlap_percentage",
    "eunis_source_version",
)


class SchemaError(ValueError, TypeError):
    """A parsed payload (JSON, GeoPackage row, Parquet column) has the wrong shape.

    Subclasses both ``ValueError`` (existing callers) and ``TypeError`` (the
    conventional type for a wrong-type value).
    """


@dataclass(frozen=True, slots=True)
class OverlapCandidate:
    """A reference geometry that can contribute actual overlap area."""

    code: str
    name: str
    geometry: BaseGeometry
    components_are_disjoint: bool = False
    error_count: int = 1


@dataclass(frozen=True, slots=True)
class EunisTaxonomy:
    """Verified code labels and parent links from one EUNIS classification source."""

    labels: Mapping[str, str]
    levels: Mapping[str, int]
    parents: Mapping[str, str]
    source_record_id: str | None = None
    source_path: str | None = None
    source_url: str | None = None
    source_size: int | None = None
    source_etag: str | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        """Require display labels for every code declared in this taxonomy."""
        taxonomy_codes = set(self.levels) | set(self.parents) | set(self.parents.values())
        missing = sorted(
            code
            for code in taxonomy_codes
            if not isinstance(self.labels.get(code), str) or not self.labels[code].strip()
        )
        if missing:
            raise ValueError(f"EUNIS taxonomy has missing label for {', '.join(missing)}")

    def ancestors_including_self(self, code: str) -> tuple[str, ...]:
        """Return a code followed by its verified parents, from specific to root."""
        ancestors = [code]
        seen = {code}
        current = code
        while current in self.parents:
            current = self.parents[current]
            if current in seen:
                raise ValueError(f"EUNIS taxonomy contains a parent cycle at {current}")
            ancestors.append(current)
            seen.add(current)
        return tuple(ancestors)

    def root_for(self, code: str) -> str:
        """Return the top verified ancestor, treating unknown codes as isolated roots."""
        return self.ancestors_including_self(code)[-1]

    @property
    def fingerprint(self) -> str:
        """Return a stable digest for taxonomy-driven checkpoints."""
        payload = json.dumps(
            {
                "labels": sorted(self.labels.items()),
                "levels": sorted(self.levels.items()),
                "parents": sorted(self.parents.items()),
            },
            separators=(",", ":"),
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class EunisResult:
    """The nullable fields written to an enriched row."""

    code: str | None
    name: str | None
    overlap_percentage: float | None
    source_version: str | None

    @property
    def is_empty(self) -> bool:
        """Return whether this result has neither a code nor an overlap measurement."""
        return self.code is None and self.overlap_percentage is None
