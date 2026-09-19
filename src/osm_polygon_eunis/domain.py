"""Small immutable domain values shared by the pure processing layers."""

from dataclasses import dataclass
from typing import Final

from shapely.geometry.base import BaseGeometry

EUNIS_FIELDS: Final[tuple[str, ...]] = (
    "eunis_code",
    "eunis_name",
    "eunis_overlap_percentage",
    "eunis_source_version",
)


@dataclass(frozen=True, slots=True)
class OverlapCandidate:
    """A reference geometry that can contribute actual overlap area."""

    code: str
    name: str
    geometry: BaseGeometry
    components_are_disjoint: bool = False


@dataclass(frozen=True, slots=True)
class EunisResult:
    """The nullable fields written to an enriched row."""

    code: str | None
    name: str | None
    overlap_percentage: float | None
    source_version: str | None

    @property
    def is_empty(self) -> bool:
        return self.code is None
