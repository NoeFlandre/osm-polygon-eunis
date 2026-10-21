"""Reference-specific measurements for complete polygon-label pools."""

from .common import (
    AttemptEvidenceRecord,
    PoolPolygonInput,
    PoolPolygonLabelRecords,
    SupportArea,
    SupportAttempt,
    build_raw_inventory_evidence,
    label_polygon,
)
from .lucas import (
    LUCAS_LEVELS,
    LucasCode,
    LucasFeatureQuery,
    LucasFootprint,
    LucasGeoPackageReader,
    LucasLayerManifest,
    LucasReferenceManifest,
    LucasTaxonomy,
    measure_lucas_supports,
    verify_lucas_file,
    verify_lucas_taxonomy_file,
)
from .runner import (
    WorldCoverSourceAttempt,
    iter_lucas_attempts,
    iter_worldcover_attempts,
    iter_worldcover_source_attempts,
)
from .worldcover import (
    WORLDCOVER_CODES,
    WORLDCOVER_LEVELS,
    WorldCoverEvidence,
    worldcover_supports,
)

__all__ = [
    "LUCAS_LEVELS",
    "WORLDCOVER_CODES",
    "WORLDCOVER_LEVELS",
    "AttemptEvidenceRecord",
    "LucasCode",
    "LucasFeatureQuery",
    "LucasFootprint",
    "LucasGeoPackageReader",
    "LucasLayerManifest",
    "LucasReferenceManifest",
    "LucasTaxonomy",
    "PoolPolygonInput",
    "PoolPolygonLabelRecords",
    "SupportArea",
    "SupportAttempt",
    "WorldCoverEvidence",
    "WorldCoverSourceAttempt",
    "build_raw_inventory_evidence",
    "iter_lucas_attempts",
    "iter_worldcover_attempts",
    "iter_worldcover_source_attempts",
    "label_polygon",
    "measure_lucas_supports",
    "verify_lucas_file",
    "verify_lucas_taxonomy_file",
    "worldcover_supports",
]
