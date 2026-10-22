"""Documented entry points and structural protocol arguments used by callers."""

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.cli import main
from osm_polygon_eunis.geometry_jobs import process_geometry_paths
from osm_polygon_eunis.pool_adapters.lucas import LucasTaxonomy
from osm_polygon_eunis.pool_adapters.source_pins import SourceTreeFile
from osm_polygon_eunis.pool_contract.export import ExportReceipt
from osm_polygon_eunis.pool_contract.streaming import (
    FilesystemArtifactReader,
    PoolArtifactReader,
    PoolStreamSpec,
)
from osm_polygon_eunis.pool_contract.models import (
    AmbiguityBasis,
    AssignmentKind,
    AttemptStatus,
    BodyStatus,
    DataSplit,
    EligibilityStatus,
    EvaluationArm,
    EvaluationCase,
    GeometryStatus,
    TextLinkStatus,
    TextSource,
)
from osm_polygon_eunis.pool_contract.streaming import PoolStreamSpec
from osm_polygon_eunis.raster_reference import RasterReference
from osm_polygon_eunis.reference_staging import open_reference_group

# Console script configured in pyproject.toml.
main

# Public observability properties documented in docs/api.md.
DatasetCardAccumulator.total_rows
DatasetCardAccumulator.invalid_geometries
RasterReference.source_extent_wgs84
RasterReference.tile_cache_hits
RasterReference.tile_cache_misses
RasterReference.tile_cache_miss_rate
RasterReference.tile_cache_byte_budget

# Library entry points that tests and downstream callers use directly.
process_geometry_paths
open_reference_group

# Public pool-contract enum values are consumed by source adapters in other repos.
TextSource.DESCRIPTION
TextSource.WEBSITE
TextSource.WIKI
GeometryStatus.USABLE
GeometryStatus.INVALID
GeometryStatus.EMPTY
GeometryStatus.TRANSFORM_FAILED
AssignmentKind.DIRECT
AssignmentKind.ANCESTOR_DERIVED
AmbiguityBasis.MULTIPLE_QUALIFIERS
AmbiguityBasis.AMBIGUOUS_EVIDENCE
TextLinkStatus.KEY_MISMATCH
TextLinkStatus.TEXT_HASH_MISMATCH
TextLinkStatus.GEOMETRY_MISSING
BodyStatus.MISSING
BodyStatus.PRIVATE
BodyStatus.NOT_RETAINED
EligibilityStatus.ELIGIBLE
EligibilityStatus.EXCLUDED
EligibilityStatus.REPLACED
EvaluationArm.YES
EvaluationArm.NO
DataSplit.TRAIN
DataSplit.VALIDATION
DataSplit.TEST

# Vulture cannot see consumers of immutable receipts and evaluation cases across repos.
ExportReceipt.manifest_sha256
ExportReceipt.table_uris
ExportReceipt.table_sha256
EvaluationCase.arm
AttemptStatus.COMPLETED
PoolStreamSpec.from_build_input

# Public constructors and raw-file metadata are consumed by source adapters downstream.
LucasTaxonomy.from_records
LucasTaxonomy.lucas_2022_c3
SourceTreeFile.lfs_oid
SourceTreeFile.lfs_size_bytes
PoolArtifactReader.iter_parquet_rows
FilesystemArtifactReader.iter_parquet_rows
