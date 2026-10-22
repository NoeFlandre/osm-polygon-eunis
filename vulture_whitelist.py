"""Documented entry points and structural protocol arguments used by callers."""

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.cli import main
from osm_polygon_eunis.eea import ClassificationTaxonomy
from osm_polygon_eunis.geometry_jobs import process_geometry_paths
from osm_polygon_eunis.pool_adapter import (
    EunisPoolSourceLayout,
    make_eunis_2021_reference_pin,
    make_eunis_taxonomy_from_classification,
    resolve_polygon_document_links,
)
from osm_polygon_eunis.pool_runner import (
    EunisPoolExport,
    export_eunis_full_pool,
    iter_attempt_checkpoint,
    iter_eunis_wiki_downstream_records,
    read_projected_geometry_wkb,
    read_raw_geometry,
    run_eunis_label_attempts,
    scan_raw_inventory,
)
from osm_polygon_eunis.raster_reference import RasterReference
from osm_polygon_eunis.reference_staging import open_reference_group
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
ClassificationTaxonomy.root_for

# EUNIS adapter settings and entry points are consumed by the bounded production runner.
EunisPoolSourceLayout.relevance_path_globs
EunisPoolSourceLayout.document_link_glob
EunisPoolSourceLayout.document_glob
EunisPoolSourceLayout.wikivoyage_document_glob
EunisPoolSourceLayout.sentence_glob
EunisPoolExport.raw_inventory_report_uri
EunisPoolExport.label_attempt_report_uri
EunisPoolExport.attempt_checkpoint_uris
EunisPoolExport.downstream_join_included
make_eunis_2021_reference_pin
make_eunis_taxonomy_from_classification
resolve_polygon_document_links
scan_raw_inventory
run_eunis_label_attempts
iter_eunis_wiki_downstream_records
export_eunis_full_pool
iter_attempt_checkpoint
read_projected_geometry_wkb
read_raw_geometry

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
PoolArtifactReader.iter_parquet_rows
LucasTaxonomy.from_records
LucasTaxonomy.lucas_2022_c3
SourceTreeFile.lfs_oid
SourceTreeFile.lfs_size_bytes
