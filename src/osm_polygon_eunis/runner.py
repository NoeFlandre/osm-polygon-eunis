"""Compatibility facade for the EUNIS release API."""

from .card_publishing import IntersectionErrorLimitError
from .geometry_jobs import process_geometry_paths
from .options import BatchLimits, ReleaseOptions, ShardContext
from .publish import VerificationError
from .references import open_reference_group
from .release_orchestration import (
    DEFAULT_WORKERS,
    ConfigError,
    DryRunDataset,
    DryRunReport,
    plan_release,
    run_release,
    validate_reference_config,
    verify_release,
)
from .release_plan import (
    DATASET_NAMES,
    DatasetPlan,
    DatasetReceipt,
    Progress,
    ReleaseReceipt,
    plan_datasets,
    selected_dataset_names,
)
from .shard_processing import FinalizeOptions, finalize_dataset

__all__ = [
    "DATASET_NAMES",
    "DEFAULT_WORKERS",
    "BatchLimits",
    "ConfigError",
    "DatasetPlan",
    "DatasetReceipt",
    "DryRunDataset",
    "DryRunReport",
    "FinalizeOptions",
    "IntersectionErrorLimitError",
    "Progress",
    "ReleaseOptions",
    "ReleaseReceipt",
    "ShardContext",
    "VerificationError",
    "finalize_dataset",
    "open_reference_group",
    "plan_datasets",
    "plan_release",
    "process_geometry_paths",
    "run_release",
    "selected_dataset_names",
    "validate_reference_config",
    "verify_release",
]
