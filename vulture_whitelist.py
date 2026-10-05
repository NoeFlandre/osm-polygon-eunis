"""Documented entry points and structural protocol arguments used by callers."""

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.cli import main
from osm_polygon_eunis.geometry_jobs import process_geometry_paths
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
