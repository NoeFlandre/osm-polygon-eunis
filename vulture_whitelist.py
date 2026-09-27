"""Documented entry points and structural protocol arguments used by callers."""

from osm_polygon_eunis.cards import DatasetCardAccumulator
from osm_polygon_eunis.cli import main
from osm_polygon_eunis.reference import RasterReference

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

# Names are part of structural client contracts; concrete adapters consume them.
method
follow_redirects
timeout
repo_type
path_in_repo
recursive
path_or_fileobj
