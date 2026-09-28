"""Exact-overlap EUNIS enrichment for OSM polygon datasets."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("osm-polygon-eunis")
except PackageNotFoundError:
    __version__ = "unknown"

__all__ = ["__version__"]
