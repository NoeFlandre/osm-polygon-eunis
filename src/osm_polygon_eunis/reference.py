"""Compatibility imports for the EEA reference readers.

Import raster and GeoPackage implementations from their specific modules.
"""

from .geopackage_reference import GeoPackageReference
from .raster_reference import (
    EPSG_LAEA_EUROPE,
    RasterLayer,
    RasterReference,
    parse_layer_code,
    wgs84_envelope,
)

__all__ = [
    "EPSG_LAEA_EUROPE",
    "GeoPackageReference",
    "RasterLayer",
    "RasterReference",
    "parse_layer_code",
    "wgs84_envelope",
]
