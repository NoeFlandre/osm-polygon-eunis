"""Deterministic, storage-bounded Hugging Face dataset-card artifacts."""

from __future__ import annotations

import copy
import html
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

import h3
from shapely.geometry.base import BaseGeometry

from .domain import EunisResult
from .fileio import sha256_file
from .geometry import (
    ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES,
    GEOMETRY_POLICY,
    has_antimeridian_span,
    parse_geometry,
)

_H3_RESOLUTION = 3
_MAP_SNAPSHOT_SCHEMA_VERSION = 1
_MAP_COUNT_ENTRY_LENGTH = 2
_LON_MIN, _LON_MAX = -180.0, 180.0
_LAT_MIN, _LAT_MAX = -90.0, 90.0
_LON_RANGE = _LON_MAX - _LON_MIN
_LAT_RANGE = _LAT_MAX - _LAT_MIN
_MAP_PATH = "eunis/world-map.svg"
_NO_LABEL = "No EUNIS label"

_LANDMASSES: tuple[tuple[tuple[float, float], ...], ...] = (
    (
        (-168, 70),
        (-150, 72),
        (-135, 65),
        (-125, 52),
        (-117, 50),
        (-110, 32),
        (-96, 20),
        (-86, 25),
        (-80, 8),
        (-72, 12),
        (-62, 25),
        (-64, 45),
        (-82, 60),
        (-105, 72),
        (-140, 78),
    ),
    ((-82, 12), (-72, 8), (-62, -8), (-52, -20), (-52, -42), (-66, -55), (-76, -35), (-80, -10)),
    ((-18, 36), (8, 37), (34, 31), (50, 12), (42, -12), (31, -35), (14, -35), (-5, -5), (-18, 14)),
    (
        (-10, 36),
        (-5, 50),
        (12, 59),
        (35, 69),
        (65, 65),
        (95, 55),
        (135, 52),
        (160, 45),
        (175, 58),
        (170, 25),
        (145, 5),
        (120, 8),
        (105, 25),
        (80, 20),
        (58, 8),
        (45, 25),
        (25, 35),
        (5, 44),
    ),
    ((95, 8), (115, 5), (132, -5), (145, -12), (153, -28), (142, -39), (115, -35), (100, -20)),
    ((-54, 60), (-42, 75), (-24, 82), (-18, 70), (-32, 58)),
    (
        (-180, -68),
        (-145, -72),
        (-105, -75),
        (-60, -72),
        (-20, -76),
        (20, -74),
        (70, -72),
        (120, -75),
        (180, -68),
        (180, -90),
        (-180, -90),
    ),
)


@dataclass(frozen=True, slots=True)
class LabelSummary:
    """One row in the dataset-card distribution table."""

    code: str | None
    name: str
    rows: int
    percentage: float


@dataclass(frozen=True, slots=True)
class CardArtifacts:
    """Local card files and their content hashes for remote verification."""

    files: Mapping[str, Path]
    hashes: Mapping[str, str]
    manifest: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _MapLayout:
    """Pixel geometry of the static SVG map canvas."""

    width: int = 1200
    height: int = 660
    plot_x: int = 50
    plot_y: int = 75
    plot_width: int = 835
    plot_height: int = 510
    subtitle_y: int = 52
    legend_x: int = 925
    legend_y: int = 105


@dataclass(frozen=True, slots=True)
class _MapCountBand:
    minimum: int
    maximum: int | None
    label: str
    color: str


_LAYOUT = _MapLayout()
_MAP_COUNT_BANDS = (
    _MapCountBand(1, 9, "1-9", "#d7ece3"),
    _MapCountBand(10, 99, "10-99", "#a4d1c2"),
    _MapCountBand(100, 999, "100-999", "#67a796"),
    _MapCountBand(1000, None, "1,000+", "#2f6f66"),
)


class DatasetCardAccumulator:
    """Collect label counts and bounded H3 cell counts while shards stream past."""

    def __init__(
        self,
        *,
        source_readme: str | None = None,
    ) -> None:
        """Create a card accumulator for a pinned source card.

        Args:
            source_readme: Pinned source card used to retain Viewer configs and license.
        """
        self._source_configs = _extract_source_configs(source_readme)
        self._source_license = _extract_source_license(source_readme)
        self._counts: dict[str | None, int] = {}
        self._names: dict[str | None, str] = {}
        self._map_counts: dict[str, int] = {}
        self._total_rows = 0
        self._invalid_geometries = 0
        self._intersection_errors = 0

    @property
    def total_rows(self) -> int:
        """Return the number of rows observed for the current dataset."""
        return self._total_rows

    @property
    def invalid_geometries(self) -> int:
        """Rows with present geometry that is not a usable areal geometry."""

        return self._invalid_geometries

    @property
    def intersection_errors(self) -> int:
        """Overlap candidates dropped because the exact GEOS intersection raised."""

        return self._intersection_errors

    def spawn(self) -> DatasetCardAccumulator:
        """Return an empty accumulator sharing this one's settings and source card."""

        child = copy.copy(self)
        child._counts, child._names, child._map_counts = {}, {}, {}
        child._total_rows = child._invalid_geometries = child._intersection_errors = 0
        return child

    def merge(self, other: DatasetCardAccumulator) -> None:
        """Add another accumulator's observations (from the same dataset) to this one."""

        for code, name in other._names.items():
            if self._names.setdefault(code, name) != name:
                raise ValueError(f"dataset card has conflicting names for {code}")
        for code, count in other._counts.items():
            self._counts[code] = self._counts.get(code, 0) + count
        for cell, count in other._map_counts.items():
            self._map_counts[cell] = self._map_counts.get(cell, 0) + count
        self._total_rows += other._total_rows
        self._invalid_geometries += other._invalid_geometries
        self._intersection_errors += other._intersection_errors

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serializable observations for a resumable release."""

        return {
            "counts": [[code, count] for code, count in self._counts.items()],
            "names": [[code, name] for code, name in self._names.items()],
            "map_snapshot_schema_version": _MAP_SNAPSHOT_SCHEMA_VERSION,
            "map_counts": [[cell, count] for cell, count in sorted(self._map_counts.items())],
            "total_rows": self._total_rows,
            "invalid_geometries": self._invalid_geometries,
            "intersection_errors": self._intersection_errors,
        }

    def restore(self, state: Mapping[str, Any]) -> None:
        """Replace the observations with a previous :meth:`snapshot`."""

        map_schema_version = state.get("map_snapshot_schema_version")
        if map_schema_version is None:
            if "bins" in state:
                raise ValueError(
                    "cannot restore legacy 2-degree map snapshot; map snapshot schema is missing"
                )
            raise ValueError("map snapshot schema version is missing")
        if map_schema_version != _MAP_SNAPSHOT_SCHEMA_VERSION:
            raise ValueError(f"unsupported map snapshot schema version: {map_schema_version}")

        counts = dict(state["counts"])
        names = dict(state["names"])
        map_counts = dict(_validated_map_counts(state["map_counts"]))
        total_rows = _validated_map_total_rows(state["total_rows"], map_counts)
        invalid_geometries = int(state["invalid_geometries"])
        intersection_errors = int(state["intersection_errors"])
        self._counts = counts
        self._names = names
        self._map_counts = map_counts
        self._total_rows = total_rows
        self._invalid_geometries = invalid_geometries
        self._intersection_errors = intersection_errors

    def record_intersection_errors(self, count: int) -> None:
        """Add intersection errors carried in a label sidecar."""

        if count < 0:
            raise ValueError("intersection error count must be non-negative")
        self._intersection_errors += count

    def observe(self, result: EunisResult, raw_geometry: object) -> None:
        """Record one output label and at most one bounded map cell."""

        code, name = _label_identity(result)
        self._record_label(code, name)
        self._record_map_cell(raw_geometry)

    def _record_label(self, code: str | None, name: str) -> None:
        previous_name = self._names.setdefault(code, name)
        if previous_name != name:
            raise ValueError(f"dataset card has conflicting names for {code}")
        self._counts[code] = self._counts.get(code, 0) + 1
        self._total_rows += 1

    def _record_map_cell(self, raw_geometry: object) -> None:
        geometry = parse_geometry(raw_geometry)
        if geometry is None:
            if raw_geometry is not None:
                self._invalid_geometries += 1
            return
        if has_antimeridian_span(geometry):
            self._invalid_geometries += 1
            return
        representative_point = _map_representative_point(geometry)
        if representative_point is None:
            return
        latitude, longitude = representative_point
        cell = h3.latlng_to_cell(latitude, longitude, _H3_RESOLUTION)
        self._map_counts[cell] = self._map_counts.get(cell, 0) + 1

    def summaries(self) -> tuple[LabelSummary, ...]:
        """Return complete deterministic label distribution statistics."""

        if self._total_rows == 0:
            return ()
        summaries = [
            LabelSummary(
                code,
                self._names[code],
                count,
                100.0 * count / self._total_rows,
            )
            for code, count in self._counts.items()
        ]
        return tuple(
            sorted(summaries, key=lambda item: (item.code is None, -item.rows, item.code or ""))
        )

    def write_artifacts(
        self,
        directory: Path,
        *,
        dataset_name: str,
        source_repo: str,
        target_repo: str,
        source_revision: str,
        reference_version: str,
    ) -> CardArtifacts:
        """Write the README and static SVG map without retaining source data."""

        if self._total_rows == 0:
            raise ValueError("cannot build a dataset card without polygon rows")
        directory.mkdir(parents=True, exist_ok=True)
        map_path = directory / _MAP_PATH
        map_path.parent.mkdir(parents=True, exist_ok=True)
        with map_path.open("w", encoding="utf-8", newline="\n") as output:
            self._write_map(output, dataset_name)
        readme_path = directory / "README.md"
        readme_path.write_text(
            self._render_readme(
                dataset_name=dataset_name,
                source_repo=source_repo,
                target_repo=target_repo,
                source_revision=source_revision,
                reference_version=reference_version,
                source_configs=self._source_configs,
            ),
            encoding="utf-8",
        )
        files = {"README.md": readme_path, _MAP_PATH: map_path}
        hashes = {path: sha256_file(local_path) for path, local_path in files.items()}
        manifest = {
            "readme_path": "README.md",
            "map_path": _MAP_PATH,
            "readme_sha256": hashes["README.md"],
            "map_sha256": hashes[_MAP_PATH],
            "total_rows": self._total_rows,
            "invalid_geometries": self._invalid_geometries,
            "intersection_errors": self._intersection_errors,
            "map_aggregation": _map_aggregation_metadata(
                mapped_rows=sum(self._map_counts.values()),
                omitted_rows=self._total_rows - sum(self._map_counts.values()),
            ),
            "geometry_policy": dict(GEOMETRY_POLICY),
            "label_distribution": [
                {
                    "code": summary.code,
                    "name": summary.name,
                    "rows": summary.rows,
                    "percentage": round(summary.percentage, 6),
                }
                for summary in self.summaries()
            ],
        }
        return CardArtifacts(files, hashes, manifest)

    def _render_readme(
        self,
        *,
        dataset_name: str,
        source_repo: str,
        target_repo: str,
        source_revision: str,
        reference_version: str,
        source_configs: str | None,
    ) -> str:
        title = _title(dataset_name)
        lines = [
            "---",
            f"pretty_name: {title}",
        ]
        if self._source_license:
            lines.append(f"license: {self._source_license}")
        lines.extend(
            [
                "tags:",
                "- geospatial",
                "- osm",
                "- eunis",
                "- ecology",
            ]
        )
        if source_configs:
            lines.extend(("", *source_configs.splitlines()))
        lines.extend(
            (
                "---",
                "",
                f"# {title}",
                "",
                "This dataset preserves the source rows and adds four nullable EUNIS fields "
                "to each polygon-bearing Parquet shard.",
                "",
                "![World map of dataset-row coverage by H3 cell](./eunis/world-map.svg)",
                "",
                "The map uses one representative point per valid polygon-bearing dataset row "
                "and H3 resolution 3. Each cell counts dataset rows, not area. Counts include "
                "every EUNIS label, including unlabeled rows. Colors use the same fixed bands "
                "across all cards: 1-9 rows, 10-99 rows, 100-999 rows, and 1,000+ rows. "
                "Missing or invalid polygon geometry is omitted. Representative points outside "
                "longitude [-180, 180] or latitude [-90, 90] are omitted.",
                "",
                "## Label distribution",
                "",
                "| EUNIS code | Label | Polygons | Share |",
                "|:--|:--|--:|--:|",
            )
        )
        lines.extend(
            f"| {_markdown_code(summary.code)} | {_markdown_cell(summary.name)} | "
            f"{summary.rows:,} | {summary.percentage:.2f}% |"
            for summary in self.summaries()
        )
        lines.extend(
            (
                "",
                "## Method",
                "",
                "EUNIS labels are selected by the largest actual polygon intersection "
                "area after transforming geometries to EPSG:3035. Bounding boxes are "
                "used only to prune candidates. Polygon and multipolygon inputs are "
                "accepted; polygon parts of mixed geometry collections are retained. "
                "WGS84 edges are densified to at most "
                f"{GEOMETRY_POLICY['wgs84_max_segment_length_degrees']} degrees before "
                "projection. Polygons spanning more than 180 degrees of longitude, "
                "collapsed polygons, and non-areal geometries are rejected and counted "
                "as invalid. Out-of-reference polygons receive null EUNIS fields.",
                "",
                f"Source: `{source_repo}` at `{source_revision}`.",
                f"Output: `{target_repo}`.",
                f"Reference: official EEA EUNIS assets pinned as `{reference_version}`.",
                "",
                "## Data terms",
                "",
                "The Apache-2.0 license applies to project code only and does not change "
                "the terms for source or derived data.",
                "OpenStreetMap data is available under the [Open Database License]"
                "(https://www.openstreetmap.org/copyright), with attribution and "
                "share-alike requirements.",
                "EEA reference assets can carry item-specific reuse terms; check the "
                "notice attached to each exact asset and the [EEA legal notice]"
                "(https://www.eea.europa.eu/en/legal-notice).",
            )
        )
        return "\n".join(lines) + "\n"

    def _write_map(self, output: TextIO, dataset_name: str) -> None:
        _write_map_header(output, dataset_name, _LAYOUT)
        _write_map_plot(output, self._map_counts, _LAYOUT)
        _write_map_footer(output, _LAYOUT)


def _map_representative_point(geometry: BaseGeometry) -> tuple[float, float] | None:
    point = geometry.representative_point()
    longitude = float(point.x)
    latitude = float(point.y)
    if not math.isfinite(longitude) or not math.isfinite(latitude):
        return None
    if not (_LON_MIN <= longitude <= _LON_MAX and _LAT_MIN <= latitude <= _LAT_MAX):
        return None
    return latitude, longitude


def _write_map_header(output: TextIO, dataset_name: str, layout: _MapLayout) -> None:
    output.write('<?xml version="1.0" encoding="UTF-8"?>\n')
    output.write(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{layout.width}" '
        f'height="{layout.height}" viewBox="0 0 {layout.width} {layout.height}" '
        'aria-labelledby="map-title map-description">\n'
    )
    output.write(
        f'<title id="map-title">{html.escape(_title(dataset_name))} H3 coverage map</title>\n'
    )
    output.write(
        '<desc id="map-description">One representative point per valid polygon-bearing '
        "dataset row is counted in one H3 resolution 3 cell across all labels. Cell colors "
        "show fixed dataset-row count bands; unlabeled rows are included.</desc>\n"
    )
    output.write('<rect width="100%" height="100%" fill="#ffffff"/>\n')
    output.write(
        f'<text x="{layout.plot_x}" y="30" font-family="sans-serif" '
        f'font-size="22" font-weight="600" fill="#1f2933">'
        f"{html.escape(_title(dataset_name))}</text>\n"
    )
    output.write(
        f'<text x="{layout.plot_x}" y="{layout.subtitle_y}" font-family="sans-serif" '
        'font-size="12" fill="#52606d">Representative point per polygon-bearing row · '
        "H3 resolution 3 · counts include unlabeled rows</text>\n"
    )
    output.write(
        '<defs><clipPath id="map-plot-clip">'
        f'<rect x="{layout.plot_x}" y="{layout.plot_y}" '
        f'width="{layout.plot_width}" height="{layout.plot_height}"/>'
        "</clipPath></defs>\n"
    )
    output.write(
        f'<rect x="{layout.plot_x}" y="{layout.plot_y}" width="{layout.plot_width}" '
        f'height="{layout.plot_height}" fill="#eaf1f3"/>\n'
    )
    output.write('<g clip-path="url(#map-plot-clip)">\n')


def _write_map_plot(output: TextIO, map_counts: Mapping[str, int], layout: _MapLayout) -> None:
    for element in _land_elements(layout):
        output.write(f"{element}\n")
    for element in _grid_elements(layout):
        output.write(f"{element}\n")
    for cell, count in sorted(map_counts.items()):
        for element in _map_cell_elements(cell, count, layout):
            output.write(f"{element}\n")


def _write_map_footer(output: TextIO, layout: _MapLayout) -> None:
    output.write("</g>\n")
    output.write(
        f'<rect x="{layout.plot_x}" y="{layout.plot_y}" width="{layout.plot_width}" '
        f'height="{layout.plot_height}" fill="none" stroke="#cbd8dc" stroke-width="1"/>\n'
    )
    for element in _map_legend_elements(layout):
        output.write(f"{element}\n")
    output.write("</svg>\n")


def _validated_map_counts(value: object) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, list):
        raise TypeError("map snapshot map_counts must be a list")
    counts = tuple(_validated_map_count_entry(entry) for entry in value)
    if len({cell for cell, _count in counts}) != len(counts):
        raise ValueError("map snapshot contains duplicate H3 cells")
    return counts


def _validated_map_count_entry(entry: object) -> tuple[str, int]:
    if not isinstance(entry, list) or len(entry) != _MAP_COUNT_ENTRY_LENGTH:
        raise TypeError("map snapshot entries must contain an H3 cell and row count")
    cell, count = entry
    _validate_map_snapshot_cell(cell)
    _validate_map_snapshot_count(count)
    return cell, count


def _validate_map_snapshot_cell(cell: object) -> None:
    if not isinstance(cell, str):
        raise TypeError("map snapshot H3 cells must be strings")
    if not h3.is_valid_cell(cell) or h3.get_resolution(cell) != _H3_RESOLUTION:
        raise ValueError("map snapshot contains an invalid resolution-3 H3 cell")


def _validate_map_snapshot_count(count: object) -> None:
    if type(count) is not int:
        raise TypeError("map snapshot row counts must be integers")
    if count < 1:
        raise ValueError("map snapshot row counts must be positive integers")


def _validated_map_total_rows(value: object, map_counts: Mapping[str, int]) -> int:
    if type(value) is not int:
        raise TypeError("map snapshot total_rows must be an integer")
    if value < 0:
        raise ValueError("map snapshot total_rows must be non-negative")
    if sum(map_counts.values()) > value:
        raise ValueError("map snapshot row counts exceed total_rows")
    return value


def _map_aggregation_metadata(*, mapped_rows: int, omitted_rows: int) -> dict[str, object]:
    return {
        "unit": "dataset rows per occupied H3 cell",
        "aggregation": "one representative point per valid polygon-bearing dataset row",
        "includes_unlabeled_rows": True,
        "h3_resolution": _H3_RESOLUTION,
        "count_bands": [
            {
                "minimum": band.minimum,
                "maximum": band.maximum,
                "label": band.label,
                "color": band.color,
            }
            for band in _MAP_COUNT_BANDS
        ],
        "mapped_rows": mapped_rows,
        "omitted_rows": omitted_rows,
        "omission_policy": (
            "Missing or invalid polygon geometry, and representative points outside longitude "
            "[-180, 180] or latitude [-90, 90], are omitted."
        ),
    }


def _map_count_band(count: int) -> _MapCountBand:
    for band in _MAP_COUNT_BANDS:
        if count >= band.minimum and (band.maximum is None or count <= band.maximum):
            return band
    raise ValueError("map cell counts must be positive integers")


def _unwrapped_boundary(cell: str) -> tuple[tuple[float, float], ...]:
    unwrapped: list[tuple[float, float]] = []
    previous_longitude: float | None = None
    for latitude, longitude_value in h3.cell_to_boundary(cell):
        unwrapped_longitude = float(longitude_value)
        if previous_longitude is not None:
            while unwrapped_longitude - previous_longitude > ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES:
                unwrapped_longitude -= _LON_RANGE
            while unwrapped_longitude - previous_longitude < -ANTIMERIDIAN_LONGITUDE_SPAN_DEGREES:
                unwrapped_longitude += _LON_RANGE
        unwrapped.append((float(latitude), unwrapped_longitude))
        previous_longitude = unwrapped_longitude
    return tuple(unwrapped)


def _map_cell_elements(cell: str, count: int, layout: _MapLayout) -> tuple[str, ...]:
    boundary = _unwrapped_boundary(cell)
    color = _map_count_band(count).color
    return tuple(
        _map_cell_element(boundary, shift, color, count, layout)
        for shift in _visible_map_shifts(boundary)
    )


def _visible_map_shifts(boundary: tuple[tuple[float, float], ...]) -> tuple[float, ...]:
    minimum_longitude = min(longitude for _latitude, longitude in boundary)
    maximum_longitude = max(longitude for _latitude, longitude in boundary)
    if minimum_longitude < _LON_MIN:
        return (0.0, _LON_RANGE)
    if maximum_longitude > _LON_MAX:
        return (-_LON_RANGE, 0.0)
    return (0.0,)


def _map_cell_element(
    boundary: tuple[tuple[float, float], ...],
    shift: float,
    color: str,
    count: int,
    layout: _MapLayout,
) -> str:
    points = [
        f"{_map_x(longitude + shift, layout):.2f},{_map_y(latitude, layout):.2f}"
        for latitude, longitude in boundary
    ]
    path = "M " + " L ".join(points) + " Z"
    return (
        f'<path d="{path}" fill="{color}" stroke="none">'
        f"<title>{count:,} dataset rows</title></path>"
    )


def _label_identity(result: EunisResult) -> tuple[str | None, str]:
    code = result.code or None
    return code, result.name or (_NO_LABEL if code is None else "Unknown EUNIS label")


def _map_x(longitude: float, layout: _MapLayout) -> float:
    return layout.plot_x + (longitude - _LON_MIN) / _LON_RANGE * layout.plot_width


def _map_y(latitude: float, layout: _MapLayout) -> float:
    return layout.plot_y + (_LAT_MAX - latitude) / _LAT_RANGE * layout.plot_height


def _grid_elements(layout: _MapLayout) -> list[str]:
    plot_x, plot_y = layout.plot_x, layout.plot_y
    plot_width, plot_height = layout.plot_width, layout.plot_height
    elements: list[str] = []
    for longitude in range(-120, 180, 60):
        x = _map_x(float(longitude), layout)
        elements.append(
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" '
            f'y2="{plot_y + plot_height}" stroke="#dfe7ef" stroke-width="0.7"/>'
        )
    for latitude in range(-60, 90, 30):
        y = _map_y(float(latitude), layout)
        elements.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" '
            f'x2="{plot_x + plot_width}" y2="{y:.2f}" '
            'stroke="#dfe7ef" stroke-width="0.7"/>'
        )
    return elements


def _land_elements(layout: _MapLayout) -> list[str]:
    elements = []
    for landmass in _LANDMASSES:
        points = " ".join(
            (f"{_map_x(longitude, layout):.2f},{_map_y(latitude, layout):.2f}")
            for longitude, latitude in landmass
        )
        elements.append(f'<polygon points="{points}" fill="#f4f1e9" stroke="none"/>')
    return elements


def _map_legend_elements(layout: _MapLayout) -> list[str]:
    x = layout.legend_x
    y = layout.legend_y
    elements = [
        (
            f'<text x="{x}" y="75" font-family="sans-serif" font-size="15" '
            'font-weight="600" fill="#1f2933">Rows per H3 cell</text>'
        ),
        (
            f'<text x="{x}" y="96" font-family="sans-serif" font-size="10" '
            'fill="#52606d">Fixed count bands across all cards</text>'
        ),
    ]
    for index, band in enumerate(_MAP_COUNT_BANDS):
        row_y = y + index * 28
        elements.append(
            f'<rect x="{x}" y="{row_y - 12}" width="15" height="15" rx="2" fill="{band.color}"/>'
        )
        elements.append(
            f'<text x="{x + 23}" y="{row_y}" font-family="sans-serif" '
            f'font-size="11" fill="#334e68">{band.label} dataset rows</text>'
        )
    elements.extend(
        (
            f'<text x="{x}" y="252" font-family="sans-serif" font-size="10" '
            'fill="#334e68">One representative point per valid</text>',
            f'<text x="{x}" y="268" font-family="sans-serif" font-size="10" '
            'fill="#334e68">polygon-bearing dataset row</text>',
            f'<text x="{x}" y="291" font-family="sans-serif" font-size="10" '
            'fill="#334e68">H3 resolution 3</text>',
            f'<text x="{x}" y="307" font-family="sans-serif" font-size="10" '
            'fill="#334e68">Counts include unlabeled rows</text>',
            f'<text x="{x}" y="338" font-family="sans-serif" font-size="10" '
            'fill="#52606d">Missing or invalid geometry is omitted</text>',
            f'<text x="{x}" y="354" font-family="sans-serif" font-size="10" '
            'fill="#52606d">Points outside longitude [-180, 180]</text>',
            f'<text x="{x}" y="370" font-family="sans-serif" font-size="10" '
            'fill="#52606d">or latitude [-90, 90] are omitted</text>',
        )
    )
    return elements


def _extract_source_configs(source_readme: str | None) -> str | None:
    """Return only the source README's top-level Dataset Viewer configs block."""

    if source_readme is None:
        return None
    lines = source_readme.splitlines()
    if not _has_frontmatter(lines):
        return None
    frontmatter_end = _frontmatter_end(lines)
    if frontmatter_end is None:
        return None
    config_start = _config_start(lines, frontmatter_end)
    if config_start is None:
        return None
    config_end = _config_end(lines, config_start, frontmatter_end)
    return _config_block(lines, config_start, config_end)


def _extract_source_license(source_readme: str | None) -> str | None:
    """Return a simple top-level license identifier from the source card."""

    lines = _source_frontmatter_lines(source_readme)
    if lines is None:
        return None
    match = re.search(
        r"(?m)^license:[ \t]*([A-Za-z0-9][A-Za-z0-9.+_-]*)[ \t]*$",
        "\n".join(lines),
    )
    return None if match is None else match.group(1)


def _source_frontmatter_lines(source_readme: str | None) -> list[str] | None:
    if source_readme is None:
        return None
    lines = source_readme.splitlines()
    if not _has_frontmatter(lines):
        return None
    frontmatter_end = _frontmatter_end(lines)
    if frontmatter_end is None:
        return None
    return lines[1:frontmatter_end]


def _has_frontmatter(lines: list[str]) -> bool:
    return bool(lines) and lines[0].strip() == "---"


def _frontmatter_end(lines: list[str]) -> int | None:
    return next(
        (index for index, line in enumerate(lines[1:], start=1) if line.strip() == "---"),
        None,
    )


def _config_start(lines: list[str], frontmatter_end: int) -> int | None:
    return next(
        (
            index
            for index, line in enumerate(lines[1:frontmatter_end], start=1)
            if line == "configs:"
        ),
        None,
    )


def _config_end(lines: list[str], config_start: int, frontmatter_end: int) -> int:
    return next(
        (
            index
            for index in range(config_start + 1, frontmatter_end)
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_-]*\s*:.*", lines[index])
        ),
        frontmatter_end,
    )


def _config_block(lines: list[str], config_start: int, config_end: int) -> str:
    return "\n".join(lines[config_start:config_end]).rstrip()


def _title(dataset_name: str) -> str:
    return dataset_name.replace("-", " ").title()


def _markdown_code(code: str | None) -> str:
    return f"`{_markdown_cell(code or '—')}`"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")
