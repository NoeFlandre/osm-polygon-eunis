"""Deterministic, storage-bounded Hugging Face dataset-card artifacts."""

from __future__ import annotations

import html
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from .domain import EunisResult
from .fileio import sha256_file
from .geometry import parse_geometry

_CELL_SIZE = 2.0
_MAP_PATH = "eunis/world-map.svg"
_NO_LABEL = "No EUNIS label"
_PALETTE = (
    "#1b9e77",
    "#d95f02",
    "#7570b3",
    "#e7298a",
    "#66a61e",
    "#e6ab02",
    "#a6761d",
    "#666666",
    "#1f78b4",
    "#b2df8a",
    "#fb9a99",
    "#6a3d9a",
    "#ff7f00",
    "#cab2d6",
    "#33a02c",
    "#a6cee3",
    "#fdbf6f",
    "#b15928",
    "#8dd3c7",
    "#bebada",
)

# Low-detail land silhouettes keep the card self-contained and avoid shipping a
# second geospatial dependency or a large Natural Earth dataset. They are only
# a visual backdrop; all data-driven positions are the colored points.
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


class DatasetCardAccumulator:
    """Collect label counts and bounded map bins while shards stream past."""

    def __init__(self, *, cell_size: float = _CELL_SIZE) -> None:
        if cell_size <= 0.0 or cell_size > 180.0:
            raise ValueError("cell_size must be in (0, 180]")
        self.cell_size = cell_size
        self._counts: dict[str | None, int] = {}
        self._names: dict[str | None, str] = {}
        self._bins: dict[tuple[str | None, int, int], int] = {}
        self._total_rows = 0

    @property
    def total_rows(self) -> int:
        return self._total_rows

    def observe(self, result: EunisResult, raw_geometry: object) -> None:
        """Record one output label and at most one bounded map cell."""

        code, name = _label_identity(result)
        self._record_label(code, name)
        self._record_map_bin(code, raw_geometry)

    def _record_label(self, code: str | None, name: str) -> None:
        previous_name = self._names.setdefault(code, name)
        if previous_name != name:
            raise ValueError(f"dataset card has conflicting names for {code}")
        self._counts[code] = self._counts.get(code, 0) + 1
        self._total_rows += 1

    def _record_map_bin(self, code: str | None, raw_geometry: object) -> None:
        geometry = parse_geometry(raw_geometry)
        if geometry is None:
            return
        point = geometry.representative_point()
        longitude = float(point.x)
        latitude = float(point.y)
        if not (-180.0 <= longitude <= 180.0 and -90.0 <= latitude <= 90.0):
            return
        longitude_index = _coordinate_index(longitude, -180.0, 180.0, self.cell_size)
        latitude_index = _coordinate_index(latitude, -90.0, 90.0, self.cell_size)
        key = (code, longitude_index, latitude_index)
        self._bins[key] = self._bins.get(key, 0) + 1

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
        map_path.write_text(self._render_map(dataset_name), encoding="utf-8")
        readme_path = directory / "README.md"
        readme_path.write_text(
            self._render_readme(
                dataset_name=dataset_name,
                source_repo=source_repo,
                target_repo=target_repo,
                source_revision=source_revision,
                reference_version=reference_version,
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
    ) -> str:
        title = _title(dataset_name)
        lines = [
            "---",
            f"pretty_name: {title}",
            "tags:",
            "- geospatial",
            "- osm",
            "- eunis",
            "- ecology",
            "---",
            "",
            f"# {title}",
            "",
            "This dataset preserves the source rows and adds four nullable EUNIS fields "
            "to each polygon-bearing Parquet shard.",
            "",
            "![Static world map of EUNIS label distribution](./eunis/world-map.svg)",
            "",
            "The map uses a representative point for each polygon and 2-degree geographic "
            "bins to keep the static artifact small. Colors identify EUNIS codes; the "
            "complete distribution is listed below. Unlabeled polygons are shown in gray.",
            "",
            "## Label distribution",
            "",
            "| EUNIS code | Label | Polygons | Share |",
            "|:--|:--|--:|--:|",
        ]
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
                "used only to prune candidates. Empty, invalid, and out-of-reference "
                "polygons receive null EUNIS fields.",
                "",
                f"Source: `{source_repo}` at `{source_revision}`.",
                f"Output: `{target_repo}`.",
                f"Reference: official EEA EUNIS assets pinned as `{reference_version}`.",
            )
        )
        return "\n".join(lines) + "\n"

    def _render_map(self, dataset_name: str) -> str:
        width, height = 1200, 660
        plot_x, plot_y, plot_width, plot_height = 50, 75, 835, 510
        summaries = self.summaries()
        colors = _label_colors(summaries)
        elements = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            (
                f'<text x="{plot_x}" y="30" font-family="sans-serif" '
                f'font-size="22" font-weight="600" fill="#1f2933">'
                f"{html.escape(_title(dataset_name))}</text>"
            ),
            (
                '<text x="50" y="52" font-family="sans-serif" font-size="12" '
                'fill="#52606d">Representative polygon locations · 2° bins · '
                "colors are EUNIS labels</text>"
            ),
            (
                f'<rect x="{plot_x}" y="{plot_y}" width="{plot_width}" '
                f'height="{plot_height}" rx="4" fill="#f7fafc" stroke="#cbd5e0"/>'
            ),
        ]
        elements.extend(_grid_elements(plot_x, plot_y, plot_width, plot_height))
        elements.extend(_land_elements(plot_x, plot_y, plot_width, plot_height))
        elements.extend(self._point_elements(plot_x, plot_y, plot_width, plot_height, colors))
        elements.extend(_legend_elements(summaries, colors, x=925, y=105))
        elements.append("</svg>")
        return "\n".join(elements) + "\n"

    def _point_elements(
        self,
        plot_x: int,
        plot_y: int,
        plot_width: int,
        plot_height: int,
        colors: Mapping[str | None, str],
    ) -> list[str]:
        elements: list[str] = []
        for (code, longitude_index, latitude_index), count in sorted(
            self._bins.items(), key=lambda item: (item[0][0] is None, item[0])
        ):
            longitude = -180.0 + (longitude_index + 0.5) * self.cell_size
            latitude = -90.0 + (latitude_index + 0.5) * self.cell_size
            x = _map_x(longitude, plot_x, plot_width)
            y = _map_y(latitude, plot_y, plot_height)
            radius = min(11.0, 3.0 + math.sqrt(count))
            label = _label_text(code, self._names.get(code, _NO_LABEL))
            elements.append(
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{colors[code]}" '
                f'fill-opacity="0.82" stroke="#ffffff" stroke-width="0.7">'
                f"<title>{html.escape(label)} · {count:,} polygon(s)</title></circle>"
            )
        return elements


def _coordinate_index(value: float, lower: float, upper: float, cell_size: float) -> int:
    cells = max(1, math.ceil((upper - lower) / cell_size))
    return min(cells - 1, max(0, math.floor((value - lower) / cell_size)))


def _label_identity(result: EunisResult) -> tuple[str | None, str]:
    code = result.code or None
    return code, result.name or (_NO_LABEL if code is None else "Unknown EUNIS label")


def _map_x(longitude: float, plot_x: int, plot_width: int) -> float:
    return plot_x + (longitude + 180.0) / 360.0 * plot_width


def _map_y(latitude: float, plot_y: int, plot_height: int) -> float:
    return plot_y + (90.0 - latitude) / 180.0 * plot_height


def _grid_elements(plot_x: int, plot_y: int, plot_width: int, plot_height: int) -> list[str]:
    elements: list[str] = []
    for longitude in range(-120, 180, 60):
        x = _map_x(float(longitude), plot_x, plot_width)
        elements.append(
            f'<line x1="{x:.2f}" y1="{plot_y}" x2="{x:.2f}" '
            f'y2="{plot_y + plot_height}" stroke="#dfe7ef" stroke-width="0.7"/>'
        )
    for latitude in range(-60, 90, 30):
        y = _map_y(float(latitude), plot_y, plot_height)
        elements.append(
            f'<line x1="{plot_x}" y1="{y:.2f}" '
            f'x2="{plot_x + plot_width}" y2="{y:.2f}" '
            'stroke="#dfe7ef" stroke-width="0.7"/>'
        )
    return elements


def _land_elements(plot_x: int, plot_y: int, plot_width: int, plot_height: int) -> list[str]:
    elements = []
    for landmass in _LANDMASSES:
        points = " ".join(
            (
                f"{_map_x(longitude, plot_x, plot_width):.2f},"
                f"{_map_y(latitude, plot_y, plot_height):.2f}"
            )
            for longitude, latitude in landmass
        )
        elements.append(
            f'<polygon points="{points}" fill="#e5ebf0" stroke="#cbd5e0" stroke-width="0.8"/>'
        )
    return elements


def _label_colors(summaries: tuple[LabelSummary, ...]) -> dict[str | None, str]:
    colors: dict[str | None, str] = {None: "#8a98a8"}
    index = 0
    for summary in summaries:
        if summary.code is not None:
            colors[summary.code] = _PALETTE[index % len(_PALETTE)]
            index += 1
    return colors


def _legend_elements(
    summaries: tuple[LabelSummary, ...],
    colors: Mapping[str | None, str],
    *,
    x: int,
    y: int,
) -> list[str]:
    elements = [
        (
            f'<text x="{x}" y="75" font-family="sans-serif" font-size="15" '
            'font-weight="600" fill="#1f2933">EUNIS labels</text>'
        ),
    ]
    visible = summaries[:18]
    for index, summary in enumerate(visible):
        row_y = y + index * 24
        code = summary.code or "—"
        label = _truncate(summary.name, 24)
        elements.append(
            f'<rect x="{x}" y="{row_y - 11}" width="12" height="12" '
            f'rx="2" fill="{colors[summary.code]}"/>'
        )
        elements.append(
            f'<text x="{x + 19}" y="{row_y}" font-family="sans-serif" '
            f'font-size="11" fill="#334e68">{html.escape(code)} · '
            f"{html.escape(label)}</text>"
        )
    if len(summaries) > len(visible):
        row_y = y + len(visible) * 24 + 5
        elements.append(
            f'<text x="{x}" y="{row_y}" font-family="sans-serif" '
            f'font-size="10" fill="#52606d">+ {len(summaries) - len(visible)} '
            "more in the table</text>"
        )
    return elements


def _label_text(code: str | None, name: str) -> str:
    return f"{code or 'No EUNIS label'} · {name}"


def _truncate(value: str, length: int) -> str:
    return value if len(value) <= length else f"{value[: length - 1]}…"


def _title(dataset_name: str) -> str:
    return dataset_name.replace("-", " ").title()


def _markdown_code(code: str | None) -> str:
    return f"`{_markdown_cell(code or '—')}`"


def _markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


