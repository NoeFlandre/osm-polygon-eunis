"""Atomic region checkpoints and bounded unique-candidate census index."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Mapping, Sequence
from importlib import import_module
from pathlib import Path

from .eunis_geometry_census_sources import (
    H3_RESOLUTION,
    MAX_PER_CELL,
    MINIMUM_H3_CELL_COUNT,
    TARGET,
    _geometry,
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_new_json(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".partial")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite census artifact {path}")
    encoded = (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode("utf-8")
    with temporary.open("xb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    temporary.replace(path)


def _region_signature(
    run_signature: str, stream: str, label_path: str, raw_paths: Sequence[str]
) -> str:
    payload = json.dumps(
        {
            "run": run_signature,
            "stream": stream,
            "label_path": label_path,
            "raw_paths": list(raw_paths),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_completed_region(
    checkpoint: Path, output: Path, signature: str
) -> dict[str, object] | None:
    if not checkpoint.exists():
        if output.exists() or output.with_name(output.name + ".partial").exists():
            raise FileExistsError(f"region output exists without a checkpoint: {output}")
        return None
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("signature") != signature:
        raise ValueError(f"region checkpoint input mismatch: {checkpoint}")
    if not output.is_file() or _sha256_file(output) != payload.get("output_sha256"):
        raise ValueError(f"region output digest mismatch: {output}")
    payload["output_path"] = str(output)
    return payload


def _open_index(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute(
        "CREATE TABLE IF NOT EXISTS eligible ("
        "source TEXT NOT NULL, decision TEXT NOT NULL, polygon_key TEXT NOT NULL, "
        "geometry_sha256 TEXT NOT NULL, h3_cell TEXT NOT NULL, "
        "PRIMARY KEY (source, decision, polygon_key))"
    )
    db.commit()
    return db


def _index_candidate(db: sqlite3.Connection, source: str, row: Mapping[str, object]) -> None:
    h3 = import_module("h3")

    geometry = _geometry(row["geometry"])
    point = geometry.representative_point()
    cell = h3.latlng_to_cell(point.y, point.x, H3_RESOLUTION)
    key_fields = tuple(
        str(row.get(key) or "") for key in ("source_pbf", "osm_type", "osm_id", "polygon_id")
    )
    polygon_key = json.dumps(key_fields, separators=(",", ":"))
    geometry_digest = hashlib.sha256(geometry.wkb).hexdigest()
    decision = str(row["decision"])
    existing = db.execute(
        "SELECT geometry_sha256, h3_cell FROM eligible "
        "WHERE source=? AND decision=? AND polygon_key=?",
        (source, decision, polygon_key),
    ).fetchone()
    if existing is not None:
        if existing != (geometry_digest, cell):
            raise ValueError(
                f"duplicate polygon has conflicting geometry: {source}/{decision}/{polygon_key}"
            )
        return
    db.execute(
        "INSERT INTO eligible VALUES (?, ?, ?, ?, ?)",
        (source, decision, polygon_key, geometry_digest, cell),
    )


def _index_region(db: sqlite3.Connection, source: str, output: Path) -> None:
    with db, output.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                _index_candidate(db, source, json.loads(line))


def _summary(db: sqlite3.Connection, requested_sources: Sequence[str]) -> dict[str, object]:
    rows = db.execute(
        "SELECT source, decision, h3_cell, count(*) FROM eligible "
        "GROUP BY source, decision, h3_cell"
    ).fetchall()
    cells: dict[str, dict[str, dict[str, int]]] = {}
    for source, decision, cell, count in rows:
        cells.setdefault(source, {}).setdefault(decision, {})[cell] = count
    summary: dict[str, object] = {}
    for source in requested_sources:
        by_decision = cells.get(source, {})
        decisions = ("yes", "no", *sorted(set(by_decision) - {"yes", "no"}))
        summary[source] = {
            decision: _decision_summary(by_decision.get(decision, {})) for decision in decisions
        }
    return {
        "h3_resolution": H3_RESOLUTION,
        "max_per_h3_cell_per_decision": MAX_PER_CELL,
        "target_per_decision": TARGET,
        "sources": summary,
    }


def _decision_summary(cell_counts: Mapping[str, int]) -> dict[str, object]:
    capped_capacity = sum(min(count, MAX_PER_CELL) for count in cell_counts.values())
    return {
        "unique_eligible_polygons": sum(cell_counts.values()),
        "h3_cells": len(cell_counts),
        f"capacity_at_{MAX_PER_CELL}_per_cell": capped_capacity,
        "max_eligible_per_cell": max(cell_counts.values(), default=0),
        f"can_reach_{TARGET}_and_{MINIMUM_H3_CELL_COUNT}_cells": (
            capped_capacity >= TARGET and len(cell_counts) >= MINIMUM_H3_CELL_COUNT
        ),
    }
