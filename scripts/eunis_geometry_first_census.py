"""Census pinned relevance labels with native geometry eligibility before text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from .eunis_geometry_census_core import (
    MINIMUM_NATIVE_OVERLAP_PERCENTAGE,
    CensusCounts,
    iter_geometry_first_candidates,
)
from .eunis_geometry_census_persistence import (
    _index_region,
    _load_completed_region,
    _open_index,
    _region_signature,
    _sha256_file,
    _summary,
    _write_new_json,
)
from .eunis_geometry_census_sources import (
    BATCH_SIZE,
    EXPECTED_NATIVE_ASSET_COUNT,
    H3_RESOLUTION,
    LABEL_PINS,
    MAX_PER_CELL,
    MINIMUM_H3_CELL_COUNT,
    RAW_PINS,
    SOURCE_VERSION,
    TARGET,
    _geometry_parameters,
    _geometry_rows,
    _json_row,
    _label_paths,
    _raw_paths,
    _text_query,
    _url,
    geometry_sql,
)


def _source_streams(source: str) -> tuple[str, ...]:
    return ("wikipedia", "wikivoyage") if source == "wiki" else (source,)


_CENSUS_ALGORITHM_VERSION = "geometry-first-v3"
_ASSIGNMENT_GATE = (
    "unique production assignment with inclusive "
    f">={MINIMUM_NATIVE_OVERLAP_PERCENTAGE:g}% polygon coverage"
)
EXPECTED_NATIVE_TOTAL_BYTES = 5_007_209_626


def _census_code_fingerprint(source_root: Path) -> str:
    """Hash the implementation modules whose behavior defines the census."""
    digest = hashlib.sha256()
    package_root = source_root / "src" / "osm_polygon_eunis"
    production_paths = sorted(package_root.rglob("*.py"))
    if not production_paths:
        raise FileNotFoundError(f"census production source tree is missing: {package_root}")
    census_paths = (
        Path(__file__),
        Path(__file__).with_name("eunis_geometry_census_core.py"),
        Path(__file__).with_name("eunis_geometry_census_sources.py"),
        Path(__file__).with_name("eunis_geometry_census_persistence.py"),
    )
    for path in (*census_paths, *production_paths):
        if not path.is_file():
            raise FileNotFoundError(f"census code fingerprint source is missing: {path}")
        label = (
            path.relative_to(package_root).as_posix()
            if path.is_relative_to(package_root)
            else path.name
        )
        digest.update(label.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as source_file:
            while chunk := source_file.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _unique_text_mapping(
    rows: Sequence[tuple[object, object]],
) -> dict[str, str | None]:
    texts: dict[str, str | None] = {}
    for key, sentence in rows:
        normalized_key = str(key)
        if normalized_key in texts:
            raise ValueError(f"text lookup returned duplicate sentence key: {normalized_key}")
        texts[normalized_key] = None if sentence is None else str(sentence)
    return texts


def _load_httpfs(db: Any) -> None:
    """Require an already available httpfs extension without installing it."""
    try:
        db.execute("SET autoinstall_known_extensions=false")
        db.execute("SET autoload_known_extensions=false")
        extension = db.execute(
            "SELECT installed FROM duckdb_extensions() WHERE extension_name='httpfs'"
        ).fetchone()
    except Exception as error:
        raise RuntimeError(
            "could not verify DuckDB's httpfs extension cache; "
            "the census does not install extensions"
        ) from error
    if extension is None or extension[0] is not True:
        raise RuntimeError(
            "DuckDB's httpfs extension must already be available in its configured "
            "extension cache; the census does not install extensions"
        )
    try:
        db.execute("LOAD httpfs")
    except Exception as error:
        raise RuntimeError(
            "DuckDB's httpfs extension must already be available in its configured "
            "extension cache before the census can read pinned remote Parquet files; "
            "the census does not install extensions"
        ) from error


def _validate_native_asset(path: Path, asset: Mapping[str, object]) -> str:
    if not path.is_file():
        raise ValueError(f"native asset path check failed (file missing): {path}")
    actual_size = path.stat().st_size
    expected_size = asset["expected_bytes"]
    if actual_size != expected_size:
        raise ValueError(
            f"native asset size check failed for {path}: "
            f"expected {expected_size} bytes, found {actual_size}"
        )
    actual_sha256 = _sha256_file(path)
    if actual_sha256 != asset.get("expected_sha256"):
        raise ValueError(
            f"native asset SHA-256 check failed for {path}: "
            f"expected {asset.get('expected_sha256')}, found {actual_sha256}"
        )
    return actual_sha256


def _read_native_inventory_source(
    native_doc: Mapping[str, object],
    native_manifest_path: Path,
    *,
    expected_count: int,
    expected_total_bytes: int,
) -> tuple[list[object], Path]:
    inventory_source = native_doc.get("inventory_source")
    if not isinstance(inventory_source, str) or not inventory_source:
        raise ValueError("native download manifest lacks its inventory source path")
    inventory_path = Path(inventory_source)
    if not inventory_path.is_absolute():
        inventory_path = native_manifest_path.parent / inventory_path
    inventory_path = inventory_path.resolve()
    if not inventory_path.is_file():
        raise FileNotFoundError(f"authoritative native inventory is missing: {inventory_path}")
    inventory_doc = json.loads(inventory_path.read_text(encoding="utf-8"))
    reference = inventory_doc.get("eea_reference")
    if not isinstance(reference, dict) or reference.get("source_version") != SOURCE_VERSION:
        raise ValueError("authoritative native inventory has an unexpected source version")
    expected_assets = reference.get("native_tiffs")
    if not isinstance(expected_assets, list) or len(expected_assets) != expected_count:
        found = len(expected_assets) if isinstance(expected_assets, list) else "invalid"
        raise ValueError(f"expected {expected_count} authoritative native assets, found {found}")
    if native_doc.get("source_version") != SOURCE_VERSION:
        raise ValueError("native download manifest has an unexpected source version")
    if (
        native_doc.get("asset_count") != expected_count
        or native_doc.get("verified_count") != expected_count
        or native_doc.get("failed_count") != 0
        or native_doc.get("expected_total_bytes") != expected_total_bytes
    ):
        raise ValueError("native download manifest summary does not match the verified inventory")
    return expected_assets, inventory_path


def _index_expected_native_assets(
    expected_assets: Sequence[object], expected_count: int, expected_total_bytes: int
) -> dict[str, Mapping[str, object]]:
    expected_by_code: dict[str, Mapping[str, object]] = {}
    source_total = 0
    for expected in expected_assets:
        if not isinstance(expected, dict):
            raise TypeError("authoritative native inventory has a malformed asset record")
        code = expected.get("code")
        size = expected.get("size_bytes")
        digest = expected.get("sha256")
        if (
            not isinstance(code, str)
            or not code
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 1
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
        ):
            raise ValueError("authoritative native inventory has invalid code, size, or SHA-256")
        if code in expected_by_code:
            raise ValueError(f"authoritative native inventory repeats code: {code}")
        expected_by_code[code] = expected
        source_total += size
    if len(expected_by_code) != expected_count or source_total != expected_total_bytes:
        raise ValueError("authoritative native inventory count or byte total is inconsistent")
    return expected_by_code


def _validate_downloaded_native_asset(  # noqa: C901, PLR0913
    asset: object,
    expected_by_code: Mapping[str, Mapping[str, object]],
    native_root: Path,
    taxonomy_labels: Mapping[str, object],
    parse_layer_code: Any,
    seen_codes: set[str],
    *,
    resolved_paths: set[Path],
) -> tuple[Mapping[str, object], Path, int]:
    if not isinstance(asset, dict):
        raise TypeError("native download manifest has a malformed asset record")
    code = asset.get("code")
    if not isinstance(code, str) or not code:
        raise ValueError("native download manifest asset is missing a code")
    if code in seen_codes:
        raise ValueError(f"native download manifest repeats code: {code}")
    expected = expected_by_code.get(code)
    if expected is None:
        raise ValueError(f"native download manifest contains unapproved code: {code}")
    expected_size = expected.get("size_bytes")
    expected_digest = expected.get("sha256")
    if not isinstance(expected_size, int) or isinstance(expected_size, bool):
        raise TypeError(f"native source inventory has invalid size for {code}")
    state = asset.get("state")
    verified_states = {
        "already_verified",
        "downloaded_verified",
        "verified_existing_partial_promoted_by_link",
    }
    if (
        asset.get("expected_bytes") != expected_size
        or asset.get("expected_sha256") != expected_digest
        or asset.get("actual_bytes") != expected_size
        or asset.get("actual_sha256") != expected_digest
        or not isinstance(state, str)
        or state not in verified_states
    ):
        raise ValueError(f"native download record is not verified against source: {code}")
    if code not in taxonomy_labels:
        raise ValueError(f"native asset code lacks a taxonomy label: {code}")
    path_value = asset.get("file")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"native download record lacks a file path: {code}")
    path = Path(path_value)
    if not path.is_absolute():
        path = native_root / path
    path = path.resolve()
    if path in resolved_paths:
        raise ValueError(f"native download manifest repeats a file path: {path}")
    if parse_layer_code(path.name) != code:
        raise ValueError(f"native asset filename/code mismatch for {code}: {path.name}")
    seen_codes.add(code)
    resolved_paths.add(path)
    return asset, path, expected_size


def _validate_native_inventory(  # noqa: PLR0913
    native_doc: Mapping[str, object],
    native_manifest_path: Path,
    native_root: Path,
    taxonomy_labels: Mapping[str, object],
    parse_layer_code: Any,
    *,
    expected_count: int = EXPECTED_NATIVE_ASSET_COUNT,
    expected_total_bytes: int = EXPECTED_NATIVE_TOTAL_BYTES,
) -> tuple[tuple[tuple[Mapping[str, object], Path], ...], Path]:
    """Validate the download report against its authoritative source inventory."""
    expected_assets, inventory_path = _read_native_inventory_source(
        native_doc,
        native_manifest_path,
        expected_count=expected_count,
        expected_total_bytes=expected_total_bytes,
    )
    expected_by_code = _index_expected_native_assets(
        expected_assets, expected_count, expected_total_bytes
    )
    assets = native_doc.get("assets")
    if not isinstance(assets, list) or len(assets) != expected_count:
        raise ValueError(f"expected {expected_count} downloaded native assets")
    resolved_paths: set[Path] = set()
    seen_codes: set[str] = set()
    actual_total = 0
    validated: list[tuple[Mapping[str, object], Path]] = []
    for asset in assets:
        record, path, size = _validate_downloaded_native_asset(
            asset,
            expected_by_code,
            native_root,
            taxonomy_labels,
            parse_layer_code,
            seen_codes,
            resolved_paths=resolved_paths,
        )
        actual_total += size
        validated.append((record, path))
    if seen_codes != expected_by_code.keys() or actual_total != expected_total_bytes:
        missing = sorted(expected_by_code.keys() - seen_codes)
        unexpected = sorted(seen_codes - expected_by_code.keys())
        raise ValueError(
            f"native asset inventory mismatch: missing={missing}; unexpected={unexpected}"
        )
    return tuple(validated), inventory_path


def _annotate_geometry_rows(
    rows: Any,
    *,
    stream: str,
    raw_urls: Sequence[str],
) -> Any:
    """Attach text-fetch URLs without materializing the bounded row stream."""
    for row in rows:
        if stream == "website":
            row["_raw_polygon_url"] = raw_urls[0]
        else:
            row["_raw_text_url"] = raw_urls[0]
            row["_raw_polygon_url"] = raw_urls[1]
        yield row


def _parse_production_geometry(value: object) -> Any:
    """Apply the production parser's repair and malformed-input policy."""
    from osm_polygon_eunis.geometry import parse_geometry  # noqa: PLC0415

    return parse_geometry(value)


def _prepare_run(
    output_dir: Path,
    run_payload: dict[str, object],
    *,
    resume: bool,
) -> str:
    identity = dict(run_payload)
    identity.pop("created_at_utc", None)
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    signature = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    run_path = output_dir / "run.json"
    if output_dir.exists():
        if not resume or not run_path.is_file():
            raise FileExistsError(
                "output directory exists; pass --resume for an identical pinned run"
            )
        existing = json.loads(run_path.read_text(encoding="utf-8"))
        if existing.get("run_signature") != signature:
            raise ValueError("resume directory does not match the current pinned inputs")
    else:
        output_dir.mkdir(parents=True, exist_ok=False)
        run_payload["run_signature"] = signature
        _write_new_json(run_path, run_payload)
    if (output_dir / "complete.json").exists():
        raise FileExistsError("this census run is already complete")
    return signature


def _positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer greater than zero") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be an integer greater than zero")
    return parsed


def _run_region_checkpoints(
    regions: Sequence[tuple[str, str, str]],
    *,
    output_dir: Path,
    max_new_regions: int | None,
    process_region: Any,
    index_region: Any,
) -> tuple[int, int, int]:
    """Visit and reindex completed checkpoints, bounding only new completions."""
    newly_completed = 0
    visited = 0
    unvisited = 0
    for source, stream, label_path in regions:
        shard = label_path.rsplit("/", 1)[-1].removesuffix("-latest.parquet")
        checkpoint = output_dir / "checkpoints" / stream / f"{shard}.json"
        output = output_dir / "regions" / stream / f"{shard}.jsonl"
        existed = checkpoint.exists()
        if not existed and max_new_regions is not None and newly_completed >= max_new_regions:
            if output.exists() or output.with_name(output.name + ".partial").exists():
                raise FileExistsError(f"region output exists without a checkpoint: {output}")
            unvisited += 1
            continue
        payload = process_region(source, stream, label_path)
        index_region(source, output_dir / "regions" / stream / f"{payload['shard']}.jsonl")
        visited += 1
        if not existed:
            newly_completed += 1
    return newly_completed, visited, unvisited


def _process_region(  # noqa: C901, PLR0913, PLR0915
    db: Any,
    output_dir: Path,
    *,
    source: str,
    stream: str,
    label_path: str,
    label_url: str,
    raw_paths: Sequence[str],
    raw_urls: Sequence[str],
    run_signature: str,
    reference: Any,
    native_extent: Any,
) -> dict[str, object]:
    shard = label_path.rsplit("/", 1)[-1].removesuffix("-latest.parquet")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", shard) or shard in {".", ".."}:
        raise ValueError(f"unsafe pinned shard basename: {shard!r}")
    signature = _region_signature(run_signature, stream, label_path, raw_paths)
    output = output_dir / "regions" / stream / f"{shard}.jsonl"
    checkpoint = output_dir / "checkpoints" / stream / f"{shard}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    completed = _load_completed_region(checkpoint, output, signature)
    if completed is not None:
        return completed
    counts = CensusCounts()

    def classify(value: object) -> Any:
        from osm_polygon_eunis.geometry import to_equal_area  # noqa: PLC0415

        geometry = _parse_production_geometry(value)
        if geometry is None or geometry.is_empty or not geometry.is_valid:
            counts.invalid_geometry_rows += 1
            return None
        if not geometry.intersects(native_extent):
            counts.outside_native_extent_rows += 1
            return None
        polygon = to_equal_area(geometry)
        if polygon is None or polygon.is_empty or not polygon.is_valid or polygon.area <= 0.0:
            counts.projection_invalid_rows += 1
            return None
        before = reference.intersection_errors
        result = reference.overlap(polygon)
        counts.native_overlap_errors += reference.intersection_errors - before
        if result.code is None:
            if (
                result.overlap_percentage is not None
                and result.overlap_percentage >= MINIMUM_NATIVE_OVERLAP_PERCENTAGE
            ):
                counts.ambiguous_native_rows += 1
            else:
                counts.unassigned_native_rows += 1
        elif (
            result.overlap_percentage is None
            or result.overlap_percentage < MINIMUM_NATIVE_OVERLAP_PERCENTAGE
        ):
            raise ValueError("production raster assignment violated the inclusive 80% gate")
        return result

    text_db = db.cursor()

    def fetch_texts(rows: Sequence[Mapping[str, object]]) -> Mapping[str, str | None]:
        sql_text, text_params = _text_query(stream, rows)
        cursor = text_db.execute(sql_text, text_params)
        return _unique_text_mapping(cursor.fetchall())

    geometry_query = geometry_sql(stream)
    geometry_params = _geometry_parameters(stream, source, label_path, shard, label_url, raw_urls)
    geometry_db = db.cursor()
    geometry_rows = _annotate_geometry_rows(
        _geometry_rows(geometry_db, geometry_query, geometry_params),
        stream=stream,
        raw_urls=raw_urls,
    )
    temporary = output.with_name(output.name + ".partial")
    if output.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite incomplete region output: {output}")
    digest = hashlib.sha256()
    row_count = 0
    with temporary.open("xb") as target:
        for candidate in iter_geometry_first_candidates(
            geometry_rows,
            source=source,
            stream=stream,
            classify=classify,
            fetch_texts=fetch_texts,
            batch_size=BATCH_SIZE,
            counts=counts,
        ):
            encoded = (
                json.dumps(_json_row(candidate), sort_keys=True, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            target.write(encoded)
            digest.update(encoded)
            row_count += 1
        target.flush()
        os.fsync(target.fileno())
    temporary.replace(output)
    checkpoint_payload: dict[str, object] = {
        "signature": signature,
        "source": source,
        "stream": stream,
        "shard": shard,
        "label_path": label_path,
        "label_repo": LABEL_PINS[stream][0],
        "label_revision": LABEL_PINS[stream][1],
        "raw_paths": list(raw_paths),
        "raw_repo": RAW_PINS[stream][0],
        "raw_revision": RAW_PINS[stream][1],
        "candidate_rows": row_count,
        "output_path": str(output),
        "output_sha256": digest.hexdigest(),
        "counts": {name: getattr(counts, name) for name in counts.__dataclass_fields__},
    }
    _write_new_json(checkpoint, checkpoint_payload)
    return checkpoint_payload


def main() -> int:  # noqa: C901, PLR0915
    """Run the bounded census over the selected pinned sources."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="checked-out PR source root containing src/osm_polygon_eunis",
    )
    parser.add_argument(
        "--native-root",
        type=Path,
        required=True,
        help="directory containing the verified 136 native EEA assets",
    )
    parser.add_argument("--native-manifest", type=Path, required=True)
    parser.add_argument("--taxonomy", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="durable project output directory, not job-local scratch",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        required=True,
        help="job-local scratch directory for DuckDB spill",
    )
    parser.add_argument(
        "--sources",
        nargs="+",
        choices=("website", "description", "wiki"),
        default=("website", "description", "wiki"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-new-regions",
        type=_positive_int,
        help="stop before the next uncheckpointed region after this many new checkpoints",
    )
    args = parser.parse_args()
    if not args.temp_dir.is_dir():
        raise SystemExit("--temp-dir must already exist as job-local scratch")
    if args.output_dir.resolve().is_relative_to(args.temp_dir.resolve()):
        raise SystemExit("--output-dir must be durable project storage, outside --temp-dir")
    import sys  # noqa: PLC0415

    sys.path.insert(0, str((args.source_root / "src").resolve()))
    duckdb = import_module("duckdb")
    from shapely.geometry import box  # noqa: PLC0415

    from osm_polygon_eunis.domain import EunisTaxonomy  # noqa: PLC0415
    from osm_polygon_eunis.raster_aggregation import (  # noqa: PLC0415
        RasterLayer,
        parse_layer_code,
    )
    from osm_polygon_eunis.raster_reference import RasterReference  # noqa: PLC0415

    with args.taxonomy.open(encoding="utf-8") as stream:
        taxonomy_doc = json.load(stream)
    with args.native_manifest.open(encoding="utf-8") as stream:
        native_doc = json.load(stream)
    taxonomy_labels = taxonomy_doc.get("labels")
    if not isinstance(taxonomy_labels, dict):
        raise TypeError("taxonomy manifest lacks a labels mapping")
    native_assets, native_inventory_path = _validate_native_inventory(
        native_doc,
        args.native_manifest,
        args.native_root,
        taxonomy_labels,
        parse_layer_code,
    )
    layers = []
    for asset, path in native_assets:
        _validate_native_asset(path, asset)
        code = str(asset["code"])
        name = taxonomy_labels.get(code)
        if not isinstance(name, str) or not name:
            raise ValueError(f"native asset lacks a taxonomy label: {path.name}")
        layers.append(RasterLayer(code, name, path, SOURCE_VERSION))
    taxonomy = EunisTaxonomy(
        labels=taxonomy_doc["labels"],
        levels=taxonomy_doc["levels"],
        parents=taxonomy_doc["parents"],
        source_record_id=taxonomy_doc.get("source_record_id"),
        source_sha256=taxonomy_doc.get("source_sha256"),
    )
    if len(set(args.sources)) != len(args.sources):
        raise ValueError("source selection must be unique")
    streams = tuple(stream for source in args.sources for stream in _source_streams(source))
    if len(streams) != len(set(streams)):
        raise ValueError("duplicate stream in source selection")
    path_inventory = {stream: _label_paths(stream) for stream in streams}
    inventory_hashes = {
        stream: hashlib.sha256("\n".join(paths).encode("utf-8")).hexdigest()
        for stream, paths in path_inventory.items()
    }
    native_manifest_sha = _sha256_file(args.native_manifest)
    native_inventory_sha = _sha256_file(native_inventory_path)
    taxonomy_sha = _sha256_file(args.taxonomy)
    run_payload: dict[str, object] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "sources": list(args.sources),
        "label_pins": {stream: LABEL_PINS[stream] for stream in path_inventory},
        "raw_pins": {stream: RAW_PINS[stream] for stream in path_inventory},
        "label_path_inventory_sha256": inventory_hashes,
        "label_path_counts": {stream: len(paths) for stream, paths in path_inventory.items()},
        "native_manifest_path": str(args.native_manifest),
        "native_manifest_sha256": native_manifest_sha,
        "native_inventory_path": str(native_inventory_path),
        "native_inventory_sha256": native_inventory_sha,
        "native_asset_count": len(layers),
        "taxonomy_path": str(args.taxonomy),
        "taxonomy_sha256": taxonomy_sha,
        "assignment_policy": (
            "production EunisResult; unique native class with inclusive "
            f">={MINIMUM_NATIVE_OVERLAP_PERCENTAGE:g}% full-polygon coverage"
        ),
        "algorithm_fingerprint": {
            "algorithm_version": _CENSUS_ALGORITHM_VERSION,
            "assignment_gate": _ASSIGNMENT_GATE,
            "code_sha256": _census_code_fingerprint(args.source_root),
            "minimum_native_overlap_percentage": MINIMUM_NATIVE_OVERLAP_PERCENTAGE,
        },
        "census_limits": {
            "h3_resolution": H3_RESOLUTION,
            "max_per_h3_cell_per_decision": MAX_PER_CELL,
            "minimum_h3_cells_per_decision": MINIMUM_H3_CELL_COUNT,
            "target_per_decision": TARGET,
        },
        "text_policy": (
            "retrieve sentence text only after native geometry qualifies; "
            "verify exact label SHA-256"
        ),
        "evaluation_run": False,
    }
    if not args.temp_dir.is_dir():
        raise ValueError("job-local scratch disappeared before census start")
    db = duckdb.connect()
    try:
        _load_httpfs(db)
    except RuntimeError:
        db.close()
        raise
    db.execute("SET threads=1")
    db.execute("SET memory_limit='1GB'")
    db.execute("SET temp_directory=?", [str(args.temp_dir)])
    db.execute("SET enable_progress_bar=false")
    run_signature = _prepare_run(args.output_dir, run_payload, resume=args.resume)
    index = _open_index(args.output_dir / "eligible-index.sqlite")
    regions = tuple(
        (source, stream, label_path)
        for source in args.sources
        for stream in _source_streams(source)
        for label_path in path_inventory[stream]
    )
    with RasterReference(
        tuple(layers),
        threshold=0,
        tile_cache_bytes=512 * 1024 * 1024,
        taxonomy=taxonomy,
    ) as reference:
        if reference.source_extent_wgs84 is None:
            raise ValueError("native EEA source extent is unavailable")
        native_extent = box(*reference.source_extent_wgs84)

        def process_region(source: str, stream: str, label_path: str) -> dict[str, object]:
            label_repo, label_revision = LABEL_PINS[stream]
            raw_repo, raw_revision = RAW_PINS[stream]
            filename = label_path.rsplit("/", 1)[-1]
            raw_paths = _raw_paths(stream, filename)
            raw_urls = tuple(_url(raw_repo, raw_revision, path) for path in raw_paths)
            payload = _process_region(
                db,
                args.output_dir,
                source=source,
                stream=stream,
                label_path=label_path,
                label_url=_url(label_repo, label_revision, label_path),
                raw_paths=raw_paths,
                raw_urls=raw_urls,
                run_signature=run_signature,
                reference=reference,
                native_extent=native_extent,
            )
            print(
                json.dumps(
                    {
                        "source": source,
                        "stream": stream,
                        "shard": payload["shard"],
                        "checkpoint": str(
                            args.output_dir / "checkpoints" / stream / f"{payload['shard']}.json"
                        ),
                        "candidate_rows": payload["candidate_rows"],
                        "output_sha256": payload["output_sha256"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return payload

        newly_completed, visited, unvisited = _run_region_checkpoints(
            regions,
            output_dir=args.output_dir,
            max_new_regions=args.max_new_regions,
            process_region=process_region,
            index_region=lambda source, output: _index_region(index, source, output),
        )
    summary = _summary(index, args.sources)
    if unvisited:
        print(
            json.dumps(
                {
                    "status": "incomplete",
                    "newly_completed_regions": newly_completed,
                    "visited_regions": visited,
                    "unvisited_regions": unvisited,
                    "summary": summary,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        index.close()
        db.close()
        return 0
    completion = {
        "run_signature": run_signature,
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "eligible_unique_polygon_summary": summary,
        "h3_resolution": H3_RESOLUTION,
        "max_per_h3_cell_per_decision": MAX_PER_CELL,
        "target_per_decision": TARGET,
        "evaluation_run": False,
    }
    _write_new_json(args.output_dir / "complete.json", completion)
    print(
        json.dumps(
            {
                "output_dir": str(args.output_dir),
                "complete": str(args.output_dir / "complete.json"),
                "summary": summary,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    index.close()
    db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
