"""Regression tests for the geometry-first EUNIS candidate census core."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from scripts.eunis_geometry_census_core import (
    CensusCounts,
    candidate_order_digest,
    iter_geometry_first_candidates,
)
from scripts.eunis_geometry_census_persistence import (
    _index_region,
    _index_region_if_needed,
    _load_completed_region,
    _open_index,
    _summary,
    _verify_and_backfill_indexed_regions,
    _write_replayed_region_output,
)
from scripts.eunis_geometry_census_sources import (
    SOURCE_VERSION,
    _geometry_parameters,
    _text_query,
    geometry_sql,
)
from scripts.eunis_geometry_first_census import (
    _annotate_geometry_rows,
    _parse_production_geometry,
    _run_region_checkpoints,
    _validate_native_inventory,
)


@dataclass
class EunisResult:
    code: str | None
    name: str | None
    overlap_percentage: float | None
    source_version: str | None


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _polygon_key(row: Mapping[str, object]) -> tuple[str, str, str, str]:
    return (
        str(row.get("source_pbf") or ""),
        str(row.get("osm_type") or ""),
        str(row.get("osm_id") or ""),
        str(row.get("polygon_id") or ""),
    )


def _legacy_candidates(
    rows: Sequence[dict[str, object]],
    texts: Mapping[str, str],
    eligible_geometries: set[str],
) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()
    for row in sorted(rows, key=lambda item: candidate_order_digest("website", item)):
        sentence = texts[str(row["sentence_key"])]
        if _sha(sentence) != row["text_sha256"]:
            continue
        if row["geometry"] not in eligible_geometries:
            continue
        key = _polygon_key(row)
        if key in seen:
            continue
        seen.add(key)
        output.append(
            {
                **row,
                "raw_sentence": sentence,
                "source": "website",
                "stream": "website",
                "eunis_code": "R11",
                "eunis_name": "steppe",
                "eunis_overlap_percentage": 80.0,
                "eunis_source_version": "EEA 2021",
            }
        )
    return output


def test_geometry_first_candidates_match_legacy_selected_rows_and_hashes() -> None:
    texts = {
        "p1-0": "old mismatch",
        "p1-1": "same polygon",
        "p1-2": "same polygon duplicate",
        "p2-0": "other",
        "p3-0": "outside",
    }
    rows: list[dict[str, object]] = [
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "decision": "yes",
            "sentence_key": "p1-0",
            "polygon_id": "1",
            "osm_type": "way",
            "osm_id": "1",
            "source_pbf": None,
            "text_sha256": _sha("expected mismatch"),
            "geometry": "covered",
        },
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "decision": "yes",
            "sentence_key": "p1-1",
            "polygon_id": "1",
            "osm_type": "way",
            "osm_id": "1",
            "source_pbf": None,
            "text_sha256": _sha(texts["p1-1"]),
            "geometry": "covered",
        },
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "decision": "yes",
            "sentence_key": "p1-2",
            "polygon_id": "1",
            "osm_type": "way",
            "osm_id": "1",
            "source_pbf": None,
            "text_sha256": _sha(texts["p1-2"]),
            "geometry": "covered",
        },
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "decision": "no",
            "sentence_key": "p2-0",
            "polygon_id": "2",
            "osm_type": "way",
            "osm_id": "2",
            "source_pbf": None,
            "text_sha256": _sha(texts["p2-0"]),
            "geometry": "covered",
        },
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "decision": "yes",
            "sentence_key": "p3-0",
            "polygon_id": "3",
            "osm_type": "way",
            "osm_id": "3",
            "source_pbf": None,
            "text_sha256": _sha(texts["p3-0"]),
            "geometry": "uncovered",
        },
    ]
    for row in rows:
        row["stream"] = "website"
    rows.sort(key=lambda item: candidate_order_digest("website", item))
    fetched_keys: list[str] = []

    def classify(geometry: object) -> EunisResult:
        if geometry == "covered":
            return EunisResult("R11", "steppe", 80.0, "EEA 2021")
        return EunisResult(None, None, 79.0, "EEA 2021")

    def fetch(eligible: Sequence[Mapping[str, object]]) -> Mapping[str, str | None]:
        fetched_keys.extend(str(row["sentence_key"]) for row in eligible)
        return {str(row["sentence_key"]): texts[str(row["sentence_key"])] for row in eligible}

    counts = CensusCounts()
    actual = list(
        iter_geometry_first_candidates(
            rows,
            source="website",
            stream="website",
            classify=classify,
            fetch_texts=fetch,
            batch_size=2,
            counts=counts,
        )
    )
    seen: set[tuple[str, str, str, str]] = set()
    unique_actual = []
    for row in actual:
        key = _polygon_key(row)
        if key not in seen:
            seen.add(key)
            unique_actual.append(row)

    expected = _legacy_candidates(rows, texts, {"covered"})
    assert unique_actual == expected
    assert counts.geometry_rows == 5
    assert counts.native_eligible_rows == 4
    assert counts.text_fetch_rows == 4
    assert counts.missing_text_rows == 0
    assert counts.text_hash_mismatches == 1
    assert counts.verified_text_rows == 3
    assert "p3-0" not in fetched_keys


def test_unassigned_and_ambiguous_geometry_never_reaches_text_fetch() -> None:
    rows = [
        {
            "decision": "yes",
            "sentence_key": "unassigned",
            "text_sha256": _sha("x"),
            "geometry": "unassigned",
        },
        {
            "decision": "no",
            "sentence_key": "ambiguous",
            "text_sha256": _sha("y"),
            "geometry": "ambiguous",
        },
    ]
    calls: list[Sequence[Mapping[str, object]]] = []

    def classify(geometry: object) -> EunisResult:
        if geometry == "ambiguous":
            return EunisResult(None, None, 82.0, "EEA 2021")
        return EunisResult(None, None, 40.0, "EEA 2021")

    output = list(
        iter_geometry_first_candidates(
            rows,
            source="website",
            stream="website",
            classify=classify,
            fetch_texts=lambda batch: calls.append(batch) or {},
        )
    )

    assert output == []
    assert calls == []


def test_missing_text_and_hash_mismatch_have_separate_counts() -> None:
    rows = [
        {
            "decision": "yes",
            "sentence_key": "missing",
            "text_sha256": _sha("x"),
            "geometry": "covered",
        },
        {
            "decision": "yes",
            "sentence_key": "wrong-hash",
            "text_sha256": _sha("wanted"),
            "geometry": "covered",
        },
    ]
    counts = CensusCounts()
    output = list(
        iter_geometry_first_candidates(
            rows,
            source="website",
            stream="website",
            classify=lambda _: EunisResult("R11", "steppe", 80.0, "EEA 2021"),
            fetch_texts=lambda batch: {"wrong-hash": "different"},
            counts=counts,
        )
    )
    assert output == []
    assert counts.missing_text_rows == 1
    assert counts.text_hash_mismatches == 1


def test_text_result_mapping_rejects_duplicate_sentence_keys() -> None:
    import scripts.eunis_geometry_first_census as runner

    try:
        runner._unique_text_mapping([("same", "first"), ("same", "second")])
    except ValueError as error:
        assert "duplicate sentence key" in str(error)
    else:
        raise AssertionError("duplicate text join results should fail closed")


def test_geometry_queries_bind_select_values_before_parquet_paths() -> None:
    cases = (
        (
            "website",
            [
                "labels/website.parquet",
                "website",
                "website",
                "places",
                "website",
                "label-url",
                "raw-url",
            ],
        ),
        (
            "description",
            [
                "labels/description.parquet",
                "description",
                "description",
                "places",
                "description",
                "label-url",
                "text-url",
                "polygon-url",
            ],
        ),
    )
    for stream, expected in cases:
        sql = geometry_sql(stream)
        params = _geometry_parameters(
            stream,
            "places",
            f"labels/{stream}.parquet",
            stream,
            "label-url",
            ("text-url", "polygon-url") if stream == "description" else ("raw-url",),
        )
        assert params == expected
        assert sql.count("?") == len(params)
        assert "read_parquet(?)" in sql
        assert "raw_sentence" not in sql
        assert "website_sentences" not in sql
        assert "sentences" not in sql

    wiki_sql = geometry_sql("wikipedia")
    wiki_params = _geometry_parameters(
        "wikipedia",
        "places",
        "labels/wikipedia.parquet",
        "wikipedia",
        "label-url",
        ("text-url", "polygon-url"),
    )
    assert wiki_sql.count("?") == len(wiki_params) == 9
    assert wiki_params == [
        "label-url",
        "text-url",
        "polygon-url",
        "labels/wikipedia.parquet",
        "wikipedia",
        "wikipedia",
        "places",
        "wikipedia",
        "wikipedia",
    ]


def test_text_query_binds_only_the_selected_batch_keys() -> None:
    selected = [
        {
            "sentence_key": "selected-a",
            "polygon_id": "polygon-a",
            "_field": "website",
            "_sentence_index": 2,
            "_raw_polygon_url": "https://example.test/polygons.parquet",
        },
        {
            "sentence_key": "selected-b",
            "polygon_id": "polygon-b",
            "_field": "contact_website",
            "_sentence_index": 4,
            "_raw_polygon_url": "https://example.test/polygons.parquet",
        },
    ]
    sql, params = _text_query("website", selected)
    assert sql.count("?") == 9
    assert "(?,?,?,?),(?,?,?,?)" in sql
    assert params == [
        "selected-a",
        "polygon-a",
        "website",
        2,
        "selected-b",
        "polygon-b",
        "contact_website",
        4,
        "https://example.test/polygons.parquet",
    ]
    assert "selected-a" not in sql
    assert "selected-b" not in sql


def test_unknown_stream_lineage_fails_closed() -> None:
    for make_query in (geometry_sql,):
        try:
            make_query("unknown")
        except ValueError as error:
            assert "unsupported census stream" in str(error)
        else:
            raise AssertionError("unknown stream lineage should fail closed")


def test_wiki_uniqueness_counts_links_before_project_hash_and_geometry_filters() -> None:
    sql = geometry_sql("wikipedia")
    matched = sql.index("matched AS (")
    polygon_count = sql.index("count(p.polygon_id) OVER", matched)
    filtered_from = sql.index("FROM matched", polygon_count)
    filtered_where = sql.index("WHERE polygon_matches = 1", filtered_from)

    assert "SELECT * FROM read_parquet(?) WHERE decision IN ('yes', 'no')" in sql
    assert polygon_count < filtered_from < filtered_where
    assert "project = ? AND text_sha256 = content_hash" in sql[filtered_where:]
    assert "geometry IS NOT NULL" in sql[filtered_where:]


def test_geometry_row_annotation_is_lazy() -> None:
    consumed: list[int] = []
    raw_geometry = "not-json"

    def rows():
        consumed.append(1)
        yield {"geometry": raw_geometry}

    annotated = _annotate_geometry_rows(rows(), stream="website", raw_urls=("raw-url",))
    assert consumed == []
    row = next(iter(annotated))
    assert consumed == [1]
    assert row["geometry"] == raw_geometry
    assert row["_raw_polygon_url"] == "raw-url"


def test_census_uses_production_geometry_repair_and_malformed_policy() -> None:
    bowtie = json.dumps(
        {
            "type": "Polygon",
            "coordinates": [[[0, 0], [2, 2], [0, 2], [2, 0], [0, 0]]],
        }
    )

    repaired = _parse_production_geometry(bowtie)

    assert repaired is not None
    assert repaired.is_valid
    assert repaired.area == 2.0
    assert _parse_production_geometry("not-json") is None


def test_region_runner_keeps_geometry_cursor_live_during_text_fetch(  # noqa: C901
    tmp_path: Path, monkeypatch
) -> None:
    from types import ModuleType

    import scripts.eunis_geometry_first_census as runner

    monkeypatch.setattr(
        runner,
        "_json_row",
        lambda row: {
            key: ("geometry" if key == "geometry" else value)
            for key, value in row.items()
            if not key.startswith("_")
        },
    )
    monkeypatch.setattr(runner, "BATCH_SIZE", 1)
    monkeypatch.setattr(runner, "_region_signature", lambda *args: "signature")
    monkeypatch.setattr(runner, "_load_completed_region", lambda *args: None)
    monkeypatch.setattr(runner, "_write_new_json", lambda *args: None)
    geometry_module = ModuleType("osm_polygon_eunis.geometry")
    monkeypatch.setattr(geometry_module, "parse_geometry", lambda geometry: geometry, raising=False)
    monkeypatch.setattr(geometry_module, "to_equal_area", lambda geometry: geometry, raising=False)
    package_module = ModuleType("osm_polygon_eunis")
    package_module.__path__ = []
    monkeypatch.setitem(__import__("sys").modules, "osm_polygon_eunis", package_module)
    monkeypatch.setitem(__import__("sys").modules, "osm_polygon_eunis.geometry", geometry_module)
    events: list[str] = []

    class Geometry:
        geom_type = "Polygon"
        is_empty = False
        is_valid = True
        area = 1.0
        wkb = b"geometry"

        def intersects(self, _other):
            return True

    geometry_rows = [
        {
            "label_path": "labels/a.parquet",
            "shard": "a",
            "stream": "website",
            "decision": "yes",
            "sentence_key": f"sentence-{index}",
            "polygon_id": str(index),
            "osm_type": "way",
            "osm_id": str(index),
            "source_pbf": None,
            "text_sha256": _sha(f"sentence text {index}"),
            "geometry": Geometry(),
            "_field": "website",
            "_sentence_index": 0,
        }
        for index in (1, 2)
    ]

    class Cursor:
        def __init__(self, kind: str):
            self.kind = kind
            self.description = None
            self.rows = []
            self.fetchmany_calls = 0

        def execute(self, sql, params):
            if self.kind == "geometry":
                self.description = [(key,) for key in geometry_rows[0]]
                self.rows = [tuple(row.values()) for row in geometry_rows]
            else:
                assert "read_parquet(?)" in sql
                geometry_cursor = next(cursor for cursor in db.created if cursor.kind == "geometry")
                events.append("text")
                if events.count("text") == 1:
                    assert len(geometry_cursor.rows) == 1
                    assert geometry_cursor.fetchmany_calls == 1
                self.rows = [(params[0], f"sentence text {params[0].split('-')[1]}")]
            return self

        def fetchmany(self, _size):
            assert self.kind == "geometry"
            self.fetchmany_calls += 1
            events.append("geometry")
            if not self.rows:
                return []
            return [self.rows.pop(0)]

        def fetchall(self):
            assert self.kind == "text"
            return self.rows

    class Connection:
        def __init__(self):
            self.created: list[Cursor] = []

        def cursor(self):
            cursor = Cursor("text" if not self.created else "geometry")
            self.created.append(cursor)
            return cursor

        def execute(self, *_args):
            raise AssertionError("runner must query through independent cursors")

    db = Connection()
    runner._process_region(
        db,
        tmp_path,
        source="website",
        stream="website",
        label_path="labels/a.parquet",
        label_url="label-url",
        raw_paths=("polygons/a.parquet",),
        raw_urls=("polygon-url",),
        run_signature="run",
        reference=type(
            "Reference",
            (),
            {
                "intersection_errors": 0,
                "overlap": lambda self, polygon: EunisResult("R11", "steppe", 80.0, "EEA 2021"),
            },
        )(),
        native_extent=object(),
    )

    assert len(db.created) == 2
    assert db.created[0] is not db.created[1]
    geometry_cursor = next(cursor for cursor in db.created if cursor.kind == "geometry")
    assert geometry_cursor.fetchmany_calls == 3
    assert events == ["geometry", "text", "geometry", "text", "geometry"]


def test_completed_region_recovery_checks_signature_and_output_digest(
    tmp_path: Path,
) -> None:
    output = tmp_path / "region.jsonl"
    checkpoint = tmp_path / "region.json"
    output.write_text("candidate\n", encoding="utf-8")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    checkpoint.write_text(
        json.dumps(
            {
                "signature": "expected",
                "output_sha256": digest,
                "output_path": str(tmp_path / "stale" / "region.jsonl"),
            }
        ),
        encoding="utf-8",
    )

    completed = _load_completed_region(checkpoint, output, "expected")
    assert completed is not None
    assert completed["output_sha256"] == digest
    assert completed["output_path"] == str(output)
    try:
        _load_completed_region(checkpoint, output, "different")
    except ValueError as error:
        assert "input mismatch" in str(error)
    else:
        raise AssertionError("mismatched checkpoint signature should fail closed")

    output.write_text("changed\n", encoding="utf-8")
    try:
        _load_completed_region(checkpoint, output, "expected")
    except ValueError as error:
        assert "digest mismatch" in str(error)
    else:
        raise AssertionError("modified region output should fail closed")


def test_run_resume_signature_includes_algorithm_and_code_fingerprint(
    tmp_path: Path,
) -> None:
    from scripts.eunis_geometry_first_census import _prepare_run

    output_dir = tmp_path / "run"
    first: dict[str, object] = {
        "created_at_utc": "first-time",
        "algorithm_fingerprint": {
            "algorithm_version": "geometry-first-v3",
            "assignment_gate": "unique >=80%",
            "code_sha256": "code-a",
        },
        "census_limits": {
            "h3_resolution": 3,
            "max_per_h3_cell_per_decision": 20,
            "target_per_decision": 500,
        },
    }
    first_signature = _prepare_run(output_dir, first, resume=False)
    assert first["run_signature"] == first_signature

    changed: dict[str, object] = {
        "created_at_utc": "later-time",
        "algorithm_fingerprint": {
            **first["algorithm_fingerprint"],
            "code_sha256": "code-b",
        },
    }
    try:
        _prepare_run(output_dir, changed, resume=True)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("code fingerprint changes must invalidate resume")


def test_run_resume_signature_includes_h3_capacity_and_target(tmp_path: Path) -> None:
    from scripts.eunis_geometry_first_census import _prepare_run

    output_dir = tmp_path / "run"
    base: dict[str, object] = {
        "algorithm_fingerprint": {"code_sha256": "same"},
        "census_limits": {
            "h3_resolution": 3,
            "max_per_h3_cell_per_decision": 20,
            "target_per_decision": 500,
        },
    }
    _prepare_run(output_dir, {**base, "created_at_utc": "first"}, resume=False)
    for key, value in (
        ("h3_resolution", 4),
        ("max_per_h3_cell_per_decision", 21),
        ("target_per_decision", 501),
    ):
        changed: dict[str, object] = {
            **base,
            "created_at_utc": "later",
            "census_limits": {**base["census_limits"], key: value},
        }
        try:
            _prepare_run(output_dir, changed, resume=True)
        except ValueError as error:
            assert "does not match" in str(error)
        else:
            raise AssertionError(f"changing {key} must invalidate resume")


def test_code_fingerprint_covers_production_assignment_modules(tmp_path: Path) -> None:
    from scripts.eunis_geometry_first_census import _census_code_fingerprint

    source_root = tmp_path / "source"
    package = source_root / "src" / "osm_polygon_eunis"
    package.mkdir(parents=True)
    modules = (
        "domain.py",
        "geometry.py",
        "raster_aggregation.py",
        "raster_reference.py",
        "matching.py",
        "grid_overlap.py",
        "raster_geometry.py",
        "raster_validation.py",
        "tiff_compression.py",
        "tiff_format.py",
        "tiff_subifd.py",
    )
    for module in modules:
        (package / module).write_text("stable content\n", encoding="utf-8")
    apple_double = package / "._metadata.py"
    apple_double.write_text("sidecar bytes\n", encoding="utf-8")
    before = _census_code_fingerprint(source_root)
    apple_double.write_text("changed sidecar bytes\n", encoding="utf-8")
    assert _census_code_fingerprint(source_root) == before
    (package / "matching.py").write_text("changed assignment\n", encoding="utf-8")
    after = _census_code_fingerprint(source_root)
    assert before != after


def test_legacy_run_signature_allows_only_pinned_code_refresh(tmp_path: Path) -> None:
    from scripts.eunis_geometry_first_census import (
        _LEGACY_RESUME_CODE_HASHES,
        _prepare_run,
    )

    old_code = next(iter(_LEGACY_RESUME_CODE_HASHES))
    current_code = "current-code"
    output_dir = tmp_path / "legacy-run"
    legacy_algorithm: dict[str, str] = {
        "algorithm_version": "geometry-first-v3",
        "assignment_gate": "unique >=80%",
        "code_sha256": old_code,
    }
    legacy: dict[str, object] = {
        "created_at_utc": "first-time",
        "native_manifest_sha256": "native-pin",
        "algorithm_fingerprint": legacy_algorithm,
    }
    saved_signature = _prepare_run(output_dir, legacy, resume=False)
    legacy_identity = {key: value for key, value in legacy.items() if key != "run_signature"}
    refreshed_algorithm = {**legacy_algorithm, "code_sha256": current_code}
    refreshed: dict[str, object] = {
        **legacy_identity,
        "created_at_utc": "later-time",
        "algorithm_fingerprint": refreshed_algorithm,
    }
    assert _prepare_run(output_dir, refreshed, resume=True) == saved_signature
    refresh_record = json.loads(
        (output_dir / "resume-compatible-code-refresh.json").read_text(encoding="utf-8")
    )
    assert refresh_record["event"] == "resume_compatible_code_refresh"
    assert refresh_record["old_code_sha256"] == old_code
    assert refresh_record["new_code_sha256"] == current_code
    assert refresh_record["run_signature"] == saved_signature

    changed_gate = {
        **refreshed,
        "algorithm_fingerprint": {
            **refreshed_algorithm,
            "assignment_gate": "different gate",
        },
    }
    try:
        _prepare_run(output_dir, changed_gate, resume=True)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("code refresh must reject changed semantic inputs")


def test_legacy_run_signature_rejects_unknown_old_code_hash(tmp_path: Path) -> None:
    from scripts.eunis_geometry_first_census import _prepare_run

    output_dir = tmp_path / "unknown-legacy-run"
    first: dict[str, object] = {
        "created_at_utc": "first-time",
        "algorithm_fingerprint": {"code_sha256": "unknown-old-code"},
    }
    _prepare_run(output_dir, first, resume=False)
    changed: dict[str, object] = {
        "created_at_utc": "later-time",
        "algorithm_fingerprint": {"code_sha256": "current-code"},
    }
    try:
        _prepare_run(output_dir, changed, resume=True)
    except ValueError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("unknown old code hash must not be accepted")


def test_summary_reports_zero_eligible_source_decisions(tmp_path: Path) -> None:
    db = _open_index(tmp_path / "eligible.sqlite")
    db.execute(
        "INSERT INTO eligible VALUES (?, ?, ?, ?, ?)",
        ("website", "yes", "polygon", "geometry-hash", "cell-a"),
    )
    summary = _summary(db, ["website", "wiki"])
    sources = cast(dict[str, dict[str, dict[str, object]]], summary["sources"])

    website = sources["website"]
    assert website["yes"]["unique_eligible_polygons"] == 1
    assert website["no"]["unique_eligible_polygons"] == 0
    assert website["no"]["can_reach_500_and_25_cells"] is False

    wiki = sources["wiki"]
    assert wiki["yes"]["unique_eligible_polygons"] == 0
    assert wiki["no"]["unique_eligible_polygons"] == 0
    assert wiki["yes"]["can_reach_500_and_25_cells"] is False
    assert wiki["no"]["can_reach_500_and_25_cells"] is False
    db.close()


def test_existing_eligible_index_is_verified_and_checkpointed_without_reindex(
    tmp_path: Path,
) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from shapely.geometry import box

    output = tmp_path / "sample.jsonl"
    geometry = box(-1, -1, 1, 1).wkb
    row = {
        "decision": "yes",
        "source_pbf": None,
        "osm_type": "relation",
        "osm_id": "42",
        "polygon_id": "polygon-42",
        "geometry": {"encoding": "wkb-hex", "value": geometry.hex()},
    }
    content = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.write_bytes(content)
    output_digest = hashlib.sha256(content).hexdigest()
    db = _open_index(tmp_path / "eligible.sqlite")
    h3 = SimpleNamespace(latlng_to_cell=lambda _lat, _lon, _resolution: "cell")
    with patch.dict("sys.modules", {"h3": h3}):
        _index_region(db, "website", output)
        artifacts = [("website", "website", "sample", output, output_digest)]
        _verify_and_backfill_indexed_regions(db, artifacts, "run-signature")
        rows_before = db.execute("SELECT * FROM eligible").fetchall()
        markers = db.execute("SELECT * FROM indexed_regions").fetchall()
        assert len(rows_before) == 1
        assert markers == [("website", "website", "sample", "run-signature", output_digest)]

        _index_region_if_needed(db, "website", "website", "sample", output, "run-signature")
        assert db.execute("SELECT * FROM eligible").fetchall() == rows_before
        try:
            _index_region_if_needed(db, "website", "website", "sample", output, "changed-run")
        except ValueError as error:
            assert "marker input mismatch" in str(error)
        else:
            raise AssertionError("index marker must reject a different run identity")
    db.close()


def test_index_backfill_recovers_missing_checkpoint_rows(tmp_path: Path) -> None:
    from types import SimpleNamespace
    from unittest.mock import patch

    from shapely.geometry import box

    output = tmp_path / "sample.jsonl"
    geometry = box(-2, -2, 2, 2).wkb
    row = {
        "decision": "no",
        "source_pbf": "region.osm.pbf",
        "osm_type": "way",
        "osm_id": "99",
        "polygon_id": "polygon-99",
        "geometry": {"encoding": "wkb-hex", "value": geometry.hex()},
    }
    content = (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode()
    output.write_bytes(content)
    db = _open_index(tmp_path / "eligible.sqlite")
    h3 = SimpleNamespace(latlng_to_cell=lambda _lat, _lon, _resolution: "cell")
    with patch.dict("sys.modules", {"h3": h3}):
        _verify_and_backfill_indexed_regions(
            db,
            [("website", "website", "sample", output, hashlib.sha256(content).hexdigest())],
            "run-signature",
        )

        assert db.execute("SELECT count(*) FROM eligible").fetchone() == (1,)
        assert db.execute("SELECT count(*) FROM indexed_regions").fetchone() == (1,)
    db.close()


def test_native_asset_hash_is_recomputed_from_file_bytes(tmp_path: Path) -> None:
    from scripts.eunis_geometry_first_census import _validate_native_asset

    path = tmp_path / "R11.tif"
    content = b"native raster bytes"
    path.write_bytes(content)
    manifest_asset = {
        "expected_bytes": len(content),
        "expected_sha256": hashlib.sha256(content).hexdigest(),
        "actual_sha256": hashlib.sha256(content).hexdigest(),
    }
    assert _validate_native_asset(path, manifest_asset) == manifest_asset["expected_sha256"]

    altered = b"native raster BYTES"
    path.write_bytes(altered)
    try:
        _validate_native_asset(path, manifest_asset)
    except ValueError as error:
        assert "SHA-256 check failed" in str(error)
        assert hashlib.sha256(altered).hexdigest() in str(error)
    else:
        raise AssertionError("asset bytes must be checked against the manifest hash")


def _native_inventory_fixture(tmp_path: Path) -> tuple[dict[str, object], Path]:
    inventory_path = tmp_path / "source-inventory.json"
    codes = ("R11", "R12")
    source_assets = [
        {
            "code": code,
            "size_bytes": index + 10,
            "sha256": hashlib.sha256(f"bytes-{code}".encode()).hexdigest(),
        }
        for index, code in enumerate(codes)
    ]
    inventory_path.write_text(
        json.dumps(
            {
                "eea_reference": {
                    "source_version": SOURCE_VERSION,
                    "native_tiffs": source_assets,
                }
            }
        ),
        encoding="utf-8",
    )
    downloaded_assets = [
        {
            "code": asset["code"],
            "file": str(tmp_path / f"Prob_{asset['code']}_100m.tif"),
            "expected_bytes": asset["size_bytes"],
            "expected_sha256": asset["sha256"],
            "actual_bytes": asset["size_bytes"],
            "actual_sha256": asset["sha256"],
            "state": "downloaded_verified",
        }
        for asset in source_assets
    ]
    expected_total = sum(int(asset["size_bytes"]) for asset in source_assets)
    native_doc: dict[str, object] = {
        "inventory_source": str(inventory_path),
        "source_version": SOURCE_VERSION,
        "asset_count": 2,
        "verified_count": 2,
        "failed_count": 0,
        "expected_total_bytes": expected_total,
        "assets": downloaded_assets,
    }
    return native_doc, inventory_path


def test_native_asset_inventory_matches_exact_source_codes_and_hashes(tmp_path: Path) -> None:
    native_doc, inventory_path = _native_inventory_fixture(tmp_path)
    assets, resolved_inventory = _validate_native_inventory(
        native_doc,
        tmp_path / "download-manifest.json",
        tmp_path,
        {"R11": "steppe", "R12": "heath"},
        lambda name: name.removeprefix("Prob_").removesuffix("_100m.tif"),
        expected_count=2,
        expected_total_bytes=21,
    )

    assert [asset[0]["code"] for asset in assets] == ["R11", "R12"]
    assert resolved_inventory == inventory_path


def test_native_asset_inventory_rejects_duplicates_and_missing_source_codes(
    tmp_path: Path,
) -> None:
    native_doc, _ = _native_inventory_fixture(tmp_path)
    assets = cast(list[dict[str, object]], native_doc["assets"])
    assets[1] = {**assets[1], "code": "R11", "file": str(tmp_path / "Prob_R11_100m.tif")}
    try:
        _validate_native_inventory(
            native_doc,
            tmp_path / "download-manifest.json",
            tmp_path,
            {"R11": "steppe", "R12": "heath"},
            lambda name: name.removeprefix("Prob_").removesuffix("_100m.tif"),
            expected_count=2,
            expected_total_bytes=21,
        )
    except ValueError as error:
        assert "repeats code: R11" in str(error)
    else:
        raise AssertionError("duplicate asset codes must not hide a missing source class")


def test_native_asset_inventory_rejects_unique_but_unapproved_code(tmp_path: Path) -> None:
    native_doc, _ = _native_inventory_fixture(tmp_path)
    assets = cast(list[dict[str, object]], native_doc["assets"])
    assets[1] = {**assets[1], "code": "R13", "file": str(tmp_path / "Prob_R13_100m.tif")}
    try:
        _validate_native_inventory(
            native_doc,
            tmp_path / "download-manifest.json",
            tmp_path,
            {"R11": "steppe", "R12": "heath", "R13": "bog"},
            lambda name: name.removeprefix("Prob_").removesuffix("_100m.tif"),
            expected_count=2,
            expected_total_bytes=21,
        )
    except ValueError as error:
        assert "unapproved code: R13" in str(error)
    else:
        raise AssertionError("codes outside the authoritative inventory must be rejected")


def test_orphan_partial_is_preserved_for_verified_replay(tmp_path: Path) -> None:
    partial = tmp_path / "region.jsonl.partial"
    partial.write_text("unfinished\n", encoding="utf-8")
    assert (
        _load_completed_region(tmp_path / "region.json", tmp_path / "region.jsonl", "signature")
        is None
    )
    assert partial.read_text(encoding="utf-8") == "unfinished\n"


def test_partial_region_replay_requires_exact_byte_prefix_and_preserves_partial(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "region.jsonl.partial"
    first = b'{"id":"first"}\n'
    second = b'{"id":"second"}\n'
    partial.write_bytes(first + second[:9])
    before = partial.read_bytes()
    output = tmp_path / "region.jsonl"
    digest, row_count = _write_replayed_region_output(output, (first, second))
    assert output.read_bytes() == first + second
    assert partial.read_bytes() == before
    assert digest == hashlib.sha256(first + second).hexdigest()
    assert row_count == 2


def test_partial_region_replay_mismatch_does_not_modify_saved_artifact(
    tmp_path: Path,
) -> None:
    partial = tmp_path / "region.jsonl.partial"
    partial.write_bytes(b'{"id":"wrong"}\n')
    before = partial.read_bytes()
    try:
        _write_replayed_region_output(tmp_path / "region.jsonl", (b'{"id":"right"}\n',))
    except ValueError as error:
        assert "differs from saved prefix" in str(error)
    else:
        raise AssertionError("replay must reject mismatched saved bytes")
    assert partial.read_bytes() == before
    assert not (tmp_path / "region.jsonl").exists()


def test_interrupted_replay_stage_continues_after_verified_saved_bytes(
    tmp_path: Path,
) -> None:
    first = b'{"id":"first"}\n'
    second = b'{"id":"second"}\n'
    third = b'{"id":"third"}\n'
    output = tmp_path / "region.jsonl"
    partial = tmp_path / "region.jsonl.partial"
    replay_stage = tmp_path / "region.jsonl.partial.replay"
    partial.write_bytes(first + second[:7])
    replay_stage.write_bytes(first + second + third[:5])
    saved_partial = partial.read_bytes()
    saved_stage = replay_stage.read_bytes()

    digest, row_count = _write_replayed_region_output(output, (first, second, third))

    assert output.read_bytes() == first + second + third
    assert partial.read_bytes() == saved_partial
    assert not replay_stage.exists()
    assert digest == hashlib.sha256(first + second + third).hexdigest()
    assert row_count == 3
    assert len(saved_stage) < output.stat().st_size


def test_uncheckpointed_final_output_is_verified_before_checkpoint_recovery(
    tmp_path: Path,
) -> None:
    content = b'{"id":"first"}\n{"id":"second"}\n'
    output = tmp_path / "region.jsonl"
    output.write_bytes(content)
    partial = tmp_path / "region.jsonl.partial"
    partial.write_bytes(content[:13])
    digest, row_count = _write_replayed_region_output(
        output, (b'{"id":"first"}\n', b'{"id":"second"}\n')
    )
    assert output.read_bytes() == content
    assert partial.read_bytes() == content[:13]
    assert digest == hashlib.sha256(content).hexdigest()
    assert row_count == 2


def test_region_budget_resumes_same_output_and_counts_only_new_checkpoints(
    tmp_path: Path,
) -> None:
    regions = [("website", "website", f"labels/{name}-latest.parquet") for name in "abcd"]
    created: list[str] = []
    indexed: list[str] = []

    def process(source: str, stream: str, label_path: str) -> dict[str, object]:
        assert source == "website"
        shard = label_path.rsplit("/", 1)[-1].removesuffix("-latest.parquet")
        output = tmp_path / "regions" / stream / f"{shard}.jsonl"
        checkpoint = tmp_path / "checkpoints" / stream / f"{shard}.json"
        if checkpoint.exists():
            return cast(dict[str, object], _load_completed_region(checkpoint, output, "sig"))
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        content = f"{shard}\n"
        output.write_text(content, encoding="utf-8")
        checkpoint.write_text(
            json.dumps(
                {
                    "signature": "sig",
                    "shard": shard,
                    "output_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
                }
            ),
            encoding="utf-8",
        )
        created.append(shard)
        return {"shard": shard}

    first = _run_region_checkpoints(
        regions,
        output_dir=tmp_path,
        max_new_regions=2,
        process_region=process,
        index_region=lambda source, output: indexed.append(output.stem),
    )
    assert first == (2, 2, 2)
    assert created == ["a", "b"]
    assert indexed == ["a", "b"]

    def resume(source: str, stream: str, label_path: str) -> dict[str, object]:
        shard = label_path.rsplit("/", 1)[-1].removesuffix("-latest.parquet")
        if shard in {"a", "b"}:
            assert (tmp_path / "checkpoints" / stream / f"{shard}.json").is_file()
        return process(source, stream, label_path)

    created.clear()
    indexed.clear()
    second = _run_region_checkpoints(
        regions,
        output_dir=tmp_path,
        max_new_regions=1,
        process_region=resume,
        index_region=lambda source, output: indexed.append(output.stem),
    )
    assert second == (1, 3, 1)
    assert created == ["c"]
    assert indexed == ["a", "b", "c"]
    assert (tmp_path / "checkpoints" / "website" / "a.json").is_file()
    assert (tmp_path / "checkpoints" / "website" / "b.json").is_file()
    assert (tmp_path / "checkpoints" / "website" / "c.json").is_file()
    assert not (tmp_path / "checkpoints" / "website" / "d.json").exists()


def test_region_budget_does_not_hide_orphan_partial_artifact(tmp_path: Path) -> None:
    regions = [("website", "website", f"labels/{name}-latest.parquet") for name in "abcd"]
    checkpoint = tmp_path / "checkpoints" / "website" / "a.json"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_text("checkpoint\n", encoding="utf-8")
    orphan = tmp_path / "regions" / "website" / "c.jsonl.partial"
    orphan.parent.mkdir(parents=True)
    orphan.write_text("preserve me\n", encoding="utf-8")
    indexed: list[str] = []

    def process(source: str, stream: str, label_path: str) -> dict[str, object]:
        assert source == "website"
        shard = label_path.rsplit("/", 1)[-1].removesuffix("-latest.parquet")
        if shard == "a":
            return {"shard": shard}
        output = tmp_path / "regions" / stream / f"{shard}.jsonl"
        checkpoint = tmp_path / "checkpoints" / stream / f"{shard}.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(f"{shard}\n", encoding="utf-8")
        checkpoint.write_text("valid checkpoint\n", encoding="utf-8")
        return {"shard": shard}

    try:
        _run_region_checkpoints(
            regions,
            output_dir=tmp_path,
            max_new_regions=1,
            process_region=process,
            index_region=lambda source, output: indexed.append(output.stem),
        )
    except FileExistsError as error:
        assert "without a checkpoint" in str(error)
    else:
        raise AssertionError("budget boundary must not skip an orphan partial artifact")
    assert orphan.read_text(encoding="utf-8") == "preserve me\n"


def test_httpfs_preflight_requires_an_existing_extension_without_installing() -> None:
    from scripts.eunis_geometry_first_census import _load_httpfs

    class Database:
        def __init__(self, *, installed: bool) -> None:
            self.statements = []
            self.installed = installed

        def execute(self, statement: str) -> Database:
            self.statements.append(statement)
            if statement.startswith("LOAD "):
                raise OSError("missing extension")

            return self

        def fetchone(self) -> tuple[bool]:
            return (self.installed,)

    database = Database(installed=False)
    try:
        _load_httpfs(database)
    except RuntimeError as error:
        assert "already be available" in str(error)
        assert "does not install extensions" in str(error)
    else:
        raise AssertionError("missing httpfs extension must fail before source access")
    assert database.statements == [
        "SET autoinstall_known_extensions=false",
        "SET autoload_known_extensions=false",
        "SELECT installed FROM duckdb_extensions() WHERE extension_name='httpfs'",
    ]


def test_httpfs_is_loaded_only_after_installed_status_is_verified() -> None:
    from scripts.eunis_geometry_first_census import _load_httpfs

    class Database:
        def __init__(self) -> None:
            self.statements = []

        def execute(self, statement: str) -> Database:
            self.statements.append(statement)
            return self

        def fetchone(self) -> tuple[bool]:
            return (True,)

    database = Database()
    _load_httpfs(database)

    assert database.statements[-1] == "LOAD httpfs"
