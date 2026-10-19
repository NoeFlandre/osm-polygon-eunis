from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from scripts import eunis_geometry_first_census as runner
from scripts.eunis_geometry_census_sources import (
    _geometry_parameters,
    _text_query,
    remap_pinned_parquet_urls,
)
from scripts.eunis_geometry_first_census import (
    _EXPECTED_STAGED_PARQUET_INPUTS,
    _load_staged_input_map,
)


def _fixture_manifest(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, str, str]:
    scratch = tmp_path / "job-scratch"
    input_dir = scratch / "inputs" / "sha256"
    input_dir.mkdir(parents=True)
    label_url, raw_url = tuple(_EXPECTED_STAGED_PARQUET_INPUTS)
    fixtures = {
        label_url: b"bounded label path fixture",
        raw_url: b"bounded raw geometry path fixture",
    }
    records = []
    for url, content in fixtures.items():
        digest = hashlib.sha256(content).hexdigest()
        monkeypatch.setitem(_EXPECTED_STAGED_PARQUET_INPUTS, url, (len(content), digest))
        staged = input_dir / f"{digest}.parquet"
        staged.write_bytes(content)
        records.append(
            {
                "source_url": url,
                "local_path": f"inputs/sha256/{digest}.parquet",
                "size_bytes": len(content),
                "sha256": digest,
            }
        )
    manifest = scratch / "staged-inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "files": records,
            }
        ),
        encoding="utf-8",
    )
    return manifest, label_url, raw_url


def test_staged_source_map_verifies_fixture_and_substitutes_query_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, label_url, raw_url = _fixture_manifest(tmp_path, monkeypatch)
    scratch = manifest.parent
    staged = _load_staged_input_map(manifest, scratch)
    local_label, local_raw = remap_pinned_parquet_urls(
        label_url, (raw_url,), staged, require_staged=True
    )
    assert local_label == str((scratch / staged[label_url]).resolve())
    assert local_raw[0] != raw_url

    geometry_params = _geometry_parameters(
        "website",
        "website",
        "labels/polygons/england-latest.parquet",
        "england",
        local_label,
        (local_raw[0],),
    )
    assert local_label in geometry_params
    assert local_raw[0] in geometry_params

    selected = [
        {
            "sentence_key": "bounded-fixture",
            "polygon_id": "fixture-polygon",
            "_field": "website",
            "_sentence_index": 0,
            "_raw_polygon_url": local_raw[0],
        }
    ]
    _, text_params = _text_query("website", selected)
    assert text_params[-1] == local_raw[0]


def test_staged_mode_fails_closed_when_a_pinned_file_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, label_url, raw_url = _fixture_manifest(tmp_path, monkeypatch)
    staged = _load_staged_input_map(manifest, manifest.parent)
    staged.pop(raw_url)
    with pytest.raises(ValueError, match="not staged"):
        remap_pinned_parquet_urls(label_url, (raw_url,), staged, require_staged=True)


def test_staged_manifest_rejects_modified_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest, label_url, _ = _fixture_manifest(tmp_path, monkeypatch)
    (manifest.parent / "inputs/sha256").joinpath(
        f"{_EXPECTED_STAGED_PARQUET_INPUTS[label_url][1]}.parquet"
    ).write_bytes(b"changed fixture")
    with pytest.raises(ValueError, match="failed size/hash verification"):
        _load_staged_input_map(manifest, manifest.parent)


def test_completed_region_does_not_require_unstaged_parquet_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = {"shard": "algeria", "candidate_rows": 0}
    monkeypatch.setattr(runner, "_load_completed_region", lambda *_args: completed)

    result = runner._process_region(
        object(),
        tmp_path,
        source="website",
        stream="website",
        label_path="labels/polygons/algeria-latest.parquet",
        label_url="remote-label-url",
        raw_paths=("polygons/algeria-latest.parquet",),
        raw_urls=("remote-raw-url",),
        staged_inputs={},
        require_staged_inputs=True,
        run_signature="run",
        code_sha256="a" * 64,
        reference=object(),
        native_extent=object(),
    )

    assert result is completed
