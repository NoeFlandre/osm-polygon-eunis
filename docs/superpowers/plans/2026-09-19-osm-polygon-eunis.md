# OSM Polygon EUNIS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (\`- [ ]\`) syntax for tracking.

**Goal:** Build and verify a storage-bounded Python pipeline that enriches the three pinned OSM polygon datasets with EUNIS labels selected by exact polygon/reference-area overlap, then publishes three public Hugging Face datasets and the tracking artifacts.

**Architecture:** Keep network, filesystem, Parquet, and Hub operations at the edges. The core matcher accepts projected geometries and overlap measurements, applies deterministic winner selection, and returns nullable output fields. An EEA raster adapter converts positive-probability raster cells into actual cell geometries; raster bounds are only candidate pruning. A shard runner streams one input Parquet shard into a bounded temporary file, writes and uploads one replacement shard, and removes local artifacts before continuing.

**Tech Stack:** Python 3.12, uv, PyArrow, Shapely 2, PyProj, Rasterio, NumPy, huggingface_hub, httpx, pytest, Hypothesis, pytest-bdd, Ruff, ty, radon, mutmut, MkDocs Material.

---

## File map

Create these focused files. Keep dependency direction as domain -> reference -> transform -> sources -> publish -> cli; no lower-level module imports a higher-level module.

- pyproject.toml, .gitignore, README.md, mkdocs.yml
- docs/index.md, docs/operations.md, docs/adr/0001-reference-data-and-overlap.md, docs/adr/0002-hub-shard-processing.md, docs/technical-debt.md
- src/osm_polygon_eunis/domain.py, geometry.py, matching.py, reference.py, transform.py, sources.py, publish.py, cli.py
- scripts/check_architecture.py, scripts/check_crap.py, scripts/smoke.py
- tests/unit, tests/integration, tests/acceptance, tests/fixtures

## Task 1: Bootstrap and baseline

**Files:** create pyproject.toml, .gitignore, README.md, mkdocs.yml, docs/index.md, docs/operations.md.

- [ ] Step 1: Run the pre-code baseline

Run:

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest
~~~

Expected in the empty repository: uv reports that project metadata is absent. This is the honest baseline and is not a passing gate.

- [ ] Step 2: Add project metadata

Use this dependency and command contract in pyproject.toml:

~~~toml
[project]
name = "osm-polygon-eunis"
version = "0.1.0"
description = "Enrich OSM polygon Hugging Face datasets with exact-overlap EUNIS labels."
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
  "httpx>=0.27,<1",
  "huggingface-hub>=0.35,<1",
  "numpy>=2,<3",
  "pyarrow>=18,<23",
  "pyproj>=3.7,<4",
  "rasterio>=1.4,<2",
  "shapely>=2.0,<3",
]
[project.scripts]
osm-polygon-eunis = "osm_polygon_eunis.cli:main"
[dependency-groups]
dev = [
  "coverage[toml]>=7.6,<8",
  "hypothesis>=6.120,<7",
  "mkdocs-material>=9.5,<10",
  "mutmut>=3.2,<4",
  "pytest>=8.3,<9",
  "pytest-bdd>=8,<9",
  "pytest-cov>=6,<7",
  "radon>=6,<7",
  "ruff>=0.9,<1",
  "ty>=0.0.1a20",
]
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
[tool.hatch.build.targets.wheel]
packages = ["src/osm_polygon_eunis"]
[tool.pytest.ini_options]
addopts = "--strict-config --strict-markers"
testpaths = ["tests"]
[tool.coverage.run]
branch = true
source = ["src/osm_polygon_eunis"]
[tool.coverage.report]
fail_under = 90
[tool.ruff]
line-length = 100
target-version = "py312"
src = ["src", "tests", "scripts"]
[tool.ruff.lint]
select = ["E", "F", "I", "B", "UP", "SIM", "C4", "RUF"]
~~~

Use a task-scoped UV_CACHE_DIR. Never put uv/HF caches into the data root.

- [ ] Step 3: Add docs and ignore rules

Ignore .venv, pytest/Ruff/ty caches, coverage output, build output, temporary staging, bytecode, and .DS_Store. Do not delete the existing .DS_Store. The README must state the three source IDs, three target IDs, four added nullable fields, exact projected-intersection formula, null behavior, provenance, bounded-storage design, and local commands. MkDocs exposes the public operations page and both ADRs.

- [ ] Step 4: Install, run the empty-suite baseline, and commit

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv sync --group dev
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest
git add pyproject.toml .gitignore README.md mkdocs.yml docs
git commit -m "chore: bootstrap EUNIS enrichment project"
~~~

Expected before tests exist: pytest exits with no tests ran; Task 2 supplies the first RED test.

## Task 2: Pure geometry and matching, RED -> GREEN -> REFACTOR

**Files:** create src/osm_polygon_eunis/domain.py, geometry.py, matching.py; test tests/unit/test_matching.py, tests/unit/test_geometry.py, tests/conftest.py.

- [ ] Step 1: Write RED tests

Start with these deterministic tests:

~~~python
from shapely.geometry import box
from osm_polygon_eunis.domain import OverlapCandidate
from osm_polygon_eunis.matching import choose_winner

def test_largest_actual_intersection_and_percentage() -> None:
    polygon = box(0, 0, 10, 10)
    candidates = (
        OverlapCandidate("R11", "Pannonian steppe", box(0, 0, 8, 10)),
        OverlapCandidate("R12", "Other grassland", box(0, 0, 7, 10)),
    )
    result = choose_winner(polygon, candidates, source_version="test")
    assert result.code == "R11"
    assert result.name == "Pannonian steppe"
    assert result.overlap_percentage == 80.0

def test_bbox_touch_without_geometry_intersection_is_null() -> None:
    result = choose_winner(
        box(0, 0, 1, 1),
        (OverlapCandidate("R11", "steppe", box(1, 1, 2, 2)),),
        source_version="test",
    )
    assert result.is_empty

def test_equal_area_tie_uses_ascending_code() -> None:
    polygon = box(0, 0, 10, 10)
    candidates = (
        OverlapCandidate("R12", "second", box(5, 0, 10, 10)),
        OverlapCandidate("R11", "first", box(0, 0, 5, 10)),
    )
    assert choose_winner(polygon, candidates, source_version="test").code == "R11"

def test_empty_polygon_returns_all_null_fields() -> None:
    result = choose_winner(box(0, 0, 0, 0), (), source_version="test")
    assert result.is_empty
    assert result.code is None
    assert result.name is None
    assert result.overlap_percentage is None
    assert result.source_version is None
~~~

Add geometry tests for GeoJSON parsing, WGS84 to EPSG:3035 transformation, valid repair, and unrecoverable geometry returning None. Run the focused tests and confirm RED.

- [ ] Step 2: Add minimal domain types

domain.py defines EUNIS_FIELDS as the four field names, frozen slotted OverlapCandidate(code, name, geometry), and frozen slotted EunisResult(code, name, overlap_percentage, source_version) with an is_empty property.

geometry.py defines parse_geometry(value), to_equal_area(geometry, source_crs="EPSG:4326"), and safe_area(geometry). Cache one pyproj Transformer per CRS pair.

- [ ] Step 3: Implement the pure matcher

choose_winner(polygon, candidates, source_version=...) calculates each actual polygon.intersection(candidate.geometry).area, discards empty/non-positive intersections, sorts by negative area then ascending code, and returns 100 * winner area / polygon area. Clamp only floating-point noise to 0..100. Never use bbox area or candidate count.

- [ ] Step 4: Add deterministic Hypothesis properties

Use a fixed profile in tests/conftest.py with derandomize=True and max_examples=100. Generate rectangle geometries and prove overlap percentages are bounded, candidate permutation does not change the result, and equal-area ties select the ascending code.

Run, refactor only after GREEN, keep matching complexity <=5, and commit:

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ruff check src tests
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/unit
git add src tests
git commit -m "feat: select EUNIS labels by exact overlap"
~~~

## Task 3: Official EEA reference catalog and exact raster-cell adapter

**Files:** create src/osm_polygon_eunis/reference.py, config/eea-2021-reference.json, docs/adr/0001-reference-data-and-overlap.md, docs/technical-debt.md; test tests/unit/test_reference.py.

- [ ] Step 1: Write RED synthetic-raster tests

Create a two-cell EPSG:3035 GeoTIFF fixture in tmp_path. Test that Prob_R11_100m.tif parses to R11, an unknown code-name mapping fails closed, a bounds hit is only a candidate filter, and a polygon crossing cells selects the greater actual intersection. Add a diagonal-polygon regression where bbox overlap exists but actual intersection is empty.

- [ ] Step 2: Record official EEA source catalog

config/eea-2021-reference.json contains:

~~~json
{
  "source_version": "EEA EUNIS habitat probability maps v1 2021",
  "crs": "EPSG:3035",
  "threshold": 0,
  "catalog_records": [
    "99498d2c-7350-4655-b914-92c5b9e016c5",
    "f425a73e-dc6c-40be-93d6-9d234d4bbe1b",
    "443cdba1-1d4b-4cae-91d5-72fa99a8a758",
    "0c4270c4-fd7e-4099-ac2f-5af2079ddbd8",
    "ae2fdada-93d6-4cf1-a11c-6ebccf25d286",
    "7a2d78ec-8a31-4e7b-91df-45db6e64e842",
    "a6c48c2d-114f-406e-ba99-c9085a5f5aee"
  ],
  "label_source": "https://biodiversity.europa.eu/resources/search-habitat"
}
~~~

The resolver reads official metadata, enumerates public WebDAV TIFF assets, rejects non-EPSG:3035 layers, and writes a run manifest with URL, ETag, size, SHA-256, code, name, and source version. Raster files never enter Git.

- [ ] Step 3: Implement the bounded adapter

Define RasterLayer(code, name, path, source_version), RasterReference(layers, threshold), RasterReference.overlap(polygon), parse_layer_code(filename), load_label_table(path), and resolve_eea_layers(config_path, workspace).

For each layer, use bounds only to select a read window. Read values greater than threshold, polygonize selected cells with the window transform, and pass actual cell geometries to the pure matcher. Use rasterio.Env(GDAL_CACHEMAX=...), close datasets with ExitStack, download one group at a time, and report peak temporary bytes.

- [ ] Step 4: Document and run GREEN

The ADR states official EEA 2021 modelled probability maps, Europe-focused coverage, intentional null outside coverage, and exact projected intersection. Technical debt records raster resolution/modelled probabilities and the adapter replacement path for an official vector layer.

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ruff check src tests
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/unit/test_reference.py
git add src config docs/adr docs/technical-debt.md tests/unit/test_reference.py
git commit -m "feat: add official EEA raster reference adapter"
~~~

## Task 4: Schema-preserving bounded Parquet transformation

**Files:** create src/osm_polygon_eunis/transform.py; test tests/unit/test_transform.py and tests/integration/test_shard_pipeline.py.

- [ ] Step 1: Write RED tests

Create a tiny Parquet shard with arbitrary columns, JSON geometry, and polygon_id. Assert row order/count and every original column/type are preserved and exactly four nullable fields are appended. Missing geometry gets four nulls. A link shard with only polygon_id gets labels from a bounded map and requires no geometry.

- [ ] Step 2: Implement batch APIs

Implement enrich_parquet_shard(source, destination, reference, batch_size, geometry_column="geometry") and enrich_link_shard(source, destination, labels_by_polygon_id, batch_size). Use ParquetFile.iter_batches and ParquetWriter with the original schema plus nullable UTF-8/float64 fields. Keep one Arrow batch and one output batch alive. Build only the current Wikidata region's polygon_id map.

- [ ] Step 3: Integrate and commit

Run a synthetic two-cell reference through a three-row geometry shard and paired link shard. Assert original-column equality and one-to-one row conservation.

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/unit/test_transform.py tests/integration/test_shard_pipeline.py
git add src/osm_polygon_eunis/transform.py tests/unit/test_transform.py tests/integration/test_shard_pipeline.py
git commit -m "feat: enrich parquet shards in bounded batches"
~~~

## Task 5: Pinned HF inventory and streamed source shards

**Files:** create src/osm_polygon_eunis/sources.py; test tests/unit/test_sources.py and extend integration tests.

- [ ] Step 1: Write RED layout/pairing tests

Assert website uses polygons/*.parquet, description uses data/*.parquet, Wikidata uses polygons/*.parquet, and Wikidata links use polygon_document_links/*.parquet. Test that france-latest.parquet pairs with the same basename and unmatched links fail rather than dropping rows.

- [ ] Step 2: Implement metadata and direct temporary downloads

Define DatasetSpec, capture_revision(api, repo_id), list_parquet_files(api, repo_id, revision), pair_region_paths(polygons, links), and download_to_temp(api, repo_id, path, revision, directory). Use Hub file metadata to obtain a signed URL, stream into a named run-local file, verify bytes, never use the persistent default HF cache for production shards, and record source commit and every path.

- [ ] Step 3: Add mocked Hub tests and commit

Mock real path shapes such as polygons/france-latest.parquet and polygon_document_links/france-latest.parquet; prove lexical order, no duplicate paths, and fail-closed missing geometry/key columns.

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/unit/test_sources.py
git add src/osm_polygon_eunis/sources.py tests/unit/test_sources.py tests/integration/test_shard_pipeline.py
git commit -m "feat: stream pinned HF shards with bounded storage"
~~~

## Task 6: Server-side duplication, replacement uploads, and verification

**Files:** create src/osm_polygon_eunis/publish.py; test tests/unit/test_publish.py and tests/acceptance/test_dataset_enrichment.py.

- [ ] Step 1: Write RED publisher tests

Mock Hub calls and assert duplicate-once, same-path replacement, no upload of unchanged shared files, idempotent skip when a matching manifest exists, and rejection of row-count/schema/source-column/path mismatches.

- [ ] Step 2: Implement publisher APIs

Define duplicate_source, upload_replacement, upload_manifest, and verify_dataset. Use server-side dataset duplication, one commit per replacement shard, eunis/manifest.json, and no broad delete. Verify remote tree independently, including representative Parquet schemas/row counts and unchanged shared files.

- [ ] Step 3: Add deterministic cards and commit

Generate deterministic card additions with source repo/revision, EEA manifest/version, fields, null semantics, and exact formula. Add three verified target URLs to the HF collection named OSM Polygon EUNIS only after all receipts pass.

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/unit/test_publish.py tests/acceptance/test_dataset_enrichment.py
git add src/osm_polygon_eunis/publish.py tests/unit/test_publish.py tests/acceptance/test_dataset_enrichment.py
git commit -m "feat: publish and verify enriched HF datasets"
~~~

## Task 7: CLI, executable Gherkin, and local smoke

**Files:** create src/osm_polygon_eunis/cli.py, tests/acceptance/features/enrich_dataset.feature, tests/acceptance/test_dataset_enrichment.py, scripts/smoke.py; modify README/docs.

- [ ] Step 1: Write the feature first

Include executable scenarios for geometry enrichment by actual overlap, preservation of shared Wikidata tables, and remote verification from a matching manifest:

~~~gherkin
Feature: EUNIS enrichment
  Scenario: Enrich a geometry shard by actual polygon overlap
    Given a pinned source dataset with one geometry parquet shard
    And an EEA reference with two overlapping raster cells
    When I run enrichment with batch size 2
    Then output rows and original columns are unchanged
    And the label is chosen by largest actual intersection
    And overlap percentage uses polygon area

  Scenario: Preserve shared wikidata tables
    Given a wikidata source with polygon and document tables
    When I enrich the source
    Then only polygons and polygon_document_links contain EUNIS fields
    And document, section, sentence, wikivoyage, and fact files are unchanged

  Scenario: Verify a completed remote output
    Given a target dataset with a matching eunis manifest
    When I run verify
    Then verification returns a successful receipt with source revision and row counts
~~~

Use fake Hub/local fixtures; acceptance tests never contact external services.

- [ ] Step 2: Implement the CLI

Use argparse and expose:

~~~text
osm-polygon-eunis plan --config config/eea-2021-reference.json --source website
osm-polygon-eunis run --source all --batch-size 2048 --max-temp-bytes 8589934592
osm-polygon-eunis verify --output-repo NoeFlandre/osm-polygon-website-tag-eunis
osm-polygon-eunis clean --run-dir /private/tmp/osm-polygon-eunis-run
~~~

run refuses source-revision drift, unexpected target state, insufficient temp budget, missing declared columns, and unmatched links. It emits JSON lines with path, rows, bytes, elapsed seconds, and peak temp bytes.

- [ ] Step 3: Implement local smoke and commit

The smoke program builds a two-cell raster and three-row Parquet source, runs the local path without credentials/network, checks labels and row conservation, and checks temp cleanup.

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest -q tests/acceptance
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run python scripts/smoke.py
git add src/osm_polygon_eunis/cli.py tests/acceptance scripts/smoke.py README.md docs
git commit -m "feat: add reproducible enrichment CLI and acceptance tests"
~~~

## Task 8: Architecture, CRAP, mutation, and CI gates

**Files:** create scripts/check_architecture.py, scripts/check_crap.py, .github/workflows/ci.yml; modify pyproject.toml and README.

- [ ] Step 1: Add architecture checks

Import every production module, reject matching imports of HF/HTTP/rasterio/filesystem code, reject higher-level imports from lower-level modules, and detect cycles with graphlib.TopologicalSorter.

- [ ] Step 2: Add the CRAP check

Read coverage.json and radon cc -j src, compute complexity squared times (1 - coverage) cubed plus complexity, print the ten highest production functions, and exit 1 for any score >= 6.0. Treat uncovered functions as 0% covered; refactor instead of excluding them.

- [ ] Step 3: Add CI in the required order

The workflow runs uv sync --locked --group dev, Ruff format/check, ty check src tests scripts, pytest with coverage, property tests, acceptance tests, architecture, CRAP, mutation, and smoke. Tests never publish or contact external services.

- [ ] Step 4: Run gates and commit

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ruff format --check .
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ruff check .
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run ty check src tests scripts
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run pytest --cov --cov-report=json --cov-report=term-missing
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run python scripts/check_architecture.py
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run python scripts/check_crap.py
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run mutmut run
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run python scripts/smoke.py
git add pyproject.toml .github scripts README.md
git commit -m "ci: enforce deterministic architecture and quality gates"
~~~

Every meaningful surviving matcher mutant gets a permanent test or a documented justified exclusion.

## Task 9: Live pinned run and publication

**Files:** modify only generated run evidence/docs as needed; never commit source data or downloaded EEA rasters.

- [ ] Step 1: Verify credentials before mutation

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run hf auth whoami
gh auth status
~~~

If either credential is unavailable, stop before external writes and report the exact missing login; public read access is not write evidence.

- [ ] Step 2: Capture revisions and plan

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run osm-polygon-eunis plan --source all --config config/eea-2021-reference.json
~~~

Verify all three source revisions, target names, target file counts, EEA IDs/CRS/labels, and the temporary-storage estimate.

- [ ] Step 3: Run one representative region first

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run osm-polygon-eunis run --source website --region andorra-latest --batch-size 2048 --max-temp-bytes 8589934592
~~~

Verify remote schema, row count, original-column hashes, labels, and manifest before expanding. Any failure becomes a permanent test before continuation.

- [ ] Step 4: Publish all outputs with conservation checks

Run separately and resumably. For Wikidata, pair polygons/region.parquet with polygon_document_links/region.parquet; leave document, section, sentence, Wikivoyage, and fact files unchanged. Verify row counts and delete local temp files after every shard.

- [ ] Step 5: Independently verify targets

~~~bash
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run osm-polygon-eunis verify --output-repo NoeFlandre/osm-polygon-website-tag-eunis
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run osm-polygon-eunis verify --output-repo NoeFlandre/osm-polygon-wikidata-and-wikipedia-eunis
UV_CACHE_DIR=/private/tmp/osm-polygon-eunis-uv uv run osm-polygon-eunis verify --output-repo NoeFlandre/osm-polygon-description-tag-eunis
~~~

Proceed only when receipts prove source conservation, target paths/fields, unchanged shared files, no missing/duplicate shard, and EEA provenance.

- [ ] Step 6: Create and link public artifacts

Create public GitHub repo NoeFlandre/osm-polygon-eunis, push reviewed main, create four issues (shared pipeline plus one per output), add issue URLs to the GeoReSeT project, and add all three verified datasets to the public HF collection OSM Polygon EUNIS. Mutate external state only after credentials and live permissions are verified.

- [ ] Step 7: Run the final gauntlet and review the diff

Run exactly: baseline -> Ruff -> ty -> tests -> property tests -> acceptance tests -> architecture checks -> CRAP -> mutation tests -> smoke test -> diff review. Then inspect git diff --check, status/log, pushed GitHub tree, every HF manifest/tree, Dataset Viewer rows, collection membership, and project links. Report local checks and remote publication separately.

## Self-review

The plan covers the approved source inventory, geometry-bearing tables, unchanged shared Wikidata/Wikipedia tables, actual polygon geometry overlap with bbox-only pruning, deterministic tie/null rules, four fields, official provenance, shard streaming, bounded disk, strict TDD, Hypothesis, integration/Gherkin, architecture, CRAP, mutation, CI, MkDocs, ADRs, technical debt, smoke testing, remote verification, GitHub issues, GeoReSeT linking, and the HF collection. Credential failure is an explicit safe stop before mutation, not a fabricated success.
