from pathlib import Path


def test_production_contract_is_grid5000_all_source_and_site_neutral() -> None:
    root = Path(__file__).parents[2]
    operations = (root / "docs" / "operations.md").read_text(encoding="utf-8")
    readme = (root / "README.md").read_text(encoding="utf-8")
    worker = (root / "scripts" / "grid5000" / "release.sh").read_text(encoding="utf-8")

    assert "Grid'5000" in operations
    assert "grid5000 submit" in operations
    assert "usagepolicycheck -t" in operations
    assert "host=1/core=16" in operations
    assert "--site SITE" in operations
    assert "--cluster CLUSTER" in operations
    assert "cluster='CLUSTER'" in operations
    assert "-q default" in operations
    assert "website" in operations
    assert "wikidata" in operations
    assert "description" in operations
    assert "--receipt" in operations
    assert "EUNIS_SIDECAR_DIR" in operations
    assert "EUNIS_SOURCE_DIR" in operations
    assert "node-local" in operations
    assert "not backed up" in operations
    assert (
        "uv run osm-polygon-eunis release --batch-size 256 --workdir .eunis-run" not in operations
    )
    assert "Grid'5000-only" in readme
    assert "all three" in readme
    assert "--dataset" not in worker
    assert '--receipt "$release_receipt"' in worker
    assert (
        worker.index('source_commit="$GRID5000_SOURCE_REVISION"')
        < worker.index('export EUNIS_SOURCE_COMMIT="$source_commit"')
        < worker.index("uv run --frozen --no-dev osm-polygon-eunis release")
    )
    for field in (
        '"site": site',
        '"frontend": frontend',
        '"cluster": cluster',
        '"queue": queue',
        '"cores": int(cores)',
        '"workers": int(workers)',
        '"walltime": walltime',
        '"batch_size": int(batch_size)',
        '"errors": {"count": int(errors)}',
    ):
        assert field in worker
