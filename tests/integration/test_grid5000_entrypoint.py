from pathlib import Path


def test_description_entrypoint_is_checkpointed_and_grid5000_only() -> None:
    script = (
        Path(__file__).parents[2] / "scripts" / "grid5000" / "description-release.sh"
    ).read_text(encoding="utf-8")

    assert "set -euo pipefail" in script
    assert ': "${OAR_JOB_ID:?' in script
    assert ': "${HF_TOKEN:?' in script
    assert 'EUNIS_SOURCE_DIR="$scratch/source"' in script
    assert 'EUNIS_REFERENCE_DIR="$scratch/reference"' in script
    assert 'UV_CACHE_DIR="$scratch/uv-cache"' in script
    assert 'EUNIS_SIDECAR_DIR="$sidecars"' in script
    assert "uv run --frozen --no-dev osm-polygon-eunis release" in script
    assert "--dataset description" in script
    assert "--execution grid5000" in script
    assert "--receipt \"$receipt\"" in script
    assert "trap write_failure_receipt EXIT" in script
