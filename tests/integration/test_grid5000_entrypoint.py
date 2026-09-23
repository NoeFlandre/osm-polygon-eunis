from pathlib import Path


def test_release_entrypoint_is_checkpointed_and_all_source_grid5000_only() -> None:
    script = (Path(__file__).parents[2] / "scripts" / "grid5000" / "release.sh").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in script
    assert ': "${OAR_JOB_ID:?' in script
    assert ': "${HF_TOKEN:?HF_TOKEN must be provided on the reserved node}"' not in script
    assert 'HF_HOME' in script
    assert '"${HF_HOME:-$HOME/.cache/huggingface}/token"' in script
    assert 'HF_TOKEN or the Hugging Face cache' in script
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script
    assert 'EUNIS_SOURCE_DIR="$scratch/source"' in script
    assert 'EUNIS_REFERENCE_DIR="$scratch/reference"' in script
    assert 'UV_CACHE_DIR="$scratch/uv-cache"' in script
    assert 'EUNIS_SIDECAR_DIR="$sidecars"' in script
    assert "uv run --frozen --no-dev osm-polygon-eunis release" in script
    assert 'max_attempts="${GRID5000_MAX_ATTEMPTS:-20}"' in script
    assert "while (( attempt <= max_attempts )); do" in script
    assert "--execution grid5000" in script
    assert "--receipt \"$receipt\"" in script
    assert "trap write_failure_receipt EXIT" in script
    assert 'workdir="$persistent_root/runs/eunis"' in script
    assert 'sidecars="$persistent_root/sidecars/eunis"' in script
    assert "description-release.sh" not in script
