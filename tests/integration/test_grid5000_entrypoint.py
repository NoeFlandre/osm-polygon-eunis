import os
import shutil
import subprocess
from pathlib import Path


def test_release_entrypoint_is_checkpointed_and_all_source_grid5000_only() -> None:
    script = (Path(__file__).parents[2] / "scripts" / "grid5000" / "release.sh").read_text(
        encoding="utf-8"
    )

    assert "set -euo pipefail" in script
    assert ': "${OAR_JOB_ID:?' in script
    assert ': "${HF_TOKEN:?HF_TOKEN must be provided on the reserved node}"' not in script
    assert "HF_HOME" in script
    assert '"${HF_HOME:-$HOME/.cache/huggingface}/token"' in script
    assert 'source "$source_root/scripts/grid5000/load_hf_token.sh"' in script
    assert 'source "$source_root/scripts/grid5000/worker_signals.sh"' in script
    assert 'eunis_load_hf_token "$hf_token_file"' in script
    assert "HF_TOKEN or the Hugging Face cache" in script
    assert 'export PATH="$HOME/.local/bin:$PATH"' in script
    assert 'EUNIS_SOURCE_DIR="$scratch/source"' in script
    assert 'reference_cache="$persistent_root/cache/reference"' in script
    assert 'EUNIS_REFERENCE_DIR="$reference_cache"' in script
    assert 'UV_CACHE_DIR="$scratch/uv-cache"' in script
    assert script.count('run_deadline_helper "$deadline_helper"') == 3
    assert 'stop_margin_seconds="${GRID5000_STOP_MARGIN_SECONDS:-300}"' in script
    assert 'termination_grace_seconds="${GRID5000_TERMINATION_GRACE_SECONDS:-20}"' in script
    assert 'stop_marker="$scratch/worker-stop-state"' in script
    assert script.count('--stop-marker "$stop_marker"') == 4
    assert 'if (( status == 124 )) && [[ "$stop_state" == "deadline" ]]' in script
    assert 'python3 "$receipt_writer"' in script
    assert script.index('-- sleep "$retry_delay"') < script.index("attempt=$((attempt + 1))")
    assert 'EUNIS_SIDECAR_DIR="$sidecars"' in script
    assert "uv run --frozen --no-dev osm-polygon-eunis release" in script
    assert 'max_attempts="${GRID5000_MAX_ATTEMPTS:-20}"' in script
    assert "while (( attempt <= max_attempts )); do" in script
    assert '--receipt "$release_receipt"' in script
    assert '--workers "${GRID5000_WORKERS:-16}"' in script
    assert "trap write_failure_receipt EXIT" in script
    assert "install_worker_signal_traps" in script
    assert 'workdir="$persistent_root/runs/eunis"' in script
    assert 'sidecars="$persistent_root/sidecars/eunis"' in script
    assert "description-release.sh" not in script


def test_grid_token_loader_exports_cached_token_without_printing_it(tmp_path: Path) -> None:
    credential_value = "hf_test_credential_value"
    token_file = tmp_path / "token"
    token_file.write_text(f"{credential_value}\n", encoding="utf-8")
    loader = Path(__file__).parents[2] / "scripts" / "grid5000" / "load_hf_token.sh"
    environment = os.environ.copy()
    environment.pop("HF_TOKEN", None)
    environment["EXPECTED_TOKEN"] = credential_value
    shell = shutil.which("bash")
    assert shell is not None

    # The script and all arguments are local test fixtures from trusted paths.
    result = subprocess.run(  # noqa: S603
        [
            shell,
            "-c",
            'source "$1"; eunis_load_hf_token "$2"; test "$HF_TOKEN" = "$EXPECTED_TOKEN"',
            "bash",
            str(loader),
            str(token_file),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert credential_value not in result.stdout
    assert credential_value not in result.stderr
