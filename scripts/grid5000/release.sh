#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

: "${OAR_JOB_ID:?this worker must run inside an OAR job}"
: "${GRID5000_PERSISTENT_ROOT:?set the remote persistent project root}"

hf_token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"
if [[ -z "${HF_TOKEN:-}" && ! -s "$hf_token_file" ]]; then
  echo "HF_TOKEN or the Hugging Face cache must be available on the reserved node" >&2
  exit 2
fi

case "$GRID5000_PERSISTENT_ROOT" in
  /home/*|/groups/*|/srv/*) ;;
  *) echo "GRID5000_PERSISTENT_ROOT must be persistent remote storage" >&2; exit 2 ;;
esac

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
persistent_root="$GRID5000_PERSISTENT_ROOT"
scratch="${TMPDIR:-/tmp}/osm-polygon-eunis-${OAR_JOB_ID}"
workdir="$persistent_root/runs/eunis"
sidecars="$persistent_root/sidecars/eunis"
logs="$persistent_root/logs"
receipts="$persistent_root/receipts"
receipt="$receipts/eunis-${OAR_JOB_ID}.json"

mkdir -p "$scratch" "$workdir" "$sidecars" "$logs" "$receipts"

write_failure_receipt() {
  status=$?
  if [[ "$status" -ne 0 && ! -e "$receipt" ]]; then
    temporary="$receipt.tmp"
    printf '{"datasets":["website","wikidata","description"],"job_id":"%s","status":"failed"}\n' \
      "$OAR_JOB_ID" > "$temporary"
    mv -- "$temporary" "$receipt"
  fi
  exit "$status"
}
trap write_failure_receipt EXIT

export EUNIS_SOURCE_DIR="$scratch/source"
export EUNIS_REFERENCE_DIR="$scratch/reference"
export EUNIS_SIDECAR_DIR="$sidecars"
export EUNIS_SOURCE_WORKERS="${GRID5000_WORKERS:-16}"
export UV_PROJECT_ENVIRONMENT="$scratch/venv"
export UV_CACHE_DIR="$scratch/uv-cache"

exec > >(tee -a "$logs/job-${OAR_JOB_ID}.log") 2>&1
cd -- "$source_root"
uv sync --frozen --no-dev
uv run --frozen --no-dev osm-polygon-eunis release \
  --execution grid5000 \
  --reference-config "$source_root/config/eea-2021-reference.json" \
  --workdir "$workdir" \
  --batch-size "${GRID5000_BATCH_SIZE:-256}" \
  --receipt "$receipt"
