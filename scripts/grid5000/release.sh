#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"

: "${OAR_JOB_ID:?this worker must run inside an OAR job}"
: "${GRID5000_PERSISTENT_ROOT:?set the remote persistent project root}"
: "${GRID5000_SOURCE_REVISION:?set the exact submitted source commit}"

hf_token_file="${HF_HOME:-$HOME/.cache/huggingface}/token"

case "$GRID5000_PERSISTENT_ROOT" in
  /home/*|/groups/*|/srv/*) ;;
  *) echo "GRID5000_PERSISTENT_ROOT must be persistent remote storage" >&2; exit 2 ;;
esac

source_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$source_root/scripts/grid5000/load_hf_token.sh"
eunis_load_hf_token "$hf_token_file"
persistent_root="$GRID5000_PERSISTENT_ROOT"
scratch="${TMPDIR:-/tmp}/osm-polygon-eunis-${OAR_JOB_ID}"
workdir="$persistent_root/runs/eunis"
reference_cache="$persistent_root/cache/reference"
sidecars="$persistent_root/sidecars/eunis"
logs="$persistent_root/logs"
receipts="$persistent_root/receipts"
receipt="$receipts/eunis-${OAR_JOB_ID}.json"
release_receipt="$receipts/eunis-${OAR_JOB_ID}.release.json"
reference_config="$source_root/config/eea-2021-reference.json"
source_commit="$GRID5000_SOURCE_REVISION"
export EUNIS_SOURCE_COMMIT="$source_commit"
site="${GRID5000_SITE:-unknown}"
frontend="${GRID5000_FRONTEND:-unknown}"
cluster="${GRID5000_CLUSTER:-unknown}"
queue="${GRID5000_QUEUE:-unknown}"
cores="${GRID5000_CORES:-0}"
workers="${GRID5000_WORKERS:-16}"
walltime="${GRID5000_WALLTIME:-unknown}"
batch_size="${GRID5000_BATCH_SIZE:-256}"
excluded_sites_json="${GRID5000_EXCLUDED_SITES:-[]}"
config_sha256="$(sha256sum "$reference_config" | cut -d ' ' -f 1)"
attempt=0
status=0
error_count=0

mkdir -p "$scratch" "$workdir" "$reference_cache" "$sidecars" "$logs" "$receipts"

write_failure_receipt() {
  status=$?
  if [[ "$status" -ne 0 && ! -e "$receipt" ]]; then
    python3 - "$receipt" "$OAR_JOB_ID" "$source_commit" "$config_sha256" \
      "$reference_config" "$site" "$frontend" "$cluster" "$queue" "$cores" \
      "$workers" "$walltime" "$batch_size" "$excluded_sites_json" \
      "$attempt" "$error_count" "$status" \
      "$logs/job-${OAR_JOB_ID}.log" <<'PY'
import json
import os
import sys

(
    path, job_id, source_commit, config_sha256, config_path, site, frontend,
    cluster, queue, cores, workers, walltime, batch_size, excluded_sites, attempts, errors,
    exit_status, log_path,
) = sys.argv[1:]
payload = {
    "status": "failed",
    "datasets": ["website", "wikidata", "description"],
    "grid5000": {
        "job_id": job_id,
        "source_commit": source_commit,
        "site": site,
        "frontend": frontend,
        "cluster": cluster,
        "queue": queue,
        "cores": int(cores),
        "workers": int(workers),
        "walltime": walltime,
        "batch_size": int(batch_size),
        "excluded_sites": json.loads(excluded_sites),
        "config": {"path": config_path, "sha256": config_sha256},
        "attempts": int(attempts),
        "retries": max(0, int(attempts) - 1),
        "errors": {"count": int(errors) + 1, "last_exit_status": int(exit_status)},
        "log_path": log_path,
    },
}
temporary = f"{path}.tmp"
with open(temporary, "w", encoding="utf-8") as output:
    json.dump(payload, output, sort_keys=True, indent=2)
    output.write("\n")
os.replace(temporary, path)
PY
  fi
  exit "$status"
}
trap write_failure_receipt EXIT

export EUNIS_SOURCE_DIR="$scratch/source"
export EUNIS_REFERENCE_DIR="$reference_cache"
export EUNIS_SIDECAR_DIR="$sidecars"
export UV_PROJECT_ENVIRONMENT="$scratch/venv"
export UV_CACHE_DIR="$scratch/uv-cache"
max_attempts="${GRID5000_MAX_ATTEMPTS:-20}"
retry_delay="${GRID5000_RETRY_DELAY:-30}"

exec > >(tee -a "$logs/job-${OAR_JOB_ID}.log") 2>&1
cd -- "$source_root"
echo "job_id=$OAR_JOB_ID"
echo "source_commit=$source_commit"
echo "grid5000_site=$site"
echo "grid5000_frontend=$frontend"
echo "grid5000_cluster=$cluster"
echo "grid5000_queue=$queue"
echo "grid5000_cores=$cores"
echo "grid5000_workers=$workers"
echo "grid5000_walltime=$walltime"
echo "grid5000_batch_size=$batch_size"
echo "grid5000_excluded_sites=$excluded_sites_json"
echo "reference_config=$reference_config"
echo "reference_config_sha256=$config_sha256"
if [[ "$site" == "unknown" || "$frontend" == "unknown" || "$cluster" == "unknown" \
  || "$queue" == "unknown" || ! "$cores" =~ ^[1-9][0-9]*$ \
  || ! "$workers" =~ ^[1-9][0-9]*$ || ! "$batch_size" =~ ^[1-9][0-9]*$ \
  || "$walltime" == "unknown" ]]; then
  echo "Grid'5000 OAR configuration metadata is incomplete" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" && ! -s "$hf_token_file" ]]; then
  echo "HF_TOKEN or the Hugging Face cache must be available on the reserved node" >&2
  exit 2
fi
uv sync --frozen --no-dev
attempt=1
status=1
while (( attempt <= max_attempts )); do
  echo "release attempt $attempt/$max_attempts"
  if uv run --frozen --no-dev osm-polygon-eunis release \
    --reference-config "$reference_config" \
    --workdir "$workdir" \
    --workers "${GRID5000_WORKERS:-16}" \
    --batch-size "${GRID5000_BATCH_SIZE:-256}" \
    --receipt "$release_receipt"; then
    python3 - "$release_receipt" "$receipt" "$OAR_JOB_ID" "$source_commit" \
      "$config_sha256" "$reference_config" "$site" "$frontend" "$cluster" \
      "$queue" "$cores" "$workers" "$walltime" "$batch_size" \
      "$excluded_sites_json" "$attempt" "$error_count" \
      "$logs/job-${OAR_JOB_ID}.log" <<'PY'
import json
import os
import sys

(
    source, destination, job_id, source_commit, config_sha256, config_path, site,
    frontend, cluster, queue, cores, workers, walltime, batch_size, excluded_sites,
    attempts, errors, log_path,
) = sys.argv[1:]
with open(source, encoding="utf-8") as input_file:
    payload = json.load(input_file)
payload["grid5000"] = {
    "job_id": job_id,
    "source_commit": source_commit,
    "site": site,
    "frontend": frontend,
    "cluster": cluster,
    "queue": queue,
    "cores": int(cores),
    "workers": int(workers),
    "walltime": walltime,
    "batch_size": int(batch_size),
    "excluded_sites": json.loads(excluded_sites),
    "config": {"path": config_path, "sha256": config_sha256},
    "attempts": int(attempts),
    "retries": max(0, int(attempts) - 1),
    "errors": {"count": int(errors)},
    "log_path": log_path,
}
temporary = f"{destination}.tmp"
with open(temporary, "w", encoding="utf-8") as output:
    json.dump(payload, output, sort_keys=True, indent=2)
    output.write("\n")
os.replace(temporary, destination)
os.unlink(source)
PY
    exit 0
  else
    status=$?
    error_count=$((error_count + 1))
  fi
  if (( status == 130 || status == 143 || attempt == max_attempts )); then
    break
  fi
  echo "release attempt $attempt failed with rc=$status; retrying after checkpoints"
  attempt=$((attempt + 1))
  sleep "$retry_delay"
done
exit "$status"
