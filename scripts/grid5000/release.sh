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
stop_marker="$scratch/worker-stop-state"
receipt_writer="$source_root/scripts/grid5000/receipts.py"
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
job_started_at="$(date +%s)"
stop_margin_seconds="${GRID5000_STOP_MARGIN_SECONDS:-300}"
termination_grace_seconds="${GRID5000_TERMINATION_GRACE_SECONDS:-20}"
batch_size="${GRID5000_BATCH_SIZE:-256}"
excluded_sites_json="${GRID5000_EXCLUDED_SITES:-[]}"
config_sha256="$(sha256sum "$reference_config" | cut -d ' ' -f 1)"
attempt=0
status=0
error_count=0
external_signal=none

mkdir -p "$scratch" "$workdir" "$reference_cache" "$sidecars" "$logs" "$receipts"

write_failure_receipt() {
  local exit_status=$?
  trap - EXIT
  if [[ "$exit_status" -ne 0 && ! -e "$receipt" ]]; then
    python3 "$receipt_writer" \
      --output "$receipt" \
      --job-id "$OAR_JOB_ID" \
      --source-commit "$source_commit" \
      --config-sha256 "$config_sha256" \
      --config-path "$reference_config" \
      --site "$site" \
      --frontend "$frontend" \
      --cluster "$cluster" \
      --queue "$queue" \
      --cores "$cores" \
      --workers "$workers" \
      --walltime "$walltime" \
      --batch-size "$batch_size" \
      --excluded-sites-json "$excluded_sites_json" \
      --attempts "$attempt" \
      --error-count "$error_count" \
      --exit-status "$exit_status" \
      --job-started-epoch "$job_started_at" \
      --stop-margin-seconds="$stop_margin_seconds" \
      --termination-grace-seconds="$termination_grace_seconds" \
      --log-path "$logs/job-${OAR_JOB_ID}.log" \
      --stop-marker "$stop_marker" \
      --external-signal "$external_signal" \
      || echo "failed to write Grid'5000 failure receipt" >&2
  fi
  exit "$exit_status"
}
trap write_failure_receipt EXIT
trap 'external_signal=INT; exit 130' INT
trap 'external_signal=TERM; exit 143' TERM

export EUNIS_SOURCE_DIR="$scratch/source"
export EUNIS_REFERENCE_DIR="$reference_cache"
export EUNIS_SIDECAR_DIR="$sidecars"
export UV_PROJECT_ENVIRONMENT="$scratch/venv"
export UV_CACHE_DIR="$scratch/uv-cache"
deadline_helper="$source_root/scripts/grid5000/worker_deadline.py"
max_attempts="${GRID5000_MAX_ATTEMPTS:-20}"
retry_delay="${GRID5000_RETRY_DELAY:-30}"
stop_state=""

load_stop_state() {
  stop_state=""
  if [[ -f "$stop_marker" ]]; then
    IFS= read -r stop_state < "$stop_marker" || true
  fi
}

is_guarded_stop() {
  case "$status:$stop_state" in
    124:deadline|130:signal:2|143:signal:15) return 0 ;;
    *) return 1 ;;
  esac
}

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
echo "grid5000_job_started_epoch=$job_started_at"
echo "grid5000_stop_margin_seconds=$stop_margin_seconds"
echo "grid5000_termination_grace_seconds=$termination_grace_seconds"
echo "grid5000_batch_size=$batch_size"
echo "grid5000_excluded_sites=$excluded_sites_json"
echo "reference_config=$reference_config"
echo "reference_config_sha256=$config_sha256"
if [[ "$site" == "unknown" || "$frontend" == "unknown" || "$cluster" == "unknown" \
  || "$queue" == "unknown" || ! "$cores" =~ ^[1-9][0-9]*$ \
  || ! "$workers" =~ ^[1-9][0-9]*$ || ! "$batch_size" =~ ^[1-9][0-9]*$ \
  || ! "$stop_margin_seconds" =~ ^[0-9]+$ \
  || ! "$termination_grace_seconds" =~ ^[0-9]+$ \
  || "$walltime" == "unknown" ]]; then
  echo "Grid'5000 OAR configuration metadata is incomplete" >&2
  exit 2
fi
if [[ -z "${HF_TOKEN:-}" && ! -s "$hf_token_file" ]]; then
  echo "HF_TOKEN or the Hugging Face cache must be available on the reserved node" >&2
  exit 2
fi
if python3 "$deadline_helper" \
  --walltime "$walltime" \
  --started-at "$job_started_at" \
  --stop-margin-seconds "$stop_margin_seconds" \
  --termination-grace-seconds "$termination_grace_seconds" \
  --stop-marker "$stop_marker" \
  -- uv sync --frozen --no-dev; then
  :
else
  status=$?
  load_stop_state
  if ! is_guarded_stop; then
    error_count=$((error_count + 1))
  fi
  exit "$status"
fi
attempt=1
status=1
while (( attempt <= max_attempts )); do
  echo "release attempt $attempt/$max_attempts"
  if python3 "$deadline_helper" \
    --walltime "$walltime" \
    --started-at "$job_started_at" \
    --stop-margin-seconds "$stop_margin_seconds" \
    --termination-grace-seconds "$termination_grace_seconds" \
    --stop-marker "$stop_marker" \
    -- uv run --frozen --no-dev osm-polygon-eunis release \
    --reference-config "$reference_config" \
    --workdir "$workdir" \
    --workers "${GRID5000_WORKERS:-16}" \
    --batch-size "${GRID5000_BATCH_SIZE:-256}" \
    --receipt "$release_receipt"; then
    python3 - "$release_receipt" "$receipt" "$OAR_JOB_ID" "$source_commit" \
      "$config_sha256" "$reference_config" "$site" "$frontend" "$cluster" \
      "$queue" "$cores" "$workers" "$walltime" "$batch_size" \
      "$excluded_sites_json" "$attempt" "$error_count" "$job_started_at" \
      "$stop_margin_seconds" "$termination_grace_seconds" \
      "$logs/job-${OAR_JOB_ID}.log" <<'PY'
import json
import os
import sys

(
    source, destination, job_id, source_commit, config_sha256, config_path, site,
    frontend, cluster, queue, cores, workers, walltime, batch_size, excluded_sites,
    attempts, errors, job_started_at, stop_margin_seconds,
    termination_grace_seconds, log_path,
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
    "job_started_epoch": int(job_started_at),
    "stop_margin_seconds": int(stop_margin_seconds),
    "termination_grace_seconds": float(termination_grace_seconds),
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
    load_stop_state
    if ! is_guarded_stop; then
      error_count=$((error_count + 1))
    fi
  fi
  load_stop_state
  if (( status == 124 )) && [[ "$stop_state" == "deadline" ]]; then
    echo "graceful worker deadline reached; saved checkpoints are ready to resume"
    break
  fi
  if [[ "$status:$stop_state" == "130:signal:2" || "$status:$stop_state" == "143:signal:15" \
    || "$attempt" -eq "$max_attempts" ]]; then
    break
  fi
  echo "release attempt $attempt failed with rc=$status; retrying after checkpoints"
  if python3 "$deadline_helper" \
    --walltime "$walltime" \
    --started-at "$job_started_at" \
    --stop-margin-seconds "$stop_margin_seconds" \
    --termination-grace-seconds "$termination_grace_seconds" \
    --stop-marker "$stop_marker" \
    -- sleep "$retry_delay"; then
    status=0
    attempt=$((attempt + 1))
  else
    status=$?
    load_stop_state
    if ! is_guarded_stop; then
      error_count=$((error_count + 1))
    fi
  fi
  load_stop_state
  if (( status == 124 )) && [[ "$stop_state" == "deadline" ]]; then
    echo "graceful worker deadline reached during retry delay; saved checkpoints are ready to resume"
    break
  fi
  if [[ "$status:$stop_state" == "130:signal:2" || "$status:$stop_state" == "143:signal:15" ]]; then
    break
  fi
  if (( status != 0 )); then
    echo "retry delay failed with rc=$status; stopping with checkpoints ready to resume"
    break
  fi
done
exit "$status"
