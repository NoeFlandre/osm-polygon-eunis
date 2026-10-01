#!/usr/bin/env bash

active_helper_pid=""
active_helper_ready_file=""
helper_ready=false
helper_launching=false
pending_worker_signal=""

run_deadline_helper() {
  pending_worker_signal=""
  helper_launching=true
  helper_ready=false
  active_helper_ready_file="${TMPDIR:-/tmp}/osm-polygon-eunis-helper-ready-$$"
  rm -f "$active_helper_ready_file"
  GRID5000_WORKER_READY_FILE="$active_helper_ready_file" python3 "$@" &
  register_worker_helper "$!"
  local helper_status=0
  if wait_for_worker_helper_ready "$active_helper_ready_file"; then
    if [[ -n "$active_helper_pid" ]]; then
      wait "$active_helper_pid" || helper_status=$?
      active_helper_pid=""
    fi
  else
    helper_status=$?
  fi
  helper_ready=false
  rm -f "$active_helper_ready_file"
  active_helper_ready_file=""
  return "$helper_status"
}

register_worker_helper() {
  active_helper_pid="$1"
  helper_launching=false
}

wait_for_worker_helper_ready() {
  local ready_file="$1"
  local helper_status=0
  while [[ ! -e "$ready_file" ]]; do
    if ! kill -0 "$active_helper_pid" 2>/dev/null; then
      wait "$active_helper_pid" || helper_status=$?
      active_helper_pid=""
      forward_pending_worker_signal
      return "$helper_status"
    fi
    sleep 0.02
  done
  helper_ready=true
  forward_pending_worker_signal
}

forward_pending_worker_signal() {
  if [[ -n "$pending_worker_signal" ]]; then
    local pending_signal="$pending_worker_signal"
    pending_worker_signal=""
    case "$pending_signal" in
      INT) handle_worker_signal INT 130 ;;
      TERM) handle_worker_signal TERM 143 ;;
    esac
  fi
}

handle_worker_signal() {
  local signal_name="$1"
  local exit_status="$2"
  external_signal="$signal_name"
  if [[ "$helper_launching" == true || ( -n "$active_helper_pid" && "$helper_ready" != true ) ]]; then
    pending_worker_signal="$signal_name"
    return 0
  fi
  if [[ -n "$active_helper_pid" ]]; then
    kill -s "$signal_name" "$active_helper_pid" 2>/dev/null || true
    wait "$active_helper_pid" 2>/dev/null || true
    active_helper_pid=""
  fi
  helper_ready=false
  if [[ -n "$active_helper_ready_file" ]]; then
    rm -f "$active_helper_ready_file"
    active_helper_ready_file=""
  fi
  exit "$exit_status"
}

install_worker_signal_traps() {
  trap 'handle_worker_signal INT 130' INT
  trap 'handle_worker_signal TERM 143' TERM
}
