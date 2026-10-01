#!/usr/bin/env bash

active_helper_pid=""
helper_launching=false
pending_worker_signal=""

run_deadline_helper() {
  pending_worker_signal=""
  helper_launching=true
  python3 "$@" &
  register_worker_helper "$!"
  local helper_status=0
  wait "$active_helper_pid" || helper_status=$?
  active_helper_pid=""
  return "$helper_status"
}

register_worker_helper() {
  active_helper_pid="$1"
  helper_launching=false
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
  if [[ -z "$active_helper_pid" && "$helper_launching" == true ]]; then
    pending_worker_signal="$signal_name"
    return 0
  fi
  if [[ -n "$active_helper_pid" ]]; then
    kill -s "$signal_name" "$active_helper_pid" 2>/dev/null || true
    wait "$active_helper_pid" 2>/dev/null || true
    active_helper_pid=""
  fi
  exit "$exit_status"
}

install_worker_signal_traps() {
  trap 'handle_worker_signal INT 130' INT
  trap 'handle_worker_signal TERM 143' TERM
}
