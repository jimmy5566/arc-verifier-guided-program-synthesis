#!/usr/bin/env bash
# Import an existing RunPod PID-1 credential only into the current child shell.
# This helper never writes or prints credential values.  It accepts only the
# two transport credentials used by ARC2 and is intended to be sourced.
arc2_import_runpod_pid1_secret() {
  local name="$1" env_path value
  case "$name" in HF_TOKEN|GH_TOKEN) ;; *) return 64 ;; esac
  env_path="${ARC2_PID1_ENV_PATH:-/proc/1/environ}"
  [ -r "$env_path" ] || return 65
  value="$(tr '\000' '\n' < "$env_path" | sed -n "s/^${name}=//p" | head -n 1)"
  [ -n "$value" ] || { unset value; return 66; }
  export "$name=$value"
  unset value
}

arc2_import_runpod_hf_token() {
  arc2_import_runpod_pid1_secret HF_TOKEN
}
