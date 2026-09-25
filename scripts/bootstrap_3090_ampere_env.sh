#!/usr/bin/env bash
# Restore the shared CUDA 12.8 runtime on RTX 3090 / sm_86 only.  This is an
# infrastructure preflight; it never launches ARC inference or reads targets.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export ARC2_ENVIRONMENT_ID="${ARC2_ENVIRONMENT_ID:-3090-ampere-env-v1}"
export ARC2_ENV_ARCHIVE_LAYOUT="${ARC2_ENV_ARCHIVE_LAYOUT:-4090-ada-env-v1}"
export ARC2_EXPECTED_GPU_NAME="${ARC2_EXPECTED_GPU_NAME:-RTX 3090}"
export ARC2_EXPECTED_GPU_CAPABILITY="${ARC2_EXPECTED_GPU_CAPABILITY:-8,6}"
export ARC2_LOCAL_RUNTIME_ROOT="${ARC2_LOCAL_RUNTIME_ROOT:-/root/arc-runtime-3090}"

exec "${SCRIPT_DIR}/bootstrap_4090_ada_env.sh"
