#!/usr/bin/env bash
# Development-only diagnostic runner. Requires an already-running Ollama
# endpoint that exposes qwen3:14b; it never invokes the main 1000-task plan.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CONFIG="$ROOT/configs/llm_retrieval_diagnostic_v1.json"
FROZEN="$ROOT/configs/frozen_llm_config_v1.json"
CHECKPOINTS="$ROOT/experiments/checkpoints"

cd "$ROOT"
for CONDITION in full top15 top30; do
  "$PYTHON_BIN" scripts/run_llm_retrieval_diagnostic.py \
    --config "$CONFIG" \
    --condition "$CONDITION" \
    --frozen-config "$FROZEN" \
    --checkpoint "$CHECKPOINTS/LLM_RETRIEVAL_DIAGNOSTIC_${CONDITION^^}.json"
done

"$PYTHON_BIN" scripts/finalize_llm_retrieval_diagnostic.py \
  --config "$CONFIG" \
  --checkpoint-dir "$CHECKPOINTS" \
  --result "$ROOT/experiments/results/LLM_CAPABILITY_RETRIEVAL_DIAGNOSTIC_V1.json"
