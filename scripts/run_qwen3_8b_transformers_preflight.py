"""Offline Qwen3-8B single-request GPU preflight for the frozen V2 transport.

This is deliberately not an ARC task run.  It validates only that the attached
competition model can load through direct Transformers and generate a bounded,
parseable response on one explicitly selected GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from inference.kaggle_l4_parallel_runner import gpu_observation, inspect_hardware
from llm.models import GenerationConfig
from llm.transformers_provider import TransformersProvider


def _parse_json_object(text: str) -> dict[str, object]:
    """Strictly accept one JSON object, allowing only surrounding whitespace."""
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError("completion is not a JSON object")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    if args.gpu != 0:
        raise ValueError("this single-process preflight must use logical cuda:0; pin physical GPU with CUDA_VISIBLE_DEVICES")
    hardware = inspect_hardware()
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    provider = TransformersProvider(model_path=args.model_path, device="cuda:0")
    before = gpu_observation(0)
    result: dict[str, object] = {
        "experiment_id": "LLM_PROGRAM_SYNTHESIS_V2",
        "stage": "QWEN3_8B_SINGLE_GPU_SINGLE_REQUEST_PREFLIGHT",
        "protocol": "offline direct Transformers; no ARC task or solution file opened",
        "hardware": hardware.to_dict(),
        "model_path": str(args.model_path),
        "availability": provider.availability().__dict__,
        "gpu_before": before,
        "http_status": None,
        "server_log_error": None,
    }
    try:
        load_seconds = provider.load()
        after_load = gpu_observation(0)
        generated = provider.generate_text(
            'Return only this JSON object: {"ok": true}',
            GenerationConfig("qwen-lm/qwen-3/Transformers/8b/1", temperature=0, top_p=1, seed=0, max_output_tokens=32, context_window=12288, hypothesis_budget=1),
        )
        parsed = _parse_json_object(generated.text)
        if generated.completion_tokens <= 0:
            raise RuntimeError("completion_tokens must be positive")
        if parsed.get("ok") is not True:
            raise RuntimeError(f"unexpected completion payload: {parsed!r}")
        after_generate = gpu_observation(0)
        if not isinstance(after_load.get("vram_used_mib"), int) or after_load["vram_used_mib"] <= (before.get("vram_used_mib") or 0):
            raise RuntimeError("model load did not produce an observable GPU0 VRAM increase")
        result.update({
            "status": "SUCCESS",
            "load_seconds": load_seconds,
            "generation_seconds": generated.elapsed_seconds,
            "prompt_tokens": generated.prompt_tokens,
            "completion_tokens": generated.completion_tokens,
            "parsed_completion": parsed,
            "gpu_after_load": after_load,
            "gpu_after_generate": after_generate,
        })
    except Exception as exc:
        result.update({"status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc), "gpu_after_failure": gpu_observation(0)})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: result.get(key) for key in ("status", "failure_type", "failure", "load_seconds", "generation_seconds", "prompt_tokens", "completion_tokens", "parsed_completion")}, sort_keys=True), flush=True)
    if result["status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
