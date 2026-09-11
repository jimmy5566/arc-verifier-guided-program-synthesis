"""Concurrent four-worker direct-Transformers preflight on the 4xL4 runtime."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
from pathlib import Path
from queue import Empty

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))


def worker(worker_id: int, model_path: str, queue: object) -> None:
    # Must precede the provider's first torch import: physical GPU N becomes
    # this worker's logical cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker_id)
    try:
        from inference.kaggle_l4_parallel_runner import gpu_observation
        from llm.models import GenerationConfig
        from llm.transformers_provider import TransformersProvider

        provider = TransformersProvider(model_path=Path(model_path), device="cuda:0")
        before = gpu_observation(worker_id)
        load_seconds = provider.load()
        after_load = gpu_observation(worker_id)
        generated = provider.generate_text('Return only this JSON object: {"ok": true}', GenerationConfig("qwen-lm/qwen-3/Transformers/8b/1", temperature=0, top_p=1, seed=0, max_output_tokens=32, context_window=12288, hypothesis_budget=1))
        parsed = json.loads(generated.text)
        if parsed.get("ok") is not True or generated.completion_tokens <= 0:
            raise RuntimeError(f"unexpected completion: {generated.text!r}")
        queue.put({"worker_id": worker_id, "physical_gpu_id": worker_id, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "logical_device": "cuda:0", "status": "SUCCESS", "load_seconds": load_seconds, "generation_seconds": generated.elapsed_seconds, "prompt_tokens": generated.prompt_tokens, "completion_tokens": generated.completion_tokens, "gpu_before": before, "gpu_after_load": after_load, "gpu_after_generate": gpu_observation(worker_id)})
    except Exception as exc:
        queue.put({"worker_id": worker_id, "physical_gpu_id": worker_id, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"), "logical_device": "cuda:0", "status": "FAILED", "failure_type": type(exc).__name__, "failure": str(exc)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    context = mp.get_context("spawn")
    queue = context.Queue()
    processes = [context.Process(target=worker, args=(worker_id, str(args.model_path), queue), daemon=False) for worker_id in range(4)]
    for process in processes:
        process.start()
    for process in processes:
        process.join()
    reports = []
    for _ in processes:
        try:
            reports.append(queue.get(timeout=10))
        except Empty:
            reports.append({"status": "FAILED", "failure_type": "MissingWorkerReport"})
    reports.sort(key=lambda item: int(item.get("worker_id", 99)))
    mapping_ok = all(item.get("status") == "SUCCESS" and item.get("worker_id") == item.get("physical_gpu_id") and item.get("cuda_visible_devices") == str(item.get("worker_id")) and item.get("logical_device") == "cuda:0" for item in reports)
    result = {"experiment_id": "LLM_PROGRAM_SYNTHESIS_V2", "stage": "QWEN3_8B_FOUR_GPU_FOUR_WORKER_PREFLIGHT", "protocol": "four independent offline model loads and bounded prompts; no ARC task or solution opened", "status": "SUCCESS" if mapping_ok else "FAILED", "worker_to_gpu_mapping": {str(item.get("worker_id")): item.get("physical_gpu_id") for item in reports if item.get("worker_id") is not None}, "workers": reports}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "worker_to_gpu_mapping": result["worker_to_gpu_mapping"]}), flush=True)
    if result["status"] != "SUCCESS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
