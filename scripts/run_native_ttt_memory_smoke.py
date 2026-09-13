"""Single-GPU, no-candidate diagnostic for native per-task LoRA memory."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", action="append", required=True)
    parser.add_argument("--steps", action="append", type=int, required=True)
    parser.add_argument("--augmentations", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--max-peak-gib", type=float, default=21.0)
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite memory-smoke artifact")
    if len(args.task_id) != len(args.steps) or any(steps <= 0 for steps in args.steps) or args.augmentations <= 0 or args.repeats <= 0:
        raise ValueError("task IDs/steps must align and all counts must be positive")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    import torch
    from arc.io import load_dataset
    from inference.nvarc_native import NVARCNativeProvider
    from inference.nvarc_native_augmentation import bounded_native_augmentations
    from inference.nvarc_native_ttt import NativeLoRAConfig, NativeTaskLoRA

    tasks = load_dataset(args.challenge_path)
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device="cuda:0")
    provider.load()
    all_augmentations = bounded_native_augmentations()
    augmentations = tuple(item for item in all_augmentations if item.pair_order == "canonical")[:args.augmentations]
    if len(augmentations) != args.augmentations: raise RuntimeError("insufficient canonical augmentations")
    adapter = NativeTaskLoRA(provider.model, NativeLoRAConfig(steps=max(args.steps)))
    before = adapter._memory_snapshot(torch)
    cases = []
    for task_id, steps in zip(args.task_id, args.steps, strict=True):
        if task_id not in tasks: raise KeyError(f"unknown task ID: {task_id}")
        runs = []
        for repeat in range(args.repeats):
            metrics = adapter.fit_task(provider, tasks[task_id], augmentations=augmentations, context_window=8192, steps=steps)
            after_fit = adapter._memory_snapshot(torch)
            adapter.finish_task()
            after_reset = adapter._memory_snapshot(torch)
            runs.append({"repeat": repeat + 1, "metrics": metrics, "after_fit": after_fit, "after_reset": after_reset})
            print(json.dumps({"event": "TTT_MEMORY_SMOKE_HEARTBEAT", "task_id": task_id, "repeat": repeat + 1, "steps": steps, "peak_allocated_bytes": metrics["memory_trace"][-1]["after_optimizer"]["max_allocated_bytes"], "loss": metrics["last_loss"], "adapter_updated": metrics["adapter_updated"]}, sort_keys=True), flush=True)
        cases.append({"task_id": task_id, "steps": steps, "runs": runs})
    peak = max(run["metrics"]["memory_trace"][-1]["after_optimizer"]["max_allocated_bytes"] for case in cases for run in case["runs"])
    limit = int(args.max_peak_gib * 1024**3)
    criteria = {
        "memory_safe": peak < limit,
        "finite_loss": all(run["metrics"]["loss_finite"] for case in cases for run in case["runs"]),
        "adapter_updated": all(run["metrics"]["adapter_updated"] for case in cases for run in case["runs"]),
        "base_model_unchanged": all(run["metrics"]["base_model_unchanged"] for case in cases for run in case["runs"]),
        "deterministic_reset": all(len({round(run["metrics"]["last_loss"], 7) for run in case["runs"]}) == 1 for case in cases),
    }
    passed = all(criteria.values())
    output = {"status": "TTT_MEMORY_SMOKE_COMPLETE" if passed else "TTT_MEMORY_SMOKE_FAILED", "protocol": "single GPU; train pairs only; no test candidate generation or solution access", "augmentation_count": len(augmentations), "repeats": args.repeats, "provider": provider.load_metadata, "before": before, "peak_allocated_bytes": peak, "peak_limit_bytes": limit, "criteria": criteria, "cases": cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"event": "TTT_MEMORY_SMOKE_CRITERIA", "status": output["status"], "task_ids": args.task_id, "steps": args.steps, "repeats": args.repeats, "peak_allocated_bytes": peak, **criteria}, sort_keys=True))
    if not passed: raise RuntimeError("native TTT memory smoke did not meet safety criteria")


if __name__ == "__main__": main()
