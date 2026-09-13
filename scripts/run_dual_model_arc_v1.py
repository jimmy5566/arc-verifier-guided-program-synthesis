"""Two-task, train-only dual-expert ARC smoke: reused Native plus SOAR-7B."""
from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.dual_reasoning_smoke import execute_program, extract_program, soar_prompt, unload_model, verify_program


TASK_IDS = ("d4c90558", "67a423a3")
PROGRAM_BUDGET = 2


def gpu_memory_mb() -> dict[str, int]:
    raw = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True
    )
    return {line.split(",", 1)[0].strip(): int(line.split(",", 1)[1].strip()) for line in raw.splitlines()}


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def native_train_reliable(record: dict[str, Any]) -> bool:
    """The immutable Native artifact's LOO metric is train-only evidence."""
    total = int(record.get("loo_total", 0))
    return total > 0 and int(record.get("loo_successes", 0)) == total


def bucket(native_ok: bool, induction_ok: bool) -> str:
    return (
        "BOTH_GOOD" if native_ok and induction_ok else "NATIVE_ONLY" if native_ok
        else "INDUCTION_ONLY" if induction_ok else "NEITHER"
    )


def discover_soar7(input_root: Path) -> Path:
    candidates: list[Path] = []
    for config in input_root.rglob("config.json"):
        marker = str(config.parent).lower()
        has_weights = any(config.parent.rglob("*.safetensors")) or any(config.parent.rglob("pytorch_model*.bin"))
        if "soar" in marker and "7b" in marker and has_weights:
            candidates.append(config.parent)
    if len(candidates) != 1:
        raise RuntimeError(f"expected one SOAR-7B checkpoint; found={[str(item) for item in candidates]}")
    return candidates[0]


def generate_program(model: Any, tokenizer: Any, prompt: str, *, seed: int) -> str:
    import torch

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if getattr(tokenizer, "chat_template", None):
        encoded = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], add_generation_prompt=True,
            tokenize=True, return_tensors="pt", return_dict=True,
        )
    else:
        encoded = tokenizer(prompt, return_tensors="pt")
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(
            **encoded, max_new_tokens=768, do_sample=True, temperature=0.2,
            top_p=0.95, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    text = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:].detach().cpu(), skip_special_tokens=True)
    del encoded, output
    return text


def induction_records(*, tasks: dict[str, Any], model_path: Path) -> tuple[dict[str, Any], dict[str, int], dict[str, int]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    before = gpu_memory_mb()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), local_files_only=True, torch_dtype=torch.bfloat16,
        device_map="auto", low_cpu_mem_usage=True,
    )
    records: dict[str, Any] = {}
    for task_index, task_id in enumerate(TASK_IDS):
        task = tasks[task_id]
        pairs = [(example.input.to_list(), example.output.to_list()) for example in task.train]
        started = perf_counter()
        candidates: list[dict[str, Any]] = []
        for candidate_index in range(PROGRAM_BUDGET):
            raw = generate_program(model, tokenizer, soar_prompt(pairs), seed=20260914 + task_index * PROGRAM_BUDGET + candidate_index)
            program = extract_program(raw)
            verification = verify_program(program, pairs) if program else {
                "program_valid": False, "train_pass_count": 0, "train_pair_count": len(pairs),
                "all_train_exact": False, "train_execution": [],
            }
            candidates.append({"candidate_index": candidate_index, "raw_model_output": raw, "extracted_code": program, "verification": verification})
        passing = next((candidate for candidate in candidates if candidate["verification"]["all_train_exact"]), None)
        test_execution = execute_program(passing["extracted_code"], task.test[0].input.to_list()) if passing else {
            "ok": False, "status": "NO_TRAIN_EXACT_PROGRAM"
        }
        records[task_id] = {
            "candidate_programs": candidates,
            "train_exact_pass": passing is not None,
            "frozen_passing_program": passing["extracted_code"] if passing else None,
            "test_prediction_execution": test_execution,
            "runtime_seconds": perf_counter() - started,
        }
    peak = gpu_memory_mb()
    del model, tokenizer
    gc.collect()
    after = unload_model(None)
    return records, before, {"peak": peak, "after_unload": after}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--native-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    native = json.loads(args.native_artifact.read_text(encoding="utf-8"))
    if native.get("status") != "BRANCH_PREDICTIONS_FROZEN" or tuple(native.get("task_ids", ())) != TASK_IDS:
        raise RuntimeError("native artifact is not the required frozen two-task result")
    tasks = load_dataset(args.challenge_path)
    print(json.dumps({"event": "NATIVE_REUSED", "artifact": str(args.native_artifact), "gpu_memory": gpu_memory_mb()}))
    write_json(args.output_root / "native.json", native)
    model_path = discover_soar7(args.input_root)
    load_started = perf_counter()
    records, before, lifecycle = induction_records(tasks=tasks, model_path=model_path)
    induction = {
        "status": "INDUCTION_PREDICTIONS_FROZEN", "task_ids": list(TASK_IDS),
        "model_path": str(model_path), "loader_backend": "transformers",
        "dtype": "torch.bfloat16", "program_budget": PROGRAM_BUDGET,
        "records": records, "lifecycle": {"before_load": before, **lifecycle},
        "wall_seconds": perf_counter() - load_started,
    }
    write_json(args.output_root / "induction.json", induction)
    unified_tasks: list[dict[str, Any]] = []
    for task_id in TASK_IDS:
        native_record = native["records"][task_id]
        induction_record = records[task_id]
        native_ok = native_train_reliable(native_record)
        induction_ok = bool(induction_record["train_exact_pass"])
        unified_tasks.append({
            "task_id": task_id, "native": native_record, "induction": induction_record,
            "native_train_reliable": native_ok, "induction_train_exact": induction_ok,
            "bucket": bucket(native_ok, induction_ok), "errors": [],
        })
    unified = {"status": "PREDICTIONS_FROZEN_TRAIN_ONLY_BUCKETS", "tasks": unified_tasks}
    summary = {name: sum(item["bucket"] == name for item in unified_tasks) for name in ("BOTH_GOOD", "NATIVE_ONLY", "INDUCTION_ONLY", "NEITHER")}
    runtime = {
        "native": {"reused": True, "historical_record_seconds": sum(float(native["records"][task_id]["runtime_seconds"]) for task_id in TASK_IDS)},
        "induction": {"wall_seconds": induction["wall_seconds"], "per_task_seconds": {task_id: records[task_id]["runtime_seconds"] for task_id in TASK_IDS}},
    }
    write_json(args.output_root / "unified.json", unified)
    write_json(args.output_root / "bucket_summary.json", summary)
    write_json(args.output_root / "runtime_summary.json", runtime)
    print(json.dumps({"event": "DUAL_V1_FROZEN", "buckets": summary, "output_root": str(args.output_root)}))


if __name__ == "__main__":
    main()
