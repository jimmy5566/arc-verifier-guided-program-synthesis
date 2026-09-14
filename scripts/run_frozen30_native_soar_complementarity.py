"""Frozen30 diagnostic: SOAR-7B induction versus immutable Native candidates.

Generation and train-only verification complete before this script opens the
cohort's test solutions.  This is a failure-decomposition study, never router
training or a deployable ranker.
"""
from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.dual_reasoning_smoke import extract_program, soar_prompt, unload_model, verify_program, execute_program


EXPECTED_NATIVE_TOP1 = 9
EXPECTED_NATIVE_ANY = 21
PROGRAM_BUDGET = 8
SEED_BASE = 20260914


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def gpu_memory_mb() -> dict[str, int]:
    import subprocess
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        return {line.split(",", 1)[0].strip(): int(line.split(",", 1)[1].strip()) for line in raw.splitlines()}
    except Exception:
        return {}


def generate_program(model: Any, tokenizer: Any, prompt: str, *, seed: int) -> str:
    import torch
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    encoded = (tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
               if getattr(tokenizer, "chat_template", None) else tokenizer(prompt, return_tensors="pt"))
    device = next(model.parameters()).device
    encoded = {name: value.to(device) for name, value in encoded.items()}
    with torch.inference_mode():
        output = model.generate(**encoded, max_new_tokens=768, do_sample=True, temperature=0.2, top_p=0.95,
                                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(output[0, encoded["input_ids"].shape[-1]:].detach().cpu(), skip_special_tokens=True)
    del encoded, output
    return text


def discover_soar7(input_root: Path) -> Path:
    candidates = [config.parent for config in input_root.rglob("config.json") if "soar" in str(config.parent).lower() and "7b" in str(config.parent).lower() and any(config.parent.rglob("*.safetensors"))]
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one SOAR-7B BF16 checkpoint; found={[str(item) for item in candidates]}")
    return candidates[0]


def native_labels(native: dict[str, Any], solutions: dict[str, Any]) -> dict[str, dict[str, Any]]:
    labels: dict[str, dict[str, Any]] = {}
    for task_id, record in native["records"].items():
        target = solutions[task_id]
        top1 = record["ranked_prediction"] == target
        any_hit = any(candidate.get("prediction") == target for candidate in record["candidates"])
        failure_class = "TOP1_SOLVED" if top1 else "SELECTION_MISS" if any_hit else "GENERATION_MISS"
        labels[task_id] = {"native_top1_correct": top1, "native_any_of_k_hit": any_hit, "native_failure_class": failure_class}
    if sum(item["native_top1_correct"] for item in labels.values()) != EXPECTED_NATIVE_TOP1 or sum(item["native_any_of_k_hit"] for item in labels.values()) != EXPECTED_NATIVE_ANY:
        raise ValueError("native frozen labels do not match the declared 9/21 baseline")
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--solutions-path", type=Path, required=True)
    parser.add_argument("--native-frozen", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    native = json.loads(args.native_frozen.read_text(encoding="utf-8"))
    records = native.get("records", {})
    if native.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or len(records) != 30:
        raise ValueError("requires the immutable 30-task Native prediction artifact")
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    timeout_seconds = float(calibration["timeout_policy"]["selected_timeout_seconds"])
    task_ids = tuple(records)
    tasks = load_dataset(args.challenge_path)
    if any(task_id not in tasks for task_id in task_ids):
        raise ValueError("a frozen Native task is missing from challenges")

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_path = discover_soar7(args.input_root)
    lifecycle = {"before_load": gpu_memory_mb()}
    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
    soar_records: dict[str, dict[str, Any]] = {}
    for task_index, task_id in enumerate(task_ids):
        task = tasks[task_id]
        pairs = [(example.input.to_list(), example.output.to_list()) for example in task.train]
        candidates: list[dict[str, Any]] = []
        for candidate_index in range(PROGRAM_BUDGET):
            raw = generate_program(model, tokenizer, soar_prompt(pairs), seed=SEED_BASE + task_index * PROGRAM_BUDGET + candidate_index)
            candidates.append({"candidate_index": candidate_index, "raw_model_output": raw, "extracted_code": extract_program(raw)})
        soar_records[task_id] = {"candidate_programs": candidates, "generation_frozen": True}
        print(json.dumps({"event": "SOAR_FROZEN30_GENERATED", "task_index": task_index + 1, "task_count": len(task_ids), "task_id": task_id, "program_count": PROGRAM_BUDGET}), flush=True)
    lifecycle["peak"] = gpu_memory_mb()
    del model, tokenizer; gc.collect(); lifecycle["after_unload"] = unload_model(None)
    print(json.dumps({"event": "SOAR_UNLOADED_BEFORE_SANDBOX", "gpu_memory": lifecycle["after_unload"]}), flush=True)

    # Phase B is strictly post-unload and uses the calibrated outer wall timeout.
    for task_index, task_id in enumerate(task_ids):
        task = tasks[task_id]
        pairs = [(example.input.to_list(), example.output.to_list()) for example in task.train]
        candidates = soar_records[task_id]["candidate_programs"]
        for candidate in candidates:
            program = candidate["extracted_code"]
            verification = verify_program(program, pairs, timeout_seconds=timeout_seconds) if program else {"parse_valid": False, "static_safe": False, "reason": "no_transform_code_extracted", "all_train_exact": False, "train_execution": []}
            executions = verification["train_execution"]
            candidate.update({
                "parse_valid": verification["parse_valid"], "static_safe": verification["static_safe"], "verification": verification,
                "all_train_exact": verification["all_train_exact"],
                "failure_reason": next((row.get("error") or row.get("reason") for row in executions if not row.get("ok")), verification.get("reason")),
            })
        # Predeclared train-only selector: lowest generation index among exact programs.
        passing = next((candidate for candidate in candidates if candidate["all_train_exact"]), None)
        test_execution = execute_program(passing["extracted_code"], task.test[0].input.to_list(), timeout_seconds=timeout_seconds) if passing else {"ok": False, "status": "NO_TRAIN_EXACT_PROGRAM"}
        soar_records[task_id].update({"train_exact_available": passing is not None, "selected_candidate_index": passing["candidate_index"] if passing else None, "selection_rule": "lowest_candidate_index_among_all_train_exact", "frozen_test_prediction": test_execution.get("grid") if test_execution.get("ok") else None, "test_execution": test_execution})
        print(json.dumps({"event": "SOAR_FROZEN30_VERIFIED", "task_index": task_index + 1, "task_count": len(task_ids), "task_id": task_id, "train_exact": passing is not None}), flush=True)

    # Oracle boundary: persist every SOAR prediction before opening targets.
    soar = {"status": "SOAR_FROZEN30_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING", "task_ids": list(task_ids), "model_path": str(model_path), "loader_backend": "transformers", "dtype": "torch.bfloat16", "program_budget": PROGRAM_BUDGET, "sampling": {"temperature": 0.2, "top_p": 0.95, "max_new_tokens": 768, "seed_base": SEED_BASE}, "sandbox_timeout_seconds": timeout_seconds, "lifecycle": lifecycle, "selection_rule": "lowest_candidate_index_among_all_train_exact", "records": soar_records, "runtime_seconds": perf_counter() - started}
    soar_path = args.output_root / "soar_frozen30.json"
    write_json(soar_path, soar)

    # Only now may retrospective frozen30 labels be derived from targets.
    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path)
    labels = native_labels(native, solutions)
    comparisons: list[dict[str, Any]] = []
    for task_id in task_ids:
        record = soar_records[task_id]; soar_exact = record["frozen_test_prediction"] == solutions[task_id] if record["frozen_test_prediction"] is not None else False
        native_label = labels[task_id]
        overlap = "BOTH_SOLVED" if native_label["native_top1_correct"] and soar_exact else "NATIVE_ONLY" if native_label["native_top1_correct"] else "SOAR_ONLY" if soar_exact else "NEITHER"
        comparisons.append({"task_id": task_id, **native_label, "soar_train_exact_available": record["train_exact_available"], "soar_final_test_exact": soar_exact, "overlap": overlap})
    overlap_summary = {name: sum(item["overlap"] == name for item in comparisons) for name in ("BOTH_SOLVED", "NATIVE_ONLY", "SOAR_ONLY", "NEITHER")}
    rescues = {failure: {"native_task_count": sum(item["native_failure_class"] == failure for item in comparisons), "soar_rescues": sum(item["native_failure_class"] == failure and item["soar_final_test_exact"] for item in comparisons)} for failure in ("SELECTION_MISS", "GENERATION_MISS")}
    comparison = {"status": "FROZEN30_RETROSPECTIVE_COMPARISON_COMPLETE", "diagnostic_only": True, "tasks": comparisons, "union_native_top1_or_soar": sum(item["native_top1_correct"] or item["soar_final_test_exact"] for item in comparisons), "leakage_audit": "SOAR generation, all train verification, deterministic selection, and test predictions were persisted in soar_frozen30.json before load_solutions."}
    runtime = {"soar_total_seconds": soar["runtime_seconds"], "model_lifecycle": lifecycle, "program_budget": PROGRAM_BUDGET, "sandbox_timeout_seconds": timeout_seconds}
    write_json(args.output_root / "native_soar_comparison.json", comparison)
    write_json(args.output_root / "overlap_summary.json", overlap_summary)
    write_json(args.output_root / "failure_rescue_matrix.json", rescues)
    write_json(args.output_root / "runtime_summary.json", runtime)
    print(json.dumps({"event": "FROZEN30_COMPLEMENTARITY_COMPLETE", "overlap": overlap_summary, "rescues": rescues, "output_root": str(args.output_root)}), flush=True)


if __name__ == "__main__":
    main()
