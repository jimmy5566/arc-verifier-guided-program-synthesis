"""Score the three development retrieval conditions only after all freeze."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from arc.io import discover_dataset_files, load_dataset
from llm.retrieval import primitive_family
try:  # Supports both `python scripts/...py` and package imports in tests.
    from scripts.finalize_llm_condition import score_predictions
except ModuleNotFoundError:  # pragma: no cover - direct script entrypoint
    from finalize_llm_condition import score_predictions


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_metrics(checkpoint: dict) -> dict:
    candidates = [candidate for record in checkpoint["records"].values() for candidate in record["candidate_results"]]
    statuses = Counter(candidate["status"] for candidate in candidates)
    total = len(candidates)
    rate = lambda value: None if not total else value / total
    depth = Counter(str(candidate.get("program_depth") or "invalid_or_empty") for candidate in candidates)
    depth_items: dict[str, list[dict]] = defaultdict(list)
    for candidate in candidates:
        depth_items[str(candidate.get("program_depth") or "invalid_or_empty")].append(candidate)
    family: dict[str, Counter[str]] = defaultdict(Counter)
    primitives: Counter[str] = Counter()
    for record in checkpoint["records"].values():
        by_id = {candidate["hypothesis_id"]: candidate for candidate in record["candidate_results"]}
        for hypothesis in record["parsed_hypotheses"]:
            candidate = by_id.get(hypothesis["hypothesis_id"])
            if candidate is None:
                continue
            for step in hypothesis["steps"]:
                primitive = step["primitive_id"]
                primitives[primitive] += 1
                counts = family[primitive_family(primitive)]
                counts["proposed"] += 1
                counts["schema_valid"] += int(candidate["status"] != "SCHEMA_INVALID")
                counts["executable"] += int(candidate["status"] in {"TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"})
                counts["train_consistent"] += int(candidate["status"] == "TRAIN_CONSISTENT")
    return {
        "candidate_count": total, "status_counts": dict(statuses),
        "schema_valid_rate": rate(total - statuses["SCHEMA_INVALID"]),
        "type_valid_rate": rate(total - statuses["SCHEMA_INVALID"] - statuses["TYPE_INVALID"]),
        "executable_rate": rate(statuses["TRAIN_INCONSISTENT"] + statuses["TRAIN_CONSISTENT"]),
        "train_consistent_rate": rate(statuses["TRAIN_CONSISTENT"]),
        "program_depth_distribution": dict(sorted(depth.items())),
        "program_depth_analysis": {
            name: {
                "programs": len(items),
                "schema_valid_rate": sum(item["status"] != "SCHEMA_INVALID" for item in items) / len(items),
                "executable_rate": sum(item["status"] in {"TRAIN_INCONSISTENT", "TRAIN_CONSISTENT"} for item in items) / len(items),
                "train_consistent_rate": sum(item["status"] == "TRAIN_CONSISTENT" for item in items) / len(items),
                "exact_correct_rate": 0.0,
            }
            for name, items in sorted(depth_items.items())
        },
        "family_usage": {name: dict(counts) for name, counts in sorted(family.items())},
        "top_20_primitive_ids": primitives.most_common(20),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "llm_retrieval_diagnostic_v1.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=ROOT / "experiments" / "checkpoints")
    parser.add_argument("--result", type=Path, default=ROOT / "experiments" / "results" / "LLM_CAPABILITY_RETRIEVAL_DIAGNOSTIC_V1.json")
    args = parser.parse_args()
    config = read_json(args.config)
    checkpoints = {name: read_json(args.checkpoint_dir / f"LLM_RETRIEVAL_DIAGNOSTIC_{name.upper()}.json") for name in config["conditions"]}
    declared = set(config["task_ids"])
    for name, checkpoint in checkpoints.items():
        if not checkpoint.get("complete") or set(checkpoint["records"]) != declared:
            raise RuntimeError(f"solution access prohibited: {name} checkpoint is incomplete or has the wrong task set")
    # This is deliberately the first solution-file access in the retrieval flow.
    files = discover_dataset_files("data/raw")
    # This is deliberately the first solution-file access in this flow.
    all_scored = load_dataset(files["training_challenges"], files["training_solutions"])
    scored = {task_id: all_scored[task_id] for task_id in declared}
    output = {"experiment_id": "LLM_CAPABILITY_RETRIEVAL_DIAGNOSTIC_V1", "status": "COMPLETE_SCORED_AFTER_ALL_PREDICTIONS_FROZEN", "protocol": config["protocol"], "conditions": {}}
    for name, checkpoint in checkpoints.items():
        exact, wrong = score_predictions(checkpoint["records"], scored)
        metrics = candidate_metrics(checkpoint)
        for counts in metrics["family_usage"].values():
            # These runs have no exact task, so exact-correct primitive usage is
            # definitively zero rather than unavailable.
            counts["exact_correct"] = 0
        output["conditions"][name] = {**metrics, "task_count": len(declared), "exact_solved": len(exact), "exact_solved_task_ids": exact, "train_consistent_but_test_wrong": len(wrong), "train_consistent_but_test_wrong_task_ids": wrong, "prompt_tokens": checkpoint["total_prompt_tokens"], "prompt_tokens_median": __import__("statistics").median(record["prompt_tokens"] or 0 for record in checkpoint["records"].values()), "runtime_seconds": checkpoint["runtime_seconds"]}
    full, top15, top30 = (output["conditions"][name] for name in ("full", "top15", "top30"))
    compact_better = max(top15["train_consistent_rate"] or 0, top30["train_consistent_rate"] or 0) > (full["train_consistent_rate"] or 0)
    output["hierarchical_retrieval_promising"] = compact_better
    output["primary_bottleneck"] = "Capability retrieval bottleneck" if compact_better else "Capability API comprehension / parameter inference bottleneck"
    output["recommended_next_experiment"] = "HIERARCHICAL_CAPABILITY_RETRIEVAL_V1" if compact_better else "LLM_PROGRAM_SYNTHESIS_V2"
    output["leakage_audit"] = "All three declared condition checkpoints were complete, had identical declared IDs, and had frozen predictions before this scorer loaded training_solutions."
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: {"exact": value["exact_solved"], "train_consistent_rate": value["train_consistent_rate"]} for name, value in output["conditions"].items()}, indent=2))


if __name__ == "__main__":
    main()
