"""Re-run only train verification of frozen SOAR candidates under a new sandbox ABI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter, time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from arc.io import load_dataset
from inference.dual_reasoning_smoke import verify_program
from inference.soar_refinement import deterministic_feedback


def atomic(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--timeout", type=float, required=True)
    args = parser.parse_args()
    started = perf_counter()
    tasks = load_dataset(args.challenge_path)
    rows = []
    for path in sorted((args.artifact_root / "tasks").glob("*.json")):
        state = json.loads(path.read_text(encoding="utf-8"))
        task_id = state["task_id"]
        pairs = [(example.input.to_list(), example.output.to_list()) for example in tasks[task_id].train]
        before = sum(bool(candidate.get("all_train_exact")) for candidate in state["candidate_programs"])
        for candidate in state["candidate_programs"]:
            code = candidate.get("extracted_code")
            candidate["verification"] = verify_program(code, pairs, timeout_seconds=args.timeout) if code else {
                "all_train_exact": False, "train_pass_count": 0, "train_pair_count": len(pairs), "train_execution": [],
            }
            candidate["all_train_exact"] = bool(candidate["verification"]["all_train_exact"])
            candidate["feedback"] = deterministic_feedback(candidate)
        after = sum(bool(candidate.get("all_train_exact")) for candidate in state["candidate_programs"])
        state["verification_completed_epoch"] = time()
        state["verification_abi"] = "safe_allowlisted_helpers_stdlib_v2"
        atomic(path, state)
        rows.append({"task_id": task_id, "candidate_count": len(state["candidate_programs"]), "train_exact_before": before, "train_exact_after": after, "best_train_pass": max(candidate["verification"]["train_pass_count"] for candidate in state["candidate_programs"])})
    summary = {"status": "TRAIN_ONLY_REVERIFICATION_COMPLETE", "sandbox_abi": "safe_allowlisted_helpers_stdlib_v2", "tasks": rows, "wall_seconds": perf_counter() - started, "leakage_audit": "No solutions file is accepted or read; only challenge train pairs are used."}
    atomic(args.artifact_root / "reverification_summary.json", summary)
    print(json.dumps({"event": "SOAR_TRAIN_ONLY_REVERIFICATION_COMPLETE", "tasks": rows, "wall_seconds": summary["wall_seconds"]}))


if __name__ == "__main__":
    main()
