"""One-time retrospective scoring for frozen untouched60 A/B predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wilson(successes: int, total: int, z: float = 1.959963984540054) -> list[float]:
    if total == 0:
        return [0.0, 0.0]
    p = successes / total; denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    radius = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return [center - radius, center + radius]


def _exact_sign_test(a_only: int, b_only: int) -> float:
    discordant, lower = a_only + b_only, min(a_only, b_only)
    if discordant == 0:
        return 1.0
    probability = sum(math.comb(discordant, index) for index in range(lower + 1)) / 2**discordant
    return min(1.0, 2 * probability)


def _validate_predictions(value: dict[str, Any], task_ids: list[str], method: str) -> dict[str, Any]:
    if value.get("status") != "PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or value.get("method") != method:
        raise ValueError(f"{method}: prediction freeze status mismatch")
    records = value.get("records")
    if list(value.get("task_ids", ())) != task_ids or not isinstance(records, dict) or set(records) != set(task_ids):
        raise ValueError(f"{method}: prediction artifact does not match immutable manifest")
    for task_id in task_ids:
        record = records[task_id]
        if record.get("attempt_1") is None or int(record.get("attempt_count", 0)) not in {1, 2}:
            raise ValueError(f"{method}/{task_id}: invalid frozen attempts")
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("manifest", "a_predictions", "b_predictions", "a_candidates", "solutions_path", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite the one-time score report")
    manifest, a_prediction, b_prediction, candidates = (_read(path) for path in (args.manifest, args.a_predictions, args.b_predictions, args.a_candidates))
    task_ids = list(manifest.get("task_ids", ()))
    if manifest.get("status") != "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS" or len(task_ids) != 60:
        raise ValueError("requires the frozen untouched60 manifest")
    # This complete structural gate is deliberately before the only target
    # boundary below.
    a_records, b_records = _validate_predictions(a_prediction, task_ids, "A"), _validate_predictions(b_prediction, task_ids, "B")
    candidate_records = candidates.get("records")
    if candidates.get("status") != "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING" or not isinstance(candidate_records, dict) or set(candidate_records) != set(task_ids):
        raise ValueError("A candidate artifact is not complete/frozen for untouched60")
    if a_prediction.get("source_candidate_artifact_sha256") != hashlib.sha256(args.a_candidates.read_bytes()).hexdigest():
        raise ValueError("A predictions are not tied to the scored candidate artifact")
    # Sole solution/target boundary.
    from arc.io import load_solutions
    solutions = load_solutions(args.solutions_path)
    if any(task_id not in solutions for task_id in task_ids):
        raise ValueError("a frozen task lacks a target solution")
    metrics: dict[str, Any] = {}
    outcomes: dict[str, dict[str, bool]] = {"A": {}, "B": {}}
    for method, records in (("A", a_records), ("B", b_records)):
        top1 = two = any_of_k = duplicate = 0
        for task_id in task_ids:
            expected, record = solutions[task_id], records[task_id]
            first, second = record["attempt_1"], record.get("attempt_2")
            outcomes[method][task_id] = first == expected or (second is not None and second == expected)
            top1 += int(first == expected)
            two += int(outcomes[method][task_id])
            any_of_k += int(any(item.get("prediction") == expected for item in candidate_records[task_id].get("candidates", ())))
            duplicate += int(bool(record.get("duplicate_attempt")))
        metrics[method] = {"n": len(task_ids), "top1_exact": top1, "two_attempt_exact": two, "any_of_k_diagnostic": any_of_k, "top1_accuracy": top1 / len(task_ids), "two_attempt_accuracy": two / len(task_ids), "two_attempt_wilson_95": _wilson(two, len(task_ids)), "duplicate_attempt_count": duplicate, "duplicate_attempt_rate": duplicate / len(task_ids)}
    a_only = sorted(task_id for task_id in task_ids if outcomes["A"][task_id] and not outcomes["B"][task_id])
    b_only = sorted(task_id for task_id in task_ids if outcomes["B"][task_id] and not outcomes["A"][task_id])
    both = sorted(task_id for task_id in task_ids if outcomes["A"][task_id] and outcomes["B"][task_id])
    neither = sorted(task_id for task_id in task_ids if not outcomes["A"][task_id] and not outcomes["B"][task_id])
    difference = metrics["B"]["two_attempt_exact"] - metrics["A"]["two_attempt_exact"]
    decision = "PASS_SUBMIT_B_UNCHANGED" if difference >= 0 else ("INCONCLUSIVE_ONE_TASK_DOWN_REVIEW_PAIRED_EVIDENCE" if difference == -1 else "FAIL_USE_A_B_IS_TWO_OR_MORE_TASKS_WORSE")
    result = {
        "experiment_id": "ARC2_UNTOUCHED60_NATIVE_PUBLIC_REFERENCE_B_V1",
        "status": "SCORED_ONCE_AFTER_A_B_PREDICTIONS_FROZEN",
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "prediction_sha256": {"A": hashlib.sha256(args.a_predictions.read_bytes()).hexdigest(), "B": hashlib.sha256(args.b_predictions.read_bytes()).hexdigest()},
        "methods": metrics,
        "paired_two_attempt": {"a_only_solves": a_only, "b_only_solves": b_only, "both_solve": both, "neither_solve": neither, "net_gain_b_vs_a": difference, "discordant_count": len(a_only) + len(b_only), "exact_two_sided_binomial_p": _exact_sign_test(len(a_only), len(b_only))},
        "public_lb_decision": decision,
        "integrity": {"solutions_loaded_only_after_a_b_prediction_structure_complete": True, "method_a_b_modified_after_manifest": False, "any_of_k_is_diagnostic_only": True, "no_target_aware_attempt_selection": True},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"A": metrics["A"], "B": metrics["B"], "net_gain_b_vs_a": difference, "decision": decision}, sort_keys=True))


if __name__ == "__main__":
    main()
