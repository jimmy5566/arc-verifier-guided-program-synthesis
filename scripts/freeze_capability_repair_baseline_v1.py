"""Freeze the forward-only baseline for capability repair; performs no inference."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "capability_repair_baseline_v1"
RECEIPT = ROOT / ".arc2-local" / "orchestration" / "receipt_cache" / "ROUND_RECONSTRUCTED_FOUNDATION_V2_V2_009" / "RECONSTRUCTED_FOUNDATION_V2_V2_009_TERMINAL_RECEIPT.json"
GATE = ROOT / "experiments" / "foundation_v2_reconstruction_and_targeted_repair_v2" / "round_009_minimum_safe_launch" / "RECONSTRUCTED_FOUNDATION_V2_V2_009_EQUIVALENCE_GATE_RESULT.json"
TARGET = ROOT / "experiments" / "targeted_capability_repair_v1" / "data" / "TARGET_DEV.jsonl"
RETENTION = ROOT / "experiments" / "targeted_capability_repair_v1" / "data" / "RETENTION_SENTINEL.jsonl"
FINAL = ROOT / "experiments" / "targeted_capability_repair_v1" / "data" / "FINAL_AUDIT_SEALED.jsonl"

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True)+"\n", encoding="utf-8", newline="\n")

def main() -> int:
    receipt, gate = json.loads(RECEIPT.read_text(encoding="utf-8")), json.loads(GATE.read_text(encoding="utf-8"))
    if receipt.get("status") != "SUCCESS" or receipt.get("optimizer_steps", 0) <= 0 or receipt.get("processed_tokens", 0) <= 0 or receipt.get("final_audit_accessed") is not False:
        raise RuntimeError("RECONSTRUCTED_CHECKPOINT_NOT_TECHNICALLY_VALID")
    selected = gate["deterministic_teacher_forced_selection"]
    used = float(receipt["scientific_gpu_training_seconds"]); cap = 28800.0
    if used < 0 or used > cap: raise RuntimeError("GPU_BUDGET_INVALID")
    model = {
        "model_id": "CAPABILITY_REPAIR_BASELINE_V1", "claim": "RECONSTRUCTED_FROM_FROZEN_RECIPE_NEW_IDENTITY",
        "not_claimed": ["EXACT_HISTORICAL_FOUNDATION_V2", "BITWISE_HISTORICAL_ADAPTER", "NODE_LEVEL_REPRODUCTION"],
        "checkpoint_path": selected["checkpoint_path"], "checkpoint_selection_source_sha256": selected["source_sha256"],
        "terminal_receipt": {"path": str(RECEIPT.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(RECEIPT)},
        "optimizer_steps": receipt["optimizer_steps"], "processed_tokens": receipt["processed_tokens"],
        "scientific_gpu_training_seconds": used, "final_audit_accessed": False,
    }
    data = {name: {"path": str(path.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(path)} for name, path in {"TARGET_DEV": TARGET, "RETENTION_SENTINEL": RETENTION, "FINAL_AUDIT_SEALED": FINAL}.items()}
    policy = {"STRICT_HISTORICAL_FOUNDATION_V2_EQUIVALENCE_REQUIRED": False, "reproducibility_level": "CONFIGURATION_LEVEL", "historical_foundation_v2": "REFERENCE_ONLY", "baseline_profile_execution": "PENDING", "final_audit_accessed": False, "no_eval60_gold_training": True, "no_diagnostic_gold_training": True}
    budget = {"cap_seconds": cap, "historical_reconstruction_seconds": used, "remaining_targeted_repair_seconds": cap-used, "accounting": "append_only_actual_optimizer_training_seconds"}
    write(OUT / "CAPABILITY_REPAIR_BASELINE_V1_IDENTITY.json", {"schema_version": 1, **model, "dataset_identities": data, "policy": policy, "budget": budget, "gate_reference": {"path": str(GATE.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(GATE), "historical_equivalence_outcome": gate["outcome"]}})
    for name, kind in (("CAPABILITY_REPAIR_BASELINE_V1_PROFILE.json", "TARGETED_ATOMIC_DEV"), ("CAPABILITY_REPAIR_BASELINE_V1_COMPOSITION.json", "TARGETED_COMPOSITION_DEV"), ("CAPABILITY_REPAIR_BASELINE_V1_RETENTION.json", "PROTECTED_RETENTION_SENTINEL")):
        write(OUT / name, {"schema_version": 1, "baseline_model_id": "CAPABILITY_REPAIR_BASELINE_V1", "surface": kind, "status": "PENDING_FORWARD_BASELINE_EVALUATION", "model_loaded": False, "final_audit_accessed": False, "not_historical_equivalence_evidence": True})
    write(OUT / "FORWARD_CAPABILITY_REPAIR_OBJECTIVE_V1.json", {"schema_version": 1, "objective": "TRAIN_VALIDATE_DIAGNOSE_ADAPT_CURRICULUM_REPEAT", "baseline": "CAPABILITY_REPAIR_BASELINE_V1", "historical_equivalence_required": False, "priority_atomic": ["connected components", "orientation", "inside contains", "same color", "relation primitives", "selector prerequisites"], "priority_composition": ["RELATION_TO_SELECTOR_ACTION", "SELECTOR_TO_ACTION", "MASK_SET_TO_CONSTRUCTION", "CONDITIONAL_TO_ACTION", "COUNTING_TO_CONSTRUCTION"], "acceptance": "paired_target_delta_plus_composition_delta_plus_retention", "budget": budget, "next_action": "RUN_FORWARD_BASELINE_TARGETED_DEV_AND_RETENTION"})
    print(json.dumps({"status": "FROZEN_PENDING_FORWARD_BASELINE_EVALUATION", "remaining_seconds": cap-used, "identity_sha256": sha(OUT / "CAPABILITY_REPAIR_BASELINE_V1_IDENTITY.json")}, sort_keys=True))
    return 0
if __name__ == "__main__": raise SystemExit(main())
