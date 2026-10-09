"""Freeze the Director-authorized target-blind Beam-4 candidate-recall package."""
from __future__ import annotations
import hashlib, json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/paired_target_blind_beam4_candidate_recall_v1"
SOURCE_MANIFEST = ROOT / "experiments/capability_repair_baseline_v1/paired_final_query_loss_decomposition_v1/TARGET_DEV_60_FINAL_QUERY_MANIFEST.json"
DEV = ROOT / "experiments/targeted_capability_repair_v1/data/TARGET_DEV.jsonl"

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main() -> None:
    manifest = json.loads(SOURCE_MANIFEST.read_text(encoding="utf-8"))
    members = [member for batch in manifest["batches"] for member in batch["members"]]
    families = tuple(manifest["families"])
    if len(members) != 60 or len({m["episode_id"] for m in members}) != 60:
        raise RuntimeError("FROZEN_60_COHORT_FAIL")
    counts = {family: sum(m["family"] == family for m in members) for family in families}
    if counts != {family: 12 for family in families}:
        raise RuntimeError("FAMILY_DENOMINATOR_FAIL")
    OUT.mkdir(parents=True, exist_ok=True)
    package = {
        "schema_version": 1,
        "protocol_id": "PAIRED_TARGET_BLIND_BEAM4_CANDIDATE_RECALL_V1",
        "director_response_path": "orchestration/director/responses/GOVERNOR_REVIEW_CPU_ONLY_PAIRED_GENERATION_ERROR_MORPHOLOGY_V1_RESULT_RESPONSE.json",
        "director_response_sha256": sha(ROOT / "orchestration/director/responses/GOVERNOR_REVIEW_CPU_ONLY_PAIRED_GENERATION_ERROR_MORPHOLOGY_V1_RESULT_RESPONSE.json"),
        "cap_seconds": 3600,
        "cohort": {"rows": 60, "families": families, "source_manifest_path": str(SOURCE_MANIFEST.relative_to(ROOT)).replace("\\", "/"), "source_manifest_sha256": sha(SOURCE_MANIFEST), "family_counts": counts},
        "inputs": {"target_dev_path": str(DEV.relative_to(ROOT)).replace("\\", "/"), "target_dev_sha256": sha(DEV),
                   "conditions": {"CAPABILITY_REPAIR_BASELINE_V1_V7": {"manifest": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json", "manifest_sha256": sha(ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json")},
                                  "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001_FINAL": {"manifest": "experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json", "manifest_sha256": sha(ROOT / "experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json")}}},
        "decode": {"batch_size": 1, "num_beams": 4, "num_return_sequences": 4, "do_sample": False, "use_cache": True, "max_new_tokens": 512, "eos_token_id": 15, "pad_token_id": 13, "attention_backend": "sdpa", "early_stopping": True},
        "candidate_freeze": {"target_blind": True, "required_fields": ["episode_id", "family", "checkpoint_id", "prompt_sha256", "observation_sha256", "rank", "sequence_score", "generated_token_ids", "generated_sha256", "parser_status", "canonical_prediction_sha256"], "before_scoring": True},
        "scoring": {"target_access": "CPU_POSTPROCESS_ONLY_AFTER_CANDIDATE_FREEZE", "primary_estimand": "family_equal_mean(FB_minus_V7_oracle_recall_at_4)", "bootstrap_replicates": 10000, "bootstrap_seed": 20261009, "stratified_within_family": True},
        "forbidden": ["training", "optimizer", "backward", "checkpoint_mutation", "Gold", "dGold", "FINAL_AUDIT", "target_use_during_generation", "adaptive_decode_extension"],
        "source_sha256": {"worker": sha(ROOT / "scripts/run_paired_target_blind_beam4_candidate_recall_v1.py"), "analyzer": sha(ROOT / "scripts/analyze_paired_target_blind_beam4_candidate_recall_v1.py"), "runtime_identity_verifier": sha(ROOT / "scripts/verify_paired_fixed_b32_runtime_identity.py")}
    }
    path = OUT / "EXECUTION_PACKAGE.json"
    path.write_text(json.dumps(package, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "FROZEN", "package_sha256": sha(path), "cohort_rows": len(members), "family_counts": counts}, sort_keys=True))

if __name__ == "__main__":
    main()
