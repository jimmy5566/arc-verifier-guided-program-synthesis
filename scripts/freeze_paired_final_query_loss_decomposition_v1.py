"""Freeze the exact 60-row target-DEV final-query decomposition package."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
from paired_final_query_loss_common import masks_for_task, sha

OUT = ROOT / "experiments/capability_repair_baseline_v1/paired_final_query_loss_decomposition_v1"
DEV = ROOT / "experiments/targeted_capability_repair_v1/data/TARGET_DEV.jsonl"
FAMILIES = ("connected components", "inside/contains", "difference", "width", "orientation")

rows = [json.loads(x) for x in DEV.read_text(encoding="utf-8").splitlines() if x.strip()]
prior = json.loads((ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/TARGET_DEV_FIVE_FAMILY_MANIFEST.json").read_text())
by_id = {row["episode_id"]: row for row in rows}
ids = [member["episode_id"] for batch in prior["batches"] for member in batch["members"]]
selected = [by_id[item] for item in ids]
if len(ids) != 60 or len(set(ids)) != 60:
    raise RuntimeError("FROZEN_DEV_60_BINDING_FAIL")
if {family: sum(row["family"] == family for row in selected) for family in FAMILIES} != {family: 12 for family in FAMILIES}:
    raise RuntimeError("FAMILY_DENOMINATOR_FAIL")
members = []
for row in selected:
    mask = masks_for_task({"source_id": row["episode_id"], **row["task"]})
    members.append({
        "episode_id": row["episode_id"], "family": row["family"],
        "task_sha256": hashlib.sha256(json.dumps(row["task"], sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
        "sequence_length": len(mask["ids"]), "assistant_token_count": mask["assistant_token_count"],
        "component_token_counts": mask["component_token_counts"],
        "final_grid_plus_eos_token_count": mask["final_grid_plus_eos_token_count"],
    })
batches = [{"effective_batch_size": len(members[i:i + 32]), "members": members[i:i + 32]} for i in range(0, 60, 32)]
cross = []
for family in FAMILIES:
    cross.extend([row["episode_id"] for row in members if row["family"] == family][:6])
cross.extend([row["episode_id"] for row in members if row["episode_id"] not in cross][:2])
OUT.mkdir(parents=True, exist_ok=True)
manifest = OUT / "TARGET_DEV_60_FINAL_QUERY_MANIFEST.json"
manifest.write_text(json.dumps({"schema_version": 1, "rows": 60, "families": FAMILIES,
    "dev_sha256": sha(DEV), "frozen_source_manifest_sha256": sha(ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/TARGET_DEV_FIVE_FAMILY_MANIFEST.json"),
    "batches": batches, "batch1_sensitivity_episode_ids": cross}, sort_keys=True, indent=2) + "\n")
config = {"schema_version": 1, "protocol_id": "PAIRED_FINAL_QUERY_LOSS_DECOMPOSITION_V1", "cap_seconds": 900,
 "seed": 20261009, "primary_estimand": "family_equal_mean(FB_minus_V7_FINAL_GRID_PLUS_EOS_per_token_nll)",
 "bootstrap": {"replicates": 10000, "family_stratified_paired": True, "raw_ci": "percentile_95"},
 "sensitivity": {"primary": "B32", "fixed_b1_subset_rows": 32, "U_definition": "abs(B1_subset_effect-B32_same_subset_effect)", "maximum": "min(0.20,0.25*abs(full_cohort_effect))"},
 "inputs": {"target_dev_path": str(DEV.relative_to(ROOT)).replace("\\", "/"), "target_dev_sha256": sha(DEV),
   "manifest_path": str(manifest.relative_to(ROOT)).replace("\\", "/"), "manifest_sha256": sha(manifest),
   "conditions": {"CAPABILITY_REPAIR_BASELINE_V1_V7": {"manifest": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"},
                  "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001": {"manifest": "experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/FINAL_CHECKPOINT_MANIFEST.json"}}},
 "masks": ["DEMONSTRATION_ASSISTANT_ALL", "FINAL_ASSISTANT_PREFIX", "FINAL_GRID_CONTENT", "FINAL_EOS"], "primary_mask": "FINAL_GRID_PLUS_EOS",
 "forbidden": ["optimizer", "backward", "training", "generation", "Gold", "dGold", "FINAL_AUDIT"], "source_sha256": {"worker": sha(ROOT / "scripts/run_paired_final_query_loss_decomposition_v1.py"), "mask_contract": sha(ROOT / "scripts/paired_final_query_loss_common.py"), "postprocessor": sha(ROOT / "scripts/analyze_paired_final_query_loss_decomposition_v1.py")}}
package = OUT / "EXECUTION_PACKAGE.json"
package.write_text(json.dumps(config, sort_keys=True, indent=2) + "\n")
print(json.dumps({"status": "FROZEN", "package_sha256": sha(package), "manifest_sha256": sha(manifest), "rows": 60, "crosscheck_rows": len(cross)}, sort_keys=True))
