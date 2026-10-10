"""CPU-only post-freeze scorer for the E04 V3 fixed-64 localization run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.e04_v3_fixed64_batch_rung_localization import (
    ARMS, PROTOCOL_ID, LocalizationFailure, atomic_json, load_and_validate_config,
    read_json, read_jsonl, sha_path, validate_arm_raw,
)


def score(output_root: Path, destination: Path) -> dict:
    config = load_and_validate_config()
    receipt = read_json(output_root / "RAW_ARMS_FREEZE_RECEIPT.json")
    if receipt.get("status") != "RAW_ARMS_FROZEN_NO_TARGETS" or receipt.get("target_sidecar_accessed") is not False:
        raise LocalizationFailure("E04_FIXED64_RAW_FREEZE_REQUIRED")
    cohort = read_json(Path(__file__).resolve().parents[1] / config["cohort_path"])
    prompt_hashes = cohort["row_input_sha256"]; indexes = cohort["row_indexes"]
    arms = {}
    for arm in ARMS:
        path = output_root / "raw_arms" / f"B{arm}_RAW.jsonl"
        if receipt.get("raw_arm_sha256", {}).get(str(arm)) != sha_path(path):
            raise LocalizationFailure("E04_FIXED64_RAW_ARM_HASH")
        arms[arm] = validate_arm_raw(path, indexes, prompt_hashes, arm)
    root = Path(__file__).resolve().parents[1]
    sidecar_path = root / config["target_sidecar_path"]
    if sha_path(sidecar_path) != config["target_sidecar_sha256"]:
        raise LocalizationFailure("E04_FIXED64_TARGET_IDENTITY")
    targets = {row["row_index"]: row["target"] for row in read_jsonl(sidecar_path)}
    historical = read_jsonl(root / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/run_003_ed3b3e71ab894bc6c6656db7a6dfc26d/remote_raw/BATCH1_SENSITIVITY_RAW_GENERATIONS.jsonl")
    historical_by_index = {row["row_index"]: row for row in historical}
    b1 = arms[1]
    repeatability = [row["row_index"] for row in b1 if any(row[key] != historical_by_index[row["row_index"]][key] for key in ("text", "parser_valid", "parsed_grid"))]
    comparisons = {}
    for arm in (16, 8, 4):
        mismatch = []
        for reference, candidate in zip(b1, arms[arm], strict=True):
            ref_exact, candidate_exact = reference["parsed_grid"] == targets[reference["row_index"]], candidate["parsed_grid"] == targets[candidate["row_index"]]
            fields = {"generated_token_ids": reference["generated_token_ids"] == candidate["generated_token_ids"], "parser_validity": reference["parser_valid"] == candidate["parser_valid"], "parsed_grid": reference["parsed_grid"] == candidate["parsed_grid"], "exact_grid_outcome": ref_exact == candidate_exact}
            if not all(fields.values()): mismatch.append({"row_index":reference["row_index"],"fields":fields})
        comparisons[str(arm)] = {"pass": not mismatch, "mismatches": mismatch}
    selected = next((arm for arm in (16, 8, 4) if comparisons[str(arm)]["pass"]), 1)
    result = {"schema_version":1,"protocol_id":PROTOCOL_ID,"status":"B1_RUNTIME_REPEATABILITY_INCONCLUSIVE" if repeatability else "SCORED_BATCH_RUNG_LOCALIZATION",
              "historical_b1_repeatability_mismatches":repeatability,"candidate_arms":comparisons,"selected_primary_rung":None if repeatability else selected,
              "selection_rule":"largest passing rung in fixed order B16,B8,B4; else B1","raw_freeze_receipt_sha256":sha_path(output_root / "RAW_ARMS_FREEZE_RECEIPT.json"),
              "target_sidecar_opened_only_after_raw_freeze":True,"target_sidecar_sha256":sha_path(sidecar_path),"target_sidecar_accessed":True,
              "rows":64,"forbidden_operations":{"optimizer_steps":0,"parameter_updates":0,"backward_calls":0,"final_audit_opened":False}}
    if destination.exists(): raise LocalizationFailure("E04_FIXED64_SCORE_OUTPUT_FRESH")
    atomic_json(destination,result); return result


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output-root",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); print(json.dumps(score(args.output_root,args.output),sort_keys=True))


if __name__ == "__main__": main()
