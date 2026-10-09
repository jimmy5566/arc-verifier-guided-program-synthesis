#!/usr/bin/env python3
"""CPU-only closure analysis for the paired fixed-B32 TRAIN/DEV NLL stage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
from statistics import mean, median


ROOT = Path(__file__).resolve().parents[1]
WEAK_FAMILIES = (
    "connected components",
    "difference",
    "inside/contains",
    "orientation",
    "width",
)
V7 = "CAPABILITY_REPAIR_BASELINE_V1_V7"
FAMILY_BALANCED = "ATOMIC_PREREQUISITE_FAMILY_BALANCED_LOSS_CONTROL_V1_001"
ATOMIC_REJECTED = "ATOMIC_PREREQUISITE_REPAIR_V1_001_REJECTED"

PATHS = {
    "dev_raw": ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/run_001_result_v1/RAW.jsonl",
    "dev_manifest": ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/TARGET_DEV_FIVE_FAMILY_MANIFEST.json",
    "v7_result": ROOT / "experiments/capability_repair_baseline_v1/corrected_baseline_v1/CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json",
    "family_balanced_result": ROOT / "experiments/capability_repair_baseline_v1/family_balanced_loss_control_v1/round_001/evaluation_v1/CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json",
    "atomic_gate": ROOT / "experiments/capability_repair_baseline_v1/forward_capability_repair_v2/atomic_prerequisite_repair_successor_proposal_v1/concrete_protocol_v1/ATOMIC_PREREQUISITE_REPAIR_V1_001_PAIRED_GATE_ANALYSIS.json",
    "atomic_checkpoint_manifest": ROOT / "experiments/capability_repair_baseline_v1/forward_capability_repair_v2/atomic_prerequisite_repair_successor_proposal_v1/concrete_protocol_v1/ATOMIC_PREREQUISITE_REPAIR_V1_001_FINAL_CHECKPOINT_MANIFEST.json",
    "train_execution_package": ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_fit_transfer_v1/PAIRED_FIXED_B32_FIT_TRANSFER_V1_EXECUTION_PACKAGE_V11.json",
    "train_result_package": ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_fit_transfer_v1/run_001_result_v1/RESULT_PACKAGE.json",
}

EXPECTED_SHA256 = {
    "dev_raw": "3a2954636b31cb7c9fb27f8935f6cffd262e923190afdb2bd433c1acf1ada473",
    "dev_manifest": "e531d07129dbbef5dd36b14ebc9e2430ca1dd2bd300aeb069e4f9e248dd9c786",
    "v7_result": "e65467ff712240348e9ef60a49797109f8ee262c69ac457be11d57a5716256d5",
    "family_balanced_result": "cc97e376e2a4703c0d73e4a79f0bbf528ac71c8750b9b5972a85f9099cb40216",
    "atomic_gate": "fd14962203c4f4631c6712214758bca1b228da97879814d2f4656454a4b08671",
    "atomic_checkpoint_manifest": "0586201a3eec4ac1c56a1de42a5b20c9ff74f52069fbe92a55691ece28ed7ddd",
    "train_execution_package": "930e1ebd15caa5312eac9eb8655a86fc16e2dab4d5610a1e5ff8ec299930622d",
    "train_result_package": "56ac80f2de646180e630ffad8d97d83df1aecc548d131f3c4f849b43d7c9bc7f",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def exact_transition(baseline: bool, candidate: bool) -> str:
    if not baseline and candidate:
        return "FIX"
    if baseline and not candidate:
        return "HARM"
    if baseline and candidate:
        return "STABLE_CORRECT"
    return "STABLE_WRONG"


def delta_summary(rows: list[dict]) -> dict:
    values = [row["per_token_nll_delta_candidate_minus_v7"] for row in rows]
    supervised_tokens = sum(row["supervised_token_count"] for row in rows)
    pooled_v7_nll = sum(
        row["v7_per_token_nll"] * row["supervised_token_count"] for row in rows
    ) / supervised_tokens
    pooled_candidate_nll = sum(
        row["family_balanced_per_token_nll"] * row["supervised_token_count"] for row in rows
    ) / supervised_tokens
    transitions = Counter(row["exact_grid_transition"] for row in rows)
    by_transition = {}
    for label in ("FIX", "HARM", "STABLE_CORRECT", "STABLE_WRONG"):
        selected = [row["per_token_nll_delta_candidate_minus_v7"] for row in rows if row["exact_grid_transition"] == label]
        by_transition[label] = {
            "episodes": len(selected),
            "mean_delta": mean(selected) if selected else None,
            "median_delta": median(selected) if selected else None,
        }
    return {
        "episodes": len(rows),
        "baseline_exact": sum(int(row["v7_exact_grid_match"]) for row in rows),
        "candidate_exact": sum(int(row["family_balanced_exact_grid_match"]) for row in rows),
        "exact_grid_transition_counts": {
            label: transitions.get(label, 0)
            for label in ("FIX", "HARM", "STABLE_CORRECT", "STABLE_WRONG")
        },
        "per_token_nll_delta_candidate_minus_v7": {
            "aggregation": "unweighted descriptive distribution over per-episode deltas",
            "mean": mean(values),
            "median": median(values),
            "minimum": min(values),
            "maximum": max(values),
            "improved_negative_count": sum(value < 0 for value in values),
            "worsened_positive_count": sum(value > 0 for value in values),
            "exact_zero_count": sum(value == 0 for value in values),
        },
        "pooled_token_weighted_nll": {
            "supervised_tokens": supervised_tokens,
            "v7": pooled_v7_nll,
            "family_balanced": pooled_candidate_nll,
            "delta_candidate_minus_v7": pooled_candidate_nll - pooled_v7_nll,
        },
        "nll_delta_by_exact_grid_transition": by_transition,
    }


def build_alignment(
    raw: list[dict], manifest: dict, v7_result: dict, family_balanced_result: dict
) -> dict:
    episode_ids = [member["episode_id"] for batch in manifest["batches"] for member in batch["members"]]
    if len(episode_ids) != 60 or len(set(episode_ids)) != 60:
        raise RuntimeError("DEV_MANIFEST_EPISODE_IDENTITY_FAIL")
    allowed = set(episode_ids)

    primary = {}
    for row in raw:
        if row.get("mode_id") != "PRIMARY_B32":
            continue
        key = (row["checkpoint_id"], row["episode_id"])
        if key in primary:
            raise RuntimeError("DUPLICATE_PRIMARY_NLL_ROW")
        primary[key] = row
    expected_keys = {(checkpoint, episode_id) for checkpoint in (V7, FAMILY_BALANCED) for episode_id in episode_ids}
    if set(primary) != expected_keys:
        raise RuntimeError("PRIMARY_NLL_KEY_SET_MISMATCH")

    def exact_map(result: dict) -> dict[str, bool]:
        found = {}
        for row in result["predictions"]:
            episode_id = row["episode_id"]
            if episode_id not in allowed:
                continue
            if episode_id in found:
                raise RuntimeError("DUPLICATE_EXACT_GRID_ROW")
            found[episode_id] = bool(row["exact_grid_match"])
        if set(found) != allowed:
            raise RuntimeError("EXACT_GRID_KEY_SET_MISMATCH")
        return found

    v7_exact = exact_map(v7_result)
    candidate_exact = exact_map(family_balanced_result)
    per_episode = []
    for episode_id in episode_ids:
        family = episode_id.split(":")[2]
        if family not in WEAK_FAMILIES:
            raise RuntimeError("UNEXPECTED_WEAK_FAMILY")
        baseline = primary[(V7, episode_id)]
        candidate = primary[(FAMILY_BALANCED, episode_id)]
        if baseline["supervised_token_count"] != candidate["supervised_token_count"]:
            raise RuntimeError("SUPERVISED_TOKEN_COUNT_MISMATCH")
        baseline_nll = baseline["row_nll_sum"] / baseline["supervised_token_count"]
        candidate_nll = candidate["row_nll_sum"] / candidate["supervised_token_count"]
        per_episode.append(
            {
                "episode_id": episode_id,
                "family": family,
                "supervised_token_count": baseline["supervised_token_count"],
                "v7_per_token_nll": baseline_nll,
                "family_balanced_per_token_nll": candidate_nll,
                "per_token_nll_delta_candidate_minus_v7": candidate_nll - baseline_nll,
                "v7_exact_grid_match": v7_exact[episode_id],
                "family_balanced_exact_grid_match": candidate_exact[episode_id],
                "exact_grid_transition": exact_transition(v7_exact[episode_id], candidate_exact[episode_id]),
            }
        )
    return {
        "schema_version": 1,
        "analysis_id": "CPU_ONLY_CURRENT_STAGE_NLL_EXACT_ALIGNMENT_CLOSURE_V1",
        "status": "COMPLETE_DESCRIPTIVE_PAIRED_ANALYSIS",
        "scope": "FROZEN_60_EPISODE_TARGET_DEV_WEAK_FAMILY_COHORT",
        "nll_definition": "sum teacher-forced token cross-entropy divided by supervised token count",
        "nll_delta_sign": "negative means Family-Balanced has lower NLL than V7",
        "overall": delta_summary(per_episode),
        "by_family": {
            family: delta_summary([row for row in per_episode if row["family"] == family])
            for family in WEAK_FAMILIES
        },
        "per_episode": sorted(per_episode, key=lambda row: row["episode_id"]),
        "interpretation_boundaries": [
            "This is paired descriptive development-set evidence, not causal identification.",
            "The 12 examples per family do not establish broad ARC generalization.",
            "Lower teacher-forced NLL does not by itself establish improved autoregressive exact-grid reasoning.",
            "The frozen DEV cohort is development evidence and must not be represented as an untouched holdout.",
        ],
        "final_audit_opened": False,
    }


def build_atomic_posthoc(gate: dict, checkpoint_manifest: dict, train_package: dict, train_result: dict) -> dict:
    conditions = train_package["inputs"]["conditions"]
    atomic_condition = conditions[ATOMIC_REJECTED]
    v7_condition = conditions[V7]
    if gate["round_id"] != "ATOMIC_PREREQUISITE_REPAIR_V1_001":
        raise RuntimeError("ATOMIC_ROUND_ID_MISMATCH")
    if gate["candidate_result"]["sha256"] != atomic_condition["result_sha256"]:
        raise RuntimeError("ATOMIC_RESULT_SHA_MISMATCH")
    if gate["baseline_result"]["sha256"] != v7_condition["result_sha256"]:
        raise RuntimeError("V7_RESULT_SHA_MISMATCH")
    if checkpoint_manifest["manifest_sha256"] != atomic_condition["checkpoint_manifest_declared_identity"]:
        raise RuntimeError("ATOMIC_CHECKPOINT_DECLARED_IDENTITY_MISMATCH")
    adapter_sha = next(
        item["sha256"]
        for item in checkpoint_manifest["adapter_files"]
        if item["name"] == "adapter_model.safetensors"
    )
    if adapter_sha != atomic_condition["adapter_sha256"]:
        raise RuntimeError("ATOMIC_ADAPTER_SHA_MISMATCH")
    train_atomic = train_result["scientific_summary"]["rejected_atomic_repair"]
    if train_atomic["decision"] != "INCONCLUSIVE_MISSING_FAMILY_DEV_EVIDENCE":
        raise RuntimeError("FROZEN_ATOMIC_CLASSIFIER_OUTCOME_CHANGED")
    family_rows = {}
    for family in WEAK_FAMILIES:
        paired = gate["families"][family]
        family_rows[family] = {
            "train_full_b32_per_token_nll_delta_candidate_minus_v7": train_atomic["full_b32_delta"][family],
            "historical_exact_grid": {
                "episodes": paired["episodes"],
                "v7_exact": paired["baseline_exact"],
                "atomic_r1_exact": paired["candidate_exact"],
                "net_paired_fixes": paired["net_paired_fixes"],
                "paired_fixes_n01": paired["paired_fixes_n01"],
                "paired_harms_n10": paired["paired_harms_n10"],
            },
        }
    return {
        "schema_version": 1,
        "analysis_id": "ATOMIC_R1_POSTHOC_SUPPLEMENTAL_NLL_EXACT_ALIGNMENT_V1",
        "status": "COMPATIBLE_POST_HOC_SUPPLEMENTAL_EVIDENCE",
        "identity_match": {
            "atomic_result_sha256": atomic_condition["result_sha256"],
            "v7_result_sha256": v7_condition["result_sha256"],
            "atomic_checkpoint_manifest_file_sha256": EXPECTED_SHA256["atomic_checkpoint_manifest"],
            "atomic_checkpoint_declared_identity": checkpoint_manifest["manifest_sha256"],
            "atomic_adapter_sha256": adapter_sha,
        },
        "families": family_rows,
        "frozen_classifier_outcome": train_atomic["decision"],
        "frozen_classifier_outcome_changed": False,
        "prospective_claim": False,
        "interpretation": (
            "Exact matching result identities make the historical family-level paired correctness evidence "
            "scientifically compatible as a separately labelled post-hoc supplement. It does not amend the "
            "frozen classifier or establish a prospective causal result."
        ),
        "final_audit_opened": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "experiments/capability_repair_baseline_v1/paired_fixed_b32_dev_nll_transfer_v1/current_stage_closure_v1",
    )
    args = parser.parse_args()
    actual = {name: sha256_file(path) for name, path in PATHS.items()}
    for name, expected in EXPECTED_SHA256.items():
        if actual[name] != expected:
            raise RuntimeError(f"INPUT_SHA256_MISMATCH:{name}")

    alignment = build_alignment(
        read_jsonl(PATHS["dev_raw"]),
        read_json(PATHS["dev_manifest"]),
        read_json(PATHS["v7_result"]),
        read_json(PATHS["family_balanced_result"]),
    )
    alignment["input_sha256"] = {
        key: actual[key]
        for key in ("dev_raw", "dev_manifest", "v7_result", "family_balanced_result")
    }
    atomic = build_atomic_posthoc(
        read_json(PATHS["atomic_gate"]),
        read_json(PATHS["atomic_checkpoint_manifest"]),
        read_json(PATHS["train_execution_package"]),
        read_json(PATHS["train_result_package"]),
    )
    atomic["input_sha256"] = {
        key: actual[key]
        for key in ("atomic_gate", "atomic_checkpoint_manifest", "train_execution_package", "train_result_package")
    }
    alignment_path = args.output_dir / "NLL_EXACT_ALIGNMENT_ANALYSIS.json"
    atomic_path = args.output_dir / "ATOMIC_R1_POSTHOC_SUPPLEMENTAL_ANALYSIS.json"
    atomic_json(alignment_path, alignment)
    atomic_json(atomic_path, atomic)
    print(
        json.dumps(
            {
                "status": "COMPLETE_CPU_ONLY",
                "alignment_sha256": sha256_file(alignment_path),
                "atomic_posthoc_sha256": sha256_file(atomic_path),
                "episodes": alignment["overall"]["episodes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
