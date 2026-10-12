"""Freeze Director-requested, target-blind measurement additions for E1.

This module deliberately creates no model, target sidecar, or executable
training/GPU job.  It is additive: the two approved training schedules are
read and hashed, never rewritten.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
MEASURE = OUT / "measurement_contract_v1"
E05_INPUT = ROOT / "experiments/capability_repair_baseline_v1/e05_structurally_held_out_marker_conditioned_rotation_confirmation_v1/INPUT_PROMPTS.jsonl"
CHECKPOINT_SHA256 = "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def fsha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def write(path: Path, value) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("wb", dir=path.parent, delete=False) as f:
        f.write(canonical(value) + b"\n")
        tmp = Path(f.name)
    os.replace(tmp, path)
    return fsha(path)


def source_prompt(row: dict) -> dict:
    """Keep train demonstrations, remove the held-out query target."""
    task = copy.deepcopy(row["task"])
    task["test"] = [{"input": task["test"][0]["input"]}]
    return {
        "arm": row["arm"], "source_slot": row["slot"],
        "family": row["structural_family"], "source_id": task["source_id"],
        "canonical_base_id": row["canonical_base_id"], "task": task,
    }


def protected_prompt(row: dict) -> dict:
    # E05 INPUT_PROMPTS contains demonstrations and a query input, but no
    # query target.  Do not open the E05 target-scorer sidecar.
    return {k: copy.deepcopy(row[k]) for k in (
        "canonical_base_id", "canonical_base_index", "condition", "episode_id",
        "fixed_native_demonstrations", "query_input", "turn")}


def main() -> None:
    control_path, treatment_path = OUT / "CONTROL_SCHEDULE.json", OUT / "TREATMENT_SCHEDULE.json"
    control, treatment = (json.loads(p.read_text(encoding="utf-8")) for p in (control_path, treatment_path))
    schedule_hashes = {"control": fsha(control_path), "treatment": fsha(treatment_path)}

    train_fit = []
    for schedule in (control, treatment):
        arm = schedule["arm"]
        primaries = [dict(r, arm=arm) for r in schedule["episodes"][:96]]
        for family in ("CONNECTED_COMPONENT_COUNT", "OBJECT_EXTRACTION"):
            picked = [r for r in primaries if r["structural_family"] == family][:12]
            if len(picked) != 12:
                raise RuntimeError("TRAIN_FIT_BALANCE")
            train_fit.extend(source_prompt(r) for r in picked)
    train_fit_sha = write(MEASURE / "TRAIN_FIT_INPUT_PROMPTS.json", {
        "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1_TRAIN_FIT_V1",
        "target_blind_query": True, "arms": [control["arm"], treatment["arm"]],
        "per_arm_per_family": 12, "records": train_fit,
    })

    e05_rows = [json.loads(line) for line in E05_INPUT.read_text(encoding="utf-8").splitlines()]
    protected = []
    for condition in ("MARKER_BINDING_CONTROL", "FIXED_TURN_ROTATION_CONTROL"):
        rows = [r for r in e05_rows if r["condition"] == condition][:8]
        if len(rows) != 8:
            raise RuntimeError("PROTECTED_PANEL_UNAVAILABLE")
        protected.extend(protected_prompt(r) for r in rows)
    protected_sha = write(MEASURE / "PROTECTED_CAPABILITY_INPUT_PROMPTS.json", {
        "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V1_PROTECTED_PANEL_V1",
        "source_input_path": str(E05_INPUT.relative_to(ROOT)).replace("\\", "/"),
        "source_input_sha256": fsha(E05_INPUT), "target_blind_query": True,
        "counts": {"MARKER_BINDING_CONTROL": 8, "FIXED_TURN_ROTATION_CONTROL": 8}, "records": protected,
    })

    base_inputs = {name: fsha(OUT / name) for name in (
        "OOD_INPUT_PROMPTS.json", "RETENTION_INPUT_PROMPTS.json", "COMPOSITION_SENTINEL_INPUT_PROMPTS.json")}
    v7_sha = write(MEASURE / "V7_EVALUATION_BINDING.json", {
        "checkpoint": "V7", "checkpoint_sha256": CHECKPOINT_SHA256,
        "same_input_files_sha256": base_inputs,
        "additional_protected_panel_sha256": protected_sha,
        "rule": "Evaluate V7, control, and treatment on the exact named input sets before opening any evaluation targets.",
    })
    oracle_sha = write(MEASURE / "ORACLE_AND_SCORER_CONTRACT.json", {
        "target_sidecars_materialized": False, "targets_accessed": False,
        "deterministic_oracle_source": "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py",
        "deterministic_oracle_source_sha256": fsha(ROOT / "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py"),
        "native_io_source": "src/inference/arc_native_io.py",
        "native_io_source_sha256": fsha(ROOT / "src/inference/arc_native_io.py"),
        "token_grid_parser_source": "scripts/arc2_token_grid_parser.py",
        "token_grid_parser_source_sha256": fsha(ROOT / "scripts/arc2_token_grid_parser.py"),
        "freeze_before_target_rule": "For every checkpoint and split, freeze SHA256 of raw generated token ids and parsed grid records before deterministic target generation or sealed target-sidecar access. Training and inference receive prompt inputs only.",
        "score": "native greedy parsed-grid exact match; malformed or unparsable output is incorrect",
    })
    analysis_sha = write(MEASURE / "MEASUREMENT_ANALYSIS_CONTRACT.json", {
        "train_fit": {"by": ["arm", "family"], "n_per_arm_family": 12, "role": "diagnostic adequacy, reported separately"},
        "ood": {"by": ["family"], "n_per_family": 24, "comparison": "paired treatment-control exact-grid delta with paired bootstrap interval"},
        "aggregate_rule": "Aggregate is reportable only with both family estimates. A positive aggregate cannot override a material negative Count or Object Extraction delta.",
        "proposed_harm_limits_for_director_review": {"NO_TRANSFORM_RETENTION_CONTROL": "at most 1 paired harm of 24", "MARKER_BINDING_CONTROL": "at most 0 paired harms of 8", "FIXED_TURN_ROTATION_CONTROL": "at most 0 paired harms of 8"},
        "composition_sentinel": "reported separately; no aggregate substitution", "target_opening": "only after all raw checkpoint/split prediction hashes are frozen",
    })
    runtime_sha = write(MEASURE / "RUNTIME_CALL_ESTIMATE.json", {
        "checkpoints": ["V7", control["arm"], treatment["arm"]],
        "native_greedy_task_evaluations": {"train_fit": 48, "per_checkpoint_holdout": 104, "holdout_total": 312, "total": 360},
        "training_conditions": "unchanged: 96 updates/arm, 4 slots/update, BF16 PagedAdamW8bit LoRA r64 alpha32 dropout0 LR5e-5",
        "runtime_bound_proposal_for_director_review": {"training_seconds_total": 7200, "evaluation_seconds_total": 10800, "combined_seconds_total": 18000, "basis": "360 task calls at a conservative 30 seconds/call; actual elapsed time must be recorded"},
    })
    manifest = {"status": "PASS_CPU_ONLY_ADDITIVE_MEASUREMENT_CONTRACT", "model_loaded": False, "gpu_used": False, "targets_accessed": False,
        "immutable_schedule_sha256": schedule_hashes, "files": {"train_fit": train_fit_sha, "protected": protected_sha, "v7_binding": v7_sha, "oracle_scorer": oracle_sha, "analysis": analysis_sha, "runtime": runtime_sha}}
    write(MEASURE / "AMENDED_MANIFEST.json", manifest)
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
