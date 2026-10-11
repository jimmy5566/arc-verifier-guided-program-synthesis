"""CPU-only freezer and verifier for E04-D local rotation/marker gradients.

This module neither imports torch nor loads a model.  It freezes a TRAIN-only
matched cohort from the E04 native generator and a no-update execution contract.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import sys
ROOT_IMPORT = Path(__file__).resolve().parents[1]
if str(ROOT_IMPORT) not in sys.path:
    sys.path.insert(0, str(ROOT_IMPORT))
from scripts import e04_orientation_marker_counterfactual_v2 as e04

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e04_d_rotation_marker_local_gradient_diagnostic_v1"
PROTOCOL = "E04_D_ROTATION_MARKER_LOCAL_GRADIENT_DIAGNOSTIC_V1"
CONDITIONS = ("FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL")
SEED = 20261011
BOOTSTRAPS = 10000
CAP_SECONDS = 1800

class FreezeFailure(RuntimeError):
    pass

def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def digest(value: Any) -> str:
    return hashlib.sha256(canon(value)).hexdigest()

def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            json.dump(value, f, sort_keys=True, indent=2)
            f.write("\n")
            f.flush(); os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def native_task(base: dict[str, Any], turn: int, condition: str) -> dict[str, Any]:
    prompt = e04.task_prompt(base, turn, condition)
    return {"train": prompt["train"], "test": [{"input": prompt["test"][0]["input"], "output": e04.target_grid(base, turn, condition)}]}

def selected_bases() -> list[dict[str, Any]]:
    """One hash-selected legitimate TRAIN base from each of the 12 allowed cells."""
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for base in e04.canonical_bases("TRAIN"):
        grouped[(base["shape_orientation"], base["reference_relation"])].append(base)
    if set(grouped) != set(e04.cells("TRAIN")) or any(len(v) != 24 for v in grouped.values()):
        raise FreezeFailure("E04D_TRAIN_CELL_IDENTITY")
    result = []
    for cell in sorted(grouped):
        result.append(min(grouped[cell], key=lambda b: digest({"seed": SEED, "cell": cell, "id": b["canonical_base_id"]})))
    return result

def cohort_rows() -> list[dict[str, Any]]:
    rows = []
    for base in selected_bases():
        for turn in e04.TURNS:
            tasks = {condition: native_task(base, turn, condition) for condition in CONDITIONS}
            pair_id = digest({"protocol": PROTOCOL, "base": base["canonical_base_id"], "turn": turn})
            rows.append({
                "pair_id": pair_id,
                "canonical_base_id": base["canonical_base_id"],
                "train_cell": {"shape_orientation": base["shape_orientation"], "reference_relation": base["reference_relation"]},
                "turn": turn,
                "tasks": tasks,
                "task_sha256": {condition: digest(task) for condition, task in tasks.items()},
            })
    return rows

def validate(rows: list[dict[str, Any]]) -> None:
    allowed = set(e04.cells("TRAIN"))
    if len(rows) != len(allowed) * len(e04.TURNS) or len({r["pair_id"] for r in rows}) != len(rows):
        raise FreezeFailure("E04D_COHORT_COUNT")
    by_cell: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    by_base: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cell = (row["train_cell"]["shape_orientation"], row["train_cell"]["reference_relation"])
        if cell not in allowed or e04.RELATIONS[cell[0]] == cell[1]: raise FreezeFailure("E04D_CELL_NOT_TRAIN")
        if set(row["tasks"]) != set(CONDITIONS) or set(row["task_sha256"]) != set(CONDITIONS): raise FreezeFailure("E04D_CONDITION_SET")
        for c, task in row["tasks"].items():
            if digest(task) != row["task_sha256"][c]: raise FreezeFailure("E04D_TASK_HASH")
            if len(task["train"]) != 4 or set(task["test"][0]) != {"input", "output"}: raise FreezeFailure("E04D_TASK_SHAPE")
        # Latent base/query inputs match across all conditions; only demonstrations/targets differ.
        query_inputs = {digest(task["test"][0]["input"]) for task in row["tasks"].values()}
        if len(query_inputs) != 1: raise FreezeFailure("E04D_QUERY_NOT_MATCHED")
        if len({digest(task["train"]) for task in row["tasks"].values()}) != 3: raise FreezeFailure("E04D_DEMONSTRATION_COLLISION")
        by_cell[cell].append(row); by_base[row["canonical_base_id"]].append(row)
    if set(by_cell) != allowed or any(len(v) != 4 for v in by_cell.values()): raise FreezeFailure("E04D_CELL_TURN_COVERAGE")
    if len(by_base) != len(allowed) or any({r["turn"] for r in v} != set(e04.TURNS) for v in by_base.values()): raise FreezeFailure("E04D_BASE_TURN_COVERAGE")

def contract(cohort_sha: str, source_sha: str) -> dict[str, Any]:
    return {
      "schema_version": 1, "protocol_id": PROTOCOL, "status": "CPU_ONLY_FROZEN_PRELAUNCH_PENDING_DIRECTOR_REVIEW",
      "scientific_question": "At exact V7 start, do matched fixed-turn rotation and marker-binding LoRA gradients show locally antagonistic directions?",
      "frozen_cohort": {"path": (OUT / "FROZEN_TRAIN_COHORT.jsonl").relative_to(ROOT).as_posix(), "sha256": cohort_sha, "canonical_bases": 12, "matched_pairs": 48, "conditions": list(CONDITIONS), "selection": "one hash-selected legitimate TRAIN base per allowed structural cell; all four turns", "selection_seed": SEED, "validation_rows_read": 0},
      "checkpoint": {"identity": "CAPABILITY_REPAIR_BASELINE_V1_V7", "manifest_path": "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json", "manifest_sha256": "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549"},
      "execution": {"physical_batch_size": 1, "precision": "BF16", "loss_reduction": "FP32 supervised-token CE", "gradient_reduction": "FP32", "objective": "same E04-C production serializer and all-assistant CE", "lora_target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], "optimizer_steps": 0, "parameter_updates": 0, "optimizer_construction": False, "generation_calls": 0, "maximum_active_seconds": CAP_SECONDS, "execution_authorized": False},
      "estimands": {"primary": "per-pair cosine(g_FIXED_TURN_ROTATION_CONTROL, g_MARKER_BINDING_CONTROL), averaged within canonical base then equally across the 12 bases", "references": ["cosine(g_FIXED_TURN_ROTATION_CONTROL,g_NO_TRANSFORM_RETENTION_CONTROL)", "cosine(g_MARKER_BINDING_CONTROL,g_NO_TRANSFORM_RETENTION_CONTROL)"], "secondary": ["per-family gradient norm", "LoRA module and layer squared-norm contributions", "negative-cosine frequency"]},
      "uncertainty": {"bootstrap": {"replicates": BOOTSTRAPS, "seed": SEED, "unit": "canonical base with all four turns resampled together", "interval": "percentile 95%"}, "repeats": {"rows": "lexicographically first matched pair in each of FIXED_TURN_ROTATION_CONTROL, MARKER_BINDING_CONTROL, NO_TRANSFORM_RETENTION_CONTROL", "replicates_per_row": 2, "directional_measure": "one minus repeat cosine", "classification": "use maximum observed directional deviation U to form [L-U,H+U]; scale variation is reported separately"}},
      "decision_rules": {"LOCAL_CONFLICT_SUPPORTED": "rotation-marker base-bootstrap CI high < 0; base-bootstrap negative-cosine-frequency CI low > 0.5; and sensitivity-adjusted cosine CI high < 0", "LOCAL_CONFLICT_DEPRIORITIZED": "rotation-marker base-bootstrap CI low > 0; negative-cosine-frequency CI high < 0.5; and sensitivity-adjusted cosine CI low > 0", "MIXED_OR_INCONCLUSIVE": "otherwise, including any interval/sensitivity envelope spanning zero or disagreement between aggregate and frequency evidence", "INVALID_NOT_INTERPRETABLE": "identity, cohort, serializer, label-mask, trainable-set, completeness, nonfinite-gradient, or sealed-boundary failure"},
      "boundaries": ["TRAIN_ONLY", "NO_VALIDATION_CORRECTNESS_OR_TARGETS_FOR_SELECTION", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT", "NO_OPTIMIZER", "NO_PARAMETER_UPDATE", "NO_TRAINING", "NO_GENERATION"],
      "source": {"generator_path": "scripts/e04_orientation_marker_counterfactual_v2.py", "generator_sha256": source_sha, "freezer_path": "scripts/freeze_e04_d_rotation_marker_gradient_diagnostic.py"}
    }

def freeze(out: Path) -> dict[str, Any]:
    if out.exists(): raise FreezeFailure("E04D_REFUSE_OVERWRITE")
    rows = cohort_rows(); validate(rows)
    out.mkdir(parents=True)
    cohort = out / "FROZEN_TRAIN_COHORT.jsonl"
    cohort.write_bytes(b"".join(canon(row) + b"\n" for row in rows))
    c = contract(sha(cohort), sha(Path(e04.__file__)))
    atomic_json(out / "E04_D_PRELAUNCH_PROTOCOL_V1.json", c)
    manifest = {"protocol_id": PROTOCOL, "cohort_sha256": sha(cohort), "protocol_sha256": sha(out / "E04_D_PRELAUNCH_PROTOCOL_V1.json"), "status": c["status"], "model_loaded": False, "gpu_used": False, "gradient_computed": False, "optimizer_steps": 0, "validation_rows_read": 0}
    atomic_json(out / "MANIFEST.json", manifest)
    return manifest

def self_test() -> None:
    rows = cohort_rows(); validate(rows)
    bad = json.loads(json.dumps(rows)); bad[0]["task_sha256"][CONDITIONS[0]] = "0" * 64
    try: validate(bad)
    except FreezeFailure as e:
        if str(e) != "E04D_TASK_HASH": raise
    else: raise FreezeFailure("E04D_SELF_TEST_REJECTS_NOT")

def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path,default=OUT); p.add_argument("--self-test",action="store_true"); a=p.parse_args()
    if a.self_test: self_test(); print("PASS_E04D_CPU_FREEZER")
    else: print(json.dumps(freeze(a.out),sort_keys=True))
if __name__ == "__main__": main()

