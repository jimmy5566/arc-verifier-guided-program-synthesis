#!/usr/bin/env python3
"""Target-blind full Eval60 V7+TTT24 canonical-AUG8 Regret-DFS R1024 run.

The script reuses the validated Clean-HF worker, recursive historical search
order, root-aware scheduler, and S/M/L fallback policy.  It first runs a fixed
short/long B1-vs-production gate, then completes all 89 outputs.  Evaluation
solutions are intentionally not accepted by this executable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_ttt24_aug8_r1024_core_v1 import (
    AUG8,
    _atomic_json,
    _freeze_generation,
    _ledger,
    _read,
    _run_surface,
    _safe,
    _sha_value,
    _verify,
)

EXPERIMENT = "V7_TTT24_AUG8_DFS_R1024_EVAL60_V1"
TASK_IDS_SHA256 = "8da66911f3cafb2652da1e5731dfeafd9100670e68de68d22d93d596c2c3abd9"
V7_ADAPTER_SHA256 = "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3"
TTT24_TARGET_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
}


def sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_json(value: Any) -> str:
    return __import__("hashlib").sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _manifest_adapter_identity(path: Path, record: dict[str, Any]) -> dict[str, Any]:
    """Verify a fresh V7+TTT24 adapter against its frozen training receipt.

    Historical Base+TTT identity checks require seven projection modules.  The
    frozen public NVARC TTT24 recipe used here also trains ``embed_tokens`` and
    ``lm_head``.  The training manifest, not that historical checker, is the
    authoritative byte identity for these newly generated adapters.
    """
    expected_files = dict(record.get("adapter_files") or {})
    required = {"adapter_config.json", "adapter_model.safetensors"}
    if not required.issubset(expected_files):
        raise RuntimeError(f"frozen adapter record is incomplete: {path}")
    if Path(str(record.get("adapter_path", ""))) != path:
        raise RuntimeError(f"frozen adapter path mismatch: {path}")
    actual_files = {
        str(file.relative_to(path)): sha256(file)
        for file in sorted(path.rglob("*")) if file.is_file()
    }
    if actual_files != expected_files:
        raise RuntimeError(f"frozen adapter bytes mismatch: {path}")
    if _sha_json(actual_files) != record.get("adapter_files_sha256"):
        raise RuntimeError(f"frozen adapter aggregate identity mismatch: {path}")
    config = _read(path / "adapter_config.json")
    semantics = {
        "peft_type": config.get("peft_type"), "task_type": config.get("task_type"),
        "r": config.get("r"), "lora_alpha": config.get("lora_alpha"),
        "use_rslora": config.get("use_rslora"),
        "target_modules": sorted(map(str, config.get("target_modules") or ())),
    }
    expected_semantics = {
        "peft_type": "LORA", "task_type": "CAUSAL_LM", "r": 256,
        "lora_alpha": 32, "use_rslora": True,
        "target_modules": sorted(TTT24_TARGET_MODULES),
    }
    if semantics != expected_semantics:
        raise RuntimeError(f"frozen TTT24 adapter semantics mismatch: {path}")
    return {
        "status": "PASS", "adapter_path": str(path),
        "adapter_model_sha256": actual_files["adapter_model.safetensors"],
        "adapter_config_sha256": actual_files["adapter_config.json"],
        "adapter_files": actual_files,
        "adapter_files_sha256": record["adapter_files_sha256"],
        "adapter_config_semantics": semantics,
        "identity_authority": "FROZEN_TTT24_ADAPTER_MANIFEST",
    }


def output_identity(template: dict[str, Any], adapter_root: Path,
                    training_manifest: dict[str, Any]) -> list[dict[str, Any]]:
    rows = list(template.get("outputs", ()))
    if len(rows) != 89 or len({row["output_id"] for row in rows}) != 89:
        raise RuntimeError("historical full cohort must contain 89 unique outputs")
    tasks = {str(row["task_id"]) for row in rows}
    if len(tasks) != 60:
        raise RuntimeError("historical full cohort must contain 60 tasks")
    records = training_manifest.get("records") or {}
    if (training_manifest.get("status") != "FROZEN"
            or training_manifest.get("target_blind") is not True
            or training_manifest.get("solutions_opened") is not False
            or set(map(str, training_manifest.get("task_ids") or ())) != tasks
            or set(map(str, records)) != tasks):
        raise RuntimeError("TTT24 training manifest identity/completeness failure")
    identities: dict[str, dict[str, Any]] = {}
    result = []
    for row in rows:
        task = str(row["task_id"])
        path = adapter_root / task / "depth_024"
        if task not in identities:
            identities[task] = _manifest_adapter_identity(path, records[task])
        result.append({**row, "adapter_path": str(path), "adapter_identity": identities[task]})
    return result


def snapshots(run: Path, output_id: str) -> dict[str, Any]:
    raw = _read(run / "RAW_OUTPUTS" / f"{_safe(output_id)}.json")
    result = {}
    for cell in raw["cells"].values():
        checkpoint = next(row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == 1024)
        result[cell["augmentation_id"]] = checkpoint["candidate_pool_snapshot"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "model_path", "foundation_adapter", "challenge", "native_config_dir", "candidate_pool",
                 "adapter_root", "training_manifest", "historical_run_cohort", "coarse_policy", "worker_python"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        if not (args.output / "PRE_GOLD_CONTRACT.json").is_file():
            raise RuntimeError("refusing unbound nonempty output directory")
    else:
        args.output.mkdir(parents=True, exist_ok=True)
    template = _read(args.historical_run_cohort)
    training_manifest = _read(args.training_manifest)
    outputs = output_identity(template, args.adapter_root, training_manifest)
    policy = _read(args.coarse_policy)
    args.device = "cuda:0"
    args.diagnostic_trace = True
    args.frontier_telemetry = False
    args.search_order_policy = "legacy"
    args.checkpoints = "128,256,512,1024"
    contract = {
        "experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
        "source_commit": __import__("subprocess").run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, check=True,
            capture_output=True, text=True,
        ).stdout.strip(),
        "cohort": "Eval60: 60 tasks / 89 outputs", "ttt_depth": 24,
        "task_count": 60, "output_count": 89, "task_ids_sha256": TASK_IDS_SHA256,
        "decoder": "CUMULATIVE_REGRET_r=4.00", "max_expanded_nodes": 1024,
        "checkpoints": [128, 256, 512, 1024], "candidate_cap": 32, "frontier_floor": 1,
        "max_new_tokens": 931, "eos": 15, "augmentation": "CANONICAL_AUG8",
        "augmentation_ids": list(AUG8),
        "cache": "ChunkedDynamicCache / valid_length rollback",
        "generation_scope": "V7_TTT24_D24_ONLY",
        "training_manifest": str(args.training_manifest),
        "training_manifest_sha256": sha256(args.training_manifest),
        "foundation_adapter": str(args.foundation_adapter),
        "foundation_adapter_sha256": sha256(args.foundation_adapter / "adapter_model.safetensors"),
        "adapter_composition": "simultaneous frozen V7 plus task-local TTT LoRA residuals",
        "search_order": "historical recursive DFS; diagnostic_trace append-only",
        "admission": "root-aware exact READY compatibility; fairness_max_wait=3",
        "worker_python": str(args.worker_python), "solutions_argument_available": False,
    }
    if contract["foundation_adapter_sha256"] != V7_ADAPTER_SHA256:
        raise RuntimeError("frozen V7 adapter identity mismatch")
    _atomic_json(args.output / "PRE_GOLD_CONTRACT.json", contract)
    _atomic_json(args.output / "CONTRACT.json", contract)
    _atomic_json(args.output / "ADAPTER_IDENTITIES.json", {
        task: next(row["adapter_identity"] for row in outputs if row["task_id"] == task)
        for task in sorted({row["task_id"] for row in outputs})
    })

    # Deterministically choose one shortest and one longest output.  The
    # historical profile already binds their exact roots; production tokenizes
    # again and records the actual per-view root classes.
    ordered = sorted(outputs, key=lambda row: (int(row["root_length_max"]), row["output_id"]))
    smoke = [ordered[0], ordered[-1]]
    smoke_b1 = args.output / "B1_PARITY"
    b1_policy = {"profiles": {name: {"primary": {"resident_capacity": 1, "physical_batch_ceiling": 1}, "fallbacks": []}
                                for name in ("PROFILE_S", "PROFILE_M", "PROFILE_L_LOW", "PROFILE_L_HIGH", "PROFILE_L")}}
    _run_surface(args, smoke_b1, smoke, depth=24, experiment=EXPERIMENT + "_B1_GATE", policy=b1_policy)
    _freeze_generation(smoke_b1, EXPERIMENT + "_B1_GATE", 2)

    # Generate the same two outputs with the frozen production policy before
    # allowing the other 87 outputs to start.
    _run_surface(args, args.output, smoke, depth=24, experiment=EXPERIMENT, policy=policy)
    parity_rows = []
    for row in smoke:
        b1, production = snapshots(smoke_b1, row["output_id"]), snapshots(args.output, row["output_id"])
        exact = b1 == production
        parity_rows.append({"output_id": row["output_id"], "profile": row["profile"], "exact": exact,
                            "b1_sha256": __import__("hashlib").sha256(json.dumps(b1, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
                            "production_sha256": __import__("hashlib").sha256(json.dumps(production, sort_keys=True, separators=(",", ":")).encode()).hexdigest()})
    gate = {"status": "PASS" if all(row["exact"] for row in parity_rows) else "FAIL",
            "criterion": "exact R1024 parsed/canonical candidate-pool agreement on fixed short and long outputs",
            "rows": parity_rows}
    _atomic_json(args.output / "B1_PARITY_GATE.json", gate)
    if gate["status"] != "PASS":
        raise RuntimeError("B1 parity gate failed; remaining Eval60 outputs were not started")

    _run_surface(args, args.output, outputs, depth=24, experiment=EXPERIMENT, policy=policy)
    _freeze_generation(args.output, EXPERIMENT, 89)
    compact = _ledger(args.output, "HASHES.json", include_raw=False)
    compact_verification = _verify(args.output, args.output / "HASHES.json")
    _atomic_json(args.output / "HASH_VERIFICATION.json", compact_verification)
    if compact_verification["status"] != "PASS":
        raise RuntimeError("compact generation hash verification failed")
    _atomic_json(args.output / "COMPACT_ARCHIVE_FREEZE.json", {
        "experiment": EXPERIMENT, "status": "FROZEN", "target_blind": True,
        "gold_loaded": False, "compact_ledger_sha256": _sha_value(compact),
    })
    print(json.dumps({"event": "V7_TTT24_AUG8_R1024_EVAL60_FROZEN", "outputs": 89,
                      "target_blind": True, "gold_loaded": False}, sort_keys=True))


if __name__ == "__main__":
    main()
