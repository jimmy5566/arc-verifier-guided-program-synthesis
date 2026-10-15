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
import statistics
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
    _run_persistent_surface,
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
        # ``adapter_sha256`` is the established worker/cohort interface.  Keep
        # the explicit model-file alias for readability without breaking the
        # production worker contract.
        "adapter_sha256": actual_files["adapter_model.safetensors"],
        "adapter_model_sha256": actual_files["adapter_model.safetensors"],
        "adapter_config_sha256": actual_files["adapter_config.json"],
        "adapter_files": actual_files,
        "adapter_files_sha256": record["adapter_files_sha256"],
        "adapter_config_semantics": semantics,
        "identity_authority": "FROZEN_TTT24_ADAPTER_MANIFEST",
    }


def _adapter_stat_snapshot(adapter_root: Path, records: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for task, record in sorted(records.items()):
        path = adapter_root / str(task) / "depth_024"
        files = {}
        for relative in sorted(record.get("adapter_files") or {}):
            file = path / relative
            if not file.is_file():
                raise RuntimeError(f"frozen adapter file is missing: {file}")
            stat = file.stat()
            files[relative] = {
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "inode": stat.st_ino,
                "device": stat.st_dev,
            }
        result[str(task)] = files
    return result


def _cached_manifest_adapter_identity(path: Path, record: dict[str, Any],
                                      cached: dict[str, Any]) -> dict[str, Any]:
    expected_files = dict(record.get("adapter_files") or {})
    if (cached.get("status") != "PASS"
            or cached.get("identity_authority") != "FROZEN_TTT24_ADAPTER_MANIFEST"
            or cached.get("adapter_path") != str(path)
            or cached.get("adapter_files") != expected_files
            or cached.get("adapter_files_sha256") != record.get("adapter_files_sha256")):
        raise RuntimeError(f"cached adapter identity does not match frozen manifest: {path}")
    actual_names = {
        str(file.relative_to(path)) for file in path.rglob("*") if file.is_file()
    }
    if actual_names != set(expected_files):
        raise RuntimeError(f"cached adapter file set changed: {path}")
    config_path = path / "adapter_config.json"
    if sha256(config_path) != expected_files.get("adapter_config.json"):
        raise RuntimeError(f"cached adapter config bytes changed: {path}")
    config = _read(config_path)
    semantics = {
        "peft_type": config.get("peft_type"), "task_type": config.get("task_type"),
        "r": config.get("r"), "lora_alpha": config.get("lora_alpha"),
        "use_rslora": config.get("use_rslora"),
        "target_modules": sorted(map(str, config.get("target_modules") or ())),
    }
    if semantics != cached.get("adapter_config_semantics"):
        raise RuntimeError(f"cached adapter semantics changed: {path}")
    return dict(cached)


def output_identity(template: dict[str, Any], adapter_root: Path,
                    training_manifest: dict[str, Any],
                    cached_identities: dict[str, Any] | None = None) -> list[dict[str, Any]]:
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
    if cached_identities is not None and set(map(str, cached_identities)) != tasks:
        raise RuntimeError("cached adapter identity coverage mismatch")
    identities: dict[str, dict[str, Any]] = {}
    result = []
    for row in rows:
        task = str(row["task_id"])
        path = adapter_root / task / "depth_024"
        if task not in identities:
            identities[task] = (
                _cached_manifest_adapter_identity(path, records[task], cached_identities[task])
                if cached_identities is not None
                else _manifest_adapter_identity(path, records[task])
            )
        result.append({**row, "adapter_path": str(path), "adapter_identity": identities[task]})
    return result


def snapshots(run: Path, output_id: str) -> dict[str, Any]:
    # The historical orchestration helper and the production worker do not
    # share the same filename escaping convention (``:`` became ``__`` in the
    # former and ``_`` in the latter).  The worker receipt is the authoritative
    # artifact binding, so resolve the raw output through it instead of
    # reconstructing a path from the logical output id.
    matches = []
    for receipt_path in sorted((run / "OUTPUT_RECEIPTS").glob("*.json")):
        receipt = _read(receipt_path)
        if str(receipt.get("output_id")) == output_id:
            matches.append(receipt)
    if len(matches) != 1:
        raise RuntimeError(f"expected one COMPLETE receipt for {output_id}, found {len(matches)}")
    receipt = matches[0]
    if (receipt.get("status") != "COMPLETE" or receipt.get("target_blind") is not True
            or receipt.get("gold_loaded") is not False):
        raise RuntimeError(f"invalid target-blind completion receipt for {output_id}")
    raw_relative = Path(str(receipt.get("raw_output", "")))
    if raw_relative.is_absolute() or ".." in raw_relative.parts:
        raise RuntimeError(f"unsafe raw-output binding for {output_id}")
    raw_path = run / raw_relative
    if not raw_path.is_file():
        raise RuntimeError(f"receipt-bound raw output missing for {output_id}: {raw_relative}")
    raw = _read(raw_path)
    raw_identity = raw.get("output") if isinstance(raw.get("output"), dict) else {}
    if str(raw_identity.get("output_id")) != output_id or raw.get("gold_loaded") is not False:
        raise RuntimeError(f"receipt/raw identity mismatch for {output_id}")
    result = {}
    for cell in raw["cells"].values():
        checkpoint = next(row for row in cell["checkpoints"] if int(row["checkpoint_requested"]) == 1024)
        result[cell["augmentation_id"]] = checkpoint["candidate_pool_snapshot"]
    return result


def b1_policy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the exact Batch1 policy for the selected parity rows.

    Historical policy labels include conservative variants, so enumerating a
    hand-maintained profile list can reject a valid frozen cohort before the
    second canary runs.
    """
    return {"profiles": {
        str(row["profile"]): {
            "primary": {"resident_capacity": 1, "physical_batch_ceiling": 1},
            "fallbacks": [],
        }
        for row in rows
    }}


def write_throughput_report(run: Path) -> None:
    """Summarize target-blind runtime costs without opening evaluation Gold."""
    rows = []
    for receipt_path in sorted((run / "OUTPUT_RECEIPTS").glob("*.json")):
        receipt = _read(receipt_path)
        if receipt.get("status") != "COMPLETE":
            continue
        raw_path = run / str(receipt["raw_output"])
        raw = _read(raw_path)
        timings = dict(raw.get("timings") or receipt.get("timings") or {})
        scheduler = dict(raw.get("scheduler") or {})
        rows.append({
            "output_id": str(receipt["output_id"]),
            "execution_mode": str(raw.get("execution_mode") or "ROLLING_READY_CELL"),
            "adapter_stage_seconds": float(timings.get("adapter_stage_seconds", 0.0)),
            "model_load_seconds": float(timings.get("model_load_seconds", 0.0)),
            "adapter_swap_seconds": float(timings.get("adapter_swap_seconds", 0.0)),
            "search_seconds": float(timings.get("search_seconds", raw.get("wall_seconds", 0.0))),
            "artifact_write_seconds": float(timings.get("artifact_write_seconds", 0.0)),
            "physical_forwards": int(scheduler.get("physical_forwards", 0)),
            "mean_active_lanes": float(scheduler.get("mean_active_lanes", 0.0)),
        })
    fixed = [row for row in rows if row["execution_mode"] == "FIXED4_D1_INDEPENDENT_LANES"]
    def total(field: str) -> float:
        return sum(float(row[field]) for row in fixed)
    measured = (
        total("adapter_stage_seconds") + total("model_load_seconds")
        + total("adapter_swap_seconds") + total("search_seconds")
        + total("artifact_write_seconds")
    )
    report = {
        "status": "COMPLETE" if len(rows) == 89 else "PARTIAL",
        "target_blind": True,
        "gold_loaded": False,
        "completed_outputs": len(rows),
        "execution_mode_counts": {
            mode: sum(row["execution_mode"] == mode for row in rows)
            for mode in sorted({row["execution_mode"] for row in rows})
        },
        "fixed4_outputs": len(fixed),
        "fixed4_totals_seconds": {
            field: total(field) for field in (
                "adapter_stage_seconds", "model_load_seconds", "adapter_swap_seconds",
                "search_seconds", "artifact_write_seconds"
            )
        },
        "fixed4_mean_active_lanes": (
            statistics.mean(row["mean_active_lanes"] for row in fixed) if fixed else None
        ),
        "fixed4_mean_search_seconds": (
            statistics.mean(row["search_seconds"] for row in fixed) if fixed else None
        ),
        "measured_pipeline_seconds": measured,
        "next_speed_priorities": [
            {
                "rank": 1,
                "change": "persistent base_plus_V7 model with exact task-adapter state swap",
                "upper_bound_seconds": total("model_load_seconds"),
                "gate": "two-task load-vs-swap native-logit and candidate parity before production",
            },
            {
                "rank": 2,
                "change": "reuse hash-verified persistent adapter files without per-output staging copy",
                "upper_bound_seconds": total("adapter_stage_seconds"),
                "gate": "source SHA256 equals runtime SHA256 and no mutable staging path",
            },
            {
                "rank": 3,
                "change": "overlap gzip and hashing with the next GPU output",
                "upper_bound_seconds": total("artifact_write_seconds"),
                "gate": "bounded one-output CPU queue with atomic publish and identical SHA256",
            },
            {
                "rank": 4,
                "change": "evaluate fixed B8 only after the fixed4 run",
                "upper_bound_seconds": None,
                "gate": "new short/long capacity and semantic-parity canary; current D1 engine is validated only to four lanes",
            },
        ],
        "rows": rows,
    }
    _atomic_json(run / "TARGET_BLIND_THROUGHPUT_REPORT.json", report)


def frozen_generation_ready(run: Path, *, experiment: str, expected: int,
                            allow_operational_receipt_drift: bool = False) -> bool:
    """Return whether an existing immutable generation can be reused safely.

    A missing freeze means generation may proceed.  A present but invalid
    freeze is never treated as an incomplete run: it fails closed so the
    worker cannot overwrite evidence that had already been declared frozen.
    """
    freeze_path = run / "GENERATION_FREEZE.json"
    if not freeze_path.is_file():
        return False
    freeze = _read(freeze_path)
    required = {
        "experiment": experiment,
        "status": "FROZEN",
        "target_blind": True,
        "gold_loaded": False,
        "raw_count": expected,
    }
    if any(freeze.get(key) != value for key, value in required.items()):
        raise RuntimeError(f"existing generation freeze contract mismatch: {run}")

    ledger_path = run / "GENERATION_HASHES.json"
    verification_path = run / "GENERATION_HASH_VERIFICATION.json"
    if not ledger_path.is_file() or not verification_path.is_file():
        raise RuntimeError(f"existing generation freeze lacks hash evidence: {run}")
    ledger = _read(ledger_path)
    if _sha_value(ledger) != freeze.get("ledger_sha256"):
        raise RuntimeError(f"existing generation freeze ledger identity mismatch: {run}")
    recorded_verification = _read(verification_path)
    if recorded_verification.get("status") != "PASS":
        raise RuntimeError(f"existing generation recorded verification failed: {run}")
    verification = _verify(run, ledger_path)
    mismatches = list(verification.get("mismatches") or ())
    operational_drift_only = (
        allow_operational_receipt_drift
        and len(mismatches) == 1
        and mismatches[0].get("path") == "OOM_FALLBACK_RECEIPTS.csv"
    )
    if verification.get("status") != "PASS" and not operational_drift_only:
        raise RuntimeError(f"existing generation bytes no longer match freeze: {run}")

    artifact_counts = {
        name: sum(path.is_file() for path in (run / name).glob("*"))
        for name in ("RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS")
    }
    if any(count != expected for count in artifact_counts.values()):
        raise RuntimeError(
            f"existing generation frozen artifact counts mismatch: {artifact_counts} expected={expected}"
        )
    receipts = [_read(path) for path in sorted((run / "OUTPUT_RECEIPTS").glob("*.json"))]
    if any(
        receipt.get("status") != "COMPLETE"
        or receipt.get("target_blind") is not True
        or receipt.get("gold_loaded") is not False
        for receipt in receipts
    ):
        raise RuntimeError(f"existing generation has invalid completion receipt: {run}")
    if len({str(receipt.get("output_id")) for receipt in receipts}) != expected:
        raise RuntimeError(f"existing generation receipt identity mismatch: {run}")
    scientific_paths = {
        str(path.relative_to(run))
        for directory in ("RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS")
        for path in (run / directory).glob("*") if path.is_file()
    }
    if not scientific_paths.issubset(set(ledger.get("files") or {})):
        raise RuntimeError(f"existing generation ledger omits scientific artifacts: {run}")
    return True


def ensure_b1_generation(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]],
                         policy: dict[str, Any]) -> str:
    experiment = EXPERIMENT + "_B1_GATE"
    if frozen_generation_ready(
        run, experiment=experiment, expected=len(outputs),
        # A prior orchestration re-entry overwrote only this attempt ledger
        # after the immutable freeze.  It contains no model output, token,
        # candidate, checkpoint, or EOS evidence.  The exact discrepancy is
        # retained outside the frozen directory while all scientific bytes
        # remain governed by the original ledger.
        allow_operational_receipt_drift=True,
    ):
        verification = _verify(run, run / "GENERATION_HASHES.json")
        _atomic_json(run.parent / "B1_FREEZE_REUSE_AUDIT.json", {
            "status": "REUSED_FROZEN_SCIENTIFIC_ARTIFACTS",
            "experiment": experiment,
            "target_blind": True,
            "gold_loaded": False,
            "scientific_artifact_classes_verified": [
                "RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS",
            ],
            "full_ledger_status": verification.get("status"),
            "allowed_post_freeze_operational_drift": verification.get("mismatches") or [],
            "allowed_paths": ["OOM_FALLBACK_RECEIPTS.csv"],
            "reason": "prior rejected orchestration re-entry rewrote attempt telemetry only",
        })
        return "REUSED_FROZEN"
    _run_surface(args, run, outputs, depth=24, experiment=experiment, policy=policy)
    _freeze_generation(run, experiment, len(outputs))
    return "GENERATED"


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
    records = dict(training_manifest.get("records") or {})
    stat_snapshot = _adapter_stat_snapshot(args.adapter_root, records)
    identity_path = args.output / "ADAPTER_IDENTITIES.json"
    reuse_path = args.output / "ADAPTER_IDENTITY_REUSE_RECEIPT.json"
    cached_identities = _read(identity_path) if identity_path.is_file() else None
    cache_mode = "FULL_SHA256_VERIFICATION"
    prior_reuse = _read(reuse_path) if reuse_path.is_file() else None
    if cached_identities is not None:
        if prior_reuse is not None and prior_reuse.get("adapter_stat_snapshot") != stat_snapshot:
            cached_identities = None
        else:
            cache_mode = "REUSED_PRIOR_EXACT_SHA256_IDENTITIES_WITH_STABLE_FILE_METADATA"
    outputs = output_identity(template, args.adapter_root, training_manifest, cached_identities)
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
        "batch_policy_estimand": "HISTORICAL_ROOT_AWARE_PRODUCTION_POLICY",
        "b1_role": "NONBLOCKING_RUNTIME_SENSITIVITY_DIAGNOSTIC",
        "worker_python": str(args.worker_python), "solutions_argument_available": False,
    }
    if contract["foundation_adapter_sha256"] != V7_ADAPTER_SHA256:
        raise RuntimeError("frozen V7 adapter identity mismatch")
    # Do not rewrite a prior scientific binding during an operational resume.
    # A later executor amendment is additive and records exactly which outputs
    # were already complete under the original rolling implementation.
    for name in ("PRE_GOLD_CONTRACT.json", "CONTRACT.json"):
        path = args.output / name
        if not path.is_file():
            _atomic_json(path, contract)
    identity_payload = {
        task: next(row["adapter_identity"] for row in outputs if row["task_id"] == task)
        for task in sorted({row["task_id"] for row in outputs})
    }
    _atomic_json(identity_path, identity_payload)
    _atomic_json(reuse_path, {
        "status": "PASS",
        "mode": cache_mode,
        "training_manifest_sha256": sha256(args.training_manifest),
        "adapter_identities_sha256": sha256(identity_path),
        "adapter_stat_snapshot": stat_snapshot,
        "full_rehash_required_if": "manifest, file set, size, mtime, inode, or device changes",
    })

    # Deterministically choose one shortest and one longest output.  The
    # historical profile already binds their exact roots; production tokenizes
    # again and records the actual per-view root classes.
    ordered = sorted(outputs, key=lambda row: (int(row["root_length_max"]), row["output_id"]))
    smoke = [ordered[0], ordered[-1]]
    smoke_b1 = args.output / "B1_PARITY"
    smoke_policy = b1_policy(smoke)
    ensure_b1_generation(args, smoke_b1, smoke, smoke_policy)

    # Generate the same two outputs with the original rolling production path.
    # On a resumed run these receipts are reused byte-for-byte.
    _run_surface(args, args.output, smoke, depth=24, experiment=EXPERIMENT, policy=policy)
    def compare_parity() -> list[dict[str, Any]]:
        rows = []
        for row in smoke:
            b1 = snapshots(smoke_b1, row["output_id"])
            production = snapshots(args.output, row["output_id"])
            exact = b1 == production
            rows.append({
                "output_id": row["output_id"], "profile": row["profile"], "exact": exact,
                "b1_sha256": __import__("hashlib").sha256(
                    json.dumps(b1, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
                "production_sha256": __import__("hashlib").sha256(
                    json.dumps(production, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            })
        return rows

    parity_rows = compare_parity()
    exact_parity = all(row["exact"] for row in parity_rows)
    gate = {"status": "PASS" if exact_parity else "CONTINUE_WITH_RUNTIME_SENSITIVITY",
            "criterion": "exact R1024 parsed/canonical candidate-pool agreement on fixed short and long outputs",
            "rows": parity_rows,
            "blocking": False,
            "formal_execution_policy": "HISTORICAL_ROOT_AWARE_PRODUCTION_POLICY",
            "interpretation": (
                "EXACT_PARITY"
                if exact_parity
                else "BATCH1_DIFFERS_FROM_DEPLOYMENT_ALIGNED_BATCHED_CANDIDATE_POOL"
            ),
            "owner_direction": "Batch1 is diagnostic; Kaggle-aligned Batch7/8 remains primary",
            "target_blind": True,
            "gold_loaded": False}
    _atomic_json(args.output / "B1_PARITY_GATE.json", gate)

    # Switch only the remaining work to the historical cross-cell D1 engine.
    # Four independent searches share each physical forward; divergence is
    # expected and completed lanes become inert PAD lanes.  Preserve the
    # already-generated rolling outputs and bind the mixed implementation
    # explicitly rather than silently rewriting their execution identity.
    completed_before_switch = []
    for receipt_path in sorted((args.output / "OUTPUT_RECEIPTS").glob("*.json")):
        receipt = _read(receipt_path)
        if receipt.get("status") == "COMPLETE":
            completed_before_switch.append(str(receipt.get("output_id")))
    fixed_worker = ROOT / "scripts" / "run_eval60_fixed4_d1_worker_v1.py"
    args.worker_entrypoint = fixed_worker
    fixed_canary = args.output.parent / f"{args.output.name}_FIXED4_CANARY_V1"
    fixed_canary.mkdir(parents=True, exist_ok=True)
    fixed_canary_experiment = EXPERIMENT + "_FIXED4_CANARY"
    if not frozen_generation_ready(
        fixed_canary, experiment=fixed_canary_experiment, expected=len(smoke)
    ):
        _run_surface(
            args, fixed_canary, smoke, depth=24,
            experiment=fixed_canary_experiment, policy=policy,
        )
        _freeze_generation(fixed_canary, fixed_canary_experiment, len(smoke))
    fixed_rows = []
    for row in smoke:
        rolling = snapshots(args.output, row["output_id"])
        fixed = snapshots(fixed_canary, row["output_id"])
        fixed_rows.append({
            "output_id": row["output_id"],
            "profile": row["profile"],
            "exact_candidate_pool_parity": rolling == fixed,
            "rolling_sha256": _sha_json(rolling),
            "fixed4_sha256": _sha_json(fixed),
            "fixed4_cell_count": len(fixed),
            "fixed4_all_aug8_present": set(fixed) == set(AUG8),
        })
    fixed_mechanical_pass = all(
        row["fixed4_cell_count"] == 8 and row["fixed4_all_aug8_present"]
        for row in fixed_rows
    )
    fixed_gate = {
        "status": "PASS" if fixed_mechanical_pass else "FAIL",
        "criterion": "short and long canaries complete all eight exact-root fixed4 lanes at R1024",
        "candidate_pool_parity_is_diagnostic": True,
        "candidate_pool_parity_blocking": False,
        "reason": "the deployed batched numerical route is the estimand; B1/rolling are sensitivity diagnostics",
        "rows": fixed_rows,
        "target_blind": True,
        "gold_loaded": False,
    }
    _atomic_json(args.output / "FIXED4_CANARY_GATE.json", fixed_gate)
    if not fixed_mechanical_pass:
        raise RuntimeError("fixed4 short/long mechanical canary failed")
    persistent_canary = args.output.parent / f"{args.output.name}_FIXED4_PERSISTENT_CANARY_V1"
    persistent_canary.mkdir(parents=True, exist_ok=True)
    persistent_canary_experiment = EXPERIMENT + "_FIXED4_PERSISTENT_CANARY"
    if not frozen_generation_ready(
        persistent_canary, experiment=persistent_canary_experiment, expected=len(smoke)
    ):
        _run_persistent_surface(
            args, persistent_canary, smoke, depth=24,
            experiment=persistent_canary_experiment,
        )
        _freeze_generation(persistent_canary, persistent_canary_experiment, len(smoke))
    persistent_rows = []
    for row in smoke:
        standalone = snapshots(fixed_canary, row["output_id"])
        persistent_result = snapshots(persistent_canary, row["output_id"])
        persistent_rows.append({
            "output_id": row["output_id"],
            "profile": row["profile"],
            "exact_candidate_pool_parity": standalone == persistent_result,
            "standalone_sha256": _sha_json(standalone),
            "persistent_sha256": _sha_json(persistent_result),
        })
    persistent_pass = all(row["exact_candidate_pool_parity"] for row in persistent_rows)
    _atomic_json(args.output / "PERSISTENT_MODEL_CANARY_GATE.json", {
        "status": "PASS" if persistent_pass else "FAIL",
        "criterion": "exact R1024 candidate-pool parity for fixed short and long outputs",
        "rows": persistent_rows,
        "target_blind": True,
        "gold_loaded": False,
    })
    if not persistent_pass:
        raise RuntimeError("persistent model/task-adapter swap canary failed exact parity")
    _atomic_json(args.output / "EXECUTION_POLICY_AMENDMENT_FIXED4_V1.json", {
        "status": "ACTIVE",
        "source_commit": contract["source_commit"],
        "worker_entrypoint": str(fixed_worker.relative_to(ROOT)),
        "execution_mode": "FIXED4_D1_INDEPENDENT_LANES",
        "physical_width": 4,
        "logical_budgets": "independent R1024 per AUG8 cell",
        "root_grouping": "two exact-root groups of four",
        "inactive_lane_policy": "PAD physical slot with no logical state mutation",
        "completed_before_switch": completed_before_switch,
        "completed_before_switch_count": len(completed_before_switch),
        "remaining_outputs_use_fixed4": True,
        "scientific_settings_changed": False,
        "candidate_generation_route_changed": True,
        "fixed4_canary_gate": str(args.output / "FIXED4_CANARY_GATE.json"),
        "persistent_model_canary_gate": str(args.output / "PERSISTENT_MODEL_CANARY_GATE.json"),
        "target_blind": True,
        "gold_loaded": False,
    })

    _run_persistent_surface(args, args.output, outputs, depth=24, experiment=EXPERIMENT)
    write_throughput_report(args.output)
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
