#!/usr/bin/env python3
"""Execute one frozen Stage-A arm from its immutable schedule.

This worker never samples a curriculum.  It reconstructs every scheduled
TRAIN episode from the frozen source, verifies its token/label identity, and
charges only elapsed time after the first successful optimizer step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_data.pipeline import IGNORE_INDEX, task_to_sample


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def token_hash(ids, labels) -> str:
    return hashlib.sha256(json.dumps({"input_ids": ids, "labels": labels}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf8", newline="\n")
    os.replace(temporary, path)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_work(config: dict) -> list[dict]:
    train = Path(config["train_path"])
    if sha(train) != config["train_sha256"]:
        raise RuntimeError("TRAIN_IDENTITY_FAIL")
    source_rows = [json.loads(line) for line in train.read_text(encoding="utf8").splitlines() if line]
    if len(source_rows) != int(config["train_rows"]):
        raise RuntimeError("TRAIN_ROW_COUNT_FAIL")
    source = {x["episode_id"]: x for x in source_rows}
    if len(source) != len(source_rows):
        raise RuntimeError("TRAIN_EPISODE_ID_DUPLICATE")
    schedule_path = resolve(ROOT, config["schedule_path"])
    if sha(schedule_path) != config["schedule_sha256"]:
        raise RuntimeError("SCHEDULE_IDENTITY_FAIL")
    schedule = json.loads(schedule_path.read_text(encoding="utf8"))
    entries = schedule.get("episodes")
    if not isinstance(entries, list) or len(entries) != 400:
        raise RuntimeError("SCHEDULE_SHAPE_FAIL")
    work = []
    for expected in entries:
        row = source.get(expected.get("episode_id"))
        if row is None or row.get("split") != "TRAIN":
            raise RuntimeError("SCHEDULE_EPISODE_NOT_TRAIN")
        sample = task_to_sample({"source_id": row["episode_id"], **row["task"]})
        labels = sample["labels"]
        if not any(v != IGNORE_INDEX for v in labels):
            raise RuntimeError("ZERO_SUPERVISION")
        observed = {"sequence_length": len(sample["input_ids"]), "supervised_token_count": sample["assistant_token_count"], "token_label_sha256": token_hash(sample["input_ids"], labels)}
        if any(observed[k] != expected.get(k) for k in observed):
            raise RuntimeError("SCHEDULE_TOKEN_BINDING_FAIL")
        # Effective role is intentionally retained from the frozen schedule;
        # same-color source rows are explicitly assigned to RETENTION_TRAIN.
        work.append({"ids": sample["input_ids"], "labels": labels, "tokens": observed["sequence_length"], "supervised": observed["supervised_token_count"], "episode_id": row["episode_id"], "role": expected["role"], "family": expected["family"]})
    return work


REQUIRED_PRE_MODEL_CONFIG_KEYS = (
    "protocol_id", "round_id", "arm", "output_root", "checkpoint_manifest_path",
    "checkpoint_manifest_sha256", "train_path", "train_sha256", "train_rows",
    "schedule_path", "schedule_sha256", "loss_aggregation_policy",
    "loss_coefficient_ledger_path", "loss_coefficient_ledger_sha256",
    "optimizer_steps", "gradient_accumulation", "final_audit_opened",
)


GOVERNOR_REVIEW_BINDING_KEYS = (
    "reviewed_brief_sha256", "protocol_id", "round_id", "config_sha256",
    "protocol_sha256", "worker_sha256", "coefficient_ledger_sha256",
    "schedule_sha256", "train_sha256", "checkpoint_manifest_sha256",
    "output_root", "reservation_seconds", "source_commit",
    "standing_user_gpu_authorization",
)


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def launch_identity(config: dict, config_path: Path, protocol_path: Path, reviewed_brief: Path) -> dict:
    return {
        "reviewed_brief_sha256": sha(reviewed_brief),
        "protocol_id": config["protocol_id"],
        "round_id": config["round_id"],
        "config_sha256": sha(config_path),
        "protocol_sha256": sha(protocol_path),
        "worker_sha256": sha(Path(__file__).resolve()),
        "coefficient_ledger_sha256": config["loss_coefficient_ledger_sha256"],
        "schedule_sha256": config["schedule_sha256"],
        "train_sha256": config["train_sha256"],
        "checkpoint_manifest_sha256": config["checkpoint_manifest_sha256"],
        "output_root": config["output_root"],
        "reservation_seconds": float(config["reservation_seconds"]),
        "source_commit": git_head(),
        "standing_user_gpu_authorization": True,
    }


def authenticate_governor_review(config: dict, config_path: Path, protocol_path: Path, reviewed_brief: Path, review_path: Path) -> dict:
    """Fail closed unless an external Director CONTINUE binds this exact launch."""
    if config.get("execution_authorized") is not False or config.get("external_governor_review_required") is not True:
        raise RuntimeError("MUTABLE_CONFIG_AUTHORIZATION_FORBIDDEN")
    if not review_path.is_file():
        raise RuntimeError("GOVERNOR_REVIEW_MISSING")
    try:
        review = json.loads(review_path.read_text(encoding="utf8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("GOVERNOR_REVIEW_MALFORMED") from exc
    if review.get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("GOVERNOR_REVIEW_DECISION_NOT_CONTINUE")
    binding = review.get("launch_binding")
    if not isinstance(binding, dict) or any(key not in binding for key in GOVERNOR_REVIEW_BINDING_KEYS):
        raise RuntimeError("GOVERNOR_REVIEW_BINDING_INCOMPLETE")
    expected = launch_identity(config, config_path, protocol_path, reviewed_brief)
    for key, value in expected.items():
        if binding.get(key) != value:
            raise RuntimeError(f"GOVERNOR_REVIEW_BINDING_MISMATCH:{key}")
    return {
        "review_sha256": sha(review_path),
        "reviewed_brief_sha256": expected["reviewed_brief_sha256"],
        "source_commit": expected["source_commit"],
        "status": "GOVERNOR_REVIEW_AUTHENTICATED",
    }


def validate_pre_model_config(config: dict, run_root_raw: str) -> dict:
    """Run every bound CPU identity check before importing model libraries."""
    missing = [key for key in REQUIRED_PRE_MODEL_CONFIG_KEYS if key not in config]
    if missing:
        raise RuntimeError("REQUIRED_CONFIG_KEY_MISSING:" + ",".join(missing))
    if run_root_raw != config["output_root"]:
        raise RuntimeError("OUTPUT_ROOT_CONFIG_MISMATCH")
    if config.get("final_audit_opened") is not False:
        raise RuntimeError("FINAL_AUDIT_FORBIDDEN")
    manifest_path = resolve(ROOT, config["checkpoint_manifest_path"])
    if sha(manifest_path) != config["checkpoint_manifest_sha256"]:
        raise RuntimeError("CHECKPOINT_MANIFEST_FAIL")
    if int(config["optimizer_steps"]) != 100 or int(config["gradient_accumulation"]) != 4:
        raise RuntimeError("FROZEN_RECIPE_SHAPE_FAIL")
    work = load_work(config)
    if len(work) != int(config["optimizer_steps"]) * int(config["gradient_accumulation"]):
        raise RuntimeError("SCHEDULE_STEP_ALIGNMENT_FAIL")
    coefficients = load_loss_coefficients(config, work)
    if len(coefficients) != int(config["optimizer_steps"]):
        raise RuntimeError("LOSS_COEFFICIENT_STEP_COUNT_FAIL")
    return {
        "status": "PRE_MODEL_CPU_VALIDATION_PASS",
        "train_path": config["train_path"],
        "train_sha256": config["train_sha256"],
        "train_rows": int(config["train_rows"]),
        "schedule_sha256": config["schedule_sha256"],
        "loss_coefficient_ledger_sha256": config["loss_coefficient_ledger_sha256"],
        "scheduled_episodes": len(work),
        "optimizer_steps": len(coefficients),
        "per_step_scalar_scale": [sum(step) for step in coefficients],
        "output_root": run_root_raw,
        "final_audit_opened": False,
    }


def load_loss_coefficients(config: dict, work: list[dict]) -> list[list[float]]:
    """Return one frozen scalar coefficient for every row of every step.

    The historical default remains token-proportional within an accumulation
    group.  A future, separately reviewed family-balanced run must bind an
    immutable ledger that names every scheduled episode and preserves scalar
    scale one per optimizer step.  This function is deliberately CPU-only so
    its identity checks can run before a model is loaded.
    """
    accumulation = int(config["gradient_accumulation"])
    groups = [work[offset:offset + accumulation] for offset in range(0, len(work), accumulation)]
    policy = config.get("loss_aggregation_policy", "TOKEN_PROPORTIONAL_PER_STEP_V1")
    if policy == "TOKEN_PROPORTIONAL_PER_STEP_V1":
        return [[row["supervised"] / sum(item["supervised"] for item in group) for row in group] for group in groups]
    if policy != "FAMILY_BALANCED_GLOBAL_IPF_V1":
        raise RuntimeError("LOSS_AGGREGATION_POLICY_UNKNOWN")
    ledger_path = resolve(ROOT, config["loss_coefficient_ledger_path"])
    if sha(ledger_path) != config["loss_coefficient_ledger_sha256"]:
        raise RuntimeError("LOSS_COEFFICIENT_LEDGER_IDENTITY_FAIL")
    ledger = json.loads(ledger_path.read_text(encoding="utf8"))
    if ledger.get("policy_id") != policy or ledger.get("schedule_sha256") != config["schedule_sha256"]:
        raise RuntimeError("LOSS_COEFFICIENT_LEDGER_BINDING_FAIL")
    steps = ledger.get("per_optimizer_step")
    if not isinstance(steps, list) or len(steps) != len(groups):
        raise RuntimeError("LOSS_COEFFICIENT_LEDGER_SHAPE_FAIL")
    coefficients: list[list[float]] = []
    for index, (group, step) in enumerate(zip(groups, steps), start=1):
        entries = step.get("entries")
        if step.get("optimizer_step") != index or not isinstance(entries, list) or len(entries) != len(group):
            raise RuntimeError("LOSS_COEFFICIENT_STEP_BINDING_FAIL")
        if [entry.get("episode_id") for entry in entries] != [row["episode_id"] for row in group]:
            raise RuntimeError("LOSS_COEFFICIENT_EPISODE_ORDER_FAIL")
        values = [float(entry.get("balanced_loss_coefficient")) for entry in entries]
        if any(value <= 0.0 or not value < 1.0 for value in values) or abs(sum(values) - 1.0) > 1e-10:
            raise RuntimeError("LOSS_COEFFICIENT_SCALE_FAIL")
        coefficients.append(values)
    return coefficients


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--validate-pre-model", action="store_true")
    parser.add_argument("--governor-review", type=Path)
    parser.add_argument("--reviewed-brief", type=Path)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf8"))
    pre_model = validate_pre_model_config(config, args.run_root)
    if args.validate_pre_model:
        print(json.dumps(pre_model, sort_keys=True))
        return 0
    run = Path(args.run_root)
    if run.exists():
        raise RuntimeError("FRESH_OUTPUT_REQUIRED")
    if config.get("external_governor_review_required") is True:
        if not args.governor_review or not args.reviewed_brief or not args.protocol:
            raise RuntimeError("GOVERNOR_REVIEW_REQUIRED")
        authorization = authenticate_governor_review(config, args.config, args.protocol, args.reviewed_brief, args.governor_review)
    elif config.get("execution_authorized") is not True:
        raise RuntimeError("SCIENTIFIC_TRAINING_NOT_AUTHORIZED")
    else:
        authorization = {"status": "LEGACY_CONFIG_AUTHORIZATION"}
    manifest_path = resolve(ROOT, config["checkpoint_manifest_path"])
    work = load_work(config)
    loss_coefficients = load_loss_coefficients(config, work)
    run.mkdir(parents=True)
    runtime, checkpoints = run / "runtime", run / "checkpoints"
    runtime.mkdir(); checkpoints.mkdir()
    source_commit = git_head()
    terminal = {"schema_version": 1, "protocol_id": config["protocol_id"], "round_id": config["round_id"], "arm": config["arm"], "scientific_training_started": False, "optimizer_steps": 0, "processed_tokens": 0, "final_audit_accessed": False, "source_commit": source_commit}
    atomic(runtime / "FROZEN_RUN_BINDING.json", {"config_sha256": sha(args.config), "schedule_sha256": config["schedule_sha256"], "checkpoint_manifest_sha256": config["checkpoint_manifest_sha256"], "run_root": str(run), "cpu_schedule_verified": True, "governor_review": authorization})
    model = None
    start_ns = None
    try:
        import torch
        import bitsandbytes as bnb
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError("CUDA_BF16_UNAVAILABLE")
        manifest = json.loads(manifest_path.read_text(encoding="utf8"))
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        model = AutoModelForCausalLM.from_pretrained(manifest["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        model = PeftModel.from_pretrained(model, manifest["adapter_path"], is_trainable=True)
        params = [p for p in model.parameters() if p.requires_grad]
        if not params or any(p.requires_grad for name, p in model.named_parameters() if "lora_" not in name):
            raise RuntimeError("LORA_PARTITION_FAIL")
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.train()
        optimizer = bnb.optim.PagedAdamW8bit(params, lr=float(config["learning_rate"]))
        start_ns = time.monotonic_ns()
        deadline = time.monotonic() + float(config["reservation_seconds"])
        done = supervised = steps = 0
        capped = False
        accumulation = int(config["gradient_accumulation"])
        for step_index, offset in enumerate(range(0, len(work), accumulation)):
            if time.monotonic() >= deadline:
                capped = True; break
            group = work[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            for row, coefficient in zip(group, loss_coefficients[step_index]):
                ids = torch.tensor([row["ids"]], device="cuda:0", dtype=torch.long)
                labels = torch.tensor([row["labels"]], device="cuda:0", dtype=torch.long)
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("NONFINITE_LOSS")
                (loss * coefficient).backward()
                done += row["tokens"]; supervised += row["supervised"]
            if time.monotonic() >= deadline:
                capped = True; break
            for group_config in optimizer.param_groups:
                group_config["lr"] = float(config["learning_rate"]) * min(1.0, (steps + 1) / 3)
            optimizer.step(); steps += 1
            atomic(runtime / "TRAINING_PROGRESS.json", {"status": "RUNNING", "optimizer_steps": steps, "processed_tokens": done, "supervised_tokens": supervised, "round_id": config["round_id"]})
            if done >= 250000 and not (checkpoints / "tokens_250000").exists():
                model.save_pretrained(checkpoints / "tokens_250000")
        model.save_pretrained(checkpoints / ("cap_finalize" if capped else "final"))
        terminal.update({"status": "CAP_REACHED_CHECKPOINTED" if capped else "COMPLETED", "scientific_training_started": steps > 0, "optimizer_steps": steps, "processed_tokens": done, "supervised_tokens": supervised, "no_optimizer_after_cap": capped, "starting_adapter_path": manifest["adapter_path"]})
    except Exception as exc:
        terminal.update({"status": "PRE_OPTIMIZER_FAILURE" if start_ns is None else "EARLY_RUNTIME_FAILURE", "error": f"{type(exc).__name__}:{exc}", "traceback": traceback.format_exc(limit=4)})
    finally:
        stop_ns = time.monotonic_ns()
        seconds = 0.0 if start_ns is None or not terminal["optimizer_steps"] else (stop_ns - start_ns) / 1e9
        terminal["scientific_gpu_training_seconds"] = seconds
        atomic(runtime / "TERMINAL_RECEIPT.json", terminal)
        with (run / "ARC2_CUMULATIVE_NEW_GPU_TRAINING_LEDGER.jsonl").open("a", encoding="utf8") as ledger:
            ledger.write(json.dumps({"schema_version": 1, "record_type": "GPU_OPTIMIZER_INTERVAL", "round_id": config["round_id"], "gpu_optimizer_seconds": seconds, "receipt_sha256": sha(runtime / "TERMINAL_RECEIPT.json")}, sort_keys=True) + "\n")
        if model is not None:
            try:
                import torch
                del model; torch.cuda.empty_cache()
            except Exception:
                pass
    return 0 if terminal.get("status") in {"COMPLETED", "CAP_REACHED_CHECKPOINTED"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
