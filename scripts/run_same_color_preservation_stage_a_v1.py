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
    source = {x["episode_id"]: x for x in (json.loads(line) for line in train.read_text(encoding="utf8").splitlines() if line)}
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf8"))
    run = args.run_root
    if run.exists():
        raise RuntimeError("FRESH_OUTPUT_REQUIRED")
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
    run.mkdir(parents=True)
    runtime, checkpoints = run / "runtime", run / "checkpoints"
    runtime.mkdir(); checkpoints.mkdir()
    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    terminal = {"schema_version": 1, "protocol_id": config["protocol_id"], "round_id": config["round_id"], "arm": config["arm"], "scientific_training_started": False, "optimizer_steps": 0, "processed_tokens": 0, "final_audit_accessed": False, "source_commit": source_commit}
    atomic(runtime / "FROZEN_RUN_BINDING.json", {"config_sha256": sha(args.config), "schedule_sha256": config["schedule_sha256"], "checkpoint_manifest_sha256": config["checkpoint_manifest_sha256"], "run_root": str(run), "cpu_schedule_verified": True})
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
        for offset in range(0, len(work), accumulation):
            if time.monotonic() >= deadline:
                capped = True; break
            group = work[offset:offset + accumulation]
            optimizer.zero_grad(set_to_none=True)
            denom = sum(x["supervised"] for x in group)
            for row in group:
                ids = torch.tensor([row["ids"]], device="cuda:0", dtype=torch.long)
                labels = torch.tensor([row["labels"]], device="cuda:0", dtype=torch.long)
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("NONFINITE_LOSS")
                (loss * (row["supervised"] / denom)).backward()
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
