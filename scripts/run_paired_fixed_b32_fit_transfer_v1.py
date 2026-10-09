#!/usr/bin/env python3
"""No-update fixed-B32 TRAIN-NLL worker for a prospective H1-versus-H3 study.

CPU self-test is deliberately the only mode usable before a later launch
authorization binds this package to an isolated remote output root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
import subprocess
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
IGNORE_INDEX = -100
PAD_TOKEN_ID = 13
WEAK = ("connected components", "inside/contains", "difference", "width", "orientation")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def load_rows(train_path: Path, manifest: dict, expected_rows: int = 400) -> list[list[dict]]:
    from training_data.pipeline import task_to_sample

    source = {row["episode_id"]: row for row in (json.loads(line) for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip())}
    all_ids, groups = set(), []
    for batch in manifest["batches"]:
        group = []
        for member in batch["members"]:
            row = source.get(member["episode_id"])
            if row is None or row.get("split") != "TRAIN":
                raise RuntimeError("TRAIN_BINDING_FAIL:" + member["episode_id"])
            sample = task_to_sample({"source_id": row["episode_id"], **row["task"]})
            if len(sample["input_ids"]) != member["sequence_length"]:
                raise RuntimeError("SEQUENCE_LENGTH_BINDING_FAIL:" + member["episode_id"])
            group.append({"episode_id": member["episode_id"], "family": row["family"], "schedule_index": member["schedule_index"], "ids": sample["input_ids"], "labels": sample["labels"]})
            all_ids.add(member["episode_id"])
        if len(group) != batch["effective_batch_size"]:
            raise RuntimeError("BATCH_SIZE_BINDING_FAIL")
        groups.append(group)
    if len(all_ids) != expected_rows:
        raise RuntimeError("ROW_BINDING_FAIL:" + str(expected_rows))
    return groups


def self_test(config: dict, train_path: Path) -> dict:
    manifest_path = ROOT / config["inputs"]["primary_manifest_path"]
    if sha(manifest_path) != config["inputs"]["primary_manifest_sha256"]:
        raise RuntimeError("PRIMARY_MANIFEST_SHA_MISMATCH")
    groups = load_rows(train_path, json.loads(manifest_path.read_text(encoding="utf-8")))
    rows = [row for group in groups for row in group]
    counts = {family: sum(row["family"] == family for row in rows) for family in WEAK}
    if counts != {family: 48 for family in WEAK}:
        raise RuntimeError("WEAK_FAMILY_COUNT_FAIL")
    return {"status": "PASS_CPU_ONLY_NO_MODEL_IMPORT", "rows": len(rows), "batches": len(groups), "weak_family_counts": counts}


def collate(group, torch):
    width = max(len(row["ids"]) for row in group)
    ids, labels, masks = [], [], []
    for row in group:
        pad = width - len(row["ids"])
        ids.append(row["ids"] + [PAD_TOKEN_ID] * pad)
        labels.append(row["labels"] + [IGNORE_INDEX] * pad)
        masks.append([1] * len(row["ids"]) + [0] * pad)
    return (torch.tensor(ids, device="cuda:0"), torch.tensor(masks, device="cuda:0"), torch.tensor(labels, device="cuda:0"))


def score(model, groups, torch, F, deadline, checkpoint_id, mode_id, repeat):
    raw = []
    with torch.no_grad():
        for batch_index, group in enumerate(groups):
            if time.monotonic() > deadline:
                raise RuntimeError("RUNTIME_CAP_REACHED")
            ids, masks, labels = collate(group, torch)
            logits = model(input_ids=ids, attention_mask=masks, use_cache=False).logits.float()
            loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.shape[-1]), labels[:, 1:].reshape(-1), ignore_index=IGNORE_INDEX, reduction="none").view_as(labels[:, 1:])
            for index, row in enumerate(group):
                valid = labels[index, 1:] != IGNORE_INDEX
                values = [float(value) for value in loss[index][valid].tolist()]
                raw.append({"checkpoint_id": checkpoint_id, "mode_id": mode_id, "repeat_index": repeat, "batch_index": batch_index, "episode_id": row["episode_id"], "family": row["family"], "schedule_index": row["schedule_index"], "row_nll_sum": sum(values), "supervised_token_count": len(values), "effective_batch_size": len(group), "max_sequence_length": len(ids[index]), "per_token_negative_log_likelihood": values})
    return raw


def validate_authorization(path: Path, config: Path) -> dict:
    auth = json.loads(path.read_text(encoding="utf-8"))
    if auth.get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("AUTHORIZATION_DECISION_NOT_CONTINUE")
    bindings = auth.get("bindings", {})
    expected = {"config_sha256": sha(config), "worker_sha256": sha(Path(__file__)), "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "runtime_cap_seconds": 1800}
    for key, value in expected.items():
        if bindings.get(key) != value:
            raise RuntimeError("AUTHORIZATION_BINDING_MISMATCH:" + key)
    return auth


def conclusion_stable(b32_deltas: dict, b1_deltas: dict, b32_conclusion: str, b1_conclusion: str) -> bool:
    """A numerical crosscheck cannot alter a checkpoint-delta sign or conclusion."""
    if b32_conclusion != b1_conclusion:
        return False
    for key, value in b32_deltas.items():
        other = b1_deltas.get(key)
        if other is None or (value < 0) != (other < 0):
            return False
    return True


def execute(config_path: Path, config: dict, train_path: Path, authorization: Path, out: Path) -> dict:
    if out.exists():
        raise RuntimeError("FRESH_OUTPUT_REQUIRED")
    auth = validate_authorization(authorization, config_path)
    manifest = json.loads((ROOT / config["inputs"]["primary_manifest_path"]).read_text(encoding="utf-8"))
    groups = load_rows(train_path, manifest)
    cross_path = ROOT / config["inputs"]["b1_crosscheck_path"]
    if sha(cross_path) != config["inputs"]["b1_crosscheck_sha256"]:
        raise RuntimeError("B1_CROSSCHECK_SHA_MISMATCH")
    cross = json.loads(cross_path.read_text(encoding="utf-8"))
    cross_b32 = load_rows(train_path, {"batches": cross["batch32_batches"]}, expected_rows=32)
    cross_b1 = load_rows(train_path, {"batches": [{"effective_batch_size": 1, "members": [member]} for member in cross["batch32_batches"][0]["members"]]}, expected_rows=32)
    if sha(train_path) != config["inputs"]["train_sha256"]:
        raise RuntimeError("TRAIN_SHA_MISMATCH")
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA_BF16_UNAVAILABLE")
    deadline = time.monotonic() + config["execution"]["cap_seconds"]
    raw = []
    for checkpoint_id in config["inputs"]["checkpoint_order"]:
        condition = config["inputs"]["conditions"][checkpoint_id]
        manifest_path = Path(condition["checkpoint_manifest_path"])
        if sha(manifest_path) != condition["checkpoint_manifest_sha256"]:
            raise RuntimeError("CHECKPOINT_MANIFEST_SHA_MISMATCH:" + checkpoint_id)
        checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        adapter = Path(checkpoint["adapter_path"]) / "adapter_model.safetensors"
        if sha(adapter) != condition["adapter_sha256"]:
            raise RuntimeError("ADAPTER_SHA_MISMATCH:" + checkpoint_id)
        model = AutoModelForCausalLM.from_pretrained(checkpoint["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda:0")
        model = PeftModel.from_pretrained(model, checkpoint["adapter_path"], is_trainable=False)
        model.eval()
        try:
            for repeat in range(3):
                raw.extend(score(model, groups, torch, F, deadline, checkpoint_id, "PRIMARY_B32", repeat))
                raw.extend(score(model, cross_b32, torch, F, deadline, checkpoint_id, "CROSSCHECK_B32", repeat))
                raw.extend(score(model, cross_b1, torch, F, deadline, checkpoint_id, "CROSSCHECK_B1", repeat))
        finally:
            del model
            torch.cuda.empty_cache()
    out.mkdir(parents=True)
    raw_path = out / "RAW_PER_ROW_NLL.jsonl"
    raw_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in raw), encoding="utf-8")
    result = {"schema_version": 1, "protocol_id": config["protocol_id"], "status": "COMPLETE_NO_UPDATE", "raw_evidence_sha256": sha(raw_path), "raw_rows": len(raw), "authorization_sha256": sha(authorization), "optimizer_steps": 0, "generation_performed": False, "checkpoint_mutated": False, "final_audit_opened": False}
    atomic(out / "RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if args.self_test:
        print(json.dumps(self_test(config, args.train), sort_keys=True))
        return
    if args.authorization is None or args.out is None:
        raise RuntimeError("EXTERNAL_STAGE_AUTHORIZATION_REQUIRED_BEFORE_MODEL_IMPORT")
    terminal = {"schema_version": 1, "protocol_id": config["protocol_id"], "status": "PRE_MODEL_FAILURE", "optimizer_steps": 0, "generation_performed": False, "checkpoint_mutated": False, "final_audit_opened": False}
    try:
        terminal.update(execute(args.config, config, args.train, args.authorization, args.out))
    except Exception as exc:
        terminal.update({"status": "CAPPED_OR_FAILURE", "error": type(exc).__name__ + ":" + str(exc)})
    finally:
        if args.out is not None:
            atomic(args.out / "TERMINAL_RECEIPT.json", terminal)
    if terminal["status"] != "COMPLETE_NO_UPDATE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
