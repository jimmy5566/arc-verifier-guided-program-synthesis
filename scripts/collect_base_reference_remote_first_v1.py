#!/usr/bin/env python3
"""Fail-closed base-reference collector for the remote-first V2 condition.

The collector never constructs LoRA or an optimizer.  ``--cpu-mock`` is the
only permitted mode before a later Director directive authorizes collection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "BASE_ONLY_TARGETED_REPAIR_REMOTE_FIRST_V2"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def rows(path: Path, allowed: set[str]) -> list[dict]:
    value = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not value or any(row.get("role") not in allowed for row in value):
        raise RuntimeError("FORBIDDEN_OR_EMPTY_EVALUATION_SURFACE")
    return value


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--target-dev", type=Path, required=True)
    p.add_argument("--retention", type=Path, required=True)
    p.add_argument("--binding", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--receipt", type=Path, required=True)
    p.add_argument("--authorization", type=Path)
    p.add_argument("--cpu-mock", action="store_true")
    args = p.parse_args()
    started = time.monotonic()
    target = rows(args.target_dev, {"TARGETED_EVALUATION", "TARGETED_COMPOSITION"})
    retention = rows(args.retention, {"RETENTION_SENTINEL"})
    binding = json.loads(args.binding.read_text(encoding="utf-8"))
    if args.cpu_mock:
        result = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "CPU_MOCK_PASS_NO_MODEL", "model_loaded": False, "lora_constructed": False, "optimizer_constructed": False, "gpu_training_started": False, "target_episode_order": [r["episode_id"] for r in target], "retention_episode_order": [r["episode_id"] for r in retention], "binding_sha256": sha(args.binding)}
        write(args.output, result)
        write(args.receipt, {"schema_version": 1, "status": "CPU_MOCK_PASS", "seconds": time.monotonic() - started, "model_loaded": False, "optimizer_constructed": False})
        return 0
    auth = json.loads(args.authorization.read_text(encoding="utf-8")) if args.authorization else {}
    if auth.get("base_reference_collection_authorized") is not True:
        raise RuntimeError("DIRECTOR_BASE_REFERENCE_AUTHORIZATION_REQUIRED")
    if binding.get("base_path") != "/workspace/arc2/models/qwen3_4b_grids15_sft139":
        raise RuntimeError("BASE_PATH_IDENTITY_INVALID")
    # Imports are deliberately behind the explicit Director gate. This branch
    # is not exercised in the current directive cycle.
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU_REQUIRED_FOR_AUTHORIZED_BASE_REFERENCE")
    tokenizer = AutoTokenizer.from_pretrained(binding["base_path"], local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(binding["base_path"], local_files_only=True, torch_dtype=torch.bfloat16).eval()
    predictions = []
    for surface, records in (("TARGET_DEV", target), ("RETENTION_SENTINEL", retention)):
        for record in records:
            # Exact output decoding is frozen by a later authorized command
            # record; keep one deterministic prediction identity per episode.
            prompt = json.dumps(record["task"], sort_keys=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output = model.generate(**inputs, do_sample=False, max_new_tokens=512)
            text = tokenizer.decode(output[0], skip_special_tokens=False)
            predictions.append({"surface": surface, "episode_id": record["episode_id"], "prediction_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest()})
    write(args.output, {"schema_version": 1, "protocol_id": PROTOCOL, "status": "COLLECTED_PASS", "predictions": predictions, "model_loaded": True, "lora_constructed": False, "optimizer_constructed": False})
    write(args.receipt, {"schema_version": 1, "status": "COLLECTED_PASS", "seconds": time.monotonic() - started, "prediction_count": len(predictions)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
