#!/usr/bin/env python3
"""Strict BatchN-versus-frozen-Batch1 inference probe; never trains.

This is deliberately an operational reproducibility study.  It uses the same
frozen target-blind cohort, checkpoint manifests, tokenizer/parser and greedy
decoder as its Batch1 reference.  A candidate is accepted only when every
stored per-episode prediction field, including generated token IDs, is exactly
equal to Batch1 for every checkpoint.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_capability_repair_baseline_v1 import (  # noqa: E402
    aggregate,
    output_row,
    prompt,
    sha,
    verify_checkpoint_manifest,
    write,
)
from scripts.arc2_token_grid_parser import tokenizer_token_contract  # noqa: E402
from scripts.run_serial_decoding_flip_replication import (  # noqa: E402
    load_contract,
    preflight,
)

ALLOWED_BATCH_SIZES = (2, 4, 8, 16, 32)
STRICT_FIELDS = (
    "episode_id",
    "generated_token_ids",
    "generated_length",
    "termination_status",
    "eos_observed",
    "trailing_pad_count",
    "content_token_ids",
    "parse_reason",
    "canonical_prediction_sha256",
    "parse_valid",
    "exact_grid_match",
)


def strict_prediction_difference(reference: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    """Return the immutable output fields that differ; never inspect targets."""
    return [field for field in STRICT_FIELDS if reference.get(field) != candidate.get(field)]


def parse_candidates(value: str) -> tuple[int, ...]:
    try:
        candidates = tuple(int(item) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise ValueError("BATCH_CANDIDATES_INVALID") from exc
    if not candidates or any(item not in ALLOWED_BATCH_SIZES for item in candidates):
        raise ValueError("BATCH_CANDIDATES_INVALID")
    if len(candidates) != len(set(candidates)) or tuple(sorted(candidates)) != candidates:
        raise ValueError("BATCH_CANDIDATES_NOT_ASCENDING_UNIQUE")
    return candidates


def reference_index(reference: dict[str, Any], model_ids: list[str], episode_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    if reference.get("status") != "COLLECTED_PASS" or reference.get("batch_size") != 1:
        raise RuntimeError("BATCH1_REFERENCE_INVALID")
    if reference.get("raw_token_evidence") is not True:
        raise RuntimeError("BATCH1_RAW_TOKEN_EVIDENCE_REQUIRED")
    models = reference.get("models")
    if not isinstance(models, list) or [item.get("model_id") for item in models] != model_ids:
        raise RuntimeError("BATCH1_REFERENCE_MODEL_BINDING_INVALID")
    indexed: dict[str, dict[str, dict[str, Any]]] = {}
    for model in models:
        predictions = model.get("predictions")
        if not isinstance(predictions, list) or [item.get("episode_id") for item in predictions] != episode_ids:
            raise RuntimeError("BATCH1_REFERENCE_EPISODE_BINDING_INVALID")
        indexed[str(model["model_id"])] = {str(item["episode_id"]): item for item in predictions}
    return indexed


def score_batch(
    *, model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int,
    inference_contract: dict[str, Any], decoder_sha256: str, manifest_sha256: str,
    token_contract: Any, device: Any, deadline: float,
) -> list[dict[str, Any]]:
    import torch

    predictions: list[dict[str, Any]] = []
    for offset in range(0, len(rows), batch_size):
        if time.monotonic() >= deadline:
            raise TimeoutError("RUNTIME_CAP_BEFORE_BATCH")
        batch = rows[offset:offset + batch_size]
        texts = [prompt(row, inference_contract) for row in batch]
        encoded = tokenizer(texts, return_tensors="pt", add_special_tokens=False, padding=True)
        full_width = int(encoded["input_ids"].shape[-1])
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                num_beams=1,
                max_new_tokens=inference_contract["runtime"]["max_new_tokens"],
                use_cache=True,
                eos_token_id=inference_contract["runtime"]["eos_token_id"],
                pad_token_id=inference_contract["runtime"]["pad_token_id"],
            )
        for index, (row, text) in enumerate(zip(batch, texts, strict=True)):
            token_ids = [int(item) for item in generated[index][full_width:].detach().cpu().tolist()]
            scored = output_row(
                "RETENTION_SENTINEL" if row["role"] == "RETENTION_SENTINEL" else "TARGET_DEV",
                row, text, token_ids, decoder_sha256, manifest_sha256, token_contract,
            )
            scored["requested_batch_size"] = batch_size
            scored["actual_batch_size"] = len(batch)
            predictions.append(scored)
    return predictions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--batch1-reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--candidates", default="2,4,8,16,32")
    parser.add_argument("--runtime-limit-seconds", type=float, default=7200)
    args = parser.parse_args()
    if args.output.exists() or args.receipt.exists():
        raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    if args.runtime_limit_seconds <= 0 or args.runtime_limit_seconds > 7200:
        raise RuntimeError("RUNTIME_CAP_BINDING_INVALID")
    candidates = parse_candidates(args.candidates)
    started = time.monotonic(); deadline = started + args.runtime_limit_seconds
    root = Path(__file__).resolve().parents[1]
    contract = load_contract(args.contract)
    bound = preflight(contract, root)
    reference = json.loads(args.batch1_reference.read_text(encoding="utf-8"))
    model_ids = [item["id"] for item in bound["manifests"]]
    episode_ids = [item["episode_id"] for item in bound["rows"]]
    reference_by_model = reference_index(reference, model_ids, episode_ids)

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA_OR_BF16_UNAVAILABLE")
    source_contract = root / "experiments/base_only_targeted_repair_remote_first_v2/execution_contract_v7/BASELINE_INFERENCE_CONTRACT_V2.json"
    inference_contract = json.loads(source_contract.read_text(encoding="utf-8"))
    runtime = inference_contract["runtime"]
    device = torch.device("cuda:0")
    all_candidates: dict[str, dict[str, Any]] = {
        str(size): {"batch_size": size, "model_results": [], "status": "PENDING"} for size in candidates
    }

    def checkpoint_progress() -> None:
        write(args.progress, {
            "schema_version": 1,
            "status": "RUNNING",
            "diagnostic_id": "V7_R1_R2_BATCH_CONSISTENCY_PROBE_V1",
            "batch1_reference_sha256": sha(args.batch1_reference),
            "candidates": all_candidates,
            "elapsed_seconds": time.monotonic() - started,
            "scientific_training_started": False,
            "optimizer_constructed": False,
            "final_audit_opened": False,
        })

    for item in bound["manifests"]:
        if time.monotonic() >= deadline:
            raise TimeoutError("RUNTIME_CAP_BEFORE_MODEL")
        manifest = item["manifest"]
        verify_checkpoint_manifest(manifest)
        base, adapter = Path(manifest["base_path"]), Path(manifest["adapter_path"])
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True)
        tokenizer.padding_side = "left"; tokenizer.pad_token_id = runtime["pad_token_id"]
        token_contract = tokenizer_token_contract(tokenizer, eos_token_id=runtime["eos_token_id"], pad_token_id=runtime["pad_token_id"])
        model = AutoModelForCausalLM.from_pretrained(
            base, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation=runtime["attention_backend"],
        ).to(device)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        for batch_size in candidates:
            record = all_candidates[str(batch_size)]
            try:
                predictions = score_batch(
                    model=model, tokenizer=tokenizer, rows=bound["rows"], batch_size=batch_size,
                    inference_contract=inference_contract, decoder_sha256=contract["decoder_contract"]["sha256"],
                    manifest_sha256=item["manifest_sha256"], token_contract=token_contract, device=device, deadline=deadline,
                )
                mismatches = [
                    {"episode_id": value["episode_id"], "differing_fields": strict_prediction_difference(reference_by_model[item["id"]][value["episode_id"]], value)}
                    for value in predictions
                    if strict_prediction_difference(reference_by_model[item["id"]][value["episode_id"]], value)
                ]
                record["model_results"].append({
                    "model_id": item["id"], "checkpoint_manifest_sha256": item["manifest_sha256"],
                    "strictly_equal_to_batch1": not mismatches, "mismatch_count": len(mismatches),
                    "mismatches": mismatches, "predictions": predictions,
                    "aggregates": {
                        "TARGET_DEV": aggregate([x for x in predictions if x["surface"] == "TARGET_DEV"]),
                        "RETENTION_SENTINEL": aggregate([x for x in predictions if x["surface"] == "RETENTION_SENTINEL"]),
                    },
                })
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                record["model_results"].append({"model_id": item["id"], "status": "OOM", "strictly_equal_to_batch1": False})
            checkpoint_progress()
        del model; torch.cuda.empty_cache()

    accepted: list[int] = []
    for batch_size in candidates:
        record = all_candidates[str(batch_size)]
        results = record["model_results"]
        record["status"] = "COMPLETE" if len(results) == len(model_ids) else "INCOMPLETE"
        record["strictly_equal_to_batch1"] = bool(results) and len(results) == len(model_ids) and all(item.get("strictly_equal_to_batch1") is True for item in results)
        if record["strictly_equal_to_batch1"]:
            accepted.append(batch_size)
    result = {
        "schema_version": 1,
        "diagnostic_id": "V7_R1_R2_BATCH_CONSISTENCY_PROBE_V1",
        "status": "COLLECTED_PASS",
        "comparison": "EXACT_PER_EPISODE_RAW_TOKEN_AND_CANONICAL_OUTPUT_EQUALITY_TO_FROZEN_BATCH1",
        "batch1_reference_path": str(args.batch1_reference),
        "batch1_reference_sha256": sha(args.batch1_reference),
        "cohort_sha256": contract["cohort"]["sha256"],
        "candidates": all_candidates,
        "largest_strictly_equal_batch_size": max(accepted) if accepted else None,
        "scientific_training_started": False,
        "optimizer_constructed": False,
        "final_audit_opened": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    write(args.output, result)
    write(args.receipt, {
        "schema_version": 1, "status": "COLLECTED_PASS", "diagnostic_id": result["diagnostic_id"],
        "result_sha256": sha(args.output), "largest_strictly_equal_batch_size": result["largest_strictly_equal_batch_size"],
        "scientific_training_started": False, "optimizer_constructed": False,
        "elapsed_seconds": time.monotonic() - started,
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
