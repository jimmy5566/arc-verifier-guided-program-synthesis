#!/usr/bin/env python3
"""One-shot Batch1 V7/R1/R2 flip-replication diagnostic; never trains."""
from __future__ import annotations

import argparse, hashlib, json, os, time
from pathlib import Path

from scripts.collect_capability_repair_baseline_v1 import (
    aggregate, observation, output_row, prompt, read_rows, sha, verify_checkpoint_manifest,
    write,
)
from scripts.arc2_token_grid_parser import tokenizer_token_contract


def require_file(path: Path, expected: str) -> None:
    if not path.is_file() or sha(path) != expected:
        raise RuntimeError(f"IDENTITY_MISMATCH:{path}")


def load_contract(path: Path) -> dict:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("diagnostic_id") != "V7_R1_R2_SERIAL_DECODING_FLIP_REPLICATION_V1":
        raise RuntimeError("DIAGNOSTIC_CONTRACT_INVALID")
    if contract.get("final_audit_opened") is not False or contract["execution"].get("batch_size") != 1:
        raise RuntimeError("DIAGNOSTIC_SAFETY_CONTRACT_INVALID")
    if contract["execution"].get("optimizer_construction") or contract["execution"].get("training"):
        raise RuntimeError("TRAINING_FORBIDDEN")
    return contract


def selected_rows(contract: dict, root: Path) -> tuple[list[dict], dict]:
    cohort_path = root / contract["cohort"]["path"]
    require_file(cohort_path, contract["cohort"]["sha256"])
    cohort = json.loads(cohort_path.read_text(encoding="utf-8"))
    if cohort.get("target_values_in_manifest") is not False or cohort.get("episode_count") != 65:
        raise RuntimeError("COHORT_CONTRACT_INVALID")
    target_spec, retention_spec = contract["datasets"]["TARGET_DEV"], contract["datasets"]["RETENTION_SENTINEL"]
    target_path, retention_path = Path(target_spec["path"]), Path(retention_spec["path"])
    require_file(target_path, target_spec["sha256"]); require_file(retention_path, retention_spec["sha256"])
    available = {row["episode_id"]: row for row in read_rows(target_path, {"TARGETED_EVALUATION", "TARGETED_COMPOSITION"}) + read_rows(retention_path, {"RETENTION_SENTINEL"})}
    ids = [row["episode_id"] for row in cohort["cohort"]]
    if len(ids) != len(set(ids)) or set(ids) - set(available):
        raise RuntimeError("COHORT_EPISODE_BINDING_INVALID")
    return [available[item] for item in ids], cohort


def preflight(contract: dict, root: Path) -> dict:
    rows, cohort = selected_rows(contract, root)
    parser_path = root / contract["decoder_contract"]["path"]
    require_file(parser_path, contract["decoder_contract"]["sha256"])
    manifests = []
    for model in contract["models"]:
        manifest_path = root / model["manifest"]
        require_file(manifest_path, model["manifest_sha256"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifests.append({"id": model["id"], "manifest": manifest, "manifest_sha256": model["manifest_sha256"]})
    return {"rows": rows, "cohort": cohort, "manifests": manifests, "parser_sha256": contract["decoder_contract"]["sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--cpu-preflight", action="store_true")
    args = parser.parse_args(); started = time.monotonic(); root = Path(__file__).resolve().parents[1]
    if args.output.exists() or args.receipt.exists():
        raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    contract = load_contract(args.contract)
    bound = preflight(contract, root)
    if args.cpu_preflight:
        write(args.output, {"status": "CPU_PREFLIGHT_PASS_NO_MODEL", "diagnostic_id": contract["diagnostic_id"], "cohort_episodes": len(bound["rows"]), "model_ids": [x["id"] for x in bound["manifests"]], "final_audit_opened": False})
        write(args.receipt, {"status": "CPU_PREFLIGHT_PASS_NO_MODEL", "scientific_training_started": False, "optimizer_constructed": False, "model_loaded": False, "elapsed_seconds": time.monotonic() - started})
        return 0
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("CUDA_OR_BF16_UNAVAILABLE")
    source_contract = root / "experiments/base_only_targeted_repair_remote_first_v2/execution_contract_v7/BASELINE_INFERENCE_CONTRACT_V2.json"
    inference_contract = json.loads(source_contract.read_text(encoding="utf-8")); runtime = inference_contract["runtime"]
    all_models = []; device = torch.device("cuda:0")
    for item in bound["manifests"]:
        manifest = item["manifest"]; verify_checkpoint_manifest(manifest)
        base, adapter = Path(manifest["base_path"]), Path(manifest["adapter_path"])
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True); tokenizer.padding_side = "left"; tokenizer.pad_token_id = runtime["pad_token_id"]
        token_contract = tokenizer_token_contract(tokenizer, eos_token_id=runtime["eos_token_id"], pad_token_id=runtime["pad_token_id"])
        model = AutoModelForCausalLM.from_pretrained(base, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=runtime["attention_backend"]).to(device)
        model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
        predictions = []
        for row in bound["rows"]:
            text = prompt(row, inference_contract)
            encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
            width = int(encoded["input_ids"].shape[-1]); encoded = {key: value.to(device) for key, value in encoded.items()}
            with torch.inference_mode():
                generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=runtime["max_new_tokens"], use_cache=True, eos_token_id=runtime["eos_token_id"], pad_token_id=runtime["pad_token_id"])
            token_ids = [int(x) for x in generated[0][width:].detach().cpu().tolist()]
            scored = output_row("RETENTION_SENTINEL" if row["role"] == "RETENTION_SENTINEL" else "TARGET_DEV", row, text, token_ids, contract["decoder_contract"]["sha256"], item["manifest_sha256"], token_contract)
            scored["requested_batch_size"] = 1; scored["actual_batch_size"] = 1; predictions.append(scored)
        del model; torch.cuda.empty_cache()
        all_models.append({"model_id": item["id"], "checkpoint_manifest_sha256": item["manifest_sha256"], "predictions": predictions, "aggregates": {"TARGET_DEV": aggregate([x for x in predictions if x["surface"] == "TARGET_DEV"]), "RETENTION_SENTINEL": aggregate([x for x in predictions if x["surface"] == "RETENTION_SENTINEL"])}})
    result = {"schema_version": 1, "diagnostic_id": contract["diagnostic_id"], "status": "COLLECTED_PASS", "batch_size": 1, "cohort_sha256": contract["cohort"]["sha256"], "models": all_models, "raw_token_evidence": True, "final_audit_opened": False}
    write(args.output, result); write(args.receipt, {"status": "COLLECTED_PASS", "scientific_training_started": False, "optimizer_constructed": False, "model_loaded": True, "elapsed_seconds": time.monotonic() - started, "completed_models": [x["model_id"] for x in all_models]})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
