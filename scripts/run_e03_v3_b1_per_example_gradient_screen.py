#!/usr/bin/env python3
"""E03 V3: bounded B1 per-example LoRA-gradient screen, with no optimizer or generation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORE = -100


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8", newline="\n")
    os.replace(temp, path)


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def check_files(manifest: dict) -> None:
    for root_key, files_key in (("base_path", "base_files"), ("adapter_path", "adapter_files")):
        for item in manifest[files_key]:
            path = Path(manifest[root_key]) / item["name"]
            if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                raise RuntimeError("E03_V3_CHECKPOINT_FILE_HASH_MISMATCH:" + item["name"])


def preflight(config_path: Path, binding_path: Path, output_root: Path) -> tuple[dict, dict, dict, dict]:
    config_path, binding_path = resolve(str(config_path)), resolve(str(binding_path))
    config = json.loads(config_path.read_text(encoding="utf-8"))
    binding = json.loads(binding_path.read_text(encoding="utf-8"))
    if os.environ.get("E03_EXTERNAL_CAP_ENFORCED") != "1":
        raise RuntimeError("E03_V3_EXTERNAL_CAP_REQUIRED")
    if config.get("execution_authorized") is not False or binding.get("execution_authorized") is not True:
        raise RuntimeError("E03_V3_BINDING_AUTHORIZATION_REQUIRED")
    if binding.get("status") != "EXECUTION_AUTHORIZED_AFTER_DIRECTOR_REVIEW":
        raise RuntimeError("E03_V3_BINDING_STATUS_INVALID")
    if str(output_root) != binding.get("output_root") or binding.get("jobs") != 1 or binding.get("retry") is not False:
        raise RuntimeError("E03_V3_OUTPUT_OR_JOB_BINDING_INVALID")
    if binding.get("runtime_cap_seconds", 1801) > 1800:
        raise RuntimeError("E03_V3_RUNTIME_CAP_INVALID")
    if output_root.exists() and any(output_root.iterdir()):
        raise RuntimeError("E03_V3_FRESH_OUTPUT_REQUIRED")
    if subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT, text=True).strip() != binding["launch_branch"]:
        raise RuntimeError("E03_V3_LAUNCH_BRANCH_MISMATCH")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if subprocess.call(["git", "merge-base", "--is-ancestor", binding["executable_source_commit"], head], cwd=ROOT) != 0:
        raise RuntimeError("E03_V3_SOURCE_NOT_ANCESTOR")
    for relative, expected in binding["bound_files"].items():
        path = ROOT / relative
        if not path.is_file() or sha(path) != expected:
            raise RuntimeError("E03_V3_BOUND_FILE_HASH_MISMATCH:" + relative)
    response = resolve(binding["director_response_path"])
    if sha(response) != binding["director_response_sha256"] or json.loads(response.read_text(encoding="utf-8")).get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("E03_V3_DIRECTOR_BINDING_INVALID")
    cohort_path, checkpoint_path = resolve(config["cohort_manifest_path"]), resolve(config["checkpoint_manifest_path"])
    if sha(cohort_path) != config["cohort_manifest_sha256"] or sha(checkpoint_path) != config["checkpoint_manifest_sha256"]:
        raise RuntimeError("E03_V3_MANIFEST_HASH_MISMATCH")
    cohort, checkpoint = json.loads(cohort_path.read_text(encoding="utf-8")), json.loads(checkpoint_path.read_text(encoding="utf-8"))
    members = [member for family in cohort["families"] for member in family["members"]]
    if len(members) != 72 or len({member["episode_id"] for member in members}) != 72 or any(not member["episode_id"].startswith("TRAIN:") for member in members):
        raise RuntimeError("E03_V3_COHORT_INVALID")
    if any(len(family["members"]) != 8 for family in cohort["families"]):
        raise RuntimeError("E03_V3_FAMILY_BALANCE_INVALID")
    allow = config["lora_parameter_name_allowlist"]
    if hashlib.sha256(json.dumps(sorted(allow), separators=(",", ":")).encode()).hexdigest() != config["lora_parameter_name_allowlist_sha256"]:
        raise RuntimeError("E03_V3_ALLOWLIST_HASH_INVALID")
    check_files(checkpoint)
    return config, binding, cohort, checkpoint


def selected_ce(logits, labels, functional):
    shifted_logits, shifted_labels = logits[:, :-1], labels[:, 1:]
    selected = shifted_labels.ne(IGNORE)
    count = int(selected.sum().item())
    if count <= 0:
        raise RuntimeError("E03_V3_ZERO_SUPERVISED_TOKENS")
    # The cast is inside the differentiable graph; gradients are then captured
    # in FP32 after per-token normalization.
    return functional.cross_entropy(shifted_logits[selected].float(), shifted_labels[selected], reduction="sum") / count, count


def module_name(parameter_name: str) -> str:
    match = re.search(r"\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.", parameter_name)
    if not match:
        raise RuntimeError("E03_V3_UNGROUPABLE_PARAMETER")
    return match.group(1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output_root / "TERMINAL_RECEIPT.json"
    start = time.monotonic()
    try:
        config, binding, cohort, checkpoint = preflight(args.config, args.binding, args.output_root)
        args.output_root.mkdir(parents=True, exist_ok=False)
        atomic(args.output_root / "PROGRESS.json", {"status": "PARTIAL_NO_UPDATE", "phase": "PREFLIGHT_COMPLETE", "completed_rows": 0, "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False})
        import numpy as np
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
        model = AutoModelForCausalLM.from_pretrained(checkpoint["base_path"], torch_dtype=torch.bfloat16, device_map="cuda:0")
        model = PeftModel.from_pretrained(model, checkpoint["adapter_path"], is_trainable=True)
        model.eval(); model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        named = dict(model.named_parameters())
        allow = tuple(config["lora_parameter_name_allowlist"])
        # No base-model parameter may silently participate in the backward
        # graph.  This check is also part of the memory contract.
        for name, parameter in named.items():
            parameter.requires_grad_(name in allow)
        actual = tuple(sorted(name for name in named if ".lora_A." in name or ".lora_B." in name))
        if actual != allow:
            raise RuntimeError("E03_V3_RUNTIME_ALLOWLIST_MISMATCH")
        trainable = tuple(sorted(name for name, parameter in named.items() if parameter.requires_grad))
        if trainable != allow:
            raise RuntimeError("E03_V3_TRAINABLE_PARAMETER_SET_MISMATCH")
        parameters = [(name, named[name]) for name in allow]
        offsets, total = {}, 0
        for name, parameter in parameters:
            offsets[name] = (total, parameter.numel()); total += parameter.numel()
        rows = [member for family in cohort["families"] for member in family["members"]]
        scratch = args.output_root / "gradient_rows.tmp.f32"
        grams = np.zeros((len(rows), len(rows)), dtype=np.float64)
        module_grams = {name: np.zeros_like(grams) for name in config["target_modules"]}
        matrix = np.memmap(scratch, dtype=np.float32, mode="w+", shape=(len(rows), total))
        for index, member in enumerate(rows):
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([member["input_ids"]], device="cuda", dtype=torch.long)
            labels = torch.tensor([member["labels"]], device="cuda", dtype=torch.long)
            attention = torch.ones_like(ids)
            positions = (attention.cumsum(-1) - 1).clamp_min(0)
            loss, count = selected_ce(model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits, labels, functional)
            loss.backward()
            for name, parameter in parameters:
                offset, length = offsets[name]
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("E03_V3_NONFINITE_GRADIENT")
                matrix[index, offset:offset + length] = parameter.grad.detach().float().flatten().cpu().numpy()
            atomic(args.output_root / "PROGRESS.json", {"status": "PARTIAL_NO_UPDATE", "phase": "B1_GRADIENTS", "completed_rows": index + 1, "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False})
        matrix.flush()
        # Re-run exactly one predeclared episode from each family.  These
        # repeated B1 gradients are only a numerical-sensitivity control and
        # cannot change the frozen cohort or scientific estimand.
        repeat_indices = [next(i for i, member in enumerate(rows) if member["episode_id"] == family["members"][0]["episode_id"]) for family in cohort["families"]]
        repeat_metrics = []
        tolerance = config["numerics"]["repeatability_contract"]
        for index in repeat_indices:
            member = rows[index]
            model.zero_grad(set_to_none=True)
            ids = torch.tensor([member["input_ids"]], device="cuda", dtype=torch.long)
            labels = torch.tensor([member["labels"]], device="cuda", dtype=torch.long)
            attention = torch.ones_like(ids); positions = (attention.cumsum(-1) - 1).clamp_min(0)
            loss, _ = selected_ce(model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits, labels, functional)
            loss.backward()
            repeated = np.empty(total, dtype=np.float32)
            for name, parameter in parameters:
                offset, length = offsets[name]
                if parameter.grad is None or not bool(torch.isfinite(parameter.grad).all()):
                    raise RuntimeError("E03_V3_REPEAT_NONFINITE_GRADIENT")
                repeated[offset:offset + length] = parameter.grad.detach().float().flatten().cpu().numpy()
            original = np.asarray(matrix[index], dtype=np.float64)
            repeat64 = repeated.astype(np.float64)
            base_norm = float(np.linalg.norm(original)); repeat_norm = float(np.linalg.norm(repeat64))
            if base_norm <= 0 or repeat_norm <= 0:
                raise RuntimeError("E03_V3_REPEAT_ZERO_NORM")
            relative_l2 = float(np.linalg.norm(repeat64 - original) / base_norm)
            cosine = float(np.dot(repeat64, original) / (base_norm * repeat_norm))
            repeat_metrics.append({"episode_id": member["episode_id"], "relative_l2": relative_l2, "cosine": cosine})
        repeatability = {"contract": tolerance, "rows": len(repeat_metrics), "episode_ids": [item["episode_id"] for item in repeat_metrics], "max_relative_l2": max(item["relative_l2"] for item in repeat_metrics), "min_cosine": min(item["cosine"] for item in repeat_metrics)}
        repeatability["status"] = "PASS" if repeatability["max_relative_l2"] <= tolerance["max_relative_l2"] and repeatability["min_cosine"] >= tolerance["min_cosine"] else "INCONCLUSIVE"
        for name, (offset, length) in offsets.items():
            group = module_name(name)
            for begin in range(offset, offset + length, 262144):
                block = np.asarray(matrix[:, begin:min(begin + 262144, offset + length)], dtype=np.float64)
                product = block @ block.T
                grams += product; module_grams[group] += product
        del matrix
        scratch.unlink(missing_ok=True)
        atomic(args.output_root / "RAW_GRAM_STATISTICS.json", {"protocol_id": config["protocol_id"], "status": "COMPLETE_NO_UPDATE", "episode_ids": [member["episode_id"] for member in rows], "families": [family["canonical_family"] for family in cohort["families"] for _ in family["members"]], "supervised_token_counts": [int(member["supervised_token_count"]) for member in rows], "per_token_normalized": True, "gram_matrix": grams.tolist(), "module_gram_matrices": {name: value.tolist() for name, value in module_grams.items()}, "repeatability": repeatability, "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False})
        atomic(receipt, {"protocol_id": config["protocol_id"], "status": "COMPLETE_NO_UPDATE", "elapsed_seconds": time.monotonic() - start, "completed_rows": len(rows), "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False})
    except Exception as exc:
        atomic(receipt, {"protocol_id": "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN", "status": "FAILED_OR_PARTIAL_NO_UPDATE", "error_class": str(exc), "elapsed_seconds": time.monotonic() - start, "optimizer_steps": 0, "parameter_updates": 0, "generation_calls": 0, "final_audit_opened": False, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
