"""Run the single authorized two-arm E04-C matched repair experiment.

The parent has one detached-process identity.  It creates a fresh Python
process for each independently initialized V7 arm, then freezes target-blind
physical-B1 generations.  It deliberately never opens the target sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.e04_c_matched_rotation_execution import (ARMS, BASE, E04CFailure, IGNORE_INDEX,
    JOINT_CAP_SECONDS, PER_ARM_CAP_SECONDS, ROOT, SCHEDULE, atomic_json, canon, fail,
    load_frozen_inputs, relative, sha_path, static_schedule_preflight)


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_file(relative_path: str, expected: str, name: str) -> Path:
    path = ROOT / relative_path
    if not path.is_file() or sha_path(path) != expected:
        fail("E04C_" + name + "_HASH")
    return path


def load_binding(path: Path, output_root: Path, *, child: bool = False) -> dict[str, Any]:
    binding = _read(path)
    required = {"schema_version", "protocol_id", "authorization_id", "director_response_path", "director_response_sha256",
                "execution_authorized", "source_commit", "worker_path", "worker_sha256", "evaluator_path", "evaluator_sha256",
                "launcher_path", "launcher_sha256", "protocol_path", "protocol_sha256", "schedule_binding_path", "schedule_binding_sha256",
                "static_preflight_path", "static_preflight_sha256", "control_cohort_path", "control_cohort_sha256",
                "treatment_cohort_path", "treatment_cohort_sha256", "checkpoint_manifest_path", "checkpoint_manifest_sha256",
                "runtime_config_path", "runtime_config_sha256", "v7_reference_raw_path", "v7_reference_raw_sha256",
                "arm_order", "seed", "arms", "jobs", "retry", "nonce", "output_root", "arm_output_roots",
                "per_arm_runtime_cap_seconds", "joint_runtime_cap_seconds", "static_preflight"}
    if set(binding) != required or binding["protocol_id"] != "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1":
        fail("E04C_BINDING_SCHEMA")
    if binding["execution_authorized"] is not True or binding["jobs"] != 1 or binding["arms"] != 2 or binding["retry"] is not False:
        fail("E04C_BINDING_AUTHORIZATION")
    if binding["arm_order"] != list(ARMS) or binding["per_arm_runtime_cap_seconds"] != PER_ARM_CAP_SECONDS or binding["joint_runtime_cap_seconds"] != JOINT_CAP_SECONDS:
        fail("E04C_BINDING_RUNTIME")
    if str(output_root).replace("\\", "/") != binding["output_root"] or not isinstance(binding["nonce"], str) or not binding["nonce"]:
        fail("E04C_BINDING_OUTPUT")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != binding["source_commit"]:
        fail("E04C_SOURCE_COMMIT")
    for key, name in (("worker_path", "WORKER"), ("evaluator_path", "EVALUATOR"), ("launcher_path", "LAUNCHER"),
                      ("protocol_path", "PROTOCOL"), ("schedule_binding_path", "SCHEDULE"),
                      ("static_preflight_path", "STATIC_PREFLIGHT"), ("control_cohort_path", "CONTROL_COHORT"),
                      ("treatment_cohort_path", "TREATMENT_COHORT"), ("checkpoint_manifest_path", "CHECKPOINT_MANIFEST"),
                      ("runtime_config_path", "RUNTIME_CONFIG"), ("v7_reference_raw_path", "V7_REFERENCE")):
        _verify_file(binding[key], binding[key.replace("_path", "_sha256")], name)
    response = _verify_file(binding["director_response_path"], binding["director_response_sha256"], "DIRECTOR_RESPONSE")
    if _read(response).get("decision") != "CONTINUE_CONTROLLER":
        fail("E04C_DIRECTOR_DECISION")
    if output_root.exists():
        allowed = {"PREFLIGHT_RECEIPT.json", "arms"}
        if not child or {item.name for item in output_root.iterdir()} - allowed:
            fail("E04C_FRESH_OUTPUT")
    return binding


def _verify_checkpoint(manifest: dict[str, Any]) -> None:
    for root_key, files_key in (("base_path", "base_files"), ("adapter_path", "adapter_files")):
        root = Path(manifest[root_key])
        for entry in manifest[files_key]:
            path = root / entry["name"]
            if not path.is_file() or path.stat().st_size != entry["bytes"] or sha_path(path) != entry["sha256"]:
                fail("E04C_CHECKPOINT_IDENTITY:" + entry["name"])


def _actual_tokenizer_preflight(binding: dict[str, Any]) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Use the checkpoint tokenizer and the worker's one-item collator.

    This is intentionally distinct from the CPU serializer audit: the real
    Transformers tokenizer is loaded, every scheduled message sequence is
    tokenized, and the exact input/labels/masks consumed in training are
    checked before model or optimizer construction.
    """
    protocol, _, static, schedules = load_frozen_inputs()
    manifest = _read(ROOT / binding["checkpoint_manifest_path"])
    from inference.nvarc_native import checkpoint_native_tokenizer
    tokenizer, token_meta = checkpoint_native_tokenizer(Path(manifest["base_path"]), Path(_read(ROOT / binding["runtime_config_path"])["native_config_dir"]))
    values: dict[str, list[dict[str, Any]]] = {}
    for arm in ARMS:
        rows: list[dict[str, Any]] = []
        for slot, episode in enumerate(schedules[arm]["episodes"]):
            task = episode["control_task"] if arm == ARMS[0] else episode["treatment_task"]
            from training_data.pipeline import task_to_sample
            sample = task_to_sample({"source_id": "e04c-runtime-" + arm.lower() + "-" + episode["pair_id"], **task})
            actual = tokenizer.apply_chat_template(sample["messages"], add_generation_prompt=False, tokenize=True, return_tensors=None)
            ids = list(actual[0] if actual and isinstance(actual[0], list) else actual)
            if ids != sample["input_ids"]:
                fail("E04C_PRODUCTION_TOKENIZER_IDS:" + str(slot))
            labels = list(sample["labels"])
            mask = [1] * len(ids)
            if len(ids) != len(labels) or len(mask) != len(ids) or sum(token != IGNORE_INDEX for token in labels) != sample["assistant_token_count"]:
                fail("E04C_PRODUCTION_COLLATOR_SHAPE:" + str(slot))
            expected = static["per_slot_rows"][slot]
            prefix = "control" if arm == ARMS[0] else "treatment"
            if len(ids) != expected[prefix + "_sequence_length"] or sample["assistant_token_count"] != expected[prefix + "_supervised_token_count"]:
                fail("E04C_PRODUCTION_STATIC_COUNT:" + str(slot))
            rows.append({"slot": slot, "pair_id": episode["pair_id"], "role": episode["role"], "input_ids": ids, "labels": labels,
                         "attention_mask": mask, "supervised": sample["assistant_token_count"]})
        values[arm] = rows
    for c, t in zip(values[ARMS[0]], values[ARMS[1]], strict=True):
        if c["slot"] >= 96 and (c["input_ids"], c["labels"], c["attention_mask"]) != (t["input_ids"], t["labels"], t["attention_mask"]):
            fail("E04C_PRODUCTION_PROTECTED_REPLAY")
        if c["slot"] < 96 and (c["input_ids"], c["labels"]) == (t["input_ids"], t["labels"]):
            fail("E04C_PRODUCTION_INTERVENTION_MISSING")
    totals = {arm: {"total_tokens": sum(len(row["input_ids"]) for row in rows), "supervised_tokens": sum(row["supervised"] for row in rows)} for arm, rows in values.items()}
    for value in totals.values():
        if value != {"total_tokens": protocol["schedule"]["per_arm_total_tokens"], "supervised_tokens": protocol["schedule"]["per_arm_supervised_tokens"]}:
            fail("E04C_PRODUCTION_TOTALS")
    return values, {"tokenizer": token_meta, "totals": totals, "static_preflight_sha256": binding["static_preflight_sha256"]}


def _load_trainable_v7(manifest: dict[str, Any], recipe: dict[str, Any]):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        fail("E04C_CUDA_BF16_UNAVAILABLE")
    model = AutoModelForCausalLM.from_pretrained(manifest["base_path"], local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
    model = PeftModel.from_pretrained(model, manifest["adapter_path"], is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable or any(parameter.requires_grad for name, parameter in model.named_parameters() if "lora_" not in name):
        fail("E04C_LORA_PARTITION")
    model.train()
    return model, trainable, torch


def train_arm(binding_path: Path, output_root: Path, arm: str) -> None:
    started = time.monotonic(); arm_root = output_root / "arms" / arm
    terminal = arm_root / "TERMINAL_RECEIPT.json"
    try:
        binding = load_binding(binding_path, output_root, child=True)
        if arm not in ARMS or str(arm_root).replace("\\", "/") != binding["arm_output_roots"][arm]:
            fail("E04C_ARM_OUTPUT_BINDING")
        if arm_root.exists():
            fail("E04C_ARM_OUTPUT_FRESH")
        schedule_proof = static_schedule_preflight()
        manifest = _read(ROOT / binding["checkpoint_manifest_path"]); _verify_checkpoint(manifest)
        runtime_rows, runtime_proof = _actual_tokenizer_preflight(binding)
        arm_root.mkdir(parents=True, exist_ok=False)
        atomic_json(arm_root / "PRE_OPTIMIZER_PREFLIGHT.json", {"status": "PASS_PRODUCTION_TOKENIZER_AND_COLLATOR", "arm": arm,
                    "schedule": schedule_proof, "runtime": runtime_proof, "model_imported": False, "optimizer_constructed": False,
                    "target_sidecar_accessed": False})
        recipe = _read(ROOT / binding["protocol_path"])["recipe"]
        import bitsandbytes as bnb
        import torch
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"; os.environ["TOKENIZERS_PARALLELISM"] = "false"
        torch.manual_seed(int(binding["seed"])); torch.cuda.manual_seed_all(int(binding["seed"]))
        model, parameters, torch = _load_trainable_v7(manifest, recipe)
        optimizer = bnb.optim.PagedAdamW8bit(parameters, lr=float(recipe["learning_rate"]))
        deadline = started + PER_ARM_CAP_SECONDS; rows = runtime_rows[arm]; steps = 0; processed = 0; supervised = 0
        for offset in range(0, len(rows), int(recipe["gradient_accumulation"])):
            if time.monotonic() >= deadline: fail("E04C_PER_ARM_CAP")
            group = rows[offset:offset + int(recipe["gradient_accumulation"])]
            if len(group) != int(recipe["gradient_accumulation"]): fail("E04C_ACCUMULATION_GROUP")
            denominator = sum(row["supervised"] for row in group)
            optimizer.zero_grad(set_to_none=True)
            for row in group:
                ids = torch.tensor([row["input_ids"]], device="cuda:0", dtype=torch.long)
                labels = torch.tensor([row["labels"]], device="cuda:0", dtype=torch.long)
                attention = torch.tensor([row["attention_mask"]], device="cuda:0", dtype=torch.long)
                loss = model(input_ids=ids, attention_mask=attention, labels=labels, use_cache=False).loss
                if not bool(torch.isfinite(loss)): fail("E04C_NONFINITE_LOSS")
                (loss * (row["supervised"] / denominator)).backward()
                processed += len(row["input_ids"]); supervised += row["supervised"]
            if time.monotonic() >= deadline: fail("E04C_PER_ARM_CAP")
            for group_config in optimizer.param_groups:
                group_config["lr"] = float(recipe["learning_rate"]) * min(1.0, (steps + 1) / 3.0)
            optimizer.step(); steps += 1
            atomic_json(arm_root / "TRAINING_PROGRESS.json", {"status": "RUNNING", "arm": arm, "optimizer_steps": steps,
                        "processed_tokens": processed, "supervised_tokens": supervised, "target_sidecar_accessed": False})
        if steps != 96 or processed != 207360 or supervised != 28800:
            fail("E04C_COMPLETION_TOTALS")
        model.save_pretrained(arm_root / "adapter")
        atomic_json(terminal, {"schema_version": 1, "status": "COMPLETED", "protocol_id": binding["protocol_id"], "arm": arm,
                    "optimizer_steps": steps, "parameter_updates": steps, "processed_tokens": processed, "supervised_tokens": supervised,
                    "target_sidecar_accessed": False, "final_audit_opened": False, "elapsed_seconds": time.monotonic() - started,
                    "runtime_cap_seconds": PER_ARM_CAP_SECONDS, "adapter_sha256": sha_path(arm_root / "adapter" / "adapter_model.safetensors")})
    except Exception as error:
        arm_root.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal, {"schema_version": 1, "status": "FAILED_OR_PARTIAL", "protocol_id": "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1",
                    "arm": arm, "error": f"{type(error).__name__}:{error}", "target_sidecar_accessed": False,
                    "final_audit_opened": False, "elapsed_seconds": time.monotonic() - started, "traceback": traceback.format_exc(limit=4)})
        raise


def _generate_arm(binding_path: Path, output_root: Path, arm: str) -> None:
    binding = load_binding(binding_path, output_root, child=True)
    arm_root = output_root / "arms" / arm
    training = _read(arm_root / "TERMINAL_RECEIPT.json")
    if training.get("status") != "COMPLETED" or training.get("optimizer_steps") != 96:
        fail("E04C_TRAINING_RECEIPT_REQUIRED")
    config = _read(ROOT / binding["runtime_config_path"]); manifest = _read(ROOT / binding["checkpoint_manifest_path"])
    _verify_checkpoint(manifest)
    from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, parse_native_grid
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch
    tokenizer, _ = checkpoint_native_tokenizer(Path(manifest["base_path"]), Path(config["native_config_dir"]))
    model = AutoModelForCausalLM.from_pretrained(manifest["base_path"], local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
    model = PeftModel.from_pretrained(model, arm_root / "adapter", is_trainable=False).eval()
    provider = NVARCNativeProvider(model_path=Path(manifest["base_path"]), tokenizer_config_dir=Path(config["native_config_dir"]), device="cuda:0")
    provider.model, provider.tokenizer = model, tokenizer
    from scripts.e04_orientation_v3_no_update import canon, prompt_messages, read_jsonl
    prompts = read_jsonl(ROOT / config["prompt_path"]); rows: list[dict[str, Any]] = []
    started = time.monotonic(); deadline = started + PER_ARM_CAP_SECONDS
    for index, prompt in enumerate(prompts):
        if time.monotonic() >= deadline: fail("E04C_EVALUATION_PER_ARM_CAP")
        generated = provider.generate_many([prompt_messages(prompt)], max_new_tokens=int(config["max_new_tokens"]), context_window=int(config["context_window"]), seeds=[int(config["seed"]) + index])
        if len(generated) != 1: fail("E04C_EVALUATION_ALIGNMENT")
        value = generated[0]; token_ids = [int(token) for token in value.token_ids]; parsed = parse_native_grid(value.text)
        rows.append({"row_index": index, "input_sha256": hashlib.sha256(canon(prompt)).hexdigest(), "generated_token_ids": token_ids,
                     "text": value.text, "parser_valid": parsed is not None, "parsed_grid": None if parsed is None else [list(row) for row in parsed],
                     "prompt_tokens": int(value.prompt_tokens), "completion_tokens": int(value.completion_tokens),
                     "elapsed_seconds": float(value.elapsed_seconds), "effective_batch_size": 1,
                     "physical_returned_token_ids": token_ids, "pad_token_count": 0})
    from scripts.e04_v3_full768_b1_remeasurement import atomic_jsonl, validate_full_raw
    raw = arm_root / "PRIMARY_B1_RAW.jsonl"; atomic_jsonl(raw, rows); validate_full_raw(raw, prompts)
    atomic_json(arm_root / "RAW_GENERATION_RECEIPT.json", {"status": "RAW_GENERATIONS_FROZEN_NO_TARGETS", "arm": arm, "rows": 768,
                "raw_sha256": sha_path(raw), "physical_batch_size": 1, "target_sidecar_accessed": False, "final_audit_opened": False,
                "elapsed_seconds": time.monotonic() - started})


def run_parent(binding_path: Path, output_root: Path) -> None:
    began = time.monotonic(); terminal = output_root / "TERMINAL_RECEIPT.json"
    try:
        binding = load_binding(binding_path, output_root)
        static_schedule_preflight()
        output_root.mkdir(parents=True, exist_ok=False); (output_root / "arms").mkdir()
        atomic_json(output_root / "PREFLIGHT_RECEIPT.json", {"status": "PASS_NO_MODEL_IMPORT", "protocol_id": binding["protocol_id"],
                    "binding_sha256": sha_path(binding_path), "source_commit": binding["source_commit"], "arm_order": list(ARMS),
                    "target_sidecar_accessed": False, "model_imported": False, "optimizer_constructed": False})
        for arm in ARMS:
            if time.monotonic() - began >= JOINT_CAP_SECONDS: fail("E04C_JOINT_CAP")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--binding", str(binding_path), "--output-root", str(output_root), "--train-arm", arm], cwd=ROOT, check=False)
            if result.returncode: fail("E04C_TRAIN_ARM_FAILED:" + arm)
        for arm in ARMS:
            if time.monotonic() - began >= JOINT_CAP_SECONDS: fail("E04C_JOINT_CAP")
            result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--binding", str(binding_path), "--output-root", str(output_root), "--evaluate-arm", arm], cwd=ROOT, check=False)
            if result.returncode: fail("E04C_EVALUATION_ARM_FAILED:" + arm)
        raw = {arm: sha_path(output_root / "arms" / arm / "PRIMARY_B1_RAW.jsonl") for arm in ARMS}
        receipt = {"schema_version": 1, "status": "TRAINING_AND_RAW_GENERATIONS_FROZEN_NO_TARGETS", "protocol_id": binding["protocol_id"],
                   "binding_sha256": sha_path(binding_path), "source_commit": binding["source_commit"], "raw_sha256": raw,
                   "arm_order": list(ARMS), "optimizer_steps_by_arm": {arm: 96 for arm in ARMS}, "parameter_updates_by_arm": {arm: 96 for arm in ARMS},
                   "physical_batch_size": 1, "target_sidecar_accessed": False, "final_audit_opened": False,
                   "wall_seconds": time.monotonic() - began, "joint_runtime_cap_seconds": JOINT_CAP_SECONDS}
        atomic_json(output_root / "RAW_FREEZE_RECEIPT.json", receipt); atomic_json(terminal, {**receipt, "status": "COMPLETE_PENDING_CPU_SCORE"})
    except Exception as error:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal, {"schema_version": 1, "status": "FAILED_OR_PARTIAL", "protocol_id": "E04_C_MATCHED_FIXED_TURN_ROTATION_REPAIR_PILOT_V1",
                    "error": f"{type(error).__name__}:{error}", "target_sidecar_accessed": False, "final_audit_opened": False,
                    "wall_seconds": time.monotonic() - began, "traceback": traceback.format_exc(limit=4)})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path); parser.add_argument("--output-root", type=Path)
    parser.add_argument("--train-arm", choices=ARMS); parser.add_argument("--evaluate-arm", choices=ARMS); parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(static_schedule_preflight(), sort_keys=True)); return
    if not args.binding or not args.output_root or bool(args.train_arm) == bool(args.evaluate_arm) and (args.train_arm or args.evaluate_arm):
        raise SystemExit("E04C_BINDING_OUTPUT_MODE_REQUIRED")
    if args.train_arm: train_arm(args.binding, args.output_root, args.train_arm)
    elif args.evaluate_arm: _generate_arm(args.binding, args.output_root, args.evaluate_arm)
    else: run_parent(args.binding, args.output_root)


if __name__ == "__main__":
    main()
