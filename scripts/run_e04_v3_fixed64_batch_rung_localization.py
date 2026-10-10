"""Run one authorized E04 V3 fixed-64 batch-rung localization job.

The parent spawns four sequential child processes, one freshly initialized
model process per arm.  It freezes target-blind raw files before a separate
CPU scorer is allowed to join the sealed target sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts.e04_orientation_v3_no_update import canon, prompt_messages
from scripts.e04_v3_fixed64_batch_rung_localization import (
    ARMS, CAP_SECONDS, CONFIG, PROTOCOL_ID, LocalizationFailure, atomic_json, atomic_jsonl,
    load_and_validate_config, read_json, read_jsonl, sha_path, validate_arm_raw,
)


def fail(code: str) -> None:
    raise LocalizationFailure(code)


def load_binding(path: Path, output_root: Path, *, allow_parent_workspace: bool = False) -> dict[str, Any]:
    binding = read_json(path)
    required = {"schema_version", "protocol_id", "authorization_id", "director_response_sha256", "execution_authorized",
                "source_commit", "worker_sha256", "config_path", "config_sha256", "output_root", "nonce",
                "hard_runtime_cap_seconds", "jobs", "retry"}
    if set(binding) != required or binding["protocol_id"] != PROTOCOL_ID:
        fail("E04_FIXED64_BINDING_SCHEMA")
    if binding["execution_authorized"] is not True or binding["hard_runtime_cap_seconds"] != CAP_SECONDS or binding["jobs"] != 1 or binding["retry"] is not False:
        fail("E04_FIXED64_BINDING_AUTHORIZATION")
    if str(output_root).replace("\\", "/") != binding["output_root"] or not isinstance(binding["nonce"], str) or not binding["nonce"]:
        fail("E04_FIXED64_BINDING_OUTPUT")
    if len(binding["director_response_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in binding["director_response_sha256"]):
        fail("E04_FIXED64_BINDING_DIRECTOR_IDENTITY")
    if sha_path(ROOT / "scripts/run_e04_v3_fixed64_batch_rung_localization.py") != binding["worker_sha256"]:
        fail("E04_FIXED64_WORKER_HASH")
    if sha_path(ROOT / binding["config_path"]) != binding["config_sha256"]:
        fail("E04_FIXED64_CONFIG_HASH")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip() != binding["source_commit"]:
        fail("E04_FIXED64_SOURCE_COMMIT")
    if output_root.exists() and any(output_root.iterdir()):
        # The parent owns freshness.  A child runs only after the parent has
        # written its target-blind preflight receipt and raw-arm directory.
        # Reapplying the parent's empty-directory gate here made every arm
        # fail before model import.
        allowed = {"PREFLIGHT_RECEIPT.json", "raw_arms"}
        entries = {item.name for item in output_root.iterdir()}
        if not allow_parent_workspace or not entries <= allowed:
            fail("E04_FIXED64_FRESH_OUTPUT")
    return binding


def _verify_runtime(config: dict[str, Any]) -> tuple[Any, Any, Any]:
    """Verify identities before imports, then return fresh loaded model/provider."""
    checkpoint = read_json(ROOT / config["checkpoint_manifest_path"])
    def entries(folder: Path, values: list[dict[str, Any]], label: str) -> None:
        for item in values:
            candidate = folder / item["name"]
            if not candidate.is_file() or candidate.stat().st_size != item["bytes"] or sha_path(candidate) != item["sha256"]:
                fail("E04_FIXED64_RUNTIME_" + label + "_IDENTITY:" + item["name"])
    base, adapter = Path(checkpoint["base_path"]), Path(checkpoint["adapter_path"])
    entries(base, checkpoint["base_files"], "BASE"); entries(adapter, checkpoint["adapter_files"], "V7_ADAPTER")
    from scripts.run_e04_orientation_v3_fixed_baseline import verify_runtime_native_config_identity
    verify_runtime_native_config_identity({"native_config_runtime_identity_path": config["runtime_identity_path"], "native_config_dir": config["native_config_dir"]})
    from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, parse_native_grid
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    import torch
    tokenizer, _ = checkpoint_native_tokenizer(base, Path(config["native_config_dir"]))
    model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, trust_remote_code=False,
                                                  torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
    model = PeftModel.from_pretrained(model, adapter, is_trainable=False).eval()
    provider = NVARCNativeProvider(model_path=base, tokenizer_config_dir=Path(config["native_config_dir"]), device="cuda:0")
    provider.model, provider.tokenizer = model, tokenizer
    return provider, parse_native_grid, torch


def _raw(index: int, prompt: dict[str, Any], generation: Any, parser: Any, arm: int) -> dict[str, Any]:
    tokens = [int(token) for token in generation.token_ids]
    grid = parser(generation.text)
    return {"row_index": index, "input_sha256": hashlib.sha256(canon(prompt)).hexdigest(),
            "generated_token_ids": tokens, "text": generation.text, "parser_valid": grid is not None,
            "parsed_grid": None if grid is None else [list(row) for row in grid], "prompt_tokens": int(generation.prompt_tokens),
            "completion_tokens": int(generation.completion_tokens), "elapsed_seconds": float(generation.elapsed_seconds),
            "effective_batch_size": arm, "physical_returned_token_ids": tokens, "pad_token_count": 0}


def run_arm(config: dict[str, Any], arm: int, output: Path) -> None:
    if arm not in ARMS:
        fail("E04_FIXED64_ARM")
    prompts = read_jsonl(ROOT / config["prompt_path"]); cohort = read_json(ROOT / config["cohort_path"]); indexes = cohort["row_indexes"]
    provider, parser, torch = _verify_runtime(config)
    rows: list[dict[str, Any]] = []
    for start in range(0, len(indexes), arm):
        selected = indexes[start:start + arm]
        generated = provider.generate_many([prompt_messages(prompts[index]) for index in selected],
                                           max_new_tokens=int(config["max_new_tokens"]), context_window=int(config["context_window"]),
                                           seeds=[int(config["seed"]) + index for index in selected])
        if len(generated) != len(selected): fail("E04_FIXED64_GENERATION_ALIGNMENT")
        rows.extend(_raw(index, prompts[index], value, parser, arm) for index, value in zip(selected, generated, strict=True))
    atomic_jsonl(output, rows)
    del provider, torch


def run_parent(config_path: Path, binding_path: Path, output_root: Path) -> None:
    started = time.monotonic(); terminal = output_root / "TERMINAL_RECEIPT.json"
    try:
        binding = load_binding(binding_path, output_root); config = load_and_validate_config(config_path)
        if config_path.resolve() != (ROOT / binding["config_path"]).resolve(): fail("E04_FIXED64_CONFIG_BINDING")
        output_root.mkdir(parents=True, exist_ok=False); arms_root = output_root / "raw_arms"; arms_root.mkdir()
        atomic_json(output_root / "PREFLIGHT_RECEIPT.json", {"status":"PASS_NO_MODEL_IMPORT", "protocol_id":PROTOCOL_ID,
                    "source_commit":binding["source_commit"], "binding_sha256":sha_path(binding_path), "config_sha256":sha_path(config_path),
                    "arms":list(ARMS), "target_sidecar_accessed":False, "model_imported":False, "optimizer_steps":0})
        for arm in ARMS:
            if time.monotonic() - started >= CAP_SECONDS: fail("E04_FIXED64_RUNTIME_CAP")
            child_output = arms_root / f"B{arm}_RAW.jsonl"
            command = [sys.executable, str(Path(__file__).resolve()), "--config", str(config_path), "--binding", str(binding_path),
                       "--output-root", str(output_root), "--arm", str(arm), "--arm-output", str(child_output)]
            completed = subprocess.run(command, cwd=ROOT, check=False)
            if completed.returncode: fail("E04_FIXED64_ARM_PROCESS_FAILED:B" + str(arm))
        cohort = read_json(ROOT / config["cohort_path"]); hashes = cohort["row_input_sha256"]
        raw_hashes = {}
        for arm in ARMS:
            raw = arms_root / f"B{arm}_RAW.jsonl"; validate_arm_raw(raw, cohort["row_indexes"], hashes, arm); raw_hashes[str(arm)] = sha_path(raw)
        receipt = {"schema_version":1,"status":"RAW_ARMS_FROZEN_NO_TARGETS","protocol_id":PROTOCOL_ID,"source_commit":binding["source_commit"],
                   "binding_sha256":sha_path(binding_path),"config_sha256":sha_path(config_path),"raw_arm_sha256":raw_hashes,"arms":list(ARMS),
                   "target_sidecar_accessed":False,"optimizer_steps":0,"parameter_updates":0,"backward_calls":0,"final_audit_opened":False,
                   "wall_seconds":time.monotonic()-started,"runtime_cap_seconds":CAP_SECONDS}
        atomic_json(output_root / "RAW_ARMS_FREEZE_RECEIPT.json", receipt); atomic_json(terminal,{**receipt,"status":"COMPLETE_NO_UPDATE","terminal":True})
    except Exception as error:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal,{"schema_version":1,"status":"FAILED_NO_UPDATE","error":str(error),"error_type":type(error).__name__,"target_sidecar_accessed":False,"optimizer_steps":0,"parameter_updates":0,"backward_calls":0,"final_audit_opened":False,"wall_seconds":time.monotonic()-started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config",type=Path); parser.add_argument("--binding",type=Path); parser.add_argument("--output-root",type=Path); parser.add_argument("--arm",type=int); parser.add_argument("--arm-output",type=Path); parser.add_argument("--self-test",action="store_true"); args=parser.parse_args()
    if args.self_test:
        print(json.dumps({"config":load_and_validate_config(),"model_imported":False,"gpu_used":False},sort_keys=True)); return
    if not args.config or not args.binding or not args.output_root: raise SystemExit("CONFIG_BINDING_OUTPUT_REQUIRED")
    if args.arm is not None:
        config=load_and_validate_config(args.config); load_binding(args.binding,args.output_root,allow_parent_workspace=True)
        if not args.arm_output: raise SystemExit("ARM_OUTPUT_REQUIRED")
        if args.arm_output.exists(): fail("E04_FIXED64_ARM_OUTPUT_NOT_FRESH")
        run_arm(config,args.arm,args.arm_output); return
    run_parent(args.config,args.binding,args.output_root)

if __name__ == "__main__": main()
