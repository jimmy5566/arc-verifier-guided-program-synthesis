"""Run the authorized E04 V3 target-blind V7 native greedy generation once.

The worker never opens the target scorer sidecar. It emits and freezes raw
generation before a separate CPU scorer may join targets.
"""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time, traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.e04_orientation_v3_no_update import (
    CAP_SECONDS, E04ExecutionFailure, CONTRACT, atomic_json, atomic_jsonl, canon,
    prompt_messages, read_jsonl, sha_path, validate_contract, validate_raw_generation,
)

def fail(code: str) -> None:
    raise E04ExecutionFailure(code)

def load_binding(path: Path, output_root: Path) -> dict[str, Any]:
    binding = json.loads(path.read_text(encoding="utf-8-sig"))
    required = {
        "schema_version", "protocol_id", "authorization_id", "director_response_sha256", "execution_authorized",
        "source_commit", "worker_sha256", "contract_path", "contract_sha256", "runtime_config_path", "runtime_config_sha256",
        "checkpoint_manifest_path", "checkpoint_manifest_sha256",
        "native_config_dir", "native_config_provenance_path", "native_config_provenance_sha256", "native_config_runtime_identity_path", "native_config_runtime_identity_sha256",
        "output_root", "nonce", "hard_runtime_cap_seconds", "jobs", "retry",
    }
    if set(binding) != required:
        fail("E04_BINDING_SCHEMA")
    if binding["protocol_id"] != "E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE":
        fail("E04_BINDING_PROTOCOL")
    if binding["authorization_id"] != "E04_V3_REPLACEMENT_EXECUTION_ONE_SHOT_CONDITIONAL_20261010":
        fail("E04_BINDING_AUTHORIZATION_ID")
    if len(binding["director_response_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in binding["director_response_sha256"]):
        fail("E04_BINDING_DIRECTOR_RESPONSE_IDENTITY")
    if binding["execution_authorized"] is not True:
        fail("E04_BINDING_EXECUTION_AUTHORIZATION")
    if binding["hard_runtime_cap_seconds"] != CAP_SECONDS or binding["jobs"] != 1 or binding["retry"] is not False:
        fail("E04_BINDING_RUNTIME")
    if str(output_root) != binding["output_root"] or not binding["nonce"]:
        fail("E04_BOUND_OUTPUT_OR_NONCE")
    if sha_path(ROOT / "scripts/run_e04_orientation_v3_fixed_baseline.py") != binding["worker_sha256"]:
        fail("E04_WORKER_HASH")
    if sha_path(path) == "":
        fail("E04_BINDING_UNREADABLE")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != binding["source_commit"]:
        fail("E04_RUNTIME_SOURCE_COMMIT")
    contract = ROOT / binding["contract_path"]
    if not contract.is_file() or sha_path(contract) != binding["contract_sha256"]:
        fail("E04_CONTRACT_HASH")
    runtime_config = ROOT / binding["runtime_config_path"]
    if not runtime_config.is_file() or sha_path(runtime_config) != binding["runtime_config_sha256"]:
        fail("E04_RUNTIME_CONFIG_HASH")
    validate_contract(contract)
    for key in ("checkpoint_manifest_path", "native_config_provenance_path", "native_config_runtime_identity_path"):
        candidate = ROOT / binding[key]
        if not candidate.is_file():
            fail("E04_RUNTIME_IDENTITY_MANIFEST_MISSING:" + key)
    if sha_path(ROOT / binding["checkpoint_manifest_path"]) != binding["checkpoint_manifest_sha256"]:
        fail("E04_CHECKPOINT_MANIFEST_HASH")
    if sha_path(ROOT / binding["native_config_provenance_path"]) != binding["native_config_provenance_sha256"]:
        fail("E04_NATIVE_CONFIG_PROVENANCE_HASH")
    if sha_path(ROOT / binding["native_config_runtime_identity_path"]) != binding["native_config_runtime_identity_sha256"]:
        fail("E04_NATIVE_CONFIG_RUNTIME_IDENTITY_MANIFEST_HASH")
    if output_root.exists() and any(output_root.iterdir()):
        fail("E04_FRESH_OUTPUT_REQUIRED")
    return binding

def verify_runtime_native_config_identity(binding: dict[str, Any]) -> None:
    runtime_identity = json.loads((ROOT / binding["native_config_runtime_identity_path"]).read_text(encoding="utf-8-sig"))
    runtime_config_dir = Path(binding["native_config_dir"])
    if runtime_identity.get("kind") != "E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY" or runtime_identity.get("schema_version") != 1:
        fail("E04_NATIVE_CONFIG_RUNTIME_IDENTITY_SCHEMA")
    files = runtime_identity.get("files")
    if not isinstance(files, list) or not files:
        fail("E04_NATIVE_CONFIG_RUNTIME_IDENTITY_SCHEMA")
    for entry in files:
        vendored = ROOT / entry["vendored_path"]
        runtime_file = runtime_config_dir / entry["runtime_relative_path"]
        if (not vendored.is_file() or not runtime_file.is_file() or vendored.stat().st_size != entry["bytes"]
                or runtime_file.stat().st_size != entry["bytes"] or sha_path(vendored) != entry["sha256"]
                or sha_path(runtime_file) != entry["sha256"]):
            fail("E04_NATIVE_CONFIG_RUNTIME_IDENTITY:" + entry["runtime_relative_path"])


def raw_record(index: int, prompt: dict[str, Any], generation: Any, parse: Any, batch_size: int) -> dict[str, Any]:
    parsed = parse(generation.text)
    grid = [list(row) for row in parsed] if parsed is not None else None
    return {
        "row_index": index,
        "input_sha256": hashlib.sha256(canon(prompt)).hexdigest(),
        "text": generation.text,
        "parser_valid": parsed is not None,
        "parsed_grid": grid,
        "prompt_tokens": int(generation.prompt_tokens),
        "completion_tokens": int(generation.completion_tokens),
        "elapsed_seconds": float(generation.elapsed_seconds),
        "effective_batch_size": batch_size,
    }

def capacity_failure(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "cuda error: out of memory" in message

def check_deadline(deadline: float) -> None:
    if time.monotonic() >= deadline:
        fail("E04_RUNTIME_CAP_EXCEEDED")

def generate_primary(provider: Any, prompts: list[dict[str, Any]], *, rung: int, deadline: float, parse: Any, config: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for start in range(0, len(prompts), rung):
        check_deadline(deadline)
        batch = prompts[start:start + rung]
        messages = [prompt_messages(prompt) for prompt in batch]
        generations = provider.generate_many(messages, max_new_tokens=int(config["max_new_tokens"]),
                                             context_window=int(config["context_window"]),
                                             seeds=[int(config["seed"]) + index for index in range(start, start + len(batch))])
        if len(generations) != len(batch):
            fail("E04_GENERATION_BATCH_ALIGNMENT")
        records.extend(raw_record(start + offset, prompt, generation, parse, len(batch))
                       for offset, (prompt, generation) in enumerate(zip(batch, generations, strict=True)))
    return records

def run(config_path: Path, binding_path: Path, output_root: Path) -> None:
    started = time.monotonic()
    terminal = output_root / "TERMINAL_RECEIPT.json"
    try:
        binding = load_binding(binding_path, output_root)
        config = json.loads(config_path.read_text(encoding="utf-8-sig"))
        required = {"schema_version", "protocol_id", "seed", "max_new_tokens", "context_window", "contract_path", "contract_sha256"}
        if set(config) != required or config["protocol_id"] != binding["protocol_id"]:
            fail("E04_RUNTIME_CONFIG_SCHEMA")
        if config_path.resolve() != (ROOT / binding["runtime_config_path"]).resolve() or sha_path(config_path) != binding["runtime_config_sha256"]:
            fail("E04_RUNTIME_CONFIG_BINDING")
        contract = ROOT / config["contract_path"]
        if sha_path(contract) != config["contract_sha256"] or config["contract_sha256"] != binding["contract_sha256"]:
            fail("E04_RUNTIME_CONFIG_CONTRACT")
        validation = validate_contract(contract)
        if output_root.exists():
            fail("E04_FRESH_OUTPUT_REQUIRED")
        output_root.mkdir(parents=True, exist_ok=False)
        deadline = started + CAP_SECONDS
        check_deadline(deadline)
        # Runtime-only imports: all prior source/config gates completed before
        # model import. Verify exact base and V7 adapter bytes before loading.
        checkpoint = json.loads((ROOT / binding["checkpoint_manifest_path"]).read_text(encoding="utf-8-sig"))
        def verify_entries(folder: Path, entries: list[dict[str, Any]], label: str) -> None:
            for entry in entries:
                candidate = folder / entry["name"]
                if (not candidate.is_file() or candidate.stat().st_size != entry["bytes"]
                        or sha_path(candidate) != entry["sha256"]):
                    fail("E04_RUNTIME_" + label + "_IDENTITY:" + entry["name"])
        base_path = Path(checkpoint["base_path"])
        adapter_path = Path(checkpoint["adapter_path"])
        verify_entries(base_path, checkpoint["base_files"], "BASE")
        verify_entries(adapter_path, checkpoint["adapter_files"], "V7_ADAPTER")
        verify_runtime_native_config_identity(binding)
        from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, parse_native_grid
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        import torch
        tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(base_path, Path(binding["native_config_dir"]))
        model = AutoModelForCausalLM.from_pretrained(str(base_path), local_files_only=True, trust_remote_code=False,
                                                     torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False).eval()
        provider = NVARCNativeProvider(model_path=base_path, tokenizer_config_dir=Path(binding["native_config_dir"]), device="cuda:0")
        provider.model, provider.tokenizer = model, tokenizer
        provider.load_metadata = {**tokenizer_metadata, "checkpoint_identity": "RECONSTRUCTED_FOUNDATION_V2_V7",
                                  "base_path": str(base_path), "adapter_path": str(adapter_path),
                                  "model_vram_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1)}
        prompts = read_jsonl(ROOT / json.loads(contract.read_text(encoding="utf-8"))["input_prompts_path"])
        failures: list[dict[str, str]] = []
        primary: list[dict[str, Any]] | None = None
        selected_rung: int | None = None
        for rung in (32, 16, 8, 4, 1):
            try:
                primary = generate_primary(provider, prompts, rung=rung, deadline=deadline, parse=parse_native_grid, config=config)
                selected_rung = rung
                break
            except RuntimeError as error:
                if not capacity_failure(error):
                    raise
                failures.append({"rung": str(rung), "error": type(error).__name__})
                import torch
                torch.cuda.empty_cache()
        if primary is None or selected_rung is None:
            fail("E04_ALL_FALLBACK_RUNGS_CAPACITY_FAILED")
        expected_primary = list(range(768))
        validate_raw_generation(output_root / "not-written.jsonl", expected_primary) if False else None
        atomic_jsonl(output_root / "PRIMARY_RAW_GENERATIONS.jsonl", primary)
        validate_raw_generation(output_root / "PRIMARY_RAW_GENERATIONS.jsonl", expected_primary)
        subset = json.loads((ROOT / json.loads(contract.read_text(encoding="utf-8"))["batch1_sensitivity_manifest_path"]).read_text(encoding="utf-8"))
        sensitivity_prompts = [prompts[index] for index in subset["row_indexes"]]
        sensitivity: list[dict[str, Any]] = []
        for index, prompt in zip(subset["row_indexes"], sensitivity_prompts, strict=True):
            check_deadline(deadline)
            generation = provider.generate_many([prompt_messages(prompt)], max_new_tokens=int(config["max_new_tokens"]),
                                                context_window=int(config["context_window"]), seeds=[int(config["seed"]) + index])[0]
            sensitivity.append(raw_record(index, prompt, generation, parse_native_grid, 1))
        atomic_jsonl(output_root / "BATCH1_SENSITIVITY_RAW_GENERATIONS.jsonl", sensitivity)
        validate_raw_generation(output_root / "BATCH1_SENSITIVITY_RAW_GENERATIONS.jsonl", subset["row_indexes"],
                                expected_input_sha=subset["row_input_sha256"])
        raw_receipt = {
            "schema_version": 1, "status": "RAW_GENERATIONS_FROZEN_NO_TARGETS",
            "protocol_id": binding["protocol_id"], "source_commit": binding["source_commit"],
            "binding_sha256": sha_path(binding_path), "contract_sha256": binding["contract_sha256"],
            "primary_raw_sha256": sha_path(output_root / "PRIMARY_RAW_GENERATIONS.jsonl"),
            "batch1_raw_sha256": sha_path(output_root / "BATCH1_SENSITIVITY_RAW_GENERATIONS.jsonl"),
            "primary_rows": 768, "batch1_rows": 64, "selected_primary_rung": selected_rung,
            "capacity_failures_before_output": failures, "target_sidecar_accessed": False,
            "optimizer_steps": 0, "parameter_updates": 0, "backward_calls": 0,
            "generation_calls": 832, "final_audit_opened": False,
            "wall_seconds": time.monotonic() - started, "runtime_cap_seconds": CAP_SECONDS,
            "model_load_metadata": provider.load_metadata,
        }
        atomic_json(output_root / "RAW_GENERATIONS_FREEZE_RECEIPT.json", raw_receipt)
        atomic_json(terminal, {**raw_receipt, "status": "COMPLETE_NO_UPDATE", "terminal": True})
    except Exception as error:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal, {"schema_version": 1, "status": "FAILED_NO_UPDATE", "error": str(error),
                               "error_type": type(error).__name__, "optimizer_steps": 0,
                               "parameter_updates": 0, "backward_calls": 0, "target_sidecar_accessed": False,
                               "final_audit_opened": False, "wall_seconds": time.monotonic() - started})
        raise

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path)
    parser.add_argument("--binding", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(validate_contract(), sort_keys=True))
        return
    if not args.config or not args.binding or not args.output_root:
        raise SystemExit("CONFIG_BINDING_AND_OUTPUT_REQUIRED")
    run(args.config, args.binding, args.output_root)

if __name__ == "__main__":
    main()

