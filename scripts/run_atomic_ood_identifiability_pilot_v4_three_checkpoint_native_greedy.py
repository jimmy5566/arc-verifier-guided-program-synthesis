"""Target-blind V4 three-checkpoint native-Greedy executor.

The import surface is deliberately CPU-only.  Torch, Transformers, PEFT and
the native provider are imported only by ``execute`` after the immutable
package and a Director-bound runtime binding have passed preflight.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "capability_repair_baseline_v1" / "atomic_ood_identifiability_pilot_v4_three_checkpoint_native_greedy_no_update"
PACKAGE = OUT / "EXECUTION_PACKAGE.json"


class V4ExecutionFailure(RuntimeError):
    pass


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical(row) + b"\n" for row in rows))
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]


def validate_input_row(row: dict[str, Any]) -> None:
    if set(row) != {"episode_id", "cell", "canonical_unit", "turn", "task"}:
        raise V4ExecutionFailure("V4_INPUT_PROMPT_SCHEMA")
    task = row["task"]
    if set(task) != {"train", "test"} or len(task["test"]) != 1 or set(task["test"][0]) != {"input"}:
        raise V4ExecutionFailure("V4_TARGET_BLIND_PROMPT_STRUCTURE")
    if any(set(pair) != {"input", "output"} for pair in task["train"]):
        raise V4ExecutionFailure("V4_DEMONSTRATION_SCHEMA")
    # This is structural target isolation.  Metadata, targets and sidecar paths
    # must not be embedded in the model-facing object.
    if any(token in canonical(task).lower() for token in (b"target", b"sidecar", b"sha256", b"canonical_unit")):
        raise V4ExecutionFailure("V4_TARGET_OR_METADATA_LEAK")


def validate_preimport(package_path: Path = PACKAGE, *, output_root: Path | None = None) -> dict[str, Any]:
    package = read_json(package_path)
    required = {
        "schema_version", "protocol_id", "status", "v4_source", "inputs", "three_checkpoints",
        "native_runtime", "decoder", "batching", "target_isolation", "execution_output",
        "terminal_receipt", "runtime_estimate", "forbidden", "entrypoint_sha256",
        "package_fingerprint_scope",
    }
    if set(package) != required or package["protocol_id"] != "ATOMIC_OOD_IDENTIFIABILITY_PILOT_V4_THREE_CHECKPOINT_NATIVE_GREEDY_NO_UPDATE":
        raise V4ExecutionFailure("V4_PACKAGE_SCHEMA")
    if package["status"] != "CPU_ONLY_FROZEN_AWAITING_DIRECTOR_EXECUTION_AUTHORIZATION":
        raise V4ExecutionFailure("V4_PACKAGE_STATUS")
    if sha(Path(__file__)) != package["entrypoint_sha256"]:
        raise V4ExecutionFailure("V4_ENTRYPOINT_IDENTITY")
    source = package["v4_source"]
    if source["sha256"] != "523b904ab3b712095009d0be3e29151aa8d8a563ac663a15ff3789a0192926bd":
        raise V4ExecutionFailure("V4_SOURCE_IDENTITY")
    for rel, expected in package["inputs"].items():
        path = ROOT / rel
        if not path.is_file() or sha(path) != expected:
            raise V4ExecutionFailure("V4_FROZEN_INPUT_IDENTITY:" + rel)
    prompts = read_jsonl(ROOT / "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl")
    if len(prompts) != 192 or len({row["episode_id"] for row in prompts}) != 192:
        raise V4ExecutionFailure("V4_PROMPT_COVERAGE")
    for row in prompts:
        validate_input_row(row)
    arms = package["three_checkpoints"]
    if [arm["arm_id"] for arm in arms] != ["V7", "E04_E_CONTROL", "E04_E_TREATMENT"]:
        raise V4ExecutionFailure("V4_ARM_ORDER")
    if any(not arm.get("adapter_model_sha256") or not arm.get("adapter_config_sha256") for arm in arms):
        raise V4ExecutionFailure("V4_CHECKPOINT_IDENTITY")
    b = package["batching"]
    if b["primary_requested_batch_size"] != 32 or b["fallback_ladder"] != [32, 16, 8, 4, 1]:
        raise V4ExecutionFailure("V4_BATCH_POLICY")
    subset = b["fixed_b1_subset"]
    ids = subset["episode_ids"]
    if len(ids) != 12 or len(set(ids)) != 12 or set(ids) - {row["episode_id"] for row in prompts}:
        raise V4ExecutionFailure("V4_B1_SUBSET")
    if b["light_numerical_drift_tolerance"] != {"max_parse_valid_rate_delta": 0.08333333333333333, "max_exact_grid_rate_delta": 0.08333333333333333}:
        raise V4ExecutionFailure("V4_NUMERICAL_TOLERANCE")
    if package["target_isolation"]["sealed_target_sidecar_may_open_before_raw_freeze"] is not False:
        raise V4ExecutionFailure("V4_TARGET_ISOLATION")
    if any(item not in package["forbidden"] for item in ("NO_TRAINING", "NO_OPTIMIZER", "NO_BACKWARD", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT")):
        raise V4ExecutionFailure("V4_FORBIDDEN_BOUNDARY")
    if output_root is not None and output_root.exists():
        raise V4ExecutionFailure("V4_FRESH_OUTPUT_REQUIRED")
    return {
        "status": "PASS_V4_THREE_ARM_PREIMPORT",
        "prompt_rows": 192,
        "arms": [arm["arm_id"] for arm in arms],
        "b1_rows_per_arm": 12,
        "model_imported": False,
        "gpu_used": False,
        "target_sidecar_accessed": False,
        "optimizer_steps": 0,
    }


def execute(package_path: Path, runtime_binding_path: Path, output_root: Path) -> None:
    """Run one authorized target-blind collection after preflight.

    The Director-bound runtime binding is intentionally outside the frozen
    package, so the eventual exact checkout SHA can be bound after the package
    commit without a circular package self-hash.
    """
    started = time.monotonic()
    partial: list[dict[str, Any]] = []
    terminal = output_root / "TERMINAL_RECEIPT.json"
    try:
        preflight = validate_preimport(package_path, output_root=output_root)
        binding = read_json(runtime_binding_path)
        package = read_json(package_path)
        if binding.get("package_sha256") != sha(package_path) or binding.get("authorization") != "DIRECTOR_ONCE":
            raise V4ExecutionFailure("V4_RUNTIME_BINDING_IDENTITY")
        if binding.get("output_root") != str(output_root) or binding.get("nonce") != package["execution_output"]["nonce"]:
            raise V4ExecutionFailure("V4_RUNTIME_BINDING_OUTPUT")
        if binding.get("active_hard_cap_seconds") != package["runtime_estimate"]["active_hard_cap_seconds"]:
            raise V4ExecutionFailure("V4_RUNTIME_CAP_BINDING")
        # Dynamic imports establish the actual model boundary.  No target or
        # sidecar is passed to, or opened by, this generator.
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from inference.nvarc_native import checkpoint_native_tokenizer, parse_native_grid
        from scripts.run_unified_native_model_capability_baseline_v2 import prompt
        from huggingface_hub import hf_hub_download

        prompts = read_jsonl(ROOT / "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v4/INPUT_PROMPTS.jsonl")
        deadline = started + int(package["runtime_estimate"]["active_hard_cap_seconds"])
        native = ROOT / package["native_runtime"]["config_identity_path"]
        if sha(native) != package["native_runtime"]["config_identity_sha256"]:
            raise V4ExecutionFailure("V4_NATIVE_TOKENIZER_IDENTITY")
        base_manifest = read_json(ROOT / package["three_checkpoints"][0]["base_manifest_path"])
        base = Path(base_manifest["base_path"])
        for entry in base_manifest["base_files"]:
            p = base / entry["name"]
            if not p.is_file() or p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
                raise V4ExecutionFailure("V4_BASE_IDENTITY:" + entry["name"])
        for arm in package["three_checkpoints"]:
            if time.monotonic() >= deadline:
                raise V4ExecutionFailure("V4_ACTIVE_CAP_EXCEEDED")
            if arm["arm_id"] == "V7":
                adapter = Path(base_manifest["adapter_path"])
                for entry in base_manifest["adapter_files"]:
                    p = adapter / entry["name"]
                    if not p.is_file() or p.stat().st_size != entry["bytes"] or sha(p) != entry["sha256"]:
                        raise V4ExecutionFailure("V4_V7_IDENTITY:" + entry["name"])
            else:
                cache = output_root / "_adapter_cache" / arm["arm_id"]
                cache.mkdir(parents=True, exist_ok=True)
                for filename, expected in (("adapter_model.safetensors", arm["adapter_model_sha256"]), ("adapter_config.json", arm["adapter_config_sha256"])):
                    downloaded = Path(hf_hub_download(arm["repository"]["repo_id"], filename=arm["repository_path"].rsplit("/", 1)[0] + "/" + filename, repo_type="model", revision=arm["repository"]["immutable_revision"], token=True, local_dir=str(cache), force_download=True))
                    if sha(downloaded) != expected:
                        raise V4ExecutionFailure("V4_ADAPTER_IDENTITY:" + arm["arm_id"])
                    os.replace(downloaded, cache / filename)
                adapter = cache
            model = AutoModelForCausalLM.from_pretrained(str(base), local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
            model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False).eval()
            tokenizer, _ = checkpoint_native_tokenizer(base, ROOT / "configs" / "nvarc_native_846d0198")
            records: list[dict[str, Any]] = []
            # Length bucketing is evaluated after tokenization; OOM is the sole
            # fallback cause.  Each accepted row records its effective batch.
            contexts = sorted(((row, prompt(row["task"])) for row in prompts), key=lambda item: (len(tokenizer(item[1], add_special_tokens=False)["input_ids"]), item[0]["episode_id"]))
            cursor, size = 0, 32
            events: list[dict[str, Any]] = []
            while cursor < len(contexts):
                count = min(size, len(contexts) - cursor)
                batch = contexts[cursor:cursor + count]
                try:
                    encoded = tokenizer([text for _, text in batch], return_tensors="pt", padding=True, add_special_tokens=False)
                    encoded = {name: value.to("cuda:0") for name, value in encoded.items()}
                    width = int(encoded["input_ids"].shape[1])
                    with torch.inference_mode():
                        generated = model.generate(**encoded, do_sample=False, num_beams=1, max_new_tokens=128, use_cache=True, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
                except RuntimeError as exc:
                    if "out of memory" not in str(exc).lower() or count == 1:
                        raise
                    next_size = next(value for value in (32, 16, 8, 4, 1) if value < count)
                    events.append({"event": "OOM_FALLBACK", "failed_batch_size": count, "fallback_batch_size": next_size})
                    torch.cuda.empty_cache(); size = next_size; continue
                for index, (row, text) in enumerate(batch):
                    token_ids = generated[index, width:].detach().cpu().tolist()
                    decoded = tokenizer.decode(token_ids, skip_special_tokens=True)
                    grid = parse_native_grid(decoded)
                    records.append({"arm_id": arm["arm_id"], "episode_id": row["episode_id"], "input_sha256": hashlib.sha256(canonical(row)).hexdigest(), "text": decoded, "parser_valid": grid is not None, "parsed_grid": [list(x) for x in grid] if grid is not None else None, "prompt_tokens": int(encoded["attention_mask"][index].sum().item()), "completion_tokens": len(token_ids), "elapsed_seconds": None, "effective_batch_size": count})
                cursor += count
                events.append({"event": "PRIMARY_BATCH", "effective_batch_size": count, "episode_ids": [row["episode_id"] for row, _ in batch]})
                if time.monotonic() >= deadline:
                    raise V4ExecutionFailure("V4_ACTIVE_CAP_EXCEEDED")
            name = arm["arm_id"] + "_RAW_TARGET_BLIND.jsonl"
            atomic_jsonl(output_root / name, records)
            partial.append({"arm_id": arm["arm_id"], "path": name, "sha256": sha(output_root / name), "rows": len(records), "batch_events": events})
            del model
            torch.cuda.empty_cache()
        atomic_json(terminal, {"status": "COMPLETE_NO_UPDATE_TARGET_BLIND", "protocol_id": package["protocol_id"], "runtime_binding_sha256": sha(runtime_binding_path), "preflight": preflight, "raw_evidence": partial, "target_sidecar_accessed": False, "optimizer_steps": 0, "parameter_updates": 0, "backward_calls": 0, "final_audit_opened": False, "wall_seconds": time.monotonic() - started})
    except Exception as exc:
        output_root.mkdir(parents=True, exist_ok=True)
        atomic_json(terminal, {"status": "FAILED_NO_UPDATE_TARGET_BLIND", "error": str(exc), "error_type": type(exc).__name__, "partial_evidence": partial, "target_sidecar_accessed": False, "optimizer_steps": 0, "parameter_updates": 0, "backward_calls": 0, "final_audit_opened": False, "wall_seconds": time.monotonic() - started})
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--package", type=Path, default=PACKAGE)
    parser.add_argument("--runtime-binding", type=Path)
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    if args.self_test:
        print(json.dumps(validate_preimport(args.package), sort_keys=True))
        return
    if args.runtime_binding is None or args.output_root is None:
        raise SystemExit("RUNTIME_BINDING_AND_OUTPUT_ROOT_REQUIRED")
    execute(args.package, args.runtime_binding, args.output_root)


if __name__ == "__main__":
    main()
