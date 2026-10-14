#!/usr/bin/env python3
"""Build the frozen V7 start and train one fresh TTT24 adapter per Eval60 task.

This executable is target blind: it accepts the challenge and cohort registry,
but has no argument for evaluation solutions.  It keeps the immutable base and
V7 adapter untouched, materializes a BF16 merged V7 copy, verifies deterministic
native-Greedy parity on frozen input-only canaries, and then resets the same
rank-256 RS-LoRA initialization before each task-local 24-step adaptation.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.kaggle_l4_parallel_runner import atomic_write_json
from scripts.run_eval3_reference_ttt import _fingerprint, _fit_task

CHALLENGE_SHA256 = "e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"
TASK_IDS_SHA256 = "8da66911f3cafb2652da1e5731dfeafd9100670e68de68d22d93d596c2c3abd9"
V7_ADAPTER_SHA256 = "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def tree_manifest(root: Path) -> dict[str, str]:
    return {str(path.relative_to(root)): sha_file(path) for path in sorted(root.rglob("*")) if path.is_file()}


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_inputs(args: argparse.Namespace) -> tuple[list[str], dict[str, Any]]:
    if sha_file(args.challenge) != CHALLENGE_SHA256:
        raise RuntimeError("Eval60 challenge SHA256 mismatch")
    cohort = read(args.cohort)
    task_ids = list(cohort.get("task_ids", ()))
    if len(task_ids) != 60 or cohort.get("task_ids_sha256") != TASK_IDS_SHA256:
        raise RuntimeError("Eval60 cohort identity mismatch")
    outputs = sum(len((cohort.get("tasks") or {}).get(task, {}).get("test_index_structure", ())) for task in task_ids)
    if outputs != 89:
        raise RuntimeError(f"Eval60 must contain exactly 89 outputs, found {outputs}")
    if sha_file(args.v7_adapter / "adapter_model.safetensors") != V7_ADAPTER_SHA256:
        raise RuntimeError("V7 adapter SHA256 mismatch")
    required = {"rank": 256, "alpha": 32, "ttt_steps": 24, "generation_augmentation_count": 8,
                "learning_rate": 5e-5, "use_rslora": True, "seed": 42}
    config = read(args.reference_config)
    if {key: config.get(key) for key in required} != required:
        raise RuntimeError("reference TTT24 config mismatch")
    return task_ids, config


def merge_v7(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if args.merged_model.exists():
        manifest_path = args.merged_model / "MERGED_V7_MANIFEST.json"
        if not manifest_path.is_file():
            raise RuntimeError("merged V7 directory exists without its manifest")
        manifest = read(manifest_path)
        actual = tree_manifest(args.merged_model)
        expected = dict(manifest["files"])
        actual.pop("MERGED_V7_MANIFEST.json", None)
        if actual != expected:
            raise RuntimeError("existing merged V7 manifest mismatch")
        return manifest
    staging = args.merged_model.with_name(args.merged_model.name + f".staging.{os.getpid()}")
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    base = AutoModelForCausalLM.from_pretrained(
        str(args.base_model), local_files_only=True, trust_remote_code=False,
        torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    model = PeftModel.from_pretrained(base, str(args.v7_adapter), is_trainable=False,
                                      autocast_adapter_dtype=False)
    merged = model.merge_and_unload(safe_merge=True).to(dtype=torch.bfloat16)
    merged.save_pretrained(str(staging), safe_serialization=True, max_shard_size="5GB")
    for name in (
        "config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json",
        "special_tokens_map.json", "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
    ):
        source = args.base_model / name
        if source.is_file() and not (staging / name).is_file():
            # Mounted RunPod storage preserves bytes but intentionally rejects
            # POSIX timestamp mutation.  Scientific identity is the SHA256 of
            # file bytes, so copy no incidental filesystem metadata.
            shutil.copyfile(source, staging / name)
    files = tree_manifest(staging)
    manifest = {
        "schema": "ARC2_V7_BF16_MERGE_V1", "base_model": str(args.base_model),
        "v7_adapter": str(args.v7_adapter), "v7_adapter_sha256": V7_ADAPTER_SHA256,
        "dtype": "torch.bfloat16", "files": files, "files_sha256": sha_json(files),
        "original_assets_modified": False,
    }
    atomic_write_json(staging / "MERGED_V7_MANIFEST.json", manifest)
    os.replace(staging, args.merged_model)
    del merged, model, base
    gc.collect()
    return manifest


def _provider(model_path: Path, native_config: Path, adapter: Path | None = None) -> Any:
    from inference.nvarc_native import NVARCNativeProvider
    provider = NVARCNativeProvider(model_path=model_path, tokenizer_config_dir=native_config, device="cuda:0")
    if adapter is None:
        # Transformers 4.55.4 accepts ``torch_dtype`` here; its Qwen3 model
        # constructor does not accept the newer AutoModel ``dtype`` alias used
        # by the generic provider on later environments.
        import torch
        from transformers import AutoModelForCausalLM
        from inference.nvarc_native import checkpoint_native_tokenizer
        provider.tokenizer, tokenizer_meta = checkpoint_native_tokenizer(model_path, native_config)
        provider.model = AutoModelForCausalLM.from_pretrained(
            str(model_path), local_files_only=True, trust_remote_code=False,
            torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        ).to("cuda:0").eval()
        provider.load_metadata = tokenizer_meta
    else:
        from inference.hf_peft_backend import load_hf_peft_inference
        provider.model, provider.tokenizer, provider.load_metadata = load_hf_peft_inference(
            model_path=model_path, adapter_path=adapter, device="cuda:0", native_config_dir=native_config,
        )
    return provider


def _attach_v7_and_zero_ttt(model: Any, args: argparse.Namespace, config: dict[str, Any]) -> Any:
    """Compose frozen V7 and a fresh zero-output task adapter in one forward."""
    import torch
    from peft import LoraConfig, PeftModel

    model = PeftModel.from_pretrained(
        model, str(args.v7_adapter), adapter_name="foundation",
        is_trainable=False, autocast_adapter_dtype=False,
    )
    torch.manual_seed(int(config["seed"])); torch.cuda.manual_seed_all(int(config["seed"]))
    model.add_adapter("default", LoraConfig(
        r=int(config["rank"]), lora_alpha=int(config["alpha"]),
        target_modules=list(config["target_modules"]), lora_dropout=0.0,
        bias="none", task_type="CAUSAL_LM", use_rslora=True,
    ))
    model.base_model.set_adapter(["foundation", "default"])
    return model


def generate_aug8_evidence(*, model: Any, tokenizer: Any, task: Any, config: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], float]:
    """Run the frozen Greedy AUG8 while retaining valid and invalid raw suffixes."""
    import torch
    from unsloth import FastLanguageModel
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
    from inference.nvarc_native_augmentation import bounded_native_augmentations, transform_tasks_for_augmentations
    from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates

    augmentations = bounded_native_augmentations()[:8]
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    candidates: list[NativeGridCandidate] = []
    raw: list[dict[str, Any]] = []
    started = time.perf_counter(); FastLanguageModel.for_inference(model)
    for view_index, (augmentation, augmented) in enumerate(zip(augmentations, transformed, strict=True)):
        outputs = []; view_tokens = 0; view_seconds = 0.0
        for test_index in range(len(task.test)):
            messages = native_messages_from_training_prefix(prefixes[view_index], augmented.test[test_index].input)
            encoded = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                    return_tensors="pt", return_dict=True)
            prompt_tokens = int(encoded["input_ids"].shape[-1])
            if prompt_tokens > int(config["generation_context_window"]):
                raise RuntimeError("Greedy prompt exceeds frozen context")
            torch.manual_seed(task_seed(task.task_id, 42, f"reference-ttt:{view_index}:{test_index}"))
            torch.cuda.manual_seed_all(torch.initial_seed()); began = time.perf_counter()
            with torch.inference_mode():
                result = model.generate(**{name: value.to(model.device) for name, value in encoded.items()},
                                        max_new_tokens=int(config["max_new_tokens"]), do_sample=False,
                                        eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            suffix = result[0, prompt_tokens:].detach().cpu()
            text = tokenizer.decode(suffix, skip_special_tokens=True)
            parsed = parse_native_grid(text)
            inverse = None if parsed is None else augmentation.inverse_grid(parsed)
            elapsed = time.perf_counter() - began; view_seconds += elapsed; view_tokens += int(suffix.shape[-1])
            raw.append({"view_index": view_index, "augmentation": augmentation.to_dict(), "test_index": test_index,
                        "prompt_tokens": prompt_tokens, "token_ids": [int(value) for value in suffix.tolist()],
                        "decoded_text": text, "parsed_augmented_grid": parsed, "canonical_grid": inverse,
                        "valid": parsed is not None, "elapsed_seconds": elapsed})
            outputs.append(inverse)
            del encoded, result, suffix
        if all(output is not None for output in outputs):
            candidates.append(NativeGridCandidate(augmentation, tuple(tuple(tuple(int(cell) for cell in row) for row in output)
                                                                        for output in outputs), view_tokens, view_seconds))
    deduped = [candidate.to_dict() for candidate in deduplicate_candidates(candidates)]
    return deduped, raw, time.perf_counter() - started


def parity(args: argparse.Namespace, task_ids: list[str], config: dict[str, Any]) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset
    from inference.dynamic_task_scheduler import task_seed
    from inference.nvarc_native import native_messages, parse_native_grid

    tasks = load_dataset(args.challenge)
    canaries = [(task_ids[0], 0), (task_ids[-1], 0)]
    results: dict[str, list[dict[str, Any]]] = {}
    for label in ("frozen_v7", "v7_plus_zero_ttt"):
        if label == "frozen_v7":
            provider = _provider(args.base_model, args.native_config_dir, args.v7_adapter)
        else:
            from transformers import AutoModelForCausalLM
            from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer
            provider = NVARCNativeProvider(model_path=args.base_model, tokenizer_config_dir=args.native_config_dir, device="cuda:0")
            provider.tokenizer, tokenizer_meta = checkpoint_native_tokenizer(args.base_model, args.native_config_dir)
            base = AutoModelForCausalLM.from_pretrained(
                str(args.base_model), local_files_only=True, trust_remote_code=False,
                torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            ).to("cuda:0").eval()
            provider.model = _attach_v7_and_zero_ttt(base, args, config).to("cuda:0").eval()
            provider.load_metadata = tokenizer_meta
        rows = []
        for task_id, test_index in canaries:
            generated = provider.generate(native_messages(tasks[task_id], test_index), max_new_tokens=931,
                                          context_window=16384,
                                          seed=task_seed(task_id, 42, f"v7-merge-canary:{test_index}"))
            rows.append({"task_id": task_id, "test_index": test_index,
                         "token_ids": list(generated.token_ids),
                         "parsed_grid": parse_native_grid(generated.text)})
        results[label] = rows
        del provider.model, provider
        gc.collect(); torch.cuda.empty_cache()
    exact = results["frozen_v7"] == results["v7_plus_zero_ttt"]
    payload = {"schema": "ARC2_V7_DUAL_LORA_COMPOSITION_PARITY_V1", "status": "PASS" if exact else "FAIL",
               "criterion": "exact generated token IDs and parsed native grids for frozen V7 versus V7 plus zero-output TTT on two input-only canaries",
               "canaries": canaries, "results": results}
    atomic_write_json(args.output / "V7_COMPOSITION_PARITY.json", payload)
    if not exact:
        raise RuntimeError("V7 dual-LoRA zero-adapter composition parity failed")
    return payload


def train(args: argparse.Namespace, task_ids: list[str], config: dict[str, Any]) -> None:
    import torch
    from arc.io import load_dataset
    from peft import get_peft_model_state_dict, set_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    tasks = load_dataset(args.challenge)
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.base_model), full_finetuning=False, load_in_4bit=False,
        local_files_only=True, use_gradient_checkpointing=False,
        max_seq_length=int(config["max_sequence_length"]),
    )
    tokenizer, tokenizer_meta = checkpoint_native_tokenizer(args.base_model, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("merged V7 tokenizer differs from native tokenizer")
    model = _attach_v7_and_zero_ttt(model, args, config)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = ".default." in name and (".lora_A." in name or ".lora_B." in name)
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    default_state = {key: value.detach().clone() for key, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or not frozen:
        raise RuntimeError("invalid task-specific adapter partition")
    adapter_before = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
    base_before = {name: _fingerprint(parameter) for name, parameter in frozen[:8]}
    records: dict[str, Any] = {}
    for position, task_id in enumerate(task_ids, start=1):
        task_dir = args.adapter_root / task_id / "depth_024"
        receipt = args.output / "TASK_RECEIPTS" / f"{task_id}.json"
        if receipt.is_file() and task_dir.is_dir():
            prior = read(receipt)
            if prior.get("status") == "COMPLETE" and tree_manifest(task_dir) == prior.get("adapter_files"):
                records[task_id] = prior
                continue
            raise RuntimeError(f"incomplete or mismatched prior task artifacts: {task_id}")
        if task_dir.exists():
            raise RuntimeError(f"refusing to overwrite adapter directory: {task_dir}")
        started = time.perf_counter()
        model.base_model.set_adapter(["foundation", "default"])
        fit = _fit_task(model=model, tokenizer=tokenizer, task=tasks[task_id], config=config,
                        default_state=default_state, adapter_before=adapter_before,
                        base_fingerprints=base_before, adapter_name="default",
                        training_backend="torch")
        if not fit["adapter_updated"] or not fit["base_model_unchanged"]:
            raise RuntimeError(f"{task_id}: task adaptation integrity failure")
        staging = task_dir.with_name(task_dir.name + f".staging.{os.getpid()}")
        staging.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(staging), safe_serialization=True, selected_adapters=["default"])
        files = tree_manifest(staging)
        required = {"adapter_config.json", "adapter_model.safetensors"}
        if not required.issubset(files):
            raise RuntimeError(f"{task_id}: incomplete adapter export")
        os.replace(staging, task_dir)
        greedy, greedy_raw, generation_seconds = generate_aug8_evidence(
            model=model, tokenizer=tokenizer, task=tasks[task_id], config=config)
        invalid = sum(not row["valid"] for row in greedy_raw)
        row = {
            "schema": "ARC2_V7_TTT24_TASK_V1", "status": "COMPLETE", "task_id": task_id,
            "position": position, "task_count": 60, "fit": fit,
            "adapter_path": str(task_dir), "adapter_files": files,
            "adapter_files_sha256": sha_json(files), "greedy_aug8": greedy,
            "greedy_raw_completions": greedy_raw, "greedy_invalid_count": invalid,
            "greedy_generation_seconds": generation_seconds,
            "tokenizer": tokenizer_meta, "elapsed_seconds": time.perf_counter() - started,
            "solutions_opened": False,
        }
        atomic_write_json(receipt, row); records[task_id] = row
        set_peft_model_state_dict(model, {key: value.clone() for key, value in default_state.items()}, adapter_name="default")
        reset = {name: _fingerprint(parameter) for name, parameter in trainable[:8]}
        if reset != adapter_before:
            raise RuntimeError(f"{task_id}: adapter reset failed")
        gc.collect(); torch.cuda.empty_cache()
        print(json.dumps({"event": "V7_TTT24_TASK_COMPLETE", "task_id": task_id,
                          "position": position, "seconds": row["elapsed_seconds"]}, sort_keys=True), flush=True)
    if set(records) != set(task_ids):
        raise RuntimeError("TTT24 task set incomplete")
    manifest = {
        "schema": "ARC2_V7_TTT24_EVAL60_ADAPTER_MANIFEST_V1", "status": "FROZEN",
        "task_ids": task_ids, "task_ids_sha256": TASK_IDS_SHA256,
        "adapter_root": str(args.adapter_root),
        "records": {task: {key: records[task][key] for key in ("adapter_path", "adapter_files", "adapter_files_sha256")}
                    for task in task_ids},
        "target_blind": True, "solutions_opened": False,
    }
    atomic_write_json(args.output / "TTT24_ADAPTER_MANIFEST.json", manifest)
    del model
    gc.collect(); torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("base_model", "v7_adapter", "merged_model", "native_config_dir", "challenge",
                 "cohort", "reference_config", "adapter_root", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--skip-merge-parity", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update({"CUDA_VISIBLE_DEVICES": "0", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                       "TOKENIZERS_PARALLELISM": "false", "TRITON_PTXAS_PATH": str(read(args.reference_config)["ptxas_path"])})
    task_ids, config = validate_inputs(args)
    atomic_write_json(args.output / "PREFLIGHT.json", {
        "status": "PASS", "task_count": 60, "output_count": 89,
        "challenge_sha256": CHALLENGE_SHA256, "task_ids_sha256": TASK_IDS_SHA256,
        "v7_adapter_sha256": V7_ADAPTER_SHA256,
        "composition": "live frozen V7 plus task-local TTT LoRA residuals",
        "bf16_merge_parity": "FAILED_AND_NOT_USED",
        "solutions_opened": False,
    })
    if not args.skip_merge_parity:
        parity(args, task_ids, config)
    train(args, task_ids, config)


if __name__ == "__main__":
    main()
