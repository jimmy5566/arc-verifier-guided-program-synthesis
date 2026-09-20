"""Build a one-task, target-blind official-NVARC adapter preflight notebook.

The notebook intentionally uses the public reference notebook's upstream
Unsloth/FlashAttention kernel source.  It does not generate test outputs, read
evaluation solutions, or invoke any candidate search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REFERENCE_SOURCE = "sorokin/pip-install-unsloth-flash-patch"
REFERENCE_NOTEBOOK = "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"
TASK_ID = "5dbc8537"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _code() -> str:
    # A raw string keeps the emitted notebook independent of the local source
    # package.  This is deliberate: the upstream kernel source provides the
    # reference environment before this cell starts.
    return r'''import gc
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import random
import subprocess
import time
import traceback
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from datasets import Dataset
from transformers import DataCollatorForLanguageModeling
from unsloth import FastLanguageModel

OUT = Path("/kaggle/working/reports/reference_ttt_preflight")
OUT.mkdir(parents=True, exist_ok=True)
TASK_ID = "5dbc8537"
MAX_PREFLIGHT_STEPS = 4
MAX_SEQUENCE_LENGTH = 8192
REFERENCE_EXPECTED = {
    "python": "3.11.13 (recorded in reference notebook metadata)",
    "torch": "2.9.1 (recorded in upstream installer output)",
    "transformers": "4.55.4 (recorded in upstream installer output)",
    "unsloth": "2025.9.7 (recorded in upstream installer)",
    "unsloth_zoo": "2025.9.9 (recorded in upstream installer)",
    "torchao": "0.10.0 (recorded as pre-existing in upstream installer output)",
    "trl": "0.22.2 (recorded in upstream installer output)",
    "bitsandbytes": "0.48.2 (recorded in upstream installer output)",
    "xformers": "0.0.33.post2 (recorded in upstream installer output)",
    "triton": "3.5.1 (recorded in upstream installer output)",
    "peft": "UNVERIFIED: no exact standalone version recorded in public notebook source",
}

def package_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "NOT_INSTALLED"

def atomic_json(path, payload):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)

def grid_text(grid):
    return "\n".join("".join(str(int(cell)) for cell in row) for row in grid)

def transform_grid(grid, geometry):
    a = np.asarray(grid)
    if geometry == "identity": return a
    if geometry == "transpose": return np.swapaxes(a, 0, 1)
    if geometry == "rot90": return np.rot90(a, 1)
    if geometry == "rot180": return np.rot90(a, 2)
    if geometry == "rot270": return np.rot90(a, 3)
    if geometry == "anti_transpose": return np.rot90(np.swapaxes(a, 0, 1), 2)
    if geometry == "flip_lr": return np.fliplr(a)
    if geometry == "flip_ud": return np.flipud(a)
    raise ValueError(geometry)

def colour_permute(grid, permutation):
    return np.asarray(permutation, dtype=np.int64)[np.asarray(grid, dtype=np.int64)]

def full_dialogue(train_pairs):
    text = ""
    for pair in train_pairs:
        text += "<|im_start|>user\n" + grid_text(pair["input"])
        text += "<|im_end|><|im_start|>assistant\n" + grid_text(pair["output"])
        text += "<|im_end|>"
    return text

def documented_variants(train_pairs):
    # Mirrors ArcDataset.augment(n=16, shfl_keys=True, seed=1) semantically:
    # eight D4 grid views, 16 arbitrary colour permutations, then a full
    # random train-pair permutation.  The direct notebook source does not
    # publish a separately versioned augmentation API, so only the public
    # sequence ordering is marked UNVERIFIED in ENVIRONMENT_LOCK.
    rng = np.random.RandomState(1)
    geometries = ("identity", "transpose", "rot90", "rot180", "rot270", "anti_transpose", "flip_lr", "flip_ud")
    variants = []
    for _ in range(16):
        permutation = rng.permutation(10).tolist()
        for geometry in geometries:
            transformed = []
            for pair in train_pairs:
                transformed.append({
                    "input": colour_permute(transform_grid(pair["input"], geometry), permutation).tolist(),
                    "output": colour_permute(transform_grid(pair["output"], geometry), permutation).tolist(),
                })
            order = rng.permutation(len(transformed)).tolist()
            variants.append([transformed[index] for index in order])
    return variants

class CompletionOnlyCollator(DataCollatorForLanguageModeling):
    # Directly follows the reference notebook's assistant-only loss masking.
    def torch_call(self, examples):
        batch = super().torch_call(examples)
        for index in range(len(examples)):
            labels = batch["input_ids"][index].clone()
            starts = sorted(
                np.where(labels.cpu().numpy() == 11)[0].tolist() +
                np.where(labels.cpu().numpy() == 12)[0].tolist()
            )
            ends = np.where(labels.cpu().numpy() == 15)[0].tolist()
            batch["labels"][index, :] = -100
            for role_index, (start, end) in enumerate(zip(starts, ends)):
                if role_index % 2 == 1:
                    batch["labels"][index, start + 2:end + 1] = labels[start + 2:end + 1]
        return batch

def fingerprint(tensor):
    sample = tensor.detach().reshape(-1)[:4096].to("cpu", dtype=torch.float32).numpy().tobytes()
    return hashlib.sha256(sample).hexdigest()

def cuda_snapshot():
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }

runtime = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "transformers": package_version("transformers"),
    "unsloth": package_version("unsloth"),
    "unsloth_zoo": package_version("unsloth-zoo"),
    "torchao": package_version("torchao"),
    "peft": package_version("peft"),
    "trl": package_version("trl"),
    "bitsandbytes": package_version("bitsandbytes"),
    "xformers": package_version("xformers"),
    "triton": package_version("triton"),
    "gpu": torch.cuda.get_device_name(0),
    "gpu_count_visible": torch.cuda.device_count(),
    "internet": os.getenv("KAGGLE_KERNEL_INTERNET_ENABLED", "UNSET"),
}
environment_lock = {
    "experiment_id": "ARC2_OFFICIAL_NVARC_SINGLE_TASK_ADAPTER_PREFLIGHT",
    "reference": {
        "repository": "https://github.com/1ytic/NVARC",
        "notebook": "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue",
        "kernel_source": "sorokin/pip-install-unsloth-flash-patch",
        "expected_recorded_versions": REFERENCE_EXPECTED,
        "unverified": ["exact standalone peft version", "exact public model file SHA256", "exact augmentation item ordering"],
    },
    "runtime": runtime,
    "parity_method": "Reference kernel source attached through kernel_sources; actual runtime versions captured before model load.",
    "solutions_opened": False,
}
atomic_json(OUT / "ENVIRONMENT_LOCK.json", environment_lock)
print(json.dumps({"event": "ENVIRONMENT_LOCKED", "runtime": runtime}, sort_keys=True), flush=True)

result = {
    "experiment_id": "ARC2_OFFICIAL_NVARC_SINGLE_TASK_ADAPTER_PREFLIGHT",
    "task_id": TASK_ID,
    "solutions_opened": False,
    "generation_or_dfs_run": False,
    "reference_adapter": {
        "rank": 256,
        "alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        "use_rslora": True,
        "bf16": True,
        "gradient_checkpointing": False,
        "optimizer": "adamw_torch",
        "learning_rate": 5e-5,
        "scheduler": "cosine",
        "intended_epoch_sequences": 128,
    },
    "measured_steps": MAX_PREFLIGHT_STEPS,
    "status": "RUNNING",
}

try:
    if runtime["internet"].strip().lower() in {"1", "true", "yes"}:
        raise RuntimeError("Internet must be off for this preflight")
    random.seed(42); np.random.seed(42); torch.manual_seed(42); torch.cuda.manual_seed_all(42)
    challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    task = json.loads(challenge.read_text(encoding="utf-8"))[TASK_ID]
    train_pairs = task["train"]
    variants = documented_variants(train_pairs)
    result["documented_variant_count"] = len(variants)
    model_path = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    torch.cuda.reset_peak_memory_stats()
    print(json.dumps({"event": "MODEL_LOAD_START", "task_id": TASK_ID, "model_path": str(model_path)}, sort_keys=True), flush=True)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(model_path), full_finetuning=False, load_in_4bit=False,
        local_files_only=True, use_gradient_checkpointing=False,
        max_seq_length=MAX_SEQUENCE_LENGTH,
    )
    result["after_model_load"] = cuda_snapshot()
    print(json.dumps({"event": "MODEL_LOAD_COMPLETE", "memory": result["after_model_load"]}, sort_keys=True), flush=True)
    model = FastLanguageModel.get_peft_model(
        model, r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32, lora_dropout=0.0, bias="none", use_gradient_checkpointing=False,
        random_state=42, use_rslora=True, loftq_config=None,
    )
    for _, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    frozen = [(name, parameter) for name, parameter in model.named_parameters() if not parameter.requires_grad]
    if not trainable or not frozen:
        raise RuntimeError("adapter parameter partition is invalid")
    result["adapter_creation"] = "PASS"
    result["adapter_parameter_count"] = int(sum(parameter.numel() for _, parameter in trainable))
    result["base_parameter_count"] = int(sum(parameter.numel() for _, parameter in frozen))
    base_fingerprints = {name: fingerprint(parameter) for name, parameter in frozen[:8]}
    adapter_before = {name: fingerprint(parameter) for name, parameter in trainable[:8]}
    result["after_adapter_creation"] = cuda_snapshot()
    model = FastLanguageModel.for_training(model)
    texts = [full_dialogue(variant) for variant in variants]
    tokenized = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    kept = [tokens for tokens in tokenized if len(tokens) <= MAX_SEQUENCE_LENGTH]
    result["tokenization"] = {"full_sequences": len(tokenized), "kept_sequences": len(kept), "dropped_over_context": len(tokenized) - len(kept), "max_tokens": max(map(len, tokenized))}
    if not kept:
        raise RuntimeError("all documented train sequences exceed reference context")
    collator = CompletionOnlyCollator(tokenizer=tokenizer, mlm=False)
    optimizer = torch.optim.AdamW([parameter for _, parameter in trainable], lr=5e-5, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: min(1.0, (step + 1) / max(1, int(0.1 * 128))) * 0.5 * (1 + np.cos(np.pi * step / 128))
    )
    losses = []
    step_seconds = []
    for step in range(MAX_PREFLIGHT_STEPS):
        started = time.perf_counter()
        batch = collator([{"input_ids": kept[step % len(kept)]}])
        batch = {key: value.to(model.device) for key, value in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        outputs = model(**batch, use_cache=False, return_dict=True)
        loss = outputs.loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        optimizer.step(); scheduler.step()
        torch.cuda.synchronize()
        losses.append(float(loss.detach().cpu()))
        step_seconds.append(time.perf_counter() - started)
        print(json.dumps({"event": "TTT_STEP", "step": step + 1, "loss": losses[-1], "seconds": step_seconds[-1]}, sort_keys=True), flush=True)
    adapter_after = {name: fingerprint(parameter) for name, parameter in trainable[:8]}
    base_after = {name: fingerprint(parameter) for name, parameter in frozen[:8]}
    result.update({
        "forward_backward": "PASS",
        "losses": losses,
        "loss_finite": all(np.isfinite(losses)),
        "adapter_updated": "PASS" if adapter_before != adapter_after else "FAIL",
        "base_model_unchanged": "PASS" if base_fingerprints == base_after else "FAIL",
        "seconds_per_ttt_step": float(sum(step_seconds) / len(step_seconds)),
        "projected_seconds_per_task": float(sum(step_seconds) / len(step_seconds) * 128),
        "projected_eval6_gpu_seconds": float(sum(step_seconds) / len(step_seconds) * 128 * 6),
        "projected_240_task_gpu_seconds": float(sum(step_seconds) / len(step_seconds) * 128 * 240),
        "peak_vram_gb": float(torch.cuda.max_memory_allocated() / 1024**3),
        "before_unload": cuda_snapshot(),
    })
    del optimizer, scheduler, batch, outputs, model, tokenizer
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()
    result["after_unload"] = cuda_snapshot()
    result["reset_unload"] = "PASS"
    required = [result["adapter_creation"] == "PASS", result["forward_backward"] == "PASS", result["adapter_updated"] == "PASS", result["base_model_unchanged"] == "PASS", result["loss_finite"]]
    result["decision"] = "PREFLIGHT_PASS" if all(required) else "PREFLIGHT_ENVIRONMENT_FAILURE"
    result["status"] = "COMPLETE"
except Exception as exc:
    result["status"] = "ERROR"
    result["decision"] = "PREFLIGHT_ENVIRONMENT_FAILURE"
    result["error"] = f"{type(exc).__name__}: {exc}"
    result["traceback"] = traceback.format_exc()
    if torch.cuda.is_available():
        result["failure_memory"] = cuda_snapshot()
    atomic_json(OUT / "SINGLE_TASK_TTT_PREFLIGHT.json", result)
    print(json.dumps({"event": "PREFLIGHT_ERROR", "error": result["error"]}, sort_keys=True), flush=True)
    raise
atomic_json(OUT / "SINGLE_TASK_TTT_PREFLIGHT.json", result)
lines = [
    "# Official NVARC Single-Task Adapter Preflight", "",
    f"- Decision: `{result['decision']}`",
    f"- Adapter creation: `{result['adapter_creation']}`",
    f"- Forward/backward: `{result['forward_backward']}`",
    f"- Base unchanged: `{result['base_model_unchanged']}`",
    f"- Adapter updated: `{result['adapter_updated']}`",
    f"- Peak VRAM: `{result['peak_vram_gb']:.3f} GiB`",
    f"- Seconds/step: `{result['seconds_per_ttt_step']:.3f}`",
    f"- Projected seconds/task: `{result['projected_seconds_per_task']:.3f}`",
    f"- Projected Eval6 GPU seconds: `{result['projected_eval6_gpu_seconds']:.3f}`",
    f"- Projected 240-task GPU seconds: `{result['projected_240_task_gpu_seconds']:.3f}`",
    "", "No evaluation solutions, test prediction, candidate generation, or DFS were used.",
]
(OUT / "SINGLE_TASK_TTT_PREFLIGHT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(json.dumps({"event": "PREFLIGHT_COMPLETE", "decision": result["decision"], "report_dir": str(OUT)}, sort_keys=True), flush=True)
'''


def build(output: Path, owner: str, kernel_slug: str) -> tuple[Path, Path]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    kernel = output / "kernel"
    kernel.mkdir(parents=True)
    notebook_name = f"{kernel_slug}.ipynb"
    notebook = {
        "cells": [{
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [line + "\n" for line in _code().splitlines()],
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"},
        },
        "nbformat": 4, "nbformat_minor": 4,
    }
    _write_json(kernel / notebook_name, notebook)
    metadata = {
        "id": f"{owner}/{kernel_slug}", "title": "ARC2 Official NVARC Single Task Adapter Preflight",
        "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu", "diagnostic"],
        "dataset_sources": [], "kernel_sources": [REFERENCE_SOURCE],
        "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": [MODEL_SOURCE],
        "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868",
        "machine_shape": "NvidiaL4",
    }
    _write_json(kernel / "kernel-metadata.json", metadata)
    return kernel, kernel / notebook_name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--kernel-slug", default="arc2-official-nvarc-adapter-preflight")
    args = parser.parse_args()
    kernel, notebook = build(args.output, args.owner, args.kernel_slug)
    print(json.dumps({
        "event": "OFFICIAL_NVARC_PREFLIGHT_STAGED", "kernel_dir": str(kernel),
        "notebook": str(notebook), "task_id": TASK_ID, "kernel_source": REFERENCE_SOURCE,
        "no_solutions": True, "no_dfs": True,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
