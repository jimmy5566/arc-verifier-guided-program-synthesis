"""Build the isolated SOAR-7B BF16 one-request Kaggle preflight notebook."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = '''import gc, json, subprocess
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

input_root = Path("/kaggle/input")
candidates = []
for config in input_root.rglob("config.json"):
    marker = str(config.parent).lower()
    has_weights = any(config.parent.rglob("*.safetensors")) or any(config.parent.rglob("pytorch_model*.bin"))
    if "soar" in marker and "7b" in marker and has_weights:
        candidates.append(config.parent)
if len(candidates) != 1:
    inventory = [str(path.relative_to(input_root)) for path in candidates]
    raise RuntimeError(f"Expected one attached SOAR-7B BF16 checkpoint; found={inventory}")
model_path = candidates[0]
print({"event": "SOAR7_RESOLVED", "model_path": str(model_path), "gpu_count": torch.cuda.device_count(), "gpus": subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()})
tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True)
prompt = "You are an ARC Python-program induction expert. Return only valid Python code in one fenced python block. Define exactly def transform(grid): that returns the input grid unchanged."
if getattr(tokenizer, "chat_template", None):
    encoded = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
else:
    encoded = tokenizer(prompt, return_tensors="pt")
device = next(model.parameters()).device
encoded = {name: value.to(device) for name, value in encoded.items()}
with torch.inference_mode():
    generated = model.generate(**encoded, max_new_tokens=128, do_sample=False, pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id, eos_token_id=tokenizer.eos_token_id)
text = tokenizer.decode(generated[0, encoded["input_ids"].shape[-1]:].detach().cpu(), skip_special_tokens=True)
memory = {line.split(",")[0].strip(): int(line.split(",")[1].strip()) for line in subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True).splitlines()}
result = {"status": "SOAR7_BF16_GENERATION_COMPLETE", "model_path": str(model_path), "gpu_count": torch.cuda.device_count(), "model_load_success": True, "dtype": str(next(model.parameters()).dtype), "device_map": getattr(model, "hf_device_map", None), "generated_output": text, "gpu_memory_mb": memory}
Path("/kaggle/working/SOAR7_BF16_PREFLIGHT.json").write_text(json.dumps(result, indent=2) + "\\n", encoding="utf-8")
print(json.dumps(result))
del generated, encoded, model, tokenizer
gc.collect()
torch.cuda.empty_cache()
'''
    notebook = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": source.splitlines(keepends=True)}],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}},
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(notebook), encoding="utf-8")


if __name__ == "__main__":
    main()
