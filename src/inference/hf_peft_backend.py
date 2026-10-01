"""Clean HuggingFace + PEFT inference boundary for ARC2 Regret DFS.

This module deliberately contains no training-backend imports. It consumes a
frozen PEFT adapter directory emitted by task adaptation and exposes only an
eval-mode causal LM plus an auditable runtime identity.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_hf_peft_inference(*, model_path: Path, adapter_path: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load a frozen adapter with stock Transformers and PEFT only."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.utils import logging as transformers_logging

    model_path, adapter_path = Path(model_path), Path(adapter_path)
    required = (adapter_path / "adapter_model.safetensors", adapter_path / "adapter_config.json")
    if not model_path.is_dir() or any(not item.is_file() for item in required):
        raise FileNotFoundError(f"missing clean-HF source or adapter files: model={model_path}, adapter={adapter_path}")
    transformers_logging.disable_progress_bar(); transformers_logging.set_verbosity_error()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False,
                                                 torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(device).eval()
    model = PeftModel.from_pretrained(base, str(adapter_path), is_trainable=False).to(device).eval()
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    identity = {
        "backend": "transformers_peft_clean", "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "peft": importlib.metadata.version("peft"),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device)),
        "model_class": type(model).__qualname__, "base_model_class": type(base).__qualname__,
        "forward_module": inspect.getmodule(model.forward).__name__,
        "attention_implementation": getattr(base.config, "_attn_implementation", None),
        "dtype": str(next(model.parameters()).dtype), "adapter_sha256": _sha256_file(adapter_path / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256_file(adapter_path / "adapter_config.json"), "tokenizer_vocab_size": len(tokenizer),
        "adapter_config": {key: adapter_config.get(key) for key in ("r", "lora_alpha", "use_rslora", "target_modules", "modules_to_save")},
    }
    return model, tokenizer, identity


def adapter_tensor_manifest(adapter_path: Path) -> list[dict[str, Any]]:
    """Return source PEFT tensor identities without materializing all weights."""
    from safetensors import safe_open
    path = Path(adapter_path) / "adapter_model.safetensors"
    rows: list[dict[str, Any]] = []
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        for name in sorted(handle.keys()):
            tensor = handle.get_tensor(name).contiguous()
            rows.append({"canonical_parameter_name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype),
                         "source_sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest()})
    return rows


def loaded_adapter_tensor_parity(model: Any, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare source safetensors against adapter tensors installed by PEFT."""
    state = model.state_dict(); rows: list[dict[str, Any]] = []
    for source in source_rows:
        name = str(source["canonical_parameter_name"])
        candidates = (name, f"base_model.model.{name.removeprefix('base_model.model.')}")
        loaded_name = next((candidate for candidate in candidates if candidate in state), None)
        tensor = state.get(loaded_name) if loaded_name else None
        loaded_sha = None if tensor is None else hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()
        rows.append({**source, "loaded_parameter_name": loaded_name, "loaded_sha256": loaded_sha,
                     "exact_equal": bool(loaded_sha == source["source_sha256"])})
    return rows
