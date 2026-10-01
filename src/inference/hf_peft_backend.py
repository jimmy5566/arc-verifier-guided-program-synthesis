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
import ast
from pathlib import Path
from typing import Any


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tensor_sha256(tensor: Any) -> str:
    """Hash exact tensor storage bytes, including BF16 without lossy casts."""
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(__import__("torch").uint8).numpy().tobytes()).hexdigest()


def load_hf_peft_inference(*, model_path: Path, adapter_path: Path, device: str) -> tuple[Any, Any, dict[str, Any]]:
    """Load a frozen adapter with stock Transformers and PEFT only."""
    import torch
    from peft import PeftConfig, PeftModel
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
    adapter_config = json.loads((adapter_path / "adapter_config.json").read_text(encoding="utf-8"))
    # Older adaptation exports serialize this PEFT field as a Python-set
    # string. PEFT 0.17 rightfully rejects that schema. Parse the exact list
    # deterministically; this changes neither safetensor keys nor values.
    parsed_targets = adapter_config.get("target_modules")
    if isinstance(parsed_targets, str):
        parsed_targets = ast.literal_eval(parsed_targets)
        if not isinstance(parsed_targets, (set, tuple, list)) or not all(isinstance(item, str) for item in parsed_targets):
            raise ValueError("adapter target_modules string is not a string collection")
        parsed_targets = sorted(parsed_targets)
    peft_config = PeftConfig.from_pretrained(str(adapter_path), local_files_only=True)
    if parsed_targets is not None:
        peft_config.target_modules = set(parsed_targets)
    # PEFT defaults to `autocast_adapter_dtype=True`, which silently promotes
    # FP16/BF16 LoRA tensors to FP32.  That is fine for generic inference but
    # invalid for this frozen-adapter parity boundary: the decoder must consume
    # the exact persisted BF16 adapter bytes.
    model = PeftModel.from_pretrained(
        base,
        str(adapter_path),
        config=peft_config,
        is_trainable=False,
        autocast_adapter_dtype=False,
    ).to(device).eval()
    identity = {
        "backend": "transformers_peft_clean", "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"), "peft": importlib.metadata.version("peft"),
        "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device)),
        "model_class": type(model).__qualname__, "base_model_class": type(base).__qualname__,
        "forward_module": inspect.getmodule(model.forward).__name__,
        "attention_implementation": getattr(base.config, "_attn_implementation", None),
        "dtype": str(next(model.parameters()).dtype), "adapter_sha256": _sha256_file(adapter_path / "adapter_model.safetensors"),
        "adapter_config_sha256": _sha256_file(adapter_path / "adapter_config.json"), "tokenizer_vocab_size": len(tokenizer),
        "adapter_config": {**{key: adapter_config.get(key) for key in ("r", "lora_alpha", "use_rslora", "modules_to_save")},
                           "target_modules": parsed_targets, "target_modules_schema_translated": isinstance(adapter_config.get("target_modules"), str)},
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
                         "source_sha256": _tensor_sha256(tensor)})
    return rows


def loaded_adapter_tensor_parity(model: Any, source_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare source safetensors using PEFT's authoritative adapter export."""
    from peft import get_peft_model_state_dict

    def canonical(name: str) -> str:
        # PEFT owns adapter-name insertion.  The frozen export omits the
        # default-adapter component, so remove only that structural segment.
        return (
            name.replace(".lora_A.default.", ".lora_A.")
            .replace(".lora_B.default.", ".lora_B.")
            .replace(".modules_to_save.default.", ".")
            .replace(".modules_to_save.", ".")
        )

    state = get_peft_model_state_dict(model)
    canonical_loaded: dict[str, tuple[str, Any]] = {}
    for loaded_name, tensor in state.items():
        key = canonical(str(loaded_name))
        if key in canonical_loaded:
            raise RuntimeError(f"ambiguous canonical PEFT adapter key: {key}")
        canonical_loaded[key] = (str(loaded_name), tensor)
    rows: list[dict[str, Any]] = []
    for source in source_rows:
        name = str(source["canonical_parameter_name"])
        loaded_name, tensor = canonical_loaded.get(name, (None, None))
        loaded_sha = None if tensor is None else _tensor_sha256(tensor)
        rows.append({**source, "loaded_parameter_name": loaded_name,
                     "loaded_shape": None if tensor is None else list(tensor.shape),
                     "loaded_dtype": None if tensor is None else str(tensor.dtype),
                     "loaded_sha256": loaded_sha,
                     "exact_equal": bool(loaded_sha == source["source_sha256"])})
    return rows
