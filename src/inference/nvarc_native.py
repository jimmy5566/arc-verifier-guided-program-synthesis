"""Verified NVARC grid serialization and checkpoint-native tokenizer transport.

The representation is frozen from 1ytic/NVARC commit 846d0198: grids are
newline-delimited digit rows and messages alternate user input/assistant
output.  It has no JSON, ARC prose, semantic IR, DSL, or target dependency.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from inference.direct_grid_solver import Grid, validate_grid


def serialize_grid(grid: Any) -> str:
    value = validate_grid(grid)
    if value is None or len(value) > 30 or len(value[0]) > 30:
        raise ValueError("NVARC grid must be rectangular 1..30 with ARC colors")
    return "\n".join("".join(str(cell) for cell in row) for row in value)


def parse_native_grid(text: str) -> Grid | None:
    """Accept only the checkpoint's newline-separated digit-grid output."""
    if not isinstance(text, str):
        return None
    lines = text.strip().splitlines()
    if not lines or len(lines) > 30 or any(not row or len(row) > 30 or any(char not in "0123456789" for char in row) for row in lines):
        return None
    if len({len(row) for row in lines}) != 1:
        return None
    return validate_grid([[int(char) for char in row] for row in lines])


def native_messages(task: Any, test_index: int) -> list[dict[str, str]]:
    if not 0 <= test_index < len(task.test):
        raise IndexError("test index out of range")
    messages: list[dict[str, str]] = []
    for example in task.train:
        if example.output is None:
            raise ValueError("train output is required")
        messages.extend(({"role": "user", "content": serialize_grid(example.input.to_list())}, {"role": "assistant", "content": serialize_grid(example.output.to_list())}))
    messages.append({"role": "user", "content": serialize_grid(task.test[test_index].input.to_list())})
    return messages


_EXPECTED_TOKENS = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7, "8": 8,
    "9": 9, "Ċ": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13,
    "<|im_start|>": 14, "<|im_end|>": 15,
}


def checkpoint_native_tokenizer(model_path: Path, native_config_dir: Path) -> tuple[Any, dict[str, Any]]:
    """Load the tokenizer packaged with the SFT checkpoint, without mutation.

    NVARC's official template is used only if the checkpoint does not package
    a chat template.  The function never adds tokens, resizes embeddings, or
    substitutes a general-Qwen tokenizer.
    """
    from transformers import AutoConfig, AutoTokenizer

    model_path, native_config_dir = Path(model_path), Path(native_config_dir)
    config = AutoConfig.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True, trust_remote_code=False, use_fast=True)
    token_mapping = {token: int(tokenizer.convert_tokens_to_ids(token)) for token in _EXPECTED_TOKENS}
    if int(getattr(config, "vocab_size", -1)) != 16 or len(tokenizer) != 16 or token_mapping != _EXPECTED_TOKENS:
        raise RuntimeError(
            "checkpoint is not the expected NVARC tiny vocabulary: "
            f"config_vocab={getattr(config, 'vocab_size', None)}, tokenizer_vocab={len(tokenizer)}, mapping={token_mapping}"
        )
    if tokenizer.eos_token_id != 15 or tokenizer.pad_token_id != 13:
        raise RuntimeError(f"checkpoint special token mismatch: eos={tokenizer.eos_token_id}, pad={tokenizer.pad_token_id}")
    template_source = "checkpoint"
    if not getattr(tokenizer, "chat_template", None):
        # This is an interface string only; it does not alter the checkpoint
        # vocabulary or embeddings. It is the official NVARC formatting rule.
        tokenizer.chat_template = (native_config_dir / "chat_template.j2").read_text(encoding="utf-8")
        template_source = "nvarc_official_for_missing_checkpoint_template"
    return tokenizer, {
        "config_vocab_size": int(config.vocab_size),
        "tokenizer_vocab_size": len(tokenizer),
        "token_mapping": token_mapping,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template_source": template_source,
    }


def native_tokenizer_preflight(model_path: Path, native_config_dir: Path) -> dict[str, Any]:
    """Verify the checkpoint tokenizer and a non-ARC native-format render."""
    tokenizer, checkpoint = checkpoint_native_tokenizer(model_path, native_config_dir)
    synthetic = [{"role": "user", "content": "01\n23"}, {"role": "assistant", "content": "32\n10"}, {"role": "user", "content": "45\n67"}]
    rendered = tokenizer.apply_chat_template(synthetic, add_generation_prompt=True, tokenize=False)
    expected = "<|im_start|>user\n01\n23<|im_end|><|im_start|>assistant\n32\n10<|im_end|><|im_start|>user\n45\n67<|im_end|><|im_start|>assistant\n"
    if rendered != expected:
        raise RuntimeError("NVARC chat template render mismatch")
    encoded = tokenizer.apply_chat_template(synthetic, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
    return {**checkpoint, "synthetic_render": rendered, "synthetic_prompt_tokens": int(encoded["input_ids"].shape[-1]), "grid_roundtrip": parse_native_grid(serialize_grid([[0, 1], [2, 3]])) == [[0, 1], [2, 3]]}


@dataclass(frozen=True)
class NativeGeneration:
    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float


class NVARCNativeProvider:
    """Local BF16 generation with the verified 16-token NVARC tokenizer."""

    def __init__(self, *, model_path: Path, tokenizer_config_dir: Path, device: str = "cuda:0") -> None:
        self.model_path, self.tokenizer_config_dir, self.device = Path(model_path), Path(tokenizer_config_dir), device
        self.model: Any | None = None
        self.tokenizer: Any | None = None
        self.load_metadata: dict[str, Any] = {}

    def load(self) -> float:
        if self.model is not None:
            return 0.0
        import torch
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("NVARC native provider requires CUDA")
        torch.cuda.reset_peak_memory_stats(self.device)
        started = time.perf_counter()
        self.tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(self.model_path, self.tokenizer_config_dir)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to(self.device).eval()
        vocab_size = int(getattr(self.model.config, "vocab_size", -1))
        if vocab_size != len(self.tokenizer):
            raise RuntimeError(f"NVARC tokenizer/model vocabulary mismatch: model={vocab_size}, tokenizer={len(self.tokenizer)}")
        elapsed = time.perf_counter() - started
        self.load_metadata = {
            **tokenizer_metadata,
            "model_load_seconds": elapsed,
            "gpu_name": torch.cuda.get_device_name(self.device),
            "peak_vram_mb": round(torch.cuda.max_memory_allocated(self.device) / (1024 * 1024), 1),
        }
        return elapsed

    def generate(self, messages: list[dict[str, str]], *, max_new_tokens: int, context_window: int, seed: int) -> NativeGeneration:
        self.load(); import torch
        assert self.model is not None and self.tokenizer is not None
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}; prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > context_window:
            raise ValueError(f"native prompt has {prompt_tokens} tokens, exceeds frozen context {context_window}")
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=False, eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id)
        generated = output[0, prompt_tokens:]
        return NativeGeneration(self.tokenizer.decode(generated, skip_special_tokens=True), prompt_tokens, int(generated.shape[-1]), time.perf_counter() - started)
