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

from inference.arc_native_io import ARCNativeInputAdapter, ARCNativeOutputParser, ARCGridValue

Grid = ARCGridValue


def serialize_grid(grid: Any) -> str:
    return ARCNativeInputAdapter.serialize_grid(grid)


def parse_native_grid(text: str) -> Grid | None:
    return ARCNativeOutputParser.parse(text)


def native_messages(task: Any, test_index: int) -> list[dict[str, str]]:
    return ARCNativeInputAdapter().messages(task, test_index)


def native_training_message_prefix(task: Any) -> tuple[tuple[str, str], ...]:
    return ARCNativeInputAdapter().training_prefix(task)


def native_messages_from_training_prefix(prefix: tuple[tuple[str, str], ...], test_input: Any) -> list[dict[str, str]]:
    return ARCNativeInputAdapter().messages_from_training_prefix(prefix, test_input)


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


@dataclass(frozen=True)
class NativeBeamGeneration:
    """One completed sequence from deterministic bounded beam search."""

    text: str
    prompt_tokens: int
    completion_tokens: int
    elapsed_seconds: float
    sequence_score: float | None


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
        from transformers.utils import logging as transformers_logging

        if not torch.cuda.is_available():
            raise RuntimeError("NVARC native provider requires CUDA")
        # ``from_pretrained`` uses a tqdm-style per-parameter materialization
        # display.  Kaggle renders every carriage-return update as a separate
        # log line (and sometimes duplicates it), producing tens of thousands
        # of misleading ``Loading weights: n/399`` lines.  Load telemetry is
        # emitted by the worker as one structured MODEL_READY event instead.
        transformers_logging.disable_progress_bar()
        transformers_logging.set_verbosity_error()
        started = time.perf_counter()
        self.tokenizer, tokenizer_metadata = checkpoint_native_tokenizer(self.model_path, self.tokenizer_config_dir)
        self.model = AutoModelForCausalLM.from_pretrained(str(self.model_path), local_files_only=True, trust_remote_code=False, dtype=torch.bfloat16, low_cpu_mem_usage=True).to(self.device).eval()
        vocab_size = int(getattr(self.model.config, "vocab_size", -1))
        if vocab_size != len(self.tokenizer):
            raise RuntimeError(f"NVARC tokenizer/model vocabulary mismatch: model={vocab_size}, tokenizer={len(self.tokenizer)}")
        elapsed = time.perf_counter() - started
        self.load_metadata = {
            **tokenizer_metadata,
            "model_load_seconds": elapsed,
            "gpu_name": torch.cuda.get_device_name(),
            "model_vram_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 1),
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
        # Move the tiny native-token result off device before the next
        # augmentation.  This keeps each long-lived worker's CUDA footprint
        # bounded rather than retaining a generation tensor through decoding.
        generated = output[0, prompt_tokens:].detach().cpu()
        text = self.tokenizer.decode(generated, skip_special_tokens=True)
        result = NativeGeneration(text, prompt_tokens, int(generated.shape[-1]), time.perf_counter() - started)
        del output, encoded, generated
        return result

    def generate_many(
        self,
        messages_batch: list[list[dict[str, str]]],
        *,
        max_new_tokens: int,
        context_window: int,
        seeds: list[int],
    ) -> list[NativeGeneration]:
        """Greedily generate a padded batch of native prompts.

        The per-request seeds remain part of the frozen task-local provenance,
        although deterministic greedy decoding does not consume randomness.
        A size-one batch deliberately uses the legacy path, giving the speed
        benchmark an exact serial reference condition.
        """
        if not messages_batch or len(messages_batch) != len(seeds):
            raise ValueError("native generation batch/messages seeds must align and be non-empty")
        if len(messages_batch) == 1:
            return [self.generate(messages_batch[0], max_new_tokens=max_new_tokens, context_window=context_window, seed=seeds[0])]
        self.load(); import torch
        assert self.model is not None and self.tokenizer is not None
        rows = [
            self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"][0]
            for messages in messages_batch
        ]
        prompt_lengths = [int(row.shape[-1]) for row in rows]
        if max(prompt_lengths) > context_window:
            raise ValueError(f"native prompt has {max(prompt_lengths)} tokens, exceeds frozen context {context_window}")
        width = max(prompt_lengths)
        input_ids = torch.full((len(rows), width), int(self.tokenizer.pad_token_id), dtype=rows[0].dtype)
        attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            input_ids[index, width - int(row.shape[-1]):] = row
            attention_mask[index, width - int(row.shape[-1]):] = 1
        # Greedy decoding is seed-independent, but retain deterministic seed
        # initialization for the same provider contract as the serial path.
        torch.manual_seed(int(seeds[0])); torch.cuda.manual_seed_all(int(seeds[0]))
        started = time.perf_counter()
        with torch.inference_mode():
            output = self.model.generate(
                input_ids=input_ids.to(self.device), attention_mask=attention_mask.to(self.device),
                max_new_tokens=max_new_tokens, do_sample=False,
                eos_token_id=self.tokenizer.eos_token_id, pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        result: list[NativeGeneration] = []
        for index in range(len(rows)):
            suffix = output[index, width:].detach().cpu()
            eos_positions = (suffix == int(self.tokenizer.eos_token_id)).nonzero(as_tuple=False)
            if len(eos_positions):
                suffix = suffix[:int(eos_positions[0].item()) + 1]
            else:
                pad_positions = (suffix == int(self.tokenizer.pad_token_id)).nonzero(as_tuple=False)
                if len(pad_positions):
                    suffix = suffix[:int(pad_positions[0].item())]
            result.append(NativeGeneration(
                self.tokenizer.decode(suffix, skip_special_tokens=True), prompt_lengths[index],
                int(suffix.shape[-1]), elapsed / len(rows),
            ))
        del output, input_ids, attention_mask, rows
        return result

    def generate_beams(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int,
        context_window: int,
        beam_width: int,
    ) -> list[NativeBeamGeneration]:
        """Return a fixed, bounded set of native-token beam completions.

        This is inference-time search only.  It uses the checkpoint's tiny
        native vocabulary, performs no adaptation, and exposes no ARC target.
        ``beam_width`` bounds every generated branch before any parse/dedup.
        """
        if not 1 <= beam_width <= 8:
            raise ValueError("native beam width must be in 1..8")
        self.load(); import torch
        assert self.model is not None and self.tokenizer is not None
        encoded = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)
        encoded = {name: value.to(self.device) for name, value in encoded.items()}
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        if prompt_tokens > context_window:
            raise ValueError(f"native prompt has {prompt_tokens} tokens, exceeds frozen context {context_window}")
        started = time.perf_counter()
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=beam_width,
                num_return_sequences=beam_width,
                early_stopping=True,
                return_dict_in_generate=True,
                output_scores=True,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        elapsed = time.perf_counter() - started
        sequence_scores = getattr(generated, "sequences_scores", None)
        result: list[NativeBeamGeneration] = []
        for index, sequence in enumerate(generated.sequences):
            suffix = sequence[prompt_tokens:].detach().cpu()
            result.append(NativeBeamGeneration(
                text=self.tokenizer.decode(suffix, skip_special_tokens=True),
                prompt_tokens=prompt_tokens,
                completion_tokens=int(suffix.shape[-1]),
                elapsed_seconds=elapsed / beam_width,
                sequence_score=float(sequence_scores[index].item()) if sequence_scores is not None else None,
            ))
            del suffix
        del generated, encoded
        return result

    def continuation_log_likelihood(self, messages: list[dict[str, str]], continuation: str, *, context_window: int) -> float:
        """Mean conditional log-likelihood for a generated native grid.

        This is a label-free candidate ranking signal: ``continuation`` must
        already be a model-generated candidate and is never an ARC target.
        """
        self.load(); import torch
        assert self.model is not None and self.tokenizer is not None
        prefix = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
        continuation_ids = self.tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"]
        # The native decoder normally terminates with <|im_end|>; include it
        # in the likelihood while keeping it out of parser-visible text.
        eos = torch.tensor([[int(self.tokenizer.eos_token_id)]], dtype=continuation_ids.dtype)
        ids = torch.cat((prefix, continuation_ids, eos), dim=1).to(self.device)
        if int(ids.shape[-1]) > context_window:
            raise ValueError(f"native candidate score has {int(ids.shape[-1])} tokens, exceeds frozen context {context_window}")
        with torch.inference_mode():
            logits = self.model(input_ids=ids).logits
            target = ids[:, int(prefix.shape[-1]):]
            predicted = logits[:, int(prefix.shape[-1]) - 1:-1, :]
            token_log_probs = torch.log_softmax(predicted.float(), dim=-1).gather(-1, target.unsqueeze(-1)).squeeze(-1)
        score = float(token_log_probs.mean().item())
        del logits, token_log_probs, predicted, target, ids, prefix, continuation_ids, eos
        return score

    def continuation_log_likelihood_many(
        self,
        requests: list[tuple[list[dict[str, str]], str]],
        *,
        context_window: int,
        batch_size: int,
    ) -> list[float]:
        """Score native continuations in padded forward-pass micro-batches.

        Each returned value is exactly the existing mean conditional
        log-likelihood definition.  Padding is left-aligned with an explicit
        attention mask, and the per-row prefix boundary selects only the
        continuation plus terminal ``<|im_end|>`` tokens.
        """
        if not requests:
            return []
        if batch_size < 1:
            raise ValueError("likelihood batch_size must be positive")
        if batch_size == 1:
            return [self.continuation_log_likelihood(messages, continuation, context_window=context_window) for messages, continuation in requests]
        self.load(); import torch
        assert self.model is not None and self.tokenizer is not None
        result: list[float] = []
        for start in range(0, len(requests), batch_size):
            batch = requests[start:start + batch_size]
            prefixes = [self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"][0] for messages, _continuation in batch]
            continuations = [self.tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"][0] for _messages, continuation in batch]
            eos = torch.tensor([int(self.tokenizer.eos_token_id)], dtype=prefixes[0].dtype)
            rows = [torch.cat((prefix, continuation, eos)) for prefix, continuation in zip(prefixes, continuations, strict=True)]
            if max(int(row.shape[-1]) for row in rows) > context_window:
                raise ValueError("native candidate score exceeds frozen context")
            width = max(int(row.shape[-1]) for row in rows)
            ids = torch.full((len(rows), width), int(self.tokenizer.pad_token_id), dtype=rows[0].dtype)
            attention_mask = torch.zeros((len(rows), width), dtype=torch.long)
            offsets: list[int] = []
            for index, row in enumerate(rows):
                offset = width - int(row.shape[-1]); offsets.append(offset)
                ids[index, offset:] = row; attention_mask[index, offset:] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids.to(self.device), attention_mask=attention_mask.to(self.device)).logits
                for index, (prefix, continuation) in enumerate(zip(prefixes, continuations, strict=True)):
                    target = ids[index, offsets[index] + int(prefix.shape[-1]):]
                    predicted = logits[index, offsets[index] + int(prefix.shape[-1]) - 1:offsets[index] + int(rows[index].shape[-1]) - 1, :]
                    token_log_probs = torch.log_softmax(predicted.float(), dim=-1).gather(-1, target.to(self.device).unsqueeze(-1)).squeeze(-1)
                    result.append(float(token_log_probs.mean().item()))
            del logits, ids, attention_mask, prefixes, continuations, rows, eos
        return result
