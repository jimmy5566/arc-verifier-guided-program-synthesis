"""Temporary train-pair-only LoRA adaptation for the native NVARC model.

This is deliberately a transport-level adaptation utility: it consumes only a
task's supplied training pairs, has no access to a test target, and is reset
before every task.  It contains no ARC rule or output-selection logic.
"""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

from arc.task import ARCTask
from inference.nvarc_native import native_messages, serialize_grid
from inference.nvarc_native_augmentation import NativeAugmentation


@dataclass(frozen=True)
class NativeLoRAConfig:
    rank: int = 8
    alpha: int = 16
    steps: int = 24
    learning_rate: float = 5e-4
    target_suffixes: tuple[str, ...] = ("q_proj", "v_proj")
    seed: int = 20260913

    def __post_init__(self) -> None:
        if self.rank <= 0 or self.alpha <= 0 or self.steps <= 0 or self.learning_rate <= 0:
            raise ValueError("LoRA rank, alpha, steps, and learning_rate must be positive")


class _LoRALinearFactory:
    """Lazily defines the torch module so this file remains importable offline."""

    @staticmethod
    def make(base: Any, rank: int, alpha: int) -> Any:
        import math
        import torch
        from torch import nn
        import torch.nn.functional as functional

        class LoRALinear(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.base = base
                for parameter in self.base.parameters(): parameter.requires_grad_(False)
                self.rank, self.scale = rank, alpha / rank
                # Match the checkpoint's BF16 execution dtype.  FP32 adapter
                # weights force an extra cast in every targeted projection and
                # can push a 22GB L4 over its ranking-time peak.
                self.down = nn.Parameter(torch.empty((rank, base.in_features), device=base.weight.device, dtype=base.weight.dtype))
                self.up = nn.Parameter(torch.zeros((base.out_features, rank), device=base.weight.device, dtype=base.weight.dtype))
                self.reset_adapter()

            def reset_adapter(self) -> None:
                nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
                nn.init.zeros_(self.up)

            def forward(self, value: Any) -> Any:
                delta = functional.linear(functional.linear(value, self.down), self.up)
                return self.base(value) + delta * self.scale

        return LoRALinear()


class NativeTaskLoRA:
    """Installs a small resettable LoRA layer set into one already-loaded model."""

    def __init__(self, model: Any, config: NativeLoRAConfig) -> None:
        self.model, self.config = model, config
        # The native checkpoint is inference-only during TTT.  Freezing the
        # entire base *before* installing adapters is essential: freezing only
        # q/v leaves gradients and activation state alive for the rest of the
        # transformer and exhausts an L4 on longer ARC prompts.
        for parameter in model.parameters(): parameter.requires_grad_(False)
        self.layers: list[Any] = []
        for name, module in list(model.named_modules()):
            if not name or not name.endswith(config.target_suffixes): continue
            if not hasattr(module, "in_features") or not hasattr(module, "out_features") or not hasattr(module, "weight"): continue
            parent, attribute = self._parent(name)
            replacement = _LoRALinearFactory.make(module, config.rank, config.alpha)
            setattr(parent, attribute, replacement); self.layers.append(replacement)
        if not self.layers: raise RuntimeError(f"no Linear modules matched native LoRA targets {config.target_suffixes}")
        self.parameters = [parameter for layer in self.layers for parameter in (layer.down, layer.up)]
        for parameter in self.parameters: parameter.requires_grad_(True)
        self._assert_trainable_boundary()

    def _parent(self, name: str) -> tuple[Any, str]:
        parts = name.split("."); parent = self.model
        for part in parts[:-1]:
            # Transformer blocks commonly live in ModuleList entries such as
            # ``model.layers.0``.  Resolve both named attributes and indexed
            # registered submodules without assuming a specific architecture.
            parent = parent._modules[part] if part in getattr(parent, "_modules", {}) else getattr(parent, part)
        return parent, parts[-1]

    def _assert_trainable_boundary(self) -> None:
        adapter_ids = {id(parameter) for parameter in self.parameters}
        unexpected = [name for name, parameter in self.model.named_parameters() if parameter.requires_grad and id(parameter) not in adapter_ids]
        if unexpected: raise RuntimeError(f"only LoRA parameters may require gradients: {unexpected[:3]}")
        if any(not parameter.requires_grad for parameter in self.parameters):
            raise RuntimeError("all LoRA parameters must require gradients")

    def _base_fingerprint(self) -> tuple[int, float]:
        """Low-memory device-side checksum of every frozen base parameter."""
        import torch
        adapter_ids = {id(parameter) for parameter in self.parameters}
        values = [parameter for parameter in self.model.parameters() if id(parameter) not in adapter_ids]
        with torch.no_grad():
            # Reduction directly to float64 avoids materializing a full FP32
            # copy of the 4B checkpoint just to audit immutability.
            total = sum(float(torch.sum(parameter, dtype=torch.float64).item()) for parameter in values)
        return len(values), total

    @staticmethod
    def _memory_snapshot(torch: Any) -> dict[str, int]:
        return {
            "allocated_bytes": int(torch.cuda.memory_allocated()),
            "reserved_bytes": int(torch.cuda.memory_reserved()),
            "max_allocated_bytes": int(torch.cuda.max_memory_allocated()),
            "max_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        }

    def reset(self, *, seed: int | None = None) -> None:
        if seed is not None:
            import torch
            torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        for layer in self.layers: layer.reset_adapter()

    def finish_task(self) -> None:
        """Remove learned task state before this worker receives another task."""
        import torch
        self.reset(seed=self.config.seed)
        # Optimizer references are local to ``fit_task``; clear caching left by
        # the adaptation/generation cycle so a later task cannot inherit its
        # VRAM peak or its learned adapter weights.
        torch.cuda.empty_cache()

    def fit_task(self, provider: Any, task: ARCTask, *, augmentations: tuple[NativeAugmentation, ...], context_window: int, steps: int | None = None) -> dict[str, Any]:
        """Optimize adapters on transformed training input→output pairs only."""
        import torch
        import torch.nn.functional as functional

        if not augmentations: raise ValueError("TTT requires at least one train-pair augmentation")
        effective_steps = self.config.steps if steps is None else steps
        if effective_steps <= 0: raise ValueError("native TTT steps must be positive")
        self.reset(seed=self.config.seed); self._assert_trainable_boundary()
        model = provider.model; original_training = bool(model.training)
        original_use_cache = getattr(model.config, "use_cache", None)
        checkpointing_was_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
        deterministic_was_enabled = torch.are_deterministic_algorithms_enabled()
        deterministic_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
        original_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
        original_cudnn_tf32 = torch.backends.cudnn.allow_tf32
        original_cudnn_benchmark = torch.backends.cudnn.benchmark
        original_cudnn_deterministic = torch.backends.cudnn.deterministic
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        if hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        if original_use_cache is not None: model.config.use_cache = False
        checkpointing_enabled = bool(getattr(model, "is_gradient_checkpointing", False))
        torch.cuda.reset_peak_memory_stats(); optimizer = torch.optim.AdamW(self.parameters, lr=self.config.learning_rate, foreach=False, fused=False)
        initial = [parameter.detach().clone() for parameter in self.parameters]
        base_before = self._base_fingerprint()
        losses: list[float] = []; memory_trace: list[dict[str, Any]] = []; started = perf_counter()
        try:
            model.train()
            for step in range(effective_steps):
                augmentation = augmentations[step % len(augmentations)]
                augmented = augmentation.transform_task(task)
                example = augmented.train[(step // len(augmentations)) % len(augmented.train)]
                messages = [{"role": "user", "content": serialize_grid(example.input.to_list())}]
                prefix = provider.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
                target = provider.tokenizer(serialize_grid(example.output.to_list()), add_special_tokens=False, return_tensors="pt")["input_ids"]
                eos = torch.tensor([[int(provider.tokenizer.eos_token_id)]], dtype=target.dtype)
                ids = torch.cat((prefix, target, eos), dim=1).to(provider.device)
                if int(ids.shape[-1]) > context_window: raise ValueError("native TTT train prompt exceeds context window")
                # Checkpointed train-mode execution may consume RNG in both
                # its forward and recomputation paths.  Pin each step's seed
                # so a reset task produces the same adapter update sequence.
                step_seed = self.config.seed + step
                torch.manual_seed(step_seed); torch.cuda.manual_seed_all(step_seed)
                trace = {"step": step + 1, "seed": step_seed, "sequence_tokens": int(ids.shape[-1]), "before_forward": self._memory_snapshot(torch)}
                optimizer.zero_grad(set_to_none=True)
                logits = model(input_ids=ids, use_cache=False).logits[:, int(prefix.shape[-1]) - 1:-1, :]
                labels = ids[:, int(prefix.shape[-1]):]
                loss = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
                trace["after_forward"] = self._memory_snapshot(torch)
                loss.backward(); trace["after_backward"] = self._memory_snapshot(torch)
                optimizer.step(); trace["after_optimizer"] = self._memory_snapshot(torch)
                value = float(loss.detach().item())
                if not torch.isfinite(loss): raise FloatingPointError("non-finite native TTT loss")
                losses.append(value); memory_trace.append(trace)
                del logits, labels, loss, ids, eos, target, prefix
            updated_l1 = float(sum((parameter.detach() - start).abs().float().sum().item() for parameter, start in zip(self.parameters, initial, strict=True)))
            base_after = self._base_fingerprint()
            return {"steps": effective_steps, "rank": self.config.rank, "alpha": self.config.alpha, "target_layer_count": len(self.layers), "adapter_parameter_count": sum(parameter.numel() for parameter in self.parameters), "adapter_dtype": str(self.parameters[0].dtype), "base_trainable_parameter_count": 0, "lora_trainable_parameter_count": sum(parameter.numel() for parameter in self.parameters), "base_fingerprint_before": base_before, "base_fingerprint_after": base_after, "base_model_unchanged": base_before == base_after, "loss_finite": all(torch.isfinite(torch.tensor(value)) for value in losses), "adapter_updated": updated_l1 > 0.0, "adapter_update_l1": updated_l1, "memory_trace": memory_trace, "first_loss": losses[0], "last_loss": losses[-1], "seconds": perf_counter() - started, "training_pairs_only": True, "use_cache_during_ttt": False, "gradient_checkpointing_during_ttt": checkpointing_enabled, "deterministic_algorithms_during_ttt": True}
        finally:
            optimizer.zero_grad(set_to_none=True)
            for parameter in self.parameters: parameter.grad = None
            del initial, optimizer
            if original_use_cache is not None: model.config.use_cache = original_use_cache
            if not checkpointing_was_enabled and hasattr(model, "gradient_checkpointing_disable"):
                model.gradient_checkpointing_disable()
            if hasattr(model, "disable_input_require_grads"):
                model.disable_input_require_grads()
            model.train(original_training)
            torch.use_deterministic_algorithms(deterministic_was_enabled, warn_only=deterministic_warn_only)
            torch.backends.cuda.matmul.allow_tf32 = original_matmul_tf32
            torch.backends.cudnn.allow_tf32 = original_cudnn_tf32
            torch.backends.cudnn.benchmark = original_cudnn_benchmark
            torch.backends.cudnn.deterministic = original_cudnn_deterministic
            torch.cuda.empty_cache()
