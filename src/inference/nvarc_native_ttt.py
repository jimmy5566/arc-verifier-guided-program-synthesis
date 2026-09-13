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
                self.down = nn.Parameter(torch.empty((rank, base.in_features), device=base.weight.device, dtype=torch.float32))
                self.up = nn.Parameter(torch.zeros((base.out_features, rank), device=base.weight.device, dtype=torch.float32))
                self.reset_adapter()

            def reset_adapter(self) -> None:
                nn.init.kaiming_uniform_(self.down, a=math.sqrt(5))
                nn.init.zeros_(self.up)

            def forward(self, value: Any) -> Any:
                delta = functional.linear(functional.linear(value, self.down.to(dtype=value.dtype)), self.up.to(dtype=value.dtype))
                return self.base(value) + delta * self.scale

        return LoRALinear()


class NativeTaskLoRA:
    """Installs a small resettable LoRA layer set into one already-loaded model."""

    def __init__(self, model: Any, config: NativeLoRAConfig) -> None:
        self.model, self.config = model, config
        self.layers: list[Any] = []
        for name, module in list(model.named_modules()):
            if not name or not name.endswith(config.target_suffixes): continue
            if not hasattr(module, "in_features") or not hasattr(module, "out_features") or not hasattr(module, "weight"): continue
            parent, attribute = self._parent(name)
            replacement = _LoRALinearFactory.make(module, config.rank, config.alpha)
            setattr(parent, attribute, replacement); self.layers.append(replacement)
        if not self.layers: raise RuntimeError(f"no Linear modules matched native LoRA targets {config.target_suffixes}")
        self.parameters = [parameter for layer in self.layers for parameter in (layer.down, layer.up)]

    def _parent(self, name: str) -> tuple[Any, str]:
        parts = name.split("."); parent = self.model
        for part in parts[:-1]: parent = getattr(parent, part)
        return parent, parts[-1]

    def reset(self) -> None:
        for layer in self.layers: layer.reset_adapter()

    def fit_task(self, provider: Any, task: ARCTask, *, augmentations: tuple[NativeAugmentation, ...], context_window: int) -> dict[str, Any]:
        """Optimize adapters on transformed training input→output pairs only."""
        import torch
        import torch.nn.functional as functional

        if not augmentations: raise ValueError("TTT requires at least one train-pair augmentation")
        self.reset(); optimizer = torch.optim.AdamW(self.parameters, lr=self.config.learning_rate)
        losses: list[float] = []; started = perf_counter()
        provider.model.train()
        for step in range(self.config.steps):
            augmentation = augmentations[step % len(augmentations)]
            augmented = augmentation.transform_task(task)
            example = augmented.train[(step // len(augmentations)) % len(augmented.train)]
            messages = [{"role": "user", "content": serialize_grid(example.input.to_list())}]
            prefix = provider.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
            target = provider.tokenizer(serialize_grid(example.output.to_list()), add_special_tokens=False, return_tensors="pt")["input_ids"]
            eos = torch.tensor([[int(provider.tokenizer.eos_token_id)]], dtype=target.dtype)
            ids = torch.cat((prefix, target, eos), dim=1).to(provider.device)
            if int(ids.shape[-1]) > context_window: raise ValueError("native TTT train prompt exceeds context window")
            optimizer.zero_grad(set_to_none=True)
            logits = provider.model(input_ids=ids).logits[:, int(prefix.shape[-1]) - 1:-1, :]
            labels = ids[:, int(prefix.shape[-1]):]
            loss = functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))
            loss.backward(); optimizer.step(); losses.append(float(loss.detach().item()))
        provider.model.eval()
        return {"steps": self.config.steps, "rank": self.config.rank, "alpha": self.config.alpha, "target_layer_count": len(self.layers), "adapter_parameter_count": sum(parameter.numel() for parameter in self.parameters), "first_loss": losses[0], "last_loss": losses[-1], "seconds": perf_counter() - started, "training_pairs_only": True}
