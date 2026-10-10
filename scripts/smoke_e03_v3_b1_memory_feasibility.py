#!/usr/bin/env python3
"""Authorized one-example E03 V3 B1 memory smoke; never an optimizer run."""
from __future__ import annotations

import argparse
import time
import traceback
from pathlib import Path

import run_e03_v3_b1_per_example_gradient_screen as e03


def cuda_memory(torch) -> dict[str, int]:
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated()),
        "reserved_bytes": int(torch.cuda.memory_reserved()),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = e03.resolve(str(args.output_root))
    receipt = output_root / "TERMINAL_RECEIPT.json"
    started = time.monotonic()
    runtime_head = None
    owns_output = False
    try:
        config, binding, cohort, checkpoint, runtime_head = e03.preflight(args.config, args.binding, output_root)
        if binding.get("kind") != "E03_V3_B1_MEMORY_FEASIBILITY_SMOKE":
            raise RuntimeError("E03_V3_SMOKE_BINDING_KIND_INVALID")
        if binding.get("authorized_examples") != 1 or binding.get("optimizer_steps") != 0 or binding.get("generation_calls") != 0:
            raise RuntimeError("E03_V3_SMOKE_SCOPE_INVALID")
        rows = [member for family in cohort["families"] for member in family["members"]]
        longest = max(rows, key=lambda member: (len(member["input_ids"]), member["episode_id"]))
        if binding.get("smoke_episode_id") != longest["episode_id"] or binding.get("selection_rule") != "FROZEN_LONGEST_SEQUENCE_THEN_EPISODE_ID":
            raise RuntimeError("E03_V3_SMOKE_EPISODE_BINDING_INVALID")
        e03.check_dependencies()
        output_root.mkdir(parents=True, exist_ok=False)
        owns_output = True
        import torch
        import torch.nn.functional as functional
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        if not torch.cuda.is_available():
            raise RuntimeError("E03_V3_SMOKE_CUDA_UNAVAILABLE")
        torch.manual_seed(int(config["seed"]))
        torch.cuda.manual_seed_all(int(config["seed"]))
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.cuda.get_device_properties(0)
        before_load = cuda_memory(torch)
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint["base_path"], torch_dtype=torch.bfloat16, device_map="cuda:0"
        )
        model = PeftModel.from_pretrained(model, checkpoint["adapter_path"], is_trainable=True)
        model.eval()
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        named = dict(model.named_parameters())
        allow = tuple(sorted(config["lora_parameter_name_allowlist"]))
        for name, parameter in named.items():
            parameter.requires_grad_(name in allow)
        if tuple(sorted(name for name, parameter in named.items() if parameter.requires_grad)) != allow:
            raise RuntimeError("E03_V3_SMOKE_TRAINABLE_PARAMETER_SET_MISMATCH")
        torch.cuda.synchronize()
        after_model_load = cuda_memory(torch)
        model.zero_grad(set_to_none=True)
        ids = torch.tensor([longest["input_ids"]], device="cuda", dtype=torch.long)
        labels = torch.tensor([longest["labels"]], device="cuda", dtype=torch.long)
        attention = torch.ones_like(ids)
        positions = (attention.cumsum(-1) - 1).clamp_min(0)
        loss, supervised_tokens = e03.selected_ce(
            model(input_ids=ids, attention_mask=attention, position_ids=positions, use_cache=False).logits,
            labels,
            functional,
        )
        loss.backward()
        torch.cuda.synchronize()
        after_backward = cuda_memory(torch)
        result = {
            "protocol_id": config["protocol_id"],
            "kind": binding["kind"],
            "status": "COMPLETE_NO_UPDATE",
            "runtime_launch_commit": runtime_head,
            "episode_id": longest["episode_id"],
            "sequence_tokens": len(longest["input_ids"]),
            "supervised_tokens": supervised_tokens,
            "loss": float(loss.detach().cpu()),
            "gpu_name": str(device.name),
            "gpu_total_memory_bytes": int(device.total_memory),
            "memory_before_load": before_load,
            "memory_after_model_load": after_model_load,
            "memory_after_backward": after_backward,
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_constructed": False,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "generation_calls": 0,
            "final_audit_opened": False,
        }
        result_path = output_root / "MEMORY_SMOKE_RESULT.json"
        e03.atomic(result_path, result)
        e03.atomic(receipt, {
            "protocol_id": config["protocol_id"],
            "kind": binding["kind"],
            "status": "COMPLETE_NO_UPDATE",
            "runtime_launch_commit": runtime_head,
            "result_sha256": e03.sha(result_path),
            "elapsed_seconds": time.monotonic() - started,
            "optimizer_steps": 0,
            "parameter_updates": 0,
            "generation_calls": 0,
            "final_audit_opened": False,
        })
        return 0
    except Exception as exc:
        if owns_output and not receipt.exists():
            e03.atomic(receipt, {
                "protocol_id": "E03_V3_B1_PER_EXAMPLE_GRADIENT_SCREEN",
                "kind": "E03_V3_B1_MEMORY_FEASIBILITY_SMOKE",
                "status": "FAILED_OR_PARTIAL_NO_UPDATE",
                "runtime_launch_commit": runtime_head,
                "error_class": f"{type(exc).__name__}:{exc}",
                "traceback": traceback.format_exc(),
                "elapsed_seconds": time.monotonic() - started,
                "optimizer_steps": 0,
                "parameter_updates": 0,
                "generation_calls": 0,
                "final_audit_opened": False,
            })
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
