"""Pure-Python contracts for the bounded Blackwell Phase-2 queue.

The runner keeps greedy decoding intact while allowing only explicit,
documented execution mechanisms. CUDA Graph replay remains unavailable for
this variable-length decode path, so its queue record is explicitly frozen as
NOT_APPLICABLE rather than silently substituting a different decoder.
"""
from __future__ import annotations

from typing import Any

BACKENDS = ("reference", "compaction", "static_kv", "torch_compile", "cuda_graph", "best_combined")


def execution_for_backend(name: str, *, successful: set[str]) -> tuple[str | None, str | None]:
    if name == "reference":
        return "baseline", None
    if name == "compaction":
        return "active_compaction", None
    if name == "static_kv":
        return "static_kv", None
    if name == "torch_compile":
        return "torch_compile", None
    if name == "best_combined":
        if {"static_kv", "torch_compile"}.issubset(successful):
            return "static_kv_torch_compile", None
        if "static_kv" in successful:
            return "static_kv", "torch_compile was not structurally valid"
        if "torch_compile" in successful:
            return "torch_compile", "static KV was not structurally valid"
        return "baseline", "no Phase-2 optimization was structurally valid"
    if name == "cuda_graph":
        return None, "NOT_APPLICABLE: greedy sequence completion produces dynamic active-batch shapes; explicit CUDA Graph replay cannot be made shape-safe without changing the decoder"
    raise ValueError(f"unknown backend: {name}")


def valid_frozen_artifact(payload: dict[str, Any], task_id: str) -> bool:
    return payload.get("status") == "EVAL3_RUNTIME_OPT_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING" and payload.get("task_ids") == [task_id] and set(payload.get("records", {})) == {task_id}
