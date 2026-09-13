"""Isolated two-branch ARC smoke components: native grids and SOAR programs.

No routing decision, task-specific solver, or hidden-target access lives here.
The program executor uses a separate constrained process and is intentionally
only suitable for a tiny feasibility smoke, not a security boundary for an
untrusted multi-tenant service.
"""
from __future__ import annotations

import ast
import gc
import json
import multiprocessing as mp
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from time import perf_counter
from typing import Any


def discover_models(models_root: Path, *, required: tuple[str, ...] = ("native", "soar")) -> dict[str, Path]:
    """Discover requested attached checkpoints without fixed user paths.

    A sequential smoke persists the completed Native result before loading SOAR.
    Consequently its Native invocation must not fail merely because a later SOAR
    attachment is absent; the induction invocation still requires SOAR itself.
    """
    unknown = set(required).difference({"native", "soar"})
    if unknown:
        raise ValueError(f"unknown required model roles: {sorted(unknown)}")
    choices: list[tuple[Path, dict[str, Any], str]] = []
    for config_path in models_root.rglob("config.json"):
        try: config = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError): continue
        marker = (str(config_path.parent) + " " + json.dumps(config)).lower()
        choices.append((config_path.parent, config, marker))
    # Kaggle model attachments occasionally expose a safetensors directory
    # before (or instead of) a conventional config.json at the same depth.
    # Keep discovery data-driven: the attachment's own name and weights decide.
    existing = {path for path, _config, _marker in choices}
    for weight in models_root.rglob("*.safetensors"):
        if weight.parent not in existing:
            choices.append((weight.parent, {}, str(weight.parent).lower())); existing.add(weight.parent)
    native = [item for item in choices if any(token in item[2] for token in ("grids15", "sft139", "native_arc"))]
    soar = [item for item in choices if "soar" in item[2]]
    selected = {"native": native, "soar": soar}
    invalid = {role: len(selected[role]) for role in required if len(selected[role]) != 1}
    if invalid:
        inventory = sorted(str(path.relative_to(models_root)) for path, _config, _marker in choices)[:32]
        counts = {role: len(values) for role, values in selected.items()}
        raise RuntimeError(
            "could not uniquely discover required model checkpoints: "
            f"required={required}, invalid={invalid}, counts={counts}, "
            f"scanned={len(choices)}, inventory={inventory}"
        )
    return {role: selected[role][0][0] for role in required}


def gpu_memory_mb() -> dict[str, int]:
    try:
        raw = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        return {index.strip(): int(memory.strip()) for line in raw.splitlines() if line.strip() for index, memory in [line.split(",", 1)]}
    except (OSError, subprocess.SubprocessError, ValueError): return {}


def unload_model(model: Any | None) -> dict[str, int]:
    if model is not None: del model
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache(); torch.cuda.synchronize()
    except Exception: pass
    return gpu_memory_mb()


def grid_text(grid: Any) -> str:
    return "[\n" + ",\n".join("  " + repr([int(cell) for cell in row]) for row in grid) + "\n]"


def soar_prompt(train_pairs: list[tuple[Any, Any]]) -> str:
    examples = "\n\n".join(f"Example {index + 1}\nInput grid:\n{grid_text(source)}\nOutput grid:\n{grid_text(target)}" for index, (source, target) in enumerate(train_pairs))
    return (
        "You are SOAR, an ARC program synthesizer. Infer one general transformation from all examples. "
        "Return only Python code in one fenced python block. Define exactly `def transform(grid):`. "
        "`grid` is a rectangular list[list[int]] with colors 0..9; return a new rectangular list[list[int]]. "
        "Use only Python list operations, range, len, enumerate, min, max, sum, abs, sorted, set, dict, tuple and bool. "
        "Do not import modules, access files/network, use eval/exec, or print.\n\n" + examples
    )


def extract_program(text: str) -> str | None:
    fenced = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = fenced.group(1).strip() if fenced else text[text.find("def transform("):] if "def transform(" in text else ""
    return candidate if candidate.strip() else None


_FORBIDDEN_AST = (ast.Import, ast.ImportFrom, ast.ClassDef, ast.Lambda, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete)
_SAFE_BUILTINS = {"range": range, "len": len, "enumerate": enumerate, "min": min, "max": max, "sum": sum, "abs": abs, "sorted": sorted, "set": set, "dict": dict, "tuple": tuple, "list": list, "int": int, "bool": bool, "zip": zip}


def validate_program(program: str) -> tuple[bool, str]:
    try: tree = ast.parse(program, mode="exec")
    except SyntaxError as error: return False, f"syntax:{error.msg}"
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    if len(tree.body) != 1 or len(functions) != 1 or functions[0].name != "transform" or len(functions[0].args.args) != 1:
        return False, "requires_one_transform_function"
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST): return False, f"forbidden:{type(node).__name__}"
        if isinstance(node, ast.Name) and node.id.startswith("__"): return False, "forbidden_dunder_name"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"): return False, "forbidden_dunder_attribute"
    return True, "ok"


def _program_worker(program: str, grid: Any, queue: Any) -> None:
    try:
        try:
            import resource
            resource.setrlimit(resource.RLIMIT_CPU, (2, 2)); resource.setrlimit(resource.RLIMIT_AS, (512 * 1024 * 1024, 512 * 1024 * 1024))
        except Exception: pass
        environment = {"__builtins__": _SAFE_BUILTINS}
        exec(compile(program, "<soar-program>", "exec"), environment, environment)
        result = environment["transform"]([[int(cell) for cell in row] for row in grid])
        if not isinstance(result, list) or not result or not all(isinstance(row, list) and row for row in result): raise ValueError("result_not_nonempty_grid")
        width = len(result[0])
        if len(result) > 30 or width > 30 or any(len(row) != width for row in result): raise ValueError("result_shape_invalid")
        if any(not isinstance(cell, int) or not 0 <= cell <= 9 for row in result for cell in row): raise ValueError("result_colors_invalid")
        queue.put({"ok": True, "grid": result})
    except BaseException as error: queue.put({"ok": False, "error": f"{type(error).__name__}:{error}"})


def execute_program(program: str, grid: Any, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    valid, reason = validate_program(program)
    if not valid: return {"ok": False, "status": "PROGRAM_INVALID", "error": reason}
    context = mp.get_context("spawn"); queue = context.Queue(maxsize=1); process = context.Process(target=_program_worker, args=(program, grid, queue)); process.start(); process.join(timeout_seconds)
    if process.is_alive(): process.terminate(); process.join(); return {"ok": False, "status": "TIMEOUT", "error": "execution_timeout"}
    try: value = queue.get_nowait()
    except Empty: return {"ok": False, "status": "EXECUTION_FAILED", "error": f"exitcode:{process.exitcode}"}
    return {"status": "SUCCESS" if value["ok"] else "EXECUTION_FAILED", **value}


def verify_program(program: str, train_pairs: list[tuple[Any, Any]]) -> dict[str, Any]:
    rows = [execute_program(program, source) for source, _target in train_pairs]
    passes = sum(int(row.get("ok") and row.get("grid") == target) for row, (_source, target) in zip(rows, train_pairs, strict=True))
    return {"program_valid": validate_program(program)[0], "train_pass_count": passes, "train_pair_count": len(train_pairs), "all_train_exact": passes == len(train_pairs), "train_execution": rows}
