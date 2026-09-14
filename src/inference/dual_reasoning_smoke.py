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
import numbers
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any


MIN_SOAR_GGUF_BYTES = 1 * 1024 * 1024 * 1024


def discover_models(
    models_root: Path,
    *,
    required: tuple[str, ...] = ("native", "soar"),
    input_root: Path | None = None,
) -> dict[str, Path]:
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
    # SOAR is attached as a notebook-input GGUF rather than a Kaggle Model
    # source.  Search every mounted input only for its own role, and reject
    # any HTML or tiny placeholder with an explicit multi-GiB size gate.
    search_root = input_root if input_root is not None else models_root
    gguf_candidates = sorted(
        path for path in search_root.rglob("*.gguf")
        if path.is_file()
        and "soar" in path.name.lower()
        and "14b" in path.name.lower()
        and "q4_k_m" in path.name.lower()
        and path.stat().st_size >= MIN_SOAR_GGUF_BYTES
    )
    selected = {"native": native, "soar": gguf_candidates}
    invalid = {role: len(selected[role]) for role in required if len(selected[role]) != 1}
    if invalid:
        inventory = sorted(str(path.relative_to(models_root)) for path, _config, _marker in choices)[:32]
        gguf_inventory = sorted(
            f"{path.relative_to(search_root)}:{path.stat().st_size}"
            for path in search_root.rglob("*.gguf") if path.is_file()
        )[:32]
        counts = {role: len(values) for role, values in selected.items()}
        raise RuntimeError(
            "could not uniquely discover required model checkpoints: "
            f"required={required}, invalid={invalid}, counts={counts}, "
            f"scanned={len(choices)}, inventory={inventory}, gguf_inventory={gguf_inventory}"
        )
    return {
        role: selected[role][0][0] if role == "native" else selected[role][0]
        for role in required
    }


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
        "Return only Python code in one fenced python block. Define one `def transform(input_grid):` function; "
        "the single parameter may have any valid name. Input is a rectangular list[list[int]] with colors 0..9. "
        "You may use Python list operations and, if useful, exactly `import numpy as np`; return a rectangular ARC grid "
        "as list[list[int]] or numpy.ndarray. Do not access files, network, shell, eval, exec, or print.\n\n" + examples
    )


def extract_program(text: str) -> str | None:
    """Use SOAR's documented fenced-code extraction transport.

    As in the official ``postprocess_transform``, retain only imports and
    function definitions.  This drops narrative/top-level scratch statements,
    never edits a function body, and is therefore parsing transport rather than
    semantic repair.
    """
    blocks = re.findall(r"```python\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    candidate = next((block for block in reversed(blocks) if "def transform(" in block), "")
    if not candidate and "def transform(" in text:
        candidate = text[text.find("def transform("):]
    if not candidate.strip():
        return None
    try:
        tree = ast.parse(candidate, mode="exec")
        nodes = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef))]
        return ast.unparse(ast.Module(body=nodes, type_ignores=[])).strip() or None
    except (SyntaxError, ValueError):
        return candidate.strip()


_FORBIDDEN_AST = (ast.ClassDef, ast.Global, ast.Nonlocal, ast.With, ast.AsyncWith, ast.Try, ast.Raise, ast.Delete)
# Deliberately boring, deterministic Python-only helpers.  SOAR's public
# programs routinely use ``all``, ``any``, numeric conversion and iteration;
# withholding those made otherwise safe programs fail after static validation.
# Capability-opening functions (file/process/introspection/dynamic execution)
# remain absent and are separately rejected by the AST policy below.
_SAFE_BUILTINS = {
    "range": range, "len": len, "enumerate": enumerate, "min": min,
    "max": max, "sum": sum, "abs": abs, "sorted": sorted, "set": set,
    "dict": dict, "tuple": tuple, "list": list, "int": int, "float": float,
    "bool": bool, "str": str, "zip": zip, "all": all, "any": any,
    "next": next, "reversed": reversed, "isinstance": isinstance,
    "map": map, "filter": filter, "frozenset": frozenset, "round": round,
}
_FORBIDDEN_NAMES = {"open", "eval", "exec", "compile", "globals", "locals", "vars", "input", "help", "breakpoint", "os", "sys", "subprocess", "pathlib", "socket", "requests", "shutil", "ctypes", "importlib", "__import__"}
_FORBIDDEN_NUMPY_ATTRIBUTES = {"load", "save", "savez", "savez_compressed", "savetxt", "loadtxt", "genfromtxt", "fromfile", "tofile", "memmap", "DataSource", "ctypeslib", "f2py"}
_ALLOWED_IMPORTS = {
    "numpy": {"np"},
    "collections": {"Counter", "defaultdict", "deque"},
    "itertools": {"combinations", "permutations", "product"},
    "math": set(),
    "typing": {"Any", "Dict", "List", "Sequence", "Set", "Tuple"},
}


def inspect_program(program: str) -> dict[str, Any]:
    """Parse and statically validate the SOAR transport contract."""
    try:
        tree = ast.parse(program, mode="exec")
    except SyntaxError as error:
        return {"parse_valid": False, "static_safe": False, "reason": f"syntax:{error.msg}"}
    functions = [item for item in tree.body if isinstance(item, ast.FunctionDef)]
    imports = [item for item in tree.body if isinstance(item, (ast.Import, ast.ImportFrom))]
    permitted_body = set(functions + imports)
    transform_functions = [item for item in functions if item.name == "transform"]
    if len(tree.body) != len(permitted_body) or len(transform_functions) != 1:
        return {"parse_valid": True, "static_safe": False, "reason": "requires_imports_helpers_and_one_transform_function"}
    arguments = transform_functions[0].args
    if len(arguments.args) != 1 or arguments.defaults or arguments.posonlyargs or arguments.vararg or arguments.kwarg or arguments.kwonlyargs:
        return {"parse_valid": True, "static_safe": False, "reason": "requires_one_required_positional_parameter"}
    for imported in imports:
        if isinstance(imported, ast.Import):
            if len(imported.names) != 1:
                return {"parse_valid": True, "static_safe": False, "reason": "invalid_import"}
            item = imported.names[0]
            if item.name not in _ALLOWED_IMPORTS or item.asname not in _ALLOWED_IMPORTS[item.name]:
                return {"parse_valid": True, "static_safe": False, "reason": "import_not_allowlisted"}
        else:
            if imported.level != 0 or imported.module not in _ALLOWED_IMPORTS or any(item.name not in _ALLOWED_IMPORTS[imported.module] or item.asname for item in imported.names):
                return {"parse_valid": True, "static_safe": False, "reason": "import_not_allowlisted"}
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_AST):
            return {"parse_valid": True, "static_safe": False, "reason": f"forbidden:{type(node).__name__}"}
        if isinstance(node, ast.Name) and (node.id.startswith("__") or node.id in _FORBIDDEN_NAMES):
            return {"parse_valid": True, "static_safe": False, "reason": f"forbidden_name:{node.id}"}
        if isinstance(node, ast.Attribute) and (node.attr.startswith("__") or node.attr in _FORBIDDEN_NUMPY_ATTRIBUTES):
            return {"parse_valid": True, "static_safe": False, "reason": f"forbidden_attribute:{node.attr}"}
    return {"parse_valid": True, "static_safe": True, "reason": "ok"}


def validate_program(program: str) -> tuple[bool, str]:
    inspection = inspect_program(program)
    return bool(inspection["static_safe"]), str(inspection["reason"])


def _safe_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
    """Trusted import bridge matching the statically validated small allowlist."""
    if level != 0 or name not in _ALLOWED_IMPORTS:
        raise ImportError("import is not allowlisted")
    requested = set(fromlist or ())
    allowed = _ALLOWED_IMPORTS[name]
    if requested and not requested.issubset(allowed):
        raise ImportError("import member is not allowlisted")
    return __import__(name, globals_, locals_, tuple(requested), 0)


def _normalise_grid(result: Any, *, numpy_allowed: bool) -> list[list[int]]:
    if numpy_allowed:
        try:
            import numpy as np
            if isinstance(result, np.ndarray):
                result = result.tolist()
        except ImportError:
            pass
    if not isinstance(result, list) or not result or not all(isinstance(row, list) and row for row in result):
        raise ValueError("result_not_nonempty_grid")
    width = len(result[0])
    if len(result) > 30 or width > 30 or any(len(row) != width for row in result):
        raise ValueError("result_shape_invalid")
    if any(not isinstance(cell, numbers.Integral) or isinstance(cell, bool) or not 0 <= int(cell) <= 9 for row in result for cell in row):
        raise ValueError("result_colors_invalid")
    return [[int(cell) for cell in row] for row in result]


def _linux_rss_mb() -> dict[str, float | None]:
    """Current and high-water RSS from Linux procfs when it is available."""
    try:
        rows = Path("/proc/self/status").read_text(encoding="utf-8").splitlines()
        values = {line.split(":", 1)[0]: int(line.split()[1]) / 1024.0 for line in rows if line.startswith(("VmRSS:", "VmHWM:"))}
        return {"rss_mb": round(values.get("VmRSS"), 3) if "VmRSS" in values else None, "peak_rss_mb": round(values.get("VmHWM"), 3) if "VmHWM" in values else None}
    except (OSError, ValueError, IndexError):
        return {"rss_mb": None, "peak_rss_mb": None}


def _sandbox_result(program: str, grid: Any) -> dict[str, Any]:
    """Run in a fresh ``exec``-ed interpreter, never in the model process."""
    try:
        try:
            import resource
            # This is a fresh Python interpreter, so its address space contains
            # no torch/CUDA model mappings.  Keep a generous data-segment cap
            # for ordinary numpy grids without using a tiny inherited RLIMIT_AS.
            if hasattr(resource, "RLIMIT_DATA"):
                resource.setrlimit(resource.RLIMIT_DATA, (1024 * 1024 * 1024, 1024 * 1024 * 1024))
        except Exception: pass
        numpy_import_seconds = 0.0

        def sandbox_import(name: str, globals_: Any = None, locals_: Any = None, fromlist: Any = (), level: int = 0) -> Any:
            nonlocal numpy_import_seconds
            started = perf_counter()
            value = _safe_import(name, globals_, locals_, fromlist, level)
            numpy_import_seconds += perf_counter() - started
            return value

        environment = {"__builtins__": {**_SAFE_BUILTINS, "__import__": sandbox_import}}
        exec(compile(program, "<soar-program>", "exec"), environment, environment)
        transform_started = perf_counter()
        result = environment["transform"]([[int(cell) for cell in row] for row in grid])
        payload: dict[str, Any] = {
            "ok": True, "grid": _normalise_grid(result, numpy_allowed="import numpy as np" in program), "output_valid": True,
            "numpy_import_seconds": numpy_import_seconds, "transform_seconds": perf_counter() - transform_started,
        }
    except BaseException as error:
        payload = {"ok": False, "error": f"{type(error).__name__}:{error}", "numpy_import_seconds": locals().get("numpy_import_seconds", 0.0), "transform_seconds": None}
    payload.update(_linux_rss_mb())
    return payload


def _sandbox_main() -> int:
    """JSON-line entrypoint deliberately free of torch/transformers imports."""
    started = perf_counter()
    try:
        request = json.loads(sys.stdin.read())
        program, grid = request["program"], request["grid"]
        inspection = inspect_program(program)
        value = _sandbox_result(program, grid) if inspection["static_safe"] else {
            "ok": False, "output_valid": False, "error": inspection["reason"]
        }
        print(json.dumps({**inspection, **value, "sandbox_work_seconds": perf_counter() - started}), flush=True)
        return 0
    except BaseException as error:
        print(json.dumps({"ok": False, "output_valid": False, "error": f"sandbox:{type(error).__name__}:{error}"}), flush=True)
        return 1


def execute_program(program: str, grid: Any, *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    inspection = inspect_program(program)
    if not inspection["static_safe"]:
        return {"ok": False, "status": "PROGRAM_INVALID", "executable": False, "output_valid": False, **inspection, "error": inspection["reason"]}
    source_root = str(Path(__file__).resolve().parents[1])
    environment = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "PYTHONPATH": source_root + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""),
    }
    outer_started = perf_counter()
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "inference.dual_reasoning_smoke", "--sandbox"],
            input=json.dumps({"program": program, "grid": grid}), text=True,
            capture_output=True, timeout=timeout_seconds, env=environment,
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "status": "TIMEOUT", "process_started": True, "executable": True, "output_valid": False, **inspection, "error": "execution_timeout", "total_wall_seconds": perf_counter() - outer_started}
    try:
        value = json.loads(completed.stdout.strip())
    except json.JSONDecodeError:
        error = (completed.stderr.strip() or completed.stdout.strip() or f"exitcode:{completed.returncode}")[-1000:]
        return {"ok": False, "status": "EXECUTION_FAILED", "process_started": True, "executable": False, "output_valid": False, **inspection, "error": error, "returncode": completed.returncode, "total_wall_seconds": perf_counter() - outer_started}
    total_wall = perf_counter() - outer_started
    work_seconds = value.get("sandbox_work_seconds")
    return {
        "status": "SUCCESS" if value["ok"] else "EXECUTION_FAILED", "process_started": True, "executable": True,
        **inspection, **value, "returncode": completed.returncode, "total_wall_seconds": total_wall,
        "python_startup_seconds": max(0.0, total_wall - float(work_seconds)) if work_seconds is not None else None,
    }


def verify_program(program: str, train_pairs: list[tuple[Any, Any]], *, timeout_seconds: float = 2.0) -> dict[str, Any]:
    rows = [execute_program(program, source, timeout_seconds=timeout_seconds) for source, _target in train_pairs]
    passes = sum(int(row.get("ok") and row.get("grid") == target) for row, (_source, target) in zip(rows, train_pairs, strict=True))
    inspection = inspect_program(program)
    return {"program_valid": inspection["static_safe"], **inspection, "train_pass_count": passes, "train_pair_count": len(train_pairs), "all_train_exact": passes == len(train_pairs), "train_execution": rows}


if __name__ == "__main__":
    if "--sandbox" not in sys.argv:
        raise SystemExit("this module only exposes the --sandbox child entrypoint")
    raise SystemExit(_sandbox_main())
