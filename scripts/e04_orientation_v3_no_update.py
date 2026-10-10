"""Frozen E04 V3 execution contract and target-blind generation helpers.

This module deliberately contains no torch/transformers imports. It freezes
the input-only generation boundary and validates raw outputs before the scorer
may open the separate target sidecar.
"""
from __future__ import annotations
import hashlib, json, os, sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
OUT = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3"
PROMPTS = OUT / "VALIDATION_INPUT_PROMPTS.jsonl"
MANIFEST = OUT / "MANIFEST.json"
CONTRACT = OUT / "E04_V3_NO_UPDATE_EXECUTION_CONTRACT_V1.json"
SENSITIVITY = OUT / "E04_V3_BATCH1_SENSITIVITY_INPUT_INDEXES_V1.json"
CONDITIONS = ("ROTATION_TARGET", "FIXED_TURN_ROTATION_CONTROL", "MARKER_BINDING_CONTROL", "NO_TRANSFORM_RETENTION_CONTROL")
TURNS = (0, 1, 2, 3)
CAP_SECONDS = 1800
SEED = 20261010

class E04ExecutionFailure(RuntimeError):
    pass

def canon(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")

def sha_path(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, sort_keys=True, indent=2).encode("utf-8") + b"\n")
    os.replace(temporary, path)

def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canon(row) + b"\n" for row in rows))
    os.replace(temporary, path)

def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line]

def _assert_input_only(prompt: dict[str, Any]) -> None:
    if set(prompt) != {"train", "test"} or len(prompt["test"]) != 1 or set(prompt["test"][0]) != {"input"}:
        raise E04ExecutionFailure("E04_INPUT_PROMPT_STRUCTURE")
    rendered = canon(prompt).decode("ascii").lower()
    if any(word in rendered for word in ("target", "condition", "canonical", "split", "sha", "hash", "orientation", "reference_relation")):
        raise E04ExecutionFailure("E04_INPUT_PROMPT_METADATA_OR_TARGET_LEAKAGE")
    for pair in prompt["train"]:
        if set(pair) != {"input", "output"}:
            raise E04ExecutionFailure("E04_DEMONSTRATION_STRUCTURE")

def prompt_messages(prompt: dict[str, Any]) -> list[dict[str, str]]:
    """Construct native messages from input-only ARC prompt data."""
    _assert_input_only(prompt)
    from inference.nvarc_native import serialize_grid
    result: list[dict[str, str]] = []
    for pair in prompt["train"]:
        result.append({"role": "user", "content": serialize_grid(pair["input"])})
        result.append({"role": "assistant", "content": serialize_grid(pair["output"])})
    result.append({"role": "user", "content": serialize_grid(prompt["test"][0]["input"])})
    return result

def sensitivity_indexes() -> tuple[list[int], list[dict[str, Any]]]:
    """Four lowest-ID bases in the four reserved cells x 4 turns x 4 conditions."""
    from scripts import e04_orientation_marker_counterfactual_v2 as v2
    by_cell: dict[tuple[int, str], list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for base_index, base in enumerate(v2.canonical_bases("VALIDATION")):
        by_cell[(base["shape_orientation"], base["reference_relation"])].append((base_index, base))
    result: list[int] = []
    selected: list[dict[str, Any]] = []
    for cell in v2.cells("VALIDATION"):
        base_index, base = min(by_cell[cell], key=lambda item: item[1]["canonical_base_id"])
        selected.append({"reserved_cell": {"shape_orientation": cell[0], "reference_relation": cell[1]},
                         "canonical_base_id": base["canonical_base_id"], "base_index": base_index})
        result.extend(base_index * 16 + turn * 4 + condition_index
                      for turn in TURNS for condition_index in range(len(CONDITIONS)))
    if len(result) != 64 or len(set(result)) != 64:
        raise E04ExecutionFailure("E04_SENSITIVITY_INDEX_CARDINALITY")
    return result, selected

def freeze_contract(destination: Path = OUT) -> dict[str, Any]:
    prompts_path = destination / PROMPTS.name
    manifest_path = destination / MANIFEST.name
    if not prompts_path.is_file() or not manifest_path.is_file():
        raise E04ExecutionFailure("E04_FROZEN_INPUTS_MISSING")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows = read_jsonl(prompts_path)
    if len(rows) != 768:
        raise E04ExecutionFailure("E04_PRIMARY_ROW_COUNT")
    for row in rows:
        _assert_input_only(row)
    indexes, selected = sensitivity_indexes()
    sensitivity = {
        "protocol_id": manifest["protocol_id"],
        "selection": "lowest canonical_base_id per reserved validation cell; all turns and conditions",
        "row_indexes": indexes,
        "row_input_sha256": {str(index): hashlib.sha256(canon(rows[index])).hexdigest() for index in indexes},
        "selected_bases": selected,
        "rows": 64,
        "target_sidecar_accessed": False,
    }
    contract = {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "authorization_id": "E04_V3_CLEAN_BASELINE_ONE_SHOT_CONDITIONAL_20261010",
        "mode": "V7_NATIVE_GREEDY_NO_UPDATE",
        "input_prompts_path": str(prompts_path.relative_to(ROOT)).replace("\\", "/"),
        "input_prompts_sha256": sha_path(prompts_path),
        "input_rows": 768,
        "requested_batch_size": 32,
        "fallback_ladder": [32, 16, 8, 4, 1],
        "fallback_rule": "complete-primary-rerun-only-after-pre-output-capacity-failure; no mixed accepted rung",
        "batch1_sensitivity_manifest_path": str((destination / SENSITIVITY.name).relative_to(ROOT)).replace("\\", "/"),
        "batch1_sensitivity_rows": 64,
        "batch_agreement": "parser-validity, parsed-grid and exact-outcome must agree for every sensitivity row",
        "hard_runtime_cap_seconds": CAP_SECONDS,
        "seed": SEED,
        "forbidden": ["NO_TRAINING", "NO_OPTIMIZER", "NO_BACKWARD", "NO_GRADIENT", "NO_TTT", "NO_AUGMENTATION", "NO_DFS", "NO_BEAM4", "NO_GOLD", "NO_DGOLD", "NO_FINAL_AUDIT"],
        "target_sidecar_accessed": False,
    }
    atomic_json(destination / SENSITIVITY.name, sensitivity)
    atomic_json(destination / CONTRACT.name, contract)
    return contract

def validate_contract(contract_path: Path = CONTRACT) -> dict[str, Any]:
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    required = {"schema_version", "protocol_id", "authorization_id", "mode", "input_prompts_path", "input_prompts_sha256",
                "input_rows", "requested_batch_size", "fallback_ladder", "fallback_rule",
                "batch1_sensitivity_manifest_path", "batch1_sensitivity_rows", "batch_agreement",
                "hard_runtime_cap_seconds", "seed", "forbidden", "target_sidecar_accessed"}
    if set(contract) != required:
        raise E04ExecutionFailure("E04_EXECUTION_CONTRACT_SCHEMA")
    if contract["protocol_id"] != "E04_ORIENTATION_V3_FIXED_INDEPENDENT_DEMONSTRATION_BASELINE":
        raise E04ExecutionFailure("E04_PROTOCOL_IDENTITY")
    if contract["requested_batch_size"] != 32 or contract["fallback_ladder"] != [32, 16, 8, 4, 1]:
        raise E04ExecutionFailure("E04_BATCH_CONTRACT")
    if contract["hard_runtime_cap_seconds"] != CAP_SECONDS or contract["input_rows"] != 768:
        raise E04ExecutionFailure("E04_RUNTIME_OR_ROWS_CONTRACT")
    if contract["target_sidecar_accessed"] is not False:
        raise E04ExecutionFailure("E04_TARGET_BOUNDARY_VIOLATION")
    prompts = ROOT / contract["input_prompts_path"]
    if not prompts.is_file() or sha_path(prompts) != contract["input_prompts_sha256"]:
        raise E04ExecutionFailure("E04_INPUT_PROMPT_IDENTITY")
    rows = read_jsonl(prompts)
    if len(rows) != 768:
        raise E04ExecutionFailure("E04_INPUT_ROW_COUNT")
    for row in rows:
        _assert_input_only(row)
    subset_path = ROOT / contract["batch1_sensitivity_manifest_path"]
    subset = json.loads(subset_path.read_text(encoding="utf-8"))
    indexes = subset.get("row_indexes")
    if not isinstance(indexes, list) or len(indexes) != 64 or len(set(indexes)) != 64 or any(not isinstance(x, int) or not 0 <= x < 768 for x in indexes):
        raise E04ExecutionFailure("E04_SENSITIVITY_INDEX_CONTRACT")
    expected = subset.get("row_input_sha256", {})
    if any(expected.get(str(index)) != hashlib.sha256(canon(rows[index])).hexdigest() for index in indexes):
        raise E04ExecutionFailure("E04_SENSITIVITY_INPUT_IDENTITY")
    return {"status": "PASS_NO_MODEL_IMPORT", "input_rows": len(rows), "sensitivity_rows": len(indexes),
            "model_imported": False, "gpu_used": False, "optimizer_steps": 0, "target_sidecar_accessed": False}

def validate_raw_generation(raw_path: Path, expected_indexes: list[int], *, expected_input_sha: dict[str, str] | None = None) -> list[dict[str, Any]]:
    rows = read_jsonl(raw_path)
    indexes = [row.get("row_index") for row in rows]
    if indexes != expected_indexes:
        raise E04ExecutionFailure("E04_RAW_ROW_ORDER_OR_COVERAGE")
    for row in rows:
        if set(row) != {"row_index", "input_sha256", "text", "parser_valid", "parsed_grid", "prompt_tokens", "completion_tokens", "elapsed_seconds", "effective_batch_size"}:
            raise E04ExecutionFailure("E04_RAW_SCHEMA")
        if not isinstance(row["parser_valid"], bool) or (row["parser_valid"] != (row["parsed_grid"] is not None)):
            raise E04ExecutionFailure("E04_RAW_PARSE_CONTRACT")
        if expected_input_sha is not None and expected_input_sha.get(str(row["row_index"])) != row["input_sha256"]:
            raise E04ExecutionFailure("E04_RAW_INPUT_IDENTITY")
    return rows

if __name__ == "__main__":
    print(json.dumps({"contract": freeze_contract(), "validation": validate_contract()}, sort_keys=True))


