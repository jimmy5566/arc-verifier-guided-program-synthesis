"""Freeze the smallest scientific repair for the E1 composition sentinel.

V1 remains immutable.  This successor replaces only the mislabeled
composition sentinel; training schedules, OOD prompts, retention prompts, and
measurement thresholds remain referenced by their existing SHA256 identities.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from connected_components_object_extraction_semantic_oracle import (
    target_for_family,
    validate_demonstrations,
)


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
OUT = BASE / "scientific_successor_v2"
FAMILY = "COUNT_THEN_OBJECT_EXTRACTION"
MARKER_COLOR = 5
SHAPES = (
    ((0, 0), (1, 0), (2, 0), (2, 1)),  # L
    ((0, 0), (0, 1), (1, 1), (1, 2)),  # Z
    ((0, 0), (0, 1), (0, 2), (1, 1)),  # T
)
COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def _rotate(points: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    current = list(points)
    for _ in range(turns % 4):
        current = [(x, 2 - y) for y, x in current]
    return tuple(current)


def _pair(variant: int) -> dict:
    """Generate without calling the independent semantic oracle."""
    grid = [[0] * 13 for _ in range(8)]
    count = 1 + (variant % 3)
    for index, x in enumerate((1, 5, 9)[:count]):
        grid[0][x] = MARKER_COLOR
    candidate_points = []
    for candidate, left in enumerate((0, 5, 10)):
        shape = _rotate(SHAPES[(variant + candidate) % len(SHAPES)], variant + 2 * candidate)
        color = COLORS[(variant * 3 + candidate) % len(COLORS)]
        candidate_points.append((shape, color))
        for y, x in shape:
            grid[4 + y][left + x] = color
    selected_shape, selected_color = candidate_points[count - 1]
    output = [[0] * 3 for _ in range(3)]
    min_y = min(y for y, _ in selected_shape)
    min_x = min(x for _, x in selected_shape)
    for y, x in selected_shape:
        output[y - min_y][x - min_x] = selected_color
    return {"input": grid, "output": output}


def _task(index: int, include_query_output: bool) -> dict:
    # Seven is coprime to the count/rotation/color cycles (3/4/8), avoiding
    # the repeated-query defect that a stride of four would introduce.
    sequence = 40000 + index * 7
    train = [_pair(sequence + offset) for offset in range(3)]
    query = _pair(sequence + 3)
    if not include_query_output:
        query = {"input": query["input"]}
    return {
        "source_id": f"COMPOSITION_SENTINEL_V2:{FAMILY}:{index:03d}",
        "train": train,
        "test": [query],
    }


def freeze(out: Path = OUT) -> dict:
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite successor freeze: {out}")
    prompts = []
    query_target_hashes = {}
    for index in range(16):
        full = _task(index, True)
        prompt = _task(index, False)
        validate_demonstrations(prompt, FAMILY)
        independent = target_for_family(FAMILY, prompt["test"][0]["input"])
        if full["test"][0]["output"] != independent:
            raise RuntimeError(f"independent composition oracle mismatch at {index}")
        prompts.append({
            "prompt_id": hashlib.sha256(canonical({"family": FAMILY, "index": index, "version": 2})).hexdigest(),
            "latent_condition": {"split": "COMPOSITION_SENTINEL", "family": FAMILY, "semantic_version": 2, "index": index},
            "task": prompt,
        })
        query_target_hashes[f"COMPOSITION:{index}"] = hashlib.sha256(canonical(independent)).hexdigest()

    prompt_path = out / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json"
    atomic(prompt_path, prompts)
    immutable = {}
    for name in (
        "CONTROL_SCHEDULE.json", "TREATMENT_SCHEDULE.json", "OOD_INPUT_PROMPTS.json",
        "RETENTION_INPUT_PROMPTS.json", "measurement_contract_v1/MEASUREMENT_ANALYSIS_CONTRACT.json",
        "measurement_contract_v1/PROTECTED_CAPABILITY_INPUT_PROMPTS.json",
        "measurement_contract_v1/TRAIN_FIT_INPUT_PROMPTS.json",
    ):
        immutable[name] = sha_file(BASE / name)
    validation = {
        "status": "PASS_CPU_ONLY_INDEPENDENT_SEMANTIC_ORACLE_VALIDATION",
        "retention": {
            "status": "PASS_NO_TRANSFORM_OBJECT_EXTRACTION",
            "records": 24,
            "semantic_rule": "extract the unique largest non-singleton component without transformation",
        },
        "composition": {
            "status": "PASS_IDENTIFIABLE_COUNT_THEN_OBJECT_EXTRACTION",
            "records": 16,
            "semantic_rule": "count top-row marker components, then extract the correspondingly indexed left-to-right candidate object",
            "query_target_sha256": query_target_hashes,
        },
        "v1_composition_disposition": "INVALID_CONSTRUCT_PRESERVED_NOT_USED_BY_SUCCESSOR",
        "generator_oracle_separation": True,
        "model_loaded": False,
        "gpu_used": False,
        "targets_exposed_to_model": False,
    }
    # Validate the immutable V1 retention prompts independently as part of the
    # successor gate.  No V1 file is modified.
    retention = json.loads((BASE / "RETENTION_INPUT_PROMPTS.json").read_text())
    for index, row in enumerate(retention):
        validate_demonstrations(row["task"], row["latent_condition"]["family"])
        target_for_family(row["latent_condition"]["family"], row["task"]["test"][0]["input"])
    validation_path = out / "SCIENTIFIC_CONSTRUCT_VALIDATION.json"
    atomic(validation_path, validation)
    manifest = {
        "schema_version": 2,
        "protocol_id": "CONNECTED_COMPONENTS_OBJECT_EXTRACTION_CAPABILITY_PILOT_V2_CONSTRUCT_REPAIR",
        "status": "CPU_ONLY_SCIENTIFIC_SUCCESSOR_FROZEN_PENDING_DIRECTOR_EXECUTION_REVIEW",
        "change_scope": "replace only invalid V1 composition sentinel; preserve training, OOD, retention, protected panels, budgets, and thresholds",
        "immutable_v1_inputs": immutable,
        "composition_prompt_path": str(
            (OUT / "COMPOSITION_SENTINEL_INPUT_PROMPTS.json").relative_to(ROOT)
        ).replace("\\", "/"),
        "composition_prompt_sha256": sha_file(prompt_path),
        "construct_validation_path": str(
            (OUT / "SCIENTIFIC_CONSTRUCT_VALIDATION.json").relative_to(ROOT)
        ).replace("\\", "/"),
        "construct_validation_sha256": sha_file(validation_path),
        "model_loaded": False,
        "gpu_used": False,
        "training_started": False,
        "sealed_targets_accessed": False,
    }
    atomic(out / "MANIFEST.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out), sort_keys=True))
