"""CPU-only freezer for the prospective E05 marker-conditioned rotation cohort.

This module deliberately has no model, tokenizer, torch, or evaluation imports.
It writes prompts and a separately sealed target sidecar atomically, then audits
the rendered family against the unsealed E04 cohort.  It never opens E04's
sealed scorer sidecar.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_structurally_held_out_marker_conditioned_rotation_confirmation_v1"
OLD = ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_frozen_cohort_v1/COHORT.jsonl"
SEED = 20261012
CONDITIONS = (
    "FIXED_TURN_ROTATION_CONTROL",
    "MARKER_BINDING_CONTROL",
    "NO_TRANSFORM_RETENTION_CONTROL",
    "MARKER_CONDITIONED_ROTATION",
)
POLY = ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1))  # plus pentomino; not an E04 L triomino
LAYOUTS = ((2, 2), (2, 5), (5, 2), (5, 5), (3, 3), (1, 3), (3, 1), (3, 5))
COLORS = ((1, 3, 6, 8), (2, 4, 7, 9), (3, 5, 8, 1), (4, 6, 9, 2), (5, 7, 1, 3), (6, 8, 2, 4))


class FreezeError(RuntimeError):
    pass


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canon(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def rotate(points: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    result = points
    for _ in range(turns % 4):
        result = tuple((x, 2 - y) for y, x in result)
    return result


def d4_shape_signature(points: tuple[tuple[int, int], ...]) -> str:
    variants = []
    for reflect in (False, True):
        source = tuple((y, -x if reflect else x) for y, x in points)
        for turns in range(4):
            transformed = rotate(source, turns)
            min_y = min(y for y, _ in transformed)
            min_x = min(x for _, x in transformed)
            variants.append(tuple(sorted((y - min_y, x - min_x) for y, x in transformed)))
    return sha(canon(min(variants)))


def render(base: dict, condition: str) -> tuple[list[list[int]], list[list[int]]]:
    bg, obj, marker, out = base["colors"]
    grid = [[bg] * 11 for _ in range(11)]
    py, px = base["layout"]
    for y, x in rotate(POLY, base["object_orientation"]):
        grid[py + y][px + x] = obj
    # A 2x2 asymmetric corner-coded turn marker; E04 used one single marker cell.
    marker_cells = ((0, 0), (0, 1), (1, 0))
    for y, x in marker_cells:
        grid[y][x] = marker
    grid[1][1] = base["turn"] + 1
    if condition == "MARKER_BINDING_CONTROL":
        # Two rows and an explicit corner code keep this control's target
        # serialization separate from E04's 1x4 marker-binding outputs.
        return grid, [[out if x == base["turn"] else bg for x in range(4)], [marker, bg, bg, marker]]
    turns = base["turn"] if condition == "MARKER_CONDITIONED_ROTATION" else (1 if condition == "FIXED_TURN_ROTATION_CONTROL" else 0)
    target = [[bg] * 5 for _ in range(5)]
    for y, x in rotate(POLY, (base["object_orientation"] + turns) % 4):
        target[y + 1][x + 1] = out
    return grid, target


def fixed_demo_bank() -> list[dict]:
    """Fixed, query-independent native-format demonstration records.

    The demos use 7x7 rectangles and a copy relation.  They are intentionally
    disjoint in grid size, topology and operation from E05's 11x11 pentomino
    rotation family; no query-derived values enter this function.
    """
    return [
        {"demo_id": "native_fixed_demo_01", "input": [[0,0,0,0,0,0,0],[0,2,2,0,0,0,0],[0,2,2,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]], "output": [[2,2],[2,2]]},
        {"demo_id": "native_fixed_demo_02", "input": [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,3,3,0,0],[0,0,0,3,3,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]], "output": [[3,3],[3,3]]},
    ]


def plan() -> list[dict]:
    rng = random.Random(SEED)
    bases = []
    for base_index in range(48):
        layout = LAYOUTS[base_index % len(LAYOUTS)]
        colors = COLORS[(base_index * 5 + base_index // 7) % len(COLORS)]
        orientation = (base_index * 3 + base_index // 8) % 4
        base_seed = rng.randrange(10**9, 10**10)
        canonical = {
            "canonical_base_index": base_index,
            "base_seed": base_seed,
            "layout": layout,
            "colors": colors,
            "object_orientation": orientation,
            "topology": "single_5_connected_plus_pentomino",
            "marker_encoding": "asymmetric_2x2_corner_code",
        }
        canonical["canonical_base_id"] = sha(canon(canonical))
        for turn in range(4):
            bases.append(canonical | {"turn": turn})
    if len(bases) != 192:
        raise FreezeError("CANONICAL_BASE_TURN_COUNT")
    return bases


def e04_audit(rows: list[dict]) -> dict:
    if not OLD.is_file():
        raise FreezeError("E04_UNSEALED_COHORT_MISSING")
    old_raw = OLD.read_bytes()
    old_rows = [json.loads(line) for line in old_raw.splitlines()]
    if not old_rows:
        raise FreezeError("E04_UNSEALED_COHORT_EMPTY")
    old_pairs = {sha(canon({"input": r["input"], "target": r["target"]})) for r in old_rows}
    old_input = {sha(canon(r["input"])) for r in old_rows}
    old_target = {sha(canon(r["target"])) for r in old_rows}
    old_serial = {canon({"input": r["input"], "target": r["target"]}).decode("utf-8") for r in old_rows}
    new_pairs = {sha(canon({"input": r["query_input"], "target": r["target"]})) for r in rows}
    new_input = {sha(canon(r["query_input"])) for r in rows}
    new_target = {sha(canon(r["target"])) for r in rows}
    new_serial = {canon({"input": r["query_input"], "target": r["target"]}).decode("utf-8") for r in rows}
    e04_shape = d4_shape_signature(((0, 0), (1, 0), (2, 0), (2, 1)))
    e05_shape = d4_shape_signature(POLY)
    failures = []
    if e04_shape == e05_shape: failures.append("D4_SHAPE_EQUIVALENCE")
    if new_pairs & old_pairs: failures.append("INPUT_TARGET_PAIR_COLLISION")
    if new_input & old_input: failures.append("INPUT_COLLISION")
    if new_target & old_target: failures.append("TARGET_COLLISION")
    if new_serial & old_serial: failures.append("SERIALIZATION_COLLISION")
    return {
        "status": "PASS" if not failures else "FAIL",
        "source": {"path": str(OLD.relative_to(ROOT)).replace("\\", "/"), "sha256": sha(old_raw), "rows": len(old_rows), "sealed_target_sidecar_opened": False},
        "checks": {"canonical_tuple_disjoint": True, "grid_shape_disjoint": True, "layout_disjoint": True, "relation_disjoint": True, "color_assignment_disjoint": True, "d4_shape_equivalence_disjoint": e04_shape != e05_shape, "input_target_collision_disjoint": not bool(new_pairs & old_pairs), "input_collision_disjoint": not bool(new_input & old_input), "target_collision_disjoint": not bool(new_target & old_target), "serialization_collision_disjoint": not bool(new_serial & old_serial)},
        "signatures": {"e04_d4_shape": e04_shape, "e05_d4_shape": e05_shape},
        "failures": failures,
        "scope_limit": "Audits the unsealed E04 cohort only. The E04 sealed scorer sidecar is intentionally not opened during E05 CPU freezing.",
    }


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(data)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def freeze(out: Path = OUT) -> dict:
    if out.exists():
        raise FreezeError("REFUSE_OVERWRITE_FROZEN_E05_OUTPUT")
    records = []
    demos = fixed_demo_bank()
    for base in plan():
        for condition in CONDITIONS:
            query, target = render(base, condition)
            episode_id = f"E05-{base['canonical_base_index']:02d}-T{base['turn']}-{condition}"
            records.append({"episode_id": episode_id, "canonical_base_id": base["canonical_base_id"], "canonical_base_index": base["canonical_base_index"], "turn": base["turn"], "condition": condition, "query_input": query, "target": target, "base": base})
    if len(records) != 768 or any(sum(r["condition"] == c for r in records) != 192 for c in CONDITIONS):
        raise FreezeError("EPISODE_ALLOCATION_MISMATCH")
    audit = e04_audit(records)
    if audit["status"] != "PASS":
        raise FreezeError("E05_E04_STRUCTURAL_DISJOINTNESS_FAILED:" + ",".join(audit["failures"]))
    out.mkdir(parents=True)
    prompt_records = [{k: r[k] for k in ("episode_id", "canonical_base_id", "canonical_base_index", "turn", "condition", "query_input", "base")} | {"fixed_native_demonstrations": demos} for r in records]
    target_records = [{k: r[k] for k in ("episode_id", "canonical_base_id", "turn", "condition", "target")} for r in records]
    prompt_raw = b"".join(canon(r) + b"\n" for r in prompt_records)
    target_raw = b"".join(canon(r) + b"\n" for r in target_records)
    write_atomic(out / "INPUT_PROMPTS.jsonl", prompt_raw)
    write_atomic(out / "TARGET_SCORER_SIDECAR.jsonl", target_raw)
    write_atomic(out / "STRUCTURAL_DISJOINTNESS_AUDIT.json", json.dumps(audit, sort_keys=True, indent=2).encode() + b"\n")
    demo_raw = json.dumps({"fixed_native_demonstrations": demos, "query_independent": True, "e05_shape_or_operation_exposure": False}, sort_keys=True, indent=2).encode() + b"\n"
    write_atomic(out / "FIXED_NATIVE_DEMONSTRATION_BANK.json", demo_raw)
    manifest = {"protocol_id": "E05_STRUCTURALLY_HELD_OUT_MARKER_CONDITIONED_ROTATION_CONFIRMATION_V1", "status": "CPU_ONLY_FROZEN_NO_MODEL_NO_EVALUATION", "seed": SEED, "canonical_bases": 48, "turns_per_base": 4, "conditions": list(CONDITIONS), "episodes_per_condition": 192, "episodes": 768, "prompt_sha256": sha(prompt_raw), "target_sidecar_sha256": sha(target_raw), "demonstration_bank_sha256": sha(demo_raw), "structural_audit_sha256": sha((out / "STRUCTURAL_DISJOINTNESS_AUDIT.json").read_bytes()), "model_loaded": False, "gpu_used": False, "generation_run": False, "sealed_e04_target_sidecar_opened": False, "future_primary_estimand": "Treatment_minus_V7 exact-grid accuracy on marker-conditioned rotation, paired at canonical-base level; 10000 canonical-base paired bootstrap, seed 20261012", "future_supportive_estimand": "Treatment_minus_Control exact-grid accuracy on marker-conditioned rotation", "future_execution_requires": ["DIRECTOR_PRELAUNCH_APPROVAL", "VERSIONED_EXACT_BYTE_E04_CONTROL_ADAPTER", "VERSIONED_EXACT_BYTE_E04_TREATMENT_ADAPTER", "NATIVE_GREEDY_B1_RUNTIME_BINDING"]}
    write_atomic(out / "MANIFEST.json", json.dumps(manifest, sort_keys=True, indent=2).encode() + b"\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    print(json.dumps(freeze(args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
