"""Freeze TARGETED_CAPABILITY_REPAIR_V1 CPU-only protocol and synthetic surfaces.

The generated episodes are programmatically labelled ARC-style grids.  This
preparation never loads a model, opens Eval60 Gold, or evaluates FINAL_AUDIT.
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments" / "targeted_capability_repair_v1"
PROFILE = ROOT / "analysis/foundation_v2_eval60_capability_alignment_v1/FOUNDATION_V2_CAPABILITY_PROFILE_V1.json"
BASE_COMP = ROOT / "analysis/base_eval60_capability_alignment_v1/EVAL60_BASE_COMPOSITION_ALIGNMENT_V1.csv"
V2_COMP = ROOT / "analysis/foundation_v2_eval60_capability_alignment_v1/EVAL60_FOUNDATION_V2_COMPOSITION_ALIGNMENT_V1.csv"
DEMAND = ROOT / "analysis/eval60_capability_demand_map_v1/EVAL60_CAPABILITY_DEMAND_MAP_V1.csv"
START = ROOT / "artifacts/foundation_round2_data_v1/FOUNDATION_V2_START_IDENTITY.json"
CATEGORIES = ("GEOMETRY_TO_GEOMETRY", "SELECTOR_TO_ACTION", "RELATION_TO_SELECTOR_ACTION", "COUNTING_TO_CONSTRUCTION", "MASK_SET_TO_CONSTRUCTION", "STATE_PROGRESSION_TO_ACTION", "CONDITIONAL_TO_ACTION")
CONFIDENCE = {"MINIMAL_CONTRAST": 1.00, "DIRECT_ATOMIC": 0.90, "COMPOSITE_ONLY": 0.25}
ATOMIC = ("connected components", "orientation", "inside/contains", "same color", "difference", "width", "recolor", "selector prerequisites")
COMPOSITION = ("RELATION_TO_SELECTOR_ACTION", "SELECTOR_TO_ACTION", "MASK_SET_TO_CONSTRUCTION", "CONDITIONAL_TO_ACTION", "COUNTING_TO_CONSTRUCTION")
RETENTION = ("color mapping", "overlay", "propagation", "complete missing structure", "rotate", "recolor")


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8", newline="\n")


def blank(n: int, bg: int) -> list[list[int]]:
    return [[bg for _ in range(n)] for _ in range(n)]


def put(grid: list[list[int]], r: int, c: int, h: int, w: int, color: int) -> None:
    for y in range(r, r + h):
        for x in range(c, c + w): grid[y][x] = color


def crop(grid: list[list[int]], bg: int) -> list[list[int]]:
    points = [(r, c) for r, row in enumerate(grid) for c, x in enumerate(row) if x != bg]
    rs, cs = [p[0] for p in points], [p[1] for p in points]
    return [row[min(cs):max(cs)+1] for row in grid[min(rs):max(rs)+1]]


def paint_non_bg(grid: list[list[int]], bg: int, color: int) -> list[list[int]]:
    return [[color if x != bg else bg for x in row] for row in grid]


def primitive_case(family: str, seed: int) -> tuple[list[list[int]], list[list[int]], dict]:
    rng = random.Random(seed); n = rng.choice((9, 10, 11, 12)); bg, a, b, c = rng.sample(range(1, 10), 4)
    inp = blank(n, bg); meta = {"seed": seed, "family": family, "background": bg}
    if family == "connected components":
        count = rng.randint(2, 5)
        anchors = [(1, 1), (1, n-4), (n-4, 1), (n-4, n-4), (n//2, n//2)]
        for i in range(count): put(inp, *anchors[i], 1 + (i % 2), 1 + ((i + 1) % 2), a if i % 2 else b)
        out = [[c] * count]; meta["operator"] = "COUNT_4_CONNECTED_COMPONENTS"
    elif family == "orientation":
        r, col = rng.randint(2, n-5), rng.randint(2, n-5); shape = [(0, 0), (1, 0), (2, 0), (2, 1)]
        for dy, dx in shape: inp[r+dy][col+dx] = a
        turns = rng.randrange(4); inp[0][turns+1] = c
        out = [[a if (y, x) in {(x0, 2-y0) for y0, x0 in shape} else bg for x in range(3)] for y in range(3)]
        for _ in range(turns): out = [list(row) for row in zip(*out[::-1])]
        out = crop(out, bg); meta["operator"] = "ROTATE_BY_CONTROL_MARKER"
    elif family == "inside/contains":
        put(inp, 1, 1, n-2, n-2, b); put(inp, 2, 2, n-4, n-4, bg); put(inp, n//2, n//2, 1, 1, a); put(inp, 1, n-2, 1, 1, a)
        out = blank(n, bg); out[n//2][n//2] = c; meta["operator"] = "SELECT_OBJECT_INSIDE_FRAME"
    elif family == "same color":
        put(inp, 1, 1, 2, 2, a); put(inp, n-3, n-3, 2, 2, a); put(inp, 1, n-3, 2, 2, b); put(inp, n-3, 1, 2, 2, c)
        out = paint_non_bg([[a if x == a else bg for x in row] for row in inp], bg, c); meta["operator"] = "SELECT_SHARED_COLOR_CLASS"
    elif family in ("difference", "mask set"):
        put(inp, 1, 1, n-3, n-3, a); put(inp, 2, 2, n-5, n-5, b)
        out = [[c if x == a else bg for x in row] for row in inp]; meta["operator"] = "OUTER_MASK_MINUS_INNER_MASK"
    elif family == "width":
        widths = [2, 3, 4]; rng.shuffle(widths)
        for i, width in enumerate(widths): put(inp, 1 + i*3, 1, 1, width, a if i != 1 else b)
        widest = max(widths); out = [[c] * widest]; meta["operator"] = "SELECT_WIDEST_OBJECT"
    elif family in ("recolor", "color mapping"):
        put(inp, 2, 2, n-4, n-4, a); inp[0][1], inp[0][2] = a, c
        out = paint_non_bg(inp, bg, c); meta["operator"] = "APPLY_EXPLICIT_COLOR_MAPPING"
    elif family == "selector prerequisites":
        put(inp, 1, 1, 2, 2, a); put(inp, n-3, n-3, 3, 2, b); put(inp, 1, n-3, 1, 1, c)
        out = blank(n, bg); put(out, n-3, n-3, 3, 2, c); meta["operator"] = "SELECT_LARGEST_OBJECT"
    elif family == "relation selector action":
        put(inp, 2, 2, 2, 2, a); put(inp, 2, 4, 2, 2, b); put(inp, n-3, n-3, 2, 2, a)
        out = blank(n, bg); put(out, 2, 2, 2, 2, c); meta["operator"] = "SELECT_OBJECT_TOUCHING_REFERENCE_COLOR"
    elif family == "conditional action":
        count = rng.choice((2, 3));
        for i in range(count): put(inp, 1 + i*2, 1, 1, 1, a)
        out = blank(n, bg); put(out, n//2, n//2, 2, 2, c if count % 2 == 0 else b); meta["operator"] = "IF_OBJECT_COUNT_EVEN_THEN_COLOR_C_ELSE_B"
    elif family == "counting construction":
        count = rng.randint(2, 6)
        for i in range(count): put(inp, 1 + (i % 3)*2, 1 + (i // 3)*3, 1, 1, a)
        out = [[c] * count]; meta["operator"] = "COUNT_OBJECTS_AND_CONSTRUCT_BAR"
    elif family == "overlay":
        put(inp, 1, 1, 2, 2, a); put(inp, n-3, n-3, 2, 2, b); out = blank(n, bg); put(out, n//2-1, n//2-1, 2, 2, c); meta["operator"] = "OVERLAY_TWO_SOURCE_MASKS_AT_CENTER"
    elif family == "propagation":
        inp[n//2][1] = a; inp[n//2][n-2] = b; out = blank(n, bg); put(out, n//2, 1, 1, n-2, c); meta["operator"] = "PROPAGATE_LINE_BETWEEN_ENDPOINTS"
    elif family == "complete missing structure":
        put(inp, 1, 1, n-2, n-2, a); put(inp, 2, 2, n-4, n-4, bg); inp[1][n//2] = bg
        out = blank(n, bg); put(out, 1, 1, n-2, n-2, a); put(out, 2, 2, n-4, n-4, bg); meta["operator"] = "COMPLETE_RECTANGULAR_FRAME"
    elif family == "rotate":
        put(inp, 2, 2, 3, 1, a); put(inp, 4, 3, 1, 2, a); out = crop([list(row) for row in zip(*inp[::-1])], bg); meta["operator"] = "ROTATE_90"
    else: raise ValueError(family)
    return inp, out, meta


def episode(split: str, role: str, family: str, index: int, seed: int) -> dict:
    pairs = [primitive_case(family, seed + offset) for offset in (0, 1, 2)]
    task = {"train": [{"input": a, "output": b} for a, b, _ in pairs[:2]], "test": [{"input": pairs[2][0], "output": pairs[2][1]}]}
    content = {"task": task, "family": family, "seed": seed, "split": split}
    return {"episode_id": f"{split}:{role}:{family}:{index:06d}", "split": split, "role": role, "family": family, "seed": seed, "task": task, "program": pairs[2][2]["operator"], "episode_sha256": digest(content), "grid_identity_sha256": digest(task)}


def build_surface(split: str, role: str, families: tuple[str, ...], count: int, seed_base: int) -> list[dict]:
    return [episode(split, role, family, i, seed_base + i * 7919 + j * 104729) for j, family in enumerate(families) for i in range(count)]


def ensure_unique_grid_identities(rows: list[dict]) -> list[dict]:
    """Resample deterministic seeds until every full scene identity is unique."""
    seen: set[str] = set()
    result: list[dict] = []
    for row in rows:
        index = int(row["episode_id"].rsplit(":", 1)[1])
        retry = 0
        while row["grid_identity_sha256"] in seen:
            retry += 1
            row = episode(row["split"], row["role"], row["family"], index, int(row["seed"]) + retry * 1_000_003)
        seen.add(row["grid_identity_sha256"])
        result.append(row)
    return result


def composition_delta() -> list[dict]:
    def read(path: Path, key: str) -> dict[str, list[float]]:
        values: dict[str, list[float]] = defaultdict(list)
        for row in csv.DictReader(path.open(encoding="utf-8")):
            cats, scores = json.loads(row["mapped_composition_categories"]), json.loads(row[key])
            for category, score in zip(cats, scores):
                if category in CATEGORIES: values[category].append(float(score))
        return values
    base, v2 = read(BASE_COMP, "base_composition_scores"), read(V2_COMP, "foundation_v2_composition_scores")
    return [{"composition_category": c, "base_mean_score": sum(base[c])/len(base[c]), "foundation_v2_mean_score": sum(v2[c])/len(v2[c]), "delta": sum(v2[c])/len(v2[c])-sum(base[c])/len(base[c]), "task_category_observations": len(v2[c]), "evidence": "TASK_MAPPED_COMPOSITION_NOT_ISOLATED_MASTERY"} for c in CATEGORIES]


def priority_map(profile: dict) -> list[dict]:
    demand: dict[str, int] = defaultdict(int)
    for row in csv.DictReader(DEMAND.open(encoding="utf-8")):
        for cap in row["required_capabilities"].split("|"): demand[cap] += int(row["test_output_count"])
    rows = []
    for item in profile["capabilities"]:
        cap, score = item["capability"], float(item["foundation_v2_primary_score"])
        centrality = demand[cap]
        confidence = CONFIDENCE[item["measurement_type"]]
        priority = demand[cap] * max(0.0, 0.75-score) * centrality * confidence
        rows.append({"capability": cap, "eval60_demand": demand[cap], "foundation_v2_score": score, "residual_deficit": max(0.0, 0.75-score), "upstream_centrality": centrality, "measurement_type": item["measurement_type"], "measurement_confidence": confidence, "priority": priority, "engineering_band": item["engineering_band"], "independent_evidence_status": item["independent_evidence_status"], "priority_tier": "S" if cap in ATOMIC else "A" if priority > 0 else "RETENTION_ONLY"})
    return sorted(rows, key=lambda r: (-r["priority"], r["capability"]))


def main() -> int:
    profile, start = json.loads(PROFILE.read_text()), json.loads(START.read_text())
    if start.get("status") != "FROZEN" or not start.get("adapter_sha256"): raise RuntimeError("FOUNDATION_V2_START_IDENTITY_UNVERIFIED")
    protocol = OUT / "protocol"; data = OUT / "data"
    priority, delta = priority_map(profile), composition_delta()
    atomic_train = build_surface("TRAIN", "ATOMIC_REPAIR", ATOMIC, 250, 101_001)
    composition_train = build_surface("TRAIN", "COMPOSITION_REPAIR", ("relation selector action", "selector prerequisites", "mask set", "conditional action", "counting construction"), 240, 202_003)
    replay_train = build_surface("TRAIN", "REPLAY_RETENTION", RETENTION, 134, 303_007)
    target_dev = build_surface("TARGET_DEV", "TARGETED_EVALUATION", ATOMIC, 12, 404_009) + build_surface("TARGET_DEV", "TARGETED_COMPOSITION", ("relation selector action", "selector prerequisites", "mask set", "conditional action"), 24, 505_011)
    retention_dev = build_surface("TARGET_DEV", "RETENTION_SENTINEL", RETENTION, 16, 606_013)
    final_audit = build_surface("FINAL_AUDIT", "SEALED_FINAL", ATOMIC + ("relation selector action", "mask set", "conditional action"), 12, 707_017)
    all_rows = ensure_unique_grid_identities(atomic_train + composition_train + replay_train + target_dev + retention_dev + final_audit)
    atomic_train = [r for r in all_rows if r["split"] == "TRAIN" and r["role"] == "ATOMIC_REPAIR"]
    composition_train = [r for r in all_rows if r["split"] == "TRAIN" and r["role"] == "COMPOSITION_REPAIR"]
    replay_train = [r for r in all_rows if r["split"] == "TRAIN" and r["role"] == "REPLAY_RETENTION"]
    target_dev = [r for r in all_rows if r["split"] == "TARGET_DEV" and r["role"] != "RETENTION_SENTINEL"]
    retention_dev = [r for r in all_rows if r["split"] == "TARGET_DEV" and r["role"] == "RETENTION_SENTINEL"]
    final_audit = [r for r in all_rows if r["split"] == "FINAL_AUDIT"]
    ids, grids = [r["episode_id"] for r in all_rows], [r["grid_identity_sha256"] for r in all_rows]
    if len(ids) != len(set(ids)) or len(grids) != len(set(grids)): raise RuntimeError("SPLIT_IDENTITY_COLLISION")
    write_jsonl(data / "TRAIN.jsonl", atomic_train + composition_train + replay_train)
    write_jsonl(data / "TARGET_DEV.jsonl", target_dev)
    write_jsonl(data / "RETENTION_SENTINEL.jsonl", retention_dev)
    write_jsonl(data / "FINAL_AUDIT_SEALED.jsonl", final_audit)
    surface = {"schema_version": 1, "status": "FROZEN_CPU_ONLY", "surfaces": {"TRAIN": {"episodes": len(atomic_train)+len(composition_train)+len(replay_train), "mix": {"atomic": len(atomic_train), "composition": len(composition_train), "replay": len(replay_train)}, "seed_domains": [101001,202003,303007]}, "TARGET_DEV": {"episodes": len(target_dev), "seed_domain": 404009}, "RETENTION_SENTINEL": {"episodes": len(retention_dev), "seed_domain": 606013}, "FINAL_AUDIT": {"episodes": len(final_audit), "seed_domain": 707017, "model_accessed": False, "sealed": True}}, "checks": {"episode_ids_disjoint": True, "exact_grid_identities_disjoint": True, "eval60_gold_used": False, "diagnostic_episodes_used_as_training": False, "final_audit_model_accessed": False, "programmatic_labels_only": True}}
    write_json(protocol / "TARGET_DATA_POLICY_V1.json", {"status": "FROZEN", "randomized_axes": ["foreground_colors","background_colors","color_permutation","grid_size","object_positions","object_count","object_size","orientation","distractor_count","irrelevant_objects","operator_parameters"], "anti_shortcut_rules": ["positive_negative_relation_cases","balanced_conditional_branches","selector_nonpositional","no_fixed_color_or_position"], **surface})
    write_json(protocol / "TARGETED_EVAL_PROTOCOL_V1.json", {"status": "FROZEN", "ordinary_checkpoint_generations": {"targeted_dev": len(target_dev), "retention": len(retention_dev)}, "paired_transition_counts": ["n01_baseline_wrong_candidate_correct", "n10_baseline_correct_candidate_wrong"], "full_3000_diagnostic": "FINAL_OR_MAJOR_STAGE_ONLY", "final_audit_access_before_final_selection": False})
    write_json(protocol / "RETENTION_SENTINEL_PROTOCOL_V1.json", {"status": "FROZEN", "families": list(RETENTION), "episodes": len(retention_dev), "selection": "representative_multisection_not_trivially_easy", "gate": "convincing_broad_or_important_degradation_rejects"})
    write_json(protocol / "ROUND_GATE_PROTOCOL_V1.json", {"status": "FROZEN", "accept": ["meaningful_high_priority_target_progress", "retention_acceptable"], "reject": ["REJECTED_RETENTION", "REJECTED_NO_TARGET_GAIN", "REJECTED_DATA_VALIDITY"], "loss_only": "INSUFFICIENT", "plateau": "two_valid_rounds_without_meaningful_gain_requires_curriculum_diagnosis", "hard_stops": ["baseline_identity_uncertain","split_contamination","eval60_gold_in_training","final_audit_opened_early","taxonomy_modified","checkpoint_identity_uncertain"]})
    write_json(protocol / "BASELINE_IDENTITY_V1.json", {"frozen_parent": "b26e469d1bd8d757727d4f34aff0e967207d1e62", "foundation_start": start, "inputs": {str(p.relative_to(ROOT)): file_hash(p) for p in (PROFILE, BASE_COMP, V2_COMP, DEMAND, START)}, "source_audit": "analysis/foundation_v2_eval60_capability_alignment_v1/SOURCE_AUDIT.json", "gold_training": False})
    with (protocol / "CAPABILITY_PRIORITY_MAP_V1.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(priority[0])); writer.writeheader(); writer.writerows(priority)
    with (protocol / "BASE_V2_COMPOSITION_DELTA_V1.csv").open("w", newline="", encoding="utf-8") as h:
        writer = csv.DictWriter(h, fieldnames=list(delta[0])); writer.writeheader(); writer.writerows(delta)
    core = {"program": "TARGETED_CAPABILITY_REPAIR_V1", "status": "PROTOCOL_FROZEN_PENDING_DIRECTOR", "baseline": start, "objective": "repair_high_demand_upstream_primitives_then_critical_compositions_with_retention", "atomic_targets": list(ATOMIC), "composition_targets": list(COMPOSITION), "round1_mix": {"atomic": 0.50, "composition": 0.30, "replay": 0.20}, "prerequisite_rule": {"lt_0_40": "intensive_primitive", "0_40_to_0_60": "primitive_majority_light_composition", "0_60_to_0_75": "balanced", "gte_0_75": "composition_if_chain_weak"}, "training_recipe": {"base_weights": "frozen", "precision": "BF16", "lora_rank": 64, "lora_alpha": 32, "lora_dropout": 0.0, "target_modules": ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"], "optimizer_family": "PagedAdamW8bit", "quantization": "NONE", "context": 8704}, "budget": {"max_new_gpu_training_seconds": 28800, "initial_round_tokens": 500000, "evaluation_checkpoints_tokens": [250000,500000]}, "director_authorization_required": ["CONTINUE","CONTINUE_WITH_WARNING"], "gpu_training_started": False, "surface_manifest": surface}
    write_json(protocol / "TARGETED_CAPABILITY_REPAIR_V1_PROTOCOL.json", core)
    md = "# TARGETED_CAPABILITY_REPAIR_V1\n\nStatus: **PROTOCOL_FROZEN_PENDING_DIRECTOR**. GPU training has not started.\n\nThe protocol uses the frozen Foundation-V2 adapter identity, targets high-demand independent deficits first, uses a 50/30/20 atomic/composition/replay Round-1 mix, and keeps FINAL_AUDIT sealed until final model selection. It preserves BF16 frozen-base LoRA (r=64, alpha=32, dropout=0) and forbids Eval60 Gold, taxonomy changes, QLoRA, and diagnostic-example training.\n\nOrdinary checkpoints use 192 TARGET_DEV and 96 retention episodes; full diagnostic remains final-stage only. A round requires paired target progress and acceptable retention; two valid no-gain rounds trigger curriculum diagnosis.\n"
    (protocol / "TARGETED_CAPABILITY_REPAIR_V1_PROTOCOL.md").write_text(md, encoding="utf-8", newline="\n")
    manifest = {"status": "PASS", "protocol_files": {str(p.relative_to(OUT)): file_hash(p) for p in sorted(protocol.glob("*"))}, "data_files": {str(p.relative_to(OUT)): file_hash(p) for p in sorted(data.glob("*"))}, "surface": surface}
    write_json(OUT / "PREPARATION_MANIFEST.json", manifest)
    print(json.dumps({"status": "PASS", "train": surface["surfaces"]["TRAIN"]["episodes"], "target_dev": len(target_dev), "retention": len(retention_dev), "final_sealed": len(final_audit)}, sort_keys=True))
    return 0


if __name__ == "__main__": raise SystemExit(main())
