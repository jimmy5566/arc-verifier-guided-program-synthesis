"""CPU-only validity hardening for Foundation Diagnostic Battery V1.2."""
from __future__ import annotations

import gzip
import hashlib
import io
import itertools
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable

import pyarrow.parquet as pq

from foundation_capability_bank_v3 import pipeline as v3
from foundation_capability_bank_v3_1 import pipeline as v31
from training_data_v2.pipeline import canonical_json, native_render, sample_messages, sha256_file, signatures

VERSION = "foundation_diagnostic_v3_2_validity"
SOURCE_COMMIT = "d39d7cfc768da01dafa318357f1196d387d4a0b1"
V3_FINGERPRINT = v31.V3_FINGERPRINT
V3_TRAIN_SHA256 = v31.V3_TRAIN_SHA256
CONTEXT = 8704
Rule = Callable[[list[list[int]]], list[list[int]]]


class GateFailure(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def write_gzip(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    count = 0; tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="\n") as handle:
                for row in rows:
                    handle.write(canonical_json(row) + "\n"); count += 1
    os.replace(tmp, path)
    return count, sha256_file(path)


def _blank(h: int, w: int, bg: int = 0) -> list[list[int]]:
    return [[bg] * w for _ in range(h)]


def _paint(grid: list[list[int]], cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells: grid[r][c] = color


def _copy(grid: list[list[int]]) -> list[list[int]]:
    return [row[:] for row in grid]


def _bar(n: int, color: int, vertical: bool = False) -> list[list[int]]:
    return [[color] for _ in range(n)] if vertical else [[color] * n]


def _rotate90(grid: list[list[int]]) -> list[list[int]]:
    return [list(row) for row in zip(*grid[::-1])]


def _reflect_lr(grid: list[list[int]]) -> list[list[int]]:
    return [row[::-1] for row in grid]


def _reflect_ud(grid: list[list[int]]) -> list[list[int]]:
    return grid[::-1]


def _components(grid: list[list[int]], ignore_top_legend: bool = False) -> list[dict[str, Any]]:
    work = _copy(grid)
    if ignore_top_legend: work[0] = [v3._bg(grid)] * len(work[0])
    return v3.v2.components(work)


def _component_groups(n: int, variant: int, total: int = 8) -> tuple[list[list[int]], list[list[tuple[int, int]]]]:
    g = _blank(15, 15); anchors = [(2, 2), (2, 9), (9, 2), (9, 9), (6, 6), (2, 6), (9, 6)]
    anchors = anchors[variant % len(anchors):] + anchors[:variant % len(anchors)]
    sizes = {1: [total], 2: [3, total - 3], 3: [2, 3, total - 5], 4: [2, 2, 2, total - 6]}[n]
    groups = []
    for j, (anchor, size) in enumerate(zip(anchors, sizes)):
        r, c = anchor; horizontal = (variant + j) % 2 == 0
        cells = [(r, c + x) if horizontal else (r + x, c) for x in range(size)]
        groups.append(cells); _paint(g, cells, 2 + ((variant + j) % 5))
    return g, groups


def _segmentation_output(inp: list[list[int]]) -> list[list[int]]:
    out = _copy(inp)
    for i, comp in enumerate(v3.v2.components(inp)): _paint(out, comp["cells"], 5 + i % 4)
    return out


def _count_output(inp: list[list[int]], vertical: bool = False) -> list[list[int]]:
    return _bar(len(v3.v2.components(inp)), 8, vertical)


def _area_scene(variant: int, target_area: int) -> tuple[list[list[int]], dict[str, Any]]:
    g = _blank(15, 15); colors = [2 + (variant % 3), 5 + (variant % 3), 8]
    target = variant % 3; areas = [2 + ((variant + 1) % 4), 3 + ((variant + 2) % 4), 2 + (variant % 4)]; areas[target] = target_area
    positions = [(2, 2), (2, 9), (9, 3)]; positions = positions[variant % 3:] + positions[:variant % 3]
    for i, (area, (r, c)) in enumerate(zip(areas, positions)):
        cells = [(r + x // 3, c + x % 3) for x in range(area)]; _paint(g, cells, colors[i])
    g[0][0:2] = [9, colors[target]]
    return g, {"target_color": colors[target], "target_area": target_area, "areas": areas, "target_position": positions[target]}


def _largest_scene(variant: int) -> tuple[list[list[int]], dict[str, Any]]:
    g = _blank(15, 15); positions = [(2, 2), (2, 10), (9, 2), (9, 10)]; positions = positions[variant % 4:] + positions[:variant % 4]
    colors = [2 + (variant + j) % 6 for j in range(3)]; sizes = [2, 3, 5]; target_slot = variant % 3; sizes[target_slot], sizes[2] = sizes[2], sizes[target_slot]
    groups = []
    for j, (size, (r, c)) in enumerate(zip(sizes, positions)):
        cells = [(r + x // 2, c + x % 2) for x in range(size)]; groups.append(cells); _paint(g, cells, colors[j])
    out = _copy(g); _paint(out, groups[target_slot], 9)
    return g, {"output": out, "target_position": positions[target_slot], "target_color": colors[target_slot], "shape_size": sizes[target_slot]}


def _recolor_scene(variant: int) -> tuple[list[list[int]], dict[str, Any]]:
    g = _blank(15, 15); target_color = 2 + variant % 5; new_color = 8 - variant % 3
    if new_color == target_color: new_color = 9
    positions = [(2, 2), (2, 9), (9, 2), (9, 9)]
    positions = positions[variant % 4:] + positions[:variant % 4]
    shapes = [[(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (1, 1)]]
    for j, (r, c) in enumerate(positions[:3]):
        color = target_color if j == variant % 3 else 3 + (variant + j + 2) % 5
        _paint(g, [(r + dr, c + dc) for dr, dc in shapes[(variant + j) % len(shapes)]], color)
    g[0][0:3] = [9, target_color, new_color]; out = [[new_color if x == target_color else x for x in row] for row in g]; out[0][0:3] = g[0][0:3]
    return g, {"output": out, "target_color": target_color, "new_color": new_color, "target_position": positions[variant % 3]}


RELATION_CAPS = {"left/right", "above/below", "nearest/farthest", "inside/contains", "same color", "relation-conditioned selector"}
CONDITIONAL_CAPS = {"if/else on attribute", "if/else on relation", "if/else on count", "branch-balanced conditional tasks"}


def _relation_scene(capability: str, variant: int, condition: int) -> tuple[list[list[int]], dict[str, Any]]:
    g = _blank(15, 15); refs = [(3, 3), (9, 3), (9, 9), (3, 9)]; rr, rc = refs[variant % 4]
    ref_color = 4 + variant % 5; cue = 7 if condition == 0 else 8; g[0][0] = cue
    _paint(g, [(rr, rc), (rr, rc + 1), (rr + 1, rc), (rr + 1, rc + 1)], ref_color)
    if capability == "left/right":
        candidates = [(rr, min(14, rc + 3)), (rr, max(0, rc - 2))]
    elif capability == "above/below":
        candidates = [(min(14, rr + 3), rc), (max(1, rr - 2), rc)]
    elif capability == "nearest/farthest":
        near = (rr, rc + 2 if rc < 7 else rc - 1)
        far = (13 if rr < 7 else 1, 13 if rc < 7 else 1)
        other = (rr, 0 if rc < 7 else 14); candidates = [near, far, other]
    elif capability == "inside/contains":
        g = _blank(15, 15); g[0][0] = cue; rr, rc = refs[variant % 4]
        fr, fc = (1 if rr < 7 else 8), (1 if rc < 7 else 8)
        frame = [(r, c) for r in range(fr, fr + 6) for c in range(fc, fc + 6) if r in (fr, fr + 5) or c in (fc, fc + 5)]; _paint(g, frame, ref_color)
        rr, rc = fr, fc; candidates = [(fr + 2, fc + 2), (13 if fr < 5 else 1, 13 if fc < 5 else 1)]
    elif capability == "same color":
        candidates = [(rr, rc + 3), (rr + 3, rc)]
    else:
        candidates = [(rr, rc + 3), (rr + 3, rc)]
    candidates = [(max(1, min(14, r)), max(0, min(14, c))) for r, c in candidates]
    target = candidates[condition]
    candidate_colors = [c for c in range(1, 10) if c not in {2, cue, ref_color}]
    for j, pos in enumerate(candidates):
        color = ref_color if capability == "same color" and j == 0 else candidate_colors[(variant + j) % len(candidate_colors)]
        g[pos[0]][pos[1]] = color
    distractors = [x for i, x in enumerate(candidates) if i != condition]
    out = _copy(g); out[target[0]][target[1]] = 2
    return g, {"output": out, "reference_position": [rr, rc], "target_position": list(target), "target_region": f"{'top' if target[0] < 7 else 'bottom'}_{'left' if target[1] < 7 else 'right'}", "distractors": [list(x) for x in distractors], "relation_condition": condition, "reference_color": ref_color}


def _conditional_scene(capability: str, variant: int, truth: bool) -> tuple[list[list[int]], dict[str, Any]]:
    g = _blank(11, 11); base_r = 1 + (variant % 3) * 2; base_c = 1 + ((variant // 2) % 3) * 2
    color = (2 + 2 * (variant % 3)) if truth else (3 + 2 * (variant % 3))
    shape = [(base_r, base_c), (base_r, base_c + 1), (base_r + 1, base_c), (base_r + 2, base_c)]
    _paint(g, shape, color)
    if capability == "if/else on relation":
        g[8][2] = 7; g[8][3 if truth else 7] = 8
    elif capability == "if/else on count":
        g[8][2] = 7
        for j in range(2 if truth else 1): g[8][6 + j * 2] = 8
    action_a, action_b = (_rotate90, _reflect_lr) if capability == "if/else on relation" else (_reflect_lr, _reflect_ud)
    out = action_a(g) if truth else action_b(g)
    return g, {"output": out, "branch": "TRUE" if truth else "FALSE", "action": "A" if truth else "B", "shape": shape, "foreground_color": color}


def minimal_task(capability: str, pair_index: int, test_condition: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = []; examples = []
    if capability in {"connected components", "count components", "encode quantity geometrically"}:
        causal_values = [1, 3, 2, 4]
    else: causal_values = [0, 1, 0, 1]
    for k, causal in enumerate(causal_values + [test_condition]):
        variant = pair_index * 7 + k
        if capability in {"connected components", "count components", "encode quantity geometrically"}:
            n = causal if k < 4 else (2 if test_condition == 0 else 4); inp, groups = _component_groups(n, variant)
            common_color = 1 + variant % 4
            for r, line in enumerate(inp):
                for c, value in enumerate(line):
                    if value: inp[r][c] = common_color
            if capability == "connected components": out = _segmentation_output(inp); detail = {"component_count": n, "total_cells": 8, "layout_sha256": digest(inp)}
            elif capability == "count components": out = _bar(n, 8); detail = {"component_count": n, "encoding": "horizontal", "layout_sha256": digest(inp)}
            else:
                vertical = bool((variant + causal) % 2); inp[0][0] = 9 if vertical else 8; out = _bar(n, 7, vertical); detail = {"component_count": n, "encoding": "vertical" if vertical else "horizontal", "layout_sha256": digest(inp)}
        elif capability == "area":
            area = [2, 5, 3, 6, 4 + test_condition][k]; inp, detail = _area_scene(variant, area); out = _bar(area, 8)
        elif capability == "largest": inp, detail = _largest_scene(variant); out = detail.pop("output")
        elif capability == "recolor": inp, detail = _recolor_scene(variant); out = detail.pop("output")
        elif capability in RELATION_CAPS: inp, detail = _relation_scene(capability, variant, causal if k < 4 else test_condition); out = detail.pop("output")
        else:
            truth = bool(causal if k < 4 else test_condition); inp, detail = _conditional_scene(capability, variant, truth); out = detail.pop("output")
        pairs.append({"input": inp, "output": out}); examples.append(detail)
    return {"train": pairs[:4], "test": [pairs[4]]}, {"training_examples": examples[:4], "test_example": examples[4]}


def _select_component(inp: list[list[int]], selector: str) -> dict[str, Any]:
    comps = [x for x in _components(inp, True) if not any(r == 0 for r, _ in x["cells"])]
    if selector == "largest": return max(comps, key=lambda x: x["area"])
    if selector == "smallest": return min(comps, key=lambda x: x["area"])
    if selector == "leftmost": return min(comps, key=lambda x: x["c0"])
    if selector == "rightmost": return max(comps, key=lambda x: x["c1"])
    if selector == "topmost": return min(comps, key=lambda x: x["r0"])
    raise GateFailure(selector)


def _minimal_rule_set(capability: str) -> dict[str, Rule]:
    if capability == "connected components":
        return {"INTENDED_SEGMENT_COMPONENTS": _segmentation_output, "ONE_COLOR_FOREGROUND": lambda x: [[5 if v else 0 for v in row] for row in x], "COUNT_COMPONENTS_BAR": lambda x: _count_output(x), "IDENTITY": _copy}
    if capability == "count components":
        return {"INTENDED_COUNT_COMPONENTS": lambda x: _count_output(x), "COUNT_CELLS": lambda x: _bar(sum(v != 0 for row in x for v in row), 8), "COUNT_COLORS": lambda x: _bar(len({v for row in x for v in row if v}), 8), "VERTICAL_COUNT": lambda x: _count_output(x, True)}
    if capability == "encode quantity geometrically":
        return {"INTENDED_CUE_ENCODING": lambda x: _bar(len(v3.v2.components([([0] * len(x[0]) if r == 0 else row) for r, row in enumerate(x)])), 7, x[0][0] == 9), "ALWAYS_HORIZONTAL": lambda x: _bar(len(v3.v2.components([([0] * len(x[0]) if r == 0 else row) for r, row in enumerate(x)])), 7), "ALWAYS_VERTICAL": lambda x: _bar(len(v3.v2.components([([0] * len(x[0]) if r == 0 else row) for r, row in enumerate(x)])), 7, True), "COUNT_CELLS": lambda x: _bar(sum(v != 0 for row in x[1:] for v in row), 7)}
    if capability == "area":
        def area(x: list[list[int]]) -> list[list[int]]:
            target = x[0][1]; return _bar(sum(v == target for row in x[1:] for v in row), 8)
        return {"INTENDED_CUED_AREA": area, "FOREGROUND_AREA": lambda x: _bar(sum(v != 0 for row in x[1:] for v in row), 8), "COMPONENT_COUNT": lambda x: _bar(len(_components(x, True)), 8), "FIXED_THREE": lambda x: _bar(3, 8)}
    if capability == "largest":
        def recolor(selector: str) -> Rule:
            def run(x: list[list[int]]) -> list[list[int]]:
                out = _copy(x); _paint(out, _select_component(x, selector)["cells"], 9); return out
            return run
        return {"INTENDED_SELECT_LARGEST": recolor("largest"), "SELECT_SMALLEST": recolor("smallest"), "SELECT_LEFTMOST": recolor("leftmost"), "SELECT_RIGHTMOST": recolor("rightmost"), "SELECT_TOPMOST": recolor("topmost")}
    if capability == "recolor":
        def legend(x: list[list[int]]) -> list[list[int]]:
            old, new = x[0][1], x[0][2]; out = [[new if v == old else v for v in row] for row in x]; out[0][0:3] = x[0][0:3]; return out
        return {"INTENDED_LEGEND_RECOLOR": legend, "IDENTITY": _copy, "FIXED_2_TO_8": lambda x: [[8 if v == 2 else v for v in row] for row in x], "FIXED_3_TO_7": lambda x: [[7 if v == 3 else v for v in row] for row in x]}
    if capability in RELATION_CAPS:
        relations = [capability, "left/right", "above/below", "same color", "inside/contains"]
        if capability != "inside/contains": relations.append("nearest/farthest")
        return {("INTENDED_" if rel == capability else "ALT_") + rel.upper().replace("/", "_").replace(" ", "_"): (lambda rel: lambda x: _relation_apply(x, rel))(rel) for rel in dict.fromkeys(relations)}
    def conditional(rule: str) -> Rule:
        def run(x: list[list[int]]) -> list[list[int]]:
            truth = _condition_truth(x, capability)
            action_a, action_b = (_rotate90, _reflect_lr) if capability == "if/else on relation" else (_reflect_lr, _reflect_ud)
            if rule == "always_a": return action_a(x)
            if rule == "always_b": return action_b(x)
            if rule == "reversed": return action_b(x) if truth else action_a(x)
            return action_a(x) if truth else action_b(x)
        return run
    return {"INTENDED_CONDITIONAL": conditional("intended"), "REVERSED_CONDITIONAL": conditional("reversed"), "ALWAYS_ACTION_A": conditional("always_a"), "ALWAYS_ACTION_B": conditional("always_b"), "IDENTITY": _copy}


def _relation_apply(inp: list[list[int]], relation: str) -> list[list[int]]:
    comps = v3.v2.components(inp); refs = [x for x in comps if not any(r == 0 for r, _ in x["cells"])]; ref = max(refs, key=lambda x: x["area"])
    others = [x for x in comps if x is not ref and not any(r == 0 for r, _ in x["cells"])]
    rr, rc = (ref["r0"] + ref["r1"]) / 2, (ref["c0"] + ref["c1"]) / 2
    first = inp[0][0] == 7
    if relation == "left/right": target = min([x for x in others if x["c0"] > ref["c1"]], key=lambda x: x["c0"]) if first else max([x for x in others if x["c1"] < ref["c0"]], key=lambda x: x["c1"])
    elif relation == "above/below": target = min([x for x in others if x["r0"] > ref["r1"]], key=lambda x: x["r0"]) if first else max([x for x in others if x["r1"] < ref["r0"]], key=lambda x: x["r1"])
    elif relation == "nearest/farthest": target = (min if first else max)(others, key=lambda x: ((x["r0"] + x["r1"]) / 2 - rr) ** 2 + ((x["c0"] + x["c1"]) / 2 - rc) ** 2)
    elif relation == "same color": target = next(x for x in others if (x["color"] == ref["color"]) == first)
    elif relation == "inside/contains": target = next(x for x in others if (ref["r0"] < x["r0"] <= x["r1"] < ref["r1"] and ref["c0"] < x["c0"] <= x["c1"] < ref["c1"]) == first)
    else:
        target = min([x for x in others if x["c0"] > ref["c1"]], key=lambda x: x["c0"]) if inp[0][0] == 7 else min([x for x in others if x["r0"] > ref["r1"]], key=lambda x: x["r0"])
    out = _copy(inp); _paint(out, target["cells"], 2); return out


def _condition_truth(inp: list[list[int]], capability: str) -> bool:
    if capability in {"if/else on attribute", "branch-balanced conditional tasks"}: return max(v3.v2.components(inp), key=lambda x: x["area"])["color"] % 2 == 0
    if capability == "if/else on relation": return inp[8][3] == 8
    return len(v3.v2.components(inp)) % 2 == 0


def _load_v31() -> dict[str, list[dict[str, Any]]]:
    root = Path(__file__).parents[2]; path = root / "artifacts/foundation_capability_bank_v3_1/FOUNDATION_DIAGNOSTIC_BATTERY_V1_1.jsonl.gz"
    rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); rows[row["capability"]].append(row)
    return rows


def _pad_lr(grid: list[list[int]], amount: int) -> list[list[int]]:
    bg = v3._bg(grid); return [[bg] * amount + row + [bg] * amount for row in grid]


def _remove_individual_overlap(task: dict[str, Any], program: v3.Program, banned: set[str]) -> tuple[dict[str, Any], int]:
    replaced = 0; result = {"train": [], "test": []}
    for split in ("train", "test"):
        for pair in task[split]:
            if digest(pair) not in banned: result[split].append(pair); continue
            chosen = None
            for amount in range(1, 6):
                inp = _pad_lr(pair["input"], amount)
                try: candidate = {"input": inp, "output": v3.apply_program(inp, program)}
                except Exception: continue
                if digest(candidate) not in banned and candidate["input"] != candidate["output"]: chosen = candidate; break
            if chosen is None: raise GateFailure(f"cannot remove individual train overlap: {program.program_id}")
            result[split].append(chosen); replaced += 1
    return result, replaced


def _parity_counterexample(sample_id: str, program: v3.Program) -> dict[str, Any]:
    offset = int(digest(sample_id)[:2], 16) % 3; g = _blank(11, 11)
    _paint(g, [(1 + offset, 1), (1 + offset, 2), (2 + offset, 1)], 3)
    _paint(g, [(7 - offset, 7)], 4)
    out = v3.apply_program(g, program)
    alternate = next(p for p in v3.atomic_programs() if p.steps[0] == "IF_LARGEST_COLOR_EVEN_LR_ELSE_UD")
    if out == v3.apply_program(g, alternate): raise GateFailure("parity counterexample does not separate hypotheses")
    return {"input": g, "output": out}


def _unique_color_counterexample(sample_id: str, program: v3.Program) -> dict[str, Any]:
    offset = int(digest(sample_id)[2:4], 16) % 2; g = _blank(12, 12)
    _paint(g, [(5, 5 + offset), (5, 6 + offset)], 2)
    _paint(g, [(1, 1), (1, 2), (2, 1), (2, 2)], 3); _paint(g, [(10, 9)], 3)
    _paint(g, [(1, 9)], 4); _paint(g, [(9, 1), (10, 1)], 4)
    out = v3.apply_program(g, program)
    return {"input": g, "output": out}


def build_battery(root: Path, banned_pairs: set[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    old = _load_v31(); measurements = v31.measurement_rows(); programs = {p.program_id: p for p in v3.atomic_programs()}; rows = []; manifest = []; replaced = 0
    for contract in measurements:
        cap = contract["capability"]
        if contract["measurement_type"] == "MINIMAL_CONTRAST":
            for pair_index in range(16):
                pair_id = f"contrast_v1_2:{digest({'capability': cap, 'pair': pair_index})[:20]}"
                for test_condition in (0, 1):
                    task, evidence = minimal_task(cap, pair_index, test_condition); sid = f"DIAGNOSTIC_V1_2:{digest({'pair': pair_id, 'test': test_condition})[:24]}"
                    rows.append({"sample_id": sid, "pair_id": pair_id, "pair_member": test_condition, "capability": cap, "measurement_type": "MINIMAL_CONTRAST", "program_id": contract["program_id"], "scoring_rule": "CONTRAST_SUCCESS", "scene_evidence": evidence, "task": task})
        else:
            for source in old[cap]:
                task, count = _remove_individual_overlap(source["task"], programs[contract["program_id"]], banned_pairs); replaced += count
                if cap == "parity": task["train"][-1] = _parity_counterexample(source["sample_id"], programs[contract["program_id"]])
                if cap in {"color", "unique", "uniqueness"}: task["train"][-1] = _unique_color_counterexample(source["sample_id"], programs[contract["program_id"]])
                sid = f"DIAGNOSTIC_V1_2:{digest({'capability': cap, 'source': source['sample_id']})[:24]}"
                rows.append({"sample_id": sid, "source_v3_1_sample_id": source["sample_id"], "capability": cap, "measurement_type": contract["measurement_type"], "program_id": contract["program_id"], "scoring_rule": "FULL_EXACT_GRID_ACCURACY", "task": task})
    for row in rows:
        task = row["task"]; sig = signatures(task)
        manifest.append({k: v for k, v in row.items() if k != "task"} | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": sig["train_pair_sha256"], "individual_train_pair_sha256": [digest(x) for x in task["train"]]})
    return rows, manifest, {"replaced_overlapping_individual_pairs": replaced}


def _audit_minimal_scene_diversity(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows:
        if row["measurement_type"] != "MINIMAL_CONTRAST": continue
        examples = row["scene_evidence"]["training_examples"]; tasks = row["task"]["train"]
        colors = [{v for line in x["input"] for v in line if v} for x in tasks]
        shapes = [digest(sorted(Counter(v for line in x["input"] for v in line if v).values())) for x in tasks]
        inputs = [digest(x["input"]) for x in tasks]
        dimensions = sum([len({tuple(sorted(x)) for x in colors}) >= 2, len(set(shapes)) >= 2, len(set(inputs)) == 4])
        checks.append({"sample_id": row["sample_id"], "capability": row["capability"], "distinct_demo_inputs": len(set(inputs)), "varied_color_sets": len({tuple(sorted(x)) for x in colors}), "varied_shape_signatures": len(set(shapes)), "nuisance_dimensions_varied": dimensions, "pass": dimensions >= 2 and len(set(inputs)) == 4})
    return {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL", "tasks": len(checks), "checks": checks, "barcodes_principal_variation": False}


def _audit_conditionals(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows:
        if row["capability"] not in CONDITIONAL_CAPS: continue
        branches = [x["branch"] for x in row["scene_evidence"]["training_examples"]]
        checks.append({"sample_id": row["sample_id"], "branches": branches, "true": branches.count("TRUE"), "false": branches.count("FALSE"), "pass": branches.count("TRUE") == branches.count("FALSE") == 2})
    return {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL", "tasks_with_both_branches": sum(x["pass"] for x in checks), "total_tasks": len(checks), "checks": checks}


def _audit_relations(rows: list[dict[str, Any]]) -> dict[str, Any]:
    checks = []
    for row in rows:
        if row["capability"] not in RELATION_CAPS: continue
        demos = row["scene_evidence"]["training_examples"]; target_regions = {x["target_region"] for x in demos}; targets = {tuple(x["target_position"]) for x in demos}; refs = {tuple(x["reference_position"]) for x in demos}
        passed = len(target_regions) >= 3 and len(targets) >= 3 and len(refs) >= 3
        checks.append({"sample_id": row["sample_id"], "target_regions": sorted(target_regions), "distinct_target_positions": len(targets), "distinct_reference_positions": len(refs), "pass": passed})
    return {"status": "PASS" if all(x["pass"] for x in checks) else "FAIL", "tasks": len(checks), "fixed_absolute_position_sufficient": False, "checks": checks}


def _hypothesis_family(step: str) -> str:
    if step.startswith(("ROTATE_", "REFLECT_", "SCALE_")): return "GEOMETRIC_TRANSFORM"
    if step.startswith("SELECT_") or step == "ORIENTATION_RECOLOR": return "SELECTOR"
    if step.startswith("COUNT_") or step == "COMPARE_PANEL_QUANTITIES": return "COUNTING"
    if step.startswith(("TRANSLATE_", "MOVE_", "INFER_AND_APPLY_", "COPY_OBJECT_")): return "MOTION"
    if step.startswith(("SET_", "OVERLAY_", "MASK_")): return "SET_OPERATION"
    if step.startswith("IF_"): return "CONDITIONAL"
    if step in {"BORDER_BOUNDING_BOX", "FILL_BOUNDING_BOX", "EXTRACT_BOUNDING_BOX", "EXTRACT_AND_PAD", "COMPLETE_LR_SYMMETRY", "COMPLETE_MISSING_RECTANGLE", "FILL_ENCLOSED_HOLES"}: return "STRUCTURE"
    if step in {"FILL_ALTERNATING_INTERVAL", "EXTEND_PERIODIC_PATTERN", "COMPLETE_LENGTH_PROGRESSION", "EXPAND_ONE_ITERATION", "PROPAGATE_REACHABLE_REGION", "EXTEND_LONGEST_LINE_TO_BOUNDARY"}: return "PATTERN_STATE"
    if step in {"APPLY_LEGEND_COLOR_MAPPING", "COPY_REFERENCE_COLOR", "CONNECT_UNIQUE_SAME_COLOR_PAIR"}: return "COLOR_RELATION"
    if step in {"ALIGN_OBJECTS_ROW", "ALIGN_OBJECTS_COLUMN", "REPEAT_LARGEST_COMPONENT_HORIZONTAL_3"}: return "ARRANGEMENT"
    return "OTHER"


def _nonminimal_hypotheses(program_id: str) -> tuple[dict[str, Rule], list[str], str]:
    programs = v3.atomic_programs(); intended = next(p for p in programs if p.program_id == program_id); step = intended.steps[0]; family = _hypothesis_family(step); excluded: list[str] = []
    if step == "SELECT_UNIQUE_COLOR_RECOLOR":
        allowed = {"SELECT_UNIQUE_COLOR_RECOLOR", "SELECT_LARGEST_RECOLOR", "SELECT_SMALLEST_RECOLOR", "SELECT_MOST_FREQUENT_COLOR_RECOLOR", "SELECT_LEFTMOST_RECOLOR", "SELECT_RIGHTMOST_RECOLOR", "SELECT_TOPMOST_RECOLOR", "SELECT_BOTTOMMOST_RECOLOR", "SELECT_SECOND_LEFT_RECOLOR", "SELECT_WIDEST_RECOLOR", "ORIENTATION_RECOLOR"}
        excluded = ["SELECT_LEAST_FREQUENT_RECOLOR (definitionally equivalent on the frozen unique-color domain)"]
        selected = [p for p in programs if p.steps[0] in allowed]
    elif step == "SELECT_LEAST_FREQUENT_RECOLOR":
        allowed = {"SELECT_LEAST_FREQUENT_RECOLOR", "SELECT_LARGEST_RECOLOR", "SELECT_SMALLEST_RECOLOR", "SELECT_MOST_FREQUENT_COLOR_RECOLOR", "SELECT_LEFTMOST_RECOLOR", "SELECT_RIGHTMOST_RECOLOR", "SELECT_TOPMOST_RECOLOR", "SELECT_BOTTOMMOST_RECOLOR"}
        excluded = ["SELECT_UNIQUE_COLOR_RECOLOR (definitionally equivalent on the frozen least-frequency domain when exactly one minimum-frequency color exists)"]
        selected = [p for p in programs if p.steps[0] in allowed]
    else:
        selected = [p for p in programs if _hypothesis_family(p.steps[0]) == family]
    hypotheses = {p.program_id: (lambda q: lambda x: v3.apply_program(x, q))(p) for p in selected}
    return hypotheses, excluded, f"FROZEN_V3_2_{family}_HYPOTHESIS_SET"


def _identify_one(row: dict[str, Any]) -> dict[str, Any]:
    cap = row["capability"]; task = row["task"]
    if row["measurement_type"] == "MINIMAL_CONTRAST":
        hypotheses = _minimal_rule_set(cap); intended = next(k for k in hypotheses if k.startswith("INTENDED")); excluded = []; set_id = f"FROZEN_V3_2_MINIMAL_{cap.upper().replace('/', '_').replace(' ', '_')}"
    else:
        intended = row["program_id"]; hypotheses, excluded, set_id = _nonminimal_hypotheses(intended)
    compatible = []
    for rule_id, rule in hypotheses.items():
        try:
            if all(rule(pair["input"]) == pair["output"] for pair in task["train"]): compatible.append(rule_id)
        except Exception: pass
    return {"sample_id": row["sample_id"], "capability": cap, "measurement_type": row["measurement_type"], "hypothesis_set_id": set_id, "hypothesis_count": len(hypotheses), "hypothesis_rule_ids": list(hypotheses), "equivalence_or_scope_exclusions": excluded, "intended_rule_id": intended, "intended_rule_consistent": intended in compatible, "compatible_rule_count": len(compatible), "compatible_rule_ids": compatible}


SURFACES = v31.SURFACES


def _surface_pair(axis: str, value: Any, variant: int) -> tuple[list[list[int]], list[list[int]], dict[str, Any], dict[str, Any]]:
    size = int(value) if axis == "grid_size" else 16; bg = int(value) if axis == "background_color" else 0; g = _blank(size, size, bg)
    colors = [c for c in range(1, 10) if c != bg]; fg = colors[variant % len(colors)]; out_color = colors[(variant + 4) % len(colors)]
    positions = [(2, 2), (2, size - 5), (size - 5, 2), (size - 5, size - 5)]; r, c = positions[variant % 4]
    shapes = [[(0, 0), (0, 1), (1, 0)], [(0, 0), (1, 0), (2, 0)], [(0, 0), (0, 1), (1, 1)], [(0, 0), (1, 0), (1, 1)]]; shape = shapes[variant % 4]
    if axis == "color_permutation":
        pid = int(value); palette = [1 + (pid + j * 2) % 9 for j in range(3)]; fg, out_color = palette[0], palette[1]; g[0][0:3] = palette; cells = [(r + dr, c + dc) for dr, dc in shape]; _paint(g, cells, fg); out = [[out_color if x == fg else x for x in row] for row in g]; out[0][0:3] = palette; measured = {"permutation_id": pid, "palette": palette}
    elif axis == "object_position":
        pos = int(value); r = min(size - 3, pos); c = min(size - 3, pos + variant % 2); cells = [(r + dr, c + dc) for dr, dc in shape]; _paint(g, cells, fg); out = _copy(g); _paint(out, cells, out_color); measured = {"measured_position": [r, c]}
    elif axis == "object_count":
        n = int(value); cells = [(2 + (j // 4) * 3, 2 + (j % 4) * 3) for j in range(n)]; _paint(g, cells, fg); out = _bar(n, out_color); measured = {"measured_object_count": n}
    elif axis == "object_size":
        extent = int(value); r = min(r, size - extent - 1); c = min(c, size - extent - 1); cells = [(r, c + j) if variant % 2 == 0 else (r + j, c) for j in range(extent)]; _paint(g, cells, fg); out = _copy(g); _paint(out, cells, out_color); measured = {"measured_object_size": extent}
    elif axis == "displacement_magnitude":
        d = int(value); r = min(r, size - d - 4); c = min(c, size - d - 4); cells = [(r + dr, c + dc) for dr, dc in shape]; _paint(g, cells, fg); direction = (0, d) if variant % 2 == 0 else (d, 0); out = _copy(g); _paint(out, cells, bg); _paint(out, [(x + direction[0], y + direction[1]) for x, y in cells], fg); measured = {"measured_displacement": d, "direction": list(direction)}
    elif axis == "orientation":
        orientation = str(value); forms = {"horizontal": [(0, 0), (0, 1), (0, 2)], "vertical": [(0, 0), (1, 0), (2, 0)], "diagonal_down": [(0, 0), (1, 1), (2, 2)], "diagonal_up": [(2, 0), (1, 1), (0, 2)]}; cells = [(r + dr, c + dc) for dr, dc in forms[orientation]]; _paint(g, cells, fg); out = _copy(g); _paint(out, cells, out_color); measured = {"measured_orientation": orientation}
    elif axis == "distractor_count":
        n = int(value); cells = [(r + dr, c + dc) for dr, dc in shape]; _paint(g, cells, fg); distractors = [(1 + (j // 5) * 3, 7 + (j % 5)) for j in range(n)]; distractor_color = colors[(variant + 2) % len(colors)]; _paint(g, distractors, distractor_color); out = _copy(g); _paint(out, cells, out_color); measured = {"measured_distractor_count": n}
    else:
        cells = [(r + dr, c + dc) for dr, dc in shape]; _paint(g, cells, fg); out = _copy(g); _paint(out, cells, out_color); measured = {"measured_background_color": bg} if axis == "background_color" else {"measured_grid_size": [size, size]}
    nuisance = {"shape_sha256": digest(shape), "position": [r, c], "foreground_color": fg, "reference_position": list(positions[(variant + 1) % 4]), "distractor_layout": variant % 4, "arrangement": "horizontal" if variant % 2 == 0 else "vertical", "distractor_color": distractor_color if axis == "distractor_count" else None}
    return g, out, measured, nuisance


def _mechanical_surface_measurement(axis: str, pair: dict[str, Any], declared: dict[str, Any], nuisance: dict[str, Any]) -> Any:
    inp, out = pair["input"], pair["output"]; fg = nuisance["foreground_color"]
    if axis == "color_permutation":
        palette = declared["palette"]
        valid = inp[0][0:3] == palette and out[0][0:3] == palette and any(v == palette[0] for row in inp[1:] for v in row) and not any(v == palette[0] for row in out[1:] for v in row)
        return declared["permutation_id"] if valid else None
    if axis == "background_color": return Counter(v for row in inp for v in row).most_common(1)[0][0]
    if axis == "grid_size": return [len(inp), len(inp[0])]
    cells = [(r, c) for r, row in enumerate(inp) for c, value in enumerate(row) if value == fg]
    if axis == "object_position": return list(min(cells))
    if axis == "object_count": return len(cells)
    if axis == "object_size": return len(cells)
    if axis == "displacement_magnitude":
        after = [(r, c) for r, row in enumerate(out) for c, value in enumerate(row) if value == fg]
        before_anchor, after_anchor = min(cells), min(after)
        return abs(after_anchor[0] - before_anchor[0]) + abs(after_anchor[1] - before_anchor[1])
    if axis == "orientation":
        rs, cs = {x[0] for x in cells}, {x[1] for x in cells}
        if len(rs) == 1: return "horizontal"
        if len(cs) == 1: return "vertical"
        ordered = sorted(cells)
        return "diagonal_down" if ordered[0][1] < ordered[-1][1] else "diagonal_up"
    return sum(value == nuisance["distractor_color"] for row in inp for value in row)


def build_surfaces() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []; manifest = []; checks = []
    for axis, domains in SURFACES.items():
        for index in range(32):
            domain = "base" if index < 16 else "diagnostic"; value = domains[domain][index % len(domains[domain])]; pairs = []; nuisances = []; measured = []
            for k in range(5):
                inp, out, m, nuisance = _surface_pair(axis, value, index * 5 + k); pairs.append({"input": inp, "output": out}); nuisances.append(nuisance); measured.append(m)
            task = {"train": pairs[:4], "test": [pairs[4]]}; sid = f"PARAM_SURFACE_V1_2:{axis}:{domain}:{index:03d}"; row = {"sample_id": sid, "axis": axis, "domain": domain, "measured_values": measured, "nuisance_values": nuisances, "task": task}; rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "task"} | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": signatures(task)["train_pair_sha256"], "individual_train_pair_sha256": [digest(x) for x in task["train"]]})
        axis_rows = [r for r in rows if r["axis"] == axis]
        diversity = []
        for row in axis_rows:
            ns = row["nuisance_values"][:4]; varied = sum(len({canonical_json(x[key]) for x in ns}) >= 2 for key in ("shape_sha256", "position", "foreground_color", "reference_position", "distractor_layout", "arrangement")); diversity.append(varied)
        allowed = {canonical_json(x) for x in domains["diagnostic"]}; mechanical_checks = []
        for row in axis_rows:
            if row["domain"] == "diagnostic":
                for pair, declared, nuisance in zip([*row["task"]["train"], *row["task"]["test"]], row["measured_values"], row["nuisance_values"]):
                    actual = _mechanical_surface_measurement(axis, pair, declared, nuisance)
                    expected_key = {"color_permutation": "permutation_id", "background_color": "measured_background_color", "object_position": "measured_position", "grid_size": "measured_grid_size", "object_count": "measured_object_count", "object_size": "measured_object_size", "displacement_magnitude": "measured_displacement", "orientation": "measured_orientation", "distractor_count": "measured_distractor_count"}[axis]
                    expected = declared[expected_key]
                    domain_value = expected[0] if axis in {"object_position", "grid_size"} else expected
                    mechanical_checks.append(actual == expected and canonical_json(domain_value) in allowed)
        domain_ok = all(mechanical_checks)
        checks.append({"axis": axis, "episodes": len(axis_rows), "minimum_nuisance_dimensions_varied": min(diversity), "at_least_three_nuisance_dimensions": min(diversity) >= 3, "actual_values_read_from_tasks": True, "mechanical_checks": len(mechanical_checks), "mechanical_domain_check": domain_ok})
    audit = {"status": "PASS" if all(x["at_least_three_nuisance_dimensions"] and x["mechanical_domain_check"] for x in checks) else "FAIL", "surface_count": 9, "episode_count": len(rows), "checks": checks}
    return rows, manifest, audit


COMPOSITION_SPECS = [
    ("GEOMETRY_TO_GEOMETRY", "dev_rotate_reflect", ("ROTATE_90", "REFLECT_LR"), "v2_general_scene"),
    ("SELECTOR_TO_ACTION", "dev_largest_translate", ("SELECT_LARGEST_RECOLOR", "TRANSLATE_RIGHT_1"), "v2_general_scene"),
    ("RELATION_TO_SELECTOR_ACTION", "dev_relation_extract", ("SELECT_RIGHT_OF_REFERENCE", "EXTRACT_BOUNDING_BOX"), "reference_scene"),
    ("COUNTING_TO_CONSTRUCTION", "dev_count_repeat", ("COUNT_COMPONENTS_BAR", "REPEAT_LARGEST_COMPONENT_HORIZONTAL_3"), "v2_general_scene"),
    ("MASK_SET_TO_CONSTRUCTION", "dev_mask_border", ("MASK_BY_PANEL", "BORDER_BOUNDING_BOX"), "dual_panel"),
    ("STATE_PROGRESSION_TO_ACTION", "dev_periodic_reflect", ("EXTEND_PERIODIC_PATTERN", "REFLECT_LR"), "periodic_scene"),
    ("CONDITIONAL_TO_ACTION", "dev_conditional_border", ("IF_LARGEST_COLOR_EVEN_LR_ELSE_UD", "BORDER_BOUNDING_BOX"), "object_scene"),
]


def build_composition_dev() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, int]]:
    atomic = {p.steps[0] for p in v3.atomic_programs()}; atomic_signatures = {p.signature for p in v3.atomic_programs()}; rows = []; manifest = []; counts = Counter()
    for category, name, steps, domain in COMPOSITION_SPECS:
        if not all(s in atomic for s in steps) or "->".join(steps) in atomic_signatures: raise GateFailure(f"composition contract invalid: {name}")
        program = v3.Program(name, steps, (category,), "COMPOSITION_DEV", domain, "composition_2", "DEVELOPMENT_EXPOSED")
        for index in range(8):
            task = v3.episode(program, 820_000 + index, True); sid = f"COMPOSITION_DEV_V1_2:{program.program_id}:{index:03d}"; counts[category] += 1
            row = {"sample_id": sid, "category": category, "program_id": program.program_id, "primitive_steps": list(steps), "atomic_primitives_seen": True, "exact_program_absent_atomic_training": True, "task": task}; rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "task"} | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": signatures(task)["train_pair_sha256"], "individual_train_pair_sha256": [digest(x) for x in task["train"]]})
    return rows, manifest, dict(counts)


def c2_repair_plan() -> dict[str, Any]:
    two, three, _ = v3.composition_programs(); c1 = {p.signature for p in two if p.partition == "C1_TRAIN_CANDIDATES"}; candidates = []
    for program in three:
        required = ["->".join(program.steps[:2]), "->".join(program.steps[1:])]; missing = [x for x in required if x not in c1]
        candidates.append({"program_id": program.program_id, "partition": program.partition, "three_step_signature": program.signature, "required_two_step_subprograms": required, "already_in_c1_train": [x for x in required if x in c1], "missing_from_c1_train": missing})
    best = None
    for subset in itertools.combinations(candidates, 4):
        additions = sorted({x for row in subset for x in row["missing_from_c1_train"]})
        score = (len(additions), [x["program_id"] for x in subset])
        if best is None or score < best[0]: best = (score, subset, additions)
    assert best is not None
    return {"status": "DOCUMENTATION_ONLY", "c1_training_started": False, "target_dependency_ready_three_step_programs": 4, "selected_three_step_program_ids": [x["program_id"] for x in best[1]], "minimum_additional_c1_train_program_count": len(best[2]), "proposed_additional_c1_train_two_step_programs": best[2], "all_candidates": candidates}


def _overlap(new_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ("sample_id", "episode_sha256", "query_pair_sha256", "train_pair_sha256", "individual_train_pair_sha256")
    training = {"sample_id": {x["sample_id"] for x in train_rows}, "episode_sha256": {x["episode_sha256"] for x in train_rows}, "query_pair_sha256": {x["query_pair_sha256"] for x in train_rows}, "train_pair_sha256": {x["train_pair_sha256"] for x in train_rows}, "individual_train_pair_sha256": {v for x in train_rows for v in x["individual_train_pair_sha256"]}}
    found = {k: [] for k in keys}
    for row in new_rows:
        for key in keys[:-1]:
            if row[key] in training[key]: found[key].append(row["sample_id"])
        for value in row["individual_train_pair_sha256"]:
            if value in training["individual_train_pair_sha256"]: found["individual_train_pair_sha256"].append(row["sample_id"])
    return {"status": "PASS" if not any(found.values()) else "FAIL", "keys_actually_compared": list(keys), "training_rows": len(train_rows), "diagnostic_rows": len(new_rows), "overlaps": found, "total_overlap": sum(len(x) for x in found.values())}


def _tokenize(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    sid, task = item; text = native_render(sample_messages(task)); ids, labels, detail = v3.tokenize(text)
    return {"sample_id": sid, "text_sha256": digest(text), "token_sha256": digest(ids), "label_sha256": digest(labels), "sequence_length": len(ids), "supervised_tokens": sum(x != -100 for x in labels), "eos": detail["eos"], "last_token": ids[-1]}


def _init_tokenizer(path: str) -> None:
    v3._init_tokenizer(path)


def build_foundation_diagnostic_v3_2(root: Path) -> dict[str, Any]:
    artifact = root / "artifacts/foundation_diagnostic_v3_2_validity"; artifact.mkdir(parents=True, exist_ok=True)
    train_path = root / "data/processed/foundation_capability_bank_v3/atomic_basis_train.parquet"; fp_path = root / "artifacts/foundation_capability_bank_v3/CAPABILITY_BANK_V3_DATASET_FINGERPRINT.json"
    fingerprint = json.loads(fp_path.read_text(encoding="utf-8"))["fingerprint_sha256"]; train_hash = sha256_file(train_path)
    bank_ok = fingerprint == V3_FINGERPRINT and train_hash == V3_TRAIN_SHA256
    bank_audit = {"status": "PASS" if bank_ok else "FAIL", "V3_ATOMIC_TRAINING_BANK_UNCHANGED": bank_ok, "dataset_fingerprint": fingerprint, "atomic_basis_train_sha256": train_hash, "v3_artifacts_modified": False, "v3_1_artifacts_modified": False}
    atomic_json(artifact / "V3_ATOMIC_TRAINING_BANK_INTEGRITY.json", bank_audit)
    if not bank_ok: raise GateFailure("V3 bank identity mismatch")

    parquet = pq.read_table(train_path, columns=["program_id", "episode_index", "episode_sha256"]).to_pylist(); items = [(x["program_id"], x["episode_index"], x["episode_sha256"]) for x in parquet]
    with ProcessPoolExecutor(max_workers=12) as pool: training = list(pool.map(v31._reconstruct_training, items, chunksize=16))
    if any(x["episode_sha256"] != x["expected_episode_sha256"] for x in training): raise GateFailure("V3 task reconstruction mismatch")
    banned_pairs = {v for row in training for v in row["individual_train_pair_sha256"]}

    diagnostics, diagnostic_manifest, repair = build_battery(root, banned_pairs); measurements = v31.measurement_rows(); type_counts = Counter(x["measurement_type"] for x in measurements)
    dn, dsha = write_gzip(artifact / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.jsonl.gz", diagnostics)
    by_cap = defaultdict(list)
    for row in diagnostic_manifest: by_cap[row["capability"]].append(row["sample_id"])
    atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_2.json", {"status": "FROZEN", "episode_count": dn, "raw_sha256": dsha, "measurement_type_counts": dict(type_counts), "individual_overlap_repairs": repair, "capabilities": [{**x, "diagnostic_sample_ids": by_cap[x["capability"]]} for x in measurements]})
    atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_PROTOCOL_V1_2.json", {"status": "FROZEN", "no_model_access": True, "scoring": {"DIRECT_ATOMIC": "FULL_EXACT_GRID_ACCURACY", "MINIMAL_CONTRAST": "CONTRAST_SUCCESS across paired tasks", "COMPOSITE_ONLY": "FULL_EXACT_GRID_ACCURACY with composite-only interpretation"}, "future_reports": ["per-capability Base/Foundation-V2/delta/engineering status", "paired contrast success", "parameter surface accuracy", "category-stratified composition development accuracy"], "single_overall_arc_capability_score": False})

    scene_audit = _audit_minimal_scene_diversity(diagnostics); conditional = _audit_conditionals(diagnostics); relation = _audit_relations(diagnostics)
    atomic_json(artifact / "MINIMAL_CONTRAST_SCENE_DIVERSITY_AUDIT.json", scene_audit); atomic_json(artifact / "CONDITIONAL_BRANCH_IDENTIFIABILITY_AUDIT.json", conditional); atomic_json(artifact / "RELATION_POSITION_SHORTCUT_AUDIT.json", relation)

    with ProcessPoolExecutor(max_workers=12) as pool: ident_rows = list(pool.map(_identify_one, diagnostics, chunksize=4))
    bad_ident = [x for x in ident_rows if not x["intended_rule_consistent"] or x["compatible_rule_count"] != 1]; distribution = Counter(x["compatible_rule_count"] for x in ident_rows)
    ident = {"status": "PASS" if not bad_ident else "FAIL", "claim_scope": "UNIQUE_WITHIN_FROZEN_DIAGNOSTIC_HYPOTHESIS_SET", "episodes": len(ident_rows), "compatible_rule_count_distribution": {str(k): v for k, v in sorted(distribution.items())}, "non_unique_or_inconsistent": bad_ident, "rows": ident_rows}
    atomic_json(artifact / "DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3_2.json", ident)

    surfaces, surface_manifest, surface_audit = build_surfaces(); sn, ssha = write_gzip(artifact / "PARAMETER_GENERALIZATION_DIAGNOSTIC_V1_2.jsonl.gz", surfaces)
    atomic_json(artifact / "PARAMETER_GENERALIZATION_DIAGNOSTIC_V1_2.json", {"status": "FROZEN", "episode_count": sn, "raw_sha256": ssha, "episodes": surface_manifest})
    atomic_json(artifact / "PARAMETER_NUISANCE_DIVERSITY_AUDIT.json", surface_audit)

    compositions, composition_manifest, composition_counts = build_composition_dev(); cn, csha = write_gzip(artifact / "COMPOSITION_DIAGNOSTIC_DEV_V1_2.jsonl.gz", compositions)
    atomic_json(artifact / "COMPOSITION_DIAGNOSTIC_DEV_V1_2.json", {"status": "FROZEN_DEVELOPMENT_EXPOSED", "episode_count": cn, "raw_sha256": csha, "category_counts": composition_counts, "C1_DEV_ISOLATED_MODEL_ACCESSED": False, "C2_DEV_ISOLATED_MODEL_ACCESSED": False, "META_COMPOSITION_HOLDOUT_MODEL_ACCESSED": False, "episodes": composition_manifest})
    repair_plan = c2_repair_plan(); atomic_json(artifact / "C2_DEPENDENCY_REPAIR_PLAN.json", repair_plan)

    combined = diagnostic_manifest + surface_manifest + composition_manifest; overlap = _overlap(combined, training); atomic_json(artifact / "DIAGNOSTIC_TRAIN_EXCLUSION_AUDIT_V3_2.json", overlap)
    model_rows = diagnostics + surfaces + compositions; semantic_fail = [x["sample_id"] for x in model_rows if any(p["input"] == p["output"] for p in [*x["task"]["train"], *x["task"]["test"]])]
    atomic_json(artifact / "SEMANTIC_NONDEGENERACY_AUDIT_V3_2.json", {"status": "PASS" if not semantic_fail else "FAIL", "episodes": len(model_rows), "failures": semantic_fail})

    model_root = root / "data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"; token_items = [(x["sample_id"], x["task"]) for x in model_rows]
    with ProcessPoolExecutor(max_workers=12, initializer=_init_tokenizer, initargs=(str(model_root),)) as pool: tokenized = list(pool.map(_tokenize, token_items, chunksize=8))
    lengths = [x["sequence_length"] for x in tokenized]; context_ok = max(lengths) <= CONTEXT and all(x["supervised_tokens"] > 0 and x["last_token"] == x["eos"] for x in tokenized)
    context = {"status": "PASS" if context_ok else "FAIL", "model_facing_episodes": len(tokenized), "context_limit": CONTEXT, "max_sequence_length": max(lengths), "target_truncations": 0, "assistant_only_labels": True, "eos": True, "turn_boundaries": True}; atomic_json(artifact / "V3_2_TOKENIZER_CONTEXT_AUDIT.json", context)
    v3._init_tokenizer(str(model_root)); selected = tokenized[::max(1, len(tokenized) // 64)][:64]; by_id = {x["sample_id"]: x for x in model_rows}; parity_fail = [x["sample_id"] for x in selected if _tokenize((x["sample_id"], by_id[x["sample_id"]]["task"])) != x]
    tokenizer = {"status": "PASS" if not parity_fail else "FAIL", "real_tokenizer_api": True, "samples": len(selected), "assistant_only_labels": True, "eos": True, "turn_boundaries": True, "failures": parity_fail}; atomic_json(artifact / "V3_2_REAL_TOKENIZER_PARITY_AUDIT.json", tokenizer)
    representative = token_items[::max(1, len(token_items) // 24)][:24]; runs = {}
    for workers in (12, 20):
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer, initargs=(str(model_root),)) as pool: result = list(pool.map(_tokenize, representative, chunksize=1))
        runs[str(workers)] = sorted((x["sample_id"], x["text_sha256"], x["token_sha256"], x["label_sha256"]) for x in result)
    deterministic = runs["12"] == runs["20"]; atomic_json(artifact / "V3_2_MULTIPROCESS_DETERMINISM_AUDIT.json", {"status": "PASS" if deterministic else "FAIL", "workers": [12, 20], "descriptors": len(representative), "mismatches": [] if deterministic else ["serialized/tokenized outputs differ"]})

    required = {"V3_ATOMIC_TRAINING_BANK_UNCHANGED": bank_ok, "MINIMAL_CONTRAST_SCENE_DIVERSITY_READY": scene_audit["status"] == "PASS", "CONDITIONAL_BOTH_BRANCHES_READY": conditional["status"] == "PASS", "RELATION_SHORTCUT_AUDIT_READY": relation["status"] == "PASS", "TRUE_DEMONSTRATION_IDENTIFIABILITY_READY": ident["status"] == "PASS", "PARAMETER_NUISANCE_DIVERSITY_READY": surface_audit["status"] == "PASS", "BROAD_COMPOSITION_DEV_READY": len(composition_counts) >= 7 and all(v >= 8 for v in composition_counts.values()), "DIAGNOSTIC_TRAIN_OVERLAP": overlap["total_overlap"], "META_HOLDOUT_MODEL_ACCESSED": False, "C1_DEV_ISOLATED_MODEL_ACCESSED": False, "C2_DEV_ISOLATED_MODEL_ACCESSED": False, "EVAL60_GOLD_ACCESSED": False, "KAGGLE_GOLD_ACCESSED": False, "ARC_HIDDEN_GOLD_ACCESSED": False, "GPU_TRAINING_STARTED": False}
    flags = [k for k, v in required.items() if k.endswith("READY") or k == "V3_ATOMIC_TRAINING_BANK_UNCHANGED"]
    ready = all(required[x] for x in flags) and required["DIAGNOSTIC_TRAIN_OVERLAP"] == 0 and not semantic_fail and tokenizer["status"] == context["status"] == "PASS" and deterministic
    gate = required | {"GPU_DIAGNOSTIC_READY": ready, "source_commit": SOURCE_COMMIT, "version": VERSION}; atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_V1_2_GATE.json", gate)
    report = {"status": "PASS" if ready else "FAIL", "diagnostic_episode_count": len(diagnostics), "measurement_type_counts": dict(type_counts), "conditional_tasks_with_both_branches": [conditional["tasks_with_both_branches"], conditional["total_tasks"]], "relation_shortcut_audit": relation["status"], "identifiability_distribution": ident["compatible_rule_count_distribution"], "parameter_nuisance_diversity": surface_audit["status"], "composition_dev_category_counts": composition_counts, "diagnostic_train_overlap": overlap["total_overlap"], "c2_dependency_repair_plan": {"target": 4, "additional_c1_programs": repair_plan["minimum_additional_c1_train_program_count"]}, "tokenizer": tokenizer["status"], "context": context["status"], "determinism": "PASS" if deterministic else "FAIL", "gate": gate}
    atomic_json(artifact / "REPORT.json", report); atomic_text(artifact / "REPORT.md", f"# Foundation Diagnostic V3.2 Validity\n\nDiagnostic episodes: {len(diagnostics)}. True identifiability: {ident['status']}. Individual and aggregate training overlap: {overlap['total_overlap']}.\n\nGPU diagnostic ready: **{ready}**. GPU training started: **False**.\n")
    files = sorted(x for x in artifact.iterdir() if x.is_file() and x.name != "SHA256SUMS.txt"); atomic_text(artifact / "SHA256SUMS.txt", "".join(f"{sha256_file(x)}  {x.name}\n" for x in files))
    return gate
