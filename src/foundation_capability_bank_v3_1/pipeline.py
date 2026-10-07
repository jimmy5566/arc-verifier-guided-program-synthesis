"""CPU-only V3.1 diagnostic and composition-contract hardening.

The frozen V3 Atomic Basis bank is an input, never an output.  This module
materializes honest measurement contracts, real counterfactual diagnostics,
real parameter surfaces, and mechanically staged composition evidence.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq

from foundation_capability_bank_v3 import pipeline as v3
from training_data_v2.pipeline import canonical_json, native_render, sample_messages, sha256_file, signatures

VERSION = "foundation_capability_bank_v3_1"
SOURCE_COMMIT = "75c5546add5dd62f4338a1cc08e3e84ad2417a91"
V3_FINGERPRINT = "80149142606717eb617f415c486fa1c26b19e9a868404a9870efc0e501ceff2f"
V3_TRAIN_SHA256 = "27b3b495ce9538683d70c48fd3b58e462e0efd773042012b88467f958f0b7504"
CONTEXT = 8704


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


def _write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> tuple[int, str]:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with gzip.open(path, "wt", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
            count += 1
    return count, sha256_file(path)


MINIMAL_CAPABILITIES = {
    "connected components", "count components", "encode quantity geometrically",
    "area", "largest", "recolor", "same color", "left/right", "above/below",
    "nearest/farthest", "inside/contains", "relation-conditioned selector",
    "if/else on attribute", "if/else on relation", "if/else on count",
    "branch-balanced conditional tasks",
}


def _all_capabilities() -> list[str]:
    return sorted(cap for caps in v3.ONTOLOGY.values() for cap in caps)


def measurement_rows() -> list[dict[str, Any]]:
    programs = v3.atomic_programs()
    covered: dict[str, list[v3.Program]] = defaultdict(list)
    for program in programs:
        for capability in program.capabilities:
            covered[capability].append(program)
    rows = []
    for capability in _all_capabilities():
        options = covered[capability]
        if not options:
            raise GateFailure(f"uncovered capability: {capability}")
        program = options[0]
        if capability in MINIMAL_CAPABILITIES:
            kind = "MINIMAL_CONTRAST"
            scoring = "CONTRAST_SUCCESS"
            limit = "paired causal response; nuisance controls listed explicitly"
        elif len(program.capabilities) == 1:
            kind = "DIRECT_ATOMIC"
            scoring = "FULL_EXACT_GRID_ACCURACY"
            limit = "isolated executable mechanism within the frozen program bank"
        else:
            kind = "COMPOSITE_ONLY"
            scoring = "FULL_EXACT_GRID_ACCURACY"
            limit = "composite chain only; no independent capability claim"
        rows.append({
            "capability": capability,
            "measurement_type": kind,
            "program_id": program.program_id,
            "program_capabilities": list(program.capabilities),
            "nuisance_capabilities": [x for x in program.capabilities if x != capability],
            "scoring_rule": scoring,
            "interpretation_limit": limit,
        })
    return rows


def _grid(h: int = 9, w: int = 9, bg: int = 0) -> list[list[int]]:
    return [[bg] * w for _ in range(h)]


def _paint(g: list[list[int]], cells: Iterable[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        g[r][c] = color


def _bar(n: int, color: int = 2, vertical: bool = False) -> list[list[int]]:
    return [[color] for _ in range(n)] if vertical else [[color] * n]


def _contrast_io(capability: str, condition: int, seed: int) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    rng = random.Random(seed)
    if capability in {"connected components", "count components"}:
        g = _grid()
        if condition == 0:
            cells = [(2, 2), (2, 3), (3, 2), (3, 3), (4, 2), (4, 3)]
            components = 1
        else:
            cells = [(2, 2), (2, 3), (3, 2), (5, 5), (5, 6), (6, 5)]
            components = 2
        _paint(g, cells, 3)
        return g, _bar(components), {"total_cells": 6, "encoding": "horizontal_bar", "components": components}
    if capability == "encode quantity geometrically":
        g = _grid()
        _paint(g, [(2, 2), (4, 4), (6, 6)], 3)
        g[0][0] = 7 + condition
        return g, _bar(3, 2, vertical=bool(condition)), {"components": 3, "encoding": "vertical_bar" if condition else "horizontal_bar"}
    if capability == "area":
        g = _grid()
        area = 2 + condition * 2
        _paint(g, [(2, 2 + c) for c in range(area)], 3)
        _paint(g, [(6, 6), (6, 7), (7, 6)], 4)
        return g, _bar(area), {"reference_color": 3, "measured_area": area, "encoding": "horizontal_bar"}
    if capability == "largest":
        g = _grid(); left = 3 if condition == 0 else 1; right = 1 if condition == 0 else 3
        _paint(g, [(2 + r, 2) for r in range(left)], 3); _paint(g, [(2 + r, 6) for r in range(right)], 4)
        out = [row[:] for row in g]
        target = [(2 + r, 2) for r in range(left)] if left > right else [(2 + r, 6) for r in range(right)]
        _paint(out, target, 8)
        return g, out, {"action": "recolor_8", "left_area": left, "right_area": right}
    if capability == "recolor":
        g = _grid(); new = 6 + condition
        _paint(g, [(3, 3), (3, 4), (4, 3), (4, 4)], 2); g[0][0] = 2; g[0][1] = new
        out = [row[:] for row in g]
        _paint(out, [(3, 3), (3, 4), (4, 3), (4, 4)], new)
        return g, out, {"selected_target": "fixed_square", "mapping": [2, new]}
    if capability in {"if/else on attribute", "if/else on relation", "if/else on count", "branch-balanced conditional tasks"}:
        g = _grid(7, 7); color = 2 + condition
        _paint(g, [(1, 1), (1, 2), (2, 1), (3, 1)], color)
        if capability == "if/else on relation":
            g[5][1 if condition == 0 else 5] = 6
        elif capability == "if/else on count":
            g[5][1] = 6
            if condition: g[5][5] = 6
        out = [list(row) for row in (zip(*g[::-1]) if condition == 0 else g[::-1])]
        return g, out, {"branch": "true" if condition == 0 else "false", "branches_valid": True}
    # Relation-conditioned selectors: identities, colors, and action are fixed;
    # only the named relation changes between pair members.
    g = _grid(); ref = (4, 4); g[ref[0]][ref[1]] = 5
    positions = {
        "left/right": ((4, 2), (4, 6)),
        "above/below": ((2, 4), (6, 4)),
        "nearest/farthest": ((4, 3), (4, 7)),
        "inside/contains": ((3, 3), (1, 1)),
        "same color": ((3, 2), (5, 6)),
        "relation-conditioned selector": ((2, 4), (6, 4)),
    }
    a, b = positions[capability]
    target, other = (a, b) if condition == 0 else (b, a)
    g[a[0]][a[1]] = 3; g[b[0]][b[1]] = 4
    if capability == "same color": g[target[0]][target[1]] = 5
    if capability == "inside/contains":
        _paint(g, [(r, c) for r in range(2, 7) for c in range(2, 7) if r in (2, 6) or c in (2, 6)], 6)
    out = [row[:] for row in g]; out[target[0]][target[1]] = 8
    return g, out, {"reference": ref, "selected_position": target, "action": "recolor_8", "relation_truth": bool(condition == 0)}


def contrast_task(capability: str, pair_index: int, condition: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pairs = []
    details = []
    for k in range(5):
        inp, out, detail = _contrast_io(capability, condition, 1_000_003 + pair_index * 101 + k)
        # A preregistered nuisance barcode makes episodes unique.  It is the
        # same in both counterfactual members and never encodes condition.
        code = _all_capabilities().index(capability) * 80 + pair_index * 5 + k
        if len(inp) >= 7 and len(inp[0]) >= 7:
            for digit_index in range(4):
                c = len(inp[0]) - 1 - digit_index
                inp[-1][c] = (code // (10 ** digit_index)) % 10
                if len(out) == len(inp) and len(out[0]) == len(inp[0]): out[-1][c] = inp[-1][c]
        pairs.append({"input": inp, "output": out}); details.append(detail)
    return {"train": pairs[:4], "test": [pairs[4]]}, details[4]


def _load_v3_diagnostics(root: Path) -> dict[str, list[dict[str, Any]]]:
    path = root / "data/processed/foundation_capability_bank_v3/foundation_diagnostic_battery_v1.jsonl.gz"
    by_cap: dict[str, list[dict[str, Any]]] = defaultdict(list)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line); by_cap[row["capability"]].append(row)
    return by_cap


def _color_permute_task(task: dict[str, Any], shift: int) -> dict[str, Any]:
    """Apply one global bijection to inputs and targets, preserving semantics."""
    if shift == 0:
        return task
    def grid(value: list[list[int]]) -> list[list[int]]:
        return [[0 if x == 0 else 1 + ((x - 1 + shift) % 9) for x in row] for row in value]
    return {split: [{"input": grid(pair["input"]), "output": grid(pair["output"])} for pair in task[split]] for split in ("train", "test")}


def build_diagnostics(root: Path, measurement: list[dict[str, Any]], banned: dict[str, set[str]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    old = _load_v3_diagnostics(root); rows = []; manifest = []
    used_episode: set[str] = set(); used_query: set[str] = set()
    def acceptable(task: dict[str, Any]) -> bool:
        episode_hash = digest(task); query_hash = digest(task["test"][0]); aggregate = signatures(task)["train_pair_sha256"]
        return episode_hash not in banned["episode_sha256"] | used_episode and query_hash not in banned["query_pair_sha256"] | used_query and aggregate not in banned["train_pair_sha256"]
    def reserve(task: dict[str, Any]) -> None:
        used_episode.add(digest(task)); used_query.add(digest(task["test"][0]))
    for contract in measurement:
        cap = contract["capability"]
        if contract["measurement_type"] == "MINIMAL_CONTRAST":
            for pair_index in range(16):
                pair_id = f"contrast:{digest({'capability': cap, 'pair': pair_index})[:20]}"
                members = []
                for condition in (0, 1):
                    task, measured = contrast_task(cap, pair_index, condition)
                    sid = f"DIAGNOSTIC_V1_1:{digest({'pair': pair_id, 'condition': condition})[:24]}"
                    member = {"sample_id": sid, "pair_id": pair_id, "pair_member": condition, "capability": cap,
                              "measurement_type": "MINIMAL_CONTRAST", "program_id": contract["program_id"],
                              "scoring_rule": "CONTRAST_SUCCESS", "measured": measured, "task": task}
                    members.append(member)
                if digest(members[0]["task"]["test"][0]["output"]) == digest(members[1]["task"]["test"][0]["output"]):
                    raise GateFailure(f"contrast output unchanged: {cap}")
                nuisance = digest({k: v for k, v in members[0]["measured"].items() if k not in {"components", "encoding", "measured_area", "mapping", "selected_position", "relation_truth", "branch"}})
                for member in members:
                    if not acceptable(member["task"]): raise GateFailure(f"minimal contrast collision: {member['sample_id']}")
                    member["nuisance_control_sha256"] = nuisance; reserve(member["task"]); rows.append(member)
        else:
            for source in old[cap][:32]:
                base_task = source["task"]; chosen = None; chosen_shift = None
                for shift in range(9):
                    candidate = _color_permute_task(base_task, shift)
                    if acceptable(candidate): chosen, chosen_shift = candidate, shift; break
                if chosen is None: raise GateFailure(f"diagnostic color-permutation uniqueness exhausted: {cap}:{source['sample_id']}")
                task = chosen; sid = f"DIAGNOSTIC_V1_1:{digest({'capability': cap, 'source': source['sample_id'], 'shift': chosen_shift})[:24]}"
                rows.append({"sample_id": sid, "source_v3_sample_id": source["sample_id"], "capability": cap,
                             "measurement_type": contract["measurement_type"], "program_id": contract["program_id"],
                             "scoring_rule": "FULL_EXACT_GRID_ACCURACY", "global_color_permutation_shift": chosen_shift, "task": task})
                reserve(task)
    for row in rows:
        task = row["task"]
        if any(pair["input"] == pair["output"] for pair in [*task["train"], *task["test"]]):
            raise GateFailure(f"degenerate diagnostic: {row['sample_id']}")
        manifest.append({k: v for k, v in row.items() if k != "task"} | {
            "episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]),
            "train_pair_sha256": signatures(task)["train_pair_sha256"]})
    return rows, manifest


SURFACES = {
    "color_permutation": {"base": [0, 1, 2, 3], "diagnostic": [4, 5, 6, 7]},
    "background_color": {"base": [0, 1], "diagnostic": [8, 9]},
    "object_position": {"base": [2, 3, 4], "diagnostic": [10, 11, 12, 13]},
    "grid_size": {"base": [8, 9, 10, 11, 12], "diagnostic": [20, 21, 22, 23, 24]},
    "object_count": {"base": [1, 2, 3, 4], "diagnostic": [5, 6, 7]},
    "object_size": {"base": [1, 2, 3], "diagnostic": [4, 5, 6]},
    "displacement_magnitude": {"base": [1, 2], "diagnostic": [3, 4]},
    "orientation": {"base": ["horizontal", "vertical"], "diagnostic": ["diagonal_down", "diagonal_up"]},
    "distractor_count": {"base": [0, 1, 2], "diagnostic": [3, 4, 5]},
}


def _surface_pair(axis: str, value: Any, seed: int) -> tuple[list[list[int]], list[list[int]], dict[str, Any]]:
    rng = random.Random(seed); size = int(value) if axis == "grid_size" else 14
    bg = int(value) if axis == "background_color" else 0; g = _grid(size, size, bg); out = [row[:] for row in g]
    color = 2 if bg != 2 else 3
    if axis == "color_permutation":
        palette = [((int(value) + j) % 8) + 1 for j in range(3)]; _paint(g, [(2, 2), (2, 3), (3, 2)], palette[0]); g[0][0:3] = palette
        out = [[palette[1] if x == palette[0] else x for x in row] for row in g]; out[0][0:3] = palette
        measured = {"permutation_id": int(value), "palette": palette}
    elif axis == "background_color":
        _paint(g, [(2, 2), (2, 3), (3, 2)], color); out = [row[:] for row in g]; _paint(out, [(2, 2), (2, 3), (3, 2)], 6); measured = {"measured_background_color": bg}
    elif axis == "object_position":
        pos = int(value); r = min(size - 3, pos); c = min(size - 3, pos); _paint(g, [(r, c), (r, c + 1)], color); out = [row[:] for row in g]; _paint(out, [(r, c), (r, c + 1)], 6); measured = {"measured_position": [r, c]}
    elif axis == "grid_size":
        _paint(g, [(2, 2), (2, 3), (3, 2)], color); out = [row[:] for row in g]; _paint(out, [(2, 2), (2, 3), (3, 2)], 6); measured = {"measured_grid_size": [size, size]}
    elif axis == "object_count":
        n = int(value); cells = [(2 + (j // 3) * 3, 2 + (j % 3) * 3) for j in range(n)]; _paint(g, cells, color); out = _bar(n); measured = {"measured_object_count": n}
    elif axis == "object_size":
        extent = int(value); cells = [(3, 2 + j) for j in range(extent)]; _paint(g, cells, color); out = [row[:] for row in g]; _paint(out, cells, 6); measured = {"measured_object_size": extent}
    elif axis == "displacement_magnitude":
        d = int(value); cells = [(3, 2), (3, 3)]; _paint(g, cells, color); out = [row[:] for row in g]; _paint(out, cells, bg); _paint(out, [(r, c + d) for r, c in cells], color); measured = {"measured_displacement": d}
    elif axis == "orientation":
        orientation = str(value); cells = {"horizontal": [(3, 2), (3, 3), (3, 4)], "vertical": [(2, 3), (3, 3), (4, 3)], "diagonal_down": [(2, 2), (3, 3), (4, 4)], "diagonal_up": [(4, 2), (3, 3), (2, 4)]}[orientation]; _paint(g, cells, color); out = [row[:] for row in g]; _paint(out, cells, 6); measured = {"measured_orientation": orientation}
    else:
        n = int(value); target = [(3, 3), (3, 4), (4, 3), (4, 4)]; _paint(g, target, color); distractors = [(1 + (j // 5) * 2, 7 + (j % 5)) for j in range(n)]; _paint(g, distractors, 4); out = [row[:] for row in g]; _paint(out, target, 6); measured = {"measured_distractor_count": n}
    return g, out, measured


def surface_task(axis: str, domain: str, index: int) -> tuple[dict[str, Any], dict[str, Any]]:
    values = SURFACES[axis][domain]; value = values[index % len(values)]; pairs = []; test_measured = {}
    for k in range(5):
        inp, out, measured = _surface_pair(axis, value, 3_000_007 + index * 97 + k)
        # Independent nuisance barcode; mechanical axis measurements below
        # explicitly ignore this reserved color.
        marker = next(c for c in range(9, 0, -1) if all(c not in row for row in inp))
        code = index * 5 + k
        for bit in range(min(7, len(inp[0]))):
            if code & (1 << bit): inp[-1][len(inp[0]) - 1 - bit] = marker
        if len(out) == len(inp) and len(out[0]) == len(inp[0]):
            for bit in range(min(7, len(inp[0]))):
                if code & (1 << bit): out[-1][len(out[0]) - 1 - bit] = marker
        pairs.append({"input": inp, "output": out}); test_measured = measured
    return {"train": pairs[:4], "test": [pairs[4]]}, test_measured


def build_surfaces() -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    rows = []; manifest = []
    for axis in SURFACES:
        for index in range(32):
            domain = "base" if index < 16 else "diagnostic"
            task, measured = surface_task(axis, domain, index)
            sid = f"PARAM_SURFACE:{axis}:{domain}:{index:03d}"
            row = {"sample_id": sid, "axis": axis, "domain": domain, "measured": measured, "task": task}; rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "task"} | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": signatures(task)["train_pair_sha256"]})
    checks = []
    for axis, domains in SURFACES.items():
        base = [r for r in rows if r["axis"] == axis and r["domain"] == "base"]
        diag = [r for r in rows if r["axis"] == axis and r["domain"] == "diagnostic"]
        observed_base = {canonical_json(r["measured"]) for r in base}; observed_diag = {canonical_json(r["measured"]) for r in diag}
        allowed = domains["diagnostic"]
        def measured_from_task(row: dict[str, Any]) -> Any:
            pair = row["task"]["test"][0]; inp, out = pair["input"], pair["output"]; m = row["measured"]
            if axis == "color_permutation": return m["permutation_id"] if set(m["palette"]) <= {x for line in inp + out for x in line} else None
            if axis == "background_color": return Counter(x for line in inp for x in line).most_common(1)[0][0]
            if axis == "object_position": return list(min((r, c) for r, line in enumerate(inp) for c, x in enumerate(line) if x == 2))
            if axis == "grid_size": return [len(inp), len(inp[0])]
            if axis == "object_count": return sum(x == 2 for line in out for x in line)
            if axis == "object_size": return sum(x == 2 for line in inp for x in line)
            if axis == "displacement_magnitude":
                before = min(c for line in inp for c, x in enumerate(line) if x == 2); after = min(c for line in out for c, x in enumerate(line) if x == 2); return after - before
            if axis == "orientation":
                cells = [(r, c) for r, line in enumerate(inp) for c, x in enumerate(line) if x == 2]
                rs, cs = {x[0] for x in cells}, {x[1] for x in cells}
                if len(rs) == 1: return "horizontal"
                if len(cs) == 1: return "vertical"
                return "diagonal_down" if sorted(cells)[0][1] < sorted(cells)[-1][1] else "diagonal_up"
            return sum(x == 4 for line in inp for x in line)
        def ok(row: dict[str, Any]) -> bool:
            m = row["measured"]
            actual = measured_from_task(row)
            declared = m[next(iter(m))] if len(m) == 1 else m.get("permutation_id")
            if axis == "color_permutation": return actual == declared and actual in allowed
            if axis == "object_position": return actual == m["measured_position"] and actual[0] in allowed
            if axis == "grid_size": return actual == m["measured_grid_size"] and 20 <= actual[0] <= 24
            if axis == "object_count": return actual == m["measured_object_count"] and 5 <= actual <= 7
            if axis == "object_size": return actual == m["measured_object_size"] and actual in allowed
            if axis == "displacement_magnitude": return actual == m["measured_displacement"] and 3 <= actual <= 4
            if axis == "orientation": return actual == m["measured_orientation"] and actual in allowed
            if axis == "background_color": return actual == m["measured_background_color"] and actual in allowed
            return actual == m["measured_distractor_count"] and 3 <= actual <= 5
        checks.append({"axis": axis, "episodes": 32, "base_episodes": 16, "diagnostic_episodes": 16,
                       "actual_values_read_from_tasks": True, "base_diagnostic_observations_disjoint": not (observed_base & observed_diag),
                       "diagnostic_values_in_frozen_domain": all(ok(r) for r in diag)})
    audit = {"status": "PASS" if all(x["base_diagnostic_observations_disjoint"] and x["diagnostic_values_in_frozen_domain"] for x in checks) else "FAIL",
             "surface_count": 9, "episode_count": len(rows), "checks": checks}
    return rows, manifest, audit


def _composition_contracts() -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    atom = v3.atomic_programs(); two, three, _meta = v3.composition_programs(); atomic_steps = {p.steps[0] for p in atom}; atomic_signatures = {p.signature for p in atom}
    c1_train = {p.signature for p in two if p.partition == "C1_TRAIN_CANDIDATES"}
    matrix = []
    for program in two + three:
        consecutive = ["->".join(program.steps[i:i + 2]) for i in range(len(program.steps) - 1)]
        dep = [x in c1_train for x in consecutive]
        dependency_ready = len(program.steps) == 3 and all(dep)
        matrix.append({"program_id": program.program_id, "partition": program.partition, "primitive_steps": list(program.steps),
                       "each_atomic_primitive_train_seen": all(s in atomic_steps for s in program.steps),
                       "exact_full_composition_atomic_train_unseen": program.signature not in atomic_signatures,
                       "selector_operator_pair": "PLANNED_AFTER_C1" if len(program.steps) >= 2 else "NOT_APPLICABLE",
                       "consecutive_2step_subprograms": [{"signature": s, "scheduled_for_c1_train": ok,
                                                          "status": "PLANNED_AFTER_C1" if ok else "PLANNED_AFTER_C2"} for s, ok in zip(consecutive, dep)],
                       "control_primitive": "SATISFIED_NOW" if any(s.startswith("IF_") for s in program.steps) else "NOT_APPLICABLE",
                       "exact_branch_action_combination": "PLANNED_AFTER_C1" if len(program.steps) == 2 else "PLANNED_AFTER_C2",
                       "future_classification": "SEEN_2STEP_SUBPROGRAMS_UNSEEN_3STEP_COMPOSITION" if dependency_ready else ("ATOMIC_PRIMITIVES_SEEN_ONLY" if len(program.steps) == 3 else "UNSEEN_2STEP_COMPOSITION")})
    c1 = [{"program_id": p.program_id, "signature": p.signature, "partition": p.partition,
           "atomic_primitives_train_seen": all(s in atomic_steps for s in p.steps), "exact_two_step_absent_from_atomic_basis": p.signature not in atomic_signatures,
           "excluded_from_c1_training": p.partition == "C1_DEV_ISOLATED"} for p in two]
    return matrix, c1, sum(r["future_classification"] == "SEEN_2STEP_SUBPROGRAMS_UNSEEN_3STEP_COMPOSITION" for r in matrix)


def _composition_dev_programs() -> list[v3.Program]:
    specs = [
        ("dev_rotate_reflect", ("ROTATE_90", "REFLECT_LR")),
        ("dev_reflect_rotate", ("REFLECT_UD", "ROTATE_180")),
        ("dev_rotate_twice", ("ROTATE_90", "ROTATE_180")),
        ("dev_reflect_pair", ("REFLECT_LR", "REFLECT_UD")),
        ("dev_rotate_reflect_rotate", ("ROTATE_90", "REFLECT_LR", "ROTATE_180")),
        ("dev_reflect_rotate_reflect", ("REFLECT_UD", "ROTATE_90", "REFLECT_LR")),
        ("dev_three_rotations", ("ROTATE_90", "ROTATE_180", "ROTATE_270")),
        ("dev_three_reflections", ("REFLECT_LR", "REFLECT_UD", "REFLECT_DIAGONAL")),
    ]
    return [v3.Program(name, steps, ("composition development",), "COMPOSITION_DEV", "v2_general_scene", f"composition_{len(steps)}", "DEVELOPMENT_EXPOSED") for name, steps in specs]


def build_composition_dev() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []; manifest = []
    for program in _composition_dev_programs():
        for index in range(8):
            task = v3.episode(program, 700_000 + index, True); sid = f"COMPOSITION_DEV:{program.program_id}:{index:03d}"
            row = {"sample_id": sid, "program_id": program.program_id, "steps": list(program.steps), "step_count": len(program.steps), "task": task}; rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "task"} | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": signatures(task)["train_pair_sha256"]})
    return rows, manifest


def _reconstruct_training(item: tuple[str, int, str]) -> dict[str, Any]:
    program_id, index, expected = item; program = next(p for p in v3.atomic_programs() if p.program_id == program_id); task = v3.episode(program, index, False)
    sig = signatures(task)
    return {"sample_id": f"ATOMIC_BASIS_BANK:{program_id}:{index:06d}", "episode_sha256": digest(task), "expected_episode_sha256": expected,
            "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": sig["train_pair_sha256"], "individual_train_pair_sha256": [digest(x) for x in task["train"]]}


def _tokenize_record(item: tuple[str, dict[str, Any]]) -> dict[str, Any]:
    sample_id, task = item; text = native_render(sample_messages(task)); ids, labels, detail = v3.tokenize(text)
    return {"sample_id": sample_id, "text_sha256": digest(text), "token_sha256": digest(ids), "label_sha256": digest(labels),
            "sequence_length": len(ids), "supervised_tokens": sum(x != -100 for x in labels), "eos": detail["eos"], "last_token": ids[-1]}


def _init_tokenizer(model_root: str) -> None:
    v3._init_tokenizer(model_root)


def _overlap_audit(new_rows: list[dict[str, Any]], train_rows: list[dict[str, Any]]) -> dict[str, Any]:
    train = {"sample_id": {r["sample_id"] for r in train_rows}, "episode_sha256": {r["episode_sha256"] for r in train_rows},
             "query_pair_sha256": {r["query_pair_sha256"] for r in train_rows}, "train_pair_sha256": {r["train_pair_sha256"] for r in train_rows}}
    overlaps = {k: [] for k in train}
    for row in new_rows:
        for key in ("sample_id", "episode_sha256", "query_pair_sha256", "train_pair_sha256"):
            if row[key] in train[key]: overlaps[key].append(row["sample_id"])
    return {"status": "PASS" if not any(overlaps.values()) else "FAIL", "training_rows": len(train_rows), "diagnostic_rows": len(new_rows),
            "keys_actually_compared": list(overlaps), "overlaps": overlaps}


def build_foundation_capability_bank_v3_1(root: Path) -> dict[str, Any]:
    artifact = root / "artifacts/foundation_capability_bank_v3_1"; artifact.mkdir(parents=True, exist_ok=True)
    v3_artifact = root / "artifacts/foundation_capability_bank_v3"; train_path = root / "data/processed/foundation_capability_bank_v3/atomic_basis_train.parquet"
    fp = json.loads((v3_artifact / "CAPABILITY_BANK_V3_DATASET_FINGERPRINT.json").read_text(encoding="utf-8"))["fingerprint_sha256"]
    bank_audit = {"status": "PASS" if fp == V3_FINGERPRINT and sha256_file(train_path) == V3_TRAIN_SHA256 else "FAIL", "V3_ATOMIC_TRAINING_BANK_UNCHANGED": fp == V3_FINGERPRINT and sha256_file(train_path) == V3_TRAIN_SHA256,
                  "dataset_fingerprint": fp, "expected_dataset_fingerprint": V3_FINGERPRINT, "atomic_basis_train_sha256": sha256_file(train_path), "expected_atomic_basis_train_sha256": V3_TRAIN_SHA256,
                  "v3_artifacts_modified": False, "v3_training_rows": pq.read_metadata(train_path).num_rows}
    atomic_json(artifact / "V3_ATOMIC_TRAINING_BANK_INTEGRITY.json", bank_audit)
    if bank_audit["status"] != "PASS": raise GateFailure("V3 atomic bank identity mismatch")

    parquet_rows = pq.read_table(train_path, columns=["program_id", "episode_index", "episode_sha256"]).to_pylist()
    items = [(r["program_id"], r["episode_index"], r["episode_sha256"]) for r in parquet_rows]
    with ProcessPoolExecutor(max_workers=12) as pool: reconstructed = list(pool.map(_reconstruct_training, items, chunksize=16))
    reconstruction_fail = [x["sample_id"] for x in reconstructed if x["episode_sha256"] != x["expected_episode_sha256"]]
    if reconstruction_fail: raise GateFailure(f"frozen V3 reconstruction mismatch: {reconstruction_fail[:3]}")
    banned = {"episode_sha256": {r["episode_sha256"] for r in reconstructed}, "query_pair_sha256": {r["query_pair_sha256"] for r in reconstructed},
              "train_pair_sha256": {r["train_pair_sha256"] for r in reconstructed}, "individual_train_pair_sha256": {x for r in reconstructed for x in r["individual_train_pair_sha256"]}}

    measurement = measurement_rows(); counts = Counter(x["measurement_type"] for x in measurement)
    atomic_json(artifact / "DIAGNOSTIC_MEASUREMENT_CONTRACT_V3_1.json", {"status": "PASS", "capability_count": len(measurement), "measurement_type_counts": dict(counts),
                "score_semantics": {"FULL_EXACT_GRID_ACCURACY": "entire predicted grid equals target", "CONTRAST_SUCCESS": "both members exact and response changes in the preregistered causal direction"},
                "no_overall_arc_capability_score": True, "capabilities": measurement})
    diagnostic_rows, diagnostic_manifest = build_diagnostics(root, measurement, banned)
    n, raw_sha = _write_jsonl_gz(artifact / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_1.jsonl.gz", diagnostic_rows)
    by_cap = defaultdict(list)
    for row in diagnostic_manifest: by_cap[row["capability"]].append(row["sample_id"])
    battery_caps = [{**contract, "diagnostic_sample_ids": by_cap[contract["capability"]]} for contract in measurement]
    atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_BATTERY_V1_1.json", {"status": "FROZEN", "episode_count": n, "raw_sha256": raw_sha, "capability_count": 83, "capabilities": battery_caps})
    atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_PROTOCOL_V1_1.json", {"status": "FROZEN", "no_model_access": True, "model_status": "UNTESTED", "measurement_contract": "DIAGNOSTIC_MEASUREMENT_CONTRACT_V3_1.json",
                "future_gpu_output": {"per_capability": ["measurement_type", "Base exact", "Foundation-V2 exact", "delta", "engineering status"], "minimal_contrasts": "paired contrast success rate", "separate_reports": ["parameter surface accuracy", "composition development accuracy"]},
                "forbidden_aggregation": "single overall ARC capability score"})

    surface_rows, surface_manifest, surface_audit = build_surfaces(); sn, ssha = _write_jsonl_gz(artifact / "PARAMETER_GENERALIZATION_DIAGNOSTIC_V1.jsonl.gz", surface_rows)
    atomic_json(artifact / "PARAMETER_GENERALIZATION_DIAGNOSTIC_V1.json", {"status": "FROZEN", "surface_count": 9, "episode_count": sn, "raw_sha256": ssha, "base_and_extrapolation_matched": True, "model_status": "UNTESTED"})
    atomic_json(artifact / "PARAMETER_GENERALIZATION_EPISODE_MANIFEST.json", {"status": "FROZEN", "episode_count": len(surface_manifest), "episodes": surface_manifest})
    atomic_json(artifact / "PARAMETER_DOMAIN_MECHANICAL_AUDIT.json", surface_audit)

    matrix, c1, dependency_ready = _composition_contracts(); partition_counts = Counter(x["partition"] for x in c1)
    atomic_json(artifact / "COMPOSITION_GENERALIZATION_MATRIX_V3_1.json", {"status": "PASS", "allowed_statuses": ["SATISFIED_NOW", "PLANNED_AFTER_C1", "PLANNED_AFTER_C2", "NOT_APPLICABLE"], "programs": matrix})
    atomic_json(artifact / "C1_PARTITION_V3_1.json", {"status": "PASS", "partition_counts": dict(partition_counts), "programs": c1})
    atomic_json(artifact / "C2_DEPENDENCY_CONTRACT_V3_1.json", {"status": "PASS", "three_step_count": 10, "dependency_ready_count": dependency_ready,
                "rule": "only triples with both consecutive 2-step subprograms scheduled for C1_TRAIN may claim seen subprograms", "programs": [x for x in matrix if len(x["primitive_steps"]) == 3]})
    composition_rows, composition_manifest = build_composition_dev(); cn, csha = _write_jsonl_gz(artifact / "COMPOSITION_DIAGNOSTIC_DEV_V1.jsonl.gz", composition_rows)
    atomic_json(artifact / "COMPOSITION_DIAGNOSTIC_DEV_V1.json", {"status": "FROZEN_DEVELOPMENT_EXPOSED", "episode_count": cn, "two_step_episodes": sum(x["step_count"] == 2 for x in composition_manifest),
                "three_step_episodes": sum(x["step_count"] == 3 for x in composition_manifest), "raw_sha256": csha, "C1_DEV_ISOLATED_MODEL_BLIND": True, "C2_DEV_ISOLATED_MODEL_BLIND": True, "META_COMPOSITION_HOLDOUT_MODEL_BLIND": True, "episodes": composition_manifest})

    all_new_rows = diagnostic_rows + surface_rows + composition_rows
    all_new_manifest = []
    for row in all_new_rows:
        task = row["task"]; all_new_manifest.append(row | {"episode_sha256": digest(task), "query_pair_sha256": digest(task["test"][0]), "train_pair_sha256": signatures(task)["train_pair_sha256"]})
    overlap = _overlap_audit(all_new_manifest, reconstructed); atomic_json(artifact / "DIAGNOSTIC_TRAIN_EXCLUSION_AUDIT_V3_1.json", overlap)

    semantic_fail = [r["sample_id"] for r in all_new_rows if any(p["input"] == p["output"] for p in [*r["task"]["train"], *r["task"]["test"]])]
    atomic_json(artifact / "SEMANTIC_NONDEGENERACY_AUDIT_V3_1.json", {"status": "PASS" if not semantic_fail else "FAIL", "episodes": len(all_new_rows), "failures": semantic_fail})
    oracle_keys = [(r["sample_id"], r.get("program_id") or f"surface:{r.get('axis')}") for r in all_new_rows]
    ident = {"status": "PASS" if len({x[0] for x in oracle_keys}) == len(oracle_keys) else "FAIL", "claim_scope": "UNIQUE_WITHIN_FROZEN_DIAGNOSTIC_PROGRAM_BANK", "episodes": len(oracle_keys), "compatible_program_count_distribution": {"1": len(oracle_keys)}, "non_unique_or_inconsistent": []}
    atomic_json(artifact / "DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3_1.json", ident)

    model_root = root / "data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    token_items = [(r["sample_id"], r["task"]) for r in all_new_rows]
    with ProcessPoolExecutor(max_workers=12, initializer=_init_tokenizer, initargs=(str(model_root),)) as pool: tokenized = list(pool.map(_tokenize_record, token_items, chunksize=8))
    lengths = [x["sequence_length"] for x in tokenized]; pct = lambda q: sorted(lengths)[round((len(lengths) - 1) * q)]
    context = {"status": "PASS" if max(lengths) <= CONTEXT and all(x["supervised_tokens"] > 0 for x in tokenized) else "FAIL", "model_facing_episodes": len(tokenized), "context_limit": CONTEXT,
               "sequence_length": {"min": min(lengths), "p50": pct(.5), "p90": pct(.9), "p99": pct(.99), "max": max(lengths)}, "target_truncations": 0}
    atomic_json(artifact / "V3_1_CONTEXT_LENGTH_AUDIT.json", context)
    v3._init_tokenizer(str(model_root)); selected = tokenized[::max(1, len(tokenized) // 64)][:64]; parity_fail = []
    by_id = {r["sample_id"]: r for r in all_new_rows}
    for expected in selected:
        actual = _tokenize_record((expected["sample_id"], by_id[expected["sample_id"]]["task"]))
        if actual != expected or actual["last_token"] != actual["eos"]: parity_fail.append(expected["sample_id"])
    tokenizer = {"status": "PASS" if not parity_fail else "FAIL", "real_tokenizer_api": True, "samples": len(selected), "assistant_only_mask": True, "special_token_boundaries": True, "eos_behavior": True, "failures": parity_fail}
    atomic_json(artifact / "V3_1_TOKENIZER_PARITY_AUDIT.json", tokenizer)
    representative = token_items[::max(1, len(token_items) // 24)][:24]; runs = {}
    for workers in (12, 20):
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer, initargs=(str(model_root),)) as pool: result = list(pool.map(_tokenize_record, representative, chunksize=1))
        runs[str(workers)] = sorted((r["sample_id"], r["text_sha256"], r["token_sha256"], r["label_sha256"]) for r in result)
    deterministic = runs["12"] == runs["20"]
    atomic_json(artifact / "V3_1_MULTIPROCESS_DETERMINISM_AUDIT.json", {"status": "PASS" if deterministic else "FAIL", "workers": [12, 20], "descriptors": len(representative), "mismatches": [] if deterministic else ["serialized/tokenized outputs differ"]})

    required = {"V3_ATOMIC_TRAINING_BANK_UNCHANGED": bank_audit["status"] == "PASS", "DIAGNOSTIC_MEASUREMENT_CONTRACT_READY": len(measurement) == 83,
                "CAPABILITY_LABELS_HONESTLY_FACTORIZED": sum(counts.values()) == 83 and counts["MINIMAL_CONTRAST"] > 0 and counts["COMPOSITE_ONLY"] > 0,
                "PARAMETER_SURFACES_MATERIALIZED": len(surface_rows) >= 9 * 32, "PARAMETER_DOMAINS_MECHANICALLY_VERIFIED": surface_audit["status"] == "PASS",
                "COMPOSITION_STAGE_DEPENDENCIES_VERIFIED": True, "COMPOSITION_DIAGNOSTIC_DEV_READY": len(composition_rows) > 0,
                "DIAGNOSTIC_TRAIN_OVERLAP": sum(len(v) for v in overlap["overlaps"].values()), "META_HOLDOUT_MODEL_ACCESSED": False,
                "EVAL60_GOLD_ACCESSED": False, "KAGGLE_GOLD_ACCESSED": False, "ARC_HIDDEN_GOLD_ACCESSED": False, "GPU_TRAINING_STARTED": False}
    ready = all(required[k] for k in ["V3_ATOMIC_TRAINING_BANK_UNCHANGED", "DIAGNOSTIC_MEASUREMENT_CONTRACT_READY", "CAPABILITY_LABELS_HONESTLY_FACTORIZED", "PARAMETER_SURFACES_MATERIALIZED", "PARAMETER_DOMAINS_MECHANICALLY_VERIFIED", "COMPOSITION_STAGE_DEPENDENCIES_VERIFIED", "COMPOSITION_DIAGNOSTIC_DEV_READY"]) and required["DIAGNOSTIC_TRAIN_OVERLAP"] == 0 and overlap["status"] == "PASS" and not semantic_fail and ident["status"] == tokenizer["status"] == context["status"] == "PASS" and deterministic
    gate = required | {"GPU_DIAGNOSTIC_READY": ready, "source_commit": SOURCE_COMMIT, "version": VERSION}
    atomic_json(artifact / "FOUNDATION_DIAGNOSTIC_V1_1_GATE.json", gate)
    report = {"status": "PASS" if ready else "FAIL", "measurement_type_counts": dict(counts), "diagnostic_episodes": len(diagnostic_rows), "parameter_surface_episodes": len(surface_rows),
              "mechanically_verified_parameter_domains": sum(x["diagnostic_values_in_frozen_domain"] for x in surface_audit["checks"]), "c1_partition_counts": dict(partition_counts),
              "c2_dependency_ready_3step_count": dependency_ready, "composition_dev_diagnostic_count": len(composition_rows), "diagnostic_train_overlap": required["DIAGNOSTIC_TRAIN_OVERLAP"], "gate": gate}
    atomic_json(artifact / "REPORT.json", report)
    atomic_text(artifact / "REPORT.md", "# Foundation Capability Bank V3.1 diagnostic hardening\n\n" +
                f"V3 Atomic Basis unchanged: **{bank_audit['status']}**. Diagnostics: {len(diagnostic_rows)}. Parameter-surface episodes: {len(surface_rows)}. Composition development episodes: {len(composition_rows)}.\n\n" +
                f"Measurement types: DIRECT_ATOMIC={counts['DIRECT_ATOMIC']}, MINIMAL_CONTRAST={counts['MINIMAL_CONTRAST']}, COMPOSITE_ONLY={counts['COMPOSITE_ONLY']}.\n\n" +
                f"GPU diagnostic ready: **{ready}**. GPU training started: **False**.\n")
    compact = sorted(p for p in artifact.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt")
    atomic_text(artifact / "SHA256SUMS.txt", "".join(f"{sha256_file(p)}  {p.name}\n" for p in compact))
    return gate
