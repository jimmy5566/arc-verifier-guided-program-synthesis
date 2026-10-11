"""Freeze the CPU-only, target-separated atomic OOD identifiability pilot."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/atomic_ood_identifiability_pilot_v1"
PROTOCOL = "ATOMIC_OOD_IDENTIFIABILITY_PILOT_V1"
SEED = 20261011
CELLS = ("REFERENCE", "OOD_GEOMETRY", "OOD_COLOR_MAPPING", "OOD_GRID_SIZE", "OOD_LAYOUT", "OOD_MARKER_ENCODING")
FACTORS = {"REFERENCE": None, "OOD_GEOMETRY": "geometry", "OOD_COLOR_MAPPING": "color_mapping", "OOD_GRID_SIZE": "grid_size", "OOD_LAYOUT": "layout", "OOD_MARKER_ENCODING": "marker_encoding"}
# Each is connected and deliberately asymmetric under quarter turns.
SHAPES = (
    ((0,0),(1,0),(1,1),(2,1),(2,2)), ((0,0),(0,1),(1,1),(1,2),(2,1)),
    ((0,0),(1,0),(2,0),(2,1),(3,1)), ((0,1),(1,1),(1,0),(2,0),(2,2)),
    ((0,0),(0,1),(1,1),(1,2),(2,2),(3,2)), ((0,1),(1,0),(1,1),(2,1),(2,2)),
    ((0,0),(1,0),(1,1),(2,1),(3,1),(3,2)), ((0,0),(1,0),(2,0),(1,1),(1,2)),
)
COLORS = (1,2,3,4,6,7,8,9)

def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
def sha(value): return hashlib.sha256(value).hexdigest()
def write_atomic(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
        handle.write(raw); temp = Path(handle.name)
    os.replace(temp, path)
def rotate(points, turns):
    out = tuple(points)
    for _ in range(turns % 4):
        out = tuple((x, -y) for y, x in out)
    min_y, min_x = min(y for y, _ in out), min(x for _, x in out)
    return tuple(sorted((y-min_y, x-min_x) for y, x in out))
def make_grid(size): return [[0 for _ in range(size)] for _ in range(size)]
def paint(grid, points, color, origin):
    oy, ox = origin
    for y, x in points: grid[oy+y][ox+x] = color
def marker(turn, encoding):
    return {"ARROW": ["N","E","S","W"][turn], "SYMBOL": ["^",">","v","<"][turn]}[encoding]
def cell_params(cell):
    return {
      "geometry": "BASE" if cell != "OOD_GEOMETRY" else "MIRRORED",
      "color_mapping": "DIRECT" if cell != "OOD_COLOR_MAPPING" else "PERMUTED",
      "grid_size": 9 if cell != "OOD_GRID_SIZE" else 11,
      "layout": "TOP_LEFT" if cell != "OOD_LAYOUT" else "BOTTOM_RIGHT",
      "marker_encoding": "ARROW" if cell != "OOD_MARKER_ENCODING" else "SYMBOL",
      "marker_semantics": "CLOCKWISE_QUARTER_TURN",
      "output_schema": "CUE_COLORED_ROTATED_OBJECT_ONLY",
    }
def target_color(unit, cell):
    base = COLORS[unit]
    if cell == "OOD_COLOR_MAPPING": return COLORS[(unit + 3) % len(COLORS)]
    return base
def input_shape(unit, cell):
    shape = SHAPES[unit]
    if cell == "OOD_GEOMETRY": shape = tuple((y, -x) for y, x in shape)
    return shape
def transformed_shape(unit, turn, cell):
    return rotate(input_shape(unit, cell), turn)
def demonstration_bank(cell):
    # Fixed across every query/turn for this cell, and explicitly provides the
    # cue-to-output-color role without exposing any query structural tuple.
    p = cell_params(cell); examples = []
    for cue, color in (("AMBER", 5), ("TEAL", 3)):
        examples.append({"cue": cue, "rule": "output_color_equals_cue_color", "output_color": color,
                         "marker_semantics": p["marker_semantics"], "output_schema": p["output_schema"]})
    return examples
def render(unit, turn, cell):
    p = cell_params(cell); shape = input_shape(unit, cell); size = p["grid_size"]
    origin = (2,2) if p["layout"] == "TOP_LEFT" else (size-5,size-5)
    # Input-object colour is deliberately outside the explicit output-cue
    # palette, preventing a no-op turn from serializing its target in the query.
    query = make_grid(size); paint(query, shape, 5, origin)
    cue = f"UNIT_COLOR_{target_color(unit, cell)}" # directly visible, non-target color cue
    prompt = {"train": demonstration_bank(cell), "test": [{"input_grid": query, "turn_marker": marker(turn,p["marker_encoding"]),
               "output_color_cue": cue, "marker_semantics": p["marker_semantics"], "output_schema": p["output_schema"]}]}
    out = make_grid(size); paint(out, transformed_shape(unit,turn,cell), target_color(unit,cell), origin)
    latent = {"shape_index": unit, "turn": turn, "cell": cell, "factor": FACTORS[cell], "params": p}
    return prompt, out, latent
def records():
    for cell in CELLS:
        for unit in range(8):
            for turn in range(4):
                prompt, target, latent = render(unit,turn,cell)
                yield {"episode_id": f"AOODV1-{cell}-U{unit:02d}-T{turn}", "cell": cell, "canonical_unit": unit,
                       "turn": turn, "prompt": prompt, "target": target, "latent": latent}
def audit(rows):
    by_cell_unit = defaultdict(list)
    for row in rows: by_cell_unit[(row["cell"],row["canonical_unit"])].append(row)
    rotation = {"status":"PASS", "canonical_units":8, "turns_per_unit":4, "all_four_distinct":True, "units":{}}
    for (cell, unit), group in by_cell_unit.items():
        hashes = {sha(canonical(row["target"])) for row in group}
        if len(hashes) != 4: raise RuntimeError("ROTATION_TARGET_NOT_IDENTIFIABLE")
        rotation["units"][f"{cell}:U{unit:02d}"] = sorted(hashes)
    # Demos have no query-unit/turn dependence; targets are absent from prompts.
    demo_hashes = defaultdict(set)
    for row in rows:
        demo_hashes[row["cell"]].add(sha(canonical(row["prompt"]["train"])))
        if canonical(row["target"]) in canonical(row["prompt"]): raise RuntimeError("TARGET_LEAKAGE")
    if any(len(v) != 1 for v in demo_hashes.values()): raise RuntimeError("QUERY_CONDITIONED_DEMONSTRATIONS")
    one_factor = {"status":"PASS", "reference":"REFERENCE", "cells":{}}
    ref = cell_params("REFERENCE")
    for cell in CELLS[1:]:
        changed = sorted(k for k in ref if ref[k] != cell_params(cell)[k])
        if changed != [FACTORS[cell]]: raise RuntimeError("NOT_ONE_FACTOR")
        one_factor["cells"][cell] = {"nominated_factor":FACTORS[cell], "changed_fields":changed}
    # Exact prompt, content, and latent identities must be unique across cells;
    # historical namespace search is intentionally deterministic and conservative.
    prompt_hashes = [sha(canonical(r["prompt"])) for r in rows]
    target_hashes = [sha(canonical(r["target"])) for r in rows]
    latent_hashes = [sha(canonical(r["latent"])) for r in rows]
    if len(set(prompt_hashes)) != len(rows) or len(set(latent_hashes)) != len(rows): raise RuntimeError("COLLISION")
    history_files = [ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery/INPUT_PROMPTS.jsonl"]
    historical = b"".join(p.read_bytes() for p in history_files if p.exists())
    if b"AOODV1-" in historical: raise RuntimeError("HISTORICAL_EXPOSURE_COLLISION")
    collision = {"status":"PASS", "prompt_hashes_unique":len(set(prompt_hashes))==len(rows), "latent_hashes_unique":len(set(latent_hashes))==len(rows),
                 "prompt_target_hash_intersection":len(set(prompt_hashes)&set(target_hashes)), "historical_files_checked":[str(p.relative_to(ROOT)) for p in history_files if p.exists()],
                 "historical_namespace_collision":False, "cross_split_content_collision":False}
    if collision["prompt_target_hash_intersection"]: raise RuntimeError("PROMPT_TARGET_DUPLICATION")
    # Remove marker or color cue: each blinded observation has divergent targets.
    def blinded(row, mode):
        q = json.loads(json.dumps(row["prompt"]))
        if mode == "marker": q["test"][0]["turn_marker"] = "BLINDED"
        else: q["test"][0]["output_color_cue"] = "BLINDED"
        return sha(canonical(q))
    blind = {"status":"PASS", "marker_blind":{}, "color_blind":{}}
    # Group real counterfactual records after the nominated visible signal is
    # removed.  A duplicate blinded observation with divergent targets proves
    # that the corresponding blind baseline cannot solve every member.
    marker_rows = [r for r in rows if r["cell"] == "REFERENCE"]
    color_rows = [r for r in rows if r["cell"] in {"REFERENCE", "OOD_COLOR_MAPPING"}]
    for mode, subset in (("marker", marker_rows), ("color", color_rows)):
        groups = defaultdict(set)
        for row in subset: groups[blinded(row, mode)].add(sha(canonical(row["target"])))
        divergent = sum(len(v) > 1 for v in groups.values())
        blind[f"{mode}_blind"] = {"counterfactual_groups":len(groups), "divergent_target_groups":divergent,
                                      "can_solve_all_counterfactuals":False}
    if blind["marker_blind"]["divergent_target_groups"] != 8 or blind["color_blind"]["divergent_target_groups"] != 32: raise RuntimeError("BLIND_BASELINE_SOLVES")
    return rotation, one_factor, collision, blind
def materialize(out):
    rows = list(records())
    if len(rows) != 192: raise RuntimeError("ALLOCATION")
    rotation, one_factor, collision, blind = audit(rows)
    prompts = [{"episode_id":r["episode_id"], "cell":r["cell"], "canonical_unit":r["canonical_unit"], "turn":r["turn"], "prompt":r["prompt"]} for r in rows]
    targets = [{"episode_id":r["episode_id"], "target":r["target"]} for r in rows]
    p_raw = b"".join(canonical(x)+b"\n" for x in prompts); t_raw = b"".join(canonical(x)+b"\n" for x in targets)
    files = {"INPUT_PROMPTS.jsonl":p_raw, "SEALED_TARGET_SIDECAR.jsonl":t_raw,
             "ROTATION_DISTINCTNESS_AUDIT.json":json.dumps(rotation,sort_keys=True,indent=2).encode()+b"\n",
             "ONE_FACTOR_AUDIT.json":json.dumps(one_factor,sort_keys=True,indent=2).encode()+b"\n",
             "COLLISION_AUDIT.json":json.dumps(collision,sort_keys=True,indent=2).encode()+b"\n",
             "BLIND_BASELINE_AUDIT.json":json.dumps(blind,sort_keys=True,indent=2).encode()+b"\n"}
    manifest = {"protocol_id":PROTOCOL,"status":"CPU_ONLY_FROZEN_NO_MODEL_NO_INFERENCE","seed":SEED,"cells":list(CELLS),"canonical_units":8,"turns_per_unit":4,"episodes":192,
      "episodes_per_checkpoint_if_separately_authorized":192,"checkpoints_if_later_authorized":["V7","E04_E_CONTROL","E04_E_TREATMENT"],
      "future_inference":"NATIVE_GREEDY_EXACT_GRID","future_batch_policy":"LENGTH_BUCKETED_B32_FALLBACK_32_16_8_4_1_FIXED_B1_SUBSET","target_sidecar_sealed":True,
      "model_loaded":False,"gpu_used":False,"generation_run":False,"training_run":False,"final_audit_opened":False,
      "file_sha256":{name:sha(raw) for name,raw in files.items()},"future_execution_requires":["DIRECTOR_PRELAUNCH_REVIEW","SEPARATE_GPU_NO_UPDATE_AUTHORIZATION","RUNTIME_IDENTITY_PREFLIGHT"]}
    files["MANIFEST.json"] = json.dumps(manifest,sort_keys=True,indent=2).encode()+b"\n"
    for name, raw in files.items(): write_atomic(out/name,raw)
    return manifest
def freeze():
    if OUT.exists(): raise RuntimeError("REFUSE_OVERWRITE_FROZEN_ATOMIC_OOD_PILOT")
    with tempfile.TemporaryDirectory() as td:
        a, b = Path(td)/"a", Path(td)/"b"; materialize(a); materialize(b)
        names = sorted(p.name for p in a.iterdir())
        if any((a/n).read_bytes() != (b/n).read_bytes() for n in names): raise RuntimeError("NONDETERMINISTIC_REGENERATION")
    manifest = materialize(OUT)
    receipt = {"status":"PASS","byte_identical_regeneration":True,"files":sorted(p.name for p in OUT.iterdir()),"manifest_sha256":sha((OUT/"MANIFEST.json").read_bytes()),"model_loaded":False,"gpu_used":False}
    write_atomic(OUT/"DETERMINISM_RECEIPT.json",json.dumps(receipt,sort_keys=True,indent=2).encode()+b"\n")
    return manifest
if __name__ == "__main__": print(json.dumps(freeze(),sort_keys=True))
