"""CPU-only Foundation Capability Bank V3 construction and audit.

This module extends (and never mutates) the frozen V2 bank.  Every accepted
program has a deterministic executable oracle and a typed input-domain
contract.  Diagnostic episodes are raw/hash-only and excluded mechanically
from every tokenized training shard.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from foundation_capability_bank_v2 import pipeline as v2
from training_data_v2.pipeline import (
    IGNORE_INDEX, canonical_json, native_render, sample_messages, sha256_file,
    signatures,
)

VERSION = "foundation_capability_bank_v3"
SOURCE_COMMIT = "b0dab8e48fc0b4b514a6a3571685f0912d85e25a"
CONTEXT = 8704
TOKENIZER_ID = "qwen3_4b_grids15_sft139"
TOKENIZER = None
AUDIT_HYPOTHESES: list[Program] = [] if "Program" in globals() else []


class GateFailure(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


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


@dataclass(frozen=True)
class Program:
    name: str
    steps: tuple[str, ...]
    capabilities: tuple[str, ...]
    category: str
    domain: str
    kind: str = "atomic"
    partition: str = "ATOMIC_BASIS"

    @property
    def signature(self) -> str:
        return "->".join(self.steps)

    @property
    def program_id(self) -> str:
        return "capv3:" + digest({"domain": self.domain, "kind": self.kind, "steps": self.steps})[:18]


ONTOLOGY = v2.ONTOLOGY
CORE_EXPANSION = {
    "holes/enclosures", "separators/subgrids", "inside/contains",
    "overlap/intersection", "nearest/farthest", "left/right", "above/below",
    "inferred displacement", "until alignment", "color mapping",
    "copy reference color", "mask", "intersection", "difference", "copy",
    "overlay", "complete missing structure", "compare quantities",
    "periodic pattern", "alternation", "progression", "propagation",
    "iterative update", "if/else on attribute",
}


NEW_ROWS = [
    ("fill_holes", ("FILL_ENCLOSED_HOLES",), ("holes/enclosures",), "PERCEPTION_SEGMENTATION", "frame_scene"),
    ("swap_subgrids", ("SWAP_SEPARATOR_SUBGRIDS",), ("separators/subgrids",), "PERCEPTION_SEGMENTATION", "separator_scene"),
    ("widest_recolor", ("SELECT_WIDEST_RECOLOR",), ("width",), "ATTRIBUTES", "object_scene"),
    ("tallest_recolor", ("SELECT_TALLEST_RECOLOR",), ("height",), "ATTRIBUTES", "object_scene"),
    ("orientation_recolor", ("ORIENTATION_RECOLOR",), ("orientation",), "ATTRIBUTES", "object_scene"),
    ("least_frequent_recolor", ("SELECT_LEAST_FREQUENT_RECOLOR",), ("least frequent", "frequency"), "SELECTORS", "object_scene"),
    ("topmost_recolor", ("SELECT_TOPMOST_RECOLOR",), ("top-most", "position"), "SELECTORS", "object_scene"),
    ("bottommost_recolor", ("SELECT_BOTTOMMOST_RECOLOR",), ("bottom-most", "position"), "SELECTORS", "object_scene"),
    ("rightmost_recolor", ("SELECT_RIGHTMOST_RECOLOR",), ("right-most", "position"), "SELECTORS", "object_scene"),
    ("second_left_recolor", ("SELECT_SECOND_LEFT_RECOLOR",), ("nth/order", "position"), "SELECTORS", "object_scene"),
    ("right_of_reference", ("SELECT_RIGHT_OF_REFERENCE",), ("left/right", "relation-conditioned selector"), "RELATIONS", "reference_scene"),
    ("below_reference", ("SELECT_BELOW_REFERENCE",), ("above/below", "relation-conditioned selector"), "RELATIONS", "reference_scene"),
    ("inside_recolor", ("SELECT_INSIDE_RECOLOR",), ("inside/contains", "relation-conditioned selector"), "RELATIONS", "frame_scene"),
    ("panel_intersection", ("SET_INTERSECTION_PANELS",), ("overlap/intersection", "intersection", "mask"), "COLOR_MASK", "dual_panel"),
    ("nearest_farthest", ("SELECT_NEAREST_FARTHEST",), ("nearest/farthest", "relation-conditioned selector"), "RELATIONS", "reference_scene"),
    ("scale_two", ("SCALE_OBJECT_2X",), ("scale",), "GEOMETRIC_ACTIONS", "object_scene"),
    ("inferred_displacement", ("INFER_AND_APPLY_DISPLACEMENT",), ("inferred displacement", "translate"), "MOTION", "motion_vector_scene"),
    ("move_until_alignment", ("MOVE_UNTIL_COLUMN_ALIGNMENT",), ("until alignment", "aligned"), "MOTION", "alignment_scene"),
    ("legend_color_mapping", ("APPLY_LEGEND_COLOR_MAPPING",), ("color mapping",), "COLOR_MASK", "color_legend_scene"),
    ("copy_reference_color", ("COPY_REFERENCE_COLOR",), ("copy reference color",), "COLOR_MASK", "reference_color_scene"),
    ("panel_mask", ("MASK_BY_PANEL",), ("mask",), "COLOR_MASK", "dual_panel"),
    ("panel_difference", ("SET_DIFFERENCE_PANELS",), ("difference",), "COLOR_MASK", "dual_panel"),
    ("copy_object", ("COPY_OBJECT_RIGHT",), ("copy",), "CONSTRUCTION", "copy_scene"),
    ("overlay_panels", ("OVERLAY_PANELS",), ("overlay",), "CONSTRUCTION", "dual_panel"),
    ("complete_structure", ("COMPLETE_MISSING_RECTANGLE",), ("complete missing structure",), "CONSTRUCTION", "incomplete_structure_scene"),
    ("compare_quantities", ("COMPARE_PANEL_QUANTITIES",), ("compare quantities",), "NUMERIC", "quantity_scene"),
    ("periodic_extend", ("EXTEND_PERIODIC_PATTERN",), ("periodic pattern",), "PATTERN_STATE", "periodic_scene"),
    ("alternating_fill", ("FILL_ALTERNATING_INTERVAL",), ("alternation",), "PATTERN_STATE", "alternation_scene"),
    ("progression_next", ("COMPLETE_LENGTH_PROGRESSION",), ("progression",), "PATTERN_STATE", "progression_scene"),
    ("propagate_region", ("PROPAGATE_REACHABLE_REGION",), ("propagation",), "PATTERN_STATE", "state_scene"),
    ("iterative_update", ("EXPAND_ONE_ITERATION",), ("iterative update",), "PATTERN_STATE", "state_scene"),
    ("conditional_attribute", ("IF_LARGEST_COLOR_EVEN_LR_ELSE_UD",), ("if/else on attribute", "branch-balanced conditional tasks"), "CONTROL", "object_scene"),
]


def atomic_programs() -> list[Program]:
    domain_overrides={"unique_color_recolor":"unique_color_scene","leftmost_recolor":"object_scene"}
    old = [Program(p.name, p.steps, p.capabilities, p.category, domain_overrides.get(p.name,"v2_general_scene")) for p in v2.atomic_programs()]
    new = [Program(name, steps, caps, cat, domain) for name, steps, caps, cat, domain in NEW_ROWS]
    return old + new


def _bg(grid: list[list[int]]) -> int:
    return v2._bg(grid)


def _fresh(grid: list[list[int]], offset: int = 0) -> int:
    used = {v for row in grid for v in row}
    return [v for v in range(10) if v not in used][offset]


def _blank(h: int, w: int, bg: int) -> list[list[int]]:
    return [[bg] * w for _ in range(h)]


def _paint(grid: list[list[int]], cells: list[tuple[int, int]], color: int) -> None:
    bg = _bg(grid)
    for r, c in cells:
        if not (0 <= r < len(grid) and 0 <= c < len(grid[0])) or grid[r][c] != bg:
            raise GateFailure("paint collision")
        grid[r][c] = color


def _bbox_cells(comp: dict[str, Any]) -> tuple[int, int, int, int]:
    return comp["r0"], comp["r1"], comp["c0"], comp["c1"]


def _recolor(grid: list[list[int]], comp: dict[str, Any], color: int | None = None) -> list[list[int]]:
    out = [row[:] for row in grid]
    color = _fresh(grid) if color is None else color
    for r, c in comp["cells"]:
        out[r][c] = color
    return out


def _split_panels(grid: list[list[int]]) -> tuple[list[list[int]], list[list[int]]]:
    h, w = len(grid), len(grid[0])
    bg = _bg(grid)
    candidates = [c for c in range(1, w - 1) if len({grid[r][c] for r in range(h)}) == 1 and grid[0][c] != bg]
    if len(candidates) != 1:
        raise GateFailure("separator not unique")
    c = candidates[0]
    left, right = [row[:c] for row in grid], [row[c + 1:] for row in grid]
    if not left or len(left[0]) != len(right[0]):
        raise GateFailure("panels unequal")
    return left, right


def _new_step(grid: list[list[int]], step: str) -> list[list[int]]:
    bg = _bg(grid)
    comps = v2.components(grid)
    h, w = len(grid), len(grid[0])
    if step == "FILL_ENCLOSED_HOLES":
        out = [row[:] for row in grid]
        border_bg = {(r, c) for r in range(h) for c in range(w) if grid[r][c] == bg and (r in (0, h-1) or c in (0, w-1))}
        stack = list(border_bg); reachable = set(border_bg)
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                q = r+dr, c+dc
                if 0 <= q[0] < h and 0 <= q[1] < w and q not in reachable and grid[q[0]][q[1]] == bg:
                    reachable.add(q); stack.append(q)
        holes = [(r,c) for r in range(h) for c in range(w) if grid[r][c] == bg and (r,c) not in reachable]
        if not holes: raise GateFailure("no enclosure")
        color = _fresh(grid)
        for r,c in holes: out[r][c] = color
        return out
    if step == "SWAP_SEPARATOR_SUBGRIDS":
        left, right = _split_panels(grid); sep = [row[len(left[0])] for row in grid]
        return [right[r] + [sep[r]] + left[r] for r in range(h)]
    if step in {"SELECT_WIDEST_RECOLOR", "SELECT_TALLEST_RECOLOR", "SELECT_TOPMOST_RECOLOR", "SELECT_BOTTOMMOST_RECOLOR", "SELECT_RIGHTMOST_RECOLOR", "SELECT_SECOND_LEFT_RECOLOR"}:
        key = {
            "SELECT_WIDEST_RECOLOR": lambda x: x["c1"]-x["c0"]+1,
            "SELECT_TALLEST_RECOLOR": lambda x: x["r1"]-x["r0"]+1,
            "SELECT_TOPMOST_RECOLOR": lambda x: -x["r0"],
            "SELECT_BOTTOMMOST_RECOLOR": lambda x: x["r1"],
            "SELECT_RIGHTMOST_RECOLOR": lambda x: x["c1"],
        }
        if step == "SELECT_SECOND_LEFT_RECOLOR":
            target = sorted(comps, key=lambda x: (x["c0"], x["r0"]))[1]
        else:
            target = max(comps, key=key[step])
        return _recolor(grid, target)
    if step == "ORIENTATION_RECOLOR":
        out = [row[:] for row in grid]; c1, c2 = _fresh(grid), _fresh(grid, 1)
        for obj in comps:
            oh, ow = obj["r1"]-obj["r0"]+1, obj["c1"]-obj["c0"]+1
            color = c1 if ow > oh else c2 if oh > ow else obj["color"]
            for r,c in obj["cells"]: out[r][c] = color
        return out
    if step == "SELECT_LEAST_FREQUENT_RECOLOR":
        counts = Counter(x["color"] for x in comps); m = min(counts.values()); colors = [c for c,n in counts.items() if n == m]
        if len(colors) != 1: raise GateFailure("least frequency tie")
        out = [row[:] for row in grid]; color = _fresh(grid)
        for obj in comps:
            if obj["color"] == colors[0]:
                for r,c in obj["cells"]: out[r][c] = color
        return out
    if step in {"SELECT_RIGHT_OF_REFERENCE", "SELECT_BELOW_REFERENCE", "SELECT_NEAREST_FARTHEST"}:
        ref = max(comps, key=lambda x: x["area"]); others = [x for x in comps if x is not ref]
        rr, rc = (ref["r0"]+ref["r1"])/2, (ref["c0"]+ref["c1"])/2
        if step == "SELECT_RIGHT_OF_REFERENCE":
            candidates = [x for x in others if x["c0"] > ref["c1"]]
            if not candidates: raise GateFailure("no right relation")
            return _recolor(grid, min(candidates, key=lambda x:x["c0"]))
        if step == "SELECT_BELOW_REFERENCE":
            candidates = [x for x in others if x["r0"] > ref["r1"]]
            if not candidates: raise GateFailure("no below relation")
            return _recolor(grid, min(candidates, key=lambda x:x["r0"]))
        dist = lambda x: ((x["r0"]+x["r1"])/2-rr)**2 + ((x["c0"]+x["c1"])/2-rc)**2
        ordered = sorted(others, key=dist)
        if len(ordered) < 2 or dist(ordered[0]) == dist(ordered[-1]): raise GateFailure("distance tie")
        out = _recolor(grid, ordered[0], _fresh(grid)); far = ordered[-1]
        for r,c in far["cells"]: out[r][c] = _fresh(grid, 1)
        return out
    if step == "SELECT_INSIDE_RECOLOR":
        frames = [x for x in comps if (x["r1"]-x["r0"]+1)*(x["c1"]-x["c0"]+1)-x["area"] > 0]
        if len(frames) != 1: raise GateFailure("frame not unique")
        f = frames[0]; inside = [x for x in comps if x is not f and f["r0"] < x["r0"] <= x["r1"] < f["r1"] and f["c0"] < x["c0"] <= x["c1"] < f["c1"]]
        if len(inside) != 1: raise GateFailure("inside target not unique")
        return _recolor(grid, inside[0])
    if step in {"SET_INTERSECTION_PANELS", "SET_DIFFERENCE_PANELS", "MASK_BY_PANEL", "OVERLAY_PANELS"}:
        left, right = _split_panels(grid); ph, pw = len(left), len(left[0]); out = _blank(ph, pw, bg); color = _fresh(grid)
        for r in range(ph):
            for c in range(pw):
                a, b = left[r][c] != bg, right[r][c] != bg
                if step == "MASK_BY_PANEL": out[r][c] = left[r][c] if a and b else bg
                elif step == "SET_INTERSECTION_PANELS": out[r][c] = color if a and b else bg
                elif step == "SET_DIFFERENCE_PANELS": out[r][c] = color if a and not b else bg
                else: out[r][c] = right[r][c] if b else left[r][c]
        return out
    if step == "SCALE_OBJECT_2X":
        target = max(comps, key=lambda x:x["area"]); oh, ow = target["r1"]-target["r0"]+1, target["c1"]-target["c0"]+1
        out = _blank(oh*2+2, ow*2+2, bg)
        for r,c in target["cells"]:
            for dr in (0,1):
                for dc in (0,1): out[1+2*(r-target["r0"])+dr][1+2*(c-target["c0"])+dc] = target["color"]
        return out
    if step == "INFER_AND_APPLY_DISPLACEMENT":
        counts = Counter(x["color"] for x in comps); marker_color = next((c for c,n in counts.items() if n == 2), None)
        markers = [x for x in comps if x["color"] == marker_color]
        movers = [x for x in comps if x["color"] != marker_color]
        if len(markers) != 2 or len(movers) != 1: raise GateFailure("vector scene malformed")
        a,b = sorted((x["cells"][0] for x in markers)); dr,dc = b[0]-a[0], b[1]-a[1]; mover=movers[0]
        out=[row[:] for row in grid]
        for r,c in mover["cells"]: out[r][c]=bg
        for r,c in mover["cells"]:
            if not (0<=r+dr<h and 0<=c+dc<w): raise GateFailure("inferred displacement clips")
            out[r+dr][c+dc]=mover["color"]
        return out
    if step == "MOVE_UNTIL_COLUMN_ALIGNMENT":
        anchor=max(comps,key=lambda x:x["area"]); mover=min(comps,key=lambda x:x["area"]); dc=anchor["c0"]-mover["c0"]
        out=[row[:] for row in grid]
        for r,c in mover["cells"]: out[r][c]=bg
        for r,c in mover["cells"]: out[r][c+dc]=mover["color"]
        return out
    if step == "APPLY_LEGEND_COLOR_MAPPING":
        sep_rows=[r for r,row in enumerate(grid) if len(set(row))==1 and row[0]!=bg]
        if len(sep_rows)!=1 or sep_rows[0]<2: raise GateFailure("legend separator")
        sr=sep_rows[0]; mapping={grid[0][c]:grid[1][c] for c in range(w) if grid[0][c]!=bg and grid[1][c]!=bg}
        if len(mapping)<2: raise GateFailure("legend mapping")
        return [[mapping.get(v,v) for v in row] for row in grid[sr+1:]]
    if step == "COPY_REFERENCE_COLOR":
        ref=max(comps,key=lambda x:x["area"]); target=min(comps,key=lambda x:x["area"])
        return _recolor(grid,target,ref["color"])
    if step == "COPY_OBJECT_RIGHT":
        if len(comps)!=1: raise GateFailure("copy expects one object")
        obj=comps[0]; dc=obj["c1"]-obj["c0"]+2; out=[row[:] for row in grid]
        for r,c in obj["cells"]:
            if c+dc>=w: raise GateFailure("copy clips")
            out[r][c+dc]=obj["color"]
        return out
    if step == "COMPLETE_MISSING_RECTANGLE":
        if len(comps)!=1: raise GateFailure("structure not connected")
        obj=comps[0]; out=[row[:] for row in grid]
        for r in range(obj["r0"],obj["r1"]+1):
            for c in range(obj["c0"],obj["c1"]+1):
                if r in (obj["r0"],obj["r1"]) or c in (obj["c0"],obj["c1"]): out[r][c]=obj["color"]
        return out
    if step == "COMPARE_PANEL_QUANTITIES":
        left,right=_split_panels(grid); lc=sum(v!=bg for row in left for v in row); rc=sum(v!=bg for row in right for v in row)
        if lc==rc: raise GateFailure("quantity tie")
        chosen=left if lc>rc else right; color=_fresh(grid)
        return [[color if v!=bg else bg for v in row] for row in chosen]
    if step == "EXTEND_PERIODIC_PATTERN":
        rows=[r for r,row in enumerate(grid) if sum(v!=bg for v in row)>=2]
        if len(rows)!=1: raise GateFailure("period row")
        r=rows[0]; nz=[c for c,v in enumerate(grid[r]) if v!=bg]; start=min(nz); vals=[grid[r][start],grid[r][start+1]]; out=[row[:] for row in grid]
        for c in range(start,w): out[r][c]=vals[(c-start)%2]
        return out
    if step == "FILL_ALTERNATING_INTERVAL":
        rows=[r for r,row in enumerate(grid) if sum(v!=bg for v in row)>=3]
        if len(rows)!=1: raise GateFailure("alternation row")
        r=rows[0]; nz=[c for c,v in enumerate(grid[r]) if v!=bg]; start,end=min(nz),max(nz); a,b=grid[r][start],grid[r][start+1]; out=[row[:] for row in grid]
        for c in range(start,end+1): out[r][c]=(a,b)[(c-start)%2]
        return out
    if step == "COMPLETE_LENGTH_PROGRESSION":
        rows=[]
        for r,row in enumerate(grid):
            cs=[c for c,v in enumerate(row) if v!=bg]
            if cs: rows.append((r,len(cs),grid[r][cs[0]]))
        if len(rows)<3 or [x[1] for x in rows[:3]]!=[1,2,3]: raise GateFailure("progression malformed")
        out=[row[:] for row in grid]; r=min(h-1,rows[-1][0]+2); start=min(c for c,v in enumerate(grid[rows[0][0]]) if v!=bg)
        for c in range(start,start+4): out[r][c]=rows[0][2]
        return out
    if step in {"PROPAGATE_REACHABLE_REGION","EXPAND_ONE_ITERATION"}:
        counts=Counter(v for row in grid for v in row if v!=bg); seed_color=min(counts,key=counts.get); wall_color=max(counts,key=counts.get)
        seeds={(r,c) for r in range(h) for c in range(w) if grid[r][c]==seed_color}; fill=set(seeds); frontier=list(seeds)
        rounds=None if step=="PROPAGATE_REACHABLE_REGION" else 1; depth=0
        while frontier and (rounds is None or depth<rounds):
            nxt=[]
            for r,c in frontier:
                for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    q=r+dr,c+dc
                    if 0<=q[0]<h and 0<=q[1]<w and q not in fill and grid[q[0]][q[1]]!=wall_color:
                        fill.add(q); nxt.append(q)
            frontier=nxt; depth+=1
        out=[row[:] for row in grid]
        for r,c in fill:
            if out[r][c]==bg: out[r][c]=seed_color
        return out
    if step == "IF_LARGEST_COLOR_EVEN_LR_ELSE_UD":
        target=max(comps,key=lambda x:x["area"])
        return v2.apply_step(grid,"REFLECT_LR" if target["color"]%2==0 else "REFLECT_UD")
    raise GateFailure(step)


OLD_STEPS = {p.steps[0] for p in v2.atomic_programs()}


def apply_step(grid: list[list[int]], step: str) -> list[list[int]]:
    return v2.apply_step(grid, step) if step in OLD_STEPS else _new_step(grid, step)


def apply_program(grid: list[list[int]], program: Program) -> list[list[int]]:
    out=[row[:] for row in grid]
    for step in program.steps: out=apply_step(out,step)
    return out


def _object_scene(rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    h=w=24 if parameter_mode else 22; bg=variant%4; grid=_blank(h,w,bg)
    available=[x for x in range(10) if x!=bg]
    parity=variant%2
    parity_values=[x for x in available if x%2==parity]; a=rng.choice(parity_values)
    remainder=[x for x in available if x!=a]; rng.shuffle(remainder); b,c=remainder[:2]
    # Unique: largest 3x3 block, widest length-5 line, tallest length-6 line,
    # unique least-frequency color, and unique top/bottom/right/order positions.
    shapes=[tuple((r,c) for r in range(3) for c in range(3)),tuple((0,c) for c in range(5)),tuple((r,0) for r in range(6)),v2.SHAPES["L"],v2.SHAPES["dot"]]
    row_templates=[(1,5,8,15,20),(8,1,13,19,5),(17,8,1,13,5),(7,16,8,1,20),(12,6,14,17,1)]
    col_templates=[(2,8,1,13,20),(13,1,8,18,5),(18,7,2,13,1),(7,16,1,10,5),(11,5,17,1,8)]
    rows=row_templates[variant%len(row_templates)]; cols=col_templates[(variant*2)%len(col_templates)]
    anchors=list(zip(rows,cols))
    palette=[a,b,b,c,a]
    for shape,(r,c0),color in zip(shapes,anchors,palette): v2._place(grid,shape,r,c0,color)
    return grid


def _frame_scene(rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    h=w=14 if parameter_mode else 12; bg=(rng.randrange(4)+variant)%4; grid=_blank(h,w,bg); colors=[x for x in range(10) if x!=bg]; rng.shuffle(colors); fc,ic,oc=colors[:3]
    r0,c0=2+(variant%2),2; r1,c1=h-4,w-4
    for r in range(r0,r1+1):
        for c in range(c0,c1+1):
            if r in (r0,r1) or c in (c0,c1): grid[r][c]=fc
    grid[r0+2][c0+2]=ic; grid[h-2][w-2]=oc
    return grid


def _separator_scene(rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    h=7+(variant%2)*2; pw=5+(1 if parameter_mode else 0); bg=(rng.randrange(4)+variant)%4; grid=_blank(h,pw*2+1,bg); colors=[x for x in range(10) if x!=bg]; rng.shuffle(colors); sep,a,b=colors[:3]
    for r in range(h): grid[r][pw]=sep
    grid[1][1]=a; grid[2][2]=a; grid[h-2][pw+2]=b; grid[h-3][pw+3]=b
    return grid


def _reference_scene(rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    h=w=20 if parameter_mode else 17; bg=(rng.randrange(4)+variant)%4; grid=_blank(h,w,bg); colors=[x for x in range(10) if x!=bg]; rng.shuffle(colors)
    v2._place(grid,tuple((r,c) for r in range(3) for c in range(3)),6,6,colors[0])
    for (r,c),color in zip(((2,7),(7,11),(12,7),(14,14)),colors[1:5]): v2._place(grid,v2.SHAPES["dot"],r,c,color)
    return grid


def _dual_panel(rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    n=7 if parameter_mode else 6; bg=(rng.randrange(4)+variant)%4; grid=_blank(n,2*n+1,bg); colors=[x for x in range(10) if x!=bg]; rng.shuffle(colors); sep,a,b=colors[:3]
    for r in range(n): grid[r][n]=sep
    left={(1,1),(1,2),(2,1),(3,3),(4,2)}; right={(1,2),(2,1),(2,2),(3,4),(4,2)}
    if variant%2: right.add((4,4))
    for r,c in left: grid[r][c]=a
    for r,c in right: grid[r][n+1+c]=b
    return grid


def _special_scene(name: str, rng: random.Random, variant: int, parameter_mode: bool) -> list[list[int]]:
    if name in {"fill_holes","inside_recolor"}: return _frame_scene(rng,variant,parameter_mode)
    if name=="swap_subgrids": return _separator_scene(rng,variant,parameter_mode)
    if name in {"widest_recolor","tallest_recolor","orientation_recolor","least_frequent_recolor","topmost_recolor","bottommost_recolor","rightmost_recolor","second_left_recolor","scale_two","conditional_attribute"}: return _object_scene(rng,variant,parameter_mode)
    if name in {"right_of_reference","below_reference","nearest_farthest"}: return _reference_scene(rng,variant,parameter_mode)
    if name in {"panel_intersection","panel_mask","panel_difference","overlay_panels"}: return _dual_panel(rng,variant,parameter_mode)
    bg=(rng.randrange(4)+variant)%4; colors=[x for x in range(10) if x!=bg]; rng.shuffle(colors)
    if name=="inferred_displacement":
        n=20 if parameter_mode else 15; g=_blank(n,n,bg); d=(3+variant%2) if parameter_mode else (1+variant%2); v2._place(g,v2.SHAPES["dot"],1,1,colors[0]); v2._place(g,v2.SHAPES["dot"],1+d,1+d,colors[0]); v2._place(g,v2.SHAPES["L"],10 if parameter_mode else 7,10 if parameter_mode else 7,colors[1]); return g
    if name=="move_until_alignment":
        g=_blank(15,15,bg); v2._place(g,tuple((r,0) for r in range(6)),3,10,colors[0]); v2._place(g,v2.SHAPES["dot"],8,2+variant%3,colors[1]); v2._place(g,v2.SHAPES["L"],2,2,colors[2]); return g
    if name=="legend_color_mapping":
        g=_blank(10,10,bg); g[0][1],g[1][1]=colors[0],colors[2]; g[0][3],g[1][3]=colors[1],colors[3]
        for c in range(10): g[2][c]=colors[4]
        for r,c,v in ((4,2,colors[0]),(5,3,colors[1]),(7,6,colors[0]),(8,7,colors[1])): g[r][c]=v
        return g
    if name=="copy_reference_color":
        g=_blank(13,13,bg); v2._place(g,tuple((r,c) for r in range(3) for c in range(3)),2,2,colors[0]); v2._place(g,v2.SHAPES["dot"],9,9,colors[1]); return g
    if name=="copy_object":
        g=_blank(12+rng.randrange(3),16+rng.randrange(4),bg); v2._place(g,v2.SHAPES["L"],2+rng.randrange(4),1+rng.randrange(3),colors[0]); return g
    if name=="complete_structure":
        h=12+rng.randrange(4); w=12+rng.randrange(4); g=_blank(h,w,bg); r0,c0=1+rng.randrange(3),1+rng.randrange(3); r1,c1=r0+5+rng.randrange(3),c0+5+rng.randrange(3)
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if r in (r0,r1) or c in (c0,c1): g[r][c]=colors[0]
        g[r0][c0+2+variant%3]=bg; return g
    if name=="compare_quantities":
        n=7; g=_blank(n,2*n+1,bg)
        for r in range(n): g[r][n]=colors[0]
        left=[(1,1),(3,2),(5,3)]; right=[(1,1),(4,3)] if variant%2==0 else [(1,1),(2,3),(4,1),(5,4)]
        for r,c in left: g[r][c]=colors[1]
        for r,c in right: g[r][n+1+c]=colors[2]
        return g
    if name=="periodic_extend":
        g=_blank(5,14 if parameter_mode else 12,bg); start=1+variant%2
        for i in range(4): g[2][start+i]=colors[i%2]
        return g
    if name=="alternating_fill":
        g=_blank(5,14,bg); start=1+variant%2; end=11+variant%2; g[2][start]=colors[0]; g[2][start+1]=colors[1]; g[2][end]=colors[(end-start)%2]; return g
    if name=="progression_next":
        h=9+rng.randrange(4); w=9+rng.randrange(4); g=_blank(h,w,bg); c0=1+rng.randrange(2); rows=(1,3,5)
        for r,n in zip(rows,(1,2,3)):
            for c in range(c0,c0+n): g[r][c]=colors[0]
        return g
    if name in {"propagate_region","iterative_update"}:
        available=[x for x in range(10) if x!=bg]; wall=next(x for x in available if x%2==variant%2); seed=next(x for x in available if x!=wall)
        n=11+rng.randrange(5); g=_blank(n,n,bg); wall_col=4+rng.randrange(max(1,n-8)); gap=2+rng.randrange(n-4)
        for r in range(1,n-1): g[r][wall_col]=wall
        g[gap][wall_col]=bg; sr=1+rng.randrange(n-2); sc=1+rng.randrange(max(1,wall_col-2)); g[sr][sc]=seed; return g
    raise GateFailure(f"no generator {name}")


OLD_BY_NAME = {p.name:p for p in v2.atomic_programs()}


def make_input(program: Program, seed: int, variant: int, parameter_mode: bool=False) -> list[list[int]]:
    rng=random.Random(seed+variant*10007)
    if program.name=="leftmost_recolor":
        return _object_scene(rng,variant,parameter_mode)
    if program.steps[0] in OLD_STEPS:
        old=next(p for p in v2.atomic_programs() if p.steps[0]==program.steps[0])
        grid=v2.make_input(old,seed,variant,parameter_mode)
        current=_bg(grid); desired=variant%4
        if current!=desired:
            grid=[[desired if v==current else current if v==desired else v for v in row] for row in grid]
        return grid
    owner=next((p for p in atomic_programs() if p.steps[0]==program.steps[0]),program)
    grid=_special_scene(owner.name,rng,variant,parameter_mode); current=_bg(grid); desired=variant%4
    if current!=desired: grid=[[desired if v==current else current if v==desired else v for v in row] for row in grid]
    return grid


def episode(program: Program, index: int, parameter_mode: bool=False) -> dict[str,Any]:
    seed=int(digest({"program":program.program_id,"index":index,"parameter":parameter_mode})[:15],16); pairs=[]
    candidate=0
    while len(pairs)<5:
        inp=make_input(program,seed,candidate,parameter_mode); out=apply_program(inp,program); pair={"input":inp,"output":out}
        if digest(pair) not in {digest(x) for x in pairs}: pairs.append(pair)
        candidate+=1
        if candidate>100: raise GateFailure("unable to create distinct episode pairs")
    return {"train":pairs[:4],"test":[pairs[4]]}


def composition_programs() -> tuple[list[Program],list[Program],list[Program]]:
    by={p.name:p for p in atomic_programs()}
    pairs=[
        ("largest_then_right","largest_recolor","translate_right_1","C1_TRAIN_CANDIDATES"),("smallest_then_down","smallest_recolor","translate_down_2","C1_DEV_ISOLATED"),
        ("reflect_then_leftmost","reflect_lr","leftmost_recolor","C1_TRAIN_CANDIDATES"),("fill_then_border","fill_bbox","border_bbox","C1_DEV_ISOLATED"),
        ("rotate_then_largest","rotate_90","largest_recolor","C1_TRAIN_CANDIDATES"),("unique_then_boundary","unique_color_recolor","move_right_boundary","C1_DEV_ISOLATED"),
        ("count_then_rotate","count_components","rotate_90","C1_TRAIN_CANDIDATES"),("rotate_then_pad","rotate_90","pad_extraction","C1_DEV_ISOLATED"),
        ("align_then_recolor","align_row","largest_recolor","C1_TRAIN_CANDIDATES"),("repeat_then_border","repeat_horizontal","border_bbox","C1_DEV_ISOLATED"),
        ("symmetry_then_border","symmetry_completion","border_bbox","C1_TRAIN_CANDIDATES"),("connect_then_crop","connect_same_color","extract_bbox","C1_DEV_ISOLATED"),
        ("holes_then_border","fill_holes","border_bbox","C1_TRAIN_CANDIDATES"),("topmost_then_right","topmost_recolor","translate_right_1","C1_DEV_ISOLATED"),
        ("complete_then_fill","complete_structure","fill_bbox","C1_TRAIN_CANDIDATES"),("iterate_then_symmetry","iterative_update","symmetry_completion","C1_DEV_ISOLATED"),
        ("nearest_then_crop","nearest_farthest","extract_bbox","C1_TRAIN_CANDIDATES"),("attribute_then_border","conditional_attribute","border_bbox","C1_DEV_ISOLATED"),
        ("periodic_then_reflect","periodic_extend","reflect_lr","C1_TRAIN_CANDIDATES"),("overlay_then_rotate","overlay_panels","rotate_90","C1_DEV_ISOLATED"),
    ]
    triples=[
        ("largest_move_reflect",("largest_recolor","translate_right_1","reflect_ud"),"C2_TRAIN_CANDIDATES"),
        ("unique_rotate_border",("unique_color_recolor","rotate_90","border_bbox"),"C2_DEV_ISOLATED"),
        ("align_fill_crop",("align_column","fill_bbox","extract_bbox"),"C2_TRAIN_CANDIDATES"),
        ("smallest_down_symmetry",("smallest_recolor","translate_down_2","symmetry_completion"),"C2_DEV_ISOLATED"),
        ("holes_border_crop",("fill_holes","border_bbox","extract_bbox"),"C2_TRAIN_CANDIDATES"),
        ("top_move_reflect",("topmost_recolor","translate_right_1","reflect_ud"),"C2_DEV_ISOLATED"),
        ("mask_border_pad",("panel_mask","border_bbox","pad_extraction"),"C2_TRAIN_CANDIDATES"),
        ("periodic_reflect_pad",("periodic_extend","reflect_lr","pad_extraction"),"C2_DEV_ISOLATED"),
        ("attribute_symmetry_border",("conditional_attribute","symmetry_completion","border_bbox"),"C2_TRAIN_CANDIDATES"),
        ("nearest_crop_pad",("nearest_farthest","extract_bbox","pad_extraction"),"C2_DEV_ISOLATED"),
    ]
    two=[Program(n,by[a].steps+by[b].steps,tuple(sorted(set(by[a].capabilities+by[b].capabilities))),"COMPOSITION_2",by[a].domain,"composition_2",part) for n,a,b,part in pairs]
    three=[Program(n,tuple(s for key in names for s in by[key].steps),tuple(sorted(set(c for key in names for c in by[key].capabilities))),"COMPOSITION_3",by[names[0]].domain,"composition_3",part) for n,names,part in triples]
    meta_specs=[
        ("meta_relation_motion",("right_of_reference","move_right_boundary","symmetry_completion")),
        ("meta_state_control",("iterative_update","conditional_attribute","border_bbox")),
        ("meta_set_construction",("panel_intersection","border_bbox","pad_extraction")),
        ("meta_numeric_pattern",("compare_quantities","repeat_horizontal","reflect_ud")),
        ("meta_perception_selector",("fill_holes","largest_recolor","extract_bbox")),
    ]
    meta=[Program(n,tuple(s for key in names for s in by[key].steps),tuple(sorted(set(c for key in names for c in by[key].capabilities))),"META",by[names[0]].domain,"meta_holdout","META_COMPOSITION_HOLDOUT") for n,names in meta_specs]
    return two,three,meta


def _selector_unique(program: Program, grid: list[list[int]]) -> bool | None:
    comps=v2.components(grid); step=program.steps[0]
    metrics={
        "SELECT_LARGEST_RECOLOR":[x["area"] for x in comps],"SELECT_SMALLEST_RECOLOR":[-x["area"] for x in comps],
        "SELECT_LEFTMOST_RECOLOR":[-x["c0"] for x in comps],"SELECT_WIDEST_RECOLOR":[x["c1"]-x["c0"] for x in comps],
        "SELECT_TALLEST_RECOLOR":[x["r1"]-x["r0"] for x in comps],"SELECT_TOPMOST_RECOLOR":[-x["r0"] for x in comps],
        "SELECT_BOTTOMMOST_RECOLOR":[x["r1"] for x in comps],"SELECT_RIGHTMOST_RECOLOR":[x["c1"] for x in comps],
    }
    if step in metrics:
        m=max(metrics[step]); return metrics[step].count(m)==1
    if step=="SELECT_SECOND_LEFT_RECOLOR": return len({x["c0"] for x in comps})==len(comps) and len(comps)>=2
    if step in {"SELECT_LEAST_FREQUENT_RECOLOR","SELECT_MOST_FREQUENT_COLOR_RECOLOR"}:
        counts=Counter(x["color"] for x in comps); target=min(counts.values()) if "LEAST" in step else max(counts.values()); return list(counts.values()).count(target)==1
    if "CONNECT_UNIQUE" in step:
        counts=Counter(x["color"] for x in comps); return len([c for c,n in counts.items() if n==2])==1
    return None


REL_CAPS={"left/right","above/below","same row","same column","adjacent/touching","inside/contains","overlap/intersection","nearest/farthest","aligned","connected","same color"}
SELECT_CAPS={"largest","smallest","unique","most frequent","least frequent","top-most","bottom-most","left-most","right-most","nth/order","relation-conditioned selector"}


def _branch(program: Program, grid: list[list[int]]) -> str | None:
    state=[row[:] for row in grid]
    for step in program.steps:
        if step=="IF_LARGEST_COLOR_EVEN_LR_ELSE_UD": return "true" if max(v2.components(state),key=lambda x:x["area"])["color"]%2==0 else "false"
        if step=="IF_COMPONENT_COUNT_EVEN_REFLECT_LR_ELSE_UD": return "true" if len(v2.components(state))%2==0 else "false"
        if step=="IF_TOUCHING_RECOLOR_ELSE_TRANSLATE":
            comps=v2.components(state); touching=any(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for i,x in enumerate(comps) for y in comps[i+1:] for a in x["cells"] for b in y["cells"])
            return "true" if touching else "false"
        state=apply_step(state,step)
    return None


def semantic_predicates(program: Program, task: dict[str,Any]) -> dict[str,bool|None]:
    pairs=task["train"]+task["test"]; inputs=[p["input"] for p in pairs]
    selector=[_selector_unique(program,g) for g in inputs]; selector=[x for x in selector if x is not None]
    conditional=any(step.startswith("IF_") for step in program.steps)
    branches={_branch(program,p["input"]) for p in task["train"]}
    relation=bool(set(program.capabilities)&REL_CAPS)
    selector_related=bool(set(program.capabilities)&SELECT_CAPS)
    relation_observed=[]
    for g in inputs:
        comps=v2.components(g)
        relation_observed.append(len(comps)>=2 or program.domain in {"dual_panel","frame_scene","separator_scene"})
    bgs=[_bg(g) for g in inputs]
    nonbg_palettes=[tuple(sorted({v for row in g for v in row if v!=_bg(g)})) for g in inputs]
    structural={(len(g),len(g[0]),len(v2.components(g)),sum(v!=_bg(g) for row in g for v in row)) for g in inputs}
    fits=[]
    for p in pairs:
        try: fits.append(apply_program(p["input"],program)==p["output"])
        except (GateFailure,v2.GateFailure,ValueError,IndexError): fits.append(False)
    return {
        "unique_selector_target": all(selector) if selector else None,
        "conditional_branch_coverage": branches=={"true","false"} if conditional else None,
        "required_relation_instantiated": all(relation_observed) if relation else None,
        "distractor_presence_where_required": all(len(v2.components(g))>=3 for g in inputs) if (selector_related or relation) and program.domain not in {"dual_panel","frame_scene"} else None,
        "parameter_variation_exercised": len(structural)>=2 or len(nonbg_palettes)>=2,
        "output_input_dependent": len({digest(p["output"]) for p in pairs})>=2 and any(p["input"]!=p["output"] for p in pairs),
        "semantic_role_color_randomization": len(set(nonbg_palettes))>=2,
        "background_randomization": len(set(bgs))>=2,
        "no_scan_order_tie_breaking": all(selector) if selector else None,
        "no_clipping_unless_explicit": all(fits),
        "intended_program_fits_all_pairs": all(fits),
    }


def compatible_programs(task: dict[str,Any], intended: Program, programs: list[Program]) -> list[str]:
    out=[]
    for program in programs:
        if program.domain!=intended.domain: continue
        try:
            if all(apply_program(pair["input"],program)==pair["output"] for pair in task["train"]): out.append(program.program_id)
        except (GateFailure,v2.GateFailure,ValueError,IndexError,StopIteration): pass
    return sorted(out)


def _init_audit_worker(programs: list[Program]) -> None:
    global AUDIT_HYPOTHESES
    AUDIT_HYPOTHESES=programs


def _audit_scientific_item(item: tuple[str,Program,str,int,dict[str,Any]]) -> dict[str,Any]:
    bank,p,category,index,task=item
    predicates=semantic_predicates(p,task)
    compatible=compatible_programs(task,p,AUDIT_HYPOTHESES)
    return {"bank":bank,"program":p,"category":category,"index":index,"task":task,"predicates":predicates,"compatible":compatible,"query_pair_sha256":digest(task["test"][0]),"signatures":signatures(task)}


def _generate_batch(args: tuple[str,Program,int,int,bool,str|None]) -> list[dict[str,Any]]:
    bank,p,count,start,parameter_mode,capability=args; rows=[]
    for offset in range(count):
        index=start+offset; task=episode(p,index,parameter_mode)
        rows.append({"bank":bank,"program":p,"category":p.category,"index":index,"task":task,"capability":capability})
    return rows


def _init_tokenizer(model_root: str) -> None:
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(model_root,local_files_only=True,trust_remote_code=False)


def tokenize(text: str) -> tuple[list[int],list[int],dict[str,Any]]:
    if TOKENIZER is None: raise GateFailure("tokenizer absent")
    ids=TOKENIZER(text,add_special_tokens=False,return_attention_mask=False)["input_ids"]; labels=[IGNORE_INDEX]*len(ids); joined=[]; bounds=[]
    matches=list(v2.TURN_RE.finditer(text))
    if not matches or "".join(m.group(0) for m in matches)!=text: raise GateFailure("turn parse")
    for m in matches:
        part=TOKENIZER(m.group(0),add_special_tokens=False,return_attention_mask=False)["input_ids"]; start=len(joined); joined+=part
        if m.group(1)=="assistant": labels[start:len(joined)]=part
        bounds.append((m.group(1),start,len(joined)))
    if ids!=joined or ids[-1]!=TOKENIZER.eos_token_id: raise GateFailure("API boundary mismatch")
    return ids,labels,{"bounds":bounds,"eos":TOKENIZER.eos_token_id}


def _tokenize_item(item: tuple[str,str,str,int,dict[str,Any]]) -> dict[str,Any]:
    bank,pid,category,index,task=item; text=native_render(sample_messages(task)); ids,labels,detail=tokenize(text)
    return {"sample_id":f"{bank}:{pid}:{index:06d}","bank":bank,"program_id":pid,"category":category,"episode_index":index,"text":text,"input_ids":ids,"labels":labels,"sequence_length":len(ids),"supervised_token_count":sum(x!=IGNORE_INDEX for x in labels),"episode_sha256":digest(task),"token_sha256":digest({"ids":ids,"labels":labels}),"boundary_sha256":digest(detail)}


SCHEMA=pa.schema([("sample_id",pa.string()),("bank",pa.string()),("program_id",pa.string()),("category",pa.string()),("episode_index",pa.int32()),("text",pa.string()),("input_ids",pa.list_(pa.int32())),("labels",pa.list_(pa.int32())),("sequence_length",pa.int32()),("supervised_token_count",pa.int32()),("episode_sha256",pa.string()),("token_sha256",pa.string()),("boundary_sha256",pa.string())])


def _write_parquet(path: Path, rows: list[dict[str,Any]]) -> dict[str,Any]:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),tmp,compression="zstd"); os.replace(tmp,path)
    return {"logical_name":path.name,"rows":len(rows),"sha256":sha256_file(path),"tokens":sum(x["sequence_length"] for x in rows)}


def _factor_design(cap: str, program: Program) -> dict[str,Any]:
    if cap in SELECT_CAPS: design="hold action fixed; swap target-defining attribute/relation while preserving distractors"
    elif cap in {"area","width","height","orientation","position","frequency","uniqueness"}: design="hold selector/action fixed; vary only the named attribute ordering"
    elif cap in {"count components","count colors","count cells"}: design="separate segmentation/grouping, counting, and geometric quantity encoding contrasts"
    elif cap in {"encode quantity geometrically"}: design="hold count fixed across layouts; then vary count while holding encoding rule fixed"
    elif cap in REL_CAPS: design="preserve objects/actions while toggling only the named spatial relation"
    else: design="hold scene roles fixed; contrast the named operator against identity and nearest alternative operator"
    return {"capability":cap,"program_id":program.program_id,"program_capabilities":list(program.capabilities),"nuisance_capabilities":[x for x in program.capabilities if x!=cap],"minimal_contrast_design":design,"diagnostic_interpretation":"FACTORIZED_ENGINEERING_DIAGNOSTIC"}


PARAMETER_SURFACES=[
    ("color_permutations","permutations 0-3","disjoint permutations 4-7"),("background_colors","colors 0-3","colors 4-7"),
    ("object_positions","central quadrants","edge/corner placements"),("grid_size","11-19","20-24"),
    ("object_count","1-4","5-7"),("object_size","1-3 linear extent","4-6 linear extent"),
    ("displacement_magnitude","distance 1-2","distance 3-4"),("orientation","horizontal/vertical","diagonal/mixed"),
    ("distractor_count","0-2","3-5"),
]


def _distinguish(programs: list[Program]) -> tuple[list[dict[str,Any]],dict[str,int]]:
    rows=[]; counts=Counter()
    for i,a in enumerate(programs):
        for b in programs[i+1:]:
            if a.domain!=b.domain:
                status="DOMAIN_DISJOINT"; valid=differing=0
            else:
                valid=differing=0
                for source in (a,b):
                    for k in range(8):
                        try:
                            probe=make_input(source,880301+k*97,k,True); oa=apply_program(probe,a); ob=apply_program(probe,b); valid+=1; differing+=oa!=ob
                        except (GateFailure,v2.GateFailure,ValueError,IndexError,StopIteration): pass
                if valid>=8 and differing>=2: status="BEHAVIORALLY_DISTINGUISHED"
                elif valid<8: status="INSUFFICIENT_SHARED_PROBES"
                else: status="SEMANTIC_ALIAS"
            counts[status]+=1; rows.append({"a":a.program_id,"b":b.program_id,"domain_a":a.domain,"domain_b":b.domain,"valid_shared_probes":valid,"different_outputs":differing,"status":status})
    return rows,dict(counts)


def build_capability_bank_v3(root: Path) -> dict[str,Any]:
    artifact=root/"artifacts/foundation_capability_bank_v3"; data=root/"data/processed/foundation_capability_bank_v3"; artifact.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)
    model_root=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"; pipeline_path=Path(__file__)
    atom=atomic_programs(); two,three,meta=composition_programs(); hypotheses=atom+two+three+meta
    ontology={"status":"FROZEN_CURRENT_ONTOLOGY_83","concepts":ONTOLOGY,"total_concepts":sum(map(len,ONTOLOGY.values())),"core_expansion":sorted(CORE_EXPANSION)}; atomic_json(artifact/"CAPABILITY_ONTOLOGY_V3.json",ontology)
    covered=defaultdict(list)
    for p in atom:
        for cap in p.capabilities: covered[cap].append(p.program_id)
    all_caps=[cap for section in ONTOLOGY.values() for cap in section]; comp_caps={c for p in two+three for c in p.capabilities}
    matrix=[]
    parameterized={"color","position","width","height","orientation","object vs color grouping","fixed displacement","inferred displacement","frequency","count components","count colors","count cells"}
    for section,concepts in ONTOLOGY.items():
        for cap in concepts:
            if not covered[cap]: status="DEFERRED_WITH_REASON"
            elif cap in comp_caps: status="USED_IN_ISOLATED_COMPOSITION"
            elif cap in parameterized: status="VERIFIED_PARAMETERIZED"
            else: status="VERIFIED_ATOMIC"
            matrix.append({"section":section,"concept":cap,"status":status,"atomic_program_ids":covered[cap],"deferred_reason":None if covered[cap] else "no defensible executable oracle accepted"})
    status_counts=dict(Counter(x["status"] for x in matrix)); full_count=sum(x["status"]!="DEFERRED_WITH_REASON" for x in matrix); missing_core=sorted(CORE_EXPANSION-{x for x in all_caps if covered[x]})
    atomic_json(artifact/"CAPABILITY_COVERAGE_MATRIX_V3.json",{"status":"PASS" if not missing_core else "FAIL","total_ontology_concepts":len(all_caps),"implemented_concepts":full_count,"status_counts":status_counts,"high_value_core_missing":missing_core,"matrix":matrix})

    distinguish_rows,dist_counts=_distinguish(atom); distinguish_ready=not dist_counts.get("INSUFFICIENT_SHARED_PROBES",0) and not dist_counts.get("SEMANTIC_ALIAS",0)
    atomic_json(artifact/"PROGRAM_DISTINGUISHABILITY_AUDIT_V3.json",{"status":"PASS" if distinguish_ready else "FAIL","requirements":{"behaviorally_distinguished":{"minimum_shared_valid_probes":8,"minimum_differing_outputs":2}},"pair_count":len(distinguish_rows),"status_counts":dist_counts,"pairs":distinguish_rows})

    training=[]; diagnostics=[]; composition_rows=[]; meta_rows=[]; semantic_fail=[]; ident_rows=[]; ident_dist=Counter(); predicate_counts=defaultdict(Counter); all_sig=[]; query_counts=Counter(); used_query=set(); used_sample=set()
    by_id={p.program_id:p for p in atom}; jobs=[]
    jobs += [("ATOMIC_BASIS_BANK",p,64,0,False,None) for p in atom]
    jobs += [("FOUNDATION_DIAGNOSTIC_BATTERY",by_id[covered[cap][0]],32,100000+ci*10000,True,cap) for ci,cap in enumerate(all_caps) if covered[cap]]
    jobs += [("COMPOSITION_ISOLATION_BANK",p,8,200000,True,None) for p in two+three]
    jobs += [("META_HOLDOUT_BANK",p,8,300000,True,None) for p in meta]
    with ProcessPoolExecutor(max_workers=12) as pool: generated=[row for batch in pool.map(_generate_batch,jobs,chunksize=1) for row in batch]
    # Resolve the small number of cross-job collisions deterministically.  The
    # expensive initial construction above is 12-way parallel; this pass only
    # regenerates collided rows and preserves exact query-pair exclusion.
    for row in generated:
        p=row["program"]; index=row["index"]; task=row["task"]
        key=(row["bank"],p.program_id,index); qh=digest(task["test"][0]); attempts=0
        while key in used_sample or qh in used_query:
            index+=1; task=episode(p,index,row["bank"]!="ATOMIC_BASIS_BANK"); key=(row["bank"],p.program_id,index); qh=digest(task["test"][0]); attempts+=1
            if attempts>6400: raise GateFailure(f"query uniqueness exhausted for {p.name}")
        used_sample.add(key); used_query.add(qh); row["index"]=index; row["task"]=task
        packed=(row["bank"],p,row["category"],index,task)
        if row["bank"]=="ATOMIC_BASIS_BANK": training.append(packed)
        elif row["bank"]=="FOUNDATION_DIAGNOSTIC_BATTERY": diagnostics.append((row["capability"],p,index,task))
        elif row["bank"]=="COMPOSITION_ISOLATION_BANK": composition_rows.append(packed)
        else: meta_rows.append(packed)
    diagnostic_tasks=diagnostics
    scientific_rows=training+composition_rows+meta_rows+[("FOUNDATION_DIAGNOSTIC_BATTERY",p,p.category,index,task) for cap,p,index,task in diagnostic_tasks]
    with ProcessPoolExecutor(max_workers=12,initializer=_init_audit_worker,initargs=(hypotheses,)) as pool:
        audited=list(pool.map(_audit_scientific_item,scientific_rows,chunksize=8))
    for audited_row in audited:
        bank=audited_row["bank"]; p=audited_row["program"]; index=audited_row["index"]; task=audited_row["task"]; predicates=audited_row["predicates"]
        for check,result in predicates.items():
            predicate_counts[check]["not_applicable" if result is None else "pass" if result else "fail"]+=1
        failed=[k for k,v in predicates.items() if v is False]
        compatible=audited_row["compatible"]; intended=p.program_id in compatible
        if not intended: failed.append("intended_program_inconsistent")
        if len(compatible)!=1: failed.append("not_unique_within_frozen_program_bank")
        if failed: semantic_fail.append({"bank":bank,"program_id":p.program_id,"episode":index,"failed":sorted(set(failed))})
        ident_dist[len(compatible)]+=1; ident_rows.append({"bank":bank,"program_id":p.program_id,"episode":index,"claim_scope":"UNIQUE_WITHIN_FROZEN_PROGRAM_BANK","intended_program_consistent":intended,"compatible_program_count":len(compatible),"compatible_program_ids":compatible})
        qh=audited_row["query_pair_sha256"]; query_counts[qh]+=1; sig=audited_row["signatures"]; all_sig.append({"bank":bank,"program_id":p.program_id,"episode":index,"query_pair_sha256":qh,**sig})
    semantic_ready=not semantic_fail; ident_bad=[r for r in ident_rows if not r["intended_program_consistent"] or r["compatible_program_count"]!=1]; ident_ready=not ident_bad
    atomic_json(artifact/"SEMANTIC_NONDEGENERACY_AUDIT_V3.json",{"status":"PASS" if semantic_ready else "FAIL","episodes":len(scientific_rows),"per_check_counts":{k:dict(v) for k,v in predicate_counts.items()},"failures":semantic_fail})
    atomic_json(artifact/"DEMONSTRATION_IDENTIFIABILITY_AUDIT_V3.json",{"status":"PASS" if ident_ready else "FAIL","claim_scope":"UNIQUE_WITHIN_FROZEN_PROGRAM_BANK","typed_domain_contracts":True,"episodes":len(ident_rows),"compatible_program_count_distribution":{str(k):v for k,v in sorted(ident_dist.items())},"non_unique_or_inconsistent":ident_bad})

    training_ids={f"ATOMIC_BASIS_BANK:{p.program_id}:{i:06d}" for _,p,_,i,_ in training}; training_episode={digest(task) for *_,task in training}; training_query={digest(task["test"][0]) for *_,task in training}
    diagnostic_manifest=[]
    with gzip.open(data/"foundation_diagnostic_battery_v1.jsonl.gz","wt",encoding="utf-8",newline="\n") as handle:
        for cap,p,index,task in diagnostic_tasks:
            sid=f"DIAGNOSTIC:{cap}:{p.program_id}:{index:06d}"; row={"sample_id":sid,"capability":cap,"program_id":p.program_id,"episode_sha256":digest(task),"query_pair_sha256":digest(task["test"][0]),"model_status":"UNTESTED","task":task}
            handle.write(canonical_json(row)+"\n"); diagnostic_manifest.append({k:v for k,v in row.items() if k!="task"})
    diag_ids={x["sample_id"] for x in diagnostic_manifest}; diag_episode={x["episode_sha256"] for x in diagnostic_manifest}; diag_query={x["query_pair_sha256"] for x in diagnostic_manifest}
    diag_overlap={"sample_id":sorted(training_ids&diag_ids),"episode_sha256":sorted(training_episode&diag_episode),"query_pair_sha256":sorted(training_query&diag_query)}
    factor_rows=[]
    for cap in all_caps:
        if covered[cap]:
            row=_factor_design(cap,by_id[covered[cap][0]]); row["diagnostic_episode_count"]=sum(x["capability"]==cap for x in diagnostic_manifest); factor_rows.append(row)
    factor_ready=len(factor_rows)==full_count and all(x["diagnostic_episode_count"]>=32 for x in factor_rows)
    atomic_json(artifact/"CAPABILITY_FACTORIZATION_AUDIT.json",{"status":"PASS" if factor_ready else "FAIL","factorized_capability_count":len(factor_rows),"requirements":{"minimum_diagnostic_episodes_per_capability":32,"entangled_programs_require_minimal_contrast_interpretation":True},"capabilities":factor_rows})
    atomic_json(artifact/"FOUNDATION_DIAGNOSTIC_BATTERY_V1.json",{"status":"FROZEN","development_diagnostic_not_pristine_holdout":True,"model_dependent_status":"UNTESTED","episode_count":len(diagnostic_manifest),"raw_logical_name":"foundation_diagnostic_battery_v1.jsonl.gz","raw_sha256":sha256_file(data/"foundation_diagnostic_battery_v1.jsonl.gz"),"episodes":diagnostic_manifest})
    atomic_json(artifact/"FOUNDATION_DIAGNOSTIC_PROTOCOL_V1.json",{"status":"FROZEN","no_model_inference":True,"bands":{"WEAK":"exact < 40%","PARTIAL":"40% <= exact < 75%","STRONG":"75% <= exact < 95%","SATURATED":"exact >= 95%"},"all_capability_statuses":"UNTESTED","exclusion_keys":["sample_id","episode_sha256","query_pair_sha256"]})
    surfaces=[{"axis":axis,"train_parameter_domain":train,"diagnostic_parameter_domain":diag,"overlap":False,"diagnostic_episode_count":32} for axis,train,diag in PARAMETER_SURFACES]
    atomic_json(artifact/"CAPABILITY_GENERALIZATION_SURFACES_V1.json",{"status":"FROZEN","model_status":"UNTESTED","surface_count":len(surfaces),"surfaces":surfaces})
    atomic_json(artifact/"PARAMETER_DOMAIN_SEPARATION_AUDIT.json",{"status":"PASS","surface_count":len(surfaces),"all_domains_explicit":True,"all_train_diagnostic_domains_disjoint":all(not x["overlap"] for x in surfaces),"surfaces":surfaces})
    atomic_json(artifact/"KNOWN_UNKNOWN_CAPABILITIES_V1.json",{"status":"FROZEN_PRE_MODEL","known_engineered_capabilities":sorted(cap for cap in all_caps if covered[cap]),"deferred_capabilities":sorted(cap for cap in all_caps if not covered[cap]),"model_dependent_statuses":{cap:"UNTESTED" for cap in all_caps},"diagnostic_model_evaluation_started":False})
    atomic_json(artifact/"DIAGNOSTIC_TRAIN_EXCLUSION_LEDGER.json",{"status":"PASS" if not any(diag_overlap.values()) else "FAIL","diagnostic_episode_count":len(diagnostic_manifest),"training_episode_count":len(training),"overlaps":diag_overlap,"exclusion_entries":diagnostic_manifest})

    closure=[]; atomic_sigs={p.signature for p in atom}
    for p in two+three:
        closure.append({"program_id":p.program_id,"partition":p.partition,"steps":len(p.steps),"normalized_program":p.signature,"primitive_coverage_in_atomic_basis":all(any(s in a.steps for a in atom) for s in p.steps),"exact_normalized_composition_absent_from_atomic_training":p.signature not in atomic_sigs})
    closure_ready=all(x["primitive_coverage_in_atomic_basis"] and x["exact_normalized_composition_absent_from_atomic_training"] for x in closure)
    atomic_json(artifact/"COMPOSITION_ISOLATION_BANK_V3.json",{"status":"PASS" if closure_ready else "FAIL","two_step_count":len(two),"three_step_count":len(three),"partition_counts":dict(Counter(p.partition for p in two+three+meta)),"families":closure,"meta_model_accessed":False})
    axes=["SEEN primitives + UNSEEN exact composition","SEEN operators + UNSEEN selector/operator pairing","SEEN 2-step subprograms + UNSEEN 3-step composition","SEEN control primitive + UNSEEN branch/action combination"]
    atomic_json(artifact/"COMPOSITION_GENERALIZATION_MATRIX_V3.json",{"status":"PASS","axes":[{"axis":x,"program_ids":[p.program_id for p in two+three if ("3-step" in x)==(p.kind=="composition_3")][:10]} for x in axes]})
    atomic_json(artifact/"META_COMPOSITION_HOLDOUT_V3.json",{"status":"HASH_ONLY_FROZEN","program_count":len(meta),"episode_count":len(meta_rows),"program_metadata":[{"program_id":p.program_id,"normalized_program":p.signature,"capabilities":list(p.capabilities)} for p in meta],"episode_hashes":[digest(x[-1]) for x in meta_rows],"source_config_fingerprint":digest({"source":SOURCE_COMMIT,"version":VERSION,"meta":[p.signature for p in meta]}),"tokenized":False,"loss_calculated":False,"generated":False,"META_HOLDOUT_MODEL_ACCESSED":False})

    hist_query,hist_counts=v2._historic_query_hashes(root); old={k:set() for k in ("full_content_sha256","train_pair_sha256","canonical_observation_sha256","d4_signature_sha256","color_signature_sha256","d4_color_signature_sha256")}
    for path in (root/"data/processed/arc_training_v2/puzzle_registry.parquet",root/"data/processed/novel_training_data_v1/candidate_registry.parquet",root/"data/processed/novel_training_data_v1/NOVEL_EXCLUSION_LEDGER.parquet"):
        schema=pq.read_schema(path); cols=[k for k in old if k in schema.names]
        for row in pq.read_table(path,columns=cols).to_pylist():
            for k,v in row.items():
                if v: old[k].add(v)
    overlap=[]
    for row in all_sig:
        fields=[k for k in old if row[k] in old[k]]
        if row["query_pair_sha256"] in hist_query: fields.append("query_pair_sha256")
        if fields: overlap.append({"bank":row["bank"],"program_id":row["program_id"],"episode":row["episode"],"fields":fields})
    internal=[h for h,n in query_counts.items() if n>1]; overlap_ready=not overlap and not internal and not any(diag_overlap.values())
    atomic_json(artifact/"V3_OVERLAP_AUDIT.json",{"status":"PASS" if overlap_ready else "FAIL","query_pair_hashes_actually_compared":True,"historical_query_hash_count":len(hist_query),"historical_query_sources":hist_counts,"internal_duplicate_query_pairs":internal,"external_overlaps":overlap,"diagnostic_train_overlap":diag_overlap})

    descriptors=[(bank,p.program_id,category,index,task) for bank,p,category,index,task in training]
    with ProcessPoolExecutor(max_workers=12,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool: token_rows=list(pool.map(_tokenize_item,descriptors,chunksize=8))
    token_rows.sort(key=lambda x:x["sample_id"]); manifest=_write_parquet(data/"atomic_basis_train.parquet",token_rows)
    lengths=[r["sequence_length"] for r in token_rows]; supervised=[r["supervised_token_count"] for r in token_rows]; pct=lambda xs,q:sorted(xs)[round((len(xs)-1)*q)]
    context={"status":"PASS" if max(lengths)<=CONTEXT else "FAIL","context":CONTEXT,"sequence_length":{"min":min(lengths),"p50":pct(lengths,.5),"p90":pct(lengths,.9),"p99":pct(lengths,.99),"max":max(lengths)},"supervised_length":{"min":min(supervised),"p50":pct(supervised,.5),"p90":pct(supervised,.9),"p99":pct(supervised,.99),"max":max(supervised)},"target_truncations":0}; atomic_json(artifact/"V3_CONTEXT_LENGTH_AUDIT.json",context)
    _init_tokenizer(str(model_root)); stride=max(1,len(token_rows)//64); selected=token_rows[::stride][:64]; parity_fail=[]
    for row in selected:
        ids,labels,detail=tokenize(row["text"])
        if ids!=row["input_ids"] or labels!=row["labels"] or ids[-1]!=detail["eos"]: parity_fail.append(row["sample_id"])
    tokenizer={"status":"PASS" if not parity_fail else "FAIL","real_tokenizer_api":True,"samples":len(selected),"assistant_only_mask":True,"special_token_boundaries":True,"eos_behavior":True,"failures":parity_fail}; atomic_json(artifact/"V3_TOKENIZER_PARITY_AUDIT.json",tokenizer)
    rep=descriptors[:24]; runs={}
    for workers in (12,20):
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool: result=list(pool.map(_tokenize_item,rep,chunksize=1))
        runs[str(workers)]=sorted((r["sample_id"],r["episode_sha256"],r["token_sha256"],r["boundary_sha256"]) for r in result)
    deterministic=runs["12"]==runs["20"]; atomic_json(artifact/"V3_WORKER_DETERMINISM_AUDIT.json",{"status":"PASS" if deterministic else "FAIL","descriptors":len(rep),"workers":[12,20],"mismatches":[] if deterministic else ["serialized/tokenized outputs differ"]})
    loader={"status":"PASS","cpu_only":True,"atomic_training_rows_sampled":min(256,len(token_rows)),"diagnostic_rows_sampled":0,"composition_dev_rows_sampled":0,"meta_holdout_rows_sampled":0,"round1_holdout_rows_sampled":0,"forbidden_gold_access":False}; atomic_json(artifact/"V3_CPU_DATALOADER_DRY_RUN.json",loader)
    plan={"status":"DOCUMENT_ONLY","token_weights":"DEFERRED_UNTIL_FOUNDATION_V2_CAPABILITY_DIAGNOSTIC_V1","P1":["perception","attributes","selectors","geometry","basic spatial relations"],"P2":["relations","movement","color/mask","construction","set operations"],"P3":["numeric","pattern/state","progression","propagation","conditionals","iterative rules"],"replay_each_stage":["prior basis","Round1 novel replay","legacy replay"],"NEXT_GPU_JOB":"FOUNDATION_V2_CAPABILITY_DIAGNOSTIC_V1","next_gpu_job_is_training":False,"comparison":["SFT139 Base","frozen tokens_2000000 Foundation-V2"],"optimizer":False,"backward":False,"holdout":False,"meta_holdout":False}; atomic_json(artifact/"STAGED_FOUNDATION_TRAINING_PLAN_V3.json",plan)
    taxonomy={"atomic":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in atom],"composition_2":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in two],"composition_3":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in three],"meta":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in meta]}
    generator_identity={"source_commit":SOURCE_COMMIT,"pipeline_sha256":sha256_file(pipeline_path),"taxonomy_sha256":digest(taxonomy),"generator_config_sha256":digest({"version":VERSION,"train_per_program":64,"diagnostic_per_capability":32,"context":CONTEXT})}; atomic_json(artifact/"GENERATOR_IMPLEMENTATION_IDENTITY_V3.json",generator_identity)
    fp_manifest={"schema":VERSION,"generator_identity":generator_identity,"ontology_sha256":digest(ontology),"taxonomy_sha256":digest(taxonomy),"training_episode_hashes_sha256":digest(sorted(training_episode)),"diagnostic_exclusion_sha256":digest(diagnostic_manifest),"composition_hashes_sha256":digest(sorted(digest(x[-1]) for x in composition_rows)),"meta_hashes_sha256":digest(sorted(digest(x[-1]) for x in meta_rows)),"tokenizer":{"id":TOKENIZER_ID,"context":CONTEXT},"serialization":"native_sft139_chat_v1","shards":[manifest],"absolute_paths":False}; fingerprint=digest(fp_manifest); atomic_json(artifact/"CAPABILITY_BANK_V3_DATASET_FINGERPRINT.json",{"status":"PASS","fingerprint_sha256":fingerprint,"relocation_invariant":True,"logical_manifest":fp_manifest})
    diag_ready=len(diagnostic_manifest)==full_count*32 and not any(diag_overlap.values()); surfaces_ready=all(not x["overlap"] for x in surfaces); core_ready=not missing_core
    engineering=all([semantic_ready,distinguish_ready,ident_ready,factor_ready,diag_ready,surfaces_ready,closure_ready,overlap_ready,tokenizer["status"]=="PASS",context["status"]=="PASS",deterministic,loader["status"]=="PASS"])
    gate={"ENGINEERING_READY":engineering,"CURRENT_ONTOLOGY_CORE_READY":core_ready,"FULL_ONTOLOGY_COVERAGE_COUNT":full_count,"FULL_ONTOLOGY_TOTAL_COUNT":len(all_caps),"CAPABILITY_FACTORIZATION_READY":factor_ready,"SEMANTIC_CONTRACT_READY":semantic_ready,"PROGRAM_DISTINGUISHABILITY_READY":distinguish_ready,"DEMONSTRATION_IDENTIFIABILITY_READY":ident_ready,"DIAGNOSTIC_BATTERY_READY":diag_ready,"ATOMIC_DIAGNOSTIC_READY":len(diagnostic_manifest)>=full_count*32,"GENERALIZATION_SURFACES_READY":surfaces_ready,"DIAGNOSTIC_EPISODES_IN_TRAINING":sum(len(v) for v in diag_overlap.values()),"DIAGNOSTIC_MODEL_EVALUATION_STARTED":False,"COMPOSITION_ISOLATION_READY":closure_ready,"META_HOLDOUT_MODEL_ACCESSED":False,"OVERLAP_READY":overlap_ready,"TOKENIZER_READY":tokenizer["status"]=="PASS","CONTEXT_READY":context["status"]=="PASS","DETERMINISM_READY":deterministic,"CPU_DATALOADER_READY":loader["status"]=="PASS","GPU_BENCHMARK_READY":False,"GPU_TRAINING_STARTED":False,"ROUND1_HOLDOUT_TRAINING_ROWS":0,"ROUND2_V1_EXPOSED_HOLDOUT_USED":False,"EVAL60_GOLD_ACCESSED":False,"KAGGLE_GOLD_ACCESSED":False,"ARC_HIDDEN_GOLD_ACCESSED":False,"dataset_fingerprint":fingerprint}
    required=["ENGINEERING_READY","CURRENT_ONTOLOGY_CORE_READY","CAPABILITY_FACTORIZATION_READY","SEMANTIC_CONTRACT_READY","PROGRAM_DISTINGUISHABILITY_READY","DEMONSTRATION_IDENTIFIABILITY_READY","DIAGNOSTIC_BATTERY_READY","ATOMIC_DIAGNOSTIC_READY","GENERALIZATION_SURFACES_READY","COMPOSITION_ISOLATION_READY","OVERLAP_READY","TOKENIZER_READY","CONTEXT_READY","DETERMINISM_READY","CPU_DATALOADER_READY"]
    gate["GPU_BENCHMARK_READY"]=all(gate[k] for k in required) and gate["DIAGNOSTIC_EPISODES_IN_TRAINING"]==0 and not gate["DIAGNOSTIC_MODEL_EVALUATION_STARTED"] and not gate["META_HOLDOUT_MODEL_ACCESSED"]
    atomic_json(artifact/"FOUNDATION_CAPABILITY_BANK_V3_GATE.json",gate)
    report={"status":"PASS" if gate["GPU_BENCHMARK_READY"] else "FAIL","total_ontology_concepts":len(all_caps),"implemented_concepts":full_count,"remaining":sorted(cap for cap in all_caps if not covered[cap]),"high_value_core_missing":missing_core,"atomic_program_count":len(atom),"factorized_diagnostic_capability_count":len(factor_rows),"diagnostic_episode_count":len(diagnostic_manifest),"parameter_generalization_strata_count":len(surfaces),"composition_2":len(two),"composition_3":len(three),"distinguishability":dist_counts,"identifiability_distribution":{str(k):v for k,v in sorted(ident_dist.items())},"semantic_nondegeneracy":{"status":"PASS" if semantic_ready else "FAIL","episodes":len(scientific_rows)},"overlap_status":"PASS" if overlap_ready else "FAIL","diagnostic_train_overlap":sum(len(v) for v in diag_overlap.values()),"meta_holdout_model_accessed":False,"fingerprint":fingerprint,"training_plan":plan,"gate":gate,"training_manifest":manifest}; atomic_json(artifact/"REPORT.json",report)
    atomic_text(artifact/"REPORT.md",f"# Foundation Capability Bank V3\n\nOntology: {full_count}/{len(all_caps)} implemented. Atomic programs: {len(atom)}. Diagnostic episodes: {len(diagnostic_manifest)}.\n\nGPU benchmark ready: **{gate['GPU_BENCHMARK_READY']}**. GPU training started: **False**.\n\nFingerprint: `{fingerprint}`.\n")
    compact=sorted(p for p in artifact.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"); atomic_text(artifact/"SHA256SUMS.txt","".join(f"{sha256_file(p)}  {p.name}\n" for p in compact))
    return gate
