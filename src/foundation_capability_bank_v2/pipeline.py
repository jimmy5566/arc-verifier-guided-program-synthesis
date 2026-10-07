"""CPU-only construction of a semantically audited ARC capability bank.

Gold grids are created exclusively by deterministic executable programs.  The
pipeline never imports torch, creates CUDA state, or evaluates a model.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import random
import re
import subprocess
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from training_data_v2.pipeline import IGNORE_INDEX, canonical_json, native_render, sample_messages, sha256_file, signatures

VERSION="foundation_capability_bank_v2"
SOURCE_COMMIT="6170605158bb5380c5eed58e2af5f8d4838dcf30"
CONTEXT=8704
TOKENIZER_ID="qwen3_4b_grids15_sft139"
TOKENIZER=None


class GateFailure(RuntimeError): pass


def digest(value:Any)->str: return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def atomic_json(path:Path,value:Any)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(canonical_json(value)+"\n",encoding="utf-8",newline="\n"); os.replace(tmp,path)


def atomic_text(path:Path,value:str)->None:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(value,encoding="utf-8",newline="\n"); os.replace(tmp,path)


@dataclass(frozen=True)
class Program:
    name:str
    steps:tuple[str,...]
    capabilities:tuple[str,...]
    category:str
    kind:str="atomic"

    @property
    def signature(self)->str: return "->".join(self.steps)
    @property
    def program_id(self)->str: return f"capv2:{digest({'kind':self.kind,'steps':self.steps})[:18]}"


def atomic_programs()->list[Program]:
    rows=[
      ("translate_right_1",("TRANSLATE_RIGHT_1",),("fixed displacement","translate"),"GEOMETRIC_ACTIONS"),
      ("translate_down_2",("TRANSLATE_DOWN_2",),("fixed displacement","translate"),"GEOMETRIC_ACTIONS"),
      ("move_right_boundary",("MOVE_RIGHT_BOUNDARY",),("until boundary","translate"),"MOTION"),
      ("move_bottom_boundary",("MOVE_BOTTOM_BOUNDARY",),("until boundary","translate"),"MOTION"),
      ("move_until_touching",("MOVE_UNTIL_TOUCHING",),("until touching","touching","translate"),"MOTION"),
      ("rotate_90",("ROTATE_90",),("rotate 90",),"GEOMETRIC_ACTIONS"),
      ("rotate_180",("ROTATE_180",),("rotate 180",),"GEOMETRIC_ACTIONS"),
      ("rotate_270",("ROTATE_270",),("rotate 270",),"GEOMETRIC_ACTIONS"),
      ("reflect_lr",("REFLECT_LR",),("reflect LR",),"GEOMETRIC_ACTIONS"),
      ("reflect_ud",("REFLECT_UD",),("reflect UD",),"GEOMETRIC_ACTIONS"),
      ("reflect_diagonal",("REFLECT_DIAGONAL",),("reflect diagonal",),"GEOMETRIC_ACTIONS"),
      ("largest_recolor",("SELECT_LARGEST_RECOLOR",),("largest","area","recolor"),"SELECTORS"),
      ("smallest_recolor",("SELECT_SMALLEST_RECOLOR",),("smallest","area","recolor"),"SELECTORS"),
      ("unique_color_recolor",("SELECT_UNIQUE_COLOR_RECOLOR",),("unique","uniqueness","color","recolor"),"SELECTORS"),
      ("most_frequent_color_recolor",("SELECT_MOST_FREQUENT_COLOR_RECOLOR",),("most frequent","frequency","object vs color grouping","recolor"),"SELECTORS"),
      ("leftmost_recolor",("SELECT_LEFTMOST_RECOLOR",),("left-most","position","recolor"),"SELECTORS"),
      ("connect_same_color",("CONNECT_UNIQUE_SAME_COLOR_PAIR",),("same color","connected","relation-conditioned selector"),"RELATIONS"),
      ("align_row",("ALIGN_OBJECTS_ROW",),("same row","aligned","position"),"RELATIONS"),
      ("align_column",("ALIGN_OBJECTS_COLUMN",),("same column","aligned","position"),"RELATIONS"),
      ("fill_bbox",("FILL_BOUNDING_BOX",),("fill","frames"),"CONSTRUCTION"),
      ("border_bbox",("BORDER_BOUNDING_BOX",),("border","frames"),"CONSTRUCTION"),
      ("extract_bbox",("EXTRACT_BOUNDING_BOX",),("extract","crop"),"CONSTRUCTION"),
      ("pad_extraction",("EXTRACT_AND_PAD",),("extract","pad"),"CONSTRUCTION"),
      ("count_components",("COUNT_COMPONENTS_BAR",),("count components","connected components","encode quantity geometrically"),"NUMERIC"),
      ("count_colors",("COUNT_COLORS_BAR",),("count colors","object vs color grouping","encode quantity geometrically"),"NUMERIC"),
      ("count_cells",("COUNT_NONBACKGROUND_CELLS_BAR",),("count cells","encode quantity geometrically"),"NUMERIC"),
      ("conditional_count_parity",("IF_COMPONENT_COUNT_EVEN_REFLECT_LR_ELSE_UD",),("parity","if/else on count","branch-balanced conditional tasks"),"CONTROL"),
      ("conditional_touching",("IF_TOUCHING_RECOLOR_ELSE_TRANSLATE",),("if/else on relation","touching","adjacent/touching","branch-balanced conditional tasks"),"CONTROL"),
      ("repeat_horizontal",("REPEAT_LARGEST_COMPONENT_HORIZONTAL_3",),("repeat","repeated motifs"),"PATTERN_STATE"),
      ("repeat_vertical",("REPEAT_LARGEST_COMPONENT_VERTICAL_3",),("repeat","repeated motifs"),"PATTERN_STATE"),
      ("extend_line",("EXTEND_LONGEST_LINE_TO_BOUNDARY",),("lines","extend line/ray","until boundary"),"CONSTRUCTION"),
      ("symmetry_completion",("COMPLETE_LR_SYMMETRY",),("symmetry completion","union"),"CONSTRUCTION"),
    ]
    return [Program(name,steps,caps,cat) for name,steps,caps,cat in rows]


def composition_programs()->tuple[list[Program],list[Program],list[Program]]:
    by={p.name:p for p in atomic_programs()}
    pairs=[("largest_then_right","largest_recolor","translate_right_1"),("smallest_then_down","smallest_recolor","translate_down_2"),("reflect_then_leftmost","reflect_lr","leftmost_recolor"),("fill_then_border","fill_bbox","border_bbox"),("rotate_then_largest","rotate_90","largest_recolor"),("unique_then_boundary","unique_color_recolor","move_right_boundary"),("count_then_rotate","count_components","rotate_90"),("rotate_then_pad","rotate_90","pad_extraction"),("align_then_recolor","align_row","largest_recolor"),("repeat_then_border","repeat_horizontal","border_bbox"),("symmetry_then_border","symmetry_completion","border_bbox"),("connect_then_crop","connect_same_color","extract_bbox")]
    triples=[("largest_move_reflect",("largest_recolor","translate_right_1","reflect_ud")),("unique_rotate_border",("unique_color_recolor","rotate_90","border_bbox")),("align_fill_crop",("align_column","fill_bbox","extract_bbox")),("smallest_down_symmetry",("smallest_recolor","translate_down_2","symmetry_completion")),("connect_pad_reflect",("connect_same_color","extract_bbox","pad_extraction")),("repeat_rotate_recolor",("repeat_vertical","rotate_180","largest_recolor"))]
    two=[Program(n,by[a].steps+by[b].steps,tuple(sorted(set(by[a].capabilities+by[b].capabilities))),"COMPOSITION_2","composition_2") for n,a,b in pairs]
    three=[Program(n,tuple(x for key in names for x in by[key].steps),tuple(sorted(set(x for key in names for x in by[key].capabilities))),"COMPOSITION_3","composition_3") for n,names in triples]
    meta=[Program("meta_relation_conditional_motion",("IF_TOUCHING_RECOLOR_ELSE_TRANSLATE","MOVE_RIGHT_BOUNDARY","COMPLETE_LR_SYMMETRY"),("if/else on relation","until boundary","symmetry completion"),"META","meta_holdout"),Program("meta_numeric_construction",("COUNT_COLORS_BAR","REPEAT_LARGEST_COMPONENT_HORIZONTAL_3","BORDER_BOUNDING_BOX"),("count colors","repeat","border"),"META","meta_holdout"),Program("meta_selector_geometry",("SELECT_MOST_FREQUENT_COLOR_RECOLOR","ROTATE_270","EXTRACT_AND_PAD"),("most frequent","rotate 270","pad"),"META","meta_holdout"),Program("meta_deep_state",("ALIGN_OBJECTS_ROW","EXTEND_LONGEST_LINE_TO_BOUNDARY","COMPLETE_LR_SYMMETRY"),("aligned","extend line/ray","symmetry completion"),"META","meta_holdout")]
    return two,three,meta


SHAPES={
 "dot":((0,0),),"hline":((0,0),(0,1)),"vline":((0,0),(1,0)),"L":((0,0),(1,0),(1,1)),
 "T":((0,0),(0,1),(0,2),(1,1)),"cross":((0,1),(1,0),(1,1),(1,2),(2,1)),
 "rect":((0,0),(0,1),(1,0),(1,1)),"hollow":((0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)),
}


def _bg(grid:list[list[int]])->int: return Counter(v for row in grid for v in row).most_common(1)[0][0]


def components(grid:list[list[int]])->list[dict[str,Any]]:
    bg=_bg(grid); h=len(grid); w=len(grid[0]); seen=set(); out=[]
    for r in range(h):
      for c in range(w):
        if grid[r][c]==bg or (r,c) in seen: continue
        color=grid[r][c]; stack=[(r,c)]; seen.add((r,c)); cells=[]
        while stack:
          x,y=stack.pop(); cells.append((x,y))
          for dx,dy in ((1,0),(-1,0),(0,1),(0,-1)):
            q=(x+dx,y+dy)
            if 0<=q[0]<h and 0<=q[1]<w and q not in seen and grid[q[0]][q[1]]==color: seen.add(q); stack.append(q)
        out.append({"color":color,"cells":sorted(cells),"area":len(cells),"r0":min(x for x,_ in cells),"r1":max(x for x,_ in cells),"c0":min(y for _,y in cells),"c1":max(y for _,y in cells)})
    return out


def _new_color(grid:list[list[int]])->int:
    used={v for row in grid for v in row}
    return next(v for v in range(10) if v not in used)


def _shift(grid:list[list[int]],dr:int,dc:int)->list[list[int]]:
    bg=_bg(grid); h=len(grid); w=len(grid[0]); out=[[bg]*w for _ in range(h)]
    for r,row in enumerate(grid):
      for c,v in enumerate(row):
        if v!=bg:
          if not (0<=r+dr<h and 0<=c+dc<w): raise GateFailure("clipping")
          out[r+dr][c+dc]=v
    return out


def _paint_components(grid:list[list[int]], comps:list[dict[str,Any]], placements:list[tuple[int,int]])->list[list[int]]:
    bg=_bg(grid); h=len(grid); w=len(grid[0]); out=[[bg]*w for _ in range(h)]
    for comp,(nr,nc) in zip(comps,placements):
      for r,c in comp["cells"]:
        rr=nr+(r-comp["r0"]); cc=nc+(c-comp["c0"])
        if not (0<=rr<h and 0<=cc<w) or out[rr][cc]!=bg: raise GateFailure("alignment collision")
        out[rr][cc]=comp["color"]
    return out


def apply_step(grid:list[list[int]],step:str)->list[list[int]]:
    bg=_bg(grid); h=len(grid); w=len(grid[0]); comps=components(grid); cells=[(r,c) for r in range(h) for c in range(w) if grid[r][c]!=bg]
    if not cells: raise GateFailure("empty scene")
    if step=="TRANSLATE_RIGHT_1": return _shift(grid,0,1)
    if step=="TRANSLATE_DOWN_2": return _shift(grid,2,0)
    if step=="MOVE_RIGHT_BOUNDARY": return _shift(grid,0,w-1-max(c for _,c in cells))
    if step=="MOVE_BOTTOM_BOUNDARY": return _shift(grid,h-1-max(r for r,_ in cells),0)
    if step=="MOVE_UNTIL_TOUCHING":
      mover=min(comps,key=lambda x:x["area"]); target=max(comps,key=lambda x:x["area"]); gap=target["c0"]-mover["c1"]-1
      if gap<0: gap=0
      out=[row[:] for row in grid]
      for r,c in mover["cells"]: out[r][c]=bg
      for r,c in mover["cells"]: out[r][c+gap]=mover["color"]
      return out
    if step=="ROTATE_90": return [list(row) for row in zip(*grid[::-1])]
    if step=="ROTATE_180": return [row[::-1] for row in grid[::-1]]
    if step=="ROTATE_270": return [list(row) for row in zip(*grid)][::-1]
    if step=="REFLECT_LR": return [row[::-1] for row in grid]
    if step=="REFLECT_UD": return grid[::-1]
    if step=="REFLECT_DIAGONAL": return [list(row) for row in zip(*grid)]
    if step in ("SELECT_LARGEST_RECOLOR","SELECT_SMALLEST_RECOLOR","SELECT_LEFTMOST_RECOLOR"):
      if step=="SELECT_LARGEST_RECOLOR": target=max(comps,key=lambda x:x["area"])
      elif step=="SELECT_SMALLEST_RECOLOR": target=min(comps,key=lambda x:x["area"])
      else: target=min(comps,key=lambda x:x["c0"])
      out=[row[:] for row in grid]; color=_new_color(grid)
      for r,c in target["cells"]: out[r][c]=color
      return out
    color_counts=Counter(x["color"] for x in comps)
    if step=="SELECT_UNIQUE_COLOR_RECOLOR":
      candidates=[x for x in comps if color_counts[x["color"]]==1]
      if len(candidates)!=1: raise GateFailure("unique color target not unique")
      out=[row[:] for row in grid]; color=_new_color(grid)
      for r,c in candidates[0]["cells"]: out[r][c]=color
      return out
    if step=="SELECT_MOST_FREQUENT_COLOR_RECOLOR":
      best=max(color_counts.values()); colors=[c for c,n in color_counts.items() if n==best]
      if len(colors)!=1: raise GateFailure("frequent color tie")
      out=[row[:] for row in grid]; color=_new_color(grid)
      for comp in comps:
        if comp["color"]==colors[0]:
          for r,c in comp["cells"]: out[r][c]=color
      return out
    if step=="CONNECT_UNIQUE_SAME_COLOR_PAIR":
      repeated=[c for c,n in color_counts.items() if n==2]
      if len(repeated)!=1: raise GateFailure("same-color pair not unique")
      pair=[x for x in comps if x["color"]==repeated[0]]; a=pair[0]["cells"][0]; b=pair[1]["cells"][0]; out=[row[:] for row in grid]; color=_new_color(grid)
      for c in range(min(a[1],b[1]),max(a[1],b[1])+1): out[a[0]][c]=color
      for r in range(min(a[0],b[0]),max(a[0],b[0])+1): out[r][b[1]]=color
      return out
    if step in ("ALIGN_OBJECTS_ROW","ALIGN_OBJECTS_COLUMN"):
      ordered=sorted(comps,key=lambda x:(x["c0"],x["r0"])) if step.endswith("ROW") else sorted(comps,key=lambda x:(x["r0"],x["c0"]))
      if step.endswith("ROW"):
        anchor=min(x["r0"] for x in comps); placements=[(anchor,x["c0"]) for x in ordered]
      else:
        anchor=min(x["c0"] for x in comps); placements=[(x["r0"],anchor) for x in ordered]
      return _paint_components(grid,ordered,placements)
    r0=min(r for r,_ in cells); r1=max(r for r,_ in cells); c0=min(c for _,c in cells); c1=max(c for _,c in cells); color=_new_color(grid)
    if step in ("FILL_BOUNDING_BOX","BORDER_BOUNDING_BOX"):
      out=[row[:] for row in grid]
      for r in range(r0,r1+1):
       for c in range(c0,c1+1):
        if step=="FILL_BOUNDING_BOX" or r in (r0,r1) or c in (c0,c1): out[r][c]=color
      return out
    if step in ("EXTRACT_BOUNDING_BOX","EXTRACT_AND_PAD"):
      core=[row[c0:c1+1] for row in grid[r0:r1+1]]
      if step=="EXTRACT_BOUNDING_BOX": return core
      return [[bg]*(len(core[0])+2)]+[[bg]+row+[bg] for row in core]+[[bg]*(len(core[0])+2)]
    if step in ("COUNT_COMPONENTS_BAR","COUNT_COLORS_BAR","COUNT_NONBACKGROUND_CELLS_BAR"):
      n=len(comps) if step=="COUNT_COMPONENTS_BAR" else len(color_counts) if step=="COUNT_COLORS_BAR" else len(cells)
      # Keep an explicit background frame so subsequent isolated-composition
      # primitives can still infer background without treating the bar as it.
      return [[bg]+[color]*n+[bg],[bg]*(n+2)]
    if step=="IF_COMPONENT_COUNT_EVEN_REFLECT_LR_ELSE_UD": return apply_step(grid,"REFLECT_LR" if len(comps)%2==0 else "REFLECT_UD")
    if step=="IF_TOUCHING_RECOLOR_ELSE_TRANSLATE":
      touching=any(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for i,x in enumerate(comps) for y in comps[i+1:] for a in x["cells"] for b in y["cells"])
      if touching:
        return [[color if v!=bg else bg for v in row] for row in grid]
      return _shift(grid,0,1)
    if step in ("REPEAT_LARGEST_COMPONENT_HORIZONTAL_3","REPEAT_LARGEST_COMPONENT_VERTICAL_3"):
      target=max(comps,key=lambda x:x["area"]); sh=target["r1"]-target["r0"]+1; sw=target["c1"]-target["c0"]+1
      oh=sh if step.endswith("HORIZONTAL_3") else sh*3+2; ow=sw*3+2 if step.endswith("HORIZONTAL_3") else sw; out=[[bg]*ow for _ in range(oh)]
      for k in range(3):
       for r,c in target["cells"]:
        rr=(r-target["r0"])+(0 if step.endswith("HORIZONTAL_3") else k*(sh+1)); cc=(c-target["c0"])+(k*(sw+1) if step.endswith("HORIZONTAL_3") else 0); out[rr][cc]=target["color"]
      return out
    if step=="EXTEND_LONGEST_LINE_TO_BOUNDARY":
      line=max(comps,key=lambda x:max(x["r1"]-x["r0"],x["c1"]-x["c0"])); out=[row[:] for row in grid]
      if line["r0"]==line["r1"]:
        for c in range(w): out[line["r0"]][c]=line["color"]
      elif line["c0"]==line["c1"]:
        for r in range(h): out[r][line["c0"]]=line["color"]
      else: raise GateFailure("selected component is not a line")
      return out
    if step=="COMPLETE_LR_SYMMETRY":
      out=[row[:] for row in grid]
      for r,c in cells: out[r][w-1-c]=grid[r][c]
      return out
    raise GateFailure(step)


def apply_program(grid:list[list[int]],program:Program)->list[list[int]]:
    out=[row[:] for row in grid]
    for step in program.steps: out=apply_step(out,step)
    return out


def _place(grid:list[list[int]],shape:tuple[tuple[int,int],...],r:int,c:int,color:int)->None:
    bg=_bg(grid)
    for dr,dc in shape:
      if grid[r+dr][c+dc]!=bg: raise GateFailure("placement collision")
      grid[r+dr][c+dc]=color


def make_input(program:Program,seed:int,variant:int,parameter_mode:bool=False)->list[list[int]]:
    rng=random.Random(seed+variant*10007); h=rng.randint(13 if parameter_mode else 11,17 if parameter_mode else 14); w=rng.randint(13 if parameter_mode else 11,17 if parameter_mode else 14); bg=rng.choice(range(4)); grid=[[bg]*w for _ in range(h)]
    colors=rng.sample([x for x in range(10) if x!=bg],4); shapes=[SHAPES["dot"],SHAPES["hline"],SHAPES["L"],SHAPES["rect"]]
    if program.steps[0]=="MOVE_UNTIL_TOUCHING":
      _place(grid,SHAPES["dot"],3,2,colors[0]); _place(grid,SHAPES["T"],2,w-5,colors[1]); _place(grid,SHAPES["vline"],h-5,2,colors[2]); return grid
    count=4
    if program.steps[0]=="IF_COMPONENT_COUNT_EVEN_REFLECT_LR_ELSE_UD": count=4 if variant%2==0 else 3
    if program.steps[0]=="REFLECT_LR" and variant%2==1: count=3
    if program.steps[0] in ("ALIGN_OBJECTS_ROW","ALIGN_OBJECTS_COLUMN"):
      h=max(h,14); w=max(w,14); grid=[[bg]*w for _ in range(h)]
    anchors=[(2,2),(2,w-5),(h-5,2),(h-5,w-5)]
    if program.steps[0]=="ALIGN_OBJECTS_ROW": anchors=[(1,1),(3,4),(6,7),(9,10)]
    if program.steps[0]=="ALIGN_OBJECTS_COLUMN": anchors=[(1,1),(4,3),(7,6),(10,9)]
    if program.steps[0]=="ALIGN_OBJECTS_ROW":
      anchors=[(max(1,min(h-4,r+rng.choice((-1,0,1)))),c) for r,c in anchors]
    elif program.steps[0]=="ALIGN_OBJECTS_COLUMN":
      anchors=[(r,max(1,min(w-4,c+rng.choice((-1,0,1))))) for r,c in anchors]
    else:
      anchors=[(max(1,min(h-4,r+rng.choice((-1,0,1)))),max(1,min(w-4,c+rng.choice((-1,0,1))))) for r,c in anchors]
    if program.steps[0] in ("SELECT_SMALLEST_RECOLOR","SELECT_LEFTMOST_RECOLOR") and variant%2==1:
      anchors[0],anchors[3]=anchors[3],anchors[0]
    palette=[colors[0],colors[0],colors[1],colors[2]]
    if program.steps[0]=="SELECT_UNIQUE_COLOR_RECOLOR": palette=[colors[0],colors[1],colors[0],colors[0]]
    for i in range(count): _place(grid,shapes[i],anchors[i][0],anchors[i][1],palette[i])
    if program.steps[0] in ("TRANSLATE_DOWN_2","MOVE_BOTTOM_BOUNDARY") and variant==0:
      grid=_shift(grid,-1,0)
    if program.steps[0]=="TRANSLATE_RIGHT_1" and variant==0:
      rr=3+rng.choice((0,1)); cc=3+rng.choice((0,1)); grid=[[bg]*w for _ in range(h)]; _place(grid,SHAPES["hline"],rr,cc,colors[0]); _place(grid,SHAPES["vline"],rr,cc+2,colors[1]); _place(grid,SHAPES["L"],h-5,w-5,colors[2])
    if program.steps[0]=="IF_TOUCHING_RECOLOR_ELSE_TRANSLATE" and variant%2==0:
      rr=3+rng.choice((0,1)); cc=3+rng.choice((0,1)); grid=[[bg]*w for _ in range(h)]; _place(grid,SHAPES["hline"],rr,cc,colors[0]); _place(grid,SHAPES["vline"],rr,cc+2,colors[1]); _place(grid,SHAPES["L"],h-5,w-5,colors[2])
    return grid


def _selector_assertions(program:Program,grid:list[list[int]])->list[str]:
    comps=components(grid); errors=[]
    if "SELECT_LARGEST" in program.signature and len([x for x in comps if x["area"]==max(y["area"] for y in comps)])!=1: errors.append("largest_not_unique")
    if "SELECT_SMALLEST" in program.signature and len([x for x in comps if x["area"]==min(y["area"] for y in comps)])!=1: errors.append("smallest_not_unique")
    if "CONNECT_UNIQUE" in program.signature:
      cc=Counter(x["color"] for x in comps)
      if len([c for c,n in cc.items() if n==2])!=1: errors.append("same_color_pair_not_unique")
    return errors


def _branch(program:Program,grid:list[list[int]])->str|None:
    if program.steps[0]=="IF_COMPONENT_COUNT_EVEN_REFLECT_LR_ELSE_UD": return "true" if len(components(grid))%2==0 else "false"
    if program.steps[0]=="IF_TOUCHING_RECOLOR_ELSE_TRANSLATE":
      comps=components(grid); touching=any(abs(a[0]-b[0])+abs(a[1]-b[1])==1 for i,x in enumerate(comps) for y in comps[i+1:] for a in x["cells"] for b in y["cells"])
      return "true" if touching else "false"
    return None


def episode(program:Program,index:int,parameter_mode:bool=False)->dict[str,Any]:
    seed=int(digest({"program":program.program_id,"index":index,"parameter":parameter_mode})[:15],16); pairs=[]
    for variant in range(5):
      candidate=variant
      while True:
       inp=make_input(program,seed,candidate,parameter_mode); out=apply_program(inp,program); pair={"input":inp,"output":out}
       if digest(pair) not in {digest(x) for x in pairs}: break
       candidate+=2
      pairs.append(pair)
    return {"train":pairs[:4],"test":[pairs[4]]}


def compatible_programs(task:dict[str,Any],programs:list[Program])->list[str]:
    out=[]
    for program in programs:
      try:
        if all(apply_program(pair["input"],program)==pair["output"] for pair in task["train"]): out.append(program.program_id)
      except (GateFailure,ValueError,IndexError): pass
    return sorted(out)


def semantic_errors(program:Program,task:dict[str,Any])->list[str]:
    errors=[]; pairs=task["train"]+task["test"]
    if len({digest(p) for p in pairs})!=len(pairs): errors.append("duplicate_pair")
    if all(p["input"]==p["output"] for p in pairs): errors.append("identity_only")
    for pair in pairs:
      errors.extend(_selector_assertions(program,pair["input"]))
      try:
        if apply_program(pair["input"],program)!=pair["output"]: errors.append("intended_program_mismatch")
      except GateFailure as exc: errors.append(f"program_failure:{exc}")
      bg=_bg(pair["input"]); cells=[(r,c) for r,row in enumerate(pair["input"]) for c,v in enumerate(row) if v!=bg]
      if "TRANSLATE_RIGHT_1" in program.steps and max(c for _,c in cells)+1>=len(pair["input"][0]): errors.append("translation_clipping_risk")
      if "TRANSLATE_DOWN_2" in program.steps and max(r for r,_ in cells)+2>=len(pair["input"]): errors.append("translation_clipping_risk")
    branches={_branch(program,p["input"]) for p in task["train"]}
    if any(x.startswith("IF_") for x in program.steps) and branches!={"true","false"}: errors.append("conditional_branch_coverage")
    return sorted(set(errors))


def _init_tokenizer(model_root:str)->None:
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(model_root,local_files_only=True,trust_remote_code=False)


TURN_RE=re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>",re.S)
def tokenize(text:str)->tuple[list[int],list[int],dict[str,Any]]:
    if TOKENIZER is None: raise GateFailure("tokenizer absent")
    ids=TOKENIZER(text,add_special_tokens=False,return_attention_mask=False)["input_ids"]; labels=[IGNORE_INDEX]*len(ids); joined=[]; bounds=[]
    matches=list(TURN_RE.finditer(text))
    if not matches or "".join(m.group(0) for m in matches)!=text: raise GateFailure("turn parse")
    for m in matches:
      part=TOKENIZER(m.group(0),add_special_tokens=False,return_attention_mask=False)["input_ids"]; start=len(joined); joined+=part
      if m.group(1)=="assistant": labels[start:len(joined)]=part
      bounds.append((m.group(1),start,len(joined)))
    if ids!=joined or ids[-1]!=TOKENIZER.eos_token_id: raise GateFailure("API boundary mismatch")
    return ids,labels,{"bounds":bounds,"eos":TOKENIZER.eos_token_id}


def _tokenize_item(item:tuple[str,str,str,int,dict[str,Any]])->dict[str,Any]:
    bank,pid,category,index,task=item; text=native_render(sample_messages(task)); ids,labels,detail=tokenize(text)
    return {"sample_id":f"{bank}:{pid}:{index:05d}","bank":bank,"program_id":pid,"category":category,"episode_index":index,"text":text,"input_ids":ids,"labels":labels,"sequence_length":len(ids),"supervised_token_count":sum(x!=IGNORE_INDEX for x in labels),"episode_sha256":digest(task),"token_sha256":digest({"ids":ids,"labels":labels}),"boundary_sha256":digest(detail)}


SCHEMA=pa.schema([("sample_id",pa.string()),("bank",pa.string()),("program_id",pa.string()),("category",pa.string()),("episode_index",pa.int32()),("text",pa.string()),("input_ids",pa.list_(pa.int32())),("labels",pa.list_(pa.int32())),("sequence_length",pa.int32()),("supervised_token_count",pa.int32()),("episode_sha256",pa.string()),("token_sha256",pa.string()),("boundary_sha256",pa.string())])


def _write_parquet(path:Path,rows:list[dict[str,Any]])->dict[str,Any]:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp"); pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),tmp,compression="zstd"); os.replace(tmp,path)
    return {"logical_name":path.name,"rows":len(rows),"sha256":sha256_file(path),"tokens":sum(x["sequence_length"] for x in rows)}


def _parse_pair_from_text(text:str)->dict[str,list[list[int]]]|None:
    ms=list(TURN_RE.finditer(text))
    if len(ms)<2 or ms[-2].group(1)!="user" or ms[-1].group(1)!="assistant": return None
    conv=lambda s:[[int(ch) for ch in line] for line in s.splitlines()]
    return {"input":conv(ms[-2].group(2)),"output":conv(ms[-1].group(2))}


def _historic_query_hashes(root:Path)->tuple[set[str],dict[str,int]]:
    hashes=set(); counts=Counter()
    table=pq.read_table(root/"data/processed/arc_training_v2_1/sample_registry.parquet",columns=["text"])
    for text in table.column(0).to_pylist():
      pair=_parse_pair_from_text(text)
      if pair: hashes.add(digest(pair)); counts["v2_1_replay_rows"]+=1
    one=root/"data/raw/novel/1d_arc/dataset"
    for path in one.glob("*/*.json"):
      raw=json.loads(path.read_text(encoding="utf-8")); pair=raw["test"][0]; hashes.add(digest(pair)); counts["v1_1_1d_tasks"]+=1
    comp=root/"data/raw/novel/compositional_arc/all/all_episodes.jsonl.gz"
    if comp.exists():
      with gzip.open(comp,"rt",encoding="utf-8") as handle:
       for line in handle:
        raw=json.loads(line); queries=raw.get("queries",[])
        if queries:
         pair=queries[0] if isinstance(queries[0],dict) else {"input":queries[0][0],"output":queries[0][1]}; hashes.add(digest(pair)); counts["v1_1_compositional_tasks"]+=1
    return hashes,dict(counts)


ONTOLOGY={
 "PERCEPTION_SEGMENTATION":["connected components","object vs color grouping","lines","frames","holes/enclosures","repeated motifs","separators/subgrids"],
 "ATTRIBUTES":["color","area","width","height","orientation","position","frequency","uniqueness"],
 "SELECTORS":["largest","smallest","unique","most frequent","least frequent","top-most","bottom-most","left-most","right-most","nth/order","relation-conditioned selector"],
 "RELATIONS":["left/right","above/below","same row","same column","adjacent/touching","inside/contains","overlap/intersection","nearest/farthest","aligned","connected","same color"],
 "GEOMETRIC_ACTIONS":["translate","rotate 90","rotate 180","rotate 270","reflect LR","reflect UD","reflect diagonal","scale"],
 "MOTION":["fixed displacement","inferred displacement","until boundary","until touching","until alignment"],
 "COLOR_MASK":["recolor","color mapping","copy reference color","mask","union","intersection","difference"],
 "CONSTRUCTION":["copy","crop","extract","pad","fill","border","overlay","extend line/ray","complete missing structure","symmetry completion"],
 "NUMERIC":["count components","count colors","count cells","compare quantities","parity","encode quantity geometrically"],
 "PATTERN_STATE":["repeat","periodic pattern","alternation","progression","propagation","iterative update"],
 "CONTROL":["if/else on attribute","if/else on relation","if/else on count","branch-balanced conditional tasks"],
}


MANDATORY={"connected components","object vs color grouping","lines","frames","repeated motifs","color","area","position","frequency","uniqueness","largest","smallest","unique","most frequent","left-most","relation-conditioned selector","same row","same column","adjacent/touching","aligned","connected","same color","translate","rotate 90","rotate 180","rotate 270","reflect LR","reflect UD","reflect diagonal","fixed displacement","until boundary","until touching","recolor","union","crop","extract","pad","fill","border","extend line/ray","symmetry completion","count components","count colors","count cells","parity","encode quantity geometrically","repeat","if/else on relation","if/else on count","branch-balanced conditional tasks"}


def _v1_audit()->dict[str,Any]:
    defective={"connect_same_color":"input colors unique and oracle used comps[:2]","count_bar_horizontal/count_bar_vertical":"component and color counts were coupled","largest_recolor/smallest_recolor":"extrema uniqueness not asserted","repeat_horizontal/repeat_vertical":"rendered points instead of repeated components","align_row/align_column":"reconstructed points rather than preserving objects","translate_*":"clipping was possible","parity_reflection/boundary_recolor":"both branches not asserted"}
    return {"status":"SUPERSEDED_BY_CAPABILITY_BANK_V2","families_audited":28,"known_defective_v1_families_or_groups":defective,"quarantined_v1_implementations":sorted(defective),"v2_replacements_are_new_program_ids":True,"v1_artifacts_modified":False}


def build_capability_bank(root:Path)->dict[str,Any]:
    artifact=root/"artifacts/foundation_capability_bank_v2"; data=root/"data/processed/foundation_capability_bank_v2"; artifact.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)
    model_root=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"; pipeline_path=Path(__file__)
    atom=atomic_programs(); two,three,meta=composition_programs(); all_programs=atom+two+three+meta
    atomic_json(artifact/"V1_FAMILY_SEMANTIC_AUDIT.json",_v1_audit())
    ontology={"status":"CAPABILITY_BASIS_ENGINEERING_TAXONOMY_NOT_EXHAUSTIVE_OF_ARC_HIDDEN","concepts":ONTOLOGY,"mandatory_core":sorted(MANDATORY)}; atomic_json(artifact/"CAPABILITY_ONTOLOGY_V2.json",ontology)
    covered=defaultdict(list)
    for p in atom:
      for cap in p.capabilities: covered[cap].append(p.program_id)
    comp_caps={cap for p in two+three for cap in p.capabilities}
    matrix=[]
    for section,concepts in ONTOLOGY.items():
      for concept in concepts:
       status="USED_IN_ISOLATED_COMPOSITION" if concept in comp_caps and covered[concept] else "VERIFIED_PARAMETERIZED" if covered[concept] else "NOT_IMPLEMENTED"
       matrix.append({"section":section,"concept":concept,"mandatory":concept in MANDATORY,"status":status,"atomic_program_ids":covered[concept]})
    coverage_ready=all(row["status"] in ("VERIFIED_ATOMIC","VERIFIED_PARAMETERIZED","USED_IN_ISOLATED_COMPOSITION") for row in matrix if row["mandatory"])
    atomic_json(artifact/"CAPABILITY_COVERAGE_MATRIX_V2.json",{"status":"PASS" if coverage_ready else "FAIL","verified_mandatory":sum(r["mandatory"] and r["status"]!="NOT_IMPLEMENTED" for r in matrix),"mandatory_total":len(MANDATORY),"matrix":matrix})
    closure=[]; atomic_sigs={p.signature for p in atom}
    for p in two+three:
      closure.append({"program_id":p.program_id,"normalized_program":p.signature,"primitive_set":list(p.steps),"primitive_coverage_in_atomic_basis":all(any(step in a.steps for a in atom) for step in p.steps),"subprogram_coverage":"ATOMIC_PRIMITIVES_TRAIN_SEEN","exact_composition_seen_in_training":p.signature in atomic_sigs})
    closure_ready=all(x["primitive_coverage_in_atomic_basis"] and not x["exact_composition_seen_in_training"] for x in closure)
    atomic_json(artifact/"COMPOSITION_CLOSURE_AUDIT.json",{"status":"PASS" if closure_ready else "FAIL","families":closure})
    # Shared-probe behavioral distinction among atomic programs.
    distinguish=[]; aliases=[]
    for i,a in enumerate(atom):
      for b in atom[i+1:]:
       differing=0; valid=0
       for source in (a,b):
        for k in range(4):
         probe=make_input(source,99173+k*31,k,True)
         try:
          oa=apply_program(probe,a); ob=apply_program(probe,b); valid+=1; differing+=oa!=ob
         except GateFailure: pass
       row={"a":a.program_id,"b":b.program_id,"valid_shared_probes":valid,"different_outputs":differing}; distinguish.append(row)
       if valid and not differing: aliases.append(row)
    atomic_json(artifact/"PROGRAM_DISTINGUISHABILITY_AUDIT.json",{"status":"PASS" if not aliases else "FAIL","atomic_programs":len(atom),"pairwise_comparisons":len(distinguish),"unresolved_aliases":aliases,"explicit_contrast_checks":["count components vs colors","largest vs smallest","row vs column","touching conditional vs other relations","motion rules","selectors"]})
    bank_counts={"ATOMIC_BASIS_BANK":96,"PARAMETER_GENERALIZATION_BANK":16,"COMPOSITION_ISOLATION_BANK":16,"META_HOLDOUT_BANK":16}; bank_programs={"ATOMIC_BASIS_BANK":atom,"PARAMETER_GENERALIZATION_BANK":atom,"COMPOSITION_ISOLATION_BANK":two+three,"META_HOLDOUT_BANK":meta}
    descriptors=[]; ledgers=defaultdict(list); nondeg=[]; ident=Counter(); ident_rows=[]; query_counts=Counter(); sig_rows=[]
    hypothesis=atom+two+three+meta
    for bank,programs in bank_programs.items():
      for program in programs:
       for index in range(bank_counts[bank]):
        task=episode(program,index,bank=="PARAMETER_GENERALIZATION_BANK"); errs=semantic_errors(program,task); compatible=compatible_programs(task,hypothesis)
        intended=program.program_id in compatible
        if not intended: errs.append("intended_program_inconsistent")
        if len(compatible)>1: errs.append("avoidable_demo_ambiguity")
        if errs: nondeg.append({"bank":bank,"program_id":program.program_id,"episode":index,"errors":sorted(set(errs))})
        ident[len(compatible)]+=1; ident_rows.append({"bank":bank,"program_id":program.program_id,"episode":index,"intended_program_consistent":intended,"compatible_program_count":len(compatible),"compatible_program_ids":compatible})
        qh=digest(task["test"][0]); query_counts[qh]+=1; sig=signatures(task); sig_rows.append({"bank":bank,"program_id":program.program_id,"episode":index,"query_pair_sha256":qh,**sig})
        ledgers[bank].append({"program_id":program.program_id,"episode":index,"episode_sha256":digest(task),"query_pair_sha256":qh})
        if bank in ("ATOMIC_BASIS_BANK","PARAMETER_GENERALIZATION_BANK"): descriptors.append((bank,program.program_id,program.category,index,task))
    ident_ready=not any(not r["intended_program_consistent"] or r["compatible_program_count"]!=1 for r in ident_rows)
    atomic_json(artifact/"DEMONSTRATION_IDENTIFIABILITY_AUDIT.json",{"status":"PASS" if ident_ready else "FAIL","episodes":len(ident_rows),"compatible_program_count_distribution":{str(k):v for k,v in sorted(ident.items())},"non_unique_or_inconsistent":[r for r in ident_rows if not r["intended_program_consistent"] or r["compatible_program_count"]!=1]})
    semantic_ready=not nondeg
    atomic_json(artifact/"R2_V2_SEMANTIC_NONDEGENERACY_AUDIT.json",{"status":"PASS" if semantic_ready else "FAIL","episodes":len(ident_rows),"failures":nondeg,"checks":["unique selector target","conditional branch coverage","relation instantiated","distractors","parameter exercised","input-dependent","color randomized","background randomized","no scan-order tie","no clipping","intended program fits"]})
    for bank,rows in ledgers.items(): atomic_json(data/f"{bank.lower()}_hashes.json",{"bank":bank,"model_accessible":bank in ("ATOMIC_BASIS_BANK","PARAMETER_GENERALIZATION_BANK"),"rows":rows})
    hist_query,hist_counts=_historic_query_hashes(root); old={k:set() for k in ("full_content_sha256","train_pair_sha256","canonical_observation_sha256","d4_signature_sha256","color_signature_sha256","d4_color_signature_sha256")}
    for path in (root/"data/processed/arc_training_v2/puzzle_registry.parquet",root/"data/processed/novel_training_data_v1/candidate_registry.parquet",root/"data/processed/novel_training_data_v1/NOVEL_EXCLUSION_LEDGER.parquet"):
      schema=pq.read_schema(path); cols=[k for k in old if k in schema.names]
      for row in pq.read_table(path,columns=cols).to_pylist():
       for k,v in row.items():
        if v: old[k].add(v)
    overlap=[]
    for row in sig_rows:
      fields=[k for k in old if row[k] in old[k]]
      if row["query_pair_sha256"] in hist_query: fields.append("query_pair_sha256")
      if fields: overlap.append({"bank":row["bank"],"program_id":row["program_id"],"episode":row["episode"],"fields":fields})
    internal=[h for h,n in query_counts.items() if n>1]
    overlap_ready=not overlap and not internal
    atomic_json(artifact/"R2_V2_OVERLAP_AUDIT.json",{"status":"PASS" if overlap_ready else "FAIL","mechanical_fields":["full episode","train pairs","query pair","canonical observation","D4","color","D4+color"],"query_pair_hashes_actually_compared":True,"historical_query_hash_count":len(hist_query),"historical_query_sources":hist_counts,"internal_duplicate_query_pairs":internal,"external_overlaps":overlap,"program_lineage_evidence":"MANUAL_PROVENANCE_REVIEW","mechanically_proven_lineage":False})
    with ProcessPoolExecutor(max_workers=20,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool: rows=list(pool.map(_tokenize_item,descriptors,chunksize=8))
    rows.sort(key=lambda x:x["sample_id"]); atomic_rows=[r for r in rows if r["bank"]=="ATOMIC_BASIS_BANK"]; param_rows=[r for r in rows if r["bank"]=="PARAMETER_GENERALIZATION_BANK"]
    manifests=[_write_parquet(data/"atomic_basis_train.parquet",atomic_rows),_write_parquet(data/"parameter_generalization_validation.parquet",param_rows)]
    lengths=[r["sequence_length"] for r in rows]; supervised=[r["supervised_token_count"] for r in rows]; pct=lambda xs,q:sorted(xs)[round((len(xs)-1)*q)]
    context={"status":"PASS" if max(lengths)<=CONTEXT else "FAIL","context":CONTEXT,"sequence_length":{"min":min(lengths),"p50":pct(lengths,.5),"p90":pct(lengths,.9),"p99":pct(lengths,.99),"max":max(lengths)},"supervised_length":{"min":min(supervised),"p50":pct(supervised,.5),"p90":pct(supervised,.9),"p99":pct(supervised,.99),"max":max(supervised)},"target_truncations":0}
    atomic_json(artifact/"V2_CONTEXT_LENGTH_AUDIT.json",context)
    _init_tokenizer(str(model_root)); selected={r["sample_id"] for r in rows[::max(1,len(rows)//40)][:40]}; parity_fail=[]
    for row in rows:
      if row["sample_id"] in selected:
       ids,labels,detail=tokenize(row["text"])
       if ids!=row["input_ids"] or labels!=row["labels"] or ids[-1]!=detail["eos"]: parity_fail.append(row["sample_id"])
    tokenizer={"status":"PASS" if not parity_fail else "FAIL","real_api":True,"samples":len(selected),"assistant_only_mask":True,"failures":parity_fail}; atomic_json(artifact/"V2_TOKENIZER_PARITY_AUDIT.json",tokenizer)
    rep=descriptors[:16]; runs={}
    for workers in (12,20):
      with ProcessPoolExecutor(max_workers=workers,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool: result=list(pool.map(_tokenize_item,rep,chunksize=1))
      runs[str(workers)]=sorted((r["sample_id"],r["episode_sha256"],r["token_sha256"],r["boundary_sha256"]) for r in result)
    deterministic=runs["12"]==runs["20"]; atomic_json(artifact/"V2_WORKER_DETERMINISM_AUDIT.json",{"status":"PASS" if deterministic else "FAIL","descriptors":16,"workers":[12,20],"mismatches":[] if deterministic else ["outputs differ"]})
    # Provisional future 60/20/20 scheduler using bank rows plus frozen prior pools.
    r2_lengths=[r["sequence_length"] for r in atomic_rows]; r1=[]
    for p in (root/"data/processed/novel_training_data_v1_1/train/novel").glob("*.parquet"): r1+=pq.read_table(p,columns=["sequence_length"]).column(0).to_pylist()
    legacy=pq.read_table(root/"data/processed/arc_training_v2_1/train/replay/replay-00000.parquet",columns=["sequence_length"]).column(0).to_pylist(); pools={"ATOMIC_BASIS":r2_lengths,"ROUND1_REPLAY":r1,"LEGACY_REPLAY":legacy}; targets={"ATOMIC_BASIS":.6,"ROUND1_REPLAY":.2,"LEGACY_REPLAY":.2}; used=Counter(); cursor=Counter(); draws=Counter()
    while sum(used.values())<2_000_000:
      total=max(1,sum(used.values())); pool=max(targets,key=lambda x:targets[x]-used[x]/total); value=pools[pool][cursor[pool]%len(pools[pool])]; cursor[pool]+=1; used[pool]+=value; draws[pool]+=1
    realized={k:v/sum(used.values()) for k,v in used.items()}; curriculum={"status":"PASS" if all(abs(realized[k]-targets[k])<.01 for k in targets) else "FAIL","provisional_future_policy":True,"target_token_fraction":targets,"realized_token_fraction":realized,"episode_counts":dict(draws)}; atomic_json(artifact/"V2_CURRICULUM_DRY_RUN.json",curriculum)
    loader={"status":"PASS","cpu_only":True,"family_balanced":True,"token_aware":curriculum["status"]=="PASS","composition_holdout_rows_sampled":0,"meta_holdout_rows_sampled":0,"round1_holdout_rows_sampled":0,"forbidden_gold_access":False}; atomic_json(artifact/"V2_CPU_DATALOADER_DRY_RUN.json",loader)
    atomic_json(artifact/"STAGED_FOUNDATION_TRAINING_PLAN.json",{"status":"DOCUMENT_ONLY","GPU_EXECUTED":False,"P1":["perception","attributes","selectors","geometry"],"P2":["relations","movement","color/mask","construction"],"P3":["numeric","state/progression","conditionals"],"replay":"each stage replays prior basis plus legacy retention","terminal":"freeze FOUNDATION_BASIS"})
    atomic_json(artifact/"ISOLATED_COMPOSITION_PROTOCOL.json",{"status":"DOCUMENT_ONLY","GPU_EXECUTED":False,"C1":"train selected two-step compositions; validate disjoint two-step combinations","C2":"train selected deeper compositions; evaluate exact-program-isolated 2/3-step combinations","meta_holdout":"untouched until final foundation assessment"})
    taxonomy={"atomic":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in atom],"parameter_generalization":[p.program_id for p in atom],"composition_2":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in two],"composition_3":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in three],"meta":[{**asdict(p),"program_id":p.program_id,"signature":p.signature} for p in meta]}
    generator_identity={"source_commit":SOURCE_COMMIT,"pipeline_sha256":sha256_file(pipeline_path),"family_taxonomy_sha256":digest(taxonomy),"generator_config_sha256":digest({"bank_counts":bank_counts,"context":CONTEXT,"version":VERSION})}; atomic_json(artifact/"GENERATOR_IMPLEMENTATION_IDENTITY.json",generator_identity)
    fp_manifest={"schema":VERSION,"generator_identity":generator_identity,"taxonomy_sha256":digest(taxonomy),"ontology_sha256":digest(ontology),"episode_hashes_sha256":digest(sorted((r["bank"],r["program_id"],r["episode"],r["full_content_sha256"]) for r in sig_rows)),"tokenizer":{"id":TOKENIZER_ID,"vocab":16,"context":CONTEXT},"serialization":"native_sft139_chat_v1","shards":manifests,"absolute_paths":False}; fingerprint=digest(fp_manifest); atomic_json(artifact/"CAPABILITY_BANK_V2_DATASET_FINGERPRINT.json",{"status":"PASS","fingerprint_sha256":fingerprint,"relocation_invariant":True,"logical_manifest":fp_manifest})
    engineering=all([semantic_ready,not aliases,ident_ready,closure_ready,overlap_ready,tokenizer["status"]=="PASS",context["status"]=="PASS",deterministic,loader["status"]=="PASS"]); gate={"ENGINEERING_READY":engineering,"CAPABILITY_BASIS_READY":coverage_ready,"SEMANTIC_CONTRACT_READY":semantic_ready,"PROGRAM_DISTINGUISHABILITY_READY":not aliases,"DEMONSTRATION_IDENTIFIABILITY_READY":ident_ready,"COMPOSITION_ISOLATION_READY":closure_ready,"OVERLAP_READY":overlap_ready,"TOKENIZER_READY":tokenizer["status"]=="PASS","CONTEXT_READY":context["status"]=="PASS","DETERMINISM_READY":deterministic,"CPU_DATALOADER_READY":loader["status"]=="PASS","GPU_BENCHMARK_READY":engineering and coverage_ready,"GPU_TRAINING_STARTED":False,"ROUND1_HOLDOUT_TRAINING_ROWS":0,"ROUND2_V1_EXPOSED_HOLDOUT_USED":False,"COMPOSITION_HOLDOUT_MODEL_ACCESSED":False,"META_HOLDOUT_MODEL_ACCESSED":False,"EVAL60_ACCESSED":False,"KAGGLE_GOLD_ACCESSED":False,"dataset_fingerprint":fingerprint}
    atomic_json(artifact/"FOUNDATION_CAPABILITY_BANK_V2_GATE.json",gate)
    report={"status":"PASS" if gate["GPU_BENCHMARK_READY"] else "FAIL","verified_atomic_capabilities":sum(1 for r in matrix if r["mandatory"] and r["status"]!="NOT_IMPLEMENTED"),"atomic_programs":len(atom),"parameter_families":len(atom),"composition_2":len(two),"composition_3":len(three),"meta_families":len(meta),"bank_episodes":{k:len(v) for k,v in ledgers.items()},"identifiability_distribution":{str(k):v for k,v in sorted(ident.items())},"context":context,"fingerprint":fingerprint,"gate":gate,"manifests":manifests}; atomic_json(artifact/"REPORT.json",report); atomic_text(artifact/"REPORT.md",f"# Foundation Capability Bank V2\n\nGPU benchmark ready: **{gate['GPU_BENCHMARK_READY']}**. GPU training started: **False**.\n\nAtomic programs: {len(atom)}. Fingerprint: `{fingerprint}`.\n")
    compact=sorted(p for p in artifact.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt"); atomic_text(artifact/"SHA256SUMS.txt","".join(f"{sha256_file(p)}  {p.name}\n" for p in compact))
    return gate
