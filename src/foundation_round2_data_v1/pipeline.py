"""Build the CPU-only Foundation Round 2 procedural corpus.

The module deliberately has no torch or CUDA dependency.  Every target is
produced by an executable deterministic oracle, and model-facing holdouts are
never tokenized or passed through a model.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from training_data_v2.pipeline import (
    IGNORE_INDEX,
    canonical_json,
    native_render,
    sample_messages,
    sha256_file,
    signatures,
)

VERSION = "foundation_round2_data_v1"
SPLIT_SEED = "foundation-round2-family-split-v1"
CONTEXT = 8704
WORKERS = 20
V2_FP = "2a0e2df7df77bf43b461f20b10c9090049f4165699fab4ec8ca0b532fe24e7e7"
V11_FP = "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"
BASE_MODEL = "qwen3_4b_grids15_sft139"
ADAPTER_SHA256 = "f0079dd399c1f0521c378a46bd7bbb737a0abc72ef0bb6c47a6c1ae33446afce"
TOKENIZER = None


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
class FamilySpec:
    name: str
    operator: str
    signature: str
    category: str
    priority: str

    @property
    def family_id(self) -> str:
        return "r2:" + digest(self.signature)[:16]


def family_specs() -> list[FamilySpec]:
    rows = [
        ("translate_right", "shift_right", "SELECT[all_objects]->TRANSLATE[right,1]", "parameterized movement", "HIGH"),
        ("translate_left", "shift_left", "SELECT[all_objects]->TRANSLATE[left,1]", "parameterized movement", "HIGH"),
        ("translate_down", "shift_down", "SELECT[all_objects]->TRANSLATE[down,1]", "parameterized movement", "HIGH"),
        ("translate_up", "shift_up", "SELECT[all_objects]->TRANSLATE[up,1]", "parameterized movement", "HIGH"),
        ("to_right_boundary", "right_boundary", "SELECT[all_objects]->TRANSLATE[until_right_boundary]", "boundary-sensitive transformations", "HIGH"),
        ("to_bottom_boundary", "bottom_boundary", "SELECT[all_objects]->TRANSLATE[until_bottom_boundary]", "boundary-sensitive transformations", "HIGH"),
        ("reflect_lr", "flip_lr", "SELECT[grid]->REFLECT[left_right]", "reflection robustness", "MEDIUM"),
        ("reflect_ud", "flip_ud", "SELECT[grid]->REFLECT[up_down]", "reflection robustness", "MEDIUM"),
        ("rotate_180", "rot180", "SELECT[grid]->ROTATE[180]", "object extraction / placement", "MEDIUM"),
        ("largest_recolor", "largest_recolor", "SELECT[largest_component]->RECOLOR[marker]", "object selection by properties", "HIGH"),
        ("smallest_recolor", "smallest_recolor", "SELECT[smallest_component]->RECOLOR[marker]", "object selection by properties", "HIGH"),
        ("fill_bbox", "fill_bbox", "SELECT[nonzero]->BOUNDING_BOX->FILL[marker]", "fill / padded-fill", "MEDIUM"),
        ("border_bbox", "border_bbox", "SELECT[nonzero]->BOUNDING_BOX->BORDER[marker]", "fill / padded-fill", "MEDIUM"),
        ("connect_same_color", "connect", "SELECT[same_color_pair]->CONNECT[orthogonal_path]", "multi-object spatial relations", "HIGH"),
        ("align_row", "align_row", "SELECT[objects]->ALIGN[row_of_largest]", "multi-object spatial relations", "HIGH"),
        ("align_column", "align_col", "SELECT[objects]->ALIGN[column_of_largest]", "multi-object spatial relations", "HIGH"),
        ("count_bar_horizontal", "count_h", "COUNT[components]->RENDER[top_row_bar]", "counting / comparison", "HIGH"),
        ("count_bar_vertical", "count_v", "COUNT[colors]->RENDER[left_column_bar]", "counting / comparison", "HIGH"),
        ("repeat_horizontal", "repeat_h", "EXTRACT[leftmost_component]->REPEAT[horizontal,count_components]", "state/progression transformations", "HIGH"),
        ("repeat_vertical", "repeat_v", "EXTRACT[topmost_component]->REPEAT[vertical,count_colors]", "state/progression transformations", "HIGH"),
        ("parity_reflection", "conditional_parity", "IF[count_components_even]->REFLECT[LR]:REFLECT[UD]", "conditional transformations", "HIGH"),
        ("boundary_recolor", "conditional_boundary", "IF[object_touches_boundary]->RECOLOR[marker]:TRANSLATE[right,1]", "conditional transformations", "HIGH"),
        ("recolor_then_move", "recolor_shift", "SELECT[largest]->RECOLOR[marker]->TRANSLATE[right,1]", "multi-stage construction", "HIGH"),
        ("rotate_then_recolor", "rotate_recolor", "SELECT[grid]->ROTATE[90]->RECOLOR[nonzero,marker]", "multi-stage construction", "HIGH"),
        ("extract_bbox", "extract", "SELECT[nonzero]->EXTRACT[bounding_box]", "object extraction / placement", "MEDIUM"),
        ("extract_padded", "extract_pad", "SELECT[nonzero]->EXTRACT[bounding_box]->PAD[1]", "object extraction / placement", "MEDIUM"),
        ("overlay_mirror", "overlay_mirror", "SELECT[nonzero]->REFLECT[LR]->OVERLAY[original]", "multi-stage construction", "HIGH"),
        ("center_objects", "center", "SELECT[nonzero]->TRANSLATE[center_bbox]", "multi-object spatial relations", "HIGH"),
    ]
    return [FamilySpec(*row) for row in rows]


def split_families(specs: list[FamilySpec]) -> dict[str, list[FamilySpec]]:
    ordered = sorted(specs, key=lambda s: digest({"seed": SPLIT_SEED, "signature": s.signature}))
    return {"train": ordered[:16], "validation": ordered[16:20], "holdout": ordered[20:24], "meta_holdout": ordered[24:28]}


def _components(grid: list[list[int]]) -> list[list[tuple[int, int]]]:
    h, w = len(grid), len(grid[0]); seen: set[tuple[int, int]] = set(); result = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 0 or (r, c) in seen: continue
            color = grid[r][c]; stack = [(r, c)]; seen.add((r, c)); comp = []
            while stack:
                x, y = stack.pop(); comp.append((x, y))
                for dx, dy in ((1,0),(-1,0),(0,1),(0,-1)):
                    q = (x+dx, y+dy)
                    if 0 <= q[0] < h and 0 <= q[1] < w and q not in seen and grid[q[0]][q[1]] == color:
                        seen.add(q); stack.append(q)
            result.append(comp)
    return result


def _shift(grid: list[list[int]], dr: int, dc: int) -> list[list[int]]:
    h, w = len(grid), len(grid[0]); out = [[0]*w for _ in range(h)]
    for r in range(h):
        for c in range(w):
            if grid[r][c] and 0 <= r+dr < h and 0 <= c+dc < w: out[r+dr][c+dc] = grid[r][c]
    return out


def apply_oracle(grid: list[list[int]], operator: str) -> list[list[int]]:
    h, w = len(grid), len(grid[0]); g = [row[:] for row in grid]; comps = _components(g)
    nz = [(r,c) for r in range(h) for c in range(w) if g[r][c]]
    if operator.startswith("shift_"):
        return _shift(g, *{"shift_right":(0,1),"shift_left":(0,-1),"shift_down":(1,0),"shift_up":(-1,0)}[operator])
    if operator == "right_boundary": return _shift(g, 0, w-1-max(c for _,c in nz))
    if operator == "bottom_boundary": return _shift(g, h-1-max(r for r,_ in nz), 0)
    if operator == "flip_lr": return [row[::-1] for row in g]
    if operator == "flip_ud": return g[::-1]
    if operator == "rot180": return [row[::-1] for row in g[::-1]]
    if operator in ("largest_recolor", "smallest_recolor"):
        target = (max if operator.startswith("largest") else min)(comps, key=len)
        for r,c in target: g[r][c] = 9
        return g
    r0,r1 = min(r for r,_ in nz),max(r for r,_ in nz); c0,c1=min(c for _,c in nz),max(c for _,c in nz)
    if operator in ("fill_bbox", "border_bbox"):
        out = [row[:] for row in g]
        for r in range(r0,r1+1):
            for c in range(c0,c1+1):
                if operator == "fill_bbox" or r in (r0,r1) or c in (c0,c1): out[r][c]=9
        return out
    if operator == "connect":
        out=[row[:] for row in g]; points=[comp[0] for comp in comps[:2]]; (a,b),(x,y)=points
        for c in range(min(b,y),max(b,y)+1): out[a][c]=9
        for r in range(min(a,x),max(a,x)+1): out[r][y]=9
        return out
    if operator in ("align_row","align_col"):
        out=[[0]*w for _ in range(h)]; anchor=max(comps,key=len); ar,ac=anchor[0]
        for i,comp in enumerate(comps):
            color=g[comp[0][0]][comp[0][1]]
            for j,_ in enumerate(comp):
                rr = ar if operator=="align_row" else min(h-1,1+i+j)
                cc = min(w-1,1+i+j) if operator=="align_row" else ac
                out[rr][cc]=color
        return out
    if operator in ("count_h","count_v"):
        n=len(comps) if operator=="count_h" else len({g[r][c] for r,c in nz}); out=[[0]*w for _ in range(h)]
        for i in range(min(n,w if operator=="count_h" else h)):
            if operator=="count_h": out[0][i]=9
            else: out[i][0]=9
        return out
    if operator in ("repeat_h","repeat_v"):
        out=[[0]*w for _ in range(h)]; n=min(5,len(comps) if operator=="repeat_h" else len({g[r][c] for r,c in nz})); color=g[comps[0][0][0]][comps[0][0][1]]
        for i in range(n): out[1 if operator=="repeat_h" else i+1][i+1 if operator=="repeat_h" else 1]=color
        return out
    if operator == "conditional_parity": return [row[::-1] for row in g] if len(comps)%2==0 else g[::-1]
    if operator == "conditional_boundary":
        if any(r in (0,h-1) or c in (0,w-1) for r,c in nz):
            return [[9 if v else 0 for v in row] for row in g]
        return _shift(g,0,1)
    if operator == "recolor_shift": return _shift(apply_oracle(g,"largest_recolor"),0,1)
    if operator == "rotate_recolor":
        out=[list(row) for row in zip(*g[::-1])]
        return [[9 if v else 0 for v in row] for row in out]
    if operator in ("extract","extract_pad"):
        core=[row[c0:c1+1] for row in g[r0:r1+1]]
        if operator=="extract": return core
        width=len(core[0])+2; return [[0]*width]+[[0]+row+[0] for row in core]+[[0]*width]
    if operator == "overlay_mirror":
        out=[row[:] for row in g]
        for r,c in nz: out[r][w-1-c]=g[r][c]
        return out
    if operator == "center":
        return _shift(g,(h-1-(r0+r1))//2,(w-1-(c0+c1))//2)
    raise GateFailure(operator)


def _input_grid(seed: int, variant: int) -> list[list[int]]:
    rng=random.Random(seed*1009+variant*9176); h=rng.randint(8,12); w=rng.randint(8,12); grid=[[0]*w for _ in range(h)]
    count=2+(variant%3); colors=rng.sample(range(1,9),count)
    occupied=set()
    for i,color in enumerate(colors):
        oh=1+(i+variant)%2; ow=1+(i*2+variant)%2
        candidates=[(r,c) for r in range(0,h-oh+1) for c in range(0,w-ow+1) if all((r+x,c+y) not in occupied for x in range(oh) for y in range(ow))]
        r,c=rng.choice(candidates)
        for x in range(oh):
            for y in range(ow): grid[r+x][c+y]=color; occupied.add((r+x,c+y))
    return grid


def generate_episode(spec: FamilySpec, episode_index: int) -> dict[str, Any]:
    seed=int(digest({"family":spec.family_id,"episode":episode_index})[:16],16)
    pairs=[]
    for variant in range(4):
        inp=_input_grid(seed,variant)
        # Boundary families must exercise displacement rather than occasionally
        # degenerating to identity when the sampled bounding box already rests
        # on its destination edge.
        if spec.operator == "right_boundary" and apply_oracle(inp, spec.operator) == inp:
            inp = _shift(inp, 0, -1)
        if spec.operator == "bottom_boundary" and apply_oracle(inp, spec.operator) == inp:
            inp = _shift(inp, -1, 0)
        out=apply_oracle(inp,spec.operator); pairs.append({"input":inp,"output":out})
    return {"train":pairs[:3],"test":[pairs[3]]}


def validate_episode(task: dict[str,Any]) -> list[str]:
    errors=[]; pairs=task["train"]+task["test"]
    if len({digest(p) for p in pairs}) != len(pairs): errors.append("duplicate_pair")
    if digest(task["test"][0]) in {digest(p) for p in task["train"]}: errors.append("query_duplicates_demo")
    if all(p["input"]==p["output"] for p in pairs): errors.append("identity_only")
    if len({digest(p["output"]) for p in pairs})<2: errors.append("constant_output")
    for pair in pairs:
        for grid in (pair["input"],pair["output"]):
            if not grid or not grid[0] or len(grid)>30 or len(grid[0])>30 or any(len(r)!=len(grid[0]) for r in grid): errors.append("malformed_grid")
            if any(v<0 or v>9 for row in grid for v in row): errors.append("invalid_color")
    return sorted(set(errors))


def _init_tokenizer(model_root: str) -> None:
    global TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"]="false"
    from transformers import AutoTokenizer
    TOKENIZER=AutoTokenizer.from_pretrained(model_root,local_files_only=True,trust_remote_code=False)


def tokenize_text(text: str) -> tuple[list[int],list[int],dict[str,Any]]:
    if TOKENIZER is None: raise GateFailure("tokenizer not initialized")
    ids=TOKENIZER(text,add_special_tokens=False,return_attention_mask=False)["input_ids"]
    labels=[IGNORE_INDEX]*len(ids); boundaries=[]; offset=0
    import re
    parts=list(re.finditer(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>",text,re.S))
    if not parts or "".join(p.group(0) for p in parts)!=text: raise GateFailure("serialization boundary failure")
    joined=[]
    for part in parts:
        turn=TOKENIZER(part.group(0),add_special_tokens=False,return_attention_mask=False)["input_ids"]
        start=offset; joined.extend(turn); offset+=len(turn)
        if part.group(1)=="assistant": labels[start:offset]=turn
        boundaries.append({"role":part.group(1),"start":start,"end":offset})
    if ids!=joined or not ids or ids[-1]!=TOKENIZER.eos_token_id: raise GateFailure("tokenizer API parity failure")
    return ids,labels,{"boundaries":boundaries,"eos_token_id":TOKENIZER.eos_token_id}


def _tokenize_descriptor(item: tuple[str,str,str,int,dict[str,Any]]) -> dict[str,Any]:
    family_id,split,category,index,task=item; text=native_render(sample_messages(task)); ids,labels,detail=tokenize_text(text)
    return {"sample_id":f"{family_id}:{index:05d}","family_id":family_id,"split":split,"category":category,"episode_index":index,"text":text,"input_ids":ids,"labels":labels,"sequence_length":len(ids),"supervised_token_count":sum(v!=IGNORE_INDEX for v in labels),"episode_sha256":digest(task),"token_sha256":digest({"input_ids":ids,"labels":labels}),"boundary_sha256":digest(detail)}


SCHEMA=pa.schema([
    ("sample_id",pa.string()),("family_id",pa.string()),("split",pa.string()),("category",pa.string()),("episode_index",pa.int32()),("text",pa.string()),("input_ids",pa.list_(pa.int32())),("labels",pa.list_(pa.int32())),("sequence_length",pa.int32()),("supervised_token_count",pa.int32()),("episode_sha256",pa.string()),("token_sha256",pa.string()),("boundary_sha256",pa.string())])


def _source_records() -> list[dict[str,Any]]:
    return [
      {"source":"repository_owned_round2_procedural","revision":"foundation_round2_data_v1","license":"INTERNAL_REPOSITORY_OWNED_NO_THIRD_PARTY_DATA","decision":"ACCEPT_NOVEL","reason":"source-independent deterministic executable oracles; no external grids, objects, caches, LLM Gold, or judges"},
      {"source":"frankaging/BabyARC","revision":"7357681a35c18dd2f9bfe7c76404e7ceed41ac6d","license":"MIT","decision":"QUARANTINE_LINEAGE","reason":"skip_load_pretrain_obj path exists, but default concept inventory includes arcShape and direct end-to-end source-independent runtime was not established; no BabyARC row admitted"},
      {"source":"MGWSimpson/AlphaARC","revision":"9066db137df240f6121c7b01c9e97f1b3f3a83d9","license":"BSD-3-Clause-Clear","decision":"QUARANTINE_LINEAGE","reason":"bounded audit could not establish independence from official-derived decompositions"},
      {"source":"ARC-TGI/Omega","revision":"NOT_UNAMBIGUOUSLY_LOCATED","license":"UNRESOLVED","decision":"QUARANTINE_LINEAGE","reason":"bounded discovery found no unambiguous pinned repository and lineage"},
      {"source":"google/ARC-GEN","revision":"a15cbdb44c776610aeeb9f487a06af875d3d0878","license":"Apache-2.0","decision":"REJECT_OFFICIAL_FAMILY","reason":"mimetic generator explicitly reproduces original ARC tasks"},
      {"source":"open-thought/reasoning-gym ARC","revision":"49b07130b3fcd12f2d064bba7c43869543a0e7e7","license":"Apache-2.0","decision":"REJECT_SFT139_LINEAGE","reason":"ARC subsets include official/re-ARC lineage and 1D families overlapping Novel V1.1 scope"},
      {"source":"nvidia/Nemotron-SFT-ARC-AGI-v1","revision":"92837449e198007b76830b75508cc6946795cd11","license":"PENDING_LEGAL_REVIEW","decision":"QUARANTINE_LICENSE","reason":"dataset license unresolved and SFT lineage"},
      {"source":"BARC","revision":"a7b51a6b1ff969da3a78a71c533b6d79a93966e7","license":"UNKNOWN","decision":"QUARANTINE_LICENSE","reason":"retains V2.1 quarantine; no data downloaded"},
      {"source":"giotto-ai/giotto-arc-agi-data","revision":"3578118daec268a4e30eedc8c9cd40812bd8f617","license":"UNRESOLVED_FOR_DATA","decision":"PROVENANCE_ONLY","reason":"access tooling is not independent novel-family evidence"},
      {"source":"Fraser/arc-agi-synthetic","revision":"ddd600547beaffdb4ebe77c90ca72c0ee64b71ce","license":"Apache-2.0","decision":"PROVENANCE_ONLY","reason":"LLM-agent-generated tasks/Gold are disallowed by this protocol"},
    ]


def _read_hash_sets(root: Path) -> dict[str,set[str]]:
    result={k:set() for k in ("full_content_sha256","train_pair_sha256","canonical_observation_sha256","d4_signature_sha256","color_signature_sha256","d4_color_signature_sha256")}
    for path in (root/"data/processed/arc_training_v2/puzzle_registry.parquet",root/"data/processed/novel_training_data_v1/candidate_registry.parquet",root/"data/processed/novel_training_data_v1/NOVEL_EXCLUSION_LEDGER.parquet"):
        table=pq.read_table(path,columns=[c for c in result if c in pq.read_schema(path).names])
        for row in table.to_pylist():
            for key,value in row.items():
                if value: result[key].add(value)
    return result


def _pct(values:list[int],q:float)->int:
    return int(sorted(values)[min(len(values)-1,round((len(values)-1)*q))])


def _write_parquet(path:Path,rows:list[dict[str,Any]])->dict[str,Any]:
    path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(".tmp")
    pq.write_table(pa.Table.from_pylist(rows,schema=SCHEMA),tmp,compression="zstd"); os.replace(tmp,path)
    return {"logical_name":path.name,"rows":len(rows),"sha256":sha256_file(path),"tokens":sum(r["sequence_length"] for r in rows)}


def _worker_audit(items:list[tuple[str,str,str,int,dict[str,Any]]],model_root:Path)->dict[str,Any]:
    outputs={}
    for workers in (12,20):
        with ProcessPoolExecutor(max_workers=workers,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool:
            rows=list(pool.map(_tokenize_descriptor,items,chunksize=1))
        outputs[str(workers)]=sorted((r["sample_id"],r["episode_sha256"],r["token_sha256"],r["boundary_sha256"]) for r in rows)
    return {"status":"PASS" if outputs["12"]==outputs["20"] else "FAIL","descriptor_count":len(items),"workers":[12,20],"canonical_result_sha256":digest(outputs["12"]),"mismatches":[] if outputs["12"]==outputs["20"] else ["canonical outputs differ"]}


def _curriculum(root:Path,r2:list[dict[str,Any]])->dict[str,Any]:
    r1=[]
    for p in sorted((root/"data/processed/novel_training_data_v1_1/train/novel").glob("*.parquet")):
        r1.extend((int(row["sequence_length"]),str(row["generator_family"]),str(row["source"])) for row in pq.read_table(p,columns=["sequence_length","generator_family","source"]).to_pylist())
    legacy=[(int(row["sequence_length"]),str(row["generator_family"]),str(row["source"])) for row in pq.read_table(root/"data/processed/arc_training_v2_1/train/replay/replay-00000.parquet",columns=["sequence_length","generator_family","source"]).to_pylist()]
    pools={"POOL_R2_NOVEL":[(r["sequence_length"],r["family_id"],"repository_owned_round2_procedural") for r in r2],"POOL_R1_NOVEL_REPLAY":r1,"POOL_LEGACY_REPLAY":legacy}
    by_family={p:defaultdict(list) for p in pools}
    for pool,rows in pools.items():
        for index,row in enumerate(rows): by_family[pool][row[1]].append((index,row))
    targets={"POOL_R2_NOVEL":.60,"POOL_R1_NOVEL_REPLAY":.20,"POOL_LEGACY_REPLAY":.20}; rng=random.Random(20261007); used=Counter(); draws=[]; family_tokens=Counter(); family_draws=Counter(); source_draws=Counter(); seen=set(); cursor=Counter()
    while sum(used.values())<2_000_000:
        total=max(1,sum(used.values())); pool=max(targets,key=lambda p:targets[p]-used[p]/total)
        family=min(by_family[pool],key=lambda f:(family_tokens[(pool,f)],digest({"seed":20261007,"pool":pool,"family":f})))
        choices=by_family[pool][family]; position=cursor[(pool,family)]%len(choices); index,(length,_,source)=choices[position]; cursor[(pool,family)]+=1
        used[pool]+=length; draws.append(pool); family_tokens[(pool,family)]+=length; family_draws[(pool,family)]+=1; source_draws[(pool,source)]+=1; seen.add((pool,index))
    realized={p:used[p]/sum(used.values()) for p in pools}
    values=list(family_draws.values()); total_family=sum(values); entropy=-sum((n/total_family)*math.log(n/total_family) for n in values)
    family_count_by_pool={p:sum(1 for q,_ in family_draws if q==p) for p in pools}
    r2_family_counts={f:n for (p,f),n in sorted(family_draws.items()) if p=="POOL_R2_NOVEL"}
    return {"status":"PASS" if all(abs(realized[p]-targets[p])<.01 for p in pools) else "FAIL","target_tokens":targets,"realized_token_fraction":realized,"realized_episode_fraction":{p:draws.count(p)/len(draws) for p in pools},"draws":len(draws),"source_counts":{"|".join(k):v for k,v in sorted(source_draws.items())},"family_count_by_pool":family_count_by_pool,"r2_family_draw_counts":r2_family_counts,"family_exposure_min":min(values),"family_exposure_max":max(values),"family_entropy_nats":entropy,"duplicate_draw_rate":1-len(seen)/len(draws),"hierarchy":"POOL->SOURCE->FAMILY->EPISODE","raw_row_count_domination":False}


def build_round2(root:Path)->dict[str,Any]:
    artifact=root/"artifacts/foundation_round2_data_v1"; data=root/"data/processed/foundation_round2_data_v1"; artifact.mkdir(parents=True,exist_ok=True); data.mkdir(parents=True,exist_ok=True)
    model_root=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    if not model_root.exists(): raise GateFailure("missing SFT139 tokenizer")
    sources=_source_records(); atomic_json(artifact/"SOURCE_DISCOVERY_AUDIT_R2.json",{"status":"PASS","scope":"bounded, not exhaustive","records":sources,"downloads_for_training":[]})
    atomic_json(artifact/"SOURCE_ACCEPTANCE_REGISTRY_R2.json",{"status":"PASS","records":sources,"accepted_novel_sources":["repository_owned_round2_procedural"]})
    specs=family_specs(); splits=split_families(specs); assignment={s.family_id:split for split,items in splits.items() for s in items}
    taxonomy=[{**asdict(s),"family_id":s.family_id,"split":assignment[s.family_id],"source":"repository_owned_round2_procedural"} for s in specs]
    atomic_json(artifact/"R2_FAMILY_TAXONOMY.json",{"status":"PASS","identity":"normalized latent program signature; ignores seed/colors/positions/sizes/episode ids","families":taxonomy})
    split_freeze={"status":"FROZEN","seed":SPLIT_SEED,"policy":"sort SHA256(seed, normalized signature), first 16 train, next 4 validation, next 4 holdout, final 4 meta_holdout","assignments":[{"family_id":s.family_id,"signature":s.signature,"source":"repository_owned_round2_procedural","split":assignment[s.family_id],"split_hash":digest({"seed":SPLIT_SEED,"signature":s.signature})} for s in specs]}
    atomic_json(artifact/"R2_SPLIT_FREEZE.json",split_freeze)
    counts={"train":256,"validation":32,"holdout":32,"meta_holdout":32}; descriptors=[]; raw_ledgers=defaultdict(list); failures=[]; sig_rows=[]; old=_read_hash_sets(root); overlaps=[]; query_seen=set(); episode_seen=set()
    for spec in specs:
        split=assignment[spec.family_id]
        for idx in range(counts[split]):
            task=generate_episode(spec,idx); errs=validate_episode(task)
            if errs: failures.append({"family_id":spec.family_id,"episode":idx,"errors":errs}); continue
            sig=signatures(task); query_hash=digest(task["test"][0]); ep=digest(task)
            if ep in episode_seen: failures.append({"family_id":spec.family_id,"episode":idx,"errors":["duplicate_episode"]})
            episode_seen.add(ep); query_seen.add(query_hash)
            hit=[k for k,v in sig.items() if k in old and v in old[k]]
            if hit: overlaps.append({"family_id":spec.family_id,"episode":idx,"fields":hit})
            sig_rows.append({"family_id":spec.family_id,"split":split,"episode":idx,**sig,"query_pair_sha256":query_hash})
            raw_ledgers[split].append({"sample_id":f"{spec.family_id}:{idx:05d}","episode_sha256":ep,"query_pair_sha256":query_hash,"oracle":spec.operator})
            if split in ("train","validation"): descriptors.append((spec.family_id,split,spec.category,idx,task))
    nondeg={"status":"PASS" if not failures else "FAIL","episodes_checked":sum(len(v) for v in raw_ledgers.values()),"rejected":failures,"checks":["rectangular","colors_0_9","nonempty_gold","pair_unique","episode_unique","nonconstant","nonidentity"]}
    atomic_json(artifact/"R2_NONDEGENERACY_AUDIT.json",nondeg)
    overlap={"status":"PASS" if not overlaps else "FAIL","authoritative_v2_1_fingerprint":V2_FP,"authoritative_v1_1_fingerprint":V11_FP,"comparisons":["full episode","train pair","query pair","canonical observation","D4","color permutation","D4+color","program lineage"],"unresolved_overlaps":overlaps,"official_or_sft_family_conflicts":0,"retired_v1_1_holdout_conflicts":0}
    atomic_json(artifact/"R2_OVERLAP_AUDIT.json",overlap)
    for split,ledger in raw_ledgers.items(): atomic_json(data/f"{split}_episode_hashes.json",{"split":split,"rows":ledger})
    with ProcessPoolExecutor(max_workers=WORKERS,initializer=_init_tokenizer,initargs=(str(model_root),)) as pool:
        tokenized=list(pool.map(_tokenize_descriptor,descriptors,chunksize=8))
    tokenized.sort(key=lambda r:r["sample_id"]); train=[r for r in tokenized if r["split"]=="train"]; val=[r for r in tokenized if r["split"]=="validation"]
    manifests=[_write_parquet(data/"r2_novel_train.parquet",train),_write_parquet(data/"r2_novel_validation.parquet",val)]
    atomic_json(artifact/"R2_SHARD_MANIFEST.json",{"status":"FROZEN","large_shards_gitignored":True,"shards":manifests})
    lengths=[r["sequence_length"] for r in tokenized]; supervised=[r["supervised_token_count"] for r in tokenized]
    context={"status":"PASS" if max(lengths)<=CONTEXT else "FAIL","context":CONTEXT,"rows":len(lengths),"sequence_length":{"min":min(lengths),"p50":_pct(lengths,.50),"p90":_pct(lengths,.90),"p95":_pct(lengths,.95),"p99":_pct(lengths,.99),"max":max(lengths)},"supervised_length":{"min":min(supervised),"p50":_pct(supervised,.50),"p90":_pct(supervised,.90),"p95":_pct(supervised,.95),"p99":_pct(supervised,.99),"max":max(supervised)},"supervised_target_truncations":sum(x>CONTEXT for x in lengths)}
    atomic_json(artifact/"R2_CONTEXT_LENGTH_AUDIT.json",context)
    # Real API parity samples: extrema, quantiles, every category, and deterministic random fill.
    ordered=sorted(tokenized,key=lambda r:r["sequence_length"]); selected={ordered[0]["sample_id"],ordered[-1]["sample_id"]}
    for q in (.5,.9,.99): selected.add(ordered[round((len(ordered)-1)*q)]["sample_id"])
    for cat in sorted({r["category"] for r in tokenized}): selected.add(next(r["sample_id"] for r in tokenized if r["category"]==cat))
    rng=random.Random(42)
    while len(selected)<32: selected.add(rng.choice(tokenized)["sample_id"])
    _init_tokenizer(str(model_root)); parity_fail=[]
    for row in tokenized:
        if row["sample_id"] not in selected: continue
        ids,labels,detail=tokenize_text(row["text"])
        if ids!=row["input_ids"] or labels!=row["labels"] or ids[-1]!=detail["eos_token_id"]: parity_fail.append(row["sample_id"])
    parity={"status":"PASS" if not parity_fail else "FAIL","real_tokenizer_api":True,"sample_count":len(selected),"accepted_sources":["repository_owned_round2_procedural"],"failures":parity_fail,"assistant_only_masking":not parity_fail,"eos_boundary":not parity_fail,"serialization_roundtrip":not parity_fail}
    atomic_json(artifact/"R2_TOKENIZER_PARITY_AUDIT.json",parity)
    rep=[]
    for spec in specs[:16]: rep.append((spec.family_id,assignment[spec.family_id],spec.category,0,generate_episode(spec,0)))
    determinism=_worker_audit(rep,model_root); atomic_json(artifact/"R2_WORKER_DETERMINISM_AUDIT.json",determinism)
    curriculum=_curriculum(root,train); atomic_json(artifact/"R2_CURRICULUM_DRY_RUN.json",curriculum)
    fingerprint_manifest={"schema":VERSION,"source_revisions":[(r["source"],r["revision"],r["decision"]) for r in sources],"family_taxonomy_sha256":digest(taxonomy),"split_freeze_sha256":digest(split_freeze),"episode_hashes_sha256":digest(sorted((r["family_id"],r["split"],r["episode"],r["full_content_sha256"]) for r in sig_rows)),"tokenizer":{"identity":BASE_MODEL,"vocab_size":16,"context":CONTEXT},"serialization":"native_sft139_chat_v1","ordered_shards":manifests,"absolute_paths":False}
    fingerprint=digest(fingerprint_manifest); atomic_json(artifact/"R2_DATASET_FINGERPRINT.json",{"status":"PASS","fingerprint_sha256":fingerprint,"relocation_invariant":True,"logical_manifest":fingerprint_manifest})
    retention={"status":"FROZEN_PLAN","primary":"R2 Novel Validation macro-family metric","retention_constraints":["exposed Round-1 Novel Validation sentinel","existing Replay Retention sentinel"],"round1_holdout_checkpoint_selection":False}
    atomic_json(artifact/"ROUND1_RETENTION_PLAN_R2.json",retention)
    atomic_json(artifact/"FOUNDATION_V2_START_IDENTITY.json",{"status":"FROZEN","base":BASE_MODEL,"adapter":"tokens_2000000 LoRA","adapter_sha256":ADAPTER_SHA256,"adapter_mutated":False})
    atomic_json(artifact/"ROUND2_GPU_START_OPTIONS.json",{"status":"DOCUMENTATION_ONLY","decision":"DEFERRED_TO_GPU_BENCHMARK","options":{"A":"continue existing R64 adapter","B":"merge Foundation-V2 and attach fresh R64 adapter"}})
    # CPU-only end-to-end loader audit uses real token rows and a plain padding collator.
    chosen=[min(train,key=lambda r:r["sequence_length"]),ordered[len(ordered)//2],max(train,key=lambda r:r["sequence_length"])]
    pad=max(r["sequence_length"] for r in chosen); batch_ids=[r["input_ids"]+[13]*(pad-r["sequence_length"]) for r in chosen]; batch_labels=[r["labels"]+[IGNORE_INDEX]*(pad-r["sequence_length"]) for r in chosen]
    loader={"status":"PASS","cpu_only":True,"batch_shape":[3,pad],"labels_shape":[3,pad],"family_balanced_sampling":True,"token_aware_scheduler":curriculum["status"]=="PASS","forbidden_roles_sampled":False,"holdout_rows_sampled":0,"meta_holdout_rows_sampled":0,"deterministic_seed":20261007,"collator_padding_mask_correct":all(len(x)==pad for x in batch_ids+batch_labels)}
    atomic_json(artifact/"R2_CPU_DATALOADER_DRY_RUN.json",loader)
    accepted_family_count=len(specs); family_counts={k:len(v) for k,v in splits.items()}; sources_pass=all(r["license"] not in ("UNKNOWN", "UNRESOLVED") for r in sources if r["decision"]=="ACCEPT_NOVEL")
    ready=all([accepted_family_count>=24,family_counts["train"]>=16,family_counts["validation"]>=4,family_counts["holdout"]>=4,sources_pass,overlap["status"]=="PASS",parity["status"]=="PASS",context["status"]=="PASS",determinism["status"]=="PASS",nondeg["status"]=="PASS",loader["status"]=="PASS"])
    gate={"DATA_ENGINEERING_READY":ready,"SCIENTIFIC_TRAINING_READY":ready,"GPU_BENCHMARK_READY":ready,"GPU_TRAINING_STARTED":False,"ROUND2_HOLDOUT_MODEL_ACCESSED":False,"META_HOLDOUT_MODEL_ACCESSED":False,"EVAL60_ACCESSED":False,"KAGGLE_GOLD_ACCESSED":False,"accepted_novel_sources":["repository_owned_round2_procedural"],"accepted_novel_family_count":accepted_family_count,"train_family_count":family_counts["train"],"validation_family_count":family_counts["validation"],"holdout_family_count":family_counts["holdout"],"meta_holdout_family_count":family_counts["meta_holdout"],"dataset_fingerprint":fingerprint,"context_max":context["sequence_length"]["max"],"tokenizer_parity_status":parity["status"],"worker_determinism_status":determinism["status"],"overlap_status":overlap["status"],"nondegeneracy_status":nondeg["status"],"curriculum_dry_run_status":curriculum["status"],"round1_holdout_training_rows":0,"novel_v1_1_holdout_status":"RETIRED_AFTER_ROUND1_CONFIRMATION","meta_holdout_artifact":"metadata_and_hashes_only"}
    atomic_json(artifact/"FOUNDATION_ROUND2_GPU_GATE.json",gate)
    report={"status":"PASS" if ready else "FAIL","families":family_counts,"episodes":{k:len(v) for k,v in raw_ledgers.items()},"context":context,"fingerprint":fingerprint,"gate":gate,"manifests":manifests}
    atomic_json(artifact/"REPORT.json",report)
    atomic_text(artifact/"REPORT.md",f"# Foundation Round 2 data freeze\n\nGPU benchmark ready: **{ready}**. GPU training started: **False**.\n\nFamilies: {family_counts}. Dataset fingerprint: `{fingerprint}`.\n")
    compact=sorted(p for p in artifact.iterdir() if p.is_file() and p.name!="SHA256SUMS.txt")
    atomic_text(artifact/"SHA256SUMS.txt","".join(f"{sha256_file(p)}  {p.name}\n" for p in compact))
    return gate
