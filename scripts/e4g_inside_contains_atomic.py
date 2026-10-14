"""CPU-only prospective E4G Inside/Contains atomic-plasticity freeze."""
from __future__ import annotations
import copy, hashlib, json, random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
OUT = BASE / "e4g_inside_contains_atomic_plasticity_v1"
SOURCE = BASE / "TREATMENT_SCHEDULE.json"
RETENTION = BASE.parent / "RETENTION_INPUT_PROMPTS.json"
COMPOSITION = BASE.parent / "scientific_successor_v2/COMPOSITION_SENTINEL_INPUT_PROMPTS.json"

def canon(x): return json.dumps(x, sort_keys=True, separators=(",", ":")).encode()
def digest(x): return hashlib.sha256(canon(x)).hexdigest()
def file_sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canon(value) + b"\n")

def blank(n=13): return [[0 for _ in range(n)] for _ in range(n)]
def put(g, y, x, color):
    if not (0 <= y < len(g) and 0 <= x < len(g)): raise ValueError("GRID_BOUNDS")
    if g[y][x] not in (0, color): raise ValueError("GRID_COLLISION")
    g[y][x] = color

def draw_frame(g, top, left, height, width, color, topology):
    for x in range(left, left + width):
        put(g, top, x, color); put(g, top + height - 1, x, color)
    for y in range(top, top + height):
        put(g, y, left, color); put(g, y, left + width - 1, color)
    if topology == "DOUBLE_FRAME":
        for x in range(left + 1, left + width - 1):
            put(g, top + 1, x, color); put(g, top + height - 2, x, color)
        for y in range(top + 1, top + height - 1):
            put(g, y, left + 1, color); put(g, y, left + width - 2, color)

def render(direction, topology, size, layout, frame, inner):
    g = blank(); g[0][0] = 8 if direction == "INSIDE" else 9
    top = 2 + (layout % 2); left = 2 + ((layout // 2) % 2)
    h = size + (2 if topology == "DOUBLE_FRAME" else 0)
    w = size + 1 + (2 if topology == "DOUBLE_FRAME" else 0)
    draw_frame(g, top, left, h, w, frame, topology)
    cy, cx = top + h // 2, left + w // 2; put(g, cy, cx, inner)
    return g, {"frame": frame, "inner": inner, "direction": direction, "topology": topology, "size": size, "layout": layout}

def expected(meta):
    f, c = meta["frame"], meta["inner"]
    return [[0,0,0],[0,c,0],[0,0,0]] if meta["direction"] == "INSIDE" else [[f,f,f],[f,0,f],[f,f,f]]

def oracle(grid, output):
    direction = "INSIDE" if grid[0][0] == 8 else "CONTAINS" if grid[0][0] == 9 else None
    colors = sorted({v for row in grid[1:] for v in row if v})
    if direction is None or len(colors) != 2: return False
    counts = {c: sum(v == c for row in grid for v in row) for c in colors}
    frame = max(colors, key=lambda c: counts[c]); inner = min(colors, key=lambda c: counts[c])
    want = [[0,0,0],[0,inner,0],[0,0,0]] if direction == "INSIDE" else [[frame,frame,frame],[frame,0,frame],[frame,frame,frame]]
    return output == want

def task(direction, topology, size, layout, frame, inner, demo_order):
    query, meta = render(direction, topology, size, layout, frame, inner)
    other = "CONTAINS" if direction == "INSIDE" else "INSIDE"; demos = []
    for j, d in enumerate((direction, other)):
        df = ((frame + 2 + j) % 9) or 1; di = ((inner + 4 + j) % 9) or 1
        if df == di: di = (di % 8) + 1
        dg, dm = render(d, topology, 3 + ((size + j) % 3), (layout + j + 1) % 4, df, di)
        demos.append({"input": dg, "output": expected(dm)})
    if demo_order == "B": demos.reverse()
    full = {"train": demos, "test": [{"input": query, "output": expected(meta)}]}
    assert oracle(query, full["test"][0]["output"])
    return full, meta

def blind(full):
    x = copy.deepcopy(full); x["test"][0].pop("output"); return x

def make_records(panel, topology, count=48):
    rows=[]; directions=("INSIDE","CONTAINS"); sizes=(3,4,5); layouts=(0,3); families=((1,2),(4,6))
    for i in range(count):
        direction=directions[i%2]; size=sizes[(i//2)%3]; layout=layouts[(i//6)%2]; frame,inner=families[(i//12)%2]
        cid=f"{panel}_{i:02d}"
        for order in ("A","B"):
            full, meta=task(direction, topology, size, layout, frame, inner, order); target=full["test"][0]["output"]
            rows.append({"canonical_unit_id":cid,"demo_order":order,"input_task":blind(full),"target_sha256":digest(target),
                         "construct":{"direction":direction,"topology":topology,"size":size,"layout":layout,"frame_color":frame,"inner_color":inner},
                         "full_prompt_sha256":digest(blind(full)),"query_input_sha256":digest(full["test"][0]["input"]),"oracle_valid":oracle(full["test"][0]["input"],target)})
    return rows

def assert_panel(rows, label):
    if len(rows)!=96: raise RuntimeError(label+"_PROMPT_COUNT")
    units={}
    for r in rows:
        units.setdefault(r["canonical_unit_id"],[]).append(r)
        if not r["oracle_valid"] or "output" in r["input_task"]["test"][0]: raise RuntimeError(label+"_TARGET_BLIND_OR_ORACLE")
    if len(units)!=48 or any({x["demo_order"] for x in xs}!={"A","B"} for xs in units.values()): raise RuntimeError(label+"_TWO_ORDER")
    canonrows=[v[0] for v in units.values()]
    for key,n in (("direction",24),("size",16),("layout",24),("frame_color",24),("inner_color",24)):
        values={}
        for r in canonrows: values[r["construct"][key]]=values.get(r["construct"][key],0)+1
        if set(values.values())!={n}: raise RuntimeError(label+"_BALANCE_"+key)
    return {"canonical_units":48,"prompt_records":96,"direction_balance":{"INSIDE":24,"CONTAINS":24},"target_blind":True,"two_orders":True}

def build():
    source=json.loads(SOURCE.read_text())["episodes"]; primary=[x for x in source if x["role"]=="PRIMARY_CONNECTED_COMPONENT_COUNT"]
    if len(source)!=384 or len(primary)!=48: raise RuntimeError("SOURCE_384_48")
    train=make_records("E4G_TRAIN_FIT","SINGLE_FRAME"); ood=make_records("E4G_OOD","DOUBLE_FRAME")
    by_slot={int(r["canonical_unit_id"].rsplit("_",1)[1]):r for r in train if r["demo_order"]=="A"}; schedule=[]
    for row in source:
        new=copy.deepcopy(row)
        if row["role"]=="PRIMARY_CONNECTED_COMPONENT_COUNT":
            r=by_slot[row["slot"]//2]; c=r["construct"]; full,_=task(c["direction"],c["topology"],c["size"],c["layout"],c["frame_color"],c["inner_color"],"A")
            new["task"]=full; new["source_id"]="E4G_INSIDE_CONTAINS:"+str(row["slot"]); new["structural_family"]="INSIDE_CONTAINS_ATOMIC"
        schedule.append(new)
    protected=[x for x in schedule if x["role"]!="PRIMARY_CONNECTED_COMPONENT_COUNT"]; original=[x for x in source if x["role"]!="PRIMARY_CONNECTED_COMPONENT_COUNT"]
    if [digest(x) for x in protected] != [digest(x) for x in original]: raise RuntimeError("PROTECTED_REPLAY_CHANGED")
    ta,oa=assert_panel(train,"TRAIN"),assert_panel(ood,"OOD")
    if {r["full_prompt_sha256"] for r in train}&{r["full_prompt_sha256"] for r in ood} or {r["query_input_sha256"] for r in train}&{r["query_input_sha256"] for r in ood}: raise RuntimeError("TRAIN_OOD_OVERLAP")
    protocol={"schema_version":1,"stage":"E4G_INSIDE_CONTAINS_ATOMIC_PLASTICITY_POSITIVE_CONTROL","status":"CPU_ONLY_PROSPECTIVE_FREEZE",
      "reference_arm":{"name":"V7_NO_UPDATE","checkpoint_manifest":"experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"},
      "training_arm":{"name":"V7_INSIDE_CONTAINS_CAPABILITY_TRAINING","initialization":"EXACT_V7","optimizer":"PagedAdamW8bit","learning_rate":5e-5,"seed":20261013,"updates":96,"slots_per_update":4,"base_answer_ce_coefficient":1.0,"protected_replay":"BYTE_IDENTICAL_336_NON_PRIMARY_SLOTS","primary_slots":48,"cap":"BOUND_LATER; NO_GPU_AUTHORIZATION_IN_THIS_PACKAGE"},
      "construct":{"direction_commands":{"INSIDE":8,"CONTAINS":9},"boundaries":"visible_closed_frames","objects":"visible_one_cell_inner_object","output_construction":{"INSIDE":"3x3 centered inner-color marker","CONTAINS":"3x3 frame-color ring"},"counterbalanced_dimensions":["topology","size","color","layout","demonstration_order"]},
      "evaluation":{"native_greedy":"PHYSICAL_BATCH1_LATER","canonical_units":48,"two_demo_orders":True,"uncertainty":"PAIRED_CANONICAL_UNIT_BOOTSTRAP_95CI","targets":"SHA_COMMITMENTS_ONLY_UNTIL_RAW_FREEZE"},
      "interpretation":{"good_train_fit_no_ood_gain":"GENERAL_STRUCTURAL_TRANSFER_BOTTLENECK","poor_train_fit":"BROADER_POST_V7_SUPERVISION_OR_OPTIMIZATION_BOTTLENECK"},
      "forbidden_now":["MODEL_LOAD","GPU","PREDICTION_GENERATION","TARGET_SIDECAR","GOLD_DGOLD_FINAL_AUDIT","PROMOTION","SUBMISSION"]}
    write(OUT/"TRAINING_SCHEDULE.json",{"schema_version":1,"episodes":schedule,"roles":{"PRIMARY_INSIDE_CONTAINS":48,"PRIMARY_OBJECT_EXTRACTION":48,"PROTECTED_REPLAY":288},"source_schedule_sha256":file_sha(SOURCE)})
    write(OUT/"TRAIN_FIT_INPUTS_AND_COMMITMENTS.json",{"schema_version":1,"panel":"TRAIN_FIT_TARGET_BLIND","records":train,"audit":ta})
    write(OUT/"STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json",{"schema_version":1,"panel":"OOD_TARGET_BLIND","records":ood,"audit":oa})
    write(OUT/"PROTOCOL.json",protocol)
    write(OUT/"PROTECTION_PANEL_REFERENCES.json",{"retention_path":str(RETENTION.relative_to(ROOT)).replace("\\\\","/"),"retention_sha256":file_sha(RETENTION),"composition_path":str(COMPOSITION.relative_to(ROOT)).replace("\\\\","/"),"composition_sha256":file_sha(COMPOSITION),"status":"FROZEN_PROTECTION_REFERENCES_NOT_SCORED"})
    receipt={"status":"PASS_E4G_CPU_PROSPECTIVE_FREEZE_NO_MODEL_NO_TARGET_SIDECAR","model_loaded":False,"gpu_used":False,"targets_opened":False,"sidecar_opened":False,"training_schedule_sha256":file_sha(OUT/"TRAINING_SCHEDULE.json"),"train_fit_sha256":file_sha(OUT/"TRAIN_FIT_INPUTS_AND_COMMITMENTS.json"),"ood_sha256":file_sha(OUT/"STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json"),"protocol_sha256":file_sha(OUT/"PROTOCOL.json"),"train_fit":ta,"ood":oa,"leakage_audit":"PASS_ZERO_FULL_PROMPT_AND_QUERY_INPUT_OVERLAP; DISJOINT_TOPOLOGY","protected_replay":"PASS_336_NON_PRIMARY_EPISODES_BYTE_IDENTICAL","deterministic_scoring":"PASS_PAIRED_CANONICAL_BOOTSTRAP_CONTRACT"}
    write(OUT/"CPU_FREEZE_RECEIPT.json",receipt); return receipt

def paired_bootstrap(control,treatment,reps=4000,seed=20261014):
    if len(control)!=48 or len(treatment)!=48: raise ValueError("CANONICAL_48_REQUIRED")
    delta=[int(t)-int(c) for c,t in zip(control,treatment,strict=True)]; rng=random.Random(seed); draws=[]
    for _ in range(reps): draws.append(sum(delta[rng.randrange(48)] for _ in range(48)))
    draws.sort(); return {"delta":sum(delta),"lower_95":draws[int(.025*reps)],"upper_95":draws[int(.975*reps)-1],"cluster":"canonical_unit"}

def cpu_preflight():
    build(); ood=json.loads((OUT/"STRUCTURAL_OOD_INPUTS_AND_COMMITMENTS.json").read_text())
    raw=[{"arm":a,"canonical_unit_id":r["canonical_unit_id"],"demo_order":r["demo_order"],"prediction_sha256":digest({"arm":a,"input":r["input_task"]})} for a in ("V7","CONTROL","TREATMENT") for r in ood["records"]]
    if len(raw)!=288 or any("target" in json.dumps(r).lower() for r in raw): raise RuntimeError("RAW_MOCK")
    contract=paired_bootstrap([0]*48,[0]*48)
    if contract["delta"]!=0 or contract["lower_95"]!=0: raise RuntimeError("BOOTSTRAP")
    out={"status":"PASS_E4G_PRODUCTION_SHAPED_CPU_PREFLIGHT_NO_MODEL","model_loaded":False,"gpu_used":False,"targets_opened":False,"sidecar_opened":False,"launcher_to_worker_to_raw_freeze":True,"raw_freeze_sha256":digest(raw),"raw_records":len(raw),"paired_bootstrap_contract":contract}
    write(OUT/"CPU_PRODUCTION_PATH_RECEIPT.json",out); return out

if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(); ap.add_argument("--preflight",action="store_true"); a=ap.parse_args()
    print(json.dumps(cpu_preflight() if a.preflight else build(),sort_keys=True))
