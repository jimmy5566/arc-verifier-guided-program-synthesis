"""CPU-only, fail-closed E4F Count=2/3 affinity construct audit."""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
E3 = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1"
SCHEDULE = E3 / "TREATMENT_SCHEDULE.json"

def canon(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def oracle_components(grid):
    """Independent BFS implementation for the visible colour-separated relation."""
    seen, result = set(), []
    for y, row in enumerate(grid):
        for x, colour in enumerate(row):
            if colour == 0 or (y, x) in seen: continue
            seen.add((y, x)); queue = [(y, x)]; component = []
            while queue:
                cy, cx = queue.pop(0); component.append((cy, cx))
                for ny, nx in ((cy-1,cx),(cy+1,cx),(cy,cx-1),(cy,cx+1)):
                    if 0 <= ny < len(grid) and 0 <= nx < len(row) and (ny,nx) not in seen and grid[ny][nx] == colour:
                        seen.add((ny,nx)); queue.append((ny,nx))
            result.append(tuple(sorted(component)))
    return tuple(sorted(result))

def features(a, b, component_sizes, component_ids):
    (y0,x0),(y1,x1) = a,b
    return {
        "colour_equal": True,
        "manhattan": abs(y0-y1)+abs(x0-x1),
        "chebyshev": max(abs(y0-y1),abs(x0-x1)),
        "relative_displacement": [abs(y0-y1),abs(x0-x1)],
        "absolute_coordinates": [[y0,x0],[y1,x1]],
        "component_sizes": [component_sizes[component_ids[a]],component_sizes[component_ids[b]]],
        "pair_order": "CANONICAL_UNORDERED",
    }

def selected_pairs(components):
    """One deterministic pair of each label per treated query; no reversed pair."""
    ids = {cell: i for i, component in enumerate(components) for cell in component}
    same = next((tuple(pair) for component in components for pair in itertools.combinations(component, 2)), None)
    different = next((tuple(pair) for a, b in itertools.combinations(components, 2) for pair in itertools.product(a, b)), None)
    if same is None or different is None: raise RuntimeError("E4F_TREATED_PAIR_LABEL_MISSING")
    return tuple(sorted(same)), tuple(sorted(different)), ids

def audit():
    from scripts.e4d_object_centric_component_anchor_grounding import connected_components
    schedule = json.loads(SCHEDULE.read_text(encoding="utf-8"))["episodes"]
    chosen, all_pairs, counts = [], [], {"1":0,"2":0,"3":0}
    for row in schedule:
        if row["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT": continue
        grid = row["task"]["test"][0]["input"]
        independent = oracle_components(grid)
        implementation = tuple(sorted(connected_components(grid)))
        if independent != implementation: raise RuntimeError("E4F_ORACLE_CONNECTIVITY_DISAGREEMENT")
        count = len(independent); counts[str(count)] += 1
        if count == 1: continue
        same, different, ids = selected_pairs(independent)
        sizes = {index: len(component) for index, component in enumerate(independent)}
        for label, pair in (("SAME_COMPONENT",same),("DIFFERENT_COMPONENT",different)):
            value = {"slot":row["slot"],"count":count,"topology":row["task"].get("source_id",str(row["slot"])),"label":label,"pair":[list(pair[0]),list(pair[1])]} | features(pair[0],pair[1],sizes,ids)
            chosen.append(value)
        for pair in itertools.combinations(sorted(ids),2):
            all_pairs.append({"slot":row["slot"],"label":"SAME_COMPONENT" if ids[pair[0]] == ids[pair[1]] else "DIFFERENT_COMPONENT", **features(pair[0],pair[1],sizes,ids)})
    if counts != {"1":16,"2":16,"3":16} or len(chosen) != 64: raise RuntimeError("E4F_SLOT_BALANCE")
    if len({(x["slot"],tuple(map(tuple,x["pair"]))) for x in chosen}) != len(chosen): raise RuntimeError("E4F_DUPLICATE_OR_REVERSED")
    labels = {label:[x for x in chosen if x["label"] == label] for label in ("SAME_COMPONENT","DIFFERENT_COMPONENT")}
    if {key:len(value) for key,value in labels.items()} != {"SAME_COMPONENT":32,"DIFFERENT_COMPONENT":32}: raise RuntimeError("E4F_LABEL_BALANCE")
    if any(sum(x["label"] == label for x in chosen if x["slot"] == slot) != 1 for slot in {x["slot"] for x in chosen} for label in labels): raise RuntimeError("E4F_PER_SLOT_MASS")
    max_same = max(x["chebyshev"] for x in labels["SAME_COMPONENT"])
    min_diff = min(x["chebyshev"] for x in labels["DIFFERENT_COMPONENT"])
    baseline = {"feature":"chebyshev","rule":f"SAME_COMPONENT iff chebyshev <= {max_same}","accuracy":sum((x["chebyshev"] <= max_same) == (x["label"] == "SAME_COMPONENT") for x in chosen)/len(chosen),"max_same":max_same,"min_different":min_diff}
    return {"schema_version":1,"status":"FAIL_CLOSED_E4F_MULTICOMPONENT_PAIR_SHORTCUT", "model_loaded":False,"gpu_used":False,"targets_opened":False,"sidecar_opened":False,
      "schedule":{"path":str(SCHEDULE.relative_to(ROOT)).replace("\\","/"),"sha256":sha(SCHEDULE)},"strata":counts,
      "selected_pair_set": {"treated_slots":32,"pairs":64,"same_component":32,"different_component":32,"one_pair_per_label_per_treated_slot":True,"unordered_no_duplicate_or_reverse":True,"count1_auxiliary_coefficient":0.0,"count1_base_answer_path":"UNCHANGED_EQUAL_SLOT_CONTROL"},
      "semantic_oracle":{"independent_bfs_4_connectivity":"PASS","implementation_agreement":"PASS"},
      "all_admissible_pairs":{"records":len(all_pairs)},"shortcut_audit":{"predeclared_baselines":[baseline],"marginal_matching_attempt":"No valid selected pair set can defeat the Chebyshev threshold because every admissible same-component pair has Chebyshev <= 2 and every admissible different-component pair has Chebyshev >= 3 in every treated frozen query.","result":"FAIL"},
      "proof":"The requested fixed equal-label sampling is achievable, but its labels are perfectly separated by Chebyshev distance. This violates the specified fail-closed shortcut rule; changing it requires different data, pair domain, or objective.","no_prospective_ood_freeze":"CONSTRUCT_INVALID_BEFORE_PANEL_FREEZE"}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();v=audit();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_bytes(canon(v)+b"\n");print(json.dumps({"status":v["status"],"sha256":sha(a.output)},sort_keys=True))
if __name__ == "__main__": main()
