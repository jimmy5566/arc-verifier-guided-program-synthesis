"""CPU-only locator and conservative loss-mass weights for count supervision."""
from __future__ import annotations

def count_terminal_label_index(task):
    pairs=[*task["train"],*task["test"]]
    out=pairs[-1]["output"]
    if len(out)!=3 or any(len(r)!=3 for r in out): raise ValueError("COUNT_OUTPUT_SHAPE")
    run=[i for i,v in enumerate(out[1]) if v!=0]
    if not run or run!=list(range(len(run))) or any(v!=0 for v in out[0]+out[2]): raise ValueError("COUNT_OUTPUT_SEMANTICS")
    # Exact NVARC assistant turn: <im_start>, assistant, newline, then 3x3 grid.
    def grid_len(grid): return len("\n".join("".join(map(str,row)) for row in grid))
    prefix=sum(3+grid_len(pair["input"])+1+3+grid_len(pair["output"])+1 for pair in pairs[:-1])
    prefix += 3+grid_len(pairs[-1]["input"])+1
    return prefix+3+4+run[-1], len(run)

def matched_weights(labels, decision_index, multiplier=2.0):
    valid=[i for i,x in enumerate(labels) if x!=-100]
    if decision_index not in valid or len(valid)<2: raise ValueError("DECISION_INDEX_INVALID")
    control={i:1.0 for i in valid}; treatment={i:(multiplier if i==decision_index else (len(valid)-multiplier)/(len(valid)-1)) for i in valid}
    if abs(sum(control.values())-sum(treatment.values()))>1e-12 or min(treatment.values())<=0: raise ValueError("LOSS_MASS_MISMATCH")
    return control,treatment
