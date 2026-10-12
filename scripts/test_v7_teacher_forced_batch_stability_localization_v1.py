#!/usr/bin/env python3
"""CPU-only regression tests for V7 teacher-forced batch localization."""
from __future__ import annotations
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("localize", ROOT / "scripts" / "v7_teacher_forced_batch_stability_localization_v1.py")
M = importlib.util.module_from_spec(spec); spec.loader.exec_module(M)

PROTOCOL = {
    "scope":"BATCH_STABILITY_ONLY_NOT_CAPABILITY_ESTIMATION",
    "measurement":{"physical_batch_sizes":[1,2,4,8,12],"minimum_repeats_per_condition":2,"records":12,"teacher_forced_only":True,"generation_calls":0,"optimizer_steps":0,"backward_calls":0},
    "fresh_authorization_required":{"gpu_execution":True},
    "within_condition_tolerance": {"rank_abs_delta_max": 0, "top_flag_abs_delta_max": 0, "margin_abs_delta_max": 1e-6, "nll_abs_delta_max": 1e-6},
    "cross_batch_tolerance": {"rank_abs_delta_max": 0, "top_flag_abs_delta_max": 0, "margin_abs_delta_max": 0.02, "nll_abs_delta_max": 0.005},
}

def records():
    return [{"cohort_record_id": f"B1:{index:02d}", "manifest_index": index,
             "prompt_token_ids": [11] * (3 + index % 5), "target_token_ids": [2, 3]}
            for index in range(12)]

class Runtime:
    def __init__(self, *, cross_drift=False, repeat_drift=False, malformed_mask=False):
        self.cross_drift=cross_drift; self.repeat_drift=repeat_drift; self.malformed_mask=malformed_mask; self.calls=[]
    def teacher_forced(self, packed, **identity):
        self.calls.append((identity["requested_batch_size"], identity["repeat_identity"], packed["effective_shape"]))
        output={}
        for row in packed["records"]:
            base=[[0,0,4,1],[0,0,1,4]]
            if self.cross_drift and identity["requested_batch_size"] == 8:
                base=[[0,4,0,1],[0,0,1,4]]
            if self.repeat_drift and identity["repeat_identity"] == 1 and identity["requested_batch_size"] == 4:
                base=[[0,4,0,1],[0,0,1,4]]
            output[row["cohort_record_id"]]=base
        if self.malformed_mask and identity["repeat_identity"] == 1 and identity["requested_batch_size"] == 2:
            # This represents a worker-side layout bug exposed by runtime metadata.
            packed["records"][0]["attention_mask"]=[1] * len(packed["records"][0]["attention_mask"])
        return output

def main():
    source=(ROOT / "scripts" / "v7_teacher_forced_batch_stability_localization_v1.py").read_text()
    assert "import torch" not in source and "import subprocess" not in source and "import transformers" not in source
    shapes={batch:[len(item) for item in M.physical_batches(records(),batch)] for batch in M.CONDITIONS}
    assert shapes == {1:[1]*12,2:[2]*6,4:[4]*3,8:[8,4],12:[12]}
    stable=M.launcher(records(),Runtime(),PROTOCOL,{"cpu_fixture_only":True,"gpu_execution_authorized":False})
    assert stable["scored"]["classification"] == "STABLE_ACROSS_BATCHES"
    assert len(stable["records"]) == 120 and stable["generation_calls"] == stable["optimizer_steps"] == stable["backward_calls"] == 0
    assert all({"rank","top1","top2","margin","nll"} == set(token) for row in stable["records"] for token in row["token_metrics"])
    assert all(row["padding_side"] == "left" and row["correct_token_indexing"] == "LEFT_PADDED_PROMPT_LAST_PLUS_TARGET_OFFSET" for row in stable["records"])
    drift=M.launcher(records(),Runtime(cross_drift=True),PROTOCOL,{"cpu_fixture_only":True,"gpu_execution_authorized":False})
    assert drift["scored"]["classification"] == "REPRODUCIBLE_BATCH_DEPENDENT_NUMERICAL_EFFECT"
    unstable=M.launcher(records(),Runtime(repeat_drift=True),PROTOCOL,{"cpu_fixture_only":True,"gpu_execution_authorized":False})
    assert unstable["scored"]["classification"] == "RUNTIME_INSTABILITY"
    malformed=M.launcher(records(),Runtime(malformed_mask=True),PROTOCOL,{"cpu_fixture_only":True,"gpu_execution_authorized":False})
    assert malformed["scored"]["classification"] == "IMPLEMENTATION_DEFECT"
    print("PASS_V7_TEACHER_FORCED_BATCH_STABILITY_LOCALIZATION_V1_CPU")

if __name__ == "__main__": main()
