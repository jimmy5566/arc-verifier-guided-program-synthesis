#!/usr/bin/env python3
"""Target-blind P4 cumulative-NLL best-first controller.

P4 changes only the logical pop order of already-retained non-EOS work.
Gold scoring is deliberately outside this controller.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.audit_search_order_adapters import audit_adapter_state
from scripts.build_search_order_p3_lds_micro12_cohort import sha_file, sha_value
from scripts.run_search_order_micro24_v1 import _frozen_input_identity, _policy, _runtime_probe
from scripts.run_ttt24_aug8_r1024_core_v1 import AUG8, _atomic_csv, _atomic_json, _read, _verify, _worker

EXPERIMENT = "SEARCH_ORDER_P4_NLL_BEST_FIRST_MICRO12_V1"
POLICY = "CUMULATIVE_NLL_BEST_FIRST_V1"
COHORT_SHA256 = "3852056112336e58559ceb0bb6f3171600ff36e0d6b02d1c9c03fc0da334443e"
CHECKPOINTS = "256,512,768,1024"
CONTROL_FILES = frozenset({"GENERATION_HASHES.json","GENERATION_HASH_VERIFICATION.json","HASHES.json","HASH_VERIFICATION.json"})


def head() -> str:
    return subprocess.run(["git","-C",str(ROOT),"rev-parse","HEAD"],text=True,capture_output=True,check=True).stdout.strip()


def safe(output_id: str) -> str:
    return output_id.replace(":","_")


def paths(run: Path, output_id: str) -> tuple[Path,Path,Path,Path,Path,Path]:
    stem=safe(output_id)
    return (
        run/"RAW_OUTPUTS"/f"{stem}.json",
        run/"OUTPUT_CHECKPOINTS"/f"{stem}.json",
        run/"EOS_EVENTS"/f"{stem}.jsonl.gz",
        run/"OUTPUT_RECEIPTS"/f"{stem}.json",
        run/"OUTPUT_HASHES"/f"{stem}.json",
        run/"OUTPUT_HASH_VERIFICATION"/f"{stem}.json",
    )


def output_is_frozen(run: Path, output_id: str) -> bool:
    raw,cp,eos,receipt,ledger,verification=paths(run,output_id)
    present=[p.is_file() for p in (raw,cp,eos,receipt,ledger,verification)]
    if not any(present):
        return False
    if not all(present):
        raise RuntimeError(f"P4_UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{output_id}")
    if _read(receipt).get("status")!="COMPLETE" or _read(verification).get("status")!="PASS":
        raise RuntimeError(f"P4_OUTPUT_RECEIPT_INVALID:{output_id}")
    if _verify(run,ledger).get("status")!="PASS":
        raise RuntimeError(f"P4_OUTPUT_HASH_REUSE_REFUSED:{output_id}")
    return True


def freeze_output(run: Path, output_id: str) -> None:
    raw,cp,eos,receipt,ledger,verification=paths(run,output_id)
    base=(raw,cp,eos,receipt)
    if any(not p.is_file() for p in base) or _read(receipt).get("status")!="COMPLETE":
        raise RuntimeError(f"P4_OUTPUT_ATOMIC_ARTIFACT_FAIL:{output_id}")
    _atomic_json(ledger,{"output_id":output_id,"files":{str(p.relative_to(run)):sha_file(p) for p in base}})
    checked=_verify(run,ledger)
    _atomic_json(verification,checked)
    if checked.get("status")!="PASS":
        raise RuntimeError(f"P4_OUTPUT_HASH_FREEZE_FAIL:{output_id}")


def ledger(run: Path, name: str, *, include_raw: bool) -> dict[str,Any]:
    files=[]
    for p in run.rglob("*"):
        if not p.is_file() or p.name in CONTROL_FILES or "WORKER_LOGS" in p.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS","OUTPUT_CHECKPOINTS","EOS_EVENTS"} for part in p.parts):
            continue
        files.append(p)
    payload={"files":{str(p.relative_to(run)):sha_file(p) for p in sorted(files)},"includes_raw":include_raw}
    _atomic_json(run/name,payload)
    return payload


def load_cohort(path: Path) -> dict[str,Any]:
    cohort=_read(path)
    if cohort.get("gold_loaded") is not False:
        raise RuntimeError("P4_COHORT_GOLD_STATE_INVALID")
    expected=sha_value({k:v for k,v in cohort.items() if k!="cohort_sha256"})
    if cohort.get("cohort_sha256")!=expected or expected!=COHORT_SHA256:
        raise RuntimeError("P4_COHORT_HASH_INVALID")
    outputs=list(cohort.get("outputs",[]))
    counts=Counter(str(x.get("category")) for x in outputs)
    if len(outputs)!=12 or len({x.get("output_id") for x in outputs})!=12 or counts!=Counter({"HIGH":3,"MID":3,"LOW":3,"CONTROL":3}):
        raise RuntimeError(f"P4_MICRO12_COHORT_INVALID:{dict(counts)}")
    return cohort


def audit_preregistration() -> dict[str,Any]:
    p=ROOT/"analysis"/"search_order_p4_nll_best_first_prereg_v1"/"PREREGISTRATION.json"
    if not p.is_file():
        raise RuntimeError("P4_PREREGISTRATION_MISSING")
    prereg=_read(p)
    policy=prereg.get("policy",{})
    if policy.get("name")!=POLICY or policy.get("pop_key")!=[
        "cumulative_nll ascending","insertion_order ascending","work_item_id ascending"
    ] or policy.get("retention_changed") is not False:
        raise RuntimeError("P4_PREREGISTRATION_METHOD_MISMATCH")
    src=(ROOT/"src"/"inference"/"search_order.py").read_text(encoding="utf-8")
    required=("class CumulativeNLLBestFirst","heapq.heappush","heapq.heappop","CUMULATIVE_NLL_BEST_FIRST_V1")
    if any(x not in src for x in required):
        raise RuntimeError("P4_IMPLEMENTATION_AUDIT_FAIL")
    return {"status":"PASS","preregistration_sha256":sha_file(p),"algorithm":POLICY,"gold_loaded":False}


def run_gate0(args: argparse.Namespace, out: Path) -> None:
    records=[]
    for files in (
        ("tests/test_search_order.py",),
        ("tests/test_nvarc_turbodfs_search_order.py",),
        ("tests/test_audit_search_order_adapters.py","tests/test_build_search_order_micro24_cohort.py"),
    ):
        cmd=[str(args.worker_python),"-m","pytest","-q",*files]
        env=dict(**__import__("os").environ); env["PYTHONPATH"]=str(ROOT)
        r=subprocess.run(cmd,cwd=ROOT,text=True,capture_output=True,env=env)
        records.append({"command":cmd,"returncode":r.returncode,"stdout":r.stdout[-4000:],"stderr":r.stderr[-4000:]})
        if r.returncode:
            _atomic_json(out/"GATE0_TESTS.json",{"status":"FAIL","gold_loaded":False,"tests":records})
            raise RuntimeError("P4_GATE0_TEST_FAILURE")
    _atomic_json(out/"GATE0_TESTS.json",{"status":"PASS","target_blind":True,"gold_loaded":False,"tests":records,"implementation_audit":audit_preregistration()})


def preflight(args: argparse.Namespace, cohort: dict[str,Any], out: Path) -> dict[str,Any]:
    runtime=_runtime_probe(args.worker_python)
    adapter=audit_adapter_state(cohort)
    if adapter.get("adapter_state")!="EXACT_HISTORICAL":
        raise RuntimeError("P4_ADAPTER_IDENTITY_FAIL")
    identity=_frozen_input_identity(args)
    profile=_read(args.coarse_policy)
    if profile.get("target_blind") is not True:
        raise RuntimeError("P4_PROFILE_POLICY_NOT_TARGET_BLIND")
    payload={"status":"PASS","experiment":EXPERIMENT,"target_blind":True,"gold_loaded":False,
             "runtime":runtime,"adapter":adapter,"input_identity":identity,"cohort_sha256":COHORT_SHA256,
             "config":{"ttt_depth":24,"augmentation":"canonical_AUG8","max_expanded_nodes":1024,
                       "decoder":"CUMULATIVE_REGRET_r=4.00","candidate_cap":32,"frontier_floor":1,
                       "eos":15,"admission":"root_aware","search_order":POLICY}}
    _atomic_json(out/"P4_PREFLIGHT.json",payload)
    return payload


def broad_profile(profile: str) -> str:
    return "PROFILE_L" if str(profile).startswith("PROFILE_L") else str(profile)


def smoke_cohort(parent: dict[str,Any]) -> list[dict[str,Any]]:
    selected=[]
    for broad in ("PROFILE_S","PROFILE_M","PROFILE_L"):
        options=[x for x in parent["outputs"] if broad_profile(str(x["profile"]))==broad]
        if not options:
            raise RuntimeError(f"P4_SMOKE_PROFILE_MISSING:{broad}")
        # Reuse the exact historical P3 deterministic smoke selector.
        selected.append(min(options,key=lambda x:hashlib.sha256(f"P3_LDS_SMOKE_V1:{x['output_id']}".encode()).hexdigest()))
    if len({x["output_id"] for x in selected})!=3:
        raise RuntimeError("P4_SMOKE_NOT_DISTINCT")
    return selected


def output_metrics(raw: dict[str,Any], category: str, profile: str) -> tuple[dict[str,Any],list[dict[str,Any]],list[dict[str,Any]]]:
    cells=list(raw["cells"].values())
    checkpoints=[r for c in cells for r in c["checkpoints"]]
    final=[r for r in checkpoints if int(r["checkpoint_requested"])==1024]
    if len(final)!=len(AUG8):
        raise RuntimeError("P4_AUG8_FINAL_CHECKPOINT_COUNT_INVALID")
    work=[]
    duplicate=0
    retention_changed=False
    for c in cells:
        rows=list(c.get("search_order_work_items",[]))
        ids=[int(r["work_item_id"]) for r in rows]
        duplicate += len(ids)-len(set(ids))
        work.extend(rows)
        retention_changed |= bool((c.get("search_order") or {}).get("retention_changed"))
    orders=[c.get("search_order") or {} for c in cells]
    scheduler=raw["scheduler"]
    useful=sum(int(o.get("useful_model_forwards",0)) for o in orders)
    replay=sum(int(o.get("replay_model_forwards",0)) for o in orders)
    row={"output_id":raw["output"]["output_id"],"category":category,"profile":profile,
         "wall_seconds":raw["wall_seconds"],"logical_advances":scheduler["logical_advances"],
         "mean_effective_batch":scheduler["mean_effective_batch"],
         "completed_candidates":sum(int(r["completed_candidate_count"]) for r in final),
         "useful_model_forwards":useful,"replay_model_forwards":replay,
         "replay_useful_ratio":replay/useful if useful else None,
         "r1024_cells":sum(bool(r["checkpoint_reached"]) for r in final),
         "terminal_carry_cells":sum(bool(r["carried_forward_terminal"]) for r in final),
         "wall_censored_cells":sum(bool(r["wall_time_censored"]) for r in final),
         "fallback_used":bool(raw.get("attempt_index",0)),"duplicate_work_item_ids":duplicate,
         "retention_changed":retention_changed}
    return row,[{"output_id":row["output_id"],"category":category,"profile":profile,**x} for x in work],[{"output_id":row["output_id"],"category":category,"profile":profile,**x} for x in checkpoints]


def materialize_runtime(run: Path, outputs: list[dict[str,Any]]) -> dict[str,Any]:
    runtime=[]; work=[]; checkpoints=[]
    for sel in outputs:
        raw=_read(paths(run,str(sel["output_id"]))[0])
        r,w,c=output_metrics(raw,str(sel["category"]),str(sel["profile"]))
        runtime.append(r);work.extend(w);checkpoints.extend(c)
    _atomic_csv(run/"OUTPUT_RUNTIME.csv",runtime,list(runtime[0]))
    _atomic_csv(run/"WORK_ITEM_TELEMETRY.csv",work,list(work[0]) if work else ["output_id"])
    _atomic_csv(run/"CHECKPOINT_CURVES.csv",checkpoints,list(checkpoints[0]))
    return {"outputs":len(runtime),"total_gpu_wall_seconds":sum(float(r["wall_seconds"]) for r in runtime),
            "oom_fallback_count":sum(bool(r["fallback_used"]) for r in runtime),
            "r1024_cells":sum(int(r["r1024_cells"]) for r in runtime),
            "terminal_carry_cells":sum(int(r["terminal_carry_cells"]) for r in runtime),
            "wall_censored_cells":sum(int(r["wall_censored_cells"]) for r in runtime),
            "completed_candidates":sum(int(r["completed_candidates"]) for r in runtime),
            "duplicate_work_item_ids":sum(int(r["duplicate_work_item_ids"]) for r in runtime),
            "retention_changed":any(bool(r["retention_changed"]) for r in runtime),
            "replay_model_forwards":sum(int(r["replay_model_forwards"]) for r in runtime),
            "useful_model_forwards":sum(int(r["useful_model_forwards"]) for r in runtime)}


def run_outputs(args: argparse.Namespace, run: Path, outputs: list[dict[str,Any]], label: str) -> dict[str,Any]:
    run.mkdir(parents=True,exist_ok=True)
    contract={"experiment":EXPERIMENT,"label":label,"target_blind":True,"gold_loaded":False,"policy":POLICY,
              "checkpoints":[256,512,768,1024],"outputs":outputs}
    cp=run/"RUN_COHORT.json"
    if cp.exists() and _read(cp)!=contract:
        raise RuntimeError("P4_RUN_COHORT_DRIFT")
    if not cp.exists():
        _atomic_json(cp,contract)
    _atomic_json(run/"AUG8_IDS.json",{"subset":"CANONICAL_GEOMETRY_AUG8","candidate_ids":list(AUG8)})
    args.search_order_policy=POLICY; args.checkpoints=CHECKPOINTS
    attempts=[]
    for sel in outputs:
        oid=str(sel["output_id"])
        if output_is_frozen(run,oid):
            continue
        done=False
        for attempt,cfg in enumerate(_policy(_read(args.coarse_policy),str(sel["profile"]))):
            code,receipt=_worker(args,run,sel,attempt,cfg,depth=24,experiment=f"{EXPERIMENT}_{label}")
            receipt["fallback_used"]=attempt>0;attempts.append(receipt)
            _atomic_csv(run/"OOM_FALLBACK_RECEIPTS.csv",attempts,list(attempts[0]))
            if code==0:
                freeze_output(run,oid);done=True;break
            if code!=2:
                raise RuntimeError(f"P4_WORKER_NON_OOM:{oid}:{receipt['log']}")
        if not done:
            raise RuntimeError(f"P4_OOM_FALLBACK_EXHAUSTED:{oid}")
    if not (run/"OOM_FALLBACK_RECEIPTS.csv").exists():
        _atomic_csv(run/"OOM_FALLBACK_RECEIPTS.csv",[],["output_id"])
    runtime=materialize_runtime(run,outputs)
    generation=ledger(run,"GENERATION_HASHES.json",include_raw=True)
    verification=_verify(run,run/"GENERATION_HASHES.json")
    _atomic_json(run/"GENERATION_HASH_VERIFICATION.json",verification)
    if verification.get("status")!="PASS":
        raise RuntimeError("P4_GENERATION_HASH_FAIL")
    _atomic_json(run/"GENERATION_FREEZE.json",{"experiment":EXPERIMENT,"label":label,"status":"FROZEN","target_blind":True,
                                                "gold_loaded":False,"raw_count":len(outputs),"ledger_sha256":sha_value(generation)})
    return runtime


def validate_smoke(run: Path, outputs: list[dict[str,Any]], runtime: dict[str,Any]) -> dict[str,Any]:
    expected=len(outputs)*len(AUG8)
    passed=(runtime["oom_fallback_count"]==0 and runtime["wall_censored_cells"]==0
            and runtime["r1024_cells"]+runtime["terminal_carry_cells"]==expected
            and runtime["duplicate_work_item_ids"]==0 and runtime["retention_changed"] is False
            and runtime["completed_candidates"]>0
            and _read(run/"GENERATION_HASH_VERIFICATION.json").get("status")=="PASS")
    result={"status":"PASS" if passed else "FAIL","experiment":EXPERIMENT,"target_blind":True,"gold_loaded":False,
            "outputs":[x["output_id"] for x in outputs],"cells":expected,**runtime}
    _atomic_json(run.parent/"GATE1_SMOKE_RESULT.json",result)
    if not passed:
        raise RuntimeError("P4_GATE1_SMOKE_FAIL")
    return result


def gate1_evidence(root: Path | None) -> dict[str,Any]:
    if root is None:
        raise RuntimeError("P4_GATE1_EVIDENCE_REQUIRED")
    result_path=root/"GATE1_SMOKE_RESULT.json"
    verify_path=root/"GATE1_SMOKE"/"GENERATION_HASH_VERIFICATION.json"
    freeze_path=root/"GATE1_SMOKE"/"GENERATION_FREEZE.json"
    if any(not p.is_file() for p in (result_path,verify_path,freeze_path)):
        raise RuntimeError("P4_GATE1_EVIDENCE_INCOMPLETE")
    result=_read(result_path); verification=_read(verify_path); freeze=_read(freeze_path)
    if result.get("status")!="PASS" or verification.get("status")!="PASS" or freeze.get("status")!="FROZEN":
        raise RuntimeError("P4_GATE1_NOT_PASSED")
    if result.get("gold_loaded") is not False or result.get("retention_changed") is not False:
        raise RuntimeError("P4_GATE1_CONTRACT_INVALID")
    return {
        "path":str(root),
        "result_sha256":sha_file(result_path),
        "generation_hash_verification_sha256":sha_file(verify_path),
        "generation_freeze_sha256":sha_file(freeze_path),
        "status":"PASS",
    }


def finalize_generation(run: Path, cohort: dict[str,Any], preflight_receipt: dict[str,Any], gate1: dict[str,Any]) -> None:
    generation=ledger(run,"GENERATION_HASHES.json",include_raw=True)
    verification=_verify(run,run/"GENERATION_HASHES.json")
    _atomic_json(run/"GENERATION_HASH_VERIFICATION.json",verification)
    if verification.get("status")!="PASS":
        raise RuntimeError("P4_FINAL_GENERATION_HASH_FAIL")
    _atomic_json(run/"PROVENANCE.json",{"experiment":EXPERIMENT,"source_commit":head(),"target_blind":True,"gold_loaded":False,
                                         "cohort_sha256":COHORT_SHA256,"runtime":preflight_receipt["runtime"],"policy":POLICY,
                                         "gate1_evidence":gate1})
    _atomic_json(run/"PRE_GOLD_FREEZE.json",{"experiment":EXPERIMENT,"status":"FROZEN","outputs":"12/12",
                                              "target_blind":True,"gold_loaded":False,"cohort_sha256":COHORT_SHA256,
                                              "scientific_policy":POLICY,"scientific_config_unchanged_except_policy":True,
                                              "generation_hash_status":"PASS","gate1_status":"PASS",
                                              "gate1_evidence_sha256":sha_value(gate1),
                                              "postfreeze_gold_scoring_required":True})
    compact=ledger(run,"HASHES.json",include_raw=False)
    checked=_verify(run,run/"HASHES.json")
    _atomic_json(run/"HASH_VERIFICATION.json",checked)
    if checked.get("status")!="PASS":
        raise RuntimeError("P4_COMPACT_HASH_FAIL")


def failure_receipt(args: argparse.Namespace, error: BaseException) -> None:
    if not args.output.exists():
        return
    _atomic_json(args.output/"FAILURE_RECEIPT.json",{"experiment":EXPERIMENT,"classification":"IMPLEMENTATION_OR_RUNTIME",
        "exception":str(error),"gpu_generation_occurred":(args.output/"RAW_OUTPUTS").exists() or (args.output/"GATE1_SMOKE"/"RAW_OUTPUTS").exists(),
        "gold_loaded":False,"current_commit":head()})


def parse_args() -> argparse.Namespace:
    p=argparse.ArgumentParser()
    p.add_argument("--mode",choices=("gate0","smoke","generate"),required=True)
    p.add_argument("--output",type=Path,required=True);p.add_argument("--cohort",type=Path,required=True);p.add_argument("--parent-cohort",type=Path,required=True)
    p.add_argument("--model-path",type=Path,required=True);p.add_argument("--challenge",type=Path,required=True)
    p.add_argument("--native-config-dir",type=Path,required=True);p.add_argument("--candidate-pool",type=Path,required=True)
    p.add_argument("--adapter-root",type=Path,required=True);p.add_argument("--coarse-policy",type=Path,required=True)
    p.add_argument("--worker-python",type=Path,required=True);p.add_argument("--source-archive",type=Path,required=True)
    p.add_argument("--gate1-evidence",type=Path)
    p.add_argument("--device",default="cuda:0")
    return p.parse_args()


def main() -> None:
    args=parse_args()
    try:
        if args.mode=="gate0":
            args.output.mkdir(parents=True,exist_ok=True);run_gate0(args,args.output);return
        cohort=load_cohort(args.cohort)
        if args.mode=="generate" and args.output.exists() and any(args.output.iterdir()):
            raise RuntimeError(f"P4_OUTPUT_DIRECTORY_MUST_BE_FRESH:{args.output}")
        args.output.mkdir(parents=True,exist_ok=True)
        pre=preflight(args,cohort,args.output)
        if args.mode=="smoke":
            smoke=smoke_cohort(_read(args.parent_cohort))
            _atomic_json(args.output/"GATE1_SMOKE_COHORT.json",{"experiment":EXPERIMENT,"target_blind":True,"gold_loaded":False,
                "selection":"reuse exact P3 deterministic S/M/L smoke selector","outputs":smoke})
            runtime=run_outputs(args,args.output/"GATE1_SMOKE",smoke,"GATE1_SMOKE")
            validate_smoke(args.output/"GATE1_SMOKE",smoke,runtime);return
        gate1=gate1_evidence(args.gate1_evidence)
        runtime=run_outputs(args,args.output,list(cohort["outputs"]),"GATE2_MICRO12")
        _atomic_json(args.output/"RUNTIME_SUMMARY.json",runtime)
        finalize_generation(args.output,cohort,pre,gate1)
    except BaseException as error:
        failure_receipt(args,error);raise


if __name__=="__main__":
    main()
