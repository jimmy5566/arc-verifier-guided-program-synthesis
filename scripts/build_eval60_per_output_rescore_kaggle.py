"""Stage a target-blind 4xL4 replay for per-output frozen-pool rescoring."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, sys, tarfile
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
MANIFEST=ROOT/"artifacts/eval60_ttt48_confirmation/eval60_manifest.json"
TTT24=ROOT/"artifacts/eval60_reference_ttt_4gpu_kernel_v5_output/artifacts/eval60_reference_ttt_4gpu"
TTT48=ROOT/"artifacts/eval60_ttt48_confirmation"

def _read(path:Path)->dict[str,Any]: return json.loads(path.read_text(encoding="utf-8"))
def _write(path:Path,value:Any)->None: path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _commit()->str: return subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip()
def _archive(destination:Path,commit:str)->None:
    tar=destination.parent/"source.tar"
    with tar.open("wb") as h: subprocess.run(["git","archive","--format=tar",commit],cwd=ROOT,stdout=h,check=True)
    with tarfile.open(tar) as h: h.extractall(destination,filter="data")
    tar.unlink()

def _inputs()->dict[str,Path]:
    values={"eval60_manifest.json":MANIFEST,"ttt24_candidates_frozen.json":TTT24/"candidates_frozen.json","ttt24_predictions_frozen.json":TTT24/"postprocessed_cpu_v3_final/predictions_frozen.json","ttt48_candidates_frozen.json":TTT48/"candidates_frozen.json","ttt48_predictions_frozen.json":TTT48/"predictions_frozen.json"}
    if any(not path.is_file() for path in values.values()): raise ValueError("missing frozen per-output rescore input")
    m,a,b,c,d=(_read(values[x]) for x in values)
    if len(m["task_ids"])!=60 or a["reference_config"]["ttt_steps"]!=24 or c["reference_config"]["ttt_steps"]!=48 or len(b["records"])!=60 or len(d["records"])!=60: raise ValueError("invalid frozen source contract")
    return values

def _notebook(slug:str)->str:
    return "\n".join([
        "import importlib.metadata,json,os,shutil,subprocess,sys", "from pathlib import Path", "",
        'inputs=Path("/kaggle/input"); runner=next(inputs.rglob("run_eval60_per_output_rescore.py"),None); source=inputs if runner else Path("/kaggle/working/per_output_source")', 'if runner is None: shutil.unpack_archive(next(inputs.rglob("ARC2.zip")),source)', 'root=(runner if runner else next(source.rglob("run_eval60_per_output_rescore.py"))).parents[1]; frozen=next((inputs if runner else source).rglob("eval60_manifest.json")).parent',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"); solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"); model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"); out=Path("/kaggle/working/artifacts/eval60_per_output_rescore")',
        'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines(); manifest=json.loads((frozen/"eval60_manifest.json").read_text()); lock=manifest["environment_lock"]', 'if len(gpus)!=4 or any("NVIDIA L4" not in x for x in gpus): raise RuntimeError(f"requires 4 L4: {gpus}")', 'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED","").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be off")', 'if not sys.version.startswith(lock["python_prefix"]) or importlib.metadata.version("unsloth")!=lock["unsloth"] or importlib.metadata.version("transformers")!=lock["transformers"]: raise RuntimeError("reference environment mismatch")', 'if subprocess.run([lock["ptxas_path"],"--version"],capture_output=True).returncode: raise RuntimeError("ptxas unavailable")', 'out.mkdir(parents=True,exist_ok=True)',
        'for name in ("eval60_manifest.json","ttt24_candidates_frozen.json","ttt24_predictions_frozen.json","ttt48_candidates_frozen.json","ttt48_predictions_frozen.json"): shutil.copy2(frozen/name,out/name)',
        f'print(json.dumps({{"event":"EVAL60_PER_OUTPUT_RESCORE_TARGET_BLIND_START","source_dataset":"{slug}","gpus":gpus,"solutions_opened":False}},sort_keys=True),flush=True)',
        'run=[sys.executable,str(root/"scripts/run_eval60_per_output_rescore.py"),"--manifest",str(out/"eval60_manifest.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(root/"configs/nvarc_native_846d0198"),"--candidates24",str(out/"ttt24_candidates_frozen.json"),"--candidates48",str(out/"ttt48_candidates_frozen.json"),"--output",str(out/"per_output_evidence_frozen.json"),"--checkpoint-dir",str(out/"checkpoints"),"--resume"]',
        'if subprocess.run(run,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0,1,2,3","TRITON_PTXAS_PATH":"/usr/local/cuda-12.5/bin/ptxas","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}).returncode: raise RuntimeError("per-output evidence run failed")', 'e=json.loads((out/"per_output_evidence_frozen.json").read_text())', 'if e.get("status")!="EVAL60_PER_OUTPUT_RESCORE_EVIDENCE_FROZEN_BEFORE_EXACT_SCORING" or len(e.get("records",{}))!=28 or e.get("solutions_opened") is not False: raise RuntimeError("invalid target-blind evidence freeze")',
        'score=[sys.executable,str(root/"scripts/score_eval60_per_output_rescore.py"),"--manifest",str(out/"eval60_manifest.json"),"--evidence",str(out/"per_output_evidence_frozen.json"),"--predictions24",str(out/"ttt24_predictions_frozen.json"),"--predictions48",str(out/"ttt48_predictions_frozen.json"),"--challenge-path",str(challenge),"--solutions-path",str(solutions),"--output-dir",str(out)]', 'if subprocess.run(score,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("per-output post-freeze scoring failed")', 'required=[out/x for x in ("per_output_evidence_frozen.json","per_output_selection_frozen.json","per_output_predictions_frozen.json","per_output_comparison.csv","EVAL60_PER_OUTPUT_RESCORE_REPORT.json","EVAL60_PER_OUTPUT_RESCORE_REPORT.md")]', 'if any(not x.is_file() for x in required): raise RuntimeError("missing per-output artifacts")', 'print(json.dumps({"event":"EVAL60_PER_OUTPUT_RESCORE_COMPLETE","artifacts":[str(x) for x in required]},sort_keys=True),flush=True)',
    ])

def main()->None:
    p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--owner",default="jimmy5566"); p.add_argument("--dataset-slug",default="arc2-eval60-per-output-rescore-source"); p.add_argument("--kernel-slug",default="arc2-eval60-per-output-rescore"); args=p.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite staging")
    inputs=_inputs(); commit=_commit(); base=args.output/"dataset"/"ARC2"; base.mkdir(parents=True); _archive(base,commit); frozen=base/"eval60_per_output_rescore"; [frozen.mkdir(parents=True,exist_ok=True)]
    for name,path in inputs.items(): (frozen/name).write_bytes(path.read_bytes())
    if list(base.rglob("*solutions*.json")): raise RuntimeError("target-blind source contains solutions")
    _write(args.output/"dataset/dataset-metadata.json",{"title":"ARC2 Eval60 per-output rescore source","subtitle":"Frozen TTT24/TTT48 candidate re-score only","description":"Private target-blind source. No solutions.","id":f"{args.owner}/{args.dataset_slug}","licenses":[{"name":"other"}]})
    kernel=args.output/"kernel"; kernel.mkdir(); nb=f"{args.kernel_slug}.ipynb"; _write(kernel/nb,{"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[x+"\n" for x in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
    _write(kernel/"kernel-metadata.json",{"id":f"{args.owner}/{args.kernel_slug}","title":"ARC2 Eval60 per-output frozen-pool rescore","code_file":nb,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"dataset_sources":[f"{args.owner}/{args.dataset_slug}"],"kernel_sources":["sorokin/pip-install-unsloth-flash-patch"],"competition_sources":["arc-prize-2026-arc-agi-2"],"model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],"docker_image":"gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868","machine_shape":"NvidiaL4"})
    print(json.dumps({"event":"PER_OUTPUT_RESCORE_STAGING_READY","commit":commit,"inputs":list(inputs)},sort_keys=True))
if __name__=="__main__": main()
