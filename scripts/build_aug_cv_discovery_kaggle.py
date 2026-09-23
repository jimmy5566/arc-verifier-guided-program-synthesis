"""Build (never submit) the budget-capped cross-validated augmentation notebook."""
from __future__ import annotations
import argparse, hashlib, json, shutil, subprocess, tarfile
from pathlib import Path
from typing import Any

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/"artifacts"/"aug_cv_discovery_r2"
TTT24=ROOT/"artifacts"/"eval60_ttt48_confirmation"/"ttt24_candidates_frozen.json"
TTT48=ROOT/"artifacts"/"eval60_ttt48_confirmation"/"candidates_frozen.json"

def _read(p:Path)->dict[str,Any]: return json.loads(p.read_text(encoding="utf-8"))
def _write(p:Path,x:Any)->None: p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(x,indent=2,sort_keys=True)+"\n",encoding="utf-8")
def _sha(p:Path)->str: return hashlib.sha256(p.read_bytes()).hexdigest()
def _archive(dst:Path,commit:str)->None:
    arc=dst.parent/"ARC2.tar"
    with arc.open("wb") as out: subprocess.run(["git","archive","--format=tar",commit],cwd=ROOT,stdout=out,check=True)
    with tarfile.open(arc) as tf: tf.extractall(dst,filter="data")
    arc.unlink()

def _notebook(slug:str)->str:
 return "\n".join([
  "import json,os,shutil,subprocess,sys,tarfile,time", "from pathlib import Path", "",
  f'dataset=Path("/kaggle/input/datasets/jimmy5566/{slug}")', 'source=dataset/"ARC2"; source_work=Path("/kaggle/working/aug_cv_source"); work=Path("/kaggle/working/artifacts/aug_cv_discovery")',
  'if not source.is_dir():\n    archive=dataset/"ARC2.tar"\n    if not archive.is_file(): raise RuntimeError("source package is neither unpacked ARC2 nor ARC2.tar")\n    with tarfile.open(archive) as handle: handle.extractall(source_work,filter="data")\n    source=source_work',
  'required={"manifest":dataset/"cv_augmentation_manifest.json","reference":dataset/"reference_ttt24_config.json","ttt24":dataset/"ttt24_candidates_frozen.json","ttt48":dataset/"ttt48_candidates_frozen.json","runner":source/"scripts/run_aug_cv_gpu_phase.py","scorer":source/"scripts/score_aug_cv.py","native":source/"configs/nvarc_native_846d0198"}',
  'missing=[name for name,path in required.items() if not path.exists()]\nif missing: raise RuntimeError(f"attached package incomplete: {missing}")',
  'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()\nif len(gpus)!=4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly 4 L4 GPUs: {gpus}")',
  'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED","").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be disabled")',
  'ptxas=Path("/usr/local/cuda-12.5/bin/ptxas")\nif not ptxas.is_file() or subprocess.run([ptxas,"--version"],capture_output=True).returncode: raise RuntimeError("verified ptxas unavailable")',
  'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"); solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"); model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
  'if not challenge.is_file() or not solutions.is_file() or not model.is_dir(): raise RuntimeError("competition/model attachment missing")',
  'if work.exists(): raise FileExistsError("refusing to overwrite frozen augmentation-CV output")\nwork.mkdir(parents=True)',
  'for name in ("cv_augmentation_manifest.json","reference_ttt24_config.json","ttt24_candidates_frozen.json","ttt48_candidates_frozen.json"): shutil.copy2(dataset/name,work/name)',
  'manifest=json.loads((work/"cv_augmentation_manifest.json").read_text())\nif manifest.get("fold_count")!=5 or manifest.get("initial_gpu_spec_count")!=8: raise RuntimeError("invalid frozen CV contract")',
  'env={**os.environ,"TRITON_PTXAS_PATH":str(ptxas),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}; began=time.monotonic(); deadline=began+5400; completed=[]',
  'print(json.dumps({"event":"AUG_CV_START","legal_specs":manifest["legal_spec_count"],"initial_specs":manifest["initial_gpu_spec_count"],"folds":5,"budget_seconds":5400,"gpus":gpus,"solutions_opened":False},sort_keys=True),flush=True)',
  'for fold in range(5):\n    if time.monotonic()>=deadline: print(json.dumps({"event":"AUG_CV_BUDGET_EXHAUSTED","before_fold":fold},sort_keys=True),flush=True); break\n    root=work/f"fold_{fold}"; discovery=root/"discovery_candidates.json"; checkpoint=root/"discovery_checkpoints"\n    run=[sys.executable,str(required["runner"]),"--manifest",str(work/"cv_augmentation_manifest.json"),"--reference-config",str(work/"reference_ttt24_config.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(required["native"]),"--fold",str(fold),"--phase","discovery","--output",str(discovery),"--checkpoint-dir",str(checkpoint),"--resume"]\n    subprocess.run(run,env=env,check=True,timeout=max(60,int(deadline-time.monotonic())))\n    score=[sys.executable,str(required["scorer"]),"discovery","--manifest",str(work/"cv_augmentation_manifest.json"),"--candidates",str(discovery),"--ttt24",str(work/"ttt24_candidates_frozen.json"),"--ttt48",str(work/"ttt48_candidates_frozen.json"),"--solutions",str(solutions),"--output-dir",str(root/"discovery_score")]\n    subprocess.run(score,env={**env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=600)\n    chosen=json.loads((root/"discovery_score"/"selected_specs_frozen.json").read_text())\n    if not chosen["selected_specs"]: print(json.dumps({"event":"AUG_CV_FOLD_EARLY_STOP","fold":fold,"reason":"no_discovery_frontier_hit"},sort_keys=True),flush=True); continue\n    if time.monotonic()>=deadline: print(json.dumps({"event":"AUG_CV_BUDGET_EXHAUSTED_AFTER_DISCOVERY","fold":fold},sort_keys=True),flush=True); break\n    held=root/"heldout_candidates.json"; run=[sys.executable,str(required["runner"]),"--manifest",str(work/"cv_augmentation_manifest.json"),"--reference-config",str(work/"reference_ttt24_config.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(required["native"]),"--fold",str(fold),"--phase","heldout","--selected-specs",str(root/"discovery_score"/"selected_specs_frozen.json"),"--output",str(held),"--checkpoint-dir",str(root/"heldout_checkpoints"),"--resume"]\n    subprocess.run(run,env=env,check=True,timeout=max(60,int(deadline-time.monotonic())))\n    score=[sys.executable,str(required["scorer"]),"heldout","--candidates",str(held),"--ttt24",str(work/"ttt24_candidates_frozen.json"),"--ttt48",str(work/"ttt48_candidates_frozen.json"),"--solutions",str(solutions),"--output-dir",str(root/"heldout_score")]\n    subprocess.run(score,env={**env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=600); completed.append(fold)',
  'subprocess.run([sys.executable,str(required["scorer"]),"summary","--root",str(work)],env={**env,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=600)',
  'print(json.dumps({"event":"AUG_CV_COMPLETE","completed_heldout_folds":completed,"elapsed_seconds":time.monotonic()-began,"budget_seconds":5400,"artifacts":str(work)},sort_keys=True),flush=True)',
 ])

def main()->None:
 p=argparse.ArgumentParser(); p.add_argument("--output",type=Path,required=True); p.add_argument("--owner",default="jimmy5566"); p.add_argument("--dataset-slug",default="arc2-aug-cv-discovery-source"); p.add_argument("--kernel-slug",default="arc2-aug-cv-discovery"); p.add_argument("--source-commit",default=None); a=p.parse_args()
 if a.output.exists(): raise FileExistsError("refusing to overwrite Kaggle staging")
 manifest=_read(FREEZE/"cv_augmentation_manifest.json"); config=_read(TTT24)["reference_config"]
 if manifest["legal_spec_count"]!=88 or manifest["initial_gpu_spec_count"]!=8 or config["ttt_steps"]!=24: raise ValueError("frozen source contract mismatch")
 commit=a.source_commit or subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(); dataset=a.output/"dataset"; source=dataset/"ARC2"; source.mkdir(parents=True); _archive(source,commit)
 for name,path in {"cv_augmentation_manifest.json":FREEZE/"cv_augmentation_manifest.json","reference_ttt24_config.json":TTT24,"ttt24_candidates_frozen.json":TTT24,"ttt48_candidates_frozen.json":TTT48}.items():
  if name=="reference_ttt24_config.json": _write(dataset/name,config)
  else: (dataset/name).write_bytes(path.read_bytes())
 if list(dataset.rglob("*solutions*.json")): raise RuntimeError("target-blind source package contains solution grids")
 _write(dataset/"dataset-metadata.json",{"id":f"{a.owner}/{a.dataset_slug}","title":"ARC2 cross-validated augmentation discovery source","subtitle":"Target-blind TTT24 challenger source","description":"Private development-only package without evaluation solutions.","licenses":[{"name":"other"}]})
 kernel=a.output/"kernel"; kernel.mkdir(); nb=f"{a.kernel_slug}.ipynb"; _write(kernel/nb,{"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in _notebook(a.dataset_slug).splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
 _write(kernel/"kernel-metadata.json",{"id":f"{a.owner}/{a.kernel_slug}","title":"ARC2 Cross-Validated Augmentation Discovery","code_file":nb,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"dataset_sources":[f"{a.owner}/{a.dataset_slug}"],"kernel_sources":["sorokin/pip-install-unsloth-flash-patch"],"competition_sources":["arc-prize-2026-arc-agi-2"],"model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],"docker_image":"gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868","machine_shape":"NvidiaL4"})
 _write(a.output/"PACKAGE_MANIFEST.json",{"status":"PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED","source_commit":commit,"dataset_slug":f"{a.owner}/{a.dataset_slug}","kernel_slug":f"{a.owner}/{a.kernel_slug}","cv_manifest_sha256":_sha(FREEZE/"cv_augmentation_manifest.json"),"files":{str(x.relative_to(a.output)).replace("\\","/"):_sha(x) for x in a.output.rglob("*") if x.is_file()}})
 print(json.dumps(_read(a.output/"PACKAGE_MANIFEST.json"),sort_keys=True),flush=True)
if __name__=="__main__": main()
