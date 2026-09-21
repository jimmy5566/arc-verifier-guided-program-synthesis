"""Stage the isolated 4xL4 Evaluation60 reference-style TTT confirmation."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_eval3_reference_ttt_kaggle import _frozen_inputs as _eval3_frozen_inputs

EVAL60_MANIFEST = ROOT / "artifacts" / "eval60_aug8_kaggle_stage" / "dataset" / "ARC2" / "eval60_aug8_baseline" / "eval60_manifest.json"
EVAL60_BASELINE = ROOT / "artifacts" / "eval60_aug8_kaggle_v2_output" / "artifacts" / "eval60_aug8" / "A_candidates_frozen.json"


def _read(path: Path) -> dict[str, Any]: return json.loads(path.read_text(encoding="utf-8"))
def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
def _sha256(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def _hash_ids(ids: list[str]) -> str: return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()
def _commit() -> str: return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle: subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source: source.extractall(destination, filter="data")
    archive.unlink()


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_manifest, baseline = _read(EVAL60_MANIFEST), _read(EVAL60_BASELINE)
    ids = list(source_manifest["task_ids"])
    if len(ids) != 60 or source_manifest["task_ids_hash"] != _hash_ids(ids) or set(baseline.get("records", ())) != set(ids):
        raise ValueError("existing Evaluation60 baseline artifact is incomplete")
    _old_manifest, _old_baseline, config = _frozen_inputs_reference()
    manifest = {"experiment_id": "ARC2_REFERENCE_STYLE_TTT_EVAL60_4GPU_CONFIRMATION", "status": "EVAL60_REFERENCE_TTT_4GPU_COHORT_FROZEN", "selection_provenance": "Existing Evaluation60 SHA256 cohort previously scored; DEVELOPMENT ONLY, not fresh held-out evidence.", "task_ids": ids, "task_ids_hash": _hash_ids(ids), "source_challenge_sha256": source_manifest["source_challenge_sha256"], "baseline": {"any_of_k": 2, "top1": 2, "top2": 2, "pool_miss_count": 58}, "integrity": {"solutions_opened_during_ttt_or_generation": False, "no_dfs": True, "no_beam": True, "no_production_change": True}}
    return manifest, baseline, config


def _frozen_inputs_reference() -> tuple[dict[str, Any], dict[str, Any]]:
    _manifest, _baseline, config = _eval3_frozen_inputs()
    return _manifest, _baseline, copy.deepcopy(config)


def _notebook(dataset_slug: str) -> str:
    return "\n".join([
        "import json, os, shutil, subprocess, sys", "from pathlib import Path", "",
        'inputs=Path("/kaggle/input")', 'root=next(inputs.rglob("run_eval60_reference_ttt_4gpu.py")).parents[1]', 'frozen=next(inputs.rglob("eval60_reference_ttt_4gpu_manifest.json")).parent',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")', 'solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")', 'model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")', 'out=Path("/kaggle/working/artifacts/eval60_reference_ttt_4gpu")',
        'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()', 'if len(gpus)!=4 or any("NVIDIA L4" not in item for item in gpus): raise RuntimeError(f"requires exactly 4 L4: {gpus}")', 'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be off")', 'out.mkdir(parents=True,exist_ok=True)',
        'for name in ("eval60_reference_ttt_4gpu_manifest.json","ttt_config_frozen.json","eval60_aug8_baseline_candidates.json"): shutil.copy2(frozen/name,out/name)',
        f'print(json.dumps({{"event":"EVAL60_REFERENCE_TTT_4GPU_TARGET_BLIND_START","source_dataset":"{dataset_slug}","gpus":gpus,"solutions_opened":False}},sort_keys=True),flush=True)',
        'generate=[sys.executable,str(root/"scripts/run_eval60_reference_ttt_4gpu.py"),"--manifest",str(out/"eval60_reference_ttt_4gpu_manifest.json"),"--reference-config",str(out/"ttt_config_frozen.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(root/"configs/nvarc_native_846d0198"),"--output",str(out/"candidates_frozen.json"),"--checkpoint-dir",str(out/"checkpoints"),"--resume"]',
        'if subprocess.run(generate,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0,1,2,3","TRITON_PTXAS_PATH":"/usr/local/cuda-12.5/bin/ptxas","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}).returncode: raise RuntimeError("Eval60 4-GPU TTT candidate freeze failed")',
        'if not (out/"candidates_frozen.json").is_file(): raise RuntimeError("candidate freeze missing; refusing target access")', 'print(json.dumps({"event":"EVAL60_REFERENCE_TTT_4GPU_CANDIDATES_FROZEN","solutions_opened":False,"gpu_after_workers":subprocess.check_output(["nvidia-smi"],text=True)[-1500:]},sort_keys=True),flush=True)',
        'score=[sys.executable,str(root/"scripts/score_eval60_reference_ttt_4gpu.py"),"--manifest",str(out/"eval60_reference_ttt_4gpu_manifest.json"),"--baseline",str(out/"eval60_aug8_baseline_candidates.json"),"--candidates",str(out/"candidates_frozen.json"),"--challenge-path",str(challenge),"--solutions-path",str(solutions),"--output-dir",str(out)]',
        'if subprocess.run(score,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("Eval60 post-freeze scoring failed")', 'required=[out/name for name in ("candidates_frozen.json","b_support_selection_frozen.json","predictions_frozen.json","runtime_per_task.csv","worker_summary.json","EVAL60_TTT_4GPU_REPORT.json","EVAL60_TTT_4GPU_REPORT.md")]', 'if any(not item.is_file() for item in required): raise RuntimeError("missing required Eval60 TTT output")', 'print(json.dumps({"event":"EVAL60_REFERENCE_TTT_4GPU_COMPLETE","artifacts":[str(item) for item in required]},sort_keys=True),flush=True)',
    ])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", default="jimmy5566"); parser.add_argument("--dataset-slug", default="arc2-eval60-reference-ttt-4gpu-source"); parser.add_argument("--kernel-slug", default="arc2-eval60-reference-ttt-4gpu-confirmation"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite Eval60 TTT staging")
    manifest, baseline, config = _frozen_inputs(); commit = _commit()
    dataset_root = args.output / "dataset" / "ARC2"; dataset_root.mkdir(parents=True); _archive(dataset_root, commit)
    frozen = dataset_root / "eval60_reference_ttt_4gpu"; _write(frozen / "eval60_reference_ttt_4gpu_manifest.json", manifest); _write(frozen / "ttt_config_frozen.json", config); _write(frozen / "eval60_aug8_baseline_candidates.json", baseline)
    if list(dataset_root.rglob("*solutions*.json")): raise RuntimeError("target-blind source dataset contains solutions")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title":"ARC2 Eval60 Reference TTT 4GPU Source","subtitle":"Development-only target-blind reference-style rank-256 TTT confirmation","description":"Private experiment source; no ARC solution grids.","id":f"{args.owner}/{args.dataset_slug}","licenses":[{"name":"other"}]})
    kernel = args.output / "kernel"; kernel.mkdir(); notebook = f"{args.kernel_slug}.ipynb"; _write(kernel / notebook, {"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
    _write(kernel / "kernel-metadata.json", {"id":f"{args.owner}/{args.kernel_slug}","title":"ARC2 Eval60 Reference TTT 4GPU Confirmation","code_file":notebook,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":["gpu"],"dataset_sources":[f"{args.owner}/{args.dataset_slug}"],"kernel_sources":["sorokin/pip-install-unsloth-flash-patch"],"competition_sources":["arc-prize-2026-arc-agi-2"],"model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],"docker_image":"gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868","machine_shape":"NvidiaL4"})
    print(json.dumps({"event":"EVAL60_REFERENCE_TTT_4GPU_STAGING_READY","source_commit":commit,"manifest":manifest,"config":config,"baseline_sha256":_sha256(EVAL60_BASELINE)},sort_keys=True))


if __name__ == "__main__": main()
