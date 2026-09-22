"""Build the always-production, fail-closed reference-TTT Kaggle notebook."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_eval3_reference_ttt_kaggle import _frozen_inputs as _reference_inputs


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_ids(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "ARC2.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    # Preserve the archive as the Kaggle source-dataset payload.  It contains
    # only committed source and cannot accidentally pick up local artifacts.


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    challenges = json.loads((ROOT / "data" / "raw" / "arc-agi_test_challenges.json").read_text(encoding="utf-8"))
    sample = json.loads((ROOT / "data" / "raw" / "sample_submission.json").read_text(encoding="utf-8"))
    task_ids = sorted(challenges)
    if len(task_ids) != 240 or set(task_ids) != set(sample):
        raise ValueError("local competition test challenge/sample contract is not 240 matching tasks")
    _manifest, _baseline, config = _reference_inputs()
    manifest = {
        "experiment_id": "ARC2_REFERENCE_TTT_PRODUCTION",
        "status": "REFERENCE_TTT_PRODUCTION_TASKS_FROZEN_BEFORE_INFERENCE",
        "task_ids": task_ids,
        "task_ids_hash": _hash_ids(task_ids),
        "source_challenge_sha256": _sha256(ROOT / "data" / "raw" / "arc-agi_test_challenges.json"),
        "source_sample_submission_sha256": _sha256(ROOT / "data" / "raw" / "sample_submission.json"),
        "protocol": "All visible ARC Prize 2026 test challenges, frozen before target-blind production inference.",
        "solutions_opened": False,
    }
    return manifest, config


def _notebook(dataset_slug: str, source_commit: str) -> str:
    return "\n".join([
        "import hashlib, json, os, shutil, subprocess, sys, tarfile, time", "from pathlib import Path", "",
        'started_unix=time.time(); inputs=Path("/kaggle/input"); work=Path("/kaggle/working")',
        'archive=next(inputs.rglob("ARC2.tar"), None); direct=next(inputs.rglob("run_reference_ttt_production_4gpu.py"), None)',
        'source=direct.parents[1] if direct is not None else work / "ARC2"',
        'if direct is None:', '    if archive is None: raise RuntimeError("reference-TTT source artifact missing")', '    source.mkdir(parents=True, exist_ok=True)', '    with tarfile.open(archive) as handle: handle.extractall(source, filter="data")',
        'root=source if (source / "scripts" / "run_reference_ttt_production_4gpu.py").is_file() else next(source.rglob("run_reference_ttt_production_4gpu.py")).parents[1]',
        'frozen=next(inputs.rglob("reference_ttt_production_manifest.json")).parent',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")',
        'sample=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/sample_submission.json")',
        'model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'ptxas=Path("/usr/local/cuda-12.5/bin/ptxas"); out=work / "artifacts" / "reference_ttt_production"',
        'gpus=subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if len(gpus)!=4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly 4 NVIDIA L4 GPUs: {gpus}")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be off")',
        'if not ptxas.is_file(): raise RuntimeError("verified PTXAS path unavailable")',
        'print(json.dumps({"event":"REFERENCE_TTT_ALWAYS_PRODUCTION_MODE","source_commit":"' + source_commit + '","gpus":gpus,"ptxas":str(ptxas),"internet":False,"solutions_opened":False},sort_keys=True),flush=True)',
        'out.mkdir(parents=True, exist_ok=True); shutil.copy2(frozen / "reference_ttt_production_manifest.json", out / "manifest.json"); shutil.copy2(frozen / "reference_ttt_config_frozen.json", out / "config.json")',
        'candidates=out / "candidates_frozen.json"; checkpoints=out / "checkpoints"',
        'generation=[sys.executable,str(root/"scripts"/"run_reference_ttt_production_4gpu.py"),"--manifest",str(out/"manifest.json"),"--reference-config",str(out/"config.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(root/"configs"/"nvarc_native_846d0198"),"--output",str(candidates),"--checkpoint-dir",str(checkpoints),"--resume"]',
        'if subprocess.run(generation,env={**os.environ,"TRITON_PTXAS_PATH":str(ptxas),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}).returncode: raise RuntimeError("REFERENCE_TTT_PRODUCTION_GENERATION_FAILED")',
        'selection=out / "b_support_selection_frozen.json"; provenance=out / "PRODUCTION_PROVENANCE.json"; submission=work / "submission.json"',
        'finalize=[sys.executable,str(root/"scripts"/"build_reference_ttt_strict_submission.py"),"--manifest",str(out/"manifest.json"),"--sample-submission",str(sample),"--candidates",str(candidates),"--selection-output",str(selection),"--provenance-output",str(provenance),"--output",str(submission)]',
        'if subprocess.run(finalize,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("STRICT_REFERENCE_TTT_COVERAGE_FAILED")',
        'if not submission.is_file(): raise RuntimeError("strict finalizer did not write submission")',
        'payload=json.loads(submission.read_text()); sample_payload=json.loads(sample.read_text())',
        'if set(payload)!=set(sample_payload) or any(len(payload[key])!=len(sample_payload[key]) for key in sample_payload): raise RuntimeError("submission mapping mismatch")',
        'runtime={"status":"VISIBLE_PRODUCTION_COMPLETE","source_commit":"' + source_commit + '","runtime_seconds":time.time()-started_unix,"expected_task_count":240,"submission_sha256":hashlib.sha256(submission.read_bytes()).hexdigest(),"submission_task_count":len(payload),"submission_test_output_count":sum(len(row) for row in payload.values()),"ptxas_path":str(ptxas),"gpu_models":gpus,"solutions_opened":False}',
        '(out / "PRODUCTION_RUNTIME_SUMMARY.json").write_text(json.dumps(runtime,indent=2,sort_keys=True)+"\\n")',
        '(out / "PRODUCTION_IDENTITY.json").write_text(json.dumps({"source_commit":"' + source_commit + '","model_source":"sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1","ptxas_path":str(ptxas),"ttt_config_sha256":hashlib.sha256((out/"config.json").read_bytes()).hexdigest(),"always_production":True,"solutions_opened":False},indent=2,sort_keys=True)+"\\n")',
        '(out / "REFERENCE_TTT_PRODUCTION_REPORT.md").write_text("# Reference-TTT production\\n\\n"+"\\n".join(f"- {key} = `{value}`" for key,value in runtime.items())+"\\n")',
        'print(json.dumps({"event":"REFERENCE_TTT_PRODUCTION_COMPLETE","validation":"PASS",**runtime},sort_keys=True),flush=True)',
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-reference-ttt-production-source")
    parser.add_argument("--kernel-slug", default="arc2-arc-prize-2026-reference-ttt-production")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite production staging")
    manifest, config = _frozen_inputs(); commit = _commit()
    source_root = args.output / "dataset" / "ARC2"; source_root.mkdir(parents=True)
    _archive(source_root, commit)
    # These two sidecars deliberately live at the dataset root.  Kaggle's
    # ``--dir-mode skip`` uploads the immutable source tar but skips nested
    # directories, so keeping them beside the archive is essential transport
    # provenance rather than a runtime fallback.
    _write(args.output / "dataset" / "reference_ttt_production_manifest.json", manifest)
    _write(args.output / "dataset" / "reference_ttt_config_frozen.json", config)
    if list(source_root.rglob("*solutions*.json")):
        raise RuntimeError("target-blind production source archive contains solution grids")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title":"ARC2 Reference TTT Production Source","subtitle":"Target-blind rank-256 reference-style TTT Aug8 source","description":"Private source attachment containing no ARC solution grids.","id":f"{args.owner}/{args.dataset_slug}","licenses":[{"name":"other"}]})
    kernel = args.output / "kernel"; kernel.mkdir(); notebook_name = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}", commit).splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.11"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
    _write(kernel / "kernel-metadata.json", {"id":f"{args.owner}/{args.kernel_slug}","title":"ARC2 Reference TTT Aug8 Production","code_file":notebook_name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":["gpu"],"dataset_sources":[f"{args.owner}/{args.dataset_slug}"],"kernel_sources":["sorokin/pip-install-unsloth-flash-patch"],"competition_sources":["arc-prize-2026-arc-agi-2"],"model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],"docker_image":"gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868","machine_shape":"NvidiaL4"})
    print(json.dumps({"event":"REFERENCE_TTT_PRODUCTION_STAGING_READY","source_commit":commit,"dataset":str(args.output/"dataset"),"kernel":str(kernel),"manifest":manifest,"config":config},sort_keys=True))


if __name__ == "__main__":
    main()
