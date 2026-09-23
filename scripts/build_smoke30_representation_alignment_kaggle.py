"""Build the isolated 4xL4 Smoke30 representation-alignment experiment."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / "artifacts" / "smoke30_representation_alignment"
TTT24 = ROOT / "artifacts" / "eval60_ttt48_confirmation" / "ttt24_candidates_frozen.json"
BASELINE_PREDICTIONS = ROOT / "artifacts" / "eval60_per_output_rescore_cpu_recovery" / "scored" / "per_output_predictions_frozen.json"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "ARC2.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as handle:
        handle.extractall(destination, filter="data")
    archive.unlink()


def _notebook(dataset_slug: str) -> str:
    return "\n".join([
        "import hashlib,json,os,shutil,subprocess,sys,time", "from pathlib import Path", "",
        f'dataset=Path("/kaggle/input/datasets/jimmy5566/{dataset_slug}")', 'source=dataset/"ARC2"; source_work=Path("/kaggle/working/smoke30_representation_source"); work=Path("/kaggle/working/artifacts/smoke30_representation_alignment")',
        'if not source.is_dir():\n    archive=dataset/"ARC2.zip"\n    if not archive.is_file(): raise RuntimeError("Smoke30 source is neither an unpacked ARC2 directory nor ARC2.zip")\n    shutil.unpack_archive(archive, source_work)\n    source=source_work',
        'required={"manifest":dataset/"smoke30_manifest.json","variants":dataset/"reference_variant_config.json","reference":dataset/"reference_ttt24_config.json","baseline_candidates":dataset/"ttt24_candidates_frozen.json","baseline_predictions":dataset/"ttt24_per_output_predictions_frozen.json","runner":source/"scripts/run_smoke30_representation_alignment.py","scorer":source/"scripts/score_smoke30_representation_alignment.py","native":source/"configs/nvarc_native_846d0198"}',
        'missing=[name for name,path in required.items() if not path.exists()]', 'if missing: raise RuntimeError(f"Smoke30 attached source incomplete: {missing}")',
        'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()', 'if len(gpus)!=4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly 4 L4 GPUs: {gpus}")', 'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED","").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be disabled")',
        'ptxas=Path("/usr/local/cuda-12.5/bin/ptxas")', 'if subprocess.run([ptxas,"--version"],capture_output=True).returncode: raise RuntimeError("verified ptxas is unavailable")',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")', 'solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")', 'model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")', 'if not challenge.is_file() or not solutions.is_file() or not model.is_dir(): raise RuntimeError("competition/model attachments missing")',
        'if work.exists(): raise FileExistsError("refusing to overwrite frozen Smoke30 output")', 'work.mkdir(parents=True)',
        'for name in ("smoke30_manifest.json","reference_variant_config.json","reference_ttt24_config.json","ttt24_candidates_frozen.json","ttt24_per_output_predictions_frozen.json"): shutil.copy2(dataset/name,work/name)',
        'manifest=json.loads((work/"smoke30_manifest.json").read_text())', 'variants=json.loads((work/"reference_variant_config.json").read_text())', 'if len(manifest.get("task_ids",[]))!=30 or variants.get("variant_count")!=8 or variants.get("config_hash")!=manifest.get("variant_config_hash"): raise RuntimeError("frozen cohort/variant contract invalid")',
        'print(json.dumps({"event":"SMOKE30_REPRESENTATION_TARGET_BLIND_START","task_count":30,"generation_count_per_task":8,"gpus":gpus,"solutions_opened":False},sort_keys=True),flush=True)',
        'run=[sys.executable,str(required["runner"]),"--manifest",str(work/"smoke30_manifest.json"),"--variant-config",str(work/"reference_variant_config.json"),"--reference-config",str(work/"reference_ttt24_config.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(required["native"]),"--output",str(work/"candidates_frozen.json"),"--checkpoint-dir",str(work/"checkpoints"),"--resume"]', 'environment={**os.environ,"TRITON_PTXAS_PATH":str(ptxas),"HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1","TOKENIZERS_PARALLELISM":"false"}', 'started=time.monotonic()', 'subprocess.run(run,env=environment,check=True,timeout=7200)',
        'candidates=json.loads((work/"candidates_frozen.json").read_text())', 'if candidates.get("status")!="SMOKE30_REFERENCE_LIKE_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING" or len(candidates.get("records",{}))!=30 or candidates.get("solutions_opened") is not False: raise RuntimeError("G1 target-blind candidate freeze failed")', 'print(json.dumps({"event":"SMOKE30_G1_CANDIDATES_FROZEN","task_count":30,"solutions_opened":False,"elapsed_seconds":time.monotonic()-started},sort_keys=True),flush=True)',
        'score=[sys.executable,str(required["scorer"]),"--manifest",str(work/"smoke30_manifest.json"),"--baseline-candidates",str(work/"ttt24_candidates_frozen.json"),"--baseline-predictions",str(work/"ttt24_per_output_predictions_frozen.json"),"--candidates",str(work/"candidates_frozen.json"),"--challenge-path",str(challenge),"--solutions-path",str(solutions),"--output-dir",str(work)]', 'subprocess.run(score,env={**environment,"CUDA_VISIBLE_DEVICES":""},check=True,timeout=600)',
        'required_outputs=[work/name for name in ("smoke30_manifest.json","reference_variant_config.json","candidates_frozen.json","per_output_selection_frozen.json","paired_results.csv","SMOKE30_REPRESENTATION_REPORT.json","SMOKE30_REPRESENTATION_REPORT.md")]', 'if any(not path.is_file() for path in required_outputs): raise RuntimeError("Smoke30 required artifact missing")', 'print(json.dumps({"event":"SMOKE30_REPRESENTATION_COMPLETE","artifacts":[str(path) for path in required_outputs]},sort_keys=True),flush=True)',
    ])


def build(*, output: Path, owner: str, dataset_slug: str, kernel_slug: str, source_commit: str) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError("refusing to overwrite Smoke30 Kaggle staging")
    manifest, variants, ttt24, predictions = _read(FREEZE / "smoke30_manifest.json"), _read(FREEZE / "reference_variant_config.json"), _read(TTT24), _read(BASELINE_PREDICTIONS)
    if len(manifest.get("task_ids", ())) != 30 or variants.get("variant_count") != 8 or manifest.get("variant_config_hash") != variants.get("config_hash") or ttt24["reference_config"].get("ttt_steps") != 24 or "TTT24" not in predictions.get("records", {}):
        raise ValueError("invalid frozen Smoke30 sources")
    dataset = output / "dataset"; source = dataset / "ARC2"; source.mkdir(parents=True); _archive(source, source_commit)
    for name, payload in {
        "smoke30_manifest.json": manifest, "reference_variant_config.json": variants, "reference_ttt24_config.json": ttt24["reference_config"],
    }.items(): _write(dataset / name, payload)
    (dataset / "ttt24_candidates_frozen.json").write_bytes(TTT24.read_bytes())
    (dataset / "ttt24_per_output_predictions_frozen.json").write_bytes(BASELINE_PREDICTIONS.read_bytes())
    if list(dataset.rglob("*solutions*.json")):
        raise RuntimeError("target-blind generation package contains solution data")
    _write(dataset / "dataset-metadata.json", {"id": f"{owner}/{dataset_slug}", "title": "ARC2 Smoke30 representation alignment source", "subtitle": "Frozen TTT24 representation-only diagnostic", "description": "Private target-blind source package. No ARC solution grids.", "licenses": [{"name": "other"}]})
    kernel = output / "kernel"; kernel.mkdir(); notebook_name = f"{kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook(dataset_slug).splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4})
    _write(kernel / "kernel-metadata.json", {"id": f"{owner}/{kernel_slug}", "title": "ARC2 Smoke30 Representation Alignment", "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "dataset_sources": [f"{owner}/{dataset_slug}"], "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868", "machine_shape": "NvidiaL4"})
    package = {"status": "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED", "source_commit": source_commit, "dataset_slug": f"{owner}/{dataset_slug}", "kernel_slug": f"{owner}/{kernel_slug}", "task_ids_hash": manifest["task_ids_hash"], "variant_config_hash": variants["config_hash"], "files": {str(path.relative_to(output)).replace("\\", "/"): _sha256(path) for path in sorted(output.rglob("*")) if path.is_file()}}
    _write(output / "PACKAGE_MANIFEST.json", package)
    return package


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-smoke30-representation-alignment-source")
    parser.add_argument("--kernel-slug", default="arc2-smoke30-representation-alignment")
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args()
    commit = args.source_commit or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    print(json.dumps(build(output=args.output, owner=args.owner, dataset_slug=args.dataset_slug, kernel_slug=args.kernel_slug, source_commit=commit), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
