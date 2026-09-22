"""Package the isolated, target-blind Eval60 TTT48 paired confirmation."""
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
from scripts.build_eval60_reference_ttt_4gpu_kaggle import EVAL60_MANIFEST, _frozen_inputs_reference

TTT24_ROOT = ROOT / "artifacts" / "eval60_reference_ttt_4gpu_kernel_v5_output" / "artifacts" / "eval60_reference_ttt_4gpu"
TTT24_CANDIDATES = TTT24_ROOT / "candidates_frozen.json"
TTT24_SCORED = TTT24_ROOT / "postprocessed_cpu_v3_final"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_manifest = _read(EVAL60_MANIFEST)
    old_manifest, _old_baseline, old_config = _frozen_inputs_reference()
    old_candidates = _read(TTT24_CANDIDATES)
    old_predictions = _read(TTT24_SCORED / "predictions_frozen.json")
    old_report = _read(TTT24_SCORED / "EVAL60_TTT_4GPU_REPORT.json")
    ids = list(source_manifest["task_ids"])
    if (len(ids) != 60 or source_manifest["task_ids_hash"] != _hash_ids(ids)
            or list(old_candidates.get("task_ids", ())) != ids
            or set(old_predictions.get("records", ())) != set(ids)
            or old_report.get("TTT_ANY_OF_K") != "12/60" or old_report.get("TTT_TOP1") != "12/60" or old_report.get("TTT_TOP2") != "12/60"):
        raise ValueError("incomplete or non-authoritative TTT24 paired baseline")
    config = copy.deepcopy(old_config)
    if config.get("ttt_steps") != 24:
        raise ValueError("TTT24 config no longer has 24 steps")
    config["ttt_steps"] = 48
    # This string is provenance only; it is updated so the frozen record does
    # not claim a 24-step run while all behavioral fields except steps match.
    if isinstance(config.get("style"), str):
        config["style"] = config["style"].replace("24_steps", "48_steps")
    manifest = {
        "experiment_id": "ARC2_EVAL60_TTT48_PAIRED_CONFIRMATION",
        "status": "EVAL60_TTT48_PAIRED_CONFIRMATION_COHORT_FROZEN",
        "selection_provenance": "Exact existing Evaluation60 SHA256 cohort used by TTT24; paired development confirmation only.",
        "task_ids": ids, "task_ids_hash": _hash_ids(ids),
        "source_challenge_sha256": source_manifest["source_challenge_sha256"],
        "ttt24_baseline": {"any_of_k": 12, "top1": 12, "top2": 12, "pool_miss_count": 48, "selection_miss_count": 0, "candidate_artifact_sha256": _sha256(TTT24_CANDIDATES), "prediction_artifact_sha256": _sha256(TTT24_SCORED / "predictions_frozen.json"), "report_sha256": _sha256(TTT24_SCORED / "EVAL60_TTT_4GPU_REPORT.json")},
        "paired_contract": {"only_scientific_change": "ttt_steps: 24 -> 48", "unchanged": ["model", "bf16", "rank", "alpha", "target_modules", "train_pair_augmentation", "masking_loss", "Aug8", "greedy", "parser", "inverse_augmentation", "dedup", "B-SUPPORT", "Top-2", "cohort"]},
        "environment_lock": {"python_prefix": "3.11.13", "unsloth": "2025.9.7", "transformers": "4.55.4", "ptxas_path": "/usr/local/cuda-12.5/bin/ptxas"},
        "integrity": {"solutions_opened_during_generation": False, "no_dfs": True, "no_beam": True, "no_production_change": True},
    }
    return manifest, config, old_candidates, old_predictions, old_report


def _notebook(dataset_slug: str) -> str:
    return "\n".join([
        "import importlib.metadata, json, os, shutil, subprocess, sys", "from pathlib import Path", "",
        'inputs=Path("/kaggle/input")', 'runner=next(inputs.rglob("run_eval60_ttt48_paired_confirmation.py"),None)', 'source=inputs if runner is not None else Path("/kaggle/working/eval60_ttt48_source")', 'if runner is None: shutil.unpack_archive(next(inputs.rglob("ARC2.zip")),source)', 'root=(runner if runner is not None else next(source.rglob("run_eval60_ttt48_paired_confirmation.py"))).parents[1]', 'frozen=next((inputs if runner is not None else source).rglob("eval60_manifest.json")).parent',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")', 'solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")', 'model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")', 'out=Path("/kaggle/working/artifacts/eval60_ttt48_confirmation")',
        'manifest=json.loads((frozen/"eval60_manifest.json").read_text())', 'lock=manifest["environment_lock"]', 'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()', 'if len(gpus)!=4 or any("NVIDIA L4" not in item for item in gpus): raise RuntimeError(f"requires exactly 4 L4: {gpus}")', 'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be off")', 'if not sys.version.startswith(lock["python_prefix"]): raise RuntimeError(f"python mismatch: {sys.version}")', 'if importlib.metadata.version("unsloth")!=lock["unsloth"] or importlib.metadata.version("transformers")!=lock["transformers"]: raise RuntimeError("frozen Unsloth/Transformers environment mismatch")', 'if subprocess.run([lock["ptxas_path"],"--version"],capture_output=True,text=True).returncode: raise RuntimeError("verified ptxas unavailable")', 'out.mkdir(parents=True,exist_ok=True)',
        'for name in ("eval60_manifest.json","ttt48_config_frozen.json","ttt24_candidates_frozen.json","ttt24_predictions_frozen.json","ttt24_report.json"): shutil.copy2(frozen/name,out/name)',
        f'print(json.dumps({{"event":"EVAL60_TTT48_PAIRED_TARGET_BLIND_START","source_dataset":"{dataset_slug}","gpus":gpus,"solutions_opened":False}},sort_keys=True),flush=True)',
        'generate=[sys.executable,str(root/"scripts/run_eval60_ttt48_paired_confirmation.py"),"--manifest",str(out/"eval60_manifest.json"),"--reference-config",str(out/"ttt48_config_frozen.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(root/"configs/nvarc_native_846d0198"),"--output",str(out/"candidates_frozen.json"),"--checkpoint-dir",str(out/"checkpoints"),"--resume"]',
        'if subprocess.run(generate,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0,1,2,3","TRITON_PTXAS_PATH":"/usr/local/cuda-12.5/bin/ptxas","HF_HUB_OFFLINE":"1","TRANSFORMERS_OFFLINE":"1"}).returncode: raise RuntimeError("Eval60 TTT48 candidate freeze failed")',
        'candidates=json.loads((out/"candidates_frozen.json").read_text())', 'if candidates.get("status")!="EVAL60_TTT48_PAIRED_CONFIRMATION_CANDIDATES_FROZEN_BEFORE_EXACT_SCORING" or len(candidates.get("records",{}))!=60 or candidates.get("solutions_opened") is not False: raise RuntimeError("candidate freeze validation failed; refusing target access")', 'print(json.dumps({"event":"EVAL60_TTT48_CANDIDATES_FROZEN","solutions_opened":False,"gpu_after_workers":subprocess.check_output(["nvidia-smi"],text=True)[-1500:]},sort_keys=True),flush=True)',
        'score=[sys.executable,str(root/"scripts/score_eval60_ttt48_paired_confirmation.py"),"--manifest",str(out/"eval60_manifest.json"),"--ttt24-candidates",str(out/"ttt24_candidates_frozen.json"),"--ttt24-predictions",str(out/"ttt24_predictions_frozen.json"),"--ttt24-report",str(out/"ttt24_report.json"),"--candidates",str(out/"candidates_frozen.json"),"--challenge-path",str(challenge),"--solutions-path",str(solutions),"--output-dir",str(out)]',
        'if subprocess.run(score,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("Eval60 TTT48 post-freeze scoring failed")', 'required=[out/name for name in ("eval60_manifest.json","ttt48_config_frozen.json","candidates_frozen.json","b_support_selection_frozen.json","predictions_frozen.json","paired_comparison.csv","runtime_per_task.csv","EVAL60_TTT48_REPORT.json","EVAL60_TTT48_REPORT.md")]', 'if any(not item.is_file() for item in required): raise RuntimeError("missing required TTT48 output")', 'print(json.dumps({"event":"EVAL60_TTT48_PAIRED_COMPLETE","artifacts":[str(item) for item in required]},sort_keys=True),flush=True)',
    ])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", default="jimmy5566"); parser.add_argument("--dataset-slug", default="arc2-eval60-ttt48-paired-source"); parser.add_argument("--kernel-slug", default="arc2-eval60-ttt48-paired-confirmation"); args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Eval60 TTT48 staging")
    manifest, config, old_candidates, old_predictions, old_report = _frozen_inputs(); commit = _commit()
    dataset_root = args.output / "dataset" / "ARC2"; dataset_root.mkdir(parents=True); _archive(dataset_root, commit)
    frozen = dataset_root / "eval60_ttt48_confirmation"
    _write(frozen / "eval60_manifest.json", manifest); _write(frozen / "ttt48_config_frozen.json", config)
    _write(frozen / "ttt24_candidates_frozen.json", old_candidates); _write(frozen / "ttt24_predictions_frozen.json", old_predictions); _write(frozen / "ttt24_report.json", old_report)
    if list(dataset_root.rglob("*solutions*.json")):
        raise RuntimeError("target-blind source dataset contains solutions")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title": "ARC2 Eval60 TTT48 Paired Source", "subtitle": "Target-blind 48-step reference-style TTT paired development confirmation", "description": "Private experiment source; no ARC solution grids.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]})
    kernel = args.output / "kernel"; kernel.mkdir(); notebook = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook, {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4})
    _write(kernel / "kernel-metadata.json", {"id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Eval60 TTT48 Paired Confirmation", "code_file": notebook, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868", "machine_shape": "NvidiaL4"})
    print(json.dumps({"event": "EVAL60_TTT48_PAIRED_STAGING_READY", "source_commit": commit, "manifest": manifest, "config": config}, sort_keys=True))


if __name__ == "__main__":
    main()
