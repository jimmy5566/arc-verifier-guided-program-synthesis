"""Stage the fixed three-task, official-style NVARC TTT recovery pilot."""
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_eval30_candidate_search_kaggle import _manifest_and_s0


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_ids(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent, baseline = _manifest_and_s0()
    task_ids = sorted(parent["task_ids"], key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))[:3]
    if len(task_ids) != 3 or set(task_ids) - set(baseline["records"]):
        raise ValueError("could not select the deterministic Eval3 pool-miss cohort")
    manifest = {
        "experiment_id": "ARC2_REFERENCE_STYLE_TTT_EVAL3_PILOT",
        "status": "EVAL3_REFERENCE_TTT_COHORT_FROZEN",
        "selection_rule": "three smallest SHA256(task_id) values from immutable Eval30 Aug8 greedy A_POOL_MISS cohort; task_id breaks ties",
        "selection_provenance": "Eval30 was previously selected from Evaluation60 pool misses. This new TTT executable contains no evaluation targets.",
        "task_ids": task_ids,
        "task_ids_hash": _hash_ids(task_ids),
        "source_challenge_sha256": parent["source_challenge_sha256"],
        "baseline_any_of_k": "0/3 by immutable parent Eval30 A_POOL_MISS definition; independently asserted only after the new candidate pool is frozen",
        "integrity": {"targets_inspected_during_ttt_or_generation": False, "no_dfs": True, "no_beam": True, "no_b_support_change": True, "selection_is_immutable": True},
        "frozen_inference": parent["frozen_inference"],
    }
    s0 = {
        "experiment_id": manifest["experiment_id"],
        "status": "EVAL3_REUSED_AUG8_GREEDY_BASELINE_FROZEN",
        "protocol": "Exact subset reuse of Evaluation60 Aug8 greedy records; baseline generation was not rerun.",
        "task_ids": task_ids,
        "task_ids_hash": manifest["task_ids_hash"],
        "records": {task_id: baseline["records"][task_id] for task_id in task_ids},
    }
    config = {
        "style": "official_nvarc_train_only_partial_24_steps",
        "reference_repository": "https://github.com/1ytic/NVARC",
        "reference_commit": "846d0198efa752534594e321fc3289fc0a06c657",
        "reference_notebook": "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue",
        "kernel_source": "sorokin/pip-install-unsloth-flash-patch",
        "ptxas_path": "/usr/local/cuda-12.5/bin/ptxas",
        "rank": 256,
        "alpha": 32,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        "use_rslora": True,
        "learning_rate": 5e-5,
        "scheduler": "cosine",
        "warmup_ratio": 0.1,
        "reference_schedule_total_steps": 128,
        "ttt_steps": 24,
        "bf16": True,
        "gradient_checkpointing": False,
        "augmentation_geometry_count": 8,
        "augmentation_colour_permutations": 16,
        "train_pair_order": "random full permutation",
        "max_sequence_length": 8192,
        "generation_augmentation_count": 8,
        "max_new_tokens": 1024,
        "generation_context_window": 16384,
        "generation": "unchanged greedy Aug8; beam=1",
        "seed": 42,
    }
    return manifest, s0, config


def _notebook(dataset_slug: str) -> str:
    return "\n".join([
        "import json, os, shutil, subprocess, sys, tarfile", "from pathlib import Path", "",
        'inputs = Path("/kaggle/input")',
        'source_root = Path("/kaggle/working/reference_ttt_source")',
        'with tarfile.open(next(inputs.rglob("ARC2.tar"))) as archive: archive.extractall(source_root, filter="data")',
        'root = source_root / "ARC2"',
        'frozen = next(inputs.rglob("eval3_manifest.json")).parent',
        'challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
        'solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")',
        'model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'out = Path("/kaggle/working/artifacts/eval3_reference_ttt")',
        'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if not gpus or "NVIDIA L4" not in gpus[0]: raise RuntimeError(f"requires an NVIDIA L4: {gpus}")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be off")',
        'if not Path("/usr/local/cuda-12.5/bin/ptxas").is_file(): raise RuntimeError("verified ptxas path unavailable")',
        'out.mkdir(parents=True, exist_ok=True)',
        'for name in ("eval3_manifest.json", "reference_ttt_config_frozen.json", "s0_greedy_reused_candidates_frozen.json"): shutil.copy2(frozen / name, out / name)',
        f'print(json.dumps({{"event": "EVAL3_REFERENCE_TTT_TARGET_BLIND_START", "source_dataset": "{dataset_slug}", "gpu": gpus[0], "solutions_opened": False, "ttt_steps": 24}}, sort_keys=True), flush=True)',
        'command = [sys.executable, str(root / "scripts/run_eval3_reference_ttt.py"), "--manifest", str(out / "eval3_manifest.json"), "--reference-config", str(out / "reference_ttt_config_frozen.json"), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(root / "configs/nvarc_native_846d0198"), "--output", str(out / "reference_ttt_candidates_frozen.json"), "--checkpoint-dir", str(out / "checkpoints"), "--resume"]',
        'if subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "TRITON_PTXAS_PATH": "/usr/local/cuda-12.5/bin/ptxas", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}).returncode: raise RuntimeError("Eval3 reference-style TTT generation failed")',
        'frozen_candidates = out / "reference_ttt_candidates_frozen.json"',
        'if not frozen_candidates.is_file(): raise RuntimeError("candidate freeze missing; refusing target access")',
        'print(json.dumps({"event": "EVAL3_REFERENCE_TTT_CANDIDATES_FROZEN", "solutions_opened": False}, sort_keys=True), flush=True)',
        'score = [sys.executable, str(root / "scripts/score_eval3_reference_ttt.py"), "--manifest", str(out / "eval3_manifest.json"), "--reference-config", str(out / "reference_ttt_config_frozen.json"), "--baseline", str(out / "s0_greedy_reused_candidates_frozen.json"), "--candidates", str(frozen_candidates), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--output-dir", str(out)]',
        'if subprocess.run(score, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}).returncode: raise RuntimeError("post-freeze Eval3 score failed")',
        'print(json.dumps({"event": "EVAL3_REFERENCE_TTT_COMPLETE", "output": str(out)}, sort_keys=True), flush=True)',
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-eval3-reference-ttt-source")
    parser.add_argument("--kernel-slug", default="arc2-eval3-reference-ttt-pilot")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Eval3 reference-TTT staging")
    manifest, baseline, config = _frozen_inputs(); commit = _commit()
    source_root = args.output / "dataset" / "ARC2"; source_root.mkdir(parents=True); _archive(source_root, commit)
    frozen = source_root / "eval3_reference_ttt"; _write(frozen / "eval3_manifest.json", manifest); _write(frozen / "reference_ttt_config_frozen.json", config); _write(frozen / "s0_greedy_reused_candidates_frozen.json", baseline)
    if list(source_root.rglob("*solutions*.json")):
        raise RuntimeError("target-blind source dataset contains evaluation solutions")
    with tarfile.open(args.output / "dataset" / "ARC2.tar", "w") as archive:
        archive.add(source_root, arcname="ARC2")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title": "ARC2 Eval3 Reference TTT Source", "subtitle": "Target-blind official-style NVARC rank-256 partial TTT pilot", "description": "Private source attachment without ARC evaluation solution grids.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]})
    kernel = args.output / "kernel"; kernel.mkdir(); notebook_name = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.11"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4})
    _write(kernel / "kernel-metadata.json", {"id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Eval3 Reference TTT Pilot", "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": ["sorokin/pip-install-unsloth-flash-patch"], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "docker_image": "gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868", "machine_shape": "NvidiaL4"})
    print(json.dumps({"event": "EVAL3_REFERENCE_TTT_STAGING_READY", "source_commit": commit, "manifest": manifest, "config": config, "dataset": str(args.output / "dataset"), "kernel": str(kernel)}, sort_keys=True))


if __name__ == "__main__":
    main()
