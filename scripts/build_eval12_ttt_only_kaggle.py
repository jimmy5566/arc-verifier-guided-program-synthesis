"""Stage the target-blind Eval12, train-pair-only native TTT diagnostic."""
from __future__ import annotations

import argparse
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

from scripts.build_eval30_candidate_search_kaggle import _manifest_and_s0


NATIVE_CONFIG = ROOT / "configs" / "QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _source_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def _ttt_config() -> dict[str, Any]:
    config = json.loads(NATIVE_CONFIG.read_text(encoding="utf-8"))
    ttt = dict(config["D_ttt_lora"])
    expected = {
        "train_pairs_only": True,
        "rank": 8,
        "alpha": 16,
        "steps": 24,
        "learning_rate": 5e-4,
        "target_suffixes": ["q_proj", "v_proj"],
    }
    if {key: ttt.get(key) for key in expected} != expected:
        raise ValueError("native TTT configuration no longer matches the frozen Eval12 contract")
    return {
        "experiment_id": "ARC2_EVAL12_TTT_ONLY_DIAGNOSTIC",
        "status": "FROZEN_BEFORE_TTT_GENERATION",
        "protocol": "Task-local LoRA uses supplied train pairs only. Candidate generation uses the unchanged native Aug8 greedy interface; target solutions are unavailable to generation.",
        "model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "native_config_sha256": _sha256(NATIVE_CONFIG),
        "augmentation_count": 8,
        "search_beams": 1,
        "generation_micro_batch_size": 1,
        "likelihood_micro_batch_size": 1,
        "worker_count": 4,
        "ttt": {**expected, "reset_between_tasks": True, "no_test_targets": True},
        "integrity": {"solutions_opened_during_generation": False, "targets_inspected_during_generation": False, "no_dfs": True, "no_beam_search": True, "no_production_code_change": True},
    }


def _manifest_and_baseline() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    eval30, s0 = _manifest_and_s0()
    source_ids = list(eval30["task_ids"])
    selected = sorted(source_ids, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))[:12]
    if len(selected) != 12 or len(selected) != len(set(selected)):
        raise ValueError("could not construct exactly 12 unique Eval30 task IDs")
    if set(selected) - set(s0["records"]):
        raise ValueError("Eval12 task absent from frozen greedy baseline")
    manifest = {
        "experiment_id": "ARC2_EVAL12_TTT_ONLY_DIAGNOSTIC",
        "status": "EVAL12_COHORT_FROZEN_BEFORE_TTT_GENERATION",
        "selection_rule": "from the immutable Eval30 Aug8 greedy pool-miss cohort, choose the 12 smallest SHA256(task_id) values; task_id breaks ties",
        "selection_provenance": "Eval30 was itself frozen from historical Evaluation60 A_POOL_MISS records. No new TTT source contains evaluation solution grids.",
        "source": eval30["source"],
        "source_challenge_sha256": eval30["source_challenge_sha256"],
        "source_task_count": 120,
        "parent_eval30_task_ids_hash": eval30["task_ids_hash"],
        "parent_baseline_candidates_sha256": eval30["baseline_candidates_sha256"],
        "baseline_any_of_k": "0/12 by immutable parent Eval30 S0 pool-miss definition; independently rechecked only after TTT candidate freeze",
        "task_ids": selected,
        "task_count": 12,
        "task_ids_hash": _task_hash(selected),
        "integrity": {"selection_is_immutable": True, "new_ttt_solutions_opened": False, "targets_inspected_during_ttt_generation": False},
        "frozen_inference": eval30["frozen_inference"],
    }
    baseline = {
        "experiment_id": manifest["experiment_id"],
        "status": "EVAL12_REUSED_AUG8_GREEDY_BASELINE_FROZEN",
        "protocol": "Exact subset reuse from the immutable Eval30 S0 greedy candidate pool; baseline generation was not rerun.",
        "task_ids": selected,
        "task_ids_hash": manifest["task_ids_hash"],
        "source_challenge_sha256": manifest["source_challenge_sha256"],
        "source_artifact_sha256": eval30["baseline_candidates_sha256"],
        "records": {task_id: s0["records"][task_id] for task_id in selected},
    }
    return manifest, baseline, _ttt_config()


def _notebook(dataset_slug: str) -> str:
    lines = [
        "import json, os, shutil, subprocess, sys", "from pathlib import Path", "",
        'inputs = Path("/kaggle/input")',
        'root = next(inputs.rglob("run_qwen4b_native_augmentation_search.py")).parents[1]',
        'frozen = next(inputs.rglob("eval12_manifest.json")).parent',
        'challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
        'solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")',
        'model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'out = Path("/kaggle/working/artifacts/eval12_ttt_only")',
        'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if len(gpus) != 4 or any("NVIDIA L4" not in item for item in gpus): raise RuntimeError(f"requires exactly 4 NVIDIA L4 GPUs: {gpus}")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be off")',
        'out.mkdir(parents=True, exist_ok=True)',
        'for name in ("eval12_manifest.json", "ttt_config_frozen.json", "s0_greedy_reused_candidates_frozen.json"): shutil.copy2(frozen / name, out / name)',
        f'print(json.dumps({{"event":"EVAL12_TTT_TARGET_BLIND_START","gpus":gpus,"internet":False,"source_dataset":"{dataset_slug}","solutions_opened":False}}, sort_keys=True), flush=True)',
        'command = [sys.executable, str(root / "scripts/run_qwen4b_native_augmentation_search.py"), "--cohort", str(out / "eval12_manifest.json"), "--config", str(root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"), "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(root / "configs/nvarc_native_846d0198"), "--output", str(out / "ttt_candidates_frozen.json"), "--checkpoint-dir", str(out / "checkpoints"), "--resume", "--stage", "external", "--external-augmentation-count", "8", "--external-worker-count", "4", "--enable-ttt", "--search-beams", "1", "--generation-micro-batch-size", "1", "--likelihood-micro-batch-size", "1"]',
        'if subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES":"0,1,2,3"}).returncode: raise RuntimeError("Eval12 TTT target-blind generation failed")',
        'frozen_candidates = out / "ttt_candidates_frozen.json"',
        'if not frozen_candidates.is_file(): raise RuntimeError("TTT candidate freeze incomplete; refusing target access")',
        'print(json.dumps({"event":"EVAL12_TTT_CANDIDATES_FROZEN","path":str(frozen_candidates),"solutions_opened":False}, sort_keys=True), flush=True)',
        'score = [sys.executable, str(root / "scripts/score_eval12_ttt_only.py"), "--manifest", str(out / "eval12_manifest.json"), "--ttt-config", str(out / "ttt_config_frozen.json"), "--baseline", str(out / "s0_greedy_reused_candidates_frozen.json"), "--ttt", str(frozen_candidates), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--output-dir", str(out)]',
        'if subprocess.run(score, env={**os.environ, "CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("post-freeze Eval12 TTT scoring failed")',
        'print(json.dumps({"event":"EVAL12_TTT_COMPLETE","output":str(out)}, sort_keys=True), flush=True)',
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-eval12-ttt-only-source")
    parser.add_argument("--kernel-slug", default="arc2-eval12-ttt-only-diagnostic")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Eval12 TTT staging")
    manifest, baseline, ttt_config = _manifest_and_baseline()
    commit = _source_commit()
    dataset_root = args.output / "dataset" / "ARC2"
    dataset_root.mkdir(parents=True)
    _archive(dataset_root, commit)
    frozen = dataset_root / "eval12_ttt_baseline"
    _write(frozen / "eval12_manifest.json", manifest)
    _write(frozen / "ttt_config_frozen.json", ttt_config)
    _write(frozen / "s0_greedy_reused_candidates_frozen.json", baseline)
    if list(dataset_root.rglob("*solutions*.json")):
        raise RuntimeError("target-blind source dataset must not contain ARC solution files")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title": "ARC2 Eval12 TTT-Only Source", "subtitle": "Target-blind train-pair-only native LoRA diagnostic", "description": "Private source attachment; no ARC evaluation solution grids.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]})
    kernel = args.output / "kernel"; kernel.mkdir()
    notebook_name = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4})
    _write(kernel / "kernel-metadata.json", {"id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Eval12 TTT-Only Diagnostic", "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "machine_shape": "NvidiaL4"})
    print(json.dumps({"status": "EVAL12_TTT_STAGING_READY", "source_commit": commit, "manifest": manifest, "ttt_config": ttt_config, "dataset": str(args.output / "dataset"), "kernel": str(kernel)}, sort_keys=True))


if __name__ == "__main__":
    main()
