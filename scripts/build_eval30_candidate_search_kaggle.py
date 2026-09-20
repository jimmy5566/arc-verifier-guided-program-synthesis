"""Stage a target-blind Kaggle Eval30 Beam/DFS candidate-recall diagnostic."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_COMMIT = "ec5a787ff16a6ff0240a76e77f6060b73e47fec3"
CHALLENGE = ROOT / "data" / "raw" / "arc-agi_evaluation_challenges.json"
BASELINE_ROOT = ROOT / "artifacts" / "eval60_aug8_kaggle_v2_output" / "artifacts" / "eval60_aug8"


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _task_hash(task_ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode()).hexdigest()


def _archive(destination: Path) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle:
        subprocess.run(["git", "archive", "--format=tar", SOURCE_COMMIT], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as source:
        source.extractall(destination, filter="data")
    archive.unlink()


def _manifest_and_s0() -> tuple[dict[str, Any], dict[str, Any]]:
    baseline = _read(BASELINE_ROOT / "A_candidates_frozen.json")
    with (BASELINE_ROOT / "per_task_results.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    misses = sorted(row["task_id"] for row in rows if row["classification"] == "A_POOL_MISS")
    if len(misses) != 58:
        raise ValueError(f"expected exactly 58 frozen Evaluation60 pool misses, got {len(misses)}")
    selected = sorted(misses, key=lambda task_id: (hashlib.sha256(task_id.encode()).hexdigest(), task_id))[:30]
    challenges = _read(CHALLENGE)
    if len(challenges) != 120 or set(selected) - set(challenges) or set(selected) - set(baseline.get("records", ())):
        raise ValueError("invalid Eval60 baseline or selected Eval30 task")
    manifest = {
        "experiment_id": "ARC2_EVAL30_CANDIDATE_SEARCH_DIAGNOSTIC",
        "status": "EVAL30_COHORT_FROZEN_BEFORE_NEW_SEARCH",
        "selection_rule": "from already frozen Evaluation60 A_POOL_MISS task IDs, choose 30 smallest SHA256(task_id) values; task_id breaks ties",
        "selection_provenance": "Previously scored Evaluation60 baseline labels; this new-search source contains no ARC evaluation solutions.",
        "source": "arc-agi_evaluation_challenges.json", "source_challenge_sha256": _sha256(CHALLENGE), "source_task_count": 120,
        "baseline_pool_miss_count": 58, "baseline_pool_miss_ids_hash": _task_hash(misses), "baseline_candidates_sha256": _sha256(BASELINE_ROOT / "A_candidates_frozen.json"),
        "task_ids": selected, "task_count": 30, "task_ids_hash": _task_hash(selected),
        "integrity": {"new_search_solutions_opened": False, "targets_inspected_during_search": False, "selection_is_immutable": True},
        "frozen_inference": {"source_commit": SOURCE_COMMIT, "model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "augmentation_count": 8, "generation_micro_batch_size": 1, "likelihood_micro_batch_size": 1, "worker_count": 4, "search_beams": 1},
    }
    s0 = {
        "experiment_id": manifest["experiment_id"], "status": "EVAL30_REUSED_AUG8_GREEDY_CANDIDATES_FROZEN",
        "protocol": "Exact subset reuse of Evaluation60 Aug8 greedy records; baseline generation was not rerun.", "condition": "s0",
        "task_ids": selected, "task_ids_hash": manifest["task_ids_hash"], "source_challenge_sha256": manifest["source_challenge_sha256"],
        "source_artifact_sha256": manifest["baseline_candidates_sha256"], "records": {task_id: baseline["records"][task_id] for task_id in selected},
    }
    return manifest, s0


def _search_config() -> dict[str, Any]:
    common = {"augmentation_count": 8, "max_new_tokens": 1024, "context_window": 16384}
    return {
        "experiment_id": "ARC2_EVAL30_CANDIDATE_SEARCH_DIAGNOSTIC", "status": "FROZEN_BEFORE_NEW_SEARCH",
        "protocol": "Budgets are fixed before model generation. DFS branches only on model probabilities constrained by the native rectangular-grid grammar.",
        "conditions": {
            "beam2": {**common, "algorithm": "native_deterministic_beam_search", "beam_width": 2},
            "dfs_small": {**common, "algorithm": "grammar_constrained_probability_only_depth_first_search", "max_completed_paths_per_view": 2, "max_pending_states_per_view": 4, "max_depth": 2, "max_branch_points_per_path": 6, "alternatives_per_branch_point": 1, "max_wall_seconds_per_path": 120.0},
            "dfs_medium": {**common, "algorithm": "grammar_constrained_probability_only_depth_first_search", "max_completed_paths_per_view": 4, "max_pending_states_per_view": 12, "max_depth": 3, "max_branch_points_per_path": 12, "alternatives_per_branch_point": 2, "max_wall_seconds_per_path": 120.0},
        },
    }


def _notebook(dataset_slug: str) -> str:
    lines = [
        "import json, os, shutil, subprocess, sys", "from pathlib import Path", "",
        'inputs = Path("/kaggle/input")',
        'root = next(inputs.rglob("run_eval30_candidate_search.py")).parents[1]',
        'baseline = next(inputs.rglob("s0_greedy_reused_candidates_frozen.json")).parent',
        'challenge = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
        'solutions = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")',
        'model = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        'out = Path("/kaggle/working/artifacts/eval30_candidate_search")',
        'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if len(gpus) != 4 or any("NVIDIA L4" not in item for item in gpus): raise RuntimeError(f"requires exactly 4 NVIDIA L4 GPUs: {gpus}")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be off")',
        'out.mkdir(parents=True, exist_ok=True)',
        'for name in ("eval30_manifest.json", "search_config.json", "s0_greedy_reused_candidates_frozen.json"): shutil.copy2(baseline / name, out / name)',
        f'print(json.dumps({{"event":"EVAL30_SEARCH_TARGET_BLIND_START","gpus":gpus,"internet":False,"source_dataset":"{dataset_slug}","solutions_opened":False}}, sort_keys=True), flush=True)',
        'for condition, output_name in (("beam2", "beam2_candidates_frozen.json"), ("dfs_small", "dfs_small_candidates_frozen.json"), ("dfs_medium", "dfs_medium_candidates_frozen.json")):',
        '    command = [sys.executable, str(root / "scripts/run_eval30_candidate_search.py"), "--manifest", str(out / "eval30_manifest.json"), "--search-config", str(out / "search_config.json"), "--condition", condition, "--challenge-path", str(challenge), "--model-path", str(model), "--native-config-dir", str(root / "configs/nvarc_native_846d0198"), "--config", str(root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json"), "--output", str(out / output_name), "--checkpoint-dir", str(out / "checkpoints" / condition), "--resume"]',
        '    print(json.dumps({"event":"EVAL30_SEARCH_CONDITION_START","condition":condition,"solutions_opened":False}, sort_keys=True), flush=True)',
        '    if subprocess.run(command, env={**os.environ, "CUDA_VISIBLE_DEVICES":"0,1,2,3"}).returncode: raise RuntimeError(f"{condition} target-blind generation failed")',
        'frozen = [out / "s0_greedy_reused_candidates_frozen.json", out / "beam2_candidates_frozen.json", out / "dfs_small_candidates_frozen.json", out / "dfs_medium_candidates_frozen.json"]',
        'if any(not item.is_file() for item in frozen): raise RuntimeError("candidate freeze incomplete; refusing target access")',
        'print(json.dumps({"event":"EVAL30_ALL_CANDIDATES_FROZEN","conditions":[str(item) for item in frozen],"solutions_opened":False}, sort_keys=True), flush=True)',
        'score = [sys.executable, str(root / "scripts/score_eval30_candidate_search.py"), "--manifest", str(out / "eval30_manifest.json"), "--search-config", str(out / "search_config.json"), "--s0", str(out / "s0_greedy_reused_candidates_frozen.json"), "--beam2", str(out / "beam2_candidates_frozen.json"), "--dfs-small", str(out / "dfs_small_candidates_frozen.json"), "--dfs-medium", str(out / "dfs_medium_candidates_frozen.json"), "--challenge-path", str(challenge), "--solutions-path", str(solutions), "--output-dir", str(out)]',
        'if subprocess.run(score, env={**os.environ, "CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("post-freeze candidate-recall score failed")',
        'print(json.dumps({"event":"EVAL30_SEARCH_COMPLETE","output":str(out)}, sort_keys=True), flush=True)',
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="jimmy5566")
    parser.add_argument("--dataset-slug", default="arc2-eval30-candidate-search-source")
    parser.add_argument("--kernel-slug", default="arc2-eval30-candidate-search-diagnostic")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("refusing to overwrite Eval30 candidate-search staging")
    manifest, s0 = _manifest_and_s0(); search_config = _search_config()
    dataset_root = args.output / "dataset" / "ARC2"; dataset_root.mkdir(parents=True)
    _archive(dataset_root)
    for name in ("run_eval30_candidate_search.py", "score_eval30_candidate_search.py"):
        shutil.copy2(ROOT / "scripts" / name, dataset_root / "scripts" / name)
    baseline = dataset_root / "eval30_candidate_search_baseline"; baseline.mkdir()
    _write(baseline / "eval30_manifest.json", manifest); _write(baseline / "search_config.json", search_config); _write(baseline / "s0_greedy_reused_candidates_frozen.json", s0)
    if list(dataset_root.rglob("*solutions*.json")):
        raise RuntimeError("target-blind source dataset must not contain ARC solution files")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title": "ARC2 Eval30 Candidate Search Source", "subtitle": "Target-blind Beam2 and bounded DFS diagnostic source", "description": "Private source attachment; no ARC target solution grids are included.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]})
    kernel = args.output / "kernel"; kernel.mkdir(); notebook_name = f"{args.kernel_slug}.ipynb"
    _write(kernel / notebook_name, {"cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [line + "\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.12"}, "kaggle": {"accelerator": "nvidiaL4", "isGpuEnabled": True, "isInternetEnabled": False, "language": "python", "sourceType": "notebook"}}, "nbformat": 4, "nbformat_minor": 4})
    _write(kernel / "kernel-metadata.json", {"id": f"{args.owner}/{args.kernel_slug}", "title": "ARC2 Eval30 Candidate Search Diagnostic", "code_file": notebook_name, "language": "python", "kernel_type": "notebook", "is_private": True, "enable_gpu": True, "enable_tpu": False, "enable_internet": False, "keywords": ["gpu"], "dataset_sources": [f"{args.owner}/{args.dataset_slug}"], "kernel_sources": [], "competition_sources": ["arc-prize-2026-arc-agi-2"], "model_sources": ["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"], "machine_shape": "NvidiaL4"})
    print(json.dumps({"status": "EVAL30_CANDIDATE_SEARCH_STAGING_READY", "manifest": manifest, "search_config": search_config, "dataset": str(args.output / "dataset"), "kernel": str(kernel)}, sort_keys=True))


if __name__ == "__main__":
    main()
