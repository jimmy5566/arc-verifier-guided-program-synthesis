#!/usr/bin/env python3
"""Build the private review package for DFS2048 Retention30; never publish."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import build_eval60_retention30_kaggle as shared


EXPERIMENT_ID = "EVAL60_DUAL_TTT_DFS2048_RETENTION30_V1"
EXPERIMENT_RELATIVE = "experiments/eval60_dual_ttt_dfs2048_retention30_v1"
RUNNER_RELATIVE = "scripts/run_eval60_dual_ttt_dfs2048_retention30.py"
SCORER_RELATIVE = "scripts/score_eval60_dfs2048_retention30.py"
OUTPUT_FOLDER = "eval60_dual_ttt_dfs2048_retention30_v1"
NOTEBOOK_TITLE = "ARC2 Eval60 Dual-TTT DFS2048 Retention30"
SOURCE_PATHS = (
    "src",
    "configs/nvarc_native_846d0198",
    "scripts/run_eval60_dual_ttt_dfs1024_retention30.py",
    "scripts/run_eval60_dual_ttt_dfs2048_retention30.py",
    "scripts/score_eval60_retention30.py",
    "scripts/score_eval60_dfs2048_retention30.py",
    "scripts/run_l4_dual_ttt_dfs1024_benchmark.py",
    "scripts/run_adaptive_ttt_loo_transfer12.py",
    "scripts/run_adaptive_ttt_step1.py",
    "scripts/run_eval3_reference_ttt.py",
    "scripts/turbodfs_d1_common.py",
    "scripts/turbodfs_v4_common.py",
    "scripts/run_eval60_adaptive_inference_joint_v2.py",
    f"{EXPERIMENT_RELATIVE}/TARGET_BLIND_RUN_MANIFEST.json",
    f"{EXPERIMENT_RELATIVE}/RETENTION30_OUTPUTS.json",
    f"{EXPERIMENT_RELATIVE}/RETENTION30_COHORT_SHA256.txt",
    f"{EXPERIMENT_RELATIVE}/HISTORICAL_GREEDY_SOURCE_CLASSES.csv",
    f"{EXPERIMENT_RELATIVE}/DFS1024_SOURCE_HIT_OVERLAP.csv",
    f"{EXPERIMENT_RELATIVE}/DFS1024_DFS_CELLS.csv",
    f"{EXPERIMENT_RELATIVE}/DFS1024_RUNTIME_BASELINE.json",
    f"{EXPERIMENT_RELATIVE}/DFS1024_TASK_CHECKPOINTS",
    f"{EXPERIMENT_RELATIVE}/PROVENANCE.json",
    f"{EXPERIMENT_RELATIVE}/PRE_RUN_STATIC_AUDIT.json",
    f"{EXPERIMENT_RELATIVE}/CPU_TEST_REPORT.json",
    f"{EXPERIMENT_RELATIVE}/retention_config.json",
    f"{EXPERIMENT_RELATIVE}/README.md",
    "pyproject.toml",
)
DATASET_COPY_NAMES = (
    "TARGET_BLIND_RUN_MANIFEST.json", "RETENTION30_OUTPUTS.json",
    "RETENTION30_COHORT_SHA256.txt", "HISTORICAL_GREEDY_SOURCE_CLASSES.csv",
    "PROVENANCE.json", "PRE_RUN_STATIC_AUDIT.json", "CPU_TEST_REPORT.json",
)
REQUIRED_OUTPUTS = [
    "PROVENANCE.json", "RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt",
    "HISTORICAL_GREEDY_SOURCE_CLASSES.csv", "TARGET_BLIND_RUN_MANIFEST.json",
    "dfs_candidates_frozen.json", "ttt24_union_candidates_frozen.json",
    "ttt48_union_candidates_frozen.json", "dual_union_candidates_frozen.json",
    "TARGET_BLIND_CANDIDATE_FREEZE.json", "dfs_cells.csv", "phase_intervals.csv",
    "gpu_telemetry_1s.csv", "gpu_phase_summary.csv", "TELEMETRY_SUMMARY.json",
    "system_telemetry_1s.csv", "task_runtime.csv", "target_blind_runtime_summary.json",
    "startup_summary.json", "worker_summary.csv", "dfs_source_hit_overlap.csv", "DFS_HIT_OVERLAP_SUMMARY.json",
    "dfs_view_gold_contribution.csv", "historical_ttt48_marginal_under_dfs2048.csv",
    "historical_ttt24_marginal_under_dfs2048.csv", "historical_top2_retention_dfs2048.csv",
    "historical_oracle_only_retention_dfs2048.csv", "dfs1024_vs_dfs2048_retention.csv",
    "dfs1024_vs_dfs2048_summary.json", "TTT_STATE_COMPARABILITY.json",
    "DFS2048_RETENTION_REPORT.md", "DECISION.json",
]
SOURCE_IDENTITY_EXTRA = {
    "scientific_change": {"search.max_expanded_nodes": {"before": 1024, "after": 2048}},
    "all_other_scientific_fields_identical": True,
    "cohort_changed": False,
    "target_blind": True,
    "kaggle_run_started": False,
}
PACKAGE_EXTRA = {
    "scientific_config_changed": True,
    "scientific_config_diff": {"search.max_expanded_nodes": {"before": 1024, "after": 2048}},
    "all_other_scientific_fields_identical": True,
    "cohort_changed": False,
    "kaggle_run_started": False,
}


def notebook_source(owner: str, dataset_slug: str) -> str:
    dataset_id = f"{owner}/{dataset_slug}"
    baseline = f'SOURCE / "{EXPERIMENT_RELATIVE}"'
    return "\n".join([
        "import json, os, shutil, subprocess, sys, time",
        "from pathlib import Path",
        "NOTEBOOK_START_MONOTONIC = time.perf_counter()",
        f'EXPERIMENT_ID = "{EXPERIMENT_ID}"',
        f'DATASET = Path("/kaggle/input/datasets/{dataset_id}")',
        'SOURCE = DATASET / "ARC2"',
        'CHALLENGE = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")',
        'MODEL = Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")',
        f'RUNNER = SOURCE / "{RUNNER_RELATIVE}"',
        f'SCORER = SOURCE / "{SCORER_RELATIVE}"',
        'NATIVE_CONFIG = SOURCE / "configs/nvarc_native_846d0198"',
        'CONFIG = DATASET / "retention_config_resolved.json"',
        'MANIFEST = DATASET / "TARGET_BLIND_RUN_MANIFEST.json"',
        'RETENTION = DATASET / "RETENTION30_OUTPUTS.json"',
        f'BASELINE = {baseline}',
        'OUTPUT = Path("/kaggle/working/analysis/eval60_dual_ttt_dfs2048_retention30_v1")',
        'required = [SOURCE, CHALLENGE, MODEL, RUNNER, SCORER, NATIVE_CONFIG, CONFIG, MANIFEST, RETENTION, BASELINE / "DFS1024_SOURCE_HIT_OVERLAP.csv", BASELINE / "DFS1024_DFS_CELLS.csv", BASELINE / "DFS1024_RUNTIME_BASELINE.json", BASELINE / "DFS1024_TASK_CHECKPOINTS"]',
        'missing = [str(path) for path in required if not path.exists()]',
        'if missing: raise RuntimeError(f"retention inputs missing: {missing}")',
        'if OUTPUT.exists() and any(OUTPUT.iterdir()): raise RuntimeError("RERUN_ENABLED=FALSE: output already exists")',
        'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1", "true", "yes"}: raise RuntimeError("Internet must be disabled")',
        'gpus = subprocess.check_output(["nvidia-smi", "-L"], text=True).splitlines()',
        'if len(gpus) != 4 or any("NVIDIA L4" not in row for row in gpus): raise RuntimeError(f"requires exactly four NVIDIA L4 GPUs: {gpus}")',
        'runner_command = [sys.executable, str(RUNNER), "--challenge", str(CHALLENGE), "--model-path", str(MODEL), "--native-config-dir", str(NATIVE_CONFIG), "--config", str(CONFIG), "--manifest", str(MANIFEST), "--output", str(OUTPUT)]',
        'print(json.dumps({"event": "TARGET_BLIND_GPU_PHASE_START", "solutions_passed_to_runner": False, "gpus": gpus}, sort_keys=True), flush=True)',
        'subprocess.run(runner_command, check=True, env={**os.environ, "TRITON_PTXAS_PATH": "/usr/local/cuda-12.5/bin/ptxas", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false"})',
        'freeze = json.loads((OUTPUT / "TARGET_BLIND_CANDIDATE_FREEZE.json").read_text(encoding="utf-8"))',
        'if freeze.get("status") != "TARGET_BLIND_CANDIDATES_FROZEN" or freeze.get("solutions_accessed") is not False: raise RuntimeError("target-blind freeze gate failed")',
        'SOLUTIONS = Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")',
        'if not SOLUTIONS.is_file(): raise RuntimeError("post-freeze Eval60 solutions file is missing")',
        'scorer_command = [sys.executable, str(SCORER), "--output", str(OUTPUT), "--solutions", str(SOLUTIONS), "--retention", str(RETENTION), "--dfs1024-overlap", str(BASELINE / "DFS1024_SOURCE_HIT_OVERLAP.csv"), "--dfs1024-cells", str(BASELINE / "DFS1024_DFS_CELLS.csv"), "--dfs1024-runtime", str(BASELINE / "DFS1024_RUNTIME_BASELINE.json"), "--dfs1024-checkpoints", str(BASELINE / "DFS1024_TASK_CHECKPOINTS"), "--notebook-start-monotonic", str(NOTEBOOK_START_MONOTONIC)]',
        'subprocess.run(scorer_command, check=True)',
        'for name in ("RETENTION30_OUTPUTS.json", "RETENTION30_COHORT_SHA256.txt", "HISTORICAL_GREEDY_SOURCE_CLASSES.csv"):',
        '    shutil.copy2(DATASET / name, OUTPUT / name)',
        f'required_outputs = {REQUIRED_OUTPUTS!r}',
        'missing_outputs = [name for name in required_outputs if not (OUTPUT / name).is_file()]',
        'if missing_outputs: raise RuntimeError(f"retention outputs missing: {missing_outputs}")',
        'decision = json.loads((OUTPUT / "DECISION.json").read_text(encoding="utf-8"))',
        'print(json.dumps({"event": "DFS2048_RETENTION30_NOTEBOOK_COMPLETE", "decision": decision, "output": str(OUTPUT), "notebook_total_wall_s": time.perf_counter() - NOTEBOOK_START_MONOTONIC}, sort_keys=True), flush=True)',
    ])


def configure_shared_builder() -> None:
    shared.EXPERIMENT_ID = EXPERIMENT_ID
    shared.EXPERIMENT = shared.ROOT / EXPERIMENT_RELATIVE
    shared.RUNNER_RELATIVE = RUNNER_RELATIVE
    shared.SCORER_RELATIVE = SCORER_RELATIVE
    shared.OUTPUT_FOLDER = OUTPUT_FOLDER
    shared.NOTEBOOK_TITLE = NOTEBOOK_TITLE
    shared.SOURCE_PATHS = SOURCE_PATHS
    shared.DATASET_COPY_NAMES = DATASET_COPY_NAMES
    shared.REQUIRED_OUTPUTS = REQUIRED_OUTPUTS
    shared.SOURCE_IDENTITY_EXTRA = SOURCE_IDENTITY_EXTRA
    shared.PACKAGE_EXTRA = PACKAGE_EXTRA
    shared.notebook_source = notebook_source
    shared.DEFAULT_DATASET_SLUG = "arc2-eval60-dual-ttt-dfs2048-retention30-v1-source"
    shared.DEFAULT_KERNEL_SLUG = "arc2-eval60-dual-ttt-dfs2048-retention30-v1"


if __name__ == "__main__":
    configure_shared_builder()
    shared.main()
