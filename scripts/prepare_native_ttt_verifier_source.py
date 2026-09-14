"""Build private, solution-free source for native verifier/TTT stages."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_FILES = (
    "pyproject.toml", "requirements.txt", "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json", "configs/ARC2_NATIVE_TTT_AND_STRONG_VERIFIER_V1.json", "configs/NVARC_NATIVE_INTERFACE_846D0198_PROVENANCE.json",
    "configs/ARC2_NATIVE_LIKELIHOOD_RANKER_V2_FAST.json", "scripts/run_qwen4b_native_augmentation_search.py", "scripts/run_native_ttt_memory_smoke.py", "scripts/rerank_native_multiview_likelihood.py", "scripts/run_dual_reasoning_smoke.py", "scripts/run_dual_model_arc_v1.py", "scripts/run_soar_numpy_sandbox_calibration.py", "scripts/run_frozen30_native_soar_complementarity.py", "scripts/run_frozen30_soar_four_workers.py", "scripts/run_soar_single_gpu_parallel_smoke.py", "src/arc", "src/inference/__init__.py", "src/inference/arc_native_io.py", "src/inference/nvarc_native.py", "src/inference/nvarc_native_augmentation.py", "src/inference/nvarc_native_candidates.py", "src/inference/native_ranker.py", "src/inference/native_multiview_likelihood.py", "src/inference/dual_reasoning_smoke.py", "src/inference/llama_cpp_backend.py", "src/inference/nvarc_native_ttt.py", "src/inference/kaggle_l4_parallel_runner.py", "src/inference/qwen3_transformers_parallel_runner.py",
)


def _copy(relative: str, project: Path) -> None:
    source, destination = ROOT / relative, project / relative; destination.parent.mkdir(parents=True, exist_ok=True)
    if source.is_dir(): shutil.copytree(source, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    else: shutil.copy2(source, destination)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", required=True); parser.add_argument("--dataset-slug", default="arc2-native-ttt-verifier-source"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite staging directory: {args.output}")
    project = args.output / "dataset" / "ARC2"
    for relative in _FILES: _copy(relative, project)
    frozen = project / "frozen_inputs"; frozen.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / "artifacts/native_multiview_frozen30_pool_v3_stage/dataset/ARC2_QWEN4B_NATIVE_CAPABILITY_PUSH_BC_PREDICTIONS_FROZEN.json", frozen / "native_frozen30.json")
    shutil.copy2(ROOT / "artifacts/soar_numpy_sandbox_calibration_v21/artifacts/soar_numpy_sandbox_calibration.json", frozen / "sandbox_calibration.json")
    resume = frozen / "resume_tasks"; resume.mkdir()
    for task_id in ("228f6490", "2bcee788"):
        shutil.copy2(ROOT / f"artifacts/frozen30_checkpoint_v27/artifacts/frozen30_native_soar_v1/generation/tasks/{task_id}.json", resume / f"{task_id}.json")
    shutil.copytree(ROOT / "configs/nvarc_native_846d0198", project / "configs/nvarc_native_846d0198")
    if list(project.rglob("*solutions*.json")): raise RuntimeError("solution-bearing files are forbidden from TTT inference source")
    forbidden = ("v3", "executor", "verifier.py", "compiler", "rulespec", "oracle", "score_")
    bad = [str(path.relative_to(project)) for path in project.rglob("*") if any(term in str(path.relative_to(project)).lower() for term in forbidden)]
    if bad: raise RuntimeError(f"native TTT source is not isolated: {bad}")
    dataset = args.output / "dataset"; shutil.make_archive(str(dataset / "ARC2"), "gztar", root_dir=dataset, base_dir="ARC2"); shutil.rmtree(project)
    metadata = {"title": "ARC2 Native TTT Verifier Source", "subtitle": "Private native TTT generation; no solutions", "description": "Source for train-pair-only native LoRA/TTT candidate generation.", "id": f"{args.owner}/{args.dataset_slug}", "licenses": [{"name": "other"}]}
    (dataset / "dataset-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": metadata["id"], "contains_solutions": False, "forbidden_files": 0}))


if __name__ == "__main__": main()
