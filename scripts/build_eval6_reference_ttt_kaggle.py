"""Stage the independent Eval6 public-reference-style TTT pilot."""
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


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(ids), separators=(",", ":")).encode()).hexdigest()


def _commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _archive(destination: Path, commit: str) -> None:
    archive = destination.parent / "source.tar"
    with archive.open("wb") as handle: subprocess.run(["git", "archive", "--format=tar", commit], cwd=ROOT, stdout=handle, check=True)
    with tarfile.open(archive) as handle: handle.extractall(destination, filter="data")
    archive.unlink()


def _frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    parent, s0 = _manifest_and_s0()
    ids = sorted(parent["task_ids"], key=lambda item: (hashlib.sha256(item.encode()).hexdigest(), item))[:6]
    manifest = {"experiment_id": "ARC2_EVAL6_REFERENCE_TTT", "status": "EVAL6_REFERENCE_TTT_COHORT_FROZEN", "selection_rule": "six smallest SHA256(task_id) values from immutable Eval30 Aug8 greedy pool-miss cohort", "selection_provenance": "Parent Eval30 was frozen from historic Evaluation60 pool misses; source contains no evaluation solutions.", "task_ids": ids, "task_ids_hash": _hash_ids(ids), "source_challenge_sha256": parent["source_challenge_sha256"], "baseline_any_of_k": "0/6 by immutable parent Eval30 S0 definition", "integrity": {"targets_inspected_during_ttt_or_generation": False, "no_dfs": True, "no_beam2": True, "no_b_support_change": True}}
    baseline = {"status": "EVAL6_REUSED_AUG8_GREEDY_BASELINE_FROZEN", "task_ids": ids, "task_ids_hash": manifest["task_ids_hash"], "records": {task_id: s0["records"][task_id] for task_id in ids}}
    config = {"style": "public_nvarc_documented_ttt", "reference_repository": "https://github.com/1ytic/NVARC", "reference_commit": "846d0198efa752534594e321fc3289fc0a06c657", "reference_notebook": "sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue", "rank": 256, "alpha": 32, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"], "learning_rate": 5e-5, "scheduler": "cosine", "warmup_ratio": 0.1, "gradient_accumulation_steps": 1, "epochs": 1, "bf16": True, "gradient_checkpointing": False, "augmentation_geometry_count": 8, "augmentation_colour_permutations": 16, "train_pair_order": "random full permutation", "max_sequence_length": 8192, "generation_augmentation_count": 8, "generation": "unchanged greedy Aug8", "max_new_tokens": 1024, "generation_context_window": 16384, "seed": 42, "maximum_expected_task_seconds": 1200}
    return manifest, baseline, config


def _notebook(dataset_slug: str) -> str:
    return "\n".join([
        "import json, os, shutil, subprocess, sys", "from pathlib import Path", "",
        'inputs=Path("/kaggle/input")', 'root=next(inputs.rglob("run_eval6_reference_ttt.py")).parents[1]', 'frozen=next(inputs.rglob("eval6_manifest.json")).parent',
        'challenge=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")', 'solutions=Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")', 'model=Path("/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1")', 'out=Path("/kaggle/working/artifacts/eval6_reference_ttt")',
        'gpus=subprocess.check_output(["nvidia-smi","-L"],text=True).splitlines()', 'if len(gpus)!=4 or any("NVIDIA L4" not in x for x in gpus): raise RuntimeError(f"requires 4 L4: {gpus}")', 'if os.environ.get("KAGGLE_KERNEL_INTERNET_ENABLED", "").strip().lower() in {"1","true","yes"}: raise RuntimeError("Internet must be off")', 'out.mkdir(parents=True,exist_ok=True)',
        'for name in ("eval6_manifest.json","reference_ttt_config_frozen.json","s0_greedy_reused_candidates_frozen.json"): shutil.copy2(frozen/name,out/name)',
        f'print(json.dumps({{"event":"EVAL6_REFERENCE_TTT_TARGET_BLIND_START","source_dataset":"{dataset_slug}","gpus":gpus,"solutions_opened":False}},sort_keys=True),flush=True)',
        'cmd=[sys.executable,str(root/"scripts/run_eval6_reference_ttt.py"),"--manifest",str(out/"eval6_manifest.json"),"--reference-config",str(out/"reference_ttt_config_frozen.json"),"--challenge-path",str(challenge),"--model-path",str(model),"--native-config-dir",str(root/"configs/nvarc_native_846d0198"),"--output",str(out/"reference_ttt_candidates_frozen.json"),"--checkpoint-dir",str(out/"checkpoints"),"--resume"]',
        'if subprocess.run(cmd,env={**os.environ,"CUDA_VISIBLE_DEVICES":"0,1,2,3"}).returncode: raise RuntimeError("reference-style Eval6 TTT failed")',
        'if not (out/"reference_ttt_candidates_frozen.json").is_file(): raise RuntimeError("candidate freeze missing; refusing target access")', 'print(json.dumps({"event":"EVAL6_REFERENCE_TTT_CANDIDATES_FROZEN","solutions_opened":False},sort_keys=True),flush=True)',
        'score=[sys.executable,str(root/"scripts/score_eval6_reference_ttt.py"),"--manifest",str(out/"eval6_manifest.json"),"--reference-config",str(out/"reference_ttt_config_frozen.json"),"--baseline",str(out/"s0_greedy_reused_candidates_frozen.json"),"--candidates",str(out/"reference_ttt_candidates_frozen.json"),"--challenge-path",str(challenge),"--solutions-path",str(solutions),"--output-dir",str(out)]',
        'if subprocess.run(score,env={**os.environ,"CUDA_VISIBLE_DEVICES":""}).returncode: raise RuntimeError("post-freeze Eval6 score failed")', 'print(json.dumps({"event":"EVAL6_REFERENCE_TTT_COMPLETE","output":str(out)},sort_keys=True),flush=True)',
    ])


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--output", type=Path, required=True); parser.add_argument("--owner", default="jimmy5566"); parser.add_argument("--dataset-slug", default="arc2-eval6-reference-ttt-source"); parser.add_argument("--kernel-slug", default="arc2-eval6-reference-ttt-diagnostic"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite Eval6 reference TTT staging")
    manifest, baseline, config = _frozen_inputs(); commit = _commit(); dataset = args.output / "dataset" / "ARC2"; dataset.mkdir(parents=True); _archive(dataset, commit)
    frozen = dataset / "eval6_reference_ttt"; _write(frozen / "eval6_manifest.json", manifest); _write(frozen / "reference_ttt_config_frozen.json", config); _write(frozen / "s0_greedy_reused_candidates_frozen.json", baseline)
    if list(dataset.rglob("*solutions*.json")): raise RuntimeError("target-blind source dataset contains solutions")
    _write(args.output / "dataset" / "dataset-metadata.json", {"title":"ARC2 Eval6 Reference TTT Source","subtitle":"Target-blind public-reference-style TTT diagnostic","description":"Private source attachment without ARC evaluation solutions.","id":f"{args.owner}/{args.dataset_slug}","licenses":[{"name":"other"}]})
    kernel=args.output/"kernel"; kernel.mkdir(); name=f"{args.kernel_slug}.ipynb"; _write(kernel/name,{"cells":[{"cell_type":"code","execution_count":None,"metadata":{},"outputs":[],"source":[line+"\n" for line in _notebook(f"{args.owner}/{args.dataset_slug}").splitlines()]}],"metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},"language_info":{"name":"python","version":"3.12"},"kaggle":{"accelerator":"nvidiaL4","isGpuEnabled":True,"isInternetEnabled":False,"language":"python","sourceType":"notebook"}},"nbformat":4,"nbformat_minor":4})
    _write(kernel/"kernel-metadata.json", {"id":f"{args.owner}/{args.kernel_slug}","title":"ARC2 Eval6 Reference TTT Diagnostic","code_file":name,"language":"python","kernel_type":"notebook","is_private":True,"enable_gpu":True,"enable_tpu":False,"enable_internet":False,"keywords":["gpu"],"dataset_sources":[f"{args.owner}/{args.dataset_slug}"],"kernel_sources":[],"competition_sources":["arc-prize-2026-arc-agi-2"],"model_sources":["sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1"],"machine_shape":"NvidiaL4"})
    print(json.dumps({"status":"EVAL6_REFERENCE_TTT_STAGING_READY","source_commit":commit,"manifest":manifest,"config":config},sort_keys=True))


if __name__ == "__main__": main()
