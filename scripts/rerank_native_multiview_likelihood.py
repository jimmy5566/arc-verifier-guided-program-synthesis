"""Re-rank an immutable native candidate artifact without generating grids."""
from __future__ import annotations
import argparse, copy, hashlib, json, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT / "src"))
from arc.io import load_dataset
from inference.nvarc_native import NVARCNativeProvider
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.native_multiview_likelihood import aggregate, calibrated, candidate_view_scores, loo_view_weights, ranks

FROZEN = "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING"

def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("frozen", "challenge_path", "model_path", "native_config_dir", "config", "output"): parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--device", default="cuda:0"); args = parser.parse_args()
    if args.output.exists(): raise FileExistsError("refusing to overwrite reranked artifact")
    frozen = json.loads(args.frozen.read_text(encoding="utf-8")); records = frozen.get("records", {})
    if frozen.get("status") != FROZEN or not records: raise ValueError("requires candidate artifact frozen before exact scoring")
    config = json.loads(args.config.read_text(encoding="utf-8")); view_config = config["views"]
    views = tuple(NativeAugmentation(geometry, int(view_config["color_offset"]), str(view_config["pair_order"])) for geometry in view_config["geometries"])
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device=args.device); provider.load()
    tasks = load_dataset(args.challenge_path); result = copy.deepcopy(frozen); started = time.perf_counter()
    for position, (task_id, record) in enumerate(sorted(result["records"].items()), 1):
        task = tasks[task_id]; weights, loo_scores = loo_view_weights(provider, task, views, context_window=8192)
        old = dict(zip(record["ranked_candidate_indices"], record["candidate_scores"], strict=True)); original = [float(old[index]) for index in range(len(record["candidates"]))]
        view_scores = [candidate_view_scores(provider, task, candidate["prediction"], views, context_window=8192) for candidate in record["candidates"]]
        methods = {"original_likelihood": original}
        for method in ("mean", "median", "trimmed_mean"): methods["multiview_" + method] = [aggregate(item, method) for item in view_scores]
        methods["calibrated_likelihood"] = [calibrated(item, weights) for item in view_scores]
        record["multiview_likelihood"] = {"views": [view.to_dict() for view in views], "loo_view_scores": loo_scores, "loo_weights": weights, "candidate_view_scores": view_scores, "methods": methods, "rankings": ranks(methods)}
        print(json.dumps({"event":"MULTIVIEW_RERANKED", "task_id":task_id, "task":f"{position}/{len(records)}", "candidate_count":len(view_scores), "rank_changed":methods["original_likelihood"] != methods["calibrated_likelihood"]}, sort_keys=True), flush=True)
    result["status"] = "MULTIVIEW_LIKELIHOOD_RERANKED_FROZEN_BEFORE_EXACT_SCORING"; result["multiview_source_frozen_sha256"] = hashlib.sha256(args.frozen.read_bytes()).hexdigest(); result["multiview_runtime_seconds"] = time.perf_counter()-started
    result["multiview_protocol"] = "teacher-forced candidate scoring under fixed reversible views; view calibration is leave-one-train-pair-out only; no generation, solutions, or task-specific rules"
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(json.dumps(result, indent=2, sort_keys=True)+"\n", encoding="utf-8")
if __name__ == "__main__": main()
