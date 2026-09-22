"""CPU/GPU-local DFS-only ablation against the frozen native V8 stack.

The task cohort is deliberately fixed from historical Frozen30 diagnostics.
Targets are not opened until every greedy and DFS prediction checkpoint has
been atomically persisted.  This script is experimental only: it never reads
or changes the V8 notebook, submission, or production checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from math import inf
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from inference.dynamic_task_scheduler import task_seed
from inference.native_multiview_likelihood import candidate_view_scores_many
from inference.nvarc_constrained_dfs import ConstrainedDFSConfig, constrained_token_dfs, decode_dfs_candidate
from inference.nvarc_native import NVARCNativeProvider, native_messages_from_training_prefix, native_training_message_prefix, parse_native_grid
from inference.nvarc_native_augmentation import NativeAugmentation, bounded_native_augmentations, transform_tasks_for_augmentations
from inference.nvarc_native_candidates import NativeGridCandidate, deduplicate_candidates, rank_candidates
from inference.nvarc_public_reference import PublicReferenceEvidence, grouped_public_reference_ranking, prediction_key, two_attempt_indices


# Historical labels chose this cohort before this DFS code existed.  They are
# selection provenance only; the labels are never passed to generation/ranking.
COHORT = (
    ("3c9b0459", "historical_top1_hit"),
    ("3f7978a0", "historical_pool_exists_top2_miss"),
    ("31aa019c", "historical_pool_miss"),
    ("996ec1f3", "historical_pool_miss"),
    ("cd3c21df", "historical_pool_miss"),
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _prediction(candidate: NativeGridCandidate) -> list[list[list[int]]]:
    return [[list(row) for row in grid] for grid in candidate.prediction]


def _candidate_from_grids(augmentation: NativeAugmentation, grids: list[list[list[int]]], *, tokens: int, seconds: float) -> NativeGridCandidate:
    return NativeGridCandidate(
        augmentation=augmentation,
        prediction=tuple(tuple(tuple(int(cell) for cell in row) for row in grid) for grid in grids),
        completion_tokens=tokens,
        generation_seconds=seconds,
    )


def _rank_b_support(provider: Any, task: Any, candidates: list[NativeGridCandidate], *, context_window: int) -> dict[str, Any]:
    if not candidates:
        return {"candidates": [], "likelihood_ranked_indices": [], "b_ranked_indices": [], "attempt_indices": [], "b_evidence": []}
    prefix = native_training_message_prefix(task)
    original_messages = [native_messages_from_training_prefix(prefix, item.input) for item in task.test]
    likelihood_ranked = rank_candidates(provider, candidates, original_messages, context_window=context_window, likelihood_batch_size=1)
    by_index = {candidates.index(candidate): float(score) for candidate, score in likelihood_ranked}
    views = tuple(NativeAugmentation(geometry=name) for name in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose"))
    serialized = [item.to_dict() for item in candidates]
    view_scores = candidate_view_scores_many(provider, task, [item["prediction"] for item in serialized], views, context_window=context_window, batch_size=1)
    evidence = [
        PublicReferenceEvidence(
            index=index,
            prediction_key=prediction_key(item["prediction"]),
            original_log_likelihood=by_index[index],
            view_negative_log_likelihoods=tuple(-float(score) for score in view_scores[index]),
            support_count=int(item["support_count"]),
        )
        for index, item in enumerate(serialized)
    ]
    b_ranked = grouped_public_reference_ranking(evidence)
    return {
        "candidates": serialized,
        "likelihood_ranked_indices": [candidates.index(candidate) for candidate, _score in likelihood_ranked],
        "b_ranked_indices": b_ranked,
        "attempt_indices": two_attempt_indices(b_ranked, serialized),
        "b_evidence": [
            {"candidate_index": item.index, "support_count": item.support_count, "original_log_likelihood": item.original_log_likelihood, "mean_view_nll": item.mean_view_nll}
            for item in evidence
        ],
    }


def _greedy(provider: Any, task: Any, augmentations: tuple[NativeAugmentation, ...], settings: dict[str, Any]) -> tuple[list[NativeGridCandidate], dict[str, Any]]:
    candidates: list[NativeGridCandidate] = []
    invalid = 0
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    for index, (augmentation, view) in enumerate(zip(augmentations, transformed, strict=True)):
        grids: list[list[list[int]]] = []
        total_tokens = 0
        total_seconds = 0.0
        valid = True
        for test_index, example in enumerate(view.test):
            message = native_messages_from_training_prefix(prefixes[index], example.input)
            generated = provider.generate(
                message,
                max_new_tokens=int(settings["decode"]["max_new_tokens"]),
                context_window=int(settings["decode"]["context_window"]),
                seed=task_seed(task.task_id, int(settings["decode"]["seed"]), f"augmentation:{index}:test:{test_index}"),
            )
            parsed = parse_native_grid(generated.text)
            total_tokens += generated.completion_tokens
            total_seconds += generated.elapsed_seconds
            if parsed is None:
                valid = False
                continue
            grids.append(augmentation.inverse_grid(parsed))
        if valid and len(grids) == len(task.test):
            candidates.append(_candidate_from_grids(augmentation, grids, tokens=total_tokens, seconds=total_seconds))
        else:
            invalid += 1
    return candidates, {"raw_generation_count": len(augmentations), "invalid_generation_count": invalid}


def _dfs(provider: Any, task: Any, augmentations: tuple[NativeAugmentation, ...], settings: dict[str, Any], config: ConstrainedDFSConfig) -> tuple[list[NativeGridCandidate], dict[str, Any]]:
    # Multi-test tasks need one grid per test input.  We retain only the
    # deterministic same-rank Cartesian alignment, avoiding an unconstrained
    # cross-product of test outputs.
    candidates: list[NativeGridCandidate] = []
    raw: list[dict[str, Any]] = []
    transformed = transform_tasks_for_augmentations(task, augmentations)
    prefixes = [native_training_message_prefix(item) for item in transformed]
    for augmentation_index, (augmentation, view) in enumerate(zip(augmentations, transformed, strict=True)):
        per_test: list[list[tuple[list[list[int]], Any]]] = []
        for test_index, example in enumerate(view.test):
            message = native_messages_from_training_prefix(prefixes[augmentation_index], example.input)
            result = constrained_token_dfs(provider, message, config)
            valid: list[tuple[list[list[int]], Any]] = []
            for item in result.candidates:
                parsed = parse_native_grid(decode_dfs_candidate(provider, item))
                if parsed is not None:
                    valid.append((augmentation.inverse_grid(parsed), item))
            raw.append({
                "augmentation_index": augmentation_index,
                "test_index": test_index,
                "terminal_candidates": len(result.candidates),
                "valid_candidates": len(valid),
                "expanded_nodes": result.expanded_nodes,
                "pruned_probability": result.pruned_probability,
                "pruned_grammar": result.pruned_grammar,
                "timed_out": result.timed_out,
            })
            per_test.append(valid)
        aligned = min((len(items) for items in per_test), default=0)
        for rank in range(aligned):
            grids = [items[rank][0] for items in per_test]
            metas = [items[rank][1] for items in per_test]
            candidates.append(_candidate_from_grids(
                augmentation, grids,
                tokens=sum(len(item.token_ids) for item in metas),
                seconds=max(item.elapsed_seconds for item in metas),
            ))
    return candidates, {"raw_generation_count": len(raw), "dfs_transport": raw}


def _selected_predictions(result: dict[str, Any]) -> list[list[list[list[int]]]]:
    candidates = result["candidates"]
    indices = result["attempt_indices"]
    return [candidates[index]["prediction"] for index in indices]


def _score(result: dict[str, Any], target: list[list[list[int]]]) -> dict[str, Any]:
    candidates = result["candidates"]
    exact_indices = [index for index, candidate in enumerate(candidates) if candidate["prediction"] == target]
    ranks = {index: position + 1 for position, index in enumerate(result["b_ranked_indices"])}
    chosen = _selected_predictions(result)
    return {
        "any_of_k": bool(exact_indices),
        "correct_candidate_indices": exact_indices,
        "correct_candidate_rank": min((ranks[index] for index in exact_indices), default=None),
        "top1": bool(chosen and chosen[0] == target),
        "top2": any(item == target for item in chosen[:2]),
    }


def _markdown(summary: dict[str, Any]) -> str:
    return "\n".join((
        "# Local DFS-only candidate-search ablation", "",
        f"TASKS_TESTED = {summary['task_count']}",
        f"GREEDY_ANY_OF_K = {summary['greedy']['any_of_k']}/{summary['task_count']}",
        f"DFS_ANY_OF_K = {summary['dfs']['any_of_k']}/{summary['task_count']}",
        f"GREEDY_TOP2 = {summary['greedy']['top2']}/{summary['task_count']}",
        f"DFS_TOP2 = {summary['dfs']['top2']}/{summary['task_count']}",
        f"MEAN_GREEDY_CANDIDATES = {summary['greedy']['mean_unique_candidates']:.2f}",
        f"MEAN_DFS_CANDIDATES = {summary['dfs']['mean_unique_candidates']:.2f}",
        f"MEAN_RUNTIME_RATIO = {summary['mean_runtime_ratio']:.2f}x",
        f"PEAK_VRAM_GB = {summary['peak_vram_gb']:.2f}", "",
        f"Decision: {summary['decision']}",
    )) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge-path", type=Path, default=ROOT / "tmp_speed_v2_scoring" / "arc-agi_training_challenges.json")
    parser.add_argument("--solution-path", type=Path, default=ROOT / "tmp_speed_v2_scoring" / "arc-agi_training_solutions.json")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json")
    parser.add_argument("--native-config-dir", type=Path, default=ROOT / "configs" / "nvarc_native_846d0198")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "local_dfs_ablation")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-candidates-view", type=int, default=4)
    parser.add_argument("--max-cumulative-nll", type=float, default=1.6094379124341003)
    parser.add_argument("--max-wall-seconds-view", type=float, default=90.0)
    args = parser.parse_args()
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"missing local Qwen3-4B checkpoint: {args.model_path}")
    for path in (args.challenge_path, args.solution_path, args.config, args.native_config_dir):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.output_dir.exists() and not args.resume and any(args.output_dir.iterdir()):
        raise FileExistsError("output exists; pass --resume to reuse valid task checkpoints")

    from arc.io import load_dataset

    config = json.loads(args.config.read_text(encoding="utf-8"))
    settings = config["B_augmentation_search"]
    tasks = load_dataset(args.challenge_path)
    task_ids = tuple(task_id for task_id, _label in COHORT)
    if any(task_id not in tasks for task_id in task_ids):
        raise ValueError("fixed DFS cohort is missing from the challenge file")
    identity = {
        "experiment": "ARC2_LOCAL_ABLATION_DFS_ONLY",
        "task_ids": task_ids,
        "cohort_labels": dict(COHORT),
        "generation_micro_batch_size": 1,
        "dfs": {"max_candidates_view": args.max_candidates_view, "max_cumulative_nll": args.max_cumulative_nll, "max_wall_seconds_view": args.max_wall_seconds_view, "max_new_tokens": int(settings["decode"]["max_new_tokens"])},
        "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
    }
    identity["identity_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    _atomic_json(args.output_dir / "cohort_manifest.json", identity)
    augmentations = bounded_native_augmentations(color_offsets=tuple(settings["color_offsets"]), pair_orders=tuple(settings["train_pair_orders"]))[:32]
    dfs_config = ConstrainedDFSConfig(
        max_candidates=args.max_candidates_view,
        max_new_tokens=int(settings["decode"]["max_new_tokens"]),
        max_cumulative_nll=args.max_cumulative_nll,
        max_wall_seconds=args.max_wall_seconds_view,
        context_window=int(settings["decode"]["context_window"]),
    )
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config_dir, device="cuda:0")
    provider.load()
    records: dict[str, Any] = {}
    for position, task_id in enumerate(task_ids, 1):
        checkpoint = args.output_dir / "checkpoints" / f"{task_id}.json"
        if args.resume and checkpoint.exists():
            saved = json.loads(checkpoint.read_text(encoding="utf-8"))
            if saved.get("identity_sha256") == identity["identity_sha256"] and "greedy" in saved and "dfs" in saved:
                records[task_id] = saved
                print(json.dumps({"event": "DFS_TASK_RESUMED", "task": f"{position}/{len(task_ids)}", "task_id": task_id}), flush=True)
                continue
        import torch
        torch.cuda.reset_peak_memory_stats()
        task = tasks[task_id]
        started = time.perf_counter()
        greedy_raw, greedy_meta = _greedy(provider, task, augmentations, settings)
        greedy_unique = deduplicate_candidates(greedy_raw)
        greedy = _rank_b_support(provider, task, greedy_unique, context_window=int(settings["decode"]["context_window"]))
        greedy.update(greedy_meta)
        greedy["unique_candidate_count"] = len(greedy_unique)
        greedy["runtime_seconds"] = time.perf_counter() - started
        dfs_started = time.perf_counter()
        dfs_raw, dfs_meta = _dfs(provider, task, augmentations, settings, dfs_config)
        dfs_unique = deduplicate_candidates(dfs_raw)
        dfs = _rank_b_support(provider, task, dfs_unique, context_window=int(settings["decode"]["context_window"]))
        dfs.update(dfs_meta)
        dfs["unique_candidate_count"] = len(dfs_unique)
        dfs["runtime_seconds"] = time.perf_counter() - dfs_started
        record = {"identity_sha256": identity["identity_sha256"], "task_id": task_id, "status": "PREDICTIONS_FROZEN_BEFORE_SCORING", "greedy": greedy, "dfs": dfs, "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3}
        _atomic_json(checkpoint, record)
        records[task_id] = record
        print(json.dumps({"event": "DFS_TASK_FROZEN", "task": f"{position}/{len(task_ids)}", "task_id": task_id, "greedy_unique": len(greedy_unique), "dfs_unique": len(dfs_unique), "elapsed_seconds": time.perf_counter() - started}), flush=True)
    # This is the freeze barrier: persist every prediction before loading targets.
    frozen = {"status": "ALL_LOCAL_DFS_PREDICTIONS_FROZEN_BEFORE_SCORING", "identity": identity, "records": records}
    _atomic_json(args.output_dir / "per_task.json", frozen)
    del provider
    import gc
    import torch
    gc.collect(); torch.cuda.empty_cache()

    solutions = json.loads(args.solution_path.read_text(encoding="utf-8"))
    if set(task_ids) - set(solutions):
        raise ValueError("ground-truth solutions missing for frozen DFS cohort")
    for task_id, record in records.items():
        record["scored_after_freeze"] = {"greedy": _score(record["greedy"], solutions[task_id]), "dfs": _score(record["dfs"], solutions[task_id])}
    methods = {name: [record["scored_after_freeze"][name] for record in records.values()] for name in ("greedy", "dfs")}
    def aggregate(name: str) -> dict[str, Any]:
        items = methods[name]
        return {"any_of_k": sum(item["any_of_k"] for item in items), "top1": sum(item["top1"] for item in items), "top2": sum(item["top2"] for item in items), "mean_unique_candidates": sum(record[name]["unique_candidate_count"] for record in records.values()) / len(records)}
    greedy_runtime = sum(record["greedy"]["runtime_seconds"] for record in records.values())
    dfs_runtime = sum(record["dfs"]["runtime_seconds"] for record in records.values())
    summary = {"status": "LOCAL_DFS_ABLATION_SCORED_AFTER_FREEZE", "task_count": len(records), "identity": identity, "greedy": aggregate("greedy"), "dfs": aggregate("dfs"), "mean_runtime_ratio": dfs_runtime / greedy_runtime if greedy_runtime else inf, "peak_vram_gb": max(record["peak_vram_gb"] for record in records.values()), "task_results": records}
    summary["decision"] = "DFS_CANDIDATE_GENERATION_BOTTLENECK_CONFIRMED" if summary["dfs"]["any_of_k"] > summary["greedy"]["any_of_k"] else "DFS_NO_RECALL_GAIN_STOP_BEFORE_TTT"
    _atomic_json(args.output_dir / "per_task.json", {**frozen, "records": records})
    _atomic_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "summary.md").write_text(_markdown(summary), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in ("task_count", "greedy", "dfs", "mean_runtime_ratio", "peak_vram_gb", "decision")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
