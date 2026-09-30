#!/usr/bin/env python3
"""REGRET_DYNAMIC_READY_BATCH_V1: contaminated, target-blind execution study.

The sole cohort is the four historically exposed d59b0160 depth-24 views.
``run`` never opens Gold.  ``score`` refuses to operate unless a target-blind
freeze plus its SHA256 manifest already exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, inference_d1_turbo_dfs
from inference.nvarc_turbodfs_dynamic_ready import (
    normalized_result_signature, ready_result, run_ready_scheduler, start_ready_cell,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file


EXPERIMENT = "REGRET_DYNAMIC_READY_BATCH_V1"
TASK_ID = "d59b0160"
OUTPUT_INDEX = 0
DEPTH = 24
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
ADAPTER_SHA = "a91e4375da1cf5f544a083221292ded0dd41e50b4a4c63e80bb4c37ab0e42854"


def json_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def decoder(caps: dict[str, Any]) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        "CUMULATIVE_REGRET_r=4.00", int(caps["max_new_tokens"]), float(caps["max_score"]), None,
        int(caps["max_expanded_nodes"]), int(caps["max_completed_candidates"]),
        frontier_floor=int(caps["frontier_floor"]), local_time_limit_seconds=float(caps["local_time_limit_seconds"]),
        pad_token_id=int(caps["pad_token_id"]), arc_tokens=tuple(int(value) for value in caps["arc_tokens"]),
        diagnostic_trace=True,
    )


def _required_caps(caps: dict[str, Any]) -> None:
    expected = {"max_expanded_nodes": 4096, "max_completed_candidates": 32, "frontier_floor": 1}
    missing = set(expected) - set(caps)
    if missing or any(int(caps[key]) != value for key, value in expected.items()):
        raise RuntimeError(f"dynamic-ready contract differs from fixed Regret4 budget: {caps}")


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite existing run {output}")
    challenge = args.challenge.resolve(); no_gold_challenge(challenge)
    source_contract = read_json(args.fixed_budget_contract.resolve())
    caps = source_contract.get("caps", {})
    _required_caps(caps)
    contract = {
        "experiment": EXPERIMENT, "execution_regime": "NEW_DECODER_EXECUTION_REGIME",
        "source_commit": args.source_commit, "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "reference_config": str(args.reference_config.resolve()), "authoritative_root": str(args.authoritative_root.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()),
        "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()), "caps": caps,
        "policy": "CUMULATIVE_REGRET_r=4.00", "candidate_cap": 32, "node_cap": 4096,
        "scheduler": {"model_instances": 1, "adapter_groups_at_once": 1, "max_physical_batch": 2,
                      "prefill": "individual_scalar_B1", "padding": "FORBIDDEN",
                      "compatible_only": ["adapter_sha256", "cache_geometry", "position", "dtype", "device"],
                      "ready_order": "cell_key_then_request_ordinal"},
        "solutions_accessed": False,
    }
    cohort = {"task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH, "views": list(VIEWS),
              "selection_scope": "HISTORICALLY_CONTAMINATED_DEVELOPMENT_ONLY",
              "known_postfreeze_canaries": {"identity": 2999, "flip_ud": 3328, "transpose": 3137},
              "evidence": "analysis/decoder_conditioned_cell_routing_v1/decoder_cell_labels.csv rows 930-933",
              "solutions_accessed": False}
    adapters = adapter_records(Path(contract["adapter_manifest"]))
    mapping = adapters.get((TASK_ID, DEPTH))
    if mapping is None:
        raise RuntimeError("authoritative d59b0160 d24 adapter is absent")
    output.mkdir(parents=True)
    atomic_json(output / "DYNAMIC_READY_CONTRACT.json", contract)
    atomic_json(output / "COHORT_PROVENANCE.json", cohort)
    atomic_json(output / "TARGET_BLIND_PREPARED.json", {"contract_sha256": json_sha(contract),
                                                          "cohort_sha256": json_sha(cohort), "solutions_accessed": False})


def _stable_result(result: Any) -> dict[str, Any]:
    return {
        "signature": normalized_result_signature(result),
        "candidate_tokens": [list(item.token_ids) for item in result.candidates[0]],
        "nodes": result.nodes, "branch_probabilities": result.branch_probabilities,
        "frontier_floor_events": result.frontier_floor_events, "search_trace": result.search_trace,
        "model_forwards": result.model_forwards, "tokens_advanced": result.tokens_advanced,
        "nodes_expanded": sum(item.get("state") == "expanded" for item in result.nodes),
        "successors_considered": sum(item.get("state") in {"expanded", "completed", "pruned"} for item in result.nodes),
        "successors_retained": sum(item.get("state") in {"expanded", "completed"} for item in result.nodes),
        "termination_reason": result.termination_reason, "budget_exhausted": result.budget_exhausted,
    }


def _record(*, result: Any, encoded: Any, augmentation: Any, tokenizer: Any, cell_key: str,
            adapter_sha: str, mode: str, runtime_seconds: float, scheduler: dict[str, Any]) -> dict[str, Any]:
    from inference.nvarc_native import parse_native_grid

    candidates = []
    for local_id, candidate in enumerate(result.candidates[0]):
        raw = parse_native_grid(tokenizer.decode(list(candidate.token_ids), skip_special_tokens=True))
        canonical = None if raw is None else augmentation.inverse_grid(raw)
        candidates.append({"candidate_id": candidate.candidate_id, "cell_candidate_id": local_id,
                           "candidate_token_ids": list(candidate.token_ids), "raw_transformed_grid": raw,
                           "canonical_candidate": canonical, "valid_grid": canonical is not None,
                           "cumulative_nll": candidate.cumulative_nll,
                           "candidate_discovery_order": candidate.candidate_id,
                           "terminal_node_id": candidate.terminal_node_id,
                           "forward_count_at_discovery": candidate.discovery_forward_index})
    stable = _stable_result(result)
    return {"cell_key": cell_key, "mode": mode, "checkpoint_sha256": adapter_sha,
            "prompt_sha256": hashlib.sha256(encoded.detach().cpu().contiguous().numpy().tobytes()).hexdigest(),
            "prompt_tokens": int(encoded.shape[-1]), "runtime_seconds": runtime_seconds,
            "candidate_count": len(candidates), "valid_grid_count": sum(item["valid_grid"] for item in candidates),
            "unique_grid_count": len({json.dumps(item["canonical_candidate"], separators=(",", ":")) for item in candidates if item["valid_grid"]}),
            "candidates": candidates, **stable, "scheduler": scheduler, "solutions_accessed": False}


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "DYNAMIC_READY_CONTRACT.json")
    no_gold_challenge(Path(contract["challenge"])); _required_caps(contract["caps"])
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel

    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
                                   reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
                                   native_config_dir=Path(contract["native_config_dir"]), gpu_id=args.gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapter_sha = ""
    try:
        adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
        if adapter_sha != ADAPTER_SHA:
            raise RuntimeError(f"unexpected d59 depth24 adapter {adapter_sha}")
        task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
        encoded_views = [(encoded["input_ids"], augmentation) for encoded, augmentation in
                         (common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config)
                          for view in VIEWS)]
        if any(int(encoded.shape[0]) != 1 for encoded, _aug in encoded_views):
            raise RuntimeError("dynamic-ready requires individual B1 view prompts")
        native = assert_native_token_contract(tokenizer)
        dec = decoder(contract["caps"]); FastLanguageModel.for_inference(model)

        scalar_results: dict[str, Any] = {}
        b1_results: dict[str, Any] = {}
        scalar_rows: list[dict[str, Any]] = []
        b1_rows: list[dict[str, Any]] = []
        for view, (encoded, augmentation) in zip(VIEWS, encoded_views, strict=True):
            key = f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}"
            started = time.perf_counter()
            scalar = inference_d1_turbo_dfs(model, input_ids=encoded.to(model.device), config=dec)
            elapsed = time.perf_counter() - started
            scalar_results[key] = scalar
            scalar_rows.append(_record(result=scalar, encoded=encoded, augmentation=augmentation, tokenizer=tokenizer,
                                       cell_key=key, adapter_sha=adapter_sha, mode="AUTHORITATIVE_SCALAR",
                                       runtime_seconds=elapsed, scheduler={"physical_batch": 1, "prefill_scalar": True}))
        b1_cells = [start_ready_cell(model=model, input_ids=encoded.to(model.device), config=dec,
                                     cell_key=f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}", normalize_root_cache=False)
                    for view, (encoded, _augmentation) in zip(VIEWS, encoded_views, strict=True)]
        b1_started = time.perf_counter(); b1_sched = run_ready_scheduler(model=model, cells=b1_cells, dynamic_batch2=False)
        b1_elapsed = time.perf_counter() - b1_started
        parity: list[dict[str, Any]] = []
        for view, cell, (encoded, augmentation) in zip(VIEWS, b1_cells, encoded_views, strict=True):
            key = cell.cell_key; refactor = ready_result(cell); b1_results[key] = refactor
            exact = normalized_result_signature(scalar_results[key]) == normalized_result_signature(refactor)
            parity.append({"cell_key": key, "authoritative_signature": normalized_result_signature(scalar_results[key]),
                           "refactor_signature": normalized_result_signature(refactor), "exact_parity": exact,
                           "authoritative_nodes": _stable_result(scalar_results[key])["nodes_expanded"],
                           "refactor_nodes": _stable_result(refactor)["nodes_expanded"]})
            b1_rows.append(_record(result=refactor, encoded=encoded, augmentation=augmentation, tokenizer=tokenizer,
                                   cell_key=key, adapter_sha=adapter_sha, mode="READY_REFACTOR_B1",
                                   runtime_seconds=b1_elapsed / len(b1_cells), scheduler=b1_sched))
        write_csv(output / "MICROBATCH1_REFACTOR_PARITY.csv", parity)
        atomic_json(output / "raw_scalar.json", scalar_rows); atomic_json(output / "raw_refactor_b1.json", b1_rows)
        if not all(row["exact_parity"] for row in parity):
            atomic_json(output / "DECISION.json", {"status": "STOP", "reason": "B1_PARITY_FAIL", "solutions_accessed": False})
            raise RuntimeError("B1 refactor state machine parity failed; dynamic B2 is prohibited")

        dynamic_cells = [start_ready_cell(model=model, input_ids=encoded.to(model.device), config=dec,
                                          cell_key=f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}", normalize_root_cache=True)
                         for view, (encoded, _augmentation) in zip(VIEWS, encoded_views, strict=True)]
        dynamic_started = time.perf_counter(); dynamic_sched = run_ready_scheduler(model=model, cells=dynamic_cells, dynamic_batch2=True)
        dynamic_elapsed = time.perf_counter() - dynamic_started
        dynamic_rows = []
        for view, cell, (encoded, augmentation) in zip(VIEWS, dynamic_cells, encoded_views, strict=True):
            dynamic_rows.append(_record(result=ready_result(cell), encoded=encoded, augmentation=augmentation, tokenizer=tokenizer,
                                        cell_key=cell.cell_key, adapter_sha=adapter_sha, mode="DYNAMIC_READY_B2",
                                        runtime_seconds=dynamic_elapsed / len(dynamic_cells), scheduler=dynamic_sched))
        atomic_json(output / "raw_dynamic_ready.json", dynamic_rows)
        occupancy = [{"forward_index": row["forward_index"], "physical_batch": row["physical_batch"],
                      "position": row["position"], "cell_keys": row["cell_keys"], "wall_seconds": row["wall_seconds"]}
                     for row in dynamic_sched["events"]]
        write_csv(output / "dynamic_forward_occupancy.csv", occupancy)
        perf = [{"mode": "AUTHORITATIVE_SCALAR", "wall_seconds": sum(row["runtime_seconds"] for row in scalar_rows),
                "physical_forwards": sum(row["model_forwards"] for row in scalar_rows), "mean_effective_batch": 1.0},
                {"mode": "READY_REFACTOR_B1", "wall_seconds": b1_elapsed, "physical_forwards": b1_sched["physical_forwards"],
                 "mean_effective_batch": b1_sched["mean_effective_batch"]},
                {"mode": "DYNAMIC_READY_B2", "wall_seconds": dynamic_elapsed, "physical_forwards": dynamic_sched["physical_forwards"],
                 "mean_effective_batch": dynamic_sched["mean_effective_batch"]}]
        write_csv(output / "dynamic_performance.csv", perf)
        write_csv(output / "dynamic_cell_results.csv", [{key: row.get(key) for key in ("cell_key", "mode", "runtime_seconds", "candidate_count", "valid_grid_count", "nodes_expanded", "successors_considered", "successors_retained", "termination_reason")} for row in dynamic_rows])
        divergence = [{"cell_key": key, "refactor_vs_scalar": normalized_result_signature(scalar_results[key]) == normalized_result_signature(b1_results[key]),
                       "dynamic_vs_refactor": normalized_result_signature(ready_result(cell)) == normalized_result_signature(b1_results[key])}
                      for key, cell in zip([cell.cell_key for cell in dynamic_cells], dynamic_cells, strict=True)]
        write_csv(output / "dynamic_path_divergence.csv", divergence)
        hashes = {path.name: sha256_file(path) for path in sorted(output.glob("raw_*.json"))}
        atomic_json(output / "RAW_FREEZE_HASHES.json", {"raw_sha256": hashes, "native_token_contract": native,
                                                          "adapter_sha256": adapter_sha, "solutions_accessed": False})
        (output / "DYNAMIC_READY_GENERATION_FROZEN.flag").write_text(json_sha(hashes) + "\n", encoding="utf-8")
    finally:
        del model


def score(args: argparse.Namespace) -> None:
    output = args.output.resolve(); frozen = output / "DYNAMIC_READY_GENERATION_FROZEN.flag"
    if not frozen.exists() or not (output / "RAW_FREEZE_HASHES.json").exists():
        raise RuntimeError("refusing Gold: dynamic raw generation is not frozen")
    raw_hashes = read_json(output / "RAW_FREEZE_HASHES.json").get("raw_sha256", {})
    if not raw_hashes or any(sha256_file(output / name) != value for name, value in raw_hashes.items()):
        raise RuntimeError("refusing Gold: frozen raw SHA256 verification failed")
    solutions = read_json(args.solutions.resolve())
    rows = read_json(output / "raw_dynamic_ready.json")
    scored: list[dict[str, Any]] = []
    for row in rows:
        task, output_index = row["cell_key"].split(":o", 1)[0], int(row["cell_key"].split(":o", 1)[1].split(":", 1)[0])
        gold = solutions[task][output_index]
        matches = [index for index, candidate in enumerate(row["candidates"], start=1) if candidate["canonical_candidate"] == gold]
        scored.append({"cell_key": row["cell_key"], "candidate_count": row["candidate_count"],
                       "gold_hit_any_candidate": bool(matches), "gold_hit_candidate_count": len(matches),
                       "first_gold_candidate_rank": matches[0] if matches else None,
                       "candidate_pool_sha256": json_sha(row["candidates"])})
    write_csv(output / "POST_FREEZE_GOLD_SCORING.csv", scored)
    by_view = {row["cell_key"].rsplit(":", 1)[-1]: row for row in scored}
    canaries = ["identity", "flip_ud", "transpose"]
    summary = {"unique_outputs": 1, "gold_hit_cells": sum(row["gold_hit_any_candidate"] for row in scored),
               "gold_hit_outputs": int(any(row["gold_hit_any_candidate"] for row in scored)),
               "known_canaries": canaries, "known_canaries_preserved": sum(bool(by_view.get(view, {}).get("gold_hit_any_candidate")) for view in canaries),
               "solutions_accessed_after_freeze": True}
    atomic_json(output / "POST_FREEZE_GOLD_SUMMARY.json", summary)
    perf_rows = list(csv.DictReader((output / "dynamic_performance.csv").open(encoding="utf-8")))
    dynamic = next(row for row in perf_rows if row["mode"] == "DYNAMIC_READY_B2")
    scalar = next(row for row in perf_rows if row["mode"] == "AUTHORITATIVE_SCALAR")
    speedup = float(scalar["wall_seconds"]) / float(dynamic["wall_seconds"])
    batch = float(dynamic["mean_effective_batch"])
    decision = {"effective_batch": batch, "end_to_end_speedup_vs_scalar": speedup,
                "canary_retention": summary["known_canaries_preserved"],
                "decision": "STOP_DYNAMIC_POOL_NOT_FEEDING_GPU" if batch < 1.40 else
                            "NO_EXPANSION_SPEEDUP_INSUFFICIENT" if speedup <= 1.05 else
                            "EXPANSION_ELIGIBLE" if batch >= 1.60 and speedup >= 1.15 and summary["known_canaries_preserved"] == 3 else
                            "NO_AUTO_EXPANSION"}
    atomic_json(output / "DECISION.json", decision)
    report = "\n".join([f"# {EXPERIMENT}", "", "Contaminated development-only execution study.", "",
                          f"- Effective dynamic batch: {batch:.6f}", f"- Scalar/dynamic wall speedup: {speedup:.6f}",
                          f"- Known canaries preserved: {summary['known_canaries_preserved']}/3", f"- Decision: {decision['decision']}", ""])
    (output / "REGRET_DYNAMIC_READY_REPORT.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "run"):
        item = sub.add_parser(name); item.add_argument("--output", type=Path, required=True); item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True); item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True); item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True); item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--source-commit", required=True); item.add_argument("--gpu-id", type=int, default=0)
    scorer = sub.add_parser("score"); scorer.add_argument("--output", type=Path, required=True); scorer.add_argument("--solutions", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run(args)
    else: score(args)


if __name__ == "__main__":
    main()
