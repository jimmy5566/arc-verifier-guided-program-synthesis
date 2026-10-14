#!/usr/bin/env python3
"""Post-freeze NONBLIND Gold-path preference and search anatomy diagnostic."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_native import NVARCNativeProvider, native_messages, serialize_grid
from inference.nvarc_native_augmentation import NativeAugmentation
from scripts.run_exact_b_selector_phase3_r1024_v1 import (
    SCORING_VIEWS,
    canonical,
    oid_parts,
    read_json,
    read_jsonl_gz,
    sha256,
    sha_value,
    verification,
    write_csv_gz,
    write_json,
    write_json_gz,
)
from scripts.run_v7_ttt24_eval60_b_selector import EXPECTED_V7_SHA256, activate_task, adapter_preflight
from scripts.score_phase3_eval60_union_v1 import validate_archive
from scripts.score_v7_ttt24_eval60_postfreeze import EXPECTED_GOLD_SHA256, archive_payload_hashes_valid, load_gold_compact


LEGAL_ARC_TOKENS = tuple(range(11)) + (15,)


def safe_id(output_id: str) -> str:
    return output_id.replace(":", "_")


def verify_b_freeze(path: Path, outputs: list[str]) -> dict[str, Any]:
    ledger = read_json(path / "PRE_GOLD_HASHES.json")
    freeze = read_json(path / "PRE_GOLD_FREEZE.json")
    if verification(path, ledger)["status"] != "PASS" or freeze.get("status") != "PASS":
        raise RuntimeError("B_PRE_GOLD_FREEZE_INVALID")
    attempts = read_json(path / "B_ATTEMPTS_FROZEN.json")
    if freeze.get("gold_loaded") is not False or set(attempts) != {"dfs", "greedy"}:
        raise RuntimeError("B_PRE_GOLD_CONTRACT_INVALID")
    if any(set(attempts[label]) != set(outputs) for label in attempts):
        raise RuntimeError("B_PRE_GOLD_OUTPUT_IDENTITY_MISMATCH")
    return {"freeze_sha256": sha256(path / "PRE_GOLD_FREEZE.json"),
            "attempts_sha256": sha256(path / "B_ATTEMPTS_FROZEN.json")}


def base_adapter_preflight(root: Path, manifest_path: Path, tasks: list[str]) -> dict[str, Any]:
    rows = list(csv.DictReader(manifest_path.open(encoding="utf-8", newline="")))
    by_task = {str(row["task_id"]): row for row in rows if int(row["depth"]) == 24}
    result: dict[str, Any] = {}
    for task in tasks:
        expected = by_task.get(task)
        if expected is None:
            raise RuntimeError(f"HISTORICAL_D24_ADAPTER_MISSING:{task}")
        path = root / task / "depth_024"
        observed = sha256(path / "adapter_model.safetensors")
        config = sha256(path / "adapter_config.json")
        if observed != expected["adapter_model_sha256"] or config != expected["adapter_config_sha256"]:
            raise RuntimeError(f"HISTORICAL_D24_ADAPTER_HASH_MISMATCH:{task}")
        result[task] = {"path": str(path), "adapter_model_sha256": observed, "adapter_config_sha256": config}
    return {"status": "PASS", "task_count": len(result), "tasks": result,
            "manifest_sha256": sha256(manifest_path)}


def load_mode(args: argparse.Namespace, mode: str, first_adapter: Path) -> tuple[Any, NVARCNativeProvider, dict[str, Any]]:
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path,
        adapter_path=first_adapter,
        foundation_adapter_path=args.foundation_adapter if mode == "v7" else None,
        device="cuda:0", native_config_dir=args.native_config,
    )
    if mode == "v7" and (identity.get("foundation_adapter") or {}).get("adapter_sha256") != EXPECTED_V7_SHA256:
        raise RuntimeError("V7_FOUNDATION_IDENTITY_MISMATCH")
    provider = NVARCNativeProvider(model_path=args.model_path, tokenizer_config_dir=args.native_config, device="cuda:0")
    provider.model, provider.tokenizer = model, tokenizer
    return model, provider, identity


def activate(model: Any, mode: str, adapter: Path) -> None:
    if mode == "v7":
        activate_task(model, adapter)
    else:
        from scripts.run_exact_b_selector_phase3_r1024_v1 import _set_adapter
        _set_adapter(model, adapter)


def score_gold_view(model: Any, tokenizer: Any, messages: list[dict[str, str]], gold_grid: Any, *, context_window: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import torch

    prefix = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_tensors="pt", return_dict=True)["input_ids"]
    continuation = tokenizer(serialize_grid(gold_grid), add_special_tokens=False, return_tensors="pt")["input_ids"]
    eos = torch.tensor([[int(tokenizer.eos_token_id)]], dtype=continuation.dtype)
    ids = torch.cat((prefix, continuation, eos), dim=1).to(model.device)
    if int(ids.shape[-1]) > context_window:
        raise RuntimeError(f"GOLD_PATH_CONTEXT_EXCEEDED:{int(ids.shape[-1])}")
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits
        target = ids[:, int(prefix.shape[-1]):]
        predicted = logits[:, int(prefix.shape[-1]) - 1:-1, :].float()
        log_probs = torch.log_softmax(predicted, dim=-1)[0].cpu()
    rows: list[dict[str, Any]] = []
    cumulative_nll = 0.0; cumulative_regret = 0.0
    targets = [int(value) for value in target[0].detach().cpu().tolist()]
    for position, token in enumerate(targets):
        legal = [(candidate, float(log_probs[position, candidate].item())) for candidate in LEGAL_ARC_TOKENS]
        ranked = sorted(legal, key=lambda value: (-value[1], value[0]))
        token_lp = float(log_probs[position, token].item())
        best_lp = ranked[0][1]
        cumulative_nll -= token_lp; cumulative_regret += best_lp - token_lp
        rows.append({
            "gold_position": position, "gold_token_id": token, "gold_token_logprob": token_lp,
            "gold_token_local_rank": next(index for index, (candidate, _lp) in enumerate(ranked, 1) if candidate == token),
            "best_legal_token_id": ranked[0][0], "best_legal_token_logprob": best_lp,
            "gold_gap_from_best": best_lp - token_lp, "cumulative_gold_nll": cumulative_nll,
            "cumulative_gold_regret": cumulative_regret,
        })
    summary = {
        "prefix_tokens": int(prefix.shape[-1]), "gold_tokens_including_eos": len(rows),
        "mean_gold_token_logprob": fmean(row["gold_token_logprob"] for row in rows),
        "mean_gold_token_rank": fmean(row["gold_token_local_rank"] for row in rows),
        "cumulative_gold_nll": cumulative_nll, "cumulative_gold_regret": cumulative_regret,
    }
    del logits, predicted, log_probs, target, ids, prefix, continuation, eos
    return rows, summary


def score_mode(args: argparse.Namespace, mode: str, outputs: list[str], tasks: dict[str, Any], gold: dict[str, Any], adapters: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import gc
    import torch

    mode_root = args.output / "TEACHER_FORCED_SHARDS" / mode
    mode_root.mkdir(parents=True, exist_ok=True)
    first_task = oid_parts(outputs[0])[0]
    model, provider, identity = load_mode(args, mode, Path(adapters["tasks"][first_task]["path"]))
    token_rows: list[dict[str, Any]] = []; summaries: list[dict[str, Any]] = []
    try:
        active_task = None
        for output_id in outputs:
            task_id, output_index = oid_parts(output_id)
            if task_id != active_task:
                activate(model, mode, Path(adapters["tasks"][task_id]["path"])); active_task = task_id
            shard_path = mode_root / f"{safe_id(output_id)}.json.gz"
            if shard_path.is_file():
                with gzip.open(shard_path, "rt", encoding="utf-8") as handle:
                    shard = json.load(handle)
                if (shard.get("status") != "COMPLETE" or shard.get("gold_sha256") != EXPECTED_GOLD_SHA256 or
                        shard.get("adapter_model_sha256") != adapters["tasks"][task_id]["adapter_model_sha256"]):
                    raise RuntimeError(f"INVALID_EXISTING_GOLD_PATH_SHARD:{mode}:{output_id}")
            else:
                per_token: list[dict[str, Any]] = []; per_view: list[dict[str, Any]] = []
                for view_name in SCORING_VIEWS:
                    view = NativeAugmentation(geometry=view_name)
                    transformed_task = view.transform_task(tasks[task_id])
                    transformed_gold = view.transform_grid(gold[output_id]).astype(int).tolist()
                    rows, summary = score_gold_view(
                        model, provider.tokenizer, native_messages(transformed_task, output_index), transformed_gold,
                        context_window=args.context_window,
                    )
                    per_token.extend({"mode": mode, "task_id": task_id, "output_id": output_id,
                                      "view": view_name, **row} for row in rows)
                    per_view.append({"mode": mode, "task_id": task_id, "output_id": output_id,
                                     "view": view_name, **summary})
                shard = {
                    "schema": "ARC2_EVAL60_GOLD_PATH_TEACHER_FORCED_SHARD_V1", "status": "COMPLETE",
                    "classification": "NONBLIND_DIAGNOSTIC", "mode": mode, "output_id": output_id,
                    "gold_sha256": EXPECTED_GOLD_SHA256,
                    "adapter_model_sha256": adapters["tasks"][task_id]["adapter_model_sha256"],
                    "runtime_identity": identity, "per_token": per_token, "per_view": per_view,
                }
                write_json_gz(shard_path, shard)
            token_rows.extend(shard["per_token"]); summaries.extend(shard["per_view"])
            print(canonical({"event": "GOLD_PATH_OUTPUT_COMPLETE", "mode": mode, "output_id": output_id}), flush=True)
    finally:
        del provider, model
        gc.collect(); torch.cuda.empty_cache()
    return token_rows, summaries


def prefix_hash(tokens: tuple[int, ...]) -> str:
    return hashlib.sha256(",".join(str(value) for value in tokens).encode("ascii")).hexdigest()


def aug_geometry(augmentation_id: str) -> str:
    for part in augmentation_id.split("__"):
        if part.startswith("geom="):
            return part.split("=", 1)[1]
    raise ValueError(f"invalid augmentation id: {augmentation_id}")


def raw_output(archive: Path, output_id: str) -> dict[str, Any]:
    index = archive / "tables" / "candidate_index" / f"{safe_id(output_id)}.jsonl.gz"
    rows = read_jsonl_gz(index)
    if not rows:
        raise RuntimeError(f"EMPTY_CANDIDATE_INDEX:{output_id}")
    source = archive / str(rows[0]["source_archive_file"])
    with gzip.open(source, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def node_paths(nodes: list[dict[str, Any]]) -> tuple[dict[tuple[int, ...], list[dict[str, Any]]], dict[int, tuple[int, ...]]]:
    by_id = {int(node["node_id"]): node for node in nodes}
    paths: dict[int, tuple[int, ...]] = {}; grouped: dict[tuple[int, ...], list[dict[str, Any]]] = {}
    for node_id in sorted(by_id):
        node = by_id[node_id]; parent = node.get("parent_node_id")
        path = () if parent is None else paths[int(parent)]
        if node.get("selected_token") is not None:
            path += (int(node["selected_token"]),)
        paths[node_id] = path; grouped.setdefault(path, []).append(node)
    return grouped, paths


def search_anatomy(archive: Path, outputs: list[str], gold: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for output_id in outputs:
        payload = raw_output(archive, output_id)
        for cell in payload["cells"].values():
            augmentation_id = str(cell["augmentation_id"])
            view = NativeAugmentation(geometry=aug_geometry(augmentation_id))
            grid = view.transform_grid(gold[output_id]).astype(int).tolist()
            tokens = tuple(10 if char == "\n" else int(char) for char in serialize_grid(grid)) + (15,)
            grouped, _paths = node_paths(cell.get("nodes", []))
            trace = (cell.get("diagnostic_trace") or {}).get("search_trace", [])
            by_hash: dict[str, list[dict[str, Any]]] = {}
            for event in trace:
                by_hash.setdefault(str(event.get("prefix_hash")), []).append(event)
            first_divergence = None
            for length in range(1, len(tokens) + 1):
                wanted = tokens[:length]; events = by_hash.get(prefix_hash(wanted), [])
                nodes = grouped.get(wanted, [])
                if any(event.get("candidate_completed") for event in events) or any(node.get("state") == "completed" for node in nodes):
                    state = "GOLD_COMPLETION_REACHED"
                elif any(event.get("frontier_pop_order") is not None for event in events) or any(node.get("state") == "expanded" for node in nodes):
                    state = "GOLD_SUCCESSOR_EXPANDED"
                elif any(event.get("frontier_insert_order") is not None for event in events):
                    state = "GOLD_SUCCESSOR_RETAINED_NOT_EXPANDED"
                elif any(event.get("prune_reason") is not None for event in events) or any(node.get("state") == "pruned" for node in nodes):
                    state = "GOLD_SUCCESSOR_PRUNED"
                else:
                    state = "UNOBSERVED_OR_BUDGET_CENSORED"
                if first_divergence is None and state in {"GOLD_SUCCESSOR_PRUNED", "GOLD_SUCCESSOR_RETAINED_NOT_EXPANDED", "UNOBSERVED_OR_BUDGET_CENSORED"}:
                    first_divergence = length
                inserted = [int(event["frontier_insert_order"]) for event in events if event.get("frontier_insert_order") is not None]
                popped = [int(event["frontier_pop_order"]) for event in events if event.get("frontier_pop_order") is not None]
                rows.append({
                    "mode": mode, "task_id": oid_parts(output_id)[0], "output_id": output_id,
                    "augmentation_id": augmentation_id, "gold_prefix_length": length,
                    "gold_token_id": tokens[length - 1], "search_state": state,
                    "matched_node_ids": [int(node["node_id"]) for node in nodes],
                    "frontier_insert_order": min(inserted) if inserted else None,
                    "frontier_pop_order": min(popped) if popped else None,
                    "frontier_wait": (min(popped) - min(inserted)) if inserted and popped else None,
                    "nodes_expanded_at_event": min((int(event.get("nodes_expanded_so_far", 10**9)) for event in events), default=None),
                    "first_divergence_position": first_divergence,
                    "wall_censored": bool(cell.get("wall_time_censored")),
                    "diagnostic_trace_available": bool(trace),
                })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("v7_archive", "base_archive", "b_freeze", "v7_training", "v7_adapter_root",
                 "base_adapter_root", "model_path", "foundation_adapter", "challenge", "solutions",
                 "native_config", "output"):
        parser.add_argument("--" + name.replace("_", "-"), required=True, type=Path)
    parser.add_argument("--context-window", type=int, default=16384)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=True)
    outputs = validate_archive(args.v7_archive, 24)
    if validate_archive(args.base_archive, 24) != outputs:
        raise RuntimeError("BASE_V7_ARCHIVE_COHORT_MISMATCH")
    if not archive_payload_hashes_valid(args.v7_archive) or not archive_payload_hashes_valid(args.base_archive):
        raise RuntimeError("ARCHIVE_PAYLOAD_HASH_FAILURE")
    task_ids = sorted({oid_parts(output_id)[0] for output_id in outputs})
    v7_adapters = adapter_preflight(args.v7_training, args.v7_adapter_root, task_ids)
    base_adapters = base_adapter_preflight(args.base_adapter_root, args.base_archive / "ADAPTER_MANIFEST.csv", task_ids)
    gate = {
        "schema": "ARC2_EVAL60_GOLD_PATH_POSTFREEZE_GATE_V1", "status": "PASS",
        "classification": "NONBLIND_DIAGNOSTIC", "v7_archive_verified": True,
        "base_archive_verified": True, "b_freeze": verify_b_freeze(args.b_freeze, outputs),
        "v7_adapter_manifest": v7_adapters["manifest_sha256"],
        "base_adapter_manifest": base_adapters["manifest_sha256"], "output_count": 89, "task_count": 60,
    }
    write_json(args.output / "POSTFREEZE_GATE.json", gate)

    # First target access occurs only after every candidate, ranking and model
    # identity above has been frozen and verified.
    gold, provenance = load_gold_compact(args.solutions, outputs, EXPECTED_GOLD_SHA256)
    write_json(args.output / "GOLD_PROVENANCE.json", provenance)
    tasks = load_dataset(args.challenge)
    all_tokens: list[dict[str, Any]] = []; all_views: list[dict[str, Any]] = []
    for mode, adapters in (("base_ttt24", base_adapters), ("v7_ttt24", v7_adapters)):
        token_rows, view_rows = score_mode(args, "v7" if mode == "v7_ttt24" else "base", outputs, tasks, gold, adapters)
        all_tokens.extend({**row, "comparison_mode": mode} for row in token_rows)
        all_views.extend({**row, "comparison_mode": mode} for row in view_rows)
    write_csv_gz(args.output / "GOLD_TOKEN_PREFERENCE.csv.gz", all_tokens, list(all_tokens[0]))
    write_csv_gz(args.output / "GOLD_VIEW_SUMMARY.csv.gz", all_views, list(all_views[0]))

    search_rows = search_anatomy(args.base_archive, outputs, gold, "base_ttt24")
    search_rows.extend(search_anatomy(args.v7_archive, outputs, gold, "v7_ttt24"))
    write_csv_gz(args.output / "GOLD_SEARCH_ANATOMY.csv.gz", search_rows, list(search_rows[0]))
    summary = {
        "schema": "ARC2_EVAL60_GOLD_PATH_DIAGNOSTIC_SUMMARY_V1", "status": "COMPLETE",
        "classification": "NONBLIND_DIAGNOSTIC", "output_count": 89, "task_count": 60,
        "teacher_forced_views_per_mode": len(SCORING_VIEWS) * 89,
        "model_preference": {
            mode: {
                "mean_cumulative_gold_nll": fmean(row["cumulative_gold_nll"] for row in all_views if row["comparison_mode"] == mode),
                "mean_cumulative_gold_regret": fmean(row["cumulative_gold_regret"] for row in all_views if row["comparison_mode"] == mode),
                "mean_gold_token_rank": fmean(row["mean_gold_token_rank"] for row in all_views if row["comparison_mode"] == mode),
            } for mode in ("base_ttt24", "v7_ttt24")
        },
        "search_state_counts": {
            mode: dict(Counter(row["search_state"] for row in search_rows if row["mode"] == mode))
            for mode in ("base_ttt24", "v7_ttt24")
        },
        "historical_base_frontier_order_limitation": "UNAVAILABLE_WHERE_HISTORICAL_TRACE_LACKS_DIAGNOSTIC_EVENTS",
    }
    write_json(args.output / "SUMMARY.json", summary)
    files = {str(path.relative_to(args.output)).replace("\\", "/"): sha256(path)
             for path in sorted(args.output.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    write_json(args.output / "HASHES.json", files)
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
