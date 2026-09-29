#!/usr/bin/env python3
"""Nonblind, resumable Eval60 Gold-path bottleneck attribution.

This program deliberately separates *measurement* from search.  It never
recreates the historical Greedy or V5 candidate pools.  G1/G2 Gold-path
measurements use the G0.5-validated incremental KV replay for each frozen
model state, while G3 runs only the four previously omitted geometry views
through the already-frozen Greedy path.

All per-cell JSON files live in a Pod-local scratch directory.  The CSV/JSON
reports written under ``reports/`` are compact, reproducible exports suitable
for Git.  Gold is an explicit input to this NONBLIND_MECHANISM_DIAGNOSTIC.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import _fingerprint, read_json, restore_adapter, view_task


CURRENT_VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
D4_VIEWS = ("rot90", "rot180", "rot270", "flip_lr")
DEPTHS = (12, 24, 48)
LEGAL_ARC_TOKENS = tuple(range(11)) + (15,)
EXPECTED_COMPACT_COMMIT = "2a64377dae0126af0c0f8947dc46efbb43a2afff"
EXPECTED_V5_CONFIG_SHA = "e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0"
EXPECTED_ADAPTER_MANIFEST_SHA = "e3e95956b0c17e3017b3bb99999c53bfc5307908df68c43fc6641e8b1459c2a6"


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Independent task shards can publish distinct cells concurrently.  A
    # process-specific temporary name prevents an unrelated shard from racing
    # on a shared manifest's temporary file before the atomic replace.
    temporary = path.with_name(path.name + f".{os.getpid()}.partial")
    temporary.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(values)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def output_key(task_id: str, output_index: int | str) -> str:
    return f"{task_id}:o{int(output_index)}"


def output_gold(solutions: dict[str, Any], task_id: str, output_index: int) -> list[list[int]]:
    try:
        return solutions[task_id][output_index]
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError(f"missing Gold for {task_id}:o{output_index}") from error


def _cell_path(scratch: Path, phase: str, task_id: str, output_index: int, depth: int, view: str) -> Path:
    return scratch / phase / "cells" / task_id / f"o{output_index:02d}_d{depth:03d}_{view}.json"


def _sharded_task_ids(task_ids: list[str], args: argparse.Namespace) -> list[str]:
    """Return a deterministic disjoint task shard without changing cell semantics."""
    if args.task_shard_count < 1:
        raise RuntimeError("task_shard_count must be positive")
    if not 0 <= args.task_shard_index < args.task_shard_count:
        raise RuntimeError(
            f"invalid task shard {args.task_shard_index}/{args.task_shard_count}"
        )
    shard = [
        task_id for index, task_id in enumerate(task_ids)
        if index % args.task_shard_count == args.task_shard_index
    ]
    # This is scheduling-only: a bounded smoke writes real, resumable cells
    # for the first deterministic task in each shard.  The later full pass
    # uses the same immutable cohort and fills the remaining cell keys.
    if args.max_tasks_per_shard is not None:
        if args.max_tasks_per_shard < 1:
            raise RuntimeError("max_tasks_per_shard must be positive")
        shard = shard[:args.max_tasks_per_shard]
    return shard


def _adapter_metadata(adapters: Path, task_id: str, depth: int) -> dict[str, Any]:
    path = adapters / task_id / f"depth_{depth:03d}" / "metadata.json"
    metadata = read_json(path)
    required = {"checkpoint_sha256", "checkpoint_path", "task_id", "depth"}
    if not required.issubset(metadata) or str(metadata["task_id"]) != task_id or int(metadata["depth"]) != depth:
        raise RuntimeError(f"invalid adapter metadata:{path}")
    return metadata


def prepare_miss_set(args: argparse.Namespace) -> list[dict[str, Any]]:
    # The immutable V2 compact archive may be intentionally absent from a
    # compute Pod.  In that case use the already Git-frozen, mechanically
    # derived 56-output manifest made from it during G0.  This never derives a
    # new cohort from Gold or from mutable runtime output.
    if args.frozen_miss_manifest is not None:
        rows = _csv_rows(args.frozen_miss_manifest)
        required_columns = {
            "task_id", "output_index", "output_id", "greedy_pool_hit",
            "v5_pool_gold_hit_available", "union_greedy_v5_hit",
            "original_v5_oracle_certainty",
        }
        present_columns = set(rows[0]) if rows else set()
        missing_columns = sorted(required_columns - present_columns)
        if missing_columns:
            raise RuntimeError(f"frozen miss manifest missing required columns:{','.join(missing_columns)}")
        misses = [{
            "task_id": str(row["task_id"]), "output_index": int(row["output_index"]),
            "output_id": str(row["output_id"]), "greedy_pool_hit": _bool(row["greedy_pool_hit"]),
            "v5_pool_gold_hit_available": _bool(row["v5_pool_gold_hit_available"]),
            "union_greedy_v5_hit": _bool(row["union_greedy_v5_hit"]),
            "original_v5_oracle_certainty": str(row["original_v5_oracle_certainty"]),
        } for row in rows]
        if len({row["output_id"] for row in misses}) != len(misses):
            raise RuntimeError("frozen miss manifest contains duplicate output_id")
        if any(row["union_greedy_v5_hit"] for row in misses):
            raise RuntimeError("frozen miss manifest contains a union Greedy/V5 hit")
        frozen_manifest_count = len(misses)
        full_miss_count = 56
        full_rows_sha256 = hashlib.sha256(json.dumps(misses, sort_keys=True).encode()).hexdigest()
        if args.cohort_output_count is None:
            if frozen_manifest_count != full_miss_count:
                raise RuntimeError(
                    "frozen miss manifest without cohort_output_count must contain the original 56 outputs"
                )
            source = "FROZEN_G0_MANIFEST"
            selection = "ALL"
        elif frozen_manifest_count == full_miss_count:
            if not 1 <= args.cohort_output_count <= full_miss_count:
                raise RuntimeError(
                    f"invalid cohort_output_count={args.cohort_output_count}; expected 1..{full_miss_count}"
                )
            if args.cohort_output_count < full_miss_count:
                misses = sorted(
                    misses,
                    key=lambda row: (
                        hashlib.sha256(row["output_id"].encode("utf-8")).hexdigest(),
                        row["output_id"],
                    ),
                )[:args.cohort_output_count]
                source = f"FROZEN_G0_MANIFEST_SHA256_OUTPUT_PREFIX_{args.cohort_output_count}_OF_{full_miss_count}"
                selection = "SHA256_OUTPUT_ID_ASCENDING_PREFIX"
            else:
                source = "FROZEN_G0_MANIFEST"
                selection = "ALL"
        elif frozen_manifest_count == args.cohort_output_count:
            # A pre-registered subset is already an immutable cohort.  Never
            # expand it, re-sort it, or derive a replacement from another
            # manifest: membership is scientific provenance, not scheduling.
            source = f"FROZEN_MANIFEST_ALREADY_SUBSET_{frozen_manifest_count}"
            selection = "ALREADY_FROZEN_SUBSET"
        else:
            raise RuntimeError(
                f"frozen manifest count {frozen_manifest_count} does not match cohort_output_count={args.cohort_output_count} "
                f"and is not the original {full_miss_count}-output manifest"
            )
    else:
        source = EXPECTED_COMPACT_COMMIT
        compact = args.compact.resolve()
        greedy = {row["output_id"]: row for row in _csv_rows(compact / "greedy" / "greedy_outputs.csv")}
        v5 = {row["output_id"]: row for row in _csv_rows(compact / "turbodfs_v5" / "v5_outputs.csv")}
        if len(greedy) != 89 or len(v5) != 89:
            raise RuntimeError(f"compact output inventory mismatch greedy={len(greedy)} v5={len(v5)}")
        misses = []
        for output_id in sorted(greedy):
            g, t = greedy[output_id], v5[output_id]
            greedy_hit = _bool(g["pool_gold_hit"])
            # ``v5_pool_gold_hit_available`` describes whether a V5 Gold verdict
            # is available at all; it is false for five provenance-limited rows,
            # including rows already solved by Greedy.  The frozen V2 union field
            # is the mechanically correct lower-bound hit indicator.
            union_hit = _bool(t["union_greedy_v5_hit"])
            if not union_hit:
                task_id, index = output_id.split(":o")
                misses.append({
                    "task_id": task_id, "output_index": int(index), "output_id": output_id,
                    "greedy_pool_hit": greedy_hit,
                    "v5_pool_gold_hit_available": _bool(t["v5_pool_gold_hit_available"]),
                    "union_greedy_v5_hit": union_hit,
                    "original_v5_oracle_certainty": (
                        "ORIGINAL_V5_ORACLE_UNCERTAIN" if output_id in {"446ef5d2:o0", "cb2d8a2c:o0"}
                        else "TRUSTWORTHY_LOWER_BOUND_MISS"
                    ),
                })
    if args.frozen_miss_manifest is None:
        if len(misses) != 56:
            raise RuntimeError(f"expected exactly 56 mechanical lower-bound misses, got {len(misses)}")
        full_miss_count = len(misses)
        full_rows_sha256 = hashlib.sha256(json.dumps(misses, sort_keys=True).encode()).hexdigest()
        selection = "ALL"
    if args.frozen_miss_manifest is None and args.cohort_output_count is not None:
        if not 1 <= args.cohort_output_count <= full_miss_count:
            raise RuntimeError(
                f"invalid cohort_output_count={args.cohort_output_count}; "
                f"expected 1..{full_miss_count}"
            )
        misses = sorted(
            misses,
            key=lambda row: (
                hashlib.sha256(row["output_id"].encode("utf-8")).hexdigest(),
                row["output_id"],
            ),
        )[:args.cohort_output_count]
        source += f"_SHA256_OUTPUT_PREFIX_{args.cohort_output_count}_OF_{full_miss_count}"
        selection = "SHA256_OUTPUT_ID_ASCENDING_PREFIX"
    # Shards all derive the same immutable list.  Only shard zero (or a
    # non-sharded finalizer) publishes shared manifests/reports.
    if args.task_shard_count == 1 or args.task_shard_index == 0:
        destination = args.report_dir / "current_union_miss_outputs.csv"
        write_csv(destination, misses)
        atomic_json(args.scratch / "manifest" / "miss_set.json", {
            "status": "FROZEN", "count": len(misses), "source": source,
            "rows_sha256": hashlib.sha256(json.dumps(misses, sort_keys=True).encode()).hexdigest(),
            "full_miss_count": full_miss_count,
            "full_rows_sha256": full_rows_sha256,
            "selection": selection,
        })
    return misses


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    """Load the exact 4B model once.  Callers must retain this one process."""
    os.environ.update({
        "TRITON_PTXAS_PATH": str(args.ptxas),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
    # Scheduler-only: honor the physical device pin supplied by the worker
    # launcher.  The default preserves the historical single-worker route.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    import torch
    from peft import get_peft_model_state_dict
    from unsloth import FastLanguageModel
    from inference.nvarc_native import checkpoint_native_tokenizer

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("requires exactly one visible CUDA device")
    if tuple(torch.cuda.get_device_capability(0)) != (8, 6) or "RTX 3090" not in torch.cuda.get_device_name(0):
        raise RuntimeError(f"requires an RTX3090/sm86, got {torch.cuda.get_device_name(0)}")
    config = read_json(args.runtime_config)
    model, checkpoint_tokenizer = FastLanguageModel.from_pretrained(
        model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False,
        local_files_only=True, use_gradient_checkpointing=False,
        max_seq_length=int(config["max_sequence_length"]),
    )
    tokenizer, tokenizer_info = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    if len(checkpoint_tokenizer) != 16 or len(tokenizer) != 16 or checkpoint_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise RuntimeError("checkpoint/native tokenizer mismatch")
    model = FastLanguageModel.get_peft_model(
        model, r=int(config["rank"]), target_modules=list(config["target_modules"]),
        lora_alpha=int(config["alpha"]), lora_dropout=0.0, bias="none",
        use_gradient_checkpointing=False, random_state=int(config["seed"]), use_rslora=True, loftq_config=None,
    )
    for _name, parameter in model.named_parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    initial = {name: value.detach().clone() for name, value in get_peft_model_state_dict(model, adapter_name="default").items()}
    if not initial:
        raise RuntimeError("empty initial adapter")
    return model, tokenizer, initial, config, tokenizer_info


def _encoded_view(tokenizer: Any, task: Any, view: str, config: dict[str, Any]) -> tuple[Any, Any]:
    return common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=config)


def teacher_force(
    *, model: Any, tokenizer: Any, task: Any, view: str, config: dict[str, Any], target_grid: list[list[int]],
    max_score: float, max_new_tokens: int, forced_token_ids: list[int] | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Exact B=1 continuation scoring plus the V5 local floor decision."""
    import torch
    from inference.nvarc_native import parse_native_grid, serialize_grid

    encoded, augmentation = _encoded_view(tokenizer, task, view, config)
    prompt = encoded["input_ids"]
    # G0 replays an already-frozen native suffix.  Its stored display grid is
    # deliberately not parsed/re-serialized here: only the original token
    # transport can test logit parity.  G1/G2/G3 always take the Gold-grid
    # branch below and therefore exercise the complete transform/serializer.
    if forced_token_ids is not None:
        tokens = [int(x) for x in forced_token_ids]
        serialized = tokenizer.decode(tokens, skip_special_tokens=True)
        serialization_ok = True
    else:
        transformed_gold = augmentation.transform_grid(target_grid).astype(int).tolist()
        serialized = serialize_grid(transformed_gold)
        tokenized = [int(x) for x in tokenizer(serialized, add_special_tokens=False)["input_ids"]]
        tokens = tokenized + [int(tokenizer.eos_token_id)]
        serialization_ok = parse_native_grid(serialized) == transformed_gold and tuple(tokenized + [int(tokenizer.eos_token_id)]) == tuple(tokens)
    illegal = [token for token in tokens if token not in LEGAL_ARC_TOKENS]
    if illegal or not serialization_ok:
        return ({
            "status": "SERIALIZATION_OR_TRANSPORT_FAILURE", "prompt_tokens": int(prompt.shape[-1]),
            "gold_token_count": len(tokens), "illegal_token_ids": illegal,
            "serialized_gold": serialized, "view": view, "target_serialization_ok": serialization_ok,
        }, [])
    ids = torch.cat((prompt, torch.tensor([tokens], dtype=prompt.dtype)), dim=1).to(model.device)
    if int(ids.shape[-1]) > int(config["generation_context_window"]):
        return ({
            "status": "SERIALIZATION_OR_TRANSPORT_FAILURE", "reason": "context_overflow",
            "prompt_tokens": int(prompt.shape[-1]), "gold_token_count": len(tokens), "view": view,
        }, [])
    with torch.inference_mode():
        logits = model(input_ids=ids, use_cache=False, return_dict=True).logits[:, int(prompt.shape[-1]) - 1:-1, :]
    if int(logits.shape[1]) != len(tokens):
        raise RuntimeError("teacher-force output/token length mismatch")
    cumulative = 0.0
    strict_path = True
    v5_path = True
    token_budget_blocked = len(tokens) > max_new_tokens
    first_strict_position: int | None = None
    first_strict_nll: float | None = None
    traces: list[dict[str, Any]] = []
    for position, token in enumerate(tokens):
        values = torch.log_softmax(logits[0, position].float(), dim=-1)
        logprob = float(values[token].item()); cumulative -= logprob
        legal_pairs = [(t, float(values[t].item())) for t in LEGAL_ARC_TOKENS]
        ranked_full = sorted(((float(values[t].item()), t) for t in range(int(values.shape[0]))), key=lambda pair: (-pair[0], pair[1]))
        ranked_legal = sorted(((lp, t) for t, lp in legal_pairs), key=lambda pair: (-pair[0], pair[1]))
        full_rank = 1 + next(index for index, (_value, candidate) in enumerate(ranked_full) if candidate == token)
        legal_rank = 1 + next(index for index, (_value, candidate) in enumerate(ranked_legal) if candidate == token)
        remaining = max_new_tokens - position
        is_legal = token == int(tokenizer.eos_token_id) or remaining > 1
        successor_scores = [(cumulative - lp, candidate, lp) for candidate, lp in legal_pairs if candidate == int(tokenizer.eos_token_id) or remaining > 1]
        strict = [(score, candidate, lp) for score, candidate, lp in successor_scores if score < max_score]
        strict_gold = bool(is_legal and cumulative < max_score)
        restored_gold = False
        if not strict and successor_scores:
            restored = sorted(successor_scores, key=lambda item: (item[0], item[1]))[0]
            restored_gold = bool(restored[1] == token)
        v5_gold = strict_gold if strict else restored_gold
        if not strict_gold and first_strict_position is None:
            first_strict_position, first_strict_nll = position, cumulative
        strict_path = strict_path and strict_gold
        v5_path = v5_path and v5_gold
        top = ranked_full[:2]
        traces.append({
            "token_position": position, "gold_token_id": token, "gold_token_logprob": logprob,
            "gold_token_probability": float(torch.exp(values[token]).item()),
            "gold_token_rank_full_vocab": full_rank, "gold_token_rank_legal_arc_vocab": legal_rank,
            "cumulative_gold_nll": cumulative, "top1_token_id": top[0][1], "top1_logprob": top[0][0],
            "top2_token_id": top[1][1], "top2_logprob": top[1][0],
            "strict_gold_survives": strict_gold, "strict_survivor_count": len(strict),
            "v5_floor_activated": not bool(strict), "v5_floor_restores_gold": restored_gold,
            "v5_local_gold_survives": v5_gold,
            "legal_arc_token_logprobs_json": json.dumps({str(t): lp for t, lp in legal_pairs}, separators=(",", ":")),
        })
    del logits, ids, encoded
    status = "STRICT_SEARCHABLE" if strict_path and not token_budget_blocked else (
        "FLOOR_REQUIRED_AND_SEARCHABLE" if v5_path and not token_budget_blocked else (
            "TOKEN_BUDGET_BLOCKED" if token_budget_blocked else "PRUNING_POLICY_BLOCKED"
        )
    )
    return ({
        "status": status, "prompt_tokens": int(prompt.shape[-1]), "gold_token_count": len(tokens),
        "serialized_gold": serialized, "target_serialization_ok": serialization_ok,
        "strict_public_searchable": bool(strict_path and not token_budget_blocked),
        "v5_local_gold_path_survives": bool(v5_path and not token_budget_blocked),
        "token_budget_blocked": bool(token_budget_blocked), "final_gold_nll": cumulative,
        "mean_gold_legal_rank": sum(row["gold_token_rank_legal_arc_vocab"] for row in traces) / len(traces),
        "worst_gold_legal_rank": max(row["gold_token_rank_legal_arc_vocab"] for row in traces),
        "first_strict_prune_position": first_strict_position, "first_strict_prune_nll": first_strict_nll,
        "view": view,
    }, traces)


def _runtime_identity(args: argparse.Namespace) -> dict[str, Any]:
    # ``--adapters`` intentionally points to the immutable checkpoints
    # directory, while the authoritative manifest lives at the run root.
    adapter_manifest = args.adapters.parent / "checkpoint_manifest.csv"
    v5_sha = sha_file(args.v5_config)
    adapter_sha = sha_file(adapter_manifest)
    if v5_sha != EXPECTED_V5_CONFIG_SHA:
        raise RuntimeError(f"V5 config hash mismatch:{v5_sha}")
    if adapter_sha != EXPECTED_ADAPTER_MANIFEST_SHA:
        raise RuntimeError(f"adapter manifest hash mismatch:{adapter_sha}")
    if len(list(args.adapters.glob("*/depth_*/adapter_model.safetensors"))) != 180:
        raise RuntimeError("expected 180 retained adapters")
    return {"v5_config_sha256": v5_sha, "adapter_manifest_sha256": adapter_sha, "compact_commit": EXPECTED_COMPACT_COMMIT}


def run_g0(args: argparse.Namespace) -> None:
    import pandas as pd
    started = time.perf_counter(); identity = _runtime_identity(args)
    cells = pd.read_parquet(args.greedy_parquet)
    selected: list[dict[str, Any]] = []
    for depth in DEPTHS:
        group = cells[(cells["depth"] == depth) & (cells["valid_grid"] == True)].sort_values(["task_id", "output_index", "view"])
        selected.extend(group.head(2).to_dict("records"))
    if len(selected) != 6:
        raise RuntimeError("could not select six trustworthy G0 cells")
    model, tokenizer, _initial, config, tokenizer_info = _load_runtime(args)
    from arc.io import load_dataset
    tasks = load_dataset(args.challenge)
    rows: list[dict[str, Any]] = []
    for frozen in selected:
        task_id, output_index, depth, view = str(frozen["task_id"]), int(frozen["output_index"]), int(frozen["depth"]), str(frozen["view"])
        metadata = _adapter_metadata(args.adapters, task_id, depth); common.load_adapter(model=model, metadata=metadata)
        summary, _trace = teacher_force(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index), view=view, config=config,
                                        target_grid=frozen["raw_transformed_candidate_grid"], max_score=float(read_json(args.v5_config)["max_score"]),
                                        max_new_tokens=int(read_json(args.v5_config)["max_new_tokens"]), forced_token_ids=[int(x) for x in frozen["token_ids"]])
        actual = float(summary.get("final_gold_nll", float("nan")))
        stored_logprob = float(frozen["sequence_logprob_sum"])
        forced_logprob = -actual
        raw_text = tokenizer.decode([int(x) for x in frozen["token_ids"]], skip_special_tokens=True)
        row = {
            "task_id": task_id, "output_index": output_index, "depth": depth, "view": view,
            "stored_checkpoint_sha256": str(frozen["checkpoint_sha256"]), "loaded_checkpoint_sha256": metadata["checkpoint_sha256"],
            "checkpoint_sha_match": str(frozen["checkpoint_sha256"]) == metadata["checkpoint_sha256"],
            "stored_prompt_tokens": int(frozen["prompt_tokens"]), "forced_prompt_tokens": summary.get("prompt_tokens"),
            "prompt_tokenization_match": int(frozen["prompt_tokens"]) == int(summary.get("prompt_tokens", -1)),
            "stored_token_count": len(frozen["token_ids"]), "forced_token_count": summary.get("gold_token_count"),
            "token_count_match": len(frozen["token_ids"]) == int(summary.get("gold_token_count", -1)),
            "candidate_serialization_match": raw_text == str(frozen["raw_transformed_text"]),
            "stored_sequence_logprob": stored_logprob, "forced_sequence_logprob": forced_logprob,
            "abs_logprob_delta": abs(stored_logprob - forced_logprob), "tolerance": args.g0_logprob_tolerance,
            "numerical_parity": abs(stored_logprob - forced_logprob) <= args.g0_logprob_tolerance,
            "teacher_force_status": summary.get("status"),
        }
        row["pass"] = all(_bool(row[key]) for key in ("checkpoint_sha_match", "prompt_tokenization_match", "token_count_match", "candidate_serialization_match", "numerical_parity"))
        rows.append(row)
    passed = len(rows) == 6 and all(_bool(row["pass"]) for row in rows)
    write_csv(args.report_dir / "g0_runtime_parity.csv", rows)
    report = ["# G0 runtime parity", "", f"STATUS: {'PASS' if passed else 'FAIL'}", f"Tolerance: {args.g0_logprob_tolerance}", "", "| depth | task/output/view | abs logprob delta | pass |", "|---|---|---:|---|"]
    report += [f"| {r['depth']} | {r['task_id']}:o{r['output_index']} / {r['view']} | {r['abs_logprob_delta']:.8g} | {r['pass']} |" for r in rows]
    (args.report_dir / "G0_RUNTIME_PARITY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(args.scratch / "g0" / "status.json", {"status": "PASS" if passed else "FAIL", "rows": rows, "identity": identity, "tokenizer": tokenizer_info, "wall_seconds": time.perf_counter() - started})
    if not passed:
        raise RuntimeError("G0_RUNTIME_PARITY_FAIL")


def _iter_phase_cells(scratch: Path, phase: str) -> list[dict[str, Any]]:
    records = []
    for path in sorted((scratch / phase / "cells").glob("*/*.json")):
        records.append(read_json(path))
    return records


def _write_g1_exports(args: argparse.Namespace, phase: str, cells: list[dict[str, Any]]) -> None:
    token_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for cell in cells:
        summaries.append({key: value for key, value in cell.items() if key != "token_trace"})
        for token in cell.get("token_trace", []): token_rows.append({key: cell.get(key) for key in ("output_id", "task_id", "output_index", "depth", "view", "phase")} | token)
    output_rows = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in summaries: grouped[cell["output_id"]].append(cell)
    for output_id, group in sorted(grouped.items()):
        statuses = {str(x.get("status")) for x in group}
        strict = any(_bool(x.get("strict_public_searchable")) for x in group)
        v5 = any(_bool(x.get("v5_local_gold_path_survives")) for x in group)
        if strict: primary = "DECODER_EXECUTION_SUSPECT"
        elif v5: primary = "DECODER_PRUNING_LIMITED"
        elif statuses == {"PRUNING_POLICY_BLOCKED"}: primary = "PRUNING_POLICY_BLOCKED"
        elif "TOKEN_BUDGET_BLOCKED" in statuses and statuses == {"TOKEN_BUDGET_BLOCKED"}: primary = "TOKEN_BUDGET_BLOCKED"
        elif "SERIALIZATION_OR_TRANSPORT_FAILURE" in statuses: primary = "SERIALIZATION_OR_TRANSPORT_FAILURE"
        else: primary = "CURRENT_PROBABILITY_STATE_LIMITED"
        output_rows.append({"output_id": output_id, "cells": len(group), "any_strict_searchable": strict, "any_v5_local_searchable": v5,
                            "output_class": primary, "best_final_gold_nll": min(float(x.get("final_gold_nll", float("inf"))) for x in group)})
    prefix = "g1" if phase == "g1" else "g3_extra_view"
    write_csv(args.report_dir / f"{prefix}_token_trace.csv", token_rows)
    write_csv(args.report_dir / f"{prefix}_cell_summary.csv", summaries)
    write_csv(args.report_dir / ("g1_output_summary.csv" if phase == "g1" else "g3_extra_view_output_summary.csv"), output_rows)


def run_g1_half_gate(args: argparse.Namespace) -> None:
    """Finalize the pre-registered 28-output G1 stop gate without model work."""
    misses = prepare_miss_set(args)
    cells = _iter_phase_cells(args.scratch, "g1")
    expected = len(misses) * len(DEPTHS) * len(CURRENT_VIEWS)
    if len(cells) != expected:
        raise RuntimeError(f"g1 half incomplete {len(cells)}/{expected}")
    _write_g1_exports(args, "g1", cells)
    outputs = _csv_rows(args.report_dir / "g1_output_summary.csv")
    if len(outputs) != len(misses):
        raise RuntimeError(f"g1 half output inventory mismatch {len(outputs)}/{len(misses)}")
    strict = sum(_bool(row["any_strict_searchable"]) for row in outputs)
    v5 = sum(_bool(row["any_v5_local_searchable"]) for row in outputs)
    pruning = sum(row["output_class"] == "PRUNING_POLICY_BLOCKED" for row in outputs)
    probability = sum(row["output_class"] == "CURRENT_PROBABILITY_STATE_LIMITED" for row in outputs)
    decision = "CONTINUE_REMAINING_G1" if v5 > 0 else "STOP_G1_AND_RUN_G2"
    payload = {
        "status": "COMPLETE", "scope": "G1_HALF_SHA256_OUTPUT_PREFIX",
        "outputs": len(outputs), "cells": len(cells),
        "A_any_current_v5_searchable": v5,
        "B_pruning_policy_blocked": pruning,
        "C_all_current_states_probability_limited": probability,
        "strict_searchable": strict,
        "decision": decision,
        "decision_rule": "continue remaining G1 iff A_any_current_v5_searchable > 0",
    }
    atomic_json(args.report_dir / "G1_HALF_GATE.json", payload)
    report = ["# G1 half stop gate", "", "NONBLIND_MECHANISM_DIAGNOSTIC", "",
              f"A = #(any current V5-searchable) = {v5}",
              f"B = #(pruning-policy blocked) = {pruning}",
              f"C = #(all current states probability-limited) = {probability}",
              f"Strict-searchable outputs = {strict}", "",
              f"Decision: {decision}",
              "Rule: complete remaining G1 only when A > 0; otherwise jump to G2."]
    (args.report_dir / "G1_HALF_GATE.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(args.scratch / "g1" / "half_gate.json", payload)


def run_gold_surface(args: argparse.Namespace, *, phase: str, depths: tuple[int, ...], views: tuple[str, ...], use_initial_adapter: bool = False) -> None:
    # Late import avoids a circular dependency: the G0.5 scorer deliberately
    # reuses this runner's immutable source-identity/load helpers.
    from scripts.run_g0_5_incremental_kv_replay import incremental_gold_path
    started = time.perf_counter(); _runtime_identity(args)
    misses = prepare_miss_set(args)
    model, tokenizer, initial, config, _info = _load_runtime(args)
    from arc.io import load_dataset
    tasks, solutions = load_dataset(args.challenge), json.loads(args.solutions.read_text(encoding="utf-8"))
    v5_config = read_json(args.v5_config); max_score, max_tokens = float(v5_config["max_score"]), int(v5_config["max_new_tokens"])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses: by_task[row["task_id"]].append(row)
    task_ids = _sharded_task_ids(sorted(by_task), args)
    expected_shard_cells = 0
    for task_id in task_ids:
        for depth in depths:
            if use_initial_adapter:
                restore_adapter(model, initial); checkpoint_sha = "INITIAL_TTT0"
            else:
                metadata = _adapter_metadata(args.adapters, task_id, depth); common.load_adapter(model=model, metadata=metadata); checkpoint_sha = metadata["checkpoint_sha256"]
            for row in by_task[task_id]:
                target_task = view_task(tasks[task_id], int(row["output_index"])); gold = output_gold(solutions, task_id, int(row["output_index"]))
                for view in views:
                    expected_shard_cells += 1
                    destination = _cell_path(args.scratch, phase, task_id, int(row["output_index"]), depth, view)
                    if destination.is_file(): continue
                    began = time.perf_counter()
                    summary, traces = incremental_gold_path(
                        model=model, tokenizer=tokenizer, task=target_task, view=view,
                        config=config, target_grid=gold, max_score=max_score,
                        max_new_tokens=max_tokens,
                    )
                    payload = {**row, **summary, "phase": phase, "depth": depth, "view": view, "checkpoint_sha256": checkpoint_sha,
                               "wall_seconds": time.perf_counter() - began, "token_trace": traces}
                    atomic_json(destination, payload)
        task_expected_cells = len(by_task[task_id]) * len(depths) * len(views)
        task_completed_cells = sum(
            _cell_path(args.scratch, phase, task_id, int(row["output_index"]), depth, view).is_file()
            for depth in depths
            for row in by_task[task_id]
            for view in views
        )
        if task_completed_cells != task_expected_cells:
            raise RuntimeError(f"{phase} task checkpoint incomplete {task_id} {task_completed_cells}/{task_expected_cells}")
        atomic_json(args.scratch / phase / "tasks" / f"{task_id}.json", {
            "status": "COMPLETE_TASK", "phase": phase, "task_id": task_id,
            "output_ids": [row["output_id"] for row in by_task[task_id]],
            "completed_cells": task_completed_cells, "expected_cells": task_expected_cells,
        })
    if args.task_shard_count > 1:
        completed_shard_cells = sum(
            _cell_path(args.scratch, phase, task_id, int(row["output_index"]), depth, view).is_file()
            for task_id in task_ids
            for depth in depths
            for row in by_task[task_id]
            for view in views
        )
        if completed_shard_cells != expected_shard_cells:
            raise RuntimeError(
                f"{phase} shard {args.task_shard_index}/{args.task_shard_count} incomplete "
                f"{completed_shard_cells}/{expected_shard_cells}"
            )
        atomic_json(
            args.scratch / phase / f"shard-{args.task_shard_index:02d}-of-{args.task_shard_count:02d}.json",
            {
                "status": "COMPLETE_SHARD",
                "phase": phase,
                "task_shard_index": args.task_shard_index,
                "task_shard_count": args.task_shard_count,
                "task_ids": task_ids,
                "completed_cells": completed_shard_cells,
                "expected_cells": expected_shard_cells,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        return
    cells = _iter_phase_cells(args.scratch, phase)
    expected = len(misses) * len(depths) * len(views)
    if len(cells) != expected: raise RuntimeError(f"{phase} incomplete {len(cells)}/{expected}")
    if phase == "g1":
        _write_g1_exports(args, phase, cells)
        outputs = _csv_rows(args.report_dir / "g1_output_summary.csv")
        report = ["# G1 Gold-path accessibility", "", "NONBLIND_MECHANISM_DIAGNOSTIC", "", f"Cells: {len(cells)}", f"Outputs: {len(outputs)}", f"Wall seconds: {time.perf_counter()-started:.3f}", "", "Exact V5 max_score is read from the frozen config; strict inequality is used."]
        (args.report_dir / "G1_GOLD_PATH_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    elif phase == "g2":
        write_csv(args.report_dir / "g2_ttt0_cell_summary.csv", [{key: value for key, value in row.items() if key != "token_trace"} for row in cells])
    elif phase == "g3":
        _write_g1_exports(args, phase, cells)
    atomic_json(args.scratch / phase / "status.json", {"status": "COMPLETE", "cells": len(cells), "expected_cells": expected, "wall_seconds": time.perf_counter() - started})


def run_g2_summary(args: argparse.Namespace) -> None:
    g1 = _iter_phase_cells(args.scratch, "g1"); g2 = _iter_phase_cells(args.scratch, "g2")
    by_output: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in g1 + g2: by_output[row["output_id"]][int(row["depth"])].append(row)
    rows: list[dict[str, Any]] = []
    for output_id, states in sorted(by_output.items()):
        best = {}
        for depth in (0, 12, 24, 48):
            rows_for_state = states.get(depth, [])
            best[depth] = min((float(x.get("final_gold_nll", float("inf"))) for x in rows_for_state), default=float("inf"))
        v5 = {depth: any(_bool(x.get("v5_local_gold_path_survives")) for x in states.get(depth, [])) for depth in (0, 12, 24, 48)}
        adapted = any(v5[d] for d in DEPTHS)
        classification = "TTT_HARMFUL_STRONG" if v5[0] and not adapted else (
            "TTT_HELPFUL_STRONG" if not v5[0] and adapted else (
                "TTT_DEPTH_SENSITIVE" if sum(v5[d] for d in DEPTHS) not in (0, 3) else "ALL_CURRENT_STATES_LOW_ACCESSIBILITY"
            )
        )
        rows.append({"output_id": output_id, "best_state_by_gold_nll": min(best, key=best.get), "ttt0_best_gold_nll": best[0], "ttt12_best_gold_nll": best[12], "ttt24_best_gold_nll": best[24], "ttt48_best_gold_nll": best[48],
                     "ttt0_v5_searchable": v5[0], "ttt12_v5_searchable": v5[12], "ttt24_v5_searchable": v5[24], "ttt48_v5_searchable": v5[48], "g2_ttt_class": classification})
    write_csv(args.report_dir / "g2_state_comparison.csv", rows); write_csv(args.report_dir / "g2_output_attribution.csv", rows)
    counts = Counter(row["g2_ttt_class"] for row in rows)
    (args.report_dir / "G2_TTT_STATE_REPORT.md").write_text("# G2 TTT state comparison\n\n" + "\n".join(f"- {key}: {value}" for key, value in sorted(counts.items())) + "\n", encoding="utf-8")


def _float_metric(cell: dict[str, Any], key: str) -> float:
    """Return a required finite numeric G2 metric, failing closed on gaps."""
    try:
        value = float(cell[key])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"missing numeric G2 metric {key} in {cell.get('output_id')}") from error
    if value != value or value in (float("inf"), -float("inf")):
        raise RuntimeError(f"non-finite G2 metric {key} in {cell.get('output_id')}")
    return value


def _first_prune_fraction(cell: dict[str, Any]) -> float:
    """Map no strict prune to the latest possible fraction (1.0).

    The frozen scorer records a zero-based first failing token.  A path that
    never fails strict pruning is later than every actual failure, so 1.0 is
    the only ordering-preserving value for the requested "latest" comparison.
    The raw position remains exported alongside this derived comparison field.
    """
    count = int(cell.get("gold_token_count", 0))
    if count <= 0:
        raise RuntimeError(f"invalid gold_token_count in {cell.get('output_id')}")
    position = cell.get("first_strict_prune_position")
    if position in (None, "", "None"):
        return 1.0
    position_float = float(position)
    if not 0.0 <= position_float < float(count):
        raise RuntimeError(f"invalid first strict prune position in {cell.get('output_id')}")
    return position_float / float(count)


def _g2_cell_metrics(cell: dict[str, Any]) -> dict[str, Any]:
    """Attach derived, non-semantic G2 comparison values to a frozen cell."""
    count = int(cell.get("gold_token_count", 0))
    nll = _float_metric(cell, "final_gold_nll")
    if count <= 0:
        raise RuntimeError(f"invalid gold_token_count in {cell.get('output_id')}")
    return {
        **cell,
        "mean_gold_nll_per_token": nll / float(count),
        "first_strict_prune_fraction": _first_prune_fraction(cell),
    }


def _metric_winners(cells: list[dict[str, Any]], key: str, *, higher_is_better: bool) -> tuple[float, list[dict[str, Any]]]:
    values = [_float_metric(cell, key) for cell in cells]
    best = max(values) if higher_is_better else min(values)
    # Exact numerical equality is intentional: G2 must not introduce an
    # unregistered epsilon/tie-breaker when comparing frozen evidence.
    return best, [cell for cell in cells if _float_metric(cell, key) == best]


def _view_names(cells: list[dict[str, Any]]) -> str:
    return ",".join(sorted(str(cell["view"]) for cell in cells))


def _state_summary(cells: list[dict[str, Any]], *, output_id: str, depth: int) -> dict[str, Any]:
    if len(cells) != len(CURRENT_VIEWS) or {str(cell.get("view")) for cell in cells} != set(CURRENT_VIEWS):
        raise RuntimeError(f"incomplete view surface for {output_id} depth={depth}")
    derived = [_g2_cell_metrics(cell) for cell in cells]
    nll, nll_winners = _metric_winners(derived, "final_gold_nll", higher_is_better=False)
    nll_per_token, nll_per_token_winners = _metric_winners(derived, "mean_gold_nll_per_token", higher_is_better=False)
    prune_fraction, prune_winners = _metric_winners(derived, "first_strict_prune_fraction", higher_is_better=True)
    rank, rank_winners = _metric_winners(derived, "mean_gold_legal_rank", higher_is_better=False)
    return {
        "depth": depth,
        "cells": derived,
        "best_final_gold_nll": nll,
        "best_final_gold_nll_views": _view_names(nll_winners),
        "best_nll_per_token": nll_per_token,
        "best_nll_per_token_views": _view_names(nll_per_token_winners),
        "latest_prune_fraction": prune_fraction,
        "latest_prune_fraction_views": _view_names(prune_winners),
        "lowest_mean_legal_rank": rank,
        "lowest_mean_legal_rank_views": _view_names(rank_winners),
        "any_strict_searchable": any(_bool(cell.get("strict_public_searchable")) for cell in derived),
        "any_v5_local_searchable": any(_bool(cell.get("v5_local_gold_path_survives")) for cell in derived),
    }


def _same_cell_harmful_soft(ttt0: dict[str, Any], adapted: list[dict[str, Any]]) -> bool:
    """Require one TTT0 view to beat every adapted depth on both metrics."""
    return any(
        _float_metric(cell, "mean_gold_nll_per_token") < min(state["best_nll_per_token"] for state in adapted)
        and _float_metric(cell, "first_strict_prune_fraction") > max(state["latest_prune_fraction"] for state in adapted)
        for cell in ttt0["cells"]
    )


def _same_cell_helpful_soft(ttt0: dict[str, Any], adapted: list[dict[str, Any]]) -> bool:
    """Require one adapted view to dominate the whole TTT0 surface."""
    return any(
        _float_metric(cell, "mean_gold_nll_per_token") < ttt0["best_nll_per_token"]
        and _float_metric(cell, "first_strict_prune_fraction") > ttt0["latest_prune_fraction"]
        for state in adapted for cell in state["cells"]
    )


def run_g2_greedy(args: argparse.Namespace) -> None:
    """Generate the frozen TTT0 four-view surface without accessing Gold.

    Gold is intentionally attached only by ``g2-finalize`` after the complete
    per-cell candidate inventory is frozen.  This preserves the requested
    generation-before-evaluation boundary even inside this nonblind study.
    """
    started = time.perf_counter(); _runtime_identity(args)
    misses = prepare_miss_set(args)
    model, tokenizer, initial, config, _info = _load_runtime(args)
    from arc.io import load_dataset
    tasks = load_dataset(args.challenge)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses:
        by_task[row["task_id"]].append(row)
    task_ids = _sharded_task_ids(sorted(by_task), args)
    expected_shard_cells = 0
    for task_id in task_ids:
        # TTT0 is the fresh base/SFT adapter state.  Restoring it per task
        # makes the no-task-update contract explicit and guards against any
        # mutable state left by model generation.
        restore_adapter(model, initial)
        for row in by_task[task_id]:
            target_task = view_task(tasks[task_id], int(row["output_index"]))
            for view in CURRENT_VIEWS:
                expected_shard_cells += 1
                destination = _cell_path(args.scratch, "g2_greedy", task_id, int(row["output_index"]), 0, view)
                if destination.is_file():
                    continue
                payload = common.greedy_cell(
                    model=model, tokenizer=tokenizer, task=target_task,
                    task_id=task_id, output_index=int(row["output_index"]),
                    depth=0, view=view, config=config, checkpoint_sha="INITIAL_TTT0",
                )
                payload.update(row)
                payload.update({"phase": "g2_greedy", "ttt_updates": 0, "gold_attached": False})
                atomic_json(destination, payload)
        task_expected_cells = len(by_task[task_id]) * len(CURRENT_VIEWS)
        task_completed_cells = sum(
            _cell_path(args.scratch, "g2_greedy", task_id, int(row["output_index"]), 0, view).is_file()
            for row in by_task[task_id] for view in CURRENT_VIEWS
        )
        if task_completed_cells != task_expected_cells:
            raise RuntimeError(f"g2_greedy task checkpoint incomplete {task_id} {task_completed_cells}/{task_expected_cells}")
        atomic_json(args.scratch / "g2_greedy" / "tasks" / f"{task_id}.json", {
            "status": "COMPLETE_TASK", "phase": "g2_greedy", "task_id": task_id,
            "output_ids": [row["output_id"] for row in by_task[task_id]],
            "completed_cells": task_completed_cells, "expected_cells": task_expected_cells,
            "gold_accessed_during_generation": False,
        })
    if args.task_shard_count > 1:
        completed_shard_cells = sum(
            _cell_path(args.scratch, "g2_greedy", task_id, int(row["output_index"]), 0, view).is_file()
            for task_id in task_ids for row in by_task[task_id] for view in CURRENT_VIEWS
        )
        if completed_shard_cells != expected_shard_cells:
            raise RuntimeError(f"g2_greedy shard {args.task_shard_index}/{args.task_shard_count} incomplete {completed_shard_cells}/{expected_shard_cells}")
        atomic_json(args.scratch / "g2_greedy" / f"shard-{args.task_shard_index:02d}-of-{args.task_shard_count:02d}.json", {
            "status": "COMPLETE_SHARD", "phase": "g2_greedy",
            "task_shard_index": args.task_shard_index, "task_shard_count": args.task_shard_count,
            "task_ids": task_ids, "completed_cells": completed_shard_cells,
            "expected_cells": expected_shard_cells, "wall_seconds": time.perf_counter() - started,
            "gold_accessed_during_generation": False,
        })
        return
    cells = _iter_phase_cells(args.scratch, "g2_greedy")
    expected = len(misses) * len(CURRENT_VIEWS)
    if len(cells) != expected:
        raise RuntimeError(f"g2_greedy incomplete {len(cells)}/{expected}")
    atomic_json(args.scratch / "g2_greedy" / "status.json", {
        "status": "COMPLETE_GENERATION_GOLD_UNREAD", "cells": len(cells), "expected_cells": expected,
        "wall_seconds": time.perf_counter() - started,
    })


def run_g2_finalize(args: argparse.Namespace) -> None:
    """Attach Gold only after both G2 surfaces are fully frozen and report G2."""
    misses = prepare_miss_set(args)
    expected_g1 = len(misses) * len(DEPTHS) * len(CURRENT_VIEWS)
    expected_g2 = len(misses) * len(CURRENT_VIEWS)
    g1 = _iter_phase_cells(args.scratch, "g1")
    g2 = _iter_phase_cells(args.scratch, "g2")
    greedy = _iter_phase_cells(args.scratch, "g2_greedy")
    if len(g1) != expected_g1:
        raise RuntimeError(f"frozen G1-half inventory mismatch {len(g1)}/{expected_g1}")
    if len(g2) != expected_g2:
        raise RuntimeError(f"TTT0 incremental-Gold inventory mismatch {len(g2)}/{expected_g2}")
    if len(greedy) != expected_g2:
        raise RuntimeError(f"TTT0 Greedy inventory mismatch {len(greedy)}/{expected_g2}")
    if any(str(cell.get("scoring_path")) != "INCREMENTAL_KV_REPLAY" for cell in g1 + g2):
        raise RuntimeError("G2 requires G0.5 incremental KV replay for every Gold-path cell")
    wanted = {row["output_id"] for row in misses}
    if {row.get("output_id") for row in g1 + g2 + greedy} != wanted:
        raise RuntimeError("G2 output inventory does not equal frozen G1-half cohort")

    g2_export = [{key: value for key, value in _g2_cell_metrics(cell).items() if key != "token_trace"} for cell in g2]
    write_csv(args.report_dir / "g2_ttt0_goldpath_cells.csv", g2_export)

    by_output: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for cell in g1 + g2:
        by_output[str(cell["output_id"])][int(cell["depth"])].append(cell)
    depth_rows: list[dict[str, Any]] = []
    output_rows: list[dict[str, Any]] = []
    classification_counts: Counter[str] = Counter()
    depth_win_counts = {
        "nll_per_token": Counter(), "prune_fraction": Counter(), "mean_legal_rank": Counter(),
    }
    tie_outputs = {"nll_per_token": 0, "prune_fraction": 0, "mean_legal_rank": 0}
    for output_id in sorted(wanted):
        states = {depth: _state_summary(by_output[output_id].get(depth, []), output_id=output_id, depth=depth) for depth in (0, *DEPTHS)}
        for depth, state in states.items():
            depth_rows.append({
                "output_id": output_id, "depth": depth,
                "best_final_gold_nll": state["best_final_gold_nll"],
                "best_final_gold_nll_views": state["best_final_gold_nll_views"],
                "best_nll_per_token": state["best_nll_per_token"],
                "best_nll_per_token_views": state["best_nll_per_token_views"],
                "latest_prune_fraction": state["latest_prune_fraction"],
                "latest_prune_fraction_views": state["latest_prune_fraction_views"],
                "lowest_mean_legal_rank": state["lowest_mean_legal_rank"],
                "lowest_mean_legal_rank_views": state["lowest_mean_legal_rank_views"],
                "any_strict_searchable": state["any_strict_searchable"],
                "any_v5_local_searchable": state["any_v5_local_searchable"],
            })
        metric_specs = (
            ("nll_per_token", "best_nll_per_token", False),
            ("prune_fraction", "latest_prune_fraction", True),
            ("mean_legal_rank", "lowest_mean_legal_rank", False),
        )
        winners: dict[str, list[int]] = {}
        for label, key, higher in metric_specs:
            values = {depth: float(state[key]) for depth, state in states.items()}
            best = max(values.values()) if higher else min(values.values())
            winners[label] = [depth for depth in sorted(values) if values[depth] == best]
            depth_win_counts[label].update(winners[label])
            tie_outputs[label] += int(len(winners[label]) > 1)
        ttt0, adapted = states[0], [states[12], states[24], states[48]]
        adapted_any_v5 = any(state["any_v5_local_searchable"] for state in adapted)
        if ttt0["any_v5_local_searchable"] and not adapted_any_v5:
            classification = "TTT_HARMFUL_STRONG"
        elif not ttt0["any_v5_local_searchable"] and adapted_any_v5:
            classification = "TTT_HELPFUL_STRONG"
        elif _same_cell_harmful_soft(ttt0, adapted):
            classification = "TTT_HARMFUL_SOFT"
        elif _same_cell_helpful_soft(ttt0, adapted):
            classification = "TTT_HELPFUL_SOFT"
        elif all(len(winners[label]) == 1 for label in winners) and len({winners[label][0] for label in winners}) > 1:
            classification = "TTT_DEPTH_SENSITIVE"
        else:
            classification = "NO_CLEAR_TTT_EFFECT"
        classification_counts[classification] += 1
        output_rows.append({
            "output_id": output_id, "g2_ttt_class": classification,
            "best_nll_per_token_depths": ",".join(map(str, winners["nll_per_token"])),
            "best_prune_fraction_depths": ",".join(map(str, winners["prune_fraction"])),
            "best_mean_legal_rank_depths": ",".join(map(str, winners["mean_legal_rank"])),
            **{f"ttt{depth}_{key}": value for depth, state in states.items() for key, value in (
                ("best_final_gold_nll", state["best_final_gold_nll"]),
                ("best_nll_per_token", state["best_nll_per_token"]),
                ("latest_prune_fraction", state["latest_prune_fraction"]),
                ("lowest_mean_legal_rank", state["lowest_mean_legal_rank"]),
                ("any_strict_searchable", state["any_strict_searchable"]),
                ("any_v5_local_searchable", state["any_v5_local_searchable"]),
            )},
        })
    write_csv(args.report_dir / "g2_depth_comparison.csv", depth_rows)
    write_csv(args.report_dir / "g2_output_attribution.csv", output_rows)

    # Only now does the Greedy evaluator open solutions.  It does not alter the
    # frozen candidate JSON cells; labels exist solely in this final report CSV.
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    greedy_export: list[dict[str, Any]] = []
    greedy_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in greedy:
        gold = output_gold(solutions, str(cell["task_id"]), int(cell["output_index"]))
        labelled = {
            **{key: value for key, value in cell.items() if key != "token_telemetry"},
            "gold_exact": bool(cell.get("valid_grid")) and cell.get("canonical_candidate") == gold,
            "gold_attached_post_generation": True,
        }
        greedy_export.append(labelled); greedy_by_output[str(cell["output_id"])].append(labelled)
    write_csv(args.report_dir / "g2_ttt0_greedy_cells.csv", greedy_export)
    rescues: list[dict[str, Any]] = []
    for output_id in sorted(wanted):
        exact = [cell for cell in greedy_by_output[output_id] if _bool(cell["gold_exact"])]
        rescues.append({
            "output_id": output_id, "ttt0_greedy_exact_output": bool(exact),
            "ttt0_greedy_exact_cell_count": len(exact),
            "exact_views": _view_names(exact) if exact else "",
            "current_union_greedy_v5_hit": False,
            "new_rescue_vs_current_union": bool(exact),
        })
    write_csv(args.report_dir / "g2_ttt0_rescues.csv", rescues)
    strict_outputs = sum(bool(states[0]["any_strict_searchable"]) for states in (
        {depth: _state_summary(by_output[output_id].get(depth, []), output_id=output_id, depth=depth) for depth in (0, *DEPTHS)} for output_id in sorted(wanted)
    ))
    v5_outputs = sum(bool(states[0]["any_v5_local_searchable"]) for states in (
        {depth: _state_summary(by_output[output_id].get(depth, []), output_id=output_id, depth=depth) for depth in (0, *DEPTHS)} for output_id in sorted(wanted)
    ))
    rescue_ids = [row["output_id"] for row in rescues if _bool(row["new_rescue_vs_current_union"])]
    adapted_best_rank_one = sum(
        any(float(state["lowest_mean_legal_rank"]) <= 1.0 for state in (
            _state_summary(by_output[output_id].get(depth, []), output_id=output_id, depth=depth) for depth in DEPTHS
        )) for output_id in sorted(wanted)
    )
    harmful = classification_counts["TTT_HARMFUL_STRONG"] + classification_counts["TTT_HARMFUL_SOFT"]
    if rescue_ids or harmful >= 0.20 * len(wanted):
        next_step, bottleneck = "INVESTIGATE_SHALLOW_OR_ADAPTIVE_TTT", "TTT_BOTTLENECK"
    elif adapted_best_rank_one > 0:
        next_step, bottleneck = "DECODER_PRUNING_REDESIGN", "DECODER_PRUNING_BOTTLENECK"
    else:
        next_step, bottleneck = "AUGMENTATION_COVERAGE_TEST", "AUGMENTATION_COVERAGE_SUSPECT"
    decision = {
        "status": "COMPLETE", "scope": "NONBLIND_G2_FROZEN_G1_HALF_28_OUTPUTS",
        "outputs": len(wanted), "new_ttt0_goldpath_cells": len(g2), "new_ttt0_greedy_cells": len(greedy),
        "ttt0_strict_searchable_outputs": strict_outputs, "ttt0_v5_searchable_outputs": v5_outputs,
        "best_by_nll_per_token": {f"TTT{depth}": depth_win_counts["nll_per_token"][depth] for depth in (0, 12, 24, 48)},
        "best_by_prune_fraction": {f"TTT{depth}": depth_win_counts["prune_fraction"][depth] for depth in (0, 12, 24, 48)},
        "best_by_mean_legal_rank": {f"TTT{depth}": depth_win_counts["mean_legal_rank"][depth] for depth in (0, 12, 24, 48)},
        "metric_tie_output_counts": tie_outputs,
        "classification_counts": dict(sorted(classification_counts.items())),
        "ttt0_greedy_exact_cells": sum(int(row["ttt0_greedy_exact_cell_count"]) for row in rescues),
        "ttt0_greedy_exact_outputs": len(rescue_ids), "ttt0_new_rescue_ids": rescue_ids,
        "current_development_union": "33/89", "new_development_union": f"{33 + len(rescue_ids)}/89",
        "adapted_rank_one_outputs": adapted_best_rank_one,
        "next_recommended_experiment": next_step, "main_g2_conclusion": bottleneck,
        "gold_path": "INCREMENTAL_KV_REPLAY", "scientific_config_changed": False,
        "g1_remaining_run": False, "v5_rerun": False, "g3_run": False, "g4_run": False,
    }
    atomic_json(args.report_dir / "G2_DECISION.json", decision)
    report = ["# G2 TTT0 hard-miss attribution", "", "NONBLIND_MECHANISM_DIAGNOSTIC", "",
              "Gold scoring: G0.5-validated INCREMENTAL_KV_REPLAY only.",
              "TTT0 Greedy Gold labels were attached only after its candidate cells were frozen.", "",
              f"Outputs: {len(wanted)}; TTT0 Gold cells: {len(g2)}; TTT0 Greedy cells: {len(greedy)}",
              f"TTT0 strict-searchable outputs: {strict_outputs}", f"TTT0 V5-searchable outputs: {v5_outputs}",
              f"TTT0 Greedy exact outputs: {len(rescue_ids)}", f"New rescue IDs: {rescue_ids}",
              f"Current development union: 33/89; new nonblind development union: {33 + len(rescue_ids)}/89", "",
              "## Depth win counts (ties count for each co-winner)"]
    for label, key in (("NLL/token", "nll_per_token"), ("prune fraction", "prune_fraction"), ("mean legal rank", "mean_legal_rank")):
        report.append(f"- {label}: " + ", ".join(f"TTT{depth}={depth_win_counts[key][depth]}" for depth in (0, 12, 24, 48)) + f"; tied outputs={tie_outputs[key]}")
    report += ["", "## Attribution classes"] + [f"- {key}: {value}" for key, value in sorted(classification_counts.items())]
    report += ["", "## Required answers",
               f"1. TTT0 better on hard misses: {harmful} harmful strong/soft outputs; exact TTT0 rescues={len(rescue_ids)}.",
               "2. Deeper TTT systematic worsening: see per-output comparison; no aggregate causal claim is made beyond these nonblind 28 outputs.",
               f"3. TTT12 better than TTT0: inspect metric win counts and per-output classes; TTT12 wins are not collapsed into a fake composite metric.",
               f"4. TTT0 V5-searchable outputs: {v5_outputs}.",
               f"5. TTT0 Greedy direct new exact solves: {len(rescue_ids)}.",
               f"6. Dominant next bottleneck: {bottleneck}; recommended next experiment: {next_step}."]
    (args.report_dir / "G2_TTT_EFFECT_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(args.scratch / "g2" / "finalized.json", decision)


def run_g3_greedy(args: argparse.Namespace) -> None:
    started = time.perf_counter(); _runtime_identity(args)
    misses = prepare_miss_set(args); model, tokenizer, _initial, config, _info = _load_runtime(args)
    from arc.io import load_dataset
    tasks, solutions = load_dataset(args.challenge), json.loads(args.solutions.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses: by_task[row["task_id"]].append(row)
    task_ids = _sharded_task_ids(sorted(by_task), args)
    expected_shard_cells = 0
    for task_id in task_ids:
        for depth in DEPTHS:
            metadata = _adapter_metadata(args.adapters, task_id, depth); common.load_adapter(model=model, metadata=metadata)
            for row in by_task[task_id]:
                target_task = view_task(tasks[task_id], int(row["output_index"])); gold = output_gold(solutions, task_id, int(row["output_index"]))
                for view in D4_VIEWS:
                    expected_shard_cells += 1
                    destination = _cell_path(args.scratch, "g3_greedy", task_id, int(row["output_index"]), depth, view)
                    if destination.is_file(): continue
                    payload = common.greedy_cell(model=model, tokenizer=tokenizer, task=target_task, task_id=task_id, output_index=int(row["output_index"]), depth=depth, view=view, config=config, checkpoint_sha=metadata["checkpoint_sha256"])
                    payload.update(row); payload.update({"phase": "g3_greedy", "gold_exact": payload.get("canonical_candidate") == gold})
                    atomic_json(destination, payload)
        task_expected_cells = len(by_task[task_id]) * len(DEPTHS) * len(D4_VIEWS)
        task_completed_cells = sum(
            _cell_path(args.scratch, "g3_greedy", task_id, int(row["output_index"]), depth, view).is_file()
            for depth in DEPTHS
            for row in by_task[task_id]
            for view in D4_VIEWS
        )
        if task_completed_cells != task_expected_cells:
            raise RuntimeError(f"g3_greedy task checkpoint incomplete {task_id} {task_completed_cells}/{task_expected_cells}")
        atomic_json(args.scratch / "g3_greedy" / "tasks" / f"{task_id}.json", {
            "status": "COMPLETE_TASK", "phase": "g3_greedy", "task_id": task_id,
            "output_ids": [row["output_id"] for row in by_task[task_id]],
            "completed_cells": task_completed_cells, "expected_cells": task_expected_cells,
        })
    if args.task_shard_count > 1:
        completed_shard_cells = sum(
            _cell_path(args.scratch, "g3_greedy", task_id, int(row["output_index"]), depth, view).is_file()
            for task_id in task_ids
            for depth in DEPTHS
            for row in by_task[task_id]
            for view in D4_VIEWS
        )
        if completed_shard_cells != expected_shard_cells:
            raise RuntimeError(
                f"g3_greedy shard {args.task_shard_index}/{args.task_shard_count} incomplete "
                f"{completed_shard_cells}/{expected_shard_cells}"
            )
        atomic_json(
            args.scratch / "g3_greedy" / f"shard-{args.task_shard_index:02d}-of-{args.task_shard_count:02d}.json",
            {
                "status": "COMPLETE_SHARD",
                "phase": "g3_greedy",
                "task_shard_index": args.task_shard_index,
                "task_shard_count": args.task_shard_count,
                "task_ids": task_ids,
                "completed_cells": completed_shard_cells,
                "expected_cells": expected_shard_cells,
                "wall_seconds": time.perf_counter() - started,
            },
        )
        return
    cells = _iter_phase_cells(args.scratch, "g3_greedy")
    expected = len(misses) * len(DEPTHS) * len(D4_VIEWS)
    if len(cells) != expected: raise RuntimeError(f"g3 greedy incomplete {len(cells)}/{expected}")
    summaries = [{key: value for key, value in row.items() if key not in {"token_telemetry"}} for row in cells]
    write_csv(args.report_dir / "g3_extra_view_greedy_cells.csv", summaries)
    per_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells: per_output[row["output_id"]].append(row)
    tf_cells = _iter_phase_cells(args.scratch, "g3")
    expected_tf = len(misses) * len(DEPTHS) * len(D4_VIEWS)
    if len(tf_cells) != expected_tf:
        raise RuntimeError(f"g3 incremental audit incomplete {len(tf_cells)}/{expected_tf}")
    tf_by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in tf_cells:
        tf_by_output[row["output_id"]].append(row)
    combined = []
    for output_id, rows in sorted(per_output.items()):
        exact = [row for row in rows if _bool(row.get("gold_exact"))]
        tf_rows = tf_by_output[output_id]
        strict = any(_bool(row.get("strict_public_searchable")) for row in tf_rows)
        v5 = any(_bool(row.get("v5_local_gold_path_survives")) for row in tf_rows)
        classification = "AUGMENTATION_EXACT_RESCUE" if exact else (
            "AUGMENTATION_ACCESSIBILITY_RESCUE" if v5 else "NO_D4_BENEFIT"
        )
        combined.append({
            "output_id": output_id, "extra_d4_exact_rescue": bool(exact),
            "best_new_view": exact[0]["view"] if exact else None,
            "new_view_exact_count": len(exact), "any_d4_strict_searchable": strict,
            "any_d4_v5_searchable": v5, "g3_classification": classification,
        })
    write_csv(args.report_dir / "g3_full_d4_union.csv", combined)
    write_csv(args.report_dir / "g3_extra_view_output_summary.csv", combined)
    view_counts = Counter(row["view"] for row in cells if _bool(row.get("gold_exact")))
    rescue_ids = [row["output_id"] for row in combined if _bool(row["extra_d4_exact_rescue"])]
    accessibility_only = [row["output_id"] for row in combined if row["g3_classification"] == "AUGMENTATION_ACCESSIBILITY_RESCUE"]
    unique_by_view = {
        view: sum(1 for row in combined if row["extra_d4_exact_rescue"] and row["best_new_view"] == view and row["new_view_exact_count"] == 1)
        for view in D4_VIEWS
    }
    report = ["# G3 full D4 representation test", "", "NONBLIND_MECHANISM_DIAGNOSTIC", "",
              f"Exact rescue outputs: {len(rescue_ids)}", f"Accessibility-only outputs: {len(accessibility_only)}",
              f"Development union after G3: {33 + len(rescue_ids)}/89", "", "## Exact rescue IDs", *[f"- {value}" for value in rescue_ids], "", "## Per-view exact cells"]
    report += [f"- {view}: exact_cells={view_counts[view]}, unique_rescue_outputs={unique_by_view[view]}" for view in D4_VIEWS]
    report.append(f"\nWall seconds: {time.perf_counter()-started:.3f}")
    (args.report_dir / "G3_FULL_D4_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    atomic_json(args.scratch / "g3_greedy" / "status.json", {"status": "COMPLETE", "cells": len(cells), "expected_cells": expected, "wall_seconds": time.perf_counter() - started})


G3A_PHASE = "g3a_ttt24_omitted_d4_greedy"


def _g3a_contract(config: dict[str, Any], tokenizer: Any) -> dict[str, Any]:
    """Return only frozen decode-contract fields relevant to G3A identity."""
    return {
        "decoder": "greedy",
        "do_sample": False,
        "return_dict_in_generate": True,
        "output_scores": True,
        "max_new_tokens": int(config["max_new_tokens"]),
        "generation_context_window": int(config["generation_context_window"]),
        "eos_token_id": int(tokenizer.eos_token_id),
        "pad_token_id": int(tokenizer.pad_token_id),
        "views": list(D4_VIEWS),
        "color_offset": 0,
        "pair_order": "canonical",
    }


def _g3a_expected_keys(misses: list[dict[str, Any]]) -> set[tuple[str, int, int, str]]:
    return {
        (str(row["task_id"]), int(row["output_index"]), 24, view)
        for row in misses for view in D4_VIEWS
    }


def _g3a_historical_predictions() -> dict[tuple[str, int, str], list[str]]:
    """Load diagnostic-only hashes from the committed non-exact audit.

    These rows never affect decoding, checkpoint selection, retries, Gold
    labels, or final rescue totals.  They only make agreement transparent.
    """
    path = ROOT / "analysis/historical_eval60_aug8_reuse_v1/hard28_ttt24_omitted_view_greedy.csv"
    if not path.is_file():
        return {}
    result: dict[tuple[str, int, str], list[str]] = defaultdict(list)
    for row in _csv_rows(path):
        if row.get("historical_prediction_available") != "TRUE":
            continue
        if row.get("prediction_sha") in {None, "", "UNAVAILABLE"}:
            continue
        result[(str(row["task_id"]), int(row["output_index"]), str(row["view"]))].append(str(row["prediction_sha"]))
    return result


def run_g3a_prepare(args: argparse.Namespace) -> None:
    """Freeze the exact existing hard28 membership without model/Gold access."""
    misses = prepare_miss_set(args)
    if len(misses) != 28:
        raise RuntimeError(f"G3A requires the frozen 28-output cohort, got {len(misses)}")
    g1_path = args.report_dir.parent / "g1_half_sha256_28of56/g1_output_summary.csv"
    if not g1_path.is_file():
        raise RuntimeError(f"missing frozen G1-half output summary:{g1_path}")
    g1_ids = {str(row["output_id"]) for row in _csv_rows(g1_path)}
    miss_ids = {str(row["output_id"]) for row in misses}
    if g1_ids != miss_ids:
        raise RuntimeError("G3A cohort does not exactly match frozen G1-half membership")
    expected = _g3a_expected_keys(misses)
    payload = {
        "experiment": "G3A_CURRENT_TTT24_OMITTED_D4_GREEDY_HALF28",
        "status": "PREPARED_GOLD_UNREAD",
        "cohort_output_ids": [row["output_id"] for row in misses],
        "cohort_output_count": len(misses),
        "expected_cells": len(expected),
        "depth": 24,
        "views": list(D4_VIEWS),
        "solutions_opened_before_generation": False,
        "g1_membership_verified": True,
        "frozen_miss_manifest_sha256": sha_file(args.frozen_miss_manifest),
    }
    atomic_json(args.report_dir / "G3A_MANIFEST.json", payload)
    atomic_json(args.scratch / G3A_PHASE / "manifest.json", payload)


def run_g3a_greedy(args: argparse.Namespace) -> None:
    """Generate only current-adapter TTT24 omitted-D4 cells, Gold unread."""
    started = time.perf_counter(); identity = _runtime_identity(args)
    misses = prepare_miss_set(args)
    if len(misses) != 28:
        raise RuntimeError(f"G3A requires exactly 28 frozen outputs, got {len(misses)}")
    model, tokenizer, _initial, config, tokenizer_info = _load_runtime(args)
    from arc.io import load_dataset
    tasks = load_dataset(args.challenge)  # Challenge only: no evaluation solutions are opened here.
    historical = _g3a_historical_predictions()
    contract = _g3a_contract(config, tokenizer)
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses:
        by_task[str(row["task_id"])].append(row)
    task_ids = _sharded_task_ids(sorted(by_task), args)
    expected_shard_cells = sum(len(by_task[task_id]) * len(D4_VIEWS) for task_id in task_ids)
    for task_id in task_ids:
        metadata = _adapter_metadata(args.adapters, task_id, 24)
        checkpoint_sha = str(metadata["checkpoint_sha256"])
        if len(checkpoint_sha) != 64:
            raise RuntimeError(f"invalid current TTT24 adapter SHA for {task_id}")
        common.load_adapter(model=model, metadata=metadata)
        for row in by_task[task_id]:
            target_task = view_task(tasks[task_id], int(row["output_index"]))
            for view in D4_VIEWS:
                destination = _cell_path(args.scratch, G3A_PHASE, task_id, int(row["output_index"]), 24, view)
                if destination.is_file():
                    continue
                payload = common.greedy_cell(
                    model=model, tokenizer=tokenizer, task=target_task, task_id=task_id,
                    output_index=int(row["output_index"]), depth=24, view=view,
                    config=config, checkpoint_sha=checkpoint_sha,
                )
                candidate = payload.get("canonical_candidate")
                prediction_sha = (
                    hashlib.sha256(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                    if candidate is not None else "UNAVAILABLE"
                )
                historical_hashes = sorted(set(historical.get((task_id, int(row["output_index"]), view), [])))
                payload.update({
                    **row,
                    "phase": G3A_PHASE,
                    "adapter_checkpoint_path": str(metadata["checkpoint_path"]),
                    "adapter_checkpoint_sha256": checkpoint_sha,
                    "generation_contract": contract,
                    "generation_contract_sha256": contract_sha,
                    "canonical_prediction": candidate,
                    "prediction_sha256": prediction_sha,
                    "parse_valid": bool(payload.get("valid_grid")),
                    "gold_attached": False,
                    "solutions_opened_before_generation": False,
                    "historical_matching_prediction_sha256": ";".join(historical_hashes) if historical_hashes else "UNAVAILABLE",
                    "historical_prediction_same_as_current": (
                        "UNAVAILABLE" if not historical_hashes or prediction_sha == "UNAVAILABLE"
                        else str(prediction_sha in historical_hashes).upper()
                    ),
                })
                atomic_json(destination, payload)
        expected_task_cells = len(by_task[task_id]) * len(D4_VIEWS)
        complete_task_cells = sum(
            _cell_path(args.scratch, G3A_PHASE, task_id, int(row["output_index"]), 24, view).is_file()
            for row in by_task[task_id] for view in D4_VIEWS
        )
        if complete_task_cells != expected_task_cells:
            raise RuntimeError(f"G3A incomplete task checkpoint {task_id}: {complete_task_cells}/{expected_task_cells}")
        atomic_json(args.scratch / G3A_PHASE / "tasks" / f"{task_id}.json", {
            "status": "COMPLETE_TASK_GOLD_UNREAD", "task_id": task_id,
            "completed_cells": complete_task_cells, "expected_cells": expected_task_cells,
            "adapter_checkpoint_sha256": checkpoint_sha,
            "solutions_opened_before_generation": False,
        })
    complete_shard_cells = sum(
        _cell_path(args.scratch, G3A_PHASE, task_id, int(row["output_index"]), 24, view).is_file()
        for task_id in task_ids for row in by_task[task_id] for view in D4_VIEWS
    )
    if complete_shard_cells != expected_shard_cells:
        raise RuntimeError(f"G3A shard incomplete {complete_shard_cells}/{expected_shard_cells}")
    atomic_json(args.scratch / G3A_PHASE / f"shard-{args.task_shard_index:02d}-of-{args.task_shard_count:02d}.json", {
        "status": "COMPLETE_SHARD_GOLD_UNREAD", "task_shard_index": args.task_shard_index,
        "task_shard_count": args.task_shard_count, "task_ids": task_ids,
        "completed_cells": complete_shard_cells, "expected_cells": expected_shard_cells,
        "wall_seconds": time.perf_counter() - started, "runtime_identity": identity,
        "tokenizer": tokenizer_info, "generation_contract_sha256": contract_sha,
        "solutions_opened_before_generation": False,
    })


def _g3a_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    return _iter_phase_cells(args.scratch, G3A_PHASE)


def run_g3a_freeze(args: argparse.Namespace) -> None:
    """Validate and mark all 112 prediction cells frozen before Gold access."""
    misses = prepare_miss_set(args)
    expected = _g3a_expected_keys(misses)
    cells = _g3a_cells(args)
    keys = {(str(cell.get("task_id")), int(cell.get("output_index")), int(cell.get("depth")), str(cell.get("view"))) for cell in cells}
    if len(cells) != 112 or keys != expected or len(keys) != 112:
        raise RuntimeError(f"G3A generation inventory mismatch cells={len(cells)} keys={len(keys)} expected={len(expected)}")
    required = {
        "adapter_checkpoint_path", "adapter_checkpoint_sha256", "generation_contract_sha256",
        "prediction_sha256", "parse_valid", "canonical_prediction", "gold_attached",
        "solutions_opened_before_generation",
    }
    for cell in cells:
        missing = required - set(cell)
        if missing or cell.get("gold_attached") is not False or cell.get("solutions_opened_before_generation") is not False:
            raise RuntimeError(f"G3A target-blind cell contract failure {cell.get('output_id')}:{missing}")
        if int(cell["depth"]) != 24 or str(cell["view"]) not in D4_VIEWS:
            raise RuntimeError("G3A cell surface drift")
        if len(str(cell["adapter_checkpoint_sha256"])) != 64:
            raise RuntimeError("G3A adapter SHA unavailable")
    frozen_sha = hashlib.sha256(json.dumps(cells, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    flag = args.scratch / G3A_PHASE / "G3A_GENERATION_FROZEN.flag"
    atomic_json(flag, {
        "status": "FROZEN_GOLD_UNREAD", "cells": len(cells), "unique_keys": len(keys),
        "cell_inventory_sha256": frozen_sha, "solutions_opened_before_generation": False,
    })
    provenance = {
        "experiment": "G3A_CURRENT_TTT24_OMITTED_D4_GREEDY_HALF28",
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "generation_frozen": True, "solutions_opened_before_generation": False,
        "current_adapter_sha_verified": True, "adapter_manifest_sha256": sha_file(args.adapters.parent / "checkpoint_manifest.csv"),
        "cell_inventory_sha256": frozen_sha, "cells": len(cells), "views": list(D4_VIEWS), "depth": 24,
        "gpu_used": True, "new_ttt": False, "new_dfs": False, "gold_path_scoring": False,
    }
    atomic_json(args.report_dir / "G3A_PROVENANCE.json", provenance)


def _g3a_runtime_incidents(scratch: Path) -> list[dict[str, str]]:
    """Retain launch faults in the post-freeze report instead of masking retries."""
    markers = (
        ("traceback", "TRACEBACK"),
        ("cuda out of memory", "OOM"),
        ("non-finite", "NONFINITE"),
    )
    incidents: list[dict[str, str]] = []
    for path in sorted((scratch / "logs").glob("*.log")):
        text = path.read_text(encoding="utf-8", errors="replace").lower()
        for needle, kind in markers:
            if needle in text:
                incidents.append({"log": path.name, "kind": kind})
    return incidents


def run_g3a_finalize(args: argparse.Namespace) -> None:
    """Attach exact Gold labels only to an already frozen G3A inventory."""
    flag = args.scratch / G3A_PHASE / "G3A_GENERATION_FROZEN.flag"
    if not flag.is_file():
        raise RuntimeError("G3A Gold attachment forbidden before generation freeze")
    freeze = read_json(flag)
    if freeze.get("status") != "FROZEN_GOLD_UNREAD" or int(freeze.get("cells", -1)) != 112:
        raise RuntimeError("invalid G3A generation freeze flag")
    misses = prepare_miss_set(args)
    cells = _g3a_cells(args)
    if len(cells) != 112:
        raise RuntimeError("G3A cells changed after freeze")
    solutions = json.loads(args.solutions.read_text(encoding="utf-8"))
    labelled: list[dict[str, Any]] = []
    by_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        gold = output_gold(solutions, str(cell["task_id"]), int(cell["output_index"]))
        row = {
            **{key: value for key, value in cell.items() if key != "token_telemetry"},
            "exact_gold_hit": bool(cell.get("parse_valid")) and cell.get("canonical_prediction") == gold,
            "gold_attached_post_generation": True,
        }
        labelled.append(row); by_output[str(row["output_id"])].append(row)
    write_csv(args.report_dir / "g3a_greedy_cells.csv", labelled)
    rescues: list[dict[str, Any]] = []
    for miss in sorted(misses, key=lambda row: str(row["output_id"])):
        output_id = str(miss["output_id"]); rows = by_output[output_id]
        if len(rows) != len(D4_VIEWS):
            raise RuntimeError(f"G3A incomplete output surface {output_id}")
        exact = [row for row in rows if _bool(row["exact_gold_hit"])]
        exact_views = sorted(str(row["view"]) for row in exact)
        rescues.append({
            "task_id": miss["task_id"], "output_id": output_id, "output_index": miss["output_index"],
            "current_union_hit": False, "new_greedy_rescue": bool(exact),
            "exact_cell_count": len(exact), "exact_views": ";".join(exact_views),
        })
    write_csv(args.report_dir / "g3a_output_rescues.csv", rescues)
    rescue_ids = [row["output_id"] for row in rescues if _bool(row["new_greedy_rescue"])]
    view_rows: list[dict[str, Any]] = []
    for view in D4_VIEWS:
        exact_cells = [row for row in labelled if row["view"] == view and _bool(row["exact_gold_hit"])]
        rescued_outputs = {row["output_id"] for row in exact_cells}
        unique = {
            row["output_id"] for row in exact_cells
            if len([x for x in by_output[row["output_id"]] if _bool(x["exact_gold_hit"])]) == 1
        }
        view_rows.append({"view": view, "exact_cells": len(exact_cells), "rescue_outputs": len(rescued_outputs), "uniquely_rescued_outputs": len(unique), "unique_rescue_ids": ";".join(sorted(unique))})
    write_csv(args.report_dir / "g3a_view_contributions.csv", view_rows)
    agreement: list[dict[str, Any]] = []
    for view in D4_VIEWS:
        view_cells = [row for row in labelled if row["view"] == view]
        comparable = [row for row in view_cells if row.get("historical_prediction_same_as_current") != "UNAVAILABLE"]
        identical = [row for row in comparable if row.get("historical_prediction_same_as_current") == "TRUE"]
        agreement.append({"view": view, "current_cells": len(view_cells), "historical_comparable_cells": len(comparable), "identical_predictions": len(identical), "different_predictions": len(comparable) - len(identical)})
    write_csv(args.report_dir / "g3a_historical_prediction_agreement.csv", agreement)
    conclusion = "AUGMENTATION_COVERAGE_SUPPORTED" if rescue_ids else "TTT24_OMITTED_D4_GREEDY_NO_GAIN"
    next_step = "TTT12_OMITTED_D4_SMALL_CONTROL" if rescue_ids else "DECODER_PRUNING_REDESIGN"
    runtime_incidents = _g3a_runtime_incidents(args.scratch)
    status = "PASS" if not runtime_incidents else "PARTIAL"
    decision = {
        "G3A_STATUS": status, "COHORT_OUTPUTS": len(misses), "EXPECTED_CELLS": 112,
        "GREEDY_CELLS_COMPLETE": f"{len(labelled)}/112", "CURRENT_ADAPTER_SHA_VERIFIED": "YES",
        "NEW_GREEDY_EXACT_CELLS": sum(_bool(row["exact_gold_hit"]) for row in labelled),
        "NEW_GREEDY_RESCUE_OUTPUTS": len(rescue_ids), "NEW_GREEDY_RESCUE_IDS": rescue_ids,
        "BY_VIEW": {row["view"]: {"exact_cells": row["exact_cells"], "rescue_outputs": row["rescue_outputs"]} for row in view_rows},
        "UNIQUE_RESCUES_BY_VIEW": {row["view"]: row["uniquely_rescued_outputs"] for row in view_rows},
        "HISTORICAL_COMPARABLE_CELLS": sum(row["historical_comparable_cells"] for row in agreement),
        "HISTORICAL_IDENTICAL_PREDICTIONS": sum(row["identical_predictions"] for row in agreement),
        "HISTORICAL_DIFFERENT_PREDICTIONS": sum(row["different_predictions"] for row in agreement),
        "CURRENT_DEVELOPMENT_UNION": "33/89", "NEW_NONBLIND_DEVELOPMENT_UNION": f"{33 + len(rescue_ids)}/89",
        "AUGMENTATION_COVERAGE_CONCLUSION": conclusion, "NEXT_RECOMMENDED_EXPERIMENT": next_step,
        "RUNTIME_INCIDENTS": runtime_incidents,
        "GPU_USED": "YES", "NEW_TTT": "NO", "NEW_DFS": "NO", "GOLD_PATH_SCORING": "NO", "SCIENTIFIC_CONFIG_CHANGED": "NO",
    }
    atomic_json(args.report_dir / "G3A_DECISION.json", decision)
    report = ["# G3A current TTT24 omitted-D4 Greedy coverage", "", "NONBLIND_DEVELOPMENT_MEASUREMENT", "",
              "Predictions were atomically frozen before Gold was opened.",
              f"- Cells: {len(labelled)}/112", f"- New exact rescue outputs: {len(rescue_ids)}", f"- New rescue IDs: {rescue_ids}",
              f"- Development union: {33 + len(rescue_ids)}/89", f"- Conclusion: {conclusion}", f"- Run status: {status}",
              f"- Runtime incidents: {runtime_incidents}", "", "## View contributions"]
    report += [f"- {row['view']}: exact cells={row['exact_cells']}; rescue outputs={row['rescue_outputs']}; unique rescues={row['uniquely_rescued_outputs']}" for row in view_rows]
    report += ["", "## Historical prediction agreement (diagnostic only)"]
    report += [f"- {row['view']}: comparable={row['historical_comparable_cells']}; identical={row['identical_predictions']}; different={row['different_predictions']}" for row in agreement]
    (args.report_dir / "G3A_CURRENT_AUGMENTATION_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def g4_discovery(args: argparse.Namespace) -> None:
    candidates = []
    for root in (args.global_root / "models", args.global_root / "assets", args.global_root / "source"):
        if root.exists():
            for path in sorted(root.rglob("model_manifest.json")):
                try: manifest = read_json(path)
                except Exception: continue
                identifier = str(manifest.get("model_identifier", ""))
                if "qwen" in identifier.lower() and "4b" not in identifier.lower():
                    candidates.append({"manifest": str(path), "model_identifier": identifier, "status": "DISCOVERED_NEEDS_TOKENIZER_GATE"})
    payload = {"status": "BLOCKED_NO_COMPATIBLE_MODEL" if not candidates else "CANDIDATES_REQUIRE_GATE", "candidates": candidates,
               "note": "No model is downloaded or used until exact native-tokenizer compatibility is proven."}
    atomic_json(args.report_dir / "g4_model_discovery.json", payload)


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser()
    item.add_argument("mode", choices=("prepare", "g0", "g1", "g1-half-finalize", "g2", "g2-greedy", "g2-finalize", "g3-tf", "g3-greedy", "g3a-prepare", "g3a-greedy", "g3a-freeze", "g3a-finalize", "g4-discover"))
    item.add_argument("--scratch", type=Path, required=True)
    item.add_argument("--report-dir", type=Path, required=True)
    item.add_argument("--compact", type=Path, required=True)
    item.add_argument("--frozen-miss-manifest", type=Path)
    item.add_argument("--global-root", type=Path, default=Path("/workspace/arc2"))
    item.add_argument("--adapters", type=Path, required=True)
    item.add_argument("--challenge", type=Path, required=True)
    item.add_argument("--solutions", type=Path, required=True)
    item.add_argument("--runtime-config", type=Path, required=True)
    item.add_argument("--v5-config", type=Path, required=True)
    item.add_argument("--model-path", type=Path, required=True)
    item.add_argument("--native-config-dir", type=Path, required=True)
    item.add_argument("--ptxas", type=Path, required=True)
    item.add_argument("--greedy-parquet", type=Path, required=True)
    item.add_argument("--g0-logprob-tolerance", type=float, default=0.05)
    item.add_argument("--task-shard-index", type=int, default=0,
                      help="Deterministic task shard index; scheduling only.")
    item.add_argument("--task-shard-count", type=int, default=1,
                      help="Total deterministic task shards; scheduling only.")
    item.add_argument("--max-tasks-per-shard", type=int,
                      help="Optional scheduling-only bounded smoke prefix; written cells remain part of the frozen cohort.")
    item.add_argument("--cohort-output-count", type=int,
                      help="Deterministic SHA256 output-prefix size; diagnostic stop gate only.")
    return item


def main() -> None:
    args = parser().parse_args(); args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare": prepare_miss_set(args)
    elif args.mode == "g0": run_g0(args)
    elif args.mode == "g1": run_gold_surface(args, phase="g1", depths=DEPTHS, views=CURRENT_VIEWS)
    elif args.mode == "g1-half-finalize": run_g1_half_gate(args)
    elif args.mode == "g2":
        run_gold_surface(args, phase="g2", depths=(0,), views=CURRENT_VIEWS, use_initial_adapter=True)
    elif args.mode == "g2-greedy": run_g2_greedy(args)
    elif args.mode == "g2-finalize": run_g2_finalize(args)
    elif args.mode == "g3-tf": run_gold_surface(args, phase="g3", depths=DEPTHS, views=D4_VIEWS)
    elif args.mode == "g3-greedy": run_g3_greedy(args)
    elif args.mode == "g3a-prepare": run_g3a_prepare(args)
    elif args.mode == "g3a-greedy": run_g3a_greedy(args)
    elif args.mode == "g3a-freeze": run_g3a_freeze(args)
    elif args.mode == "g3a-finalize": run_g3a_finalize(args)
    elif args.mode == "g4-discover": g4_discovery(args)


if __name__ == "__main__":
    main()
