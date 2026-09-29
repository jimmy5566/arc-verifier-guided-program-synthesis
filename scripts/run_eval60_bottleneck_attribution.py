#!/usr/bin/env python3
"""Nonblind, resumable Eval60 Gold-path bottleneck attribution.

This program deliberately separates *measurement* from search.  It never
recreates the historical Greedy or V5 candidate pools.  G1/G2 use a single
teacher-forced forward pass for each frozen model state and G3 runs only the
four previously omitted geometry views through the already-frozen Greedy path.

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
    temporary = path.with_name(path.name + ".partial")
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


def _adapter_metadata(adapters: Path, task_id: str, depth: int) -> dict[str, Any]:
    path = adapters / task_id / f"depth_{depth:03d}" / "metadata.json"
    metadata = read_json(path)
    required = {"checkpoint_sha256", "checkpoint_path", "task_id", "depth"}
    if not required.issubset(metadata) or str(metadata["task_id"]) != task_id or int(metadata["depth"]) != depth:
        raise RuntimeError(f"invalid adapter metadata:{path}")
    return metadata


def prepare_miss_set(args: argparse.Namespace) -> list[dict[str, Any]]:
    compact = args.compact.resolve()
    greedy = {row["output_id"]: row for row in _csv_rows(compact / "greedy" / "greedy_outputs.csv")}
    v5 = {row["output_id"]: row for row in _csv_rows(compact / "turbodfs_v5" / "v5_outputs.csv")}
    if len(greedy) != 89 or len(v5) != 89:
        raise RuntimeError(f"compact output inventory mismatch greedy={len(greedy)} v5={len(v5)}")
    misses: list[dict[str, Any]] = []
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
    if len(misses) != 56:
        raise RuntimeError(f"expected exactly 56 mechanical lower-bound misses, got {len(misses)}")
    destination = args.report_dir / "current_union_miss_outputs.csv"
    write_csv(destination, misses)
    atomic_json(args.scratch / "manifest" / "miss_set.json", {
        "status": "FROZEN", "count": len(misses), "source_commit": EXPECTED_COMPACT_COMMIT,
        "rows_sha256": hashlib.sha256(json.dumps(misses, sort_keys=True).encode()).hexdigest(),
    })
    return misses


def _load_runtime(args: argparse.Namespace) -> tuple[Any, Any, Any, dict[str, Any], dict[str, Any]]:
    """Load the exact 4B model once.  Callers must retain this one process."""
    os.environ.update({
        "CUDA_VISIBLE_DEVICES": "0", "TRITON_PTXAS_PATH": str(args.ptxas),
        "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false",
    })
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
        if v5: primary = "CURRENT_STATE_SEARCHABLE"
        elif "TOKEN_BUDGET_BLOCKED" in statuses and statuses == {"TOKEN_BUDGET_BLOCKED"}: primary = "TOKEN_BUDGET_BLOCKED"
        elif "SERIALIZATION_OR_TRANSPORT_FAILURE" in statuses: primary = "SERIALIZATION_OR_TRANSPORT_FAILURE"
        else: primary = "CURRENT_PROBABILITY_STATE_LIMITED"
        output_rows.append({"output_id": output_id, "cells": len(group), "any_strict_searchable": strict, "any_v5_local_searchable": v5,
                            "output_class": primary, "best_final_gold_nll": min(float(x.get("final_gold_nll", float("inf"))) for x in group)})
    prefix = "g1" if phase == "g1" else "g3_extra_view"
    write_csv(args.report_dir / f"{prefix}_token_trace.csv", token_rows)
    write_csv(args.report_dir / f"{prefix}_cell_summary.csv", summaries)
    write_csv(args.report_dir / ("g1_output_summary.csv" if phase == "g1" else "g3_extra_view_output_summary.csv"), output_rows)


def run_gold_surface(args: argparse.Namespace, *, phase: str, depths: tuple[int, ...], views: tuple[str, ...], use_initial_adapter: bool = False) -> None:
    started = time.perf_counter(); _runtime_identity(args)
    misses = prepare_miss_set(args)
    model, tokenizer, initial, config, _info = _load_runtime(args)
    from arc.io import load_dataset
    tasks, solutions = load_dataset(args.challenge), json.loads(args.solutions.read_text(encoding="utf-8"))
    v5_config = read_json(args.v5_config); max_score, max_tokens = float(v5_config["max_score"]), int(v5_config["max_new_tokens"])
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses: by_task[row["task_id"]].append(row)
    for task_id in sorted(by_task):
        for depth in depths:
            if use_initial_adapter:
                restore_adapter(model, initial); checkpoint_sha = "INITIAL_TTT0"
            else:
                metadata = _adapter_metadata(args.adapters, task_id, depth); common.load_adapter(model=model, metadata=metadata); checkpoint_sha = metadata["checkpoint_sha256"]
            for row in by_task[task_id]:
                target_task = view_task(tasks[task_id], int(row["output_index"])); gold = output_gold(solutions, task_id, int(row["output_index"]))
                for view in views:
                    destination = _cell_path(args.scratch, phase, task_id, int(row["output_index"]), depth, view)
                    if destination.is_file(): continue
                    began = time.perf_counter()
                    summary, traces = teacher_force(model=model, tokenizer=tokenizer, task=target_task, view=view, config=config, target_grid=gold,
                                                    max_score=max_score, max_new_tokens=max_tokens)
                    payload = {**row, **summary, "phase": phase, "depth": depth, "view": view, "checkpoint_sha256": checkpoint_sha,
                               "wall_seconds": time.perf_counter() - began, "token_trace": traces}
                    atomic_json(destination, payload)
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


def run_g3_greedy(args: argparse.Namespace) -> None:
    started = time.perf_counter(); _runtime_identity(args)
    misses = prepare_miss_set(args); model, tokenizer, _initial, config, _info = _load_runtime(args)
    from arc.io import load_dataset
    tasks, solutions = load_dataset(args.challenge), json.loads(args.solutions.read_text(encoding="utf-8"))
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in misses: by_task[row["task_id"]].append(row)
    for task_id in sorted(by_task):
        for depth in DEPTHS:
            metadata = _adapter_metadata(args.adapters, task_id, depth); common.load_adapter(model=model, metadata=metadata)
            for row in by_task[task_id]:
                target_task = view_task(tasks[task_id], int(row["output_index"])); gold = output_gold(solutions, task_id, int(row["output_index"]))
                for view in D4_VIEWS:
                    destination = _cell_path(args.scratch, "g3_greedy", task_id, int(row["output_index"]), depth, view)
                    if destination.is_file(): continue
                    payload = common.greedy_cell(model=model, tokenizer=tokenizer, task=target_task, task_id=task_id, output_index=int(row["output_index"]), depth=depth, view=view, config=config, checkpoint_sha=metadata["checkpoint_sha256"])
                    payload.update(row); payload.update({"phase": "g3_greedy", "gold_exact": payload.get("canonical_candidate") == gold})
                    atomic_json(destination, payload)
    cells = _iter_phase_cells(args.scratch, "g3_greedy")
    expected = len(misses) * len(DEPTHS) * len(D4_VIEWS)
    if len(cells) != expected: raise RuntimeError(f"g3 greedy incomplete {len(cells)}/{expected}")
    summaries = [{key: value for key, value in row.items() if key not in {"token_telemetry"}} for row in cells]
    write_csv(args.report_dir / "g3_extra_view_greedy_cells.csv", summaries)
    per_output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cells: per_output[row["output_id"]].append(row)
    combined = []
    for output_id, rows in sorted(per_output.items()):
        exact = [row for row in rows if _bool(row.get("gold_exact"))]
        combined.append({"output_id": output_id, "extra_d4_exact_rescue": bool(exact), "best_new_view": exact[0]["view"] if exact else None,
                         "new_view_exact_count": len(exact)})
    write_csv(args.report_dir / "g3_full_d4_union.csv", combined)
    write_csv(args.report_dir / "g3_extra_view_output_summary.csv", combined)
    view_counts = Counter(row["view"] for row in cells if _bool(row.get("gold_exact")))
    (args.report_dir / "G3_FULL_D4_REPORT.md").write_text("# G3 full D4 representation test\n\n" + "\n".join(f"- {view}: {view_counts[view]} exact cells" for view in D4_VIEWS) + f"\n\nWall seconds: {time.perf_counter()-started:.3f}\n", encoding="utf-8")
    atomic_json(args.scratch / "g3_greedy" / "status.json", {"status": "COMPLETE", "cells": len(cells), "expected_cells": expected, "wall_seconds": time.perf_counter() - started})


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
    item.add_argument("mode", choices=("prepare", "g0", "g1", "g2", "g3-tf", "g3-greedy", "g4-discover"))
    item.add_argument("--scratch", type=Path, required=True)
    item.add_argument("--report-dir", type=Path, required=True)
    item.add_argument("--compact", type=Path, required=True)
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
    return item


def main() -> None:
    args = parser().parse_args(); args.report_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "prepare": prepare_miss_set(args)
    elif args.mode == "g0": run_g0(args)
    elif args.mode == "g1": run_gold_surface(args, phase="g1", depths=DEPTHS, views=CURRENT_VIEWS)
    elif args.mode == "g2":
        run_gold_surface(args, phase="g2", depths=(0,), views=CURRENT_VIEWS, use_initial_adapter=True); run_g2_summary(args)
    elif args.mode == "g3-tf": run_gold_surface(args, phase="g3", depths=DEPTHS, views=D4_VIEWS)
    elif args.mode == "g3-greedy": run_g3_greedy(args)
    elif args.mode == "g4-discover": g4_discovery(args)


if __name__ == "__main__":
    main()
