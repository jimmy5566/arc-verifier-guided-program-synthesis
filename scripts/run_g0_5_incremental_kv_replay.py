#!/usr/bin/env python3
"""G0.5: compare historical greedy telemetry with two frozen-token replay paths.

This is a diagnostic-only companion to ``run_eval60_bottleneck_attribution``.
It never calls ``generate``, TTT, or TurboDFS.  For the six cells selected by
the already-frozen G0 CSV it scores the *existing* continuation in two ways:

* a one-token-at-a-time ``past_key_values`` replay, matching autoregressive
  generation's cache shape; and
* the original G0 full-sequence teacher-forced forward pass.

The reports intentionally keep only compact per-token native-vocabulary
statistics.  The model outputs, caches, adapters, and candidate pools remain
Pod-local and immutable.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_eval60_bottleneck_attribution import (
    LEGAL_ARC_TOKENS,
    _adapter_metadata,
    _encoded_view,
    _load_runtime,
    _runtime_identity,
)


def _bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    partial.replace(path)


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    values = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        path.write_text("", encoding="utf-8")
        return
    fields = sorted({key for row in values for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(values)


def _native_map(value: Any) -> dict[int, float]:
    """Normalise the historical compact native distribution representation."""
    if not isinstance(value, list):
        raise RuntimeError("missing historical full_native_logprobs")
    result: dict[int, float] = {}
    for entry in value:
        if not isinstance(entry, dict) or "token_id" not in entry or "logprob" not in entry:
            raise RuntimeError("malformed historical full_native_logprobs")
        result[int(entry["token_id"])] = float(entry["logprob"])
    expected = set(LEGAL_ARC_TOKENS)
    if set(result) != expected:
        raise RuntimeError(f"historical native token inventory mismatch: {sorted(result)}")
    return result


def _rank(probabilities: dict[int, float], token: int) -> int:
    ordered = sorted(probabilities.items(), key=lambda item: (-item[1], item[0]))
    return 1 + next(index for index, (candidate, _value) in enumerate(ordered) if candidate == token)


def _decision(distribution: dict[int, float], prefix_nll: float, max_score: float) -> dict[str, Any]:
    """Exact public strict inequality and deterministic V5 frontier-floor rule."""
    successors = sorted(
        ((prefix_nll - value, token) for token, value in distribution.items()),
        key=lambda item: (item[0], item[1]),
    )
    strict = [token for score, token in successors if score < max_score]
    floor = None if strict else successors[0][1]
    return {
        "strict_survivors": strict,
        "frontier_floor_token": floor,
        "classification": "STRICT_SEARCHABLE" if strict else "FLOOR_REQUIRED",
    }


def _distribution_row(*, values: Any, token: int, prefix_nll: float, max_score: float, tokenizer: Any) -> dict[str, Any]:
    """Extract only the fixed native grid vocabulary from logits."""
    import torch

    logprobs = torch.log_softmax(values.float(), dim=-1)
    probs = torch.exp(logprobs)
    native = {candidate: float(logprobs[candidate].item()) for candidate in LEGAL_ARC_TOKENS}
    top_values, top_ids = torch.topk(probs, 2)
    entropy = float((-(probs * logprobs)).sum().item())
    decision = _decision(native, prefix_nll, max_score)
    return {
        "chosen_logprob": float(logprobs[token].item()),
        "native": native,
        "top1_token_id": int(top_ids[0].item()),
        "top2_token_id": int(top_ids[1].item()),
        "top1_prob": float(top_values[0].item()),
        "top2_prob": float(top_values[1].item()),
        "margin": float(top_values[0].item() - top_values[1].item()),
        "entropy": entropy,
        "chosen_rank_native": _rank(native, token),
        **decision,
    }


def _historical_row(token: dict[str, Any], frozen_token_id: int, prefix_nll: float, max_score: float) -> dict[str, Any]:
    if int(token["chosen_token_id"]) != frozen_token_id:
        raise RuntimeError("historical token-id transport mismatch")
    native = _native_map(token.get("full_native_logprobs"))
    decision = _decision(native, prefix_nll, max_score)
    return {
        "chosen_logprob": float(token["chosen_token_logprob"]),
        "native": native,
        "top1_token_id": int(token["top1_token_id"]),
        "top2_token_id": int(token["top2_token_id"]),
        "top1_prob": float(token["top1_prob"]),
        "top2_prob": float(token["top2_prob"]),
        "margin": float(token["top1_minus_top2_margin"]),
        "entropy": float(token["entropy"]),
        "chosen_rank_native": _rank(native, frozen_token_id),
        **decision,
    }


def _full_teacher_force(*, model: Any, encoded: dict[str, Any], tokens: list[int]) -> list[Any]:
    import torch

    prompt = encoded["input_ids"]
    suffix = torch.tensor([tokens], dtype=prompt.dtype, device=model.device)
    ids = torch.cat((prompt.to(model.device), suffix), dim=1)
    inputs = {"input_ids": ids, "use_cache": False, "return_dict": True}
    # The historical generate call was given exactly the encoded mapping.  It did
    # not supply position_ids.  A full-length attention mask is the semantic
    # equivalent of concatenating its all-one prompt mask with the frozen suffix.
    if "attention_mask" in encoded:
        mask = encoded["attention_mask"].to(model.device)
        inputs["attention_mask"] = torch.cat((mask, torch.ones_like(suffix)), dim=1)
    with torch.inference_mode():
        output = model(**inputs)
    return [output.logits[0, int(prompt.shape[-1]) - 1 + index, :] for index in range(len(tokens))]


def _incremental_kv_replay(*, model: Any, encoded: dict[str, Any], tokens: list[int]) -> list[Any]:
    """Replay exact tokens through the cache path without sampling/argmax.

    We pass the original encoded prompt mapping on the first call.  Afterwards
    we pass only input_ids, the cache, and an incrementally extended attention
    mask when the historical encoded mapping contained one.  No custom
    ``position_ids`` are supplied: neither did the original ``generate`` call.
    """
    import torch

    first = {key: value.to(model.device) for key, value in encoded.items()}
    # ``GenerationMixin`` creates an all-one mask when generate receives no
    # attention mask for this unpadded B=1 prompt.  The Unsloth cache-forward
    # wrapper requires that materialised mask on subsequent calls (otherwise it
    # dereferences ``None``).  This therefore mirrors generate's default rather
    # than introducing any selective masking or position IDs.
    if "attention_mask" not in first:
        first["attention_mask"] = torch.ones_like(first["input_ids"])
    # Crucially, use the model's *actual* generation preparation hook.  Unsloth
    # supplies this hook for Qwen/Llama and derives its cache position / position
    # IDs there.  Calling ``model`` directly after prefill skips that generation
    # path and cannot reproduce the historical cached decode semantics.
    sequence = first["input_ids"]
    first_inputs = model.prepare_inputs_for_generation(
        sequence, attention_mask=first["attention_mask"], use_cache=True,
    )
    first_inputs["return_dict"] = True
    with torch.inference_mode():
        output = model(**first_inputs)
    replay_logits = [output.logits[0, -1, :]]
    cache = output.past_key_values
    attention_mask = first["attention_mask"]
    for token in tokens[:-1]:
        one = torch.tensor([[token]], dtype=first["input_ids"].dtype, device=model.device)
        sequence = torch.cat((sequence, one), dim=1)
        attention_mask = torch.cat((attention_mask, torch.ones_like(one)), dim=1)
        next_inputs = model.prepare_inputs_for_generation(
            sequence, past_key_values=cache, attention_mask=attention_mask,
            use_cache=True,
        )
        next_inputs["return_dict"] = True
        with torch.inference_mode():
            output = model(**next_inputs)
        cache = output.past_key_values
        replay_logits.append(output.logits[0, -1, :])
    return replay_logits


def _raw_cell_path(root: Path, task_id: str, output_index: int, depth: int, view: str) -> Path:
    return root / task_id / f"o{output_index:02d}_d{depth:03d}_{view}.json"


def _selected_cells(args: argparse.Namespace) -> list[dict[str, Any]]:
    import pandas as pd

    g0 = _csv_rows(args.g0_csv)
    if len(g0) != 6:
        raise RuntimeError(f"G0 CSV must contain exactly six cells, got {len(g0)}")
    source = pd.read_parquet(args.greedy_parquet)
    selected: list[dict[str, Any]] = []
    for row in g0:
        match = source[
            (source["task_id"] == row["task_id"])
            & (source["output_index"] == int(row["output_index"]))
            & (source["depth"] == int(row["depth"]))
            & (source["view"] == row["view"])
        ]
        if len(match) != 1:
            raise RuntimeError(f"could not uniquely recover G0 cell {row}")
        selected.append(match.iloc[0].to_dict())
    return selected


def _runtime_current(model: Any) -> dict[str, Any]:
    import importlib.metadata
    import torch

    def version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=False,
    )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": version("transformers"),
        "unsloth": version("unsloth"),
        "peft": version("peft"),
        "triton": version("triton"),
        "xformers": version("xformers"),
        "cuda_driver": smi.stdout.strip() if smi.returncode == 0 else None,
        "gpu_name": torch.cuda.get_device_name(0),
        "compute_capability": list(torch.cuda.get_device_capability(0)),
        "model_dtype": str(next(model.parameters()).dtype),
        "model_attn_implementation": getattr(model.config, "_attn_implementation", None),
        "allow_tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "kv_cache_replay": "past_key_values, batch=1, no custom position_ids",
    }


def _historical_environment(authoritative_run: Path) -> dict[str, Any]:
    candidates = sorted(authoritative_run.rglob("environment*.json")) + sorted(authoritative_run.rglob("*runtime*.json"))
    rows = []
    for path in candidates[:20]:
        try:
            rows.append({"path": str(path), "payload": read_json(path)})
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return {"status": "AVAILABLE" if rows else "NOT_AVAILABLE", "files": rows}


def _compare_cell(*, frozen: dict[str, Any], historical: dict[str, Any], incremental: list[Any], full: list[Any], tokenizer: Any, max_score: float) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    tokens = [int(value) for value in frozen["token_ids"]]
    telemetry = historical.get("token_telemetry")
    if not isinstance(telemetry, list) or len(telemetry) != len(tokens):
        raise RuntimeError("historical token telemetry/token count mismatch")
    if len(incremental) != len(tokens) or len(full) != len(tokens):
        raise RuntimeError("replay logits/token count mismatch")
    rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    prefix_nll = 0.0
    for index, token in enumerate(tokens):
        a = _historical_row(telemetry[index], token, prefix_nll, max_score)
        b = _distribution_row(values=incremental[index], token=token, prefix_nll=prefix_nll, max_score=max_score, tokenizer=tokenizer)
        c = _distribution_row(values=full[index], token=token, prefix_nll=prefix_nll, max_score=max_score, tokenizer=tokenizer)
        inc_deltas = [abs(a["native"][candidate] - b["native"][candidate]) for candidate in LEGAL_ARC_TOKENS]
        full_deltas = [abs(a["native"][candidate] - c["native"][candidate]) for candidate in LEGAL_ARC_TOKENS]
        strict_inc = a["strict_survivors"] == b["strict_survivors"]
        strict_full = a["strict_survivors"] == c["strict_survivors"]
        floor_inc = (a["frontier_floor_token"] == b["frontier_floor_token"])
        floor_full = (a["frontier_floor_token"] == c["frontier_floor_token"])
        row = {
            "task_id": frozen["task_id"], "output_index": int(frozen["output_index"]), "depth": int(frozen["depth"]), "view": frozen["view"],
            "token_index": index, "frozen_token_id": token,
            "historical_chosen_token_logprob": a["chosen_logprob"],
            "incremental_replay_token_logprob": b["chosen_logprob"],
            "full_teacher_force_token_logprob": c["chosen_logprob"],
            "historical_top1_token_id": a["top1_token_id"], "incremental_top1_token_id": b["top1_token_id"], "full_tf_top1_token_id": c["top1_token_id"],
            "historical_top2_token_id": a["top2_token_id"], "incremental_top2_token_id": b["top2_token_id"], "full_tf_top2_token_id": c["top2_token_id"],
            "historical_top1_prob": a["top1_prob"], "incremental_top1_prob": b["top1_prob"], "full_tf_top1_prob": c["top1_prob"],
            "historical_top2_prob": a["top2_prob"], "incremental_top2_prob": b["top2_prob"], "full_tf_top2_prob": c["top2_prob"],
            "historical_margin": a["margin"], "incremental_margin": b["margin"], "full_tf_margin": c["margin"],
            "historical_entropy": a["entropy"], "incremental_entropy": b["entropy"], "full_tf_entropy": c["entropy"],
            "historical_chosen_rank_native": a["chosen_rank_native"], "incremental_chosen_rank_native": b["chosen_rank_native"], "fulltf_chosen_rank_native": c["chosen_rank_native"],
            "historical_strict_survivors_json": json.dumps(a["strict_survivors"]), "incremental_strict_survivors_json": json.dumps(b["strict_survivors"]), "fulltf_strict_survivors_json": json.dumps(c["strict_survivors"]),
            "historical_frontier_floor_token": a["frontier_floor_token"], "incremental_frontier_floor_token": b["frontier_floor_token"], "fulltf_frontier_floor_token": c["frontier_floor_token"],
            "historical_searchability_class": a["classification"], "incremental_searchability_class": b["classification"], "fulltf_searchability_class": c["classification"],
            "chosen_token_rank_parity_incremental": a["chosen_rank_native"] == b["chosen_rank_native"],
            "chosen_token_rank_parity_fulltf": a["chosen_rank_native"] == c["chosen_rank_native"],
            "strict_survivor_set_parity_incremental": strict_inc, "strict_survivor_set_parity_fulltf": strict_full,
            "frontier_floor_choice_parity_incremental": floor_inc, "frontier_floor_choice_parity_fulltf": floor_full,
            "searchability_class_parity_incremental": a["classification"] == b["classification"],
            "searchability_class_parity_fulltf": a["classification"] == c["classification"],
            "max_native_abs_delta_incremental_vs_historical": max(inc_deltas), "mean_native_abs_delta_incremental_vs_historical": sum(inc_deltas) / len(inc_deltas),
            "max_native_abs_delta_fulltf_vs_historical": max(full_deltas), "mean_native_abs_delta_fulltf_vs_historical": sum(full_deltas) / len(full_deltas),
        }
        rows.append(row)
        for candidate in LEGAL_ARC_TOKENS:
            native_rows.append({
                "task_id": frozen["task_id"], "output_index": int(frozen["output_index"]), "depth": int(frozen["depth"]), "view": frozen["view"], "token_index": index, "native_token_id": candidate,
                "historical_logprob": a["native"][candidate], "incremental_logprob": b["native"][candidate], "fulltf_logprob": c["native"][candidate],
                "incremental_abs_delta": abs(a["native"][candidate] - b["native"][candidate]), "fulltf_abs_delta": abs(a["native"][candidate] - c["native"][candidate]),
            })
        prefix_nll -= a["chosen_logprob"]
    historical_sum = sum(float(row["historical_chosen_token_logprob"]) for row in rows)
    incremental_sum = sum(float(row["incremental_replay_token_logprob"]) for row in rows)
    full_sum = sum(float(row["full_teacher_force_token_logprob"]) for row in rows)
    summary = {
        "task_id": frozen["task_id"], "output_index": int(frozen["output_index"]), "depth": int(frozen["depth"]), "view": frozen["view"],
        "token_count": len(rows), "historical_sequence_logprob_sum": historical_sum,
        "incremental_sequence_logprob_sum": incremental_sum, "fulltf_sequence_logprob_sum": full_sum,
        "abs_delta_incremental": abs(incremental_sum - historical_sum), "abs_delta_fulltf": abs(full_sum - historical_sum),
        "per_token_mean_abs_delta_incremental": sum(abs(float(row["incremental_replay_token_logprob"]) - float(row["historical_chosen_token_logprob"])) for row in rows) / len(rows),
        "per_token_mean_abs_delta_fulltf": sum(abs(float(row["full_teacher_force_token_logprob"]) - float(row["historical_chosen_token_logprob"])) for row in rows) / len(rows),
        "max_token_native_abs_delta_incremental": max(float(row["max_native_abs_delta_incremental_vs_historical"]) for row in rows),
        "max_token_native_abs_delta_fulltf": max(float(row["max_native_abs_delta_fulltf_vs_historical"]) for row in rows),
        "ALL_TOKEN_TOP1_PARITY_INCREMENTAL": all(int(row["historical_top1_token_id"]) == int(row["incremental_top1_token_id"]) for row in rows),
        "ALL_TOKEN_TOP1_PARITY_FULLTF": all(int(row["historical_top1_token_id"]) == int(row["full_tf_top1_token_id"]) for row in rows),
        "ALL_CHOSEN_RANK_PARITY_INCREMENTAL": all(_bool(row["chosen_token_rank_parity_incremental"]) for row in rows),
        "ALL_CHOSEN_RANK_PARITY_FULLTF": all(_bool(row["chosen_token_rank_parity_fulltf"]) for row in rows),
        "ALL_STRICT_SURVIVOR_SET_PARITY_INCREMENTAL": all(_bool(row["strict_survivor_set_parity_incremental"]) for row in rows),
        "ALL_STRICT_SURVIVOR_SET_PARITY_FULLTF": all(_bool(row["strict_survivor_set_parity_fulltf"]) for row in rows),
        "ALL_FRONTIER_FLOOR_PARITY_INCREMENTAL": all(_bool(row["frontier_floor_choice_parity_incremental"]) for row in rows),
        "ALL_FRONTIER_FLOOR_PARITY_FULLTF": all(_bool(row["frontier_floor_choice_parity_fulltf"]) for row in rows),
        "ALL_SEARCHABILITY_CLASS_PARITY_INCREMENTAL": all(_bool(row["searchability_class_parity_incremental"]) for row in rows),
        "ALL_SEARCHABILITY_CLASS_PARITY_FULLTF": all(_bool(row["searchability_class_parity_fulltf"]) for row in rows),
    }
    return rows, summary, native_rows


def run(args: argparse.Namespace) -> None:
    import torch
    from unsloth import FastLanguageModel
    from arc.io import load_dataset

    started = time.perf_counter()
    _runtime_identity(args)  # validates V5 and adapter source identities before GPU work
    cells = _selected_cells(args)
    model, tokenizer, _initial, config, tokenizer_info = _load_runtime(args)
    FastLanguageModel.for_inference(model)
    tasks = load_dataset(args.challenge)
    max_score = float(read_json(args.v5_config)["max_score"])
    token_rows: list[dict[str, Any]] = []
    cell_rows: list[dict[str, Any]] = []
    native_rows: list[dict[str, Any]] = []
    for frozen in cells:
        task_id, output_index, depth, view = str(frozen["task_id"]), int(frozen["output_index"]), int(frozen["depth"]), str(frozen["view"])
        raw_path = _raw_cell_path(args.raw_greedy_cells, task_id, output_index, depth, view)
        historical = read_json(raw_path)
        tokens = [int(value) for value in frozen["token_ids"]]
        if historical.get("token_ids") != tokens:
            raise RuntimeError(f"historical/raw token mismatch:{raw_path}")
        metadata = _adapter_metadata(args.adapters, task_id, depth)
        if str(frozen["checkpoint_sha256"]) != metadata["checkpoint_sha256"] or historical.get("checkpoint_sha256") != metadata["checkpoint_sha256"]:
            raise RuntimeError(f"adapter identity mismatch:{task_id}:d{depth}:{view}")
        common.load_adapter(model=model, metadata=metadata)
        encoded, _augmentation = _encoded_view(tokenizer, view_task(tasks[task_id], output_index), view, config)
        if int(encoded["input_ids"].shape[-1]) != int(frozen["prompt_tokens"]):
            raise RuntimeError(f"prompt token mismatch:{task_id}:d{depth}:{view}")
        torch.cuda.synchronize()
        incremental = _incremental_kv_replay(model=model, encoded=encoded, tokens=tokens)
        torch.cuda.synchronize()
        full = _full_teacher_force(model=model, encoded=encoded, tokens=tokens)
        torch.cuda.synchronize()
        details, summary, native = _compare_cell(frozen=frozen, historical=historical, incremental=incremental, full=full, tokenizer=tokenizer, max_score=max_score)
        summary.update({"checkpoint_sha256": metadata["checkpoint_sha256"], "prompt_tokens": int(encoded["input_ids"].shape[-1])})
        token_rows.extend(details); cell_rows.append(summary); native_rows.extend(native)
        del incremental, full, encoded
        torch.cuda.empty_cache()
    numerical_incremental = sum(float(row["abs_delta_incremental"]) <= args.tolerance for row in cell_rows)
    numerical_full = sum(float(row["abs_delta_fulltf"]) <= args.tolerance for row in cell_rows)
    inc_decision = all(_bool(row["ALL_CHOSEN_RANK_PARITY_INCREMENTAL"]) and _bool(row["ALL_STRICT_SURVIVOR_SET_PARITY_INCREMENTAL"]) and _bool(row["ALL_FRONTIER_FLOOR_PARITY_INCREMENTAL"]) and _bool(row["ALL_SEARCHABILITY_CLASS_PARITY_INCREMENTAL"]) for row in cell_rows)
    full_decision = all(_bool(row["ALL_CHOSEN_RANK_PARITY_FULLTF"]) and _bool(row["ALL_STRICT_SURVIVOR_SET_PARITY_FULLTF"]) and _bool(row["ALL_FRONTIER_FLOOR_PARITY_FULLTF"]) and _bool(row["ALL_SEARCHABILITY_CLASS_PARITY_FULLTF"]) for row in cell_rows)
    if numerical_incremental == 6 and inc_decision:
        status, root_cause, g1_path = "PASS_NUMERICAL", "NO_RUNTIME_PARITY_BLOCKER", "INCREMENTAL_KV_REPLAY"
    elif inc_decision:
        status, root_cause, g1_path = "PASS_DECISION_PARITY", "FULL_SEQUENCE_FORWARD_NUMERIC_PATH_DIFFERS_FROM_INCREMENTAL_KV_PATH", "INCREMENTAL_KV_REPLAY"
    else:
        status, root_cause, g1_path = "FAIL", "INCREMENTAL_KV_REPLAY_DOES_NOT_PRESERVE_HISTORICAL_SEARCH_DECISIONS", "NONE"
    report_dir = args.report_dir
    _write_csv(report_dir / "g0_5_token_parity.csv", token_rows)
    _write_csv(report_dir / "g0_5_cell_parity.csv", cell_rows)
    _write_csv(report_dir / "g0_5_native_distribution_parity.csv", native_rows)
    _write_csv(report_dir / "g0_5_search_decision_parity.csv", token_rows)
    current = _runtime_current(model)
    historical_env = _historical_environment(args.authoritative_run)
    _atomic_json(report_dir / "runtime_current.json", {"tokenizer": tokenizer_info, "runtime": current})
    _atomic_json(report_dir / "runtime_environment_diff.json", {"historical": historical_env, "current": current, "comparison_status": "HISTORICAL_METADATA_AVAILABLE" if historical_env["status"] == "AVAILABLE" else "HISTORICAL_METADATA_NOT_AVAILABLE"})
    report = ["# G0.5 incremental KV replay parity", "", f"STATUS: {status}", f"Tolerance: {args.tolerance}", "", "The incremental path starts with the original encoded prompt, uses `past_key_values` one frozen token at a time, and does not supply custom position IDs.  Full TF is retained only as a diagnostic control.", "", "| depth | task/output/view | historical LP | incremental LP | full-TF LP | inc abs delta | full abs delta | inc max native delta | full max native delta | inc strict parity | full strict parity |", "|---|---|---:|---:|---:|---:|---:|---:|---:|---|---|"]
    report += [f"| {row['depth']} | {row['task_id']}:o{row['output_index']} / {row['view']} | {row['historical_sequence_logprob_sum']:.8g} | {row['incremental_sequence_logprob_sum']:.8g} | {row['fulltf_sequence_logprob_sum']:.8g} | {row['abs_delta_incremental']:.8g} | {row['abs_delta_fulltf']:.8g} | {row['max_token_native_abs_delta_incremental']:.8g} | {row['max_token_native_abs_delta_fulltf']:.8g} | {row['ALL_STRICT_SURVIVOR_SET_PARITY_INCREMENTAL']} | {row['ALL_STRICT_SURVIVOR_SET_PARITY_FULLTF']} |" for row in cell_rows]
    report += ["", f"Root cause classification: `{root_cause}`.", f"G1 allowed: `{status != 'FAIL'}`.", f"Required G1 scoring path: `{g1_path}`.", "Scientific configuration changed: NO.  TTT rerun: NO.  Greedy rerun: NO.  V5 rerun: NO."]
    (report_dir / "G0_5_INCREMENTAL_REPLAY_REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    _atomic_json(args.scratch / "g0_5" / "status.json", {
        "status": status, "cells_tested": len(cell_rows), "incremental_numerical_pass": numerical_incremental, "fulltf_numerical_pass": numerical_full,
        "incremental_decision_parity": inc_decision, "fulltf_decision_parity": full_decision, "root_cause": root_cause, "g1_required_scoring_path": g1_path,
        "wall_seconds": time.perf_counter() - started,
    })
    if status == "FAIL":
        raise RuntimeError("G0_5_INCREMENTAL_REPLAY_PARITY_FAIL")


def parser() -> argparse.ArgumentParser:
    item = argparse.ArgumentParser()
    item.add_argument("--scratch", type=Path, required=True)
    item.add_argument("--report-dir", type=Path, required=True)
    item.add_argument("--g0-csv", type=Path, required=True)
    item.add_argument("--greedy-parquet", type=Path, required=True)
    item.add_argument("--raw-greedy-cells", type=Path, required=True)
    item.add_argument("--authoritative-run", type=Path, required=True)
    item.add_argument("--adapters", type=Path, required=True)
    item.add_argument("--challenge", type=Path, required=True)
    item.add_argument("--runtime-config", type=Path, required=True)
    item.add_argument("--v5-config", type=Path, required=True)
    item.add_argument("--model-path", type=Path, required=True)
    item.add_argument("--native-config-dir", type=Path, required=True)
    item.add_argument("--ptxas", type=Path, required=True)
    item.add_argument("--tolerance", type=float, default=0.05)
    return item


if __name__ == "__main__":
    run(parser().parse_args())
