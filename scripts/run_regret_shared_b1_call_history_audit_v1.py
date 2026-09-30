#!/usr/bin/env python3
"""Target-blind call-history diagnosis for the scalar dynamic-ready B1 decoder.

This is deliberately a *diagnostic* of one frozen d59b0160/d24 four-view
surface.  It neither instantiates Dynamic Batch2 nor reads Gold.  In
particular, it does not try to repair the scheduler: it records enough of the
existing scalar execution to distinguish a pre-created-cell issue from a
model/runtime call-history numerical effect.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_dynamic_ready import (
    _reply, clone_legacy_cache, ready_result, run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_dynamic_ready_b1_1 import _cell_key, _parity, write_csv
from scripts.run_regret_dynamic_ready_v1 import ADAPTER_SHA, DEPTH, OUTPUT_INDEX, TASK_ID, VIEWS, decoder
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file


EXPERIMENT = "REGRET_SHARED_B1_CALL_HISTORY_AUDIT_V1"
ARC_TOKEN_IDS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)
PRIMARY = "flip_ud"
HISTORY = "anti_transpose"


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _tensor_sha(tensor: Any) -> str:
    return _sha_bytes(tensor.detach().float().cpu().contiguous().numpy().tobytes())


def _cache_sha(cache: Any) -> str:
    """Stable content hash, intentionally not a storage-address comparison."""
    import torch
    digest = hashlib.sha256()

    def walk(value: Any) -> None:
        if isinstance(value, torch.Tensor):
            digest.update(str(tuple(value.shape)).encode()); digest.update(str(value.dtype).encode())
            digest.update(value.detach().float().cpu().contiguous().numpy().tobytes())
        elif isinstance(value, (tuple, list)):
            digest.update(f"[{len(value)}]".encode())
            for item in value:
                walk(item)
        else:
            digest.update(repr(type(value)).encode())
    walk(cache)
    return digest.hexdigest()


def _cache_geometry(cache: Any) -> str:
    def walk(value: Any) -> Any:
        shape = getattr(value, "shape", None)
        if shape is not None:
            return tuple(int(item) for item in shape)
        if isinstance(value, (tuple, list)):
            return tuple(walk(item) for item in value)
        return type(value).__name__
    return json.dumps(walk(cache), separators=(",", ":"))


def _logit_summary(logits: Any) -> dict[str, Any]:
    import torch
    row = logits[:, -1].float()
    logprob = torch.log_softmax(row, dim=-1)
    top = torch.topk(row, k=2, dim=-1)
    arc_logits = {str(token): float(row[0, token].item()) for token in ARC_TOKEN_IDS}
    arc_logprobs = {str(token): float(logprob[0, token].item()) for token in ARC_TOKEN_IDS}
    arc_rank = [int(token) for token in sorted(ARC_TOKEN_IDS, key=lambda token: (-arc_logits[str(token)], token))]
    return {"full_logits_sha256": _tensor_sha(row), "arc_logits": arc_logits,
            "arc_logprobs": arc_logprobs, "top1": int(top.indices[0, 0]), "top2": int(top.indices[0, 1]),
            "arc_ranking": arc_rank}


def _delta(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    full_same = lhs["full_logits_sha256"] == rhs["full_logits_sha256"]
    arc_delta = max(abs(float(lhs["arc_logits"][str(t)]) - float(rhs["arc_logits"][str(t)])) for t in ARC_TOKEN_IDS)
    lp_delta = max(abs(float(lhs["arc_logprobs"][str(t)]) - float(rhs["arc_logprobs"][str(t)])) for t in ARC_TOKEN_IDS)
    return {"full_logits_same": full_same, "max_abs_full_logit_delta": 0.0 if full_same else None,
            "max_abs_arc_logit_delta": arc_delta, "max_abs_arc_logprob_delta": lp_delta,
            "top1_same": lhs["top1"] == rhs["top1"], "top2_same": lhs["top2"] == rhs["top2"],
            "arc_ranking_same": lhs["arc_ranking"] == rhs["arc_ranking"]}


def _attach_full_delta(comparison: dict[str, Any], lhs_outputs: Any, rhs_outputs: Any) -> dict[str, Any]:
    import torch
    comparison = dict(comparison)
    comparison["max_abs_full_logit_delta"] = float(torch.max(torch.abs(
        lhs_outputs.logits[:, -1].float() - rhs_outputs.logits[:, -1].float())).item())
    return comparison


def _one_forward(model: Any, request: Any, *, cache: Any | None = None) -> tuple[Any, dict[str, Any]]:
    import torch
    actual_cache = request.cache if cache is None else cache
    with torch.no_grad():
        outputs = model(input_ids=torch.tensor([[request.token_id]], device=model.device, dtype=torch.long),
                        position_ids=torch.tensor([[request.position]], device=model.device, dtype=torch.long),
                        past_key_values=actual_cache, return_dict=True, use_cache=True)
    return outputs, _logit_summary(outputs.logits)


def _advance_one(model: Any, cell: Any, trace: list[dict[str, Any]] | None = None) -> None:
    request = cell.request
    if request is None:
        raise RuntimeError("attempted to advance completed cell")
    before = _cache_sha(request.cache)
    outputs, summary = _one_forward(model, request)
    if trace is not None:
        trace.append({"ordinal": int(request.ordinal), "input_token": int(request.token_id),
                      "position": int(request.position), "kv_checksum": before, **summary})
    _reply(cell, outputs)


def _run_cell_with_trace(model: Any, cell: Any) -> tuple[Any, list[dict[str, Any]]]:
    trace: list[dict[str, Any]] = []
    while cell.request is not None:
        _advance_one(model, cell, trace)
    return ready_result(cell), trace


def _runtime(contract: dict[str, Any], gpu_id: int) -> tuple[Any, dict[str, Any], Any, Any, Any]:
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel
    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
        reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
        native_config_dir=Path(contract["native_config_dir"]), gpu_id=gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
    if adapter_sha != ADAPTER_SHA:
        raise RuntimeError(f"unexpected d59 d24 adapter {adapter_sha}")
    task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
    encoded = {view: value["input_ids"] for view, (value, _aug) in zip(
        VIEWS, (common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config) for view in VIEWS), strict=True)}
    if any(int(value.shape[0]) != 1 for value in encoded.values()):
        raise RuntimeError("call-history audit requires scalar prompts")
    FastLanguageModel.for_inference(model)
    return model, encoded, decoder(contract["caps"]), adapter_sha, assert_native_token_contract(tokenizer)


def _release(model: Any) -> None:
    import torch
    del model
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize()


def _new_cell(model: Any, encoded: dict[str, Any], dec: Any, view: str) -> Any:
    return start_ready_cell(model=model, input_ids=encoded[view].to(model.device), config=dec,
                            cell_key=_cell_key(view), normalize_root_cache=False, active_time_accounting=True)


def _first_diff(reference: list[dict[str, Any]], affected: list[dict[str, Any]]) -> dict[str, Any]:
    for index, (left, right) in enumerate(zip(reference, affected, strict=False)):
        logical = (left["input_token"] == right["input_token"], left["position"] == right["position"],
                   left["kv_checksum"] == right["kv_checksum"])
        if left["full_logits_sha256"] != right["full_logits_sha256"]:
            return {"status": "DIVERGED", "ordinal": index, "input_token": right["input_token"],
                    "position": right["position"], "same_input_token": logical[0], "same_position": logical[1],
                    "same_kv_checksum": logical[2], "reference_logits_sha256": left["full_logits_sha256"],
                    "affected_logits_sha256": right["full_logits_sha256"], **_delta(left, right)}
    if len(reference) != len(affected):
        return {"status": "TRACE_LENGTH_DIFFERENCE", "ordinal": min(len(reference), len(affected))}
    return {"status": "NO_DIVERGENCE", "ordinal": None}


def _prior_scheduler_rows(prior_output: Path) -> tuple[list[dict[str, Any]], bool]:
    raw = read_json(prior_output / "raw_refactor_b1.json")
    events = raw[0]["scheduler"]["events"]
    rows = []
    for event in events[:50]:
        rows.append({"forward_index": event["forward_index"], "physical_batch": event["physical_batch"],
                     "cell_key": event["cell_keys"][0], "position": event["position"],
                     "wall_seconds": event["wall_seconds"]})
    keys = [row["cell_key"] for row in rows]
    interleaves = any(keys[i] != keys[i - 1] for i in range(1, len(keys)))
    return rows, interleaves


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output}")
    challenge = args.challenge.resolve(); no_gold_challenge(challenge)
    fixed = read_json(args.fixed_budget_contract.resolve()); caps = fixed.get("caps", {})
    if int(caps.get("max_expanded_nodes", -1)) != 4096 or int(caps.get("max_completed_candidates", -1)) != 32:
        raise RuntimeError("requires frozen Regret4 4096/32 contract")
    prior = args.prior_output.resolve()
    for name in ("raw_scalar.json", "raw_refactor_b1.json"):
        if not (prior / name).is_file():
            raise RuntimeError(f"missing prior evidence {prior / name}")
    output.mkdir(parents=True)
    contract = {"experiment": EXPERIMENT, "source_commit": args.source_commit, "target_blind": True,
        "gold_accessed": False, "dynamic_batch2_executed": False, "untouched12_used": False, "untouched24_used": False,
        "task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH, "views": list(VIEWS),
        "primary_view": PRIMARY, "history_view": HISTORY, "policy": "CUMULATIVE_REGRET_r=4.00", "caps": caps,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge), "model_path": str(args.model_path.resolve()),
        "native_config_dir": str(args.native_config_dir.resolve()), "reference_config": str(args.reference_config.resolve()),
        "authoritative_root": str(args.authoritative_root.resolve()), "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()), "fixed_budget_contract": str(args.fixed_budget_contract.resolve()),
        "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()), "prior_output": str(prior),
        "prior_scalar_sha256": sha256_file(prior / "raw_scalar.json"), "prior_b1_sha256": sha256_file(prior / "raw_refactor_b1.json"),
        "fresh_process_per_checkpoint": False, "fresh_model_per_checkpoint": True,
        "execution_protocol": "diagnostic only; no policy/cap/precision changes"}
    atomic_json(output / "CALL_HISTORY_CONTRACT.json", contract)
    rows, interleaves = _prior_scheduler_rows(prior)
    write_csv(output / "SCHEDULER_SEQUENCE.csv", rows)
    atomic_json(output / "SCHEDULER_SEQUENCE_SUMMARY.json", {"first_50_rows": len(rows),
        "actually_interleaves": interleaves, "observed_sequence": [row["cell_key"] for row in rows],
        "interpretation": "frozen B1 trace; a non-interleaving result excludes coroutine alternation as the shared-B1 drift mechanism"})


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "CALL_HISTORY_CONTRACT.json")
    no_gold_challenge(Path(contract["challenge"]))
    prior = {row["cell_key"]: row for row in read_json(Path(contract["prior_output"]) / "raw_scalar.json")}
    # Phase B: a fresh model and one immutable first flip request, replayed twice.
    model, encoded, dec, adapter_sha, native = _runtime(contract, args.gpu_id)
    try:
        flip = _new_cell(model, encoded, dec, PRIMARY); request = flip.request
        if request is None:
            raise RuntimeError("flip prefill did not expose an incremental request")
        frozen_cache = clone_legacy_cache(request.cache)
        frozen = {"token_id": int(request.token_id), "position": int(request.position), "ordinal": int(request.ordinal),
                  "cache_sha256": _cache_sha(frozen_cache), "cache_geometry": _cache_geometry(frozen_cache)}
        o1, one = _one_forward(model, request, cache=clone_legacy_cache(frozen_cache))
        o2, two = _one_forward(model, request, cache=clone_legacy_cache(frozen_cache))
        replay_rows = [{"replay": 1, **frozen, **one}, {"replay": 2, **frozen, **two}]
        write_csv(output / "SAME_REQUEST_REPLAY.csv", replay_rows)
        same = _attach_full_delta(_delta(one, two), o1, o2); atomic_json(output / "FROZEN_FLIP_REQUEST.json", frozen)
        atomic_json(output / "SAME_REQUEST_REPLAY_SUMMARY.json", {"fresh_same_request_deterministic": bool(same["full_logits_same"]), **same})
    finally:
        _release(model)

    # Phase C: independent model objects for each history length.  The frozen
    # flip request is reconstructed from the same prompt each time, and replay
    # always uses a deep cache clone.
    history_rows: list[dict[str, Any]] = []
    first_drift: int | None = None
    for steps in (0, 1, 8, 32, 128, 512):
        model, encoded, dec, _adapter_sha, _native = _runtime(contract, args.gpu_id)
        try:
            flip = _new_cell(model, encoded, dec, PRIMARY); request = flip.request
            baseline_cache = clone_legacy_cache(request.cache)
            base_o, baseline = _one_forward(model, request, cache=clone_legacy_cache(baseline_cache))
            anti = _new_cell(model, encoded, dec, HISTORY)
            actual = 0
            while actual < steps and anti.request is not None:
                _advance_one(model, anti); actual += 1
            anti_complete = anti.request is None
            before = _cache_sha(baseline_cache)
            after_o, after = _one_forward(model, request, cache=clone_legacy_cache(baseline_cache))
            comparison = _attach_full_delta(_delta(baseline, after), base_o, after_o)
            row = {"history_steps_requested": steps, "history_steps_executed": actual, "history_completed": anti_complete,
                   "flip_token_id": int(request.token_id), "flip_position": int(request.position),
                   "flip_cache_sha_before": before, "flip_cache_sha_after": _cache_sha(baseline_cache),
                   "regret_retained_set_same": comparison["arc_ranking_same"],
                   "frontier_decision_would_change": not comparison["arc_ranking_same"], **comparison,
                   "baseline_full_logits_sha256": baseline["full_logits_sha256"], "after_full_logits_sha256": after["full_logits_sha256"]}
            history_rows.append(row)
            if first_drift is None and not comparison["full_logits_same"]:
                first_drift = steps
                break
        finally:
            _release(model)
    write_csv(output / "CALL_HISTORY_LOGIT_DRIFT.csv", history_rows)

    # Phase D.  Construct all shared-style cells, but deliberately service
    # only the requested first cell.  Thus creation/prefill stays shared while
    # no other decoder cell contributes model-call history.
    forced_rows: list[dict[str, Any]] = []
    for selected in (PRIMARY, "identity"):
        model, encoded, dec, _adapter_sha, _native = _runtime(contract, args.gpu_id)
        try:
            cells = {view: _new_cell(model, encoded, dec, view) for view in VIEWS}
            result, trace = _run_cell_with_trace(model, cells[selected])
            parity = _parity(prior[_cell_key(selected)], result)
            forced_rows.append({"forced_first_view": selected, "all_ready_cells_constructed": True,
                "only_selected_cell_executed": True, "physical_forwards": len(trace), **parity,
                "nodes_expanded": sum(row.get("state") == "expanded" for row in result.nodes)})
        finally:
            _release(model)
    write_csv(output / "FORCED_FIRST_PARITY.csv", forced_rows)

    # Phase E/F records an isolated reference and then same-model history with
    # flip pre-created, plus newly prefilling flip after full anti completion.
    model, encoded, dec, _adapter_sha, _native = _runtime(contract, args.gpu_id)
    try:
        fresh_flip = _new_cell(model, encoded, dec, PRIMARY)
        base_result, base_trace = _run_cell_with_trace(model, fresh_flip)
        base_parity = _parity(prior[_cell_key(PRIMARY)], base_result)
    finally:
        _release(model)

    model, encoded, dec, _adapter_sha, _native = _runtime(contract, args.gpu_id)
    try:
        anti = _new_cell(model, encoded, dec, HISTORY)
        _anti_result, _anti_trace = _run_cell_with_trace(model, anti)
        post_flip = _new_cell(model, encoded, dec, PRIMARY)
        post_result, post_trace = _run_cell_with_trace(model, post_flip)
        post_parity = _parity(prior[_cell_key(PRIMARY)], post_result)
    finally:
        _release(model)
    write_csv(output / "PREFILL_AFTER_HISTORY.csv", [
        {"condition": "fresh_flip_isolated", "flip_prefill_timing": "before_any_history", **base_parity},
        {"condition": "anti_complete_then_new_flip_prefill", "flip_prefill_timing": "after_anti_completion", **post_parity},
    ])

    # A pre-created flip state affected by a completed anti cell provides the
    # direct per-forward trace required by Phase F.
    model, encoded, dec, _adapter_sha, _native = _runtime(contract, args.gpu_id)
    try:
        affected_flip = _new_cell(model, encoded, dec, PRIMARY)
        anti = _new_cell(model, encoded, dec, HISTORY)
        _anti_result, _anti_trace = _run_cell_with_trace(model, anti)
        affected_result, affected_trace = _run_cell_with_trace(model, affected_flip)
        affected_parity = _parity(prior[_cell_key(PRIMARY)], affected_result)
    finally:
        _release(model)
    first = _first_diff(base_trace, affected_trace)
    write_csv(output / "FLIP_FIRST_TRUE_DIVERGENCE.csv", [first])
    atomic_json(output / "FLIP_TRACE_COMPARISON.json", {"reference_trace_forwards": len(base_trace),
        "affected_trace_forwards": len(affected_trace), "affected_precreated_flip_parity": affected_parity,
        "first_difference": first})

    forced = {row["forced_first_view"]: row for row in forced_rows}
    post_exact = bool(post_parity["exact_parity"])
    if first.get("status") == "DIVERGED" and all(first.get(key) for key in ("same_input_token", "same_position", "same_kv_checksum")):
        root_cause = "MODEL_RUNTIME_CALL_HISTORY_NUMERICAL_EFFECT"
        next_step = "FIX_MODEL_RUNTIME_STATE"
    elif first.get("status") == "DIVERGED":
        root_cause = "EXECUTOR_STATE_DIVERGENCE"
        next_step = "FIX_RESUME_STATE"
    elif post_exact and not bool(affected_parity["exact_parity"]):
        root_cause = "PRECREATED_STATE_RESUME_EFFECT"
        next_step = "FIX_RESUME_STATE"
    else:
        root_cause = "NOT_ESTABLISHED"; next_step = "OTHER"
    decision = {"experiment": EXPERIMENT, "source_commit": contract["source_commit"], "adapter_sha256": adapter_sha,
        "current_b1_actually_interleaves": bool(read_json(output / "SCHEDULER_SEQUENCE_SUMMARY.json")["actually_interleaves"]),
        "fresh_same_request_deterministic": bool(read_json(output / "SAME_REQUEST_REPLAY_SUMMARY.json")["fresh_same_request_deterministic"]),
        "call_history_drift_first_seen_at": first_drift if first_drift is not None else "NONE",
        "flip_first_parity": "PASS" if bool(forced[PRIMARY]["exact_parity"]) else "FAIL",
        "identity_first_parity": "PASS" if bool(forced["identity"]["exact_parity"]) else "FAIL",
        "flip_prefilled_after_anti_parity": "PASS" if post_exact else "FAIL",
        "first_divergence_forward": first.get("ordinal"), "first_divergence_same_kv": first.get("same_kv_checksum"),
        "first_divergence_same_position": first.get("same_position"), "first_divergence_same_token": first.get("same_input_token"),
        "root_cause": root_cause, "next": next_step, "dynamic_b2_executed": False,
        "gold_accessed": False, "untouched12_used": False, "untouched24_used": False,
        "native_token_contract": native, "fresh_model_per_checkpoint": True, "fresh_process_per_checkpoint": False}
    atomic_json(output / "DECISION.json", decision)
    lines = [f"# {EXPERIMENT}", "", "Target-blind diagnostic. Gold was not opened and Dynamic Batch2 was not instantiated.", "",
        f"- Current B1 interleaves: {decision['current_b1_actually_interleaves']}",
        f"- Fresh same-request deterministic: {decision['fresh_same_request_deterministic']}",
        f"- First observed call-history drift: {decision['call_history_drift_first_seen_at']}",
        f"- Flip forced-first parity: {decision['flip_first_parity']}", f"- Identity forced-first parity: {decision['identity_first_parity']}",
        f"- New flip prefilled after anti parity: {decision['flip_prefilled_after_anti_parity']}",
        f"- First per-forward divergence: {decision['first_divergence_forward']}", f"- Root cause: {decision['root_cause']}", "",
        "Fresh *model objects* were used for each call-history checkpoint. They were initialized in one diagnostic process; this is recorded rather than described as a fresh OS process."]
    (output / "CALL_HISTORY_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    names = {"CALL_HISTORY_CONTRACT.json", "SCHEDULER_SEQUENCE.csv", "SCHEDULER_SEQUENCE_SUMMARY.json", "FROZEN_FLIP_REQUEST.json",
             "SAME_REQUEST_REPLAY.csv", "SAME_REQUEST_REPLAY_SUMMARY.json", "CALL_HISTORY_LOGIT_DRIFT.csv", "FORCED_FIRST_PARITY.csv",
             "PREFILL_AFTER_HISTORY.csv", "FLIP_FIRST_TRUE_DIVERGENCE.csv", "FLIP_TRACE_COMPARISON.json", "CALL_HISTORY_AUDIT.md", "DECISION.json"}
    atomic_json(output / "HASHES.json", {"sha256": {name: sha256_file(output / name) for name in sorted(names)}, "gold_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        item = sub.add_parser(command)
        item.add_argument("--output", type=Path, required=True); item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True); item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True); item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True); item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--prior-output", type=Path, required=True); item.add_argument("--source-commit", required=True); item.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    else: run(args)


if __name__ == "__main__": main()
