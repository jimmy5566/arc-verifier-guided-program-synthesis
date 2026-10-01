#!/usr/bin/env python3
"""Fresh-process, target-blind prefill-state causality micro experiment.

This tool deliberately performs *one* continuation forward per fresh model.
It never invokes DFS, never opens a solution file, and never advances the
foreign ``anti_transpose`` cell.  Its job is to decide whether that root
prefill alone changes the pre-created flip continuation.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import _retained
from inference.nvarc_turbodfs_dynamic_ready import clone_legacy_cache
from scripts.run_d1_real_decoder_ab import atomic_json, no_gold_challenge
from scripts.run_regret_dynamic_ready_b1_1 import write_csv
from scripts.run_regret_dynamic_ready_v1 import ADAPTER_SHA, DEPTH, OUTPUT_INDEX, TASK_ID
from scripts.run_regret_shared_b1_call_history_audit_v1 import (
    ARC_TOKEN_IDS,
    HISTORY,
    PRIMARY,
    _reply,
    _cache_geometry,
    _cache_sha,
    _logit_summary,
    _new_cell,
    _one_forward,
    _release,
    _runtime,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "REGRET_PREFILL_STATE_ROOT_CAUSE_V1"
REQUIRED = (
    "candidate_tokens", "nodes", "branch_probabilities", "frontier_floor_events",
    "search_trace", "nodes_expanded", "completed_candidates", "termination_reason",
)


def _sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _cache_details(cache: Any) -> dict[str, Any]:
    """Public cache information plus content hashes, never storage addresses."""
    values: dict[str, Any] = {"cache_python_type": f"{type(cache).__module__}.{type(cache).__qualname__}"}
    for name in ("seen_tokens", "_seen_tokens", "cache_position", "max_cache_len", "max_length", "sliding_window"):
        value = getattr(cache, name, None)
        if isinstance(value, (int, float, str, bool)) or value is None:
            values[name] = value
        elif hasattr(value, "detach"):
            values[name] = {"tensor_shape": list(value.shape), "tensor_sha256": _tensor_sha(value)}
    for name in ("get_seq_length", "get_max_cache_shape"):
        method = getattr(cache, name, None)
        if callable(method):
            try:
                values[name] = method()
            except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
                values[name] = f"UNAVAILABLE:{type(exc).__name__}"
    try:
        legacy = clone_legacy_cache(cache)
        values["legacy_geometry"] = json.loads(_cache_geometry(legacy))
        values["content_sha256"] = _cache_sha(legacy)
        values["layer_tensor_sha256"] = [
            [_tensor_sha(part) for part in layer] for layer in legacy
        ]
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        values["content_sha256"] = f"UNAVAILABLE:{type(exc).__name__}"
    return values


def _tensor_sha(value: Any) -> str:
    return hashlib.sha256(value.detach().float().cpu().contiguous().numpy().tobytes()).hexdigest()


def _retained_successors(logits: Any, config: Any) -> list[dict[str, Any]]:
    import torch

    logprobs = torch.log_softmax(logits[:, -1].float(), dim=-1)[0]
    ranked = sorted(((int(token), float(logprobs[token].item())) for token in config.arc_tokens),
                    key=lambda pair: (-pair[1], pair[0]))
    kept, reason = _retained(config, ranked, score_before=0.0, regret_before=0.0,
                             remaining=int(config.max_new_tokens), generated_length=0)
    return [{"token_id": int(token), "logprob": float(logprob), "cumulative_nll": float(score),
             "cumulative_regret": float(regret), "reason": reason}
            for score, regret, token, logprob in kept]


def _write_condition(args: argparse.Namespace) -> None:
    contract = read_json(args.contract.resolve())
    no_gold_challenge(Path(contract["challenge"]))
    model, encoded, dec, adapter_sha, native = _runtime(contract, args.gpu_id)
    try:
        flip = _new_cell(model, encoded, dec, PRIMARY)
        request = flip.request
        if request is None:
            raise RuntimeError("flip root prefill did not yield an incremental request")
        request_before = _cache_details(request.cache)
        frozen_legacy = clone_legacy_cache(request.cache) if args.cache_mode == "LEGACY_DEEP_CLONE" else None
        anti = None
        if args.foreign_prefill:
            anti = _new_cell(model, encoded, dec, HISTORY)
            if anti.request is None:
                raise RuntimeError("anti root prefill did not yield an incremental request")
        request_after = _cache_details(request.cache)
        actual_cache = request.cache if frozen_legacy is None else clone_legacy_cache(frozen_legacy)
        cache_used = _cache_details(actual_cache)
        outputs, summary = _one_forward(model, request, cache=actual_cache)
        record = {
            "condition_id": args.condition_id, "cache_mode": args.cache_mode,
            "foreign_prefill": bool(args.foreign_prefill), "foreign_incremental_forwards": 0,
            "cell_key": f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{PRIMARY}",
            "foreign_cell_key": f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{HISTORY}",
            "request_token_id": int(request.token_id), "request_position": int(request.position),
            "request_ordinal": int(request.ordinal), "adapter_sha256": adapter_sha,
            "native_token_contract": native, "request_cache_before": request_before,
            "request_cache_after_foreign_prefill": request_after, "cache_used": cache_used,
            "full_logits_sha256": summary["full_logits_sha256"], "arc_logits": summary["arc_logits"],
            "arc_logprobs": summary["arc_logprobs"], "arc_ranking": summary["arc_ranking"],
            "top1": summary["top1"], "top2": summary["top2"],
            "retained_regret_successors": _retained_successors(outputs.logits, dec),
            "gold_accessed": False, "dfs_executed": False, "dynamic_b2_executed": False,
        }
        atomic_json(args.result.resolve(), record)
    finally:
        _release(model)


def _write_history_condition(args: argparse.Namespace) -> None:
    """One fresh process/model for exactly N foreign incremental forwards."""
    contract = read_json(args.contract.resolve())
    no_gold_challenge(Path(contract["challenge"]))
    model, encoded, dec, adapter_sha, native = _runtime(contract, args.gpu_id)
    try:
        flip = _new_cell(model, encoded, dec, PRIMARY)
        request = flip.request
        if request is None:
            raise RuntimeError("flip root prefill did not yield an incremental request")
        flip_cache_before = _cache_details(request.cache)
        anti = _new_cell(model, encoded, dec, HISTORY)
        actual = 0
        while actual < args.history_length:
            if anti.request is None:
                raise RuntimeError(f"anti completed before requested history {args.history_length}")
            outputs, _summary = _one_forward(model, anti.request)
            _reply(anti, outputs); actual += 1
        flip_cache_after = _cache_details(request.cache)
        outputs, summary = _one_forward(model, request)
        atomic_json(args.result.resolve(), {
            "history_length_requested": int(args.history_length), "history_length_executed": actual,
            "foreign_incremental_forwards": actual, "foreign_prefill": True,
            "cell_key": f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{PRIMARY}",
            "foreign_cell_key": f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{HISTORY}",
            "request_token_id": int(request.token_id), "request_position": int(request.position),
            "request_ordinal": int(request.ordinal), "adapter_sha256": adapter_sha,
            "native_token_contract": native, "request_cache_before": flip_cache_before,
            "request_cache_after_foreign_history": flip_cache_after,
            "full_logits_sha256": summary["full_logits_sha256"], "arc_logits": summary["arc_logits"],
            "arc_logprobs": summary["arc_logprobs"], "arc_ranking": summary["arc_ranking"],
            "top1": summary["top1"], "top2": summary["top2"],
            "retained_regret_successors": _retained_successors(outputs.logits, dec),
            "gold_accessed": False, "dfs_executed": False, "dynamic_b2_executed": False,
        })
    finally:
        _release(model)


def _same(lhs: dict[str, Any], rhs: dict[str, Any]) -> dict[str, Any]:
    fields = ("full_logits_sha256", "arc_logits", "arc_logprobs", "arc_ranking", "retained_regret_successors")
    row = {f"{field}_same": lhs[field] == rhs[field] for field in fields}
    row["strict_forward_parity"] = all(row.values())
    return row


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    allowed_existing = {"COMPARATOR_FIX_AUDIT.md", "ISOLATED_PARITY_REAUDIT.csv"}
    existing = {path.name for path in output.iterdir()} if output.exists() else set()
    unexpected = existing - allowed_existing
    if unexpected:
        raise RuntimeError(f"refusing to overwrite {output}; unexpected files: {sorted(unexpected)}")
    challenge = args.challenge.resolve(); no_gold_challenge(challenge)
    source = read_json(args.fixed_budget_contract.resolve())
    caps = source.get("caps", {})
    if int(caps.get("max_expanded_nodes", -1)) != 4096 or int(caps.get("max_completed_candidates", -1)) != 32:
        raise RuntimeError("requires fixed Regret 4096/32 contract")
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "experiment": EXPERIMENT, "source_commit": args.source_commit, "target_blind": True,
        "gold_accessed": False, "dynamic_b2_executed": False, "dfs_executed": False,
        "task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH,
        "primary_view": PRIMARY, "foreign_view": HISTORY, "caps": caps,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "authoritative_root": str(args.authoritative_root.resolve()),
        "reference_config": str(args.reference_config.resolve()), "model_path": str(args.model_path.resolve()),
        "native_config_dir": str(args.native_config_dir.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()),
        "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()),
        "required_semantic_fields": list(REQUIRED),
        "execution": "four separate Python child processes; one fresh model each; one flip continuation each",
    }
    atomic_json(output / "PREFILL_STATE_ROOT_CAUSE_CONTRACT.json", contract)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = output / "PREFILL_STATE_ROOT_CAUSE_CONTRACT.json"
    if not contract.is_file():
        raise RuntimeError("prepare must write PREFILL_STATE_ROOT_CAUSE_CONTRACT.json first")
    conditions = (
        ("A1_NATIVE_NO_FOREIGN", "NATIVE", False), ("A2_NATIVE_FOREIGN_PREFILL", "NATIVE", True),
        ("B1_LEGACY_NO_FOREIGN", "LEGACY_DEEP_CLONE", False), ("B2_LEGACY_FOREIGN_PREFILL", "LEGACY_DEEP_CLONE", True),
    )
    for condition_id, cache_mode, foreign in conditions:
        result = output / "raw" / f"{condition_id}.json"
        if result.exists():
            raise RuntimeError(f"refusing to overwrite existing condition {result}")
        result.parent.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(Path(__file__).resolve()), "condition", "--contract", str(contract),
                   "--result", str(result), "--condition-id", condition_id, "--cache-mode", cache_mode,
                   "--gpu-id", str(args.gpu_id)]
        if foreign:
            command.append("--foreign-prefill")
        subprocess.run(command, check=True)
    records = {path.stem: read_json(path) for path in sorted((output / "raw").glob("*.json"))}
    if set(records) != {item[0] for item in conditions}:
        raise RuntimeError(f"incomplete condition records: {sorted(records)}")
    native = _same(records["A1_NATIVE_NO_FOREIGN"], records["A2_NATIVE_FOREIGN_PREFILL"])
    legacy = _same(records["B1_LEGACY_NO_FOREIGN"], records["B2_LEGACY_FOREIGN_PREFILL"])
    foreign_changes = not native["strict_forward_parity"]
    if foreign_changes and legacy["strict_forward_parity"]:
        decision = "NATIVE_CACHE_OBJECT_STATE_DEPENDENCY"; next_step = "NATIVE_CACHE_METADATA_AUDIT"
    elif foreign_changes and not legacy["strict_forward_parity"]:
        decision = "MODEL_LEVEL_PREFILL_MUTABLE_STATE_DEPENDENCY"; next_step = "MODEL_STATE_PREFILL_AUDIT"
    elif not foreign_changes and legacy["strict_forward_parity"]:
        decision = "FOREIGN_INCREMENTAL_HISTORY_REQUIRED"; next_step = "FRESH_PROCESS_HISTORY_LENGTH_SWEEP"
    else:
        decision = "NOT_ESTABLISHED"; next_step = "STOP_AND_REVIEW_ASYMMETRIC_CACHE_REPRESENTATION_RESULT"
    rows = []
    for key in sorted(records):
        record = records[key]
        rows.append({"condition_id": key, "cache_mode": record["cache_mode"], "foreign_prefill": record["foreign_prefill"],
                     "token_id": record["request_token_id"], "position": record["request_position"],
                     "ordinal": record["request_ordinal"], "cache_type": record["cache_used"]["cache_python_type"],
                     "cache_content_sha256": record["cache_used"].get("content_sha256"),
                     "full_logits_sha256": record["full_logits_sha256"], "arc_ranking": record["arc_ranking"],
                     "retained_regret_successors": record["retained_regret_successors"]})
    write_csv(output / "PREFILL_ONLY_CAUSAL_TEST.csv", rows[:2] + [
        {"comparison": "A1_NATIVE_NO_FOREIGN vs A2_NATIVE_FOREIGN_PREFILL", **native}
    ])
    write_csv(output / "CACHE_REPRESENTATION_2X2.csv", rows + [
        {"comparison": "A1 vs A2", **native}, {"comparison": "B1 vs B2", **legacy}
    ])
    placeholders = {
        "FOREIGN_HISTORY_LENGTH_SWEEP.csv": "NOT_REQUIRED_BY_DECISION_TREE" if foreign_changes else "PENDING_NEXT_DECISION_TREE_PHASE",
        "FIRST_DIVERGENT_LAYER.csv": "NOT_RUN_DUE_TO_PRIOR_GATE",
        "NATIVE_CACHE_METADATA_DIFF.json": "NOT_REQUIRED_BY_DECISION_TREE" if decision != "NATIVE_CACHE_OBJECT_STATE_DEPENDENCY" else "NOT_RUN_DUE_TO_PRIOR_GATE",
        "MODEL_STATE_PREFILL_DIFF.json": "NOT_REQUIRED_BY_DECISION_TREE" if decision != "MODEL_LEVEL_PREFILL_MUTABLE_STATE_DEPENDENCY" else "NOT_RUN_DUE_TO_PRIOR_GATE",
        "MICRO_FIX_PARITY.csv": "NOT_RUN_DUE_TO_PRIOR_GATE", "FINAL_ISOLATED_PARITY.csv": "NOT_RUN_DUE_TO_PRIOR_GATE",
        "FINAL_SHARED_B1_PARITY.csv": "NOT_RUN_DUE_TO_PRIOR_GATE",
    }
    for name, status in placeholders.items():
        path = output / name
        if path.suffix == ".json": atomic_json(path, {"status": status})
        else: path.write_text(f"status\n{status}\n", encoding="utf-8")
    outcome = {"experiment": EXPERIMENT, "causal_classification": decision, "next": next_step,
               "foreign_prefill_alone_changes_flip": foreign_changes, "native_pair": native, "legacy_pair": legacy,
               "gold_accessed": False, "dynamic_b2_safe_next": False, "dynamic_b2_executed": False}
    atomic_json(output / "DECISION.json", outcome)
    report = [f"# {EXPERIMENT}", "", "Target-blind prefill-only micro experiment; no Gold, DFS, or Dynamic B2.", "",
              f"- Foreign prefill changes native continuation: {foreign_changes}",
              f"- Native A1/A2 strict forward parity: {native['strict_forward_parity']}",
              f"- Legacy B1/B2 strict forward parity: {legacy['strict_forward_parity']}",
              f"- Causal classification: {decision}", f"- Next: {next_step}"]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    hashes = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*")) if path.is_file() and path.name != "HASHES.json"}
    atomic_json(output / "HASHES.json", {"sha256": hashes, "gold_accessed": False})


def history(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = output / "PREFILL_STATE_ROOT_CAUSE_CONTRACT.json"
    decision_path = output / "DECISION.json"
    decision = read_json(decision_path)
    if decision["causal_classification"] != "FOREIGN_INCREMENTAL_HISTORY_REQUIRED":
        raise RuntimeError(f"history sweep is not authorized by current decision: {decision['causal_classification']}")
    no_gold_challenge(Path(read_json(contract)["challenge"]))
    prior = output / "history_raw"
    prior.mkdir(exist_ok=True)
    for length in (0, 1, 2, 8, 32):
        result = prior / f"N{length}.json"
        if result.exists():
            raise RuntimeError(f"refusing to overwrite existing history condition {result}")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "history-condition", "--contract", str(contract),
                        "--result", str(result), "--history-length", str(length), "--gpu-id", str(args.gpu_id)], check=True)
    rows = [read_json(prior / f"N{length}.json") for length in (0, 1, 2, 8, 32)]
    baseline = rows[0]
    summary = []
    first_drift: int | None = None
    for row in rows:
        parity = _same(baseline, row)
        summary.append({"history_length": row["history_length_requested"], "history_length_executed": row["history_length_executed"],
                        "flip_token_id": row["request_token_id"], "flip_position": row["request_position"],
                        "flip_cache_before_sha256": row["request_cache_before"].get("content_sha256"),
                        "full_logits_sha256": row["full_logits_sha256"], "arc_ranking": row["arc_ranking"],
                        "retained_regret_successors": row["retained_regret_successors"], **parity})
        if first_drift is None and not parity["strict_forward_parity"]:
            first_drift = int(row["history_length_requested"])
    write_csv(output / "FOREIGN_HISTORY_LENGTH_SWEEP.csv", summary)
    if first_drift is None:
        decision["causal_classification"] = "NOT_ESTABLISHED"
        decision["next"] = "STOP_AND_REVIEW_NO_DRIFT_THROUGH_HISTORY32"
        decision["foreign_incremental_history_first_drift"] = "NONE_THROUGH_32"
    else:
        decision["causal_classification"] = "FOREIGN_INCREMENTAL_HISTORY_REQUIRED"
        decision["next"] = "FIRST_LEVEL_ROOT_CAUSE_LOCALIZATION"
        decision["foreign_incremental_history_first_drift"] = first_drift
    decision["history_sweep_source_commit"] = args.source_commit
    atomic_json(decision_path, decision)
    hashes = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*"))
              if path.is_file() and path.name != "HASHES.json"}
    atomic_json(output / "HASHES.json", {"sha256": hashes, "gold_accessed": False})


def _first_tensor(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def _trace_modules(model: Any) -> list[tuple[str, Any]]:
    # Unsloth may return a PEFT wrapper whose Qwen stack is at
    # ``base_model.model.model`` rather than at ``model.model``.  Locate the
    # real block container structurally, without assuming one wrapper layout.
    pending = [model]
    visited: set[int] = set()
    core = None
    while pending:
        candidate = pending.pop(0)
        if id(candidate) in visited:
            continue
        visited.add(id(candidate))
        layers = getattr(candidate, "layers", None)
        if layers is not None and len(layers) and all(hasattr(layer, "forward") for layer in layers):
            core = candidate
            break
        for name in ("base_model", "model"):
            child = getattr(candidate, name, None)
            if child is not None and hasattr(child, "forward"):
                pending.append(child)
    if core is None:
        raise RuntimeError("cannot locate transformer layer stack for diagnostic trace")
    modules: list[tuple[str, Any]] = []
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    embed = get_input_embeddings() if callable(get_input_embeddings) else getattr(core, "embed_tokens", None)
    if embed is not None:
        modules.append(("input_embedding", embed))
    layers = core.layers
    modules.extend((f"transformer_block_{index:02d}", layer) for index, layer in enumerate(layers))
    norm = getattr(core, "norm", None)
    if norm is not None:
        modules.append(("final_norm", norm))
    get_output_embeddings = getattr(model, "get_output_embeddings", None)
    head = get_output_embeddings() if callable(get_output_embeddings) else getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("cannot locate lm_head for diagnostic trace")
    modules.append(("lm_head", head))
    return modules


def _write_trace_condition(args: argparse.Namespace) -> None:
    """Trace good/bad fresh history conditions, with no search continuation."""
    import torch

    contract = read_json(args.contract.resolve())
    no_gold_challenge(Path(contract["challenge"]))
    model, encoded, dec, adapter_sha, native = _runtime(contract, args.gpu_id)
    captured: dict[str, dict[str, Any]] = {}
    try:
        flip = _new_cell(model, encoded, dec, PRIMARY); request = flip.request
        if request is None:
            raise RuntimeError("flip root prefill did not yield an incremental request")
        anti = _new_cell(model, encoded, dec, HISTORY)
        for _ in range(args.history_length):
            if anti.request is None:
                raise RuntimeError("anti completed before requested trace history")
            outputs, _summary = _one_forward(model, anti.request); _reply(anti, outputs)
        # Unsloth routes most blocks through patched fast-forward functions,
        # bypassing ordinary PyTorch module hooks.  The model's public
        # ``hidden_states`` output is therefore the first faithful per-block
        # observation point for this diagnostic forward.
        with torch.no_grad():
            outputs = model(input_ids=torch.tensor([[request.token_id]], device=model.device, dtype=torch.long),
                            position_ids=torch.tensor([[request.position]], device=model.device, dtype=torch.long),
                            past_key_values=request.cache, return_dict=True, use_cache=True,
                            output_hidden_states=True)
        summary = _logit_summary(outputs.logits)
        hidden = getattr(outputs, "hidden_states", None)
        if not hidden:
            raise RuntimeError("model did not return hidden_states for first-level diagnostic trace")
        for index, tensor in enumerate(hidden):
            label = "input_embedding" if index == 0 else f"transformer_block_{index - 1:02d}"
            cpu = tensor.detach().float().cpu().contiguous()
            captured[label] = {"shape": list(cpu.shape), "dtype": str(tensor.dtype),
                               "sha256": _tensor_sha(cpu), "values": cpu.tolist()}
        logits_cpu = outputs.logits[:, -1].detach().float().cpu().contiguous()
        captured["lm_head"] = {"shape": list(logits_cpu.shape), "dtype": str(outputs.logits.dtype),
                               "sha256": _tensor_sha(logits_cpu), "values": logits_cpu.tolist()}
        atomic_json(args.result.resolve(), {
            "history_length": args.history_length, "cell_key": f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{PRIMARY}",
            "request_token_id": int(request.token_id), "request_position": int(request.position),
            "request_ordinal": int(request.ordinal), "adapter_sha256": adapter_sha,
            "native_token_contract": native, "full_logits_sha256": summary["full_logits_sha256"],
            "arc_logits": summary["arc_logits"], "arc_logprobs": summary["arc_logprobs"],
            "arc_ranking": summary["arc_ranking"], "retained_regret_successors": _retained_successors(outputs.logits, dec),
            "stages": captured, "gold_accessed": False, "dfs_executed": False,
        })
    finally:
        _release(model)


def trace(args: argparse.Namespace) -> None:
    import torch

    output = args.output.resolve(); contract = output / "PREFILL_STATE_ROOT_CAUSE_CONTRACT.json"
    decision_path = output / "DECISION.json"; decision = read_json(decision_path)
    if decision.get("foreign_incremental_history_first_drift") in (None, "NONE_THROUGH_32"):
        raise RuntimeError("first-level trace requires an established history-length drift")
    trace_root = output / "layer_trace_hidden_states"; trace_root.mkdir(exist_ok=True)
    for history_length in (0, int(decision["foreign_incremental_history_first_drift"])):
        result = trace_root / f"N{history_length}.json"
        if result.exists():
            raise RuntimeError(f"refusing to overwrite trace record {result}")
        subprocess.run([sys.executable, str(Path(__file__).resolve()), "trace-condition", "--contract", str(contract),
                        "--result", str(result), "--history-length", str(history_length), "--gpu-id", str(args.gpu_id)], check=True)
    good, bad = (read_json(trace_root / "N0.json"), read_json(trace_root / f"N{decision['foreign_incremental_history_first_drift']}.json"))
    rows = []
    first = "NONE"
    for stage in good["stages"]:
        lhs, rhs = good["stages"][stage], bad["stages"].get(stage)
        if rhs is None:
            raise RuntimeError(f"bad trace omitted stage {stage}")
        left = torch.tensor(lhs["values"]); right = torch.tensor(rhs["values"])
        same = lhs["sha256"] == rhs["sha256"]
        row = {"stage": stage, "shape": lhs["shape"], "dtype": lhs["dtype"], "tensor_sha256_same": same,
               "max_abs_delta": float(torch.max(torch.abs(left - right)).item())}
        rows.append(row)
        if first == "NONE" and not same:
            first = stage
    write_csv(output / "FIRST_DIVERGENT_LAYER.csv", rows)
    decision["first_divergent_layer_or_operation"] = first
    decision["first_level_trace_source_commit"] = args.source_commit
    decision["next"] = "STOP_AFTER_FIRST_LEVEL_ROOT_CAUSE_LOCALIZATION"
    atomic_json(decision_path, decision)
    hashes = {path.relative_to(output).as_posix(): sha256_file(path) for path in sorted(output.rglob("*"))
              if path.is_file() and path.name != "HASHES.json"}
    atomic_json(output / "HASHES.json", {"sha256": hashes, "gold_accessed": False})


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for item in (prep,):
        item.add_argument("--output", type=Path, required=True); item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True); item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True); item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True); item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--source-commit", required=True)
    run_parser = sub.add_parser("run"); run_parser.add_argument("--output", type=Path, required=True); run_parser.add_argument("--gpu-id", type=int, default=0)
    history_parser = sub.add_parser("history"); history_parser.add_argument("--output", type=Path, required=True); history_parser.add_argument("--gpu-id", type=int, default=0); history_parser.add_argument("--source-commit", required=True)
    trace_parser = sub.add_parser("trace"); trace_parser.add_argument("--output", type=Path, required=True); trace_parser.add_argument("--gpu-id", type=int, default=0); trace_parser.add_argument("--source-commit", required=True)
    condition = sub.add_parser("condition"); condition.add_argument("--contract", type=Path, required=True); condition.add_argument("--result", type=Path, required=True)
    condition.add_argument("--condition-id", required=True); condition.add_argument("--cache-mode", choices=("NATIVE", "LEGACY_DEEP_CLONE"), required=True)
    condition.add_argument("--foreign-prefill", action="store_true"); condition.add_argument("--gpu-id", type=int, default=0)
    history_condition = sub.add_parser("history-condition"); history_condition.add_argument("--contract", type=Path, required=True); history_condition.add_argument("--result", type=Path, required=True)
    history_condition.add_argument("--history-length", type=int, choices=(0, 1, 2, 8, 32), required=True); history_condition.add_argument("--gpu-id", type=int, default=0)
    trace_condition = sub.add_parser("trace-condition"); trace_condition.add_argument("--contract", type=Path, required=True); trace_condition.add_argument("--result", type=Path, required=True)
    trace_condition.add_argument("--history-length", type=int, choices=(0, 1, 2, 8, 32), required=True); trace_condition.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    elif args.command == "run": run(args)
    elif args.command == "condition": _write_condition(args)
    elif args.command == "history": history(args)
    elif args.command == "trace": trace(args)
    elif args.command == "trace-condition": _write_trace_condition(args)
    else: _write_history_condition(args)


if __name__ == "__main__": main()
