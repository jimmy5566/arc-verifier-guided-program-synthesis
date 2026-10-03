#!/usr/bin/env python3
"""Same-contract, target-blind reproducibility gate for the AUG8 R1024 Core.

RUN_A and RUN_B invoke the existing production worker in separate fresh
subprocesses.  This controller intentionally has no solutions argument and
never imports evaluation solutions.  It preserves bounded local disk by
freezing each completed run as independently hash-verifiable gzip raw records;
the content hash is verified by streaming decompression before its temporary
JSON source is released.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "run_eval60_budget_eos_pilot18_v1.py"
SOURCE_COMMIT = "b849e821deb69f6c6cbfa193164aace020b00efd"
EXPERIMENT = "TTT24_AUG8_R1024_REPRODUCIBILITY_GATE_V1"
FRESH_PARTIAL_EXPERIMENT = "TTT24_AUG8_R1024_REPRODUCIBILITY_RERUN3_V1"
AUG8 = (
    "geom=identity__color=id__order=canonical",
    "geom=flip_ud__color=id__order=canonical",
    "geom=transpose__color=id__order=canonical",
    "geom=anti_transpose__color=id__order=canonical",
    "geom=rot90__color=id__order=canonical",
    "geom=rot180__color=id__order=canonical",
    "geom=rot270__color=id__order=canonical",
    "geom=flip_lr__color=id__order=canonical",
)
PROCESS_LOCAL_FIELDS = frozenset({
    "cache_owner_id", "cache_owner_id_after_reply", "process_id", "pid",
    "elapsed_seconds", "active_elapsed_seconds", "wall_seconds", "started_unix",
    "ended_unix", "prefill_seconds", "model_forward_seconds", "host_model_call_seconds",
    "host_cache_pack_seconds", "host_cache_adoption_seconds", "host_scheduler_overhead_seconds",
    "memory_allocated_bytes", "memory_driver_free_bytes", "memory_fragmentation_ratio",
    "memory_reserved_bytes", "memory_reserved_unallocated_bytes", "b16_peak_memory",
    "peak_allocated_bytes", "peak_reserved_bytes", "cuda_memory_address",
})
NUMERIC_FIELDS = frozenset({
    "token_logprob", "cumulative_score", "cumulative_regret", "path_cumulative_nll",
    "path_cumulative_regret", "top1_logprob", "top2_logprob", "entropy", "margin",
    "incremental_regret", "selected_token_logprob", "logprob",
})


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_canonical(payload) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _canonical(value) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})
    os.replace(temporary, path)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha_gzip_content(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_raw(run: Path, output_id: str) -> dict[str, Any]:
    plain = run / "RAW_OUTPUTS" / f"{_safe(output_id)}.json"
    frozen = run / "FROZEN_RAW_OUTPUTS" / f"{_safe(output_id)}.json.gz"
    if plain.exists():
        return _read_json(plain)
    if frozen.exists():
        with gzip.open(frozen, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    raise FileNotFoundError(f"missing raw output for {output_id}: {plain} or {frozen}")


def _clean(value: Any, *, numeric: bool = False) -> Any:
    """Normalize process-local records without excluding scientific values."""
    if isinstance(value, dict):
        return {str(key): _clean(item, numeric=numeric) for key, item in value.items()
                if key not in PROCESS_LOCAL_FIELDS and (numeric or key not in NUMERIC_FIELDS)}
    if isinstance(value, list):
        return [_clean(item, numeric=numeric) for item in value]
    if isinstance(value, tuple):
        return [_clean(item, numeric=numeric) for item in value]
    return value


def _prefill_projection(cell: dict[str, Any]) -> dict[str, Any]:
    trace = cell.get("diagnostic_trace", {}).get("prefill")
    if not isinstance(trace, dict):
        return {"missing": True}
    request = trace.get("first_ready_request")
    if isinstance(request, dict):
        request = {key: value for key, value in request.items() if key != "cache_owner_id"}
    return {
        "prompt_token_ids": trace.get("prompt_token_ids"),
        "prompt_length": trace.get("prompt_length"),
        "full_logits_sha256": trace.get("full_logits_sha256"),
        "root_cache_sha256": trace.get("root_cache_sha256"),
        "root_cache_valid_length": trace.get("root_cache_valid_length"),
        "root_cache_geometry": trace.get("root_cache_geometry"),
        "first_ready_request": request,
    }


def _scheduler_projection(raw: dict[str, Any]) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for event in raw.get("scheduler_events", []):
        projected.append({
            "event": event.get("event"),
            "cell_key": event.get("cell_key"),
            "selected_cell_keys": event.get("selected_cell_keys", event.get("cell_keys", [])),
            "physical_batch": event.get("physical_batch", event.get("actual_batch_width")),
            "request_position": event.get("request_position", event.get("position")),
            "compatibility_class": event.get("compatibility_class", event.get("cache_geometry")),
            "termination_reason": event.get("termination_reason"),
            "resident_count": event.get("resident_count"),
            "pending_count": event.get("pending_count"),
            "request_ordinals": event.get("request_ordinals"),
        })
    return projected


def _logical_projection(cell: dict[str, Any]) -> dict[str, Any]:
    trace = cell.get("diagnostic_trace", {})
    return {
        "nodes": _clean(cell.get("nodes", [])),
        "search_trace": _clean(trace.get("search_trace", [])),
        "frontier_samples": _clean(trace.get("frontier_samples", [])),
        "events": _clean(cell.get("events", [])),
        "final_candidate_pool": _clean(cell.get("final_candidate_pool", [])),
        "nodes_expanded": cell.get("nodes_expanded"),
        "completed_candidates": cell.get("completed_candidates"),
        "termination_reason": cell.get("termination_reason"),
        "budget_exhausted": cell.get("budget_exhausted"),
    }


def _numeric_projection(cell: dict[str, Any]) -> dict[str, Any]:
    trace = cell.get("diagnostic_trace", {})
    return {
        "nodes": _clean(cell.get("nodes", []), numeric=True),
        "branch_probabilities": _clean(trace.get("branch_probabilities", []), numeric=True),
        "search_trace": _clean(trace.get("search_trace", []), numeric=True),
    }


def _first_difference(left: Any, right: Any, *, label: str) -> dict[str, Any] | None:
    if left == right:
        return None
    if isinstance(left, list) and isinstance(right, list):
        for index, (a, b) in enumerate(zip(left, right, strict=False)):
            if a != b:
                return {"sequence": label, "step": index, "left": a, "right": b}
        return {"sequence": label, "step": min(len(left), len(right)),
                "left_count": len(left), "right_count": len(right)}
    if isinstance(left, dict) and isinstance(right, dict):
        fields = sorted(key for key in set(left) | set(right) if left.get(key) != right.get(key))
        return {"sequence": label, "step": 0, "fields": fields,
                "left": {key: left.get(key) for key in fields}, "right": {key: right.get(key) for key in fields}}
    return {"sequence": label, "step": 0, "left": left, "right": right}


def _numeric_drift(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    token_deltas: list[float] = []
    score_deltas: list[float] = []
    regret_deltas: list[float] = []

    def scalar_pairs(a: Any, b: Any) -> None:
        if isinstance(a, dict) and isinstance(b, dict):
            for key in set(a).intersection(b):
                if key in {"token_logprob", "top1_logprob", "top2_logprob", "selected_token_logprob", "logprob"}:
                    if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
                        token_deltas.append(abs(float(a[key]) - float(b[key])))
                elif key in {"cumulative_score", "path_cumulative_nll"}:
                    if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
                        score_deltas.append(abs(float(a[key]) - float(b[key])))
                elif key in {"cumulative_regret", "path_cumulative_regret", "incremental_regret"}:
                    if isinstance(a[key], (int, float)) and isinstance(b[key], (int, float)):
                        regret_deltas.append(abs(float(a[key]) - float(b[key])))
                else:
                    scalar_pairs(a[key], b[key])
        elif isinstance(a, list) and isinstance(b, list):
            for x, y in zip(a, b, strict=False):
                scalar_pairs(x, y)

    scalar_pairs(left, right)
    return {
        "max_abs_token_logprob_difference": max(token_deltas, default=0.0),
        "median_abs_token_logprob_difference": statistics.median(token_deltas) if token_deltas else 0.0,
        "max_cumulative_score_difference": max(score_deltas, default=0.0),
        "max_cumulative_regret_difference": max(regret_deltas, default=0.0),
        "numeric_values_compared": len(token_deltas) + len(score_deltas) + len(regret_deltas),
        "number_of_logical_steps_with_numeric_drift": sum(value > 0.0 for value in token_deltas + score_deltas + regret_deltas),
    }


def _cache_invariants(raw: dict[str, Any]) -> dict[str, Any]:
    owner_violations: list[dict[str, Any]] = []
    rollback_violations: list[dict[str, Any]] = []
    alias_violations: list[dict[str, Any]] = []
    batch_violations: list[dict[str, Any]] = []
    ceiling = int(raw["profile_configuration"]["physical_batch_ceiling"])
    seen_owner: dict[str, int] = {}
    for cell_key, cell in raw.get("cells", {}).items():
        trace = cell.get("diagnostic_trace", {})
        prefill = trace.get("prefill", {}) if isinstance(trace, dict) else {}
        owner = prefill.get("cache_owner_id")
        if not isinstance(owner, int):
            owner_violations.append({"cell_key": cell_key, "reason": "MISSING_PREFILL_OWNER"})
        elif owner in seen_owner:
            alias_violations.append({"cell_key": cell_key, "other_cell_key": seen_owner[owner], "owner": owner})
        else:
            seen_owner[owner] = cell_key
        advances = trace.get("logical_advances", []) if isinstance(trace, dict) else []
        if not advances:
            owner_violations.append({"cell_key": cell_key, "reason": "MISSING_PER_FORWARD_OWNER_TRACE"})
        for step, advance in enumerate(advances):
            if advance.get("cache_owner_id") != owner or advance.get("cache_owner_id_after_reply") != owner:
                owner_violations.append({"cell_key": cell_key, "step": step, "reason": "OWNER_SWAP"})
            position = advance.get("absolute_position")
            valid = advance.get("valid_kv_length")
            output_valid = advance.get("output_valid_kv_length")
            if valid != position or output_valid != (int(position) + 1 if isinstance(position, int) else None):
                rollback_violations.append({"cell_key": cell_key, "step": step, "position": position,
                                            "valid": valid, "output_valid": output_valid})
            width = int(advance.get("physical_batch_width", 0))
            if width < 1 or width > ceiling:
                batch_violations.append({"cell_key": cell_key, "step": step, "width": width, "ceiling": ceiling,
                                         "reason": "CEILING"})
    for event in raw.get("scheduler_events", []):
        if event.get("event") not in {"FORWARD", "FORWARD_COMPLETE"}:
            continue
        width = event.get("physical_batch", event.get("actual_batch_width"))
        keys = event.get("selected_cell_keys", event.get("cell_keys", []))
        if width is not None and int(width) != len(keys):
            batch_violations.append({"event": event.get("event_index"), "reason": "WIDTH_MEMBER_MISMATCH",
                                     "width": width, "members": keys})
    return {
        "status": "PASS" if not (owner_violations or rollback_violations or alias_violations or batch_violations) else "FAIL",
        "no_owner_swap": not owner_violations,
        "no_cross_cell_cache_aliasing": not alias_violations,
        "rollback_valid_lengths_coherent": not rollback_violations,
        "no_incompatible_physical_batch": not batch_violations,
        "packed_batch_temporaries_durable_state": "NOT_OBSERVED_IN_RAW_TRACE",
        "reply_returns_to_correct_logical_cell": not owner_violations,
        "owner_violations": owner_violations,
        "rollback_violations": rollback_violations,
        "alias_violations": alias_violations,
        "batch_violations": batch_violations,
    }


def _compare_cell(output_id: str, augmentation_id: str, run_a: dict[str, Any], run_b: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    a = next(cell for cell in run_a["cells"].values() if cell["augmentation_id"] == augmentation_id)
    b = next(cell for cell in run_b["cells"].values() if cell["augmentation_id"] == augmentation_id)
    prefill_equal = _prefill_projection(a) == _prefill_projection(b)
    logical_a, logical_b = _logical_projection(a), _logical_projection(b)
    logical_equal = logical_a == logical_b
    candidates_a = logical_a["final_candidate_pool"]
    candidates_b = logical_b["final_candidate_pool"]
    candidate_order_equal = candidates_a == candidates_b
    candidate_set_equal = set(_canonical(value) for value in candidates_a) == set(_canonical(value) for value in candidates_b)
    termination_equal = logical_a["termination_reason"] == logical_b["termination_reason"]
    numeric_a, numeric_b = _numeric_projection(a), _numeric_projection(b)
    numeric_exact = numeric_a == numeric_b
    drift = _numeric_drift(numeric_a, numeric_b)
    first: dict[str, Any] | None = None
    if not logical_equal:
        for name in ("nodes", "search_trace", "frontier_samples", "events", "final_candidate_pool"):
            first = _first_difference(logical_a[name], logical_b[name], label=name)
            if first is not None:
                break
        if first is None:
            first = _first_difference(logical_a, logical_b, label="endpoint")
        classification = "LOGICAL_DIVERGENCE"
    elif numeric_exact:
        classification = "EXACT_SEMANTIC_AND_NUMERIC"
    else:
        classification = "SEMANTIC_EXACT_NUMERIC_DRIFT"
    row = {
        "output_id": output_id, "profile": run_a["output"]["profile"], "augmentation_id": augmentation_id,
        "root_length": run_a["root_admission"]["root_lengths"].get(a["cell_key"]),
        "run_a_nodes": a.get("nodes_expanded"), "run_b_nodes": b.get("nodes_expanded"),
        "logical_trace_equal": logical_equal, "candidate_set_equal": candidate_set_equal,
        "candidate_order_equal": candidate_order_equal, "termination_equal": termination_equal,
        "scheduler_semantic_equal": None, "prefill_semantic_equal": prefill_equal,
        "numeric_exact": numeric_exact, "first_divergence_step": None if first is None else first.get("step"),
        "classification": classification,
    }
    detail = {"output_id": output_id, "augmentation_id": augmentation_id, "classification": classification,
              "first_divergence": first, "numeric_drift": drift,
              "prefill_a": _prefill_projection(a), "prefill_b": _prefill_projection(b)}
    return row, detail


def _profile_configs(policy: dict[str, Any], profile: str) -> list[dict[str, int]]:
    row = policy["profiles"][profile]
    candidates = [row["primary"], *row["fallbacks"]]
    values: list[dict[str, int]] = []
    for item in candidates:
        bounded = {"resident_capacity": min(8, int(item["resident_capacity"])),
                   "physical_batch_ceiling": min(8, int(item["physical_batch_ceiling"]), min(8, int(item["resident_capacity"]))) }
        if bounded not in values:
            values.append(bounded)
    return values


def _experiment_id(args: argparse.Namespace) -> str:
    """Keep a fresh, user-authorized partial rerun distinct from reused evidence."""
    return FRESH_PARTIAL_EXPERIMENT if args.fresh_partial_rerun else EXPERIMENT


def _prepare_static(args: argparse.Namespace) -> list[dict[str, Any]]:
    output = args.output
    source_cohort = _read_json(args.source_core / "RUN_COHORT.json")
    outputs = source_cohort.get("outputs", [])
    if len(outputs) != 6:
        raise RuntimeError(f"expected frozen 2S+2M+2L source cohort, found {len(outputs)}")
    profiles = [str(item["profile"]) for item in outputs]
    if sum(profile == "PROFILE_S" for profile in profiles) != 2 or sum(profile == "PROFILE_M" for profile in profiles) != 2 or sum(profile.startswith("PROFILE_L") for profile in profiles) != 2:
        raise RuntimeError("source cohort is not the required deterministic 2S+2M+2L Core smoke")
    source_aug = _read_json(args.source_core / "AUG8_IDS.json")
    if tuple(source_aug.get("candidate_ids", [])) != AUG8:
        raise RuntimeError("source Core AUG8 identity differs from canonical AUG8")
    policy = _read_json(args.coarse_policy)
    model_identity = {
        "model_path": str(args.model_path), "model_contract": "Qwen3ForCausalLM BF16 Clean HuggingFace Transformers + PEFT",
        "native_config_dir": str(args.native_config_dir), "tokenizer_contract": "native ARC tokenizer",
        "eos": 15, "pad": 13, "source_core": str(args.source_core), "source_core_commit": SOURCE_COMMIT,
    }
    adapters = {item["output_id"]: item["adapter_identity"] for item in outputs}
    contract = {
        "experiment": _experiment_id(args), "target_blind": True, "gold_loaded": False,
        "contract": "TTT24_AUG8_R1024_ROOT_AWARE", "ttt_depth": 24, "augmentation_ids": list(AUG8),
        "max_expanded_nodes": 1024, "decoder": "CUMULATIVE_REGRET_r=4.00", "max_new_tokens": 931,
        "max_completed_candidates": 32, "frontier_floor": 1, "eos": 15,
        "admission_policy": "root_aware", "fairness_max_wait": 3,
        "cache": "ChunkedDynamicCache 256-token blocks; stable owner; rollback valid_length",
        "physical_batch_rule": "min(compatible READY class size, frozen safe ceiling, resident availability)",
        "fresh_worker_subprocess_per_output": True, "raw_codec_after_freeze": "gzip with raw-content SHA256 verification",
        "source_core_commit": SOURCE_COMMIT,
        "profile_safe_configs": {item["profile"]: _profile_configs(policy, item["profile"]) for item in outputs},
    }
    gate = {
        "frozen_before_gpu": True, "semantic_reproducibility_required": 1.0,
        "classifications": ["EXACT_SEMANTIC_AND_NUMERIC", "SEMANTIC_EXACT_NUMERIC_DRIFT", "LOGICAL_DIVERGENCE"],
        "excluded_process_local_fields": sorted(PROCESS_LOCAL_FIELDS),
        "numerical_fields_measured_separately": sorted(NUMERIC_FIELDS),
        "acceptance": "all cells semantic-exact and all runtime invariants PASS",
        "phase2_authorized_only_if_pass": True, "phase2_started": False,
    }
    _write_json(output / "CONTRACT.json", contract)
    _write_json(output / "COHORT.json", {"target_blind": True, "gold_loaded": False, "outputs": outputs})
    _write_json(output / "MODEL_IDENTITY.json", model_identity)
    _write_json(output / "ADAPTER_IDENTITIES.json", adapters)
    _write_json(output / "AUG8_IDS.json", source_aug)
    _write_json(output / "GATE_DEFINITION.json", gate)
    return outputs


def _run_worker(args: argparse.Namespace, run: Path, output_row: dict[str, Any], attempt: int, config: dict[str, int]) -> dict[str, Any]:
    command = [str(args.python), str(WORKER), "--mode", "worker", "--output", str(run), "--cohort-file", "RUN_COHORT.json",
               "--output-id", str(output_row["output_id"]), "--attempt", str(attempt),
               "--resident", str(config["resident_capacity"]), "--ceiling", str(config["physical_batch_ceiling"]),
               "--model-path", str(args.model_path), "--challenge", str(args.challenge),
               "--native-config-dir", str(args.native_config_dir), "--candidate-pool", str(args.candidate_pool),
               "--aug16-ids", str(run / "AUG8_IDS.json"), "--adapter-stage", str(run / "ADAPTER_STAGE"),
               "--max-expanded-nodes", "1024", "--ttt-depth", "24", "--augmentation-label", "aug8",
               "--admission-policy", "root_aware", "--fairness-max-wait", "3", "--checkpoints", "512,1024",
               "--diagnostic-trace", "--experiment", _experiment_id(args), "--device", str(args.device)]
    log = run / "WORKER_LOGS" / f"{_safe(str(output_row['output_id']))}_attempt{attempt}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    with log.open("w", encoding="utf-8", newline="\n") as handle:
        result = subprocess.run(command, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return {"output_id": output_row["output_id"], "attempt": attempt, "configuration": config,
            "fresh_subprocess": True, "returncode": result.returncode, "started_unix": started,
            "ended_unix": time.time(), "log": str(log.relative_to(run))}


def _freeze_completed_raw(run: Path, output_id: str) -> None:
    """Atomically retain one frozen raw result without accumulating six JSONs.

    This never changes a worker result before its content hash has been
    reproduced from the compressed frozen record.  It is an execution-storage
    measure only; the decompressed bytes remain the worker's original JSON.
    """
    raw_dir = run / "RAW_OUTPUTS"
    frozen_dir = run / "FROZEN_RAW_OUTPUTS"; frozen_dir.mkdir(parents=True, exist_ok=True)
    name = f"{_safe(output_id)}.json"
    raw, gz = raw_dir / name, frozen_dir / f"{name}.gz"
    if gz.exists():
        return
    if not raw.exists():
        raise RuntimeError(f"raw output missing before per-output freeze: {raw}")
    content_path = run / "PARTIAL_RAW_CONTENT.json"
    content = _read_json(content_path) if content_path.exists() else {}
    if str(gz.relative_to(run)) in content:
        raise RuntimeError(f"raw-content ledger already contains missing frozen output: {output_id}")
    raw_hash = _sha_file(raw)
    temporary = gz.with_suffix(gz.suffix + ".tmp")
    with raw.open("rb") as source, gzip.open(temporary, "wb", compresslevel=6) as sink:
        shutil.copyfileobj(source, sink, 1024 * 1024)
    os.replace(temporary, gz)
    if _sha_gzip_content(gz) != raw_hash:
        raise RuntimeError(f"gzip content hash mismatch: {gz}")
    content[str(gz.relative_to(run))] = {"raw_sha256": raw_hash, "gzip_sha256": _sha_file(gz),
                                          "raw_bytes": raw.stat().st_size, "gzip_bytes": gz.stat().st_size}
    _write_json(content_path, content)
    raw.unlink()


def _freeze_run(run: Path, outputs: list[dict[str, Any]], *, allow_preserved_extra: bool = False) -> dict[str, Any]:
    content_path = run / "PARTIAL_RAW_CONTENT.json"
    content = _read_json(content_path) if content_path.exists() else {}
    selected_names = {f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz" for row in outputs}
    missing = sorted(selected_names.difference(content))
    if missing:
        raise RuntimeError(f"frozen raw content missing selected outputs: {missing}")
    preserved_extra = sorted(set(content).difference(selected_names))
    if preserved_extra and not allow_preserved_extra:
        raise RuntimeError(f"unexpected extra independently frozen raw outputs: {preserved_extra}")
    selected_content = {name: content[name] for name in sorted(selected_names)}
    for row in outputs:
        name = f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz"
        if name not in content:
            raise RuntimeError(f"frozen raw content missing from ledger: {name}")
        gz = run / name
        expected = content[name]
        if not gz.exists() or _sha_file(gz) != expected["gzip_sha256"] or _sha_gzip_content(gz) != expected["raw_sha256"]:
            raise RuntimeError(f"frozen raw content failed verification before run freeze: {gz}")
    # Generated checkpoint/receipt/EOS evidence is still retained, while raw
    # JSON is represented by its independently verified compressed content.
    files = []
    for directory in ("FROZEN_RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "OUTPUT_RECEIPTS", "EOS_EVENTS", "OOM_FALLBACK_RECEIPTS.csv", "RUN_COHORT.json", "AUG8_IDS.json"):
        item = run / directory
        if item.is_file(): files.append(item)
        elif item.exists(): files.extend(path for path in item.rglob("*") if path.is_file())
    ledger = {str(path.relative_to(run)): _sha_file(path) for path in sorted(files)}
    generation = {"status": "FROZEN", "target_blind": True, "gold_loaded": False,
                  "raw_codec": "gzip", "raw_content": selected_content, "files": ledger,
                  "preserved_extra_frozen_records": preserved_extra}
    _write_json(run / "GENERATION_HASHES.json", generation)
    mismatches = []
    for rel, expected in ledger.items():
        observed = _sha_file(run / rel) if (run / rel).exists() else None
        if observed != expected: mismatches.append({"path": rel, "expected": expected, "actual": observed})
    for rel, expected in selected_content.items():
        path = run / rel
        if not path.exists() or _sha_file(path) != expected["gzip_sha256"] or _sha_gzip_content(path) != expected["raw_sha256"]:
            mismatches.append({"path": rel, "reason": "COMPRESSED_RAW_HASH_MISMATCH"})
    verification = {"status": "PASS" if not mismatches else "FAIL", "checked": len(ledger) + len(content), "mismatches": mismatches}
    _write_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS": raise RuntimeError(f"generation hash verification failed: {run}")
    _write_json(run / "GENERATION_FREEZE.json", {"status": "FROZEN", "target_blind": True, "gold_loaded": False,
                                                    "raw_count": len(selected_content), "raw_codec": "gzip", "hash_verification": "PASS",
                                                    "preserved_extra_frozen_records": preserved_extra})
    return verification


def _validated_frozen_prefix(run: Path, outputs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return the verified completed source-order prefix, rejecting any gap."""
    content_path = run / "PARTIAL_RAW_CONTENT.json"
    if not content_path.exists():
        raise RuntimeError(f"missing independent raw-content ledger: {content_path}")
    content = _read_json(content_path)
    completed: list[dict[str, Any]] = []
    for row in outputs:
        rel = f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz"
        if rel not in content:
            break
        expected = content[rel]
        path = run / rel
        if not path.exists() or _sha_file(path) != expected["gzip_sha256"] or _sha_gzip_content(path) != expected["raw_sha256"]:
            raise RuntimeError(f"independently frozen raw record failed verification: {path}")
        completed.append(row)
    completed_names = {f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz" for row in completed}
    selected_names = {f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz" for row in outputs}
    unexpected = sorted(set(content).difference(selected_names))
    if unexpected:
        raise RuntimeError(f"partial RUN_B contains out-of-scope frozen raw records: {unexpected}")
    later = selected_names.difference(completed_names)
    if any(name in content for name in later):
        raise RuntimeError("partial RUN_B frozen records are not a source-order prefix")
    return completed, content


def _pause_partial_run(run: Path, outputs: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze an interrupted RUN_B prefix without pretending the run completed."""
    completed, content = _validated_frozen_prefix(run, outputs)
    if not completed:
        raise RuntimeError("refusing to pause-freeze RUN_B without a completed, independently verified raw output")
    raw_dir = run / "RAW_OUTPUTS"
    plain = sorted(raw_dir.glob("*.json")) if raw_dir.exists() else []
    if plain:
        raise RuntimeError(f"refusing pause freeze with mutable plaintext raw outputs: {plain}")
    completed_names = [f"FROZEN_RAW_OUTPUTS/{_safe(str(row['output_id']))}.json.gz" for row in completed]
    files = [run / name for name in completed_names]
    for name in ("RUN_COHORT.json", "AUG8_IDS.json", "PARTIAL_RAW_CONTENT.json"):
        path = run / name
        if path.exists():
            files.append(path)
    ledger = {str(path.relative_to(run)): _sha_file(path) for path in sorted(files)}
    payload = {
        "status": "PAUSED_FROZEN_PREFIX", "target_blind": True, "gold_loaded": False,
        "completed_output_ids": [str(row["output_id"]) for row in completed],
        "remaining_output_ids": [str(row["output_id"]) for row in outputs[len(completed):]],
        "raw_content": {name: content[name] for name in completed_names}, "files": ledger,
        "resume_allowed_only_with": "--resume-partial-run-b and the same PARTIAL_EXECUTION_AMENDMENT",
    }
    _write_json(run / "PAUSED_GENERATION_HASHES.json", payload)
    mismatches = [{"path": rel, "expected": expected, "actual": _sha_file(run / rel) if (run / rel).exists() else None}
                  for rel, expected in ledger.items() if not (run / rel).exists() or _sha_file(run / rel) != expected]
    for name in completed_names:
        expected = content[name]
        path = run / name
        if not path.exists() or _sha_file(path) != expected["gzip_sha256"] or _sha_gzip_content(path) != expected["raw_sha256"]:
            mismatches.append({"path": name, "reason": "COMPRESSED_RAW_HASH_MISMATCH"})
    verification = {"status": "PASS" if not mismatches else "FAIL", "checked": len(ledger) + len(completed_names), "mismatches": mismatches}
    _write_json(run / "PAUSED_GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError("paused RUN_B hash verification failed")
    _write_json(run / "PAUSED_STATE.json", {
        "status": "PAUSED", "target_blind": True, "gold_loaded": False,
        "completed_output_count": len(completed), "total_output_count": len(outputs),
        "completed_output_ids": payload["completed_output_ids"], "remaining_output_ids": payload["remaining_output_ids"],
        "hash_verification": "PASS", "generation_complete": False, "comparison_complete": False,
    })
    return verification


def _run_one(args: argparse.Namespace, run: Path, outputs: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    if args.reuse_partial_run_a and run.name == "RUN_A":
        return _freeze_run(run, outputs, allow_preserved_extra=True)
    if (run / "GENERATION_FREEZE.json").exists():
        verification = _read_json(run / "GENERATION_HASH_VERIFICATION.json")
        if verification.get("status") != "PASS": raise RuntimeError(f"pre-existing frozen run fails verification: {run}")
        return verification
    completed_ids: set[str] = set()
    if run.exists() and any(run.iterdir()):
        if not (args.resume_partial_run_b and run.name == "RUN_B"):
            raise RuntimeError(f"refusing to reuse incomplete mutable runtime state: {run}")
        paused = _read_json(run / "PAUSED_STATE.json") if (run / "PAUSED_STATE.json").exists() else {}
        paused_verification = _read_json(run / "PAUSED_GENERATION_HASH_VERIFICATION.json") if (run / "PAUSED_GENERATION_HASH_VERIFICATION.json").exists() else {}
        if paused.get("status") != "PAUSED" or paused_verification.get("status") != "PASS":
            raise RuntimeError("partial RUN_B lacks a verified paused-state receipt")
        completed, _ = _validated_frozen_prefix(run, outputs)
        if [str(row["output_id"]) for row in completed] != paused.get("completed_output_ids"):
            raise RuntimeError("paused RUN_B selected-prefix identity changed")
        completed_ids = {str(row["output_id"]) for row in completed}
        _write_json(run / "RESUME_RECEIPT.json", {"status": "RESUMED", "target_blind": True, "gold_loaded": False,
                                                  "completed_output_ids_reused": sorted(completed_ids),
                                                  "remaining_output_ids": [str(row["output_id"]) for row in outputs if str(row["output_id"]) not in completed_ids]})
    else:
        free = shutil.disk_usage(run.parent).free
        if free < args.minimum_free_bytes:
            raise RuntimeError(f"insufficient local disk for independent raw run: free={free}, required={args.minimum_free_bytes}")
        run.mkdir(parents=True, exist_ok=False)
        _write_json(run / "RUN_COHORT.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "outputs": outputs})
        _write_json(run / "AUG8_IDS.json", {"subset": "CANONICAL_GEOMETRY_AUG8", "candidate_ids": list(AUG8)})
    receipts: list[dict[str, Any]] = []
    for output_row in outputs:
        if str(output_row["output_id"]) in completed_ids:
            continue
        done = False
        for attempt, config in enumerate(_profile_configs(policy, str(output_row["profile"]))):
            receipt = _run_worker(args, run, output_row, attempt, config)
            # Adapter staging is process-local load state.  It is not a
            # candidate/checkpoint artifact and must not crowd out the next
            # independently frozen raw output on the 30-GB local volume.
            shutil.rmtree(run / "ADAPTER_STAGE", ignore_errors=True)
            receipt["fallback_used"] = attempt > 0
            receipts.append(receipt)
            if receipt["returncode"] == 0:
                _freeze_completed_raw(run, str(output_row["output_id"]))
                done = True; break
            if receipt["returncode"] != 2:
                _write_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
                raise RuntimeError(f"worker failed outside frozen OOM fallback: {receipt['log']}")
        if not done:
            _write_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
            raise RuntimeError(f"all frozen OOM fallbacks failed: {output_row['output_id']}")
    if receipts:
        _write_csv(run / "OOM_FALLBACK_RECEIPTS.csv", receipts, receipts[0].keys())
    return _freeze_run(run, outputs)


def _compare_runs(output: Path, outputs: list[dict[str, Any]], *, partial_validation: bool = False) -> dict[str, Any]:
    a_root, b_root = output / "RUN_A", output / "RUN_B"
    cell_rows: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    prefill_rows: list[dict[str, Any]] = []
    scheduler_rows: list[dict[str, Any]] = []
    drift_rows: list[dict[str, Any]] = []
    cache = {"RUN_A": [], "RUN_B": []}
    for output_row in outputs:
        output_id = str(output_row["output_id"])
        raw_a, raw_b = _read_raw(a_root, output_id), _read_raw(b_root, output_id)
        scheduler_equal = _scheduler_projection(raw_a) == _scheduler_projection(raw_b)
        scheduler_rows.append({"output_id": output_id, "event_count_a": len(raw_a.get("scheduler_events", [])),
                               "event_count_b": len(raw_b.get("scheduler_events", [])), "semantic_equal": scheduler_equal})
        cache["RUN_A"].append({"output_id": output_id, **_cache_invariants(raw_a)})
        cache["RUN_B"].append({"output_id": output_id, **_cache_invariants(raw_b)})
        for augmentation_id in AUG8:
            row, detail = _compare_cell(output_id, augmentation_id, raw_a, raw_b)
            row["scheduler_semantic_equal"] = scheduler_equal
            # A scheduler mismatch is scientifically logical even if a one-cell
            # projection happens to finish at the same endpoint.
            if not scheduler_equal and row["classification"] != "LOGICAL_DIVERGENCE":
                row["classification"] = "LOGICAL_DIVERGENCE"
                detail["first_divergence"] = _first_difference(_scheduler_projection(raw_a), _scheduler_projection(raw_b), label="scheduler")
            cell_rows.append(row); details.append(detail)
            drift_rows.append({"output_id": output_id, "augmentation_id": augmentation_id, **detail["numeric_drift"]})
            prefill_rows.append({"output_id": output_id, "augmentation_id": augmentation_id,
                                 "semantic_equal": row["prefill_semantic_equal"],
                                 "excluded": "first_ready_request.cache_owner_id"})
    _write_csv(output / "CELL_REPRODUCIBILITY.csv", cell_rows, cell_rows[0].keys())
    _write_csv(output / "PREFILL_REPRODUCIBILITY.csv", prefill_rows, prefill_rows[0].keys())
    _write_csv(output / "SCHEDULER_REPRODUCIBILITY.csv", scheduler_rows, scheduler_rows[0].keys())
    _write_csv(output / "NUMERICAL_DRIFT_SUMMARY.csv", drift_rows, drift_rows[0].keys())
    divergences = [detail for detail in details if detail["classification"] == "LOGICAL_DIVERGENCE"]
    _write_csv(output / "FIRST_DIVERGENCE.csv", [
        {"output_id": row["output_id"], "augmentation_id": row["augmentation_id"],
         "first_divergence": _canonical(row["first_divergence"])} for row in divergences
    ] or [{"output_id": "", "augmentation_id": "", "first_divergence": "NONE"}],
        ("output_id", "augmentation_id", "first_divergence"))
    cache_audit = {name: entries for name, entries in cache.items()}
    cache_audit["status"] = "PASS" if all(entry["status"] == "PASS" for entries in cache.values() for entry in entries) else "FAIL"
    _write_json(output / "CACHE_INVARIANT_AUDIT.json", cache_audit)
    endpoint_rows = []
    for row in cell_rows:
        endpoint_rows.append({key: row[key] for key in ("output_id", "augmentation_id", "run_a_nodes", "run_b_nodes", "candidate_set_equal", "candidate_order_equal", "termination_equal")})
    _write_csv(output / "ENDPOINT_REPRODUCIBILITY.csv", endpoint_rows, endpoint_rows[0].keys())
    total = len(cell_rows)
    counts = {name: sum(row["classification"] == name for row in cell_rows) for name in
              ("EXACT_SEMANTIC_AND_NUMERIC", "SEMANTIC_EXACT_NUMERIC_DRIFT", "LOGICAL_DIVERGENCE")}
    semantic = counts["EXACT_SEMANTIC_AND_NUMERIC"] + counts["SEMANTIC_EXACT_NUMERIC_DRIFT"]
    passed = semantic == total and cache_audit["status"] == "PASS"
    decision = {
        "classification": ("PARTIAL_REPRODUCIBILITY_PASS" if passed else "PARTIAL_REPRODUCIBILITY_FAIL") if partial_validation
        else ("CORE_CLASS_READY" if passed else "CORE_CLASS_NOT_READY"),
        "contract": "TTT24_AUG8_R1024_ROOT_AWARE", "historical_pilot_parity": "RETIRED_AS_CONFOUNDED",
        "same_contract_reproducibility": "PASS" if passed else "FAIL",
        "partial_validation": partial_validation,
        "phase2_authorized": False if partial_validation else passed, "phase2_started": False,
        "target_blind": True, "gold_loaded": False, "total_cells": total, "counts": counts,
        "semantic_reproducibility_rate": semantic / total if total else 0.0,
        "prefill_semantic_reproducibility_rate": sum(bool(row["semantic_equal"]) for row in prefill_rows) / total if total else 0.0,
        "scheduler_semantic_reproducibility_rate": sum(bool(row["semantic_equal"]) for row in scheduler_rows) / len(scheduler_rows) if scheduler_rows else 0.0,
        "max_abs_token_logprob_difference": max(float(row["max_abs_token_logprob_difference"]) for row in drift_rows),
        "max_cumulative_score_difference": max(float(row["max_cumulative_score_difference"]) for row in drift_rows),
        "max_cumulative_regret_difference": max(float(row["max_cumulative_regret_difference"]) for row in drift_rows),
        "earliest_logical_divergence": divergences[0] if divergences else "NONE",
    }
    _write_json(output / "REPRODUCIBILITY_SUMMARY.json", decision)
    _write_json(output / "CORE_DECISION.json", decision)
    if divergences:
        _write_json(output / "REPRODUCIBILITY_FAILURE.json", {
            "MEASURED": divergences, "INFERRED": [], "NOT_ESTABLISHED": ["root cause absent concrete first-divergence defect"],
            "categories": ["NONDETERMINISTIC_MODEL_FORWARD", "SCHEDULER_NONDETERMINISM", "CACHE_STATE_NONDETERMINISM",
                           "ROLLBACK_NONDETERMINISM", "PACK_SPLIT_ADOPTION_NONDETERMINISM", "FLOATING_POINT_BRANCH_INSTABILITY", "OTHER"],
        })
    report = "# TTT24 AUG8 R1024 same-contract reproducibility gate\n\n```json\n" + json.dumps(decision, indent=2, sort_keys=True) + "\n```\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    return decision


def _compact_hashes(output: Path) -> dict[str, Any]:
    ignored_parts = {"RUN_A", "RUN_B", "WORKER_LOGS", "RAW_OUTPUTS", "FROZEN_RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS", "OUTPUT_RECEIPTS", "FAILED_ATTEMPTS"}
    files = [path for path in output.rglob("*") if path.is_file() and path.name not in {"HASHES.json", "HASH_VERIFICATION.json"}
             and not any(part in ignored_parts for part in path.relative_to(output).parts)]
    ledger = {str(path.relative_to(output)): _sha_file(path) for path in sorted(files)}
    _write_json(output / "HASHES.json", {"files": ledger, "target_blind": True, "gold_loaded": False})
    mismatches = [{"path": rel, "expected": expected, "actual": _sha_file(output / rel) if (output / rel).exists() else None}
                  for rel, expected in ledger.items() if not (output / rel).exists() or _sha_file(output / rel) != expected]
    verification = {"status": "PASS" if not mismatches else "FAIL", "checked": len(ledger), "mismatches": mismatches}
    _write_json(output / "HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS": raise RuntimeError("compact artifact hash verification failed")
    return verification


def _parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-core", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--minimum-free-bytes", type=int, default=1_500_000_000)
    parser.add_argument("--partial-output-count", type=int, default=0,
                        help="Authorized execution-only subset of the frozen source order; 0 keeps all six.")
    parser.add_argument("--reuse-partial-run-a", action="store_true",
                        help="Freeze selected pre-existing RUN_A records without rerunning them.")
    parser.add_argument("--fresh-partial-rerun", action="store_true",
                        help="Run both A and B afresh on the selected source-order prefix in a new output directory.")
    parser.add_argument("--pause-partial-run-b", action="store_true",
                        help="CPU-only: hash-freeze the completed RUN_B prefix without completing or comparing it.")
    parser.add_argument("--resume-partial-run-b", action="store_true",
                        help="Resume only the missing source-order suffix of a verified paused RUN_B prefix.")
    return parser.parse_args()


def main() -> None:
    args = _parse()
    if args.fresh_partial_rerun and (args.reuse_partial_run_a or args.pause_partial_run_b or args.resume_partial_run_b):
        raise RuntimeError("fresh partial rerun is mutually exclusive with reuse, pause, and resume modes")
    # This controller never accepts a solutions path; fail closed if the
    # supplied challenge is not an ARC challenge-only mapping.
    challenge = _read_json(args.challenge)
    if any("output" in test for task in challenge.values() for test in task.get("test", [])):
        raise RuntimeError("refusing a challenge file containing evaluation outputs")
    outputs = _prepare_static(args)
    partial_validation = args.partial_output_count > 0
    if partial_validation:
        if not (args.reuse_partial_run_a or args.fresh_partial_rerun):
            raise RuntimeError("partial-output execution requires explicit RUN_A reuse or a fresh partial rerun")
        if args.partial_output_count >= len(outputs):
            raise RuntimeError("partial-output count must select a strict subset of the frozen six-output Core")
        outputs = outputs[:args.partial_output_count]
        _write_json(args.output / "PARTIAL_EXECUTION_AMENDMENT.json", {
            "status": "AUTHORIZED_EXECUTION_AMENDMENT", "target_blind": True, "gold_loaded": False,
            "reason": ("RUN_A was user-terminated after independently frozen outputs; compare only frozen source-order prefix"
                       if args.reuse_partial_run_a else
                       "User-authorized fresh A/B rerun on a source-order prefix; no historical raw evidence is reused"),
            "selected_output_count": len(outputs), "selected_output_ids": [row["output_id"] for row in outputs],
            "run_a_reused": bool(args.reuse_partial_run_a), "run_a_not_rerun": bool(args.reuse_partial_run_a),
            "fresh_partial_rerun": bool(args.fresh_partial_rerun),
            "full_six_output_core_gate_completed": False, "phase2_authorized": False, "phase2_started": False,
        })
    if args.pause_partial_run_b:
        if not partial_validation or not args.reuse_partial_run_a or args.resume_partial_run_b:
            raise RuntimeError("pause mode requires the authorized partial RUN_A reuse and forbids resume mode")
        _pause_partial_run(args.output / "RUN_B", outputs)
        _compact_hashes(args.output)
        return
    policy = _read_json(args.coarse_policy)
    verify_a = _run_one(args, args.output / "RUN_A", outputs, policy)
    verify_b = _run_one(args, args.output / "RUN_B", outputs, policy)
    decision = _compare_runs(args.output, outputs, partial_validation=partial_validation)
    decision["run_a_generation_hash_verification"] = verify_a
    decision["run_b_generation_hash_verification"] = verify_b
    _write_json(args.output / "REPRODUCIBILITY_SUMMARY.json", decision)
    _write_json(args.output / "CORE_DECISION.json", decision)
    _compact_hashes(args.output)


if __name__ == "__main__":
    main()
