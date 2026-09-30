#!/usr/bin/env python3
"""Target-blind KV-isolation audit for the four contaminated d59 Regret cells.

This is a diagnostic, not a decoder experiment: it keeps the fixed Regret4
policy and B1 scheduler, never constructs Dynamic Batch2, and never opens a
solution file.  Its only model work is the minimum needed to determine whether
root KV storage or model-level mutable state can explain B1 divergence.
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
    _legacy_cache,
    _reply,
    clone_legacy_cache,
    normalized_result_signature,
    run_ready_scheduler,
    start_ready_cell,
)
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_dynamic_ready_b1_1 import _parity
from scripts.run_regret_dynamic_ready_v1 import (
    ADAPTER_SHA,
    DEPTH,
    OUTPUT_INDEX,
    TASK_ID,
    VIEWS,
    decoder,
    json_sha,
)
from scripts.turbodfs_v4_common import assert_native_token_contract, sha256_file


EXPERIMENT = "REGRET_DYNAMIC_READY_KV_ISOLATION_AUDIT_V1"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True, separators=(",", ":"))
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _cell_key(view: str) -> str:
    return f"{TASK_ID}:o{OUTPUT_INDEX}:d{DEPTH}:{view}"


def _checksum(tensor: Any) -> str:
    import torch

    with torch.no_grad():
        raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def _tensor_bounds(tensor: Any) -> tuple[int, int, int]:
    element_size = int(tensor.element_size())
    storage_offset = int(tensor.storage_offset())
    storage_ptr = int(tensor.untyped_storage().data_ptr())
    if tensor.numel() == 0:
        return storage_ptr, storage_offset * element_size, storage_offset * element_size
    # A conservative storage interval is enough to detect possible overlap for
    # non-contiguous views; exact aliasing is separately identified by ptr.
    extent = storage_offset
    for size, stride in zip(tensor.shape, tensor.stride(), strict=True):
        extent += max(0, int(size) - 1) * abs(int(stride))
    return storage_ptr, storage_offset * element_size, (extent + 1) * element_size


def cache_inventory(cache: Any, *, view: str, stage: str) -> list[dict[str, Any]]:
    legacy = _legacy_cache(cache)
    rows: list[dict[str, Any]] = []
    for layer_index, layer in enumerate(legacy):
        for component_index, tensor in enumerate(layer):
            storage_ptr, start, end = _tensor_bounds(tensor)
            rows.append({
                "record_type": "COMPONENT", "stage": stage, "view": view,
                "root_cache_object_id": id(cache), "root_cache_class": type(cache).__qualname__,
                "layer": layer_index, "component": component_index,
                "tensor_object_id": id(tensor), "data_ptr": int(tensor.data_ptr()),
                "storage_ptr": storage_ptr, "storage_start_byte": start, "storage_end_byte": end,
                "storage_offset": int(tensor.storage_offset()), "shape": tuple(int(v) for v in tensor.shape),
                "stride": tuple(int(v) for v in tensor.stride()), "dtype": str(tensor.dtype),
                "device": str(tensor.device), "requires_grad": bool(tensor.requires_grad),
                "tensor_class": type(tensor).__qualname__,
                "sequence_length": int(tensor.shape[2]) if tensor.ndim >= 3 else None,
                "checksum_sha256": _checksum(tensor),
            })
    return rows


def _component_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return str(row["view"]), int(row["layer"]), int(row["component"])


def _digest(rows: list[dict[str, Any]]) -> dict[tuple[str, int, int], str]:
    return {_component_key(row): str(row["checksum_sha256"]) for row in rows if row["record_type"] == "COMPONENT"}


def _overlap_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    components = [row for row in rows if row["record_type"] == "COMPONENT"]
    output: list[dict[str, Any]] = []
    for index, left in enumerate(components):
        for right in components[index + 1:]:
            if left["view"] == right["view"]:
                continue
            same_storage = left["storage_ptr"] == right["storage_ptr"] and left["device"] == right["device"]
            overlaps = same_storage and max(left["storage_start_byte"], right["storage_start_byte"]) < min(left["storage_end_byte"], right["storage_end_byte"])
            if same_storage or left["data_ptr"] == right["data_ptr"]:
                output.append({
                    "record_type": "CROSS_VIEW_STORAGE_RELATION", "stage": left["stage"],
                    "left_view": left["view"], "left_layer": left["layer"], "left_component": left["component"],
                    "right_view": right["view"], "right_layer": right["layer"], "right_component": right["component"],
                    "same_storage": same_storage, "same_data_ptr": left["data_ptr"] == right["data_ptr"],
                    "possible_byte_overlap": overlaps,
                })
    return output


def _model_state_snapshot(model: Any, label: str) -> dict[str, Any]:
    """Capture only cache/state-bearing public runtime attributes, not weights."""
    def one(owner: Any, name: str) -> dict[str, Any]:
        value = getattr(owner, name)
        row: dict[str, Any] = {"owner": type(owner).__qualname__, "name": name,
                               "object_id": id(value), "class": type(value).__qualname__}
        if hasattr(value, "shape") and hasattr(value, "data_ptr"):
            row.update({"shape": tuple(int(v) for v in value.shape), "data_ptr": int(value.data_ptr()),
                        "dtype": str(value.dtype), "device": str(value.device), "checksum_sha256": _checksum(value)})
        elif isinstance(value, (str, int, float, bool, type(None))):
            row["scalar"] = value
        elif isinstance(value, (tuple, list, dict)):
            row["container_len"] = len(value)
        return row

    snapshots: list[dict[str, Any]] = []
    for owner_name, owner in (("model", model), ("base_model", getattr(model, "model", None))):
        if owner is None:
            continue
        names = sorted(name for name in vars(owner) if any(token in name.lower() for token in ("cache", "past", "state", "generation")))
        snapshots.extend({"scope": owner_name, **one(owner, name)} for name in names)
    return {"label": label, "entries": snapshots}


def _state_diff(before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    lhs = {(row["scope"], row["owner"], row["name"]): row for row in before["entries"]}
    rhs = {(row["scope"], row["owner"], row["name"]): row for row in after["entries"]}
    return [{"key": key, "before": lhs.get(key), "after": rhs.get(key)} for key in sorted(set(lhs) | set(rhs)) if lhs.get(key) != rhs.get(key)]


def _advance_one(model: Any, cell: Any) -> None:
    import torch

    request = cell.request
    if request is None:
        raise RuntimeError("cannot increment a completed ready cell")
    with torch.no_grad():
        started = time.perf_counter()
        outputs = model(input_ids=torch.tensor([[request.token_id]], device=model.device, dtype=torch.long),
                        position_ids=torch.tensor([[request.position]], device=model.device, dtype=torch.long),
                        past_key_values=request.cache, return_dict=True, use_cache=True)
        elapsed = time.perf_counter() - started
    cell.state["model_forward_seconds"] += elapsed
    cell.active_elapsed_seconds += elapsed
    cell.state["active_elapsed_seconds"] += elapsed
    _reply(cell, outputs)
    del outputs


def _new_cells(model: Any, encoded: dict[str, Any], dec: Any, *, order: tuple[str, ...], clone_root: bool = False) -> dict[str, Any]:
    transform = clone_legacy_cache if clone_root else None
    return {view: start_ready_cell(model=model, input_ids=encoded[view].to(model.device), config=dec,
                                   cell_key=_cell_key(view), normalize_root_cache=False,
                                   active_time_accounting=True, root_cache_transform=transform)
            for view in order}


def _short_incremental_mutation(model: Any, encoded: dict[str, Any], dec: Any, target: str) -> list[dict[str, Any]]:
    cells = _new_cells(model, encoded, dec, order=VIEWS)
    initial = {view: _digest(cache_inventory(cell.request.cache, view=view, stage=f"{target}:initial"))
               for view, cell in cells.items()}
    rows: list[dict[str, Any]] = []
    completed = 0
    for mark in (1, 8, 32, 128):
        while completed < mark:
            _advance_one(model, cells[target]); completed += 1
        for idle_view in VIEWS:
            if idle_view == target:
                continue
            current = _digest(cache_inventory(cells[idle_view].request.cache, view=idle_view, stage=f"{target}:{mark}"))
            for key, before_sha in initial[idle_view].items():
                rows.append({"target_view": target, "incremental_steps": mark, "idle_view": idle_view,
                             "layer": key[1], "component": key[2], "checksum_before": before_sha,
                             "checksum_after": current[key], "mutated": before_sha != current[key]})
    del cells
    gc.collect()
    return rows


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _first_difference(reference: dict[str, Any], result: Any) -> dict[str, Any]:
    current = {
        "candidate_tokens": [list(item.token_ids) for item in result.candidates[0]],
        "nodes": list(result.nodes), "branch_probabilities": list(result.branch_probabilities),
        "frontier_floor_events": list(result.frontier_floor_events), "search_trace": list(result.search_trace),
    }
    for name, observed in current.items():
        expected = reference.get(name, [])
        for index, (left, right) in enumerate(zip(expected, observed, strict=False)):
            if _canonical(left) != _canonical(right):
                return {"status": "DIVERGED", "field": name, "index": index, "reference": left, "observed": right}
        if len(expected) != len(observed):
            return {"status": "DIVERGED", "field": name, "index": min(len(expected), len(observed)),
                    "reference_length": len(expected), "observed_length": len(observed)}
    return {"status": "NO_DIVERGENCE"}


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"refusing to overwrite {output}")
    challenge = args.challenge.resolve(); no_gold_challenge(challenge)
    fixed = read_json(args.fixed_budget_contract.resolve()); caps = fixed.get("caps", {})
    if int(caps.get("max_expanded_nodes", -1)) != 4096 or int(caps.get("max_completed_candidates", -1)) != 32:
        raise RuntimeError("KV audit requires frozen fixed Regret4 caps")
    prior = args.prior_output.resolve() / "raw_scalar.json"
    if not prior.exists():
        raise RuntimeError(f"missing exact scalar reference {prior}")
    contract = {
        "experiment": EXPERIMENT, "source_commit": args.source_commit, "target_blind": True,
        "solutions_accessed": False, "dynamic_batch2_executed": False,
        "untouched12_used": False, "untouched24_used": False,
        "task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH, "views": list(VIEWS),
        "policy": "CUMULATIVE_REGRET_r=4.00", "caps": caps,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
        "reference_config": str(args.reference_config.resolve()), "authoritative_root": str(args.authoritative_root.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()), "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "fixed_budget_contract": str(args.fixed_budget_contract.resolve()), "fixed_budget_contract_sha256": sha256_file(args.fixed_budget_contract.resolve()),
        "prior_scalar": str(prior), "prior_scalar_sha256": sha256_file(prior),
        "execution_orders": {"A": list(VIEWS), "B": ["flip_ud", "anti_transpose", "identity", "transpose"]},
    }
    output.mkdir(parents=True)
    atomic_json(output / "KV_ISOLATION_CONTRACT.json", contract)


def run(args: argparse.Namespace) -> None:
    output = args.output.resolve(); contract = read_json(output / "KV_ISOLATION_CONTRACT.json")
    no_gold_challenge(Path(contract["challenge"]))
    prior = {row["cell_key"]: row for row in read_json(Path(contract["prior_scalar"]))}
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from unsloth import FastLanguageModel

    runtime_args = SimpleNamespace(output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
                                   reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
                                   native_config_dir=Path(contract["native_config_dir"]), gpu_id=args.gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    try:
        adapter_sha = load_adapter(model, adapter_records(Path(contract["adapter_manifest"])).get((TASK_ID, DEPTH)))
        if adapter_sha != ADAPTER_SHA:
            raise RuntimeError(f"unexpected adapter SHA {adapter_sha}")
        task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
        encoded = {view: value["input_ids"] for view, (value, _aug) in zip(
            VIEWS, (common.encoded_view(tokenizer=tokenizer, task=task, view=view, config=generation_config) for view in VIEWS), strict=True)}
        if any(int(value.shape[0]) != 1 for value in encoded.values()):
            raise RuntimeError("KV audit requires scalar prompt prefills")
        native = assert_native_token_contract(tokenizer); dec = decoder(contract["caps"]); FastLanguageModel.for_inference(model)

        model_audit: dict[str, Any] = {"native_token_contract": native, "snapshots": [], "diffs": []}
        before = _model_state_snapshot(model, "BEFORE_ALL_PREFILLS"); model_audit["snapshots"].append(before)
        cells: dict[str, Any] = {}
        storage_rows: list[dict[str, Any]] = []
        mutation_rows: list[dict[str, Any]] = []
        known_digests: dict[str, dict[tuple[str, int, int], str]] = {}
        for view in VIEWS:
            prior_state = _model_state_snapshot(model, f"BEFORE_PREFILL_{view}")
            cell = start_ready_cell(model=model, input_ids=encoded[view].to(model.device), config=dec,
                                    cell_key=_cell_key(view), normalize_root_cache=False, active_time_accounting=True)
            cells[view] = cell
            own = cache_inventory(cell.request.cache, view=view, stage=f"ROOT_AFTER_PREFILL_{view}")
            storage_rows.extend(own)
            known_digests[view] = _digest(own)
            after_state = _model_state_snapshot(model, f"AFTER_PREFILL_{view}")
            model_audit["snapshots"].extend([prior_state, after_state])
            model_audit["diffs"].append({"transition": f"PREFILL_{view}", "changes": _state_diff(prior_state, after_state)})
            for old_view, old_cell in cells.items():
                if old_view == view:
                    continue
                current = _digest(cache_inventory(old_cell.request.cache, view=old_view, stage=f"AFTER_PREFILL_{view}"))
                for key, before_sha in known_digests[old_view].items():
                    mutation_rows.append({"prior_view": old_view, "after_prefill_view": view, "layer": key[1], "component": key[2],
                                          "checksum_before": before_sha, "checksum_after": current[key], "mutated": before_sha != current[key]})
        storage_rows.extend(_overlap_rows(storage_rows))
        write_csv(output / "root_cache_storage_map.csv", storage_rows)
        write_csv(output / "prefill_cross_mutation.csv", mutation_rows)
        del cells
        gc.collect()

        incremental_rows = _short_incremental_mutation(model, encoded, dec, "anti_transpose")
        incremental_rows.extend(_short_incremental_mutation(model, encoded, dec, "flip_ud"))
        write_csv(output / "incremental_cross_mutation.csv", incremental_rows)

        state_before_short = _model_state_snapshot(model, "BEFORE_SHORT_INCREMENTAL")
        short = _new_cells(model, encoded, dec, order=("identity",))
        _advance_one(model, short["identity"])
        state_after_short = _model_state_snapshot(model, "AFTER_SHORT_INCREMENTAL")
        model_audit["snapshots"].extend([state_before_short, state_after_short])
        model_audit["diffs"].append({"transition": "SHORT_INCREMENTAL_IDENTITY", "changes": _state_diff(state_before_short, state_after_short)})
        del short
        gc.collect()

        # Clone each root before its generator binds it, then execute the exact
        # shared B1 scheduler.  This is the only full shared run in the audit.
        cloned_cells = _new_cells(model, encoded, dec, order=VIEWS, clone_root=True)
        clone_scheduler = run_ready_scheduler(model=model, cells=list(cloned_cells.values()), dynamic_batch2=False)
        clone_rows: list[dict[str, Any]] = []
        clone_results: dict[str, Any] = {}
        for view, cell in cloned_cells.items():
            from inference.nvarc_turbodfs_dynamic_ready import ready_result
            result = ready_result(cell); clone_results[view] = result
            parity = _parity(prior[_cell_key(view)], result)
            clone_rows.append({"cell_key": _cell_key(view), "root_cache_mode": "DEEP_CLONED_LEGACY", **parity,
                               "authoritative_nodes": prior[_cell_key(view)]["nodes_expanded"],
                               "clone_shared_nodes": sum(row.get("state") == "expanded" for row in result.nodes),
                               "authoritative_termination": prior[_cell_key(view)]["termination_reason"],
                               "clone_shared_termination": result.termination_reason,
                               "physical_forwards": clone_scheduler["physical_forwards"],
                               "mean_effective_batch": clone_scheduler["mean_effective_batch"]})
        write_csv(output / "ROOT_CLONE_SHARED_B1_PARITY.csv", clone_rows)
        clone_pass = all(bool(row["exact_parity"]) for row in clone_rows)
        divergence = [{"status": "NOT_REQUIRED", "reason": "ROOT_CLONE_SHARED_B1_PARITY_4_OF_4"}]
        if not clone_pass:
            divergence = []
            for view, result in clone_results.items():
                if normalized_result_signature(result) != prior[_cell_key(view)]["signature"]:
                    divergence.append({"cell_key": _cell_key(view), **_first_difference(prior[_cell_key(view)], result)})
        write_csv(output / "FLIP_UD_FIRST_DIVERGENCE.csv", divergence)
        del cloned_cells, clone_results
        gc.collect()

        # Construction order can affect prefill state, while the documented
        # scheduler itself remains lexical.  Run only that first lexical cell
        # to test whether its full B1 trace stays exact in each construction order.
        order_rows: list[dict[str, Any]] = []
        for label, raw_order in contract["execution_orders"].items():
            order = tuple(raw_order); ordered_cells = _new_cells(model, encoded, dec, order=order)
            first = sorted(ordered_cells.values(), key=lambda cell: (cell.cell_key, cell.request.ordinal))[0]
            scheduler = run_ready_scheduler(model=model, cells=[first], dynamic_batch2=False)
            from inference.nvarc_turbodfs_dynamic_ready import ready_result
            result = ready_result(first); parity = _parity(prior[first.cell_key], result)
            order_rows.append({"order_label": label, "requested_prefill_order": list(order),
                               "scheduler_order": "cell_key_then_request_ordinal", "first_scheduled_cell": first.cell_key,
                               "first_scheduled_exact_parity": parity["exact_parity"], "first_scheduled_nodes": sum(row.get("state") == "expanded" for row in result.nodes),
                               "physical_forwards": scheduler["physical_forwards"], "dynamic_batch2": False})
            del ordered_cells
            gc.collect()
        write_csv(output / "execution_order_audit.csv", order_rows)

        shared_state_changes = sum(len(row["changes"]) for row in model_audit["diffs"])
        model_audit["shared_mutable_model_state_detected"] = shared_state_changes > 0
        model_audit["note"] = "Runtime state metadata only; model weights and Gold were not inspected."
        atomic_json(output / "model_mutable_state_audit.json", model_audit)

        aliases = any(row.get("record_type") == "CROSS_VIEW_STORAGE_RELATION" and row.get("same_storage") for row in storage_rows)
        prefill_mutation = any(str(row["mutated"]).lower() == "true" for row in mutation_rows)
        idle_mutation = any(str(row["mutated"]).lower() == "true" for row in incremental_rows)
        first_exact = all(bool(row["first_scheduled_exact_parity"]) for row in order_rows)
        cause = "NOT_ESTABLISHED"
        if clone_pass and not aliases and not prefill_mutation and not idle_mutation:
            cause = "ROOT_CACHE_ISOLATION_NOT_SUPPORTED_BY_AUDIT"
        elif not clone_pass and (aliases or prefill_mutation or idle_mutation):
            cause = "ROOT_CACHE_OR_CACHE_MUTATION_SUPPORTED"
        elif not clone_pass:
            # A deep clone itself changes the cache container to legacy tuples.
            # With no storage alias, checksum mutation, or observed model-level
            # mutable state, this audit cannot attribute divergence beyond a
            # non-root-cache shared-execution effect.
            cause = "NON_ROOT_CACHE_SHARED_EXECUTION_EFFECT_NOT_ESTABLISHED"
        decision = {
            "experiment": EXPERIMENT, "source_commit": contract["source_commit"], "adapter_sha256": adapter_sha,
            "cross_cell_root_storage_aliases": aliases, "cross_prefill_mutation": prefill_mutation,
            "idle_root_mutation_during_incremental": idle_mutation,
            "model_level_shared_mutable_state": model_audit["shared_mutable_model_state_detected"],
            "root_clone_shared_B1_parity": f"{sum(bool(row['exact_parity']) for row in clone_rows)}/4",
            "first_scheduled_cell_exact_in_both_orders": first_exact,
            "first_divergence": divergence[0].get("status"), "root_cause": cause,
            "isolated_B1_parity": "4/4 (previous frozen B1.1 evidence)", "dynamic_B2_executed": False,
            "gold_accessed": False, "untouched12_used": False, "untouched24_used": False,
            "next": "CPU_REVIEW_REQUIRED",
        }
        atomic_json(output / "DECISION.json", decision)
        lines = [f"# {EXPERIMENT}", "", "Target-blind diagnostic. Gold was not opened and Dynamic Batch2 was not instantiated.", "",
                 f"- Root storage aliases across views: {aliases}", f"- Cross-prefill mutation: {prefill_mutation}",
                 f"- Idle-root mutation during 1/8/32/128 incremental forwards: {idle_mutation}",
                 f"- Model mutable-state entries changed: {shared_state_changes}",
                 f"- Root-clone shared B1 parity: {decision['root_clone_shared_B1_parity']}",
                 f"- First scheduled cell exact in both construction orders: {first_exact}", f"- Root-cause classification: {cause}", "",
                 "## Scope", "", "- Cell family: d59b0160:o0:d24 {identity, flip_ud, transpose, anti_transpose}",
                 "- Decoder: fixed CUMULATIVE_REGRET_r=4.00, 4096 nodes, 32 candidates", "- Dynamic Batch2: not run", "- Gold: not accessed", "- Untouched12/Untouched24: not used"]
        (output / "KV_ISOLATION_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        # Runtime logs/PIDs are deliberately local operational state, not
        # immutable research evidence.  Hash only the portable audit payload.
        artifact_names = {
            "KV_ISOLATION_CONTRACT.json", "root_cache_storage_map.csv",
            "prefill_cross_mutation.csv", "incremental_cross_mutation.csv",
            "model_mutable_state_audit.json", "ROOT_CLONE_SHARED_B1_PARITY.csv",
            "execution_order_audit.csv", "FLIP_UD_FIRST_DIVERGENCE.csv",
            "KV_ISOLATION_AUDIT.md", "DECISION.json",
        }
        files = [output / name for name in sorted(artifact_names)]
        if any(not path.is_file() for path in files):
            raise RuntimeError("KV audit core artifact missing before hash freeze")
        atomic_json(output / "HASHES.json", {"sha256": {path.name: sha256_file(path) for path in sorted(files)}, "solutions_accessed": False})
    finally:
        del model


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("prepare", "run"):
        item = sub.add_parser(command)
        item.add_argument("--output", type=Path, required=True)
        item.add_argument("--challenge", type=Path, required=True)
        item.add_argument("--authoritative-root", type=Path, required=True)
        item.add_argument("--reference-config", type=Path, required=True)
        item.add_argument("--model-path", type=Path, required=True)
        item.add_argument("--native-config-dir", type=Path, required=True)
        item.add_argument("--adapter-manifest", type=Path, required=True)
        item.add_argument("--fixed-budget-contract", type=Path, required=True)
        item.add_argument("--prior-output", type=Path, required=True)
        item.add_argument("--source-commit", required=True)
        item.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
    else:
        run(args)


if __name__ == "__main__":
    main()
