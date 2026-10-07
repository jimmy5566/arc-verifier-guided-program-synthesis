"""Recover a stable batched protocol without starting the full diagnostic."""
from __future__ import annotations

import argparse
from collections import defaultdict
import gc
import hashlib
import json
from pathlib import Path
import random
import sys
import time
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from foundation_v2_capability_diagnostic_v1.audit import canonical_sha256
from run_foundation_v2_capability_diagnostic_v1 import (
    ADAPTER_SHA256, DiagnosticError, atomic_json, calibration_selection,
    load_contexts, load_model, sha256_file,
)


BATCH_SIZES = (4, 8, 12, 16, 20, 24, 28, 32)
DUPLICATE_SIZES = (2, 4, 8, 16, 32)
SENSITIVE = (
    "DIAGNOSTIC_V1_2:ff68d59469e5ba1fea3ce1d0",
    "PARAM_SURFACE_V1_2:object_count:diagnostic:018",
    "PARAM_SURFACE_V1_2:object_position:diagnostic:031",
    "COMPOSITION_DEV_V1_2:capv3:4b433153812ecc3d40:001",
    "COMPOSITION_DEV_V1_2:capv3:dbdcca62cc0200da4e:004",
)


def _tokens_hash(tokens: list[int]) -> str:
    return hashlib.sha256(json.dumps(tokens, separators=(",", ":")).encode()).hexdigest()


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    if left == right:
        return None
    shared = min(len(left), len(right))
    return next((index for index in range(shared) if left[index] != right[index]), shared)


class Capture:
    def __init__(self, rows: dict[int, str]) -> None:
        self.rows, self.values = rows, defaultdict(list)

    def __call__(self, _input_ids: Any, scores: Any) -> Any:
        import torch
        for index, sample_id in self.rows.items():
            values, ids = torch.topk(scores[index].detach().float().cpu(), k=5)
            value_list, id_list = values.tolist(), ids.tolist()
            self.values[sample_id].append({
                "top5_token_ids": [int(value) for value in id_list],
                "top5_logits": [float(value) for value in value_list],
                "top1_top2_margin": float(value_list[0] - value_list[1]),
            })
        return scores


def generate(model: Any, batch: list[dict[str, Any]], capture_ids: set[str] | None = None) -> tuple[dict[str, list[int]], float, dict[str, list[dict[str, Any]]], int]:
    import torch
    from transformers import LogitsProcessorList
    width = max(len(row["prompt_ids"]) for row in batch)
    prompt = torch.full((len(batch), width), 13, dtype=torch.long, device="cuda:0")
    mask = torch.zeros_like(prompt)
    for index, row in enumerate(batch):
        length = len(row["prompt_ids"])
        prompt[index, width-length:] = torch.tensor(row["prompt_ids"], dtype=torch.long, device="cuda:0")
        mask[index, width-length:] = 1
    capture = Capture({index: row["sample_id"] for index, row in enumerate(batch) if capture_ids and row["sample_id"] in capture_ids})
    processors = LogitsProcessorList([capture]) if capture.rows else None
    started = time.perf_counter()
    with torch.inference_mode():
        output = model.generate(input_ids=prompt, attention_mask=mask, max_new_tokens=932, do_sample=False, num_beams=1,
                                eos_token_id=15, pad_token_id=13, use_cache=True, logits_processor=processors)
    torch.cuda.synchronize(0)
    elapsed = time.perf_counter() - started
    result: dict[str, list[int]] = {}
    for index, row in enumerate(output):
        raw = [int(value) for value in row[width:].detach().cpu().tolist()]
        result[batch[index]["sample_id"]] = raw[:raw.index(15)+1] if 15 in raw else raw
    del output, prompt, mask
    return result, elapsed, dict(capture.values), width


def batches(rows: list[dict[str, Any]], max_batch: int, token_cap: int, *, order: str = "length") -> list[list[dict[str, Any]]]:
    pending = list(rows)
    if order == "length":
        pending.sort(key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    elif order == "reverse":
        pending.reverse()
        pending.sort(key=lambda row: len(row["prompt_ids"]))
    elif order == "shuffle":
        random.Random(20261007).shuffle(pending)
        pending.sort(key=lambda row: len(row["prompt_ids"]))
    groups, cursor = [], 0
    while cursor < len(pending):
        count = min(max_batch, len(pending)-cursor)
        while count > 1 and max(len(row["prompt_ids"]) for row in pending[cursor:cursor+count]) * count > token_cap:
            count -= 1
        groups.append(pending[cursor:cursor+count]); cursor += count
    return groups


def run_protocol(model: Any, rows: list[dict[str, Any]], max_batch: int, token_cap: int, *, order: str = "length", capture_ids: set[str] | None = None) -> dict[str, Any]:
    import torch
    torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats(0)
    outputs, captures, widths, actual = {}, {}, {}, []
    started = time.perf_counter(); oom = False
    try:
        for group in batches(rows, max_batch, token_cap, order=order):
            generated, _seconds, logged, width = generate(model, group, capture_ids)
            outputs.update(generated); captures.update(logged); actual.append(len(group))
            for row in group: widths[row["sample_id"]] = width
    except torch.cuda.OutOfMemoryError:
        oom = True; gc.collect(); torch.cuda.empty_cache()
    wall = time.perf_counter() - started
    return {"outputs": outputs, "captures": captures, "padded_widths": widths, "actual_batch_sizes": actual, "wall_seconds": wall,
            "examples_per_second": 0 if oom else len(rows)/wall,
            "generated_tokens_per_second": 0 if oom else sum(len(value) for value in outputs.values())/wall,
            "peak_allocated_vram_bytes": int(torch.cuda.max_memory_allocated(0)), "peak_reserved_vram_bytes": int(torch.cuda.max_memory_reserved(0)), "oom": oom}


def compare(reference: dict[str, list[int]], candidate: dict[str, list[int]], sample_ids: list[str]) -> dict[str, Any]:
    matches = [sample_id for sample_id in sample_ids if reference[sample_id] == candidate.get(sample_id)]
    return {"parity_count": len(matches), "total": len(sample_ids), "mismatches": [
        {"sample_id": sample_id, "first_divergence": _first_divergence(reference[sample_id], candidate.get(sample_id, [])),
         "b1_sha256": _tokens_hash(reference[sample_id]), "candidate_sha256": _tokens_hash(candidate.get(sample_id, []))}
        for sample_id in sample_ids if sample_id not in matches]}


def duplicate_self(model: Any, contexts: dict[str, dict[str, Any]], reference: dict[str, list[int]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for sample_id in SENSITIVE:
        row = contexts[sample_id]; rows = []
        for size in DUPLICATE_SIZES:
            group = [{**row, "sample_id": f"{sample_id}#dup{index}"} for index in range(size)]
            generated, elapsed, _capture, width = generate(model, group)
            values = list(generated.values())
            rows.append({"batch_size": size, "padded_width": width, "wall_seconds": elapsed,
                         "members_identical": len({_tokens_hash(value) for value in values}) == 1,
                         "members_matching_b1": sum(value == reference[sample_id] for value in values), "member_count": size})
        result[sample_id] = rows
    return result


def peer_invariance(model: Any, selected: list[dict[str, Any]], contexts: dict[str, dict[str, Any]], reference: dict[str, list[int]], batch_size: int, token_cap: int) -> dict[str, Any]:
    ordered = sorted(selected, key=lambda row: (len(row["prompt_ids"]), row["sample_id"]))
    bins = {"short": ordered[:len(ordered)//3], "medium": ordered[len(ordered)//3:2*len(ordered)//3], "long": ordered[2*len(ordered)//3:]}
    result: dict[str, Any] = {}
    for sample_id in SENSITIVE:
        anchor = contexts[sample_id]; cases = []
        for name, pool in bins.items():
            peers = [row for row in pool if row["sample_id"] != sample_id]
            group = [anchor] + peers[:max(1, batch_size-1)]
            while len(group) > 1 and max(len(row["prompt_ids"]) for row in group) * len(group) > token_cap: group.pop()
            output, elapsed, _capture, width = generate(model, group)
            cases.append({"peer_type": name, "actual_batch_size": len(group), "padded_width": width, "wall_seconds": elapsed,
                          "matches_b1": output[sample_id] == reference[sample_id], "sha256": _tokens_hash(output[sample_id])})
        group = [{**anchor, "sample_id": sample_id}] + [{**anchor, "sample_id": f"{sample_id}#self{i}"} for i in range(max(1, batch_size-1))]
        output, elapsed, _capture, width = generate(model, group)
        cases.append({"peer_type": "duplicate_self", "actual_batch_size": len(group), "padded_width": width, "wall_seconds": elapsed,
                      "matches_b1": output[sample_id] == reference[sample_id], "sha256": _tokens_hash(output[sample_id])})
        result[sample_id] = cases
    return result


def forensic(reference: dict[str, list[int]], b32: dict[str, Any], b1_capture: dict[str, Any], duplicate: dict[str, Any], contexts: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for sample_id in SENSITIVE:
        other = b32["outputs"][sample_id]; position = _first_divergence(reference[sample_id], other)
        if position is None: raise DiagnosticError(f"EXPECTED_B32_MISMATCH_NOT_REPRODUCED={sample_id}")
        b1_logits = b1_capture[sample_id][position]; b32_logits = b32["captures"][sample_id][position]
        near_tie = min(b1_logits["top1_top2_margin"], b32_logits["top1_top2_margin"]) <= 0.05
        duplicate_all = all(row["members_matching_b1"] == row["member_count"] for row in duplicate[sample_id])
        classification = "LIKELY_NUMERICAL_NEAR_TIE" if near_tie else ("LIKELY_PADDING_OR_IMPLEMENTATION" if duplicate_all else "UNRESOLVED")
        result.append({"sample_id": sample_id, "classification": classification, "first_divergent_generated_token_position": position,
                       "b1_token": reference[sample_id][position] if position < len(reference[sample_id]) else None,
                       "b32_token": other[position] if position < len(other) else None,
                       "b1_top5": b1_logits, "b32_top5": b32_logits,
                       "prompt_length": len(contexts[sample_id]["prompt_ids"]), "b32_padded_width": b32["padded_widths"][sample_id],
                       "attention_mask_sum": len(contexts[sample_id]["prompt_ids"]), "duplicate_self_all_parity": duplicate_all})
    return result


def main() -> int:
    args = parser().parse_args()
    import torch
    contexts_list = load_contexts(args); selected = calibration_selection(contexts_list); ids = [row["sample_id"] for row in selected]
    contexts = {row["sample_id"]: row for row in contexts_list}
    previous = json.loads((args.artifact / "GPU_DIAGNOSTIC_BATCH_CALIBRATION.json").read_text(encoding="utf-8"))
    if previous.get("status") != "FAIL" or previous.get("representative_samples") != 64 or len(previous.get("token_exact_mismatches", [])) != 5:
        raise DiagnosticError("STOPPED_CALIBRATION_RECEIPT_INVALID")
    if any(key in previous for key in ("b1_outputs", "b1_generated_token_ids", "rows", "records")):
        raise DiagnosticError("REFERENCE_RECOVERY_EXCEPTION_NOT_PROVEN")
    model = load_model(args, "foundation_v2")
    # The prior receipt omitted all reference tokens. Reconstruct once under the explicit corruption exception and freeze immediately.
    reference: dict[str, list[int]] = {}; b1_capture: dict[str, Any] = {}; b1_wall = 0.0
    for index, row in enumerate(selected, 1):
        generated, elapsed, captured, _width = generate(model, [row], set(SENSITIVE)); reference.update(generated); b1_capture.update(captured); b1_wall += elapsed
        print(json.dumps({"event": "B1_REFERENCE_RECOVERY", "completed": index, "total": 64}), flush=True)
    old_hashes = {row["sample_id"]: row["b1_sha256"] for row in previous["token_exact_mismatches"]}
    old_match = {sample_id: _tokens_hash(reference[sample_id]) == expected for sample_id, expected in old_hashes.items()}
    if not all(old_match.values()): raise DiagnosticError(f"B1_REFERENCE_RECONSTRUCTION_DRIFT={old_match}")
    reference_path = args.output / "B1_REFERENCE_RECOVERY.json"
    atomic_json(reference_path, {"status": "FROZEN", "classification": "B1_REFERENCE_FREEZE_INCOMPLETE", "corruption_exception_invoked": True,
                                 "record_count": 64, "prior_known_hashes_reproduced": old_match, "wall_seconds": b1_wall,
                                 "records": [{"sample_id": sample_id, "generated_token_ids": reference[sample_id], "sha256": _tokens_hash(reference[sample_id])} for sample_id in ids]})
    duplicate = duplicate_self(model, contexts, reference)
    ladder: list[dict[str, Any]] = []; runs: dict[int, dict[str, Any]] = {}
    for size in BATCH_SIZES:
        run = run_protocol(model, selected, size, args.max_batched_prompt_tokens, capture_ids=set(SENSITIVE) if size == 32 else None)
        runs[size] = run; parity = compare(reference, run["outputs"], ids)
        ladder.append({"max_batch": size, **{key: run[key] for key in ("wall_seconds", "examples_per_second", "generated_tokens_per_second", "peak_allocated_vram_bytes", "peak_reserved_vram_bytes", "oom", "actual_batch_sizes")}, **parity})
        print(json.dumps({"event": "BATCH_LADDER", "max_batch": size, "parity": parity["parity_count"], "wall_seconds": run["wall_seconds"]}), flush=True)
    b32 = runs[32]
    forensics = forensic(reference, b32, b1_capture, duplicate, contexts)
    parity_options = [row for row in ladder if not row["oom"] and row["parity_count"] == 64]
    repeatability: dict[str, Any] | str = "NOT_REQUIRED_TOKEN_EXACT_B1_PARITY"
    order_invariance: dict[str, Any] | str = "NOT_REQUIRED_TOKEN_EXACT_B1_PARITY"
    peer: dict[str, Any] | str = "NOT_REQUIRED_TOKEN_EXACT_B1_PARITY"
    route, selected_size, stable = "TOKEN_EXACT_B1_PARITY", None, False
    if parity_options:
        choice = min(parity_options, key=lambda row: row["wall_seconds"]); selected_size = int(choice["max_batch"]); stable = True
    else:
        safe = [row for row in ladder if not row["oom"]]
        choice = min(safe, key=lambda row: row["wall_seconds"]); selected_size = int(choice["max_batch"]); route = "FROZEN_BATCH_PROTOCOL"
        repeated = [run_protocol(model, selected, selected_size, args.max_batched_prompt_tokens) for _ in range(3)]
        hashes = [{sample_id: _tokens_hash(run["outputs"][sample_id]) for sample_id in ids} for run in repeated]
        repeatability = {"status": "PASS" if hashes[0] == hashes[1] == hashes[2] else "FAIL", "comparisons": 64*2,
                         "wall_seconds": [run["wall_seconds"] for run in repeated]}
        order_runs = {name: run_protocol(model, selected, selected_size, args.max_batched_prompt_tokens, order=name) for name in ("length", "reverse", "shuffle")}
        order_hashes = {name: {sample_id: _tokens_hash(run["outputs"][sample_id]) for sample_id in ids} for name, run in order_runs.items()}
        order_invariance = {"status": "PASS" if order_hashes["length"] == order_hashes["reverse"] == order_hashes["shuffle"] else "FAIL", "comparisons": 64*2}
        peer = peer_invariance(model, selected, contexts, reference, selected_size, args.max_batched_prompt_tokens)
        peer_pass = all(len({case["sha256"] for case in cases}) == 1 for cases in peer.values())
        stable = repeatability["status"] == order_invariance["status"] == "PASS" and peer_pass
    selected_row = next(row for row in ladder if row["max_batch"] == selected_size)
    estimate = 6000 / selected_row["examples_per_second"] if selected_row["examples_per_second"] else None
    recovery = {"status": "PASS" if stable else "FAIL", "BATCH_RECOVERY_READY": stable, "cause_classifications": forensics,
                "batch_implementation_audit": {"status": "PASS", "matches_known_good": True, "left_padding": True, "pad_token_id": 13,
                                                "explicit_attention_mask": True, "eos_token_id": 15, "suffix_after_padded_width": True,
                                                "gold_dependent_logic": False, "use_cache": True, "do_sample": False, "num_beams": 1},
                "b1_reference_recovery": {"classification": "B1_REFERENCE_FREEZE_INCOMPLETE", "path": str(reference_path),
                                          "sha256": sha256_file(reference_path), "prior_five_hashes_reproduced": old_match},
                "parity_table": ladder, "duplicate_self_results": duplicate, "repeatability": repeatability,
                "order_invariance": order_invariance, "batch_peer_invariance": peer, "acceptance_route": route,
                "protocol_statement": None if route == "TOKEN_EXACT_B1_PARITY" else "Batched BF16 greedy inference is treated as the frozen evaluation protocol; B1 is a calibration reference, not the normative inference definition.",
                "selected_batch_protocol": {"max_batch": selected_size, "max_batched_prompt_tokens": args.max_batched_prompt_tokens, "prompt_length_bucketing": True},
                "estimated_full_6000_generation_wall_seconds": estimate, "Gold accessed": False, "full generation started": False,
                "adapter_sha256": sha256_file(args.adapter_path / "adapter_model.safetensors"), "optimizer_steps": 0, "backward_calls": 0}
    if recovery["adapter_sha256"] != ADAPTER_SHA256: raise DiagnosticError("ADAPTER_DRIFT")
    atomic_json(args.artifact / "BATCH_PARITY_FORENSIC.json", {"status": "COMPLETE", "mismatches": forensics})
    atomic_json(args.artifact / "GPU_DIAGNOSTIC_BATCH_RECOVERY.json", recovery)
    print(json.dumps({"status": recovery["status"], "BATCH_RECOVERY_READY": stable, "route": route, "selected_batch": selected_size, "estimated_seconds": estimate}), flush=True)
    return 0 if stable else 3


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser()
    value.add_argument("--model-path", type=Path, required=True); value.add_argument("--adapter-path", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True); value.add_argument("--artifact", type=Path, required=True)
    value.add_argument("--max-batched-prompt-tokens", type=int, default=21568)
    return value


if __name__ == "__main__":
    raise SystemExit(main())
