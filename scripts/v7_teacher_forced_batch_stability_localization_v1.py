#!/usr/bin/env python3
"""CPU-testable worker and scorer for the prospective V7 batch-localization study.

This module deliberately has no model, CUDA, subprocess, or network dependency.
The separately authorised future launcher must provide a ``teacher_forced``
runtime adapter.  The adapter receives the physical packed batches produced
here; the scorer records their effective shapes and classifies repeatability
before it can call a cross-batch result stable.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict


CONDITIONS = (1, 2, 4, 8, 12)
REPEATS = (0, 1)
TOKEN_METRICS = ("rank", "top1", "top2", "margin", "nll")


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def metric(logits, target):
    values = [float(item) for item in logits]
    target = int(target)
    require(values and 0 <= target < len(values), "TARGET_LOGIT_INVALID")
    correct = values[target]
    maximum = max(values)
    logz = maximum + math.log(sum(math.exp(item - maximum) for item in values))
    rank = 1 + sum(item > correct for item in values)
    other = max(item for index, item in enumerate(values) if index != target) if len(values) > 1 else float("-inf")
    return {"rank": rank, "top1": rank == 1, "top2": rank <= 2,
            "margin": correct - other, "nll": logz - correct}


def _record_key(row):
    return str(row["cohort_record_id"])


def deterministic_order(records):
    """Freeze length-bucketed order without looking at targets or logits."""
    checked = []
    for index, row in enumerate(records):
        require("cohort_record_id" in row and "prompt_token_ids" in row and "target_token_ids" in row,
                "RECORD_LAYOUT")
        prompt = row["prompt_token_ids"]
        target = row["target_token_ids"]
        require(isinstance(prompt, list) and prompt and isinstance(target, list) and target, "TOKEN_LAYOUT")
        checked.append(dict(row, manifest_index=int(row.get("manifest_index", index)),
                            sequence_length=len(prompt) + len(target)))
    require(len(checked) == 12 and len({_record_key(row) for row in checked}) == 12, "BATCH1_SUBSET_EXACTLY_12")
    # Four-token buckets are deterministic and allow reproducible padding layouts.
    return sorted(checked, key=lambda row: (row["sequence_length"] // 4, row["sequence_length"], row["manifest_index"], _record_key(row)))


def physical_batches(records, requested_batch_size):
    require(requested_batch_size in CONDITIONS, "UNAUTHORISED_PHYSICAL_BATCH")
    ordered = deterministic_order(records)
    return [ordered[offset:offset + requested_batch_size]
            for offset in range(0, len(ordered), requested_batch_size)]


def _packed_batch(batch):
    """Create left-padded production-shaped metadata without model tensors."""
    maximum = max(row["sequence_length"] for row in batch)
    packed = []
    for row in batch:
        padding = maximum - row["sequence_length"]
        attention_mask = [0] * padding + [1] * row["sequence_length"]
        # Correct targets are teacher-forced at prompt positions shifted by one.
        indexes = [padding + len(row["prompt_token_ids"]) - 1 + offset for offset in range(len(row["target_token_ids"]))]
        packed.append({"cohort_record_id": _record_key(row), "prompt_token_ids": list(row["prompt_token_ids"]),
                       "target_token_ids": list(row["target_token_ids"]), "sequence_length": row["sequence_length"],
                       "attention_mask": attention_mask, "correct_token_indexes": indexes})
    return {"records": packed, "effective_batch_size": len(batch), "effective_shape": [len(batch), maximum],
            "padding_side": "left", "attention_mask_sha256": sha256([row["attention_mask"] for row in packed]),
            "sequence_lengths": [row["sequence_length"] for row in packed],
            "correct_token_indexing": "LEFT_PADDED_PROMPT_LAST_PLUS_TARGET_OFFSET"}


def _validate_runtime_payload(packed, payload):
    require(isinstance(payload, dict) and set(payload) == {row["cohort_record_id"] for row in packed["records"]},
            "RUNTIME_RECORD_SET_MISMATCH")
    result = []
    for row in packed["records"]:
        expected_mask = [0] * (len(row["attention_mask"]) - row["sequence_length"]) + [1] * row["sequence_length"]
        expected_indexes = [len(row["attention_mask"]) - row["sequence_length"] + len(row["prompt_token_ids"]) - 1 + offset
                            for offset in range(len(row["target_token_ids"]))]
        logits = payload[row["cohort_record_id"]]
        require(isinstance(logits, list) and len(logits) == len(row["target_token_ids"]), "RUNTIME_TARGET_TOKEN_LAYOUT")
        token_metrics = [metric(vector, target) for vector, target in zip(logits, row["target_token_ids"], strict=True)]
        result.append(dict(row, token_metrics=token_metrics,
                           packing_valid=(row["attention_mask"] == expected_mask and row["correct_token_indexes"] == expected_indexes)))
    return result


def worker(records, runtime, batch_size, repeat_identity):
    """Run the physical batches for one predeclared condition and repeat."""
    require(repeat_identity in REPEATS, "REPEAT_IDENTITY")
    output = []
    for ordinal, batch in enumerate(physical_batches(records, batch_size)):
        packed = _packed_batch(batch)
        payload = runtime.teacher_forced(packed, requested_batch_size=batch_size, repeat_identity=repeat_identity,
                                         physical_batch_ordinal=ordinal)
        for row in _validate_runtime_payload(packed, payload):
            output.append({"cohort_record_id": row["cohort_record_id"], "requested_batch_size": batch_size,
                           "repeat_identity": repeat_identity, "physical_batch_ordinal": ordinal,
                           "effective_batch_size": packed["effective_batch_size"], "effective_shape": packed["effective_shape"],
                           "padding_side": packed["padding_side"], "attention_mask": row["attention_mask"],
                           "attention_mask_sha256": sha256(row["attention_mask"]), "batch_attention_mask_sha256": packed["attention_mask_sha256"],
                           "sequence_length": row["sequence_length"], "sequence_lengths_in_physical_batch": packed["sequence_lengths"],
                           "prompt_token_ids_sha256": sha256(row["prompt_token_ids"]), "target_token_ids_sha256": sha256(row["target_token_ids"]),
                           "correct_token_indexes": row["correct_token_indexes"],
                           "correct_token_indexing": packed["correct_token_indexing"], "token_metrics": row["token_metrics"]})
            output[-1]["packing_valid"] = row["packing_valid"]
    require(len(output) == 12 and len({row["cohort_record_id"] for row in output}) == 12, "CONDITION_COMPLETENESS")
    return sorted(output, key=lambda row: row["cohort_record_id"])


def _identity(row, *, physical_layout):
    fields = ["cohort_record_id", "sequence_length", "prompt_token_ids_sha256", "target_token_ids_sha256",
              "correct_token_indexing", "padding_side"]
    if physical_layout:
        fields.extend(["attention_mask", "correct_token_indexes"])
    return {key: row[key] for key in fields}


def _compare(left, right, tolerance, *, physical_layout):
    require(len(left) == len(right) == 12, "COMPARISON_CARDINALITY")
    by_left, by_right = ({row["cohort_record_id"]: row for row in item} for item in (left, right))
    require(set(by_left) == set(by_right), "COMPARISON_RECORD_SET")
    maximum = {"rank": 0, "top1": 0, "top2": 0, "margin": 0.0, "nll": 0.0}
    implementation_mismatches = []
    for record_id in sorted(by_left):
        a, b = by_left[record_id], by_right[record_id]
        if not a["packing_valid"] or not b["packing_valid"] or _identity(a, physical_layout=physical_layout) != _identity(b, physical_layout=physical_layout):
            implementation_mismatches.append(record_id)
            continue
        require(len(a["token_metrics"]) == len(b["token_metrics"]), "TOKEN_COUNT_MISMATCH")
        for ma, mb in zip(a["token_metrics"], b["token_metrics"], strict=True):
            maximum["rank"] = max(maximum["rank"], abs(ma["rank"] - mb["rank"]))
            maximum["top1"] = max(maximum["top1"], abs(int(ma["top1"]) - int(mb["top1"])))
            maximum["top2"] = max(maximum["top2"], abs(int(ma["top2"]) - int(mb["top2"])))
            maximum["margin"] = max(maximum["margin"], abs(ma["margin"] - mb["margin"]))
            maximum["nll"] = max(maximum["nll"], abs(ma["nll"] - mb["nll"]))
    numeric_pass = (maximum["rank"] <= tolerance["rank_abs_delta_max"] and
                    maximum["top1"] <= tolerance["top_flag_abs_delta_max"] and
                    maximum["top2"] <= tolerance["top_flag_abs_delta_max"] and
                    maximum["margin"] <= tolerance["margin_abs_delta_max"] and
                    maximum["nll"] <= tolerance["nll_abs_delta_max"])
    return {"implementation_mismatch_record_ids": implementation_mismatches, "deltas": maximum, "numeric_pass": numeric_pass}


def scorer(all_rows, protocol):
    """Classify stability only after completeness and repeatability gates."""
    expected = len(CONDITIONS) * len(REPEATS) * 12
    require(len(all_rows) == expected, "GLOBAL_COMPLETENESS")
    grouped = defaultdict(list)
    for row in all_rows:
        grouped[(row["requested_batch_size"], row["repeat_identity"])].append(row)
    require(set(grouped) == {(batch, repeat) for batch in CONDITIONS for repeat in REPEATS}, "CONDITION_MATRIX_COMPLETENESS")
    within, cross = {}, {}
    for batch in CONDITIONS:
        comparison = _compare(grouped[(batch, 0)], grouped[(batch, 1)], protocol["within_condition_tolerance"], physical_layout=True)
        within[str(batch)] = comparison
        if comparison["implementation_mismatch_record_ids"]:
            return {"classification": "IMPLEMENTATION_DEFECT", "within_condition": within, "cross_batch": cross}
        if not comparison["numeric_pass"]:
            return {"classification": "RUNTIME_INSTABILITY", "within_condition": within, "cross_batch": cross}
    reference = grouped[(1, 0)]
    for batch in CONDITIONS[1:]:
        comparison = _compare(reference, grouped[(batch, 0)], protocol["cross_batch_tolerance"], physical_layout=False)
        cross[str(batch)] = comparison
        if comparison["implementation_mismatch_record_ids"]:
            return {"classification": "IMPLEMENTATION_DEFECT", "within_condition": within, "cross_batch": cross}
    if all(item["numeric_pass"] for item in cross.values()):
        classification = "STABLE_ACROSS_BATCHES"
    else:
        classification = "REPRODUCIBLE_BATCH_DEPENDENT_NUMERICAL_EFFECT"
    return {"classification": classification, "within_condition": within, "cross_batch": cross}


def validate_protocol(protocol):
    require(protocol["scope"] == "BATCH_STABILITY_ONLY_NOT_CAPABILITY_ESTIMATION", "SCOPE_MISMATCH")
    measurement = protocol["measurement"]
    require(measurement["physical_batch_sizes"] == list(CONDITIONS), "PHYSICAL_BATCH_MATRIX_MISMATCH")
    require(measurement["minimum_repeats_per_condition"] == 2 and measurement["records"] == 12, "REPEAT_OR_SUBSET_MISMATCH")
    require(measurement["teacher_forced_only"] and measurement["generation_calls"] == 0 and
            measurement["optimizer_steps"] == 0 and measurement["backward_calls"] == 0, "NO_UPDATE_OR_GENERATION_CONTRACT")
    require(protocol["fresh_authorization_required"]["gpu_execution"], "FRESH_GPU_AUTHORIZATION_REQUIRED")
    return protocol


def launcher(records, runtime, protocol, authorization):
    """CPU-safe entrypoint used by the fixture and future preflight wrapper.

    The authorization object may *only* say that CPU preparation is permitted;
    it cannot turn this import-safe module into a GPU launch mechanism.
    """
    validate_protocol(protocol)
    require(authorization == {"cpu_fixture_only": True, "gpu_execution_authorized": False}, "CPU_ONLY_AUTHORIZATION_REQUIRED")
    return run(records, runtime, {"minimum_repeats_per_condition": protocol["measurement"]["minimum_repeats_per_condition"],
                                  "within_condition_tolerance": protocol["within_condition_tolerance"],
                                  "cross_batch_tolerance": protocol["cross_batch_tolerance"]})


def run(records, runtime, protocol):
    require(protocol["minimum_repeats_per_condition"] == 2, "REPEATS_NOT_FROZEN")
    rows = []
    for batch in CONDITIONS:
        for repeat in REPEATS:
            rows.extend(worker(records, runtime, batch, repeat))
    scored = scorer(rows, protocol)
    return {"protocol_sha256": sha256(protocol), "records": rows, "scored": scored,
            "generation_calls": 0, "optimizer_steps": 0, "backward_calls": 0,
            "physical_batch_sizes": list(CONDITIONS), "repeats": list(REPEATS)}
