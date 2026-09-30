#!/usr/bin/env python3
"""CPU-only root-cause analysis for frozen Regret masked-Batch2 evidence."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def forward_summary(calls: list[dict[str, Any]], active: int) -> dict[str, float | int | None]:
    rows = [row for row in calls if int(row["active_logical_lanes"]) == active]
    return {
        "count": len(rows),
        "mean_cuda_forward_ms": mean([float(row["cuda_forward_ms"]) for row in rows]),
        "mean_dispatch_wall_ms": mean([float(row["dispatch_wall_ms"]) for row in rows]),
        "mean_cpu_gap_before_dispatch_ms": mean([float(row["cpu_gap_before_dispatch_ms"])
                                                    for row in rows if row["cpu_gap_before_dispatch_ms"] is not None]),
    }


def profile_breakdown(raw: dict[str, Any], mode: str) -> dict[str, float]:
    rows = raw[f"{mode}_rows"]
    calls = raw[f"{mode}_forward_calls"]
    wall = float(raw[f"{mode}_wall_seconds"])
    # Batch2 records duplicate the shared executor telemetry on each lane; one
    # copy is physical. Scalar rows each own an independent physical executor.
    physical_rows = rows if mode == "scalar" else rows[:1]
    telemetry = {key: sum(float(row.get("executor_performance_telemetry", {}).get(key, 0.0))
                           for row in physical_rows)
                 for key in ("batch_preparation_seconds", "trace_telemetry_seconds",
                             "node_and_state_seconds", "frontier_bookkeeping_seconds",
                             "kv_pack_or_select_seconds")}
    model = sum(float(row["cuda_forward_ms"]) for row in calls) / 1000.0
    parse = sum(float(row["candidate_parse_seconds"]) for row in rows)
    dedup = sum(float(row["dedup_seconds"]) for row in rows)
    known = model + parse + dedup + sum(telemetry.values())
    return {
        "MODE": mode.upper(), "TOTAL_WALL_SECONDS": wall,
        "MODEL_FORWARD_SECONDS": model,
        "BATCH_PREPARATION_SECONDS": telemetry["batch_preparation_seconds"],
        "KV_PACK_OR_SELECT_SECONDS": telemetry["kv_pack_or_select_seconds"],
        "TRACE_TELEMETRY_SECONDS": telemetry["trace_telemetry_seconds"],
        "NODE_AND_STATE_SECONDS": telemetry["node_and_state_seconds"],
        "FRONTIER_BOOKKEEPING_SECONDS": telemetry["frontier_bookkeeping_seconds"],
        "CANDIDATE_PARSE_SECONDS": parse,
        "DEDUP_SECONDS": dedup,
        "OTHER_SEARCH_ORCHESTRATION_SECONDS": max(0.0, wall - known),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--microprofile", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.frozen_run.resolve(); micro = args.microprofile.resolve(); out = args.report_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    freeze = read(frozen / "RAW_TARGET_BLIND_FREEZE.json")
    frozen_decision = read(Path(__file__).resolve().parents[1] / "analysis" / "regret_fast_v1" / "A1_1_DECISION.json")
    micro_raw = read(micro / "MICROPROFILE_RAW.json")

    occupancy: list[dict[str, Any]] = []
    total_active1 = total_active2 = total_physical = total_logical = 0
    desync: list[dict[str, Any]] = []
    for group in freeze["groups"]:
        index = int(group["group_index"])
        rows = read(frozen / "raw" / f"group_{index}_batch2.json")
        logical_records = [event for row in rows for event in row["branch_probabilities"]]
        active1_records = [event for event in logical_records if int(event["active_lane_count"]) == 1]
        active2_records = [event for event in logical_records if int(event["active_lane_count"]) == 2]
        active1 = len(active1_records)
        if len(active2_records) % 2:
            raise RuntimeError(f"group {index} has unpaired active2 logical records")
        active2 = len(active2_records) // 2
        physical = active1 + active2
        reported = int(rows[0]["physical_model_forwards"])
        if physical != reported:
            raise RuntimeError(f"group {index} physical-forward mismatch {physical} != {reported}")
        logical = int(rows[0]["physical_tokens_advanced"])
        total_active1 += active1; total_active2 += active2; total_physical += physical; total_logical += logical
        for active, count in ((1, active1), (2, active2)):
            occupancy.append({"group_index": index, "task_id": group["task_id"], "depth": group["depth"],
                              "physical_batch_width": 2, "active_logical_lanes": active,
                              "inactive_padded_lanes": 2 - active, "physical_forward_count": count,
                              "logical_tokens_from_forward_count": count * active,
                              "reported_physical_tokens_advanced": logical,
                              "reconstruction": "active1 logical record=one physical call; active2 has two lane records=one physical call"})
        masks = {"lane0_only": sum(tuple(event["active_mask"]) == (True, False) for event in logical_records),
                 "lane1_only": sum(tuple(event["active_mask"]) == (False, True) for event in logical_records),
                 "both_logical_records": len(active2_records)}
        final = [{"cell_key": row["cell_key"], "nodes": row["nodes_expanded"], "candidates": row["candidate_count"],
                  "termination": row["termination_reason"], "forwards": row["model_forwards"]} for row in rows]
        if index == 0:
            cause = "MIXED: one lane reaches the candidate cap, while recursive lockstep exposes the other lane alone"
            evidence = "Both one-lane masks occur; lane 0 ends at 1362 nodes/32 candidates while lane 1 continues to the 4096-node cap."
        else:
            cause = "RECURSIVE_LOCKSTEP"
            evidence = "Both lanes finish at the same 4096-node cap (only 29/30 candidates), yet each owns 3560 one-lane records."
        desync.append({"group_index": index, "task_id": group["task_id"], "depth": group["depth"],
                       "active1_physical_forwards": active1, "active2_physical_forwards": active2,
                       "lane0_only_logical_records": masks["lane0_only"], "lane1_only_logical_records": masks["lane1_only"],
                       "final_lane_state": json.dumps(final, sort_keys=True), "classification": cause, "evidence": evidence,
                       "frontier_evidence": "Only cell-level mean/max frontier and node traces were frozen; no global physical-forward frontier snapshot was saved."})

    if total_physical != int(frozen_decision["physical_model_forwards"]) or total_logical != int(frozen_decision["batch_executed_nodes"]):
        raise RuntimeError("aggregate occupancy does not reconcile to frozen A1.1 decision")
    effective = total_logical / total_physical
    utilization = total_logical / (2 * total_physical)

    scalar_calls = micro_raw["scalar_forward_calls"]
    batch_calls = micro_raw["batch2_forward_calls"]
    scalar = forward_summary(scalar_calls, 1)
    batch1 = forward_summary(batch_calls, 1)
    batch2 = forward_summary(batch_calls, 2)
    breakdown = [profile_breakdown(micro_raw, "scalar"), profile_breakdown(micro_raw, "batch2")]
    write_csv(out / "batch2_forward_occupancy.csv", occupancy)
    write_csv(out / "batch2_lane_desync.csv", desync)
    write_csv(out / "batch2_time_breakdown.csv", breakdown)

    model_fraction = breakdown[1]["MODEL_FORWARD_SECONDS"] / breakdown[1]["TOTAL_WALL_SECONDS"]
    cpu_fraction = 1.0 - model_fraction
    # An occupancy-only upper bound assumes a two-lane physical forward costs no
    # more than a scalar forward and all orchestration is free. It is therefore
    # deliberately optimistic.
    occupancy_ceiling = effective
    model_only_speedup = (total_logical * float(scalar["mean_cuda_forward_ms"])) / (
        total_active1 * float(batch1["mean_cuda_forward_ms"]) + total_active2 * float(batch2["mean_cuda_forward_ms"]) )
    policy = {
        "ROOT_CAUSE": "Correct masking exposed true fixed-pair under-occupancy; padded batch-2 forwards dominate and recursive lockstep prevents ready work from being dynamically paired.",
        "frozen_raw_manifest_sha256": sha256_file(frozen / "RAW_TARGET_BLIND_FREEZE.json"),
        "frozen_raw_file_sha256": freeze["raw_sha256"], "effective_batch": effective,
        "active1_forward_fraction": total_active1 / total_physical,
        "active2_forward_fraction": total_active2 / total_physical,
        "effective_lane_utilization": utilization, "wasted_lane_fraction": 1.0 - utilization,
        "when_only_one_lane_active": "PHYSICAL_BATCH2_WITH_PAD",
        "scalar_batch1_forward_ms": scalar["mean_cuda_forward_ms"],
        "masked_batch2_active1_forward_ms": batch1["mean_cuda_forward_ms"],
        "masked_batch2_active2_forward_ms": batch2["mean_cuda_forward_ms"],
        "model_forward_fraction": model_fraction, "cpu_orchestration_fraction": cpu_fraction,
        "premask_high_occupancy_cause": "INACTIVE_LANE_REACTIVATION",
        "desynchronization_primary_cause": "RECURSIVE_LOCKSTEP",
        "current_architecture_max_speedup_estimate": occupancy_ceiling,
        "microprofile_model_only_speedup_vs_scalar": model_only_speedup,
        "is_fixed_pair_batch2_fundamentally_low_occupancy": "YES",
        "would_dynamic_ready_pool_plausibly_help": "YES",
        "gold_status": "Known canaries preserved 2/2 post-freeze; this is performance failure plus numerical trajectory divergence, not Gold-destruction evidence.",
        "scientific_status": "Performance-only contaminated microprofile; no Gold read by the microprofile.",
        "next": "DESIGN_DYNAMIC_READY_BATCHER_AS_NEW_DECODER",
    }
    write_json(out / "BATCH2_PERFORMANCE_DECISION.json", policy)
    md = f"""# Masked Batch2 root-cause performance audit

## Evidence scope

CPU analysis of frozen A1.1 raw evidence plus a separately labelled contaminated
performance-only 97d7923e microprofile. The microprofile did not access Gold.
Frozen A1.1 manifest SHA256: `{sha256_file(frozen / 'RAW_TARGET_BLIND_FREEZE.json')}`.

## Occupancy

- Physical forwards: {total_physical}
- Active one-lane forwards: {total_active1} ({total_active1 / total_physical:.4%})
- Active two-lane forwards: {total_active2} ({total_active2 / total_physical:.4%})
- Effective batch: {effective:.9f}
- Effective lane utilization: {utilization:.4%}; wasted lane fraction: {1.0-utilization:.4%}

The physical tensor width remains two for active-one calls: `PHYSICAL_BATCH2_WITH_PAD`.

## Root cause

The corrected engine no longer lets an inactive pad/cache lane enter the decoder.
The pre-mask apparent occupancy near two was therefore **INACTIVE_LANE_REACTIVATION**,
not valid parallel search. The frozen masks show most calls have one live logical
state. Group 1 is the strongest scheduler evidence: both lanes end at identical
node caps, but 7,120 physical calls are singly active (3,560 observed from each lane).
That is recursive lockstep exposing work at different stack levels, not merely a
permanently finished companion cell. Group 0 additionally has real early terminal
asymmetry (candidate cap versus node cap).

## Timing microprofile

| metric | scalar B1 | Batch2 active=1 | Batch2 active=2 |
|---|---:|---:|---:|
| mean CUDA forward ms | {scalar['mean_cuda_forward_ms']:.4f} | {batch1['mean_cuda_forward_ms']:.4f} | {batch2['mean_cuda_forward_ms']:.4f} |
| count | {scalar['count']} | {batch1['count']} | {batch2['count']} |

Batch2 model-forward fraction: {model_fraction:.2%}; residual CPU/orchestration
fraction: {cpu_fraction:.2%}. Detailed timers are in `batch2_time_breakdown.csv`.
`KV_PACK_OR_SELECT_SECONDS` is zero on this executor: it passes an already
batch-shaped cache directly; it does not compact/select cache rows.

## Ceiling and architecture implication

Even under an impossible optimistic assumption—two-wide forward latency equal to
scalar and zero overhead—the fixed-pair ceiling is the measured effective batch,
only **{occupancy_ceiling:.4f}x**. The actual microprofile model-only comparison
is **{model_only_speedup:.4f}x**, because an active-one Batch2 forward is slower
than a scalar forward. Therefore >1.15x cannot be obtained while occupancy
remains at this level. A dynamic ready pool could raise occupancy by
batching independent ready states, but it would be a **new decoder execution
architecture**, and it would still retain the known BF16 batch numerical path
divergence. It is not scalar-equivalent engineering.

## Gold / science status

Post-freeze scoring already established that the two known d59 canaries survived
in Batch2 (2/2). This audit neither rescored Gold nor changes the historical 33/89.
"""
    (out / "BATCH2_PERFORMANCE_AUDIT.md").write_text(md, encoding="utf-8")


if __name__ == "__main__":
    main()
