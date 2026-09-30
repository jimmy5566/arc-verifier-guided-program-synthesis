#!/usr/bin/env python3
"""Contaminated, performance-only microprofile for the Regret Batch2 executor.

This runs the pre-registered 97d7923e pair only.  It never opens evaluation
solutions and writes to a new output directory; it is not a candidate-parity or
scientific-accuracy experiment.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.run_regret_fast_v1 import sha256_file
from scripts.turbodfs_d1_common import d1_cells_batch


TASK_ID = "97d7923e"
OUTPUT_INDEX = 0
DEPTH = 12
VIEWS = ("identity", "flip_ud")


class ForwardProfiler:
    """Attach CUDA-event timing without changing model inputs or outputs."""

    def __init__(self, model: Any, pad_token_id: int) -> None:
        import torch

        self.torch = torch
        self.model = model
        self.pad_token_id = int(pad_token_id)
        self.original = model.forward
        self.calls: list[dict[str, Any]] = []
        self.previous_dispatch_end: float | None = None

    def install(self) -> None:
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            input_ids = kwargs.get("input_ids", args[0] if args else None)
            if input_ids is None:
                raise RuntimeError("microprofile could not identify input_ids")
            width = int(input_ids.shape[0])
            sequence = int(input_ids.shape[1])
            # Recurrence inputs are one token wide.  The padded inactive lane is
            # exactly the frozen pad token, which is outside the legal ARC set.
            active = width if sequence > 1 else int((input_ids[:, -1] != self.pad_token_id).sum().item())
            started = time.perf_counter()
            before = self.torch.cuda.Event(enable_timing=True)
            after = self.torch.cuda.Event(enable_timing=True)
            before.record()
            output = self.original(*args, **kwargs)
            after.record()
            ended = time.perf_counter()
            self.calls.append({
                "physical_batch_width": width,
                "active_logical_lanes": active,
                "inactive_padded_lanes": width - active,
                "sequence_tokens": sequence,
                "dispatch_wall_ms": (ended - started) * 1000.0,
                "cpu_gap_before_dispatch_ms": None if self.previous_dispatch_end is None else (started - self.previous_dispatch_end) * 1000.0,
                "cuda_start": before,
                "cuda_end": after,
            })
            self.previous_dispatch_end = ended
            return output

        self.model.forward = wrapped

    def remove(self) -> None:
        self.model.forward = self.original

    def finalize(self) -> list[dict[str, Any]]:
        self.torch.cuda.synchronize()
        rows: list[dict[str, Any]] = []
        for index, call in enumerate(self.calls, start=1):
            rows.append({
                "forward_index": index,
                **{key: value for key, value in call.items() if key not in {"cuda_start", "cuda_end"}},
                "cuda_forward_ms": float(call["cuda_start"].elapsed_time(call["cuda_end"])),
            })
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) if rows else ["empty"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def decoder(caps: dict[str, Any]) -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        "CUMULATIVE_REGRET_r=4.00", int(caps["max_new_tokens"]), float(caps["max_score"]), None,
        int(caps["max_expanded_nodes"]), int(caps["max_completed_candidates"]),
        frontier_floor=int(caps["frontier_floor"]), local_time_limit_seconds=float(caps["local_time_limit_seconds"]),
        pad_token_id=int(caps["pad_token_id"]), arc_tokens=tuple(int(item) for item in caps["arc_tokens"]),
        diagnostic_trace=True, independent_lane_budgets=True, performance_profile=True,
    )


def aggregate(calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for active in (1, 2):
        subset = [row for row in calls if int(row["active_logical_lanes"]) == active]
        if not subset:
            continue
        values = sorted(float(row["cuda_forward_ms"]) for row in subset)
        gaps = [float(row["cpu_gap_before_dispatch_ms"]) for row in subset if row["cpu_gap_before_dispatch_ms"] is not None]
        rows.append({
            "active_logical_lanes": active,
            "physical_batch_width": 2 if any(int(row["physical_batch_width"]) == 2 for row in subset) else 1,
            "forward_count": len(subset),
            "mean_cuda_forward_ms": sum(values) / len(values),
            "median_cuda_forward_ms": values[len(values) // 2],
            "mean_dispatch_wall_ms": sum(float(row["dispatch_wall_ms"]) for row in subset) / len(subset),
            "mean_cpu_gap_before_dispatch_ms": sum(gaps) / len(gaps) if gaps else None,
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frozen = args.frozen_run.resolve()
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite {output}")
    contract = read_json(frozen / "REGRET_FAST_CONTRACT.json")
    no_gold_challenge(Path(contract["challenge"]))
    caps = contract["caps"]
    from scripts import run_eval60_authoritative_greedy_v1 as greedy

    runtime_args = SimpleNamespace(
        output=Path(contract["authoritative_root"]), challenge=Path(contract["challenge"]),
        reference_config=Path(contract["reference_config"]), model_path=Path(contract["model_path"]),
        native_config_dir=Path(contract["native_config_dir"]), gpu_id=0,
    )
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapter_map = adapter_records(Path(contract["adapter_manifest"]))
    adapter_sha = load_adapter(model, adapter_map[(TASK_ID, DEPTH)])
    task = view_task(tasks[TASK_ID], OUTPUT_INDEX)
    dec = decoder(caps)
    output.mkdir(parents=True)
    raw: dict[str, Any] = {"experiment": "CONTAMINATED_PERFORMANCE_ONLY_DEVELOPMENT", "solutions_accessed": False,
                           "frozen_run": str(frozen), "frozen_raw_sha256": sha256_file(frozen / "RAW_TARGET_BLIND_FREEZE.json"),
                           "task_id": TASK_ID, "output_index": OUTPUT_INDEX, "depth": DEPTH, "views": list(VIEWS),
                           "contract_sha256": contract["contract_sha256"], "adapter_sha256": adapter_sha}
    try:
        scalar_profiler = ForwardProfiler(model, int(caps["pad_token_id"])); scalar_profiler.install()
        scalar_started = time.perf_counter()
        scalar_rows = [d1_cells_batch(model=model, tokenizer=tokenizer, task=task, task_id=TASK_ID,
                                      output_index=OUTPUT_INDEX, depth=DEPTH, views=(view,),
                                      generation_config=generation_config, decoder=dec,
                                      checkpoint_sha=adapter_sha, diagnostic_trace=True)[0]
                       for view in VIEWS]
        scalar_wall = time.perf_counter() - scalar_started
        scalar_calls = scalar_profiler.finalize(); scalar_profiler.remove()

        batch_profiler = ForwardProfiler(model, int(caps["pad_token_id"])); batch_profiler.install()
        batch_started = time.perf_counter()
        batch_rows = d1_cells_batch(model=model, tokenizer=tokenizer, task=task, task_id=TASK_ID,
                                    output_index=OUTPUT_INDEX, depth=DEPTH, views=VIEWS,
                                    generation_config=generation_config, decoder=dec,
                                    checkpoint_sha=adapter_sha, diagnostic_trace=True)
        batch_wall = time.perf_counter() - batch_started
        batch_calls = batch_profiler.finalize(); batch_profiler.remove()
    finally:
        del model

    raw.update({"scalar_wall_seconds": scalar_wall, "batch2_wall_seconds": batch_wall,
                "scalar_rows": scalar_rows, "batch2_rows": batch_rows,
                "scalar_forward_calls": scalar_calls, "batch2_forward_calls": batch_calls,
                "scalar_forward_summary": aggregate(scalar_calls), "batch2_forward_summary": aggregate(batch_calls)})
    atomic_json(output / "MICROPROFILE_RAW.json", raw)
    write_csv(output / "scalar_forward_calls.csv", scalar_calls)
    write_csv(output / "batch2_forward_calls.csv", batch_calls)
    write_csv(output / "forward_summary.csv", [{"mode": "SCALAR", **row} for row in aggregate(scalar_calls)] +
              [{"mode": "BATCH2", **row} for row in aggregate(batch_calls)])
    atomic_json(output / "MICROPROFILE_MANIFEST.json", {
        "raw_sha256": sha256_file(output / "MICROPROFILE_RAW.json"),
        "solutions_accessed": False, "performance_only": True,
        "source_contract_sha256": contract["contract_sha256"],
    })


if __name__ == "__main__":
    main()
