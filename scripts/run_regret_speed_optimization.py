#!/usr/bin/env python3
"""Same-semantics, target-blind performance profiling for Regret4 TurboDFS.

This runner intentionally owns only the contaminated D1/D2 performance cohort.
It must never be pointed at Router-v0 Untouched12/24 artifacts.  The baseline
uses the existing D1 bridge unchanged; all raw records remain outside the Git
working tree and compact summaries can be copied in after a target-blind freeze.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import adapter_records, atomic_json, load_adapter, no_gold_challenge
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "REGRET_SPEED_OPTIMIZATION_V1"
POLICY = "CUMULATIVE_REGRET_r=4.00"
# All entries are historically contaminated D1/D2 development cells.  The last
# two were selected from frozen D2 runtime telemetry, not target correctness.
PERFORMANCE_CELLS: tuple[dict[str, str], ...] = (
    {"cell_key": "20270e3b:o0:d24:identity", "role": "cheap_success", "source": "D2 reference"},
    {"cell_key": "97d7923e:o0:d48:identity", "role": "medium", "source": "D2 validation"},
    {"cell_key": "d59b0160:o0:d24:identity", "role": "deep_success", "source": "D2 validation"},
    {"cell_key": "d59b0160:o0:d24:flip_ud", "role": "deep_success", "source": "D2 validation"},
    {"cell_key": "f931b4a8:o1:d12:transpose", "role": "two_to_three_k", "source": "D2 runtime telemetry"},
    {"cell_key": "d59b0160:o0:d12:identity", "role": "four_k_nonrescue_runaway", "source": "D2 runtime telemetry"},
)


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def parse_cell_key(value: str) -> tuple[str, int, int, str]:
    output, remainder = value.rsplit(":d", 1)
    depth_s, view = remainder.split(":", 1)
    task_id, output_index = output.split(":o", 1)
    return task_id, int(output_index), int(depth_s), view


def cell_path(root: Path, key: str) -> Path:
    return root / "raw" / (key.replace(":", "_") + ".json")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


class Sampler:
    """Observational telemetry; no decoder state is read or mutated."""

    def __init__(self, interval_seconds: float = 0.5) -> None:
        self.interval_seconds = interval_seconds
        self.rows: list[dict[str, Any]] = []
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def close(self) -> None:
        self.stop.set(); self.thread.join(timeout=self.interval_seconds * 3)

    def _run(self) -> None:
        try:
            import psutil
            process = psutil.Process(os.getpid())
            process.cpu_percent(None)
        except Exception:  # telemetry must never break target-blind generation
            process = None
        while not self.stop.is_set():
            row: dict[str, Any] = {"unix": time.time()}
            if process is not None:
                try: row["cpu_process_percent"] = process.cpu_percent(None)
                except Exception: row["cpu_process_percent"] = None
            try:
                value = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                ).strip().splitlines()[0]
                util, memory, power = [item.strip() for item in value.split(",")]
                row.update({"gpu_util_percent": float(util), "gpu_memory_mb": float(memory), "gpu_power_w": float(power)})
            except Exception:
                row.update({"gpu_util_percent": None, "gpu_memory_mb": None, "gpu_power_w": None})
            self.rows.append(row)
            self.stop.wait(self.interval_seconds)


def prepare(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError(f"performance output exists: {output}")
    challenge = args.challenge.resolve()
    no_gold_challenge(challenge)
    contract = read_json(args.contract.resolve())
    caps = contract.get("caps", {})
    required = {"max_new_tokens", "max_score", "max_expanded_nodes", "max_completed_candidates", "frontier_floor", "local_time_limit_seconds", "pad_token_id", "arc_tokens"}
    if not required.issubset(caps):
        raise RuntimeError("D1 fixed-budget contract incomplete")
    adapters = adapter_records(args.adapter_manifest.resolve())
    for item in PERFORMANCE_CELLS:
        task, _output, depth, _view = parse_cell_key(item["cell_key"])
        if (task, depth) not in adapters:
            raise RuntimeError(f"authoritative adapter missing for {item['cell_key']}")
    output.mkdir(parents=True)
    cohort = {
        "experiment": EXPERIMENT,
        "scope": "CONTAMINATED_D1_D2_PERFORMANCE_ENGINEERING_ONLY",
        "forbidden_sources": ["regret_router_v0_untouched12", "regret_router_v0_untouched24"],
        "cells": list(PERFORMANCE_CELLS),
        "cell_count": len(PERFORMANCE_CELLS),
        "selection": "four protocol-required cells plus D2 2k-3k and 4096 non-rescue runtime representatives",
        "solutions_accessed": False,
    }
    config = {
        "experiment": EXPERIMENT,
        "source_commit": args.source_commit,
        "challenge": str(challenge), "challenge_sha256": sha256_file(challenge),
        "authoritative_root": str(args.authoritative_root.resolve()),
        "reference_config": str(args.reference_config.resolve()),
        "model_path": str(args.model_path.resolve()),
        "native_config_dir": str(args.native_config_dir.resolve()),
        "adapter_manifest": str(args.adapter_manifest.resolve()),
        "adapter_manifest_sha256": sha256_file(args.adapter_manifest.resolve()),
        "contract": str(args.contract.resolve()), "contract_sha256": sha256_file(args.contract.resolve()),
        "caps": caps, "policy": POLICY, "diagnostic_trace": True,
        "single_model_process": True, "single_gpu_worker": True,
        "solutions_accessed": False,
        "cohort_sha256": sha_json(cohort),
    }
    atomic_json(output / "PERFORMANCE_COHORT.json", cohort)
    atomic_json(output / "RUN_CONFIG.json", config)


def _decoder(config: dict[str, Any]) -> D1TurboDFSConfig:
    caps = config["caps"]
    return D1TurboDFSConfig(
        POLICY, int(caps["max_new_tokens"]), float(caps["max_score"]), None,
        int(caps["max_expanded_nodes"]), int(caps["max_completed_candidates"]),
        frontier_floor=int(caps["frontier_floor"]), local_time_limit_seconds=float(caps["local_time_limit_seconds"]),
        pad_token_id=int(caps["pad_token_id"]), arc_tokens=tuple(int(value) for value in caps["arc_tokens"]),
        diagnostic_trace=bool(config["diagnostic_trace"]),
    )


def _existing(path: Path, config: dict[str, Any]) -> bool:
    try:
        row = read_json(path)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (row.get("experiment") == EXPERIMENT and row.get("policy") == POLICY and
            row.get("config_sha256") == sha_json(config) and row.get("solutions_accessed") is False)


def baseline(args: argparse.Namespace) -> None:
    output = args.output.resolve(); config = read_json(output / "RUN_CONFIG.json")
    cohort = read_json(output / "PERFORMANCE_COHORT.json")
    no_gold_challenge(Path(config["challenge"]))
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(
        output=Path(config["authoritative_root"]), challenge=Path(config["challenge"]),
        reference_config=Path(config["reference_config"]), model_path=Path(config["model_path"]),
        native_config_dir=Path(config["native_config_dir"]), gpu_id=0,
    )
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    adapters = adapter_records(Path(config["adapter_manifest"]))
    decoder = _decoder(config)
    sampler = Sampler(); sampler.start(); started = time.perf_counter(); last_adapter: tuple[str, int] | None = None
    completed: list[str] = []
    try:
        for item in cohort["cells"]:
            key = str(item["cell_key"]); destination = cell_path(output, key)
            if _existing(destination, config):
                completed.append(key); continue
            task_id, output_index, depth, view = parse_cell_key(key)
            adapter_key = (task_id, depth)
            adapter_sha = load_adapter(model, adapters[adapter_key]) if adapter_key != last_adapter else str(adapters[adapter_key]["sha256"])
            last_adapter = adapter_key
            row = d1_cells_batch(
                model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index), task_id=task_id,
                output_index=output_index, depth=depth, views=(view,), generation_config=generation_config,
                decoder=decoder, checkpoint_sha=adapter_sha, diagnostic_trace=True,
            )[0]
            row.update({"experiment": EXPERIMENT, "policy": POLICY, "config_sha256": sha_json(config),
                        "cohort_sha256": config["cohort_sha256"], "role": item["role"], "cell_key": key,
                        "adapter_sha256": adapter_sha, "solutions_accessed": False})
            atomic_json(destination, row); completed.append(key)
    finally:
        sampler.close()
        del model
    atomic_json(output / "BASELINE_TARGET_BLIND_COMPLETE.json", {
        "experiment": EXPERIMENT, "cells": completed, "count": len(completed),
        "elapsed_seconds": time.perf_counter() - started, "solutions_accessed": False,
        "raw_sha256": {key: sha256_file(cell_path(output, key)) for key in completed},
    })
    write_csv(output / "gpu_utilization.csv", sampler.rows)


def summarize(args: argparse.Namespace) -> None:
    output = args.output.resolve(); config = read_json(output / "RUN_CONFIG.json"); cohort = read_json(output / "PERFORMANCE_COHORT.json")
    rows = []
    timing = []
    for item in cohort["cells"]:
        raw = read_json(cell_path(output, str(item["cell_key"])))
        if raw.get("solutions_accessed") is not False:
            raise RuntimeError("target-blind boundary violated")
        nodes = max(1, int(raw["nodes_expanded"])); forwards = int(raw["batch_forward_passes"])
        rows.append({
            "cell_key": item["cell_key"], "role": item["role"], "runtime_seconds": raw["runtime_seconds"],
            "nodes_expanded": raw["nodes_expanded"], "seconds_per_node": float(raw["runtime_seconds"]) / nodes,
            "batch_forward_passes": forwards, "forward_passes_per_node": forwards / nodes,
            "tokens_advanced": raw["tokens_advanced"], "successors_considered": raw["successors_considered"],
            "successors_retained": raw["successors_retained"], "candidate_count": raw["candidate_count"],
            "max_frontier_size": raw["max_frontier_size"], "mean_frontier_size": raw["mean_frontier_size"],
            "peak_vram_mb": raw["peak_vram_mb"], "peak_reserved_vram_mb": raw["peak_reserved_vram_mb"],
            "termination_reason": raw["termination_reason"], "candidate_pool_sha256": sha_json(raw["candidates"]),
        })
        timing.append({"cell_key": item["cell_key"], "model_forward_seconds": raw["model_forward_seconds"],
                       "candidate_parse_seconds": raw["candidate_parse_seconds"], "dedup_seconds": raw["dedup_seconds"],
                       "other_python_seconds": raw["python_overhead_seconds"], "runtime_seconds": raw["runtime_seconds"]})
    report = args.report_dir.resolve(); report.mkdir(parents=True, exist_ok=True)
    write_csv(report / "baseline_profile.csv", rows); write_csv(report / "timing_breakdown.csv", timing)
    source_telemetry = output / "gpu_utilization.csv"
    (report / "gpu_utilization.csv").write_bytes(source_telemetry.read_bytes())
    (report / "PERFORMANCE_COHORT.json").write_bytes((output / "PERFORMANCE_COHORT.json").read_bytes())
    environment = {"experiment": EXPERIMENT, "source_commit": config["source_commit"], "policy": POLICY,
                   "contract_sha256": config["contract_sha256"], "single_model_process": True, "single_gpu_worker": True,
                   "solutions_accessed": False}
    atomic_json(report / "POD_ENVIRONMENT.json", environment)
    atomic_json(report / "provenance.json", {"run_config_sha256": sha256_file(output / "RUN_CONFIG.json"),
                                               "raw_complete_sha256": sha256_file(output / "BASELINE_TARGET_BLIND_COMPLETE.json"),
                                               "sealed_router_sources_accessed": False, "solutions_accessed": False})
    (report / "REGRET_SPEED_PROFILE.md").write_text(
        "# Regret4 same-semantics baseline\n\n"
        "Target-blind baseline for the contaminated D1/D2-only six-cell performance cohort. "
        "No Router-v0 Untouched12/24 artifact or evaluation solution was accessed.\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    for name in ("output", "challenge", "authoritative_root", "reference_config", "model_path", "native_config_dir", "adapter_manifest", "contract"):
        prepare_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    prepare_parser.add_argument("--source-commit", required=True)
    baseline_parser = sub.add_parser("baseline"); baseline_parser.add_argument("--output", type=Path, required=True)
    summary_parser = sub.add_parser("summarize"); summary_parser.add_argument("--output", type=Path, required=True); summary_parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args(); {"prepare": prepare, "baseline": baseline, "summarize": summarize}[args.command](args)


if __name__ == "__main__":
    main()
