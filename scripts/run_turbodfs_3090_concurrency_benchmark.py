#!/usr/bin/env python3
"""Target-blind RTX3090 TurboDFS V5 concurrency-capacity benchmark.

This is deliberately a small engineering benchmark.  It only loads retained
authoritative adapters, never calls TTT/Greedy, and never opens evaluation
solutions.  The controller writes a few aggregate CSV/JSON files rather than
one artifact per search cell.
"""
from __future__ import annotations

import argparse
import csv
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
from types import SimpleNamespace
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

DEPTHS = (12, 24, 48)
VIEWS = ("identity", "flip_ud", "transpose", "anti_transpose")
PARTNER = {"identity": "flip_ud", "flip_ud": "identity", "transpose": "anti_transpose", "anti_transpose": "transpose"}


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha_text(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def adapter_map(path: Path) -> dict[tuple[str, int], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    result = {(str(row["task_id"]), int(row["depth"])): row for row in rows}
    if len(rows) != 180 or len(result) != 180:
        raise RuntimeError("expected exactly 180 retained adapter mappings")
    return result


def select_cells(calibration: Path) -> list[dict[str, Any]]:
    """Choose eight calibration identities, using no target/outcome information."""
    with calibration.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if len(raw) != 24:
        raise RuntimeError(f"frozen full calibration cohort must have 24 rows, got {len(raw)}")
    rows = []
    for row in raw:
        item = {"task_id": str(row["task_id"]), "output_index": int(row["output_index"]),
                "depth": int(row["depth"]), "view": str(row["view"])}
        if item["depth"] not in DEPTHS or item["view"] not in VIEWS:
            raise RuntimeError(f"invalid calibration identity:{item}")
        item["selection_hash"] = hashlib.sha256(
            f"turbodfs_3090_concurrency_v1:{item['task_id']}:{item['output_index']}:{item['depth']}:{item['view']}".encode()
        ).hexdigest()
        rows.append(item)
    rows.sort(key=lambda row: (row["selection_hash"], row["task_id"], row["output_index"], row["depth"], row["view"]))
    chosen: list[dict[str, Any]] = []
    # Fixed coverage priority, then the deterministic hash order fills to eight.
    for depth in DEPTHS:
        candidate = next((row for row in rows if row["depth"] == depth and row not in chosen), None)
        if candidate is None: raise RuntimeError(f"calibration has no depth {depth}")
        chosen.append(candidate)
    for view in VIEWS:
        candidate = next((row for row in rows if row["view"] == view and row not in chosen), None)
        if candidate is not None: chosen.append(candidate)
    for row in rows:
        if len(chosen) >= 8: break
        if row not in chosen: chosen.append(row)
    if len(chosen) != 8 or len({(row["task_id"], row["output_index"], row["depth"], row["view"]) for row in chosen}) != 8:
        raise RuntimeError("could not freeze eight unique benchmark cells")
    return chosen


def candidate_signature(cell: dict[str, Any]) -> dict[str, Any]:
    candidates = []
    for row in cell.get("candidates", []):
        candidates.append({
            "token_ids": row.get("candidate_token_ids"),
            "cumulative_nll": row.get("cumulative_nll"),
            "canonical": row.get("canonical_candidate"),
            "order": row.get("candidate_discovery_order"),
        })
    return {
        "termination_reason": cell.get("termination_reason"),
        "complete_candidate_count": cell.get("complete_candidate_count"),
        "candidate_signature": sha_text(candidates),
        "candidate_payload": candidates,
        "tree_signature": sha_text({"nodes": cell.get("nodes", []), "branch_probabilities": cell.get("branch_probabilities", [])}),
        "nodes_expanded": cell.get("nodes_expanded"),
        "model_forwards": cell.get("model_forwards"),
        "tokens_advanced": cell.get("tokens_advanced"),
    }


def safe_adapter_load(model: Any, metadata: dict[str, Any], mapping: dict[str, str]) -> None:
    """Load bytes once; identity comes from the already completed Global audit.

    Loading a safetensors adapter is unavoidable.  We intentionally do not do
    a second one-gigabyte SHA256 pass before that load.  Header/tensor-count,
    Global mapping, size and authoritative metadata remain fail-closed.
    """
    from peft import set_peft_model_state_dict
    from safetensors import safe_open
    from safetensors.torch import load_file

    path = Path(mapping["global_path"])
    if metadata.get("checkpoint_sha256") != mapping.get("sha256"):
        raise RuntimeError("authoritative/global adapter identity disagreement")
    if not path.is_file() or path.stat().st_size != int(mapping["size"]):
        raise RuntimeError(f"adapter path/size mismatch:{path}")
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        tensor_count = len(handle.keys())
    if tensor_count != int(metadata["tensor_count"]):
        raise RuntimeError("adapter header tensor count mismatch")
    set_peft_model_state_dict(model, load_file(str(path), device="cpu"), adapter_name="default")


def run_child(args: argparse.Namespace) -> None:
    # Heavy imports are child-only so the controller never owns a CUDA context.
    import torch
    from scripts import run_eval60_adaptive_inference_joint_v2 as common
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    from scripts.run_adaptive_ttt_loo_transfer12 import view_task
    from scripts.turbodfs_v5_common import decoder_from, turbo_cells_v5_batch

    spec = read_json(args.spec)
    result_path = Path(spec["result_path"]); ready_path = Path(spec["ready_path"])
    go_path = Path(spec["go_path"]); abort_path = Path(spec["abort_path"])
    started = time.perf_counter()
    try:
        runtime_args = SimpleNamespace(output=Path(spec["authoritative_root"]), challenge=Path(spec["challenge"]),
            reference_config=Path(spec["reference_config"]), model_path=Path(spec["model_path"]),
            native_config_dir=Path(spec["native_config_dir"]), gpu_id=0)
        auth_root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
        model_load_seconds = time.perf_counter() - started
        adapters = adapter_map(Path(spec["adapter_manifest"]))
        _payload, decoder = decoder_from(Path(spec["final_config"]))
        representative = spec["cells"][0]
        mapping = adapters[(representative["task_id"], int(representative["depth"]))]
        metadata = read_json(auth_root / "checkpoints" / representative["task_id"] / f"depth_{int(representative['depth']):03d}" / "metadata.json")
        safe_adapter_load(model, metadata, mapping)
        torch.cuda.synchronize();
        ready = {"status": "READY", "worker_index": spec["worker_index"], "model_load_seconds": model_load_seconds,
                 "allocated_bytes": int(torch.cuda.memory_allocated()), "reserved_bytes": int(torch.cuda.memory_reserved()),
                 "adapter_checkpoint_sha256": mapping["sha256"], "pid": os.getpid()}
        atomic_json(ready_path, ready)
        while not go_path.exists() and not abort_path.exists(): time.sleep(0.05)
        if abort_path.exists():
            atomic_json(result_path, {**ready, "status": "ABORTED_SAFETY", "records": []}); return
        records: list[dict[str, Any]] = []
        for cell in spec["cells"]:
            mapping = adapters[(cell["task_id"], int(cell["depth"]))]
            metadata = read_json(auth_root / "checkpoints" / cell["task_id"] / f"depth_{int(cell['depth']):03d}" / "metadata.json")
            safe_adapter_load(model, metadata, mapping)
            torch.cuda.reset_peak_memory_stats()
            target, _companion = turbo_cells_v5_batch(
                model=model, tokenizer=tokenizer, task=view_task(tasks[cell["task_id"]], int(cell["output_index"])),
                task_id=cell["task_id"], output_index=int(cell["output_index"]), depth=int(cell["depth"]),
                views=(cell["view"], PARTNER[cell["view"]]), generation_config=generation_config,
                decoder=decoder, checkpoint_sha=mapping["sha256"],
            )
            signature = candidate_signature(target)
            records.append({
                **cell, "worker_index": spec["worker_index"], "checkpoint_sha256": mapping["sha256"],
                "wall_seconds": float(target["runtime_seconds"]), "shared_batch_wall_seconds": float(target["shared_batch_runtime_seconds"]),
                "nodes_expanded": int(target["nodes_expanded"]), "model_forwards": int(target["model_forwards"]),
                "tokens_advanced": int(target["tokens_advanced"]), "complete_candidate_count": int(target["complete_candidate_count"]),
                "valid_grid_count": int(target["valid_grid_count"]), "termination_reason": str(target["termination_reason"]),
                "frontier_floor_activation_count": int(target.get("frontier_floor_activation_count", 0)),
                "peak_allocated_bytes": int(target["peak_allocated_bytes"]), "peak_reserved_bytes": int(target["peak_reserved_bytes"]),
                **signature,
            })
        torch.cuda.synchronize()
        atomic_json(result_path, {**ready, "status": "COMPLETE", "records": records,
                                  "worker_wall_seconds": time.perf_counter() - started})
    except BaseException as error:  # serialize any OOM/driver/runtime failure to the controller
        try:
            peak_allocated = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else None
            peak_reserved = int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else None
        except Exception:
            peak_allocated = peak_reserved = None
        atomic_json(result_path, {"status": "ERROR", "worker_index": spec.get("worker_index"),
                                  "error_type": type(error).__name__, "error": str(error),
                                  "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved})
        raise
    finally:
        try: del model
        except Exception: pass


def gpu_sample(gpu: int) -> dict[str, float] | None:
    command = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,power.draw", "--format=csv,noheader,nounits", "-i", str(gpu)]
    try:
        text = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
        util, used, power = (float(value.strip()) for value in text.split(","))
        return {"unix": time.time(), "gpu_util_pct": util, "used_mib": used, "power_w": power}
    except Exception:
        return None


def run_config(args: argparse.Namespace, *, config_id: str, worker_count: int, cells: list[dict[str, Any]], load_only: bool = False) -> dict[str, Any]:
    root = args.output.resolve(); config_root = root / "_control" / config_id
    if config_root.exists(): shutil.rmtree(config_root)
    config_root.mkdir(parents=True)
    assignments = [cells[index::worker_count] for index in range(worker_count)]
    processes: list[subprocess.Popen[str]] = []
    specs: list[Path] = []
    for index, assigned in enumerate(assignments):
        spec = {
            "worker_index": index, "cells": assigned, "result_path": str(config_root / f"worker_{index}.json"),
            "ready_path": str(config_root / f"worker_{index}.ready.json"), "go_path": str(config_root / "GO"),
            "abort_path": str(config_root / "ABORT"), "authoritative_root": str(args.authoritative_root.resolve()),
            "challenge": str(args.challenge.resolve()), "reference_config": str(args.reference_config.resolve()),
            "model_path": str(args.model_path.resolve()), "native_config_dir": str(args.native_config_dir.resolve()),
            "adapter_manifest": str(args.adapter_manifest.resolve()), "final_config": str(args.final_config.resolve()),
        }
        path = config_root / f"worker_{index}.spec.json"; atomic_json(path, spec); specs.append(path)
        environment = os.environ.copy(); environment.update({"CUDA_VISIBLE_DEVICES": str(args.gpu_id), "TRITON_PTXAS_PATH": args.ptxas})
        processes.append(subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--child", "--spec", str(path)],
                                          cwd=str(ROOT), env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    telemetry: list[dict[str, float]] = []; started = time.perf_counter(); deadline = started + args.load_timeout_seconds
    readiness: list[dict[str, Any]] = []
    while time.perf_counter() < deadline:
        sample = gpu_sample(args.gpu_id)
        if sample: telemetry.append(sample)
        if all((config_root / f"worker_{index}.ready.json").is_file() for index in range(worker_count)):
            readiness = [read_json(config_root / f"worker_{index}.ready.json") for index in range(worker_count)]; break
        if any(process.poll() is not None for process in processes): break
        time.sleep(.5)
    max_used = max((row["used_mib"] for row in telemetry), default=0.0)
    ready_failed = len(readiness) != worker_count
    reserved_sum = sum(int(row.get("reserved_bytes") or 0) for row in readiness)
    safety_fail = worker_count == 3 and (ready_failed or max_used >= 23 * 1024 or (24 * 1024 - max_used) < 768 or reserved_sum >= int(23.0 * 1024**3))
    startup_fail = ready_failed and not safety_fail
    if safety_fail or startup_fail or load_only:
        (config_root / "ABORT").write_text("safety/load-only\n", encoding="utf-8")
    else:
        (config_root / "GO").write_text("go\n", encoding="utf-8")
    while any(process.poll() is None for process in processes):
        sample = gpu_sample(args.gpu_id)
        if sample: telemetry.append(sample)
        time.sleep(.5)
    stdout_stderr: list[dict[str, Any]] = []
    for index, process in enumerate(processes):
        stdout, stderr = process.communicate()
        stdout_stderr.append({"worker_index": index, "returncode": process.returncode, "stdout": stdout[-4000:], "stderr": stderr[-4000:]})
    elapsed = time.perf_counter() - started
    worker_results = [read_json(path) if path.is_file() else {"status": "MISSING_RESULT"}
                      for path in (config_root / f"worker_{index}.json" for index in range(worker_count))]
    records = [record for result in worker_results for record in result.get("records", [])]
    status = "PASS"
    if safety_fail: status = "INVALID_SAFETY"
    elif startup_fail: status = "ERROR_STARTUP"
    elif any(result.get("status") == "ERROR" or item["returncode"] != 0 for result, item in zip(worker_results, stdout_stderr, strict=True)):
        status = "OOM" if any("out of memory" in str(result).lower() or "out of memory" in item["stderr"].lower() for result, item in zip(worker_results, stdout_stderr, strict=True)) else "ERROR"
    elif len(records) != len(cells): status = "INVALID"
    return {"config_id": config_id, "worker_count": worker_count, "model_copy_count": worker_count,
            "shared_model_batch_size": 1, "status": status, "load_only": load_only,
            "elapsed_batch_wall_seconds": elapsed, "readiness": readiness, "records": records,
            "telemetry": telemetry, "worker_results": worker_results, "worker_logs": stdout_stderr,
            "max_nvidia_used_mib": max_used, "reserved_sum_bytes": reserved_sum, "control_root": str(config_root)}


def parity_against_serial(serial: dict[str, Any], candidate: dict[str, Any]) -> list[dict[str, Any]]:
    base = {(row["task_id"], int(row["output_index"]), int(row["depth"]), row["view"]): row for row in serial["records"]}
    result = []
    for row in candidate["records"]:
        key = (row["task_id"], int(row["output_index"]), int(row["depth"]), row["view"])
        reference = base.get(key)
        ref_candidates = reference.get("candidate_payload", []) if reference else []
        run_candidates = row.get("candidate_payload", [])
        tokens_equal = reference is not None and len(ref_candidates) == len(run_candidates) and all(
            left.get("token_ids") == right.get("token_ids") and left.get("canonical") == right.get("canonical") and
            left.get("order") == right.get("order") and abs(float(left.get("cumulative_nll")) - float(right.get("cumulative_nll"))) <= 1e-5
            for left, right in zip(ref_candidates, run_candidates, strict=True)
        )
        same_termination = reference is not None and row["termination_reason"] == reference["termination_reason"]
        same_complete_count = reference is not None and row["complete_candidate_count"] == reference["complete_candidate_count"]
        same_tree = reference is not None and row["tree_signature"] == reference["tree_signature"]
        passed = bool(tokens_equal and same_termination and same_complete_count)
        result.append({"configuration_id": candidate["config_id"], "task_id": key[0], "output_index": key[1],
                       "depth": key[2], "view": key[3], "candidate_tokens_equal": tokens_equal,
                       "termination_equal": same_termination, "complete_candidate_count_equal": same_complete_count,
                       "search_tree_equal": same_tree, "parity_pass": passed,
                       "serial_candidate_signature": reference.get("candidate_signature") if reference else None,
                       "candidate_signature": row["candidate_signature"]})
    return result


def summary_row(config: dict[str, Any], parity_rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = config["records"]
    wall_values = [float(row["wall_seconds"]) for row in records]
    nodes = [int(row["nodes_expanded"]) for row in records]
    forwards = [int(row["model_forwards"]) for row in records]
    peak = [int(row["peak_reserved_bytes"]) for row in records]
    telemetry = config["telemetry"]
    elapsed = float(config["elapsed_batch_wall_seconds"])
    return {
        "configuration_id": config["config_id"], "status": config["status"], "worker_count": config["worker_count"],
        "model_copy_count": config["model_copy_count"], "shared_model_batch_size": 1,
        "median_cell_latency_sec": statistics.median(wall_values) if wall_values else None,
        "p90_cell_latency_sec": sorted(wall_values)[math.ceil(.9 * len(wall_values)) - 1] if wall_values else None,
        "total_cells_completed": len(records), "elapsed_batch_wall_sec": elapsed,
        "cells_per_minute_per_gpu": len(records) * 60.0 / elapsed if elapsed else None,
        "gpu_seconds_per_cell": elapsed / len(records) if records else None,
        "median_nodes_per_sec": statistics.median([n / w for n, w in zip(nodes, wall_values, strict=True)]) if records else None,
        "median_model_forwards_per_sec": statistics.median([f / w for f, w in zip(forwards, wall_values, strict=True)]) if records else None,
        "median_nodes_expanded": statistics.median(nodes) if nodes else None,
        "median_model_forwards": statistics.median(forwards) if forwards else None,
        "peak_reserved_gib": max(peak, default=0) / 1024**3,
        "peak_nvidia_used_gib": float(config["max_nvidia_used_mib"]) / 1024,
        "oom_count": sum(result.get("status") == "ERROR" for result in config["worker_results"]),
        "error_count": sum(result.get("status") not in {"COMPLETE", "ABORTED_SAFETY"} for result in config["worker_results"]),
        "output_parity_rate": (sum(bool(row["parity_pass"]) for row in parity_rows) / len(parity_rows)) if parity_rows else None,
        "gpu_util_avg_pct": statistics.mean(row["gpu_util_pct"] for row in telemetry) if telemetry else None,
        "gpu_util_max_pct": max((row["gpu_util_pct"] for row in telemetry), default=None),
        "power_avg_w": statistics.mean(row["power_w"] for row in telemetry) if telemetry else None,
        "power_max_w": max((row["power_w"] for row in telemetry), default=None),
        "model_load_seconds": statistics.median([float(row["model_load_seconds"]) for row in config["readiness"]]) if config["readiness"] else None,
    }


def render_report(root: Path, *, source_commit: str, config_sha: str, cells: list[dict[str, Any]], summaries: list[dict[str, Any]],
                  parity: list[dict[str, Any]], best: dict[str, Any] | None, repeat: dict[str, Any] | None) -> None:
    lines = ["# RTX3090 TurboDFS concurrency capacity benchmark V1", "", "Target-blind infrastructure measurement only.", "",
             f"- Source commit: `{source_commit}`", f"- Frozen V5 config SHA256: `{config_sha}`", "- Gold/solutions accessed: `false`", f"- Benchmark cells: {len(cells)}", "",
             "## Configuration summary", "", "|config|status|cells/min/GPU|median cell s|peak reserved GiB|parity|", "|---|---:|---:|---:|---:|---:|"]
    for row in summaries:
        lines.append(f"|{row['configuration_id']}|{row['status']}|{row.get('cells_per_minute_per_gpu')}|{row.get('median_cell_latency_sec')}|{row.get('peak_reserved_gib')}|{row.get('output_parity_rate')}|")
    lines += ["", "## Decision", "", f"Best valid configuration: `{best['configuration_id'] if best else 'NONE'}`.",
              f"Output parity rows passing: {sum(bool(row['parity_pass']) for row in parity)}/{len(parity)}."]
    if repeat: lines.append(f"Repeat status: `{repeat['status']}`.")
    (root / "CONCURRENCY_BENCHMARK_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", action="store_true"); parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path); parser.add_argument("--authoritative-root", type=Path)
    parser.add_argument("--challenge", type=Path); parser.add_argument("--reference-config", type=Path)
    parser.add_argument("--model-path", type=Path); parser.add_argument("--native-config-dir", type=Path)
    parser.add_argument("--adapter-manifest", type=Path); parser.add_argument("--final-config", type=Path)
    parser.add_argument("--gpu-id", type=int, default=0); parser.add_argument("--ptxas", default="/usr/local/cuda/bin/ptxas")
    parser.add_argument("--source-commit", default="UNKNOWN"); parser.add_argument("--load-timeout-seconds", type=float, default=360.0)
    args = parser.parse_args()
    if args.child:
        if args.spec is None: parser.error("--child requires --spec")
        run_child(args); return
    required = ("output", "authoritative_root", "challenge", "reference_config", "model_path", "native_config_dir", "adapter_manifest", "final_config")
    if any(getattr(args, name) is None for name in required): parser.error("controller requires all runtime paths")
    for path in (args.authoritative_root, args.challenge, args.reference_config, args.model_path, args.native_config_dir, args.adapter_manifest, args.final_config):
        if not path.exists(): raise FileNotFoundError(path)
    if args.output.exists(): raise FileExistsError(f"refusing to overwrite benchmark output:{args.output}")
    if not (args.authoritative_root / "GREEDY_GENERATION_FROZEN.flag").is_file(): raise RuntimeError("authoritative Greedy freeze required")
    if not os.path.isfile(args.ptxas): raise RuntimeError(f"PTXAS missing:{args.ptxas}")
    final = read_json(args.final_config)
    if final.get("decoder_id") != "TURBODFS_OPT_V5_FRONTIER_FLOOR" or final.get("solutions_accessed") is not False:
        raise RuntimeError("final V5 target-blind config required")
    calibration = args.authoritative_root.parent / "eval60_turbodfs_v5" / "turbodfs_v5_full_cohort.csv"
    cells = select_cells(calibration)
    args.output.mkdir(parents=True)
    write_csv(args.output / "benchmark_cells.csv", cells, ["task_id", "output_index", "depth", "view", "selection_hash"])
    shutil.copy2(args.final_config, args.output / "FINAL_TURBODFS_CONFIG.json")
    atomic_json(args.output / "run_manifest.json", {"run_id": "turbodfs_3090_concurrency_benchmark_v1", "source_commit": args.source_commit,
        "decoder_config_sha256": sha_file(args.final_config), "decoder_id": final["decoder_id"], "target_blind": True,
        "solutions_accessed": False, "benchmark_cells_sha256": sha_file(args.output / "benchmark_cells.csv"), "cells": len(cells),
        "shared_model_multi_cell_batching": "NOT_CURRENTLY_SUPPORTED", "reason": "existing V5 interface is fixed to one task and two paired view lanes"})
    serial = run_config(args, config_id="A1_serial", worker_count=1, cells=cells)
    if serial["status"] != "PASS": raise RuntimeError(f"serial baseline failed:{serial['status']}")
    serial_parity = [{"configuration_id": "A1_serial", "task_id": row["task_id"], "output_index": row["output_index"], "depth": row["depth"], "view": row["view"],
                      "candidate_tokens_equal": True, "termination_equal": True, "complete_candidate_count_equal": True, "search_tree_equal": True, "parity_pass": True,
                      "serial_candidate_signature": row["candidate_signature"], "candidate_signature": row["candidate_signature"]} for row in serial["records"]]
    configs = [serial]; parity = list(serial_parity); summaries = [summary_row(serial, serial_parity)]
    two = run_config(args, config_id="A2_workers2", worker_count=2, cells=cells)
    two_parity = parity_against_serial(serial, two) if two["status"] == "PASS" else []
    configs.append(two); parity.extend(two_parity); summaries.append(summary_row(two, two_parity))
    stop_scaling = two["status"] != "PASS" or not two_parity or summaries[-1]["output_parity_rate"] != 1.0 or summaries[-1]["cells_per_minute_per_gpu"] < summaries[0]["cells_per_minute_per_gpu"] * 1.10 or summaries[-1]["median_nodes_expanded"] < summaries[0]["median_nodes_expanded"] * .85
    if stop_scaling:
        three = {"config_id": "A3_workers3", "worker_count": 3, "model_copy_count": 3, "status": "NOT_RUN_STOP_RULE", "records": [], "telemetry": [], "worker_results": [], "readiness": [], "elapsed_batch_wall_seconds": 0.0, "max_nvidia_used_mib": 0.0}
        three_parity: list[dict[str, Any]] = []
    else:
        three = run_config(args, config_id="A3_workers3", worker_count=3, cells=cells)
        three_parity = parity_against_serial(serial, three) if three["status"] == "PASS" else []
    configs.append(three); parity.extend(three_parity); summaries.append(summary_row(three, three_parity))
    valid = [row for row in summaries if row["status"] == "PASS" and row.get("output_parity_rate") == 1.0 and row.get("median_nodes_expanded", 0) >= summaries[0].get("median_nodes_expanded", 0) * .85]
    best_summary = max(valid, key=lambda row: float(row["cells_per_minute_per_gpu"])) if valid else None
    best_config = next((config for config in configs if best_summary and config["config_id"] == best_summary["configuration_id"]), None)
    repeat = None; repeat_parity: list[dict[str, Any]] = []
    if best_config is not None:
        repeat = run_config(args, config_id=f"{best_config['config_id']}_repeat", worker_count=int(best_config["worker_count"]), cells=cells)
        repeat_parity = parity_against_serial(serial, repeat) if repeat["status"] == "PASS" else []
        parity.extend(repeat_parity); summaries.append(summary_row(repeat, repeat_parity))
    serial_rows = [{**row, "configuration_id": "A1_serial"} for row in serial["records"]]
    worker_rows = [{**row, "configuration_id": config["config_id"]} for config in configs[1:] for row in config["records"]]
    if repeat: worker_rows.extend({**row, "configuration_id": repeat["config_id"]} for row in repeat["records"])
    write_csv(args.output / "serial_baseline.csv", serial_rows)
    write_csv(args.output / "independent_worker_results.csv", worker_rows)
    write_csv(args.output / "shared_model_batch_results.csv", [], ["configuration_id", "status", "reason"])
    write_csv(args.output / "configuration_summary.csv", summaries)
    write_csv(args.output / "output_parity.csv", parity)
    telemetry = [{"configuration_id": config["config_id"], **row} for config in configs for row in config["telemetry"]]
    if repeat: telemetry.extend({"configuration_id": repeat["config_id"], **row} for row in repeat["telemetry"])
    write_csv(args.output / "gpu_telemetry.csv", telemetry)
    atomic_json(args.output / "benchmark_result.json", {"status": "FROZEN_TARGET_BLIND", "solutions_accessed": False,
        "source_commit": args.source_commit, "decoder_config_sha256": sha_file(args.final_config), "cells": cells,
        "summaries": summaries, "best_configuration": best_summary["configuration_id"] if best_summary else None,
        "shared_model_multi_cell_batching": "NOT_CURRENTLY_SUPPORTED"})
    render_report(args.output, source_commit=args.source_commit, config_sha=sha_file(args.final_config), cells=cells, summaries=summaries, parity=parity, best=best_summary, repeat=repeat)
    hashes = {path.name: sha_file(path) for path in sorted(args.output.iterdir()) if path.is_file() and path.name != "hashes.json"}
    atomic_json(args.output / "hashes.json", hashes)
    print(json.dumps({"BENCHMARK_COMPLETED": "YES", "BEST_CONFIGURATION": best_summary["configuration_id"] if best_summary else None,
                      "OUTPUT_PARITY_PASS": all(row["parity_pass"] for row in parity), "output": str(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
