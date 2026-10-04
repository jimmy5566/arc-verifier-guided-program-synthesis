#!/usr/bin/env python3
"""One-L4 compatible physical-batch scaling, with one load and one root.

This is deliberately a hardware microbenchmark.  It has no production-view
bucket discovery, TTT, LoRA, candidate generation, solutions, or submission.
"""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback
from typing import Any

# Must run before the first torch import, including imports made by the audited
# forward core below.
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))
if str(ROOT / "src") not in os.sys.path:
    os.sys.path.insert(0, str(ROOT / "src"))

from scripts import run_l4_native_base_physical_batch_scaling_b1_b16_v1 as physical  # noqa: E402


EXPERIMENT = "L4_SINGLE_GPU_PHYSICAL_BATCH_SCALING_V1"
WIDTHS = (1, 2, 4, 8, 12, 16)
WARMUPS, MEASUREMENTS = 2, 12
GLOBAL_LIMIT, MODEL_READY_LIMIT, NO_PROGRESS_LIMIT, WIDTH_LIMIT = 6900, 600, 180, 600
VIEW = "identity"


def _gate(args: argparse.Namespace, *, stage: str, width: int | None = None) -> None:
    """Fail closed on the explicit single-process timing limits.

    A CUDA call itself cannot safely be interrupted in-process.  The notebook
    owns the bounded outer process group; this guard covers every boundary
    between load, root construction, warmups, and measured forwards.
    """
    now = time.monotonic()
    elapsed = now - args.started
    if elapsed > GLOBAL_LIMIT:
        raise RuntimeError(f"GLOBAL_TIME_GATE:{stage}:{elapsed:.3f}")
    if now - args.last_progress_monotonic > NO_PROGRESS_LIMIT:
        raise RuntimeError(f"NO_PROGRESS_TIME_GATE:{stage}:{now - args.last_progress_monotonic:.3f}")
    if width is not None and now - args.width_started_monotonic > WIDTH_LIMIT:
        raise RuntimeError(f"WIDTH_TIME_GATE:B{width}:{now - args.width_started_monotonic:.3f}")


def _json(path: Path, payload: Any) -> None:
    physical._atomic_json(path, payload)


def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row}) or ["status"]
    physical._atomic_csv(path, rows, fields)


def _progress(args: argparse.Namespace, event: str, **extra: Any) -> None:
    args.last_progress_monotonic = time.monotonic()
    physical.emit_progress(args, event, **extra)


def _specs(width: int) -> list[dict[str, Any]]:
    return [{"replica_index": index, "lane_index": index, "view": VIEW} for index in range(width)]


def _is_clean_capacity_width(width: int) -> bool:
    """Only the predeclared upper widths may become capacity endpoints."""
    return width in (12, 16)


def _quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] if low == high else ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _single_template(*, model: Any, context: dict[str, Any], config: Any, args: argparse.Namespace, torch: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    _progress(args, "ROOT_PREFILL_START", physical_batch=1, view=VIEW)
    cell = None
    try:
        cell = physical.start_ready_cell(
            model=model, input_ids=context["prompts"][VIEW], config=config,
            cell_key=f"{args.task_id}:o{args.output_index}:d{args.depth}:{VIEW}:single_root",
            normalize_root_cache=True, root_cache_transform=physical._cache_transform, cache_strategy="rollback",
        )
        template = physical._template_from_cell(cell, view=VIEW)
        templates = {VIEW: template}
        audit = physical._require_cpu_root_templates(templates)
        root_hash = physical._cache_tensor_hash(torch, template.legacy_cache)
    finally:
        if cell is not None:
            del cell
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)
    payload = {
        "MODEL_LOAD_COUNT": 1, "ROOT_PREFILL_COUNT": 1, "ROOT_TEMPLATE_MUTATED": False,
        "ROOT_TEMPLATE_GPU_TENSOR_COUNT": audit["template_gpu_tensor_count"],
        "ROOT_TEMPLATE_GPU_BYTES": audit["template_gpu_bytes"], "view": VIEW,
        "root_sequence_length": template.sequence_length, "position": template.position,
        "cache_key": list(template.cache_key), "cache_geometry": [list(t.shape) for layer in template.legacy_cache for t in layer],
        "root_template_sha256": root_hash, "task_id": args.task_id,
        "output_index": args.output_index, "depth": args.depth, "budget": args.budget,
        "memory": audit,
    }
    _progress(args, "ROOT_PREFILL_DONE", physical_batch=1, view=VIEW, root_sequence_length=template.sequence_length)
    return templates, payload


def _one_forward(*, templates: dict[str, Any], width: int, model: Any, torch: Any, args: argparse.Namespace, timed: bool) -> dict[str, Any]:
    cloned: list[Any] = []
    requests: list[Any] = []
    replies: list[Any] = []
    try:
        cloned = [physical._clone_template_lane(templates[VIEW], spec=spec, device=args.device) for spec in _specs(width)]
        torch.cuda.synchronize(args.device)  # clone/materialisation is outside timing.
        clone_integrity = physical._template_clone_integrity(templates=templates, cells=cloned)
        required_clone_checks = ("template_storage_disjoint", "sample_lane_storage_independent", "cache_owners_unique", "dynamic_caches_unique")
        if not all(clone_integrity[key] for key in required_clone_checks):
            raise RuntimeError("ROOT_TEMPLATE_LANE_ISOLATION_FAILED")
        requests = physical._requests(cloned)
        owners = [id(request.cache_owner.cache) for request in requests]
        storage = [physical._all_storage_pointers(request.cache_owner.cache) for request in requests]
        lengths = [physical._cache_sequence_length(request.cache_owner.cache) for request in requests]
        torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats(args.device)
        before = physical._memory(torch, args.device)
        started = time.perf_counter()  # Exact audited incremental-forward boundary.
        requests, replies, telemetry = physical._execute_once(model=model, cells=cloned)
        torch.cuda.synchronize(args.device)
        latency_ms = (time.perf_counter() - started) * 1000.0
        integrity = physical._lane_integrity(torch=torch, requests=requests, replies=replies, owner_ids_before=owners, storage_before=storage, lengths_before=lengths)
        safety = all(bool(integrity[key]) for key in ("owner_identity_preserved", "cache_owners_unique", "input_cache_storage_independent", "output_cache_storage_independent", "sequence_length_incremented_exactly_once", "finite_logits")) and integrity["owner_count"] == width
        peak_allocated = int(torch.cuda.max_memory_allocated(args.device))
        peak_reserved = int(torch.cuda.max_memory_reserved(args.device))
    finally:
        # All clone/reply state is sample-local.  The CPU root is deliberately
        # outside this scope and remains byte-identical for the full sweep.
        replies.clear(); requests.clear(); cloned.clear()
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(args.device)
    after = physical._memory(torch, args.device)
    thermal = physical._nvidia_telemetry(0)
    return {"latency_ms": latency_ms, "lanes_per_second": width / (latency_ms / 1000.0), "physical_batch": width,
            "cache_pack_seconds": float(telemetry["cache_pack_seconds"]), "model_call_seconds": float(telemetry["model_call_seconds"]),
            "cache_adoption_seconds": float(telemetry["cache_adoption_seconds"]), "execute_total_seconds": float(telemetry["scheduler_elapsed_seconds"]),
            "allocated_before_bytes": before["allocated_bytes"], "reserved_before_bytes": before["reserved_bytes"], "free_before_bytes": before["free_vram_bytes"],
            "peak_allocated_bytes": peak_allocated, "peak_reserved_bytes": peak_reserved,
            "allocated_after_release_bytes": after["allocated_bytes"], "reserved_after_release_bytes": after["reserved_bytes"], "free_after_release_bytes": after["free_vram_bytes"],
            "temperature_c": thermal.get("temperature_c"), "power_draw_w": thermal.get("power_draw_w"), "power_limit_w": thermal.get("power_limit_w"),
            "sm_clock_mhz": thermal.get("sm_clock_mhz"), "memory_clock_mhz": thermal.get("memory_clock_mhz"), "finite_logits": integrity["finite_logits"],
            "owner_identity_preserved": integrity["owner_identity_preserved"], "output_cache_storage_independent": integrity["output_cache_storage_independent"],
            "sequence_increment_once": integrity["sequence_length_incremented_exactly_once"], "status": "PASS" if safety else "SAFETY_FAIL", "timed": timed,
            "root_template_sha256": physical._cache_tensor_hash(torch, templates[VIEW].legacy_cache)}


def _measure_width(*, templates: dict[str, Any], width: int, model: Any, torch: Any, args: argparse.Namespace, warmups: int, measurements: int, label: str) -> dict[str, Any]:
    _progress(args, f"{label}_START", physical_batch=width)
    for _ in range(warmups):
        _gate(args, stage=f"{label}:warmup", width=width)
        _one_forward(templates=templates, width=width, model=model, torch=torch, args=args, timed=False)
    samples = []
    for index in range(measurements):
        _gate(args, stage=f"{label}:sample:{index}", width=width)
        sample = _one_forward(templates=templates, width=width, model=model, torch=torch, args=args, timed=True)
        sample["sample_index"] = index; sample["label"] = label; samples.append(sample)
        _progress(args, "SAMPLE_DONE", physical_batch=width, sample_index=index, label=label, latency_ms=sample["latency_ms"])
    _progress(args, f"{label}_DONE", physical_batch=width)
    return {"status": "PASS" if all(row["status"] == "PASS" for row in samples) else "SAFETY_FAIL", "samples": samples}


def _summary(rows: list[dict[str, Any]], width: int, baseline: float | None, previous: float | None) -> dict[str, Any]:
    latency, lanes = [float(row["latency_ms"]) for row in rows], [float(row["lanes_per_second"]) for row in rows]
    median = statistics.median(lanes)
    return {"physical_batch": width, "status": "PASS", "median_latency_ms": statistics.median(latency), "mean_latency_ms": statistics.mean(latency),
            **{f"p{int(q*100)}_latency_ms": _quantile(latency, q) for q in (.10, .25, .75, .90)},
            "median_lanes_s": median, "mean_lanes_s": statistics.mean(lanes), **{f"p{int(q*100)}_lanes_s": _quantile(lanes, q) for q in (.10, .25, .75, .90)},
            "vs_b1": None if baseline is None else median / baseline, "vs_previous": None if previous is None else median / previous,
            "batch_efficiency": None if baseline is None else median / baseline / width,
            "peak_allocated_gib": max(row["peak_allocated_bytes"] for row in rows) / 1024**3,
            "peak_reserved_gib": max(row["peak_reserved_bytes"] for row in rows) / 1024**3}


def _ratio_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = {row["physical_batch"]: row for row in summaries if row["status"] == "PASS"}
    return [
        {"ratio": f"B{after}_OVER_B{before}", "value": None if before not in summary or after not in summary else summary[after]["median_lanes_s"] / summary[before]["median_lanes_s"]}
        for before, after in zip(WIDTHS, WIDTHS[1:])
    ]


def _validate_model_mount(model_path: Path) -> None:
    if not model_path.is_dir():
        raise FileNotFoundError(f"MODEL_DIRECTORY_MISSING:{model_path}")
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"MODEL_CONFIG_MISSING:{config_path}")
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"MODEL_CONFIG_INVALID:{exc!r}") from exc
    if not isinstance(config, dict) or not config:
        raise RuntimeError("MODEL_CONFIG_INVALID_EMPTY")
    tokenizer_files = ("tokenizer.json", "tokenizer_config.json", "vocab.json")
    if not any((model_path / name).is_file() for name in tokenizer_files):
        raise FileNotFoundError("MODEL_TOKENIZER_FILES_MISSING")


def _report_markdown(summaries: list[dict[str, Any]], ratios: list[dict[str, Any]], stability: list[dict[str, Any]], decision: dict[str, Any]) -> str:
    lines = [f"# {EXPERIMENT}", "", "Single-GPU descriptive hardware timing only.", "", "| Batch | Median latency ms | Median lanes/s | vs B1 | vs previous | Efficiency | Peak alloc GiB | Peak reserved GiB | Status |", "|---|---:|---:|---:|---:|---:|---:|---:|---|"]
    for row in summaries:
        fmt = lambda value: "—" if value is None else f"{float(value):.4f}"
        lines.append("| B{physical_batch} | {latency} | {lanes} | {b1} | {previous} | {efficiency} | {allocated} | {reserved} | {status} |".format(
            physical_batch=row["physical_batch"], latency=fmt(row.get("median_latency_ms")), lanes=fmt(row.get("median_lanes_s")),
            b1=fmt(row.get("vs_b1")), previous=fmt(row.get("vs_previous")), efficiency=fmt(row.get("batch_efficiency")),
            allocated=fmt(row.get("peak_allocated_gib")), reserved=fmt(row.get("peak_reserved_gib")), status=row["status"]))
    lines.extend(["", f"B8_OVER_B4={decision['B8_OVER_B4']}", f"B12_OVER_B8={next((row['value'] for row in ratios if row['ratio'] == 'B12_OVER_B8'), None)}", f"B16_OVER_B12={next((row['value'] for row in ratios if row['ratio'] == 'B16_OVER_B12'), None)}", f"FASTEST_BATCH_BY_LANES_PER_SECOND={decision['fastest_batch']}", f"HIGHEST_BATCH_WITHOUT_OOM={decision['highest_batch_without_oom']}", "", "## B4/B8 stability", ""])
    for row in stability:
        lines.append(f"- B{row['physical_batch']}: repeat drift {row['relative_drift_percent']:.3f}%")
    return "\n".join(lines) + "\n"


def run(args: argparse.Namespace) -> None:
    args.output.mkdir(parents=True, exist_ok=True); args.device = "cuda:0"; args.run_started_unix = time.time(); args.last_progress_monotonic = args.started; args.width_started_monotonic = args.started
    if args.benchmark_model_mode != physical.BENCHMARK_MODEL_MODE: raise RuntimeError("BASE_MODEL_ONLY_REQUIRED")
    if time.monotonic() - args.started > GLOBAL_LIMIT: raise RuntimeError("GLOBAL_TIME_GATE")
    _json(args.output / "CONTRACT.json", {"experiment": EXPERIMENT, "GPU_COUNT_USED_FOR_TIMING": 1, "MODEL_LOAD_COUNT": 1, "ROOT_PREFILL_COUNT": 1, "WIDTHS": list(WIDTHS), "WARMUP_PER_WIDTH": WARMUPS, "MEASURED_PER_WIDTH": MEASUREMENTS, "BENCHMARK_MODEL_MODE": "BASE_MODEL_ONLY", "DTYPE": "BF16", "TTT_USED": False, "PEFT_USED": False, "GOLD_LOADED": False, "SUBMISSION_CREATED": False, "REAL_8VIEW_BUCKET_DISCOVERY_USED": False, "FOUR_GPU_AGGREGATION_USED": False})
    _json(args.output / "TIME_GATE_CONFIG.json", {"global_notebook_limit_seconds": GLOBAL_LIMIT, "model_ready_limit_seconds": MODEL_READY_LIMIT, "no_progress_limit_seconds": NO_PROGRESS_LIMIT, "per_width_limit_seconds": WIDTH_LIMIT, "cleanup_grace_seconds": 10})
    _validate_model_mount(args.model_path)
    _progress(args, "NOTEBOOK_START", physical_batch=1)
    _progress(args, "MODEL_LOAD_START", physical_batch=1)
    load_start = time.perf_counter(); torch, model, identity, config, context = physical._load_context(args); load_seconds = time.perf_counter()-load_start
    if load_seconds > MODEL_READY_LIMIT:
        raise RuntimeError(f"MODEL_READY_TIME_GATE:{load_seconds:.3f}")
    if not torch.cuda.is_bf16_supported(): raise RuntimeError("BF16_UNAVAILABLE")
    hardware = physical._nvidia_telemetry(0)
    if hardware.get("gpu_name") != "NVIDIA L4": raise RuntimeError(f"SELECTED_GPU_NOT_L4:{hardware}")
    _json(args.output / "HARDWARE.json", {"selected_physical_gpu": 0, "visible_cuda_devices": os.environ["CUDA_VISIBLE_DEVICES"], **hardware, "nvidia_smi": physical._nvidia_snapshot()})
    _json(args.output / "RUNTIME_ENVIRONMENT.json", {**identity, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "selected_torch_device": args.device})
    _json(args.output / "SOURCE_IDENTITY.json", {"experiment": EXPERIMENT, "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "model_path": str(args.model_path), "native_config_dir": str(args.native_config_dir), "challenge": str(args.challenge), "model_identity": identity})
    _progress(args, "MODEL_LOAD_DONE", physical_batch=1, model_load_seconds=load_seconds)
    _gate(args, stage="post_model_load")
    templates, root = _single_template(model=model, context=context, config=config, args=args, torch=torch)
    _gate(args, stage="post_root_prefill")
    _json(args.output / "ROOT_TEMPLATE_AUDIT.json", root)
    args.width_started_monotonic = time.monotonic()
    preflight = _measure_width(templates=templates, width=4, model=model, torch=torch, args=args, warmups=0, measurements=1, label="DYNAMICCACHE_PREFLIGHT")
    if preflight["status"] != "PASS": raise RuntimeError("DYNAMICCACHE_PREFLIGHT_FAILED")
    _progress(args, "ROOT_PARITY_PASS", physical_batch=1)
    results: dict[int, dict[str, Any]] = {}; raw: list[dict[str, Any]] = []
    for width in WIDTHS:
        start = time.monotonic(); args.width_started_monotonic = start; _progress(args, f"WIDTH_START_B{width}", physical_batch=width)
        try:
            result = _measure_width(templates=templates, width=width, model=model, torch=torch, args=args, warmups=WARMUPS, measurements=MEASUREMENTS, label=f"WIDTH_B{width}")
        except BaseException as exc:
            if physical._is_oom(exc) and _is_clean_capacity_width(width): result = {"status": "CLEAN_CAPACITY_FAILURE", "exception": repr(exc), "samples": []}; torch.cuda.empty_cache()
            else: raise
        if time.monotonic()-start > WIDTH_LIMIT: raise RuntimeError(f"WIDTH_TIME_GATE_B{width}")
        results[width] = result; raw.extend(result["samples"]); _json(args.output / f"B{width}_PARTIAL.json", result); _progress(args, f"WIDTH_DONE_B{width}", physical_batch=width, status=result["status"])
    repeats = {}
    for width in (8, 4):
        if results[width]["status"] == "PASS":
            args.width_started_monotonic = time.monotonic()
            repeats[width] = _measure_width(templates=templates, width=width, model=model, torch=torch, args=args, warmups=1, measurements=6, label=f"B{width}_REPEAT")
    initial_hash = root["root_template_sha256"]; final_hash = physical._cache_tensor_hash(torch, templates[VIEW].legacy_cache)
    if initial_hash != final_hash: raise RuntimeError("ROOT_TEMPLATE_MUTATED")
    _csv(args.output / "L4_SINGLE_GPU_BATCH_RAW.csv", raw)
    summaries=[]; baseline=previous=None
    for width in WIDTHS:
        if results[width]["status"] != "PASS": summaries.append({"physical_batch":width,"status":results[width]["status"]}); continue
        row=_summary(results[width]["samples"],width,baseline,previous); summaries.append(row); baseline=row["median_lanes_s"] if baseline is None else baseline; previous=row["median_lanes_s"]
    _csv(args.output / "L4_SINGLE_GPU_BATCH_SUMMARY.csv", summaries)
    summary={row["physical_batch"]:row for row in summaries if row["status"]=="PASS"}; ratios = _ratio_rows(summaries)
    _csv(args.output / "L4_SINGLE_GPU_BATCH_RATIOS.csv", ratios)
    stability=[]
    for width, repeat in repeats.items():
        primary=summary[width]["median_lanes_s"]; repeated=statistics.median(row["lanes_per_second"] for row in repeat["samples"]); stability.append({"physical_batch":width,"primary_median_lanes_s":primary,"repeat_median_lanes_s":repeated,"relative_drift_percent":(repeated/primary-1)*100})
    _csv(args.output / "L4_SINGLE_GPU_B4_B8_STABILITY.csv", stability)
    b84=next((row["value"] for row in ratios if row["ratio"]=="B8_OVER_B4"),None); decision="NOT_AVAILABLE" if b84 is None else ("MATERIAL_B8_GAIN" if b84>=1.10 else "SMALL_B8_GAIN" if b84>=1.03 else "NO_MATERIAL_B8_GAIN")
    decision_payload = {"experiment":EXPERIMENT,"B8_OVER_B4":b84,"B8_INTERPRETATION":decision,"fastest_batch":max(summary,key=lambda width:summary[width]["median_lanes_s"]) if summary else None,"highest_batch_without_oom":max(summary) if summary else None,"MODEL_LOAD_COUNT":1,"ROOT_PREFILL_COUNT":1,"MODEL_LOAD_SECONDS":load_seconds,"B4_B8_STABILITY_WARNING":any(abs(row["relative_drift_percent"]) > 10.0 for row in stability)}
    _json(args.output / "DECISION.json", decision_payload)
    (args.output / "REPORT.md").write_text(_report_markdown(summaries, ratios, stability, decision_payload), encoding="utf-8")
    _json(args.output / "PROCESS_CLEANUP_AUDIT.json", {"single_process":True,"model_load_count":1,"root_prefill_count":1,"root_template_mutated":False})
    _progress(args,"SUMMARY_WRITTEN",physical_batch=1); _progress(args,"BENCHMARK_COMPLETE",physical_batch=1)
    _json(args.output / "HASHES.json", {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.output.glob("*") if p.is_file() and p.name!="HASHES.json"})


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--model-path",type=Path,required=True); parser.add_argument("--challenge",type=Path,required=True); parser.add_argument("--native-config-dir",type=Path,required=True); parser.add_argument("--task-id",default="d59b0160"); parser.add_argument("--output-index",type=int,default=0); parser.add_argument("--depth",type=int,default=24); parser.add_argument("--budget",type=int,default=128); parser.add_argument("--benchmark-model-mode",default="BASE_MODEL_ONLY"); args=parser.parse_args(); args.started=time.monotonic()
    try:
        run(args)
    except BaseException as exc:
        failure = {"exception":repr(exc),"traceback":traceback.format_exc()}
        _json(args.output / "FAILURE.json", failure)
        if any(marker in repr(exc) for marker in ("TIME_GATE", "MODEL_READY_TIME_GATE")):
            _json(args.output / "TIME_GATE_FAILURE.json", {"experiment": EXPERIMENT, **failure})
        raise

if __name__=="__main__": main()
