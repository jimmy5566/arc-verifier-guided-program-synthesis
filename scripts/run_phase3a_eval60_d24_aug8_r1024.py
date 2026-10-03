#!/usr/bin/env python3
"""Target-blind Phase 3 Eval60 canonical-AUG8 R1024 generation.

This controller intentionally owns no model forward.  Each output is delegated
to the frozen fresh-process Core worker and is independently hash-verified
before a later invocation can reuse it.  It never accepts a solutions path and
therefore cannot perform Gold scoring.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset  # noqa: E402
from inference.nvarc_native import checkpoint_native_tokenizer  # noqa: E402
from inference.root_length_memory_profile import select_memory_profile  # noqa: E402
from scripts.run_clean_hf_parallel_regret_dfs_v1 import _assert_challenge_only  # noqa: E402
from scripts.run_non_s_rolling_resident_v1 import _adapter_identity, _native_prompt_record  # noqa: E402
from scripts.run_real_project_aug16_dynamic_b16_pilot_v1 import _load_aug16  # noqa: E402
from scripts.run_ttt24_aug8_r1024_core_v1 import (  # noqa: E402
    AUG8,
    _atomic_csv,
    _atomic_json,
    _canonical,
    _policy,
    _read,
    _safe,
    _sha_file,
    _sha_value,
    _subset_file,
    _verify,
    _worker,
)


DEPTH = 24
EXPERIMENT = "PHASE3A_EVAL60_D24_AUG8_R1024_V1"
EXPECTED_RUNTIME = {
    "torch": "2.8.0+cu128",
    "torch_cuda": "12.8",
    "transformers": "4.55.4",
    "peft": "0.17.1",
}
GENERATION_LEDGER_EXCLUDED = {
    "GENERATION_HASHES.json", "GENERATION_HASH_VERIFICATION.json",
    "HASHES.json", "HASH_VERIFICATION.json", "GENERATION_FREEZE.json",
    "COMPACT_ARCHIVE_FREEZE.json", "DECISION.json", "REPORT.md",
}


def _configure_depth(depth: int) -> None:
    """Select only the pre-registered phase identity; Core execution is shared."""
    global DEPTH, EXPERIMENT
    if depth not in {24, 48}:
        raise ValueError(f"unsupported Phase 3 depth: {depth}")
    DEPTH = int(depth)
    phase = "PHASE3A" if DEPTH == 24 else "PHASE3B"
    EXPERIMENT = f"{phase}_EVAL60_D{DEPTH}_AUG8_R1024_V1"


def _other_depth_status() -> str:
    return "NOT_STARTED" if DEPTH == 24 else "SEPARATE"


def _head() -> str:
    result = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, capture_output=True, check=True)
    return result.stdout.strip()


def _task_output(output_id: str) -> tuple[str, int]:
    task_id, marker, output = str(output_id).partition(":o")
    if marker != ":o" or not task_id or not output.isdecimal():
        raise RuntimeError(f"malformed output id: {output_id}")
    return task_id, int(output)


def _runtime_probe(worker_python: Path) -> dict[str, Any]:
    if not worker_python.is_file():
        raise FileNotFoundError(f"WORKER_PYTHON_MISSING:{worker_python}")
    code = (
        "import json,sys,torch,transformers,peft;"
        "print(json.dumps({'sys_executable':sys.executable,'torch':torch.__version__,"
        "'torch_cuda':str(torch.version.cuda),'transformers':transformers.__version__,"
        "'peft':peft.__version__,'cuda_available':torch.cuda.is_available()},sort_keys=True))"
    )
    result = subprocess.run([str(worker_python), "-c", code], text=True, capture_output=True, check=False)
    if result.returncode:
        raise RuntimeError(f"WORKER_RUNTIME_IMPORT_FAIL:{result.stderr.strip()[-500:]}")
    try:
        value = json.loads([line for line in result.stdout.splitlines() if line.strip()][-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise RuntimeError("WORKER_RUNTIME_PROBE_PARSE_FAIL") from error
    if {key: value.get(key) for key in EXPECTED_RUNTIME} != EXPECTED_RUNTIME or value.get("cuda_available") is not True:
        raise RuntimeError(f"WORKER_RUNTIME_VERSION_DRIFT:{_canonical(value)}")
    if Path(str(value["sys_executable"])).resolve() != worker_python.resolve():
        raise RuntimeError("WORKER_RUNTIME_EXECUTABLE_DRIFT")
    return value


def _profile_name(root_length: int) -> str:
    profile = select_memory_profile(int(root_length)).name
    if profile == "PROFILE_L":
        return "PROFILE_L_LOW" if int(root_length) <= 4096 else "PROFILE_L_HIGH"
    if profile == "PROFILE_XL":
        return "PROFILE_XL_CONSERVATIVE"
    if profile in {"PROFILE_S", "PROFILE_M"}:
        return profile
    raise RuntimeError(f"FROZEN_PROFILE_POLICY_UNAVAILABLE:{profile}:{root_length}")


def _challenge_outputs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _assert_challenge_only(args.challenge)
    registry = _read(args.cohort_registry)
    raw = _read(args.challenge)
    if (registry.get("cohort_id") != "eval60" or int(registry.get("task_count", -1)) != 60
            or _sha_file(args.challenge) != registry.get("source_challenge_sha256")):
        raise RuntimeError("EVAL60_COHORT_OR_CHALLENGE_IDENTITY_FAIL")
    task_ids = [str(item) for item in registry.get("task_ids", [])]
    if len(task_ids) != 60 or len(set(task_ids)) != 60 or _sha_value(task_ids) != registry.get("task_ids_sha256"):
        raise RuntimeError("EVAL60_TASK_SET_FAIL")
    rows: list[dict[str, Any]] = []
    for task_id in task_ids:
        task = raw.get(task_id)
        expected = registry.get("tasks", {}).get(task_id)
        if not isinstance(task, dict) or not isinstance(expected, dict):
            raise RuntimeError(f"EVAL60_TASK_MISSING:{task_id}")
        if _sha_value(task) != expected.get("task_sha256"):
            raise RuntimeError(f"EVAL60_TASK_SHA_FAIL:{task_id}")
        tests = task.get("test")
        expected_tests = expected.get("test_index_structure")
        if not isinstance(tests, list) or not isinstance(expected_tests, list) or len(tests) != len(expected_tests):
            raise RuntimeError(f"EVAL60_TEST_STRUCTURE_FAIL:{task_id}")
        for index, (test, identity) in enumerate(zip(tests, expected_tests, strict=True)):
            if int(identity.get("test_index", -1)) != index or _sha_value(test.get("input")) != identity.get("input_sha256"):
                raise RuntimeError(f"EVAL60_TEST_INPUT_SHA_FAIL:{task_id}:o{index}")
            rows.append({"output_id": f"{task_id}:o{index}", "task_id": task_id, "output_index": index})
    if len(rows) != 89 or len({row["output_id"] for row in rows}) != 89:
        raise RuntimeError(f"EVAL60_EXPECTED_89_OUTPUTS_GOT:{len(rows)}")
    return rows, {"registry_sha256": _sha_file(args.cohort_registry), "challenge_sha256": _sha_file(args.challenge),
                  "task_count": len(task_ids), "output_count": len(rows), "task_ids_sha256": _sha_value(task_ids)}


def _adapter_manifest(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    with args.adapter_manifest.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    selected = {str(row["task_id"]): dict(row) for row in rows if str(row.get("depth")) == str(DEPTH)}
    if len(selected) != 60:
        raise RuntimeError(f"D{DEPTH}_ADAPTER_MANIFEST_EXPECTED_60_GOT:{len(selected)}")
    return selected


def _make_cohort(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, cohort_identity = _challenge_outputs(args)
    manifest = _adapter_manifest(args)
    tasks = load_dataset(args.challenge)
    candidates = _load_aug16(args.candidate_pool, args.aug8_ids)
    candidate_ids = tuple(str(candidate["candidate_id"]) for candidate in candidates)
    if candidate_ids != AUG8:
        raise RuntimeError("CANONICAL_AUG8_IDENTITY_FAIL")
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(args.model_path, args.native_config_dir)
    adapters: dict[str, Any] = {}
    prepared: list[dict[str, Any]] = []
    for row in rows:
        task_id = row["task_id"]
        adapter = args.adapter_root / task_id / f"depth_{DEPTH:03d}"
        identity = _adapter_identity(adapter)
        expected = manifest.get(task_id)
        if identity.get("status") != "PASS" or expected is None:
            raise RuntimeError(f"D{DEPTH}_ADAPTER_IDENTITY_FAIL:{task_id}")
        if identity.get("adapter_sha256") != expected.get("sha256") or str(adapter / "adapter_model.safetensors") != expected.get("global_path"):
            raise RuntimeError(f"D{DEPTH}_ADAPTER_MANIFEST_MISMATCH:{task_id}")
        adapters.setdefault(task_id, {"adapter_path": str(adapter), "identity": identity, "manifest_sha256": expected["sha256"]})
        task = tasks.get(task_id)
        if task is None or int(row["output_index"]) >= len(task.test):
            raise RuntimeError(f"PUBLIC_OUTPUT_UNAVAILABLE:{row['output_id']}")
        prompts = [_native_prompt_record(tokenizer=tokenizer, task=task, output_index=int(row["output_index"]), candidate=candidate)[1]
                   for candidate in candidates]
        root_lengths = [int(prompt["prompt_token_length"]) for prompt in prompts]
        maximum = max(root_lengths)
        prepared.append({**row, "profile": _profile_name(maximum), "root_length_min": min(root_lengths),
                         "root_length_max": maximum, "adapter_path": str(adapter), "adapter_identity": identity,
                         "canonical_aug8_prompt_sha256": {prompt["augmentation_id"]: prompt["prompt_sha256"] for prompt in prompts}})
    broad_counts = Counter("PROFILE_L" if row["profile"].startswith("PROFILE_L") else "PROFILE_XL" if row["profile"].startswith("PROFILE_XL") else row["profile"] for row in prepared)
    return prepared, {"status": "PASS", "cohort": cohort_identity, "candidate_pool_sha256": _sha_file(args.candidate_pool),
                      "aug8_ids_sha256": _sha_file(args.aug8_ids), "tokenizer_identity": tokenizer_identity,
                      "adapter_manifest_sha256": _sha_file(args.adapter_manifest), "adapter_identities": adapters,
                      "profile_counts": {name: int(broad_counts.get(name, 0)) for name in ("PROFILE_S", "PROFILE_M", "PROFILE_L", "PROFILE_XL")}}


def _existing_output_gate(run: Path, selected: dict[str, Any]) -> dict[str, Any] | None:
    safe = _safe(selected["output_id"])
    required = [run / "RAW_OUTPUTS" / f"{safe}.json", run / "OUTPUT_CHECKPOINTS" / f"{safe}.json",
                run / "EOS_EVENTS" / f"{safe}.jsonl.gz", run / "OUTPUT_RECEIPTS" / f"{safe}.json",
                run / "OUTPUT_HASHES" / f"{safe}.json", run / "OUTPUT_HASH_VERIFICATION" / f"{safe}.json"]
    present = [path.exists() for path in required]
    if not any(present):
        return None
    if not all(present):
        raise RuntimeError(f"UNVERIFIED_PARTIAL_OUTPUT_REFUSED:{selected['output_id']}")
    receipt = _read(required[3])
    ledger = _read(required[4])
    verification = _verify(run, required[4])
    if receipt.get("status") != "COMPLETE" or verification.get("status") != "PASS" or _read(required[5]).get("status") != "PASS":
        raise RuntimeError(f"OUTPUT_HASH_REUSE_REFUSED:{selected['output_id']}")
    if set(ledger.get("files", {})) != {str(path.relative_to(run)) for path in required[:4]}:
        raise RuntimeError(f"OUTPUT_LEDGER_SCOPE_FAIL:{selected['output_id']}")
    return {"output_id": selected["output_id"], "resumed": True, "hash_checked": len(verification["checked"])}


def _freeze_output(run: Path, selected: dict[str, Any]) -> dict[str, Any]:
    safe = _safe(selected["output_id"])
    files = [run / "RAW_OUTPUTS" / f"{safe}.json", run / "OUTPUT_CHECKPOINTS" / f"{safe}.json",
             run / "EOS_EVENTS" / f"{safe}.jsonl.gz", run / "OUTPUT_RECEIPTS" / f"{safe}.json"]
    if any(not path.is_file() for path in files) or _read(files[-1]).get("status") != "COMPLETE":
        raise RuntimeError(f"OUTPUT_ATOMIC_ARTIFACT_FAIL:{selected['output_id']}")
    ledger = {"output_id": selected["output_id"], "files": {str(path.relative_to(run)): _sha_file(path) for path in files}}
    ledger_path = run / "OUTPUT_HASHES" / f"{safe}.json"
    _atomic_json(ledger_path, ledger)
    verification = _verify(run, ledger_path)
    _atomic_json(run / "OUTPUT_HASH_VERIFICATION" / f"{safe}.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"OUTPUT_HASH_FREEZE_FAIL:{selected['output_id']}")
    return {"output_id": selected["output_id"], "resumed": False, "hash_checked": len(verification["checked"])}


def _write_progress(run: Path, completed: list[dict[str, Any]], expected: int) -> None:
    _atomic_json(run / "OUTPUT_PROGRESS.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False,
                                                   "completed_outputs": len(completed), "expected_outputs": expected,
                                                   "output_ids": [row["output_id"] for row in completed]})


def _run_all(args: argparse.Namespace, run: Path, cohort: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    attempts: list[dict[str, Any]] = []
    receipt_path = run / "OOM_FALLBACK_RECEIPTS.csv"
    for selected in cohort:
        reused = _existing_output_gate(run, selected)
        if reused is not None:
            completed.append(reused); _write_progress(run, completed, len(cohort)); continue
        success = False
        for attempt, config in enumerate(_policy(policy, selected["profile"])):
            code, receipt = _worker(args, run, selected, attempt, config, depth=DEPTH, experiment=EXPERIMENT)
            receipt["fallback_used"] = attempt > 0
            attempts.append(receipt)
            _atomic_csv(receipt_path, attempts, attempts[0].keys())
            if code == 0:
                completed.append(_freeze_output(run, selected)); _write_progress(run, completed, len(cohort)); success = True; break
            if code != 2:
                raise RuntimeError(f"WORKER_NON_OOM_FAIL:{selected['output_id']}:{receipt['log']}")
        if not success:
            raise RuntimeError(f"FROZEN_OOM_FALLBACK_EXHAUSTED:{selected['output_id']}")
    return completed


def _ledger(run: Path, name: str, *, include_raw: bool) -> dict[str, Any]:
    files: list[Path] = []
    for path in run.rglob("*"):
        if not path.is_file() or path.name in GENERATION_LEDGER_EXCLUDED or "WORKER_LOGS" in path.parts:
            continue
        if not include_raw and any(part in {"RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS"} for part in path.parts):
            continue
        files.append(path)
    value = {str(path.relative_to(run)): _sha_file(path) for path in sorted(files)}
    payload = {"files": value, "includes_raw": include_raw}
    _atomic_json(run / name, payload)
    return payload


def _runtime_summaries(run: Path, cohort: list[dict[str, Any]], attempts: list[dict[str, Any]]) -> dict[str, Any]:
    output_rows: list[dict[str, Any]] = []
    histogram: Counter[tuple[str, int]] = Counter()
    grouped: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for selected in cohort:
        raw = _read(run / "RAW_OUTPUTS" / f"{_safe(selected['output_id'])}.json")
        scheduler = raw["scheduler"]
        row = {"output_id": selected["output_id"], "profile": selected["profile"], "root_length_min": selected["root_length_min"],
               "root_length_max": selected["root_length_max"], "wall_seconds": raw["wall_seconds"],
               "logical_advances": scheduler["logical_advances"], "logical_nodes_per_second": scheduler["logical_advances"] / raw["wall_seconds"] if raw["wall_seconds"] else None,
               "mean_effective_batch": scheduler["mean_effective_batch"], "physical_forwards": scheduler["physical_forwards"],
               "resident_capacity": raw["profile_configuration"]["resident_capacity"], "physical_batch_ceiling": raw["profile_configuration"]["physical_batch_ceiling"],
               "attempt_index": raw["attempt_index"], "fallback_used": bool(raw["attempt_index"]),
               "peak_allocated_bytes": raw["memory"]["peak_allocated_bytes"], "peak_reserved_bytes": raw["memory"]["peak_reserved_bytes"]}
        output_rows.append(row); grouped[selected["profile"]].append(row)
        for width, count in scheduler["physical_batch_histogram"].items(): histogram[(selected["profile"], int(width))] += int(count)
    _atomic_csv(run / "OUTPUT_RUNTIME.csv", output_rows, output_rows[0].keys())
    summary_rows: list[dict[str, Any]] = []
    for profile, rows in sorted(grouped.items()):
        wall = sum(float(row["wall_seconds"]) for row in rows); logical = sum(int(row["logical_advances"]) for row in rows)
        summary_rows.append({"profile": profile, "output_count": len(rows), "total_wall_seconds": wall,
                             "mean_wall_seconds": wall / len(rows), "logical_advances": logical,
                             "logical_nodes_per_second": logical / wall if wall else None,
                             "mean_effective_batch": sum(float(row["mean_effective_batch"]) for row in rows) / len(rows),
                             "fallback_count": sum(int(row["fallback_used"]) for row in rows)})
    _atomic_csv(run / "PROFILE_RUNTIME.csv", summary_rows, summary_rows[0].keys())
    histogram_rows = [{"profile": profile, "physical_batch": width, "physical_forwards": count}
                      for (profile, width), count in sorted(histogram.items())]
    _atomic_csv(run / "PHYSICAL_BATCH_HISTOGRAM.csv", histogram_rows, ["profile", "physical_batch", "physical_forwards"])
    return {"output_runtime": output_rows, "profile_runtime": summary_rows,
            "total_gpu_wall_seconds": sum(float(row["wall_seconds"]) for row in output_rows),
            "fallback_count": sum(int(row["fallback_used"]) for row in output_rows),
            "attempt_count": len(attempts)}


def _freeze_generation(run: Path, cohort: list[dict[str, Any]], runtime: dict[str, Any]) -> dict[str, Any]:
    expected = len(cohort)
    required = ("RAW_OUTPUTS", "OUTPUT_CHECKPOINTS", "EOS_EVENTS", "OUTPUT_RECEIPTS", "OUTPUT_HASHES", "OUTPUT_HASH_VERIFICATION")
    actual = {name: len(list((run / name).glob("*.json*"))) for name in required}
    if any(actual[name] != expected for name in required):
        raise RuntimeError(f"PHASE3_D{DEPTH}_INCOMPLETE_GENERATION:{_canonical(actual)}")
    for selected in cohort:
        if _existing_output_gate(run, selected) is None:
            raise RuntimeError(f"PHASE3_D{DEPTH}_OUTPUT_REVERIFY_FAIL:{selected['output_id']}")
    manifest = {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "completed_outputs": expected,
                "d24_generation": "COMPLETE" if DEPTH == 24 else _other_depth_status(),
                "d48_generation": "COMPLETE" if DEPTH == 48 else _other_depth_status(),
                "gold_scoring": "DEFERRED_UNTIL_D24_AND_D48_BOTH_FROZEN", "runtime": runtime}
    _atomic_json(run / "GENERATION_MANIFEST.json", manifest)
    ledger = _ledger(run, "GENERATION_HASHES.json", include_raw=True)
    verification = _verify(run, run / "GENERATION_HASHES.json")
    _atomic_json(run / "GENERATION_HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"PHASE3_D{DEPTH}_GENERATION_HASH_FAIL")
    _atomic_json(run / "GENERATION_FREEZE.json", {"experiment": EXPERIMENT, "status": "FROZEN", "target_blind": True,
                                                     "gold_loaded": False, "raw_count": expected, "ledger_sha256": _sha_value(ledger)})
    return {"ledger": ledger, "verification": verification, "manifest": manifest}


def _finalize(run: Path, cohort: list[dict[str, Any]], runtime: dict[str, Any], generation: dict[str, Any]) -> None:
    decision = {"experiment": EXPERIMENT,
                "PHASE3_D24_GENERATION": "COMPLETE" if DEPTH == 24 else _other_depth_status(),
                "PHASE3_D48_GENERATION": "COMPLETE" if DEPTH == 48 else _other_depth_status(),
                "GOLD_SCORING": "DEFERRED_UNTIL_D24_AND_D48_BOTH_FROZEN", "gold_loaded": False,
                "completed_outputs": len(cohort), "generation_hash_status": generation["verification"]["status"],
                "total_gpu_wall_seconds": runtime["total_gpu_wall_seconds"], "oom_fallback_count": runtime["fallback_count"]}
    _atomic_json(run / "DECISION.json", decision)
    report = "\n".join([f"# {EXPERIMENT}", "", f"Target-blind D{DEPTH}-only Eval60 generation is frozen.", "",
                          f"- Outputs: {len(cohort)}/89", f"- Other depth: {_other_depth_status()}", "- Gold scoring: DEFERRED_UNTIL_D24_AND_D48_BOTH_FROZEN",
                          f"- Generation hashes: {generation['verification']['status']}", f"- Total GPU wall seconds: {runtime['total_gpu_wall_seconds']:.3f}",
                          f"- OOM/fallback count: {runtime['fallback_count']}", ""])
    (run / "REPORT.md").write_text(report, encoding="utf-8")
    compact = _ledger(run, "HASHES.json", include_raw=False)
    verification = _verify(run, run / "HASHES.json")
    _atomic_json(run / "HASH_VERIFICATION.json", verification)
    if verification["status"] != "PASS":
        raise RuntimeError(f"PHASE3_D{DEPTH}_COMPACT_HASH_FAIL")
    _atomic_json(run / "COMPACT_ARCHIVE_FREEZE.json", {"status": "FROZEN", "compact_ledger_sha256": _sha_value(compact),
                                                          "hash_status": verification["status"], "gold_loaded": False})


def _preflight(args: argparse.Namespace, source_commit: str) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    required = {"model_path": args.model_path, "challenge": args.challenge, "native_config_dir": args.native_config_dir,
                "candidate_pool": args.candidate_pool, "adapter_root": args.adapter_root, "adapter_manifest": args.adapter_manifest,
                "coarse_policy": args.coarse_policy, "cohort_registry": args.cohort_registry, "runtime_ready": args.runtime_ready}
    missing = [name for name, path in required.items() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"PHASE3_D{DEPTH}_REQUIRED_PATHS_MISSING:{','.join(missing)}")
    ready = _read(args.runtime_ready)
    if ready.get("event") != "READY_FOR_ARC2_AMPERE_V2" or ready.get("source_commit") != source_commit:
        raise RuntimeError("RUNTIME_READY_IDENTITY_FAIL")
    runtime = _runtime_probe(args.worker_python)
    cohort, identities = _make_cohort(args)
    policy = _read(args.coarse_policy)
    if policy.get("target_blind") is not True or any(row["profile"] not in policy.get("profiles", {}) for row in cohort):
        raise RuntimeError("FROZEN_COARSE_POLICY_FAIL")
    return cohort, policy, {"status": "PASS", "source_commit": source_commit, "target_blind": True, "gold_loaded": False,
                              "gold_data_loaded": False, "worker_python": str(args.worker_python), "worker_runtime": runtime,
                              "runtime_ready_sha256": _sha_file(args.runtime_ready), "model_manifest_sha256": _sha_file(args.model_path / "model_manifest.json"),
                              "coarse_policy_sha256": _sha_file(args.coarse_policy), **identities}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--challenge", type=Path, required=True)
    parser.add_argument("--native-config-dir", type=Path, required=True)
    parser.add_argument("--candidate-pool", type=Path, required=True)
    parser.add_argument("--adapter-root", type=Path, required=True)
    parser.add_argument("--adapter-manifest", type=Path, required=True)
    parser.add_argument("--coarse-policy", type=Path, required=True)
    parser.add_argument("--cohort-registry", type=Path, required=True)
    parser.add_argument("--runtime-ready", type=Path, required=True)
    parser.add_argument("--worker-python", type=Path, required=True)
    parser.add_argument("--ttt-depth", type=int, choices=(24, 48), default=24)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    _configure_depth(args.ttt_depth)
    source_commit = _head()
    if args.output.exists():
        if not (args.output / "CONTRACT.json").is_file():
            raise RuntimeError(f"OUTPUT_DIR_EXISTS_WITHOUT_RESUMABLE_CONTRACT:{args.output}")
        contract = _read(args.output / "CONTRACT.json")
        if contract.get("experiment") != EXPERIMENT or contract.get("source_commit") != source_commit:
            raise RuntimeError("RESUME_CONTRACT_IDENTITY_FAIL")
        cohort = _read(args.output / "RUN_COHORT.json")["outputs"]
        policy = _read(args.coarse_policy)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        args.aug8_ids = _subset_file(args.output)
        cohort, policy, preflight = _preflight(args, source_commit)
        _atomic_json(args.output / "PREFLIGHT.json", preflight)
        _atomic_json(args.output / "ADAPTER_IDENTITIES.json", preflight["adapter_identities"])
        _atomic_json(args.output / "RUN_COHORT.json", {"experiment": EXPERIMENT, "target_blind": True, "gold_loaded": False, "outputs": cohort})
        _atomic_json(args.output / "CONTRACT.json", {"experiment": EXPERIMENT, "source_commit": source_commit,
            "target_blind": True, "gold_loaded": False, "cohort": "Eval60: 60 tasks / 89 outputs", "ttt_depth": DEPTH,
            "augmentation_ids": list(AUG8), "max_expanded_nodes": 1024, "decoder": "CUMULATIVE_REGRET_r=4.00",
            "max_new_tokens": 931, "candidate_cap": 32, "frontier_floor": 1, "eos": 15,
            "admission": "root_aware", "cache": "ChunkedDynamicCache / valid_length rollback",
            "profile_policy": "existing S/M/L/XL frozen coarse policy", "worker_python": str(args.worker_python),
            "worker_runtime": preflight["worker_runtime"], "generation_scope": f"D{DEPTH}_ONLY",
            "d24": "COMPLETE" if DEPTH == 24 else _other_depth_status(),
            "d48": "COMPLETE" if DEPTH == 48 else _other_depth_status(),
            "gold_scoring": "DEFERRED_UNTIL_D24_AND_D48_BOTH_FROZEN"})
    args.aug8_ids = args.output / "AUG8_IDS.json"
    completed = _run_all(args, args.output, cohort, policy)
    attempts = []
    receipt_path = args.output / "OOM_FALLBACK_RECEIPTS.csv"
    if receipt_path.exists():
        with receipt_path.open(encoding="utf-8", newline="") as handle:
            attempts = list(csv.DictReader(handle))
    runtime = _runtime_summaries(args.output, cohort, attempts)
    generation = _freeze_generation(args.output, cohort, runtime)
    _finalize(args.output, cohort, runtime, generation)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
