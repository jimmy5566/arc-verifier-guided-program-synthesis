#!/usr/bin/env python3
"""Post-freeze exact B-selector evaluation for the Phase-3 R1024 archives.

The ``prepare`` command is deliberately target-blind.  It freezes the ORC
cohort membership from an already-frozen score receipt, extracts only existing
R1024 candidates, validates the historical adapters, and teacher-forces every
candidate under the eight fixed reversible views.  ``evaluate`` refuses to open
solutions until that frozen ledger verifies exactly.

No path in this module performs DFS, decoding, sampling, adaptation, or any
other candidate generation.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any, Iterable


# Match the established Eval60 launchers: serializing only the Inductor compile
# pool avoids a 32-worker CPU oversubscription at the first long-context forward.
# This influences compilation resource use only, never the frozen model or score.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from arc.io import load_dataset
from inference.hf_peft_backend import load_hf_peft_inference
from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer, native_messages, serialize_grid
from inference.nvarc_native_augmentation import NativeAugmentation


DEPTHS = (24, 48)
SCORING_VIEWS = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
CANONICAL_AUG8 = (
    "geom=identity__color=id__order=canonical",
    "geom=flip_ud__color=id__order=canonical",
    "geom=transpose__color=id__order=canonical",
    "geom=anti_transpose__color=id__order=canonical",
    "geom=rot90__color=id__order=canonical",
    "geom=rot180__color=id__order=canonical",
    "geom=rot270__color=id__order=canonical",
    "geom=flip_lr__color=id__order=canonical",
)
EXPECTED_RUNTIME = {"torch": "2.8.0+cu128", "torch_cuda": "12.8", "transformers": "4.55.4", "peft": "0.17.1"}
EXPECTED_GOLD_SHA256 = "84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1"
NEW7 = ("20270e3b:o0", "332f06d7:o0", "7491f3cf:o0", "78332cb0:o0", "97d7923e:o0", "981571dc:o0", "de809cff:o0")


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha_value(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False, suffix=".tmp") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_jsonl_gz(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as raw:
        temporary = Path(raw.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical(row) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False, suffix=".tmp") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv_gz(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False, suffix=".tmp") as raw:
        temporary = Path(raw.name)
    try:
        with gzip.open(temporary, "wt", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: canonical(value) if isinstance(value, (dict, list)) else value for key, value in row.items()})
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def oid_parts(output_id: str) -> tuple[str, int]:
    task, marker, index = output_id.partition(":o")
    if marker != ":o" or not task or not index.isdecimal():
        raise ValueError(f"invalid output id: {output_id}")
    return task, int(index)


def safe_id(output_id: str) -> str:
    return output_id.replace(":", "_")


def bool_cell(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def source_score_cohort(score_csv: Path) -> tuple[list[dict[str, str]], dict[str, Any]]:
    rows = list(csv.DictReader(score_csv.open(encoding="utf-8", newline="")))
    selected = [row for row in rows if bool_cell(row.get("ORC_UNION"))]
    outputs = [str(row["output_id"]) for row in selected]
    tasks = sorted({str(row["task_id"]) for row in selected})
    if len(rows) != 89 or len(selected) != 35 or len(set(outputs)) != 35 or len(tasks) != 27:
        raise RuntimeError(f"ORC_COHORT_EXPECTED_35_OUTPUTS_27_TASKS_GOT:{len(rows)}:{len(selected)}:{len(tasks)}")
    return selected, {
        "status": "PASS",
        "statement": "COHORT_IS_POSTFREEZE_GOLD_DERIVED_ORACLE_HIT_DIAGNOSTIC",
        "output_ids": outputs,
        "task_ids": tasks,
        "output_count": len(outputs),
        "task_count": len(tasks),
        "source_score_artifact_sha256": sha256(score_csv),
        "source_score_commit": _artifact_commit(score_csv.parent),
    }


def _artifact_commit(path: Path) -> str:
    import subprocess
    relative_path = path.resolve().relative_to(ROOT.resolve())
    result = subprocess.run(
        ["git", "-C", str(ROOT), "log", "-1", "--format=%H", "--", str(relative_path)],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "UNKNOWN"


def archive_contract(archive: Path, depth: int, outputs: list[str]) -> dict[str, Any]:
    contract = read_json(archive / "CONTRACT.json")
    verification = read_json(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json")
    if int(contract.get("ttt_depth", -1)) != depth or contract.get("gold_loaded") is not False:
        raise RuntimeError(f"ARCHIVE_CONTRACT_FAIL_D{depth}")
    if verification.get("status") != "PASS":
        raise RuntimeError(f"ARCHIVE_HASH_FAIL_D{depth}")
    for output_id in outputs:
        path = archive / "tables" / "candidate_index" / f"{safe_id(output_id)}.jsonl.gz"
        if not path.is_file():
            raise FileNotFoundError(path)
    return {
        "archive": str(archive), "archive_contract_sha256": sha256(archive / "CONTRACT.json"),
        "archive_verification_sha256": sha256(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json"),
        "archive_hash_status": verification["status"], "depth": depth,
    }


def build_source_candidates(archive: Path, depth: int, output_id: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    index_path = archive / "tables" / "candidate_index" / f"{safe_id(output_id)}.jsonl.gz"
    rows = [row for row in read_jsonl_gz(index_path) if int(row.get("checkpoint", -1)) == 1024]
    if not rows:
        raise RuntimeError(f"NO_R1024_CANDIDATES:{output_id}:d{depth}")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("canonical_grid") is None or int(row.get("depth", depth)) != depth:
            continue
        groups[canonical(row["canonical_grid"])].append(row)
    if not groups:
        raise RuntimeError(f"NO_VALID_R1024_GRID:{output_id}:d{depth}")
    result: list[dict[str, Any]] = []
    aug_order = {name: index for index, name in enumerate(CANONICAL_AUG8)}
    for grid_key, members in groups.items():
        for row in members:
            if str(row["augmentation_id"]) not in aug_order:
                raise RuntimeError(f"NONCANONICAL_AUGMENTATION:{row['augmentation_id']}")
        representative = min(members, key=lambda row: (
            aug_order[str(row["augmentation_id"])], int(row.get("candidate_completion_index", 10**9)),
            int(row.get("nodes_expanded_so_far") or 10**18), str(row.get("candidate_id", "")), grid_key,
        ))
        augmentations = sorted({str(row["augmentation_id"]) for row in members}, key=aug_order.__getitem__)
        result.append({
            "task_id": oid_parts(output_id)[0], "output_id": output_id, "depth": depth,
            "canonical_grid": representative["canonical_grid"], "grid_key": grid_key,
            "supporting_augmentation_ids": augmentations, "support_count": len(augmentations),
            "original_candidate_ids": sorted({str(row.get("candidate_id")) for row in members}),
            "terminal_node_ids": sorted({int(row["terminal_node_id"]) for row in members if row.get("terminal_node_id") is not None}),
            "candidate_completion_indices": sorted({int(row.get("candidate_completion_index", 0)) for row in members}),
            "earliest_candidate_completion_index": int(representative.get("candidate_completion_index", 0)),
            "earliest_nodes_expanded_so_far": int(representative.get("nodes_expanded_so_far") or 10**18),
            "stable_representative": {
                "augmentation_id": str(representative["augmentation_id"]), "candidate_id": str(representative.get("candidate_id")),
                "candidate_completion_index": int(representative.get("candidate_completion_index", 0)),
                "nodes_expanded_so_far": int(representative.get("nodes_expanded_so_far") or 10**18),
            },
            "occurrence_count": len(members), "candidate_index_sha256": sha256(index_path),
        })
    return sorted(result, key=lambda row: row["grid_key"]), {"candidate_index_sha256": sha256(index_path), "occurrence_count": len(rows), "unique_grid_count": len(result)}


def adapter_preflight(adapter_root: Path, d24_manifest: Path, d48_manifest: Path, task_ids: list[str]) -> dict[str, Any]:
    entries: dict[int, dict[str, dict[str, str]]] = {}
    for depth, manifest_path in ((24, d24_manifest), (48, d48_manifest)):
        with manifest_path.open(encoding="utf-8", newline="") as handle:
            rows = [dict(row) for row in csv.DictReader(handle) if int(row["depth"]) == depth]
        entries[depth] = {str(row["task_id"]): row for row in rows}
    report: dict[str, Any] = {"status": "PASS", "adapter_root": str(adapter_root), "tasks": {}, "manifest_sha256": {"d24": sha256(d24_manifest), "d48": sha256(d48_manifest)}}
    for task_id in task_ids:
        report["tasks"][task_id] = {}
        for depth in DEPTHS:
            expected = entries[depth].get(task_id)
            if expected is None:
                raise RuntimeError(f"ADAPTER_MANIFEST_MISSING:{task_id}:d{depth}")
            path = adapter_root / task_id / f"depth_{depth:03d}"
            model, config = path / "adapter_model.safetensors", path / "adapter_config.json"
            model_sha = sha256(model) if model.is_file() else None
            config_sha = sha256(config) if config.is_file() else None
            state = {
                "status": "EXACT_HISTORICAL_ADAPTER" if model_sha == expected["adapter_model_sha256"] and config_sha == expected["adapter_config_sha256"] else "MISSING_OR_MISMATCHED",
                "adapter_path": str(path), "adapter_model_sha256_expected": expected["adapter_model_sha256"],
                "adapter_config_sha256_expected": expected["adapter_config_sha256"],
                "adapter_model_sha256_observed": model_sha, "adapter_config_sha256_observed": config_sha,
                "lora_rank": int(expected["lora_rank"]), "lora_alpha": int(expected["lora_alpha"]),
                "target_modules": json.loads(expected["target_modules"]), "depth": depth,
            }
            if state["status"] != "EXACT_HISTORICAL_ADAPTER" or state["lora_rank"] != 256 or state["lora_alpha"] != 32:
                report["status"] = "FAIL"
            report["tasks"][task_id][f"d{depth}"] = state
    if report["status"] != "PASS":
        raise RuntimeError("EXACT_HISTORICAL_ADAPTERS_UNAVAILABLE_RECONSTRUCTION_REQUIRED")
    return report


def runtime_identity(model_path: Path, native_config: Path) -> dict[str, Any]:
    import peft
    import torch
    import transformers
    tokenizer, tokenizer_identity = checkpoint_native_tokenizer(model_path, native_config)
    observed = {"torch": torch.__version__, "torch_cuda": str(torch.version.cuda), "transformers": transformers.__version__, "peft": peft.__version__}
    if observed != EXPECTED_RUNTIME or not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"RUNTIME_IDENTITY_FAIL:{canonical(observed)}")
    model_manifest = model_path / "model_manifest.json"
    return {"status": "PASS", "python": sys.executable, "runtime": observed, "gpu": torch.cuda.get_device_name(0), "bf16": True,
            "model_path": str(model_path), "model_manifest_sha256": sha256(model_manifest) if model_manifest.is_file() else None,
            "native_config": str(native_config), "tokenizer_identity": tokenizer_identity, "tokenizer_length": len(tokenizer)}


def _load_model(model_path: Path, initial_adapter: Path, native_config: Path) -> tuple[Any, NVARCNativeProvider]:
    model, tokenizer, _identity = load_hf_peft_inference(
        model_path=model_path,
        adapter_path=initial_adapter,
        device="cuda:0",
        native_config_dir=native_config,
    )
    provider = NVARCNativeProvider(model_path=model_path, tokenizer_config_dir=native_config, device="cuda:0")
    provider.model, provider.tokenizer = model, tokenizer
    return model, provider


def _set_adapter(model: Any, adapter_path: Path) -> None:
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    state = load_file(str(adapter_path / "adapter_model.safetensors"), device="cpu")
    result = set_peft_model_state_dict(model, state, adapter_name="default")
    if getattr(result, "unexpected_keys", ()):
        raise RuntimeError(f"ADAPTER_LOAD_UNEXPECTED_KEYS:{result.unexpected_keys}")
    model.set_adapter("default")
    model.eval()


def _request_tokens(provider: NVARCNativeProvider, continuation: str) -> int:
    assert provider.tokenizer is not None
    return int(provider.tokenizer(continuation, add_special_tokens=False, return_tensors="pt")["input_ids"].shape[-1]) + 1


def score_requests(provider: NVARCNativeProvider, requests: list[tuple[list[dict[str, str]], str]], context_window: int, batch_size: int) -> tuple[list[float], int]:
    """Exact existing mean-logprob scorer with deterministic OOM size fallback."""
    if not requests:
        return [], 0
    import torch
    current, fallback_count = min(batch_size, len(requests)), 0
    while True:
        try:
            return provider.continuation_log_likelihood_many(requests, context_window=context_window, batch_size=current), fallback_count
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if current == 1:
                raise
            current = max(1, current // 2)
            fallback_count += 1


def score_task_depth(model: Any, provider: NVARCNativeProvider, task: Any, output_id: str, depth: int, rows: list[dict[str, Any]], adapter_path: Path, context_window: int, batch_size: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    _set_adapter(model, adapter_path)
    # This identity was established during adapter preflight.  Keep a single
    # source-local copy in the expensive evidence rows rather than rehashing a
    # several-hundred-MiB adapter for every candidate and scoring view.
    adapter_model_sha256 = sha256(adapter_path / "adapter_model.safetensors")
    _task, output_index = oid_parts(output_id)
    continuations = [serialize_grid(row["canonical_grid"]) for row in rows]
    identity_messages = native_messages(task, output_index)
    original, fallback = score_requests(provider, [(identity_messages, continuation) for continuation in continuations], context_window, batch_size)
    detail_rows: list[dict[str, Any]] = []
    by_grid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for view_name in SCORING_VIEWS:
        view = NativeAugmentation(geometry=view_name)
        transformed_task = view.transform_task(task)
        transformed = [serialize_grid(view.transform_grid(row["canonical_grid"]).astype(int).tolist()) for row in rows]
        scores, used = score_requests(provider, [(native_messages(transformed_task, output_index), continuation) for continuation in transformed], context_window, batch_size)
        fallback += used
        for candidate, continuation, logprob in zip(rows, transformed, scores, strict=True):
            token_count = _request_tokens(provider, continuation)
            record = {"task_id": candidate["task_id"], "output_id": output_id, "depth": depth, "grid_key": candidate["grid_key"],
                      "scoring_view": view_name, "token_count": token_count, "sequence_log_likelihood": float(logprob) * token_count,
                      "sequence_negative_log_likelihood": -float(logprob) * token_count, "mean_token_nll": -float(logprob),
                      "adapter_model_sha256": adapter_model_sha256}
            detail_rows.append(record); by_grid[candidate["grid_key"]].append(record)
    scored: list[dict[str, Any]] = []
    for candidate, original_score in zip(rows, original, strict=True):
        values = by_grid[candidate["grid_key"]]
        if [value["scoring_view"] for value in values] != list(SCORING_VIEWS):
            raise RuntimeError("SCORING_VIEW_COMPLETENESS_FAIL")
        scored.append({**candidate, "original_log_likelihood": float(original_score), "mean_view_nll": fmean(float(value["mean_token_nll"]) for value in values),
                       "view_negative_log_likelihoods": [float(value["mean_token_nll"]) for value in values]})
    return scored, detail_rows, fallback


def representative_key(row: dict[str, Any]) -> tuple[Any, ...]:
    rep = row["stable_representative"]
    return (CANONICAL_AUG8.index(rep["augmentation_id"]), int(rep["candidate_completion_index"]), int(rep["nodes_expanded_so_far"]), str(rep["candidate_id"]), row["grid_key"])


def source_ranking(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda row: (-float(row["b_support_score"]), representative_key(row)))
    return [{**row, "source_rank": index} for index, row in enumerate(ordered, 1)]


def compose_rankings(scored_by_source: dict[tuple[str, int], list[dict[str, Any]]]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_result: dict[str, Any] = {}; rrf_result: dict[str, Any] = {}; attempts: dict[str, Any] = {}
    by_output: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(dict)
    for (output_id, depth), rows in scored_by_source.items():
        enriched = [{**row, "b_support_score": float(row["support_count"]) - float(row["mean_view_nll"])} for row in rows]
        by_output[output_id][depth] = source_ranking(enriched)
    for output_id, sources in sorted(by_output.items()):
        source_result[output_id] = {f"d{depth}": rows for depth, rows in sorted(sources.items())}
        merged: dict[str, dict[str, Any]] = {}
        for depth, rows in sources.items():
            for row in rows:
                entry = merged.setdefault(row["grid_key"], {"grid_key": row["grid_key"], "canonical_grid": row["canonical_grid"], "sources": {}, "representatives": []})
                entry["sources"][f"d{depth}"] = {"rank": row["source_rank"], "support_count": row["support_count"], "mean_view_nll": row["mean_view_nll"], "b_support_score": row["b_support_score"], "original_log_likelihood": row["original_log_likelihood"]}
                entry["representatives"].append(row)
        fused = []
        for entry in merged.values():
            entry["b_rrf"] = sum(1.0 / int(item["rank"]) for item in entry["sources"].values())
            entry["stable_representative"] = min(entry.pop("representatives"), key=representative_key)["stable_representative"]
            fused.append(entry)
        ordered = sorted(fused, key=lambda row: (-float(row["b_rrf"]), representative_key(row)))
        ranked = [{**row, "final_b_rrf_rank": index} for index, row in enumerate(ordered, 1)]
        rrf_result[output_id] = ranked
        selected = ranked[:2]
        if not selected:
            raise RuntimeError(f"EMPTY_UNION_POOL:{output_id}")
        attempts[output_id] = {"output_id": output_id, "attempt_1": selected[0], "attempt_2": selected[1] if len(selected) > 1 else selected[0], "unique_candidate_count": len(ranked), "duplicate_second_attempt": len(selected) == 1}
    return source_result, rrf_result, attempts


def verification(path: Path, ledger: dict[str, str]) -> dict[str, Any]:
    observed = {name: sha256(path / name) for name in ledger if (path / name).is_file()}
    return {"status": "PASS" if observed == ledger else "FAIL", "expected": ledger, "observed": observed}


def prepare(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite selector output: {args.output}")
    output = args.output; output.mkdir(parents=True)
    source_rows, cohort = source_score_cohort(args.source_score_csv)
    output_ids = list(cohort["output_ids"]); task_ids = list(cohort["task_ids"])
    cohort["cohort_sha256"] = sha_value({"output_ids": output_ids, "task_ids": task_ids})
    write_json(output / "B_SELECTOR_COHORT.json", cohort)
    archives = {24: archive_contract(args.d24, 24, output_ids), 48: archive_contract(args.d48, 48, output_ids)}
    adapters = adapter_preflight(args.adapter_root, args.d24 / "ADAPTER_MANIFEST.csv", args.d48 / "ADAPTER_MANIFEST.csv", task_ids)
    write_json(output / "ADAPTER_PREFLIGHT.json", adapters)
    identity = runtime_identity(args.model_path, args.native_config)
    write_json(output / "MODEL_RUNTIME_IDENTITY.json", identity)
    tasks = load_dataset(args.challenge)
    all_candidates: list[dict[str, Any]] = []
    audit_outputs: dict[str, Any] = {}; source_hashes: dict[tuple[str, int], str] = {}
    for output_id in output_ids:
        task_id, output_index = oid_parts(output_id)
        if task_id not in tasks or output_index >= len(tasks[task_id].test):
            raise RuntimeError(f"CHALLENGE_OUTPUT_UNAVAILABLE:{output_id}")
        audit_outputs[output_id] = {}
        for depth, archive in ((24, args.d24), (48, args.d48)):
            candidates, metadata = build_source_candidates(archive, depth, output_id)
            all_candidates.extend(candidates); audit_outputs[output_id][f"d{depth}"] = metadata
            source_hashes[(output_id, depth)] = metadata["candidate_index_sha256"]
    audit = {"status": "PASS", "checkpoint": 1024, "archives": archives, "candidate_generation_performed": False,
             "dfs_rerun": False, "output_count": len(output_ids), "source_local_unique_candidate_counts": {"d24": sum(v["d24"]["unique_grid_count"] for v in audit_outputs.values()), "d48": sum(v["d48"]["unique_grid_count"] for v in audit_outputs.values())},
             "outputs": audit_outputs, "canonical_aug8": list(CANONICAL_AUG8)}
    write_json(output / "SOURCE_POOL_AUDIT.json", audit)
    # The first adapter is only used to construct the fixed PEFT wrapper.  All
    # source states, including this one, are then loaded by exact SHA-bound path.
    first_task = task_ids[0]; model, provider = _load_model(args.model_path, args.adapter_root / first_task / "depth_024", args.native_config)
    scored_by_source: dict[tuple[str, int], list[dict[str, Any]]] = {}; likelihood_rows: list[dict[str, Any]] = []; fallback_count = 0
    try:
        for output_id in output_ids:
            task_id, _ = oid_parts(output_id)
            for depth in DEPTHS:
                candidates = [row for row in all_candidates if row["output_id"] == output_id and row["depth"] == depth]
                adapter = args.adapter_root / task_id / f"depth_{depth:03d}"
                scored, details, fallbacks = score_task_depth(model, provider, tasks[task_id], output_id, depth, candidates, adapter, args.context_window, args.batch_size)
                scored_by_source[(output_id, depth)] = scored; likelihood_rows.extend(details); fallback_count += fallbacks
                print(canonical({"event": "B_SOURCE_SCORED", "output_id": output_id, "depth": depth, "candidate_count": len(scored), "view_calls": len(details), "oom_batch_fallbacks": fallbacks}), flush=True)
    finally:
        import torch
        del provider, model
        torch.cuda.empty_cache()
    source_candidates = [row for rows in scored_by_source.values() for row in rows]
    source_rankings, rrf_rankings, attempts = compose_rankings(scored_by_source)
    write_jsonl_gz(output / "B_VIEW_LIKELIHOOD.jsonl.gz", likelihood_rows)
    write_jsonl_gz(output / "B_SOURCE_CANDIDATES.jsonl.gz", source_candidates)
    write_json(output / "B_SOURCE_RANKINGS.json", source_rankings)
    write_json(output / "B_RRF_RANKINGS.json", rrf_rankings)
    write_json(output / "B_ATTEMPTS_FROZEN.json", attempts)
    evidence_rows = []
    for row in source_candidates:
        evidence_rows.append({"task_id": row["task_id"], "output_id": row["output_id"], "depth": row["depth"], "grid_key": row["grid_key"], "support_count": row["support_count"], "mean_view_nll": row["mean_view_nll"], "original_log_likelihood": row["original_log_likelihood"], "b_support_score": row["support_count"] - row["mean_view_nll"], "supporting_augmentation_ids": row["supporting_augmentation_ids"], "view_negative_log_likelihoods": row["view_negative_log_likelihoods"]})
    write_csv_gz(output / "B_CANDIDATE_EVIDENCE.csv.gz", evidence_rows, list(evidence_rows[0]))
    source_rank_rows = [{"output_id": oid, "depth": depth, "grid_key": row["grid_key"], "source_rank": row["source_rank"], "b_support_score": row["b_support_score"], "support_count": row["support_count"], "mean_view_nll": row["mean_view_nll"]} for oid, sources in source_rankings.items() for label, values in sources.items() for depth in [int(label[1:])] for row in values]
    write_csv(output / "B_SOURCE_RANKS.csv", source_rank_rows, list(source_rank_rows[0]))
    final_rank_rows = [{"output_id": oid, "grid_key": row["grid_key"], "final_b_rrf_rank": row["final_b_rrf_rank"], "b_rrf": row["b_rrf"], "sources": row["sources"]} for oid, rows in rrf_rankings.items() for row in rows]
    write_csv(output / "B_FINAL_RANKS.csv", final_rank_rows, list(final_rank_rows[0]))
    freeze_files = ["B_SELECTOR_COHORT.json", "ADAPTER_PREFLIGHT.json", "MODEL_RUNTIME_IDENTITY.json", "SOURCE_POOL_AUDIT.json", "B_VIEW_LIKELIHOOD.jsonl.gz", "B_SOURCE_CANDIDATES.jsonl.gz", "B_SOURCE_RANKINGS.json", "B_RRF_RANKINGS.json", "B_ATTEMPTS_FROZEN.json", "B_CANDIDATE_EVIDENCE.csv.gz", "B_SOURCE_RANKS.csv", "B_FINAL_RANKS.csv"]
    ledger = {name: sha256(output / name) for name in freeze_files}
    write_json(output / "PRE_GOLD_HASHES.json", ledger)
    receipt = {"status": "PASS", "candidate_generation_performed": False, "DFS_rerun": False, "candidate_pool_modified": False, "target_grid_loaded_before_freeze": False, "selector_uses_output_target": False, "B_attempts_frozen": True, "likelihood_calls": len(likelihood_rows), "candidate_count": len(source_candidates), "oom_batch_fallback_count": fallback_count, "hash_verification": verification(output, ledger)}
    if receipt["hash_verification"]["status"] != "PASS":
        raise RuntimeError("PRE_GOLD_HASH_VERIFICATION_FAIL")
    write_json(output / "PRE_GOLD_FREEZE.json", receipt)


def gold_for_outputs(path: Path, output_ids: list[str]) -> dict[str, Any]:
    if sha256(path) != EXPECTED_GOLD_SHA256:
        raise RuntimeError("GOLD_SHA256_MISMATCH")
    source = read_json(path); result: dict[str, Any] = {}
    for output_id in output_ids:
        task_id, index = oid_parts(output_id); task = source.get(task_id)
        if isinstance(task, list): value = task[index]
        elif isinstance(task, dict):
            value = task.get("test", [])[index]
            if isinstance(value, dict): value = value.get("output")
        else: raise RuntimeError(f"GOLD_TASK_MISSING:{task_id}")
        if not isinstance(value, list): raise RuntimeError(f"GOLD_OUTPUT_MISSING:{output_id}")
        result[output_id] = value
    return result


def historical_retained(greedy_cells: Path) -> set[str]:
    rows = list(csv.DictReader(greedy_cells.open(encoding="utf-8", newline="")))
    retained = {str(row["output_id"]) for row in rows if int(row.get("depth", -1)) in {24, 48} and bool_cell(row.get("exact_gold_hit"))}
    if len(retained) != 28:
        raise RuntimeError(f"HISTORICAL_RETAINED_28_EXPECTED_GOT:{len(retained)}")
    return retained


def evaluate(args: argparse.Namespace) -> None:
    output = args.output
    ledger = read_json(output / "PRE_GOLD_HASHES.json")
    if verification(output, ledger)["status"] != "PASS" or read_json(output / "PRE_GOLD_FREEZE.json").get("status") != "PASS":
        raise RuntimeError("PRE_GOLD_FREEZE_NOT_VALID")
    cohort = read_json(output / "B_SELECTOR_COHORT.json"); output_ids = list(cohort["output_ids"])
    source_candidates = read_jsonl_gz(output / "B_SOURCE_CANDIDATES.jsonl.gz")
    rankings, attempts = read_json(output / "B_RRF_RANKINGS.json"), read_json(output / "B_ATTEMPTS_FROZEN.json")
    gold = gold_for_outputs(args.solutions, output_ids)
    result_rows: list[dict[str, Any]] = []
    for output_id in output_ids:
        target_key = canonical(gold[output_id])
        source_rows = [row for row in source_candidates if row["output_id"] == output_id]
        source_by_depth = {depth: next((row for row in source_rows if row["depth"] == depth and row["grid_key"] == target_key), None) for depth in DEPTHS}
        if source_by_depth[24] is None and source_by_depth[48] is None:
            raise RuntimeError(f"CANDIDATE_EXTRACTION_BUG_GOLD_ABSENT:{output_id}")
        ranked = rankings[output_id]; gold_rank = next((int(row["final_b_rrf_rank"]) for row in ranked if row["grid_key"] == target_key), None)
        if gold_rank is None: raise RuntimeError(f"RRF_GOLD_MISSING:{output_id}")
        selected = attempts[output_id]
        row = {"task_id": oid_parts(output_id)[0], "output_id": output_id, "gold_present_in_pool": True,
               "gold_sources": "+".join(f"d{d}" for d in DEPTHS if source_by_depth[d] is not None), "gold_d24_source_rank": None, "gold_d48_source_rank": None,
               "gold_final_b_rrf_rank": gold_rank, "B_TOP1_HIT": selected["attempt_1"]["grid_key"] == target_key,
               "B_TOP2_HIT": target_key in {selected["attempt_1"]["grid_key"], selected["attempt_2"]["grid_key"]},
               "attempt_1_grid_key": selected["attempt_1"]["grid_key"], "attempt_2_grid_key": selected["attempt_2"]["grid_key"],
               "d24_support_count": None, "d48_support_count": None, "d24_mean_view_nll": None, "d48_mean_view_nll": None,
               "d24_b_support_score": None, "d48_b_support_score": None, "gold_b_rrf": next(row["b_rrf"] for row in ranked if row["grid_key"] == target_key)}
        for depth in DEPTHS:
            candidate = source_by_depth[depth]
            if candidate is not None:
                source_rank = next(item["source_rank"] for item in read_json(output / "B_SOURCE_RANKINGS.json")[output_id][f"d{depth}"] if item["grid_key"] == target_key)
                row[f"gold_d{depth}_source_rank"] = source_rank; row[f"d{depth}_support_count"] = candidate["support_count"]
                row[f"d{depth}_mean_view_nll"] = candidate["mean_view_nll"]; row[f"d{depth}_b_support_score"] = candidate["support_count"] - candidate["mean_view_nll"]
        result_rows.append(row)
    score_rows = {str(row["output_id"]): row for row in csv.DictReader(args.source_score_csv.open(encoding="utf-8", newline=""))}
    for row in result_rows: row["NLL_TOP2_HIT"] = bool_cell(score_rows[row["output_id"]]["TOP2_NLL_UNION"])
    retained = historical_retained(args.greedy_cells)
    selected_ids = {row["output_id"] for row in result_rows}
    if not retained <= selected_ids: raise RuntimeError(f"HISTORICAL_RETAINED_NOT_IN_ORC:{sorted(retained-selected_ids)}")
    new7 = set(NEW7)
    if not new7 <= selected_ids: raise RuntimeError(f"NEW7_NOT_IN_ORC:{sorted(new7-selected_ids)}")
    top1, top2 = sum(bool(row["B_TOP1_HIT"]) for row in result_rows), sum(bool(row["B_TOP2_HIT"]) for row in result_rows)
    fixes = [row for row in result_rows if row["B_TOP2_HIT"] and not row["NLL_TOP2_HIT"]]
    harms = [row for row in result_rows if row["NLL_TOP2_HIT"] and not row["B_TOP2_HIT"]]
    misses = [row for row in result_rows if not row["B_TOP2_HIT"]]
    summary = {"cohort_outputs": 35, "cohort_tasks": 27, "pool_gold_presence": f"{len(result_rows)}/35", "B_TOP1_HITS": top1, "B_TOP2_HITS": top2,
               "B_TOP2_FULL89": f"{top2}/89", "B_ORACLE_TO_TOP2_CONVERSION": top2 / 35, "HISTORICAL_28_B_TOP2_RETAINED": f"{sum(bool(row['B_TOP2_HIT']) for row in result_rows if row['output_id'] in retained)}/28",
               "DFS_NEW7_B_TOP2": f"{sum(bool(row['B_TOP2_HIT']) for row in result_rows if row['output_id'] in new7)}/7", "NLL_TOP2_BASELINE": "25/35",
               "B_FIXES_VS_NLL": len(fixes), "B_HARMS_VS_NLL": len(harms), "B_NET_GAIN_VS_NLL": len(fixes)-len(harms),
               "classification": "B_SELECTOR_TRANSFER_STRONG" if top2 >= 32 else "B_SELECTOR_TRANSFER_PARTIAL" if top2 >= 29 else "DFS_POOL_SELECTION_DISTRIBUTION_SHIFT",
               "gold_accessed_only_after_selector_freeze": True, "adapter_status": "EXACT_HISTORICAL_ADAPTER", "model_state_status": "BYTE_IDENTICAL_HISTORICAL_ADAPTERS"}
    write_csv(output / "OUTPUT_RESULTS.csv", result_rows, list(result_rows[0]))
    write_csv(output / "B_MISSES.csv", misses, list(result_rows[0]))
    write_csv(output / "B_FIXES_VS_NLL.csv", fixes, list(result_rows[0]))
    write_csv(output / "B_HARMS_VS_NLL.csv", harms, list(result_rows[0]))
    write_json(output / "SUMMARY.json", summary)
    manifest = read_json(ROOT / "artifacts" / "eval60_authoritative_greedy_v1" / "RUN_COMPLETION_MANIFEST.json")
    if manifest["gold"]["solution_sha256"] != EXPECTED_GOLD_SHA256: raise RuntimeError("HISTORICAL_GOLD_BINDING_FAIL")
    write_json(output / "GOLD_PROVENANCE.json", {"status": "PASS", "solutions_sha256": sha256(args.solutions), "expected_solutions_sha256": EXPECTED_GOLD_SHA256, "historical_manifest": "artifacts/eval60_authoritative_greedy_v1/RUN_COMPLETION_MANIFEST.json", "historical_manifest_gold_sha256": manifest["gold"]["solution_sha256"], "gold_opened_after_pre_gold_freeze_only": True})
    report = ["# Exact historical B-selector on frozen Phase-3 R1024 pool", "", f"- Cohort: 35 ORC-hit outputs / 27 tasks.", f"- B Top-1 / Top-2: {top1}/35 / {top2}/35.", f"- Full Eval60 Top-2 equivalent: {top2}/89.", f"- Historical retained 28: {summary['HISTORICAL_28_B_TOP2_RETAINED']}; DFS-new 7: {summary['DFS_NEW7_B_TOP2']}.", f"- NLL baseline: 25/35; B fixes/harms/net: {len(fixes)}/{len(harms)}/{len(fixes)-len(harms)}.", f"- Classification: `{summary['classification']}`.", "", "## Measured", "- Frozen-pool B Top-2 conversion with exact historical adapter bytes.", "", "## Not established", "- Generalization outside this frozen Eval60 diagnostic cohort."]
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    files = {path.name: sha256(path) for path in sorted(output.iterdir()) if path.is_file() and path.name != "HASHES.json"}
    write_json(output / "HASHES.json", files)
    final = verification(output, files)
    if final["status"] != "PASS": raise RuntimeError("FINAL_HASH_VERIFICATION_FAIL")
    print(canonical(summary))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    for name in ("d24", "d48", "source_score_csv", "challenge", "model_path", "native_config", "adapter_root", "output"):
        prep.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    prep.add_argument("--context-window", type=int, default=8192)
    prep.add_argument("--batch-size", type=int, default=4)
    evaluate_parser = sub.add_parser("evaluate")
    for name in ("output", "solutions", "source_score_csv", "greedy_cells"):
        evaluate_parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare": prepare(args)
    else: evaluate(args)


if __name__ == "__main__":
    main()
