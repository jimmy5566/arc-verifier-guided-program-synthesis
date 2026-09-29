"""D2 nonblind mechanism diagnosis with target-blind decoder tracing.

The worker never opens solutions.  Gold is accepted only by ``annotate-gold``
after a verified trace freeze.  This deliberately reuses the immutable D1
model/adapters/configuration and changes only append-only telemetry.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.d1_shared_queue import claim_cell, release_claim
from inference.nvarc_turbodfs_d1 import POLICIES
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.run_d1_real_decoder_ab import (
    adapter_records, atomic_json, cell_key, config_for, existing_cell,
    load_adapter, no_gold_challenge, output_key,
)
from scripts.turbodfs_d1_common import d1_cells_batch
from scripts.turbodfs_v4_common import sha256_file


EXPERIMENT = "D2_RETRIEVAL_AND_SEARCH_EFFICIENCY_DIAGNOSIS_V1"


def sha_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def candidate_signature(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [{key: candidate.get(key) for key in (
        "candidate_id", "cell_candidate_id", "candidate_token_ids", "canonical_candidate",
        "valid_grid", "cumulative_nll", "candidate_discovery_order", "terminal_node_id",
    )} for candidate in row["candidates"]]


def row_path(root: Path, policy: str, key: str, kind: str) -> Path:
    safe = hashlib.sha256(policy.encode("utf-8")).hexdigest()[:12]
    return root / kind / safe / (key.replace(":", "_") + ".json")


def required_items(manifest: dict[str, Any], keys: list[str]) -> list[tuple[str, str]]:
    return [(policy, key) for policy in POLICIES for key in keys]


def load_keys(path: Path) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("selected_cell_keys")
    if not isinstance(payload, list) or not payload or not all(isinstance(value, str) for value in payload):
        raise RuntimeError("D2 cell manifest must contain nonempty selected_cell_keys")
    return list(payload)


def prepare(args: argparse.Namespace) -> None:
    root = args.output.resolve()
    if root.exists():
        raise RuntimeError(f"D2 output already exists: {root}")
    d1_root = args.d1_root.resolve(); d1_manifest = read_json(d1_root / "D1_MANIFEST.json")
    cohort = args.cohort.resolve(); keys = load_keys(cohort)
    if not (d1_root / "D1_FIXED_BUDGET_CONTRACT_SHA.json").is_file():
        raise RuntimeError("D2 requires frozen D1 fixed-budget contract")
    challenge = d1_root / "generation_inputs" / "evaluation_challenges.json"
    no_gold_challenge(challenge)
    root.mkdir(parents=True)
    (root / "generation_inputs").mkdir()
    shutil.copyfile(challenge, root / "generation_inputs" / "evaluation_challenges.json")
    manifest = {
        "experiment_id": EXPERIMENT,
        "scope": "NONBLIND_MECHANISM_DIAGNOSTIC; target-blind generation then post-freeze Gold annotation",
        "source_commit": args.source_commit,
        "d1_root": str(d1_root), "d1_manifest_sha256": sha256_file(d1_root / "D1_MANIFEST.json"),
        "d1_contract_sha256": read_json(d1_root / "D1_FIXED_BUDGET_CONTRACT_SHA.json")["contract_sha256"],
        "cohort_file_sha256": sha256_file(cohort), "selected_cell_keys": keys,
        "selected_cell_keys_sha256": sha_json(keys), "challenge_sha256": sha256_file(challenge),
        "challenge_path": str(root / "generation_inputs" / "evaluation_challenges.json"),
        "adapter_manifest": d1_manifest["adapter_manifest"], "adapter_manifest_sha256": d1_manifest["adapter_manifest_sha256"],
        "authoritative_root": d1_manifest["authoritative_root"], "model_path": d1_manifest["model_path"],
        "native_config_dir": d1_manifest["native_config_dir"],
        "reference_config": str(d1_root / "generation_inputs" / "reference_ttt_config.json"),
        "g1_root": str(args.g1_root.resolve()), "g1_root_note": "Gold trace source; forbidden to workers",
        "policies": list(POLICIES), "workers": "one worker per GPU; shared reclaimable queue",
        "solutions_accessed": False,
    }
    atomic_json(root / "D2_MANIFEST.json", manifest)
    atomic_json(root / "D2_PROVENANCE.json", {"manifest_sha256": sha256_file(root / "D2_MANIFEST.json"),
                                                 "gold_status": "FORBIDDEN_UNTIL_D2_GENERATION_FROZEN.flag"})


def verify_reference(d1_root: Path, policy: str, row: dict[str, Any]) -> dict[str, Any]:
    reference = existing_cell(d1_root, policy, str(row["output_id"]), int(row["depth"]), str(row["view"]), str(row["decoder_config_sha256"]))
    if reference is None:
        raise RuntimeError(f"missing frozen D1 reference cell: {policy} {row['output_id']}")
    expected = candidate_signature(reference); actual = candidate_signature(row)
    return {
        "candidate_pool_identical": expected == actual,
        "candidate_order_identical": expected == actual,
        "nodes_expanded_identical": int(reference["nodes_expanded"]) == int(row["nodes_expanded"]),
        "termination_reason_identical": reference["termination_reason"] == row["termination_reason"],
        "reference_candidate_sha256": sha_json(expected), "instrumented_candidate_sha256": sha_json(actual),
        "reference_nodes_expanded": int(reference["nodes_expanded"]), "instrumented_nodes_expanded": int(row["nodes_expanded"]),
        "reference_termination_reason": reference["termination_reason"], "instrumented_termination_reason": row["termination_reason"],
    }


def worker(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D2_MANIFEST.json"); d1_root = Path(manifest["d1_root"])
    no_gold_challenge(Path(manifest["challenge_path"]))
    keys = load_keys(args.cells)
    if args.mode == "trace" and not (root / "D2_TRACE_PARITY_PASSED.flag").is_file():
        raise RuntimeError("D2 trace generation requires successful no-Gold parity freeze")
    from scripts import run_eval60_authoritative_greedy_v1 as greedy
    runtime_args = SimpleNamespace(output=Path(manifest["authoritative_root"]), challenge=Path(manifest["challenge_path"]),
                                   reference_config=Path(manifest["reference_config"]), model_path=Path(manifest["model_path"]),
                                   native_config_dir=Path(manifest["native_config_dir"]), gpu_id=args.gpu_id)
    _root, _runtime_manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    d1_manifest = read_json(d1_root / "D1_MANIFEST.json"); adapters = adapter_records(Path(manifest["adapter_manifest"]))
    last_adapter: tuple[str, int] | None = None
    for policy, key in required_items(manifest, keys):
        destination = row_path(root, policy, key, "parity" if args.mode == "parity" else "raw")
        if destination.is_file():
            continue
        task_id, output_index = output_key(key.rsplit(":d", 1)[0]); depth = int(key.split(":d", 1)[1].split(":", 1)[0]); view = key.rsplit(":", 1)[1]
        claim = claim_cell(claims_root=root / "claims" / args.mode, policy=policy, output_id=f"{task_id}:o{output_index}",
                           depth=depth, view=view, worker_id=f"d2-{args.worker_index}@gpu{args.gpu_id}", stale_seconds=float(args.claim_stale_seconds))
        if claim is None:
            continue
        try:
            if destination.is_file():
                continue
            adapter_key = (task_id, depth)
            if adapter_key != last_adapter:
                adapter_sha = load_adapter(model, adapters[adapter_key]); last_adapter = adapter_key
            else:
                adapter_sha = str(adapters[adapter_key]["sha256"])
            decoder, config_sha = config_for(d1_manifest, policy)
            decoder = replace(decoder, diagnostic_trace=True)
            rows = d1_cells_batch(model=model, tokenizer=tokenizer, task=view_task(tasks[task_id], output_index),
                                  task_id=task_id, output_index=output_index, depth=depth, views=(view,),
                                  generation_config=generation_config, decoder=decoder, checkpoint_sha=adapter_sha,
                                  diagnostic_trace=True)
            row = rows[0]
            row.update({"output_id": f"{task_id}:o{output_index}", "decoder_config_sha256": config_sha,
                        "adapter_sha": adapter_sha, "worker_index": args.worker_index, "gpu_id": args.gpu_id,
                        "solutions_accessed": False, "d2_mode": args.mode, "d2_cell_key": key,
                        "d1_fixed_budget_contract_sha256": manifest["d1_contract_sha256"]})
            if args.mode == "parity":
                row["parity"] = verify_reference(d1_root, policy, row)
            atomic_json(destination, row)
        finally:
            if destination.is_file():
                release_claim(claim)
    del model


def all_rows(root: Path, manifest: dict[str, Any], keys: list[str], kind: str) -> list[dict[str, Any]]:
    rows = []
    for policy, key in required_items(manifest, keys):
        path = row_path(root, policy, key, kind)
        if not path.is_file():
            raise RuntimeError(f"missing {kind} record: {policy} {key}")
        row = read_json(path)
        if row.get("solutions_accessed") is not False or row.get("decoder_policy") != policy or row.get("d2_cell_key") != key:
            raise RuntimeError(f"invalid {kind} record: {path}")
        rows.append(row)
    return rows


def freeze_parity(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D2_MANIFEST.json"); keys = load_keys(args.cells)
    rows = all_rows(root, manifest, keys, "parity")
    result = []
    for row in rows:
        parity = row.get("parity") or {}
        passed = all(bool(parity.get(key)) for key in ("candidate_pool_identical", "candidate_order_identical", "nodes_expanded_identical", "termination_reason_identical"))
        result.append({"decoder_policy": row["decoder_policy"], "cell_key": row["d2_cell_key"], "passed": passed, **parity})
    if not all(row["passed"] for row in result):
        raise RuntimeError("D2 diagnostic instrumentation changed frozen D1 search behavior")
    atomic_json(root / "D2_TRACE_PARITY_PASSED.flag", {"records": len(rows), "result_sha256": sha_json(result),
                                                         "solutions_accessed": False, "results": result})


def freeze_trace(args: argparse.Namespace) -> None:
    root = args.output.resolve(); manifest = read_json(root / "D2_MANIFEST.json"); keys = load_keys(args.cells)
    rows = all_rows(root, manifest, keys, "raw")
    hashes = {str(row_path(root, str(row["decoder_policy"]), str(row["d2_cell_key"]), "raw").relative_to(root)):
              sha256_file(row_path(root, str(row["decoder_policy"]), str(row["d2_cell_key"]), "raw")) for row in rows}
    atomic_json(root / "D2_GENERATION_FROZEN.flag", {"records": len(rows), "raw_hashes": hashes,
                                                       "manifest_sha256": sha256_file(root / "D2_MANIFEST.json"),
                                                       "solutions_accessed": False})


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n"); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--output", type=Path, required=True); p.add_argument("--d1-root", type=Path, required=True); p.add_argument("--cohort", type=Path, required=True); p.add_argument("--g1-root", type=Path, required=True); p.add_argument("--source-commit", required=True)
    p = sub.add_parser("worker"); p.add_argument("--output", type=Path, required=True); p.add_argument("--cells", type=Path, required=True); p.add_argument("--gpu-id", type=int, required=True); p.add_argument("--worker-index", type=int, required=True); p.add_argument("--mode", choices=("parity", "trace"), required=True); p.add_argument("--claim-stale-seconds", type=float, default=900.0)
    p = sub.add_parser("freeze-parity"); p.add_argument("--output", type=Path, required=True); p.add_argument("--cells", type=Path, required=True)
    p = sub.add_parser("freeze-trace"); p.add_argument("--output", type=Path, required=True); p.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args(); {"prepare": prepare, "worker": worker, "freeze-parity": freeze_parity, "freeze-trace": freeze_trace}[args.command](args)


if __name__ == "__main__": main()
