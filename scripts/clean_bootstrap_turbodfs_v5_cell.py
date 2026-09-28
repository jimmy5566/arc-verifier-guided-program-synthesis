#!/usr/bin/env python3
"""One target-blind TurboDFS cell used solely to prove a clean V5 bootstrap."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_eval60_adaptive_inference_joint_v2 as common
from scripts import run_eval60_authoritative_greedy_v1 as greedy
from scripts.run_adaptive_ttt_loo_transfer12 import read_json, view_task
from scripts.turbodfs_v5_common import decoder_from, sha256_file, turbo_cells_v5_batch

PARTNER = {"identity": "flip_ud", "flip_ud": "identity", "transpose": "anti_transpose", "anti_transpose": "transpose"}


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("output", "authoritative-root", "challenge", "reference-config", "model-path", "native-config-dir", "adapter-manifest", "final-config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--output-index", type=int, required=True)
    parser.add_argument("--depth", type=int, choices=(12, 24, 48), required=True)
    parser.add_argument("--view", choices=tuple(PARTNER), required=True)
    parser.add_argument("--gpu-id", type=int, default=0)
    args = parser.parse_args()
    with args.adapter_manifest.open(newline="", encoding="utf-8") as handle:
        row = next((item for item in csv.DictReader(handle) if item["task_id"] == args.task_id and int(item["depth"]) == args.depth), None)
    if row is None:
        raise RuntimeError("requested clean-bootstrap adapter mapping missing")
    adapter = Path(row["global_path"])
    if not adapter.is_file() or sha256_file(adapter) != row["sha256"]:
        raise RuntimeError("clean-bootstrap global adapter identity mismatch")
    payload, decoder = decoder_from(args.final_config)
    runtime_args = SimpleNamespace(output=args.authoritative_root, challenge=args.challenge, reference_config=args.reference_config,
        model_path=args.model_path, native_config_dir=args.native_config_dir, gpu_id=args.gpu_id)
    _root, _manifest, generation_config, tasks, model, tokenizer, _initial, _base = greedy._runtime(runtime_args)
    try:
        metadata = read_json(args.authoritative_root / "checkpoints" / args.task_id / f"depth_{args.depth:03d}" / "metadata.json")
        if metadata["checkpoint_sha256"] != row["sha256"]:
            raise RuntimeError("authoritative/global checkpoint disagreement")
        metadata["checkpoint_path"] = str(adapter)
        common.load_adapter(model=model, metadata=metadata)
        target, _companion = turbo_cells_v5_batch(
            model=model, tokenizer=tokenizer, task=view_task(tasks[args.task_id], args.output_index),
            task_id=args.task_id, output_index=args.output_index, depth=args.depth,
            views=(args.view, PARTNER[args.view]), generation_config=generation_config,
            decoder=decoder, checkpoint_sha=row["sha256"],
        )
        target.update({"clean_bootstrap_test": True, "solutions_accessed": False,
                       "decoder_config_sha256": sha256_file(args.final_config), "decoder_id": payload["decoder_id"]})
        if not target.get("search_tree_reconstructible") or not target.get("full_branch_probabilities_saved"):
            raise RuntimeError("clean bootstrap trace contract failure")
        common.atomic_json(args.output, target)
        print(json.dumps({"GLOBAL_BOOTSTRAP_TEST": "PASS", "valid_grid_count": target["valid_grid_count"],
                          "candidate_count": target["candidate_count"], "solutions_accessed": False}, sort_keys=True))
    finally:
        del model


if __name__ == "__main__":
    main()
