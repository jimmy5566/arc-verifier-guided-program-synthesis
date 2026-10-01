#!/usr/bin/env python3
"""Target-blind sibling/backtrack proof for rollback-owned Clean-HF DFS.

This is intentionally a small physical CUDA test, separate from the R32 whole
search oracle.  It freezes a naturally occurring multi-sibling point, advances
one sibling one level deeper, crops to the parent's real sequence length, and
then compares the next sibling's actual model reply with one produced from an
independent clone of the same parent cache.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from inference.hf_peft_backend import load_hf_peft_inference  # noqa: E402
from inference.nvarc_turbodfs_d1 import D1TurboDFSConfig, _retained  # noqa: E402
from inference.nvarc_turbodfs_dynamic_ready import (  # noqa: E402
    _legacy_cache,
    _logits_diagnostic,
    _restore_cache_kind,
    cache_sha256,
    clone_legacy_cache,
    ready_incremental_forward_kwargs,
)
from scripts.run_clean_hf_parallel_regret_dfs_v1 import (  # noqa: E402
    _assert_challenge_only,
    _cache_transform,
    _prompt_ids,
    _verify_frozen_foundation,
)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2)
            handle.write("\n")
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _config() -> D1TurboDFSConfig:
    return D1TurboDFSConfig(
        policy_id="CUMULATIVE_REGRET_r=4.00", max_new_tokens=931,
        max_score=1.6094379124341003, absolute_end_time_unix=None,
        max_expanded_nodes=32, max_completed_candidates=32, frontier_floor=1,
        diagnostic_trace=True, independent_lane_budgets=True,
    )


def _sequence_length(cache: Any) -> int:
    return int(_legacy_cache(cache)[0][0].shape[-2])


def _retained_tokens(logits: Any, config: D1TurboDFSConfig) -> tuple[list[int], list[int]]:
    import torch
    values = [(token, float(torch.log_softmax(logits[0, -1].float(), dim=-1)[token].item())) for token in config.arc_tokens]
    ranked = sorted(values, key=lambda pair: (-pair[1], pair[0]))
    kept, _reason = _retained(config, ranked, score_before=0.0, regret_before=0.0,
                              remaining=config.max_new_tokens, generated_length=0)
    return [int(value[2]) for value in kept if int(value[2]) != 15], [int(token) for token, _value in ranked]


def _row(*, path: str, token: int, position: int, input_sha: str, output: Any,
         config: D1TurboDFSConfig) -> dict[str, Any]:
    diagnostic = _logits_diagnostic(output.logits, config.arc_tokens)
    retained, _ranked = _retained_tokens(output.logits, config)
    return {
        "path": path,
        "token": token,
        "position": position,
        "input_cache_sha256": input_sha,
        "full_logits_sha256": diagnostic["full_logits_sha256"],
        "arc_logits_json": json.dumps(diagnostic["arc_logits"], sort_keys=True),
        "arc_logprobs_json": json.dumps(diagnostic["arc_logprobs"], sort_keys=True),
        "arc_ranking_json": json.dumps(diagnostic["arc_ranking"]),
        "retained_successors_json": json.dumps(retained),
        "output_cache_sha256": cache_sha256(output.past_key_values),
        "output_sequence_length": _sequence_length(output.past_key_values),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from arc.io import load_dataset

    _assert_challenge_only(args.challenge)
    foundation = json.loads(args.adapter_foundation.read_text(encoding="utf-8"))
    model, tokenizer, identity = load_hf_peft_inference(
        model_path=args.model_path, adapter_path=args.adapter_path, device=args.device,
        native_config_dir=args.native_config_dir,
        frozen_adapter_identity={
            "adapter_sha256": str(foundation["adapter_sha256"]),
            "adapter_config_sha256": str(foundation["adapter_config_sha256"]),
        },
    )
    adapter = _verify_frozen_foundation(args.adapter_foundation, identity)
    task = load_dataset(args.challenge)[args.task_id]
    config = _config()
    input_ids = _prompt_ids(tokenizer=tokenizer, task=task, output_index=args.output_index,
                             view=args.view, device=args.device)
    with torch.no_grad():
        root_output = model(input_ids=input_ids, return_dict=True, use_cache=True)
    cache = _cache_transform(root_output.past_key_values)
    root_length = _sequence_length(cache)

    def forward(target_cache: Any, token: int, position: int) -> Any:
        before = id(target_cache)
        with torch.no_grad():
            output = model(**ready_incremental_forward_kwargs(
                token_ids=[token], position=position, cache=target_cache, device=model.device,
            ))
        if output.past_key_values is not target_cache or id(target_cache) != before:
            raise RuntimeError("Clean-HF B1 did not preserve the mutable DynamicCache identity")
        return output

    # This prefix was selected target-blind from the R32 trace: it reaches the
    # first real non-EOS multi-sibling parent for this frozen view.  Recheck it
    # from current logits instead of trusting a historical trace.
    first = forward(cache, 7, root_length)
    second = forward(cache, 7, root_length + 1)
    parent_length = _sequence_length(cache)
    siblings, parent_ranking = _retained_tokens(second.logits, config)
    if len(siblings) < 2:
        raise RuntimeError(f"MULTI_SIBLING_CANARY_NOT_REPRODUCED: retained={siblings}")
    sibling_a, sibling_b = siblings[:2]
    parent_before = cache_sha256(cache)
    reference_parent = _restore_cache_kind(clone_legacy_cache(cache), cache)

    # Production path: enter sibling A, advance one deterministic real child
    # step, then roll the one owned cache back to the parent checkpoint.
    output_a = forward(cache, sibling_a, parent_length)
    descendants, descendant_ranking = _retained_tokens(output_a.logits, config)
    child = descendants[0] if descendants else next(token for token in descendant_ranking if token != 15)
    _ = forward(cache, child, parent_length + 1)
    cache.crop(parent_length)
    production_input = cache_sha256(cache)
    production = forward(cache, sibling_b, parent_length)
    production_row = _row(path="production_after_sibling_A_subtree_and_crop", token=sibling_b,
                          position=parent_length, input_sha=production_input, output=production, config=config)

    # Independent pristine parent snapshot -> exact sibling-B control.
    reference_input = cache_sha256(reference_parent)
    reference = forward(reference_parent, sibling_b, parent_length)
    reference_row = _row(path="independent_pristine_parent_to_sibling_B", token=sibling_b,
                         position=parent_length, input_sha=reference_input, output=reference, config=config)
    compare_keys = [
        "token", "position", "input_cache_sha256", "full_logits_sha256", "arc_logits_json",
        "arc_logprobs_json", "arc_ranking_json", "retained_successors_json",
        "output_cache_sha256", "output_sequence_length",
    ]
    exact = all(production_row[key] == reference_row[key] for key in compare_keys)
    rows = [
        {"comparison": "production_rollback_vs_snapshot", "strict_exact": exact,
         "sibling_A": sibling_a, "sibling_B": sibling_b, "subtree_child": child,
         **{f"production_{key}": production_row[key] for key in compare_keys},
         **{f"reference_{key}": reference_row[key] for key in compare_keys}},
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    csv_path = args.output / "BRANCH_BACKTRACK_PARITY.csv"
    fields = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    canary = {
        "experiment": "CLEAN_HF_ROLLBACK_DFS_AND_B2_OWNERSHIP_V1",
        "target_blind": True, "gold_loaded": False, "task_id": args.task_id,
        "output_index": args.output_index, "depth": args.depth, "view": args.view,
        "prefix_tokens": [7, 7], "root_sequence_length": root_length,
        "parent_sequence_length": parent_length, "parent_cache_sha256": parent_before,
        "parent_arc_ranking": parent_ranking, "retained_non_eos_successors": siblings,
        "frozen_sibling_A": sibling_a, "frozen_sibling_B": sibling_b,
        "sibling_A_real_descendant": child, "strict_backtrack_parity": exact,
        "adapter": adapter, "runtime_identity": identity,
    }
    _atomic_json(args.output / "MULTI_SIBLING_CANARY.json", canary)
    _atomic_json(args.output / "BRANCH_BACKTRACK_DECISION.json", {
        "status": "PASS" if exact else "DFS_BACKTRACK_SEMANTIC_FAIL",
        "strict_exact": exact,
        "comparison_keys": compare_keys,
    })
    _atomic_json(args.output / "HASHES.json", {
        path.name: _sha(path) for path in args.output.iterdir() if path.is_file()
    })
    if not exact:
        raise RuntimeError("DFS_BACKTRACK_SEMANTIC_FAIL")
    return canary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for name in ("model_path", "adapter_path", "challenge", "native_config_dir", "adapter_foundation", "output"):
        parser.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    parser.add_argument("--task-id", default="d59b0160")
    parser.add_argument("--output-index", type=int, default=0)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--view", default="anti_transpose")
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


if __name__ == "__main__":
    result = run(parse_args())
    print(json.dumps({"strict_backtrack_parity": result["strict_backtrack_parity"],
                      "retained_non_eos_successors": result["retained_non_eos_successors"]}, sort_keys=True))
