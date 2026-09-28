#!/usr/bin/env python3
"""Dependency-free CPU contracts for the one-model V5 shared executor."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from inference.nvarc_turbodfs_v5 import FrontierFloorTurboDFSConfig, inference_frontier_floor_turbo_dfs
from inference.nvarc_turbodfs_v5_shared import SearchContext, SharedForwardExecutor


class TransitionModel:
    device = torch.device("cpu")

    def __call__(self, *, input_ids, **_kwargs):
        logits = torch.zeros((int(input_ids.shape[0]), 1, 16))
        for lane, token in enumerate(input_ids[:, -1].tolist()):
            logits[lane, 0, 15 if token == 1 else 1] = 5.0
        return SimpleNamespace(logits=logits, past_key_values=object())


def cfg() -> FrontierFloorTurboDFSConfig:
    return FrontierFloorTurboDFSConfig(max_new_tokens=3, max_score=0.01, absolute_end_time_unix=None)


def ctx(name: str, token: int = 2, adapter: str = "task:d24") -> SearchContext:
    return SearchContext(name, f"cell:{name}", adapter, torch.tensor([[token, token]]))


def expect_error(fn, fragment: str) -> bool:
    try:
        fn()
    except ValueError as exc:
        return fragment in str(exc)
    return False


def main() -> None:
    model = TransitionModel(); executor = SharedForwardExecutor(model, cfg())
    serial_a = inference_frontier_floor_turbo_dfs(model, input_ids=ctx("a").input_ids, config=cfg())
    serial_b = inference_frontier_floor_turbo_dfs(model, input_ids=ctx("b").input_ids, config=cfg())
    shared = executor.execute((ctx("a"), ctx("b")))
    left = shared.context_nodes("a"); right = shared.context_nodes("b")
    tests = {
        "same_adapter_serial_candidate_parity": [[x.token_ids for x in lane] for lane in shared.result.candidates] == [[x.token_ids for x in serial_a.candidates[0]], [x.token_ids for x in serial_b.candidates[0]]],
        "distinct_node_id_namespaces": bool(left and right and {x["node_id"] for x in left}.isdisjoint({x["node_id"] for x in right})),
        "context_a_cache_slot": bool(left and all(x["search_context_id"] == "a" and x["cache_slot"] == 0 for x in left)),
        "context_b_cache_slot": bool(right and all(x["search_context_id"] == "b" and x["cache_slot"] == 1 for x in right)),
        "finished_context_isolation": bool(executor.execute((ctx("eos", 1), ctx("long", 2))).result.candidates[1]),
        "frontier_floor_context_local": bool([x for x in shared.result.frontier_floor_events if x["lane"] == 0] and [x for x in shared.result.frontier_floor_events if x["lane"] == 1]),
        "same_adapter_enforced": expect_error(lambda: executor.execute((ctx("x", adapter="x"), ctx("y", adapter="y"))), "different adapter"),
        "duplicate_context_rejected": expect_error(lambda: executor.execute((ctx("dup"), ctx("dup"))), "context IDs"),
        "unequal_width_rejected": expect_error(lambda: executor.execute((ctx("narrow"), SearchContext("wide", "cell:wide", "task:d24", torch.tensor([[2, 2, 2]])))), "equal frozen prompt widths"),
        "odd_single_context": len(executor.execute((ctx("only"),)).result.candidates) == 1,
        "probability_row_routing": all(row["cache_slot"] == 0 for row in shared.context_probabilities("a")) and all(row["cache_slot"] == 1 for row in shared.context_probabilities("b")),
    }
    output = {"unit_tests": tests, "passed": sum(bool(x) for x in tests.values()), "total": len(tests), "status": "PASS" if all(tests.values()) else "FAIL"}
    args = argparse.ArgumentParser(); args.add_argument("--output", type=Path, required=True); ns = args.parse_args()
    ns.output.parent.mkdir(parents=True, exist_ok=True); ns.output.write_text(json.dumps(output, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(output, sort_keys=True))
    if output["status"] != "PASS": raise SystemExit(1)


if __name__ == "__main__":
    main()
