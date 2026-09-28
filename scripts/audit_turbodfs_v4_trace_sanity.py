#!/usr/bin/env python3
"""CPU-only V1--V3 trace sanity audit before V4 GPU calibration.

This audit never reads targets or solutions.  It is deliberately limited to
token-domain, cumulative-score, node-parent and completion evidence available
in retained calibration traces.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from inference.nvarc_turbodfs_reference import PUBLIC_ARC_TOKENS
from scripts import run_eval60_adaptive_inference_joint_v2 as common


def read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--authoritative-root", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(); source, output = args.authoritative_root.resolve(), args.output.resolve()
    paths = sorted((source / "raw" / "turbodfs_v3_calibration" / "micro").glob("*.json"))
    if len(paths) != 6: raise RuntimeError("expected six retained V3 micro traces")
    failures: list[dict[str, Any]] = []
    details: list[dict[str, Any]] = []
    for path in paths:
        row = read(path)
        node_ids = {node.get("node_id") for node in row.get("nodes", [])}
        parents_ok = all(node.get("parent_node_id") is None or node.get("parent_node_id") in node_ids for node in row.get("nodes", []))
        tokens_ok = all(set(item.get("token_ids", [])) <= set(PUBLIC_ARC_TOKENS) for item in row.get("candidates", []))
        probabilities = row.get("branch_probabilities", [])
        distributions_ok = all({entry["token_id"] for entry in item.get("full_arc_logprobs", [])} == set(PUBLIC_ARC_TOKENS) for item in probabilities)
        candidate_scores_ok = all(float(item.get("cumulative_nll", 0.0)) < 1.6094379124341003 for item in row.get("candidates", []))
        item = {"path": str(path), "parent_links_valid": parents_ok, "native_token_domain_valid": tokens_ok,
                "branch_distribution_domain_valid": distributions_ok, "completed_scores_under_public_threshold": candidate_scores_ok,
                "candidate_count": len(row.get("candidates", [])), "node_count": len(row.get("nodes", [])),
                "probability_rows": len(probabilities), "termination_reason": row.get("termination_reason")}
        details.append(item)
        if not all((parents_ok, tokens_ok, distributions_ok, candidate_scores_ok)): failures.append(item)
    report = {"status": "PASS" if not failures else "FAIL", "audit": "CPU_TRACE_SANITY_ONLY", "solutions_accessed": False,
              "public_arc_tokens": list(PUBLIC_ARC_TOKENS), "traces": len(paths), "failures": failures, "details": details,
              "scope": "V3 trace inspection cannot prove GPU/model parity; it checks no obvious token-domain, score or tree-link semantic mismatch before V4."}
    common.atomic_json(output / "v4_cpu_trace_sanity.json", report)
    (output / "V4_CPU_TRACE_SANITY.md").write_text("# V4 CPU/trace sanity\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in report.items() if key not in {"details", "failures"}) + "\n", encoding="utf-8")
    print(json.dumps(report, sort_keys=True))
    if failures: raise RuntimeError("V4_TRACE_SANITY_FAILED")


if __name__ == "__main__": main()
