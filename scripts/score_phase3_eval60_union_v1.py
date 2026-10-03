#!/usr/bin/env python3
"""CPU-only post-freeze Phase-3 scorer for frozen d24+d48 ARC-AGI-2 archives.

Primary metric:
  ORC / FIX = Gold exists anywhere in the frozen R1024 candidate set.

Descriptive target-blind rankings (post-freeze, not pre-registered Phase-3 metrics):
  NLL: unique canonical grids ordered by best terminal cumulative NLL, then
       support count, earliest discovery, stable grid key.
  SUPPORT_NLL: support count first, then best NLL, earliest discovery.

No model inference is performed.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


AUG8 = (
    "geom=identity__color=id__order=canonical",
    "geom=flip_ud__color=id__order=canonical",
    "geom=transpose__color=id__order=canonical",
    "geom=anti_transpose__color=id__order=canonical",
    "geom=rot90__color=id__order=canonical",
    "geom=rot180__color=id__order=canonical",
    "geom=rot270__color=id__order=canonical",
    "geom=flip_lr__color=id__order=canonical",
)
DEPTHS = (24, 48)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def output_index(output_id: str) -> int:
    return int(output_id.rsplit(":o", 1)[1])


def task_id(output_id: str) -> str:
    return output_id.split(":o", 1)[0]


def safe(output_id: str) -> str:
    return output_id.replace(":", "_")


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def validate_archive(archive: Path, depth: int) -> list[str]:
    contract = read_json(archive / "CONTRACT.json")
    audit = read_json(archive / "SOURCE_COMPLETENESS_AUDIT.json")
    av = read_json(archive / "manifests" / "ARCHIVE_HASH_VERIFICATION.json")
    if int(contract["ttt_depth"]) != depth:
        raise RuntimeError(f"{archive}: expected d{depth}")
    if contract.get("gold_loaded") is not False or audit.get("gold_loaded") is not False:
        raise RuntimeError(f"{archive}: not target-blind")
    if audit.get("status") != "PASS" or int(audit.get("outputs_found", -1)) != 89 or int(audit.get("canonical_aug8_cells_found", -1)) != 712:
        raise RuntimeError(f"{archive}: completeness audit failed")
    if av.get("status") != "PASS":
        raise RuntimeError(f"{archive}: archive hash verification failed")
    cohort = read_json(archive / "RUN_COHORT.json")
    outputs = [str(x["output_id"]) for x in cohort["outputs"]]
    if len(outputs) != 89 or len(set(outputs)) != 89:
        raise RuntimeError(f"{archive}: invalid output cohort")
    for oid in outputs:
        for sub in ("candidate_index", "node_index"):
            p = archive / "tables" / sub / f"{safe(oid)}.jsonl.gz"
            if not p.is_file():
                raise RuntimeError(f"missing {p}")
    return outputs


def load_gold_compact(solutions_path: Path, outputs: list[str], expected_sha256: str) -> tuple[dict[str, Any], dict[str, Any]]:
    observed = sha256(solutions_path)
    if observed != expected_sha256:
        raise RuntimeError(f"Gold SHA256 mismatch: expected {expected_sha256}, observed {observed}")
    payload = read_json(solutions_path)
    gold: dict[str, Any] = {}
    for oid in outputs:
        task = task_id(oid)
        idx = output_index(oid)
        if task not in payload:
            raise RuntimeError(f"missing Gold task: {task}")
        task_solutions = payload[task]
        if isinstance(task_solutions, list):
            if idx >= len(task_solutions):
                raise RuntimeError(f"{oid}: output index out of range in compact Gold")
            target = task_solutions[idx]
        elif isinstance(task_solutions, dict):
            tests = task_solutions.get("test", [])
            if idx >= len(tests):
                raise RuntimeError(f"{oid}: output index out of range in object Gold")
            item = tests[idx]
            target = item["output"] if isinstance(item, dict) else item
        else:
            raise RuntimeError(f"{task}: unsupported Gold schema")
        gold[oid] = target
    if set(gold) != set(outputs):
        raise RuntimeError("Gold output coverage mismatch")
    return gold, {
        "source": str(solutions_path),
        "solutions_sha256": observed,
        "output_count": len(gold),
        "gold_digest": hashlib.sha256(canonical({k: gold[k] for k in sorted(gold)}).encode()).hexdigest(),
    }


def node_scores(archive: Path, oid: str) -> dict[tuple[str, int], tuple[float | None, float | None]]:
    rows = read_jsonl_gz(archive / "tables" / "node_index" / f"{safe(oid)}.jsonl.gz")
    out: dict[tuple[str, int], tuple[float | None, float | None]] = {}
    for row in rows:
        node = row.get("node")
        if not isinstance(node, dict):
            continue
        node_id = node.get("node_id")
        if node_id is None:
            continue
        score = node.get("cumulative_score")
        regret = node.get("cumulative_regret")
        out[(str(row["augmentation_id"]), int(node_id))] = (
            float(score) if isinstance(score, (int, float)) and math.isfinite(float(score)) else None,
            float(regret) if isinstance(regret, (int, float)) and math.isfinite(float(regret)) else None,
        )
    return out


def occurrences(archive: Path, oid: str, checkpoint: int) -> list[dict[str, Any]]:
    depth = int(read_json(archive / "CONTRACT.json")["ttt_depth"])
    score_map = node_scores(archive, oid)
    rows = read_jsonl_gz(archive / "tables" / "candidate_index" / f"{safe(oid)}.jsonl.gz")
    result = []
    for row in rows:
        if int(row["checkpoint"]) != checkpoint:
            continue
        grid = row.get("canonical_grid")
        if grid is None:
            continue
        terminal = row.get("terminal_node_id")
        nll = regret = None
        if terminal is not None:
            nll, regret = score_map.get((str(row["augmentation_id"]), int(terminal)), (None, None))
        result.append({
            "depth": depth,
            "augmentation_id": str(row["augmentation_id"]),
            "candidate_completion_index": int(row.get("candidate_completion_index", 0)),
            "candidate_id": row.get("candidate_id"),
            "terminal_node_id": terminal,
            "nodes_expanded_so_far": row.get("nodes_expanded_so_far"),
            "canonical_grid": grid,
            "grid_key": canonical(grid),
            "nll": nll,
            "regret": regret,
        })
    return result


def aggregate_candidates(occs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in occs:
        grouped[row["grid_key"]].append(row)
    agg = []
    for key, rows in grouped.items():
        sources = {(int(r["depth"]), str(r["augmentation_id"])) for r in rows}
        nlls = [float(r["nll"]) for r in rows if r["nll"] is not None]
        regrets = [float(r["regret"]) for r in rows if r["regret"] is not None]
        nodes = [int(r["nodes_expanded_so_far"]) for r in rows if isinstance(r.get("nodes_expanded_so_far"), int)]
        agg.append({
            "grid_key": key,
            "canonical_grid": rows[0]["canonical_grid"],
            "support_count": len(sources),
            "depth_support": len({d for d, _ in sources}),
            "augmentation_support": len({a for _, a in sources}),
            "occurrence_count": len(rows),
            "best_nll": min(nlls) if nlls else math.inf,
            "best_regret": min(regrets) if regrets else math.inf,
            "earliest_node": min(nodes) if nodes else 10**18,
            "sources": sorted([f"d{d}:{a}" for d, a in sources]),
        })
    return agg


def ranking(candidates: list[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    if method == "NLL":
        return sorted(candidates, key=lambda x: (
            float(x["best_nll"]), -int(x["support_count"]), int(x["earliest_node"]), x["grid_key"]
        ))
    if method == "SUPPORT_NLL":
        return sorted(candidates, key=lambda x: (
            -int(x["support_count"]), float(x["best_nll"]), int(x["earliest_node"]), x["grid_key"]
        ))
    raise ValueError(method)


def hit_rank(candidates: list[dict[str, Any]], target_key: str, method: str) -> int | None:
    for i, row in enumerate(ranking(candidates, method), 1):
        if row["grid_key"] == target_key:
            return i
    return None


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            out = {}
            for key in fields:
                value = row.get(key)
                if isinstance(value, (dict, list)):
                    value = canonical(value)
                out[key] = value
            w.writerow(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d24", required=True, type=Path)
    ap.add_argument("--d48", required=True, type=Path)
    ap.add_argument("--solutions-json", required=True, type=Path)
    ap.add_argument("--expected-solutions-sha256", required=True)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)

    out24 = validate_archive(args.d24, 24)
    out48 = validate_archive(args.d48, 48)
    if out24 != out48:
        raise RuntimeError("d24/d48 cohort ordering differs")
    outputs = out24

    gold, gold_manifest = load_gold_compact(args.solutions_json, outputs, args.expected_solutions_sha256)

    rows = []
    task_to_outputs: dict[str, list[str]] = defaultdict(list)
    for oid in outputs:
        task_to_outputs[task_id(oid)].append(oid)
        target_key = canonical(gold[oid])
        depth_occs: dict[int, list[dict[str, Any]]] = {
            24: occurrences(args.d24, oid, 1024),
            48: occurrences(args.d48, oid, 1024),
        }
        depth512: dict[int, list[dict[str, Any]]] = {
            24: occurrences(args.d24, oid, 512),
            48: occurrences(args.d48, oid, 512),
        }
        ag24 = aggregate_candidates(depth_occs[24])
        ag48 = aggregate_candidates(depth_occs[48])
        agunion = aggregate_candidates(depth_occs[24] + depth_occs[48])
        agunion512 = aggregate_candidates(depth512[24] + depth512[48])

        orc24 = any(x["grid_key"] == target_key for x in ag24)
        orc48 = any(x["grid_key"] == target_key for x in ag48)
        orcu = any(x["grid_key"] == target_key for x in agunion)
        orcu512 = any(x["grid_key"] == target_key for x in agunion512)
        cls = "SHARED" if orc24 and orc48 else "D24_ONLY" if orc24 else "D48_ONLY" if orc48 else "NEITHER"

        hit_occ24 = [x for x in depth_occs[24] if x["grid_key"] == target_key]
        hit_occ48 = [x for x in depth_occs[48] if x["grid_key"] == target_key]
        row = {
            "task_id": task_id(oid), "output_id": oid, "classification": cls,
            "candidate_unique_d24": len(ag24), "candidate_unique_d48": len(ag48), "candidate_unique_union": len(agunion),
            "ORC_D24": orc24, "ORC_D48": orc48, "ORC_UNION": orcu, "ORC_UNION_R512": orcu512,
            "first_gold_node_d24": min([x["nodes_expanded_so_far"] for x in hit_occ24 if isinstance(x.get("nodes_expanded_so_far"), int)], default=None),
            "first_gold_node_d48": min([x["nodes_expanded_so_far"] for x in hit_occ48 if isinstance(x.get("nodes_expanded_so_far"), int)], default=None),
            "gold_views_d24": sorted({x["augmentation_id"] for x in hit_occ24}),
            "gold_views_d48": sorted({x["augmentation_id"] for x in hit_occ48}),
        }
        for label, candidates in (("D24", ag24), ("D48", ag48), ("UNION", agunion)):
            for method in ("NLL", "SUPPORT_NLL"):
                rank = hit_rank(candidates, target_key, method)
                row[f"gold_rank_{method}_{label}"] = rank
                row[f"TOP1_{method}_{label}"] = rank is not None and rank <= 1
                row[f"TOP2_{method}_{label}"] = rank is not None and rank <= 2
        rows.append(row)

    summary: dict[str, Any] = {
        "scope": "POST_FREEZE_CPU_ONLY",
        "outputs": 89,
        "tasks": 60,
        "primary_metric_definition": "ORC/FIX: Gold appears anywhere in frozen R1024 candidate set.",
        "top2_status": "POST_FREEZE_DESCRIPTIVE_TARGET_BLIND_RANKING_NOT_PRE_REGISTERED_PHASE3_METRIC",
        "ranker_NLL": "best terminal cumulative NLL ascending; ties support desc, earliest node, stable grid key",
        "ranker_SUPPORT_NLL": "unique (depth,augmentation) support desc; ties best terminal NLL, earliest node, stable grid key",
        "FIX_D24": sum(bool(r["ORC_D24"]) for r in rows),
        "FIX_D48": sum(bool(r["ORC_D48"]) for r in rows),
        "FIX_UNION": sum(bool(r["ORC_UNION"]) for r in rows),
        "FIX_UNION_R512": sum(bool(r["ORC_UNION_R512"]) for r in rows),
        "D24_ONLY": sum(r["classification"] == "D24_ONLY" for r in rows),
        "D48_ONLY": sum(r["classification"] == "D48_ONLY" for r in rows),
        "SHARED": sum(r["classification"] == "SHARED" for r in rows),
        "NEITHER": sum(r["classification"] == "NEITHER" for r in rows),
    }
    for method in ("NLL", "SUPPORT_NLL"):
        for label in ("D24", "D48", "UNION"):
            summary[f"TOP1_{method}_{label}_OUTPUTS"] = sum(bool(r[f"TOP1_{method}_{label}"]) for r in rows)
            summary[f"TOP2_{method}_{label}_OUTPUTS"] = sum(bool(r[f"TOP2_{method}_{label}"]) for r in rows)

    # Task-level ARC score: every test output for a task must succeed.
    task_rows = []
    by_oid = {r["output_id"]: r for r in rows}
    for task, oids in sorted(task_to_outputs.items()):
        tr = {"task_id": task, "output_count": len(oids)}
        for label, field in (("D24", "ORC_D24"), ("D48", "ORC_D48"), ("UNION", "ORC_UNION")):
            tr[f"ORC_{label}_TASK"] = all(bool(by_oid[oid][field]) for oid in oids)
        for method in ("NLL", "SUPPORT_NLL"):
            for label in ("D24", "D48", "UNION"):
                tr[f"TOP2_{method}_{label}_TASK"] = all(bool(by_oid[oid][f"TOP2_{method}_{label}"]) for oid in oids)
        task_rows.append(tr)

    for label in ("D24", "D48", "UNION"):
        summary[f"ORC_{label}_TASKS"] = sum(bool(r[f"ORC_{label}_TASK"]) for r in task_rows)
    for method in ("NLL", "SUPPORT_NLL"):
        for label in ("D24", "D48", "UNION"):
            summary[f"TOP2_{method}_{label}_TASKS"] = sum(bool(r[f"TOP2_{method}_{label}_TASK"]) for r in task_rows)

    miss = sorted(r["output_id"] for r in rows if not r["ORC_UNION"])
    rescued_after_512 = sorted(r["output_id"] for r in rows if r["ORC_UNION"] and not r["ORC_UNION_R512"])
    summary["NEW_MISS_POOL_SIZE"] = len(miss)
    summary["NEW_MISS_POOL"] = miss
    summary["UNION_R512_TO_R1024_RESCUES"] = rescued_after_512

    gold_manifest = dict(gold_manifest)

    fields = list(rows[0].keys())
    write_csv(args.output / "OUTPUT_RESULTS.csv", rows, fields)
    write_csv(args.output / "TASK_RESULTS.csv", task_rows, list(task_rows[0].keys()))
    (args.output / "SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.output / "NEW_MISS_POOL.json").write_text(json.dumps({"output_ids": miss}, indent=2, sort_keys=True) + "\n")
    (args.output / "GOLD_PROVENANCE.json").write_text(json.dumps(gold_manifest, indent=2, sort_keys=True) + "\n")
    contract = {
        "d24_archive": str(args.d24), "d48_archive": str(args.d48),
        "generation_rerun": False, "gpu_used": False,
        "archives_validated_before_scoring": True,
        "gold_opened_post_freeze_only": True,
        "primary": "ORC/FIX R1024",
        "top2": "descriptive post-freeze target-blind rankings; not used to retune generation",
    }
    (args.output / "SCORING_CONTRACT.json").write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n")

    report = [
        "# Phase 3 Eval60 d24+d48 post-freeze score",
        "",
        f"- FIX_D24 / ORC: {summary['FIX_D24']}/89",
        f"- FIX_D48 / ORC: {summary['FIX_D48']}/89",
        f"- FIX_UNION / ORC: {summary['FIX_UNION']}/89",
        f"- D24_ONLY / D48_ONLY / SHARED / NEITHER: {summary['D24_ONLY']} / {summary['D48_ONLY']} / {summary['SHARED']} / {summary['NEITHER']}",
        f"- NEW_MISS_POOL: {summary['NEW_MISS_POOL_SIZE']}",
        f"- ORC task-level d24/d48/union: {summary['ORC_D24_TASKS']} / {summary['ORC_D48_TASKS']} / {summary['ORC_UNION_TASKS']} of 60",
        f"- Top2 NLL output d24/d48/union: {summary['TOP2_NLL_D24_OUTPUTS']} / {summary['TOP2_NLL_D48_OUTPUTS']} / {summary['TOP2_NLL_UNION_OUTPUTS']} of 89",
        f"- Top2 NLL task d24/d48/union: {summary['TOP2_NLL_D24_TASKS']} / {summary['TOP2_NLL_D48_TASKS']} / {summary['TOP2_NLL_UNION_TASKS']} of 60",
        f"- Top2 support->NLL output d24/d48/union: {summary['TOP2_SUPPORT_NLL_D24_OUTPUTS']} / {summary['TOP2_SUPPORT_NLL_D48_OUTPUTS']} / {summary['TOP2_SUPPORT_NLL_UNION_OUTPUTS']} of 89",
        f"- Top2 support->NLL task d24/d48/union: {summary['TOP2_SUPPORT_NLL_D24_TASKS']} / {summary['TOP2_SUPPORT_NLL_D48_TASKS']} / {summary['TOP2_SUPPORT_NLL_UNION_TASKS']} of 60",
        "",
        "Top-2 numbers are descriptive post-freeze rankings; ORC/FIX is the primary frozen candidate-coverage result.",
    ]
    (args.output / "REPORT.md").write_text("\n".join(report) + "\n")

    # Final result hashes.
    hashes = {}
    for p in sorted(args.output.iterdir()):
        if p.is_file() and p.name != "HASHES.json":
            hashes[p.name] = sha256(p)
    (args.output / "HASHES.json").write_text(json.dumps({"files": hashes}, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
