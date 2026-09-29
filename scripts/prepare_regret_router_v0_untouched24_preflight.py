"""Freeze the additive, leakage-free Untouched24 cohort before GPU work."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ID_RE = __import__("re").compile(r"\b[0-9a-f]{8}:o[0-9]+\b")
TRANCHE_A = [
    "dfadab01:o0", "8b7bacbf:o1", "13e47133:o1", "e3721c99:o0",
    "4e34c42c:o0", "247ef758:o0", "cbebaa4b:o1", "7b3084d4:o0",
    "dd6b8c4b:o0", "36a08778:o1", "88bcf3b4:o0", "80a900e0:o0",
]
DEV_DIRS = (
    "decoder_pruning_parallel_v1",
    "decoder_retrieval_diagnosis_v1",
    "regret_budget_and_retrieval_v1",
    "regret_budget_router_v1",
)


def sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_hash(value: Any) -> str:
    return sha(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def output_ids_in(path: Path) -> set[str]:
    result: set[str] = set()
    for item in sorted(path.rglob("*")):
        if item.is_file():
            result.update(OUTPUT_ID_RE.findall(item.read_text(encoding="utf-8", errors="ignore")))
    return result


def main() -> None:
    repo = ROOT
    out = repo / "analysis" / "regret_router_v0_untouched24"
    compact = repo / "artifacts" / "eval60_compact_analysis_v2" / "turbodfs_v5" / "v5_outputs.csv"
    with compact.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    misses = [row["output_id"] for row in rows if row["union_greedy_v5_hit"].lower() == "false"]
    if len(rows) != 89 or len(misses) != 56 or len(set(misses)) != 56:
        raise RuntimeError("expected the frozen original 56-output current-union miss pool")
    previous = json.loads((repo / "analysis" / "regret_router_v0_untouched12" / "UNTOUCHED12_MANIFEST.json").read_text())
    original_contract = json.loads((repo / "analysis" / "regret_router_v0_untouched12" / "ROUTER_V0_CONTRACT.json").read_text())
    if previous.get("output_ids") != TRANCHE_A:
        raise RuntimeError("original tranche-A membership differs from the immutable Untouched12 freeze")
    per_dir = {name: sorted(output_ids_in(repo / "analysis" / name)) for name in DEV_DIRS}
    used = set().union(*(set(values) for values in per_dir.values()))
    eligible = sorted(set(misses) - used - set(TRANCHE_A), key=lambda key: (sha(key.encode("utf-8")), key))
    if len(eligible) < 12:
        raise RuntimeError(f"need 12 new leakage-free outputs, found {len(eligible)}")
    tranche_b = eligible[:12]
    combined = TRANCHE_A + tranche_b
    leakage = sorted(set(combined) & used)
    if len(combined) != 24 or len(set(combined)) != 24 or leakage:
        raise RuntimeError("invalid additive cohort or router-development leakage")
    static = {
        "experiment_id": "REGRET_ROUTER_V0_UNTOUCHED24_VALIDATION",
        "rule_id": "REGRET_ROUTER_V0_STATIC_DEPTH_BUDGET",
        "router_decision_changed": False,
        "execution_protocol_changed": True,
        "reason": "exact resume engine unavailable",
        "original_router_contract_sha256": original_contract["contract_sha256"],
        "policy": "CUMULATIVE_REGRET_r=4.0",
        "candidate_cap": 32,
        "depth_budgets": {"12": 1024, "24": 4096, "48": 1024},
        "shadow_depth_budgets": {"12": 4096, "48": 4096},
        "views": ["identity", "flip_ud", "transpose", "anti_transpose"],
        "forbidden": ["gold", "confidence", "task_exceptions", "view_exceptions", "2048_tier", "candidate_cap_change"],
        "gold_accessed": False,
        "status": "FROZEN_BEFORE_GPU",
    }
    static["contract_sha256"] = canonical_hash(static)
    manifest = {
        "experiment_id": static["experiment_id"],
        "status": "FROZEN_BEFORE_GPU",
        "scope": "held-out nonblind development; original current-union miss pool was historically established",
        "tranche_a": TRANCHE_A,
        "tranche_a_sha256": canonical_hash(TRANCHE_A),
        "tranche_b": tranche_b,
        "tranche_b_sha256": canonical_hash(tranche_b),
        "output_ids": combined,
        "output_ids_sha256": canonical_hash(combined),
        "total_unique_outputs": 24,
        "selection": "preserve original tranche A; sort SHA256(output_id) over remaining leakage-free miss outputs and take first 12 as tranche B",
        "original_miss_pool_path": str(compact.relative_to(repo)).replace("\\", "/"),
        "original_miss_pool_sha256": sha(compact.read_bytes()),
        "original_miss_pool_count": 56,
        "eligible_count_after_exclusions": len(eligible),
        "gold_accessed": False,
    }
    audit = {
        "status": "PASS",
        "development_descendants": per_dir,
        "original_miss_count": 56,
        "tranche_a_preserved_exactly": True,
        "tranche_b_count": len(tranche_b),
        "combined_count": len(combined),
        "LEAKAGE_WITH_ROUTER_DEV": 0,
        "leakage_output_ids": leakage,
        "gold_accessed": False,
    }
    dump(out / "ROUTER_V0_CONTRACT.json", static)
    dump(out / "UNTOUCHED24_MANIFEST.json", manifest)
    dump(out / "LEAKAGE_AUDIT.json", audit)
    source_commit = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    dump(out / "provenance.json", {"source_commit": source_commit, "gpu_used": False, "gold_accessed": False, "status": "PRELAUNCH_FROZEN", "static_contract_sha256": static["contract_sha256"], "cohort_sha256": manifest["output_ids_sha256"]})
    hashes = {item.name: sha(item.read_bytes()) for item in sorted(out.iterdir()) if item.is_file()}
    dump(out / "PRELAUNCH_HASHES.json", hashes)
    print(json.dumps({"tranche_b": tranche_b, "contract_sha256": static["contract_sha256"], "cohort_sha256": manifest["output_ids_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
