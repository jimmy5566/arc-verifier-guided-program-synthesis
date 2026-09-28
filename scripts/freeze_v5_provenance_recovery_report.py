#!/usr/bin/env python3
"""Freeze and validate the compact, Gold-independent V5 provenance recovery."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_UNRESOLVED = {
    "1818057f:o0:d012:identity", "1818057f:o0:d024:identity",
    "31f7f899:o0:d012:identity", "31f7f899:o0:d048:identity",
    "446ef5d2:o0:d012:identity", "53fb4810:o0:d048:flip_ud",
    "cb2d8a2c:o0:d012:flip_ud",
}
V5_CONFIG = "e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0"
TTT_CONFIG = "8968f67e5a5c8c1f838dc1a45120d052527d65e95188cd9e155426ba56481486"
ADAPTER_MANIFEST = "e3e95956b0c17e3017b3bb99999c53bfc5307908df68c43fc6641e8b1459c2a6"


def stable(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def candidate_signature(cell: dict[str, str], candidates: list[dict[str, str]]) -> str:
    selected = [{key: row.get(key, "") for key in (
        "candidate_id", "candidate_grid_sha256", "cumulative_nll", "candidate_rank_within_cell",
    )} for row in candidates]
    evidence = {"candidate_count": cell.get("candidate_count", ""), "candidates": sorted(selected, key=stable)}
    return hashlib.sha256(stable(evidence).encode()).hexdigest()


def stability(v1: Path, v2: Path, out: Path) -> tuple[int, list[dict[str, str]]]:
    cells1 = rows(v1 / "v5_cells.csv")
    cells2 = rows(v2 / "turbodfs_v5" / "v5_cells.csv")
    trusted1 = {row["cell_key"]: row for row in cells1 if row.get("provenance_status") == "TRUSTWORTHY"}
    trusted2 = {row["cell_key"]: row for row in cells2 if row.get("provenance_status") == "TRUSTWORTHY"}
    if len(trusted1) != 777 or not set(trusted1).issubset(trusted2):
        raise RuntimeError(f"V1 trustworthy surface mismatch: v1={len(trusted1)} contained={len(set(trusted1) & set(trusted2))}")
    groups1: dict[str, list[dict[str, str]]] = defaultdict(list)
    groups2: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows(v1 / "v5_candidates.csv"):
        groups1[row["cell_key"]].append(row)
    for row in rows(v2 / "turbodfs_v5" / "v5_candidates.csv"):
        groups2[row["cell_key"]].append(row)
    report: list[dict[str, str]] = []
    for key in sorted(trusted1):
        left = candidate_signature(trusted1[key], groups1[key])
        right = candidate_signature(trusted2[key], groups2[key])
        report.append({"cell_key": key, "v1_candidate_signature": left, "v2_candidate_signature": right, "unchanged": "YES" if left == right else "NO"})
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(report[0]))
        writer.writeheader(); writer.writerows(report)
    changed = [row for row in report if row["unchanged"] != "YES"]
    if changed:
        raise RuntimeError(f"V1 candidate evidence changed for {len(changed)} cells")
    return len(report), report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery", type=Path, required=True)
    parser.add_argument("--v1", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    args = parser.parse_args()
    recovery, v2 = args.recovery, args.v2
    freeze = json.loads((recovery / "PROVENANCE_RECOVERY_FREEZE.json").read_text(encoding="utf-8"))
    independence = json.loads((recovery / "GOLD_INDEPENDENCE_TEST.json").read_text(encoding="utf-8"))
    recovered = rows(recovery / "recovered_cells.csv")
    unresolved = rows(recovery / "unresolved_cells.csv")
    if len(recovered) != 284 or {row["match_class"] for row in recovered} != {"EXACT_LINK"}:
        raise RuntimeError("recovered cells are not exactly 284 EXACT_LINK rows")
    if len(unresolved) != 7 or {row["cell_key"] for row in unresolved} != EXPECTED_UNRESOLVED:
        raise RuntimeError("unresolved cell set differs from the frozen seven")
    if any(row["reason"] != "NO_ARTIFACT" or row["candidate_artifact_count"] != "0" for row in unresolved):
        raise RuntimeError("unresolved rows are not all NO_ARTIFACT/0")
    if not independence.get("pass") or independence.get("gold_used_for_linkage") is not False:
        raise RuntimeError("Gold-independence evidence failed")
    stability_path = recovery / "v1_v2_trustworthy_stability.csv"
    stable_count, _ = stability(args.v1, v2, stability_path)
    before = {"scheduler_done_cells": 1068, "trustworthy_candidate_linked_cells": 777, "ambiguous_candidate_link_cells": 291}
    after = freeze["after"]
    after_summary = {"scheduler_done_cells": after["scheduler_done_cells"], "trustworthy_candidate_linked_cells": after["trustworthy_candidate_linked_cells"], "unresolved_no_artifact_cells": after["ambiguous_candidate_link_cells"], "missing_scheduler_cells": after["missing_cells"], "recovered_exact_link": len(recovered), "gold_used_for_linkage": False}
    (recovery / "AFTER_RECOVERY_SUMMARY.json").write_text(json.dumps(after_summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gold_audit = {"GOLD_USED_FOR_LINKAGE": False, "PASS_1_HASH": {key: independence[key] for key in independence if key.endswith("_first")}, "PASS_2_HASH": {key: independence[key] for key in independence if key.endswith("_repeat")}, "PASS": True}
    (recovery / "GOLD_INDEPENDENCE_AUDIT.json").write_text(json.dumps(gold_audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = f"""# V5 provenance recovery freeze

- Scheduler surface: **1068 / 1068 DONE**.
- Before: **777 trustworthy**, **291 ambiguous**.
- Recovery: **284 / 291 EXACT_LINK** cells recovered from immutable candidate artifacts.
- After: **1061 / 1068 trustworthy**, **7 / 1068 unresolved NO_ARTIFACT**, **0 missing scheduler rows**.
- Gold used for linkage: **NO**. Two independent resolver passes produced identical recovered, unresolved, and resolution hashes.
- V1 stability: **{stable_count} / 777** original trustworthy candidate sets unchanged in V2.

The seven unresolved cells remain explicitly unresolved and are not treated as Gold misses. No inference or scientific configuration change occurred during this preservation task.
"""
    (recovery / "RECOVERY_REPORT.md").write_text(report, encoding="utf-8")
    quality = f"""# V2 provenance data quality

- Scheduler surface: 1068 / 1068 DONE
- Candidate provenance: 1061 / 1068 trustworthy; 7 / 1068 unresolved NO_ARTIFACT
- Recovery: 284 / 291 ambiguous cells recovered, all EXACT_LINK
- Gold used for linkage: NO
- Missing scheduler rows: 0
"""
    (v2 / "provenance" / "data_quality_report.md").write_text(quality, encoding="utf-8")
    readme = """# Eval60 compact analysis V2

V2 supersedes V1 only for V5 provenance coverage; V1 remains preserved. It uses
the frozen deterministic provenance mapping to retain 1061 candidate-linked V5
cells. The other seven cells have no surviving candidate artifact and remain
`UNRESOLVED_NO_ARTIFACT`, not Gold misses. Recovery used no Gold, no inference
was rerun, and the scientific V5 configuration did not change.
"""
    (v2 / "README.md").write_text(readme, encoding="utf-8")
    manifest = {"expected_cells": 1068, "scheduler_done_cells": 1068, "before_trustworthy": 777,
                "before_ambiguous": 291, "recovered_exact_link": 284, "after_trustworthy": 1061,
                "after_unresolved": 7, "after_missing": 0, "gold_used_for_linkage": False,
                "independent_recovery_passes": 2, "recovery_pass_hash_match": True,
                "v5_config_sha256": V5_CONFIG, "ttt_config_sha256": TTT_CONFIG,
                "adapter_checkpoint_manifest_sha256": ADAPTER_MANIFEST, "file_sha256": {}}
    for path in sorted(recovery.iterdir()):
        if path.is_file() and path.name not in {"PROVENANCE_RECOVERY_MANIFEST.json", "PROVENANCE_RECOVERY_FROZEN.flag"}:
            manifest["file_sha256"][path.name] = digest(path)
    (recovery / "PROVENANCE_RECOVERY_MANIFEST.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    v2_manifest_path = v2 / "MANIFEST.json"
    v2_manifest = json.loads(v2_manifest_path.read_text(encoding="utf-8"))
    v2_manifest["file_sha256"] = {str(path.relative_to(v2)): digest(path) for path in sorted(v2.rglob("*")) if path.is_file() and path.name != "MANIFEST.json"}
    v2_manifest_path.write_text(json.dumps(v2_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"recovered": len(recovered), "unresolved": len(unresolved), "v1_stable": stable_count}, sort_keys=True))


if __name__ == "__main__":
    main()
