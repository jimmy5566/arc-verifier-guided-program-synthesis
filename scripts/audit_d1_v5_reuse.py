"""CPU-only D1 versus authoritative-V5 identity and parity audit.

No solutions are read.  This audit intentionally fails closed: an added search
or candidate cap makes a control ineligible for exact historical V5 reuse.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from scripts.run_d1_real_decoder_ab import DEPTHS, GROUPS, VIEWS, adapter_records, config_for, output_key, read_json, unit_path, valid_unit

EXPECTED_CONFIG_SHA = "e5beec92c9992930f4d6b468db1806ee2f3b9224e13179af7d4e8bb5897148e0"


def sha_grid(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def current_rows(root: Path, manifest: dict[str, Any]) -> dict[tuple[str, int, str], dict[str, Any]]:
    _config, config_sha = config_for(manifest, "V5_CURRENT")
    result: dict[tuple[str, int, str], dict[str, Any]] = {}
    for output in manifest["cohort_output_ids"]:
        for depth in DEPTHS:
            for group in GROUPS:
                path = unit_path(root, "V5_CURRENT", str(output), depth, group, smoke=False)
                if not valid_unit(path, "V5_CURRENT", str(output), depth, group, config_sha):
                    continue
                for row in json.loads(path.read_text(encoding="utf-8")):
                    result[(str(output), depth, str(row["view"]))] = row
    return result


def historic_rows(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    return {(row["output_id"], int(row["depth"]), row["view"]): row for row in csv.DictReader(path.open(encoding="utf-8", newline=""))}


def historic_candidate_hashes(path: Path) -> dict[tuple[str, int, str], list[str]]:
    values: dict[tuple[str, int, str], list[str]] = {}
    for row in csv.DictReader(path.open(encoding="utf-8", newline="")):
        values.setdefault((row["output_id"], int(row["depth"]), row["view"]), []).append(row["candidate_grid_sha256"])
    return {key: sorted(value) for key, value in values.items()}


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--d1-root", type=Path, required=True)
    parser.add_argument("--historical-dir", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.d1_root.resolve(); history = args.historical_dir.resolve(); report = args.report_dir.resolve(); report.mkdir(parents=True, exist_ok=True)
    manifest = read_json(root / "D1_MANIFEST.json")
    adapters = adapter_records(Path(manifest["adapter_manifest"]))
    historical = historic_rows(history / "v5_cells.csv")
    candidates = historic_candidate_hashes(history / "v5_candidates.csv")
    current = current_rows(root, manifest)
    policy = manifest["policies"]["V5_CURRENT"]
    cap_mismatch = bool(policy.get("max_expanded_nodes")) or bool(policy.get("max_completed_candidates"))
    config_ok = manifest.get("v5_config_sha256") == EXPECTED_CONFIG_SHA
    rows: list[dict[str, Any]] = []
    parity_mismatches = 0
    for output in manifest["cohort_output_ids"]:
        task_id, output_index = output_key(str(output))
        for depth in DEPTHS:
            expected_adapter = str(adapters[(task_id, depth)]["sha256"])
            for view in VIEWS:
                key = (str(output), depth, view); old = historical.get(key); new = current.get(key)
                old_candidate_hashes = candidates.get(key, [])
                new_candidate_hashes = sorted(sha_grid(value["canonical_candidate"]) for value in (new or {}).get("candidates", []) if value.get("valid_grid"))
                comparable = new is not None and old is not None
                parity = "NOT_RERUN" if new is None else "PARITY_CONFIRMED"
                if comparable and (str(new.get("adapter_sha")) != expected_adapter or str(old["checkpoint_sha256"]) != expected_adapter or
                                   str(new.get("termination_reason")) != old["termination_reason"] or int(new.get("candidate_count", -1)) != int(old["candidate_count"]) or
                                   int(new.get("nodes_expanded", -1)) != int(old["nodes_expanded"]) or new_candidate_hashes != old_candidate_hashes):
                    parity = "PARITY_MISMATCH"; parity_mismatches += 1
                rows.append({
                    "output_id": output, "depth": depth, "view": view, "historical_present": old is not None,
                    "historical_v5_config_sha256": old.get("v5_config_sha256") if old else None,
                    "current_v5_config_sha256": manifest.get("v5_config_sha256"), "historical_adapter_sha256": old.get("checkpoint_sha256") if old else None,
                    "expected_adapter_sha256": expected_adapter, "current_adapter_sha256": new.get("adapter_sha") if new else None,
                    "historical_termination_reason": old.get("termination_reason") if old else None, "current_termination_reason": new.get("termination_reason") if new else None,
                    "historical_candidate_count": old.get("candidate_count") if old else None, "current_candidate_count": new.get("candidate_count") if new else None,
                    "historical_nodes_expanded": old.get("nodes_expanded") if old else None, "current_nodes_expanded": new.get("nodes_expanded") if new else None,
                    "candidate_hash_match": None if new is None else new_candidate_hashes == old_candidate_hashes,
                    "parity_class": parity, "gold_compared": False,
                })
    exact_identity = config_ok and not cap_mismatch
    exact_rows = sum(bool(row["historical_present"]) for row in rows)
    outcome = "EXACT_REUSE" if exact_identity and parity_mismatches == 0 else "PARITY_MISMATCH" if parity_mismatches or cap_mismatch else "INSUFFICIENT_IDENTITY"
    write_csv(report / "v5_reuse_cells.csv", rows)
    summary = {
        "V5_EXACT_REUSE": "YES" if outcome == "EXACT_REUSE" else "NO", "V5_REUSED_CELLS": 144 if outcome == "EXACT_REUSE" else 0,
        "V5_NEW_PARITY_CELLS_OBSERVED": len(current), "V5_PARITY_MISMATCHES": parity_mismatches,
        "historical_cells_present": exact_rows, "expected_v5_config_sha256": EXPECTED_CONFIG_SHA,
        "current_v5_config_sha256": manifest.get("v5_config_sha256"), "config_hash_match": config_ok,
        "identity_blockers": [
            "D1 adds max_expanded_nodes=4096 whereas historical V5 declares non_reference_caps=[]" if policy.get("max_expanded_nodes") else None,
            "D1 adds max_completed_candidates=32 whereas historical V5 declares non_reference_caps=[]" if policy.get("max_completed_candidates") else None,
            "D1 uses a separate decoder implementation module; source identity is not the frozen V5 implementation" ,
        ],
        "classification": outcome, "gold_read": False, "scientific_config_changed": "NO",
    }
    summary["identity_blockers"] = [value for value in summary["identity_blockers"] if value]
    (report / "D1_EXECUTION_STATUS.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    lines = ["# V5 exact-reuse audit", "", "This CPU audit did not read evaluation solutions.", "",
             f"- Historical cells available: {exact_rows}/144", f"- Current D1 V5 parity cells observed: {len(current)}", f"- Historical config hash matches: `{config_ok}`", f"- Exact historical reuse: **{summary['V5_EXACT_REUSE']}**", f"- Parity mismatches among rerun cells: {parity_mismatches}", "", "## Identity blockers", ""]
    lines += [f"- {value}" for value in summary["identity_blockers"]]
    lines += ["", "Because identity differs materially, no finalist GPU smoke is authorized by this audit. The shared-queue scheduler may be used only after an explicitly frozen compatible decoder contract exists."]
    (report / "V5_REUSE_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
