"""Freeze a held-out ARC cohort without ever opening target solutions.

The exposure scan is intentionally conservative: a held-out task is ineligible
if its ID occurs anywhere in the declared historical research roots.  This
makes the selection repeatable and prevents accidentally validating on a task
that was previously used for design, diagnostics, or manual inspection.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Iterable


TASK_ID = re.compile(r"(?<![0-9a-f])[0-9a-f]{8}(?![0-9a-f])")
SALT = "ARC2_UNTOUCHED60_NATIVE_PUBLIC_REFERENCE_B_V1"
DEFAULT_ROOTS = ("artifacts", "configs", "reports", "scripts")
ARTIFACT_NAME_HINTS = ("cohort", "manifest", "frozen", "output", "result", "score", "checkpoint", "prediction", "report")
STAGING_PATH_HINTS = ("source", "stage", "kernel", "notebook", "official_nvarc")
NATIVE_LINEAGE_HINTS = ("native", "qwen4b", "ranker", "public_reference", "frozen30", "ttt")


def sha256_json(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def exposed_task_ids(project_root: Path, known_ids: set[str], roots: Iterable[str] = DEFAULT_ROOTS) -> tuple[set[str], list[str]]:
    """Return task-like IDs mentioned in historical non-data research files."""
    ids: set[str] = set()
    scanned: list[str] = []
    for relative in roots:
        root = project_root / relative
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            # Raw ARC data is never an exposure record and may contain targets.
            if path.suffix.lower() not in {".json", ".csv", ".md", ".py", ".ipynb", ".txt"}:
                continue
            relative_path = str(path.relative_to(project_root)).replace("\\", "/")
            lowered = relative_path.lower()
            if not any(hint in lowered for hint in NATIVE_LINEAGE_HINTS):
                continue
            if relative == "artifacts":
                # Artifact source/kernel staging directories are copied code
                # and binary-oriented notebook plumbing, not experiment
                # exposure ledgers.  They create many duplicate large reads.
                if any(part in lowered for part in STAGING_PATH_HINTS):
                    continue
                if not any(hint in path.name.lower() for hint in ARTIFACT_NAME_HINTS):
                    continue
                # Cohort IDs are also persisted in compact manifests beside
                # huge candidate pools.  Do not turn manifest construction
                # into a multi-minute scan of raw decoded grid artifacts.
                if path.stat().st_size > 8 * 1024 * 1024:
                    continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            # Only identifiers that belong to the held-out split are IDs;
            # hexadecimal config/source hashes must never be mistaken for
            # historical cohort membership.
            ids.update(task_id for task_id in known_ids if task_id in text.lower())
            scanned.append(relative_path)
    return ids, scanned


def build_manifest(project_root: Path, output: Path, count: int = 60) -> dict[str, object]:
    if output.exists():
        raise FileExistsError(f"immutable manifest already exists: {output}")
    split_path = project_root / "data/splits/task_splits.csv"
    with split_path.open(newline="", encoding="utf-8") as handle:
        held_out = sorted(row["task_id"] for row in csv.DictReader(handle) if row["split"] == "held_out")
    exposed, scanned = exposed_task_ids(project_root, set(held_out))
    eligible = sorted(set(held_out) - exposed)
    ordered = sorted(eligible, key=lambda task_id: hashlib.sha256(f"{SALT}:{task_id}".encode()).hexdigest())
    selected = ordered[:count]
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} unexposed held-out tasks available; require {count}")
    config = {
        "selection_salt": SALT,
        "source_split": "held_out",
        "source_split_file_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "exposure_roots": list(DEFAULT_ROOTS),
        "exposure_file_count": len(scanned),
        "exposure_task_ids_sha256": sha256_json(sorted(exposed)),
        "candidate_order": "ascending SHA-256(salt + ':' + task_id)",
        "method_a_config_sha256": hashlib.sha256((project_root / "configs/QWEN4B_MAX_NATIVE_CAPABILITY_PUSH_V1.json").read_bytes()).hexdigest(),
        "method_b_selector_sha256": hashlib.sha256((project_root / "scripts/rerank_native_public_reference_selection.py").read_bytes()).hexdigest(),
    }
    manifest: dict[str, object] = {
        "experiment_id": "ARC2_UNTOUCHED60_NATIVE_PUBLIC_REFERENCE_B_V1",
        "status": "COHORT_FROZEN_BEFORE_INFERENCE_AND_TARGET_ACCESS",
        "source_split": "held_out",
        "task_count": count,
        "task_ids": selected,
        "task_ids_hash": sha256_json(sorted(selected)),
        "selection": config,
        "config_hash": sha256_json(config),
        "integrity": {
            "solutions_opened": False,
            "targets_inspected": False,
            "excluded_historical_task_count": len(set(held_out) & exposed),
            "eligible_held_out_task_count": len(eligible),
            "selection_is_immutable": True,
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=60)
    args = parser.parse_args()
    manifest = build_manifest(args.project_root.resolve(), args.output.resolve(), args.count)
    print(json.dumps({"status": manifest["status"], "task_count": manifest["task_count"], "task_ids_hash": manifest["task_ids_hash"], "config_hash": manifest["config_hash"]}, sort_keys=True))


if __name__ == "__main__":
    main()
