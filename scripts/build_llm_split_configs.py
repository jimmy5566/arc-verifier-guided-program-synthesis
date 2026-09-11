"""Write deterministic task manifests for frozen LLM split evaluations."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from arc.io import discover_dataset_files, load_challenges


ROOT = Path(__file__).resolve().parents[1]


def write_config(split: str, config_id: str, filename: str) -> None:
    with (ROOT / "data" / "splits" / "task_splits.csv").open(encoding="utf-8", newline="") as handle:
        task_ids = sorted(row["task_id"] for row in csv.DictReader(handle) if row["split"] == split)
    payload = {
        "config_id": config_id,
        "selection": f"existing fixed {split} split; lexicographic task-ID order; no task-specific filtering",
        "split": split,
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "frozen_config": "configs/frozen_llm_config_v1.json",
        "protocol": "No solution file is opened by inference. Checkpoint after every task. Score only after all declared predictions are frozen.",
    }
    path = ROOT / "configs" / filename
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(path.relative_to(ROOT)), "task_count": len(task_ids), "split": split}))


def main() -> None:
    write_config("held_out", "LLM_HELD_OUT_FROZEN_V1", "llm_held_out_frozen_v1.json")
    write_config("challenge_like", "LLM_CHALLENGE_LIKE_FROZEN_V1", "llm_challenge_like_frozen_v1.json")
    task_ids = sorted(load_challenges(discover_dataset_files(ROOT / "data" / "raw")["training_challenges"]))
    payload = {
        "config_id": "LLM_FULL_1000_FROZEN_V1",
        "selection": "all official training challenges; lexicographic task-ID order; execute only after frozen split evaluations",
        "split": "official_training_all",
        "task_count": len(task_ids),
        "task_ids": task_ids,
        "frozen_config": "configs/frozen_llm_config_v1.json",
        "protocol": "No solution file is opened by inference. Checkpoint after every task. Score only after all 1000 predictions are frozen.",
    }
    path = ROOT / "configs" / "llm_full_1000_frozen_v1.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"config": str(path.relative_to(ROOT)), "task_count": len(task_ids), "split": "official_training_all"}))


if __name__ == "__main__":
    main()
