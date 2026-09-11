"""Freeze a deterministic, V1-comparable development-only V2 pilot set."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    confirmation = json.loads((ROOT / "configs" / "llm_confirmation_dev_100.json").read_text(encoding="utf-8"))
    retrieval = json.loads((ROOT / "configs" / "llm_retrieval_diagnostic_v1.json").read_text(encoding="utf-8"))
    excluded = set(retrieval["task_ids"])
    eligible = [task_id for task_id in confirmation["task_ids"] if task_id not in excluded]
    task_ids = sorted(eligible, key=lambda task_id: hashlib.sha256(f"LLM_PROGRAM_SYNTHESIS_V2_PILOT:{task_id}".encode()).hexdigest())[:50]
    payload = {"config_id": "LLM_PROGRAM_SYNTHESIS_V2_PILOT_50", "split": "development", "selection": "deterministic SHA-256 order from V1 confirmation development set; excludes all 12 retrieval-diagnostic tasks", "task_count": len(task_ids), "task_ids": task_ids, "frozen_config": "configs/frozen_llm_program_synthesis_v2.json", "conditions": ["symbolic", "direct_parameter_ablation"], "protocol": "No solution file during inference. Each condition checkpoints every task; compare only after its predictions freeze."}
    path = ROOT / "configs" / "llm_program_synthesis_v2_pilot_50.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(path), "task_count": len(task_ids), "overlap_with_v1_confirmation": len(set(task_ids) & set(confirmation["task_ids"])), "excluded_retrieval": len(excluded)}, indent=2))


if __name__ == "__main__":
    main()
