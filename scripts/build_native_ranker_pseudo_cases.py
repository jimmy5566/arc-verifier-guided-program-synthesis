"""Build train-pair-only leave-one-out native pseudo-test cases."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _hash(values: list[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(values), separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-cohort", type=Path, required=True); parser.add_argument("--cohort-key", default="development_pseudo_validation"); parser.add_argument("--challenge-path", type=Path, required=True)
    parser.add_argument("--output-challenges", type=Path, required=True); parser.add_argument("--output-cohort", type=Path, required=True); parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()
    config = json.loads(args.task_cohort.read_text(encoding="utf-8")); cohort = config.get(args.cohort_key, config); task_ids = list(cohort["task_ids"])
    if _hash(task_ids) != cohort["task_ids_hash"]: raise ValueError("development cohort hash mismatch")
    challenges = json.loads(args.challenge_path.read_text(encoding="utf-8")); pseudo: dict[str, Any] = {}; manifest: dict[str, Any] = {}
    for task_id in task_ids:
        train = challenges[task_id]["train"]
        if len(train) < 2: raise ValueError(f"{task_id}: pseudo-validation requires at least two train pairs")
        for held_out_index, example in enumerate(train):
            case_id = f"{task_id}__train_{held_out_index}"
            pseudo[case_id] = {"train": [row for index, row in enumerate(train) if index != held_out_index], "test": [{"input": example["input"]}]}
            manifest[case_id] = {"source_task_id": task_id, "held_out_train_index": held_out_index}
    ids = sorted(pseudo)
    for path in (args.output_challenges, args.output_cohort, args.output_manifest): path.parent.mkdir(parents=True, exist_ok=True)
    args.output_challenges.write_text(json.dumps(pseudo), encoding="utf-8")
    args.output_cohort.write_text(json.dumps({"task_ids": ids, "task_ids_hash": _hash(ids), "kind": "train_pair_leave_one_out_pseudo_cases", "source_task_cohort_hash": cohort["task_ids_hash"]}, indent=2) + "\n", encoding="utf-8")
    args.output_manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"pseudo_case_count": len(ids), "task_count": len(task_ids), "pseudo_cohort_hash": _hash(ids)}, sort_keys=True))


if __name__ == "__main__": main()
