from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from novel_training_data_v1.pipeline import digest
from training_data_v2.pipeline import sha256_file


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    artifacts = root / "artifacts/novel_training_data_v1_1"
    processed = root / "data/processed/novel_training_data_v1_1"
    holdout = json.loads((artifacts / "V1_1_HOLDOUT_FREEZE.json").read_text(encoding="utf-8"))
    holdout_ids = set(holdout["base_puzzle_ids"])
    checks: dict[str, bool] = {}
    counts = {}
    materialized_ids: set[str] = set()
    for split, filename in (("train", "NOVEL_TRAIN_SHARD_MANIFEST.json"), ("validation", "NOVEL_VAL_SHARD_MANIFEST.json")):
        manifest = json.loads((artifacts / filename).read_text(encoding="utf-8"))
        row_count = 0
        for shard in manifest["shards"]:
            path = processed / shard["logical_name"]
            checks[f"{split}:{shard['logical_name']}:sha256"] = path.exists() and sha256_file(path) == shard["sha256"]
            table = pq.read_table(path, columns=["base_puzzle_id", "split", "final_training_role"])
            rows = table.to_pylist()
            row_count += len(rows)
            for row in rows:
                base_id = row["base_puzzle_id"]
                if base_id in materialized_ids:
                    raise RuntimeError(f"duplicate materialized row: {base_id}")
                materialized_ids.add(base_id)
                checks[f"{split}:{base_id}:split"] = row["split"] == split
                checks[f"{split}:{base_id}:role"] = all(token not in row["final_training_role"] for token in ("HOLDOUT", "HARD_EXCLUDE", "QUARANTINE"))
        counts[split] = row_count
    checks["holdout_absent"] = not bool(materialized_ids & holdout_ids)
    report = json.loads((artifacts / "REPORT.json").read_text(encoding="utf-8"))
    checks["train_count"] = counts["train"] == report["accepted_episode_counts"]["train"]
    checks["validation_count"] = counts["validation"] == report["accepted_episode_counts"]["validation"]
    fingerprint = json.loads((artifacts / "NOVEL_DATASET_FINGERPRINT.json").read_text(encoding="utf-8"))
    checks["portable_fingerprint"] = (
        not fingerprint["absolute_paths"]
        and digest(fingerprint["logical_manifest"]) == fingerprint["fingerprint_sha256"]
    )
    systematicity = json.loads((artifacts / "COMPOSITIONAL_SYSTEMATICITY_AUDIT.json").read_text(encoding="utf-8"))
    for split, row in systematicity["upstream_files"].items():
        path = root / row["logical_name"]
        checks[f"systematicity:{split}:sha256"] = path.exists() and sha256_file(path) == row["sha256"]
    gate = json.loads((artifacts / "SCIENTIFIC_TRAINING_GATE_V1_1.json").read_text(encoding="utf-8"))
    checks["gpu_training_not_started"] = gate["GPU_TRAINING_STARTED"] is False
    checks["gate_derived"] = gate["status"] == ("PASS_READY_FOR_GPU_BENCHMARK" if all(gate["checks"].values()) else "FAIL_NOT_READY")
    failed = sorted(key for key, value in checks.items() if not value)
    result = {"status": "PASS" if not failed else "FAIL", "checks": len(checks), "failed": failed, "rows": counts}
    print(json.dumps(result, sort_keys=True))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
