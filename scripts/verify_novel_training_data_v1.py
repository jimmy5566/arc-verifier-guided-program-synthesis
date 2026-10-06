"""Independent CPU-only verification of the frozen novel-data-v1 receipts."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def sha256_file(path: Path) -> str:
    value=hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda:handle.read(1024*1024),b""): value.update(block)
    return value.hexdigest()


def load(path: Path): return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    root=Path(__file__).resolve().parents[1]; art=root/"artifacts/novel_training_data_v1"; data=root/"data/processed/novel_training_data_v1"
    train=load(art/"NOVEL_TRAIN_SHARD_MANIFEST.json"); val=load(art/"NOVEL_VAL_SHARD_MANIFEST.json")
    splits={name:set(load(art/file)["families"]) for name,file in (("train","NOVEL_TRAIN_FAMILIES.json"),("validation","NOVEL_VALIDATION_FAMILIES.json"),("holdout","NOVEL_HOLDOUT_FAMILIES.json"))}
    checks={}; totals={"train":0,"validation":0}; families={"train":set(),"validation":set()}
    for split,manifest in (("train",train),("validation",val)):
        for row in manifest["shards"]:
            path=data/row["logical_name"]
            checks[f"{split}:{row['logical_name']}:exists"]=path.exists()
            if not path.exists(): continue
            checks[f"{split}:{row['logical_name']}:sha256"]=sha256_file(path)==row["sha256"]
            table=pq.read_table(path,columns=["generator_family","final_training_role","supervised_token_count"])
            values=table.to_pylist(); totals[split]+=len(values); families[split].update(item["generator_family"] for item in values)
            checks[f"{split}:{row['logical_name']}:rows"]=len(values)==row["rows"]
            checks[f"{split}:{row['logical_name']}:roles"]=all(item["final_training_role"]=="DEFENSIBLE_NOVEL_TRAINABLE" for item in values)
            checks[f"{split}:{row['logical_name']}:supervision"]=all(item["supervised_token_count"]>0 for item in values)
    checks["train_rows_exact"]=totals["train"]==80856; checks["validation_rows_exact"]=totals["validation"]==10226
    checks["train_families_exact"]=families["train"]==splits["train"]; checks["validation_families_exact"]=families["validation"]==splits["validation"]
    checks["split_family_overlap_zero"]=not ((splits["train"]&splits["validation"])|(splits["train"]&splits["holdout"])|(splits["validation"]&splits["holdout"]))
    checks["holdout_absent_from_shards"]=not ((families["train"]|families["validation"])&splits["holdout"])
    replay=root/"data/processed/arc_training_v2_1/train/replay/replay-00000.parquet"
    checks["v2_1_replay_unchanged"]=sha256_file(replay)=="32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc"
    fingerprint=load(art/"NOVEL_DATASET_FINGERPRINT.json"); checks["portable_fingerprint"]=fingerprint["status"]=="PASS" and not fingerprint["absolute_paths"]
    assets=load(art/"RAW_NOVEL_ASSET_MANIFEST.json"); raw=root/"data/raw/novel"; asset_failures=[]
    for item in assets["assets"]:
        path=raw/item["logical_name"]
        if not path.exists() or sha256_file(path)!=item["sha256"]: asset_failures.append(item["logical_name"])
    checks["raw_asset_hashes"]=not asset_failures
    novelty=load(art/"NOVELTY_AUDIT.json"); checks["all_overlap_checks"]=all(novelty["overlap_checks"].values())
    coverage=load(art/"ALL_CANDIDATE_TOKENIZATION_COVERAGE.json"); checks["all_candidate_actual_tokenizer_coverage"]=coverage["status"]=="PASS" and coverage["actual_tokenizer_covered"]==100901 and not coverage["holdout_labels_or_ids_materialized"]
    compact_ledger=art/"NOVEL_EXCLUSION_LEDGER.parquet"; processed_ledger=data/"NOVEL_EXCLUSION_LEDGER.parquet"; checks["compact_exclusion_ledger_exact"]=compact_ledger.exists() and sha256_file(compact_ledger)==sha256_file(processed_ledger)
    dry=load(art/"CPU_DATALOADER_DRY_RUN.json"); checks["dataloader_dry_run"]=dry["status"]=="PASS" and not dry["holdout_or_quarantine_sampled"]
    gate=load(art/"SCIENTIFIC_TRAINING_GATE.json"); checks["derived_gate"]=gate["status"]=="PASS_READY_FOR_GPU_BENCHMARK" and all(gate["checks"].values())
    checks["gpu_training_not_started"]=gate["GPU_TRAINING_STARTED"] is False
    result={"status":"PASS" if all(checks.values()) else "FAIL","checks":checks,"totals":totals,"family_counts":{key:len(value) for key,value in splits.items()},"raw_asset_failures":asset_failures,"v2_1_replay_sha256":sha256_file(replay),"novel_dataset_fingerprint":fingerprint["fingerprint_sha256"],"GPU_TRAINING_STARTED":False}
    (art/"FINAL_VERIFICATION.json").write_text(json.dumps(result,sort_keys=True,separators=(",",":"))+"\n",encoding="utf-8",newline="\n")
    print(json.dumps(result,sort_keys=True)); return 0 if result["status"]=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
