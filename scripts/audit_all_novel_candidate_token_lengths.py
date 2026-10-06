from pathlib import Path

from novel_training_data_v1.pipeline import (
    WORKERS,
    atomic_json,
    atomic_text,
    classify_candidates,
    split_families,
    tokenize_all_candidate_coverage,
)
import json


def main() -> int:
    root=Path(__file__).resolve().parents[1]
    registry=root/"data/processed/novel_training_data_v1/candidate_registry.parquet"
    accepted,rejected,_=classify_candidates(root,registry)
    _,mapping=split_families(accepted)
    art=root/"artifacts/novel_training_data_v1"
    manifests={
        "train":json.loads((art/"NOVEL_TRAIN_SHARD_MANIFEST.json").read_text(encoding="utf-8"))["shards"],
        "validation":json.loads((art/"NOVEL_VAL_SHARD_MANIFEST.json").read_text(encoding="utf-8"))["shards"],
    }
    receipt=tokenize_all_candidate_coverage(root,accepted,rejected,mapping,manifests,WORKERS)
    gate_path=art/"SCIENTIFIC_TRAINING_GATE.json"
    gate=json.loads(gate_path.read_text(encoding="utf-8"))
    gate["checks"]["all_candidate_tokenizer_coverage"]=receipt["actual_tokenizer_covered"]==receipt["candidate_count"]
    gate["checks"]["tokenizer_audit_pass"] = gate["checks"]["tokenizer_audit_pass"] and receipt["status"]=="PASS"
    ready=all(gate["checks"].values()); gate["status"]="PASS_READY_FOR_GPU_BENCHMARK" if ready else "FAIL_NOT_READY"; gate["DATA_ENGINEERING_READY"]=ready; gate["SCIENTIFIC_TRAINING_READY"]=ready
    atomic_json(gate_path,gate)
    atomic_text(art/"SCIENTIFIC_TRAINING_GATE.md","# Scientific training gate\n\nStatus: **"+gate["status"]+"**.\n\n"+"\n".join(f"- {'PASS' if value else 'FAIL'}: {name}" for name,value in gate["checks"].items())+"\n\nGPU TRAINING STARTED = FALSE\n")
    source=root/"data/processed/novel_training_data_v1/NOVEL_EXCLUSION_LEDGER.parquet"
    (art/"NOVEL_EXCLUSION_LEDGER.parquet").write_bytes(source.read_bytes())
    print(json.dumps(receipt,sort_keys=True)); return 0 if receipt["status"]=="PASS" else 1


if __name__=="__main__": raise SystemExit(main())
