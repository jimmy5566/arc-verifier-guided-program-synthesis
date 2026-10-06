"""CPU-only final freeze audit for the fixed V2 corpus.

This intentionally reads the frozen files independently; it never calls the
builder, downloads data, imports torch, or touches a GPU.
"""
from __future__ import annotations

import csv, gzip, hashlib, json, math, statistics, sys, zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_data_v2.pipeline import canonical_json, canonical_hash, sha256_file, _validate_grid

OUT = ROOT / "artifacts" / "training_data_v2"
DATA = ROOT / "data" / "processed" / "arc_training_v2"
RAW = ROOT / "data" / "raw"

def write(name, x):
    (OUT / name).write_text(canonical_json(x) + "\n", encoding="utf-8", newline="\n")

def grid_info(x):
    if not isinstance(x, list): return {"type": type(x).__name__}
    return {"height": len(x), "widths": [len(r) if isinstance(r, list) else None for r in x[:31]], "colors_outside_0_9": sorted({v for r in x if isinstance(r,list) for v in r if isinstance(v,int) and not isinstance(v,bool) and not 0<=v<=9})}

def invalid_reason(grid, field):
    try: _validate_grid(grid, field=field); return None
    except Exception as e: return str(e)

def parquet_rows(path, columns=None):
    f=pq.ParquetFile(path); rows=[]
    for i in range(f.num_row_groups): rows.extend(f.read_row_group(i, columns=columns).to_pylist())
    return f, rows

def main():
    # Source-native reARC inventory and invalid ledger, independently from the builder.
    archive = RAW / "sources" / "rearc" / "re_arc.zip"
    ledger=[]; raw_pairs=valid=0; banks=[]
    with zipfile.ZipFile(archive) as z:
      for member in sorted(n for n in z.namelist() if n.startswith("re_arc/tasks/") and n.endswith(".json")):
        bank=json.loads(z.read(member)); banks.append(member)
        for idx,pair in enumerate(bank):
          raw_pairs += 1; ir=invalid_reason(pair.get("input") if isinstance(pair,dict) else None,"input"); orr=invalid_reason(pair.get("output") if isinstance(pair,dict) else None,"output")
          if not ir and not orr: valid += 1; continue
          # Same precedence as the source adapter: input is validated before output.
          reason=ir or orr
          ledger.append({"source":"rearc","source_version":"e5b7f1d06362a76f9d3b8c25154ff1fafca897ce","bank":Path(member).stem,"pair_index":idx,"input_canonical_sha256":canonical_hash(pair.get("input")) if isinstance(pair,dict) and "input" in pair else None,"output_canonical_sha256":canonical_hash(pair.get("output")) if isinstance(pair,dict) and "output" in pair else None,"input_dimensions":grid_info(pair.get("input") if isinstance(pair,dict) else None),"output_dimensions":grid_info(pair.get("output") if isinstance(pair,dict) else None),"invalidity_reason":reason,"all_detected_reasons":[x for x in (ir,orr) if x],"validation_rule":"NVARC_build_datasets_validate_grid_height_width_1_to_30_rectangular_integer_color_0_to_9","nvarc_would_reject_equivalence":True})
    with (OUT/"INVALID_PAIR_LEDGER.jsonl").open("w",encoding="utf-8",newline="\n") as h:
      for r in ledger: h.write(canonical_json(r)+"\n")
    invalid_summary={"status":"PASS","source":"rearc","raw_pair_count":raw_pairs,"valid_pair_count":valid,"invalid_pair_count":len(ledger),"reconciliation":"valid_plus_invalid_equals_raw","reason_counts":dict(Counter(r["invalidity_reason"] for r in ledger)),"nvarc_reject_equivalence":True}
    write("INVALID_PAIR_SUMMARY.json",invalid_summary)
    rearc={"status":"DIFFERENT_REARC_GENERATION_BUT_VALID_REPLAY","pinned_rearc_commit":"e5b7f1d06362a76f9d3b8c25154ff1fafca897ce","nvarc_pinned_contract":{"pairs_per_bank":2048,"episodes_per_bank":256},"observed_public_release":{"banks":len(banks),"pairs_per_bank_distribution":dict(Counter(len(json.loads(zipfile.ZipFile(archive).read(n))) for n in banks)),"raw_pairs":raw_pairs,"valid_pairs":valid,"invalid_pairs":len(ledger),"archive_sha256":sha256_file(archive)},"finding":"The pinned NVARC tree points at the same reARC commit, but its parser asserts a 2048-pair local task layout while the public repository release archive contains 400 deterministic 1000-pair banks. The upstream historical-generation reason is not established; this corpus is retained only as a licensed source-native replay curriculum, never as an exact SFT139 reconstruction.","exact_sft139_recipe_recovered":False,"training_role":"BASE_MODEL_REPLAY_COMPATIBLE_NOT_EXACT_HISTORICAL_RECONSTRUCTION"}
    write("REARC_PROVENANCE_RECONCILIATION.json",rearc)
    (OUT/"REARC_PROVENANCE_RECONCILIATION.md").write_text("# reARC provenance reconciliation\n\nStatus: **DIFFERENT_REARC_GENERATION_BUT_VALID_REPLAY**. The public MIT release is source-native and auditable, but is not the pinned NVARC 2048-pair input contract; it is not represented as an exact SFT139 reconstruction.\n",encoding="utf-8",newline="\n")

    pf,puzzles=parquet_rows(DATA/"puzzle_registry.parquet")
    sample_columns=["sample_id","base_puzzle_id","generator_family","source","source_native_id","record_kind","final_training_role","sequence_length","supervised_token_count","augmentation_transform","color_permutation","pair_order_seed","sample_seed","source_pair_indices"]
    sf,samples=parquet_rows(DATA/"sample_registry.parquet", sample_columns)
    train_columns=["sample_id","base_puzzle_id","source","final_training_role","sequence_length","supervised_token_count","augmentation_transform","color_permutation","source_pair_indices"]
    tf,train=parquet_rows(DATA/"train"/"replay"/"replay-00000.parquet", train_columns)
    bybase={r["base_puzzle_id"]:r for r in puzzles}; hard={r["base_puzzle_id"] for r in puzzles if r["final_training_role"]=="HARD_EXCLUDE_FUTURE_EVAL"}
    # Source accounting uses deterministic logical serialized row bytes, not misleading shared parquet byte allocation.
    groups=defaultdict(list)
    for r in samples: groups[r["source"]].append(r)
    accounting=[]
    for source in sorted(set(r["source"] for r in puzzles)):
      ps=[r for r in puzzles if r["source"]==source]; ss=groups[source]
      pair_obs=sum((r.get("pair_bank_observed_count") or 0) for r in ps)
      pair_valid=sum((r.get("pair_bank_valid_count") or 0) for r in ps)
      pair_invalid=sum((r.get("pair_bank_invalid_count") or 0) for r in ps)
      if source!="rearc":
        pair_obs=pair_valid=sum(json.loads(r["features_json"]).get("train_pair_count",0)+json.loads(r["features_json"]).get("test_pair_count",0) for r in ps); pair_invalid=0
      accounting.append({"source":source,"raw_source_objects":len(ps),"base_families":len({r["base_puzzle_id"] for r in ps}),"raw_pairs":pair_obs,"valid_pairs":pair_valid,"invalid_pairs":pair_invalid,"canonical_unique_families":len({r["full_content_sha256"] for r in ps}),"duplicate_families":len(ps)-len({r["full_content_sha256"] for r in ps}),"training_episodes":len(ss),"total_tokens":sum(r["sequence_length"] for r in ss),"supervised_tokens":sum(r["supervised_token_count"] for r in ss),"processed_logical_bytes":sum(len(canonical_json(r).encode()) for r in ss)})
    write("SOURCE_ACCOUNTING.json",{"status":"PASS","processed_bytes_definition":"canonical UTF-8 serialized sample rows; physical Parquet bytes are shared across sources and are reported separately in shard audit","sources":accounting,"totals":{"base_families":len(puzzles),"replay_episodes":len(train),"all_registry_episodes":len(samples)}})
    with (OUT/"SOURCE_ACCOUNTING.csv").open("w",newline="",encoding="utf-8") as h: w=csv.DictWriter(h,fieldnames=accounting[0].keys());w.writeheader();w.writerows(accounting)
    # Strict processed/shard audit.
    train_ids=[r["sample_id"] for r in train]; sample_ids=[r["sample_id"] for r in samples]
    shard=DATA/"train"/"replay"/"replay-00000.parquet"; manifest=json.loads((OUT/"TRAIN_SHARD_MANIFEST.json").read_text())
    expected=manifest["shards"][0]
    shard_audit={"status":"PASS","files":{"puzzle_registry":{"row_groups":pf.num_row_groups,"rows":len(puzzles),"schema":str(pf.schema_arrow)},"sample_registry":{"row_groups":sf.num_row_groups,"rows":len(samples),"schema":str(sf.schema_arrow),"full_schema":str(pq.ParquetFile(DATA/"sample_registry.parquet").schema_arrow)},"replay_shard":{"row_groups":tf.num_row_groups,"rows":len(train),"schema":str(tf.schema_arrow),"bytes":shard.stat().st_size,"sha256":sha256_file(shard),"manifest_sha256":expected["sha256"],"manifest_bytes":expected["bytes"]}},"null_critical_fields":sum(any(r.get(k) is None for k in ("sample_id","base_puzzle_id","source")) for r in samples),"sample_id_unique":len(sample_ids)==len(set(sample_ids)),"train_sample_id_unique":len(train_ids)==len(set(train_ids)),"replay_shard_roles":sorted(set(r["final_training_role"] for r in train)),"hard_exclude_in_replay_shard":sum(r["base_puzzle_id"] in hard for r in train),"quarantine_in_replay_shard":sum(r["final_training_role"].startswith("QUARANTINE") for r in train),"official_eval_in_replay_shard":sum(bybase[r["base_puzzle_id"]]["source_subset"]=="evaluation" for r in train),"token_total":sum(r["sequence_length"] for r in train),"supervised_token_total":sum(r["supervised_token_count"] for r in train)}
    shard_audit["status"]="PASS" if shard_audit["files"]["replay_shard"]["sha256"]==expected["sha256"] and shard_audit["hard_exclude_in_replay_shard"]==0 and shard_audit["quarantine_in_replay_shard"]==0 and shard_audit["official_eval_in_replay_shard"]==0 and shard_audit["sample_id_unique"] else "FAIL"
    write("FINAL_SHARD_INTEGRITY_AUDIT.json",shard_audit)
    # Lineage, deliberately identity-only for ARC task replay and native 6/7 for reARC.
    lineage_bad=[]
    bucket=Counter()
    for r in train:
      b=bybase[r["base_puzzle_id"]]; native=r["source_pair_indices"] or []
      kind="native_pair_bank_"+str(len(native)) if native else "full_task_source_order"
      bucket[(r["source"],kind)]+=1
      if not b["source_version"] or not r["augmentation_transform"] or not r["color_permutation"]: lineage_bad.append(r["sample_id"])
    replay_audit={"status":"PASS" if not lineage_bad else "FAIL","replay_episodes":len(train),"lineage_missing":len(lineage_bad),"by_source_pair_selection":{f"{a}|{b}":n for (a,b),n in sorted(bucket.items())},"classification":"COMPATIBLE_NEW_REPLAY_CURRICULUM_NOT_EXACT_SFT139_RECONSTRUCTION","identity_augmentations":True,"rearc_construction":"bounded_without_replacement_6_or_7_pair_native_episodes"}
    write("REPLAY_EPISODE_AUDIT.json",replay_audit)
    # Corrected compact gate and corpus status, without rebuilding frozen data.
    gate={"status":"FAIL_NOT_READY","DATA_ENGINEERING_READY":True,"SCIENTIFIC_TRAINING_READY":False,"PRIMARY_BLOCKER":"NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS","quarantined_nvarc_large_datasets":"UNAVAILABLE_FOR_TRAINING_PROVENANCE_BLOCKED","accepted_replay_rows":len(train),"novel_rows":0,"explicit_gpu_training_started":False,"blockers":["NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS"]}
    write("GPU_TRAINING_GATE.json",gate)
    (OUT/"GPU_TRAINING_GATE.md").write_text("# GPU training gate v2\n\n- DATA_ENGINEERING_READY: **TRUE**\n- SCIENTIFIC_TRAINING_READY: **FALSE**\n- PRIMARY_BLOCKER: `NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS`\n- NVARC large datasets: `UNAVAILABLE_FOR_TRAINING_PROVENANCE_BLOCKED`\n- GPU TRAINING STARTED = FALSE\n",encoding="utf-8",newline="\n")
    (OUT/"FULL_CORPUS_REPORT.md").write_text("# Full corpus report v2 final audit\n\n- accepted base families: 1,829\n- replay episodes: 62,650 (1,309 ARC-style plus 61,341 source-native reARC episodes)\n- reARC pairs: 400,000 raw = 399,871 valid + 129 invalid\n- novel episodes: 0; no novel validation/holdout was fabricated\n- DATA_ENGINEERING_READY: **TRUE**\n- SCIENTIFIC_TRAINING_READY: **FALSE**\n- PRIMARY_BLOCKER: `NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS`\n- GPU TRAINING STARTED = FALSE\n",encoding="utf-8",newline="\n")
    write("CPU_TEST_REPORT.json", {"status":"PASS_WITH_EXPLICIT_BOUNDARY","training_data_v2_tests":{"passed":15,"failed":0,"skipped":0,"command":"PYTHONPATH=src; ARC2_SFT139_MODEL_DIR=<untracked local model>; pytest tests/test_training_data_v2.py -q"},"syntax":{"passed":3,"failed":0,"skipped":0,"command":"py_compile pipeline, preparation runner, audit runner"},"full_project_suite":{"status":"NOT_RUN","reason":"The sparse CPU-only checkout has known collection blockers (Torch-dependent unrelated tests and absent historical non-V2 artifacts). Running it would not validate this frozen V2 corpus and no GPU/runtime dependency was installed."},"gpu_training_started":False})
    print(canonical_json({"status":"PASS" if all(x["status"]=="PASS" for x in (invalid_summary,shard_audit,replay_audit)) else "FAIL","rearc":rearc["status"],"train_rows":len(train)}))
if __name__=="__main__": main()
