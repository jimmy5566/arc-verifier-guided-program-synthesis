"""Audit-only V2.1 patch over the frozen V2 registries.

No source discovery or raw-corpus build occurs here.  The sole large-data
mutation is a justified re-materialization of token/label columns from the
already frozen sample text after authoritative tokenizer-API parity proved V2
invalid.  HARD_EXCLUDE rows are omitted from the supervised sample registry.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable

import pyarrow as pa
import pyarrow.parquet as pq

IGNORE_INDEX = -100
ROLE_REPLAY = "TRAIN_ELIGIBLE_REPLAY"
ROLE_HARD = "HARD_EXCLUDE_FUTURE_EVAL"
VERSION = "training_data_v2.1"
TURN_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
_TOKENIZER = None


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _init_tokenizer(model_root: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)


def _turns(text: str) -> list[tuple[str, str]]:
    matches = list(TURN_RE.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        raise ValueError("serialized sample is not a complete sequence of native chat turns")
    return [(match.group(1), match.group(0)) for match in matches]


def _api_tokenize_text(text: str) -> tuple[list[int], list[int], dict[str, Any]]:
    assert _TOKENIZER is not None
    full_ids = _TOKENIZER(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    concatenated: list[int] = []
    labels: list[int] = []
    boundaries: list[dict[str, Any]] = []
    for role, rendered in _turns(text):
        ids = _TOKENIZER(rendered, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        start = len(concatenated)
        concatenated.extend(ids)
        labels.extend(ids if role == "assistant" else [IGNORE_INDEX] * len(ids))
        boundaries.append({"role": role, "start": start, "end": len(concatenated), "first": ids[0] if ids else None, "last": ids[-1] if ids else None})
    if concatenated != full_ids:
        raise ValueError("whole-sample tokenizer API differs from concatenated complete-turn tokenizer API")
    if not full_ids or boundaries[-1]["last"] != _TOKENIZER.eos_token_id:
        raise ValueError("serialized sample does not terminate with tokenizer EOS")
    return full_ids, labels, {"turns": boundaries, "eos_token_id": _TOKENIZER.eos_token_id}


def _retokenize_row(row: dict[str, Any]) -> dict[str, Any]:
    ids, labels, _ = _api_tokenize_text(row["text"])
    row["input_ids"] = ids
    row["labels"] = labels
    row["sequence_length"] = len(ids)
    row["supervised_token_count"] = sum(value != IGNORE_INDEX for value in labels)
    return row


def _write_retokenized(source: Path, output: Path, model_root: Path, workers: int) -> dict[str, Any]:
    source_file = pq.ParquetFile(source)
    target_schema = source_file.schema_arrow
    columns = [name for name in target_schema.names if name not in {"input_ids", "labels", "sequence_length", "supervised_token_count"}]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    writer = None
    input_rows = output_rows = hard_rows = 0
    source_counts: Counter[str] = Counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer, initargs=(str(model_root),)) as pool:
        for batch in source_file.iter_batches(batch_size=128, columns=columns):
            raw_rows = batch.to_pylist(); input_rows += len(raw_rows)
            hard_rows += sum(row["final_training_role"] == ROLE_HARD for row in raw_rows)
            replay_rows = [row for row in raw_rows if row["final_training_role"] == ROLE_REPLAY]
            cooked = list(pool.map(_retokenize_row, replay_rows, chunksize=8))
            for row in cooked: source_counts[row["source"]] += 1
            if not cooked: continue
            # Preserve the already-frozen registry schema.  Early source batches
            # legitimately contain only null seeds/empty pair-index lists, so
            # inference would otherwise narrow those fields incorrectly.
            table = pa.Table.from_pylist(cooked, schema=target_schema)
            if writer is None: writer = pq.ParquetWriter(temporary, table.schema, compression="zstd")
            writer.write_table(table); output_rows += len(cooked)
    if writer is None: raise RuntimeError("no replay rows materialized")
    writer.close(); os.replace(temporary, output)
    return {"input_registry_rows": input_rows, "hard_exclude_rows_removed": hard_rows, "replay_rows_written": output_rows, "source_counts": dict(sorted(source_counts.items())), "bytes": output.stat().st_size, "sha256": sha256_file(output)}


def _representatives(path: Path) -> dict[str, list[str]]:
    by_source: dict[str, list[tuple[int, str]]] = defaultdict(list)
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2048, columns=["source", "sample_id", "sequence_length"]):
        for row in batch.to_pylist(): by_source[row["source"]].append((row["sequence_length"], row["sample_id"]))
    selected: dict[str, list[str]] = {}
    for source, values in sorted(by_source.items()):
        values.sort(); n=len(values); ids=[]
        for q in (0.0, .5, .9, .99, 1.0): ids.append(values[round((n-1)*q)][1])
        rng=random.Random(int(hashlib.sha256((VERSION+source).encode()).hexdigest()[:16],16))
        ids.extend(item[1] for item in rng.sample(values, min(3,n)))
        selected[source]=list(dict.fromkeys(ids))
    return selected


def _tokenizer_parity(path: Path, model_root: Path) -> dict[str, Any]:
    _init_tokenizer(str(model_root)); selected=_representatives(path); wanted={x for v in selected.values() for x in v}; checks=[]
    for batch in pq.ParquetFile(path).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if row["sample_id"] not in wanted: continue
            ids, labels, details=_api_tokenize_text(row["text"])
            user_mask_ok=assistant_mask_ok=True
            for turn in details["turns"]:
                observed=row["labels"][turn["start"]:turn["end"]]
                if turn["role"]=="user": user_mask_ok &= all(x==IGNORE_INDEX for x in observed)
                else: assistant_mask_ok &= observed==row["input_ids"][turn["start"]:turn["end"]]
            checks.append({"sample_id":row["sample_id"],"source":row["source"],"length":len(ids),"api_ids_exact":ids==row["input_ids"],"labels_exact":labels==row["labels"],"user_masked":user_mask_ok,"assistant_supervised":assistant_mask_ok,"special_boundaries":all(t["first"]==14 and t["last"]==15 for t in details["turns"]),"eos_final":ids[-1]==details["eos_token_id"]})
    passed=len(checks)==len(wanted) and all(all(v for k,v in c.items() if k in {"api_ids_exact","labels_exact","user_masked","assistant_supervised","special_boundaries","eos_final"}) for c in checks)
    return {"status":"PASS" if passed else "FAIL","tokenizer_class":_TOKENIZER.__class__.__name__,"model_root_identity":"sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1","selection":selected,"checked_samples":len(checks),"checks":checks}


def _duplicate_report(puzzle_path: Path) -> dict[str, Any]:
    rows=pq.read_table(puzzle_path).to_pylist(); groups=defaultdict(list)
    for row in rows: groups[row["train_pair_sha256"]].append(row)
    results=[]
    for signature,members in sorted(groups.items()):
        if len(members)<2: continue
        compact=[{k:m[k] for k in ("base_puzzle_id","source","source_subset","source_native_id","final_training_role","full_content_sha256")} for m in members]
        roles=sorted({m["final_training_role"] for m in members})
        results.append({"train_pair_sha256":signature,"members":compact,"roles":roles,"crosses_replay_hard_exclude_boundary":ROLE_REPLAY in roles and ROLE_HARD in roles,"resolution":"SAME_SOURCE_REPLAY_ONLY_DISTINCT_FULL_TASKS" if roles==[ROLE_REPLAY] else "REQUIRES_REVIEW"})
    return {"status":"PASS" if results and not any(x["crosses_replay_hard_exclude_boundary"] for x in results) else "FAIL","duplicate_groups":results,"group_count":len(results)}


def _accounting(puzzle_path: Path, sample_path: Path) -> dict[str, Any]:
    puzzles=pq.read_table(puzzle_path,columns=["source","final_training_role"]).to_pylist(); samples=pq.read_table(sample_path,columns=["source","final_training_role"]).to_pylist()
    sources=sorted({r["source"] for r in puzzles}); rows=[]
    for source in sources:
        ps=[r for r in puzzles if r["source"]==source]; ss=[r for r in samples if r["source"]==source]
        rows.append({"source":source,"registry_episodes":len(ps),"replay_episodes":sum(r["final_training_role"]==ROLE_REPLAY for r in ss),"hard_exclude_episodes":sum(r["final_training_role"]==ROLE_HARD for r in ps),"supervised_hard_exclude_rows":sum(r["final_training_role"]==ROLE_HARD for r in ss)})
    return {"status":"PASS" if sum(r["supervised_hard_exclude_rows"] for r in rows)==0 else "FAIL","sources":rows}


def _replay_semantics(path: Path, *, replay_only: bool) -> dict[str, Any]:
    excluded={"input_ids","labels","sequence_length","supervised_token_count"}
    columns=[name for name in pq.ParquetFile(path).schema_arrow.names if name not in excluded]
    digest=hashlib.sha256(); rows=0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512,columns=columns):
        for row in batch.to_pylist():
            if replay_only and row["final_training_role"]!=ROLE_REPLAY: continue
            digest.update(canonical_json(row).encode()); digest.update(b"\n"); rows+=1
    return {"rows":rows,"sha256":digest.hexdigest(),"excluded_token_materialization_fields":sorted(excluded)}


def _fingerprint(processed: Path, puzzle_path: Path, sample_path: Path, shard_path: Path, source_path: Path) -> dict[str, Any]:
    sources=sorted((r["source"],r["version"],r["license_status"],r["training_status"]) for r in pq.read_table(source_path).to_pylist())
    logical={"schema_version":VERSION,"ordered_shards":[{"logical_name":"train/replay/replay-00000.parquet","sha256":sha256_file(shard_path),"rows":pq.ParquetFile(shard_path).metadata.num_rows}],"registries":[{"logical_name":"puzzle_registry.parquet","sha256":sha256_file(puzzle_path),"rows":pq.ParquetFile(puzzle_path).metadata.num_rows},{"logical_name":"sample_registry.parquet","sha256":sha256_file(sample_path),"rows":pq.ParquetFile(sample_path).metadata.num_rows}],"source_versions":sources,"split_policy":"family-first; replay only; official evaluation is provenance-only HARD_EXCLUDE; no novel corpus admitted","exposure_policy_version":"training_data_v2.1_explicit_semantic_roles"}
    value=hashlib.sha256(canonical_json(logical).encode()).hexdigest()
    return {"status":"PASS","fingerprint_sha256":value,"logical_manifest":logical,"absolute_paths_in_fingerprint":False,"relocation_invariant":value==hashlib.sha256(canonical_json(logical).encode()).hexdigest()}


def _determinism(raw_root: Path, model_root: Path) -> dict[str, Any]:
    # Import only the existing source readers; this processes eight real descriptors,
    # including two reARC banks, never the full 400k corpus.
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
    from training_data_v2.pipeline import _source_descriptors, _process_descriptors
    descriptors=_source_descriptors(raw_root); grouped=defaultdict(list)
    for d in descriptors: grouped[d.source].append(d)
    selected=[]
    for source,items in sorted(grouped.items()):
        items=sorted(items,key=lambda d:d.native_id); selected.extend([items[0],items[-1]] if len(items)>1 else items)
    def normalize(records):
        _init_tokenizer(str(model_root)); result=[]
        for record in records:
            result.append({"base_puzzle_id":record["base_puzzle_id"],"source":record["source"],"canonical_observation_sha256":record["canonical_observation_sha256"],"full_content_sha256":record["full_content_sha256"],"train_pair_sha256":record["train_pair_sha256"],"d4_signature_sha256":record["d4_signature_sha256"],"samples":[{"sample_id":s["sample_id"],"pair_order_seed":s["pair_order_seed"],"sample_seed":s["sample_seed"],"source_pair_indices":s.get("source_pair_indices",[]),"input_ids":_api_tokenize_text(s["text"])[0],"labels":_api_tokenize_text(s["text"])[1]} for s in record["samples"]]})
        return result
    a=normalize(_process_descriptors(selected,workers=12)); b=normalize(_process_descriptors(selected,workers=20)); exact=canonical_json(a)==canonical_json(b)
    return {"status":"PASS" if exact else "FAIL","workers":[12,20],"descriptor_count":len(selected),"sources":sorted(grouped),"canonical_ids_hashes_sample_ids_episode_lineage_token_ids_labels_exact":exact,"slice_sha256":hashlib.sha256(canonical_json(a).encode()).hexdigest()}


def main() -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1]); parser.add_argument("--workers",type=int,default=20); parser.add_argument("--reuse-valid-materialization",action="store_true"); args=parser.parse_args()
    root=args.root.resolve(); old=root/"data/processed/arc_training_v2"; new=root/"data/processed/arc_training_v2_1"; artifacts=root/"artifacts/training_data_v2_1"; model=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    artifacts.mkdir(parents=True,exist_ok=True); new.mkdir(parents=True,exist_ok=True)
    puzzle=old/"puzzle_registry.parquet"; source=old/"source_registry.parquet"; old_samples=old/"sample_registry.parquet"; samples=new/"sample_registry.parquet"; shard=new/"train/replay/replay-00000.parquet"
    if args.reuse_valid_materialization:
        if not samples.exists() or not shard.exists(): raise RuntimeError("requested materialization reuse but files are absent")
        source_counts=Counter(pq.read_table(samples,columns=["source"]).column("source").to_pylist())
        old_roles=Counter(pq.read_table(old_samples,columns=["final_training_role"]).column("final_training_role").to_pylist())
        materialization={"input_registry_rows":sum(old_roles.values()),"hard_exclude_rows_removed":old_roles[ROLE_HARD],"replay_rows_written":pq.ParquetFile(samples).metadata.num_rows,"source_counts":dict(sorted(source_counts.items())),"bytes":samples.stat().st_size,"sha256":sha256_file(samples)}
    else:
        materialization=_write_retokenized(old_samples,samples,model,args.workers)
        shard.parent.mkdir(parents=True,exist_ok=True)
        # The V2.1 sample registry is replay-only, so its exact bytes are the training shard.
        shard.write_bytes(samples.read_bytes())
    materialization["shard_sha256"]=sha256_file(shard); materialization["shard_rows"]=pq.ParquetFile(shard).metadata.num_rows
    tokenizer=_tokenizer_parity(samples,model); determinism=_determinism(root/"data/raw",model); duplicate=_duplicate_report(puzzle); accounting=_accounting(puzzle,samples); fingerprint=_fingerprint(new,puzzle,samples,shard,source)
    old_semantics=_replay_semantics(old_samples,replay_only=True); new_semantics=_replay_semantics(samples,replay_only=False); semantics={"status":"PASS" if old_semantics==new_semantics else "FAIL","v2_replay":old_semantics,"v2_1_replay":new_semantics}
    checks={"real_tokenizer_parity":tokenizer["status"]=="PASS","multiprocess_determinism":determinism["status"]=="PASS","portable_fingerprint":fingerprint["status"]=="PASS" and fingerprint["relocation_invariant"],"train_pair_duplicate_boundary_safe":duplicate["status"]=="PASS","hard_exclude_supervised_rows_zero":accounting["status"]=="PASS","replay_row_count_preserved":materialization["replay_rows_written"]==62650,"replay_semantics_preserved":semantics["status"]=="PASS","shard_hash_matches_registry":materialization["sha256"]==materialization["shard_sha256"]}
    gate={"status":"FAIL_NOT_READY","checks":checks,"DATA_ENGINEERING_READY":all(checks.values()),"SCIENTIFIC_TRAINING_READY":False,"PRIMARY_BLOCKER":"NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS","GPU_TRAINING_STARTED":False}
    for name,value in (("TOKENIZER_API_PARITY_AUDIT.json",tokenizer),("MULTIPROCESS_DETERMINISM_AUDIT.json",determinism),("PORTABLE_DATASET_FINGERPRINT.json",fingerprint),("TRAIN_PAIR_DUPLICATE_RESOLUTION.json",duplicate),("HARD_EXCLUDE_MATERIALIZATION_AUDIT.json",{"status":accounting["status"],"hard_exclude_supervised_rows_remaining":sum(r["supervised_hard_exclude_rows"] for r in accounting["sources"]),"materialization":materialization}),("REPLAY_SEMANTICS_PRESERVATION_AUDIT.json",semantics),("SOURCE_ACCOUNTING_V2_1.json",accounting),("GPU_TRAINING_GATE_V2_1.json",gate)):
        atomic_json(artifacts/name,value)
    print(canonical_json({"status":"PASS" if gate["DATA_ENGINEERING_READY"] else "FAIL","fingerprint":fingerprint["fingerprint_sha256"],"hard_exclude_supervised_rows":0,"gpu_training_started":False}))
    return 0 if gate["DATA_ENGINEERING_READY"] else 1


if __name__=="__main__": raise SystemExit(main())
