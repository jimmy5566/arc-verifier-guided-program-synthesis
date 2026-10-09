"""Target-blind, no-update, Batch-1 native Beam-4 candidate freezer."""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from training_data.pipeline import ARCNativeInputAdapter, EXPECTED_TOKENS, canonical_hash, task_observation

FAMILIES = ("connected components", "inside/contains", "difference", "width", "orientation")
EOS, PAD, NEWLINE = 15, 13, 10

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
def sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
def atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)
def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")

def candidate_parse(ids: list[int]) -> tuple[str, list[list[int]] | None]:
    values = list(ids)
    while values and values[-1] == PAD: values.pop()
    if values and values[-1] == EOS: values.pop()
    if not values: return "EMPTY", None
    if EOS in values: return "EOS_NOT_SINGLE_TERMINAL", None
    if any(v not in tuple(range(10)) + (NEWLINE,) for v in values): return "NON_GRID_TOKEN", None
    rows, row = [], []
    for value in values:
        if value == NEWLINE:
            if not row: return "EMPTY_ROW", None
            rows.append(row); row = []
        else: row.append(value)
    if not row: return "TRAILING_NEWLINE", None
    rows.append(row)
    if not rows or any(len(r) != len(rows[0]) for r in rows): return "RAGGED_GRID", None
    return "VALID", rows

def prompt_ids(task: dict) -> tuple[list[int], str, str]:
    observation = task_observation(task)
    ids: list[int] = []
    for pair in observation["train"]:
        for role, grid in (("user", pair["input"]), ("assistant", pair["output"])):
            ids += [14, EXPECTED_TOKENS[role], NEWLINE]
            ids += [EXPECTED_TOKENS[c] if c != "\n" else NEWLINE for c in ARCNativeInputAdapter.serialize_grid(grid)] + [EOS]
    test = observation["test"][0]
    ids += [14, EXPECTED_TOKENS["user"], NEWLINE]
    ids += [EXPECTED_TOKENS[c] if c != "\n" else NEWLINE for c in ARCNativeInputAdapter.serialize_grid(test["input"])] + [EOS]
    ids += [14, EXPECTED_TOKENS["assistant"], NEWLINE]
    return ids, sha_bytes(bytes(ids)), canonical_hash(observation)

def load(config: dict) -> list[dict]:
    declared = ROOT / config["inputs"]["target_dev_path"]
    aliases = [declared, Path("/workspace/arc2/mounted_inputs/base_only_targeted_repair_available_data_v1/TARGET_DEV.jsonl"), Path("/workspace/arc2/mounted_inputs/base_only_targeted_repair_remote_first_v2/TARGET_DEV.jsonl")]
    dev = next((p for p in aliases if p.is_file() and sha(p) == config["inputs"]["target_dev_sha256"]), None)
    if dev is None: raise RuntimeError("TARGET_DEV_EXACT_IDENTITY_RESOLUTION_FAIL")
    source = {row["episode_id"]: row for row in (json.loads(line) for line in dev.read_text(encoding="utf-8").splitlines() if line.strip())}
    manifest = json.loads((ROOT / config["cohort"]["source_manifest_path"]).read_text(encoding="utf-8"))
    if sha(ROOT / config["cohort"]["source_manifest_path"]) != config["cohort"]["source_manifest_sha256"]: raise RuntimeError("COHORT_MANIFEST_SHA_FAIL")
    rows = []
    for batch in manifest["batches"]:
        for member in batch["members"]:
            row = source.get(member["episode_id"])
            if not row or row["family"] != member["family"]: raise RuntimeError("COHORT_ROW_BINDING_FAIL")
            ids, prompt_sha, observation_sha = prompt_ids(row["task"])
            rows.append({"episode_id": row["episode_id"], "family": row["family"], "input_ids": ids, "prompt_sha256": prompt_sha, "observation_sha256": observation_sha})
    counts = {f: sum(r["family"] == f for r in rows) for f in FAMILIES}
    if len(rows) != 60 or counts != {f: 12 for f in FAMILIES}: raise RuntimeError("FROZEN_COHORT_DENOMINATOR_FAIL")
    return rows

def self_test(config: dict) -> dict:
    rows = load(config)
    valid_status, valid_grid = candidate_parse([1,2,NEWLINE,3,4,EOS,PAD])
    if valid_status != "VALID" or valid_grid != [[1,2],[3,4]]: raise RuntimeError("PARSER_VALID_FIXTURE_FAIL")
    if candidate_parse([EOS,1])[0] == "VALID" or candidate_parse([1,EOS,2])[0] == "VALID": raise RuntimeError("PARSER_SPECIAL_FIXTURE_FAIL")
    # `load` constructs every prompt only through `task_observation`; verify
    # the target-blind observation identity is present for every frozen row.
    if any(not row["observation_sha256"] or not row["prompt_sha256"] for row in rows):
        raise RuntimeError("OBSERVATION_TARGET_BLINDNESS_FAIL")
    return {"status": "PASS_CPU_ONLY_NO_MODEL_IMPORT", "rows": len(rows), "family_counts": {f: sum(r["family"] == f for r in rows) for f in FAMILIES}, "decode_batch_size": 1, "beam_width": 4}

def runtime_tokenizer_check(tokenizer, decode: dict) -> None:
    # The frozen tokenizer's printable vocabulary calls newline `Ċ`; verify
    # semantics through its encoder and IDs rather than assuming that spelling.
    if any(tokenizer.convert_ids_to_tokens(i) != str(i) for i in range(10)):
        raise RuntimeError("TOKENIZER_NATIVE_DIGIT_ID_FAIL")
    if tokenizer.encode("\n", add_special_tokens=False) != [NEWLINE]:
        raise RuntimeError("TOKENIZER_NATIVE_NEWLINE_ID_FAIL")
    if tokenizer.eos_token_id != decode["eos_token_id"] or tokenizer.pad_token_id not in (None, decode["pad_token_id"]): raise RuntimeError("TOKENIZER_SPECIAL_TOKEN_ID_FAIL")

def execute(config: dict, out: Path) -> dict:
    if out.exists(): raise RuntimeError("FRESH_OUTPUT_REQUIRED")
    rows = load(config)
    import subprocess
    verify = [sys.executable, str(ROOT / "scripts/verify_paired_fixed_b32_runtime_identity.py")]
    for condition in config["inputs"]["conditions"].values():
        manifest_path = ROOT / condition["manifest"]
        if sha(manifest_path) != condition["manifest_sha256"]:
            raise RuntimeError("CHECKPOINT_MANIFEST_SHA_FAIL")
        verify += ["--checkpoint-manifest", str(manifest_path)]
    # This complete base/adapter identity check is deliberately before every
    # CUDA/model import, so a bad mounted model never reaches generation.
    subprocess.run(verify, check=True)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("CUDA_BF16_UNAVAILABLE")
    deadline = time.monotonic() + int(config["cap_seconds"])
    all_rows = []
    for checkpoint_id, condition in config["inputs"]["conditions"].items():
        manifest_path = ROOT / condition["manifest"]
        checkpoint = json.loads(manifest_path.read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(checkpoint["base_path"], local_files_only=True)
        tokenizer.pad_token_id = config["decode"]["pad_token_id"]
        runtime_tokenizer_check(tokenizer, config["decode"])
        model = AutoModelForCausalLM.from_pretrained(checkpoint["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=config["decode"]["attention_backend"]).to("cuda:0")
        model = PeftModel.from_pretrained(model, checkpoint["adapter_path"], is_trainable=False).eval()
        try:
            for index, row in enumerate(rows):
                if time.monotonic() >= deadline: raise RuntimeError("RUNTIME_CAP_REACHED")
                inputs = torch.tensor([row["input_ids"]], dtype=torch.long, device="cuda:0")
                attention = torch.ones_like(inputs)
                with torch.inference_mode():
                    generated = model.generate(input_ids=inputs, attention_mask=attention, do_sample=False, num_beams=4, num_return_sequences=4, max_new_tokens=config["decode"]["max_new_tokens"], use_cache=True, eos_token_id=EOS, pad_token_id=PAD, early_stopping=True, return_dict_in_generate=True, output_scores=True)
                seqs = generated.sequences.detach().cpu().tolist()
                scores = generated.sequences_scores.detach().cpu().tolist()
                if len(seqs) != 4 or len(scores) != 4: raise RuntimeError("BEAM_CANDIDATE_COUNT_FAIL")
                for rank, (seq, score) in enumerate(zip(seqs, scores, strict=True), 1):
                    suffix = [int(x) for x in seq[len(row["input_ids"]):]]
                    parse_status, grid = candidate_parse(suffix)
                    all_rows.append({"checkpoint_id": checkpoint_id, "episode_id": row["episode_id"], "family": row["family"], "prompt_sha256": row["prompt_sha256"], "observation_sha256": row["observation_sha256"], "rank": rank, "sequence_score": float(score), "generated_token_ids": suffix, "generated_sha256": sha_bytes(canonical(suffix)), "parser_status": parse_status, "canonical_prediction_sha256": sha_bytes(canonical(grid)) if grid is not None else None})
                print(json.dumps({"event":"TARGET_BLIND_BEAM_PROGRESS","checkpoint_id":checkpoint_id,"completed":index+1,"total":len(rows)},sort_keys=True), flush=True)
        finally:
            del model; torch.cuda.empty_cache()
    expected = len(rows) * len(config["inputs"]["conditions"]) * 4
    if len(all_rows) != expected: raise RuntimeError("CANDIDATE_ROW_COUNT_FAIL")
    out.mkdir(parents=True)
    raw = out / "CANDIDATES_TARGET_BLIND.jsonl"
    raw.write_text("".join(json.dumps(r,sort_keys=True)+"\n" for r in all_rows), encoding="utf-8")
    result = {"schema_version":1,"protocol_id":config["protocol_id"],"status":"COMPLETE_CANDIDATES_FROZEN_NO_UPDATE","candidate_rows":len(all_rows),"candidate_sha256":sha(raw),"optimizer_steps":0,"backward_performed":False,"generation_performed":True,"target_labels_accessed_during_generation":False,"final_audit_opened":False}
    atomic(out / "TERMINAL_RECEIPT.json", result); return result

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--config",type=Path,required=True); ap.add_argument("--self-test",action="store_true"); ap.add_argument("--out",type=Path); a=ap.parse_args()
    config=json.loads(a.config.read_text(encoding="utf-8"))
    if a.self_test: print(json.dumps(self_test(config),sort_keys=True)); return
    if a.out is None: raise RuntimeError("OUTPUT_REQUIRED")
    try: result=execute(config,a.out)
    except Exception as exc:
        result={"schema_version":1,"protocol_id":config.get("protocol_id"),"status":"FAILED_OR_CAPPED","error":type(exc).__name__+":"+str(exc),"optimizer_steps":0,"backward_performed":False,"generation_performed":False,"target_labels_accessed_during_generation":False,"final_audit_opened":False}
        atomic(a.out / "TERMINAL_RECEIPT.json",result); raise
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__": main()
