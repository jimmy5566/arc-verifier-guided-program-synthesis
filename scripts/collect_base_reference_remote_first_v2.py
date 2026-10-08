#!/usr/bin/env python3
"""Read-only, fail-closed base-reference collector for V2. Additive correction; never trains."""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
from typing import Any

PROTOCOL = "BASE_ONLY_TARGETED_REPAIR_REMOTE_FIRST_V2"
CONTRACT_NAME = "ARC_NATIVE_OBSERVATION_ONLY_V1"

def canonical(v: Any) -> str: return json.dumps(v, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
def sha_bytes(b: bytes) -> str: return hashlib.sha256(b).hexdigest()
def sha(p: Path) -> str: return sha_bytes(p.read_bytes())
def dig(v: Any) -> str: return sha_bytes(canonical(v).encode("utf-8"))
def write(p: Path, v: Any) -> None:
    p.parent.mkdir(parents=True, exist_ok=True); t = p.with_suffix(p.suffix + ".tmp")
    t.write_text(json.dumps(v, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n"); t.replace(p)
def reject_path(p: Path) -> None:
    if "TRAIN.JSONL" in str(p).upper() or "FINAL_AUDIT" in str(p).upper(): raise RuntimeError("FORBIDDEN_INPUT_PATH")
def read_rows(p: Path, roles: set[str]) -> list[dict[str, Any]]:
    reject_path(p)
    if not p.is_file(): raise RuntimeError("EVALUATION_SURFACE_MISSING")
    rows = [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]
    if not rows or any(r.get("role") not in roles for r in rows): raise RuntimeError("FORBIDDEN_OR_EMPTY_EVALUATION_SURFACE")
    ids = [str(r.get("episode_id", "")) for r in rows]
    if len(ids) != len(set(ids)) or any(not i for i in ids): raise RuntimeError("DUPLICATE_OR_MISSING_EPISODE_ID")
    return rows
def observation(r: dict[str, Any]) -> dict[str, Any]:
    task = r.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list): raise RuntimeError("INVALID_ARC_TASK")
    train = []; test = []
    for x in task["train"]:
        if not isinstance(x, dict) or "input" not in x or "output" not in x: raise RuntimeError("INVALID_TRAIN_PAIR")
        train.append({"input": x["input"], "output": x["output"]})
    for x in task["test"]:
        if not isinstance(x, dict) or "input" not in x: raise RuntimeError("INVALID_TEST_PAIR")
        test.append({"input": x["input"]})  # Structural redaction: no held-out output is copied.
    return {"schema": CONTRACT_NAME, "train": train, "test": test}
def native_grid(value: Any) -> str:
    if not isinstance(value, list) or not value or any(not isinstance(row, list) or not row for row in value):
        raise RuntimeError("INVALID_ARC_GRID")
    width = len(value[0])
    if any(len(row) != width or any(isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9 for cell in row) for row in value):
        raise RuntimeError("INVALID_ARC_GRID")
    return "\n".join("".join(str(cell) for cell in row) for row in value)
def prompt(r: dict[str, Any], c: dict[str, Any]) -> str:
    # This exactly follows configs/nvarc_native_846d0198/chat_template.j2:
    # alternating native digit-grid messages plus one assistant generation turn.
    pieces = []
    for pair in observation(r)["train"]:
        pieces.extend(("<|im_start|>user\n" + native_grid(pair["input"]) + "<|im_end|>",
                       "<|im_start|>assistant\n" + native_grid(pair["output"]) + "<|im_end|>"))
    for pair in observation(r)["test"]:
        pieces.append("<|im_start|>user\n" + native_grid(pair["input"]) + "<|im_end|>")
    return c["prompt"]["prefix"] + "".join(pieces) + c["prompt"]["suffix"]
def parse_grid(s: str) -> list[list[int]] | None:
    # Exact ARCNativeOutputParser policy: only newline-delimited digit grids.
    body = s.strip("\r\n")
    if not body or "\r" in body: return None
    rows = body.split("\n")
    if any(not row or any(char not in "0123456789" for char in row) for row in rows): return None
    width = len(rows[0])
    if any(len(row) != width for row in rows): return None
    return [[int(char) for char in row] for row in rows]
def verify_file(p: Path, want: str) -> None:
    if not p.is_file(): raise RuntimeError(f"REQUIRED_FILE_MISSING:{p}")
    if sha(p) != want: raise RuntimeError(f"REQUIRED_FILE_HASH_MISMATCH:{p}")
def verify_binding(b: dict[str, Any], base: Path) -> str:
    if b.get("base_path") != "/workspace/arc2/models/qwen3_4b_grids15_sft139": raise RuntimeError("BASE_PATH_IDENTITY_INVALID")
    files = b.get("base_files")
    if not isinstance(files, dict) or not files: raise RuntimeError("BASE_MANIFEST_INVALID")
    for name, want in sorted(files.items()): verify_file(base / name, want)
    return dig(files)
def expected(r: dict[str, Any]) -> list[list[int]]:
    test = r["task"]["test"]
    if len(test) != 1 or "output" not in test[0]: raise RuntimeError("BASE_REFERENCE_REQUIRES_SINGLE_HELDOUT_SCORING_OUTPUT")
    return test[0]["output"]
def output_row(surface: str, r: dict[str, Any], p: str, text: str, cid: str, mid: str) -> dict[str, Any]:
    grid = parse_grid(text)
    return {"surface": surface, "episode_id": r["episode_id"], "observation_sha256": dig(observation(r)), "prompt_sha256": sha_bytes(p.encode("utf-8")), "canonical_prediction_sha256": dig({"grid": grid} if grid is not None else {"invalid_output": True}), "parse_valid": grid is not None, "exact_grid_match": bool(grid == expected(r)) if grid is not None else False, "model_base_manifest_identity": mid, "inference_contract_identity": cid}
def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"episodes": len(rows), "parse_valid": sum(bool(x["parse_valid"]) for x in rows), "exact_grid_match": sum(bool(x["exact_grid_match"]) for x in rows), "episode_ids_sha256": dig([x["episode_id"] for x in rows]), "outcome_rows_sha256": dig(rows)}
def receipt(status: str, started: float, **more: Any) -> dict[str, Any]:
    return {"schema_version": 2, "protocol_id": PROTOCOL, "collector": "collect_base_reference_remote_first_v2.py", "status": status, "baseline_inference_seconds": time.monotonic() - started, "scientific_training_started": False, "lora_constructed": False, "optimizer_constructed": False, **more}
def main() -> int:
    a = argparse.ArgumentParser()
    for name in ("target-dev", "retention", "binding", "contract", "output", "receipt"): a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--base", type=Path); a.add_argument("--authorization", type=Path); a.add_argument("--runtime-limit-seconds", type=float, default=7200); a.add_argument("--cpu-mock", action="store_true"); a.add_argument("--cpu-mock-simulate-runtime-cap", action="store_true")
    z = a.parse_args(); started = time.monotonic()
    if z.runtime_limit_seconds <= 0: raise RuntimeError("RUNTIME_LIMIT_INVALID")
    if z.output.exists() or z.receipt.exists(): raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    target = read_rows(z.target_dev, {"TARGETED_EVALUATION", "TARGETED_COMPOSITION"}); retention = read_rows(z.retention, {"RETENTION_SENTINEL"})
    if len({r["episode_id"] for r in target + retention}) != len(target) + len(retention): raise RuntimeError("DUPLICATE_EPISODE_ACROSS_SURFACES")
    b = json.loads(z.binding.read_text(encoding="utf-8")); craw = z.contract.read_bytes(); c = json.loads(craw.decode("utf-8"))
    if c.get("contract_id") != CONTRACT_NAME: raise RuntimeError("INFERENCE_CONTRACT_INVALID")
    cid = sha_bytes(craw)
    if z.cpu_mock:
        ps = [prompt(r, c) for r in target + retention]
        status = "PARTIAL_RUNTIME_CAP" if z.cpu_mock_simulate_runtime_cap else "CPU_MOCK_PASS_NO_MODEL"
        write(z.output, {"schema_version": 2, "protocol_id": PROTOCOL, "status": status, "model_loaded": False, "lora_constructed": False, "optimizer_constructed": False, "gpu_training_started": False, "inference_contract_identity": cid, "target_episode_order": [r["episode_id"] for r in target], "retention_episode_order": [r["episode_id"] for r in retention], "prompt_sha256": dig(ps), "completed_episode_count": 0 if z.cpu_mock_simulate_runtime_cap else len(ps)})
        write(z.receipt, receipt(status, started, model_loaded=False, model_released=True)); return 1 if z.cpu_mock_simulate_runtime_cap else 0
    auth = json.loads(z.authorization.read_text(encoding="utf-8")) if z.authorization else {}
    if auth.get("base_reference_collection_authorized") is not True: raise RuntimeError("DIRECTOR_BASE_REFERENCE_AUTHORIZATION_REQUIRED")
    if z.base is None: raise RuntimeError("BASE_PATH_REQUIRED")
    mid = verify_binding(b, z.base)  # Complete base identity before model-library import.
    deadline = time.monotonic() + z.runtime_limit_seconds; rows = []; model = None; status = "COLLECTED_PASS"; failure = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("CUDA_OR_BF16_UNAVAILABLE")
        device = torch.device("cuda:0"); rt = c["runtime"]; tok = AutoTokenizer.from_pretrained(z.base, local_files_only=True)
        if tok.get_vocab().get(rt["required_special_token"]) != rt["required_special_token_id"]: raise RuntimeError("TOKENIZER_SPECIAL_TOKEN_ID_MISMATCH")
        model = AutoModelForCausalLM.from_pretrained(z.base, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=rt["attention_backend"]).to(device).eval()
        for surface, records in (("TARGET_DEV", target), ("RETENTION_SENTINEL", retention)):
            for r in records:
                if time.monotonic() >= deadline: status = "PARTIAL_RUNTIME_CAP"; raise TimeoutError("RUNTIME_CAP_BEFORE_EPISODE")
                p = prompt(r, c); encoded = tok(p, return_tensors="pt", add_special_tokens=False); ins = {k: v.to(device) for k, v in encoded.items()}
                with torch.inference_mode(): gen = model.generate(**ins, do_sample=False, max_new_tokens=rt["max_new_tokens"], use_cache=True, eos_token_id=rt["eos_token_id"], pad_token_id=rt["pad_token_id"])
                text = tok.decode(gen[0][ins["input_ids"].shape[-1]:], skip_special_tokens=False); rows.append(output_row(surface, r, p, text, cid, mid))
                if time.monotonic() >= deadline: status = "PARTIAL_RUNTIME_CAP"; raise TimeoutError("RUNTIME_CAP_AFTER_EPISODE")
    except Exception as exc:
        if status == "COLLECTED_PASS": status = "PARTIAL_FAILURE"
        failure = f"{type(exc).__name__}:{exc}"
    finally:
        if model is not None:
            del model
            try:
                import torch; torch.cuda.empty_cache()
            except Exception: pass
    n = len(target) + len(retention)
    if status == "COLLECTED_PASS" and len(rows) != n: status, failure = "PARTIAL_FAILURE", "INCOMPLETE_EPISODE_SET"
    result = {"schema_version": 2, "protocol_id": PROTOCOL, "status": status, "inference_contract_identity": cid, "model_base_manifest_identity": mid, "predictions": rows, "aggregates": {"TARGET_DEV": aggregate([x for x in rows if x["surface"] == "TARGET_DEV"]), "RETENTION_SENTINEL": aggregate([x for x in rows if x["surface"] == "RETENTION_SENTINEL"])}, "expected_episode_count": n, "completed_episode_count": len(rows)}
    write(z.output, result); write(z.receipt, receipt(status, started, model_loaded=True, completed_episode_ids=[x["episode_id"] for x in rows], expected_episode_count=n, completed_episode_count=len(rows), model_base_manifest_identity=mid, inference_contract_identity=cid, failure=failure, runtime_cap_seconds=z.runtime_limit_seconds, model_released=True))
    return 0 if status == "COLLECTED_PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
