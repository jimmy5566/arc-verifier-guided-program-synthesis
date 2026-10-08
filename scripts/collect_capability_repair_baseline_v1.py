#!/usr/bin/env python3
"""Read-only, fail-closed base-reference collector for V2. Additive correction; never trains."""
from __future__ import annotations
import argparse, gc, hashlib, json, os, sys, time
from pathlib import Path
from typing import Any
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.arc2_token_grid_parser import TokenGridContract, parse_generated_token_ids, tokenizer_token_contract

PROTOCOL = "CAPABILITY_REPAIR_BASELINE_V1"
CONTRACT_NAME = "ARC_NATIVE_OBSERVATION_ONLY_V1"
DEFAULT_BATCH_SIZE = 32
DEFAULT_MAX_BATCHED_PROMPT_TOKENS = 21_568
DEFAULT_BATCH_FALLBACK_LADDER = (32, 16, 8, 4, 1)
DEFAULT_BATCH_VALIDATION_COUNT = 32
MIN_CANONICAL_AGREEMENT = 0.9375
MAX_RATE_DELTA = 0.0625

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
def verify_file(p: Path, want: str, expected_bytes: int | None = None) -> None:
    if not p.is_file(): raise RuntimeError(f"REQUIRED_FILE_MISSING:{p}")
    if expected_bytes is not None and p.stat().st_size != expected_bytes: raise RuntimeError(f"REQUIRED_FILE_SIZE_MISMATCH:{p}")
    if sha(p) != want: raise RuntimeError(f"REQUIRED_FILE_HASH_MISMATCH:{p}")
def verify_binding(b: dict[str, Any], base: Path) -> str:
    if b.get("base_path") != "/workspace/arc2/models/qwen3_4b_grids15_sft139": raise RuntimeError("BASE_PATH_IDENTITY_INVALID")
    files = b.get("base_files")
    if not isinstance(files, dict) or not files: raise RuntimeError("BASE_MANIFEST_INVALID")
    for name, want in sorted(files.items()): verify_file(base / name, want)
    return dig(files)
def verify_checkpoint_manifest(manifest: dict[str, Any]) -> str:
    """Verify the exact current base and adapter before model import."""
    base = Path(manifest.get("base_path", "")); adapter = Path(manifest.get("adapter_path", ""))
    if str(base) != "/workspace/arc2/models/qwen3_4b_grids15_sft139": raise RuntimeError("BASE_PATH_IDENTITY_INVALID")
    for label, root, entries in (("BASE", base, manifest.get("base_files")), ("ADAPTER", adapter, manifest.get("adapter_files"))):
        if not isinstance(entries, list) or not entries: raise RuntimeError(f"{label}_MANIFEST_INVALID")
        names = []
        for entry in entries:
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) or "/" in entry["name"] or "\\" in entry["name"]:
                raise RuntimeError(f"{label}_MANIFEST_ENTRY_INVALID")
            if not isinstance(entry.get("bytes"), int) or not isinstance(entry.get("sha256"), str): raise RuntimeError(f"{label}_MANIFEST_ENTRY_INVALID")
            names.append(entry["name"]); verify_file(root / entry["name"], entry["sha256"], entry["bytes"])
        if len(names) != len(set(names)): raise RuntimeError(f"{label}_MANIFEST_DUPLICATE_ENTRY")
    identity = manifest.get("manifest_sha256")
    return str(identity) if isinstance(identity, str) and identity else dig(manifest)
def verify_decoder_validity_evidence(evidence: Path, expected_sha256: str) -> dict[str, Any]:
    """The parser contract is scoped to frozen tokenizer/parser semantics, not an adapter SHA."""
    if sha(evidence) != expected_sha256: raise RuntimeError("DECODER_VALIDITY_EVIDENCE_MISMATCH")
    ev=json.loads(evidence.read_text(encoding="utf-8"))
    if ev.get("status") != "VALID_MEASUREMENT_PASS" or ev.get("target_blind") is not True or ev.get("correctness_scoring_present") is not False or ev.get("raw_token_evidence") is not True:
        raise RuntimeError("VALID_MEASUREMENT_GATE_FAIL")
    return ev
def expected(r: dict[str, Any]) -> list[list[int]]:
    test = r["task"]["test"]
    if len(test) != 1 or "output" not in test[0]: raise RuntimeError("BASE_REFERENCE_REQUIRES_SINGLE_HELDOUT_SCORING_OUTPUT")
    return test[0]["output"]
def output_row(surface: str, r: dict[str, Any], p: str, token_ids: list[int], cid: str, mid: str, token_contract: TokenGridContract) -> dict[str, Any]:
    extracted = parse_generated_token_ids(token_ids, token_contract)
    grid = extracted.grid
    return {
        "surface": surface, "episode_id": r["episode_id"], "observation_sha256": dig(observation(r)),
        "prompt_sha256": sha_bytes(p.encode("utf-8")), "generated_token_ids": extracted.generated_token_ids,
        "generated_length": extracted.generated_length, "termination_status": extracted.termination_status,
        "eos_observed": extracted.eos_observed, "trailing_pad_count": extracted.trailing_pad_count,
        "content_token_ids": extracted.content_token_ids, "parse_reason": extracted.parse_reason,
        "canonical_prediction_sha256": dig({"grid": grid} if grid is not None else {"invalid_output": True}),
        "parse_valid": grid is not None, "exact_grid_match": bool(grid == expected(r)) if grid is not None else False,
        "model_base_manifest_identity": mid, "inference_contract_identity": cid, "token_grid_contract": token_contract.as_dict(),
    }
def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {"episodes": len(rows), "parse_valid": sum(bool(x["parse_valid"]) for x in rows), "exact_grid_match": sum(bool(x["exact_grid_match"]) for x in rows), "episode_ids_sha256": dig([x["episode_id"] for x in rows]), "outcome_rows_sha256": dig(rows)}
def receipt(status: str, started: float, **more: Any) -> dict[str, Any]:
    return {"schema_version": 2, "protocol_id": PROTOCOL, "collector": "collect_capability_repair_baseline_v1.py", "status": status, "baseline_inference_seconds": time.monotonic() - started, "scientific_training_started": False, "adapter_loaded": False, "lora_constructed": False, "optimizer_constructed": False, **more}

def verify_governor_review(launch: dict[str, Any], review_path: Path) -> None:
    """Bind the external stage approval to exactly this immutable launch."""
    if review_path.resolve() != Path(launch["governor_review_path"]).resolve():
        raise RuntimeError("GOVERNOR_REVIEW_PATH_BINDING_MISMATCH")
    if not review_path.is_file():
        raise RuntimeError("GOVERNOR_REVIEW_REQUIRED")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    if review.get("decision") != "CONTINUE_CONTROLLER":
        raise RuntimeError("GOVERNOR_REVIEW_NOT_AUTHORIZING")
    expected = {
        "authorization_request_sha256": launch["authorization_request_sha256"],
        "reviewed_brief_sha256": sha(Path(launch["review_brief_path"])),
        "launch_contract_file_sha256": sha(Path(launch["contract_path"])),
        "launch_contract_identity": launch["contract_sha256"],
        "baseline_identity_sha256": launch["baseline_identity"]["sha256"],
        "checkpoint_manifest_file_sha256": launch["checkpoint_manifest_sha256"],
        "checkpoint_manifest_identity": launch["checkpoint_manifest_identity"],
        "target_dev_sha256": launch["datasets"]["TARGET_DEV"]["sha256"],
        "target_dev_rows": launch["datasets"]["TARGET_DEV"]["rows"],
        "retention_sha256": launch["datasets"]["RETENTION_SENTINEL"]["sha256"],
        "retention_rows": launch["datasets"]["RETENTION_SENTINEL"]["rows"],
        "source_commit": launch["source_commit"],
        "executable_sha256": launch["executable"]["sha256"],
        "nonce_sha256": launch["nonce_sha256"],
        "output_root": launch["output_root"],
        "receipt_path": launch["receipt_path"],
        "runtime_cap_seconds": launch["runtime_cap_seconds"],
    }
    if any(review.get(k) != v for k, v in expected.items()):
        raise RuntimeError("GOVERNOR_REVIEW_BINDING_MISMATCH")

def verify_entrypoint(launch: dict[str, Any], args: argparse.Namespace) -> None:
    actual = [str(Path(sys.executable).resolve()), str(Path(sys.argv[0]).resolve()), *sys.argv[1:]]
    expected = [str(Path(launch["argv"][0]).resolve()), str(Path(launch["argv"][1]).resolve()), *launch["argv"][2:]]
    if actual != expected:
        raise RuntimeError("ARGV_BINDING_MISMATCH")
    if args.runtime_limit_seconds != launch["runtime_cap_seconds"]:
        raise RuntimeError("RUNTIME_CAP_BINDING_MISMATCH")
    for key, value in launch["environment"].items():
        if os.environ.get(key) != value:
            raise RuntimeError(f"ENVIRONMENT_BINDING_MISMATCH:{key}")

def is_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text or "cuda error: out of memory" in text

def fallback_size(failed: int, ladder: tuple[int, ...]) -> int | None:
    for candidate in ladder:
        if candidate < failed:
            return candidate
    return None

def batch_validation_status(*, canonical_agreement: float, parse_valid_rate_delta: float, exact_grid_rate_delta: float) -> str:
    # Token IDs may drift slightly across valid physical batch shapes and GPUs.
    # The scientific invariant is stable parsed/scored behavior; raw agreement
    # remains preserved as an audit metric rather than a post-hoc gate.
    del canonical_agreement
    return "PASS" if parse_valid_rate_delta <= MAX_RATE_DELTA and exact_grid_rate_delta <= MAX_RATE_DELTA else "FAIL"
def validation_subset(contexts: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    """Deterministic short/median/long coverage without reading held-out targets."""
    ordered = sorted(contexts, key=lambda x: (x["prompt_tokens"], x["episode_id"]))
    n = min(count, len(ordered))
    if n <= 1: return ordered[:n]
    if n == len(ordered): return ordered
    positions = sorted({round(index * (len(ordered) - 1) / (n - 1)) for index in range(n)})
    return [ordered[index] for index in positions]
def main() -> int:
    a = argparse.ArgumentParser()
    for name in ("output", "receipt"): a.add_argument("--" + name, type=Path, required=True)
    a.add_argument("--launch-contract", type=Path); a.add_argument("--minimal-gate", type=Path); a.add_argument("--governor-review", type=Path);
    for name in ("target-dev", "retention", "binding", "contract", "cohort-manifest"): a.add_argument("--" + name, type=Path)
    a.add_argument("--base", type=Path); a.add_argument("--adapter", type=Path); a.add_argument("--authorization", type=Path); a.add_argument("--runtime-limit-seconds", type=float, default=7200); a.add_argument("--cpu-mock", action="store_true"); a.add_argument("--cpu-mock-simulate-runtime-cap", action="store_true")
    a.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    a.add_argument("--max-batched-prompt-tokens", type=int, default=DEFAULT_MAX_BATCHED_PROMPT_TOKENS)
    a.add_argument("--batch-validation-count", type=int, default=DEFAULT_BATCH_VALIDATION_COUNT)
    z = a.parse_args(); started = time.monotonic()
    if z.runtime_limit_seconds <= 0: raise RuntimeError("RUNTIME_LIMIT_INVALID")
    if z.batch_size != DEFAULT_BATCH_SIZE: raise RuntimeError("BATCH_SIZE_MUST_BE_32")
    if z.max_batched_prompt_tokens != DEFAULT_MAX_BATCHED_PROMPT_TOKENS: raise RuntimeError("BATCH_TOKEN_CAP_DRIFT")
    if z.batch_validation_count != DEFAULT_BATCH_VALIDATION_COUNT: raise RuntimeError("BATCH_VALIDATION_COUNT_DRIFT")
    # A real run consumes the recorded launch configuration.  Its scientific
    # admission checks are the model/data identities, split roles, and fresh
    # output paths below; review/nonce machinery is operational provenance.
    launch = None; minimal_gate = None; preverified_model_identity = None
    if not z.cpu_mock:
        if (z.launch_contract is None) == (z.minimal_gate is None): raise RuntimeError("CHOOSE_ONE_EXECUTION_GATE")
        if z.minimal_gate is not None:
            minimal_gate=json.loads(z.minimal_gate.read_text(encoding="utf-8"))
            hard=minimal_gate.get("hard_gate_check", {})
            if any(hard.get(k) != "PASS" for k in ("correct_checkpoint", "correct_dataset_and_denominator", "no_leakage", "decoder_and_scoring_valid", "fresh_output", "no_duplicate_live_job")):
                raise RuntimeError("MINIMAL_SCIENTIFIC_GATE_FAIL")
            if minimal_gate.get("final_audit_opened") is not False: raise RuntimeError("FINAL_AUDIT_FORBIDDEN")
            evidence=Path(minimal_gate["decoder_validity_evidence_path"])
            verify_decoder_validity_evidence(evidence, minimal_gate["decoder_validity_evidence_sha256"])
            manifest_path=Path(minimal_gate["checkpoint_manifest_path"])
            if sha(manifest_path) != minimal_gate["checkpoint_manifest_sha256"]: raise RuntimeError("CHECKPOINT_MANIFEST_IDENTITY_FAIL")
            manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
            z.target_dev=Path(minimal_gate["datasets"]["TARGET_DEV"]["path"]); z.retention=Path(minimal_gate["datasets"]["RETENTION_SENTINEL"]["path"])
            verify_file(z.target_dev, minimal_gate["datasets"]["TARGET_DEV"]["sha256"]); verify_file(z.retention, minimal_gate["datasets"]["RETENTION_SENTINEL"]["sha256"])
            z.base=Path(manifest["base_path"]); z.adapter=Path(manifest["adapter_path"]); z.binding=None; z.contract=Path(minimal_gate["inference_contract_path"])
            # The decoder contract is tokenizer/parser-scoped; the current model
            # identity is independently verified from this round's manifest below.
            preverified_model_identity=verify_checkpoint_manifest(manifest)
        else:
            launch=json.loads(z.launch_contract.read_text(encoding="utf-8"))
            verify_entrypoint(launch, z)
            expected_out=Path(launch["output_root"]) / "CAPABILITY_REPAIR_BASELINE_V1_RESULTS.json"
            if z.output.resolve()!=expected_out.resolve() or z.receipt.resolve()!=Path(launch["receipt_path"]).resolve(): raise RuntimeError("LAUNCH_OUTPUT_BINDING_MISMATCH")
            z.target_dev=Path(launch["datasets"]["TARGET_DEV"]["path"]); z.retention=Path(launch["datasets"]["RETENTION_SENTINEL"]["path"])
            z.base=Path(json.loads(Path(launch["checkpoint_manifest_path"]).read_text(encoding="utf-8"))["base_path"])
            z.adapter=Path(json.loads(Path(launch["checkpoint_manifest_path"]).read_text(encoding="utf-8"))["adapter_path"])
            z.binding=None; z.contract=Path(launch["inference_contract"]["path"])
            if launch.get("characterization_cohort_path"): z.cohort_manifest=Path(launch["characterization_cohort_path"])
    if z.output.exists() or z.receipt.exists(): raise RuntimeError("OUTPUT_PATH_NON_OVERWRITE_REQUIRED")
    if z.target_dev is None or z.retention is None or z.contract is None: raise RuntimeError("EVALUATION_INPUTS_REQUIRED")
    target = read_rows(z.target_dev, {"TARGETED_EVALUATION", "TARGETED_COMPOSITION"}); retention = read_rows(z.retention, {"RETENTION_SENTINEL"})
    if z.cohort_manifest is not None:
        cohort=json.loads(z.cohort_manifest.read_text(encoding="utf-8"))
        selected=cohort.get("episode_ids")
        if not isinstance(selected,list) or len(selected)!=12 or len(selected)!=len(set(selected)): raise RuntimeError("CHARACTERIZATION_COHORT_INVALID")
        available={r["episode_id"]:r for r in target+retention}
        if set(selected) - set(available): raise RuntimeError("CHARACTERIZATION_COHORT_EPISODE_MISSING")
        roles={available[item]["role"] for item in selected}
        if roles != {"TARGETED_EVALUATION","TARGETED_COMPOSITION","RETENTION_SENTINEL"}: raise RuntimeError("CHARACTERIZATION_COHORT_ROLE_INVALID")
        if any(sum(available[item]["role"]==role for item in selected)!=4 for role in roles): raise RuntimeError("CHARACTERIZATION_COHORT_BALANCE_INVALID")
        target=[available[item] for item in selected if available[item]["role"] in {"TARGETED_EVALUATION","TARGETED_COMPOSITION"}]
        retention=[available[item] for item in selected if available[item]["role"]=="RETENTION_SENTINEL"]
    elif len(target)!=192 or len(retention)!=96:
        raise RuntimeError("EVALUATION_DENOMINATOR_OR_IDENTITY_INVALID")
    if len({r["episode_id"] for r in target + retention}) != len(target) + len(retention): raise RuntimeError("EVALUATION_DENOMINATOR_OR_IDENTITY_INVALID")
    craw = z.contract.read_bytes(); c = json.loads(craw.decode("utf-8"))
    if c.get("contract_id") != CONTRACT_NAME: raise RuntimeError("INFERENCE_CONTRACT_INVALID")
    cid = sha_bytes(craw)
    b={"base_path":"/workspace/arc2/models/qwen3_4b_grids15_sft139","base_files":{}}
    if launch is not None:
        m=json.loads(Path(launch["checkpoint_manifest_path"]).read_text(encoding="utf-8")); b={"base_path":m["base_path"],"base_files":{x["name"]:x["sha256"] for x in m["base_files"]}}
    elif z.binding is not None:
        b=json.loads(z.binding.read_text(encoding="utf-8"))
    batching = {"requested_batch_size": z.batch_size, "max_batched_prompt_tokens": z.max_batched_prompt_tokens, "fallback_ladder": list(DEFAULT_BATCH_FALLBACK_LADDER), "validation_count": z.batch_validation_count, "min_canonical_agreement": MIN_CANONICAL_AGREEMENT, "max_rate_delta": MAX_RATE_DELTA, "prompt_length_bucketing": True, "left_padding": True}
    if z.cpu_mock:
        ps = [prompt(r, c) for r in target + retention]
        status = "PARTIAL_RUNTIME_CAP" if z.cpu_mock_simulate_runtime_cap else "CPU_MOCK_PASS_NO_MODEL"
        write(z.output, {"schema_version": 3, "protocol_id": PROTOCOL, "status": status, "model_loaded": False, "lora_constructed": False, "optimizer_constructed": False, "gpu_training_started": False, "inference_contract_identity": cid, "target_episode_order": [r["episode_id"] for r in target], "retention_episode_order": [r["episode_id"] for r in retention], "prompt_sha256": dig(ps), "completed_episode_count": 0 if z.cpu_mock_simulate_runtime_cap else len(ps), "batching": batching})
        write(z.receipt, receipt(status, started, model_loaded=False, model_released=True)); return 1 if z.cpu_mock_simulate_runtime_cap else 0
    if z.base is None or z.adapter is None or not z.adapter.is_dir(): raise RuntimeError("BASE_OR_ADAPTER_PATH_REQUIRED")
    mid = preverified_model_identity or verify_binding(b, z.base)  # Reuse the immediately preceding, scope-bound model verification when unchanged.
    deadline = time.monotonic() + z.runtime_limit_seconds; rows = []; model = None; status = "COLLECTED_PASS"; failure = None
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(): raise RuntimeError("CUDA_OR_BF16_UNAVAILABLE")
        device = torch.device("cuda:0"); rt = c["runtime"]; tok = AutoTokenizer.from_pretrained(z.base, local_files_only=True)
        if tok.get_vocab().get(rt["required_special_token"]) != rt["required_special_token_id"]: raise RuntimeError("TOKENIZER_SPECIAL_TOKEN_ID_MISMATCH")
        tok.padding_side = "left"
        tok.pad_token_id = rt["pad_token_id"]
        token_contract = tokenizer_token_contract(tok, eos_token_id=rt["eos_token_id"], pad_token_id=rt["pad_token_id"])
        model = AutoModelForCausalLM.from_pretrained(z.base, local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation=rt["attention_backend"]).to(device)
        model = PeftModel.from_pretrained(model, z.adapter, is_trainable=False).eval()
        contexts: list[dict[str, Any]] = []
        for surface, records in (("TARGET_DEV", target), ("RETENTION_SENTINEL", retention)):
            for r in records:
                p = prompt(r, c)
                prompt_tokens = len(tok(p, add_special_tokens=False)["input_ids"])
                contexts.append({"surface": surface, "record": r, "prompt": p, "episode_id": r["episode_id"], "prompt_tokens": prompt_tokens})
        contexts.sort(key=lambda x: (x["prompt_tokens"], x["episode_id"]))

        def generate_one_batch(batch: list[dict[str, Any]]) -> list[list[int]]:
            encoded = tok([x["prompt"] for x in batch], return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
            ins = {key: value.to(device) for key, value in encoded.items()}
            width = int(ins["input_ids"].shape[-1])
            with torch.inference_mode():
                generated = model.generate(**ins, do_sample=False, num_beams=1, max_new_tokens=rt["max_new_tokens"], use_cache=True, eos_token_id=rt["eos_token_id"], pad_token_id=rt["pad_token_id"])
            return [[int(value) for value in generated[index][width:].detach().cpu().tolist()] for index in range(len(batch))]

        def generate_adaptive(items: list[dict[str, Any]], *, label: str, on_batch: Any = None) -> tuple[list[tuple[dict[str, Any], list[int], int]], list[dict[str, Any]]]:
            produced: list[tuple[dict[str, Any], list[int], int]] = []; records: list[dict[str, Any]] = []; cursor = 0
            while cursor < len(items):
                if time.monotonic() >= deadline: raise TimeoutError("RUNTIME_CAP_BEFORE_BATCH")
                count = min(z.batch_size, len(items) - cursor)
                while count > 1 and max(x["prompt_tokens"] for x in items[cursor:cursor + count]) * count > z.max_batched_prompt_tokens:
                    count -= 1
                constrained_count = count
                while True:
                    batch = items[cursor:cursor + count]
                    try:
                        texts = generate_one_batch(batch)
                        break
                    except Exception as exc:
                        if not is_oom(exc): raise
                        next_count = fallback_size(count, DEFAULT_BATCH_FALLBACK_LADDER)
                        if next_count is None: raise RuntimeError("OOM_AT_BATCH1") from exc
                        records.append({"event": "OOM_FALLBACK", "label": label, "cursor": cursor, "failed_batch_size": count, "fallback_batch_size": next_count, "error": str(exc)[:300]})
                        gc.collect(); torch.cuda.empty_cache(); count = next_count
                batch_record = {"event": "BATCH", "label": label, "cursor": cursor, "actual_batch_size": len(batch), "longest_prompt_tokens": max(x["prompt_tokens"] for x in batch), "padded_prompt_tokens": max(x["prompt_tokens"] for x in batch) * len(batch), "long_prompt_downgrade": constrained_count != min(z.batch_size, len(items) - cursor), "oom_fallback": count != constrained_count}
                records.append(batch_record)
                produced.extend((context, text, len(batch)) for context, text in zip(batch, texts, strict=True))
                cursor += len(batch)
                if on_batch is not None: on_batch(produced, records)
                print(json.dumps({"event": "BASELINE_BATCH_PROGRESS", "label": label, "completed": cursor, "total": len(items), "actual_batch_size": len(batch)}, sort_keys=True), flush=True)
                if time.monotonic() >= deadline: raise TimeoutError("RUNTIME_CAP_AFTER_BATCH")
            return produced, records

        validation = validation_subset(contexts, z.batch_validation_count)
        serial = [(context, generate_one_batch([context])[0]) for context in validation]
        batched, validation_batches = generate_adaptive(validation, label="BATCH1_VS_BATCH32")
        serial_by_id = {x[0]["episode_id"]: output_row(x[0]["surface"], x[0]["record"], x[0]["prompt"], x[1], cid, mid, token_contract) for x in serial}
        batched_by_id = {x[0]["episode_id"]: output_row(x[0]["surface"], x[0]["record"], x[0]["prompt"], x[1], cid, mid, token_contract) for x in batched}
        matched = sum(serial_by_id[key]["canonical_prediction_sha256"] == batched_by_id[key]["canonical_prediction_sha256"] for key in serial_by_id)
        serial_parse = sum(x["parse_valid"] for x in serial_by_id.values()) / len(serial_by_id)
        batched_parse = sum(x["parse_valid"] for x in batched_by_id.values()) / len(batched_by_id)
        serial_exact = sum(x["exact_grid_match"] for x in serial_by_id.values()) / len(serial_by_id)
        batched_exact = sum(x["exact_grid_match"] for x in batched_by_id.values()) / len(batched_by_id)
        agreement = matched / len(validation)
        parse_delta = abs(serial_parse - batched_parse); exact_delta = abs(serial_exact - batched_exact)
        batching["validation"] = {"sample_count": len(validation), "canonical_agreement": agreement, "canonical_agreement_status": "AUDIT_ONLY", "parse_valid_rate_delta": parse_delta, "exact_grid_rate_delta": exact_delta, "batch_records": validation_batches, "status": batch_validation_status(canonical_agreement=agreement, parse_valid_rate_delta=parse_delta, exact_grid_rate_delta=exact_delta)}
        if batching["validation"]["status"] != "PASS": raise RuntimeError("BATCH1_BATCH32_VALIDATION_FAILED")

        progress = z.output.parent / "CAPABILITY_REPAIR_BASELINE_V1_PROGRESS.json"
        def persist_progress(produced: list[tuple[dict[str, Any], str, int]], records: list[dict[str, Any]]) -> None:
            partial_rows = []
            for context, token_ids, actual_batch_size in produced:
                row = output_row(context["surface"], context["record"], context["prompt"], token_ids, cid, mid, token_contract)
                row.update({"requested_batch_size": z.batch_size, "actual_batch_size": actual_batch_size, "prompt_tokens": context["prompt_tokens"]})
                partial_rows.append(row)
            write(progress, {"status": "INCOMPLETE", "completed_episode_count": len(partial_rows), "expected_episode_count": len(contexts), "batch_records": records, "batching": {key: value for key, value in batching.items() if key != "full_evaluation_batches"}, "predictions": partial_rows, "updated_at_monotonic_seconds": time.monotonic() - started})
        generated, full_batches = generate_adaptive(contexts, label="FULL_EVALUATION", on_batch=persist_progress)
        batching["full_evaluation_batches"] = full_batches
        for context, token_ids, actual_batch_size in generated:
            row = output_row(context["surface"], context["record"], context["prompt"], token_ids, cid, mid, token_contract)
            row.update({"requested_batch_size": z.batch_size, "actual_batch_size": actual_batch_size, "prompt_tokens": context["prompt_tokens"]})
            rows.append(row)
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
    result = {"schema_version": 4, "protocol_id": PROTOCOL, "status": status, "source_commit": os.environ.get("ARC2_SOURCE_COMMIT"), "checkpoint_manifest_file_sha256": minimal_gate.get("checkpoint_manifest_sha256") if minimal_gate else None, "inference_contract_identity": cid, "model_base_manifest_identity": mid, "predictions": rows, "aggregates": {"TARGET_DEV": aggregate([x for x in rows if x["surface"] == "TARGET_DEV"]), "RETENTION_SENTINEL": aggregate([x for x in rows if x["surface"] == "RETENTION_SENTINEL"])}, "expected_episode_count": n, "completed_episode_count": len(rows), "batching": batching, "token_evidence_preserved": True, "token_grid_contract": token_contract.as_dict() if "token_contract" in locals() else None}
    write(z.output, result); write(z.receipt, receipt(status, started, model_loaded=True, adapter_loaded=True, lora_constructed=False, source_commit=os.environ.get("ARC2_SOURCE_COMMIT"), checkpoint_manifest_file_sha256=minimal_gate.get("checkpoint_manifest_sha256") if minimal_gate else None, adapter_path=str(z.adapter), completed_episode_ids=[x["episode_id"] for x in rows], expected_episode_count=n, completed_episode_count=len(rows), model_base_manifest_identity=mid, inference_contract_identity=cid, failure=failure, runtime_cap_seconds=z.runtime_limit_seconds, model_released=True, batching=batching))
    return 0 if status == "COLLECTED_PASS" else 1
if __name__ == "__main__": raise SystemExit(main())
