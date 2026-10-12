#!/usr/bin/env python3
"""Executable V7 component diagnostic with an import-safe CPU integration path.

The only code path that can load a model is ``RealRuntime`` and it is reached
only by the future, separately authorised ``--launch`` command.  CPU tests use
the same launcher, worker and scorer through ``FakeRuntime``; they never import
torch, transformers or the sealed sidecar.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW_COUNT = 72
B1_COUNT = 12
LADDER = (32, 16, 8, 4, 1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n")
    os.replace(temporary, path)


def write_jsonl(path, values):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical(row) + b"\n" for row in values))
    os.replace(temporary, path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, code):
    if not condition:
        raise RuntimeError(code)


def require_sha(path, expected, code):
    require(sha_path(path) == expected, code)


def exclusive_claim(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc:
        raise RuntimeError("DUPLICATE_OR_CONSUMED_ONE_SHOT") from exc
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(canonical(payload) + b"\n")


def validate_binding(binding_path):
    binding = load_json(binding_path)
    execution = binding.get("execution", {})
    require(binding.get("protocol_sha256"), "BINDING_PROTOCOL_MISSING")
    require(execution.get("checkpoint") == "V7", "CHECKPOINT_NOT_V7")
    require(execution.get("retry") is False, "RETRY_MUST_BE_FALSE")
    require(execution.get("optimizer_steps") == 0 and execution.get("backward_calls") == 0,
            "NO_UPDATE_CONTRACT")
    require(execution.get("hard_cap_seconds") == 3600, "HARD_CAP_MISMATCH")
    require(execution.get("requested_batch_size") == 32, "BATCH32_REQUIRED")
    require(execution.get("fallback_ladder") == list(LADDER), "OOM_LADDER_MISMATCH")
    require(isinstance(binding.get("tokenizer"), dict) and binding["tokenizer"].get("runtime_path") and binding["tokenizer"].get("files"), "TOKENIZER_BINDING_MISSING")
    for relative, digest in binding.get("source_hashes", {}).items():
        require_sha(ROOT / relative, digest, "SOURCE_IDENTITY_MISMATCH:" + relative)
    for label in ("protocol", "cohort"):
        item = binding.get(label, {})
        require(item.get("path") and item.get("sha256"), "BINDING_" + label.upper() + "_MISSING")
        require_sha(item["path"], item["sha256"], label.upper() + "_IDENTITY_MISMATCH")
    return binding


def metric(logits, target):
    values = [float(value) for value in logits]
    require(values and 0 <= int(target) < len(values), "TARGET_LOGIT_INVALID")
    correct = values[int(target)]
    maximum = max(values)
    logz = maximum + math.log(sum(math.exp(value - maximum) for value in values))
    rank = 1 + sum(value > correct for value in values)
    other = max(value for index, value in enumerate(values) if index != int(target)) if len(values) > 1 else float("-inf")
    return {"rank": rank, "top1": rank == 1, "top2": rank <= 2,
            "margin": correct - other, "nll": logz - correct}


def aggregate(metrics):
    require(bool(metrics), "EMPTY_METRICS")
    return {
        "tokens": len(metrics),
        "top1_count": sum(item["top1"] for item in metrics),
        "top2_count": sum(item["top2"] for item in metrics),
        "mean_margin": sum(item["margin"] for item in metrics) / len(metrics),
        "token_weighted_nll": sum(item["nll"] for item in metrics) / len(metrics),
        "sequence_normalized_nll": sum(item["nll"] for item in metrics) / len(metrics),
    }


def first_divergence(left, right):
    for index, (a, b) in enumerate(zip(left, right)):
        if int(a) != int(b):
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def taxonomy(parsed_grid, expected_grid, parser_valid):
    if not parser_valid:
        return {"primary": "PARSER_FAILURE", "axes": ["PARSER_FAILURE"]}
    if parsed_grid == expected_grid:
        return {"primary": "EXACT", "axes": []}
    if not isinstance(parsed_grid, list) or not parsed_grid or len(parsed_grid) != len(expected_grid) or any(len(a) != len(b) for a, b in zip(parsed_grid, expected_grid)):
        return {"primary": "DIMENSIONS", "axes": ["DIMENSIONS"]}
    axes = []
    if sum(x != 0 for row in parsed_grid for x in row) != sum(x != 0 for row in expected_grid for x in row):
        axes.append("CONNECTED_COMPONENT_COUNT")
    if sorted(x for row in parsed_grid for x in row if x) != sorted(x for row in expected_grid for x in row if x):
        axes.append("COLOR")
    if not axes:
        axes.append("OUTPUT_CONSTRUCTION")
    order = ("CONNECTED_COMPONENT_COUNT", "SELECTED_OBJECT_IDENTITY", "GEOMETRY", "COLOR", "OUTPUT_CONSTRUCTION")
    return {"primary": next(axis for axis in order if axis in axes), "axes": axes}


def _generate(records, runtime, max_new_tokens, deadline):
    size, offset, raw, fallback = 32, 0, [], []
    while offset < len(records):
        require(time.monotonic() <= deadline, "EXECUTION_TIMEOUT")
        batch = records[offset:offset + size]
        try:
            got = runtime.generate(batch, batch_size=size, max_new_tokens=max_new_tokens)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower() or size == 1:
                raise
            next_size = LADDER[LADDER.index(size) + 1]
            fallback.append({"from": size, "to": next_size, "reason": "CUDA_OOM"})
            size = next_size
            runtime.empty_cache()
            continue
        require(len(got) == len(batch), "MALFORMED_GENERATION_COMPLETENESS")
        for record, result in zip(batch, got, strict=True):
            tokens = result.get("generated_token_ids")
            require(isinstance(tokens, list) and all(isinstance(x, int) for x in tokens), "MALFORMED_GENERATION_TOKENS")
            raw.append({"cohort_record_id": record["cohort_record_id"], "prompt_sha256": record["prompt_task_sha256"],
                        "generated_token_ids": tokens, "parsed_grid": result.get("parsed_grid"),
                        "parser_valid": bool(result.get("parser_valid")), "effective_batch_size": size,
                        "task": record.get("task")})
        offset += len(batch)
    return raw, fallback


def worker(binding, runtime, output_root, deadline):
    """Generate and freeze all raw output before this function returns."""
    cohort = load_json(binding["cohort"]["path"])
    records = cohort.get("records", cohort)
    require(isinstance(records, list) and len(records) == RAW_COUNT, "COHORT_CARDINALITY")
    require(len({row.get("cohort_record_id") for row in records}) == RAW_COUNT, "COHORT_DUPLICATE")
    raw, fallback = _generate(records, runtime, 192, deadline)
    require(len(raw) == RAW_COUNT and len({row["cohort_record_id"] for row in raw}) == RAW_COUNT, "RAW_COMPLETENESS")
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=False)
    raw_path = output_root / "RAW_FROZEN.jsonl"
    write_jsonl(raw_path, raw)
    receipt = {"status": "RAW_FROZEN", "raw_sha256": sha_path(raw_path), "records": RAW_COUNT,
               "fallback_events": fallback, "sidecar_accessed": False, "optimizer_steps": 0,
               "backward_calls": 0, "raw_generation_calls": RAW_COUNT}
    write_json(output_root / "RAW_FREEZE_RECEIPT.json", receipt)
    return raw, receipt


def _batch_agreement(primary, batch1):
    require(set(primary) == set(batch1) and len(primary) == B1_COUNT, "BATCH1_PAIRING")
    # The frozen tolerance is deliberately strict for injected comparison.
    passed = all(primary[key]["generated_token_ids"] == batch1[key]["generated_token_ids"] for key in primary)
    return {"status": "PASS" if passed else "INVALID_NOT_INTERPRETABLE", "physical_primary": B1_COUNT,
            "physical_batch1": B1_COUNT, "generated_token_identity": passed}


def scorer(binding, runtime, raw, raw_receipt, output_root, deadline):
    """The sole sidecar-opening boundary, valid only after complete raw freeze."""
    require(raw_receipt.get("status") == "RAW_FROZEN" and len(raw) == RAW_COUNT, "RAW_FREEZE_REQUIRED")
    require(raw_receipt.get("raw_sha256") == sha_path(Path(output_root) / "RAW_FROZEN.jsonl"), "RAW_FREEZE_IDENTITY")
    sidecar = runtime.open_sidecar(binding["sidecar"])  # intentionally deferred
    require(len(sidecar) == RAW_COUNT and set(sidecar) == {row["cohort_record_id"] for row in raw}, "SIDECAR_LAYOUT")
    b1_ids = sorted(sidecar)[:B1_COUNT]
    by_id = {row["cohort_record_id"]: row for row in raw}
    primary_logits = runtime.teacher_forced([by_id[key] for key in sorted(by_id)], sidecar, batch_size=32)
    b1_logits = runtime.teacher_forced([by_id[key] for key in b1_ids], sidecar, batch_size=1)
    b1_generated = runtime.generate([by_id[key] for key in b1_ids], batch_size=1, max_new_tokens=192)
    require(len(primary_logits) == RAW_COUNT and len(b1_logits) == B1_COUNT and len(b1_generated) == B1_COUNT, "PHYSICAL_BATCH1_COMPLETENESS")
    primary_b1 = {key: by_id[key] for key in b1_ids}
    observed_b1 = {key: {"generated_token_ids": row.get("generated_token_ids")} for key, row in zip(b1_ids, b1_generated, strict=True)}
    gate = _batch_agreement(primary_b1, observed_b1)
    rows = []
    for key in sorted(by_id):
        target = sidecar[key]
        token_metrics = [metric(logit, token) for logit, token in zip(primary_logits[key], target["target_token_ids"], strict=True)]
        expected_grid = target.get("target_grid")
        if expected_grid is None:
            expected_grid = runtime.decode_grid(target["target_token_ids"])
        require(isinstance(expected_grid, list), "TARGET_GRID_DECODE_FAILURE")
        rows.append({"cohort_record_id": key, "prompt_sha256": by_id[key]["prompt_sha256"],
                     "teacher_forced": aggregate(token_metrics),
                     "first_divergence": first_divergence(by_id[key]["generated_token_ids"], target["target_token_ids"]),
                     "exact_grid": by_id[key]["parsed_grid"] == expected_grid,
                     "family_aware_semantic_taxonomy": taxonomy(by_id[key]["parsed_grid"], expected_grid, by_id[key]["parser_valid"])})
    require(time.monotonic() <= deadline, "EXECUTION_TIMEOUT")
    write_jsonl(Path(output_root) / "SCORED_METRICS.jsonl", rows)
    receipt = {"status": "COMPLETE" if gate["status"] == "PASS" else "INVALID_NOT_INTERPRETABLE",
               "records": RAW_COUNT, "teacher_forced_forwards": RAW_COUNT, "batch1_teacher_forced_forwards": B1_COUNT,
               "batch1_generations": B1_COUNT, "batch1_gate": gate, "optimizer_steps": 0, "backward_calls": 0,
               "metrics_sha256": sha_path(Path(output_root) / "SCORED_METRICS.jsonl"), "target_token_ids_persisted": False}
    write_json(Path(output_root) / "TERMINAL_RECEIPT.json", receipt)
    return receipt


def launch(binding_path, runtime, authorization_path, output_root, control_root):
    """One executable process path; caller supplies real or injected runtime."""
    binding = validate_binding(binding_path)
    authorization = load_json(authorization_path)
    require(authorization.get("status") == "VALID_UNCONSUMED", "AUTHORIZATION_INVALID")
    require(authorization.get("binding_sha256") == sha_path(binding_path), "AUTHORIZATION_BINDING_MISMATCH")
    output_root, control_root = Path(output_root), Path(control_root)
    require(not output_root.exists(), "FRESH_OUTPUT_ROOT_REQUIRED")
    require(not (control_root / "TERMINAL_RECEIPT.json").exists(), "DUPLICATE_TERMINAL_RECEIPT")
    runtime.check_dependencies()
    runtime.verify_tokenizer(binding["tokenizer"])
    runtime.verify_adapter(binding["checkpoint"]["adapter_model_path"], binding["checkpoint"]["adapter_model_sha256"])
    claim = Path(authorization_path).with_suffix(Path(authorization_path).suffix + ".claimed")
    exclusive_claim(claim, {"binding_sha256": sha_path(binding_path), "output_root": str(output_root), "claimed_at": time.time()})
    deadline = time.monotonic() + 3600
    try:
        raw, freeze = worker(binding, runtime, output_root, deadline)
        terminal = scorer(binding, runtime, raw, freeze, output_root, deadline)
        write_json(control_root / "TERMINAL_RECEIPT.json", terminal)
        return terminal
    except Exception as exc:
        terminal = {"status": "FAILED_NO_UPDATE", "exception_type": type(exc).__name__, "exception": str(exc),
                    "optimizer_steps": 0, "backward_calls": 0, "targets_accessed": False}
        write_json(control_root / "TERMINAL_RECEIPT.json", terminal)
        raise


class RealRuntime:
    """Lazy RunPod implementation.  It is never instantiated by CPU validation."""
    def __init__(self, runtime_config): self.config, self.model, self.tokenizer = runtime_config, None, None
    def check_dependencies(self):
        import torch, transformers, peft  # noqa: F401
        require(torch.cuda.is_available(), "CUDA_REQUIRED")
    def verify_adapter(self, path, digest): require_sha(path, digest, "V7_ADAPTER_IDENTITY_MISMATCH")
    def verify_tokenizer(self, descriptor):
        root = Path(descriptor["runtime_path"])
        for name, digest in descriptor["files"].items():
            require_sha(root / name, digest, "TOKENIZER_IDENTITY_MISMATCH:" + name)
    def _load(self):
        if self.model is not None: return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base = AutoModelForCausalLM.from_pretrained(self.config["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True).to("cuda:0")
        self.model = PeftModel.from_pretrained(base, self.config["adapter_dir"], is_trainable=False).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config["base_path"], local_files_only=True)
    def _prompt(self, record):
        """Build the native chat prompt only from the target-free cohort task."""
        task = record.get("task")
        require(isinstance(task, dict) and isinstance(task.get("train"), list) and isinstance(task.get("test"), list), "TARGET_FREE_PROMPT_TASK_REQUIRED")
        def grid(value): return json.dumps(value, separators=(",", ":"))
        messages = []
        for pair in task["train"]:
            messages.extend(({"role": "user", "content": grid(pair["input"])}, {"role": "assistant", "content": grid(pair["output"])}))
        messages.append({"role": "user", "content": grid(task["test"][0]["input"])})
        try:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except (AttributeError, TypeError):
            return "\n".join(item["role"] + ":" + item["content"] for item in messages) + "\nassistant:"
    def generate(self, records, batch_size, max_new_tokens):
        self._load(); import torch
        prompts = [self._prompt(record) for record in records]
        encoded = self.tokenizer(prompts, return_tensors="pt", padding=True).to("cuda:0")
        with torch.inference_mode(): output = self.model.generate(**encoded, do_sample=False, max_new_tokens=max_new_tokens)
        from inference.nvarc_native import parse_native_grid
        width = int(encoded.input_ids.shape[1])
        result = []
        for row in output:
            tokens = [int(x) for x in row[width:].tolist()]
            parsed = parse_native_grid(self.tokenizer.decode(tokens, skip_special_tokens=False))
            result.append({"generated_token_ids": tokens, "parsed_grid": parsed, "parser_valid": parsed is not None})
        return result
    def teacher_forced(self, raw_rows, sidecar, batch_size):
        self._load(); import torch
        out = {}
        for offset in range(0, len(raw_rows), batch_size):
            for row in raw_rows[offset:offset + batch_size]:
                prompt = self.tokenizer(self._prompt(row), add_special_tokens=False)["input_ids"]
                target = sidecar[row["cohort_record_id"]]["target_token_ids"]
                ids = torch.tensor([prompt + target], device="cuda:0")
                with torch.inference_mode(): logits = self.model(input_ids=ids, use_cache=False).logits[0]
                out[row["cohort_record_id"]] = [logits[len(prompt) - 1 + i].float().tolist() for i in range(len(target))]
        return out
    def open_sidecar(self, descriptor):
        require_sha(descriptor["path"], descriptor["sha256"], "SIDECAR_IDENTITY_MISMATCH")
        rows = load_json(descriptor["path"]).get("records", [])
        return {row["cohort_record_id"]: row for row in rows}
    def decode_grid(self, token_ids):
        from inference.nvarc_native import parse_native_grid
        return parse_native_grid(self.tokenizer.decode(token_ids, skip_special_tokens=False))
    def empty_cache(self):
        import torch; torch.cuda.empty_cache()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--control-root", type=Path); parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not args.launch or not args.authorization or not args.output_root or not args.control_root:
        raise SystemExit("REFUSE_UNAUTHORIZED_EXECUTION")
    binding = validate_binding(args.binding)
    runtime = RealRuntime(binding["runtime"])
    print(json.dumps(launch(args.binding, runtime, args.authorization, args.output_root, args.control_root), sort_keys=True))


if __name__ == "__main__": main()
