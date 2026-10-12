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
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
RAW_COUNT = 72
B1_COUNT = 12
LADDER = (32, 16, 8, 4, 1)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def sha_path(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree_sha256(path):
    root = Path(path)
    require(root.is_dir(), "NATIVE_CONFIG_DIRECTORY_MISSING")
    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    require(bool(files), "NATIVE_CONFIG_DIRECTORY_EMPTY")
    for item in files:
        digest.update(item.relative_to(root).as_posix().encode("utf-8") + b"\0")
        digest.update(hashlib.sha256(item.read_bytes()).digest())
    return digest.hexdigest()


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


def bound_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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
    subset = binding.get("batch1_subset", {})
    require(subset.get("path") and subset.get("sha256"), "BATCH1_SUBSET_BINDING_MISSING")
    require_sha(bound_path(subset["path"]), subset["sha256"], "BATCH1_SUBSET_IDENTITY_MISMATCH")
    require(isinstance(binding.get("tokenizer"), dict) and binding["tokenizer"].get("runtime_path") and binding["tokenizer"].get("files"), "TOKENIZER_BINDING_MISSING")
    runtime = binding.get("runtime", {})
    require(runtime.get("native_config_dir") and runtime.get("native_config_tree_sha256"), "NATIVE_CONFIG_BINDING_MISSING")
    require(tree_sha256(runtime["native_config_dir"]) == runtime["native_config_tree_sha256"], "NATIVE_CONFIG_IDENTITY_MISMATCH")
    require(runtime.get("native_token_contract") == {"eos_token_id": 15, "pad_token_id": 13, "padding": "LEFT_EXPLICIT", "num_beams": 1}, "NATIVE_TOKEN_CONTRACT_BINDING")
    require(execution.get("mandatory_watchdog") is True, "MANDATORY_WATCHDOG_BINDING")
    for relative, digest in binding.get("source_hashes", {}).items():
        require_sha(ROOT / relative, digest, "SOURCE_IDENTITY_MISMATCH:" + relative)
    for label in ("protocol", "cohort"):
        item = binding.get(label, {})
        require(item.get("path") and item.get("sha256"), "BINDING_" + label.upper() + "_MISSING")
        require_sha(bound_path(item["path"]), item["sha256"], label.upper() + "_IDENTITY_MISMATCH")
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


def _components(grid):
    seen, output = set(), []
    for r,row in enumerate(grid):
        for c,value in enumerate(row):
            if not value or (r,c) in seen: continue
            todo=[(r,c)]; seen.add((r,c)); cells=[]
            while todo:
                x,y=todo.pop(); cells.append((x,y))
                for nx,ny in ((x-1,y),(x+1,y),(x,y-1),(x,y+1)):
                    if 0<=nx<len(grid) and 0<=ny<len(grid[0]) and (nx,ny) not in seen and grid[nx][ny]==value:
                        seen.add((nx,ny)); todo.append((nx,ny))
            minx,miny=min(x for x,_ in cells),min(y for _,y in cells)
            output.append((value,tuple(sorted((x-minx,y-miny) for x,y in cells)),tuple(sorted(cells))))
    return sorted(output)

def taxonomy(parsed_grid, expected_grid, parser_valid):
    if not parser_valid:
        return {"primary": "PARSER_FAILURE", "axes": ["PARSER_FAILURE"]}
    if parsed_grid == expected_grid:
        return {"primary": "EXACT", "axes": []}
    if not isinstance(parsed_grid, list) or not parsed_grid or len(parsed_grid) != len(expected_grid) or any(len(a) != len(b) for a, b in zip(parsed_grid, expected_grid)):
        return {"primary": "DIMENSIONS", "axes": ["DIMENSIONS"]}
    axes=[]; predicted,expected=_components(parsed_grid),_components(expected_grid)
    if len(predicted)!=len(expected):
        axes.append("CONNECTED_COMPONENT_COUNT")
    if sorted(x[0] for x in predicted)!=sorted(x[0] for x in expected):
        axes.append("COLOR")
    same_shape = sorted((x[0],x[1]) for x in predicted)==sorted((x[0],x[1]) for x in expected)
    if not same_shape: axes.append("GEOMETRY")
    if same_shape and sorted((x[0],x[1],x[2]) for x in predicted)!=sorted((x[0],x[1],x[2]) for x in expected): axes.append("SELECTED_OBJECT_IDENTITY")
    if not axes:
        axes.append("OUTPUT_CONSTRUCTION")
    order = ("CONNECTED_COMPONENT_COUNT", "COLOR", "GEOMETRY", "SELECTED_OBJECT_IDENTITY", "OUTPUT_CONSTRUCTION")
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
    cohort = load_json(bound_path(binding["cohort"]["path"]))
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

def _teacher_batch_agreement(primary, batch1, tolerance):
    require(set(primary)==set(batch1) and len(primary)==B1_COUNT, "BATCH1_TEACHER_PAIRING")
    p=[metric for key in sorted(primary) for metric in primary[key]]; b=[metric for key in sorted(batch1) for metric in batch1[key]]
    require(len(p)==len(b), "BATCH1_TEACHER_TOKEN_LAYOUT")
    rank=max(abs(x['rank']-y['rank']) for x,y in zip(p,b)); top1=abs(sum(x['top1'] for x in p)-sum(y['top1'] for y in b)); top2=abs(sum(x['top2'] for x in p)-sum(y['top2'] for y in b)); margin=max(abs(x['margin']-y['margin']) for x,y in zip(p,b)); nll=max(abs(x['nll']-y['nll']) for x,y in zip(p,b))
    passed=rank<=tolerance['rank_abs_delta_max'] and top1<=tolerance['top_count_delta_max'] and top2<=tolerance['top_count_delta_max'] and margin<=tolerance['margin_abs_delta_max'] and nll<=tolerance['nll_abs_delta_max']
    return {'status':'PASS' if passed else 'INVALID_NOT_INTERPRETABLE','rank_abs_delta_max':rank,'top1_count_delta':top1,'top2_count_delta':top2,'margin_abs_delta_max':margin,'nll_abs_delta_max':nll}


def scorer(binding, runtime, raw, raw_receipt, output_root, deadline):
    """The sole sidecar-opening boundary, valid only after complete raw freeze."""
    require(raw_receipt.get("status") == "RAW_FROZEN" and len(raw) == RAW_COUNT, "RAW_FREEZE_REQUIRED")
    require(raw_receipt.get("raw_sha256") == sha_path(Path(output_root) / "RAW_FROZEN.jsonl"), "RAW_FREEZE_IDENTITY")
    sidecar = runtime.open_sidecar(binding["sidecar"])  # intentionally deferred
    require(len(sidecar) == RAW_COUNT and set(sidecar) == {row["cohort_record_id"] for row in raw}, "SIDECAR_LAYOUT")
    subset=load_json(bound_path(binding['batch1_subset']['path'])); b1_ids=subset.get('record_ids')
    require(isinstance(b1_ids,list) and len(b1_ids)==B1_COUNT and len(set(b1_ids))==B1_COUNT and set(b1_ids)<=set(sidecar), 'BATCH1_SUBSET_LAYOUT')
    by_id = {row["cohort_record_id"]: row for row in raw}
    primary_logits = runtime.teacher_forced([by_id[key] for key in sorted(by_id)], sidecar, batch_size=32)
    b1_logits = runtime.teacher_forced([by_id[key] for key in b1_ids], sidecar, batch_size=1)
    # ``batch_size=1`` is a physical execution contract, not merely a model
    # option.  RealRuntime rejects a multi-record list at B1, so invoke it once
    # per predeclared subset record and preserve the frozen subset ordering.
    b1_generated = []
    for key in b1_ids:
        b1_generated.extend(runtime.generate([by_id[key]], batch_size=1, max_new_tokens=192))
    require(len(primary_logits) == RAW_COUNT and len(b1_logits) == B1_COUNT and len(b1_generated) == B1_COUNT, "PHYSICAL_BATCH1_COMPLETENESS")
    primary_b1 = {key: by_id[key] for key in b1_ids}
    observed_b1 = {key: {"generated_token_ids": row.get("generated_token_ids")} for key, row in zip(b1_ids, b1_generated, strict=True)}
    generation_gate = _batch_agreement(primary_b1, observed_b1)
    primary_metrics={key:[metric(logit,token) for logit,token in zip(primary_logits[key],sidecar[key]['target_token_ids'],strict=True)] for key in primary_logits}
    b1_metrics={key:[metric(logit,token) for logit,token in zip(b1_logits[key],sidecar[key]['target_token_ids'],strict=True)] for key in b1_logits}
    teacher_gate = _teacher_batch_agreement({key:primary_metrics[key] for key in b1_ids}, b1_metrics, binding['teacher_forced_tolerance'])
    rows = []
    for key in sorted(by_id):
        target = sidecar[key]
        token_metrics = primary_metrics[key]
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
    receipt = {"status": "COMPLETE" if generation_gate["status"] == "PASS" and teacher_gate['status']=='PASS' else "INVALID_NOT_INTERPRETABLE",
               "records": RAW_COUNT, "teacher_forced_forwards": RAW_COUNT, "batch1_teacher_forced_forwards": B1_COUNT,
                "batch1_generations": B1_COUNT, "batch1_generation_gate": generation_gate, "batch1_teacher_forced_gate":teacher_gate,"optimizer_steps": 0, "backward_calls": 0,
               "metrics_sha256": sha_path(Path(output_root) / "SCORED_METRICS.jsonl"), "target_token_ids_persisted": False,
               "teacher_forward_events": list(getattr(runtime, "teacher_forward_events", []))}
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

def enforce_hard_cap(command, terminal_path, seconds=3600, *, environment=None):
    """Run exactly one child at the production wall-clock boundary.

    The parent owns timeout receipts.  A timed-out model child is killed as a
    process group where the platform supports it, so it cannot continue after
    a timeout terminal receipt has become durable.
    """
    terminal_path = Path(terminal_path)
    child = subprocess.Popen(command, start_new_session=True, env=environment)
    try:
        returncode = child.wait(timeout=seconds)
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                import signal
                os.killpg(child.pid, signal.SIGKILL)
            else:
                child.kill()
        finally:
            child.wait()
        if not terminal_path.exists():
            write_json(terminal_path, {"status": "FAILED_NO_UPDATE_TIMEOUT", "hard_cap_seconds": seconds,
                                       "optimizer_steps": 0, "backward_calls": 0,
                                       "targets_accessed": False})
        return None
    if returncode and not terminal_path.exists():
        write_json(terminal_path, {"status": "FAILED_NO_UPDATE_CHILD_EXIT", "returncode": returncode,
                                   "optimizer_steps": 0, "backward_calls": 0,
                                   "targets_accessed": False})
    return returncode


def native_prompt_messages(record):
    """Return the established NVARC message contract for one target-free row."""
    task = record.get("task")
    require(isinstance(task, dict) and isinstance(task.get("train"), list) and isinstance(task.get("test"), list)
            and len(task["test"]) == 1 and "output" not in task["test"][0], "TARGET_FREE_PROMPT_TASK_REQUIRED")
    from types import SimpleNamespace
    from inference.nvarc_native import native_messages
    def grid(value): return SimpleNamespace(values=value)
    def pair(value): return SimpleNamespace(input=grid(value["input"]), output=grid(value["output"]))
    native_task = SimpleNamespace(train=[pair(item) for item in task["train"]],
                                  test=[SimpleNamespace(input=grid(task["test"][0]["input"]))])
    return native_messages(native_task, 0)


def native_prompt_token_ids(tokenizer, record):
    """Tokenize only through the checkpoint-native chat template."""
    encoded = tokenizer.apply_chat_template(native_prompt_messages(record), add_generation_prompt=True,
                                            tokenize=True, return_tensors="pt", return_dict=True)
    ids = encoded["input_ids"][0]
    return [int(value) for value in ids.tolist()]


class RealRuntime:
    """Lazy RunPod implementation using the established NVARC runtime contract.

    This class is never instantiated by CPU validation.  Both greedy
    generation and teacher forcing build target-free native prompts using the
    checkpoint-native tokenizer; sidecar token IDs enter only after raw freeze.
    """
    def __init__(self, runtime_config):
        self.config, self.model, self.tokenizer = runtime_config, None, None
        self.teacher_forward_events = []

    def check_dependencies(self):
        import torch, transformers, peft  # noqa: F401
        require(torch.cuda.is_available(), "CUDA_REQUIRED")

    def verify_adapter(self, path, digest):
        require_sha(path, digest, "V7_ADAPTER_IDENTITY_MISMATCH")

    def verify_tokenizer(self, descriptor):
        root = Path(descriptor["runtime_path"])
        for name, digest in descriptor["files"].items():
            require_sha(root / name, digest, "TOKENIZER_IDENTITY_MISMATCH:" + name)

    def _load(self):
        if self.model is not None:
            return
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM
        from inference.nvarc_native import checkpoint_native_tokenizer
        native_config = Path(self.config["native_config_dir"])
        require(native_config.is_dir(), "NATIVE_TOKENIZER_CONFIG_MISSING")
        self.tokenizer, metadata = checkpoint_native_tokenizer(Path(self.config["base_path"]), native_config)
        contract = self.config.get("native_token_contract", {})
        require(contract == {"eos_token_id": 15, "pad_token_id": 13, "padding": "LEFT_EXPLICIT", "num_beams": 1},
                "NATIVE_TOKEN_CONTRACT_MISMATCH")
        require(metadata["eos_token_id"] == 15 and metadata["pad_token_id"] == 13,
                "NATIVE_TOKENIZER_SPECIAL_IDS")
        base = AutoModelForCausalLM.from_pretrained(self.config["base_path"], local_files_only=True,
                                                     trust_remote_code=False, torch_dtype=torch.bfloat16,
                                                     low_cpu_mem_usage=True).to("cuda:0")
        self.model = PeftModel.from_pretrained(base, self.config["adapter_dir"], is_trainable=False).eval()

    def _prompt_ids(self, record):
        self._load()
        return native_prompt_token_ids(self.tokenizer, record)

    def generate(self, records, batch_size, max_new_tokens):
        """Physical native-Greedy batch using explicit left padding and mask."""
        self._load()
        import torch
        from inference.nvarc_native import parse_native_grid
        require(len(records) <= batch_size and batch_size in LADDER, "GENERATION_BATCH_CONTRACT")
        rows = [self._prompt_ids(record) for record in records]
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
        attention = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            ids[index, width - len(row):] = torch.tensor(row, dtype=torch.long)
            attention[index, width - len(row):] = 1
        with torch.inference_mode():
            generated = self.model.generate(input_ids=ids.to("cuda:0"), attention_mask=attention.to("cuda:0"),
                                             do_sample=False, num_beams=1, use_cache=True,
                                             max_new_tokens=max_new_tokens, eos_token_id=self.tokenizer.eos_token_id,
                                             pad_token_id=self.tokenizer.pad_token_id)
        result = []
        for row in generated:
            suffix = row[width:].detach().cpu()
            eos = (suffix == int(self.tokenizer.eos_token_id)).nonzero(as_tuple=False)
            if len(eos):
                suffix = suffix[:int(eos[0].item()) + 1]
            else:
                pad = (suffix == int(self.tokenizer.pad_token_id)).nonzero(as_tuple=False)
                if len(pad):
                    suffix = suffix[:int(pad[0].item())]
            token_ids = [int(value) for value in suffix.tolist()]
            parsed = parse_native_grid(self.tokenizer.decode(suffix, skip_special_tokens=True))
            result.append({"generated_token_ids": token_ids, "parsed_grid": parsed, "parser_valid": parsed is not None})
        return result

    def teacher_forced(self, raw_rows, sidecar, batch_size):
        """Actual padded/masked forward batches; never a serial Batch1 loop."""
        self._load()
        import torch
        require(batch_size in LADDER, "TEACHER_BATCH_CONTRACT")
        materialized = [(row["cohort_record_id"], self._prompt_ids(row),
                         [int(value) for value in sidecar[row["cohort_record_id"]]["target_token_ids"]])
                        for row in raw_rows]
        output, cursor, size = {}, 0, batch_size
        while cursor < len(materialized):
            batch = materialized[cursor:cursor + size]
            sequences = [prompt + target for _key, prompt, target in batch]
            width = max(map(len, sequences))
            ids = torch.full((len(batch), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            positions = torch.zeros((len(batch), width), dtype=torch.long)
            offsets = []
            for index, sequence in enumerate(sequences):
                offset = width - len(sequence); offsets.append(offset)
                ids[index, offset:] = torch.tensor(sequence, dtype=torch.long)
                mask[index, offset:] = 1
                positions[index, offset:] = torch.arange(len(sequence), dtype=torch.long)
            try:
                with torch.inference_mode():
                    logits = self.model(input_ids=ids.to("cuda:0"), attention_mask=mask.to("cuda:0"),
                                        position_ids=positions.to("cuda:0"), use_cache=False).logits
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or size == 1:
                    raise
                next_size = LADDER[LADDER.index(size) + 1]
                self.teacher_forward_events.append({"event": "OOM_FALLBACK", "from": size, "to": next_size,
                                                    "reason": "CUDA_OOM"})
                self.empty_cache(); size = next_size; continue
            self.teacher_forward_events.append({"event": "PHYSICAL_TEACHER_FORWARD", "effective_batch_size": len(batch),
                                                "requested_batch_size": batch_size, "records": [key for key, _p, _t in batch]})
            for index, (key, prompt, target) in enumerate(batch):
                start = offsets[index] + len(prompt) - 1
                selected = logits[index, start:start + len(target)].float().detach().cpu()
                require(int(selected.shape[0]) == len(target), "TEACHER_CAUSAL_ALIGNMENT")
                output[key] = [[float(value) for value in vector] for vector in selected.tolist()]
            cursor += len(batch)
        return output

    def open_sidecar(self, descriptor):
        remote_path = descriptor.get("remote_path")
        require(isinstance(remote_path, str) and remote_path.startswith("/"), "REMOTE_SIDECAR_PATH_REQUIRED")
        require_sha(remote_path, descriptor["sha256"], "SIDECAR_IDENTITY_MISMATCH")
        rows = load_json(remote_path).get("records", [])
        return {row["cohort_record_id"]: row for row in rows}

    def decode_grid(self, token_ids):
        self._load()
        from inference.nvarc_native import parse_native_grid
        return parse_native_grid(self.tokenizer.decode(token_ids, skip_special_tokens=True))

    def empty_cache(self):
        import torch
        torch.cuda.empty_cache()


def production_child_command(args):
    return [sys.executable, str(Path(__file__).resolve()), "--binding", str(args.binding),
            "--authorization", str(args.authorization), "--output-root", str(args.output_root),
            "--control-root", str(args.control_root), "--execute-claimed"]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--control-root", type=Path)
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--execute-claimed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.authorization or not args.output_root or not args.control_root:
        raise SystemExit("REFUSE_UNAUTHORIZED_EXECUTION")
    binding = validate_binding(args.binding)
    require(binding["execution"].get("mandatory_watchdog") is True, "MANDATORY_WATCHDOG_BINDING")
    if args.execute_claimed:
        require(os.environ.get("ARC2_V7_WATCHDOG_CHILD") == "1", "DIRECT_CHILD_EXECUTION_REFUSED")
        runtime = RealRuntime(binding["runtime"])
        print(json.dumps(launch(args.binding, runtime, args.authorization, args.output_root, args.control_root), sort_keys=True))
        return
    if not args.launch:
        raise SystemExit("REFUSE_UNAUTHORIZED_EXECUTION")
    require(not args.output_root.exists(), "FRESH_OUTPUT_ROOT_REQUIRED")
    require(not (args.control_root / "TERMINAL_RECEIPT.json").exists(), "DUPLICATE_TERMINAL_RECEIPT")
    write_json(args.control_root / "WATCHDOG_INVOCATION.json", {
        "status": "WATCHDOG_PARENT_STARTED", "binding_sha256": sha_path(args.binding),
        "hard_cap_seconds": 3600, "child_mode": "--execute-claimed",
    })
    environment = os.environ.copy(); environment["ARC2_V7_WATCHDOG_CHILD"] = "1"
    code = enforce_hard_cap(production_child_command(args), args.control_root / "TERMINAL_RECEIPT.json",
                            seconds=3600, environment=environment)
    if code != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
