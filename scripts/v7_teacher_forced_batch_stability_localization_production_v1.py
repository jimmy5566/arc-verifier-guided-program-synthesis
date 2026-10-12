#!/usr/bin/env python3
"""Bounded production entrypoint for V7 Batch1 teacher-forced localization.

The module is import-safe for CPU fixtures.  ``ProductionRuntime`` is lazy and
is instantiated only in a separately authorised watchdog child.  Target token
IDs are admitted only after all identity, output-root and one-shot checks pass,
and only to form the twelve teacher-forced inputs.  They are never written.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


L = _module("v7_batch_localization", "v7_teacher_forced_batch_stability_localization_v1.py")
V5 = _module("v7_v5_runtime", "v7_component_diagnostic_executable_v2.py")


def canonical(value): return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
def sha_path(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def require(condition, code):
    if not condition: raise RuntimeError(code)


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(value) + b"\n"); os.replace(temporary, path)


def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(b"".join(canonical(row) + b"\n" for row in rows)); os.replace(temporary, path)


def load_json(path): return json.loads(Path(path).read_text(encoding="utf-8"))


def exclusive_claim(path, payload):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    try: descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    except FileExistsError as exc: raise RuntimeError("DUPLICATE_OR_CONSUMED_ONE_SHOT") from exc
    with os.fdopen(descriptor, "wb") as handle: handle.write(canonical(payload) + b"\n")


def _bound_path(value):
    path = Path(value); return path if path.is_absolute() else ROOT / path


def _clean_tracked_source():
    require(subprocess.run(["git", "diff", "--quiet"], cwd=ROOT).returncode == 0, "DIRTY_TRACKED_SOURCE")
    require(subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0, "DIRTY_STAGED_SOURCE")


def validate_binding(binding_path):
    binding = load_json(binding_path)
    require(binding.get("protocol_id") == "V7_TEACHER_FORCED_BATCH_STABILITY_LOCALIZATION_V1", "PROTOCOL_ID_MISMATCH")
    require(binding.get("status") == "PREAUTH_CPU_REPAIRED_NOT_GPU_AUTHORIZED", "BINDING_NOT_PREAUTH")
    execution = binding.get("execution", {})
    require(execution == {"generation_calls": 0, "optimizer_steps": 0, "backward_calls": 0, "retry": False,
                          "physical_batch_sizes": [1, 2, 4, 8, 12], "minimum_repeats_per_condition": 2,
                          "records": 12, "hard_cap_seconds": 3600, "mandatory_watchdog": True}, "EXECUTION_CONTRACT")
    for label in ("protocol", "cohort", "batch1_subset"):
        item = binding.get(label, {})
        require(item.get("path") and item.get("sha256"), "BINDING_" + label.upper() + "_MISSING")
        require(sha_path(_bound_path(item["path"])) == item["sha256"], label.upper() + "_IDENTITY_MISMATCH")
    sidecar = binding.get("sidecar", {})
    require(sidecar.get("path") and sidecar.get("sha256") and sidecar.get("access") == "TEACHER_FORCED_WORKER_ONLY", "SIDE_CAR_ACCESS_CONTRACT")
    # The Windows path is provenance only.  Remote preflight hashes precisely
    # the bound POSIX runtime path immediately before worker-only access.
    require(isinstance(sidecar.get("remote_path"), str) and sidecar["remote_path"].startswith("/"), "REMOTE_SIDECAR_PATH_REQUIRED")
    for source, digest in binding.get("source_hashes", {}).items():
        require(sha_path(ROOT / source) == digest, "SOURCE_IDENTITY_MISMATCH:" + source)
    identities = binding.get("identities", {})
    require(identities.get("checkpoint", {}).get("adapter_model_sha256") == "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3", "V7_ADAPTER_IDENTITY")
    require(identities.get("runtime", {}).get("native_config_tree_sha256") == "f2701b5c1458d29a4be62933ef7bcd910ad9ae75a671cb55341a85e43896a17b", "TOKENIZER_CONFIG_IDENTITY")
    return binding


def _runtime_protocol(binding):
    protocol = load_json(_bound_path(binding["protocol"]["path"]))
    L.validate_protocol(protocol)
    return protocol


def _verify_worker_only_sidecar(binding, cohort_records, runtime):
    """Open exactly twelve hashed target-token rows only after preclaim gates."""
    subset = load_json(_bound_path(binding["batch1_subset"]["path"]))["record_ids"]
    require(isinstance(subset, list) and len(subset) == 12 and len(set(subset)) == 12, "BATCH1_SUBSET_LAYOUT")
    by_id = {str(row["cohort_record_id"]): row for row in cohort_records}
    require(len(by_id) == 72 and set(subset) <= set(by_id), "COHORT_SUBSET_LAYOUT")
    sidecar_rows = runtime.open_worker_only_sidecar(binding["sidecar"])
    require(isinstance(sidecar_rows, list) and len(sidecar_rows) == 72, "SIDE_CAR_CARDINALITY")
    sidecar = {str(row.get("cohort_record_id")): row for row in sidecar_rows}
    require(len(sidecar) == 72 and set(sidecar) == set(by_id), "SIDE_CAR_UNIQUE_COHORT_IDENTITY")
    prepared = []
    for manifest_index, record_id in enumerate(subset):
        target = sidecar[record_id]
        tokens = target.get("target_token_ids")
        require(isinstance(tokens, list) and tokens and all(isinstance(token, int) for token in tokens), "SIDE_CAR_TARGET_LAYOUT")
        prepared.append({"cohort_record_id": record_id, "manifest_index": manifest_index,
                         "prompt_token_ids": runtime.prompt_token_ids(by_id[record_id]), "target_token_ids": tokens})
    return prepared


def launch(binding_path, runtime, authorization_path, output_root, control_root):
    """One preclaimed launcher path; only the CLI watchdog may instantiate production runtime."""
    binding = validate_binding(binding_path); protocol = _runtime_protocol(binding)
    authorization = load_json(authorization_path)
    require(authorization.get("status") == "VALID_UNCONSUMED", "AUTHORIZATION_INVALID")
    require(authorization.get("binding_sha256") == sha_path(binding_path), "AUTHORIZATION_BINDING_MISMATCH")
    output_root, control_root = Path(output_root), Path(control_root)
    require(not output_root.exists(), "FRESH_OUTPUT_ROOT_REQUIRED")
    require(not (control_root / "TERMINAL_RECEIPT.json").exists(), "DUPLICATE_TERMINAL_RECEIPT")
    _clean_tracked_source(); runtime.check_dependencies(); runtime.verify_identities(binding["identities"])
    exclusive_claim(Path(authorization_path).with_suffix(Path(authorization_path).suffix + ".claimed"),
                    {"binding_sha256": sha_path(binding_path), "output_root": str(output_root), "claimed_at": time.time()})
    try:
        cohort = load_json(_bound_path(binding["cohort"]["path"]))
        records = cohort.get("records", cohort)
        require(isinstance(records, list) and len(records) == 72, "COHORT_CARDINALITY")
        prepared = _verify_worker_only_sidecar(binding, records, runtime)
        result = L.run(prepared, runtime, {"minimum_repeats_per_condition": 2,
                                           "within_condition_tolerance": protocol["within_condition_tolerance"],
                                           "cross_batch_tolerance": protocol["cross_batch_tolerance"]})
        require(len(result["records"]) == 120, "GLOBAL_COMPLETENESS")
        for row in result["records"]:
            require("target_token_ids" not in row and "prompt_token_ids" not in row, "RAW_TARGET_PERSISTENCE_FORBIDDEN")
        output_root.mkdir(parents=True, exist_ok=False)
        write_jsonl(output_root / "LOCALIZATION_METRICS.jsonl", result["records"])
        terminal = {"status": "COMPLETE", "classification": result["scored"]["classification"], "records": 120,
                    "metrics_sha256": sha_path(output_root / "LOCALIZATION_METRICS.jsonl"), "generation_calls": 0,
                    "optimizer_steps": 0, "backward_calls": 0, "target_token_ids_persisted": False,
                    "sidecar_access": "TEACHER_FORCED_WORKER_ONLY_EXACT_12"}
        write_json(control_root / "TERMINAL_RECEIPT.json", terminal); return terminal
    except Exception as exc:
        terminal = {"status": "FAILED_NO_UPDATE", "exception_type": type(exc).__name__, "exception": str(exc),
                    "generation_calls": 0, "optimizer_steps": 0, "backward_calls": 0, "target_token_ids_persisted": False}
        write_json(control_root / "TERMINAL_RECEIPT.json", terminal); raise


class ProductionRuntime(V5.RealRuntime):
    """Exact V5 loader with a bounded packed-batch teacher-forced adapter."""
    def __init__(self, config): super().__init__(config)
    def verify_identities(self, identities):
        self.verify_tokenizer(identities["tokenizer"])
        self.verify_adapter(identities["checkpoint"]["adapter_model_path"], identities["checkpoint"]["adapter_model_sha256"])
        require(V5.tree_sha256(identities["runtime"]["native_config_dir"]) == identities["runtime"]["native_config_tree_sha256"], "NATIVE_CONFIG_IDENTITY_MISMATCH")
    def prompt_token_ids(self, record): return self._prompt_ids(record)
    def open_worker_only_sidecar(self, descriptor):
        require(descriptor.get("access") == "TEACHER_FORCED_WORKER_ONLY", "SIDE_CAR_ACCESS_CONTRACT")
        path = descriptor.get("remote_path"); require(isinstance(path, str) and path.startswith("/"), "REMOTE_SIDECAR_PATH_REQUIRED")
        require(sha_path(path) == descriptor["sha256"], "SIDECAR_IDENTITY_MISMATCH")
        return load_json(path).get("records", [])
    def teacher_forced(self, packed, *, requested_batch_size, repeat_identity, physical_batch_ordinal):
        self._load(); import torch
        rows = packed["records"]; require(len(rows) <= requested_batch_size and len(rows) == packed["effective_batch_size"], "PHYSICAL_BATCH_CONTRACT")
        width = packed["effective_shape"][1]
        ids = torch.full((len(rows), width), int(self.tokenizer.pad_token_id), dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long); positions = torch.zeros((len(rows), width), dtype=torch.long)
        for index, row in enumerate(rows):
            sequence = row["prompt_token_ids"] + row["target_token_ids"]; offset = width - len(sequence)
            ids[index, offset:] = torch.tensor(sequence, dtype=torch.long); mask[index, offset:] = 1
            positions[index, offset:] = torch.arange(len(sequence), dtype=torch.long)
        with torch.inference_mode():
            logits = self.model(input_ids=ids.to("cuda:0"), attention_mask=mask.to("cuda:0"), position_ids=positions.to("cuda:0"), use_cache=False).logits
        output = {}
        for index, row in enumerate(rows):
            start = row["correct_token_indexes"][0]; count = len(row["target_token_ids"])
            selected = logits[index, start:start + count].float().detach().cpu(); require(int(selected.shape[0]) == count, "TEACHER_CAUSAL_ALIGNMENT")
            output[row["cohort_record_id"]] = [[float(value) for value in vector] for vector in selected.tolist()]
        return output


def production_child_command(args):
    return [sys.executable, str(Path(__file__).resolve()), "--binding", str(args.binding), "--authorization", str(args.authorization),
            "--output-root", str(args.output_root), "--control-root", str(args.control_root), "--execute-claimed"]


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--binding", type=Path, required=True); parser.add_argument("--authorization", type=Path)
    parser.add_argument("--output-root", type=Path); parser.add_argument("--control-root", type=Path); parser.add_argument("--launch", action="store_true"); parser.add_argument("--execute-claimed", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not args.authorization or not args.output_root or not args.control_root: raise SystemExit("REFUSE_UNAUTHORIZED_EXECUTION")
    binding = validate_binding(args.binding)
    if args.execute_claimed:
        require(os.environ.get("ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD") == "1", "DIRECT_CHILD_EXECUTION_REFUSED")
        print(json.dumps(launch(args.binding, ProductionRuntime(binding["identities"]["runtime"]), args.authorization, args.output_root, args.control_root), sort_keys=True)); return
    if not args.launch: raise SystemExit("REFUSE_UNAUTHORIZED_EXECUTION")
    require(not args.output_root.exists(), "FRESH_OUTPUT_ROOT_REQUIRED")
    write_json(args.control_root / "WATCHDOG_INVOCATION.json", {"status":"WATCHDOG_PARENT_STARTED","binding_sha256":sha_path(args.binding),"hard_cap_seconds":3600,"child_mode":"--execute-claimed"})
    environment=os.environ.copy(); environment["ARC2_V7_BATCH_LOCALIZATION_WATCHDOG_CHILD"]="1"
    code=V5.enforce_hard_cap(production_child_command(args), args.control_root / "TERMINAL_RECEIPT.json", seconds=3600, environment=environment)
    if code != 0: raise SystemExit(1)


if __name__ == "__main__": main()
