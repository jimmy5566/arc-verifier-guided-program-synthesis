"""Target-blind V7 versus E4G evaluation on the frozen E4H panel."""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(os.environ.get("ARC2_ROOT", Path(__file__).resolve().parents[1]))
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
PKG = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4h_inside_contains_counterfactual_ood_v1"
PANEL = PKG / "COUNTERFACTUAL_OOD_INPUTS_AND_COMMITMENTS.json"
PROTOCOL = PKG / "PROTOCOL.json"
V7_MANIFEST = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
EXPERIMENT_ID = "E4H_INSIDE_CONTAINS_COUNTERFACTUAL_OOD_V1"


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canon(value)).hexdigest()


def sha(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(canon(value) + b"\n")
    os.replace(tmp, path)


def write_jsonl(path, rows):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(b"".join(canon(row) + b"\n" for row in rows))
    os.replace(tmp, path)


def fail(reason):
    raise RuntimeError("E4H:" + reason)


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def clean_source():
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout:
        fail("SOURCE_DIRTY")


def resolve(binding_path, verify_assets=True, require_clean=True):
    binding = load(binding_path)
    if require_clean:
        clean_source()
    if binding.get("experiment_id") != EXPERIMENT_ID or binding.get("retry") is not False:
        fail("BINDING_IDENTITY")
    if binding.get("execution_source_commit") != git_head():
        fail("SOURCE_COMMIT")
    expected_inputs = {"panel": PANEL, "protocol": PROTOCOL, "v7_manifest": V7_MANIFEST}
    for name, path in expected_inputs.items():
        item = binding.get("inputs", {}).get(name, {})
        if item.get("path") != str(path.relative_to(ROOT)).replace("\\", "/") or item.get("sha256") != sha(path):
            fail("INPUT_" + name.upper())
    worker = ROOT / binding.get("worker_path", "")
    if sha(worker) != binding.get("worker_sha256"):
        fail("WORKER_HASH")
    protocol = load(PROTOCOL)
    runtime = binding.get("runtime", {})
    if runtime.get("requested_batch_size") != protocol["requested_batch_size"]:
        fail("REQUESTED_BATCH")
    if runtime.get("fallback_ladder") != protocol["frozen_fallback_ladder"]:
        fail("FALLBACK_LADDER")
    if runtime.get("decoder") != "NVARC_NATIVE_GREEDY" or runtime.get("context_window") != 16384:
        fail("RUNTIME_CONTRACT")
    if binding.get("hard_cap_seconds") != 3600:
        fail("HARD_CAP")
    output = Path(binding["output_root"])
    if output.exists():
        fail("OUTPUT_EXISTS")
    guard = Path(binding["duplicate_guard_path"])
    if guard.exists():
        fail("DUPLICATE_ACTIVE_PROCESS")
    for module in ("inference.nvarc_native", "peft", "transformers"):
        importlib.import_module(module)
    manifest = load(V7_MANIFEST)
    if binding["v7"]["manifest_sha256"] != sha(V7_MANIFEST):
        fail("V7_MANIFEST")
    if verify_assets:
        for root, files in ((manifest["adapter_path"], manifest["adapter_files"]), (manifest["base_path"], manifest["base_files"])):
            for item in files:
                path = Path(root) / item["name"]
                if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                    fail("V7_ASSET_" + item["name"])
        e4g = binding["e4g_adapter"]
        adapter_path = Path(e4g["path"]) / "adapter_model.safetensors"
        if not adapter_path.is_file() or adapter_path.stat().st_size != e4g["bytes"] or sha(adapter_path) != e4g["sha256"]:
            fail("E4G_ADAPTER")
        config = Path(e4g["path"]) / "adapter_config.json"
        if not config.is_file():
            fail("E4G_ADAPTER_CONFIG")
    if not Path(binding["runtime_python"]).is_file():
        fail("RUNTIME_PYTHON")
    panel = load(PANEL)
    if panel.get("status") != "FROZEN_TARGET_BLIND_CPU_VALIDATED" or len(panel.get("records", [])) != 96:
        fail("PANEL")
    if any("output" in row["input_task"]["test"][0] for row in panel["records"]):
        fail("TARGET_LEAK")
    return binding, manifest, panel


def messages(task):
    from inference.nvarc_native import serialize_grid

    result = []
    for pair in task["train"]:
        result.extend(
            [
                {"role": "user", "content": serialize_grid(pair["input"])},
                {"role": "assistant", "content": serialize_grid(pair["output"])},
            ]
        )
    result.append({"role": "user", "content": serialize_grid(task["test"][0]["input"])})
    return result


def load_checkpoint(manifest, adapter_path):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        fail("CUDA_BF16")
    base = AutoModelForCausalLM.from_pretrained(
        manifest["base_path"],
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to("cuda:0")
    model = PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
    model.eval()
    return model, torch


def native_provider(model, manifest):
    from inference.nvarc_native import NVARCNativeProvider, checkpoint_native_tokenizer

    identity = load(ROOT / "experiments/capability_repair_baseline_v1/e04_orientation_fixed_demonstration_baseline_v3/E04_V3_RUNTIME_NATIVE_CONFIG_IDENTITY_V1.json")
    config_dir = ROOT / "configs" / identity["runtime_config_dir_name"]
    tokenizer, _ = checkpoint_native_tokenizer(Path(manifest["base_path"]), config_dir)
    provider = NVARCNativeProvider(model_path=Path(manifest["base_path"]), tokenizer_config_dir=config_dir, device="cuda:0")
    provider.model = model
    provider.tokenizer = tokenizer
    return provider


def _generate_at_batch(provider, records, batch_size, binding, deadline):
    from inference.nvarc_native import parse_native_grid

    rows = []
    ordered = sorted(records, key=lambda row: (len(canon(row["input_task"])), row["record_id"]))
    for offset in range(0, len(ordered), batch_size):
        if time.monotonic() > deadline:
            fail("TIME_CAP")
        chunk = ordered[offset : offset + batch_size]
        outputs = provider.generate_many(
            [messages(row["input_task"]) for row in chunk],
            max_new_tokens=binding["runtime"]["max_new_tokens"],
            context_window=binding["runtime"]["context_window"],
            seeds=[binding["seed"] + offset + index for index in range(len(chunk))],
        )
        if len(outputs) != len(chunk):
            fail("GENERATION_COMPLETENESS")
        for row, output in zip(chunk, outputs, strict=True):
            parsed = parse_native_grid(output.text)
            rows.append(
                {
                    "record_id": row["record_id"],
                    "canonical_unit_id": row["canonical_unit_id"],
                    "demo_order": row["demo_order"],
                    "parsed_grid": None if parsed is None else [list(line) for line in parsed],
                    "parser_valid": parsed is not None,
                    "generated_token_ids": [int(token) for token in output.token_ids],
                    "prompt_sha256": digest(row["input_task"]),
                    "effective_batch_size": batch_size,
                }
            )
    return sorted(rows, key=lambda row: row["record_id"])


def generate_arm(model, torch, manifest, records, binding, arm, output_root, deadline):
    provider = native_provider(model, manifest)
    fallback_reason = None
    selected_batch = None
    generated = None
    for batch_size in binding["runtime"]["fallback_ladder"]:
        try:
            generated = _generate_at_batch(provider, records, batch_size, binding, deadline)
            selected_batch = batch_size
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            fallback_reason = "CUDA_OOM_AT_BATCH_" + str(batch_size)
            torch.cuda.empty_cache()
    if generated is None or selected_batch is None:
        fail("FALLBACK_EXHAUSTED")
    subset = set(binding["runtime"]["b1_validation_subset"])
    subset_records = [row for row in records if row["canonical_unit_id"] in subset]
    b1 = _generate_at_batch(provider, subset_records, 1, binding, deadline)
    main = {row["record_id"]: row for row in generated}
    mismatches = [
        row["record_id"]
        for row in b1
        if row["parser_valid"] != main[row["record_id"]]["parser_valid"]
        or row["parsed_grid"] != main[row["record_id"]]["parsed_grid"]
    ]
    token_identity = sum(row["generated_token_ids"] == main[row["record_id"]]["generated_token_ids"] for row in b1)
    validation = {
        "records": len(b1),
        "parser_and_exact_grid_mismatches": mismatches,
        "generated_token_identity_records": token_identity,
        "status": "PASS" if not mismatches else "FAIL_MATERIAL_BATCH_DRIFT",
    }
    if mismatches:
        fail("B1_VALIDATION")
    raw_path = Path(output_root) / f"{arm}_RAW.jsonl"
    write_jsonl(raw_path, generated)
    receipt = {
        "status": "COMPLETE_TARGET_BLIND_NATIVE_GREEDY",
        "arm": arm,
        "rows": len(generated),
        "requested_batch_size": binding["runtime"]["requested_batch_size"],
        "effective_batch_size": selected_batch,
        "fallback_reason": fallback_reason,
        "b1_validation": validation,
        "raw_sha256": sha(raw_path),
        "targets_accessed": False,
    }
    write(Path(output_root) / f"{arm}_GENERATION_RECEIPT.json", receipt)
    return raw_path, receipt


def paired_bootstrap(deltas, repetitions=10000, seed=20261014):
    rng = random.Random(seed)
    draws = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(repetitions))
    return [draws[250], draws[9749]]


def score(records, raw_paths, targets):
    arms = {}
    for arm, path in raw_paths.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        arms[arm] = {row["record_id"]: row for row in rows}
    expected = {row["record_id"] for row in records}
    if any(set(rows) != expected for rows in arms.values()) or set(targets) != expected:
        fail("SCORE_COMPLETENESS")

    def correct(arm, record_id):
        return arms[arm][record_id]["parser_valid"] and arms[arm][record_id]["parsed_grid"] == targets[record_id]

    units = sorted({row["canonical_unit_id"] for row in records})
    by_unit = {unit: [row for row in records if row["canonical_unit_id"] == unit] for unit in units}
    summary = {}
    both_vectors = {}
    for arm in ("V7", "E4G"):
        paired = {"BOTH": 0, "A_ONLY": 0, "B_ONLY": 0, "NEITHER": 0}
        order = {"A": 0, "B": 0}
        direction = {"INSIDE": 0, "CONTAINS": 0}
        numeric = {"FRAME_LT_INNER": 0, "FRAME_GT_INNER": 0}
        both = []
        for unit in units:
            rows = by_unit[unit]
            outcome = {row["demo_order"]: correct(arm, row["record_id"]) for row in rows}
            if outcome["A"] and outcome["B"]:
                category = "BOTH"
            elif outcome["A"]:
                category = "A_ONLY"
            elif outcome["B"]:
                category = "B_ONLY"
            else:
                category = "NEITHER"
            paired[category] += 1
            both.append(category == "BOTH")
            for row in rows:
                ok = correct(arm, row["record_id"])
                order[row["demo_order"]] += int(ok)
                direction[row["construct"]["direction"]] += int(ok)
                numeric[row["construct"]["numeric_order"]] += int(ok)
        any_count = 48 - paired["NEITHER"]
        summary[arm] = {
            "order_correct": order,
            "paired_canonical_units": paired,
            "both_orders_exact": paired["BOTH"],
            "any_order_exact": any_count,
            "order_invariance_both_over_any": paired["BOTH"] / any_count if any_count else 0.0,
            "direction_record_correct": direction,
            "numeric_order_record_correct": numeric,
            "parser_valid": sum(row["parser_valid"] for row in arms[arm].values()),
        }
        both_vectors[arm] = both
    deltas = [int(treatment) - int(control) for control, treatment in zip(both_vectors["V7"], both_vectors["E4G"], strict=True)]
    delta = sum(deltas)
    ci = paired_bootstrap(deltas)
    e4g = summary["E4G"]
    numeric_deltas = {
        key: e4g["numeric_order_record_correct"][key] - summary["V7"]["numeric_order_record_correct"][key]
        for key in ("FRAME_LT_INNER", "FRAME_GT_INNER")
    }
    positive = (
        delta >= 6
        and ci[0] > 0
        and e4g["order_invariance_both_over_any"] >= 0.8
        and min(numeric_deltas.values()) >= 0
    )
    return {
        "status": "COMPLETE_VALID_COUNTERFACTUAL_OOD_DIAGNOSTIC",
        "arms": summary,
        "primary_delta": delta,
        "paired_bootstrap_95_ci": ci,
        "numeric_order_record_deltas": numeric_deltas,
        "classification": "POSITIVE_TRANSFER" if positive else "NO_CONFIRMED_TRANSFER_SHORTCUT_SENSITIVE",
        "promotion": "FORBIDDEN_DIAGNOSTIC_ONLY",
    }


def cpu_self_test(binding_path):
    binding, _, panel = resolve(binding_path, verify_assets=False, require_clean=False)
    records = panel["records"]
    # Deliberately use synthetic constant targets.  The production-shaped CPU
    # fixture validates completeness and scorer wiring without deriving or
    # opening any real E4H evaluation target before raw model predictions.
    targets = {row["record_id"]: [[0]] for row in records}
    temp = Path(binding_path).parent / "e4h_cpu_fixture"
    temp.mkdir(exist_ok=True)
    raw = {}
    for arm in ("V7", "E4G"):
        rows = [
            {
                "record_id": row["record_id"],
                "canonical_unit_id": row["canonical_unit_id"],
                "demo_order": row["demo_order"],
                "parsed_grid": targets[row["record_id"]],
                "parser_valid": True,
                "generated_token_ids": [],
                "prompt_sha256": digest(row["input_task"]),
                "effective_batch_size": 1,
            }
            for row in records
        ]
        path = temp / f"{arm}.jsonl"
        write_jsonl(path, rows)
        raw[arm] = path
    result = score(records, raw, targets)
    receipt = {
        "status": "PASS_E4H_PRODUCTION_SHAPED_CPU_FIXTURE_NO_MODEL",
        "records_per_arm": 96,
        "primary_delta": result["primary_delta"],
        "model_loaded": False,
        "gpu_used": False,
        "real_targets_opened": False,
    }
    write(Path(binding_path).parent / "E4H_CPU_INTEGRATION_RECEIPT.json", receipt)
    return receipt


def execute(binding_path):
    binding, manifest, panel = resolve(binding_path)
    output = Path(binding["output_root"])
    output.mkdir(parents=True)
    guard = Path(binding["duplicate_guard_path"])
    guard.parent.mkdir(parents=True, exist_ok=True)
    try:
        with guard.open("x", encoding="utf-8") as handle:
            json.dump({"experiment_id": EXPERIMENT_ID, "nonce": binding["nonce"], "pid": os.getpid()}, handle, sort_keys=True)
    except FileExistsError:
        fail("DUPLICATE_ACTIVE_PROCESS")
    deadline = time.monotonic() + binding["hard_cap_seconds"]
    try:
        write(
            output / "PREFLIGHT_RECEIPT.json",
            {
                "status": "PASS_E4H_PRODUCTION_PREFLIGHT_NO_MODEL",
                "source_commit": git_head(),
                "binding_sha256": sha(binding_path),
                "panel_sha256": sha(PANEL),
                "model_loaded": False,
                "gpu_used": False,
                "targets_opened": False,
            },
        )
        write(output / "SCIENTIFIC_BOUNDARY_RECEIPT.json", {"status": "FIRST_MODEL_LOAD_NEXT", "nonce": binding["nonce"]})
        records = panel["records"]
        raw_paths = {}
        generation = {}
        v7, torch = load_checkpoint(manifest, manifest["adapter_path"])
        raw_paths["V7"], generation["V7"] = generate_arm(v7, torch, manifest, records, binding, "V7", output, deadline)
        del v7
        torch.cuda.empty_cache()
        e4g, torch = load_checkpoint(manifest, binding["e4g_adapter"]["path"])
        raw_paths["E4G"], generation["E4G"] = generate_arm(e4g, torch, manifest, records, binding, "E4G", output, deadline)
        del e4g
        torch.cuda.empty_cache()
        raw_hashes = {arm: sha(path) for arm, path in raw_paths.items()}
        write(
            output / "RAW_FREEZE_RECEIPT.json",
            {
                "status": "COMPLETE_RAW_FROZEN_NO_TARGETS",
                "raw_sha256": raw_hashes,
                "complete_arms": ["E4G", "V7"],
                "targets_accessed": False,
                "nonce": binding["nonce"],
                "generation": generation,
            },
        )
        # Target construction is intentionally below the complete raw freeze.
        from scripts.e4h_inside_contains_counterfactual_ood import build_panel

        _, targets = build_panel()
        target_commitment = load(PKG / "CPU_FREEZE_RECEIPT.json")["target_sidecar_sha256_commitment"]
        if digest(targets) != target_commitment:
            fail("TARGET_COMMITMENT")
        sidecar = output / "COUNTERFACTUAL_OOD_TARGET_SIDECAR.json"
        write(sidecar, {"status": "DERIVED_AFTER_COMPLETE_RAW_FREEZE", "targets": targets})
        result = score(records, raw_paths, targets)
        result["raw_sha256"] = raw_hashes
        result["target_sidecar_sha256"] = sha(sidecar)
        result["targets_opened_after_raw_freeze"] = True
        write(output / "RESULT_RECEIPT.json", result)
        write(
            output / "TERMINAL_RECEIPT.json",
            {
                "status": "COMPLETE",
                "raw_freeze_before_sidecar": True,
                "result_sha256": sha(output / "RESULT_RECEIPT.json"),
            },
        )
    except Exception as error:
        write(
            output / "TERMINAL_RECEIPT.json",
            {
                "status": "FAILED_OR_PARTIAL_NO_RETRY",
                "error": type(error).__name__ + ":" + str(error),
                "scientific_boundary_entered": (output / "SCIENTIFIC_BOUNDARY_RECEIPT.json").exists(),
                "raw_frozen": (output / "RAW_FREEZE_RECEIPT.json").exists(),
            },
        )
        raise
    finally:
        if guard.exists():
            guard.unlink()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--cpu-self-test", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.cpu_self_test:
        print(json.dumps(cpu_self_test(args.binding), sort_keys=True))
    elif args.execute:
        execute(args.binding)
    else:
        parser.error("choose --cpu-self-test or --execute")
