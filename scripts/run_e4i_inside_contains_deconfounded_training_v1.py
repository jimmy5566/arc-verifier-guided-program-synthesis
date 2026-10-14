"""E4I matched-control training and target-blind fresh-OOD evaluation worker."""
from __future__ import annotations

import argparse
import copy
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
PKG = ROOT / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/deconfounded_count_curriculum_causal_pilot_v1/e4i_inside_contains_deconfounded_order_balanced_training_v1"
V7 = ROOT / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
ID = "E4I_MATCHED_DECONFOUNDED_ORDER_BALANCED_INSIDE_CONTAINS_TRAINING_V1"


def canon(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def digest(value):
    return hashlib.sha256(canon(value)).hexdigest()


def sha(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            result.update(block)
    return result.hexdigest()


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canon(value) + b"\n")
    os.replace(temporary, path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def fail(reason):
    raise RuntimeError("E4I:" + reason)


def git_head():
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def clean_source():
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, text=True, capture_output=True).stdout:
        fail("SOURCE_DIRTY")


def inputs():
    return {
        "control_schedule": PKG / "CONTROL_TRAINING_SCHEDULE.json",
        "treatment_schedule": PKG / "TREATMENT_TRAINING_SCHEDULE.json",
        "train_fit": PKG / "TRAIN_FIT_INPUTS_AND_COMMITMENTS.json",
        "ood": PKG / "FRESH_OOD_INPUTS_AND_COMMITMENTS.json",
        "protection": PKG / "PROTECTION_PANEL_REFERENCES.json",
        "protocol": PKG / "PROTOCOL.json",
        "freeze": PKG / "CPU_FREEZE_RECEIPT.json",
        "v7_manifest": V7,
    }


def schedule_task(row):
    task = copy.deepcopy(row["task"])
    nested = task.get("source_id")
    enclosing = row.get("source_id")
    if nested is not None and enclosing is not None and nested != enclosing:
        fail("SOURCE_ID_CONFLICT")
    resolved = nested if nested is not None else enclosing
    if not isinstance(resolved, str) or not resolved:
        fail("SOURCE_ID_MISSING")
    task["source_id"] = resolved
    return task


def validate_schedules():
    from training_data.pipeline import task_to_sample

    schedules = {arm: load(PKG / f"{arm}_TRAINING_SCHEDULE.json")["episodes"] for arm in ("CONTROL", "TREATMENT")}
    for arm, rows in schedules.items():
        if len(rows) != 384 or [row["optimizer_step"] for row in rows] != [index // 4 for index in range(384)]:
            fail(arm + "_SCHEDULE")
        roles = {name: sum(row["role"] == name for row in rows) for name in ("PRIMARY_CONNECTED_COMPONENT_COUNT", "PRIMARY_OBJECT_EXTRACTION", "PROTECTED_REPLAY")}
        if roles != {"PRIMARY_CONNECTED_COMPONENT_COUNT": 48, "PRIMARY_OBJECT_EXTRACTION": 48, "PROTECTED_REPLAY": 288}:
            fail(arm + "_ROLES")
    matched = 0
    for control, treatment in zip(schedules["CONTROL"], schedules["TREATMENT"], strict=True):
        left = task_to_sample(schedule_task(control))
        right = task_to_sample(schedule_task(treatment))
        if (left["sequence_length"], left["assistant_token_count"]) != (right["sequence_length"], right["assistant_token_count"]):
            fail("TOKEN_BUDGET")
        if control["role"] != "PRIMARY_CONNECTED_COMPONENT_COUNT" and digest(control) != digest(treatment):
            fail("PROTECTED_REPLAY_CHANGED")
        matched += 1
    return schedules, matched


def resolve(binding_path, verify_assets=True, require_clean=True, check_output=True):
    binding = load(binding_path)
    if require_clean:
        clean_source()
    if binding.get("experiment_id") != ID or binding.get("retry") is not False or binding.get("hard_cap_seconds") != 14400:
        fail("BINDING_CONTRACT")
    if binding.get("execution_source_commit") != git_head():
        fail("SOURCE_COMMIT")
    for name, path in inputs().items():
        item = binding.get("inputs", {}).get(name, {})
        if item.get("path") != str(path.relative_to(ROOT)).replace("\\", "/") or item.get("sha256") != sha(path):
            fail("INPUT_" + name.upper())
    worker = ROOT / binding.get("worker_path", "")
    if sha(worker) != binding.get("worker_sha256"):
        fail("WORKER_HASH")
    if check_output and Path(binding["output_root"]).exists():
        fail("OUTPUT_EXISTS")
    if Path(binding["duplicate_guard_path"]).exists():
        fail("DUPLICATE_ACTIVE_PROCESS")
    schedules, matched = validate_schedules()
    for module in ("training_data.pipeline", "scripts.e04_e_equal_slot_loss", "inference.nvarc_native", "peft", "transformers", "bitsandbytes"):
        importlib.import_module(module)
    manifest = load(V7)
    if binding["v7"]["manifest_sha256"] != sha(V7) or binding["v7"]["adapter_model_sha256"] != "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3":
        fail("V7_IDENTITY")
    if verify_assets:
        for root, files in ((manifest["adapter_path"], manifest["adapter_files"]), (manifest["base_path"], manifest["base_files"])):
            for item in files:
                path = Path(root) / item["name"]
                if not path.is_file() or path.stat().st_size != item["bytes"] or sha(path) != item["sha256"]:
                    fail("V7_ASSET_" + item["name"])
    if not Path(binding["runtime_python"]).is_file():
        fail("RUNTIME_PYTHON")
    runtime = binding["runtime"]
    if runtime["requested_batch_size"] != 32 or runtime["fallback_ladder"] != [32, 16, 8, 4, 1] or runtime["decoder"] != "NVARC_NATIVE_GREEDY":
        fail("RUNTIME_CONTRACT")
    return binding, manifest, schedules, matched


def load_model(trainable=False):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        fail("CUDA_BF16")
    manifest = load(V7)
    base = AutoModelForCausalLM.from_pretrained(
        manifest["base_path"], local_files_only=True, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True
    ).to("cuda:0")
    model = PeftModel.from_pretrained(base, manifest["adapter_path"], is_trainable=trainable)
    if trainable:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()
    else:
        model.eval()
    return model, torch


def train_arm(arm, rows, binding, output, deadline):
    import bitsandbytes as bnb
    import torch
    from scripts.e04_e_equal_slot_loss import backward_equal_slot_term
    from training_data.pipeline import task_to_sample

    model, _ = load_model(True)
    torch.manual_seed(binding["seed"])
    optimizer = bnb.optim.PagedAdamW8bit(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=binding["training"]["learning_rate"],
    )
    accounting = []
    for offset in range(0, 384, 4):
        if time.monotonic() > deadline:
            fail("TRAIN_CAP")
        optimizer.zero_grad(set_to_none=True)
        for row in rows[offset : offset + 4]:
            sample = task_to_sample(schedule_task(row))
            ids = torch.tensor([sample["input_ids"]], device="cuda:0")
            labels = torch.tensor([sample["labels"]], device="cuda:0")
            result = model(input_ids=ids, attention_mask=torch.ones_like(ids), use_cache=False)
            _, selected = backward_equal_slot_term(result.logits, labels, expected_slots=4)
            accounting.append(
                {
                    "slot": row["slot"],
                    "role": row["role"],
                    "selected_tokens": selected,
                    "sequence_length": sample["sequence_length"],
                    "assistant_token_count": sample["assistant_token_count"],
                    "slot_coefficient": 0.25,
                }
            )
            del result
        optimizer.step()
    if len(accounting) != 384:
        fail("TRAIN_ACCOUNTING")
    adapter = Path(output) / f"{arm}_ADAPTER"
    model.save_pretrained(adapter)
    receipt = {
        "status": "COMPLETE_E4I_MATCHED_TRAINING",
        "arm": arm,
        "optimizer": "PagedAdamW8bit",
        "optimizer_steps": 96,
        "slot_count": 384,
        "slot_coefficient": 0.25,
        "adapter_sha256": sha(adapter / "adapter_model.safetensors"),
        "accounting": accounting,
    }
    write(Path(output) / f"{arm}_TRAINING_RECEIPT.json", receipt)
    model.eval()
    return model, torch, receipt


def resolve_reference(refs, field):
    raw = refs[field].replace("\\", "/")
    candidate = (ROOT / raw).resolve()
    if ROOT.resolve() not in candidate.parents or not candidate.is_file():
        fail("PROTECTION_REFERENCE")
    expected = refs[field.replace("_path", "_sha256")]
    if sha(candidate) != expected:
        fail("PROTECTION_HASH")
    return candidate


def task_records():
    records = []
    for split, filename in (("TRAIN_FIT", "TRAIN_FIT_INPUTS_AND_COMMITMENTS.json"), ("OOD", "FRESH_OOD_INPUTS_AND_COMMITMENTS.json")):
        for row in load(PKG / filename)["records"]:
            if "output" in row["input_task"]["test"][0]:
                fail("TARGET_LEAK")
            records.append({**row, "split": split, "semantic_family": "E4I"})
    refs = load(PKG / "PROTECTION_PANEL_REFERENCES.json")
    for row in load(resolve_reference(refs, "retention_path")):
        records.append(
            {
                "record_id": "RETENTION:" + row["prompt_id"],
                "canonical_unit_id": "RETENTION:" + row["prompt_id"],
                "demo_order": "SINGLE",
                "input_task": copy.deepcopy(row["task"]),
                "split": "RETENTION",
                "semantic_family": row["latent_condition"]["family"],
            }
        )
    for row in load(resolve_reference(refs, "composition_path")):
        records.append(
            {
                "record_id": "COMPOSITION:" + row["prompt_id"],
                "canonical_unit_id": "COMPOSITION:" + row["prompt_id"],
                "demo_order": "SINGLE",
                "input_task": copy.deepcopy(row["task"]),
                "split": "COMPOSITION",
                "semantic_family": row["latent_condition"]["family"],
            }
        )
    if any("output" in row["input_task"]["test"][0] for row in records):
        fail("TARGET_LEAK")
    return records


def build_targets_after_raw(records):
    from scripts.connected_components_object_extraction_semantic_oracle import target_for_family
    from scripts.e4i_inside_contains_deconfounded_training import build_schedules, ood_records, train_fit_records

    _, _ = build_schedules()
    _, train_targets = train_fit_records()
    _, ood_targets = ood_records()
    targets = {**train_targets, **ood_targets}
    for row in records:
        if row["split"] in ("RETENTION", "COMPOSITION"):
            targets[row["record_id"]] = target_for_family(row["semantic_family"], row["input_task"]["test"][0]["input"])
    commitment = load(PKG / "CPU_FREEZE_RECEIPT.json")["target_map_sha256_commitment"]
    scientific = {key: value for key, value in targets.items() if key.startswith("TRAIN_FIT:") or key.startswith("OOD:")}
    if digest(scientific) != commitment:
        fail("TARGET_COMMITMENT")
    return targets


def paired_ci(deltas, seed=20261014):
    rng = random.Random(seed)
    draws = sorted(sum(deltas[rng.randrange(len(deltas))] for _ in deltas) for _ in range(10000))
    return [draws[250], draws[9749]]


def score(records, raw_paths, targets, output):
    arms = {arm: {row["record_id"]: row for row in read_jsonl(path)} for arm, path in raw_paths.items()}
    expected = {row["record_id"] for row in records}
    if set(arms) != {"CONTROL", "TREATMENT"} or any(set(rows) != expected for rows in arms.values()):
        fail("RAW_COMPLETENESS")

    def good(arm, record_id):
        return arms[arm][record_id]["parsed_grid"] == targets[record_id]

    def paired_split(split):
        units = sorted({row["canonical_unit_id"] for row in records if row["split"] == split})
        result = {}
        for arm in arms:
            both = []
            any_order = []
            for unit in units:
                values = [good(arm, row["record_id"]) for row in records if row["split"] == split and row["canonical_unit_id"] == unit]
                both.append(all(values))
                any_order.append(any(values))
            result[arm] = {"both": both, "any": any_order}
        return units, result

    _, train = paired_split("TRAIN_FIT")
    units, ood = paired_split("OOD")
    deltas = [int(treatment) - int(control) for control, treatment in zip(ood["CONTROL"]["both"], ood["TREATMENT"]["both"], strict=True)]
    retention = {}
    for family in sorted({row["semantic_family"] for row in records if row["split"] == "RETENTION"}):
        retention[family] = sum(
            int(good("TREATMENT", row["record_id"])) - int(good("CONTROL", row["record_id"]))
            for row in records
            if row["split"] == "RETENTION" and row["semantic_family"] == family
        )
    composition = sum(
        int(good("TREATMENT", row["record_id"])) - int(good("CONTROL", row["record_id"]))
        for row in records
        if row["split"] == "COMPOSITION"
    )
    ci = paired_ci(deltas)
    control_any = sum(ood["CONTROL"]["any"])
    treatment_any = sum(ood["TREATMENT"]["any"])
    metrics = {
        "train_fit_both_orders": {arm: sum(value["both"]) for arm, value in train.items()},
        "ood_both_orders": {arm: sum(value["both"]) for arm, value in ood.items()},
        "ood_any_order": {arm: sum(value["any"]) for arm, value in ood.items()},
        "ood_delta": sum(deltas),
        "paired_bootstrap_95_ci": ci,
        "order_invariance": {
            "CONTROL": sum(ood["CONTROL"]["both"]) / control_any if control_any else 0.0,
            "TREATMENT": sum(ood["TREATMENT"]["both"]) / treatment_any if treatment_any else 0.0,
        },
        "retention_family_delta": retention,
        "retention_aggregate_delta": sum(retention.values()),
        "composition_delta": composition,
        "canonical_units": len(units),
    }
    reasons = []
    if metrics["ood_delta"] < 6:
        reasons.append("OOD_DELTA")
    if ci[0] <= 0:
        reasons.append("OOD_UNCERTAINTY")
    if metrics["order_invariance"]["TREATMENT"] < 0.8:
        reasons.append("ORDER_INVARIANCE")
    if metrics["retention_aggregate_delta"] < 0 or any(value < 0 for value in retention.values()):
        reasons.append("RETENTION")
    if composition < 0:
        reasons.append("COMPOSITION")
    receipt = {
        "status": "COMPLETE_VALID_E4I_MATCHED_CAUSAL_RESULT",
        "metrics": metrics,
        "decision": "PASS_SCIENTIFIC_RESULT_NOT_PROMOTED_PENDING_CONFIRMATION" if not reasons else "REJECT",
        "failure_reasons": reasons,
        "raw_sha256": {arm: sha(path) for arm, path in raw_paths.items()},
        "targets_opened_after_raw_freeze": True,
        "promotion": "FORBIDDEN_PENDING_INDEPENDENT_CONFIRMATION",
    }
    write(Path(output) / "RESULT_RECEIPT.json", receipt)
    return receipt


def execute(binding_path):
    binding, manifest, schedules, matched = resolve(binding_path)
    output = Path(binding["output_root"])
    output.mkdir(parents=True)
    guard = Path(binding["duplicate_guard_path"])
    guard.parent.mkdir(parents=True, exist_ok=True)
    try:
        with guard.open("x", encoding="utf-8") as handle:
            json.dump({"experiment_id": ID, "nonce": binding["nonce"], "pid": os.getpid()}, handle, sort_keys=True)
    except FileExistsError:
        fail("DUPLICATE_ACTIVE_PROCESS")
    deadline = time.monotonic() + binding["hard_cap_seconds"]
    try:
        write(
            output / "PREFLIGHT_RECEIPT.json",
            {
                "status": "PASS_E4I_PRODUCTION_PREFLIGHT_NO_MODEL",
                "source_commit": git_head(),
                "binding_sha256": sha(binding_path),
                "matched_slot_token_budgets": matched,
                "model_loaded": False,
                "gpu_used": False,
                "targets_opened": False,
            },
        )
        write(output / "SCIENTIFIC_BOUNDARY_RECEIPT.json", {"status": "FIRST_MODEL_LOAD_NEXT", "nonce": binding["nonce"]})
        records = task_records()
        from scripts.run_e4h_counterfactual_ood_v1 import generate_arm

        raw_paths = {}
        generation = {}
        training = {}
        for arm in ("CONTROL", "TREATMENT"):
            model, torch, training[arm] = train_arm(arm, schedules[arm], binding, output, deadline)
            raw_paths[arm], generation[arm] = generate_arm(model, torch, manifest, records, binding, arm, output, deadline)
            del model
            torch.cuda.empty_cache()
        raw_hashes = {arm: sha(path) for arm, path in raw_paths.items()}
        write(
            output / "RAW_FREEZE_RECEIPT.json",
            {
                "status": "COMPLETE_RAW_FROZEN_NO_TARGETS",
                "complete_arms": ["CONTROL", "TREATMENT"],
                "raw_sha256": raw_hashes,
                "generation": generation,
                "targets_accessed": False,
                "nonce": binding["nonce"],
            },
        )
        targets = build_targets_after_raw(records)
        sidecar = output / "SYNTHETIC_TARGET_SIDECAR.json"
        write(sidecar, {"status": "DERIVED_AFTER_COMPLETE_RAW_FREEZE", "targets": targets})
        result = score(records, raw_paths, targets, output)
        result["target_sidecar_sha256"] = sha(sidecar)
        write(output / "RESULT_RECEIPT.json", result)
        write(
            output / "TERMINAL_RECEIPT.json",
            {"status": "COMPLETE", "raw_freeze_before_sidecar": True, "result_sha256": sha(output / "RESULT_RECEIPT.json")},
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


def cpu_self_test(binding_path):
    binding, _, schedules, matched = resolve(binding_path, verify_assets=False, require_clean=False, check_output=False)
    records = task_records()
    if len(records) < 192 or matched != 384:
        fail("CPU_FIXTURE")
    return {
        "status": "PASS_E4I_PRODUCTION_SHAPED_CPU_NO_MODEL",
        "records": len(records),
        "schedule_slots_per_arm": {arm: len(rows) for arm, rows in schedules.items()},
        "matched_slot_token_budgets": matched,
        "model_loaded": False,
        "gpu_used": False,
        "targets_opened": False,
        "output_root": binding["output_root"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binding", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--cpu-self-test", action="store_true")
    arguments = parser.parse_args()
    if arguments.execute == arguments.cpu_self_test:
        parser.error("choose exactly one mode")
    if arguments.execute:
        execute(arguments.binding)
    else:
        print(json.dumps(cpu_self_test(arguments.binding), sort_keys=True))
