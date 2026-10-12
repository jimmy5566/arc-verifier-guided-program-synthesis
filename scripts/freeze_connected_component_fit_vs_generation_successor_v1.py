"""Freeze Director-requested successor artifacts without model loading or GPU use.

Creates a sealed correct-token sidecar for the existing 24 TRAIN_FIT CCC and
48 structural-OOD prompts, validates byte-identical RUN_002 reuse, and freezes
one corrected successor protocol.  Target grid strings never appear in any
non-sealed output or console text.
"""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, os, subprocess, tempfile
from collections import Counter
from pathlib import Path

PROTOCOL_ID = "CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1_SUCCESSOR"
OLD_PROTOCOL_ID = "CONNECTED_COMPONENT_COUNT_FIT_VS_GENERATION_DIAGNOSTIC_V1"
RUN_ID = "RUN_002_09ff2d41a20e291a5900c6f73897fed5"
CHECKPOINTS = {
    "V7": "1639bf6665ccfbb99cf547f59fcd581663c95b6bad770ae7846fee90f66be1c3",
    "CONTROL_NARROW_STRUCTURE": "4a5e621f22dc5eb3a3e3e7d2eab009559e276a2c20fd05675f2265ad23ce13e2",
    "TREATMENT_STRUCTURE_DIVERSE": "5dae97898461b6b618328154bfb4271199a57b6cec68e8e7bb90209002618978",
}
TOKENIZER = {
    "distribution": "KAGGLE",
    "repository": "sorokin/qwen3_4b_grids15_sft139",
    "revision": "Transformers/bfloat16/1",
    "runtime_path": "/workspace/arc2/models/qwen3_4b_grids15_sft139",
    "tokenizer_files": {
        "tokenizer.json": "af6b9e855f9db3192bd9152607708ebd7700b40f9fc5f792a26af70150a96cd5",
        "tokenizer_config.json": "e1912aeba4050616cbfe01b34df3f3df2818207eda0056f6eb87c2336936ea90",
        "vocab.json": "8141a61e47eceec466524bc178bc9c6d5076b7d1b18b456fbd3fdf66cae57f82",
        "special_tokens_map.json": "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b",
    },
    "grid_token_ids": {"digits": list(range(10)), "newline": 10, "eos": 15, "pad": 13},
}


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha_value(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load(path: Path, expected: str | None = None) -> object:
    if not path.is_file():
        raise RuntimeError(f"MISSING:{path}")
    actual = sha_file(path)
    if expected and actual != expected:
        raise RuntimeError(f"SHA_MISMATCH:{path}:{actual}")
    return json.loads(path.read_text(encoding="utf-8"))


def write(path: Path, value: object, sealed: bool = False) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        handle.write(payload)
        temporary = Path(handle.name)
    os.replace(temporary, path)
    if sealed:
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass
    return hashlib.sha256(payload).hexdigest()


def module_from(path: Path):
    spec = importlib.util.spec_from_file_location("arc2_connected_components_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("ORACLE_IMPORT_SPEC_FAILURE")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def serialize_grid(grid: list[list[int]]) -> str:
    if not grid or any(not row for row in grid):
        raise RuntimeError("EMPTY_TARGET_GRID")
    if any(any(type(cell) is not int or not 0 <= cell <= 9 for cell in row) for row in grid):
        raise RuntimeError("INVALID_TARGET_GRID_CELL")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise RuntimeError("NON_RECTANGULAR_TARGET_GRID")
    return "\n".join("".join(str(cell) for cell in row) for row in grid)


def encode_target(text: str) -> list[int]:
    mapping = {str(i): i for i in range(10)} | {"\n": 10}
    try:
        return [mapping[ch] for ch in text] + [15]
    except KeyError as error:
        raise RuntimeError("TARGET_TOKENIZATION_FAILURE") from error


def decode_target(ids: list[int]) -> str:
    if not ids or ids[-1] != 15 or 15 in ids[:-1]:
        raise RuntimeError("TARGET_EOS_CONTRACT_FAILURE")
    mapping = {i: str(i) for i in range(10)} | {10: "\n"}
    try:
        return "".join(mapping[token] for token in ids[:-1])
    except KeyError as error:
        raise RuntimeError("TARGET_TOKEN_DECODE_FAILURE") from error


def parse_grid(text: str) -> list[list[int]]:
    if not text or "\r" in text:
        raise RuntimeError("TARGET_PARSE_FAILURE")
    rows = text.split("\n")
    if any(not row or any(ch not in "0123456789" for ch in row) for row in rows):
        raise RuntimeError("TARGET_PARSE_FAILURE")
    grid = [[int(ch) for ch in row] for row in rows]
    if any(len(row) != len(grid[0]) for row in grid):
        raise RuntimeError("TARGET_PARSE_FAILURE")
    return grid


def git_commit(root: Path) -> str:
    return subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()


def task_target(task: dict) -> list[list[int]]:
    test = task.get("test")
    if not isinstance(test, list) or len(test) != 1 or "output" not in test[0]:
        raise RuntimeError("ORACLE_TARGET_NOT_MATERIALIZED")
    return test[0]["output"]


def expected_rows(root: Path, oracle_source: Path) -> list[dict]:
    oracle = module_from(oracle_source)
    pilot = root / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1"
    train = load(pilot / "measurement_contract_v1/TRAIN_FIT_INPUT_PROMPTS.json", "708596c8715c5380903ab1af89852c32f05bb9176fb45cfd6c95f33e63dff981")["records"]
    ood = load(pilot / "OOD_INPUT_PROMPTS.json", "b50687c9d9493a3f2dc1ce6f8e369aa48915277b0818a71b22bec16a34f57a3b")
    rows: list[dict] = []
    for item in train:
        if item["family"] != "CONNECTED_COMPONENT_COUNT":
            continue
        source = item["source_id"].split(":")
        rebuilt = oracle.task(source[2], source[3], int(source[4]) * 4, item["source_id"], True)
        prompt = oracle.task(source[2], source[3], int(source[4]) * 4, item["source_id"], False)
        if prompt != item["task"]:
            raise RuntimeError("TRAIN_PROMPT_ORACLE_MISMATCH")
        rows.append({
            "cohort_record_id": "TRAIN_FIT:" + item["source_id"], "split": "TRAIN_FIT",
            "family": item["family"], "source_arm": item["arm"], "source_slot": item["source_slot"],
            "prompt_task": prompt, "target_grid": task_target(rebuilt),
        })
    for item in ood:
        source = item["task"]["source_id"].split(":")
        index = int(source[-1])
        latent = item["latent_condition"]
        rebuilt = oracle.task(latent["family"], latent["topology"], 10000 + index * 4, item["task"]["source_id"], True)
        prompt = oracle.task(latent["family"], latent["topology"], 10000 + index * 4, item["task"]["source_id"], False)
        if prompt != item["task"]:
            raise RuntimeError("OOD_PROMPT_ORACLE_MISMATCH")
        rows.append({
            "cohort_record_id": f"OOD:{index}", "split": "OOD", "family": latent["family"],
            "source_arm": None, "source_slot": index, "prompt_task": prompt, "target_grid": task_target(rebuilt),
        })
    if len(rows) != 72 or Counter(row["split"] for row in rows) != {"TRAIN_FIT": 24, "OOD": 48}:
        raise RuntimeError("SUCCESSOR_COHORT_COUNT_MISMATCH")
    if Counter(row["family"] for row in rows if row["split"] == "TRAIN_FIT") != {"CONNECTED_COMPONENT_COUNT": 24}:
        raise RuntimeError("TRAIN_FAMILY_MISMATCH")
    return sorted(rows, key=lambda row: row["cohort_record_id"])


def sidecar_rows(rows: list[dict]) -> list[dict]:
    output = []
    for row in rows:
        target_text = serialize_grid(row["target_grid"])
        ids = encode_target(target_text)
        decoded = decode_target(ids)
        if decoded != target_text or parse_grid(decoded) != row["target_grid"]:
            raise RuntimeError("TARGET_ROUNDTRIP_FAILURE")
        output.append({
            "cohort_record_id": row["cohort_record_id"], "split": row["split"], "family": row["family"],
            "source_arm": row["source_arm"], "source_slot": row["source_slot"],
            "prompt_task_sha256": sha_value(row["prompt_task"]),
            "canonical_target_grid_sha256": sha_value(row["target_grid"]),
            "target_serialization_sha256": hashlib.sha256(target_text.encode("utf-8")).hexdigest(),
            "target_token_ids": ids, "target_token_ids_sha256": sha_value(ids),
            "supervised_token_mask": [1] * len(ids), "supervised_token_count": len(ids),
            "target_eos_token_id": 15,
        })
    return output


def raw_rows(recovered: Path, checkpoint: str) -> list[dict]:
    name = "V7_RAW.jsonl" if checkpoint == "V7" else checkpoint + "_RAW.jsonl"
    path = recovered / name
    if not path.is_file():
        raise RuntimeError(f"MISSING_RAW:{name}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def reuse_inventory(sidecar: list[dict], recovered: Path) -> tuple[dict, list[dict]]:
    expected_by_sha = {row["prompt_task_sha256"]: row for row in sidecar}
    inventory: list[dict] = []
    missing: list[dict] = []
    for checkpoint in CHECKPOINTS:
        raw = raw_rows(recovered, checkpoint)
        matching: dict[str, dict] = {}
        for entry in raw:
            digest = entry.get("prompt_sha256")
            if digest in expected_by_sha:
                if digest in matching:
                    raise RuntimeError("DUPLICATE_COMPATIBLE_RAW_PROMPT")
                matching[digest] = entry
        for row in sidecar:
            entry = matching.get(row["prompt_task_sha256"])
            data = {
                "checkpoint": checkpoint, "adapter_model_sha256": CHECKPOINTS[checkpoint],
                "cohort_record_id": row["cohort_record_id"], "split": row["split"],
                "prompt_task_sha256": row["prompt_task_sha256"],
            }
            if entry is None:
                data["disposition"] = "GENERATE_NEW_ONCE_AFTER_FRESH_AUTHORIZATION"
                missing.append(data)
            else:
                if entry.get("effective_batch_size") not in {1, 4, 8, 16, 32}:
                    raise RuntimeError("INVALID_REUSED_EFFECTIVE_BATCH")
                data.update({
                    "disposition": "REUSE_IMMUTABLE_RUN_002_RAW",
                    "run_002_raw_file": "V7_RAW.jsonl" if checkpoint == "V7" else checkpoint + "_RAW.jsonl",
                    "run_002_record_id": entry["record_id"], "run_002_raw_record_sha256": sha_value(entry),
                    "run_002_effective_batch_size": entry["effective_batch_size"],
                })
            inventory.append(data)
    if len(inventory) != 216 or len(missing) != 48:
        raise RuntimeError(f"REUSE_SET_DIFFERENCE_INVALID:{len(inventory)}:{len(missing)}")
    if Counter(item["checkpoint"] for item in missing) != {"V7": 24, "CONTROL_NARROW_STRUCTURE": 12, "TREATMENT_STRUCTURE_DIVERSE": 12}:
        raise RuntimeError("MISSING_CHECKPOINT_BALANCE_INVALID")
    if any(item["split"] != "TRAIN_FIT" for item in missing):
        raise RuntimeError("OOD_REGENERATION_PROHIBITED")
    summary = {
        "schema_version": 1, "run_id": RUN_ID, "table_shape": "3 checkpoints x 72 frozen cohort records",
        "expected_checkpoint_record_rows": 216,
        "reused_immutable_run_002_rows": 168,
        "new_free_running_rows_required": 48,
        "reuse_rule": "Only exact prompt_task_sha256 matches are reusable; reused rows retain their original raw-record SHA256.",
        "new_generation_rule": "Generate only the listed missing rows once, freeze their raw token IDs before any scorer-side sidecar access, and never replace a reused row.",
        "inventory": inventory,
    }
    return summary, missing


def select_positions(rows: list[dict], count: int) -> list[dict]:
    ordered = sorted(rows, key=lambda x: (x["cohort_record_id"], x.get("checkpoint", "")))
    return [ordered[(i * len(ordered)) // count] for i in range(count)]


def sensitivity(sidecar: list[dict], missing: list[dict]) -> dict:
    train = [row for row in sidecar if row["split"] == "TRAIN_FIT"]
    ood = [row for row in sidecar if row["split"] == "OOD"]
    teacher_records = select_positions(train, 6) + select_positions(ood, 6)
    teacher = [{"cohort_record_id": r["cohort_record_id"], "split": r["split"], "prompt_task_sha256": r["prompt_task_sha256"]} for r in teacher_records]
    generation = select_positions(missing, 12)
    if len({x["cohort_record_id"] for x in teacher}) != 12 or len(generation) != 12:
        raise RuntimeError("B1_SUBSET_INVALID")
    return {
        "requested_batch_size": 32, "fallback_ladder": [32, 16, 8, 4, 1], "fallback_permitted_only_for": "CUDA_OOM",
        "teacher_forced_batch1_subset": {"cohort_rows": teacher, "checkpoint_record_forwards": 36},
        "new_free_running_batch1_subset": {"checkpoint_record_rows": generation, "generations": 12},
        "teacher_forced_comparison": {
            "correct_token_rank_abs_delta_max": 1,
            "Top1_and_Top2_aggregate_count_delta_max": 0,
            "per_row_mean_margin_abs_delta_max": 0.02,
            "token_weighted_NLL_abs_delta_max": 0.005,
            "sequence_normalized_NLL_abs_delta_max": 0.005,
            "failure_disposition": "INVALID_NOT_INTERPRETABLE",
        },
        "free_running_comparison": {
            "parser_validity_agreement_required": 1.0,
            "canonical_parsed_grid_agreement_required": 1.0,
            "scorer_exactness_agreement_required": 1.0,
            "aggregate_exact_count_delta_max": 0,
            "failure_disposition": "INVALID_NOT_INTERPRETABLE",
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--recovered-root", type=Path, required=True)
    ap.add_argument("--sidecar-out", type=Path, required=True)
    ap.add_argument("--inventory-out", type=Path, required=True)
    ap.add_argument("--protocol-out", type=Path, required=True)
    ap.add_argument("--validation-out", type=Path, required=True)
    args = ap.parse_args()
    root = args.repo.resolve()
    source = root / "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py"
    if sha_file(source) != "a2cf87957fb0e29c412b850acd64a098695570c50fa739f9bccb6942a33f7de2":
        raise RuntimeError("ORACLE_SOURCE_IDENTITY_MISMATCH")
    predecessor = root / "experiments/capability_repair_baseline_v1/connected_components_object_extraction_capability_pilot_v1/connected_component_count_fit_vs_generation_diagnostic_v1/CPU_ONLY_FROZEN_PROTOCOL.json"
    if sha_file(predecessor) != "718b4e1b525fa5672d9db2ba78ea97458d02f464cce9fa618315ba31f9a20d49":
        raise RuntimeError("PREDECESSOR_PROTOCOL_IDENTITY_MISMATCH")
    manifest = root / "experiments/capability_repair_baseline_v1/CHECKPOINT_MANIFEST_REMOTE_V1.json"
    if sha_file(manifest) != "1e124cc4f43530bbbc71103703d404b38df83a6998a3e39d7c1da0676c800549":
        raise RuntimeError("CHECKPOINT_MANIFEST_IDENTITY_MISMATCH")
    rows = expected_rows(root, source)
    sidecar = sidecar_rows(rows)
    sidecar_doc = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "SEALED_SCORER_ONLY_CORRECT_TOKEN_SIDECAR",
        "access_boundary": "Scorer-side only. Never serialize into prompts or target-blind generation bundles; raw new generations must freeze before sidecar access.",
        "oracle_source": {"path": "scripts/freeze_connected_components_object_extraction_capability_pilot_v1.py", "sha256": sha_file(source)},
        "tokenizer": TOKENIZER, "records": sidecar,
    }
    sidecar_sha = write(args.sidecar_out, sidecar_doc, sealed=True)
    inventory, missing = reuse_inventory(sidecar, args.recovered_root)
    inventory_sha = write(args.inventory_out, inventory)
    sensitivity_contract = sensitivity(sidecar, missing)
    protocol = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "CPU_ONLY_FROZEN_SUCCESSOR_PENDING_DIRECTOR_REVIEW",
        "successor_of": {"protocol_id": OLD_PROTOCOL_ID, "predecessor_protocol_sha256": sha_file(predecessor)},
        "exact_source_commit": git_commit(root),
        "immutable_conditions": {"checkpoint_adapter_model_sha256": CHECKPOINTS, "checkpoint_manifest_sha256": sha_file(manifest), "decoder": "NATIVE_GREEDY", "max_new_tokens": 192, "optimizer_steps": 0, "backward_calls": 0, "training": False, "retry": False, "one_detached_job_only": True, "no_new_cohort": True, "no_target_release": True, "run_002_retry": False},
        "correct_token_sidecar": {"path": str(args.sidecar_out.resolve()), "sha256": sidecar_sha, "records": 72, "tokenizer": TOKENIZER},
        "reuse_and_new_generation": {"inventory_path": str(args.inventory_out.resolve()), "inventory_sha256": inventory_sha, "reuse_rows": 168, "new_free_running_rows": 48, "required_set_difference": "48", "target_boundary": "Freeze new raw rows before scorer-side sidecar access."},
        "teacher_forced": {"checkpoint_record_forwards": 216, "metrics": ["correct_token_rank_distribution", "Top1_rate_rank_eq_1", "Top2_rate_rank_le_2", "mean_correct_token_margin", "token_weighted_NLL", "sequence_normalized_NLL"]},
        "batch_sensitivity": sensitivity_contract,
        "execution_shape_and_budget": {"primary_teacher_forced_forwards": 216, "Batch1_teacher_forced_forwards": 36, "primary_new_free_running_generations": 48, "Batch1_new_free_running_generations": 12, "total_model_operations": 312, "planning_estimate_seconds": 3600, "hard_cap_seconds": 10800},
        "fresh_execution_authorization": {"required": True, "unconsumed": True, "must_bind": ["this protocol SHA256", "sealed sidecar SHA256", "reuse inventory SHA256", "exact source commit", "all three checkpoint hashes", "one detached job", "retry=false", "hard cap 10800"], "authorization_not_granted_by_this_artifact": True},
        "forbidden": ["GPU use during this CPU freezer", "model loading", "optimizer", "backward", "parameter update", "target content in prompts", "Gold", "dGold", "FINAL_AUDIT", "E05", "V4", "promotion"],
    }
    protocol_sha = write(args.protocol_out, protocol)
    validation = {
        "schema_version": 1, "protocol_id": PROTOCOL_ID, "status": "PASS_CPU_ONLY_SUCCESSOR_VALIDATION",
        "protocol_sha256": protocol_sha, "sidecar_sha256": sidecar_sha, "inventory_sha256": inventory_sha,
        "checks": {"oracle_prompt_reconstruction_72": "PASS", "correct_token_encode_decode_and_grid_roundtrip_72": "PASS", "sealed_sidecar_has_no_grid_text_field": "PASS", "RUN_002_exact_reuse_rows": 168, "RUN_002_missing_rows": 48, "OOD_regeneration": "PROHIBITED_AND_NOT_SCHEDULED", "B1_contract": "FROZEN", "model_loaded": False, "gpu_used": False, "generation_started": False, "optimizer_constructed": False, "backward_calls": 0},
    }
    validation_sha = write(args.validation_out, validation)
    print(json.dumps({"status": validation["status"], "protocol_sha256": protocol_sha, "sidecar_sha256": sidecar_sha, "inventory_sha256": inventory_sha, "validation_sha256": validation_sha}, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())