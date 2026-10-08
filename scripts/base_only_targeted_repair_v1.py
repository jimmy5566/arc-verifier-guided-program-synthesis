#!/usr/bin/env python3
"""Frozen, fail-closed base-only targeted-repair execution contract.

The command has three deliberately separated phases.  ``freeze`` is CPU-only
and turns only TRAIN JSONL into native-token samples and a deterministic
schedule.  ``preflight`` hashes bytes and source without importing a model.
``launch`` is the sole model/optimizer entry point; it rejects work without a
later Director authorization, a passing base reference, a passing preflight,
and a reserved slice of the shared 28,800-second ledger.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from training_data.pipeline import IGNORE_INDEX, task_to_sample  # noqa: E402

PROTOCOL = "BASE_ONLY_TARGETED_REPAIR_AVAILABLE_DATA_V1"
CONTEXT = 8704
SEED = 2_000_031
ACCUMULATION = 4
NOMINAL_TOKENS = 500_000
RESERVATION_SECONDS = 7_200
CAP_SECONDS = 28_800
BASE_FILES = {
    "added_tokens.json": "cf1224e6a594b51a34b8b9ca6c6dce3026683777521f361e37526b8a33258bce",
    "config.json": "932e9430395ba3eea4087cba1769241bc7d0dc8d574846862d6983659456a2d0",
    "generation_config.json": "fd0abf1e2baa1f9407803a83486767a6e7841164481d750fcaa9b71d31a0ab02",
    "model-00001-of-00002.safetensors": "379de219f74a1402a67d26fe7dcacdacadbf686947e08e41117dd9c884ee4529",
    "model-00002-of-00002.safetensors": "3b8470b83355ccbc527afcb7a77664bd679e63899a80e79169c016658e6b92e0",
    "model.safetensors.index.json": "d2d5b2bffaf2fd27bad4452dfb4d4df80c56a4a77f1ac9c40fdd634845680cd4",
    "model_manifest.json": "9e995b611936791884063b69ef85cba12d89c53dd1c6cebe33ac155145c9aee7",
    "special_tokens_map.json": "f4f79e08d97f4d1c87f8d89264f525c8789da3b73b3bb55d1e12f692f41a7b1b",
    "tokenizer.json": "af6b9e855f9db3192bd9152607708ebd7700b40f9fc5f792a26af70150a96cd5",
    "tokenizer_config.json": "e1912aeba4050616cbfe01b34df3f3df2818207eda0056f6eb87c2336936ea90",
    "vocab.json": "8141a61e47eceec466524bc178bc9c6d5076b7d1b18b456fbd3fdf66cae57f82",
}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    tmp.replace(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise RuntimeError(f"EMPTY_JSONL:{path}")
    return rows


def training_samples(path: Path) -> dict[str, dict[str, Any]]:
    samples: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(path):
        if row.get("split") != "TRAIN" or row.get("role") not in {"ATOMIC_REPAIR", "COMPOSITION_REPAIR", "REPLAY_RETENTION"}:
            raise RuntimeError("TRAIN_SURFACE_ROLE_OR_SPLIT_INVALID")
        if not isinstance(row.get("episode_id"), str) or not isinstance(row.get("task"), dict):
            raise RuntimeError("TRAIN_RECORD_INVALID")
        sample = task_to_sample({"source_id": row["episode_id"], **row["task"]})
        if sample["sequence_length"] > CONTEXT or len(sample["input_ids"]) != len(sample["labels"]):
            raise RuntimeError("TRAIN_SERIALIZATION_CONTEXT_OR_LABEL_INVALID")
        if not any(token != IGNORE_INDEX for token in sample["labels"]):
            raise RuntimeError("TRAIN_SERIALIZATION_ZERO_SUPERVISION")
        if sample["sample_id"] in samples:
            raise RuntimeError("TRAIN_SAMPLE_ID_COLLISION")
        samples[sample["sample_id"]] = {**sample, "episode_id": row["episode_id"], "role": row["role"], "family": row["family"]}
    return samples


def schedule_for(samples: dict[str, dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(samples, key=lambda key: hashlib.sha256(f"{SEED}:{key}".encode("utf-8")).hexdigest())
    episodes: list[dict[str, Any]] = []
    total = 0
    cursor = 0
    # Stop only after a complete gradient-accumulation group crosses the cap.
    while total < NOMINAL_TOKENS or len(episodes) % ACCUMULATION:
        key = ordered[cursor % len(ordered)]
        item = samples[key]
        episodes.append({
            "sample_id": key, "episode_id": item["episode_id"], "role": item["role"], "family": item["family"],
            "sequence_length": item["sequence_length"], "supervised_token_count": item["assistant_token_count"],
            "token_label_sha256": digest({"input_ids": item["input_ids"], "labels": item["labels"]}),
        })
        total += int(item["sequence_length"])
        cursor += 1
    if len(episodes) % ACCUMULATION or total < NOMINAL_TOKENS:
        raise RuntimeError("SCHEDULE_STOP_RULE_INVALID")
    return {"schema_version": 1, "protocol_id": PROTOCOL, "seed": SEED,
            "sampling_policy": "SORT_SHA256(seed:sample_id)_THEN_CYCLIC_REPEAT", "source_pool": "TRAIN_JSONL_ONLY",
            "nominal_transformer_tokens": NOMINAL_TOKENS, "actual_transformer_tokens": total,
            "optimizer_steps": len(episodes) // ACCUMULATION, "gradient_accumulation_steps": ACCUMULATION,
            "stop_rule": "first_complete_optimizer_step_at_or_above_nominal_transformer_tokens", "episodes": episodes}


def identity_rows(path: Path, expected_role: str | None = None) -> list[dict[str, str]]:
    result: list[dict[str, str]] = []
    # Evaluation bytes are read only to freeze identifiers; no grid is supplied
    # to the serializer or schedule builder.
    for row in load_jsonl(path):
        if expected_role and row.get("role") != expected_role:
            raise RuntimeError("EVALUATION_ROLE_INVALID")
        values = {key: row.get(key) for key in ("episode_id", "episode_sha256", "grid_identity_sha256")}
        if not all(isinstance(value, str) and len(value) == 64 for key, value in values.items() if key != "episode_id") or not isinstance(values["episode_id"], str):
            raise RuntimeError("EVALUATION_IDENTITY_INVALID")
        result.append(values)  # type: ignore[arg-type]
    return result


def source_entries() -> dict[str, str]:
    paths = [
        ROOT / "scripts" / "base_only_targeted_repair_v1.py",
        ROOT / "src" / "training_data" / "pipeline.py",
        ROOT / "src" / "inference" / "arc_native_io.py",
        ROOT / "configs" / "nvarc_native_846d0198" / "chat_template.j2",
        ROOT / "scripts" / "record_targeted_capability_repair_gpu_time.py",
    ]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def freeze(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"REFUSE_OVERWRITE_FROZEN_PACKAGE:{args.output}")
    samples = training_samples(args.train)
    schedule = schedule_for(samples)
    out = args.output
    vectors = []
    for key in sorted(samples)[:3]:
        item = samples[key]
        vectors.append({"sample_id": key, "episode_id": item["episode_id"], "input_ids": item["input_ids"], "labels": item["labels"],
                        "sequence_length": item["sequence_length"], "assistant_token_count": item["assistant_token_count"],
                        "token_label_sha256": digest({"input_ids": item["input_ids"], "labels": item["labels"]})})
    data = {
        "TRAIN": {"local_path": str(args.train.resolve()), "remote_path": args.remote_inputs.rstrip("/") + "/TRAIN.jsonl", "bytes": args.train.stat().st_size, "sha256": sha(args.train)},
        "TARGET_DEV": {"local_path": str(args.target_dev.resolve()), "remote_path": args.remote_inputs.rstrip("/") + "/TARGET_DEV.jsonl", "bytes": args.target_dev.stat().st_size, "sha256": sha(args.target_dev)},
        "RETENTION_SENTINEL": {"local_path": str(args.retention.resolve()), "remote_path": args.remote_inputs.rstrip("/") + "/RETENTION_SENTINEL.jsonl", "bytes": args.retention.stat().st_size, "sha256": sha(args.retention)},
        "FINAL_AUDIT_SEALED": {"local_path": str(args.final_audit.resolve()), "remote_path": args.remote_inputs.rstrip("/") + "/FINAL_AUDIT_SEALED.jsonl", "bytes": args.final_audit.stat().st_size, "sha256": sha(args.final_audit), "model_accessed": False},
    }
    serialization = {
        "schema_version": 1, "protocol_id": PROTOCOL, "status": "FROZEN", "transport": "NVARC_NATIVE_16_TOKEN",
        "prompt_template": "<|im_start|>{role}\\n{grid}<|im_end|> repeated in task pair order; no generation prompt in supervised samples",
        "pair_order": "all task.train pairs then task.test pairs, each user input followed by assistant output",
        "token_ids": {"0..9": [0,1,2,3,4,5,6,7,8,9], "newline": 10, "user": 11, "assistant": 12, "pad": 13, "im_start": 14, "im_end_and_eos": 15},
        "label_masking": "all user-turn and delimiters labels=-100; all assistant-turn tokens including delimiters labels=input_ids",
        "truncation": "REJECT sequence_length > 8704; no truncation", "padding": "none per microbatch=1", "sequence_packing": "forbidden",
        "source_hashes": source_entries(), "tokenizer_files": {name: BASE_FILES[name] for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json")},
    }
    recipe = {"base_weights": "FROZEN", "precision": "BF16", "quantization": "NONE", "lora_rank": 64, "lora_alpha": 32,
              "lora_dropout": 0.0, "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
              "optimizer": "bitsandbytes.optim.PagedAdamW8bit", "learning_rate": 0.00005,
              "learning_rate_schedule": "linear_warmup_optimizer_steps_1_to_3_then_constant", "context_length": CONTEXT,
              "micro_batch_size": 1, "gradient_accumulation_steps": ACCUMULATION, "effective_episode_batch": ACCUMULATION,
              "attention_backend": "sdpa", "gradient_checkpointing": True, "seed": SEED,
              "checkpoint_landmarks_transformer_tokens": [250000, 500000], "initial_gpu_reservation_seconds": RESERVATION_SECONDS,
              "cumulative_gpu_cap_seconds": CAP_SECONDS}
    gate = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "FROZEN_BASE_REFERENCE_PENDING_SEPARATE_AUTHORIZATION",
            "TARGET_DEV": {"file_sha256": data["TARGET_DEV"]["sha256"], "episodes": identity_rows(args.target_dev), "metric": "exact_grid_match; paired n01/n10 vs frozen base predictions"},
            "RETENTION_SENTINEL": {"file_sha256": data["RETENTION_SENTINEL"]["sha256"], "episodes": identity_rows(args.retention, "RETENTION_SENTINEL"), "metric": "exact_grid_match; paired n01/n10 vs frozen base predictions"},
            "metric_source_hash": sha(Path(__file__).resolve()), "base_reference_status": "NOT_COLLECTED_MODEL_LOADING_OR_GPU_EXECUTION_NOT_AUTHORIZED_BY_DIRECTIVE_010",
            "selection": "highest TARGET_DEV net paired fixes; then lowest retention net harms; then earliest checkpoint token count", "final_audit": "SEALED_UNOPENED"}
    binding = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "FROZEN_PENDING_READ_ONLY_PREFLIGHT", "source_hashes": source_entries(),
               "base_path": "/workspace/arc2/models/qwen3_4b_grids15_sft139", "base_files": BASE_FILES, "data": data,
               "serialization_contract_sha256": digest(serialization), "schedule_sha256": digest(schedule), "recipe_sha256": digest(recipe),
               "base_reference_gate_sha256": digest(gate), "output": {"run_root": args.run_root, "runtime": args.run_root.rstrip("/") + "/runtime", "checkpoints": args.run_root.rstrip("/") + "/checkpoints", "reservation": args.run_root.rstrip("/") + "/runtime/GPU_RESERVATION.json"},
               "shared_ledger": "/root/arc-runtime-3090-gpu-benchmark-v1/arc2/experiments/targeted_capability_repair_v1/TARGETED_CAPABILITY_REPAIR_GPU_TIME_LEDGER.jsonl",
               "forbidden": ["historical replay", "replay substitution", "Eval60 Gold", "diagnostic Gold", "TARGET_DEV training", "retention training", "FINAL_AUDIT access"]}
    write(out / "SERIALIZATION_CONTRACT_V1.json", serialization)
    write(out / "SERIALIZATION_TEST_VECTORS_V1.json", {"schema_version": 1, "status": "FROZEN", "vectors": vectors})
    write(out / "TRAINING_SCHEDULE_ROUND_001.json", schedule)
    write(out / "EXECUTION_CONFIG_ROUND_001.json", recipe)
    write(out / "BASE_REFERENCE_AND_GATE_IDENTITIES_V1.json", gate)
    write(out / "LAUNCH_BINDING_V1.json", binding)
    write(out / "CUMULATIVE_BUDGET_BINDING_V1.json", {"schema_version": 1, "status": "FROZEN", "cap_seconds": CAP_SECONDS, "initial_reservation_seconds": RESERVATION_SECONDS, "actual_charges_only": True, "ledger": binding["shared_ledger"], "dummy_tests_charged_gpu_seconds": 0})
    write(out / "PACKAGE_MANIFEST_V2.json", {"schema_version": 1, "status": "FROZEN_PENDING_READ_ONLY_PREFLIGHT", "files": {p.name: sha(p) for p in sorted(out.glob("*.json"))}})
    print(json.dumps({"status": "FROZEN", "samples": len(samples), "tokens": schedule["actual_transformer_tokens"], "steps": schedule["optimizer_steps"]}, sort_keys=True))
    return 0


def verify_file(path: Path, expected: str) -> None:
    if not path.is_file():
        raise RuntimeError(f"REQUIRED_FILE_MISSING:{path}")
    if sha(path) != expected:
        raise RuntimeError(f"REQUIRED_FILE_HASH_MISMATCH:{path}")


def preflight(args: argparse.Namespace) -> int:
    raw = args.binding.read_bytes(); binding = json.loads(raw.decode("utf-8"))
    out: dict[str, Any] = {"schema_version": 1, "protocol_id": PROTOCOL, "status": "FAIL_CLOSED", "launch_binding_sha256": hashlib.sha256(raw).hexdigest(), "no_model_loaded": True, "no_optimizer_constructed": True}
    try:
        if binding.get("status") != "FROZEN_PENDING_READ_ONLY_PREFLIGHT": raise RuntimeError("BINDING_STATUS_INVALID")
        source_root = args.source_root.resolve()
        for rel, expected in binding["source_hashes"].items(): verify_file(source_root / rel, expected)
        for name, expected in binding["base_files"].items(): verify_file(args.base / name, expected)
        expected_inputs = {str(spec["remote_path"]): spec for spec in binding["data"].values()}
        actual_inputs = {str(path): path for path in args.inputs.iterdir() if path.is_file()}
        if set(actual_inputs) != set(expected_inputs): raise RuntimeError("MOUNTED_DATA_FILE_SET_MISMATCH")
        mounted = []
        for path_text, spec in sorted(expected_inputs.items()):
            path = Path(path_text); verify_file(path, spec["sha256"])
            if path.stat().st_size != int(spec["bytes"]): raise RuntimeError(f"MOUNTED_DATA_SIZE_MISMATCH:{path}")
            mounted.append({"path": str(path), "sha256": spec["sha256"], "bytes": spec["bytes"]})
        vectors = json.loads(args.freeze.joinpath("SERIALIZATION_TEST_VECTORS_V1.json").read_text(encoding="utf-8"))
        samples = training_samples(args.inputs / "TRAIN.jsonl")
        for vector in vectors["vectors"]:
            got = samples.get(vector["sample_id"])
            if not got or got["input_ids"] != vector["input_ids"] or got["labels"] != vector["labels"]: raise RuntimeError("SERIALIZATION_VECTOR_MISMATCH")
        schedule = json.loads(args.freeze.joinpath("TRAINING_SCHEDULE_ROUND_001.json").read_text(encoding="utf-8"))
        if digest(schedule) != binding["schedule_sha256"]: raise RuntimeError("SCHEDULE_HASH_MISMATCH")
        if any(row["sample_id"] not in samples or digest({"input_ids": samples[row["sample_id"]]["input_ids"], "labels": samples[row["sample_id"]]["labels"]}) != row["token_label_sha256"] for row in schedule["episodes"]): raise RuntimeError("SCHEDULE_SAMPLE_MISMATCH")
        if args.run_root.exists(): raise RuntimeError("RUN_OUTPUT_PATH_ALREADY_EXISTS")
        ledger = args.ledger.read_text(encoding="utf-8").splitlines()
        if not ledger or json.loads(ledger[0]).get("cap_seconds") != CAP_SECONDS: raise RuntimeError("SHARED_LEDGER_INVALID")
        source_sha = subprocess.check_output(["git", "-C", str(source_root), "rev-parse", "HEAD"], text=True).strip()
        gate = json.loads(args.freeze.joinpath("BASE_REFERENCE_AND_GATE_IDENTITIES_V1.json").read_text(encoding="utf-8"))
        out.update({"status": "PASS_TECHNICAL_BASE_REFERENCE_PENDING", "source_commit": source_sha, "verified_mounted_files": mounted, "schedule_steps": schedule["optimizer_steps"], "schedule_tokens": schedule["actual_transformer_tokens"], "base_reference_status": gate["base_reference_status"], "launch_blocked_until_base_reference_and_director_authorization": True})
    except Exception as exc:
        out["failure"] = f"{type(exc).__name__}:{exc}"
    write(args.output, out)
    print(json.dumps({"status": out["status"], "launch_binding_sha256": out["launch_binding_sha256"]}, sort_keys=True))
    return 0 if out["status"].startswith("PASS_TECHNICAL") else 1


def launch(args: argparse.Namespace) -> int:
    """Protected launcher.  Dummy mode exercises only cap/reservation plumbing."""
    pre = json.loads(args.preflight.read_text(encoding="utf-8"))
    if not pre.get("status", "").startswith("PASS_TECHNICAL"):
        raise RuntimeError("PREFLIGHT_NOT_PASS")
    if args.dummy:
        if args.dummy_seconds <= 0 or args.dummy_seconds > 5: raise RuntimeError("DUMMY_DURATION_INVALID")
        started = time.monotonic(); subprocess.run([sys.executable, "-c", f"import time; time.sleep({args.dummy_seconds})"], check=True); elapsed = time.monotonic() - started
        write(args.output, {"status": "DUMMY_CPU_ONLY_PASS", "gpu_training_started": False, "optimizer_constructed": False, "charged_gpu_training_seconds": 0, "reservation_seconds": RESERVATION_SECONDS, "elapsed_seconds": elapsed})
        return 0
    authorization = json.loads(args.authorization.read_text(encoding="utf-8")) if args.authorization else {}
    if authorization.get("decision") not in {"CONTINUE", "CONTINUE_WITH_WARNING"} or authorization.get("scientific_training_authorized") is not True:
        raise RuntimeError("DIRECTOR_AUTHORIZATION_REQUIRED")
    if pre.get("base_reference_status") != "COLLECTED_PASS":
        raise RuntimeError("FROZEN_BASE_REFERENCE_REQUIRED_BEFORE_TRAINING")
    raise RuntimeError("TRAINING_IMPLEMENTATION_REQUIRES_SEPARATE_REVIEWED_TRAIN_COMMAND")


def main() -> int:
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest="mode", required=True)
    f = sub.add_parser("freeze"); f.add_argument("--train", type=Path, required=True); f.add_argument("--target-dev", type=Path, required=True); f.add_argument("--retention", type=Path, required=True); f.add_argument("--final-audit", type=Path, required=True); f.add_argument("--output", type=Path, required=True); f.add_argument("--remote-inputs", type=str, required=True); f.add_argument("--run-root", type=str, required=True)
    q = sub.add_parser("preflight"); q.add_argument("--binding", type=Path, required=True); q.add_argument("--freeze", type=Path, required=True); q.add_argument("--source-root", type=Path, required=True); q.add_argument("--base", type=Path, required=True); q.add_argument("--inputs", type=Path, required=True); q.add_argument("--run-root", type=Path, required=True); q.add_argument("--ledger", type=Path, required=True); q.add_argument("--output", type=Path, required=True)
    l = sub.add_parser("launch"); l.add_argument("--preflight", type=Path, required=True); l.add_argument("--authorization", type=Path); l.add_argument("--dummy", action="store_true"); l.add_argument("--dummy-seconds", type=float, default=0.05); l.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    return freeze(args) if args.mode == "freeze" else preflight(args) if args.mode == "preflight" else launch(args)


if __name__ == "__main__":
    raise SystemExit(main())
