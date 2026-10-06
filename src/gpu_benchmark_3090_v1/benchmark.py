"""Pure contracts and CPU-side helpers for the bounded RTX 3090 benchmark."""
from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "a25c9b313bf0c1571857e149350f305d9e0a8f4c"
DATASET_FINGERPRINT = "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"
REPLAY_ROWS = 62_650
REPLAY_SHA256 = "32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc"
SEED = 31_090
POOL_WEIGHTS = {"POOL_NOVEL_V1_1": 0.75, "POOL_REPLAY_V2_1": 0.25}
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


class BenchmarkGateError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def validate_model_package(model_dir: Path) -> dict[str, Any]:
    required = ["config.json", "tokenizer_config.json", "tokenizer.json", "model.safetensors.index.json"]
    missing = [name for name in required if not (model_dir / name).is_file()]
    if missing:
        raise BenchmarkGateError(f"INFRA_MODEL_PATH_INVALID missing={missing}")
    index = json.loads((model_dir / "model.safetensors.index.json").read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    missing_shards = [name for name in shards if not (model_dir / name).is_file()]
    if not shards or missing_shards:
        raise BenchmarkGateError(f"INFRA_MODEL_PATH_INVALID shards={missing_shards or 'EMPTY'}")
    manifest_path = model_dir / "model_manifest.json"
    checked: dict[str, str] = {}
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for name, expected in sorted(manifest.get("file_sha256", {}).items()):
            path = model_dir / name
            if not path.is_file() or sha256_file(path) != expected:
                raise BenchmarkGateError(f"INFRA_MODEL_PATH_INVALID hash={name}")
            checked[name] = expected
    return {"required_files": required, "safetensor_shards": shards, "manifest_hashes": checked}


def require_single_cuda_device(device_count: int, device_name: str) -> None:
    if device_count != 1:
        raise BenchmarkGateError(f"EXACT_SINGLE_GPU_REQUIRED count={device_count}")
    if "RTX 3090" not in device_name:
        raise BenchmarkGateError(f"RTX3090_REQUIRED name={device_name}")


def assert_lora_partition(named_parameters: Iterable[tuple[str, Any]]) -> dict[str, int]:
    trainable = 0
    total = 0
    invalid: list[str] = []
    for name, parameter in named_parameters:
        count = int(parameter.numel())
        total += count
        if bool(parameter.requires_grad):
            trainable += count
            if "lora_" not in name:
                invalid.append(name)
    if not trainable or invalid:
        raise BenchmarkGateError(f"LORA_PARTITION_INVALID trainable={trainable} invalid={invalid[:5]}")
    return {"trainable_parameters": trainable, "total_parameters": total, "base_trainable_parameters": 0}


def make_family_schedule(
    novel_rows: Sequence[dict[str, Any]],
    replay_rows: Sequence[dict[str, Any]],
    draws: int,
    seed: int = SEED,
) -> list[dict[str, Any]]:
    by_pool: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for pool, rows in (("POOL_NOVEL_V1_1", novel_rows), ("POOL_REPLAY_V2_1", replay_rows)):
        families: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            role = str(row.get("final_training_role", ""))
            split = str(row.get("split", "train"))
            if split != "train" or any(term in role for term in ("HOLDOUT", "HARD_EXCLUDE", "QUARANTINE")):
                continue
            families[str(row["generator_family"])].append(dict(row))
        if not families:
            raise BenchmarkGateError(f"EMPTY_BENCHMARK_POOL={pool}")
        by_pool[pool] = dict(families)
    rng = random.Random(seed)
    novel_draws = round(draws * POOL_WEIGHTS["POOL_NOVEL_V1_1"])
    pools = ["POOL_NOVEL_V1_1"] * novel_draws + ["POOL_REPLAY_V2_1"] * (draws - novel_draws)
    rng.shuffle(pools)
    schedule: list[dict[str, Any]] = []
    for index, pool in enumerate(pools):
        family = rng.choice(sorted(by_pool[pool]))
        row = rng.choice(by_pool[pool][family])
        schedule.append({
            "schedule_index": index,
            "pool": pool,
            "sample_id": str(row["sample_id"]),
            "source": str(row["source"]),
            "generator_family": family,
            "sequence_length": int(row["sequence_length"]),
            "supervised_token_count": int(row["supervised_token_count"]),
        })
    return schedule


def truncate_example(input_ids: Sequence[int], labels: Sequence[int], max_length: int) -> dict[str, Any]:
    if len(input_ids) != len(labels):
        raise BenchmarkGateError("TOKEN_LABEL_LENGTH_MISMATCH")
    kept_ids = list(input_ids[:max_length])
    kept_labels = list(labels[:max_length])
    removed = list(labels[max_length:])
    return {
        "input_ids": kept_ids,
        "labels": kept_labels,
        "original_tokens": len(input_ids),
        "processed_tokens": len(kept_ids),
        "original_supervised_tokens": sum(int(value != -100) for value in labels),
        "processed_supervised_tokens": sum(int(value != -100) for value in kept_labels),
        "supervised_tokens_lost": sum(int(value != -100) for value in removed),
        "truncated": len(input_ids) > max_length,
    }


def runtime_extrapolation(tokens_per_second: float) -> dict[str, dict[str, float]]:
    if tokens_per_second <= 0:
        raise BenchmarkGateError("NONPOSITIVE_THROUGHPUT")
    result = {}
    for tokens in (2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000):
        seconds = tokens / tokens_per_second
        result[str(tokens)] = {"seconds": seconds, "hours": seconds / 3600.0}
    return result


def vram_safety(total_bytes: int, peak_reserved_bytes: int) -> dict[str, Any]:
    headroom = total_bytes - peak_reserved_bytes
    gib = headroom / (1024**3)
    level = "SAFE" if gib >= 1.5 else "MARGINAL" if gib >= 0.5 else "UNSAFE"
    return {
        "headroom_bytes": headroom,
        "headroom_gib": gib,
        "headroom_percentage": 100.0 * headroom / total_bytes,
        "classification": level,
    }


def derive_gate(checks: dict[str, bool], result_8192: dict[str, Any] | None, result_8704: dict[str, Any] | None,
                diagnostic_8512: dict[str, Any] | None = None) -> str:
    if not checks.get("data_access_policy", False):
        return "FAIL_DATA_ACCESS_POLICY"
    if not checks.get("infrastructure", False):
        return "FAIL_INFRASTRUCTURE"
    if not checks.get("numerical", False):
        return "FAIL_NUMERICAL"
    if result_8704 and result_8704.get("status") == "PASS" and result_8704.get("vram_safety") != "UNSAFE":
        return "PASS_8704_RECOMMENDED"
    if result_8192 and result_8192.get("status") == "PASS":
        if diagnostic_8512 and diagnostic_8512.get("status") == "PASS" and diagnostic_8512.get("vram_safety") != "UNSAFE":
            return "PASS_8512_FULL_CONTENT_CANDIDATE"
        return "PASS_8192_ONLY"
    if any(value and value.get("failure_class") == "OOM" for value in (result_8192, result_8704) if value):
        return "FAIL_GPU_OOM"
    return "FAIL_INFRASTRUCTURE"
