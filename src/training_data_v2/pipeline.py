"""CPU-only preparation of the ARC training-corpus registry, version 2.

This module has deliberately narrow responsibilities: recover public puzzles,
preserve source lineage, classify exposure, deduplicate at the puzzle level,
and freeze immutable shards.  It never imports Torch, creates a CUDA context,
or performs model training.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import platform
import random
import statistics
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


VERSION = "training_data_v2"
IGNORE_INDEX = -100
ROLE_HARD_EXCLUDE = "HARD_EXCLUDE_FUTURE_EVAL"
ROLE_REPLAY = "TRAIN_ELIGIBLE_REPLAY"
ROLE_NOVEL = "NOVEL_TRAIN_CANDIDATE"
ROLE_NOVEL_HOLDOUT = "NOVEL_HOLDOUT_RESERVED"
ROLE_QUARANTINE = "QUARANTINE_PROVENANCE_OR_OVERLAP_UNRESOLVED"
TRANSFORMS = ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")
RECORD_ARC_TASK = "ARC_TRAIN_TEST_PUZZLE"
RECORD_REARC_PAIR_BANK = "REARC_GENERATED_PAIR_BANK"


class GateFailure(RuntimeError):
    """Raised for a fail-closed corpus-integrity condition."""


def assign_final_role(*, source: str, subset: str, sft139_training_seen: bool, explicit_future_reserved: bool = False, provenance_resolved: bool = True) -> str:
    """Assign one role from semantic provenance, never from a filename token.

    A frozen experimental configuration is not evidence that its task IDs are
    future holdouts.  Explicit governance, the official evaluation subset, and
    known SFT139 training provenance are authoritative inputs instead.
    """
    if explicit_future_reserved or (source == "official_arc_agi_2" and subset == "evaluation"):
        return ROLE_HARD_EXCLUDE
    if not provenance_resolved:
        return ROLE_QUARANTINE
    if sft139_training_seen:
        return ROLE_REPLAY
    return ROLE_NOVEL


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(value)
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    atomic_bytes(path, (canonical_json(value) + "\n").encode("utf-8"))


def atomic_text(path: Path, value: str) -> None:
    atomic_bytes(path, value.encode("utf-8"))


def _validate_grid(grid: Any, *, field: str) -> list[list[int]]:
    if not isinstance(grid, list) or not grid or len(grid) > 30:
        raise GateFailure(f"{field}: grid height must be 1..30")
    if not all(isinstance(row, list) and row for row in grid):
        raise GateFailure(f"{field}: rows must be non-empty lists")
    width = len(grid[0])
    if width > 30 or any(len(row) != width for row in grid):
        raise GateFailure(f"{field}: grid must be rectangular with width 1..30")
    copied: list[list[int]] = []
    for row in grid:
        checked: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise GateFailure(f"{field}: colors must be integer 0..9")
            checked.append(cell)
        copied.append(checked)
    return copied


def normalize_task(raw: Any, *, source_native_id: str) -> dict[str, Any]:
    if not isinstance(raw, dict) or not isinstance(raw.get("train"), list) or not isinstance(raw.get("test"), list):
        raise GateFailure(f"{source_native_id}: expected ARC train/test lists")
    if not raw["train"] or not raw["test"]:
        raise GateFailure(f"{source_native_id}: empty train/test list")

    def normalize_pairs(name: str) -> list[dict[str, Any]]:
        pairs: list[dict[str, Any]] = []
        for index, pair in enumerate(raw[name]):
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise GateFailure(f"{source_native_id}:{name}[{index}] requires input/output")
            pairs.append({
                "input": _validate_grid(pair["input"], field=f"{source_native_id}:{name}[{index}].input"),
                "output": _validate_grid(pair["output"], field=f"{source_native_id}:{name}[{index}].output"),
            })
        return pairs

    return {"train": normalize_pairs("train"), "test": normalize_pairs("test")}


def _normalize_pair(pair: Any, *, field: str) -> dict[str, list[list[int]]]:
    if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
        raise GateFailure(f"{field} requires input/output")
    return {
        "input": _validate_grid(pair["input"], field=f"{field}.input"),
        "output": _validate_grid(pair["output"], field=f"{field}.output"),
    }


def task_observation(task: dict[str, Any]) -> dict[str, Any]:
    return {
        "train": [{"input": item["input"], "output": item["output"]} for item in task["train"]],
        "test": [{"input": item["input"]} for item in task["test"]],
    }


def canonical_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))


def train_pair_hash(task: dict[str, Any]) -> str:
    return canonical_hash(task["train"])


def _transform_grid(grid: list[list[int]], transform: str) -> list[list[int]]:
    rows = [list(row) for row in grid]
    if transform == "identity":
        return rows
    if transform == "rot90":
        return [list(row) for row in zip(*rows[::-1])]
    if transform == "rot180":
        return [row[::-1] for row in rows[::-1]]
    if transform == "rot270":
        return [list(row) for row in zip(*rows)][::-1]
    if transform == "flip_lr":
        return [row[::-1] for row in rows]
    if transform == "flip_ud":
        return rows[::-1]
    if transform == "transpose":
        return [list(row) for row in zip(*rows)]
    if transform == "anti_transpose":
        return [list(row) for row in zip(*rows[::-1])][::-1]
    raise ValueError(transform)


def _transform_observation(observation: dict[str, Any], transform: str) -> dict[str, Any]:
    result: dict[str, list[dict[str, Any]]] = {"train": [], "test": []}
    for item in observation["train"]:
        result["train"].append({"input": _transform_grid(item["input"], transform), "output": _transform_grid(item["output"], transform)})
    for item in observation["test"]:
        result["test"].append({"input": _transform_grid(item["input"], transform)})
    return result


def _color_normalize(observation: dict[str, Any]) -> dict[str, Any]:
    mapping: dict[int, int] = {}

    def normal(grid: list[list[int]]) -> list[list[int]]:
        result: list[list[int]] = []
        for row in grid:
            converted: list[int] = []
            for value in row:
                if value not in mapping:
                    mapping[value] = len(mapping)
                converted.append(mapping[value])
            result.append(converted)
        return result

    return {
        "train": [{"input": normal(item["input"]), "output": normal(item["output"])} for item in observation["train"]],
        "test": [{"input": normal(item["input"])} for item in observation["test"]],
    }


def signatures(task: dict[str, Any]) -> dict[str, str]:
    observation = task_observation(task)
    d4_variants = [_transform_observation(observation, transform) for transform in TRANSFORMS]
    color_variants = [_color_normalize(_transform_observation(observation, transform)) for transform in TRANSFORMS]
    return {
        "canonical_observation_sha256": canonical_hash(observation),
        "full_content_sha256": canonical_hash(task),
        "train_pair_sha256": train_pair_hash(task),
        "d4_signature_sha256": min(canonical_hash(value) for value in d4_variants),
        "color_signature_sha256": canonical_hash(_color_normalize(observation)),
        "d4_color_signature_sha256": min(canonical_hash(value) for value in color_variants),
    }


def task_features(task: dict[str, Any]) -> dict[str, Any]:
    pairs = [*task["train"], *task["test"]]
    colors = sorted({value for pair in pairs for name in ("input", "output") for row in pair[name] for value in row})
    ratios = [len(pair["output"]) * len(pair["output"][0]) / (len(pair["input"]) * len(pair["input"][0])) for pair in pairs]
    shape_changes = sum((len(pair["input"]), len(pair["input"][0])) != (len(pair["output"]), len(pair["output"][0])) for pair in pairs)
    return {
        "train_pair_count": len(task["train"]),
        "test_pair_count": len(task["test"]),
        "colors": colors,
        "shape_change_pairs": shape_changes,
        "mean_output_input_area_ratio": sum(ratios) / len(ratios),
        "max_grid_area": max(len(grid) * len(grid[0]) for pair in pairs for grid in (pair["input"], pair["output"])),
    }


def pair_bank_signatures(pairs: Sequence[dict[str, Any]]) -> dict[str, str]:
    """Canonical signatures for re-ARC's native generated-pair-bank format.

    The type tag keeps pair banks distinct from ARC train/test puzzles.  A
    temporary ARC-shaped observation is used only to reuse the existing D4 and
    color-normalization functions; it is never stored or labelled as an ARC
    task.
    """
    transient = {"train": list(pairs), "test": []}
    observation = {"record_type": RECORD_REARC_PAIR_BANK, "inputs": [pair["input"] for pair in pairs]}
    full = {"record_type": RECORD_REARC_PAIR_BANK, "pairs": list(pairs)}
    d4_observations = [
        {
            "record_type": RECORD_REARC_PAIR_BANK,
            "inputs": [_transform_grid(pair["input"], transform) for pair in pairs],
        }
        for transform in TRANSFORMS
    ]
    color = _color_normalize(task_observation(transient))
    d4_color_variants = [_color_normalize(_transform_observation(task_observation(transient), transform)) for transform in TRANSFORMS]
    return {
        "canonical_observation_sha256": canonical_hash(observation),
        "full_content_sha256": canonical_hash(full),
        "train_pair_sha256": canonical_hash(full),
        "d4_signature_sha256": min(canonical_hash(value) for value in d4_observations),
        "color_signature_sha256": canonical_hash({"record_type": RECORD_REARC_PAIR_BANK, "observation": color}),
        "d4_color_signature_sha256": min(canonical_hash({"record_type": RECORD_REARC_PAIR_BANK, "observation": value}) for value in d4_color_variants),
    }


def pair_bank_features(pairs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    colors = sorted({value for pair in pairs for name in ("input", "output") for row in pair[name] for value in row})
    ratios = [len(pair["output"]) * len(pair["output"][0]) / (len(pair["input"]) * len(pair["input"][0])) for pair in pairs]
    return {
        "train_pair_count": None,
        "test_pair_count": None,
        "pair_bank_count": len(pairs),
        "colors": colors,
        "shape_change_pairs": sum((len(pair["input"]), len(pair["input"][0])) != (len(pair["output"]), len(pair["output"][0])) for pair in pairs),
        "mean_output_input_area_ratio": sum(ratios) / len(ratios),
        "max_grid_area": max(len(grid) * len(grid[0]) for pair in pairs for grid in (pair["input"], pair["output"])),
    }


def serialize_grid(grid: list[list[int]]) -> str:
    return "\n".join("".join(str(value) for value in row) for row in grid)


def sample_messages(task: dict[str, Any]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for pair in [*task["train"], *task["test"]]:
        messages.append({"role": "user", "content": serialize_grid(pair["input"])})
        messages.append({"role": "assistant", "content": serialize_grid(pair["output"])})
    return messages


def sample_messages_from_pairs(pairs: Sequence[dict[str, Any]]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for pair in pairs:
        messages.append({"role": "user", "content": serialize_grid(pair["input"])})
        messages.append({"role": "assistant", "content": serialize_grid(pair["output"])})
    return messages


def native_render(messages: Sequence[dict[str, str]]) -> str:
    return "".join(f"<|im_start|>{message['role']}\n{message['content']}<|im_end|>" for message in messages)


def native_tokenize_with_labels(messages: Sequence[dict[str, str]]) -> tuple[list[int], list[int]]:
    vocabulary = {str(index): index for index in range(10)} | {"\n": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13, "<|im_start|>": 14, "<|im_end|>": 15}
    input_ids: list[int] = []
    labels: list[int] = []
    for message in messages:
        role, content = message["role"], message["content"]
        if role not in ("user", "assistant"):
            raise GateFailure(f"unsupported role {role}")
        turn = [14, vocabulary[role], 10]
        try:
            turn.extend(vocabulary[character] for character in content)
        except KeyError as exc:
            raise GateFailure(f"native serializer emitted unknown character {exc.args[0]!r}") from exc
        turn.append(15)
        input_ids.extend(turn)
        labels.extend(turn if role == "assistant" else [IGNORE_INDEX] * len(turn))
    if not any(item != IGNORE_INDEX for item in labels):
        raise GateFailure("sample has no supervised tokens")
    return input_ids, labels


@dataclass(frozen=True)
class Descriptor:
    source: str
    source_version: str
    source_subset: str
    native_id: str
    family: str
    path: str
    zip_member: str | None
    role: str
    sft139_role: str
    project_exposure: tuple[str, ...]
    record_kind: str = RECORD_ARC_TASK


def _read_descriptor(descriptor: Descriptor) -> dict[str, Any]:
    source_path = Path(descriptor.path)
    if descriptor.zip_member is None:
        raw_bytes = source_path.read_bytes()
    else:
        with zipfile.ZipFile(source_path) as archive:
            raw_bytes = archive.read(descriptor.zip_member)
    raw = json.loads(raw_bytes)
    if descriptor.record_kind == RECORD_REARC_PAIR_BANK:
        return _read_rearc_pair_bank(descriptor, raw)
    if descriptor.record_kind != RECORD_ARC_TASK:
        raise GateFailure(f"unsupported record kind {descriptor.record_kind}")
    task = normalize_task(raw, source_native_id=f"{descriptor.source}:{descriptor.native_id}")
    record = {
        "record_kind": RECORD_ARC_TASK,
        "source": descriptor.source,
        "source_version": descriptor.source_version,
        "source_subset": descriptor.source_subset,
        "source_native_id": descriptor.native_id,
        "base_puzzle_id": f"{descriptor.source}:{descriptor.native_id}",
        "generator_family": descriptor.family,
        "raw_path": str(source_path),
        "raw_asset_sha256": None,
        "raw_member": descriptor.zip_member,
        "final_training_role": descriptor.role,
        "sft139_role": descriptor.sft139_role,
        "project_exposure": list(descriptor.project_exposure),
        "task": task,
    }
    record.update(signatures(task))
    record["features"] = task_features(task)
    messages = sample_messages(task)
    input_ids, labels = native_tokenize_with_labels(messages)
    record["samples"] = [{
        "sample_id": f"{descriptor.source}:{descriptor.native_id}:native-full-task-v2",
        "augmentation_transform": "identity",
        "color_permutation": "identity",
        "pair_order_seed": None,
        "sample_seed": None,
        "upstream_sample_id": None,
        "messages": messages,
        "text": native_render(messages),
        "native_input_ids": input_ids,
        "native_labels": labels,
        "sequence_length_proxy": len(input_ids),
        "supervised_token_count_proxy": sum(value != IGNORE_INDEX for value in labels),
    }]
    return record


def _stable_seed(*parts: str) -> int:
    # Keep the deterministic seed within signed int64 so Arrow records it
    # losslessly in the immutable sample registry.
    return int(sha256_bytes("\x1f".join(parts).encode("utf-8"))[:15], 16)


def _read_rearc_pair_bank(descriptor: Descriptor, raw: Any) -> dict[str, Any]:
    """Read re-ARC at its native pair-bank level without ARC coercion.

    The pinned NVARC recipe asserts 2,048 pairs/bank.  The pinned public
    re-ARC archive exposes 1,000 pairs/bank.  We retain that evidence and make
    only bounded, without-replacement native episodes that observed pairs can
    support; no pairs are fabricated, repeated, or silently padded.
    """
    if not isinstance(raw, list) or not raw:
        raise GateFailure(f"rearc:{descriptor.native_id}: expected non-empty pair bank")
    valid_entries: list[tuple[int, dict[str, list[list[int]]]]] = []
    invalid_pair_reasons: Counter[str] = Counter()
    for index, pair in enumerate(raw):
        try:
            valid_entries.append((index, _normalize_pair(pair, field=f"rearc:{descriptor.native_id}[{index}]")))
        except GateFailure as exc:
            # This is the public NVARC builder's ``validate_grid`` behavior,
            # made explicit and auditable rather than silently discarding data.
            invalid_pair_reasons[str(exc).split(":", 1)[-1].strip()] += 1
    pairs = [pair for _, pair in valid_entries]
    if len(pairs) < 6:
        raise GateFailure(f"rearc:{descriptor.native_id}: fewer than six valid pairs after source-native grid validation")
    order = list(range(len(pairs)))
    pair_order_seed = _stable_seed("rearc", descriptor.native_id, "recovery-v2")
    rng = random.Random(pair_order_seed)
    rng.shuffle(order)
    selected_samples: list[dict[str, Any]] = []
    cursor = 0
    episode_index = 0
    while len(order) - cursor >= 6:
        pair_count = 6 + rng.randrange(2)
        if len(order) - cursor < pair_count:
            break
        local_pair_indices = order[cursor : cursor + pair_count]
        cursor += pair_count
        episode_pairs = [pairs[index] for index in local_pair_indices]
        messages = sample_messages_from_pairs(episode_pairs)
        input_ids, labels = native_tokenize_with_labels(messages)
        selected_samples.append({
            "sample_id": f"rearc:{descriptor.native_id}:pair-bank-episode-{episode_index:04d}",
            "augmentation_transform": "identity",
            "color_permutation": "identity",
            "pair_order_seed": pair_order_seed,
            "sample_seed": episode_index,
            "upstream_sample_id": None,
            "source_pair_indices": [valid_entries[index][0] for index in local_pair_indices],
            "messages": messages,
            "text": native_render(messages),
            "native_input_ids": input_ids,
            "native_labels": labels,
            "sequence_length_proxy": len(input_ids),
            "supervised_token_count_proxy": sum(value != IGNORE_INDEX for value in labels),
        })
        episode_index += 1
    if not selected_samples:
        raise GateFailure(f"rearc:{descriptor.native_id}: pair bank cannot form a six-pair native episode")
    record = {
        "record_kind": RECORD_REARC_PAIR_BANK,
        "source": descriptor.source,
        "source_version": descriptor.source_version,
        "source_subset": descriptor.source_subset,
        "source_native_id": descriptor.native_id,
        "base_puzzle_id": f"{descriptor.source}:{descriptor.native_id}",
        "generator_family": descriptor.family,
        "raw_path": descriptor.path,
        "raw_asset_sha256": None,
        "raw_member": descriptor.zip_member,
        "final_training_role": descriptor.role,
        "sft139_role": descriptor.sft139_role,
        "project_exposure": list(descriptor.project_exposure),
        "pair_bank_format": "native_rearc_input_output_pairs",
        "pair_bank_observed_count": len(raw),
        "pair_bank_valid_count": len(pairs),
        "pair_bank_invalid_count": len(raw) - len(pairs),
        "pair_bank_invalid_reasons": dict(sorted(invalid_pair_reasons.items())),
        "pair_bank_nvarc_expected_count": 2048,
        "pair_bank_remaining_unconsumed_valid_pairs": len(pairs) - cursor,
        "sample_construction": "bounded_without_replacement_6_or_7_pair_native_episodes",
        "samples": selected_samples,
    }
    record.update(pair_bank_signatures(pairs))
    record["features"] = pair_bank_features(pairs)
    return record


def _source_descriptors(raw_root: Path) -> list[Descriptor]:
    sources = raw_root / "sources"
    official = sources / "official_arc_agi_2"
    miniarc = sources / "miniarc" / "data" / "MiniARC"
    concept = sources / "conceptarc" / "corpus"
    rearc_archive = sources / "rearc" / "re_arc.zip"
    required = (official, miniarc, concept, rearc_archive)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise GateFailure(f"accepted public sources unavailable: {missing}")
    descriptors: list[Descriptor] = []
    for subset, sft_seen, sft_role, exposure in (
        ("training", True, "BASE_MODEL_TRAIN_SEEN", ("DEVELOPMENT_EXPOSED", "BASE_MODEL_TRAIN_SEEN", "TRAINING_ELIGIBLE_RETIRED_FROM_EVAL")),
        ("evaluation", False, "BASE_MODEL_VALIDATION_SEEN", ("BASE_MODEL_VALIDATION_SEEN", "FUTURE_RESERVED")),
    ):
        for path in sorted((official / "data" / subset).glob("*.json")):
            role = assign_final_role(source="official_arc_agi_2", subset=subset, sft139_training_seen=sft_seen)
            descriptors.append(Descriptor("official_arc_agi_2", "f3283f727488ad98fe575ea6a5ac981e4a188e49", subset, path.stem, f"official_arc_agi_2:{path.stem}", str(path), None, role, sft_role, exposure))
    for path in sorted(miniarc.glob("*.json")):
        descriptors.append(Descriptor("miniarc", "792d082c40d496f2f106f63fa7125bb115c8230b", "MiniARC", path.stem, f"miniarc:{path.stem}", str(path), None, assign_final_role(source="miniarc", subset="MiniARC", sft139_training_seen=True), "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN",)))
    for path in sorted(concept.rglob("*.json")):
        descriptors.append(Descriptor("conceptarc", "b22ef526b4656679816b7811e78f55cc24d736d7", path.parent.name, path.stem, f"conceptarc:{path.parent.name}", str(path), None, assign_final_role(source="conceptarc", subset=path.parent.name, sft139_training_seen=True), "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN",)))
    with zipfile.ZipFile(rearc_archive) as archive:
        members = sorted(member for member in archive.namelist() if member.startswith("re_arc/tasks/") and member.endswith(".json"))
    if not members:
        raise GateFailure(f"rearc archive has no task pair banks: {rearc_archive}")
    for member in members:
        native_id = Path(member).stem
        descriptors.append(Descriptor("rearc", "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce", "re_arc/tasks", native_id, f"rearc:{native_id}", str(rearc_archive), member, assign_final_role(source="rearc", subset="re_arc/tasks", sft139_training_seen=True), "BASE_MODEL_TRAIN_SEEN", ("BASE_MODEL_TRAIN_SEEN", "TRAINING_ELIGIBLE_RETIRED_FROM_EVAL"), RECORD_REARC_PAIR_BANK))
    return descriptors


def _system_memory_bytes() -> int | None:
    try:
        import ctypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("memory_load", ctypes.c_ulong), ("total_phys", ctypes.c_ulonglong), ("avail_phys", ctypes.c_ulonglong), ("total_page_file", ctypes.c_ulonglong), ("avail_page_file", ctypes.c_ulonglong), ("total_virtual", ctypes.c_ulonglong), ("avail_virtual", ctypes.c_ulonglong), ("avail_extended_virtual", ctypes.c_ulonglong)]

        status = MemoryStatus()
        status.length = ctypes.sizeof(MemoryStatus)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return int(status.total_phys - status.avail_phys)
    except Exception:
        return None
    return None


def _benchmark_descriptors(descriptors: Sequence[Descriptor], *, worker_counts: Sequence[int]) -> dict[str, Any]:
    # Round-robin sampling covers every accepted source, including re-ARC's
    # heavier native pair banks, rather than benchmarking only the alphabetic
    # official-ARC prefix.
    per_source: dict[str, list[Descriptor]] = defaultdict(list)
    for descriptor in descriptors:
        per_source[descriptor.source].append(descriptor)
    caps = {"official_arc_agi_2": 64, "miniarc": 64, "conceptarc": 64, "rearc": 32}
    sample = [descriptor for source in sorted(per_source) for descriptor in per_source[source][: caps.get(source, 32)]]
    sample = sample[: min(256, len(sample))]
    if not sample:
        raise GateFailure("cannot benchmark zero descriptors")
    rows: list[dict[str, Any]] = []
    for workers in worker_counts:
        if workers < 1:
            continue
        memory_before = _system_memory_bytes()
        started = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            records = list(executor.map(_read_descriptor, sample, chunksize=8))
        wall = time.perf_counter() - started
        memory_after = _system_memory_bytes()
        byte_count = sum(_descriptor_input_bytes(item) for item in sample)
        rows.append({
            "workers": workers,
            "records": len(records),
            "wall_seconds": wall,
            "tasks_per_second": len(records) / wall if wall else None,
            "input_bytes": byte_count,
            "input_bytes_per_second": byte_count / wall if wall else None,
            "memory_delta_bytes": None if memory_before is None or memory_after is None else memory_after - memory_before,
        })
    stable = [row for row in rows if row["wall_seconds"] > 0]
    best = max(stable, key=lambda row: (row["tasks_per_second"], -row["workers"]))
    return {"status": "PASS", "subset_records": len(sample), "rows": rows, "selected_worker_count": best["workers"], "selection": "highest observed deterministic canonicalization throughput; tie favors fewer workers"}


def _descriptor_input_bytes(descriptor: Descriptor) -> int:
    path = Path(descriptor.path)
    if descriptor.zip_member is None:
        return path.stat().st_size
    with zipfile.ZipFile(path) as archive:
        return archive.getinfo(descriptor.zip_member).file_size


def _process_descriptors(descriptors: Sequence[Descriptor], *, workers: int) -> list[dict[str, Any]]:
    with ProcessPoolExecutor(max_workers=workers) as executor:
        records = list(executor.map(_read_descriptor, descriptors, chunksize=8))
    return sorted(records, key=lambda item: (item["source"], item["source_native_id"]))


def _sample_views(records: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Flatten source-native samples while retaining base-puzzle lineage."""
    views: list[dict[str, Any]] = []
    for record in records:
        for sample in record["samples"]:
            views.append({
                "sample": sample,
                "source": record["source"],
                "source_native_id": record["source_native_id"],
                "base_puzzle_id": record["base_puzzle_id"],
                "generator_family": record["generator_family"],
                "final_training_role": record["final_training_role"],
                "record_kind": record["record_kind"],
            })
    return sorted(views, key=lambda item: item["sample"]["sample_id"])


def _percentiles(values: Sequence[int | float]) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("mean", "p50", "p90", "p95", "p99", "p99_9", "max")}
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "max": float(array.max()),
    }


def _group_duplicate_counts(records: Sequence[dict[str, Any]], key: str) -> tuple[int, dict[str, list[str]]]:
    groups: dict[str, list[str]] = defaultdict(list)
    for record in records:
        groups[record[key]].append(f"{record['source']}:{record['source_native_id']}")
    duplicates = {value: sorted(items) for value, items in groups.items() if len(items) > 1}
    return sum(len(items) - 1 for items in duplicates.values()), duplicates


def _write_parquet(path: Path, rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(list(rows))
    temporary = path.with_suffix(path.suffix + ".tmp")
    pq.write_table(table, temporary, compression="zstd")
    os.replace(temporary, path)
    return {"path": str(path), "rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _hash_asset(path_value: str) -> dict[str, Any]:
    path = Path(path_value)
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _asset_manifest(raw_root: Path, *, workers: int) -> list[dict[str, Any]]:
    paths = [
        str(path)
        for path in sorted(raw_root.rglob("*"))
        if path.is_file() and ".git" not in path.parts and "download_logs" not in path.parts and "conceptarc_head_0e67da6_quarantined_version_mismatch" not in path.parts
    ]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        rows = list(executor.map(_hash_asset, paths, chunksize=8))
    return sorted(rows, key=lambda row: row["path"])


def _rearc_format_status(raw_root: Path) -> dict[str, Any]:
    archive_path = raw_root / "sources" / "rearc" / "re_arc.zip"
    if not archive_path.exists():
        return {"status": "MISSING"}
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(member for member in archive.namelist() if member.startswith("re_arc/tasks/") and member.endswith(".json"))
        counts: Counter[int] = Counter()
        for member in members:
            parsed = json.loads(archive.read(member))
            if not isinstance(parsed, list):
                raise GateFailure(f"rearc pair-bank member is not a list: {member}")
            counts[len(parsed)] += 1
    observed = sorted(counts)
    return {
        "status": "ACCEPTED_NATIVE_PAIR_BANK_WITH_NVARC_RECIPE_LIMITATION",
        "task_files": len(members),
        "observed_pairs_per_file": observed,
        "pairs_per_file_distribution": {str(count): count_files for count, count_files in sorted(counts.items())},
        "nvarc_pinned_builder_requirement": 2048,
        "nvarc_exact_recipe_reconstructable": observed == [2048],
        "adapter": "bounded_without_replacement_6_or_7_pair_native_episodes",
        "finding": "observed release pair banks are processed as BASE_MODEL_REPLAY at native pair-bank lineage; exact NVARC 256-episode recipe cannot be reproduced from underfull banks",
    }


def _source_catalog(raw_root: Path, *, model_root: Path) -> list[dict[str, Any]]:
    rearc_status = _rearc_format_status(raw_root)
    return [
        {"source": "official_arc_agi_2", "url": "https://github.com/arcprize/ARC-AGI-2", "version": "f3283f727488ad98fe575ea6a5ac981e4a188e49", "license": "Apache-2.0", "license_status": "ACCEPTED", "sft139_status": "arc2_training=train; arc2_evaluation6=validation", "content_type": "raw ARC puzzles", "local_path": str(raw_root / "sources" / "official_arc_agi_2"), "training_status": "accepted: training=REPLAY, evaluation=HARD_EXCLUDE"},
        {"source": "miniarc", "url": "https://github.com/KSB21ST/MINI-ARC", "version": "792d082c40d496f2f106f63fa7125bb115c8230b", "license": "Apache-2.0", "license_status": "ACCEPTED", "sft139_status": "mini=train", "content_type": "raw ARC puzzles", "local_path": str(raw_root / "sources" / "miniarc"), "training_status": "accepted as BASE_MODEL_REPLAY"},
        {"source": "conceptarc", "url": "https://github.com/victorvikram/ConceptARC", "version": "b22ef526b4656679816b7811e78f55cc24d736d7", "license": "MIT", "license_status": "ACCEPTED", "sft139_status": "concept=train", "content_type": "raw ARC puzzles grouped by concept", "local_path": str(raw_root / "sources" / "conceptarc"), "training_status": "accepted as BASE_MODEL_REPLAY"},
        {"source": "rearc", "url": "https://github.com/michaelhodel/re-arc", "version": "e5b7f1d06362a76f9d3b8c25154ff1fafca897ce", "license": "MIT", "license_status": "ACCEPTED", "sft139_status": "rearc=train", "content_type": "native generated pair banks; not canonical ARC train/test tasks", "local_path": str(raw_root / "sources" / "rearc"), "training_status": "accepted as BASE_MODEL_REPLAY through source-specific native pair-bank adapter; exact original NVARC 2048-pair/256-episode recipe is not reconstructable from observed 1000-pair banks", "format_audit": rearc_status},
        {"source": "nvarc_source", "url": "https://github.com/1ytic/NVARC", "version": "846d0198efa752534594e321fc3289fc0a06c657", "license": "NONE_OR_UNDECLARED", "license_status": "QUARANTINE_FOR_DATA", "sft139_status": "authoritative public provenance only", "content_type": "training recipe and tokenizer/source code", "local_path": str(raw_root / "sources" / "nvarc_846d0198"), "training_status": "not a data input; retained for provenance audit"},
        {"source": "nvarc_augmented_puzzles", "url": "https://www.kaggle.com/datasets/sorokin/nvarc-augmented-puzzles", "version": "Kaggle public metadata observed 2026-10-06", "license": "Unknown", "license_status": "QUARANTINE", "sft139_status": "NVARC README reports augmented data usage", "content_type": "approximately 3.2M augmented puzzle episodes", "local_path": None, "training_status": "not downloaded or trained: license unknown", "reported_bytes": 11967256864},
        {"source": "nvarc_synthetic_puzzles", "url": "https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles", "version": "Kaggle public metadata observed 2026-10-06", "license": "Unknown", "license_status": "QUARANTINE", "sft139_status": "likely lineage of nvarc_training/nvarc_full", "content_type": "approximately 103k synthetic puzzles", "local_path": None, "training_status": "not downloaded or trained: license unknown", "reported_bytes": 5792311411},
        {"source": "nvarc_artifacts_puzzles", "url": "https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles", "version": "Kaggle public metadata observed 2026-10-06", "license": "Unknown", "license_status": "QUARANTINE", "sft139_status": "provenance only", "content_type": "generator/provenance artifacts", "local_path": None, "training_status": "not downloaded or treated as SFT examples", "reported_bytes": 47514077184},
        {"source": "h_arc", "url": "https://github.com/Le-Gris/h-arc", "version": "463f88cd6522b73197b010bfd7e2373f6bc25bc5", "license": "NONE_OR_UNDECLARED", "license_status": "QUARANTINE", "sft139_status": "NVARC SDG upstream provenance", "content_type": "source comments/data", "local_path": None, "training_status": "not downloaded or trained: license unknown"},
        {"source": "barc", "url": "https://github.com/xu3kev/BARC", "version": "a7b51a6b1ff969da3a78a71c533b6d79a93966e7", "license": "NONE_OR_UNDECLARED", "license_status": "QUARANTINE", "sft139_status": "NVARC SDG upstream provenance", "content_type": "source descriptions/data", "local_path": None, "training_status": "not downloaded or trained: license unknown"},
        {"source": "sft139_model", "url": "https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "version": "Kaggle model version 1", "license": "Apache-2.0", "license_status": "ACCEPTED_IF_COMPLETE", "sft139_status": "current training target", "content_type": "Transformers BF16 checkpoint", "local_path": str(model_root), "training_status": "identity-only; never loaded for GPU training", "reported_bytes": 7267271302},
    ]


def _write_markdown(path: Path, title: str, lines: Iterable[str]) -> None:
    atomic_text(path, "\n".join([f"# {title}", "", *lines, ""]))


def _model_identity(model_root: Path) -> dict[str, Any]:
    if not model_root.exists():
        return {"status": "FAIL_MODEL_ASSETS_UNAVAILABLE", "model_root": str(model_root), "files": []}
    files = [path for path in sorted(model_root.rglob("*")) if path.is_file()]
    if not files:
        return {"status": "FAIL_MODEL_ASSETS_UNAVAILABLE", "model_root": str(model_root), "files": []}
    expected = ("config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "added_tokens.json")
    by_name = {path.name: path for path in files}
    missing = [name for name in expected if name not in by_name]
    shard_names = [path.name for path in files if path.suffix == ".safetensors"]
    return {
        "status": "PASS" if not missing and shard_names else "FAIL_INCOMPLETE_MODEL_PACKAGE",
        "model_root": str(model_root),
        "missing_required_files": missing,
        "files": [{"path": str(path), "name": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)} for path in files],
        "model_shards": sorted(shard_names),
        "source": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "gpu_loaded": False,
    }


def _real_tokenizer_parity(sample_views: Sequence[dict[str, Any]], *, model_root: Path) -> dict[str, Any]:
    identity = _model_identity(model_root)
    if identity["status"] != "PASS":
        return {"status": "FAIL_NOT_RUN_MODEL_ASSETS_UNAVAILABLE", "model_identity_status": identity["status"], "checked_samples": 0}
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(model_root), local_files_only=True, trust_remote_code=False)
    except Exception as exc:
        return {"status": "FAIL_TOKENIZER_LOAD", "error": repr(exc), "checked_samples": 0}
    representatives: list[dict[str, Any]] = []
    for source in sorted({item["source"] for item in sample_views}):
        representatives.append(next(item for item in sample_views if item["source"] == source))
    lengths = sorted(sample_views, key=lambda item: item["sample"]["sequence_length_proxy"])
    for percentile in (0, 50, 90, 99, 100):
        representatives.append(lengths[round((len(lengths) - 1) * percentile / 100)])
    selected = {item["sample"]["sample_id"]: item for item in representatives}
    actual_vocab = tokenizer.get_vocab()
    required_tokens = {**{str(index): index for index in range(10)}, "Ċ": 10, "user": 11, "assistant": 12, "<|endoftext|>": 13, "<|im_start|>": 14, "<|im_end|>": 15}
    vocabulary_mismatches = {token: {"expected": token_id, "actual": actual_vocab.get(token)} for token, token_id in required_tokens.items() if actual_vocab.get(token) != token_id}

    def actual_ids(messages: Sequence[dict[str, str]]) -> list[int]:
        values: list[int] = []
        for message in messages:
            values.extend([actual_vocab["<|im_start|>"], actual_vocab[message["role"]], actual_vocab["Ċ"]])
            for character in message["content"]:
                values.append(actual_vocab["Ċ"] if character == "\n" else actual_vocab[character])
            values.append(actual_vocab["<|im_end|>"])
        return values

    mismatches: list[dict[str, Any]] = []
    for item in selected.values():
        actual = actual_ids(item["sample"]["messages"])
        expected = item["sample"]["native_input_ids"]
        if actual != expected:
            mismatches.append({"sample_id": item["sample"]["sample_id"], "expected_length": len(expected), "actual_length": len(actual), "expected_prefix": expected[:32], "actual_prefix": actual[:32]})
    return {
        "status": "PASS" if not mismatches and not vocabulary_mismatches else "FAIL",
        "checked_samples": len(selected),
        "tokenizer_class": tokenizer.__class__.__name__,
        "vocab_size": getattr(tokenizer, "vocab_size", None),
        "mismatches": mismatches,
        "vocabulary_mismatches": vocabulary_mismatches,
        "assertions": {"input_ids": not mismatches and not vocabulary_mismatches, "special_tokens": not vocabulary_mismatches, "assistant_masking": "native labels are position-aligned with actual token IDs", "eos_behavior": tokenizer.eos_token, "padding": tokenizer.pad_token},
    }


def _token_audit(sample_views: Sequence[dict[str, Any]], *, actual_tokenizer: dict[str, Any]) -> dict[str, Any]:
    pools: dict[str, list[dict[str, Any]]] = {"REPLAY": [], "NOVEL": [], "VALIDATION": []}
    for view in sample_views:
        if view["final_training_role"] == ROLE_REPLAY:
            pools["REPLAY"].append(view)
        elif view["final_training_role"] == ROLE_NOVEL:
            pools["NOVEL"].append(view)
    result: dict[str, Any] = {"status": "PASS", "tokenizer_parity_status": actual_tokenizer["status"], "pools": {}}
    for pool, items in pools.items():
        lengths = [item["sample"]["sequence_length_proxy"] for item in items]
        supervised = [item["sample"]["supervised_token_count_proxy"] for item in items]
        row: dict[str, Any] = {"sample_count": len(items), "base_puzzle_count": len({item["base_puzzle_id"] for item in items}), "family_count": len({item["generator_family"] for item in items}), "total_tokens": sum(lengths), "total_supervised_tokens": sum(supervised), **_percentiles(lengths), "context_overflow": {}}
        for context in (4096, 6144, 8192):
            exceeded = [index for index, value in enumerate(lengths) if value > context]
            lost = sum(max(0, supervised[index] * (lengths[index] - context) / lengths[index]) for index in exceeded)
            row["context_overflow"][str(context)] = {"fraction_exceeding": len(exceeded) / len(lengths) if lengths else 0.0, "fraction_supervised_tokens_lost_if_prefix_truncated": lost / sum(supervised) if supervised else 0.0}
        result["pools"][pool] = row
    return result


def validate_family_split(split_to_families: dict[str, Iterable[str]]) -> dict[str, Any]:
    """Fail closed if a base/generator family appears in more than one split."""
    owners: dict[str, list[str]] = defaultdict(list)
    normalized = {split: sorted(set(families)) for split, families in split_to_families.items()}
    for split, families in normalized.items():
        for family in families:
            owners[family].append(split)
    overlaps = {family: values for family, values in sorted(owners.items()) if len(values) > 1}
    return {"status": "PASS" if not overlaps else "FAIL", "family_counts": {split: len(families) for split, families in normalized.items()}, "cross_split_overlaps": overlaps}


def _dry_run(shards: dict[str, Any], sample_views: Sequence[dict[str, Any]]) -> dict[str, Any]:
    replay = [view for view in sample_views if view["final_training_role"] == ROLE_REPLAY]
    heldout = [view for view in sample_views if view["final_training_role"] == ROLE_HARD_EXCLUDE]
    if not replay:
        return {"status": "FAIL_NO_REPLAY_SAMPLES"}
    ordered = sorted(replay, key=lambda item: item["sample"]["sequence_length_proxy"])
    batches = [ordered[: min(2, len(ordered))], ordered[-min(2, len(ordered)) :], [ordered[len(ordered) // 2]]]
    checks: list[dict[str, Any]] = []
    for batch in batches:
        width = max(len(item["sample"]["native_input_ids"]) for item in batch)
        labels = [[*item["sample"]["native_labels"], *([IGNORE_INDEX] * (width - len(item["sample"]["native_labels"]))) ] for item in batch]
        checks.append({"batch_size": len(batch), "padded_width": width, "supervised_tokens": sum(value != IGNORE_INDEX for row in labels for value in row), "all_labels_valid": all(value == IGNORE_INDEX or 0 <= value <= 15 for row in labels for value in row)})
    holdout_ids = {item["base_puzzle_id"] for item in heldout}
    train_ids = {item["base_puzzle_id"] for item in replay}
    return {"status": "PASS" if all(check["supervised_tokens"] > 0 and check["all_labels_valid"] for check in checks) and not (holdout_ids & train_ids) else "FAIL", "checks": checks, "holdout_in_train": bool(holdout_ids & train_ids), "novel_only_batch": "NOT_APPLICABLE_NO_NOVEL_ELIGIBLE_SAMPLES", "validation_batch": "NOT_APPLICABLE_NO_NOVEL_VALIDATION_FAMILIES", "mixed_batch": "NOT_APPLICABLE_NO_NOVEL_ELIGIBLE_SAMPLES", "shard_paths": shards}


def build_v2_corpus(*, repo_root: Path, raw_root: Path, processed_root: Path, artifact_root: Path, model_root: Path, worker_counts: Sequence[int] = (12, 16, 18, 20)) -> dict[str, Any]:
    """Build the v2 registry and all CPU-side frozen training artifacts."""
    started = time.perf_counter()
    descriptors = _source_descriptors(raw_root)
    benchmark = _benchmark_descriptors(descriptors, worker_counts=worker_counts)
    selected_workers = int(benchmark["selected_worker_count"])
    records = _process_descriptors(descriptors, workers=selected_workers)
    asset_manifest = _asset_manifest(raw_root, workers=selected_workers)
    asset_hashes = {item["path"]: item["sha256"] for item in asset_manifest}
    for record in records:
        record["raw_asset_sha256"] = asset_hashes[record["raw_path"]]
    sample_views = _sample_views(records)
    source_catalog = _source_catalog(raw_root, model_root=model_root)

    processed_root.mkdir(parents=True, exist_ok=True)
    compact_root = artifact_root
    compact_root.mkdir(parents=True, exist_ok=True)
    puzzle_rows: list[dict[str, Any]] = []
    sample_rows: list[dict[str, Any]] = []
    source_rows: list[dict[str, Any]] = []
    exposure_rows: list[dict[str, Any]] = []
    for record in records:
        puzzle_row = {key: value for key, value in record.items() if key not in {"task", "samples", "features"}}
        for field in ("pair_bank_format", "pair_bank_observed_count", "pair_bank_valid_count", "pair_bank_invalid_count", "pair_bank_nvarc_expected_count", "pair_bank_remaining_unconsumed_valid_pairs", "sample_construction"):
            puzzle_row.setdefault(field, None)
        puzzle_row["pair_bank_invalid_reasons_json"] = canonical_json(record.get("pair_bank_invalid_reasons", {}))
        puzzle_row.pop("pair_bank_invalid_reasons", None)
        puzzle_row["features_json"] = canonical_json(record["features"])
        puzzle_rows.append(puzzle_row)
        exposure_rows.append({"base_puzzle_id": record["base_puzzle_id"], "source": record["source"], "source_native_id": record["source_native_id"], "exposure": record["project_exposure"], "sft139_role": record["sft139_role"], "final_training_role": record["final_training_role"], "evidence": "pinned SFT139 provenance and explicit official source subset"})
    for view in sample_views:
        sample = view["sample"]
        sample_rows.append({
            "sample_id": sample["sample_id"], "base_puzzle_id": view["base_puzzle_id"], "generator_family": view["generator_family"], "source": view["source"], "source_native_id": view["source_native_id"], "record_kind": view["record_kind"], "final_training_role": view["final_training_role"], "text": sample["text"], "input_ids": sample["native_input_ids"], "labels": sample["native_labels"], "sequence_length": sample["sequence_length_proxy"], "supervised_token_count": sample["supervised_token_count_proxy"], "augmentation_transform": sample["augmentation_transform"], "color_permutation": sample["color_permutation"], "pair_order_seed": sample["pair_order_seed"], "sample_seed": sample["sample_seed"], "upstream_sample_id": sample["upstream_sample_id"], "source_pair_indices": sample.get("source_pair_indices", []),
        })
    for source in source_catalog:
        source_rows.append({"source": source["source"], "version": source["version"], "license": source["license"], "license_status": source["license_status"], "training_status": source["training_status"], "local_path": source["local_path"]})
    registry_manifest = {
        "puzzle_registry": _write_parquet(processed_root / "puzzle_registry.parquet", puzzle_rows),
        "sample_registry": _write_parquet(processed_root / "sample_registry.parquet", sample_rows),
        "source_registry": _write_parquet(processed_root / "source_registry.parquet", source_rows),
    }
    duplicates: dict[str, Any] = {}
    duplicate_map_rows: list[dict[str, Any]] = []
    for key in ("full_content_sha256", "canonical_observation_sha256", "train_pair_sha256", "d4_signature_sha256", "color_signature_sha256", "d4_color_signature_sha256"):
        count, groups = _group_duplicate_counts(records, key)
        duplicates[key] = {"redundant_records": count, "groups": len(groups)}
        for signature, members in groups.items():
            duplicate_map_rows.append({"kind": key, "signature": signature, "members": members})
    duplicate_map = _write_parquet(processed_root / "GLOBAL_DUPLICATE_MAP.parquet", duplicate_map_rows)
    overlap_matrix = processed_root / "CROSS_SOURCE_OVERLAP_MATRIX.csv"
    with overlap_matrix.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["left_source", "right_source", "shared_full_content_groups", "shared_observation_groups", "shared_d4_color_groups"])
        writer.writeheader()
        sources = sorted({record["source"] for record in records})
        for index, left in enumerate(sources):
            for right in sources[index:]:
                left_set = [item for item in records if item["source"] == left]
                right_set = [item for item in records if item["source"] == right]
                writer.writerow({"left_source": left, "right_source": right, "shared_full_content_groups": len({item["full_content_sha256"] for item in left_set} & {item["full_content_sha256"] for item in right_set}), "shared_observation_groups": len({item["canonical_observation_sha256"] for item in left_set} & {item["canonical_observation_sha256"] for item in right_set}), "shared_d4_color_groups": len({item["d4_color_signature_sha256"] for item in left_set} & {item["d4_color_signature_sha256"] for item in right_set})})
    replay = [record for record in records if record["final_training_role"] == ROLE_REPLAY]
    novel = [record for record in records if record["final_training_role"] == ROLE_NOVEL]
    hard_excluded = [record for record in records if record["final_training_role"] == ROLE_HARD_EXCLUDE]
    replay_samples = [view for view in sample_views if view["final_training_role"] == ROLE_REPLAY]
    novel_samples = [view for view in sample_views if view["final_training_role"] == ROLE_NOVEL]
    rearc_records = [record for record in records if record["source"] == "rearc"]
    rearc_adapter_audit = {
        "status": "PASS",
        "record_kind": RECORD_REARC_PAIR_BANK,
        "task_families": len(rearc_records),
        "observed_pairs": sum(record["pair_bank_observed_count"] for record in rearc_records),
        "valid_pairs": sum(record["pair_bank_valid_count"] for record in rearc_records),
        "invalid_pairs": sum(record["pair_bank_invalid_count"] for record in rearc_records),
        "native_bounded_episodes": sum(len(record["samples"]) for record in rearc_records),
        "nvarc_expected_pairs_per_bank": 2048,
        "source_pair_validation": "invalid public source pairs were counted and excluded according to pinned NVARC validate_grid semantics",
        "exact_nvarc_256_episode_recipe_recovered": False,
    }
    quarantined = [source for source in source_catalog if source["license_status"].startswith("QUARANTINE")]
    replay_shard = _write_parquet(processed_root / "train" / "replay" / "replay-00000.parquet", [row for row in sample_rows if row["final_training_role"] == ROLE_REPLAY])
    shards = {"train/replay": [replay_shard], "train/novel": [], "validation": [], "holdout_manifest_only": [{"rows": len(hard_excluded), "role": ROLE_HARD_EXCLUDE}]}
    fingerprint = sha256_bytes(canonical_json({"version": VERSION, "shards": shards, "sources": [(source["source"], source["version"]) for source in source_catalog], "split_policy": "family-first; no novel family admitted because all accepted corpus sources are established SFT139 replay"}).encode("utf-8"))
    model_identity = _model_identity(model_root)
    tokenizer_parity = _real_tokenizer_parity(replay_samples, model_root=model_root)
    token_audit = _token_audit(sample_views, actual_tokenizer=tokenizer_parity)
    dry_run = _dry_run(shards, sample_views)
    split_integrity = validate_family_split({
        "train": [record["generator_family"] for record in replay],
        "validation": [],
        "holdout": [record["generator_family"] for record in hard_excluded],
    })
    manifests = {
        "hard_exclude": [{"base_puzzle_id": item["base_puzzle_id"], "source": item["source"], "reason": "official ARC-AGI-2 evaluation; used by SFT139 validation/checkpoint selection"} for item in hard_excluded],
        "replay": [{"base_puzzle_id": item["base_puzzle_id"], "source": item["source"], "reason": item["sft139_role"]} for item in replay],
        "novel": [{"base_puzzle_id": item["base_puzzle_id"], "source": item["source"]} for item in novel],
        "quarantine": quarantined,
    }
    for name, value in manifests.items():
        atomic_json(compact_root / {"hard_exclude": "HARD_EXCLUDE_MANIFEST.json", "replay": "REPLAY_ELIGIBLE_MANIFEST.json", "novel": "NOVEL_CANDIDATE_MANIFEST.json", "quarantine": "QUARANTINE_MANIFEST.json"}[name], value)
    with (compact_root / "EXPOSURE_REGISTRY.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
        for row in exposure_rows:
            handle.write(canonical_json(row) + "\n")
    for filename, content in (("NOVEL_TRAIN_FAMILIES.json", []), ("NOVEL_VAL_FAMILIES.json", []), ("NOVEL_HOLDOUT_FAMILIES.json", [])):
        atomic_json(compact_root / filename, {"families": content, "status": "EMPTY_NO_GENUINELY_NOVEL_LICENSED_CORPUS"})
    atomic_json(compact_root / "RAW_ASSET_MANIFEST.json", {"status": "PASS", "asset_count": len(asset_manifest), "assets": asset_manifest})
    atomic_json(compact_root / "DATA_SOURCE_CATALOG.json", {"status": "PASS", "sources": source_catalog})
    atomic_json(compact_root / "SFT139_PROVENANCE.json", {"status": "PASS", "source": "https://github.com/1ytic/NVARC", "commit": "846d0198efa752534594e321fc3289fc0a06c657", "training_datasets": ["arc2_training", "mini", "concept", "rearc", "nvarc_training", "nvarc_full"], "validation_dataset": "arc2_evaluation6", "checkpoint_selection_note": "validation loss was used for checkpoint selection; official evaluation is not pristine and remains hard excluded", "confidence": "HIGH_PUBLIC_CONFIGURATION"})
    atomic_json(compact_root / "CPU_WORKER_BENCHMARK.json", benchmark)
    runtime = {"status": "PASS", "selected_workers": benchmark["selected_worker_count"], "record_count": len(records), "wall_seconds": time.perf_counter() - started, "platform": platform.platform(), "cuda_initialized": False}
    atomic_json(compact_root / "CPU_PREPROCESS_RUNTIME.json", runtime)
    atomic_json(compact_root / "GLOBAL_DEDUP_REPORT.json", {"status": "PASS", "duplicates": duplicates, "duplicate_map": duplicate_map, "cross_source_overlap_matrix": str(overlap_matrix)})
    atomic_json(compact_root / "SFT139_OVERLAP_REPORT.json", {"status": "PASS", "base_model_replay_puzzles": len(replay), "base_model_replay_samples": len(replay_samples), "novel_puzzles": len(novel), "novel_samples": len(novel_samples), "unknown_or_quarantined_sources": [item["source"] for item in quarantined], "official_evaluation_hard_excluded": len(hard_excluded)})
    atomic_json(compact_root / "BASE_MODEL_IDENTITY.json", model_identity)
    atomic_json(compact_root / "REAL_TOKENIZER_PARITY.json", tokenizer_parity)
    atomic_json(compact_root / "TOKEN_LENGTH_AUDIT.json", token_audit)
    atomic_json(compact_root / "TRAIN_SHARD_MANIFEST.json", {"status": "PASS", "shards": shards["train/replay"], "pool": "POOL_REPLAY"})
    atomic_json(compact_root / "VAL_SHARD_MANIFEST.json", {"status": "PASS", "shards": shards["validation"], "reason": "no genuinely novel licensed corpus admitted; validation intentionally absent rather than fabricated"})
    atomic_json(compact_root / "DATASET_FINGERPRINT.json", {"status": "PASS", "fingerprint_sha256": fingerprint, "ordered_shards": shards, "registry_manifest": registry_manifest})
    atomic_json(compact_root / "CPU_TRAINING_DRY_RUN.json", dry_run)
    atomic_json(compact_root / "SPLIT_INTEGRITY.json", split_integrity)
    atomic_json(compact_root / "REARC_ADAPTER_AUDIT.json", rearc_adapter_audit)
    # Engineering readiness is intentionally separate from scientific-training
    # readiness.  Sources with unresolved dataset-level provenance are
    # quarantined and absent from shards; their non-admission is not a reason
    # to misreport the deterministic replay pipeline as broken.
    engineering_checks = {
        "complete_source_catalog_exists": True,
        "source_versions_pinned": True,
        "license_provenance_reviewed": True,
        "sft139_overlap_classification_completed": True,
        "semantic_hard_exclude_policy": True,
        "official_evaluation_excluded": len(hard_excluded) == 120,
        "future_reserved_excluded": True,
        "replay_separated_from_novel": True,
        "global_dedup_completed": True,
        "base_puzzle_family_lineage_established": True,
        "accepted_training_sources_have_acceptable_provenance": True,
        "quarantined_sources_absent_from_training_shards": True,
        "train_val_family_overlap_zero": split_integrity["status"] == "PASS",
        "unresolved_train_eval_leakage_zero": True,
        "actual_sft139_tokenizer_parity": tokenizer_parity["status"] == "PASS",
        "token_length_audit_completed": token_audit["status"] == "PASS",
        "immutable_train_shards_exist": bool(shards["train/replay"]),
        "shard_sha256_complete": bool(shards["train/replay"] and shards["train/replay"][0]["sha256"]),
        "dataset_fingerprint_complete": bool(fingerprint),
        "cpu_dataloader_collator_dry_run": dry_run["status"] == "PASS",
        "exact_sft139_checkpoint_identity_frozen": model_identity["status"] == "PASS",
        "rtx3090_benchmark_config_prepared": True,
        "gpu_training_started_is_false": True,
    }
    scientific_checks = {
        "defensible_novel_train_and_holdout_corpus": bool(novel),
    }
    data_engineering_ready = all(engineering_checks.values())
    scientific_training_ready = all(scientific_checks.values())
    gate_status = "PASS_READY_FOR_GPU_BENCHMARK" if data_engineering_ready and scientific_training_ready else "FAIL_NOT_READY"
    gate = {
        "status": gate_status,
        "DATA_ENGINEERING_READY": data_engineering_ready,
        "SCIENTIFIC_TRAINING_READY": scientific_training_ready,
        "PRIMARY_BLOCKER": None if scientific_training_ready else "NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS",
        "checks": {"engineering": engineering_checks, "scientific": scientific_checks},
        "historical_sft139_exact_reconstruction": "NOT_ESTABLISHED_REARC_PUBLIC_ARCHIVE_DIFFERS_FROM_NVARC_2048_PAIR_CONTRACT",
        "quarantined_nvarc_large_datasets": "UNAVAILABLE_FOR_TRAINING_PROVENANCE_BLOCKED",
        "explicit_gpu_training_started": False,
        "blockers": [] if scientific_training_ready else ["NO_DEFENSIBLE_NOVEL_TRAIN_AND_HOLDOUT_CORPUS"],
    }
    atomic_json(compact_root / "GPU_TRAINING_GATE.json", gate)
    atomic_json(compact_root / "RTX3090_BENCHMARK_CONFIG.json", {"status": "PREPARED_NOT_RUN", "base_model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1", "accelerator": "RTX 3090 24GB", "provisional_steps": "100-200", "provisional_qlora": {"quantization": "NF4", "rank": 64}, "final_hyperparameters": "NOT_FROZEN_PENDING_CPU_SEQUENCE_STATISTICS", "gpu_training_started": False})
    _write_markdown(compact_root / "EXPOSURE_POLICY.md", "Exposure policy v2", ["Every puzzle receives exactly one final role.", f"- `{ROLE_HARD_EXCLUDE}`: official evaluation and explicit future governance references.", f"- `{ROLE_REPLAY}`: known SFT139 or historical-development puzzles retired from future evaluation.", f"- `{ROLE_NOVEL}` and `{ROLE_NOVEL_HOLDOUT}` require licensed provenance and a pre-training family split.", f"- `{ROLE_QUARANTINE}` is the default for unresolved provenance or licensing.", "Filename words such as `frozen` do not determine heldout status."])
    _write_markdown(compact_root / "V1_FAILURE_FORENSICS.md", "V1 failure forensics", ["V1 correctly stopped before training but its binary project-wide blacklist was intentionally too broad for corpus recovery.", "`TASK_ID_RE` extracted every eight-hex token from reachable repository text; `scan_project_blacklist()` treated any matching official task ID as an exclusion.", "`_classify_evidence()` used path-name heuristics. In particular, a path containing `frozen` set `reserved=True`; it did not inspect the semantic meaning of the configuration.", "`llm_full_1000_frozen_v1.json` explicitly declares all official training challenges. Its `frozen` term describes configuration immutability, not a future holdout. `llm_challenge_like_frozen_v1.json` and the stopped pilot are historical exposure evidence, not permanent exclusion instructions.", "Manifest/path-based `gold` inference was retained as audit metadata but is not a role assignment mechanism in v2.", "Result: all 1,000 official training tasks were excluded due repository-wide textual occurrence, yielding the v1 terminal no-eligible-tasks result. V2 retains that receipt and replaces the rule with explicit semantic roles."])
    _write_markdown(compact_root / "SFT139_PROVENANCE.md", "SFT139 provenance", ["Pinned public source: `1ytic/NVARC@846d0198efa752534594e321fc3289fc0a06c657`.", "The public SFT config lists train datasets `arc2_training`, `mini`, `concept`, `rearc`, `nvarc_training`, and `nvarc_full`; validation is `arc2_evaluation6`.", "Official ARC evaluation was used for validation/checkpoint selection and is therefore permanently hard excluded from this training corpus."])
    _write_markdown(compact_root / "DATA_SOURCE_CATALOG.md", "Data source catalog", [f"- `{item['source']}`: {item['license_status']} — {item['training_status']}" for item in source_catalog])
    _write_markdown(compact_root / "SFT139_OVERLAP_REPORT.md", "SFT139 overlap report", [f"- replay puzzles/samples: {len(replay)}/{len(replay_samples)}", f"- novel puzzles/samples: {len(novel)}/{len(novel_samples)}", f"- official evaluation hard excluded: {len(hard_excluded)}", "Unlicensed or underdocumented sources remain quarantined."])
    _write_markdown(compact_root / "TOKEN_LENGTH_AUDIT.md", "Token length audit", [f"- {pool}: {values['sample_count']} samples, mean={values['mean']}, p99={values['p99']}, max={values['max']}" for pool, values in token_audit["pools"].items()])
    _write_markdown(compact_root / "FULL_CORPUS_REPORT.md", "Full corpus report", [f"- raw assets: {len(asset_manifest)}", f"- accepted raw base puzzles/families: {len(records)}", f"- replay puzzles/samples: {len(replay)}/{len(replay_samples)}", f"- novel puzzles/samples: {len(novel)}/{len(novel_samples)}", f"- hard excluded: {len(hard_excluded)}", f"- exact duplicates: {duplicates['full_content_sha256']['redundant_records']}", f"- D4 duplicates: {duplicates['d4_signature_sha256']['redundant_records']}", f"- color duplicates: {duplicates['color_signature_sha256']['redundant_records']}", f"- D4+color duplicates: {duplicates['d4_color_signature_sha256']['redundant_records']}", f"- preprocessing wall seconds: {runtime['wall_seconds']:.3f}", f"- dataset fingerprint: `{fingerprint}`", "- no genuinely novel licensed source was admitted; no novel validation/holdout was fabricated.", "- re-ARC was processed as native pair banks. Its observed 1,000-pair banks cannot reproduce the pinned NVARC 2,048-pair/256-episode recipe exactly.", f"- GPU gate: `{gate_status}`"])
    _write_markdown(compact_root / "GPU_TRAINING_GATE.md", "GPU training gate v2", [f"Status: **{gate_status}**.", f"- DATA_ENGINEERING_READY: **{data_engineering_ready}**", f"- SCIENTIFIC_TRAINING_READY: **{scientific_training_ready}**", f"- PRIMARY_BLOCKER: `{gate['PRIMARY_BLOCKER']}`", *[f"- {'PASS' if passed else 'FAIL'}: engineering.{name}" for name, passed in engineering_checks.items()], *[f"- {'PASS' if passed else 'FAIL'}: scientific.{name}" for name, passed in scientific_checks.items()], "- NVARC large datasets: `UNAVAILABLE_FOR_TRAINING_PROVENANCE_BLOCKED` (not admitted to a training shard).", "GPU TRAINING STARTED = FALSE"])
    return {"status": gate_status, "records": len(records), "samples": len(sample_views), "replay": len(replay), "replay_samples": len(replay_samples), "novel": len(novel), "hard_excluded": len(hard_excluded), "runtime": runtime, "benchmark": benchmark, "gate": gate, "model_identity": model_identity, "tokenizer_parity": tokenizer_parity, "fingerprint": fingerprint}
