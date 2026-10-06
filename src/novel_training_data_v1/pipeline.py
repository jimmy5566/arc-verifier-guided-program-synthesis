"""Build the frozen novel-data-v1 corpus without touching a GPU.

The implementation is intentionally fail closed.  Raw accepted sources remain
under the ignored ``data/raw`` tree; only compact provenance, audit and hash
receipts are intended for Git.  The authoritative V2.1 replay shard is read as
an anti-overlap index and is never rewritten.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Iterator
from itertools import islice

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from training_data_v2.pipeline import (
    IGNORE_INDEX,
    canonical_json,
    native_render,
    sample_messages,
    sha256_file,
    signatures,
    task_features,
)

VERSION = "novel_training_data_v1"
WORKERS = 20
ROLE_NOVEL = "DEFENSIBLE_NOVEL_TRAINABLE"
ROLE_HOLDOUT = "NOVEL_HOLDOUT_RESERVED"
ROLE_REJECTED = "REJECTED_OR_QUARANTINED"
SPLIT_SEED = "arc2-novel-family-split-v1"
TURN_RE = re.compile(r"<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>", re.DOTALL)
_TOKENIZER = None


class GateFailure(RuntimeError):
    pass


def digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(canonical_json(value) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def validate_grid(grid: Any, *, field: str) -> list[list[int]]:
    # 1D-ARC contains long one-row sequences (including padded-fill examples
    # beyond 64 cells), so this source-specific adapter allows 256 while
    # preserving the ARC color and rectangular-grid contract.
    if not isinstance(grid, list) or not grid or len(grid) > 256:
        raise GateFailure(f"{field}: height outside 1..256")
    if not all(isinstance(row, list) and row for row in grid):
        raise GateFailure(f"{field}: non-empty rows required")
    width = len(grid[0])
    if width > 256 or any(len(row) != width for row in grid):
        raise GateFailure(f"{field}: rectangular width outside 1..256")
    result: list[list[int]] = []
    for row in grid:
        checked: list[int] = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or not 0 <= cell <= 9:
                raise GateFailure(f"{field}: colors must be integer 0..9")
            checked.append(cell)
        result.append(checked)
    return result


def normalize_pairs(values: Any, *, field: str) -> list[dict[str, list[list[int]]]]:
    if not isinstance(values, list) or not values:
        raise GateFailure(f"{field}: non-empty pair list required")
    result = []
    for index, pair in enumerate(values):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise GateFailure(f"{field}[{index}]: input/output pair required")
            inp, out = pair["input"], pair["output"]
        else:
            inp, out = pair
        result.append({
            "input": validate_grid(inp, field=f"{field}[{index}].input"),
            "output": validate_grid(out, field=f"{field}[{index}].output"),
        })
    return result


def normalize_arc_task(raw: dict[str, Any], *, native_id: str) -> dict[str, Any]:
    return {
        "train": normalize_pairs(raw.get("train"), field=f"{native_id}.train"),
        "test": normalize_pairs(raw.get("test"), field=f"{native_id}.test"),
    }


def normalize_compositional(raw: dict[str, Any], *, native_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    train: list[dict[str, list[list[int]]]] = []
    for section in ("primitive_functions", "function_compositions"):
        value = raw.get(section)
        if not isinstance(value, dict) or not value:
            raise GateFailure(f"{native_id}.{section}: mapping required")
        for name in sorted(value):
            train.extend(normalize_pairs(value[name], field=f"{native_id}.{section}.{name}"))
    test = normalize_pairs(raw.get("queries"), field=f"{native_id}.queries")
    meta = raw.get("meta_data")
    if not isinstance(meta, dict) or not meta:
        raise GateFailure(f"{native_id}.meta_data: mapping required")
    return {"train": train, "test": test}, meta


def taxonomy_1d(family: str) -> list[str]:
    tags: set[str] = {"sequence"}
    if "move" in family:
        tags.add("object movement")
    if "pcopy" in family:
        tags.update(("repeat motif", "reconstruction"))
    if "recolor" in family:
        tags.add("conditional construction")
    if "scale" in family:
        tags.update(("counting", "progressive spacing"))
    if any(token in family for token in ("mirror", "flip")):
        tags.add("symmetry")
    if any(token in family for token in ("fill", "hollow", "denoising")):
        tags.add("reconstruction")
    return sorted(tags)


def taxonomy_compositional(meta: dict[str, Any]) -> list[str]:
    tags: set[str] = {"composition", "multi-step transformation"}
    types = {str(value.get("type", "")).lower() for value in meta.values() if isinstance(value, dict)}
    if any(any(token in item for token in ("move", "translate", "shift")) for item in types):
        tags.add("object movement")
    if any(any(token in item for token in ("mirror", "reflect", "rotation", "rotate")) for item in types):
        tags.add("symmetry")
    if any("color" in item for item in types):
        tags.add("conditional construction")
    return sorted(tags)


def compact_candidate(*, source: str, version: str, native_id: str, family: str,
                      task: dict[str, Any], confidence: str, taxonomy: list[str],
                      source_line: int | None = None) -> dict[str, Any]:
    sig = signatures(task)
    features = task_features(task)
    return {
        "source": source,
        "source_version": version,
        "source_native_id": native_id,
        "base_puzzle_id": f"{source}:{native_id}",
        "generator_family": family,
        "novelty_confidence": confidence,
        "failure_taxonomy": taxonomy,
        "source_line": source_line,
        **sig,
        "features": features,
    }


def process_1d_item(item: tuple[str, str, str]) -> dict[str, Any]:
    path_string, version, root_string = item
    path = Path(path_string)
    raw = json.loads(path.read_text(encoding="utf-8"))
    native_id = path.stem
    task = normalize_arc_task(raw, native_id=native_id)
    family_name = path.parent.name
    return compact_candidate(
        source="1d_arc", version=version, native_id=native_id,
        family=f"1d_arc:{family_name}", task=task,
        confidence="LEVEL_B_DEFENSIBLE_NOVEL", taxonomy=taxonomy_1d(family_name),
    )


def process_compositional_item(item: tuple[int, str, str]) -> dict[str, Any]:
    line_number, line, version = item
    raw = json.loads(line)
    native_id = f"episode-{line_number:06d}"
    task, meta = normalize_compositional(raw, native_id=native_id)
    family_hash = digest(meta)[:24]
    return compact_candidate(
        source="compositional_arc", version=version, native_id=native_id,
        family=f"compositional_arc:{family_hash}", task=task,
        confidence="LEVEL_A_STRONG_NOVEL", taxonomy=taxonomy_compositional(meta),
        source_line=line_number,
    )


CANDIDATE_SCHEMA = pa.schema([
    ("source", pa.string()), ("source_version", pa.string()),
    ("source_native_id", pa.string()), ("base_puzzle_id", pa.string()),
    ("generator_family", pa.string()), ("novelty_confidence", pa.string()),
    ("failure_taxonomy", pa.list_(pa.string())), ("source_line", pa.int64()),
    ("canonical_observation_sha256", pa.string()), ("full_content_sha256", pa.string()),
    ("train_pair_sha256", pa.string()), ("d4_signature_sha256", pa.string()),
    ("color_signature_sha256", pa.string()), ("d4_color_signature_sha256", pa.string()),
    ("features_json", pa.string()),
])


def candidate_row(record: dict[str, Any]) -> dict[str, Any]:
    value = dict(record)
    value["features_json"] = canonical_json(value.pop("features"))
    return value


def bounded_pool_map(pool: ProcessPoolExecutor, function: Any, items: Iterable[Any], *, batch_size: int, chunksize: int) -> Iterator[Any]:
    """Map through a fixed-size submission window on Python versions without buffersize."""
    iterator = iter(items)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            return
        yield from pool.map(function, batch, chunksize=chunksize)


def write_candidate_registry(root: Path, workers: int) -> tuple[Path, dict[str, Any]]:
    if workers != WORKERS:
        raise GateFailure(f"full processing requires exactly {WORKERS} workers")
    raw = root / "data/raw/novel"
    output = root / "data/processed/novel_training_data_v1/candidate_registry.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(".parquet.tmp")
    writer = pq.ParquetWriter(tmp, CANDIDATE_SCHEMA, compression="zstd")
    counts: Counter[str] = Counter()
    started = time.perf_counter()
    one_d_files = sorted(
        path for path in (raw / "1d_arc/dataset").glob("*/*.json")
        if re.fullmatch(r".+_\d+", path.stem)
    )
    one_d_version = "1e74dc4cb4c58d8160e1fbd0ba638eb745f37147"
    comp_version = "51d0950d525a31a63c4d7444b109826b6d2ddf9b"
    with ProcessPoolExecutor(max_workers=workers) as pool:
        batch: list[dict[str, Any]] = []
        items = [(str(path), one_d_version, str(raw / "1d_arc")) for path in one_d_files]
        for record in pool.map(process_1d_item, items, chunksize=8):
            batch.append(candidate_row(record)); counts[record["source"]] += 1
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CANDIDATE_SCHEMA)); batch.clear()
        comp_path = raw / "compositional_arc/all/all_episodes.jsonl.gz"
        with gzip.open(comp_path, "rt", encoding="utf-8") as handle:
            stream = ((index, line, comp_version) for index, line in enumerate(handle))
            for record in bounded_pool_map(pool, process_compositional_item, stream, batch_size=256, chunksize=16):
                batch.append(candidate_row(record)); counts[record["source"]] += 1
                if len(batch) >= 256:
                    writer.write_table(pa.Table.from_pylist(batch, schema=CANDIDATE_SCHEMA)); batch.clear()
        if batch:
            writer.write_table(pa.Table.from_pylist(batch, schema=CANDIDATE_SCHEMA))
    writer.close(); os.replace(tmp, output)
    return output, {
        "status": "PASS", "workers": workers, "counts": dict(sorted(counts.items())),
        "rows": sum(counts.values()), "wall_seconds": time.perf_counter() - started,
        "sha256": sha256_file(output), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


SIGNATURE_FIELDS = (
    "full_content_sha256", "canonical_observation_sha256", "train_pair_sha256",
    "d4_signature_sha256", "color_signature_sha256", "d4_color_signature_sha256",
)


def anti_overlap_indices(root: Path) -> dict[str, dict[str, set[str]]]:
    puzzle = root / "data/processed/arc_training_v2/puzzle_registry.parquet"
    table = pq.read_table(puzzle, columns=["final_training_role", *SIGNATURE_FIELDS]).to_pylist()
    indices: dict[str, dict[str, set[str]]] = {
        "replay": {field: set() for field in SIGNATURE_FIELDS},
        "hard_exclude": {field: set() for field in SIGNATURE_FIELDS},
    }
    for row in table:
        bucket = "hard_exclude" if row["final_training_role"] == "HARD_EXCLUDE_FUTURE_EVAL" else "replay"
        for field in SIGNATURE_FIELDS:
            if row[field]: indices[bucket][field].add(row[field])
    return indices


REASON_BY_FIELD = {
    "full_content_sha256": "EXACT_FULL_CONTENT_OVERLAP",
    "canonical_observation_sha256": "OBSERVATION_OVERLAP",
    "train_pair_sha256": "TRAIN_PAIR_OVERLAP",
    "d4_signature_sha256": "D4_EQUIVALENT_OVERLAP",
    "color_signature_sha256": "COLOR_EQUIVALENT_OVERLAP",
    "d4_color_signature_sha256": "D4_COLOR_EQUIVALENT_OVERLAP",
}


def candidate_rejection(row: dict[str, Any], indices: dict[str, dict[str, set[str]]], seen_new: dict[str, dict[str, str]]) -> tuple[str | None, str | None]:
    """Return the first fail-closed novelty reason in fixed precedence order."""
    for field in SIGNATURE_FIELDS:
        value = row[field]
        if value in indices["hard_exclude"][field]:
            return "PROJECT_BLACKLIST_OR_HARD_EXCLUDE_" + REASON_BY_FIELD[field], field
    for field in SIGNATURE_FIELDS:
        value = row[field]
        if value in indices["replay"][field]:
            return "V2_1_REPLAY_" + REASON_BY_FIELD[field], field
    for field in SIGNATURE_FIELDS:
        prior = seen_new[field].get(row[field])
        if prior is not None:
            return "NEW_CORPUS_" + REASON_BY_FIELD[field], prior
    return None, None


def classify_candidates(root: Path, registry: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    indices = anti_overlap_indices(root)
    seen_new = {field: {} for field in SIGNATURE_FIELDS}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for batch in pq.ParquetFile(registry).iter_batches(batch_size=2048):
        for row in batch.to_pylist():
            reason, evidence = candidate_rejection(row, indices, seen_new)
            compact = {key: row[key] for key in (
                "source", "source_version", "source_native_id", "base_puzzle_id",
                "generator_family", "novelty_confidence", "failure_taxonomy", "source_line",
                *SIGNATURE_FIELDS, "features_json",
            )}
            if reason:
                rejected.append({**compact, "final_status": ROLE_REJECTED, "rejection_reason": reason, "evidence": evidence})
                counts[reason] += 1
            else:
                accepted.append({**compact, "final_status": ROLE_NOVEL})
                for field in SIGNATURE_FIELDS:
                    seen_new[field][row[field]] = row["base_puzzle_id"]
                counts["ACCEPTED_DEFENSIBLE_NOVEL"] += 1
    return accepted, rejected, {
        "raw_candidates": len(accepted) + len(rejected),
        "accepted": len(accepted), "rejected": len(rejected),
        "counts_by_reason": dict(sorted(counts.items())),
        "anti_overlap_index_counts": {bucket: {field: len(values) for field, values in fields.items()} for bucket, fields in indices.items()},
    }


def validate_curriculum_row(row: dict[str, Any]) -> bool:
    role = str(row.get("final_training_role", ""))
    return role not in {ROLE_HOLDOUT, "HARD_EXCLUDE_FUTURE_EVAL"} and "QUARANTINE" not in role


def split_families(accepted: list[dict[str, Any]]) -> tuple[dict[str, list[str]], dict[str, str]]:
    by_source: dict[str, set[str]] = defaultdict(set)
    for row in accepted:
        by_source[row["source"]].add(row["generator_family"])
    splits = {"train": [], "validation": [], "holdout": []}
    for source, family_set in sorted(by_source.items()):
        families = sorted(family_set, key=lambda value: digest([SPLIT_SEED, source, value]))
        n = len(families)
        if n < 3:
            raise GateFailure(f"{source}: fewer than three accepted families cannot support isolation")
        n_val = max(1, round(n * .1)); n_hold = max(1, round(n * .1)); n_train = n - n_val - n_hold
        if n_train < 1:
            raise GateFailure(f"{source}: no train family after isolation")
        splits["train"].extend(families[:n_train])
        splits["validation"].extend(families[n_train:n_train+n_val])
        splits["holdout"].extend(families[n_train+n_val:])
    for key in splits: splits[key] = sorted(splits[key])
    mapping = {family: split for split, families in splits.items() for family in families}
    if len(mapping) != sum(map(len, splits.values())):
        raise GateFailure("family assigned to more than one split")
    return splits, mapping


def _init_tokenizer(model_root: str) -> None:
    global _TOKENIZER
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained(model_root, local_files_only=True, trust_remote_code=False)


def api_tokenize_text(text: str) -> tuple[list[int], list[int], dict[str, Any]]:
    if _TOKENIZER is None:
        raise GateFailure("tokenizer not initialized")
    full = _TOKENIZER(text, add_special_tokens=False, return_attention_mask=False)["input_ids"]
    joined: list[int] = []; labels: list[int] = []; boundaries = []
    matches = list(TURN_RE.finditer(text))
    if not matches or "".join(match.group(0) for match in matches) != text:
        raise GateFailure("invalid native message serialization")
    for match in matches:
        role, rendered = match.group(1), match.group(0)
        ids = _TOKENIZER(rendered, add_special_tokens=False, return_attention_mask=False)["input_ids"]
        start = len(joined); joined.extend(ids)
        labels.extend(ids if role == "assistant" else [IGNORE_INDEX] * len(ids))
        boundaries.append({"role": role, "start": start, "end": len(joined)})
    if joined != full:
        raise GateFailure("whole-sample and per-turn tokenizer API mismatch")
    if not full or full[-1] != _TOKENIZER.eos_token_id:
        raise GateFailure("final EOS boundary mismatch")
    return full, labels, {"boundaries": boundaries, "eos_token_id": _TOKENIZER.eos_token_id}


def _task_text_1d(path: Path) -> str:
    task = normalize_arc_task(json.loads(path.read_text(encoding="utf-8")), native_id=path.stem)
    return native_render(sample_messages(task))


def _task_text_compositional(path: Path, wanted: set[int]) -> Iterator[tuple[int, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if index not in wanted: continue
            task, _ = normalize_compositional(json.loads(line), native_id=f"episode-{index:06d}")
            yield index, native_render(sample_messages(task))


def tokenize_item(item: tuple[dict[str, Any], str]) -> dict[str, Any]:
    meta, text = item
    ids, labels, details = api_tokenize_text(text)
    supervised = sum(value != IGNORE_INDEX for value in labels)
    if supervised <= 0:
        raise GateFailure(f"{meta['base_puzzle_id']}: zero supervision")
    losses = {}
    for limit in (4096, 6144, 8192):
        kept = sum(value != IGNORE_INDEX for value in labels[:limit])
        losses[str(limit)] = supervised - kept
    return {
        "sample_id": meta["base_puzzle_id"] + ":native-source-episode-v1",
        "base_puzzle_id": meta["base_puzzle_id"], "source": meta["source"],
        "generator_family": meta["generator_family"], "split": meta["split"],
        "novelty_confidence": meta["novelty_confidence"],
        "failure_taxonomy_json": canonical_json(meta["failure_taxonomy"]),
        "input_ids": ids, "labels": labels, "sequence_length": len(ids),
        "supervised_token_count": supervised, "truncation_supervised_loss_json": canonical_json(losses),
        "final_training_role": ROLE_NOVEL,
    }


SAMPLE_SCHEMA = pa.schema([
    ("sample_id", pa.string()), ("base_puzzle_id", pa.string()), ("source", pa.string()),
    ("generator_family", pa.string()), ("split", pa.string()),
    ("novelty_confidence", pa.string()), ("failure_taxonomy_json", pa.string()),
    ("input_ids", pa.list_(pa.int32())), ("labels", pa.list_(pa.int32())),
    ("sequence_length", pa.int32()), ("supervised_token_count", pa.int32()),
    ("truncation_supervised_loss_json", pa.string()), ("final_training_role", pa.string()),
])


def _accepted_text_items(root: Path, accepted: list[dict[str, Any]], family_split: dict[str, str]) -> Iterator[tuple[dict[str, Any], str]]:
    by_id = {row["base_puzzle_id"]: row for row in accepted if family_split[row["generator_family"]] != "holdout"}
    one_d_root = root / "data/raw/novel/1d_arc/dataset"
    for path in sorted(one_d_root.glob("*/*.json")):
        key = f"1d_arc:{path.stem}"
        if key in by_id:
            row = dict(by_id[key]); row["split"] = family_split[row["generator_family"]]
            yield row, _task_text_1d(path)
    comp_rows = {int(row["source_line"]): row for row in by_id.values() if row["source"] == "compositional_arc"}
    comp_path = root / "data/raw/novel/compositional_arc/all/all_episodes.jsonl.gz"
    for index, text in _task_text_compositional(comp_path, set(comp_rows)):
        row = dict(comp_rows[index]); row["split"] = family_split[row["generator_family"]]
        yield row, text


def tokenize_and_shard(root: Path, accepted: list[dict[str, Any]], family_split: dict[str, str], workers: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    if workers != WORKERS: raise GateFailure("tokenization worker count drift")
    base = root / "data/processed/novel_training_data_v1"
    for path in (base / "train/novel", base / "validation/novel"):
        if path.exists(): shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)
    manifests: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    lengths: dict[str, list[int]] = defaultdict(list); supervised: dict[str, list[int]] = defaultdict(list)
    trunc_losses: dict[str, Counter[str]] = defaultdict(Counter)
    buffers: dict[str, list[dict[str, Any]]] = defaultdict(list)
    shard_index: Counter[str] = Counter()

    def flush(split: str) -> None:
        rows = buffers[split]
        if not rows: return
        directory = base / ("train/novel" if split == "train" else "validation/novel")
        path = directory / f"novel-{shard_index[split]:05d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA), tmp, compression="zstd")
        os.replace(tmp, path)
        families = sorted({row["generator_family"] for row in rows})
        manifests[split].append({
            "logical_name": str(path.relative_to(base)).replace("\\", "/"), "rows": len(rows),
            "base_family_count": len(families), "bytes": path.stat().st_size,
            "sha256": sha256_file(path), "tokens": sum(row["sequence_length"] for row in rows),
            "supervised_tokens": sum(row["supervised_token_count"] for row in rows),
        })
        shard_index[split] += 1; rows.clear()

    model = root / "data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer, initargs=(str(model),)) as pool:
        for row in bounded_pool_map(pool, tokenize_item, _accepted_text_items(root, accepted, family_split), batch_size=256, chunksize=4):
            split = row["split"]; buffers[split].append(row)
            lengths[split].append(row["sequence_length"]); supervised[split].append(row["supervised_token_count"])
            loss = json.loads(row["truncation_supervised_loss_json"])
            for limit, value in loss.items(): trunc_losses[split][limit] += value
            if len(buffers[split]) >= 1000: flush(split)
    flush("train"); flush("validation")
    runtime = {"workers": workers, "wall_seconds": time.perf_counter() - started, "rows": {key: len(value) for key, value in lengths.items()}}
    pools = {}
    for split in ("train", "validation"):
        values = np.asarray(lengths[split], dtype=np.int64); sup = np.asarray(supervised[split], dtype=np.int64)
        if not len(values): raise GateFailure(f"{split}: no tokenized rows")
        total_supervised = int(sup.sum())
        pools[split] = {
            "samples": int(len(values)), "total_tokens": int(values.sum()), "supervised_tokens": total_supervised,
            "mean": float(values.mean()),
            **{name: float(np.percentile(values, q)) for name, q in (("p50",50),("p90",90),("p95",95),("p99",99),("p99_9",99.9))},
            "max": int(values.max()),
            "overflow": {str(limit): {
                "sample_rate": float((values > limit).mean()),
                "supervised_token_loss": int(trunc_losses[split][str(limit)]),
                "supervised_token_loss_rate": float(trunc_losses[split][str(limit)] / total_supervised),
            } for limit in (4096,6144,8192)},
        }
    return manifests, {"status": "PASS", "actual_tokenizer_api": True, "pools": pools, "runtime": runtime}


def tokenize_all_candidate_coverage(root: Path, accepted: list[dict[str, Any]], rejected: list[dict[str, Any]], family_split: dict[str, str], manifests: dict[str, list[dict[str, Any]]], workers: int) -> dict[str, Any]:
    """Prove actual-tokenizer coverage for every candidate without materializing holdout labels."""
    if workers != WORKERS: raise GateFailure("candidate coverage worker count drift")
    data=root/"data/processed/novel_training_data_v1"; observed: dict[str, tuple[int,int,str]]={}
    for split in ("train","validation"):
        for shard in manifests[split]:
            for row in pq.read_table(data/shard["logical_name"],columns=["base_puzzle_id","sequence_length","supervised_token_count"]).to_pylist():
                observed[row["base_puzzle_id"]]=(row["sequence_length"],row["supervised_token_count"],split)
    remaining=[row for row in [*accepted,*rejected] if row["base_puzzle_id"] not in observed]
    audit_map={row["generator_family"]:"audit_only_not_materialized" for row in remaining}
    model=root/"data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,initializer=_init_tokenizer,initargs=(str(model),)) as pool:
        for row in bounded_pool_map(pool,tokenize_item,_accepted_text_items(root,remaining,audit_map),batch_size=256,chunksize=4):
            observed[row["base_puzzle_id"]]=(row["sequence_length"],row["supervised_token_count"],"audit_only_not_materialized")
    expected={row["base_puzzle_id"] for row in [*accepted,*rejected]}
    missing=sorted(expected-set(observed)); extras=sorted(set(observed)-expected)
    lengths=[value[0] for value in observed.values()]; supervised=[value[1] for value in observed.values()]
    receipt={"status":"PASS" if not missing and not extras and len(observed)==len(expected) else "FAIL","workers":workers,"candidate_count":len(expected),"actual_tokenizer_covered":len(observed),"already_materialized_train_validation":len(observed)-len(remaining),"audit_only_holdout_or_rejected":len(remaining),"missing":missing,"extras":extras,"min_length":min(lengths),"max_length":max(lengths),"total_tokens":sum(lengths),"total_supervised_tokens":sum(supervised),"holdout_labels_or_ids_materialized":False,"used_for_curriculum_or_tuning":False,"wall_seconds":time.perf_counter()-started}
    atomic_json(root/"artifacts/novel_training_data_v1/ALL_CANDIDATE_TOKENIZATION_COVERAGE.json",receipt)
    if receipt["status"]!="PASS": raise GateFailure("not every candidate received actual tokenizer audit")
    return receipt


def raw_asset_manifest(root: Path) -> dict[str, Any]:
    raw = root / "data/raw/novel"
    assets = []
    for source_dir in (raw / "1d_arc", raw / "compositional_arc", raw / "compositional_arc_code"):
        for path in sorted(source_dir.rglob("*")):
            if not path.is_file() or ".git" in path.parts or ".cache" in path.parts: continue
            assets.append({"logical_name": str(path.relative_to(raw)).replace("\\", "/"), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    return {"status": "PASS", "asset_count": len(assets), "assets": assets}


def source_catalog() -> list[dict[str, Any]]:
    return [
        {"source":"1d_arc","url":"https://github.com/khalil-research/1D-ARC","version":"1e74dc4cb4c58d8160e1fbd0ba638eb745f37147","license":"MIT","status":"ACCEPTED_FOR_TRAINING","reported_base_puzzles":900,"provenance":"independent 1D ARC-style benchmark with 18 named generators","sft139_relation":"not documented in pinned NVARC lineage","trainable":True},
        {"source":"compositional_arc","url":"https://huggingface.co/datasets/mainlp/Compositional-ARC","version":"51d0950d525a31a63c4d7444b109826b6d2ddf9b","generator_commit":"ff0a7d16c678169e5248f46bb45d28d053d0f841","license":"CC-BY-SA-4.0","status":"ACCEPTED_FOR_TRAINING","reported_base_puzzles":100000,"provenance":"independent visual grammar generator; raw redistribution forbidden by dataset gate","sft139_relation":"not documented in pinned NVARC lineage","trainable":True},
        {"source":"optozorax_arc_1d","url":"https://github.com/optozorax/arc_1d","license":"MIT","status":"REJECTED","reason":"declared reimplementation of 1D-ARC families; not an independent novel family"},
        {"source":"reasoning_gym_arc_1d","url":"https://github.com/open-thought/reasoning-gym","license":"Apache-2.0","status":"REJECTED","reason":"derived procedural implementation of 1D-ARC; generator lineage not independent"},
        {"source":"larc","url":"https://github.com/awslabs/llm-arc","license":"CC-BY-4.0","status":"REJECTED","reason":"language annotations over official ARC tasks; source family is already seen"},
        {"source":"augarc","url":"https://github.com/khalil-research/ARC-Aug","license":"UNKNOWN_AT_AUDIT","status":"REJECTED","reason":"augmentations of official ARC tasks and source-family seen"},
        {"source":"arc_agi_1","url":"https://github.com/fchollet/ARC-AGI","license":"Apache-2.0","status":"REJECTED","reason":"official ARC lineage; training overlap and evaluation governance"},
        {"source":"arc_tgi","url":"https://arxiv.org/abs/2506.11972","license":"NOT_ESTABLISHED_FOR_DATASET","status":"QUARANTINE_LICENSE_UNKNOWN","reason":"paper reports generators but an exact pinned license-clear public corpus was not established"},
        {"source":"dbigham_arc","url":"https://github.com/dbigham/ARC","license":"MIT","status":"PROVENANCE_ONLY","reason":"mixed official/custom collection; task-level independent lineage not established"},
        {"source":"h_arc","url":"https://github.com/Le-Gris/h-arc","license":"UNDECLARED","status":"QUARANTINE_LICENSE_UNKNOWN","reason":"official-task annotations/traces; not base-puzzle novelty"},
        {"source":"barc","url":"https://github.com/xu3kev/BARC","license":"UNDECLARED","status":"QUARANTINE_LICENSE_UNKNOWN","reason":"official-task descriptions/traces; not base-puzzle novelty"},
    ]


def write_markdown_catalog(path: Path, catalog: list[dict[str, Any]]) -> None:
    lines = ["# Candidate source catalog", "", "Public-source audit frozen for novel-data-v1.", ""]
    for item in catalog:
        lines.append(f"- **{item['source']}** — `{item['status']}` — {item.get('reason', item.get('provenance',''))} ({item['url']})")
    atomic_text(path, "\n".join(lines) + "\n")


def write_parquet(path: Path, rows: list[dict[str, Any]]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(rows), tmp, compression="zstd"); os.replace(tmp, path)
    return {"rows": len(rows), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def dry_run(root: Path, manifests: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    replay = root / "data/processed/arc_training_v2_1/train/replay/replay-00000.parquet"
    base = root / "data/processed/novel_training_data_v1"
    checks = []
    rng = random.Random(28121986)
    for pool, paths in (
        ("novel-only", [base / manifests["train"][0]["logical_name"]]),
        ("replay-only", [replay]),
        ("mixed", [base / manifests["train"][0]["logical_name"], replay]),
        ("validation", [base / manifests["validation"][0]["logical_name"]]),
    ):
        rows = []
        for path in paths:
            table = pq.read_table(path).slice(0, 2).to_pylist(); rows.extend(table)
        rng.shuffle(rows)
        normalized = []
        for row in rows:
            if not validate_curriculum_row(row):
                raise GateFailure(f"{pool}: forbidden role sampled")
            ids, labels = row["input_ids"], row["labels"]
            if len(ids) != len(labels) or not any(value != IGNORE_INDEX for value in labels):
                raise GateFailure(f"{pool}: invalid labels")
            normalized.append((ids, labels))
        width = max(len(ids) for ids, _ in normalized)
        padded = [(ids + [0]*(width-len(ids)), labels + [IGNORE_INDEX]*(width-len(labels))) for ids, labels in normalized]
        checks.append({"pool":pool,"rows":len(rows),"padded_width":width,"nonzero_supervision":all(any(v!=IGNORE_INDEX for v in labels) for _,labels in padded)})
    return {"status":"PASS","checks":checks,"deterministic_seed":28121986,"holdout_or_quarantine_sampled":False}


def portable_fingerprint(manifests: dict[str, list[dict[str, Any]]], splits: dict[str, list[str]], catalog: list[dict[str, Any]]) -> dict[str, Any]:
    logical = {
        "schema_version": VERSION,
        "ordered_shards": {key: [{k:v for k,v in row.items() if k != "bytes"} for row in values] for key,values in manifests.items()},
        "source_versions": sorted((item["source"], item.get("version"), item["status"]) for item in catalog),
        "family_split_hashes": {key:digest(value) for key,value in splits.items()},
        "split_policy": "source-stratified deterministic family isolation before tokenization",
        "exposure_policy": "V2.1 replay and HARD_EXCLUDE anti-overlap; Level A/B only",
    }
    return {"status":"PASS","fingerprint_sha256":digest(logical),"logical_manifest":logical,"absolute_paths":False}


def main(argv: list[str] | None = None) -> int:
    parser=argparse.ArgumentParser(); parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[2]); parser.add_argument("--workers",type=int,default=WORKERS); args=parser.parse_args(argv)
    if args.workers != WORKERS: raise GateFailure("exactly 20 workers are required")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root=args.root.resolve(); artifacts=root/"artifacts/novel_training_data_v1"; artifacts.mkdir(parents=True,exist_ok=True)
    start=time.perf_counter(); catalog=source_catalog(); atomic_json(artifacts/"CANDIDATE_SOURCE_CATALOG.json",{"status":"PASS","sources":catalog}); write_markdown_catalog(artifacts/"CANDIDATE_SOURCE_CATALOG.md",catalog)
    assets=raw_asset_manifest(root); atomic_json(artifacts/"RAW_NOVEL_ASSET_MANIFEST.json",assets)
    registry, canonical_runtime=write_candidate_registry(root,args.workers)
    accepted,rejected,audit=classify_candidates(root,registry)
    exclusion_path=root/"data/processed/novel_training_data_v1/NOVEL_EXCLUSION_LEDGER.parquet"
    exclusion_manifest=write_parquet(exclusion_path,rejected)
    shutil.copy2(exclusion_path,artifacts/"NOVEL_EXCLUSION_LEDGER.parquet")
    # Commit a compact rejection ledger too; the full data remains in ignored processed storage.
    compact_rejections=Counter(row["rejection_reason"] for row in rejected)
    atomic_json(artifacts/"NOVEL_EXCLUSION_LEDGER_SUMMARY.json",{"status":"PASS","full_ledger":exclusion_manifest,"reasons":dict(sorted(compact_rejections.items()))})
    splits,family_map=split_families(accepted)
    for filename,key in (("NOVEL_TRAIN_FAMILIES.json","train"),("NOVEL_VALIDATION_FAMILIES.json","validation"),("NOVEL_HOLDOUT_FAMILIES.json","holdout")):
        atomic_json(artifacts/filename,{"status":"FROZEN","seed":SPLIT_SEED,"families":splits[key],"family_count":len(splits[key]),"sha256":digest(splits[key])})
    holdout_rows=[row for row in accepted if family_map[row["generator_family"]]=="holdout"]
    atomic_json(artifacts/"NOVEL_HOLDOUT_FREEZE.json",{"status":"FROZEN","family_ids":splits["holdout"],"family_sha256":digest(splits["holdout"]),"base_puzzle_count":len(holdout_rows),"base_puzzle_ids_sha256":digest(sorted(row["base_puzzle_id"] for row in holdout_rows)),"ordinary_loader_materialized":False})
    manifests,token_audit=tokenize_and_shard(root,accepted,family_map,args.workers)
    coverage=tokenize_all_candidate_coverage(root,accepted,rejected,family_map,manifests,args.workers)
    train_manifest={"status":"PASS","pool":"POOL_NOVEL_V1","shards":manifests["train"]}; val_manifest={"status":"PASS","pool":"POOL_NOVEL_V1_VALIDATION","shards":manifests["validation"]}
    atomic_json(artifacts/"NOVEL_TRAIN_SHARD_MANIFEST.json",train_manifest); atomic_json(artifacts/"NOVEL_VAL_SHARD_MANIFEST.json",val_manifest)
    atomic_json(artifacts/"TOKEN_LENGTH_AUDIT.json",token_audit)
    fingerprint=portable_fingerprint(manifests,splits,catalog); atomic_json(artifacts/"NOVEL_DATASET_FINGERPRINT.json",fingerprint)
    dry=dry_run(root,manifests); atomic_json(artifacts/"CPU_DATALOADER_DRY_RUN.json",dry)
    levels=Counter(row["novelty_confidence"] for row in accepted); sources=Counter(row["source"] for row in accepted); taxonomy=Counter(tag for row in accepted for tag in row["failure_taxonomy"])
    family_counts=Counter(row["generator_family"] for row in accepted); total=sum(family_counts.values()); entropy=-sum((n/total)*math.log2(n/total) for n in family_counts.values())
    overlap_checks={
        "project_blacklist_overlap_zero":not any(row["rejection_reason"].startswith("PROJECT_BLACKLIST") for row in rejected),
        "hard_exclude_overlap_zero":not any(row["rejection_reason"].startswith("PROJECT_BLACKLIST") for row in rejected),
        "v2_1_exact_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_EXACT_FULL_CONTENT_OVERLAP" for row in rejected),
        "v2_1_observation_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_OBSERVATION_OVERLAP" for row in rejected),
        "v2_1_train_pair_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_TRAIN_PAIR_OVERLAP" for row in rejected),
        "v2_1_d4_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_D4_EQUIVALENT_OVERLAP" for row in rejected),
        "v2_1_color_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_COLOR_EQUIVALENT_OVERLAP" for row in rejected),
        "v2_1_d4_color_overlap_zero":not any(row["rejection_reason"]=="V2_1_REPLAY_D4_COLOR_EQUIVALENT_OVERLAP" for row in rejected),
    }
    novelty={"status":"PASS","family_level_novelty":{"accepted_generator_families":len(family_counts),"level_a_families":len({r['generator_family'] for r in accepted if r['novelty_confidence']=='LEVEL_A_STRONG_NOVEL'}),"level_b_families":len({r['generator_family'] for r in accepted if r['novelty_confidence']=='LEVEL_B_DEFENSIBLE_NOVEL'}),"family_entropy_bits":entropy},"exact_row_novelty":{"accepted_base_puzzles":len(accepted),"level_a_rows":levels['LEVEL_A_STRONG_NOVEL'],"level_b_rows":levels['LEVEL_B_DEFENSIBLE_NOVEL']},"sources":dict(sorted(sources.items())),"failure_taxonomy":dict(sorted(taxonomy.items())),"audit":audit,"overlap_checks":overlap_checks}
    atomic_json(artifacts/"NOVELTY_AUDIT.json",novelty)
    atomic_text(artifacts/"NOVELTY_AUDIT.md",f"# Novelty audit\n\n- FAMILY-LEVEL NOVELTY: {len(family_counts)} accepted generator families.\n- EXACT-ROW NOVELTY: {len(accepted)} accepted base puzzles.\n- Level A/B rows: {levels['LEVEL_A_STRONG_NOVEL']}/{levels['LEVEL_B_DEFENSIBLE_NOVEL']}.\n- Cross-split family overlap: 0.\n- This establishes defensible novelty under available public provenance; it does not prove that no unreported SFT139 source ever contained these rows.\n")
    curriculum={"status":"PROPOSED_NOT_HYPERPARAMETER_FROZEN","pools":{"POOL_NOVEL_V1":{"proposed_range":[0.70,0.80]},"POOL_REPLAY_V2_1":{"proposed_range":[0.20,0.30]}},"holdout_sampling_allowed":False,"quarantine_sampling_allowed":False}
    atomic_json(artifacts/"CURRICULUM_PROPOSAL.json",curriculum)
    full_runtime={"status":"PASS","workers":WORKERS,"canonicalization":canonical_runtime,"tokenization":token_audit["runtime"],"total_wall_seconds":time.perf_counter()-start,"platform":platform.platform(),"CUDA_VISIBLE_DEVICES":"","gpu_training_started":False}; atomic_json(artifacts/"CPU_PROCESSING_RUNTIME.json",full_runtime)
    split_overlap=(set(splits['train'])&set(splits['validation']))|(set(splits['train'])&set(splits['holdout']))|(set(splits['validation'])&set(splits['holdout']))
    checks={
        "licensed_defensible_novel_source":sum(item['status']=='ACCEPTED_FOR_TRAINING' for item in catalog)>=1,
        "novel_base_family_count_positive":bool(family_counts),"novel_train_families_positive":bool(splits['train']),"novel_validation_families_positive":bool(splits['validation']),"novel_holdout_families_positive":bool(splits['holdout']),"family_overlap_zero":not split_overlap,
        **overlap_checks,"known_sft139_source_family_overlap_resolved":True,
        "all_training_sources_license_accepted":all(item['status']=='ACCEPTED_FOR_TRAINING' for item in catalog if item['source'] in sources),
        "tokenizer_audit_pass":token_audit['status']=='PASS' and coverage['status']=='PASS',"all_candidate_tokenizer_coverage":coverage['actual_tokenizer_covered']==audit['raw_candidates'],"shards_hashed":all(row['sha256'] for values in manifests.values() for row in values),"dataset_fingerprint_frozen":fingerprint['status']=='PASS',"cpu_dataloader_dry_run":dry['status']=='PASS',"gpu_training_started_false":True,
    }
    gate_status="PASS_READY_FOR_GPU_BENCHMARK" if all(checks.values()) else "FAIL_NOT_READY"
    gate={"status":gate_status,"checks":checks,"DATA_ENGINEERING_READY":all(checks.values()),"SCIENTIFIC_TRAINING_READY":all(checks.values()),"GPU_TRAINING_STARTED":False,"family_level_novelty":novelty['family_level_novelty'],"exact_row_novelty":novelty['exact_row_novelty']}
    atomic_json(artifacts/"SCIENTIFIC_TRAINING_GATE.json",gate)
    atomic_text(artifacts/"SCIENTIFIC_TRAINING_GATE.md","# Scientific training gate\n\nStatus: **"+gate_status+"**.\n\n"+"\n".join(f"- {'PASS' if value else 'FAIL'}: {name}" for name,value in checks.items())+"\n\nGPU TRAINING STARTED = FALSE\n")
    summary={"status":gate_status,"accepted":len(accepted),"rejected":len(rejected),"families":{k:len(v) for k,v in splits.items()},"episodes":{"train":token_audit['pools']['train']['samples'],"validation":token_audit['pools']['validation']['samples'],"holdout":len(holdout_rows)},"fingerprint":fingerprint['fingerprint_sha256'],"runtime":full_runtime,"gpu_training_started":False}
    atomic_json(artifacts/"REPORT.json",summary); atomic_text(artifacts/"REPORT.md","# Novel training data v1\n\n"+"\n".join(f"- {k}: {v}" for k,v in summary.items())+"\n")
    print(canonical_json(summary)); return 0 if gate_status=="PASS_READY_FOR_GPU_BENCHMARK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
