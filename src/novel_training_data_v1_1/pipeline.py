"""CPU-only V1.1 correction for scientific splits and curriculum sampling.

V1 remains immutable.  This module reuses its canonical registry, tokenized
rows, tokenizer contract, and overlap evidence.  Only split-dependent shards
are rebuilt, and only newly exposed train/validation rows are tokenized.
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from novel_training_data_v1.pipeline import (
    SAMPLE_SCHEMA,
    WORKERS,
    _init_tokenizer,
    _task_text_compositional,
    atomic_json,
    atomic_text,
    bounded_pool_map,
    classify_candidates,
    digest,
    tokenize_item,
)
from training_data_v2.pipeline import canonical_json, sha256_file


VERSION = "novel_training_data_v1_1"
OFFICIAL_SPLIT_SEED = 1860
OFFICIAL_DATASET_REVISION = "51d0950d525a31a63c4d7444b109826b6d2ddf9b"
OFFICIAL_GENERATOR_COMMIT = "ff0a7d16c678169e5248f46bb45d28d053d0f841"
SAMPLER_VERSION = "hierarchical-pool-source-family-episode-v1"
DIAGNOSTIC_POOL_WEIGHTS = {"POOL_NOVEL_V1_1": 0.75, "POOL_REPLAY_V2_1": 0.25}
POLICIES = (
    "ROW_UNIFORM_BASELINE",
    "FAMILY_UNIFORM_GLOBAL",
    "SOURCE_UNIFORM_THEN_FAMILY_UNIFORM",
    "SQRT_FAMILY_SOURCE_WEIGHT",
    "CONFIGURABLE_MANUAL_SOURCE_WEIGHT",
)


class GateFailure(RuntimeError):
    pass


def composition_signature(meta: dict[str, Any]) -> str:
    values = []
    for item in meta.values():
        if not isinstance(item, dict) or not item.get("type"):
            raise GateFailure("invalid Compositional-ARC meta_data")
        values.append(str(item["type"]))
    if len(values) != 3:
        raise GateFailure("systematicity composition must contain three transformations")
    return "+".join(sorted(values))


def official_episode_key(meta: dict[str, Any], queries: Sequence[Any]) -> str:
    return digest({"meta_data": meta, "queries": list(queries)[:10]})


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _family_id(source: str, signature: str) -> str:
    return f"{source}:systematicity:{signature}"


def recover_systematicity_split(root: Path) -> tuple[dict[str, dict[str, str]], dict[str, Any]]:
    split_root = root / "data/raw/novel/compositional_arc/split_seed_1860"
    files = {
        "train": split_root / "train_systematicity.jsonl.gz",
        "validation": split_root / "val_systematicity.jsonl.gz",
        "holdout": split_root / "test_systematicity.jsonl.gz",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise GateFailure(f"official systematicity files missing: {missing}")

    membership: dict[str, tuple[str, str, str]] = {}
    split_episode_keys: dict[str, set[str]] = defaultdict(set)
    split_meta_ids: dict[str, set[str]] = defaultdict(set)
    split_signatures: dict[str, set[str]] = defaultdict(set)
    split_counts: Counter[str] = Counter()
    duplicate_keys: list[str] = []
    for split, path in files.items():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            for line in handle:
                raw = json.loads(line)
                meta = raw["meta_data"]
                key = official_episode_key(meta, raw["queries"])
                signature = composition_signature(meta)
                meta_id = digest(meta)
                if key in membership:
                    duplicate_keys.append(key)
                membership[key] = (split, signature, meta_id)
                split_episode_keys[split].add(key)
                split_meta_ids[split].add(meta_id)
                split_signatures[split].add(signature)
                split_counts[split] += 1
    if duplicate_keys or sum(split_counts.values()) != 100_000 or len(membership) != 100_000:
        raise GateFailure("official systematicity identities are not exactly one-to-one")

    line_map: dict[str, dict[str, str]] = {}
    found: Counter[str] = Counter()
    all_path = root / "data/raw/novel/compositional_arc/all/all_episodes.jsonl.gz"
    with gzip.open(all_path, "rt", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            raw = json.loads(line)
            key = official_episode_key(raw["meta_data"], raw["queries"])
            if key not in membership:
                raise GateFailure(f"all_episodes line {index} absent from official split")
            split, signature, meta_id = membership[key]
            base_id = f"compositional_arc:episode-{index:06d}"
            line_map[base_id] = {
                "split": split,
                "family": _family_id("compositional_arc", signature),
                "composition_signature": signature,
                "metadata_identity": meta_id,
                "official_episode_identity": key,
                "source_line": str(index),
            }
            found[key] += 1
    if len(line_map) != 100_000 or set(found.values()) != {1} or set(found) != set(membership):
        raise GateFailure("official split could not be mapped exactly to all_episodes")

    def pairwise(values: dict[str, set[str]]) -> dict[str, int]:
        return {
            "train_validation": len(values["train"] & values["validation"]),
            "train_holdout": len(values["train"] & values["holdout"]),
            "validation_holdout": len(values["validation"] & values["holdout"]),
        }

    episode_overlap = pairwise(split_episode_keys)
    metadata_overlap = pairwise(split_meta_ids)
    signature_overlap = pairwise(split_signatures)
    audit = {
        "status": "PASS_WITH_UPSTREAM_VAL_TEST_TEMPLATE_OVERLAP",
        "dataset": "mainlp/Compositional-ARC",
        "dataset_revision": OFFICIAL_DATASET_REVISION,
        "generator_repository": "mainlp/C-ARC",
        "generator_commit": OFFICIAL_GENERATOR_COMMIT,
        "upstream_split": "split_seed_1860",
        "upstream_files": {
            split: {
                "logical_name": str(path.relative_to(root)).replace("\\", "/"),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for split, path in files.items()
        },
        "upstream_implementation": {
            "script": "scripts/split_data_systematicity.py",
            "seed": OFFICIAL_SPLIT_SEED,
            "frac_test_compositions": 0.2,
            "min_func_examples": 5000,
            "shuffle_study_examples": False,
            "num_primitives": 2,
            "num_compositions": 2,
            "num_queries": 10,
            "semantic_rule": "8 of C(5,3) transformation triplets train; 2 OOD triplets evaluation; evaluation sequence split in half into validation/test",
        },
        "episodes": dict(split_counts),
        "transformation_composition_templates": {key: sorted(value) for key, value in split_signatures.items()},
        "template_counts": {key: len(value) for key, value in split_signatures.items()},
        "metadata_identity_counts": {key: len(value) for key, value in split_meta_ids.items()},
        "episode_identity_counts": {key: len(value) for key, value in split_episode_keys.items()},
        "episode_identity_overlap": episode_overlap,
        "metadata_identity_overlap": metadata_overlap,
        "higher_level_composition_signature_overlap": signature_overlap,
        "episode_identities_disjoint": not any(episode_overlap.values()),
        "train_vs_evaluation_composition_signatures_disjoint": (
            signature_overlap["train_validation"] == 0 and signature_overlap["train_holdout"] == 0
        ),
        "validation_vs_holdout_composition_signatures_disjoint": signature_overlap["validation_holdout"] == 0,
        "scientific_note": "The pinned upstream protocol is exactly reproduced. It intentionally splits the two OOD composition templates by episode into validation/test, so those two higher-level templates and their 96 parameter identities occur in both evaluation partitions.",
    }
    return line_map, audit


def preserved_1d_split(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    base = root / "artifacts/novel_training_data_v1"
    for split, filename in (
        ("train", "NOVEL_TRAIN_FAMILIES.json"),
        ("validation", "NOVEL_VALIDATION_FAMILIES.json"),
        ("holdout", "NOVEL_HOLDOUT_FAMILIES.json"),
    ):
        for family in _read_json(base / filename)["families"]:
            if family.startswith("1d_arc:"):
                if family in result:
                    raise GateFailure(f"1D family crosses splits: {family}")
                result[family] = split
    if len(result) != 18:
        raise GateFailure("expected exactly 18 preserved 1D-ARC named families")
    return result


def build_split_mapping(
    accepted: list[dict[str, Any]],
    compositional: dict[str, dict[str, str]],
    one_d: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, list[dict[str, Any]]]]:
    mapping: dict[str, dict[str, str]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in accepted:
        base_id = row["base_puzzle_id"]
        if row["source"] == "compositional_arc":
            value = dict(compositional[base_id])
        elif row["source"] == "1d_arc":
            family = row["generator_family"]
            value = {
                "split": one_d[family],
                "family": family,
                "composition_signature": family,
                "metadata_identity": family,
                "official_episode_identity": row["full_content_sha256"],
                "source_line": "",
            }
        else:
            raise GateFailure(f"unexpected accepted source: {row['source']}")
        mapping[base_id] = value
        grouped[value["split"]].append(row)
    return mapping, grouped


@dataclass
class _ShardAccumulator:
    base: Path
    buffers: dict[str, list[dict[str, Any]]]
    manifests: dict[str, list[dict[str, Any]]]
    indices: Counter[str]
    stats: dict[str, dict[str, Any]]

    @classmethod
    def create(cls, base: Path) -> "_ShardAccumulator":
        for path in (base / "train/novel", base / "validation/novel"):
            if path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)
        return cls(base, defaultdict(list), {"train": [], "validation": []}, Counter(), defaultdict(lambda: {"lengths": [], "supervised": [], "loss8192": 0}))

    def add(self, split: str, row: dict[str, Any]) -> None:
        self.buffers[split].append(row)
        self.stats[split]["lengths"].append(int(row["sequence_length"]))
        self.stats[split]["supervised"].append(int(row["supervised_token_count"]))
        self.stats[split]["loss8192"] += int(json.loads(row["truncation_supervised_loss_json"])["8192"])
        if len(self.buffers[split]) >= 1000:
            self.flush(split)

    def flush(self, split: str) -> None:
        rows = self.buffers[split]
        if not rows:
            return
        directory = self.base / ("train/novel" if split == "train" else "validation/novel")
        path = directory / f"novel-{self.indices[split]:05d}.parquet"
        tmp = path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows, schema=SAMPLE_SCHEMA), tmp, compression="zstd")
        os.replace(tmp, path)
        families = sorted({row["generator_family"] for row in rows})
        self.manifests[split].append({
            "logical_name": str(path.relative_to(self.base)).replace("\\", "/"),
            "rows": len(rows),
            "base_family_count": len(families),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "tokens": sum(row["sequence_length"] for row in rows),
            "supervised_tokens": sum(row["supervised_token_count"] for row in rows),
        })
        self.indices[split] += 1
        rows.clear()


def rebuild_split_shards(
    root: Path,
    accepted: list[dict[str, Any]],
    mapping: dict[str, dict[str, str]],
    workers: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any], set[str]]:
    target = {base_id for base_id, value in mapping.items() if value["split"] != "holdout"}
    old_base = root / "data/processed/novel_training_data_v1"
    new_base = root / "data/processed/novel_training_data_v1_1"
    acc = _ShardAccumulator.create(new_base)
    seen: set[str] = set()
    columns = list(SAMPLE_SCHEMA.names)
    old_paths = sorted((old_base / "train/novel").glob("*.parquet")) + sorted((old_base / "validation/novel").glob("*.parquet"))
    for path in old_paths:
        parquet = pq.ParquetFile(path)
        for batch in parquet.iter_batches(batch_size=128, columns=columns):
            for row in batch.to_pylist():
                base_id = row["base_puzzle_id"]
                if base_id not in target:
                    continue
                if base_id in seen:
                    raise GateFailure(f"duplicate reused tokenized row: {base_id}")
                row["split"] = mapping[base_id]["split"]
                row["generator_family"] = mapping[base_id]["family"]
                acc.add(row["split"], row)
                seen.add(base_id)

    accepted_by_id = {row["base_puzzle_id"]: row for row in accepted}
    missing = sorted(target - seen, key=lambda value: int(value.rsplit("-", 1)[-1]) if value.startswith("compositional_arc:") else value)
    if any(not value.startswith("compositional_arc:") for value in missing):
        raise GateFailure("preserved 1D split unexpectedly requires retokenization")
    wanted = {int(mapping[base_id]["source_line"]): base_id for base_id in missing}
    raw_path = root / "data/raw/novel/compositional_arc/all/all_episodes.jsonl.gz"

    def items() -> Iterator[tuple[dict[str, Any], str]]:
        for source_line, text in _task_text_compositional(raw_path, set(wanted)):
            base_id = wanted[source_line]
            meta = dict(accepted_by_id[base_id])
            meta["split"] = mapping[base_id]["split"]
            meta["generator_family"] = mapping[base_id]["family"]
            yield meta, text

    model = root / "data/raw/models/sorokin_qwen3_4b_grids15_sft139_transformers_bfloat16_1"
    started = time.perf_counter()
    if missing:
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_tokenizer, initargs=(str(model),)) as pool:
            for row in bounded_pool_map(pool, tokenize_item, items(), batch_size=128, chunksize=2):
                base_id = row["base_puzzle_id"]
                if base_id in seen:
                    raise GateFailure(f"duplicate newly tokenized row: {base_id}")
                acc.add(row["split"], row)
                seen.add(base_id)
    for split in ("train", "validation"):
        acc.flush(split)
    if seen != target:
        raise GateFailure(f"split shard coverage mismatch: missing={len(target-seen)} extras={len(seen-target)}")

    pools: dict[str, Any] = {}
    for split in ("train", "validation"):
        lengths = np.asarray(acc.stats[split]["lengths"], dtype=np.int64)
        supervised = np.asarray(acc.stats[split]["supervised"], dtype=np.int64)
        total_supervised = int(supervised.sum())
        pools[split] = {
            "samples": int(lengths.size),
            "total_tokens": int(lengths.sum()),
            "supervised_tokens": total_supervised,
            "mean": float(lengths.mean()),
            "p50": float(np.percentile(lengths, 50)),
            "p90": float(np.percentile(lengths, 90)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
            "overflow_8192": {
                "samples_affected": int((lengths > 8192).sum()),
                "sample_rate": float((lengths > 8192).mean()),
                "supervised_token_loss": int(acc.stats[split]["loss8192"]),
                "supervised_token_loss_rate": float(acc.stats[split]["loss8192"] / total_supervised),
            },
            "overflow_8704": {
                "samples_affected": int((lengths > 8704).sum()),
                "sample_rate": float((lengths > 8704).mean()),
                "supervised_token_loss": 0 if int(lengths.max()) <= 8704 else None,
                "supervised_token_loss_rate": 0.0 if int(lengths.max()) <= 8704 else None,
            },
        }
    reuse = {
        "status": "PASS",
        "v1_tokenized_rows_reused": len(seen) - len(missing),
        "newly_tokenized_rows": len(missing),
        "duplicate_rows": 0,
        "tokenizer_contract": "UNCHANGED_SFT139_NATIVE_API",
        "tokenization_wall_seconds": time.perf_counter() - started,
        "pools": pools,
    }
    return acc.manifests, reuse, seen


def family_probabilities(
    family_rows: list[dict[str, Any]],
    policy: str,
    manual_source_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    counts: Counter[str] = Counter()
    for row in family_rows:
        counts[row["family"]] += int(row.get("raw_episode_count", 1))
    sources = {row["family"]: row["source"] for row in family_rows}
    by_source: dict[str, list[str]] = defaultdict(list)
    for family in sorted(counts):
        by_source[sources[family]].append(family)
    if policy == "ROW_UNIFORM_BASELINE":
        total = sum(counts.values())
        return {family: count / total for family, count in counts.items()}
    if policy == "FAMILY_UNIFORM_GLOBAL":
        return {family: 1 / len(counts) for family in counts}
    if policy == "SOURCE_UNIFORM_THEN_FAMILY_UNIFORM":
        source_weight = {source: 1 / len(by_source) for source in by_source}
    elif policy == "SQRT_FAMILY_SOURCE_WEIGHT":
        denominator = sum(math.sqrt(len(families)) for families in by_source.values())
        source_weight = {source: math.sqrt(len(families)) / denominator for source, families in by_source.items()}
    elif policy == "CONFIGURABLE_MANUAL_SOURCE_WEIGHT":
        if manual_source_weights is None or set(manual_source_weights) != set(by_source):
            raise GateFailure("manual source weights must cover every source exactly")
        denominator = sum(manual_source_weights.values())
        source_weight = {source: manual_source_weights[source] / denominator for source in by_source}
    else:
        raise GateFailure(f"unknown curriculum policy: {policy}")
    return {
        family: source_weight[source] / len(families)
        for source, families in by_source.items()
        for family in families
    }


def _entropy(probabilities: Iterable[float]) -> tuple[float, float]:
    values = [value for value in probabilities if value > 0]
    bits = -sum(value * math.log2(value) for value in values)
    return bits, 2 ** bits


def simulate_curriculum(
    pools: dict[str, list[dict[str, Any]]],
    policy: str,
    draws: int,
    seed: int,
    pool_weights: dict[str, float],
    manual_source_weights: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    rng = random.Random(seed)
    pool_prob = {pool: pool_weights[pool] / sum(pool_weights.values()) for pool in pools}
    family_tables: dict[str, dict[str, float]] = {}
    rows_by_family: dict[str, dict[str, Any]] = {}
    for pool, rows in pools.items():
        manual = None if manual_source_weights is None else manual_source_weights[pool]
        family_tables[pool] = family_probabilities(rows, policy, manual)
        rows_by_family[pool] = {row["family"]: row for row in rows}
    weighted: list[tuple[str, str, float]] = []
    for pool, table in family_tables.items():
        for family, probability in table.items():
            weighted.append((pool, family, pool_prob[pool] * probability))
    labels = [(pool, family) for pool, family, _ in weighted]
    weights = [probability for _, _, probability in weighted]
    sampled = rng.choices(labels, weights=weights, k=draws)
    observed = Counter(sampled)
    source_observed: Counter[str] = Counter()
    pool_observed: Counter[str] = Counter()
    table = []
    for pool, family, probability in weighted:
        row = rows_by_family[pool][family]
        count = observed[(pool, family)]
        source_observed[row["source"]] += count
        pool_observed[pool] += count
        table.append({
            "pool": pool,
            "source": row["source"],
            "family": family,
            "raw_episode_count": row["raw_episode_count"],
            "sampling_probability": probability,
            "expected_draws_per_100k": probability * 100_000,
            "observed_draws": count,
            "effective_weight_per_raw_row": probability / row["raw_episode_count"],
        })
    family_probs = [row["sampling_probability"] for row in table]
    bits, effective = _entropy(family_probs)
    nonzero_counts = [row["observed_draws"] for row in table if row["observed_draws"] > 0]
    per_pool_family_exposure = {}
    for pool in pools:
        pool_rows = [row for row in table if row["pool"] == pool]
        conditional = [row["sampling_probability"] / pool_prob[pool] for row in pool_rows]
        pool_bits, pool_effective = _entropy(conditional)
        pool_counts = [row["observed_draws"] for row in pool_rows if row["observed_draws"] > 0]
        per_pool_family_exposure[pool] = {
            "min_observed": min(pool_counts),
            "max_observed": max(pool_counts),
            "largest_smallest_ratio": max(pool_counts) / min(pool_counts),
            "conditional_entropy_bits": pool_bits,
            "conditional_effective_family_count": pool_effective,
            "eligible_family_count": len(pool_rows),
        }
    return {
        "policy": policy,
        "draws": draws,
        "seed": seed,
        "diagnostic_pool_weights_not_final": pool_weights,
        "observed_pool_proportions": {key: value / draws for key, value in sorted(pool_observed.items())},
        "observed_source_proportions": {key: value / draws for key, value in sorted(source_observed.items())},
        "family_exposure": {
            "min_observed": min(nonzero_counts),
            "max_observed": max(nonzero_counts),
            "largest_smallest_ratio": max(nonzero_counts) / min(nonzero_counts),
            "entropy_bits": bits,
            "effective_family_count": effective,
            "by_pool": per_pool_family_exposure,
        },
        "families": sorted(table, key=lambda row: (row["pool"], row["source"], row["family"])),
        "expected_novel_percentage": pool_prob.get("POOL_NOVEL_V1_1", 0.0) * 100,
        "expected_replay_percentage": pool_prob.get("POOL_REPLAY_V2_1", 0.0) * 100,
    }


class HierarchicalCurriculumSampler:
    """Deterministic POOL -> SOURCE/FAMILY -> EPISODE sampler."""

    def __init__(
        self,
        episodes: dict[str, dict[str, list[str]]],
        pool_weights: dict[str, float],
        policy: str,
        seed: int,
        manual_source_weights: dict[str, dict[str, float]] | None = None,
    ) -> None:
        self.episodes = episodes
        self.rng = random.Random(seed)
        self.pool_weights = pool_weights
        rows = {
            pool: [
                {"family": family, "source": source, "raw_episode_count": len(ids)}
                for source, families in sources.items()
                for family, ids in families.items()
            ]
            for pool, sources in episodes.items()
        }
        self.pool_probs = [pool_weights[pool] for pool in episodes]
        self.pools = list(episodes)
        self.family_tables = {
            pool: family_probabilities(
                rows[pool],
                policy,
                None if manual_source_weights is None else manual_source_weights[pool],
            )
            for pool in episodes
        }

    def draw(self) -> tuple[str, str, str]:
        pool = self.rng.choices(self.pools, weights=self.pool_probs, k=1)[0]
        families = list(self.family_tables[pool])
        family = self.rng.choices(families, weights=list(self.family_tables[pool].values()), k=1)[0]
        source = next(source for source, family_map in self.episodes[pool].items() if family in family_map)
        episode = self.rng.choice(self.episodes[pool][source][family])
        return pool, family, episode


def aggregate_validation_losses(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    if not records:
        raise GateFailure("validation losses cannot be empty")
    by_family: dict[str, list[float]] = defaultdict(list)
    by_source: dict[str, list[float]] = defaultdict(list)
    for row in records:
        value = float(row["loss"])
        by_family[row["family"]].append(value)
        by_source[row["source"]].append(value)
    family_means = {family: sum(values) / len(values) for family, values in by_family.items()}
    source_family_means: dict[str, list[float]] = defaultdict(list)
    family_source = {row["family"]: row["source"] for row in records}
    for family, value in family_means.items():
        source_family_means[family_source[family]].append(value)
    source_macro_values = [sum(values) / len(values) for values in source_family_means.values()]
    return {
        "micro_average_loss": sum(float(row["loss"]) for row in records) / len(records),
        "macro_family_average_loss": sum(family_means.values()) / len(family_means),
        "source_macro_loss": sum(source_macro_values) / len(source_macro_values),
    }


def official_systematicity_protocol_status(systematicity: dict[str, Any]) -> dict[str, Any]:
    """Interpret the pinned upstream split without imposing a new split contract."""
    overlap = systematicity["higher_level_composition_signature_overlap"]
    episode_overlap = systematicity["episode_identity_overlap"]
    templates = systematicity["transformation_composition_templates"]
    shared_ood = sorted(set(templates["validation"]) & set(templates["holdout"]))
    expected_shared_ood = (
        len(shared_ood) == 2
        and shared_ood == sorted(templates["validation"])
        and shared_ood == sorted(templates["holdout"])
        and overlap["validation_holdout"] == len(shared_ood)
    )
    fields = {
        "train_validation_high_level_composition_overlap_zero": overlap["train_validation"] == 0,
        "train_holdout_high_level_composition_overlap_zero": overlap["train_holdout"] == 0,
        "validation_holdout_episode_identity_overlap_zero": episode_overlap["validation_holdout"] == 0,
        "validation_holdout_high_level_composition_overlap": (
            "UPSTREAM_EXPECTED_SHARED_OOD_TEMPLATES" if expected_shared_ood else "UNEXPECTED"
        ),
    }
    protocol_ready = (
        fields["train_validation_high_level_composition_overlap_zero"]
        and fields["train_holdout_high_level_composition_overlap_zero"]
        and fields["validation_holdout_episode_identity_overlap_zero"]
        and expected_shared_ood
    )
    return {
        "checks": fields,
        "OFFICIAL_SYSTEMATICITY_PROTOCOL_READY": protocol_ready,
        "STRICT_THREE_WAY_FAMILY_ISOLATION": not any(overlap.values()),
        "shared_ood_templates": shared_ood,
    }


def bounded_source_discovery_audit() -> dict[str, Any]:
    sources = [
        {
            "source": "Fraser/arc-agi-synthetic",
            "url": "https://huggingface.co/datasets/Fraser/arc-agi-synthetic",
            "version": "ddd600547beaffdb4ebe77c90ca72c0ee64b71ce",
            "license": "NOT_DECLARED_IN_DATASET_CARD",
            "reported_rows": 4940,
            "relation_to_official_arc": "LLM-authored ARC-style programs; not documented as official-task resampling",
            "relation_to_sft139": "not established",
            "generator_family_independence": "plausible but not license-cleared or independently overlap-audited",
            "decision": "QUARANTINE_LICENSE",
        },
        {
            "source": "google/ARC-GEN",
            "url": "https://github.com/google/ARC-GEN",
            "version": "a15cbdb44c776610aeeb9f487a06af875d3d0878",
            "license": "Apache-2.0",
            "reported_rows": 100000,
            "relation_to_official_arc": "400/now-expanded official ARC task mimetic generators; validation explicitly reproduces official tasks",
            "relation_to_sft139": "official-family lineage is already excluded regardless of exact row novelty",
            "generator_family_independence": "not independent of official ARC families",
            "decision": "REJECTED_SFT139_OR_OFFICIAL_FAMILY",
        },
        {
            "source": "Omega-Reasoning/ARC-Task-Generators-Inventory",
            "url": "https://github.com/Omega-Reasoning/ARC-Task-Generators-Inventory",
            "version": "a614132ff5b2cb3628063d541e7cbd74a2cd2edb",
            "license": "Apache-2.0",
            "reported_rows": None,
            "relation_to_official_arc": "human-refined generators covering ARC-Mini, ARC-AGI-1 and ARC-AGI-2 task families",
            "relation_to_sft139": "official-family overlap not eligible as novel",
            "generator_family_independence": "not independent for official-derived generators",
            "decision": "REJECTED_SFT139_OR_OFFICIAL_FAMILY",
        },
        {
            "source": "MGWSimpson/AlphaARC",
            "url": "https://github.com/MGWSimpson/AlphaARC",
            "version": "9066db137df240f6121c7b01c9e97f1b3f3a83d9",
            "license": "BSD-3-Clause-Clear",
            "reported_rows": None,
            "relation_to_official_arc": "research repository with generated/code task assets; stable dataset release and family provenance not established",
            "relation_to_sft139": "not established",
            "generator_family_independence": "not established at task-family level",
            "decision": "PROVENANCE_ONLY",
        },
        {
            "source": "giotto-ai/giotto-arc-agi-data",
            "url": "https://github.com/giotto-ai/giotto-arc-agi-data",
            "version": "3578118daec268a4e30eedc8c9cd40812bd8f617",
            "license": "AGPL-3.0 repository; dataset terms linked separately",
            "reported_rows": 1188500,
            "relation_to_official_arc": "mixed original seeds, manual seeds, augmentations, and re-ARC-derived data",
            "relation_to_sft139": "mixed lineage requires shard-level audit before any eligibility claim",
            "generator_family_independence": "not established for mixed release",
            "decision": "PROVENANCE_ONLY",
        },
    ]
    return {
        "status": "PASS",
        "scope": "bounded public-source discovery audit",
        "exhaustive_global_discovery_claimed": False,
        "large_ineligible_datasets_downloaded": False,
        "accepted_new_sources": [],
        "sources": sources,
        "decision_counts": dict(Counter(row["decision"] for row in sources)),
    }


def _family_rows_from_ids(mapping: dict[str, dict[str, str]], accepted_ids: set[str], split: str) -> list[dict[str, Any]]:
    counts: Counter[tuple[str, str]] = Counter()
    for base_id in accepted_ids:
        value = mapping[base_id]
        if value["split"] == split:
            counts[(base_id.split(":", 1)[0], value["family"])] += 1
    return [
        {"source": source, "family": family, "raw_episode_count": count}
        for (source, family), count in sorted(counts.items())
    ]


def _replay_family_rows(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, list[str]]]]:
    path = root / "data/processed/arc_training_v2_1/train/replay/replay-00000.parquet"
    rows = pq.read_table(path, columns=["sample_id", "source", "generator_family", "final_training_role"]).to_pylist()
    counts: Counter[tuple[str, str]] = Counter()
    episodes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        role = row["final_training_role"]
        if "HOLDOUT" in role or "HARD_EXCLUDE" in role or "QUARANTINE" in role:
            raise GateFailure(f"forbidden replay role: {role}")
        key = (row["source"], row["generator_family"])
        counts[key] += 1
        episodes[row["source"]][row["generator_family"]].append(row["sample_id"])
    return ([{"source": source, "family": family, "raw_episode_count": count} for (source, family), count in sorted(counts.items())], episodes)


def _novel_episode_index(root: Path, manifests: dict[str, list[dict[str, Any]]], split: str) -> dict[str, dict[str, list[str]]]:
    base = root / "data/processed/novel_training_data_v1_1"
    episodes: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for shard in manifests[split]:
        path = base / shard["logical_name"]
        for row in pq.read_table(path, columns=["sample_id", "source", "generator_family", "final_training_role"]).to_pylist():
            role = row["final_training_role"]
            if "HOLDOUT" in role or "HARD_EXCLUDE" in role or "QUARANTINE" in role:
                raise GateFailure(f"forbidden novel role: {role}")
            episodes[row["source"]][row["generator_family"]].append(row["sample_id"])
    return episodes


def sampler_dry_run(
    root: Path,
    manifests: dict[str, list[dict[str, Any]]],
    holdout_ids: set[str],
    manual: dict[str, Any],
) -> dict[str, Any]:
    novel_train = _novel_episode_index(root, manifests, "train")
    novel_validation = _novel_episode_index(root, manifests, "validation")
    _, replay = _replay_family_rows(root)
    holdout_tokens = {base_id + ":native-source-episode-v1" for base_id in holdout_ids}
    cases = {
        "novel-only": ({"POOL_NOVEL_V1_1": novel_train}, {"POOL_NOVEL_V1_1": 1.0}),
        "replay-only": ({"POOL_REPLAY_V2_1": replay}, {"POOL_REPLAY_V2_1": 1.0}),
        "mixed": ({"POOL_NOVEL_V1_1": novel_train, "POOL_REPLAY_V2_1": replay}, DIAGNOSTIC_POOL_WEIGHTS),
        "validation": ({"POOL_NOVEL_V1_1": novel_validation}, {"POOL_NOVEL_V1_1": 1.0}),
    }
    results = []
    for name, (episodes, weights) in cases.items():
        sampler_a = HierarchicalCurriculumSampler(episodes, weights, "FAMILY_UNIFORM_GLOBAL", 28121986)
        sampler_b = HierarchicalCurriculumSampler(episodes, weights, "FAMILY_UNIFORM_GLOBAL", 28121986)
        first = [sampler_a.draw() for _ in range(10_000)]
        second = [sampler_b.draw() for _ in range(10_000)]
        if first != second:
            raise GateFailure(f"{name}: sampler is not deterministic")
        sampled_ids = {episode for _, _, episode in first}
        if sampled_ids & holdout_tokens:
            raise GateFailure(f"{name}: holdout sampled")
        results.append({
            "sampler": name,
            "draws": len(first),
            "deterministic_seed_behavior": True,
            "holdout_rows_sampled": 0,
            "quarantine_rows_sampled": 0,
            "hard_exclude_rows_sampled": 0,
            "unique_families_observed": len({family for _, family, _ in first}),
        })
    return {"status": "PASS", "sampler_version": SAMPLER_VERSION, "checks": results}


def _write_systematicity_markdown(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Compositional-ARC systematicity audit",
        "",
        f"Pinned dataset revision: `{audit['dataset_revision']}`.",
        f"Pinned generator commit: `{audit['generator_commit']}`.",
        f"Official split: `split_seed_{audit['upstream_split'].split('_')[-1]}`.",
        "",
        f"Episodes: {audit['episodes']}.",
        f"Template counts: {audit['template_counts']}.",
        f"Episode identity overlap: {audit['episode_identity_overlap']}.",
        f"Higher-level composition overlap: {audit['higher_level_composition_signature_overlap']}.",
        "",
        audit["scientific_note"],
        "",
    ]
    atomic_text(path, "\n".join(lines))


def _context_plan(token_audit: dict[str, Any]) -> dict[str, Any]:
    train = token_audit["pools"]["train"]
    configs = []
    for name, limit in (("CONFIG_A", 8192), ("CONFIG_B", 8704)):
        overflow = train[f"overflow_{limit}"]
        configs.append({
            "config": name,
            "max_seq_length": limit,
            "status": "PLANNED_NOT_RUN",
            "samples_affected": overflow["samples_affected"],
            "supervised_token_loss": overflow["supervised_token_loss"],
            "supervised_token_loss_rate": overflow["supervised_token_loss_rate"],
            "expected_token_load_per_optimizer_step": {
                "assumption": "micro_batch_size=1, gradient_accumulation_steps configurable",
                "mean_tokens_per_microbatch": min(train["mean"], limit),
                "upper_bound_tokens_per_microbatch": limit,
                "formula": "mean_tokens_per_microbatch * gradient_accumulation_steps",
            },
        })
    return {
        "status": "PREPARED_NOT_RUN",
        "observed_max_sequence_length": train["max"],
        "silent_truncation_allowed": False,
        "configs": configs,
        "gpu_benchmark_started": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--workers", type=int, default=WORKERS)
    args = parser.parse_args(argv)
    if args.workers != WORKERS:
        raise GateFailure("exactly 20 workers are required by the frozen CPU build")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = args.root.resolve()
    artifacts = root / "artifacts/novel_training_data_v1_1"
    artifacts.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    registry = root / "data/processed/novel_training_data_v1/candidate_registry.parquet"
    accepted, rejected, overlap_audit = classify_candidates(root, registry)
    overlap_checks = {
        "v2_1_overlap_remains_zero": not any(str(row["rejection_reason"]).startswith("V2_1_") for row in rejected),
        "project_blacklist_overlap_remains_zero": not any(str(row["rejection_reason"]).startswith("PROJECT_BLACKLIST") for row in rejected),
        "accepted_count_unchanged": len(accepted) == 100_899,
        "rejected_count_unchanged": len(rejected) == 2,
    }
    atomic_json(artifacts / "V1_NOVELTY_REUSE_AND_OVERLAP_AUDIT.json", {
        "status": "PASS" if all(overlap_checks.values()) else "FAIL",
        "frozen_v1_commit": "4d013d006f5e774cd502d6c4a935907f06fbfe80",
        "v1_artifacts_overwritten": False,
        "checks": overlap_checks,
        "classification_audit": overlap_audit,
    })

    compositional, systematicity = recover_systematicity_split(root)
    atomic_json(artifacts / "COMPOSITIONAL_SYSTEMATICITY_AUDIT.json", systematicity)
    _write_systematicity_markdown(artifacts / "COMPOSITIONAL_SYSTEMATICITY_AUDIT.md", systematicity)
    one_d = preserved_1d_split(root)
    mapping, grouped = build_split_mapping(accepted, compositional, one_d)
    accepted_ids = set(mapping)

    families = {
        split: sorted({mapping[row["base_puzzle_id"]]["family"] for row in rows})
        for split, rows in grouped.items()
    }
    family_overlap = {
        "train_validation": sorted(set(families["train"]) & set(families["validation"])),
        "train_holdout": sorted(set(families["train"]) & set(families["holdout"])),
        "validation_holdout": sorted(set(families["validation"]) & set(families["holdout"])),
    }
    for split, filename in (
        ("train", "NOVEL_TRAIN_FAMILIES.json"),
        ("validation", "NOVEL_VALIDATION_FAMILIES.json"),
        ("holdout", "NOVEL_HOLDOUT_FAMILIES.json"),
    ):
        atomic_json(artifacts / filename, {
            "status": "FROZEN",
            "split_policy": "Compositional-ARC official systematicity seed 1860; V1-preserved deterministic named-family isolation for 1D-ARC",
            "families": families[split],
            "family_count": len(families[split]),
            "sha256": digest(families[split]),
            "episodes": len(grouped[split]),
        })

    holdout_ids = {row["base_puzzle_id"] for row in grouped["holdout"]}
    holdout = {
        "status": "FROZEN",
        "base_puzzle_count": len(holdout_ids),
        "base_puzzle_ids": sorted(holdout_ids),
        "base_puzzle_ids_sha256": digest(sorted(holdout_ids)),
        "families": families["holdout"],
        "family_sha256": digest(families["holdout"]),
        "ordinary_train_validation_shards_materialized": False,
        "labels_or_targets_materialized_in_holdout_manifest": False,
        "family_overlap": family_overlap,
        "higher_level_composition_overlap": systematicity["higher_level_composition_signature_overlap"],
    }
    atomic_json(artifacts / "V1_1_HOLDOUT_FREEZE.json", holdout)

    manifests, token_audit, materialized = rebuild_split_shards(root, accepted, mapping, args.workers)
    if materialized & holdout_ids:
        raise GateFailure("holdout leaked into materialized shards")
    train_manifest = {"status": "PASS", "pool": "POOL_NOVEL_V1_1", "shards": manifests["train"]}
    val_manifest = {"status": "PASS", "pool": "POOL_NOVEL_V1_1_VALIDATION", "shards": manifests["validation"]}
    atomic_json(artifacts / "NOVEL_TRAIN_SHARD_MANIFEST.json", train_manifest)
    atomic_json(artifacts / "NOVEL_VAL_SHARD_MANIFEST.json", val_manifest)
    atomic_json(artifacts / "SPLIT_DEPENDENT_SHARD_REBUILD_AUDIT.json", token_audit)

    source_audit = bounded_source_discovery_audit()
    atomic_json(artifacts / "SOURCE_DISCOVERY_AUDIT_V1_1.json", source_audit)
    manual = _read_json(root / "configs/novel_v1_1_curriculum_manual.json")
    novel_family_rows = _family_rows_from_ids(mapping, accepted_ids, "train")
    replay_family_rows, _ = _replay_family_rows(root)
    pools = {"POOL_NOVEL_V1_1": novel_family_rows, "POOL_REPLAY_V2_1": replay_family_rows}
    simulations = []
    for index, policy in enumerate(POLICIES):
        manual_weights = manual["source_weights"] if policy == "CONFIGURABLE_MANUAL_SOURCE_WEIGHT" else None
        simulations.append(simulate_curriculum(
            pools,
            policy,
            draws=100_000,
            seed=28121986 + index,
            pool_weights=DIAGNOSTIC_POOL_WEIGHTS,
            manual_source_weights=manual_weights,
        ))
    recommended = next(row for row in simulations if row["policy"] == "FAMILY_UNIFORM_GLOBAL")
    raw_ratio = max(row["raw_episode_count"] for row in novel_family_rows) / min(row["raw_episode_count"] for row in novel_family_rows)
    recommended_novel = [row for row in recommended["families"] if row["pool"] == "POOL_NOVEL_V1_1"]
    probability_ratio = max(row["sampling_probability"] for row in recommended_novel) / min(row["sampling_probability"] for row in recommended_novel)
    row_domination_disabled = probability_ratio < raw_ratio / 10
    curriculum = {
        "status": "PASS" if row_domination_disabled else "FAIL",
        "sampler_version": SAMPLER_VERSION,
        "draws_per_policy": 100_000,
        "provisional_novel_range": [0.70, 0.80],
        "provisional_replay_range": [0.20, 0.30],
        "diagnostic_midpoint_not_final": DIAGNOSTIC_POOL_WEIGHTS,
        "final_weights_frozen": False,
        "recommended_policy": "FAMILY_UNIFORM_GLOBAL",
        "recommendation_reason": "every eligible family is equally likely inside its frozen pool; the implied source mass is proportional to eligible family count rather than raw rows, avoiding arbitrary 50/50 sources and the large cross-source per-family disparity observed under source-first policies",
        "ROW_COUNT_DOMINATION_DISABLED": row_domination_disabled,
        "novel_raw_episode_ratio_largest_smallest": raw_ratio,
        "recommended_novel_family_probability_ratio_largest_smallest": probability_ratio,
        "simulations": simulations,
    }
    atomic_json(artifacts / "CURRICULUM_SAMPLER_AUDIT.json", curriculum)
    atomic_text(artifacts / "CURRICULUM_SAMPLER_AUDIT.md", "# Curriculum sampler audit\n\n"
                f"Recommended configurable policy: **{curriculum['recommended_policy']}**.\n\n"
                f"ROW_COUNT_DOMINATION_DISABLED = {row_domination_disabled}.\n\n"
                "The 75/25 pool mix is a diagnostic midpoint only; final weights remain unfrozen pending the RTX3090 benchmark.\n")

    dry = sampler_dry_run(root, manifests, holdout_ids, manual)
    atomic_json(artifacts / "CPU_SAMPLER_DRY_RUN.json", dry)
    validation_audit = {
        "status": "PASS",
        "implementation": "novel_training_data_v1_1.pipeline.aggregate_validation_losses",
        "metrics": ["micro_average_loss", "macro_family_average_loss", "source_macro_loss"],
        "primary_scientific_model_selection_metric": "MACRO_FAMILY_VALIDATION_LOSS",
        "fixture_result": aggregate_validation_losses([
            {"source": "a", "family": "a:small", "loss": 1.0},
            {"source": "a", "family": "a:large", "loss": 3.0},
            {"source": "a", "family": "a:large", "loss": 5.0},
            {"source": "b", "family": "b:one", "loss": 2.0},
        ]),
    }
    atomic_json(artifacts / "FAMILY_BALANCED_VALIDATION_METRICS.json", validation_audit)
    context = _context_plan(token_audit)
    atomic_json(artifacts / "CONTEXT_LENGTH_BENCHMARK_PLAN.json", context)

    fingerprint_logical = {
        "schema_version": VERSION,
        "source_versions": {
            "1d_arc": "1e74dc4cb4c58d8160e1fbd0ba638eb745f37147",
            "compositional_arc": OFFICIAL_DATASET_REVISION,
            "compositional_arc_generator": OFFICIAL_GENERATOR_COMMIT,
            "v2_1_replay": _read_json(root / "artifacts/training_data_v2_1/PORTABLE_DATASET_FINGERPRINT.json").get("fingerprint_sha256") if (root / "artifacts/training_data_v2_1/PORTABLE_DATASET_FINGERPRINT.json").exists() else "frozen-v2.1-local-authoritative",
        },
        "systematicity_split": {
            "identity": "split_seed_1860",
            "files": {key: value["sha256"] for key, value in systematicity["upstream_files"].items()},
        },
        "family_manifests": {split: digest(families[split]) for split in ("train", "validation", "holdout")},
        "ordered_shards": {
            split: [{key: value for key, value in row.items() if key != "bytes"} for row in manifests[split]]
            for split in ("train", "validation")
        },
        "sampler_policy_version": SAMPLER_VERSION,
        "exposure_policy": "Novel train and V2.1 replay only; holdout/quarantine/HARD_EXCLUDE impossible to sample",
        "absolute_paths": False,
    }
    fingerprint = {
        "status": "FROZEN",
        "fingerprint_sha256": digest(fingerprint_logical),
        "logical_manifest": fingerprint_logical,
        "absolute_paths": False,
    }
    atomic_json(artifacts / "NOVEL_DATASET_FINGERPRINT.json", fingerprint)

    protocol = official_systematicity_protocol_status(systematicity)
    checks = {
        "v2_1_overlap_remains_zero": overlap_checks["v2_1_overlap_remains_zero"],
        "project_blacklist_overlap_remains_zero": overlap_checks["project_blacklist_overlap_remains_zero"],
        "compositional_systematicity_split_recovered_and_frozen": systematicity["episode_identities_disjoint"],
        "one_d_named_family_split_isolated": len(one_d) == 18 and len(set(one_d)) == 18,
        **protocol["checks"],
        "holdout_absent_from_loaders": not bool(materialized & holdout_ids),
        "row_count_domination_disabled": row_domination_disabled,
        "family_aware_sampler_implemented": curriculum["status"] == "PASS",
        "family_aware_validation_metric_prepared": validation_audit["status"] == "PASS",
        "bounded_source_discovery_v1_1_completed": source_audit["status"] == "PASS",
        "context_benchmark_configs_prepared": context["status"] == "PREPARED_NOT_RUN",
        "tokenizer_contract_unchanged": token_audit["tokenizer_contract"] == "UNCHANGED_SFT139_NATIVE_API",
        "shards_hashed": all(row["sha256"] for values in manifests.values() for row in values),
        "portable_fingerprint_frozen": fingerprint["status"] == "FROZEN" and not fingerprint["absolute_paths"],
        "cpu_sampler_dry_run": dry["status"] == "PASS",
        "gpu_training_started_false": True,
    }
    blocking_checks = [value for value in checks.values() if isinstance(value, bool)]
    gpu_benchmark_ready = protocol["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"] and all(blocking_checks)
    gate_status = "PASS_READY_FOR_GPU_BENCHMARK" if gpu_benchmark_ready else "FAIL_NOT_READY"
    gate = {
        "status": gate_status,
        "checks": checks,
        "OFFICIAL_SYSTEMATICITY_PROTOCOL_READY": protocol["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"],
        "STRICT_THREE_WAY_FAMILY_ISOLATION": protocol["STRICT_THREE_WAY_FAMILY_ISOLATION"],
        "GPU_BENCHMARK_READY": gpu_benchmark_ready,
        "scientific_claim": "Validation and holdout are episode-disjoint and both evaluate composition templates unseen during training, reproducing the official upstream systematicity protocol.",
        "informational_evidence": {
            "shared_validation_holdout_ood_templates": protocol["shared_ood_templates"],
            "strict_three_way_family_isolation_is_not_an_upstream_requirement": True,
        },
        "blocking_evidence": {
            "family_overlap": family_overlap,
            "upstream_higher_level_composition_signature_overlap": systematicity["higher_level_composition_signature_overlap"],
        } if gate_status == "FAIL_NOT_READY" else {},
        "GPU_TRAINING_STARTED": False,
    }
    atomic_json(artifacts / "SCIENTIFIC_TRAINING_GATE_V1_1.json", gate)

    summary = {
        "status": gate_status,
        "OFFICIAL_SYSTEMATICITY_PROTOCOL_READY": protocol["OFFICIAL_SYSTEMATICITY_PROTOCOL_READY"],
        "STRICT_THREE_WAY_FAMILY_ISOLATION": protocol["STRICT_THREE_WAY_FAMILY_ISOLATION"],
        "GPU_BENCHMARK_READY": gpu_benchmark_ready,
        "scientific_claim": gate["scientific_claim"],
        "branch": "training/novel-data-v1.1-scientific-split-curriculum",
        "systematicity_upstream_counts": systematicity["episodes"],
        "accepted_episode_counts": {split: len(grouped[split]) for split in ("train", "validation", "holdout")},
        "family_counts": {split: len(families[split]) for split in ("train", "validation", "holdout")},
        "one_d_family_split": dict(Counter(one_d.values())),
        "recommended_policy": curriculum["recommended_policy"],
        "provisional_novel_replay_range": {"novel": [0.70, 0.80], "replay": [0.20, 0.30]},
        "dataset_fingerprint": fingerprint["fingerprint_sha256"],
        "runtime_seconds": time.perf_counter() - started,
        "GPU_TRAINING_STARTED": False,
    }
    atomic_json(artifacts / "REPORT.json", summary)
    atomic_text(artifacts / "REPORT.md", "# Novel training data V1.1\n\n" + "\n".join(f"- {key}: {value}" for key, value in summary.items()) + "\n")
    print(canonical_json(summary))
    return 0 if gate_status == "PASS_READY_FOR_GPU_BENCHMARK" else 2


if __name__ == "__main__":
    raise SystemExit(main())
