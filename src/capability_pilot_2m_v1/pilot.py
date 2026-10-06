"""Pure, testable contracts for the first scientific capability pilot."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "d7b489e73835011230a6d2c4fd70e21f515ac152"
DATASET_FINGERPRINT = "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"
REPLAY_SHA256 = "32823bea01d69bead992abe5bd7c88ce463e4b82e32dc60f237343b7b8c854dc"
SEED = 2_000_031
TOKEN_BUDGET = 2_000_000
CONTEXT = 8_704
EFFECTIVE_BATCH = 4
NOVEL_DRAWS_PER_STEP = 3
REPLAY_DRAWS_PER_STEP = 1
CHECKPOINT_THRESHOLDS = (500_000, 1_000_000, 1_500_000, 2_000_000)
NOVEL_POOL = "POOL_NOVEL_V1_1"
REPLAY_POOL = "POOL_REPLAY_V2_1"
REPLAY_SOURCES = ("official_arc_agi_2", "miniarc", "conceptarc", "rearc")
FORBIDDEN_ROLE_TERMS = ("HOLDOUT", "HARD_EXCLUDE", "QUARANTINE")
FORBIDDEN_PATH_TERMS = ("holdout", "eval60", "gold", "solution")


class PilotGateError(RuntimeError):
    """Raised when a frozen pilot invariant is violated."""


def validate_allowed_data_paths(*paths: Path | str) -> None:
    """Fail closed if any configured input could expose a forbidden evaluation set."""
    for path in paths:
        lowered = str(path).replace("\\", "/").lower()
        if any(term in lowered for term in FORBIDDEN_PATH_TERMS):
            raise PilotGateError(f"FORBIDDEN_DATA_PATH={path}")


def frozen_training_config() -> dict[str, Any]:
    return {
        "status": "FROZEN",
        "source_commit": SOURCE_COMMIT,
        "base_model": "sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
        "dataset_fingerprint": DATASET_FINGERPRINT,
        "replay_shard_sha256": REPLAY_SHA256,
        "token_budget": TOKEN_BUDGET,
        "seed": SEED,
        "precision": "BF16",
        "base_model_frozen": True,
        "quantization": "NONE",
        "lora": {
            "rank": 64,
            "alpha": 32,
            "dropout": 0.0,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        },
        "context": CONTEXT,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "effective_episode_batch": EFFECTIVE_BATCH,
        "gradient_checkpointing": True,
        "attention_backend": "sdpa",
        "optimizer": "bitsandbytes.optim.PagedAdamW8bit",
        "learning_rate": 5e-5,
        "lr_schedule": "linear warmup for optimizer steps 1-3, then constant",
        "pool_weight_policy": "PILOT_V1_FROZEN",
        "pool_draws_per_optimizer_step": {NOVEL_POOL: 3, REPLAY_POOL: 1},
        "sampling_hierarchy": "POOL_TO_FAMILY_TO_EPISODE",
        "novel_holdout_forbidden": True,
        "eval60_forbidden": True,
        "gold_forbidden": True,
    }


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def row_metadata(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": str(row["source"]),
        "family": str(row["generator_family"]),
        "sample_id": str(row["sample_id"]),
        "sequence_length": int(row["sequence_length"]),
        "supervised_token_count": int(row["supervised_token_count"]),
    }


def context_eligible(row: dict[str, Any]) -> bool:
    return 0 < int(row["sequence_length"]) <= CONTEXT and int(row["supervised_token_count"]) > 0


def _stable_rng(seed: int, label: str) -> random.Random:
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def select_validation_sentinel(
    rows: Sequence[dict[str, Any]], per_family: int = 32, seed: int = SEED
) -> dict[str, Any]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split")) != "validation":
            raise PilotGateError("NOVEL_VALIDATION_SPLIT_MISMATCH")
        if any(term in str(row.get("final_training_role", "")) for term in FORBIDDEN_ROLE_TERMS):
            raise PilotGateError("FORBIDDEN_ROLE_IN_VALIDATION")
        if context_eligible(row):
            by_family[str(row["generator_family"])].append(row)
    if len(by_family) != 4:
        raise PilotGateError(f"VALIDATION_FAMILY_COUNT={len(by_family)}")
    selected: list[dict[str, Any]] = []
    for family in sorted(by_family):
        candidates = sorted(by_family[family], key=lambda row: str(row["sample_id"]))
        count = min(per_family, len(candidates))
        chosen = _stable_rng(seed, f"validation:{family}").sample(candidates, count)
        selected.extend(row_metadata(row) for row in sorted(chosen, key=lambda row: str(row["sample_id"])))
    payload = {
        "status": "FROZEN",
        "seed": seed,
        "selection_policy": "32 deterministic context-compatible episodes per Novel validation family",
        "family_count": len(by_family),
        "episode_count": len(selected),
        "episodes": selected,
    }
    payload["selection_sha256"] = canonical_sha256(selected)
    return payload


def select_replay_retention_sentinel(
    rows: Sequence[dict[str, Any]], families_per_source: int = 16, seed: int = SEED
) -> dict[str, Any]:
    by_source_family: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        if str(row.get("final_training_role")) != "TRAIN_ELIGIBLE_REPLAY":
            raise PilotGateError("NON_REPLAY_ROW_IN_REPLAY_SHARD")
        if context_eligible(row):
            by_source_family[str(row["source"])][str(row["generator_family"])].append(row)
    if set(by_source_family) != set(REPLAY_SOURCES):
        raise PilotGateError(f"REPLAY_SOURCE_SET={sorted(by_source_family)}")
    selected: list[dict[str, Any]] = []
    for source in REPLAY_SOURCES:
        family_map = by_source_family[source]
        families = sorted(family_map)
        if len(families) < families_per_source:
            raise PilotGateError(f"REPLAY_FAMILIES_TOO_FEW={source}:{len(families)}")
        chosen_families = sorted(_stable_rng(seed, f"retention-families:{source}").sample(families, families_per_source))
        for family in chosen_families:
            candidates = sorted(family_map[family], key=lambda row: str(row["sample_id"]))
            chosen = _stable_rng(seed, f"retention-episode:{source}:{family}").choice(candidates)
            selected.append(row_metadata(chosen))
    payload = {
        "status": "FROZEN",
        "seed": seed,
        "selection_policy": "16 deterministic families per replay source and one context-compatible episode per family",
        "source_count": len(REPLAY_SOURCES),
        "family_count": len(selected),
        "episode_count": len(selected),
        "episodes": selected,
    }
    payload["selection_sha256"] = canonical_sha256(selected)
    return payload


def _family_index(
    rows: Sequence[dict[str, Any]], *, pool: str, excluded_ids: set[str]
) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        sample_id = str(row["sample_id"])
        role = str(row.get("final_training_role", ""))
        split = str(row.get("split", "train"))
        if sample_id in excluded_ids or not context_eligible(row):
            continue
        if pool == NOVEL_POOL:
            if split != "train" or role != "DEFENSIBLE_NOVEL_TRAINABLE":
                continue
        elif pool == REPLAY_POOL:
            if role != "TRAIN_ELIGIBLE_REPLAY":
                continue
        else:
            raise PilotGateError(f"UNKNOWN_POOL={pool}")
        if any(term in role for term in FORBIDDEN_ROLE_TERMS):
            raise PilotGateError(f"FORBIDDEN_ROLE_IN_TRAINING={role}")
        index[str(row["generator_family"])].append(row)
    if not index or any(not rows_for_family for rows_for_family in index.values()):
        raise PilotGateError(f"EMPTY_FAMILY_INDEX={pool}")
    return {family: sorted(values, key=lambda row: str(row["sample_id"])) for family, values in sorted(index.items())}


def build_training_schedule(
    novel_rows: Sequence[dict[str, Any]],
    replay_rows: Sequence[dict[str, Any]],
    retention_sample_ids: Iterable[str],
    token_budget: int = TOKEN_BUDGET,
    seed: int = SEED,
) -> dict[str, Any]:
    retention_ids = set(retention_sample_ids)
    novel = _family_index(novel_rows, pool=NOVEL_POOL, excluded_ids=set())
    replay = _family_index(replay_rows, pool=REPLAY_POOL, excluded_ids=retention_ids)
    rng = random.Random(seed)
    schedule: list[dict[str, Any]] = []
    total_tokens = total_supervised = step = 0
    while total_tokens < token_budget:
        pools = [NOVEL_POOL] * NOVEL_DRAWS_PER_STEP + [REPLAY_POOL] * REPLAY_DRAWS_PER_STEP
        rng.shuffle(pools)
        for microstep, pool in enumerate(pools):
            family_map = novel if pool == NOVEL_POOL else replay
            family = rng.choice(sorted(family_map))
            row = rng.choice(family_map[family])
            item = row_metadata(row) | {
                "optimizer_step_index": step,
                "microstep_index": microstep,
                "pool": pool,
            }
            schedule.append(item)
            total_tokens += item["sequence_length"]
            total_supervised += item["supervised_token_count"]
        step += 1
    if len(schedule) % EFFECTIVE_BATCH:
        raise PilotGateError("PARTIAL_OPTIMIZER_STEP")
    if total_tokens < token_budget or total_tokens - sum(item["sequence_length"] for item in schedule[-4:]) >= token_budget:
        raise PilotGateError("TOKEN_BUDGET_STOP_RULE_VIOLATION")
    if retention_ids & {item["sample_id"] for item in schedule}:
        raise PilotGateError("RETENTION_SENTINEL_IN_TRAINING")
    pool_counts = Counter(item["pool"] for item in schedule)
    expected_novel = step * NOVEL_DRAWS_PER_STEP
    expected_replay = step * REPLAY_DRAWS_PER_STEP
    if pool_counts != Counter({NOVEL_POOL: expected_novel, REPLAY_POOL: expected_replay}):
        raise PilotGateError("POOL_WEIGHT_CONTRACT_VIOLATION")
    payload = {
        "status": "FROZEN",
        "seed": seed,
        "token_budget": token_budget,
        "stop_rule": "first complete optimizer step whose cumulative actual transformer tokens reach or exceed budget",
        "sampling_hierarchy": "POOL_TO_FAMILY_TO_EPISODE",
        "pool_weight_policy": "PILOT_V1_FROZEN",
        "pool_draws_per_optimizer_step": {NOVEL_POOL: NOVEL_DRAWS_PER_STEP, REPLAY_POOL: REPLAY_DRAWS_PER_STEP},
        "row_count_domination_disabled": True,
        "optimizer_steps": step,
        "episode_count": len(schedule),
        "actual_transformer_tokens": total_tokens,
        "scheduled_supervised_tokens": total_supervised,
        "episodes": schedule,
    }
    payload["schedule_sha256"] = canonical_sha256(schedule)
    return payload


def checkpoint_crossings(previous_tokens: int, current_tokens: int) -> list[int]:
    return [threshold for threshold in CHECKPOINT_THRESHOLDS if previous_tokens < threshold <= current_tokens]


def aggregate_losses(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise PilotGateError("EMPTY_EVALUATION_RECORDS")
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if not math.isfinite(float(record["loss"])) or int(record["supervised_tokens"]) <= 0:
            raise PilotGateError("INVALID_EVALUATION_RECORD")
        by_family[str(record["family"])].append(record)
    family_losses = {
        family: sum(float(row["nll_sum"]) for row in rows) / sum(int(row["supervised_tokens"]) for row in rows)
        for family, rows in sorted(by_family.items())
    }
    family_sources = {str(row["family"]): str(row["source"]) for row in records}
    source_families: dict[str, list[float]] = defaultdict(list)
    for family, loss in family_losses.items():
        source_families[family_sources[family]].append(loss)
    per_source = {source: sum(values) / len(values) for source, values in sorted(source_families.items())}
    total_tokens = sum(int(row["supervised_tokens"]) for row in records)
    return {
        "micro_average_loss": sum(float(row["nll_sum"]) for row in records) / total_tokens,
        "macro_family_average_loss": sum(family_losses.values()) / len(family_losses),
        "source_macro_loss": sum(per_source.values()) / len(per_source),
        "supervised_token_count": total_tokens,
        "episode_count": len(records),
        "family_count": len(family_losses),
        "per_family_loss": family_losses,
        "per_source_loss": per_source,
    }


def relative_changes(base_novel: float, novel: float, base_replay: float, replay: float) -> dict[str, float]:
    if base_novel <= 0 or base_replay <= 0:
        raise PilotGateError("NONPOSITIVE_BASELINE_LOSS")
    return {
        "novel_gain_relative": (base_novel - novel) / base_novel,
        "replay_loss_change": (replay - base_replay) / base_replay,
    }


def select_best_safe_checkpoint(points: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        point for point in points
        if int(point.get("training_tokens", 0)) > 0 and float(point["replay_loss_change"]) <= 0.05
    ]
    return min(eligible, key=lambda point: float(point["novel_macro_family_loss"]), default=None)


def classify_results(novel_gain: float, replay_change: float) -> dict[str, str]:
    if novel_gain < 0:
        novel = "NOVEL_WORSENED"
    elif novel_gain >= 0.10:
        novel = "STRONG_NOVEL_GAIN"
    elif novel_gain >= 0.05:
        novel = "PROMISING_NOVEL_GAIN"
    elif novel_gain >= 0.02:
        novel = "WEAK_NOVEL_SIGNAL"
    else:
        novel = "NO_CLEAR_LEARNING"
    if replay_change > 0.10:
        replay = "SEVERE_FORGETTING"
    elif replay_change > 0.05:
        replay = "FORGETTING_WARNING"
    else:
        replay = "RETENTION_WITHIN_5_PERCENT"
    if novel_gain < 0 and replay_change > 0:
        gate = "HARMFUL"
    elif replay_change > 0.10:
        gate = "SEVERE_FORGETTING"
    elif replay_change > 0.05:
        gate = "FORGETTING_WARNING"
    elif novel_gain >= 0.10 and replay_change <= 0.03:
        gate = "STRONG_PROMISING"
    elif novel_gain >= 0.05 and replay_change <= 0.05:
        gate = "PROMISING"
    elif novel_gain >= 0.02 and replay_change <= 0.05:
        gate = "WEAK_SIGNAL"
    else:
        gate = "NO_CLEAR_LEARNING"
    return {"novel_capability_result": novel, "replay_retention_result": replay, "gate": gate}


def next_phase_recommendation(gate: str, novel_gain: float, replay_change: float) -> str:
    if gate in {"HARMFUL", "SEVERE_FORGETTING"}:
        return "F_STOP_DUE_TO_HARM"
    if replay_change > 0.05:
        return "C_INCREASE_REPLAY_WEIGHT"
    if novel_gain < 0.02:
        return "E_INVESTIGATE_NO_LEARNING"
    return "A_EXTEND_SAME_CONFIG_TO_5M"


def distribution_audit(schedule: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not schedule:
        raise PilotGateError("EMPTY_SCHEDULE")
    pool_counts = Counter(str(row["pool"]) for row in schedule)
    pool_tokens = Counter()
    source_counts = Counter()
    source_tokens = Counter()
    family_counts = Counter()
    for row in schedule:
        pool_tokens[str(row["pool"])] += int(row["sequence_length"])
        source_counts[str(row["source"])] += 1
        source_tokens[str(row["source"])] += int(row["sequence_length"])
        family_counts[f"{row['pool']}::{row['family']}"] += 1
    total = len(schedule)
    total_tokens = sum(pool_tokens.values())
    probabilities = [count / total for count in family_counts.values()]
    entropy = -sum(p * math.log2(p) for p in probabilities)
    novel_exposure = [count for key, count in family_counts.items() if key.startswith(f"{NOVEL_POOL}::")]
    replay_exposure = [count for key, count in family_counts.items() if key.startswith(f"{REPLAY_POOL}::")]
    return {
        "status": "PASS",
        "row_count_domination_disabled": True,
        "episode_counts_by_pool": dict(sorted(pool_counts.items())),
        "episode_percentage_by_pool": {pool: count / total for pool, count in sorted(pool_counts.items())},
        "token_counts_by_pool": dict(sorted(pool_tokens.items())),
        "token_percentage_by_pool": {pool: count / total_tokens for pool, count in sorted(pool_tokens.items())},
        "episode_counts_by_source": dict(sorted(source_counts.items())),
        "token_counts_by_source": dict(sorted(source_tokens.items())),
        "family_draw_counts": dict(sorted(family_counts.items())),
        "novel_family_exposure": {"min": min(novel_exposure), "max": max(novel_exposure)},
        "replay_family_exposure": {"min": min(replay_exposure), "max": max(replay_exposure)},
        "family_entropy_bits": entropy,
        "episode_pool_policy_exact_75_25": pool_counts[NOVEL_POOL] * 1 == 3 * pool_counts[REPLAY_POOL],
    }
