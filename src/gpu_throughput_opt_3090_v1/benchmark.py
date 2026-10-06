"""Pure contracts for the bounded RTX3090 throughput optimization run."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


SOURCE_COMMIT = "36c7b22b47a5ee1c70631b90bc082189dbf43abb"
DATASET_FINGERPRINT = "3b040b3fcd45d84361aa5286c3b762e268a5f7c98dae5b3c87ddeb16c5105c82"
SEED = 31_090
CONTEXT = 8_704
EFFECTIVE_BATCH = 4
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class RunConfig:
    name: str
    precision: str
    quantization: str
    micro_batch: int
    grad_accumulation: int
    attention_backend: str = "sdpa"
    gradient_checkpointing: bool = True

    def validate(self) -> None:
        if self.micro_batch * self.grad_accumulation != EFFECTIVE_BATCH:
            raise ValueError(f"effective batch mismatch: {self.name}")
        if self.precision not in ("NF4", "BF16"):
            raise ValueError(f"unsupported precision: {self.precision}")

    def json(self) -> dict[str, Any]:
        self.validate()
        return asdict(self) | {
            "effective_batch": EFFECTIVE_BATCH,
            "context": CONTEXT,
            "lora_rank": 64,
            "lora_alpha": 32,
            "lora_dropout": 0.0,
            "learning_rate": 5e-5,
            "target_modules": TARGET_MODULES,
        }


CONFIG_A = RunConfig("A_NF4_MB1_GA4_SDPA", "NF4", "NF4_DOUBLE_QUANT", 1, 4)
CONFIG_B = RunConfig("B_NF4_MB2_GA2_SDPA", "NF4", "NF4_DOUBLE_QUANT", 2, 2)
CONFIG_C = RunConfig("C_NF4_MB4_GA1_SDPA", "NF4", "NF4_DOUBLE_QUANT", 4, 1)
CONFIG_D = RunConfig("D_BF16_MB1_GA4_SDPA", "BF16", "NONE", 1, 4)


def effective_episode_groups(schedule: Sequence[dict[str, Any]], steps: int) -> list[list[str]]:
    required = steps * EFFECTIVE_BATCH
    if len(schedule) < required:
        raise ValueError(f"schedule too short: {len(schedule)} < {required}")
    return [
        [str(row["sample_id"]) for row in schedule[offset:offset + EFFECTIVE_BATCH]]
        for offset in range(0, required, EFFECTIVE_BATCH)
    ]


def loss_scaling_weights(supervised_counts: Sequence[int], micro_batch: int) -> list[float]:
    if len(supervised_counts) != EFFECTIVE_BATCH or EFFECTIVE_BATCH % micro_batch:
        raise ValueError("invalid effective batch partition")
    total = sum(supervised_counts)
    if total <= 0:
        raise ValueError("effective batch has no supervised tokens")
    return [
        sum(supervised_counts[offset:offset + micro_batch]) / total
        for offset in range(0, EFFECTIVE_BATCH, micro_batch)
    ]


def vram_classification(total_bytes: int, peak_reserved_bytes: int) -> dict[str, Any]:
    headroom = total_bytes - peak_reserved_bytes
    gib = headroom / 2**30
    status = "SAFE" if gib >= 2.0 else "MARGINAL" if gib >= 1.0 else "UNSAFE"
    return {
        "classification": status,
        "headroom_bytes": headroom,
        "headroom_gib": gib,
        "headroom_percentage": 100.0 * headroom / total_bytes,
    }


def runtime_extrapolation(tokens_per_second: float) -> dict[str, dict[str, float]]:
    if tokens_per_second <= 0:
        raise ValueError("non-positive throughput")
    return {
        str(tokens): {
            "seconds": tokens / tokens_per_second,
            "hours": tokens / tokens_per_second / 3600.0,
        }
        for tokens in (2_000_000, 5_000_000, 10_000_000, 20_000_000, 30_000_000, 40_000_000)
    }


def select_fastest_safe(results: Sequence[dict[str, Any]]) -> dict[str, Any] | None:
    eligible = [
        row for row in results
        if row.get("status") == "PASS"
        and row.get("finite_losses") is True
        and row.get("gradient_semantics_pass") is True
        and row.get("checkpoint_reload_pass") is True
        and row.get("vram", {}).get("classification") == "SAFE"
        and row.get("supervised_token_loss_rate") == 0
        and row.get("data_access_policy_pass") is True
    ]
    return max(eligible, key=lambda row: row["tokens_per_second"], default=None)


def update_fairness(reference: dict[str, Any], candidate: dict[str, Any], threshold: float = 0.25) -> dict[str, Any]:
    if reference["parameter_names"] != candidate["parameter_names"] or reference["parameter_shapes"] != candidate["parameter_shapes"]:
        return {"status": "FAIL", "reason": "PARAMETER_SHAPE_MISMATCH"}
    a = reference["update_samples"]
    b = candidate["update_samples"]
    if len(a) != len(b) or not a:
        return {"status": "FAIL", "reason": "UPDATE_SAMPLE_MISMATCH"}
    import math
    diff = math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    ref = math.sqrt(sum(x * x for x in a))
    cand = math.sqrt(sum(x * x for x in b))
    normalized = diff / max(ref, 1e-30)
    ratio = cand / max(ref, 1e-30)
    passed = (
        reference["initial_sample_sha256"] == candidate["initial_sample_sha256"]
        and reference["finite"] is True
        and candidate["finite"] is True
        and normalized <= threshold
        and 0.5 <= ratio <= 2.0
    )
    return {
        "status": "PASS" if passed else "FAIL",
        "normalized_sample_update_difference": normalized,
        "sample_update_norm_ratio": ratio,
        "threshold": threshold,
        "same_initialization": reference["initial_sample_sha256"] == candidate["initial_sample_sha256"],
    }
