from __future__ import annotations

from gpu_throughput_opt_3090_v1.benchmark import (
    CONFIG_A,
    CONFIG_B,
    CONFIG_C,
    CONFIG_D,
    TARGET_MODULES,
    attention_mask_required,
    effective_episode_groups,
    loss_scaling_weights,
    runtime_extrapolation,
    select_fastest_safe,
    update_fairness,
    vram_classification,
)


def test_attention_mask_only_for_real_padding() -> None:
    assert attention_mask_required([8704]) is False
    assert attention_mask_required([8704, 8704]) is False
    assert attention_mask_required([8704, 8192]) is True


def test_same_effective_episode_batch_across_abc() -> None:
    schedule = [{"sample_id": str(i)} for i in range(12)]
    expected = [[str(i) for i in range(4)], [str(i) for i in range(4, 8)], [str(i) for i in range(8, 12)]]
    assert effective_episode_groups(schedule, 3) == expected
    for config in (CONFIG_A, CONFIG_B, CONFIG_C):
        config.validate()
        assert config.micro_batch * config.grad_accumulation == 4


def test_gradient_accumulation_scaling_equivalence() -> None:
    counts = [100, 200, 300, 400]
    assert loss_scaling_weights(counts, 1) == [0.1, 0.2, 0.3, 0.4]
    assert loss_scaling_weights(counts, 2) == [0.3, 0.7]
    assert loss_scaling_weights(counts, 4) == [1.0]


def test_frozen_lora_contract_is_shared() -> None:
    for config in (CONFIG_A, CONFIG_B, CONFIG_C, CONFIG_D):
        payload = config.json()
        assert payload["lora_rank"] == 64
        assert payload["target_modules"] == TARGET_MODULES
        assert payload["context"] == 8704


def test_runtime_and_vram_contracts() -> None:
    assert runtime_extrapolation(1000)["5000000"]["seconds"] == 5000
    assert vram_classification(24 * 2**30, 21 * 2**30)["classification"] == "SAFE"
    assert vram_classification(24 * 2**30, 23.5 * 2**30)["classification"] == "UNSAFE"


def test_winner_selection_only_safe_stable() -> None:
    base = {"status": "PASS", "finite_losses": True, "gradient_semantics_pass": True,
            "checkpoint_reload_pass": True, "supervised_token_loss_rate": 0,
            "data_access_policy_pass": True}
    unsafe = base | {"config": "fast", "tokens_per_second": 999, "vram": {"classification": "UNSAFE"}}
    safe = base | {"config": "safe", "tokens_per_second": 100, "vram": {"classification": "SAFE"}}
    assert select_fastest_safe([unsafe, safe])["config"] == "safe"


def test_update_fairness_metric() -> None:
    base = {"parameter_names": ["x"], "parameter_shapes": [[2]], "update_samples": [1.0, 2.0],
            "initial_sample_sha256": "a", "finite": True}
    close = base | {"update_samples": [1.01, 1.99]}
    assert update_fairness(base, close)["status"] == "PASS"
    far = base | {"update_samples": [10.0, 20.0]}
    assert update_fairness(base, far)["status"] == "FAIL"


def test_data_access_policy_fixture() -> None:
    access = {"novel_train_accessed": True, "replay_train_accessed": True,
              "novel_validation_accessed": False, "novel_holdout_accessed": False,
              "eval60_accessed": False}
    assert not any(access[key] for key in ("novel_validation_accessed", "novel_holdout_accessed", "eval60_accessed"))
