from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_l4_clean_hf_physical_batch_scaling_b1_b16_v2.py"
BUILDER_PATH = ROOT / "scripts" / "build_l4_clean_hf_physical_batch_scaling_b1_b16_v2_kaggle.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load(RUNNER_PATH, "l4_batch_scaling_runner")
builder = _load(BUILDER_PATH, "l4_batch_scaling_builder")


class L4CleanHFPhysicalBatchScalingV2Tests(unittest.TestCase):
 def test_frozen_hardware_contract_and_adapter_shape(self) -> None:
    contract = runner.experiment_contract(source_commit=runner.AUTHORITATIVE_SOURCE_COMMIT)
    assert contract["experiment"] == "L4_CLEAN_HF_PHYSICAL_BATCH_SCALING_B1_B16_V2"
    assert contract["widths"] == [1, 2, 4, 8, 12, 16]
    assert contract["hardware"] == {"gpu_count": 4, "gpu_name": "NVIDIA L4", "tensor_parallelism": False}
    assert contract["target_blind"] is True and contract["gold_loaded"] is False
    adapter = contract["benchmark_adapter"]
    assert adapter["mode"] == "DETERMINISTIC_BENCHMARK_LORA"
    assert adapter["seed"] == 42 and adapter["r"] == 256 and adapter["lora_alpha"] == 32
    assert adapter["use_rslora"] is True and adapter["dtype"] == "torch.bfloat16"
    assert adapter["target_modules"] == ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"]
    assert contract["measurement"]["warmup_forwards"] == 2
    assert contract["measurement"]["measurement_forwards"] == 12
    assert contract["measurement"]["streaming_split_and_adopt_for_batched_widths"] is True
    assert contract["benchmark_model_mode"]["global_24_worker_identity_required"] is True
    assert contract["benchmark_model_mode"]["per_worker_fallback_forbidden"] is True
    assert contract["benchmark_model_mode"]["allowed_final_states"] == [
        "ALL_DETERMINISTIC_BENCHMARK_LORA", "ALL_BASE_MODEL_ONLY",
    ]
    assert contract["benchmark_model_mode"]["base_model_only_requires_fresh_experiment_from_b1"] is True
    assert contract["runtime_dynamiccache_preflight"]["physical_batch"] == 4
    assert contract["runtime_dynamiccache_preflight"]["timed"] is False
    with self.assertRaisesRegex(RuntimeError, "source commit"):
        runner.experiment_contract(source_commit="not-the-frozen-source")


 def test_lane_layout_is_deterministic_and_replicas_are_not_new_views(self) -> None:
    assert [(x["replica_index"], x["view"]) for x in runner.lane_specs(1)] == [(0, "anti_transpose")]
    assert [(x["replica_index"], x["view"]) for x in runner.lane_specs(2)] == [(0, "anti_transpose"), (0, "flip_ud")]
    assert [x["view"] for x in runner.lane_specs(4)] == list(runner.BASE_VIEWS)
    assert [(x["replica_index"], x["view"]) for x in runner.lane_specs(8)] == [
        (0, "anti_transpose"), (0, "flip_ud"), (0, "identity"), (0, "transpose"),
        (1, "anti_transpose"), (1, "flip_ud"), (1, "identity"), (1, "transpose"),
    ]
    for width in (4, 8, 12, 16):
        specs = runner.lane_specs(width)
        assert len(specs) == width
        assert [x["lane_index"] for x in specs] == list(range(width))
        assert all(x["view"] in runner.BASE_VIEWS for x in specs)


 def test_every_batched_width_uses_streaming_split_and_adopt(self) -> None:
    calls: list[dict] = []

    def fake_execute(**kwargs):
        calls.append(kwargs)
        return [SimpleNamespace(logits=None) for _ in kwargs["selected"]], {"cache_pack_seconds": 0.0, "model_call_seconds": 0.0, "cache_adoption_seconds": 0.0, "scheduler_elapsed_seconds": 0.0}

    original = runner.execute_ready_forward
    try:
        runner.execute_ready_forward = fake_execute
        request = SimpleNamespace(cache_key="same", position=7)
        b1 = [SimpleNamespace(request=request)]
        runner._execute_once(model=object(), cells=b1)
        assert "streaming_split_and_adopt" not in calls[-1]
        b2 = [SimpleNamespace(request=SimpleNamespace(cache_key="same", position=7)) for _ in range(2)]
        runner._execute_once(model=object(), cells=b2)
        assert calls[-1]["streaming_split_and_adopt"] is True
        assert calls[-1]["release_batch_temporaries_for_audit"] is True
    finally:
        runner.execute_ready_forward = original


 def test_bootstrap_preserves_gpu_identity_and_positive_transition(self) -> None:
    old = {gpu: [10.0, 10.5, 9.5] for gpu in range(4)}
    new = {gpu: [12.0, 12.5, 11.5] for gpu in range(4)}
    evidence = runner._bootstrap_ratio(old, new)
    assert evidence is not None
    assert evidence["trials"] == 10_000 and evidence["seed"] == 20261001
    assert evidence["ratio_ci95_low"] > 1.0
    assert runner._bootstrap_ratio({0: [1.0]}, new) is None


 def test_fail_closed_ready_identity_and_model_mode_gates(self) -> None:
    def row(gpu: int, *, mode: str = "DETERMINISTIC_BENCHMARK_LORA", state: str = "state", config: str = "config") -> dict:
        return {"type": "READY", "gpu_id": gpu, "benchmark_model_mode": mode,
                "benchmark_adapter_state_sha256": state, "benchmark_adapter_config_sha256": config}

    good = [row(gpu) for gpu in range(4)]
    accepted = runner._ready_identity_gate(ready_rows=good, expected_mode="DETERMINISTIC_BENCHMARK_LORA",
                                           prior_identity=None)
    assert accepted["status"] == "PASS"
    assert accepted["identity"]["benchmark_adapter_state_sha256"] == "state"

    mixed_adapter = good[:3] + [row(3, state="other-state")]
    assert runner._ready_identity_gate(ready_rows=mixed_adapter, expected_mode="DETERMINISTIC_BENCHMARK_LORA",
                                       prior_identity=None)["status"] == "INVALID_MIXED_ADAPTER_IDENTITY"
    mixed_mode = good[:3] + [row(3, mode="BASE_MODEL_ONLY")]
    assert runner._ready_identity_gate(ready_rows=mixed_mode, expected_mode="DETERMINISTIC_BENCHMARK_LORA",
                                       prior_identity=None)["status"] == "INVALID_MIXED_BENCHMARK_MODEL_MODE"
    assert runner._ready_identity_gate(ready_rows=good, expected_mode="DETERMINISTIC_BENCHMARK_LORA",
                                       prior_identity={**accepted["identity"], "benchmark_adapter_state_sha256": "old"})[
                                           "status"] == "INVALID_MIXED_ADAPTER_IDENTITY"


 def test_global_identity_requires_exactly_twenty_four_same_instances(self) -> None:
    def ready(gpu: int, *, state: str = "s", config: str = "c") -> dict:
        return {"type": "READY", "gpu_id": gpu, "benchmark_model_mode": "DETERMINISTIC_BENCHMARK_LORA",
                "benchmark_adapter_state_sha256": state, "benchmark_adapter_config_sha256": config}

    widths = {width: {"ready": [ready(gpu) for gpu in range(4)]} for width in runner.WIDTHS}
    accepted = runner._global_identity_gate(all_widths=widths, expected_mode="DETERMINISTIC_BENCHMARK_LORA")
    assert accepted["status"] == "PASS" and accepted["instance_count"] == 24
    assert accepted["global_benchmark_model_mode"] == "ALL_DETERMINISTIC_BENCHMARK_LORA"
    widths[16]["ready"][0]["benchmark_adapter_state_sha256"] = "different"
    assert runner._global_identity_gate(all_widths=widths, expected_mode="DETERMINISTIC_BENCHMARK_LORA")[
        "status"] == "INVALID_MIXED_ADAPTER_IDENTITY"


 def test_runner_has_no_target_path_and_no_submission_behavior(self) -> None:
    source = RUNNER_PATH.read_text(encoding="utf-8")
    for forbidden in ("evaluation_solutions", "arc-agi_evaluation_solutions", "submission.json", "kaggle kernels push"):
        assert forbidden not in source
    assert "streaming_split_and_adopt" in source
    assert "release_batch_temporaries_for_audit" in source
    assert "RUNTIME_DYNAMICCACHE_PREFLIGHT_FAILED" in source
    assert "DETERMINISTIC_BENCHMARK_LORA_ATTACH_FAILED" in source
    assert "fallback_reason" not in source


 def test_private_review_package_is_target_blind_and_not_launched(self) -> None:
    import tempfile
    staged = Path(tempfile.mkdtemp()) / "stage"
    self.addCleanup(lambda: __import__("shutil").rmtree(staged.parent, ignore_errors=True))
    manifest = builder.build(output=staged, owner="private-owner", dataset_slug="private-source", kernel_slug="private-kernel")
    assert manifest["status"] == "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED"
    assert manifest["authoritative_source_commit"] == runner.AUTHORITATIVE_SOURCE_COMMIT
    notebook = json.loads((staged / "kernel" / "private-kernel.ipynb").read_text(encoding="utf-8"))
    source = "".join(notebook["cells"][0]["source"])
    assert "controller" in source and "nvidia-smi" in source and "runtime_preflight" in source
    assert source.index("runtime_preflight") < source.index("controller")
    assert "submission.json" not in source and "evaluation_solutions" not in source
    packaged = staged / "dataset" / "ARC2" / "scripts" / RUNNER_PATH.name
    assert packaged.is_file()
    assert (staged / "dataset" / "ARC2" / "src" / "inference" / "hf_peft_backend.py").is_file()
    assert not list((staged / "dataset").rglob("*solution*.json"))
