from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py"
BUILDER_PATH = ROOT / "scripts" / "build_l4_native_base_physical_batch_scaling_b1_b16_v1_kaggle.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load(RUNNER_PATH, "l4_native_base_runner")
builder = _load(BUILDER_PATH, "l4_native_base_builder")


class L4NativeBasePhysicalBatchScalingV1Tests(unittest.TestCase):
    def test_frozen_native_base_contract(self) -> None:
        contract = runner.experiment_contract(source_commit=runner.AUTHORITATIVE_SOURCE_COMMIT)
        assert contract["experiment"] == "L4_NATIVE_BASE_PHYSICAL_BATCH_SCALING_B1_B16_V1"
        assert contract["benchmark_model_mode"] == "BASE_MODEL_ONLY"
        assert contract["peft_used"] is False
        assert contract["torchao_required"] is False
        assert contract["native_kaggle_environment"] is True
        assert contract["widths"] == [1, 2, 4, 8, 12, 16]
        assert contract["hardware"] == {"gpu_count": 4, "gpu_name": "NVIDIA L4", "tensor_parallelism": False}
        assert contract["base_views"] == ["anti_transpose", "flip_ud", "identity", "transpose"]
        assert contract["measurement"]["warmup_forwards"] == 2
        assert contract["measurement"]["measurement_forwards"] == 12
        assert contract["measurement"]["bootstrap_seed"] == 20261002
        assert contract["runtime_dynamiccache_preflight"]["physical_batch"] == 4
        assert contract["runtime_dynamiccache_preflight"]["timed"] is False
        with self.assertRaisesRegex(RuntimeError, "source commit"):
            runner.experiment_contract(source_commit="wrong")

    def test_lane_layout_is_deterministic_physical_replication_only(self) -> None:
        assert [(item["replica_index"], item["view"]) for item in runner.lane_specs(1)] == [(0, "anti_transpose")]
        assert [(item["replica_index"], item["view"]) for item in runner.lane_specs(2)] == [(0, "anti_transpose"), (0, "flip_ud")]
        assert [item["view"] for item in runner.lane_specs(4)] == list(runner.BASE_VIEWS)
        for width, replicas in ((8, 2), (12, 3), (16, 4)):
            lanes = runner.lane_specs(width)
            assert len(lanes) == width
            assert [item["lane_index"] for item in lanes] == list(range(width))
            assert max(item["replica_index"] for item in lanes) == replicas - 1

    def test_batched_widths_all_use_same_streaming_path(self) -> None:
        calls: list[dict] = []

        def fake_execute(**kwargs):
            calls.append(kwargs)
            return [SimpleNamespace(logits=None) for _ in kwargs["selected"]], {
                "cache_pack_seconds": 0.0, "model_call_seconds": 0.0,
                "cache_adoption_seconds": 0.0, "scheduler_elapsed_seconds": 0.0,
            }

        original = runner.execute_ready_forward
        try:
            runner.execute_ready_forward = fake_execute
            b1 = [SimpleNamespace(request=SimpleNamespace(cache_key="same", position=7))]
            runner._execute_once(model=object(), cells=b1)
            assert "streaming_split_and_adopt" not in calls[-1]
            for width in (2, 4, 8, 12, 16):
                cells = [SimpleNamespace(request=SimpleNamespace(cache_key="same", position=7)) for _ in range(width)]
                runner._execute_once(model=object(), cells=cells)
                assert calls[-1]["streaming_split_and_adopt"] is True
                assert calls[-1]["release_batch_temporaries_for_audit"] is True
        finally:
            runner.execute_ready_forward = original

    def test_base_model_identity_gate_requires_four_matching_workers(self) -> None:
        def ready(gpu: int, *, sha: str = "base") -> dict:
            return {"type": "READY", "gpu_id": gpu, "benchmark_model_mode": "BASE_MODEL_ONLY",
                    "benchmark_model_config_sha256": sha}

        rows = [ready(gpu) for gpu in range(4)]
        accepted = runner._ready_identity_gate(ready_rows=rows, expected_mode="BASE_MODEL_ONLY", prior_identity=None)
        assert accepted["status"] == "PASS"
        assert runner._ready_identity_gate(ready_rows=rows[:3] + [ready(3, sha="different")],
                                           expected_mode="BASE_MODEL_ONLY", prior_identity=None)["status"] == "INVALID_MIXED_BASE_MODEL_IDENTITY"
        widths = {width: {"ready": [ready(gpu) for gpu in range(4)]} for width in runner.WIDTHS}
        assert runner._global_identity_gate(all_widths=widths, expected_mode="BASE_MODEL_ONLY")["status"] == "PASS"

    def test_runner_excludes_adapter_and_target_behaviour(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for forbidden in ("from peft import", "import peft", "from torchao import", "import torchao", "evaluation_solutions", "submission.json", "kaggle kernels push"):
            assert forbidden not in source
        assert "AutoModelForCausalLM.from_pretrained" in source
        assert "torch_dtype=torch.bfloat16" in source
        assert "DYNAMICCACHE_PREFLIGHT.json" in source
        assert "for gpu_id in range(4)" in source
        assert "color_offset=0, pair_order=\"canonical\"" in source

    def test_bootstrap_preserves_gpu_identity(self) -> None:
        old = {gpu: [10.0, 10.5, 9.5] for gpu in range(4)}
        new = {gpu: [12.0, 12.5, 11.5] for gpu in range(4)}
        evidence = runner._bootstrap_ratio(old, new)
        assert evidence is not None
        assert evidence["trials"] == 10_000 and evidence["seed"] == 20261002
        assert evidence["ratio_ci95_low"] > 1.0

    def test_private_package_is_native_base_target_blind_and_not_launched(self) -> None:
        staged = Path(tempfile.mkdtemp()) / "stage"
        self.addCleanup(lambda: __import__("shutil").rmtree(staged.parent, ignore_errors=True))
        manifest = builder.build(output=staged, owner="private-owner", dataset_slug="private-source", kernel_slug="private-kernel")
        assert manifest["status"] == "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED"
        assert manifest["github_push"] == "BLOCKED_PENDING_TOKEN_ROTATION"
        notebook = json.loads((staged / "kernel" / "private-kernel.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][0]["source"])
        assert "BASE_MODEL_ONLY" in source
        assert "runtime_preflight" in source and "controller" in source
        assert "bootstrap_l4" not in source and "adapter_smoke" not in source
        assert "submission.json" not in source and "evaluation_solutions" not in source
        packaged = staged / "dataset" / "ARC2"
        assert (packaged / "scripts" / RUNNER_PATH.name).is_file()
        assert not (packaged / "src" / "inference" / "hf_peft_backend.py").exists()
        assert not list(packaged.rglob("__pycache__"))
        assert not list((staged / "dataset").rglob("*solution*.json"))


if __name__ == "__main__":
    unittest.main()
