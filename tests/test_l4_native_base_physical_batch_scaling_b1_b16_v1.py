from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py"
BUILDER_PATH = ROOT / "scripts" / "build_l4_native_base_physical_batch_scaling_b1_b16_v1_kaggle.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
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

    def test_root_template_prefill_plan_is_frozen_and_not_per_sample(self) -> None:
        assert {width: runner.expected_template_prefills(width) for width in runner.WIDTHS} == {
            1: 1, 2: 2, 4: 4, 8: 4, 12: 4, 16: 4,
        }
        assert runner.expected_controller_prefills() == 19
        assert runner.expected_controller_prefills() + 4 == 23
        assert runner.WARMUP_FORWARDS == 2
        assert runner.MEASUREMENT_FORWARDS == 12

    def test_build_templates_prefills_each_distinct_view_once(self) -> None:
        calls: list[str] = []
        fake_template = runner.FrozenRootTemplate(
            view="", cell_key="k", token_id=1, position=2, cache_key=("same",), parent_node_id=0,
            ordinal=0, legacy_cache=((object(),),), sequence_length=2,
        )

        def fake_start(**kwargs):
            calls.append(kwargs["cell_key"])
            return SimpleNamespace(request=object(), prefill_seconds=0.25)

        def fake_template_from_cell(cell, *, view):
            return fake_template.__class__(view=view, cell_key=f"{view}:k", token_id=1, position=2,
                                           cache_key=("same",), parent_node_id=0, ordinal=0,
                                           legacy_cache=fake_template.legacy_cache, sequence_length=2)

        output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(output, ignore_errors=True))
        args = SimpleNamespace(task_id="d59b0160", output_index=0, depth=24, output=output,
                               run_started_unix=0.0, phase="controller")
        with mock.patch.object(runner, "start_ready_cell", side_effect=fake_start), \
                mock.patch.object(runner, "_template_from_cell", side_effect=fake_template_from_cell), \
                mock.patch.object(runner, "_cache_tensor_hash", return_value="root"), \
                mock.patch.object(runner, "_emit_worker_progress"):
            templates, parity_cells, audit = runner._build_root_templates(
                model=object(), prompts={view: object() for view in runner.BASE_VIEWS}, config=object(), args=args,
                width=16, torch=object(), physical_gpu_id=0, progress=object(),
            )
        assert list(templates) == list(runner.BASE_VIEWS)
        assert len(parity_cells) == 4
        assert len(calls) == 4
        assert audit["actual_template_prefills"] == 4

    def test_template_clones_have_independent_owner_and_dynamiccache_identities(self) -> None:
        template = runner.FrozenRootTemplate(
            view="anti_transpose", cell_key="root", token_id=1, position=2, cache_key=("same",),
            parent_node_id=0, ordinal=0, legacy_cache=((object(),),), sequence_length=2,
        )
        with mock.patch.object(runner, "_clone_legacy_tensors", side_effect=lambda _cache: ((object(),),)), \
                mock.patch.object(runner, "dynamic_cache_from_legacy", side_effect=lambda _legacy: object()):
            left = runner._clone_template_lane(template, spec={"replica_index": 0, "lane_index": 0})
            right = runner._clone_template_lane(template, spec={"replica_index": 1, "lane_index": 1})
        assert left.request.cache_owner is not right.request.cache_owner
        assert left.request.cache_owner.cache is not right.request.cache_owner.cache
        assert left.request.token_id == right.request.token_id == template.token_id
        assert left.request.cache_key == right.request.cache_key == template.cache_key

    def test_progress_jsonl_is_valid_jsonl(self) -> None:
        output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(output, ignore_errors=True))
        args = SimpleNamespace(output=output, run_started_unix=0.0, phase="controller")
        row = runner.emit_progress(args, "SAMPLE_DONE", physical_batch=4, gpu_id=1, sample_index=0)
        stored = [json.loads(line) for line in (output / "PROGRESS.jsonl").read_text(encoding="utf-8").splitlines()]
        assert stored == [row]

    def test_engineering_smoke_is_explicit_and_distinct_from_science(self) -> None:
        original = sys.argv[:]
        try:
            sys.argv = ["runner", "--phase", "preflight", "--output", tempfile.mkdtemp(), "--harness-commit", "x", "--engineering-smoke"]
            args = runner.parse_args()
        finally:
            sys.argv = original
        assert args.engineering_smoke is True
        assert (args.warmup_forwards, args.measurement_forwards) == (1, 2)
        assert runner.WIDTHS == (1, 2, 4, 8, 12, 16)

    def test_operational_guards_are_not_scientific_changes(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        assert "NO_PROGRESS_TIMEOUT_SECONDS = 300" in source
        assert "WIDTH_HARD_TIMEOUT_SECONDS = 900" in source
        assert "timeout=10" in source
        assert "def _nvidia_snapshot" in source
        assert "GPU_WORKER_COULD_NOT_BE_KILLED" in source
        assert "FAILURE.json" in source
        assert "B{width}_PARTIAL.json" in source
        assert "ROOT_TEMPLATE_CLONE_PARITY_FAILED" in source
        assert "CONTROLLER_ROOT_TEMPLATE_PREFILL_COUNT_MISMATCH" in source

    def test_notebook_uses_unbuffered_streaming_subprocesses(self) -> None:
        notebook_source = builder._notebook_source("private-source", "abc")
        assert "PYTHONUNBUFFERED':'1" in notebook_source
        assert "sys.executable, '-u'" in notebook_source
        assert "NOTEBOOK_START" in notebook_source
        assert "NOTEBOOK_PHASE_ERROR" in notebook_source

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
                    "benchmark_model_config_sha256": sha, "startup_milestone": "MODEL_READY_SENT"}

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
        assert "dynamic_cache_from_legacy" in source
        assert "DynamicCache.from_legacy_cache" not in source
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
        assert "source = dataset / 'ARC2'" in source
        assert "bootstrap_l4" not in source and "adapter_smoke" not in source
        assert "submission.json" not in source and "evaluation_solutions" not in source
        packaged = staged / "dataset" / "ARC2"
        assert (packaged / "scripts" / RUNNER_PATH.name).is_file()
        assert (packaged / "src" / "inference" / "nvarc_turbodfs_dynamic_ready.py").is_file()
        assert not (packaged / "src" / "inference" / "hf_peft_backend.py").exists()
        assert not list(packaged.rglob("__pycache__"))
        assert not list((staged / "dataset").rglob("*solution*.json"))


if __name__ == "__main__":
    unittest.main()
