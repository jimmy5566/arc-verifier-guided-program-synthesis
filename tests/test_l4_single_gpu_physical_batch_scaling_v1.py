from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_l4_single_gpu_physical_batch_scaling_v1.py"
BUILDER_PATH = ROOT / "scripts" / "build_l4_single_gpu_physical_batch_scaling_v1_kaggle.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load(RUNNER_PATH, "l4_single_gpu_batch_runner")
builder = _load(BUILDER_PATH, "l4_single_gpu_batch_builder")


class L4SingleGpuPhysicalBatchScalingV1Tests(unittest.TestCase):
    def test_frozen_single_gpu_contract(self) -> None:
        contract = builder.experiment_contract(source_commit="review")
        self.assertEqual(runner.EXPERIMENT, "L4_SINGLE_GPU_PHYSICAL_BATCH_SCALING_V1")
        self.assertEqual(runner.WIDTHS, (1, 2, 4, 8, 12, 16))
        self.assertEqual((runner.WARMUPS, runner.MEASUREMENTS), (2, 12))
        self.assertEqual(contract["hardware"]["gpu_count_used_for_timing"], 1)
        self.assertEqual(contract["model"]["model_load_count"], 1)
        self.assertEqual(contract["root"]["prefill_count"], 1)
        self.assertEqual(contract["model"]["mode"], "BASE_MODEL_ONLY")
        self.assertFalse(contract["ttt_used"] or contract["peft_used"] or contract["gold_loaded"] or contract["submission_created"])

    def test_every_lane_is_a_same_root_identity_clone(self) -> None:
        for width in runner.WIDTHS:
            specs = runner._specs(width)
            self.assertEqual(len(specs), width)
            self.assertEqual([row["lane_index"] for row in specs], list(range(width)))
            self.assertEqual({row["view"] for row in specs}, {"identity"})
            self.assertEqual([row["replica_index"] for row in specs], list(range(width)))

    def test_single_load_single_prefill_and_timing_boundary_are_explicit(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("physical.start_ready_cell("), 1)
        self.assertIn('os.environ["CUDA_VISIBLE_DEVICES"] = "0"', source)
        self.assertIn('args.device = "cuda:0"', source)
        sample = source[source.index("def _one_forward"):source.index("def _measure_width")]
        self.assertLess(sample.index("physical._clone_template_lane"), sample.index("started = time.perf_counter()"))
        self.assertLess(sample.index("torch.cuda.synchronize(args.device); torch.cuda.reset_peak_memory_stats"), sample.index("started = time.perf_counter()"))
        self.assertIn("ROOT_TEMPLATE_MUTATED", source)
        self.assertIn("ROOT_TEMPLATE_GPU_TENSOR_COUNT", source)

    def test_ratio_calculation_and_capacity_policy(self) -> None:
        rows = [
            {"physical_batch": 1, "status": "PASS", "median_lanes_s": 10.0},
            {"physical_batch": 2, "status": "PASS", "median_lanes_s": 18.0},
            {"physical_batch": 4, "status": "PASS", "median_lanes_s": 30.0},
            {"physical_batch": 8, "status": "PASS", "median_lanes_s": 45.0},
            {"physical_batch": 12, "status": "CLEAN_CAPACITY_FAILURE"},
            {"physical_batch": 16, "status": "CLEAN_CAPACITY_FAILURE"},
        ]
        ratios = {row["ratio"]: row["value"] for row in runner._ratio_rows(rows)}
        self.assertEqual(ratios["B8_OVER_B4"], 1.5)
        self.assertIsNone(ratios["B12_OVER_B8"])
        self.assertTrue(runner._is_clean_capacity_width(12))
        self.assertTrue(runner._is_clean_capacity_width(16))
        self.assertFalse(runner._is_clean_capacity_width(8))

    def test_time_and_scope_guards_are_frozen(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        for token in ("GLOBAL_LIMIT, MODEL_READY_LIMIT, NO_PROGRESS_LIMIT, WIDTH_LIMIT = 6900, 600, 180, 600", "MODEL_READY_TIME_GATE", "NO_PROGRESS_TIME_GATE", "WIDTH_TIME_GATE", "DYNAMICCACHE_PREFLIGHT"):
            self.assertIn(token, source)
        for forbidden in ("from peft import", "import peft", "evaluation_solutions", "submission.json", "kaggle kernels push"):
            self.assertNotIn(forbidden, source)

    def test_generated_notebook_metadata_and_contract_compile(self) -> None:
        staged = Path(tempfile.mkdtemp()) / "stage"
        self.addCleanup(lambda: shutil.rmtree(staged.parent, ignore_errors=True))
        manifest = builder.build(output=staged, owner="private-owner", dataset_slug="private-source", kernel_slug="private-kernel")
        self.assertEqual(manifest["status"], "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED")
        notebook = json.loads((staged / "kernel" / "private-kernel.ipynb").read_text(encoding="utf-8"))
        metadata = json.loads((staged / "kernel" / "kernel-metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(notebook["metadata"]["kaggle"]["accelerator"], "NvidiaL4")
        self.assertEqual(metadata["machine_shape"], "NvidiaL4")
        source = "".join(notebook["cells"][0]["source"])
        compile(source, "single_gpu_generated.ipynb", "exec")
        self.assertIn("CUDA_VISIBLE_DEVICES':'0", source)
        self.assertIn("start_new_session=True", source)
        self.assertIn("cleanup_process_group", source)
        self.assertNotIn("submission.json", source)
        self.assertNotIn("evaluation_solutions", source)
        self.assertTrue((staged / "dataset" / "ARC2" / "scripts" / RUNNER_PATH.name).is_file())


if __name__ == "__main__":
    unittest.main()
