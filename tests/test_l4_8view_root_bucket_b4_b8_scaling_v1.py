from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = ROOT / "scripts" / "run_l4_8view_root_bucket_b4_b8_scaling_v1.py"
BUILDER_PATH = ROOT / "scripts" / "build_l4_8view_root_bucket_b4_b8_scaling_v1_kaggle.py"
OLD_RUNNER_PATH = ROOT / "scripts" / "run_l4_native_base_physical_batch_scaling_b1_b16_v1.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


runner = _load(RUNNER_PATH, "l4_8view_root_bucket_runner")
builder = _load(BUILDER_PATH, "l4_8view_root_bucket_builder")


class _Queue:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def cancel_join_thread(self) -> None:
        self.calls.append("cancel_join_thread")

    def close(self) -> None:
        self.calls.append("close")

    def join_thread(self) -> None:
        raise AssertionError("unbounded Queue.join_thread must never be called")


class L48ViewRootBucketB4B8ScalingTests(unittest.TestCase):
    def test_contract_uses_real_fixed_4plus4_sources_not_synthetic_aug8(self) -> None:
        slots = runner.production_view_slots()
        assert [(slot.source, slot.geometry) for slot in slots] == [
            ("TTT24", "flip_lr"), ("TTT24", "flip_ud"), ("TTT24", "transpose"), ("TTT24", "anti_transpose"),
            ("TTT48", "identity"), ("TTT48", "rot90"), ("TTT48", "flip_ud"), ("TTT48", "anti_transpose"),
        ]
        contract = runner.experiment_contract(source_commit=runner.AUTHORITATIVE_SOURCE_COMMIT)
        assert contract["experiment"] == "L4_8VIEW_ROOT_BUCKET_B4_B8_SCALING_V1"
        assert contract["test_widths"] == [4, 8]
        assert contract["measurement"] == {**contract["measurement"], "warmup_forwards": 2, "measurement_forwards": 12}
        assert contract["bucket_requirement"]["bucket_count"] == 2
        assert contract["peft_used"] is False and contract["ttt_used"] is False

    def test_bucket_gate_requires_exactly_two_groups_of_four(self) -> None:
        signature_a = (10, 9, ("a",), ((1, "bf16"),))
        signature_b = (20, 19, ("b",), ((2, "bf16"),))
        fake = {slot.slot_id: object() for slot in runner.production_view_slots()}
        mapping = {slot: signature_a if index < 4 else signature_b for index, slot in enumerate(fake)}
        with mock.patch.object(runner, "compatibility_signature", side_effect=lambda template: mapping[next(key for key, value in fake.items() if value is template)]):
            buckets = runner.discover_root_buckets(fake)
        self.assertEqual(sorted(len(bucket.slots) for bucket in buckets), [4, 4])
        runner.validate_root_buckets(buckets)
        with self.assertRaisesRegex(RuntimeError, "8VIEW_BUCKET_ASSUMPTION_FAILED"):
            runner.validate_root_buckets((buckets[0],))

    def test_b4_is_real_bucket_and_b8_is_two_replicas_only(self) -> None:
        bucket = runner.RootBucket("A", tuple(), "hash", ("a", "b", "c", "d"))
        b4 = runner.lane_specs(bucket, 4)
        b8 = runner.lane_specs(bucket, 8)
        assert [row["view"] for row in b4] == ["a", "b", "c", "d"]
        assert [row["replica_index"] for row in b8] == [0, 0, 0, 0, 1, 1, 1, 1]
        assert [row["lane_index"] for row in b8] == list(range(8))

    def test_failure_queue_cleanup_has_no_unbounded_join_thread(self) -> None:
        queue = _Queue()
        runner._close_queue(queue)
        assert queue.calls == ["cancel_join_thread", "close"]
        old = _load(OLD_RUNNER_PATH, "l4_native_base_runner_queue_cleanup")
        old_queue = _Queue()
        old._close_queue(old_queue)
        assert old_queue.calls == ["cancel_join_thread", "close"]

    def test_time_gate_and_orphan_protection_are_present_without_science_drift(self) -> None:
        source = RUNNER_PATH.read_text(encoding="utf-8")
        old = OLD_RUNNER_PATH.read_text(encoding="utf-8")
        for token in ("MODEL_READY_LIMIT_SECONDS = 300", "NO_PROGRESS_LIMIT_SECONDS = 180",
                      "WIDTH_HARD_LIMIT_SECONDS = 600", "GLOBAL_NOTEBOOK_BENCHMARK_LIMIT_SECONDS = 1800",
                      "8VIEW_BUCKET_ASSUMPTION_FAILED", "COMPATIBLE_LANE_HARDWARE_SCALING"):
            assert token in source
        assert "PR_SET_PDEATHSIG" in old
        assert "torch.bfloat16" in source
        assert "from peft import" not in source and "evaluation_solutions" not in source and "submission.json" not in source

    def test_generated_notebook_has_bounded_process_group_popen(self) -> None:
        source = builder._notebook_source("review-sha")
        for token in ("subprocess.Popen", "start_new_session=True", "os.killpg", "SIGTERM_PROCESS_GROUP",
                      "SIGKILL_PROCESS_GROUP", "process.wait(timeout=10)", "TIME_GATE_FAILURE.json",
                      "GLOBAL_LIMIT = 1800", "MODEL_READY_LIMIT = 300", "NO_PROGRESS_LIMIT = 180",
                      "WIDTH_HARD_LIMIT = 600"):
            assert token in source
        assert "subprocess.run(command, env=env)" not in source
        assert "submission.json" not in source and "KAGGLE_IS_COMPETITION_RERUN" not in source

    def test_runtime_bucket_artifact_includes_all_required_view_provenance(self) -> None:
        slots = runner.production_view_slots()
        records = [{"slot_id": slot.slot_id, "source": slot.source, "geometry": slot.geometry,
                    "bucket_id": "A" if index < 4 else "B", "root_length": 10 if index < 4 else 20,
                    "position": 9 if index < 4 else 19, "cache_key": ["a"], "cache_geometry": [],
                    "compatibility_signature_sha256": "a" if index < 4 else "b"}
                   for index, slot in enumerate(slots)]
        ready = [{"bucket_layout_sha256": "layout", "view_records": records,
                  "buckets": [{"bucket_id": "A", "signature_sha256": "a", "slots": [slot.slot_id for slot in slots[:4]]},
                              {"bucket_id": "B", "signature_sha256": "b", "slots": [slot.slot_id for slot in slots[4:]]}]}
                 for _ in range(4)]
        output = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: shutil.rmtree(output, ignore_errors=True))
        runner._write_full_bucket_artifact_from_ready(output, ready)
        artifact = json.loads((output / "EIGHT_VIEW_ROOT_BUCKETS.json").read_text(encoding="utf-8"))
        assert artifact["status"] == "PASS"
        assert artifact["root_length_bucket_counts"] == [4, 4]
        assert len(artifact["views"]) == 8
        assert all({"root_length", "position", "cache_key", "cache_geometry", "bucket_id"} <= set(row) for row in artifact["views"])

    def test_private_package_is_target_blind_and_not_launched(self) -> None:
        staged = Path(tempfile.mkdtemp()) / "stage"
        self.addCleanup(lambda: shutil.rmtree(staged.parent, ignore_errors=True))
        manifest = builder.build(output=staged, owner="private-owner", dataset_slug="private-source", kernel_slug="private-kernel")
        assert manifest["status"] == "PACKAGE_BUILT_NOT_PUSHED_NOT_LAUNCHED"
        notebook = json.loads((staged / "kernel" / "private-kernel.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][0]["source"])
        assert "start_new_session=True" in source and "TIME_GATE_FAILURE.json" in source
        assert "submission.json" not in source and "evaluation_solutions" not in source
        assert (staged / "dataset" / "ARC2" / "scripts" / RUNNER_PATH.name).is_file()
        assert (staged / "dataset" / "ARC2" / "scripts" / OLD_RUNNER_PATH.name).is_file()
        assert (staged / "dataset" / "ARC2" / "src" / "inference" / "d1_release_contract.py").is_file()
        assert not list(staged.rglob("*solution*.json"))


if __name__ == "__main__":
    unittest.main()
