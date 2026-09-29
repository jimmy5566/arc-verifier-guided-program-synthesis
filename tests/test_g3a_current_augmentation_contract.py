"""CPU contracts for the isolated current-adapter G3A surface."""
from __future__ import annotations

import argparse
import unittest

from scripts.run_eval60_bottleneck_attribution import D4_VIEWS, _g3a_expected_keys, _sharded_task_ids


class G3AContractTests(unittest.TestCase):
    def test_exact_surface_has_28_times_four_unique_ttt24_keys(self) -> None:
        rows = [
            {"task_id": f"{index:08x}", "output_index": index % 2, "output_id": f"{index:08x}:o{index % 2}"}
            for index in range(28)
        ]
        keys = _g3a_expected_keys(rows)
        self.assertEqual(len(keys), 112)
        self.assertEqual({depth for _, _, depth, _ in keys}, {24})
        self.assertEqual({view for _, _, _, view in keys}, set(D4_VIEWS))

    def test_bounded_smoke_is_a_prefix_of_the_same_deterministic_shard(self) -> None:
        task_ids = [f"{index:08x}" for index in range(24)]
        full = _sharded_task_ids(task_ids, argparse.Namespace(task_shard_index=2, task_shard_count=4, max_tasks_per_shard=None))
        smoke = _sharded_task_ids(task_ids, argparse.Namespace(task_shard_index=2, task_shard_count=4, max_tasks_per_shard=1))
        self.assertEqual(smoke, full[:1])
        self.assertEqual(set(full), set(task_ids[2::4]))

    def test_invalid_smoke_limit_is_rejected(self) -> None:
        args = argparse.Namespace(task_shard_index=0, task_shard_count=1, max_tasks_per_shard=0)
        with self.assertRaisesRegex(RuntimeError, "max_tasks_per_shard"):
            _sharded_task_ids(["aaaaaaaa"], args)


if __name__ == "__main__":
    unittest.main()
