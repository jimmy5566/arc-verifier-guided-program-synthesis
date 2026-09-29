from __future__ import annotations

import unittest

from arc.solution_normalization import normalize_arc_solutions


TASKS = ["a", "b"]
COUNTS = {"a": 1, "b": 2}
GRID0 = [[0, 1], [2, 3]]
GRID1 = [[4]]
GRID2 = [[5, 6]]


class TestD2SolutionNormalization(unittest.TestCase):
    def test_dict_form_normalizes_compact_and_task_object_values(self) -> None:
        raw = {"a": [GRID0], "b": {"test": [{"output": GRID1}, {"output": GRID2}]}}
        self.assertEqual(normalize_arc_solutions(raw, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS), {"a": [GRID0], "b": [GRID1, GRID2]})


    def test_list_form_uses_explicit_challenge_order(self) -> None:
        raw = [[GRID0], [GRID1, GRID2]]
        self.assertEqual(normalize_arc_solutions(raw, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS), {"a": [GRID0], "b": [GRID1, GRID2]})


    def test_list_length_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            normalize_arc_solutions([[GRID0]], task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


    def test_output_count_mismatch_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "output count mismatch"):
            normalize_arc_solutions({"a": [GRID0], "b": [GRID1]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


    def test_unknown_task_id_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not exactly match"):
            normalize_arc_solutions({"a": [GRID0], "unknown": [GRID1, GRID2]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)


    def test_duplicate_challenge_task_id_and_malformed_grid_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate"):
            normalize_arc_solutions([[GRID0], [GRID1, GRID2]], task_ids_in_challenge_order=["a", "a"], expected_output_counts={"a": 1})
        with self.assertRaisesRegex(ValueError, "rectangular"):
            normalize_arc_solutions({"a": [[[0], [1, 2]]], "b": [GRID1, GRID2]}, task_ids_in_challenge_order=TASKS, expected_output_counts=COUNTS)
