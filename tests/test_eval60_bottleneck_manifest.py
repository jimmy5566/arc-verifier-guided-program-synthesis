"""Regression tests for frozen G2 cohort-manifest handling.

These tests cover parsing and deterministic provenance only; they never load a
model or touch CUDA.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_eval60_bottleneck_attribution import prepare_miss_set


FIELDS = [
    "task_id", "output_index", "output_id", "greedy_pool_hit",
    "v5_pool_gold_hit_available", "union_greedy_v5_hit",
    "original_v5_oracle_certainty",
]


def make_rows(count: int, *, start: int = 0) -> list[dict[str, str]]:
    return [
        {
            "task_id": f"{index:08x}",
            "output_index": "0",
            "output_id": f"{index:08x}:o0",
            "greedy_pool_hit": "False",
            "v5_pool_gold_hit_available": "True",
            "union_greedy_v5_hit": "False",
            "original_v5_oracle_certainty": "TRUSTWORTHY_LOWER_BOUND_MISS",
        }
        for index in range(start, start + count)
    ]


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def make_args(root: Path, manifest: Path, cohort_output_count: int | None) -> argparse.Namespace:
    return argparse.Namespace(
        frozen_miss_manifest=manifest,
        cohort_output_count=cohort_output_count,
        scratch=root / "scratch",
        report_dir=root / "reports",
        task_shard_index=0,
        task_shard_count=1,
    )


class FrozenG2ManifestContractTests(unittest.TestCase):
    def new_root(self) -> tuple[tempfile.TemporaryDirectory[str], Path]:
        temporary = tempfile.TemporaryDirectory()
        return temporary, Path(temporary.name)

    def test_full_frozen_56_manifest_without_cohort_count_is_accepted(self) -> None:
        temporary, root = self.new_root()
        with temporary:
            manifest = root / "full.csv"
            source = make_rows(56)
            write_manifest(manifest, source)
            selected = prepare_miss_set(make_args(root, manifest, None))
            self.assertEqual([row["output_id"] for row in selected], [row["output_id"] for row in source])
            state = json.loads((root / "scratch" / "manifest" / "miss_set.json").read_text())
            self.assertEqual(state["source"], "FROZEN_G0_MANIFEST")
            self.assertEqual(state["selection"], "ALL")

    def test_full_56_manifest_with_cohort_28_derives_sha256_prefix(self) -> None:
        temporary, root = self.new_root()
        with temporary:
            manifest = root / "full.csv"
            source = make_rows(56)
            write_manifest(manifest, source)
            selected = prepare_miss_set(make_args(root, manifest, 28))
            expected = sorted(
                source,
                key=lambda row: (hashlib.sha256(row["output_id"].encode()).hexdigest(), row["output_id"]),
            )[:28]
            self.assertEqual([row["output_id"] for row in selected], [row["output_id"] for row in expected])
            state = json.loads((root / "scratch" / "manifest" / "miss_set.json").read_text())
            self.assertEqual(state["selection"], "SHA256_OUTPUT_ID_ASCENDING_PREFIX")

    def test_preregistered_28_manifest_is_accepted_without_reordering(self) -> None:
        temporary, root = self.new_root()
        with temporary:
            manifest = root / "half.csv"
            source = list(reversed(make_rows(28)))
            write_manifest(manifest, source)
            selected = prepare_miss_set(make_args(root, manifest, 28))
            self.assertEqual([row["output_id"] for row in selected], [row["output_id"] for row in source])
            state = json.loads((root / "scratch" / "manifest" / "miss_set.json").read_text())
            self.assertEqual(state["source"], "FROZEN_MANIFEST_ALREADY_SUBSET_28")
            self.assertEqual(state["selection"], "ALREADY_FROZEN_SUBSET")

    def test_wrong_preregistered_subset_size_is_rejected(self) -> None:
        for count in (27, 29):
            with self.subTest(count=count):
                temporary, root = self.new_root()
                with temporary:
                    manifest = root / "wrong.csv"
                    write_manifest(manifest, make_rows(count))
                    with self.assertRaisesRegex(RuntimeError, "does not match cohort_output_count"):
                        prepare_miss_set(make_args(root, manifest, 28))

    def test_duplicate_output_id_is_rejected(self) -> None:
        temporary, root = self.new_root()
        with temporary:
            manifest = root / "duplicate.csv"
            source = make_rows(28)
            source[-1]["output_id"] = source[0]["output_id"]
            write_manifest(manifest, source)
            with self.assertRaisesRegex(RuntimeError, "duplicate output_id"):
                prepare_miss_set(make_args(root, manifest, 28))

    def test_union_hit_in_frozen_miss_manifest_is_rejected(self) -> None:
        temporary, root = self.new_root()
        with temporary:
            manifest = root / "hit.csv"
            source = make_rows(28)
            source[0]["union_greedy_v5_hit"] = "True"
            write_manifest(manifest, source)
            with self.assertRaisesRegex(RuntimeError, "union Greedy/V5 hit"):
                prepare_miss_set(make_args(root, manifest, 28))


if __name__ == "__main__":
    unittest.main()
