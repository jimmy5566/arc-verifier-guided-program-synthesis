"""CPU-only guards for Speed V2 transport optimizations.

These tests deliberately use no Transformers import or CUDA device.  They
prove the batch interfaces preserve candidate ordering/likelihood aggregation
and that cached B-SUPPORT evidence prevents a serial model-scoring tail.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from arc.task import ARCExample, ARCGrid, ARCTask
from inference.native_multiview_likelihood import candidate_view_scores, candidate_view_scores_many
from inference.nvarc_native_augmentation import NativeAugmentation
from inference.nvarc_native_candidates import NativeGridCandidate, rank_candidates


ROOT = Path(__file__).resolve().parents[1]


class _BatchedProvider:
    def __init__(self) -> None:
        self.scalar_calls = 0
        self.batch_calls: list[tuple[int, int]] = []

    def continuation_log_likelihood(self, messages, continuation, *, context_window):
        self.scalar_calls += 1
        return -float(len(continuation) + len(messages) / 100.0)

    def continuation_log_likelihood_many(self, requests, *, context_window, batch_size):
        self.batch_calls.append((len(requests), batch_size))
        return [self.continuation_log_likelihood(messages, continuation, context_window=context_window) for messages, continuation in requests]


def _task() -> ARCTask:
    return ARCTask(
        "speed-v2-test",
        (ARCExample(ARCGrid([[0, 1]]), ARCGrid([[1, 0]])),),
        (ARCExample(ARCGrid([[2, 3]])), ARCExample(ARCGrid([[4]]))),
    )


def test_batched_candidate_likelihood_preserves_scalar_scores_and_order() -> None:
    task = _task()
    candidates = [
        NativeGridCandidate(NativeAugmentation(), (((0, 1),), ((2,),)), 2, 0.1),
        NativeGridCandidate(NativeAugmentation(color_offset=1), (((1, 0),), ((3,),)), 2, 0.1),
    ]
    messages = [[{"role": "user", "content": "x"}], [{"role": "user", "content": "y"}]]
    scalar = _BatchedProvider()
    batched = _BatchedProvider()
    old = rank_candidates(scalar, candidates, messages, context_window=100, likelihood_batch_size=1)
    new = rank_candidates(batched, candidates, messages, context_window=100, likelihood_batch_size=4)
    assert [(item.key(), score) for item, score in new] == [(item.key(), score) for item, score in old]
    assert batched.batch_calls == [(4, 4)]


def test_batched_multiview_likelihood_preserves_scalar_definition() -> None:
    task = _task()
    views = (NativeAugmentation(), NativeAugmentation(geometry="flip_lr"))
    predictions = [
        [[[0, 1]], [[2]]],
        [[[1, 0]], [[3]]],
    ]
    scalar = _BatchedProvider()
    batched = _BatchedProvider()
    expected = [candidate_view_scores(scalar, task, prediction, views, context_window=100) for prediction in predictions]
    actual = candidate_view_scores_many(batched, task, predictions, views, context_window=100, batch_size=4)
    assert actual == expected
    assert batched.batch_calls == [(8, 4)]


def test_cached_b_support_selection_needs_no_model_or_challenge_input(tmp_path: Path) -> None:
    views = [NativeAugmentation(geometry=geometry).to_dict() for geometry in ("identity", "rot90", "rot180", "rot270", "flip_lr", "flip_ud", "transpose", "anti_transpose")]
    source = {
        "status": "CANDIDATES_AND_RANKED_PREDICTIONS_FROZEN_BEFORE_EXACT_SCORING",
        "records": {
            "task": {
                "candidates": [
                    {"prediction": [[[0]]], "support_count": 3},
                    {"prediction": [[[1]]], "support_count": 1},
                ],
                "ranked_candidate_indices": [0, 1],
                "candidate_scores": [-0.2, -0.3],
                "b_support_view_spec": views,
                "b_support_evidence": [
                    {"candidate_index": 0, "original_log_likelihood": -0.2, "view_negative_log_likelihoods": [0.1] * 8},
                    {"candidate_index": 1, "original_log_likelihood": -0.3, "view_negative_log_likelihoods": [0.2] * 8},
                ],
            },
        },
    }
    frozen, output = tmp_path / "frozen.json", tmp_path / "selection.json"
    frozen.write_text(json.dumps(source), encoding="utf-8")
    subprocess.run([
        sys.executable, str(ROOT / "scripts/rerank_native_public_reference_selection.py"),
        "--frozen", str(frozen), "--challenge-path", str(tmp_path / "absent_challenges.json"),
        "--model-path", str(tmp_path / "absent_model"), "--native-config-dir", str(tmp_path / "absent_native_config"),
        "--output", str(output),
    ], check=True)
    selected = json.loads(output.read_text(encoding="utf-8"))
    record = selected["records"]["task"]["public_reference_selection"]
    assert selected["public_reference_evidence_mode"] == "task_local_inline_cache"
    assert record["evidence_source"] == "task_local_inline_cache"
    assert record["attempt_candidate_indices"] == [0, 1]


def test_speed_v2_runner_keeps_b_support_task_local_and_no_per_candidate_empty_cache() -> None:
    provider = (ROOT / "src/inference/nvarc_native.py").read_text(encoding="utf-8")
    runner = (ROOT / "scripts/run_qwen4b_native_augmentation_search.py").read_text(encoding="utf-8")
    reranker = (ROOT / "scripts/rerank_native_public_reference_selection.py").read_text(encoding="utf-8")
    assert "def generate_many" in provider and "def continuation_log_likelihood_many" in provider
    assert provider.count("torch.cuda.empty_cache()") == 0
    assert "candidate_view_scores_many" in runner and '"b_support_evidence"' in runner
    assert '"generation_micro_batch_size"' in runner and '"likelihood_micro_batch_size"' in runner
    assert "task_local_inline_cache" in reranker and "needs_model" in reranker
