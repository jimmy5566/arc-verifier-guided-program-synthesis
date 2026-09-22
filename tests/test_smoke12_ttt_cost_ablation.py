from __future__ import annotations

from types import SimpleNamespace

import torch

from scripts.prepare_smoke12_ttt_reused_pools import _union_candidates
from scripts.run_smoke12_ttt_cost_ablation import _frozen_task_ids, _independent_cache_beams, _source_challenge_sha256


def test_smoke12_beam2_keeps_exact_aug8_transport_contract() -> None:
    source = open("scripts/run_smoke12_ttt_cost_ablation.py", encoding="utf-8").read()
    assert '"color_offsets": [0]' in source
    assert '"train_pair_orders": ["canonical"]' in source
    assert '"color_offsets": [0, 1]' not in source
    assert '"train_pair_orders": ["canonical", "reversed"]' not in source


def test_smoke12_beam2_restores_unsloth_generation_state_after_ttt() -> None:
    source = open("scripts/run_smoke12_ttt_cost_ablation.py", encoding="utf-8").read()
    beam_start = source.index("def _beam2_candidates")
    settings_start = source.index("    settings = {", beam_start)
    transition = "FastLanguageModel.for_inference(provider.model)"
    assert transition in source[beam_start:settings_start]


def test_smoke12_beam2_uses_independent_prefixes_without_legacy_cache_transport() -> None:
    class Tokenizer:
        eos_token_id = 15

        @staticmethod
        def apply_chat_template(*_args: object, **_kwargs: object) -> dict[str, torch.Tensor]:
            return {"input_ids": torch.tensor([[14]], dtype=torch.long)}

        @staticmethod
        def decode(token_ids: list[int], *, skip_special_tokens: bool) -> str:
            assert skip_special_tokens is True
            return "".join(str(token) for token in token_ids if token != 15)

    class Model:
        def __init__(self) -> None:
            self.calls: list[tuple[int, object | None]] = []

        def __call__(self, *, input_ids: torch.Tensor, past_key_values: object | None = None, **_kwargs: object) -> object:
            last = int(input_ids[0, -1].item())
            self.calls.append((last, past_key_values))
            logits = torch.full((1, 1, 16), -100.0)
            if last == 14:
                logits[0, 0, 1], logits[0, 0, 2] = 10.0, 9.0
            else:
                logits[0, 0, 15] = 10.0
            # This is deliberately a legacy tuple: generic BeamSearch tries
            # to call tuple.reorder_cache(), so the experimental decoder must
            # not pass it back to any child forward call.
            return SimpleNamespace(logits=logits, past_key_values=("legacy-cache", last))

    model = Model()
    provider = SimpleNamespace(model=model, tokenizer=Tokenizer(), device="cpu")
    beams, stats = _independent_cache_beams(
        provider, [{"role": "user", "content": "grid"}], max_new_tokens=8, context_window=64
    )

    assert [beam["text"] for beam in beams] == ["1", "2"]
    assert stats["backend"] == "independent_prefix_beam2_no_cache"
    assert all(cache is None for _token, cache in model.calls)


def test_smoke12_union_is_base_then_ttt_and_preserves_duplicate_provenance() -> None:
    base = [
        {"prediction": [[[1]]], "support_count": 1},
        {"prediction": [[[2]]], "support_count": 1},
    ]
    ttt = [
        {"prediction": [[[2]]], "support_count": 1},
        {"prediction": [[[3]]], "support_count": 1},
    ]
    union = _union_candidates(base, ttt)
    assert [item["prediction"] for item in union] == [[[[1]]], [[[2]]], [[[3]]]]
    assert union[0]["union_sources"] == ["base_aug8"]
    assert union[1]["union_sources"] == ["base_aug8", "strong_ttt_24_greedy"]
    assert union[2]["union_sources"] == ["strong_ttt_24_greedy"]


def test_smoke12_reused_pool_builder_is_target_blind() -> None:
    source = open("scripts/prepare_smoke12_ttt_reused_pools.py", encoding="utf-8").read().lower()
    assert "solutions_path" not in source
    assert "load_solutions" not in source
    assert "torch" not in source


def test_smoke12_protocol_manifest_uses_nested_selection_hash_contract() -> None:
    task_ids = [
        "5dbc8537", "97d7923e", "cb2d8a2c", "b99e7126", "7b3084d4", "dfadab01",
        "58490d8a", "142ca369", "446ef5d2", "b5ca7ac4", "80a900e0", "16de56c4",
    ]
    from scripts.run_eval3_reference_ttt import _task_hash

    selected, task_hash = _frozen_task_ids({"selection": {"task_ids": task_ids, "task_ids_hash": _task_hash(task_ids)}})
    assert selected == task_ids
    assert task_hash == _task_hash(task_ids)


def test_smoke12_protocol_manifest_resolves_nested_evaluation_challenge_hash() -> None:
    assert _source_challenge_sha256({
        "source_artifacts": {"eval60_manifest": {"source_challenge_sha256": "nested-hash"}},
    }) == "nested-hash"
