from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts.run_5090_blackwell_unleashed_phase1_queue import _json_digest, _load_queue, _resolved_config, _telemetry
from scripts.run_eval3_runtime_opt import _generated_suffix


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_phase1_queue_has_exactly_the_authorized_two_gpu_schedule() -> None:
    queue, runs = _load_queue(ROOT / "governance" / "queues" / "5090-unleashed-phase1-v1.json")
    assert queue["queue_id"] == "5090-unleashed-phase1-v1"
    assert [(item.physical_gpu_id, item.queue_position, item.generation_micro_batch_size) for item in runs] == [
        (0, 1, 1), (1, 1, 2), (0, 2, 4), (1, 2, 4),
    ]
    assert all(item.mode == "serial" for item in runs)


def test_single_gpu_amendment_keeps_all_four_batch_conditions_on_gpu_zero() -> None:
    queue, runs = _load_queue(ROOT / "governance" / "queues" / "5090-unleashed-phase1-single-gpu-v1.json")
    assert queue["execution_policy"] == "single_gpu_ephemeral_offpod_backup"
    assert [item.physical_gpu_id for item in runs] == [0, 0, 0, 0]
    assert [item.generation_micro_batch_size for item in runs] == [1, 2, 4, 4]
    assert [item.queue_position for item in runs] == [1, 2, 3, 4]


def test_phase1_reference_config_has_portable_historical_provenance() -> None:
    config_path = ROOT / "governance" / "benchmarks" / "legacy-eval3" / "reference_ttt_config_frozen.json"
    provenance_path = config_path.with_name("reference_ttt_config_provenance.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    assert provenance["historical_export_file_sha256"] == "63379036f4881f8649a2a61b003d60b3af5977743ad0472301fed307f1e4dc26"
    assert _json_digest(config) == provenance["canonical_json_sha256"]


def test_queue_and_cpu_scorer_bootstrap_their_staged_src_tree() -> None:
    for relative in (
        "scripts/run_5090_blackwell_unleashed_phase1_queue.py",
        "scripts/score_eval3_blackwell_unleashed_phase1.py",
    ):
        source = (ROOT / relative).read_text(encoding="utf-8")
        assert 'sys.path.insert(0, str(ROOT / "src"))' in source


def test_batched_suffix_keeps_eos_and_drops_only_rectangular_completion_pad() -> None:
    sequence = torch.tensor([0, 11, 12, 13, 15, 0, 0])
    assert _generated_suffix(sequence, input_width=2, eos_token_id=15, pad_token_id=0).tolist() == [12, 13, 15]
    without_eos = torch.tensor([0, 11, 12, 13, 0, 0])
    assert _generated_suffix(without_eos, input_width=2, eos_token_id=15, pad_token_id=0).tolist() == [12, 13]


def test_resolved_config_changes_only_execution_ptxas_and_backend_metadata() -> None:
    _queue, runs = _load_queue(ROOT / "governance" / "queues" / "5090-unleashed-phase1-v1.json")
    frozen = {"rank": 256, "alpha": 32, "ttt_steps": 24, "ptxas_path": "/historical/ptxas", "seed": 42}
    resolved, scientific_hash = _resolved_config(frozen_config=frozen, ptxas_path=Path("/usr/local/cuda-12.8/bin/ptxas"), spec=runs[1])
    unchanged = {key: value for key, value in resolved.items() if key not in {"ptxas_path", "execution_backend", "generation_micro_batch_size", "execution_mode"}}
    assert unchanged == {key: value for key, value in frozen.items() if key != "ptxas_path"}
    assert resolved["ptxas_path"] == "/usr/local/cuda-12.8/bin/ptxas"
    assert resolved["generation_micro_batch_size"] == 2
    assert scientific_hash


def test_phase1_telemetry_retains_padding_and_stage_fields() -> None:
    records = {
        "task": {
            "generation_micro_batch_size": 4,
            "max_ttt_sequence_tokens": 128,
            "invalid_candidate_count": 0,
            "attention_backend": "xformers",
            "KV_CACHE": {"KV_CACHE_ACTIVE": True},
            "telemetry": {
                "ttt": {"seconds": 2.0},
                "generation": {"seconds": 1.0, "generated_token_count": 20, "generation_tokens_per_second": 20.0, "padding_tokens": 3, "padding_ratio": 0.1},
                "whole_task": {"seconds": 3.0},
            },
        }
    }
    value = _telemetry(records)
    assert value["tasks"][0]["padding_tokens"] == 3
    assert value["tasks"][0]["KV_CACHE_ACTIVE"] is True
    assert value["aggregate"]["generation_tokens_per_second"] == 20.0
