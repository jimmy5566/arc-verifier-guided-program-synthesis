from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts/run_foundation_v2_b32_root_cause_forensic_v1.py"
    spec = importlib.util.spec_from_file_location("b32_forensic", path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def test_scope_is_b32_only() -> None:
    loaded = module()
    source = (ROOT / "scripts/run_foundation_v2_b32_root_cause_forensic_v1.py").read_text(encoding="utf-8")
    assert loaded.NORMAL_MAX_BATCH == 32
    assert "BATCH_SIZES" not in source
    assert '"lower_batch_ladder": "NOT_RUN_BY_PRIORITY_CHANGE"' in source
    assert '"Gold accessed": False' in source
    assert '"full 6000 generation started": False' in source


def test_generation_contract_and_oom_recovery() -> None:
    source = (ROOT / "scripts/run_foundation_v2_b32_root_cause_forensic_v1.py").read_text(encoding="utf-8")
    assert 'prompt[index, width - length :]' in source
    assert 'mask[index, width - length :] = 1' in source
    assert 'row[width:]' in source
    assert "do_sample=False" in source and "num_beams=1" in source and "use_cache=True" in source
    assert "except torch.cuda.OutOfMemoryError" in source
    assert "gc.collect()" in source and "torch.cuda.empty_cache()" in source


def test_required_conditions_and_classifications_exist() -> None:
    source = (ROOT / "scripts/run_foundation_v2_b32_root_cause_forensic_v1.py").read_text(encoding="utf-8")
    for name in (
        "B1_PADDED_TO_W",
        "BN_SELF_NATIVE",
        "BN_SELF_PADDED_TO_W",
        "BN_ORIGINAL_PEERS",
        "BN_ORIGINAL_PEERS_REPEAT",
        "BN_ORIGINAL_PEERS_REVERSED",
        "PADDING_WIDTH_EFFECT",
        "BATCH_DIMENSION_NUMERICAL_EFFECT",
        "HETEROGENEOUS_BATCH_NUMERICAL_EFFECT",
        "ORDER_SENSITIVE_BATCH_EFFECT",
        "NUMERICAL_NEAR_TIE",
        "UNRESOLVED",
    ):
        assert name in source


def test_first_divergence() -> None:
    loaded = module()
    assert loaded._first_divergence([1, 2], [1, 3]) == 1
    assert loaded._first_divergence([1], [1]) is None
    assert loaded._first_divergence([1], [1, 2]) == 1
