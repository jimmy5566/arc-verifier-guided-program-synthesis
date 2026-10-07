from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def module():
    path = ROOT / "scripts/run_foundation_v2_batch_recovery_v1.py"
    spec = importlib.util.spec_from_file_location("batch_recovery", path)
    assert spec and spec.loader
    loaded = importlib.util.module_from_spec(spec); spec.loader.exec_module(loaded)
    return loaded


def test_first_divergence_and_hash_are_exact() -> None:
    loaded = module()
    assert loaded._first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert loaded._first_divergence([1, 2], [1, 2]) is None
    assert loaded._first_divergence([1, 2], [1, 2, 3]) == 2
    assert loaded._tokens_hash([1, 2]) == loaded._tokens_hash([1, 2])


def test_batch_contract_and_no_full_generation() -> None:
    source = (ROOT / "scripts/run_foundation_v2_batch_recovery_v1.py").read_text(encoding="utf-8")
    assert "prompt[index, width-length:]" in source
    assert "mask[index, width-length:] = 1" in source
    assert "row[width:]" in source
    assert "do_sample=False" in source and "num_beams=1" in source and "use_cache=True" in source
    assert '"Gold accessed": False' in source and '"full generation started": False' in source
    assert "generate_state" not in source


def test_ladder_and_duplicate_sizes_are_frozen() -> None:
    loaded = module()
    assert loaded.BATCH_SIZES == (32, 28, 24, 20, 16, 12, 8, 4, 2)
    assert loaded.DUPLICATE_SIZES == (2, 4, 8, 16, 32)
    assert len(loaded.SENSITIVE) == 5
