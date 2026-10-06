from scripts.model_training.verify_dots3_arc_programs import lint, run_program, valid_grid, split_family
import numpy as np

INPUT_CODE = r'''
def generate_input(seed: int):
    rng = np.random.default_rng(seed)
    h = 5 + seed % 4
    w = 6 + (seed // 3) % 4
    g = np.zeros((h, w), dtype=np.int8)
    y = 1 + seed % (h - 2)
    x = 1 + (seed * 3) % (w - 2)
    g[y, x] = 2
    g[y, min(w-1, x+1)] = 2
    g[0, seed % w] = 4
    return g
'''
OUTPUT_CODE = r'''
def transform(grid):
    out = np.fliplr(grid.copy())
    out[out == 2] = 8
    return out
'''

def test_program_contract():
    lint(INPUT_CODE); lint(OUTPUT_CODE)
    result=run_program(INPUT_CODE,OUTPUT_CODE,list(range(12)),5)
    assert result["ok"], result
    assert len(result["pairs"])==12
    assert all(valid_grid(np.asarray(p["input"])) and valid_grid(np.asarray(p["output"])) for p in result["pairs"])
    assert len({str(p["input"]) for p in result["pairs"]}) > 8
    assert all(p["input"] != p["output"] for p in result["pairs"])

def test_family_split_is_deterministic():
    assert split_family("F00__a__b") == split_family("F00__a__b")

def test_sandbox_rejects_imports():
    try:
        lint("import os\ndef generate_input(seed): return np.zeros((2,2), dtype=np.int8)")
    except ValueError:
        return
    raise AssertionError("import was not rejected")
