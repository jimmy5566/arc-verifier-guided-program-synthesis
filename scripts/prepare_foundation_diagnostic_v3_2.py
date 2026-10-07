#!/usr/bin/env python
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_capability_bank_v3_2.pipeline import build_foundation_diagnostic_v3_2

if __name__ == "__main__":
    gate = build_foundation_diagnostic_v3_2(ROOT)
    print(gate)
    raise SystemExit(0 if gate["GPU_DIAGNOSTIC_READY"] else 2)
