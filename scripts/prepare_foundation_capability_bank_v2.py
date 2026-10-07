from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_capability_bank_v2.pipeline import build_capability_bank

if __name__ == "__main__":
    print(build_capability_bank(ROOT))
