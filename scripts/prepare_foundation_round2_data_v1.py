from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from foundation_round2_data_v1.pipeline import build_round2


if __name__ == "__main__":
    result = build_round2(ROOT)
    print(result)
