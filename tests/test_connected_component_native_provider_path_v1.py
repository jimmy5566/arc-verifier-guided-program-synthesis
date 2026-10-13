from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT / "src"))
from inference.nvarc_native import parse_native_grid, serialize_grid

def test_native_provider_import_and_grid_surface():
 assert callable(parse_native_grid) and callable(serialize_grid)
 assert serialize_grid([[1, 2], [3, 4]]) == "12\n34"
