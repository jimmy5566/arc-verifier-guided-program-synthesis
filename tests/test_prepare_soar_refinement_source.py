import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).parents[1]


def test_source_package_is_target_free_and_not_native(tmp_path: Path):
    output = tmp_path / "stage"
    script = ROOT / "scripts" / "prepare_soar_refinement_source.py"
    subprocess.run([sys.executable, str(script), "--output", str(output), "--owner", "owner"], check=True)
    metadata = json.loads((output / "dataset" / "dataset-metadata.json").read_text(encoding="utf-8"))
    assert metadata["id"] == "owner/arc2-soar-refinement-source"
    assert (output / "dataset" / "ARC2.tar.gz").is_file()
