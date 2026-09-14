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


def test_notebook_builder_accepts_fixed_budget_parameters(tmp_path: Path):
    notebook = tmp_path / "search.ipynb"
    script = ROOT / "scripts" / "prepare_soar_refinement_notebook.py"
    subprocess.run([sys.executable, str(script), "--output", str(notebook), "--initial-samples", "32", "--refinement-rounds", "4", "--refinements-per-round", "16", "--artifact-name", "ladder"], check=True)
    content = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]["source"])
    assert '"32"' in content and '"16"' in content and "artifacts/ladder" in content
