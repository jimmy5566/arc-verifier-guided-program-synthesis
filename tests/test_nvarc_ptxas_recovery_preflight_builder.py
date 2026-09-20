import json

from scripts.build_nvarc_ptxas_recovery_preflight_kaggle import build


def test_recovery_builder_uses_discovery_before_model_preflight(tmp_path):
    kernel, notebook = build(tmp_path / "stage", "owner", "ptxas-recovery")
    metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    code = "".join(json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]["source"])

    assert metadata["enable_internet"] is False
    assert metadata["kernel_sources"] == ["sorokin/pip-install-unsloth-flash-patch"]
    assert "discover_ptxas" in code
    assert "TRITON_PTXAS_PATH" in code
    assert "TRITON_SMOKE_PASS" in code
    assert "evaluation_solutions" not in code
    assert "turbo_dfs" not in code
