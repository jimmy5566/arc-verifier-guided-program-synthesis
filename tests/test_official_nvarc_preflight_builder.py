import json

from scripts.build_official_nvarc_preflight_kaggle import (
    MODEL_SOURCE,
    REFERENCE_SOURCE,
    TASK_ID,
    build,
)


def test_preflight_builder_uses_official_kernel_source_and_no_solutions(tmp_path):
    kernel, notebook = build(tmp_path / "stage", "owner", "preflight")
    metadata = json.loads((kernel / "kernel-metadata.json").read_text(encoding="utf-8"))
    source = json.loads(notebook.read_text(encoding="utf-8"))["cells"][0]["source"]
    source_text = "".join(source)

    assert metadata["kernel_sources"] == [REFERENCE_SOURCE]
    assert metadata["model_sources"] == [MODEL_SOURCE]
    assert metadata["enable_internet"] is False
    assert TASK_ID in source_text
    assert "evaluation_solutions" not in source_text
    assert "turbo_dfs" not in source_text
    assert "MAX_PREFLIGHT_STEPS = 4" in source_text
