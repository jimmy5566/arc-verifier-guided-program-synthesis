from pathlib import Path


def test_reentrant_checkpointing_has_input_gradient_bridge():
    source = Path("scripts/run_forward_targeted_repair_v1.py").read_text(encoding="utf-8")
    assert "model.enable_input_require_grads()" in source
    assert source.index("model.enable_input_require_grads()") < source.index(
        "model.gradient_checkpointing_enable"
    )


def test_zero_optimizer_steps_are_not_charged_to_scientific_gpu_budget():
    source = Path("scripts/run_forward_targeted_repair_v1.py").read_text(encoding="utf-8")
    assert "int(terminal.get('optimizer_steps',0))==0" in source
