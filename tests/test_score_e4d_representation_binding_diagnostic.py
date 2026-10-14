from scripts.score_e4d_representation_binding_diagnostic_v1 import self_test


def test_linear_probe_scorer_is_cpu_only():
    result = self_test()
    assert result["status"] == "PASS_E4D_REPRESENTATION_BINDING_SCORER_CPU_TEST"
    assert result["model_loaded"] is False and result["gpu_used"] is False
