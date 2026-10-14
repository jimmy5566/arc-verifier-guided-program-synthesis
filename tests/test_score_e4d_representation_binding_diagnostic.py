from scripts.score_e4d_representation_binding_diagnostic_v1 import classify, self_test


def test_linear_probe_scorer_is_cpu_only():
    result = self_test()
    assert result["status"] == "PASS_E4D_REPRESENTATION_BINDING_SCORER_CPU_TEST"
    assert result["model_loaded"] is False and result["gpu_used"] is False


def test_mechanism_classification_is_exhaustive():
    assert classify(True, False) == "REPRESENTATION_TO_GENERATION_BINDING_SUPPORTED"
    assert classify(False, False) == "ANCHOR_REPRESENTATION_INTERVENTION_FAILED"
    assert classify(True, True) == "BOTH_IMPROVE_DIAGNOSTIC_SUPPORT_ONLY"
    assert classify(False, True) == "NATIVE_ONLY_DISCORDANT_DIAGNOSTIC_INCONCLUSIVE"
