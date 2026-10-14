from scripts.prepare_e4d_representation_binding_diagnostic_v1 import LAYERS, TOPOLOGIES, freeze, normal, rotate


def test_fixed_layers_and_structurally_distinct_orbits():
    assert LAYERS == (8, 16, 24, 31)
    orbits = {name: {rotate(points, turn) for turn in range(4)} for name, points in TOPOLOGIES.items()}
    assert all(len(points) == 11 for points in TOPOLOGIES.values())
    assert all(len(orbit) == 4 for orbit in orbits.values())
    assert len({normal(points) for points in TOPOLOGIES.values()}) == 4


def test_freeze_is_target_blind_balanced_and_cpu_only():
    receipt = freeze()
    assert receipt["status"] == "PASS_E4D_REPRESENTATION_BINDING_CPU_VALIDATION"
    assert receipt["model_loaded"] is False and receipt["gpu_used"] is False
    assert receipt["targets_opened"] is False and receipt["target_sidecar_written"] is False
    assert receipt["fresh_ood"] == {"canonical_units": 48, "records": 96, "count_balance": {"1": 16, "2": 16, "3": 16}, "orders": 2, "zero_overlap_prior_prompt_query_content_role": True}
