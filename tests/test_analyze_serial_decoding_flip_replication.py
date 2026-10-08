from scripts.analyze_serial_decoding_flip_replication import MODEL_TO_SOURCE


def test_serial_batch1_model_mapping_is_explicit_and_complete():
    assert MODEL_TO_SOURCE == {
        "CAPABILITY_REPAIR_BASELINE_V1_V7": "v7",
        "FORWARD_TARGETED_CAPABILITY_REPAIR_V1_003_FINAL": "r1",
        "FORWARD_TARGETED_CAPABILITY_REPAIR_V1_004_FINAL": "r2",
    }
