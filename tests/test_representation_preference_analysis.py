from scripts.analyze_representation_preference import analyze_training
from scripts.build_candidate_generation_root_cause_report import build


def test_training_view_analysis_reports_fixed_and_oracle_bounds():
    artifact = {
        "records": {
            "a": {
                "generated_candidate_count": 2,
                "invalid_candidate_count": 0,
                "candidates": [
                    {
                        "prediction": [[[1]]],
                        "support_augmentations": [
                            {"geometry": "identity", "pair_order": "canonical", "color_offset": 0}
                        ],
                    },
                    {
                        "prediction": [[[9]]],
                        "support_augmentations": [
                            {"geometry": "rot90", "pair_order": "canonical", "color_offset": 0}
                        ],
                    },
                ],
            },
            "b": {
                "generated_candidate_count": 1,
                "invalid_candidate_count": 0,
                "candidates": [
                    {
                        "prediction": [[[2]]],
                        "support_augmentations": [
                            {"geometry": "rot90", "pair_order": "reversed", "color_offset": 1}
                        ],
                    }
                ],
            },
            "c": {
                "generated_candidate_count": 1,
                "invalid_candidate_count": 0,
                "candidates": [
                    {
                        "prediction": [[[3]]],
                        "support_augmentations": [
                            {"geometry": "rot90", "pair_order": "canonical", "color_offset": 0}
                        ],
                    }
                ],
            },
        }
    }
    report, correct_by_view = analyze_training(
        artifact, {"a": [[[1]]], "b": [[[2]]], "c": [[[3]]]}
    )

    assert report["best_global_view"] == "rot90"
    assert report["best_global_view_anyk"] == 2
    assert report["oracle_view_anyk"] == 3
    assert report["routing_headroom"] == 1
    assert correct_by_view["identity"] == {"a"}
    assert correct_by_view["rot90"] == {"b", "c"}


def test_root_report_keeps_blocked_ttt_as_not_measured():
    parity = {
        "status": "COMPLETE",
        "high_impact_mismatch_count": 1,
        "unverified_critical_count": 0,
        "top_5_most_important_differences": [{"item": "colour augmentation"}],
    }
    representation = {
        "best_global_view": "identity",
        "best_global_view_anyk": 2,
        "oracle_view_anyk": 3,
        "routing_headroom": 1,
        "view_preference_stability": "UNAVAILABLE",
        "router_feasibility": "INSUFFICIENT_DATA_FOR_ROUTER",
    }
    report = build(parity, representation)

    assert report["final_diagnosis"] == "REFERENCE_INTERFACE_MISMATCH"
    assert report["stage_b_reference_style_ttt_eval6"]["strong_ttt_anyk"] == "NOT_MEASURED"
    assert report["stage_c_ttt_search_interaction"]["status"] == "SCIENTIFICALLY_RULED_OUT_NOT_RUN"
