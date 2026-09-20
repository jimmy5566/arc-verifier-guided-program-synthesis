"""Assemble the controlled candidate-generation root-cause synthesis.

No inference or target scoring happens here: all inputs are frozen reports or
the exact failure log from the blocked reference-style pilot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TORCHAO_ERROR = (
    "ImportError: Found an incompatible version of torchao. Found version 0.10.0, "
    "but only versions above 0.16.0 are supported"
)


def load(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def build(parity: dict[str, Any], representation: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment_id": "ARC2_CANDIDATE_GENERATION_ROOT_CAUSE_REFERENCE_PARITY",
        "status": "COMPLETE_WITH_REFERENCE_TTT_STOP_GATE",
        "final_diagnosis": "REFERENCE_INTERFACE_MISMATCH",
        "scientific_conclusion": (
            "The available evidence does not isolate an intrinsic base-model capability "
            "limit. It identifies material documented reference-interface mismatches, and "
            "the closest reference-style TTT pilot was blocked before any task adaptation "
            "or generation by an incompatible local backend."
        ),
        "current_evidence": {
            "frozen60_aug8": {"any_of_k": "21/60", "top1": "19/60", "top2": "19/60"},
            "eval60_aug8": {
                "any_of_k": "2/60",
                "top1": "2/60",
                "top2": "2/60",
                "pool_miss": "58/60",
                "selection_miss": "0/60",
            },
            "eval30_pool_miss_search": {
                "greedy": "0/30",
                "beam2": "2/30",
                "dfs_small": "0/30",
                "dfs_medium": "CANCELLED_NOT_RESUMED",
            },
            "eval12_light_ttt": {"baseline": "0/12", "ttt": "0/12"},
        },
        "stage_a_reference_parity": {
            "status": parity["status"],
            "high_impact_mismatch_count": parity["high_impact_mismatch_count"],
            "unverified_critical_count": parity["unverified_critical_count"],
            "top_5_most_important_differences": parity["top_5_most_important_differences"],
            "faithful_to_known_strong_reference": False,
            "faithfulness_explanation": (
                "Tokenizer-level semantics match, but high-impact differences remain in "
                "pair-order diversity, arbitrary colour permutations, generation and TTT "
                "view counts, LoRA rank/targets, adaptation batch construction, and "
                "Turbo-DFS decoding."
            ),
        },
        "stage_b_reference_style_ttt_eval6": {
            "status": "BLOCKED_BEFORE_TASK_EXECUTION",
            "kaggle_kernel": "jimmy5566/arc2-eval6-reference-ttt-diagnostic",
            "kernel_status": "KernelWorkerStatus.ERROR",
            "wall_seconds_before_stop": 80.13,
            "candidate_artifacts_created": False,
            "solutions_opened": False,
            "exact_failure": TORCHAO_ERROR,
            "failure_stage": "PEFT adapter creation after base-model load",
            "backend_mismatch": (
                "The public notebook uses a separate Unsloth/Unsloth Zoo/FlashAttention "
                "kernel source in a Python 3.11 environment. The attempted isolated pilot "
                "used standard PEFT in Kaggle Python 3.12, where the installed torchao 0.10.0 "
                "is incompatible with that PEFT path."
            ),
            "stop_rule_applied": (
                "Do not silently substitute a reduced adapter or patch around the dependency; "
                "the experiment cannot claim reference-faithful adaptation in this runtime."
            ),
            "strong_ttt_anyk": "NOT_MEASURED",
            "new_recoveries": "NOT_MEASURED",
        },
        "stage_c_ttt_search_interaction": {
            "status": "SCIENTIFICALLY_RULED_OUT_NOT_RUN",
            "reason": (
                "Stage C requires a valid reference-style TTT base. Stage B produced no "
                "adapted model or frozen candidates, so Beam2/DFS-small after TTT would not "
                "test the stated interaction."
            ),
        },
        "stage_d_representation_preference": {
            "best_global_view": representation["best_global_view"],
            "best_global_view_anyk": representation["best_global_view_anyk"],
            "oracle_view_anyk": representation["oracle_view_anyk"],
            "routing_headroom": representation["routing_headroom"],
            "view_preference_stability": representation["view_preference_stability"],
            "router_feasibility": representation["router_feasibility"],
        },
        "answers": {
            "is_our_model_interface_faithful": "NO",
            "highest_impact_mismatches": [entry["item"] for entry in parity["top_5_most_important_differences"]],
            "did_reference_style_ttt_recover_eval_pool_misses": "NOT_MEASURED: blocked before task execution",
            "did_search_become_more_useful_after_ttt": "NOT_MEASURED: Stage C correctly not run",
            "does_model_show_stable_representation_preferences": "NOT_ESTABLISHED",
            "is_representation_routing_worth_pursuing_now": "NO: INSUFFICIENT_DATA_FOR_ROUTER",
            "evidence_base_checkpoint_needs_retraining": "NO: prerequisite reference-faithful TTT/search evidence is missing",
        },
        "next_best_experiment": {
            "name": "Reference-environment adapter preflight",
            "design": (
                "Use the public notebook's documented Unsloth kernel source/runtime as an "
                "isolated environment. Run one target-blind task through adapter creation, "
                "one epoch, reset, and unload. Only if that passes, run the already-frozen "
                "Eval6 reference-TTT greedy pilot."
            ),
            "expected_gpu_cost": "Preflight only: bounded single-task run; no full Eval6 until adapter fidelity is demonstrated.",
            "expected_information_gain": "HIGH: distinguishes runtime/backend incompatibility from adaptation effectiveness.",
            "do_not_do_next": [
                "Do not resume DFS-medium.",
                "Do not run full Eval60/Eval120 or a competition submission.",
                "Do not reduce rank/targets or monkey-patch torchao and label the result reference-style TTT.",
                "Do not infer a base-model capability limit or begin base retraining.",
            ],
        },
    }


def write_markdown(report: dict[str, Any], destination: Path) -> None:
    stage_b = report["stage_b_reference_style_ttt_eval6"]
    stage_d = report["stage_d_representation_preference"]
    lines = [
        "# Candidate Generation Root Cause + Reference Parity Study",
        "",
        f"## Final diagnosis: `{report['final_diagnosis']}`",
        "",
        report["scientific_conclusion"],
        "",
        "## Evidence",
        "",
        "- Frozen60 Aug8: Any-of-K 21/60; Top-1/Top-2 19/60.",
        "- Eval60 Aug8: Any-of-K/Top-1/Top-2 2/60; 58/60 pool misses and 0 selection misses.",
        "- Eval30: greedy 0/30, Beam2 2/30, DFS-small 0/30; DFS-medium was not resumed.",
        "- Eval12 light TTT: 0/12 before and after adaptation.",
        "",
        "## Stage A — parity",
        "",
        f"- High-impact confirmed mismatches: **{report['stage_a_reference_parity']['high_impact_mismatch_count']}**.",
        f"- Critical unverified items: **{report['stage_a_reference_parity']['unverified_critical_count']}**.",
        "- The known strong public reference is not faithfully reproduced by the current stack.",
        "",
        "## Stage B — reference-style Eval6 TTT",
        "",
        "The target-blind pilot stopped before any task execution. Its exact adapter-creation error was:",
        "",
        f"`{stage_b['exact_failure']}`",
        "",
        stage_b["backend_mismatch"],
        "",
        "No candidates were frozen and no solutions were opened. Per the predefined stop rule, no reduced or patched substitute was run.",
        "",
        "## Stage C — TTT + search",
        "",
        report["stage_c_ttt_search_interaction"]["reason"],
        "",
        "## Stage D — representation preference",
        "",
        f"- Best global training-side view: `{stage_d['best_global_view']}` ({stage_d['best_global_view_anyk']}/60).",
        f"- Per-task oracle view upper bound: {stage_d['oracle_view_anyk']}/60.",
        f"- Apparent routing headroom: {stage_d['routing_headroom']} tasks.",
        f"- Stability: {stage_d['view_preference_stability']}",
        f"- Router decision: `{stage_d['router_feasibility']}`.",
        "",
        "## Next experiment",
        "",
        report["next_best_experiment"]["design"],
        "",
        f"Expected GPU cost: {report['next_best_experiment']['expected_gpu_cost']}",
    ]
    destination.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parity-report", type=Path, required=True)
    parser.add_argument("--representation-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = build(load(args.parity_report), load(args.representation_report))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CANDIDATE_GENERATION_ROOT_CAUSE.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_markdown(report, args.output_dir / "CANDIDATE_GENERATION_ROOT_CAUSE.md")
    print(json.dumps({
        "event": "CANDIDATE_GENERATION_ROOT_CAUSE_COMPLETE",
        "final_diagnosis": report["final_diagnosis"],
        "stage_b": report["stage_b_reference_style_ttt_eval6"]["status"],
        "stage_c": report["stage_c_ttt_search_interaction"]["status"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
