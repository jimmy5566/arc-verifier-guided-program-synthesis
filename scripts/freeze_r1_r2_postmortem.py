#!/usr/bin/env python3
"""Freeze the R1/R2 paired-interference postmortem without changing results."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "experiments/capability_repair_baseline_v1/forward_targeted_repair_v1/r2_paired_evaluation_v1"
ANALYSIS = PACKAGE / "R1_R2_PER_FAMILY_PAIRED_ANALYSIS.json"
TABLE = PACKAGE / "R1_R2_PER_FAMILY_PAIRED_ANALYSIS.md"
SCOPE = PACKAGE / "CURRENT_PROTOCOL_TERMINAL_SCOPE_INTERPRETATION_V1.json"
BRIEF = ROOT / "orchestration/director/briefs/TARGETED_REPAIR_POSTMORTEM_REVIEW_V1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


def compact_delta(value: dict[str, int]) -> str:
    return f"{value['paired_fixes_n01']}/{value['paired_harms_n10']}/{value['net_fixes']:+d}"


def family_lines(analysis: dict) -> list[str]:
    role_order = ("TARGETED_EVALUATION", "TARGETED_COMPOSITION", "RETENTION_SENTINEL")
    labels = {
        "TARGETED_EVALUATION": "ATOMIC",
        "TARGETED_COMPOSITION": "COMPOSITION",
        "RETENTION_SENTINEL": "RETENTION",
    }
    rows = list(analysis["families"].values())
    lines = [
        "| 组别 | 家族 | n | V7 | R1 | R2 | R1−V7 修复/伤害/净值 | R2−V7 修复/伤害/净值 | R2−R1 修复/伤害/净值 | 分类 |",
        "|---|---|---:|---:|---:|---:|---|---|---|---|",
    ]
    for role in role_order:
        for item in sorted((x for x in rows if x["role"] == role), key=lambda x: x["family"]):
            lines.append(
                "| {role} | {family} | {episodes} | {baseline_exact} | {r1_exact} | {r2_exact} | {r1} | {r2} | {r21} | {classification} |".format(
                    role=labels.get(role, role),
                    family=item["family"],
                    episodes=item["episodes"],
                    baseline_exact=item["baseline_exact"],
                    r1_exact=item["r1_exact"],
                    r2_exact=item["r2_exact"],
                    r1=compact_delta(item["r1_vs_v7"]),
                    r2=compact_delta(item["r2_vs_v7"]),
                    r21=compact_delta(item["r2_vs_r1"]),
                    classification=item["classification"],
                )
            )
    return lines


def names(items: list[dict]) -> str:
    return ", ".join(x["family"] for x in items) or "无"


def main() -> None:
    analysis = json.loads(ANALYSIS.read_text(encoding="utf-8"))
    if analysis.get("status") != "COMPLETE" or analysis.get("final_audit_opened") is not False:
        raise RuntimeError("PAIRED_ANALYSIS_NOT_FROZEN_OR_FINAL_AUDIT_OPENED")
    analysis_sha = sha256(ANALYSIS)
    summary = analysis["interference_summary"]
    scope = {
        "schema_version": 1,
        "status": "CURRENT_PROTOCOL_CONTROLLED_STOP_INTERPRETATION",
        "historical_director_response_path": "orchestration/director/responses/GOVERNOR_REVIEW_TARGETED_REPAIR_R2_COMPLETION_V1.json",
        "historical_director_decision": "TERMINAL",
        "terminal_scope": "CURRENT_PROTOCOL",
        "protocol_id": "FORWARD_TARGETED_CAPABILITY_REPAIR_V1",
        "meaning": "STOP_AUTOMATIC_R3_UNDER_THE_CURRENT_R1_R2_CURRICULUM_ROUTE",
        "entire_experiment_terminal": False,
        "permitted_next_action": "CPU_ONLY_PER_FAMILY_POSTMORTEM_AND_ONE_DIRECTOR_SCIENTIFIC_REVIEW",
        "forbidden_actions": [
            "DO_NOT_LAUNCH_R3_AUTOMATICALLY",
            "DO_NOT_MUTATE_V7_R1_R2_OR_EXISTING_GATES",
            "DO_NOT_OPEN_FINAL_AUDIT",
            "DO_NOT_USE_TARGET_DEV_OR_RETENTION_SENTINEL_AS_TRAINING_DATA",
        ],
        "final_audit_opened": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    atomic_write(SCOPE, json.dumps(scope, sort_keys=True, indent=2) + "\n")
    markdown = [
        "# R1 / R2 逐家族配对分析",
        "",
        "此分析只读取冻结的 V7、R1 和 R2 逐样本预测；不改变既有结果，不访问 FINAL_AUDIT，不启动 R3。",
        "",
        f"- 配对分析 JSON SHA256：`{analysis_sha}`",
        "- 记号：`修复/伤害/净值` 分别为 n01/n10/(n01−n10)。",
        "",
        *family_lines(analysis),
        "",
        "## 干扰解释",
        "",
        f"- R1 的组成能力净收益来自：{names(summary['r1_composition_gain_families'])}。",
        f"- R1 伤害的原子能力：{names(summary['r1_atomic_harm_families'])}。",
        f"- R2 相比 R1 进一步伤害的原子能力：{names(summary['r2_atomic_further_harm_families'])}。",
        f"- R1→R2 恢复的保持能力：{names(summary['retention_recovered_r1_to_r2'])}。",
        f"- R2 仍低于 V7 的保持能力：{names(summary['retention_below_v7_after_r2'])}。",
        f"- 同一示例在轮次间反复翻转的家族：{names(summary['repeated_flip_families'])}。",
        "",
        "分类基于同一冻结 episode 集的逐例正确性转换；它描述训练干扰，不构成任何 R3 启动授权。",
        "",
    ]
    atomic_write(TABLE, "\n".join(markdown))
    brief = {
        "schema_version": 1,
        "brief_id": "TARGETED_REPAIR_POSTMORTEM_REVIEW_V1",
        "scope": "TARGETED_REPAIR_CURRENT_PROTOCOL_POSTMORTEM",
        "terminal_scope_interpretation": {
            "terminal_scope": "CURRENT_PROTOCOL",
            "meaning": "STOP_AUTOMATIC_R3_UNDER_THE_CURRENT_R1_R2_CURRICULUM_ROUTE",
            "entire_experiment_terminal": False,
            "scope_artifact": str(SCOPE.relative_to(ROOT)).replace("\\", "/"),
            "scope_artifact_sha256": sha256(SCOPE),
        },
        "frozen_evidence": {
            "paired_analysis": {
                "path": str(ANALYSIS.relative_to(ROOT)).replace("\\", "/"),
                "sha256": analysis_sha,
            },
            "human_table": {
                "path": str(TABLE.relative_to(ROOT)).replace("\\", "/"),
                "sha256": sha256(TABLE),
            },
            "reference_baseline": "CAPABILITY_REPAIR_BASELINE_V1_V7",
            "v7_results": {"target_dev": 76, "atomic": 45, "composition": 31, "retention": 70},
            "r1_results": {"target_dev": 77, "atomic": 43, "composition": 34, "retention": 64},
            "r2_results": {"target_dev": 74, "atomic": 39, "composition": 35, "retention": 68},
        },
        "budget": {
            "cumulative_scientific_gpu_seconds": 4778.014247704957,
            "remaining_scientific_gpu_seconds": 24021.985752295043,
            "r3_started": False,
        },
        "scientific_interpretation": {
            "r1_composition_gain_families": [x["family"] for x in summary["r1_composition_gain_families"]],
            "r1_atomic_harm_families": [x["family"] for x in summary["r1_atomic_harm_families"]],
            "r2_atomic_further_harm_families": [x["family"] for x in summary["r2_atomic_further_harm_families"]],
            "retention_recovered_r1_to_r2": [x["family"] for x in summary["retention_recovered_r1_to_r2"]],
            "retention_below_v7_after_r2": [x["family"] for x in summary["retention_below_v7_after_r2"]],
            "repeated_flip_families": [x["family"] for x in summary["repeated_flip_families"]],
        },
        "frozen_conditions": [
            "V7, R1, R2, current R2 gate outcome, and all existing receipts remain byte-preserved.",
            "RETENTION_SENTINEL and TARGET_DEV remain evaluation-only and are not training data.",
            "FINAL_AUDIT remains sealed and unopened.",
            "No automatic R3 is authorized by this postmortem request.",
        ],
        "director_question": "Given the per-family paired fixes/harms across V7 -> R1 -> R2, is there a materially different R3 curriculum that is scientifically justified, or should this targeted-repair approach be stopped?",
        "allowed_director_outcomes": [
            "NEW_R3_PROTOCOL_RECOMMENDED",
            "STOP_TARGETED_REPAIR_APPROACH",
            "RUN_ONE_DIAGNOSTIC_BEFORE_DECIDING",
        ],
        "required_content_if_new_r3_protocol_recommended": [
            "families to upweight",
            "families to downweight or remove",
            "atomic/composition/replay mix",
            "whether learning rate or token budget changes",
            "starting checkpoint: V7, R1-final, or R2",
            "explicit expected benefit",
            "explicit failure condition",
        ],
        "required_content_if_one_diagnostic": "Specify exactly one diagnostic and its decision rule.",
        "final_audit_opened": False,
        "scientific_training_requested": False,
    }
    atomic_write(BRIEF, json.dumps(brief, sort_keys=True, indent=2) + "\n")
    print(json.dumps({
        "analysis_sha256": analysis_sha,
        "table_sha256": sha256(TABLE),
        "scope_sha256": sha256(SCOPE),
        "brief_sha256": sha256(BRIEF),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
