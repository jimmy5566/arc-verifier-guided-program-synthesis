"""Run the development-only solution-aware DOWNSTREAM_V1 capability audit."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from time import perf_counter
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recognition.downstream_capability_gap_forensics import (  # noqa: E402
    aggregate_records,
    audit_legacy_serialized_scores,
    cohort_public_record,
    DEVELOPMENT_SOLUTION_FILENAME,
    legacy_individual_dispositions,
    run_solution_aware_audit,
    select_development_cohort,
    stable_hash,
)
from build_grid_recognition_diagnostic_tasks import _exposed_config_ids  # noqa: E402


EXPERIMENT_ID = "DOWNSTREAM_CAPABILITY_GAP_FORENSICS_V1"
SALT = EXPERIMENT_ID


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cohort() -> tuple[tuple[str, ...], dict[str, Any]]:
    rows = list(csv.DictReader((ROOT / "data/splits/task_splits.csv").open(encoding="utf-8")))
    development = tuple(sorted(row["task_id"] for row in rows if row["split"] == "development"))
    exposed = _exposed_config_ids()
    task_ids = select_development_cohort(development, exposed, size=30, salt=SALT)
    return task_ids, cohort_public_record(task_ids, eligible_count=len(set(development) - exposed), excluded_count=len(exposed), salt=SALT)


def _prior_score_audit() -> dict[str, Any]:
    """Audit the former tuple-vs-list comparison without rewriting its files."""
    legacy_path = ROOT / "artifacts/grid_recognition_diagnostic_v1_capability_gate_private.json"
    return audit_legacy_serialized_scores(ROOT / "data/raw" / DEVELOPMENT_SOLUTION_FILENAME, legacy_path)


def _report(public: dict[str, Any]) -> str:
    aggregate = public["aggregate"]
    top = sorted(((name, count) for name, count in aggregate["classification_counts"].items() if name.startswith("MISSING_") or name == "OTHER_CAPABILITY_GAP"), key=lambda item: (-item[1], item[0]))[:5]
    rows = "\n".join(f"| {name} | {count} |" for name, count in top)
    return f"""# DOWNSTREAM Capability Gap Forensics V1

这是一个 solution-aware、development-only 的离线能力取证；没有运行 Grid Recognizer、Qwen 或新推理。公开产物不含任务 ID、ARC grids、solutions、oracle programs 或 test outputs。

## 冻结 cohort

- Tasks: {aggregate['task_count']} development
- Cohort commitment hash: `{public['cohort']['cohort_commitment_hash']}`
- Frozen DOWNSTREAM_V1 candidate-space hash: `{public['frozen_downstream']['candidate_space_hash']}`

## 结果

- Representable: **{aggregate['representable']}/{aggregate['task_count']} = {aggregate['representability_rate']:.1%}**
- Search found oracle/equivalent: **{aggregate['search_found_oracle']}**
- Search missed: **{aggregate['search_missed']}**
- Compiler gap: **{aggregate['compiler_gap']}**
- Capability gap: **{aggregate['capability_gap']}**

| 分类 | Tasks |
| --- | ---: |
{chr(10).join(f"| {name} | {count} |" for name, count in public['aggregate']['classification_counts'].items())}

## Top missing capabilities

| 缺口分类 | Tasks |
| --- | ---: |
{rows}

## 历史 4 个 apparent train-consistent/test-wrong

历史私有记录中有 {public['legacy_score_audit']['legacy_records']} 个 train-consistent 候选。重新用逐个 `numpy.array_equal` 比较 JSON-normalized grids 后，发现 {public['legacy_score_audit']['serialization_false_negative_count']} 个是 tuple/list 容器比较造成的计分假阴性，真实 test-wrong 为 {public['legacy_score_audit']['genuine_test_wrong']}。因此它们不是 coordinate/color/size/object-identity overfit 证据，也不支持据此引入 ranking 或 generalization 修改。

## 决策

主瓶颈：**{public['decision']['main_bottleneck']}**。

建议的唯一下一实验：**{public['decision']['next_experiment']}**。

## 完整性与泄漏审计

- solutions 仅在 `recognition.downstream_capability_gap_forensics` 的离线 oracle 边界读取。
- 30 题 cohort 在读取 solutions 前按 development split 和 SHA-256 固定；不会按 solver 成功筛选。
- 全部 representable 结论均要求 schema → type → parameter → compiler → executor → train exact → test exact。
- 无 task-ID production branch；私有 task IDs、program fingerprints 及任何逐题解释位于 ignored `artifacts/`。
"""


def main() -> None:
    started = perf_counter()
    task_ids, cohort = _cohort()
    _write_json(ROOT / "configs/DOWNSTREAM_CAPABILITY_GAP_FORENSICS_V1_TASKS.json", cohort)
    private_cohort = ROOT / "artifacts/downstream_capability_gap_forensics_v1_cohort_private.json"
    _write_json(private_cohort, {"task_ids": list(task_ids), "cohort_commitment_hash": cohort["cohort_commitment_hash"]})
    records, private_details = run_solution_aware_audit(
        ROOT / "data/raw/arc-agi_training_challenges.json",
        ROOT / "data/raw" / DEVELOPMENT_SOLUTION_FILENAME,
        task_ids,
        ROOT / "artifacts/grid_recognition_diagnostic_v1_capability_gate_private.json",
    )
    aggregate = aggregate_records(records)
    frozen_hash = stable_hash({"candidate_module": (ROOT / "src/recognition/downstream_v1_capability_audit.py").read_text(encoding="utf-8"), "compiler": (ROOT / "src/llm/macro_compiler_v1.py").read_text(encoding="utf-8")})
    decision = {"main_bottleneck": "CAPABILITY_COVERAGE" if aggregate["representability_rate"] < .7 else ("HYPOTHESIS_RANKING" if aggregate["compiler_gap"] else "PROGRAM_SEARCH"), "next_experiment": "CAPABILITY_LIBRARY_EXPANSION_V1" if aggregate["representability_rate"] < .7 else "GUIDED_PROGRAM_SEARCH_V1"}
    public = {
        "experiment_id": EXPERIMENT_ID,
        "status": "COMPLETE_SOLUTION_AWARE_DEVELOPMENT_ONLY",
        "cohort": cohort,
        "frozen_downstream": {"candidate_space_hash": frozen_hash, "historical_candidate_audit": {"generated": 8223505, "compiled": 3739652, "train_consistent": 4}},
        "aggregate": aggregate,
        "legacy_score_audit": _prior_score_audit(),
        "decision": decision,
        "runtime_seconds": perf_counter() - started,
        "leakage_audit": ["Development-only cohort commitment precedes oracle analysis.", "Solutions are read only in the diagnostic oracle boundary.", "No task IDs, raw grids, solutions, oracle programs, or test outputs appear in public outputs.", "No task-specific production rule or frozen DOWNSTREAM_V1 change was made."],
    }
    _write_json(ROOT / "artifacts/downstream_capability_gap_forensics_v1_private.json", {
        "cohort_commitment_hash": cohort["cohort_commitment_hash"],
        "records": private_details,
        "legacy_individual_dispositions": legacy_individual_dispositions(ROOT / "data/raw" / DEVELOPMENT_SOLUTION_FILENAME, ROOT / "artifacts/grid_recognition_diagnostic_v1_capability_gate_private.json"),
    })
    _write_json(ROOT / "experiments/results/DOWNSTREAM_CAPABILITY_GAP_FORENSICS_V1.json", public)
    (ROOT / "reports/downstream_capability_gap_forensics_v1.md").write_text(_report(public), encoding="utf-8")
    with (ROOT / "experiments/experiments.csv").open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["experiment_id", "date", "git_commit", "solver", "representation", "search_method", "llm_model", "candidate_budget", "validation_split", "tasks_solved", "accuracy", "runtime_seconds", "gpu_hours", "notes"])
        writer.writerow({"experiment_id": EXPERIMENT_ID, "date": str(date.today()), "git_commit": "pending", "solver": "Frozen DOWNSTREAM_V1 capability forensics", "representation": "Macro DSL + compiler + primitive executor", "search_method": "solution-aware oracle diagnostic over frozen finite candidate space", "llm_model": "", "candidate_budget": 8223505, "validation_split": "development_only_30", "tasks_solved": aggregate["representable"], "accuracy": aggregate["representability_rate"], "runtime_seconds": public["runtime_seconds"], "gpu_hours": 0, "notes": json.dumps({"decision": decision, "legacy_score_audit": public["legacy_score_audit"]}, sort_keys=True)})
    print(json.dumps({"event": "forensics_complete", "aggregate": aggregate, "decision": decision, "runtime_seconds": public["runtime_seconds"]}, sort_keys=True))


if __name__ == "__main__":
    main()
