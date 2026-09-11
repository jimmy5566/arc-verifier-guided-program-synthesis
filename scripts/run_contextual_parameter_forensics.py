"""Write the required offline contextual Q1 failure forensics artifact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from llm.contextual_parameter_forensics import build_contextual_forensics


def main() -> None:
    frozen = json.loads((ROOT / "configs/PARAMETER_SEMANTIC_RETRIEVAL_V1_FROZEN_CONFIG.json").read_text())
    benchmark = json.loads((ROOT / "configs/macro_api_comprehension_benchmark_v1.json").read_text())
    checkpoint_root = ROOT / "artifacts/parameter_semantic_retrieval_output_v22_full/checkpoints/parameter_semantic_retrieval_v1/Q1_DETERMINISTIC_ONTOLOGY_REPAIR"
    result = build_contextual_forensics(benchmark=benchmark, frozen=frozen, checkpoint_root=checkpoint_root)
    if result["q1_baseline"]["remaining_pure_parameter_failures"] != 12:
        raise ValueError("expected exactly 12 unresolved pure parameter failures")
    output = ROOT / "experiments/results/CONTEXTUAL_PARAMETER_FORENSICS_V1.json"
    if output.exists():
        raise FileExistsError(output)
    output.write_text(json.dumps(result, indent=2) + "\n")
    lines = ["# Contextual Parameter Forensics V1", "", "- 完全离线 Q1 后验审计；未调用模型，未使用 ARC 数据。", f"- Q1 remaining pure parameter failures: {result['q1_baseline']['remaining_pure_parameter_failures']}。", "", "## 分布", "", f"- primary taxonomy: `{result['taxonomy_counts_primary']}`", f"- aspects: `{result['taxonomy_counts_aspects']}`", f"- fields: `{result['per_field_distribution']}`", f"- lexical / relational / contextual: `{result['lexical_vs_relational_vs_contextual']}`", f"- truly repairable: {result['truly_repairable_count']}；ambiguous: {result['ambiguous_count']}。", "", "## 结论", "", "- path 链失败主要是 source-wrapper 的隐式关系；count→generate 链主要是跨 slot 的 contract-scoped 参数关系。", "- 后续 R1 仅使用全局 Macro contract 与 typed relation features；R2/R3 不接收 canonical 或 failure labels。", ""]
    (ROOT / "reports/contextual_parameter_forensics_v1.md").write_text("\n".join(lines))
    print(json.dumps({"output": str(output), "remaining": 12}))


if __name__ == "__main__":
    main()
