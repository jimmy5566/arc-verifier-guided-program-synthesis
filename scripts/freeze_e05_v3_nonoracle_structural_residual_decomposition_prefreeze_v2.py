#!/usr/bin/env python3
"""Freeze the Director-required conditional-estimand repair for E05 residual decomposition.

This script only validates and writes a prospective CPU-only V2 contract.  It never
opens model weights, classifies a raw prediction, or reads a sealed target value.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "experiments/capability_repair_baseline_v1/e05_v3_semantically_identifiable_demonstrations_and_adapter_storage_recovery"
V1_SPEC = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V1.json"
V1_BRIEF = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF.json"
DIRECTOR_RESPONSE = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF_RESPONSE.json"
OUTPUT_SPEC = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_SPEC_V2.json"
OUTPUT_BRIEF = OUT / "E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREEXECUTION_REVIEW_BRIEF_V2.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    v1 = json.loads(V1_SPEC.read_text(encoding="utf-8-sig"))
    response = json.loads(DIRECTOR_RESPONSE.read_text(encoding="utf-8-sig"))
    if response.get("reviewed_brief_sha256") != sha256(V1_BRIEF):
        raise RuntimeError("INVALID_V2_REPAIR_DIRECTOR_BRIEF_BINDING")
    if response.get("decision") != "REQUIRE_CHANGES":
        raise RuntimeError("INVALID_V2_REPAIR_DIRECTOR_DECISION")
    if response.get("root_cause") != "The population.base_aggregation rule defines each category as eligible category count divided by four views per canonical base and then averages across all 48 bases, while prospective_decision_rule.mechanism_dominant requires a category share among eligible parser-valid nonoracle views. When parser-invalid or exact-target views are present, the unconditional four-view prevalence and the conditional eligible-residual share are not equal. The bootstrap denominator-instability rule also presupposes a conditional eligible denominator that base_aggregation never defines. This ambiguity can reverse the frozen 0.50 decision.":
        raise RuntimeError("INVALID_V2_REPAIR_DIRECTOR_SCOPE")

    spec = copy.deepcopy(v1)
    spec.update({
        "schema_version": 2,
        "stage": "CPU_ONLY_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_PREFREEZE_V2",
        "status": "CPU_ONLY_PREFREEZE_V2_NO_RESIDUAL_CLASSIFICATION_EXECUTED",
        "predecessor_specification_sha256": sha256(V1_SPEC),
        "director_repair_response_sha256": sha256(DIRECTOR_RESPONSE),
        "source_freezer_sha256": sha256(Path(__file__)),
        "execution_authorization": False,
        "preexecution_review_required": True,
    })
    spec.pop("director_response_sha256", None)
    population = spec["population"]
    population["primary_category_estimand"] = (
        "For each arm and required condition, the primary category share is "
        "sum over all resampled canonical bases of n_category(base) divided by "
        "sum over those same bases of n_eligible(base). Every selected base retains "
        "all four views. A zero-eligible base contributes zero to both sums and is "
        "not dropped."
    )
    population["base_aggregation"] = (
        "The primary estimator is conditional on eligible parser-valid nonoracle "
        "views using pooled numerator and denominator across the same resampled "
        "canonical bases; it is not the mean of four-view prevalences."
    )
    population["unconditional_diagnostics"] = (
        "Separately report mean over all 48 bases of n_category(base)/4 as labeled "
        "unconditional four-view prevalence, and mean n_eligible(base)/4 as eligible-view coverage. "
        "Neither diagnostic substitutes for the primary conditional estimator."
    )
    spec["required_reports"] = [
        "eligible, parser-invalid, and exact-target view counts by arm and condition",
        "primary conditional eligible-residual category shares by arm and condition, with pooled numerator and denominator across the same canonical bases",
        "separately labeled unconditional four-view category prevalence and eligible-view coverage by arm and condition",
        "Treatment versus Control and Treatment versus V7 per-base category-share deltas as descriptive evidence only",
        "all per-base four-view category vectors, with no adaptive filtering",
    ]
    spec["bootstrap"] = {
        "seed": 20261013,
        "replicates": 10000,
        "resampling": "paired canonical-base bootstrap within each arm and condition; retain all four views for each sampled base; calculate each replicate as sum n_category(base) / sum n_eligible(base) over the same draw",
        "interval": "percentile 95 percent",
        "denominator_instability": "If the total eligible-residual denominator is zero in any required bootstrap draw, classify that arm-condition primary estimand as INCONCLUSIVE exactly; do not drop, repair, or reweight the draw.",
    }
    spec["prospective_decision_rule"] = {
        "mechanism_dominant": "For Treatment, one and only one category has lower 95 percent sensitivity-interval bound above 0.50 for the primary conditional eligible-residual share in both required conditions, and neither condition has denominator instability.",
        "mixed_or_inconclusive": "Otherwise, including disagreement across conditions, tied or absent dominance, invalid identities, nonexclusive taxonomy, or denominator instability.",
        "interpretation_limit": "Descriptive morphology on reused synthetic development evidence only; no causal training mechanism, checkpoint promotion, intervention selection, or independent ARC generalization claim.",
    }
    spec["repair_scope"] = {
        "only_changed": ["primary conditional estimand", "bootstrap computation", "decision-rule binding", "associated regression fixture"],
        "unchanged": ["artifact SHA256 identities", "three arms", "two required conditions", "48 canonical bases", "four views per base", "taxonomy order and precedence", "task-schema semantics", "seed", "replicate count", "percentile interval", "prohibitions"],
    }
    OUTPUT_SPEC.write_text(json.dumps(spec, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    brief = {
        "brief_type": "DIRECTOR_E05_V3_NONORACLE_STRUCTURAL_RESIDUAL_DECOMPOSITION_V2_PREEXECUTION_REVIEW",
        "protocol_id": spec["protocol_id"],
        "request": "Review only the Director-required V2 conditional-estimand repair before any CPU residual classification. No model loading, GPU work, training, inference, generation, gradient work, target expansion, or intervention selection is requested.",
        "specification_path": OUTPUT_SPEC.name,
        "specification_sha256": sha256(OUTPUT_SPEC),
        "predecessor_specification_sha256": sha256(V1_SPEC),
        "director_repair_response_sha256": sha256(DIRECTOR_RESPONSE),
        "review_scope": ["conditional numerator/denominator", "zero-denominator handling", "dominance-rule binding", "mixed-view regression fixture"],
        "unchanged_scope": spec["repair_scope"]["unchanged"],
        "allowed_outcomes": ["CONTINUE_CONTROLLER only for unchanged CPU-only execution", "REQUIRE_CHANGES", "PAUSED", "TERMINAL"],
        "boundaries": spec["prohibitions"],
    }
    OUTPUT_BRIEF.write_text(json.dumps(brief, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print("PASS_E05_V3_NONORACLE_RESIDUAL_PREFREEZE_V2")


if __name__ == "__main__":
    main()

