# Supervisor project summary

## 50 words

I built an independent ARC-AGI research system that separates neural candidate generation from symbolic execution, hard train-pair verification, and target-blind selection. Frozen-cohort studies exposed a candidate-recall versus ranking gap and motivated provenance-aware two-attempt inference. The project demonstrates experimental discipline, failure analysis, and reliable AI systems engineering.

## 100 words

This independent project investigates reliable test-time inference for ARC-AGI. I developed a neuro-symbolic pipeline with evidence extraction, typed RuleSpec representations, deterministic execution, hard verification, reversible augmentation search, candidate provenance, and two-attempt selection. Rather than optimizing only a final score, I froze cohorts and predictions before scoring to distinguish candidate recall from Top-1 and two-attempt accuracy. On one 60-task target-blind A/B cohort, provenance-aware selection achieved 21 versus 18 two-attempt exact solves, although the paired p=0.375 result is not statistically significant. The work is relevant to trustworthy ML, LLM reasoning, agentic systems, and neuro-symbolic AI because it makes failures inspectable rather than hidden.

## 200 words

I pursued ARC-AGI as an independent research project on how reasoning systems can make reliable decisions when the correct answer is unavailable at inference time. The work began with bounded symbolic program libraries and moved through LLM program-synthesis failures to a typed RuleSpec architecture that separates evidence extraction, recognition, parameter inference, deterministic execution, and hard train-pair verification. I then studied specialized public-model test-time inference using reversible augmentations, bounded search, exact-output deduplication, provenance-aware ranking, and predeclared two-attempt output policies.

The project’s central methodological contribution is its evaluation discipline. Task cohorts, inference settings, and selectors are frozen before inference; predictions are frozen before solution files are opened. This revealed that candidate existence and candidate selection are different problems: on frozen30, correct candidates existed for 21/30 tasks but likelihood Top-1 was exact for 9/30. In a separate untouched60 A/B comparison, provenance-aware selection achieved 21/60 versus 18/60 two-attempt exact solves; the paired p=0.375 result is directional rather than significant.

I also implemented atomic checkpointing, resumable workers, and dynamic 4×L4 scheduling. The project suggests research readiness through careful negative-result reporting, controlled experimentation, reproducible interfaces, and a clear agenda: learn target-blind candidate-quality signals without compromising evaluation integrity.
