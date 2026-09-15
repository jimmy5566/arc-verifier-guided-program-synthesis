# ARC2 research audit

This is the canonical claim ledger for the portfolio package. It prioritizes configs, machine-readable artifacts, and reports over prior README wording. “Local/private” means the source exists in the working research archive but is ignored by Git because it may expose restricted inputs or generated predictions; a public document must retain that caveat.

## Reconstructed trajectory

| Stage | Question / hypothesis | What was attempted | Evidence-backed outcome |
| --- | --- | --- | --- |
| Symbolic libraries | Can bounded transformations provide an interpretable base? | Whole-grid, object, relation, pattern, and typed capability libraries | V0 solved 14/1,000 training tasks; V1 and V2 each 6/1,000. Coverage, composition, and object ambiguity limited progress. |
| LLM programs | Can a planner propose verifiable symbolic programs? | Qwen3 typed programs, Macro DSL, compiler, parameter solvers, hard verifier | Frozen V2 Pilot 50: 0/50 exact for both symbolic and direct routes. Forensics identified interface/parameter grounding rather than verifier operation as the major blocker. |
| Semantic representation | Can structured evidence and RuleSpec separate recognition from execution? | Evidence extraction, skeleton/RuleSpec, typed values/selectors, binding, executor | A train-only construction push improved exact coverage 3/30 → 10/30 with 0 regressions; no end-to-end held-out claim follows. |
| Native candidate generation | Do reversible views and ARC-specialized public model inference create usable candidate pools? | Grid serialization, augmentation, greedy and bounded beam branches, deduplication | Candidate recall became non-trivial but varied sharply by cohort; a correct candidate was frequently not Top-1. |
| Selection | Does selection help once a correct output exists? | Multi-view likelihood, consensus/provenance, two-attempt selection | On untouched60 A/B, recall was 30/60 for both while B two-attempt exact was 21/60 vs A 18/60; paired p=0.375. |
| Systems | Can production inference be made robust without changing semantics? | Complete per-GPU workers, atomic checkpoints, resume, dynamic queue | Historical CPU-only replay estimated 26.38% less makespan with dynamic scheduling; this is not a live GPU benchmark. |

## Evidence table

| Claim | Supporting artifact / code | Exact metric | Confidence and caveat |
| --- | --- | ---: | --- |
| Symbolic baseline was deliberately narrow | `experiments/results/SOLVER_LIB_V0.json`; `experiments/experiments.csv` | 14/1,000 exact | High for recorded train-only metric; not a hidden-test result. |
| V2 Macro DSL did not solve its frozen pilot | `experiments/results/LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json`; `reports/llm_program_synthesis_v2.md` | 0/50 symbolic; 0/50 direct | High; negative result retained. |
| Macro API syntax could be repaired but semantic grounding remained weak | `experiments/results/MACRO_API_COMPILER_AWARE_ABLATION_V1.json`; `C3_SEMANTIC_INTENT_AUDIT_V1.json` | compile-valid 60/60; semantic audit 31/60 | High for API-only benchmark; ARC data was not used. |
| Complete RuleSpec improved representability | `reports/arc2_v3_complete_rulespec_construction_push.md`; local `artifacts/arc2_v3_construction_push_audit_v17.json` | 3/30 → 10/30 exact train coverage | High for train-only oracle-family audit; no recognition/E2E scoring. |
| Candidate availability and ranking diverge | local `artifacts/ARC2_NATIVE_RANKER_FROZEN30_FORENSICS.json` | Any-of-K 21/30; Top-1 9/30; MRR 0.389; median correct rank 2 | High for frozen30 forensic; diagnostic cohort only. |
| Robust likelihood had limited untouched benefit | `reports/arc2_native_likelihood_ranker_v2_fast.md` | 3/8 Any-of-K; 2/8 Top-1 under all reported methods | High, but N=8 and one one-shot cohort. |
| TTT path was memory-safe and had a narrow recall gain | `reports/arc2_native_ttt_and_strong_verifier_v1.md`; config `ARC2_NATIVE_TTT_AND_STRONG_VERIFIER_V1.json` | 1/5 → 2/5 Any-of-K; 1/5 Top-1 with TTT | High for one clean N=5 validation; broad gain not established. |
| Search increased candidate recall on diagnostic cohort | `reports/public_reference_ablation_partial_v32.md`; config `PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json` | A 19/25 → C 22/25 Any-of-K | High for partial frozen30 cohort; five incomplete tasks excluded. |
| Provenance-aware selection improved two-attempt diagnostic outcome | same partial report | C 9/25 → D 13/25 two-attempt exact | High for diagnostic completion subset; not target-tuned or generalizable without validation. |
| A/B held-out direction favored B for two attempts | local `artifacts/untouched60_manifest.json`, `untouched60_scored.json` | A 18/60; B 21/60; net +3; p=0.375 | High artifact integrity, limited statistical power; do not call significant. |
| Dynamic scheduling improves modeled utilization | local `artifacts/untouched60_dynamic_queue_simulation.json` | 7,557.76 s → 5,564.09 s; utilization 72.81% → 98.90% | High for deterministic replay; not measured live throughput. |
| Four L4 runtime was available historically | `experiments/results/QWEN3_8B_COMPETITION_RUNTIME_AUDIT.json` | 4 × NVIDIA L4, 23,034 MiB each | High for runtime audit; unrelated to solver success. |

## Controlled evaluation discipline

The relevant frozen configurations specify target-blind generation and post-freeze scoring. `untouched60_manifest.json` states that 60 task IDs were immutably selected before inference and target access; `untouched60_scored.json` states that solutions loaded only after A/B predictions were complete. The two-attempt comparison is paired: B-only solved 4 tasks and A-only solved 1, yielding an exact two-sided binomial p-value of 0.375 over five discordant pairs. This does not meet a conventional significance threshold.

Candidate recall/Any-of-K is a diagnostic: it asks whether at least one candidate equals a target after scoring. Top-1 asks whether the first predeclared rank is exact. Two-attempt exact asks whether either of two distinct predeclared outputs is exact. None should be substituted for another.

## Current architecture

The research code contains two complementary paths:

- **Symbolic/neuro-symbolic:** `Evidence → Rule Recognition → Parameter Inference → RuleSpec → deterministic Executor → HardVerifier`. Its purpose is transparency and train-pair constraint checking. Current known limits include representation coverage, repeat semantics, composition/state updates, and recognition.
- **Native test-time candidate path:** a public Qwen/NVARC-derived model receives a strict grid serialization, candidates are generated under fixed reversible views/search budgets, exact outputs are deduplicated with provenance, and a fixed train-only score selects two distinct attempts. It is a deployment and selection study around public pretrained weights, not an author-trained foundation model.

## Limitations and open questions

- Small and heterogeneous diagnostic cohorts make strong causal or statistical claims inappropriate.
- Candidate pool size can improve oracle recall without improving Top-1; selection is unresolved.
- Train-exact RuleSpec coverage does not test unseen-output generalization.
- Public pretrained model dependence makes results sensitive to model/interface availability.
- Dynamic scheduling evidence replays historical runtimes; it does not measure a post-change GPU run.
- Cohort construction, even when precommitted, may create cohort-specific effects.
- Internal ARC diagnostics and competition hidden evaluation differ.

Open questions are therefore: Which train-only features predict candidate correctness? Can provenance improve a selector on a larger preregistered cohort? Which RuleSpec extensions improve representability without introducing brittle templates? How can test-time compute be allocated adaptively without changing task semantics or leaking labels?
