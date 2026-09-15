# Experiment index

| Experiment | Configuration / report | Artifact | Result | Status |
| --- | --- | --- | --- | --- |
| Solver libraries V0–V3 | `scripts/run_solver_library_*.py` | `experiments/results/SOLVER_LIB_*.json` | V0 14/1,000; V1 6/1,000; V2 6/1,000 | completed, train-only baselines |
| V2 Macro DSL pilot | `configs/llm_program_synthesis_v2_pilot_50.json` | `LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json` | 0/50 in each symbolic/direct condition | completed negative result |
| Macro API/compiler studies | `configs/MACRO_API_*` | matching `experiments/results/` and `reports/` | compiler-valid 60/60 after constraints; semantic grounding 31/60 | completed API-only diagnostics |
| V3 RuleSpec construction | `scripts/run_v3_complete_rulespec_construction_forensics.py` | report `arc2_v3_complete_rulespec_construction_push.md` | 3/30 → 10/30 train coverage | completed train-only audit |
| Native likelihood ranker | `configs/ARC2_NATIVE_LIKELIHOOD_RANKER_V2_FAST.json` | report `arc2_native_likelihood_ranker_v2_fast.md` | 2/8 Top-1, 3/8 Any-of-K | one untouched cohort |
| TTT + verifier | `configs/ARC2_NATIVE_TTT_AND_STRONG_VERIFIER_V1.json` | report `arc2_native_ttt_and_strong_verifier_v1.md` | 1/5 → 2/5 Any-of-K | clean N=5 validation |
| Public-reference A/B/C/D | `configs/PUBLIC_REFERENCE_SELECTION_SEARCH_ABLATION_V1.json` | report `public_reference_ablation_partial_v32.md` | D: 22/25 Any-of-K, 9/25 Top-1, 13/25 two attempts | partial frozen30 diagnostic |
| Untouched60 A/B | `scripts/build_untouched60_manifest.py`, `score_untouched60_native_b.py` | local `artifacts/untouched60_*.json` | A 18/60; B 21/60 two attempts; p=0.375 | completed once; local/private detail |
| Dynamic scheduler replay | `scripts/simulate_dynamic_queue_historical.py` | local `artifacts/untouched60_dynamic_queue_simulation.json` | 26.38% modeled makespan reduction | CPU-only replay |

Historical/legacy components remain available for audit. “Completed” does not imply a claim of solver quality; read the linked report before comparing results across cohorts.
