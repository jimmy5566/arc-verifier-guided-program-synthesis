# Research contribution matrix

## Scientific contributions

| Contribution | Research question | Implementation | Experiment | Evidence | Result | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| RuleSpec pipeline | Can representation separate recognition from execution? | evidence, typed values/selectors, executor, verifier | V3 construction push | `reports/arc2_v3_complete_rulespec_construction_push.md` | 3/30 → 10/30 train coverage | train-only, oracle-family audit |
| Hard verification | Can train-pair consistency be made deterministic? | executor + exact verifier | V2/V3 protocols | configs and tests | auditable acceptance gate | does not verify test correctness |
| Candidate/selection analysis | Are recall and ranking distinct? | dedup, scoring, rank forensics | frozen30 | local ranker forensic artifact | 21/30 Any-of-K vs 9/30 Top-1 | diagnostic cohort |
| Provenance-aware selection | Can output support improve fixed two-attempt choice? | support/persisted-likelihood aggregation | untouched60 A/B | local untouched60 artifacts | 18/60 → 21/60 two-attempt | p=0.375; limited power |

## Engineering contributions

| Contribution | Research question | Implementation | Experiment | Evidence | Result | Limitation |
| --- | --- | --- | --- | --- | --- | --- |
| Reversible augmentation/search | Can controlled views diversify outputs? | 32 views, bounded branches, dedup | frozen30 A/C/D | `reports/public_reference_ablation_partial_v32.md` | 19/25 → 22/25 Any-of-K | partial cohort |
| Checkpoint/resume | Can long runs remain recoverable? | atomic task checkpoints, no-duplicate resume | production runners | `src/inference/`, tests, reports | operational infrastructure | not a quality result |
| Dynamic 4-worker scheduler | Can scheduling improve throughput without semantic change? | shared queue, complete model per L4 | historical runtime replay | local dynamic simulation artifact | 26.38% modeled makespan reduction | CPU-only replay |

## Negative findings

| Finding | Evidence | Interpretation |
| --- | --- | --- |
| V2 Macro DSL 0/50 exact in both routes | `LLM_PROGRAM_SYNTHESIS_V2_PILOT_50.json` | end-to-end architecture operated but hypothesis quality/interface grounding was inadequate |
| Semantic compiler validity was not semantic success | `C3_SEMANTIC_INTENT_AUDIT_V1.json` | 60/60 compile validity did not remove grounding errors |
| Multi-view likelihood did not raise Top-1 in N=8 | `reports/arc2_native_likelihood_ranker_v2_fast.md` | robustness features alone were insufficient |

## Open questions

| Contribution area | Open question |
| --- | --- |
| Candidate selection | Which train-only or provenance features generalize to a larger preregistered cohort? |
| RuleSpec | How can stateful composition and repeat semantics expand coverage without brittle task templates? |
| Evaluation | How can adaptive compute be studied while preserving target-blind frozen protocols? |
