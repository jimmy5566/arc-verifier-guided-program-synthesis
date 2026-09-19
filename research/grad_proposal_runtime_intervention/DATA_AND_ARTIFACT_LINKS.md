# Data and artifact links

Paths below are exact repository-relative source paths at archive time. Links
are relative to this archive directory. `Copied` means the compact derived file
is versioned under `preliminary_results/` or `figures/`; `Referenced` means the
potentially large or frozen source is deliberately not duplicated.

| Stage | Source path and repository link | Contents | Archive status | SHA-256 |
|---|---|---|---|---|
| Frozen30 / Stage-0 | `artifacts/ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json` ([link](../../artifacts/ARC2_STRATEGY_AWARE_RANKER_FROZEN30_STRATEGY_ONLY_RERANKED_FROZEN.json)) | Frozen 30-task candidates, rankings, and scores used for responsibility-mismatch motivation. | Referenced; 2.35 MB frozen artifact. | `0D95D6B1582A010D9384C89D25E4C25A0FF09BB38DD7AAF7E6C77D3DB0843F82` |
| Frozen30 / Stage-0 | `experiments/stage0_attribution_pilot/outputs/descriptive_summary.csv` ([link](../../experiments/stage0_attribution_pilot/outputs/descriptive_summary.csv)) | Derived Frozen30 failure decomposition. | Copied as [stage0_frozen30_descriptive_summary.csv](preliminary_results/stage0_frozen30_descriptive_summary.csv). | `B936E56E48BCB3F280FEEEBD318543DA04CF3500FDA5FA4374109628848A45D0` |
| Frozen60 / Stage-0 | `experiments/stage0_attribution_pilot/frozen60/outputs/descriptive_summary.csv` ([link](../../experiments/stage0_attribution_pilot/frozen60/outputs/descriptive_summary.csv)) | Derived Frozen60 Any-of-K, Top-1/Top-2, and failure taxonomy. | Copied as [stage0_frozen60_descriptive_summary.csv](preliminary_results/stage0_frozen60_descriptive_summary.csv). | `E589E2FB7CB766F95C84E106B5A2824E2BD88197D9F28FE6B975743091CDA710` |
| Frozen60 source | `artifacts/untouched60_manifest.json` ([link](../../artifacts/untouched60_manifest.json)) | Immutable untouched60 membership, selection method, and source/config hashes. | Copied as [untouched60_manifest.json](preliminary_results/untouched60_manifest.json). | `4E361ABF8B99DB06B506B903BEA217A2431D137D3361A90C58379E3EE090CD98` |
| Frozen60 source | `artifacts/untouched60_v33_output_final/artifacts/untouched60/A_candidates_frozen.json` ([link](../../artifacts/untouched60_v33_output_final/artifacts/untouched60/A_candidates_frozen.json)) | Frozen untouched60 32-view native candidate pool. | Referenced; 5.10 MB frozen candidate pool. | `72B3E856CDC1259FD524D314A9109A1EECBDBBE1301D6230A500C97CD5AE5F15` |
| Frozen60 source | `artifacts/frozen60_support_ablation/B_SUPPORT_predictions_frozen.json` ([link](../../artifacts/frozen60_support_ablation/B_SUPPORT_predictions_frozen.json)) | Frozen B-SUPPORT two-attempt predictions. | Referenced; 0.80 MB frozen prediction artifact. | `EDE89ABE402062091DAD9FECFCE8842C8F6032915AC338E5AEE87FD283A52816` |
| Stage-1 | `experiments/stage1_interactive_attribution/outputs/attribution_metrics.csv` ([link](../../experiments/stage1_interactive_attribution/outputs/attribution_metrics.csv)) | Telemetry-attribution, ablation, and OOD metric rows. | Copied as [stage1_attribution_metrics.csv](preliminary_results/stage1_attribution_metrics.csv). | `7D6780A293207886BB86732CFB472173E0A66521B19D4B66E77BC80980AE7BE4` |
| Stage-1 | `experiments/stage1_interactive_attribution/outputs/reliability_results.csv` ([link](../../experiments/stage1_interactive_attribution/outputs/reliability_results.csv)) | Reliability update and policy results. | Copied as [stage1_reliability_results.csv](preliminary_results/stage1_reliability_results.csv). | `25CBCF11ED13E03BF2E6F209635A05CFA9BE963E585902202212ECB11073F876` |
| Stage-2A | `experiments/stage2a_state_conditional_attribution/outputs/aggregate_decision_results.csv` ([link](../../experiments/stage2a_state_conditional_attribution/outputs/aggregate_decision_results.csv)) | State-conditional calibration and fixed-policy utility aggregate. | Copied as [stage2a_aggregate_decision_results.csv](preliminary_results/stage2a_aggregate_decision_results.csv). | `1246FF133E83A0B94066EFD1039F749D828E4A6B01FF14A9284E0D2AC4F387C0` |
| Stage-2B | `experiments/stage2b_decision_utility/outputs/policy_sensitivity.csv` ([link](../../experiments/stage2b_decision_utility/outputs/policy_sensitivity.csv)) | Strong-verifier repair/cost utility sweep. | Copied as [stage2b_policy_sensitivity.csv](preliminary_results/stage2b_policy_sensitivity.csv). | `7D387FBB2B37DBF478A4544B336599D0347EE02ED352ADBF82A1565A36B5D89D` |
| Stage-2C | `experiments/stage2c_multi_intervention/outputs/policy_results.csv` ([link](../../experiments/stage2c_multi_intervention/outputs/policy_results.csv)) | Heterogeneous intervention policy, cost, and OOD results. | Copied as [stage2c_policy_results.csv](preliminary_results/stage2c_policy_results.csv). | `04848D7C9739FE81BE475AC9A9C50A5F2A9ACB0FA3536DB7E2A7AE6F7AD9EAD3` |
| Stage-3 | `experiments/stage3_multifault_budget/outputs/policy_results.csv` ([link](../../experiments/stage3_multifault_budget/outputs/policy_results.csv)) | Budgeted single/dual-fault policy utility results. | Copied as [stage3_policy_results.csv](preliminary_results/stage3_policy_results.csv). | `C625A1BF7761405DB88FCB372FF41F6CD774EAFB31DDAFA6DC5FED48DB5FC342` |
| Stage-3 | `experiments/stage3_multifault_budget/outputs/multilabel_attribution_metrics.csv` ([link](../../experiments/stage3_multifault_budget/outputs/multilabel_attribution_metrics.csv)) | Per-component and aggregate multi-label attribution metrics. | Copied as [stage3_multilabel_attribution_metrics.csv](preliminary_results/stage3_multilabel_attribution_metrics.csv). | Small derived CSV; hash retained by copied archive commit. |

## Copied figures

The `figures/` directory contains compact proposal-supporting renders copied
from Stage-0 Frozen60, Stage-2C, and Stage-3 outputs. They are illustrative
derived figures only. Their source-stage reports remain the authoritative method
and limitation record.

No model checkpoint, raw ARC challenge/solution dataset, Kaggle notebook,
submission, candidate-generation code, or competition result is copied into
this research archive.
