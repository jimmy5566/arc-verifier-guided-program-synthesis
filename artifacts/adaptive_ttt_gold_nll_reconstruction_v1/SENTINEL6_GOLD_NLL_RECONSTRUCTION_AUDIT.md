# Sentinel6 Gold-NLL Reconstruction Audit

## 1. Executive conclusion
- **REJECTED_FOR_RECONSTRUCTED_TRAJECTORY_GOLD_NLL_ROUTING**; H3b: **NOT_SUPPORTED_ON_RECONSTRUCTED_SENTINEL6**; interpretation: **CASE_C_RECONSTRUCTION_LIMITED**.
- All seven frozen LOO Top1/Top2/Top4 selections missed the reconstructed Gold-NLL-best depth-view cell; loss curves were correlated but not identical, so this does not prove the same failure for unpreserved original adapter states.
- This is Sentinel6-only reconstructed-trajectory evidence, not a hidden-test or production-decoding claim.

## 2. Provenance
- Experiment: `RECONSTRUCTED_TRAJECTORY_GOLD_NLL_AUDIT`.
- Frozen raw Transfer30 archive SHA256: `84cdf9faced48ac98503f580c4355de234ced2b397c5d5155c72cf615e136ce4`.
- Source archive commit: `7466df3d336335218463d341cee141b975311995`; task IDs SHA256: `b25ba34222f79529f90e91e6e94c81d5efd6005c3fbac85fbee11cbe61c0ba3b`.
- Tasks/outputs/teacher-forced cells: 6 / 7 / 280. Gold answers were used only after the target-blind frozen LOO reference was loaded, and only as teacher-forced targets.

## 3. Why exact frozen-state replay is impossible
- Original task checkpoints preserved loss curves and summary metadata, not LoRA adapter tensors. Exact adapter replay is therefore impossible.
- Per-step historical learning rates, token counts, and exact sample-order metadata were also not retained; all are explicitly marked `NOT_RETAINED_IN_FROZEN_TRAJECTORY` in the reconstruction table.

## 4. Reconstruction quality
- `21897d95`: CLOSE; loss correlation=0.998944, mean |delta|=0.001888, max |delta|=0.009423, first measurable divergence step=2.
- `28a6681f`: CLOSE; loss correlation=0.998504, mean |delta|=0.001058, max |delta|=0.010256, first measurable divergence step=2.
- `89565ca0`: MODERATELY_DIFFERENT; loss correlation=0.994862, mean |delta|=0.011059, max |delta|=0.063601, first measurable divergence step=2.
- `a47bf94d`: MODERATELY_DIFFERENT; loss correlation=0.992417, mean |delta|=0.000659, max |delta|=0.005472, first measurable divergence step=2.
- `e87109e9`: MODERATELY_DIFFERENT; loss correlation=0.990230, mean |delta|=0.001221, max |delta|=0.009211, first measurable divergence step=2.
- `fc7cae8d`: MODERATELY_DIFFERENT; loss correlation=0.980503, mean |delta|=0.003077, max |delta|=0.020028, first measurable divergence step=2.
- These labels are descriptive post-freeze summaries, not preregistered acceptance thresholds; numerical columns are authoritative.

## 5. Gold Test NLL surface results
- Every task was reconstructed once along continuous all-train TTT 0→72; all 8 geometry views were teacher-forced at depths 0, 12, 24, 48, and 72.
- Gold-best depth counts: `{'0': 1, '12': 0, '24': 3, '48': 2, '72': 1}`; Gold-best view counts: `{'identity': 1, 'rot90': 4, 'rot180': 1, 'rot270': 0, 'flip_lr': 0, 'flip_ud': 0, 'transpose': 0, 'anti_transpose': 1}`.

## 6. Frozen LOO vs Gold-NLL ranking
- Exact Gold-best joint-cell retrieval: Top1=0/7, Top2=0/7, Top4=0/7.
- Median LOO rank of Gold Top1=25.0; median Gold rank of LOO Top1=12.0; median Spearman=0.248968; median Kendall=0.141026.
- Mean Gold-NLL regret (Top1/Top2/Top4)=0.023496/0.023496/0.015691.

## 7. Depth recovery
- Top1/Top2/Top4 depth coverage=2/7, 3/7, 3/7.

## 8. View recovery
- Top1/Top2/Top4 view coverage=0/7, 0/7, 2/7.

## 9. Depth×view recovery
- Exact joint Top1/Top2/Top4=0/7, 0/7, 0/7. All observed misses are preserved in `sentinel6_loo_vs_gold_rank.csv`.

## 10. Greedy vs Gold-NLL
- Frozen old greedy-exact cells (4): fc7cae8d:0@d48/anti_transpose -> Gold rank 2 (NLL 0.003784); fc7cae8d:0@d72/anti_transpose -> Gold rank 1 (NLL 0.003178); a47bf94d:0@d48/rot90 -> Gold rank 1 (NLL 0.003308); a47bf94d:0@d72/flip_ud -> Gold rank 2 (NLL 0.003816).
- Outputs whose Gold-best reconstructed path is also an old greedy-exact cell: 2/7: a47bf94d:0@d48/rot90, fc7cae8d:0@d72/anti_transpose.
- Gold-best reconstructed paths that were old-greedy-wrong: 5/7.

## 11. Gold path accessibility
- Accessibility counts: `{'SEARCH_FRIENDLY': 5, 'GREEDY_FRIENDLY': 2}`. This measures teacher-forced token accessibility, not achieved candidate recall.
- No autoregressive generation was performed, so this audit cannot establish that a search procedure would actually recover a candidate.

## 12. MEASURED
- Frozen LOO rank rows, 280 Gold-NLL cells, real-test teacher-forced token ranks/margins, reconstruction losses, and all derived ranks in the CSV/JSON artifacts.

## 13. INFERRED
- The frozen LOO ranking has no observed exact Gold-best joint-cell retrieval on this reconstructed Sentinel6 surface; the evidence does not support using it as a faithful real-test Gold-NLL router here.

## 14. NOT ESTABLISHED
- Exact original-adapter-state parity; hidden-test performance; candidate-generation recall; selector improvement; or TurboDFS benefit.

## 15. Decision gate
- CASE_C_RECONSTRUCTION_LIMITED: reconstructed trajectory differences limit causal attribution, while the measured rank-transfer failure provides no affirmative H3b evidence.
- Therefore no TurboDFS or expanded LOO-routing experiment is authorized by this audit. Stop after Sentinel6.
