# LOO Transfer Failure Anatomy

## 1. Executive conclusion
- This CPU-only audit dissects a previously measured reconstructed-Sentinel6 rank-transfer failure; it does not retest whether LOO works.
- The reconstruction caveat applies to every causal interpretation.

## 2. What exactly was measured
- Seven outputs × forty depth/view cells, frozen LOO fold-NLL arrays, reconstructed teacher-forced Gold Test NLL, and target-blind train/test input features.

## 3. Reconstruction caveat
- Original adapter tensors were not retained. Gold NLL was measured on reconstructed trajectories, not exact original adapter states.

## 4. Cell-level rank failure
- Gold Top1 present in frozen LOO Top1/Top2/Top4: 0/7, 0/7, 0/7.

## 5. Depth bias
- LOO best depth early (0/12): 6/7; Gold best depth early: 1/7. This is descriptive shallow-to-late displacement only.

## 6. View bias
- All seven LOO best views differ from the Gold-best view; four Gold winners use rot90, while LOO winners use transpose, flip_ud, rot270, or identity.

## 7. Depth×view interaction
- Additive residuals are reported per output. Joint Top1/Top2/Top4 recovery is 0/7 despite nonzero marginal coverage, but Gold is not uniformly more interaction-heavy than LOO.

## 8. Fold-level anatomy
- Gold-cell fold NLL/rank distributions are in `gold_cell_fold_anatomy.csv`; some cells are never competitive, while one is strong on one fold and poor under mean aggregation.

## 9. Aggregation diagnosis
- Predefined mean/median/min/max/std summary ranks are diagnostic only in `aggregation_diagnostic.csv`; the evidence is mixed, not mean-only.

## 10. Train→test input shift
- Descriptive Spearman: nearest input shift vs Gold rank of LOO Top1=0.6071428571428571; vs regret=0.14285714285714285.

## 11. Reconstruction-drift analysis
- Descriptive Spearman: trajectory mean loss delta vs LOO Top1 Gold regret=0.5405624776173353. CLOSE trajectories also miss joint Top4, so drift is not sufficient evidence alone.

## 12. Failure taxonomy
- Labels are descriptive and secondary to the per-cell/fold measurements; ambiguous outputs remain mixed/unresolved.

## 13. Hypothesis status
- H-A: WEAKLY_SUPPORTED — Input-shift associations are descriptive only; N=7 and no classifier/regression was fit.
- H-B: WEAKLY_SUPPORTED — LOO early-depth winners=6/7; Gold early-depth winners=1/7. This is a descriptive shallow-to-late displacement in a reconstructed N=7 audit.
- H-C: WEAKLY_SUPPORTED — Joint Top1/Top2/Top4 recovery was 0/7 while marginal depth and view coverage were nonzero. Gold is not uniformly more interaction-heavy than LOO, so this is not a universal interaction claim.
- H-D: WEAKLY_SUPPORTED — One Gold cell improves to rank 3 under its frozen best-fold summary while mean ranks it 32, but other Gold cells remain poor under all predefined summaries; aggregation is not the whole failure.
- H-E: INSUFFICIENT — Fold-winner instability versus regret is descriptive at N=7.
- H-F: INSUFFICIENT — All results use reconstructed trajectories. CLOSE tasks also miss joint Top4, so drift alone is not established as sufficient.

## 14. MEASURED
- Derived ranks, curves, additive residuals, fold distributions, input-only shift features, and trajectory-loss associations.

## 15. INFERRED
- Any explanation of train-to-test mismatch is hypothesis-level because there are seven reconstructed outputs.

## 16. NOT ESTABLISHED
- Original adapter-state equivalence, hidden-test generalization, a replacement aggregation rule, or a production routing improvement.

## 17. Whether LOO deserves another routing experiment
- NO. The additive multi-fold basin diagnostics find no stable shared training-side competence basin or cross-fold winner transfer. This is still an explanatory reconstructed-Sentinel6 audit, not hidden-test evidence.

## 18. Multi-fold basin overlap
- Every output has median pairwise Top4 Jaccard 0 at the frozen fold surfaces. At epsilon=1e-3, every output also has median basin Jaccard 0 and an empty all-fold intersection.

## 19. Cross-fold transfer
- Fold winners are rank 1 on their own fold but have poor off-diagonal ranks; no output has a stable shared near-optimal region at epsilon=1e-3.

## 20. Train basin versus Gold Test NLL
- The Gold-NLL Top1 is in an epsilon=1e-3 majority-fold basin for 0/7 outputs. This is measured on reconstructed trajectories and does not establish hidden-test behavior.

## 21. Added hypotheses
- H-G: NOT_SUPPORTED. H-H: NOT_SUPPORTED. H-I: NOT_SUPPORTED.

## 22. Final routing decision
- NO. The frozen fold surfaces show disjoint training-side basins and poor cross-fold winner transfer, so another LOO routing experiment is not justified by this evidence.
