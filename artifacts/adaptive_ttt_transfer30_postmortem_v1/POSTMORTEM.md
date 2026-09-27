# Adaptive TTT / LOO Transfer30 post-mortem

## Executive conclusion

**Recommendation: B — stop LOO as selector, but retain adaptive depth/view as a candidate-generation axis.**
The frozen LOO ranking did not transfer to the two Sentinel outputs for which the 40-cell surface contains an exact candidate: all exact cells ranked outside Top4. The result is a nonblind development mechanism result, not held-out generalization evidence.

## Final measured score reproduction

- Fixed TTT24 identity: 3/41
- LOO Top1: 2/41
- LOO Top2: 2/41
- LOO Top4: 4/41

## Sentinel6 forensic

- Outputs: 7; exact full-surface cells: 4; oracle outputs: 2.
- Exact-cell LOO ranks: [18, 24, 25, 32].
- Exact-cell rank coverage: Top1=0, Top2=0, Top4=0.
- Lower-NLL-is-better association: AUROC=0.75, AUPRC=0.04018927044078418, Spearman=0.044323166141441266.

| oracle output | exact depth×view cells (LOO ranks) | Top1 depth×view | Top4 depth-family / view-family coverage | basin |
|---|---|---|---|---|
| fc7cae8d:0 | 48/anti_transpose (r24), 72/anti_transpose (r25) | 12/flip_ud | depth=True; view=False; exact=False | MEDIUM |
| a47bf94d:0 | 48/rot90 (r18), 72/flip_ud (r32) | 12/identity | depth=False; view=True; exact=False | MEDIUM |

## Why LOO did not transfer

MEASURED: all real-test exact Sentinel cells were ranked 18, 24, 25, or 32 of 40 by the frozen mean held-out-train NLL. The selected Top4 never contained one. This is direct evidence against transfer of this ranking to the sampled real-test-optimal depth×view states.

INFERRED: the frozen teacher-forced objective rewards reconstruction of held-out train-pair trajectories, which can prefer states that do not extrapolate to the test input. The evidence is consistent with train-pair competence and test extrapolation being decoupled; it does not isolate a causal mechanism inside the adapter.

NOT ESTABLISHED: a universal claim that LOO can never be useful, a population estimate from the two positive Sentinel outputs, or a replacement selector.

## What Step1 actually established

The previously reported 8/8 Step1 pseudo-test result supports held-out-train reconstruction/competence retrieval in that pseudo-test regime. It did not establish that the same ranking chooses a state that extrapolates to a real test input. Transfer30 directly tests the latter and fails on the available Sentinel oracle outputs.

## Adaptive state versus router value

Adaptive-state value: **WEAK**. 2/2 Sentinel oracle outputs were exact under adaptive depth/view cells while Fixed TTT24 identity was not exact. The sample is only 2 oracle outputs.

## Failure-layer attribution

- **A — wrong ability measured:** not established as the primary failure. The historical Step1 pseudo-test supports a held-out-train reconstruction signal, but Transfer30 does not re-run that historical comparison.
- **B — train competence versus test extrapolation:** inferred plausible. Exact cells have ranks [18, 24, 25, 32] despite being real-test correct.
- **C — aggregation alone:** not sufficient. Correct cells are outside Top4, so a fixed Top1/2/4 aggregation boundary misses them before any final choice among those cells.
- **D — fold instability:** measured on exact cells in `sentinel6_exact_cell_ranks.csv`; fold-rank standard deviations are nonzero and range from 7.587 to 13.140.
- **E — narrow/exceptional real-test states:** measured descriptively. Each oracle output has 2/40 exact cells, hence MEDIUM under the predeclared 1 / 2-4 / >=5 bins but still only 5% of its surface.
- **F — state-space value:** weak positive evidence: both oracle outputs are absent from Fixed TTT24 identity and present at adaptive depth/view cells.
- **G — depth×view interaction:** primary observed selector failure. One oracle output has Top4 correct-view coverage but no correct-depth coverage; the other has Top2 correct-depth coverage but no correct-view coverage; neither has an exact selected cell.

## Hypotheses

- **H1: SUPPORTED** — Historical Step1's reported 8/8 pseudo-test result supports held-out-train competence retrieval in that train-reconstruction regime; it was not re-run here.
- **H2: SUPPORTED** — The frozen LOO protocol defines and scores train-pair competence basins. This support is limited to the train-pair teacher-forced objective, not test extrapolation.
- **H3: NOT SUPPORTED** — All 4 Sentinel exact cells ranked [18, 24, 25, 32]; none was in frozen Top4.
- **H4: WEAK** — 2/2 Sentinel oracle outputs were exact under adaptive depth/view cells while Fixed TTT24 identity was not exact. The sample is only 2 oracle outputs.

## Provenance and scope

This analysis is CPU-only and reads frozen raw artifacts. Evaluation solutions were opened only after the verified target-blind GPU freeze. The cohort is historically exposed but not adaptive-designed; it is not untouched, held-out, or final generalization evidence.
