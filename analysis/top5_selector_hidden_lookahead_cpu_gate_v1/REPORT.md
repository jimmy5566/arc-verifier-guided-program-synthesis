# Top5 selector / hidden-state / shallow-lookahead CPU audit v1

Frozen frontier source: `2da4ef598cc8e792ddab5b9f9d85036e1b0792ce`

## 1. Top5 telemetry selector

Cumulative-NLL baseline over 64,005 true Gold-opportunity states:
- Top1: 32.91%
- Top2: 48.43%
- Top5: 80.72%
- MRR: 0.5204

12-way LOOTO Top5-only selectors:
- Candidate Logistic: Top1 32.48% (-0.42pp), Top2 55.13%, MRR 0.5336; bootstrap DeltaTop1 95% CI [-8.54pp,+5.18pp]; 1 catastrophic task.
- Pairwise Logistic: Top1 30.84% (-2.07pp), Top2 49.01%, MRR 0.5121; bootstrap DeltaTop1 95% CI [-11.24pp,+4.13pp]; 3 catastrophic tasks.

Decision: **NO GPU Top5 telemetry selector.** Existing scalar telemetry can improve Top2/MRR in one formulation but does not improve the primary Top1 decision robustly.

## 2. Passive hidden-state selector

No usable hidden-state archive exists in the frozen frontier dataset. A passive parent-prefix representation is feasible without extra model forwards:
- 33,203 unique Top5 work items
- 32,680 unique Top5 parent keys
- max simultaneous active parent groups: 268
- frozen layers: 11, 23, 35 (0-based)
- hidden size: 2560
- estimated 3-layer BF16 archive: 478.7 MiB
- worst observed active cache estimate: 3.93 MiB

Decision: **authorize CPU Gate0 implementation and a later 3-output GPU smoke for passive hidden-state collection.** Child-prefix hidden states remain forbidden because unexpanded children would require extra forwards.

## 3. Shallow lookahead

Existing P3 trees cannot causally evaluate equal-budget Top5 lookahead:
- q=8: all five Top5 branches have >=8 observed subtree expansions in only 5,062/64,005 states (7.91%), covering only 3 tasks.
- q=16: 4,839/64,005 (7.56%), covering only 3 tasks.

On this selection-biased observational subset, Top1 does not beat cumulative NLL for q=8 or q=16. Therefore no accuracy claim is authorized.

Decision: **prepare a runtime-safe q=8 Gate0 only; do not authorize Micro12 accuracy evaluation before a target-blind 3-output smoke proves sane runtime.**
