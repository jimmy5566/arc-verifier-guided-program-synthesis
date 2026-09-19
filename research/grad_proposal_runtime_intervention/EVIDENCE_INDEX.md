# Evidence index

| Stage | Question | Frozen result | Proposal role |
|---|---|---|---|
| Stage-0 | Can task outcome mislabel selector/verifier reliability? | Frozen30/Frozen60 distinguish candidate-generation limits from selection failures. | ARC motivation for responsibility mismatch. |
| Stage-1 | Is telemetry-only attribution feasible? | Feasible in a controlled synthetic CPU-only sandbox. | Establishes a learnable telemetry signal. |
| Stage-2A | Does calibration automatically improve decisions? | No: a cheap, effective strong verifier can make conservative routing optimal. | Prevents overclaiming from calibration alone. |
| Stage-2B | When is attribution economically useful? | Positive utility region emerges at intermediate/high verification cost. | Connects attribution to intervention economics. |
| Stage-2C | Does responsibility beat uncertainty for multiple interventions? | Attribution-aware routing beats uncertainty-only and simple routing. | Core heterogeneous-intervention evidence. |
| Stage-3 | Does the effect survive ambiguity and limited budget? | Positive under single/dual faults, partial repairs, budgets 1/2, and OOD dual shift. | Strongest current controlled preliminary evidence. |

The causal chain is deliberately narrow: frozen ARC evidence motivates the
problem; synthetic stages test the proposed runtime-intervention mechanism.
Neither is a real interactive-LLM benchmark.
