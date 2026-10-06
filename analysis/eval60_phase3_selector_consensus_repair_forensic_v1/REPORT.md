# Consensus Repair Forensic V1

Classification: `CONSENSUS_REPAIR_SIGNAL_WEAK`.

## MEASURED

- CPU-only. No GPU, model call, DFS, TTT, candidate regeneration, or S2 modification.
- C1 was frozen before Gold: take the first five frozen S2 candidates having the same shape as S2 rank 1, then compute cellwise modal color; ties use the highest-ranked S2 candidate among tied colors.
- Pre-Gold C1 receipt SHA256: `06dc00b58ae1a11fb87905a24043c9722d450c3d03fb2cf201c8949d7c21bf91`.
- C1 produced a repair on 34/35 outputs.
- 26/34 C1 outputs were already existing frozen candidates; 8 were genuinely new grids.
- Among the eight S2 Top2 misses, C1 exactly recovered Gold on only `b5ca7ac4:o0`, where Gold was already frozen S2 rank 4.
- Exact repairs among S2 misses: 1/8.
- Hypothetical distinct-slot2 substitution: baseline 27/35 -> 25/35, with 1 FIX, 3 HARM, net -2.
- HARM outputs: `20270e3b:o1`, `8e5c0c38:o1`, `78332cb0:o1`.

## INFERRED

Local cellwise consensus sometimes identifies a lower-ranked correct candidate, but the mechanism is not selective enough to allocate a submission slot safely.

## NOT ESTABLISHED

This does not rule out more semantic or learned consensus methods. It does not justify tuning K, disagreement thresholds, shape groups, or task-specific repair rules on this exposed cohort.

## Decision

Do not promote C1 into an operational selector. Close this exact consensus-repair rule unless genuinely new evidence appears.
