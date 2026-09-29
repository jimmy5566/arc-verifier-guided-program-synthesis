# V5 exact-reuse audit

This CPU audit did not read evaluation solutions.

- Historical cells available: 144/144
- Current D1 V5 parity cells observed: 90
- Historical config hash matches: `True`
- Exact historical reuse: **NO**
- Parity mismatches among rerun cells: 46

## Identity blockers

- D1 adds max_expanded_nodes=4096 whereas historical V5 declares non_reference_caps=[]
- D1 adds max_completed_candidates=32 whereas historical V5 declares non_reference_caps=[]
- D1 uses a separate decoder implementation module; source identity is not the frozen V5 implementation

Because identity differs materially, no finalist GPU smoke is authorized by this audit. The shared-queue scheduler may be used only after an explicitly frozen compatible decoder contract exists.
