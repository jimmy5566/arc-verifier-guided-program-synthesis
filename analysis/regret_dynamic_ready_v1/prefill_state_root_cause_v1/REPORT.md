# REGRET_PREFILL_STATE_ROOT_CAUSE_V1

## MEASURED

- The tuple/list plus elapsed-time comparator false-failure mechanism is unit-tested.
- The old Phase-2 repeat payload was not frozen, so its corrected strict 4/4 value is INDETERMINATE rather than reconstructed.
- Foreign anti root prefill alone leaves both native and legacy-deep-clone flip continuation outputs strictly identical.
- One foreign anti incremental forward changes the frozen flip continuation full-logit hash; N=2/8/32 retain the same changed hash.

## NOT ESTABLISHED

- The first divergent transformer block cannot be observed in this runtime: Unsloth bypasses ordinary hooks and does not return hidden states.
- No minimal repair is identified; no micro-fix, final isolated B1, final Shared B1, or Dynamic B2 was run.

## DECISION

Stop for review. Dynamic B2 remains unsafe because Shared-B1 parity is not repaired.
