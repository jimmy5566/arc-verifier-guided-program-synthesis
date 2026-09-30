# REGRET_SHARED_B1_CALL_HISTORY_AUDIT_V1 — partial stop snapshot

This snapshot was deliberately stopped by the user before the forced-first,
prefill-after-history, and per-forward Phase-F diagnostics completed.  It is
not a completed call-history audit and must not be used to claim a final root
cause.

## Preserved target-blind evidence

- Source implementation commit: `dfea27d923651be09caa31e8ac041df1706ebcdc`.
- Remote partial-evidence commit: `e45b2d6c2212e69e028ff5b3926deb48d028879b`.
- Gold was not opened; Dynamic Batch2 and Untouched12/Untouched24 were not
  used.
- The first 50 frozen B1 scheduler selections are all `anti_transpose`; the
  scheduler therefore did not interleave cells in that observed prefix.
- Two immediate fresh-model replays of the frozen `flip_ud` first incremental
  request were byte-identical for the recorded full-logits SHA and had zero
  recorded full/ARC logit and ARC-logprob delta.
- The history-length-0 replay was identical.  After exactly one normal
  `anti_transpose` incremental forward, replay of a deep clone of the same
  frozen `flip_ud` request retained the same cache checksum but differed:
  `max_abs_full_logit_delta=38.375`,
  `max_abs_arc_logit_delta=50.79766845703125`, and
  `max_abs_arc_logprob_delta=38.375`.

The result is evidence worth preserving for a later diagnosis, but it does
not distinguish model-runtime state from every possible unmeasured executor
state mechanism.  The stopped phases are explicitly `NOT_RUN`.
