# Comparator fix audit

- The former raw comparator treated JSON list `[true]` and in-memory tuple `(true,)` as different.
- It also compared `elapsed_seconds`, which is explicitly wall-clock telemetry.
- Canonical parity now recursively normalizes tuple/list and removes only explicit timing fields.
- The Phase-2 full repeat payload was not frozen, so its new canonical parity cannot be recomputed honestly from the surviving CSV alone.
- Its `exact_parity=True` field and raw mismatches are preserved as historical evidence, not reinterpreted as a model finding.
