# SEARCH_ORDER_FRONTIER_TELEMETRY_MICRO12_V1 test plan

This plan was frozen before telemetry implementation and before any GPU work.

## Gate 0: CPU implementation and semantic audit

Run the established search-order test suites:

```text
pytest -q tests/test_search_order.py
pytest -q tests/test_nvarc_turbodfs_search_order.py
pytest -q tests/test_audit_search_order_adapters.py
pytest -q tests/test_build_search_order_micro24_cohort.py
```

Add telemetry-specific tests that prove all of the following.

1. Every retained non-EOS work item receives exactly one insertion record.
2. `work_item_id` is unique within a cell.
3. Parent-work-item lineage is valid.
4. Prefix hashes are deterministic under the frozen canonical byte representation.
5. Each child hash is its parent prefix plus the selected token.
6. Discrepancy count follows exact LDS lineage.
7. Local rank is the pre-retention legal-token rank, never a frontier-floor restore rank.
8. Insertion expanded-node counters are monotonic and legal.
9. Expanded work items receive exactly one pop record.
10. Never-expanded retained work items retain null pop fields.
11. Enabling telemetry preserves retained membership.
12. Enabling telemetry preserves scheduler pop order.
13. Enabling telemetry adds no model forward.

The source-diff audit must explicitly confirm no scientific change to `_retained`, LDS priority, regret threshold, candidate cap, node budget, EOS behavior, frontier floor, cache behavior, physical scheduler, or model call sites.

## Gate 1: target-blind semantic parity smoke

Use the already-frozen deterministic P3 S/M/L smoke selection: three outputs, canonical AUG8, d24, R1024, `LDS_UNIT_DISCREPANCY_V1`. Gold remains unopened.

At R256, R512, R768, and R1024 compare against the frozen P3 reference:

- candidate-pool snapshot and SHA256;
- completed candidate count;
- actual node count;
- termination reason;
- R1024 or terminal-carry classification; and
- available search-order behavior.

Require zero duplicate inserts and pops, invalid lineage, prefix-hash failures, discrepancy-lineage failures, expanded-without-pop, and pop-without-insert. Final pending records must have an insertion, null pop fields, and a legal status.

Record wall time, useful/replay forwards, replay ratio, effective batch, wall censoring, OOM fallback, telemetry rows, and compressed telemetry bytes. Stop before Micro12 if any semantic discrepancy, telemetry-integrity failure, OOM, wall censoring, or pathological replay increase occurs.

## Gate 2: frozen Micro12 collection

Only after Gate 0 and all Gate 1 requirements pass, collect exactly the frozen 12-output Micro12 cohort under canonical AUG8, d24, R1024, and P3 LDS. Freeze each output atomically only after raw/checkpoint/candidate/runtime/telemetry receipts and hash verification pass.

Gold remains unavailable throughout Gate 2. After 12/12 outputs, generate and verify the generation and frontier-telemetry manifests, freeze and push compact pre-Gold evidence, and only then begin the separately controlled post-freeze analysis.
