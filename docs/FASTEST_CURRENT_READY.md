# Frozen current fastest configuration reference

The owner's terms **最快配置 / 历史最快配置 / 当前最快配置** now refer to
[`FASTEST_CURRENT_READY_V1`](../configs/runtime/FASTEST_CURRENT_READY_V1.json).
This is a stable configuration alias, not a claim of mathematically optimal
throughput or an all-hardware benchmark winner.

Use the exact READY engine at execution commit
`cde188b600600c17cf5c9419db1cfd8cd4898f74`: persistent model, same-task adapter
pooling, exact-root admission, independently owned chunked KV, dynamic vacant
lane refill, per-forward compatibility selection, fairness and live-KV capacity
protection. Do not revert to a fixed B1/B4 loop or force eight incompatible
views into one physical batch.

The current Eval60 CPU audit covered **89 outputs x 8 views** using the actual
native tokenizer and production prompt builder: **53 outputs have 4+4 root
groups, 36 have eight equal roots, none have 7+1**. The groups describe root
compatibility, not guaranteed physical batch width throughout DFS branching.
Single-output AUG8 has at most eight lanes; B16 requires enough compatible
same-adapter cells across multiple outputs. Long roots and live KV growth
reduce the ceiling to B8/B4/B2 as recorded in the JSON profile.

At the audit snapshot (`2026-10-15T15:04:38Z`), the restored worker was running
attempt zero, with no READY OOM failure receipts and no OOM/Traceback in its
active generation log. Its isolated short/long R128 canaries passed. The formal
pool remained **59/89 complete**, including preserved earlier outputs. Therefore
the evidence supports **no OOM observed to date**, not guaranteed zero OOM for
the unfinished full run. Full R1024 throughput optimality is not established.

This registration changes neither live code nor the running checkout, source
identity, scientific budgets, candidate evidence or authorization. It does not
restart generation, open Gold or freeze an incomplete scientific result.

Before future reuse, resolve the profile and its source hashes. Report any
different hardware, source, cache/batch policy or fallback as a different
configuration rather than silently calling it this frozen fastest reference.
