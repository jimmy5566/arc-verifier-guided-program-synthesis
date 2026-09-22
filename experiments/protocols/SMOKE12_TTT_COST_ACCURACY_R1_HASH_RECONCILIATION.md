# Smoke12 Cohort Hash Reconciliation

This is a provenance-only repair. Smoke12 task membership, selection order, scientific conditions, and frozen historical artifacts are unchanged.

## Preserved history

- Old declared hash: `ca4de9de17d90cbc958606bc8a5aab9729e028a742bf5756d97c5dae2e6b8561`.
- Old repository `_task_hash`: `4513f6e1f2aaa759344b9ce8c5b9f09cc6efbb9480dbd75b0566071920d823de`.

The old declared value was SHA-256 of newline-delimited IDs in selection order. The repository runner and scorer use SHA-256 of compact JSON containing lexicographically sorted task IDs. The encodings differ even though membership is identical.

## Canonical replacement

`SMOKE12_TASK_IDS_JSON_V1` is now authoritative:

```python
sha256(json.dumps(sorted(task_ids), separators=(",", ":")).encode("utf-8")).hexdigest()
```

The canonical replacement is `4513f6e1f2aaa759344b9ce8c5b9f09cc6efbb9480dbd75b0566071920d823de`.
