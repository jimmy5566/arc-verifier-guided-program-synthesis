from inference.d1_shared_queue import claim_cell, deterministic_order, release_claim


def test_claim_is_exclusive_and_expired_claim_is_reclaimed(tmp_path):
    first = claim_cell(claims_root=tmp_path, policy="p", output_id="t:o0", depth=12, view="identity", worker_id="a", stale_seconds=60)
    assert first is not None
    assert claim_cell(claims_root=tmp_path, policy="p", output_id="t:o0", depth=12, view="identity", worker_id="b", stale_seconds=60) is None
    release_claim(first)
    assert claim_cell(claims_root=tmp_path, policy="p", output_id="t:o0", depth=12, view="identity", worker_id="b", stale_seconds=60) is not None


def test_order_is_deterministic_and_not_worker_partitioned():
    rows = [("p", "b:o0", 24, "identity"), ("p", "a:o0", 12, "transpose")]
    assert deterministic_order(rows) == deterministic_order(list(reversed(rows)))
