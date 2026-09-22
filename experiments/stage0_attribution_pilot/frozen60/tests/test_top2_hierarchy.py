def test_top2_hierarchy_is_nested_in_candidate_coverage():
    rows = ([{"any": True, "top1": True, "top2": True}] * 16 +
            [{"any": True, "top1": False, "top2": True}] * 5 +
            [{"any": True, "top1": False, "top2": False}] * 9 +
            [{"any": False, "top1": False, "top2": False}] * 30)
    assert sum(row["any"] for row in rows) == 30
    assert sum(row["top1"] for row in rows) == 16
    assert sum(row["top2"] for row in rows) == 21
    assert all(not row["top2"] or row["any"] for row in rows)
