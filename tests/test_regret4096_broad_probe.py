from __future__ import annotations

from scripts.run_regret4096_broad_probe_v1 import DEPTH_VIEW_ORDER, build_schedule


def test_round_robin_schedule_is_deterministic_and_broad_first() -> None:
    schedule = build_schedule(["b:o0", "a:o0"])
    assert len(schedule) == 24
    assert [row["pass_index"] for row in schedule[:2]] == [1, 1]
    assert {row["output_id"] for row in schedule[:2]} == {"a:o0", "b:o0"}
    assert [row["pass_index"] for row in schedule[2:4]] == [2, 2]
    assert schedule[0]["depth"] == DEPTH_VIEW_ORDER[0][0]
    assert schedule[-1]["view"] == DEPTH_VIEW_ORDER[-1][1]
