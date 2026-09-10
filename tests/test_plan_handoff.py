"""The plan-handoff policies, without a robot.

`--mock` runs `_run_serial`; every line of handoff logic lives in
`_run_overlapped`, which only a real run reaches. So the timing arithmetic
is tested here directly instead, by replaying synthetic solve-time traces
through the same `publish_index` the publisher thread calls and the same
anchor rule each policy uses.

The publisher anchors each plan at `t_loop + lat` and indexes it by how far
past that anchor the present is. A solve that beats `lat` publishes before
its own anchor, making that offset negative -- which is the case these pin.
"""

from __future__ import annotations

import pytest

from oim.worlds.real3d.run_real import publish_index


CONTROL_DT = 0.02          # 50 Hz publisher
PLAN_SPAN = 1.6            # horizon 32 x planning_dt 0.05
N_SAMPLES = int(PLAN_SPAN / CONTROL_DT)


# --------------------------------------------------------------------------
# publish_index: the one branch the publisher actually takes
# --------------------------------------------------------------------------

def test_publish_index_enters_at_zero_on_anchor():
    assert publish_index(0.0, N_SAMPLES, CONTROL_DT) == 0


def test_publish_index_walks_forward_with_elapsed():
    assert publish_index(0.13, N_SAMPLES, CONTROL_DT) == 6
    assert publish_index(0.40, N_SAMPLES, CONTROL_DT) == 20


def test_publish_index_clamps_negative_elapsed_to_the_head():
    """The bug this replaces: int() truncates toward zero, so a plan
    published 0.07 s before its anchor indexed s[-3] -- near-goal commands
    from the plan's TAIL, sent while the arm is nowhere near the goal."""
    assert int(-0.07 / CONTROL_DT) == -3          # the old expression
    assert publish_index(-0.07, N_SAMPLES, CONTROL_DT) == 0


@pytest.mark.parametrize("early", [-0.001, -0.07, -0.25, -10.0])
def test_publish_index_never_returns_a_negative_index(early):
    assert publish_index(early, N_SAMPLES, CONTROL_DT) == 0


def test_publish_index_returns_none_past_the_plan():
    assert publish_index(PLAN_SPAN + 0.01, N_SAMPLES, CONTROL_DT) is None


def test_publish_index_stays_in_range_at_the_boundary():
    idx = publish_index(PLAN_SPAN, N_SAMPLES, CONTROL_DT)
    assert idx == N_SAMPLES - 1
