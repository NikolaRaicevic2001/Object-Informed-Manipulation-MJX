"""The plan-handoff policies, without a robot.

`--mock` runs `_run_serial`; every line of handoff logic lives in
`_run_overlapped`, which only a real run reaches. So the timing arithmetic
is tested here directly instead, by replaying synthetic solve-time traces
through the same `publish_index` the publisher thread calls and the same
anchor rule each policy uses.

What the two policies are, in one line each (full version in
`_run_overlapped`'s docstring):

* responsive    -- anchor = t_loop + EMA(solve); publish when ready.
* deterministic -- anchor = t_loop + t_c (fixed); wait for the anchor.

`simulate` below is the loop's timing skeleton and nothing else: no JAX, no
MuJoCo, no threads. It is deliberately a re-statement of the schedule rather
than a call into it, because the schedule is spread across a thread closure
and a solve loop that cannot run without hardware -- but the one piece with
a real branch in it, `publish_index`, is imported, not copied.
"""

from __future__ import annotations

import numpy as np
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


# --------------------------------------------------------------------------
# the two schedules
# --------------------------------------------------------------------------

def simulate(solve_times, policy, t_c=0.5, latency_comp=0.27):
    """Replay one solve-time trace through one handoff policy.

    Returns a dict of per-cycle series:
        period    -- t_loop to t_loop, what the arm sees as replan spacing
        staleness -- age of the state a freshly published plan was solved
                     from, measured at the instant it is published. This is
                     the quantity that matters: it is how out of date the
                     command stream is.
        wait      -- idle time spent holding a finished plan for its anchor
        early     -- cycles where the plan was ready before its anchor
    """
    lat = float(latency_comp)
    t = 0.0
    out = {"period": [], "staleness": [], "wait": [], "early": []}

    for solve in solve_times:
        t_loop = t
        anchor = t_loop + (t_c if policy == "deterministic" else lat)
        ready = t_loop + solve

        if policy == "deterministic":
            t_pub = max(ready, anchor)          # wait for the anchor
        else:
            t_pub = ready                       # publish immediately

        out["wait"].append(t_pub - ready)
        out["early"].append(ready < anchor)
        # The plan carries the state read at t_loop, predicted forward to
        # the anchor. Its staleness on publication is measured against that
        # prediction, which is what the planner believes it solved for.
        out["staleness"].append(t_pub - t_loop)

        if policy == "responsive":
            lat = 0.8 * lat + 0.2 * (t_pub - t_loop)

        t = t_pub
        if out["period"]:
            pass
        out["period"].append(t_pub - t_loop)

    return {k: np.asarray(v) for k, v in out.items()}


# a realistic ADMM trace: ~0.28 s solve with jitter, plus two slow outliers
RNG = np.random.default_rng(0)
SOLVES = np.clip(RNG.normal(0.28, 0.05, 200), 0.12, None)
SOLVES[50] = 0.62
SOLVES[120] = 0.71


def test_deterministic_gives_a_constant_period():
    r = simulate(SOLVES, "deterministic", t_c=0.5)
    # Every cycle that finished inside t_c is exactly t_c long.
    normal = r["period"][r["early"]]
    assert np.allclose(normal, 0.5)
    assert normal.size > 190          # the overwhelming majority


def test_responsive_period_tracks_the_solve():
    r = simulate(SOLVES, "responsive")
    assert np.allclose(r["period"], SOLVES)


def test_responsive_is_faster_but_more_variable():
    a = simulate(SOLVES, "responsive")
    b = simulate(SOLVES, "deterministic", t_c=0.5)
    assert a["staleness"].mean() < b["staleness"].mean()
    assert a["staleness"].std() > b["staleness"].std()


def test_deterministic_never_publishes_before_its_anchor():
    """The property that makes the clamp unnecessary in this mode."""
    r = simulate(SOLVES, "deterministic", t_c=0.5)
    assert (r["wait"] >= 0.0).all()
    # and the slow outliers are the only cycles that do not wait
    assert (r["wait"][~r["early"]] == 0.0).all()


def test_responsive_never_waits():
    r = simulate(SOLVES, "responsive")
    assert (r["wait"] == 0.0).all()


def test_a_too_small_t_c_degenerates_to_responsive():
    """`t_c` below the solve time makes every cycle late, so the mode stops
    being deterministic -- it becomes `responsive` with a stale constant
    anchor. Guards against sizing `t_c` off the loop period by mistake."""
    r = simulate(SOLVES, "deterministic", t_c=0.05)
    assert not r["early"].any()
    assert np.allclose(r["period"], SOLVES)
