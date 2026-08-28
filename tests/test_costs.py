"""The cost breakdown must agree with the cost the planner actually used.

`oim.utils.costs` restates the task's cost terms in numpy so a finished run
can be decomposed without re-entering `jit`. That restatement is the thing
most likely to drift: someone retunes a weight on the task and the figure
keeps reporting the old one, silently. These tests pin the two together.
"""

import copy
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oim.experiment import load_config
from oim.objects.planar_pushing import PlanarPushingObject, t_shape_footprint
from oim.tasks.pusht import PushT
from oim.utils.costs import TERM_ORDER, cost_series, cost_totals, summarize


@pytest.fixture(scope="module")
def _built() -> PushT:
    """One real task, built once: constructing it loads an MJCF.

    Built with the shipped `costs:` block, not `DEFAULT_COSTS`: the
    decomposition these tests pin is the one real runs are plotted with,
    and the two differ (`shaping_fade_dist` is 0.25 in the config and 0.0
    in the defaults, which alone zeroes every faded term).
    """
    return PushT(
        clutter=True,
        robot="xarm6",
        env="clutter",
        costs=load_config("xarm6")["costs"],
    )


@pytest.fixture
def task(_built: PushT) -> PushT:
    """A shallow copy of it, per test.

    Several tests below zero or raise a weight to isolate one term, and a
    shared instance would carry that into the next. Copying is enough
    because every one of them assigns a plain float attribute; the one
    that reaches into `object_model` restores it itself.
    """
    return copy.copy(_built)


def _log(task: PushT, poses: np.ndarray, robot: np.ndarray) -> Dict[str, Any]:
    """A log shaped like a finished run's: states run one longer than inputs."""
    return {
        "object_pose": poses,
        "robot_pos": robot,
        "robot_control": np.zeros((len(poses) - 1, 2)),
    }


def _sample_poses(n: int = 6) -> np.ndarray:
    rng = np.random.default_rng(0)
    return np.column_stack(
        [
            rng.uniform(-0.1, 0.35, n),
            rng.uniform(-0.1, 0.3, n),
            rng.uniform(-np.pi, np.pi, n),
        ]
    )


def test_terms_match_the_task_object_cost(task: PushT) -> None:
    """goal_pos + goal_theta + obstacle equals the object block's own cost.

    `object_running_cost` at zero wrench drops only its effort term, so the
    remainder is exactly the two goal halves plus the clearance hinge.
    """
    poses = _sample_poses()
    robot = poses[:, :2] - 0.05
    series = cost_series(task, _log(task, poses, robot))

    for i, pose in enumerate(poses[1:]):
        mine = (
            series["goal_pos"][i]
            + series["goal_theta"][i]
            + series["obstacle"][i]
        )
        theirs = float(
            task.object_running_cost(jnp.asarray(pose), jnp.zeros(3))
        )
        assert mine == pytest.approx(theirs, rel=1e-5, abs=1e-6)


def test_effort_uses_the_task_weight(task: PushT) -> None:
    """Effort is the task's own `w_robot_effort` times the squared command."""
    poses = _sample_poses(3)
    log = _log(task, poses, poses[:, :2])
    log["robot_control"] = np.array([[1.0, 2.0], [0.5, 0.0]])
    series = cost_series(task, log)
    assert series["effort"][0] == pytest.approx(task.w_robot_effort * 5.0)
    assert series["effort"][1] == pytest.approx(task.w_robot_effort * 0.25)


def _object(**kwargs: Any) -> PlanarPushingObject:
    """A bare object model at the shipped timestep."""
    return PlanarPushingObject(
        dt=0.05, goal=jnp.zeros(3), footprint=t_shape_footprint(), **kwargs
    )


def test_step_subtracts_friction_rather_than_gating_on_it() -> None:
    """Motion goes continuously to zero at the friction cone, not off a cliff.

    The earlier form zeroed sub-threshold wrenches and passed the *full*
    wrench above threshold, so one-step displacement jumped straight from 0
    to `dt * 1.0` = 0.05 m -- the goal tolerance -- and no smaller
    correction was representable at all. Subtracting the friction is the
    standard Coulomb form and is what makes a smaller sampled force
    actually produce a smaller step.
    """
    obj = _object()
    limit = np.asarray(obj.wrench_limit)

    def travel(multiple: float) -> float:
        wrench = jnp.array([multiple * limit[0], 0.0, 0.0])
        return float(np.linalg.norm(np.asarray(obj.step(jnp.zeros(3), wrench))))

    assert travel(0.99) == 0.0  # inside the cone: sticking, exactly
    # On the boundary, up to float32: `k * limit / limit` is not exactly 1,
    # so `slip = 1 - 1/s` lands at ~6e-9 rather than a hard zero. Bounded
    # rather than asserted equal, since the alternative is an epsilon in
    # the dynamics to make a test pass.
    assert travel(1.0) < 1e-6
    # Just outside, the old form gave 0.05 -- the whole tolerance ball.
    assert travel(1.05) == pytest.approx(0.0025, rel=1e-3)
    assert travel(2.0) == pytest.approx(0.05, rel=1e-3)
    # Monotone and continuous across the boundary.
    assert travel(1.001) < travel(1.05) < travel(1.2) < travel(2.0)


def test_step_is_nan_safe_at_zero_wrench() -> None:
    """`w = 0` is ordinary input and a singularity of both norm and 1/s.

    Guarding only the output would leave a nan in the gradient, which
    nothing on the sampling path would notice today and everything using
    `grad` later would.
    """
    obj = _object()
    zero = jnp.zeros(3)
    assert np.asarray(obj.step(zero, zero)) == pytest.approx(np.zeros(3))
    grad = jax.grad(lambda w: jnp.sum(obj.step(zero, w)))(zero)
    assert bool(np.all(np.isfinite(np.asarray(grad))))


def test_rate_cost_weights_each_wrench_channel_separately() -> None:
    """`w_rate` is per-channel, and a scalar broadcasts to all three.

    The torque limit is ~17x smaller than the force limit, so a run that
    silently collapsed the triple to its first entry would still look
    sane -- it would just stop damping rotation, which is the channel the
    weighting exists to treat differently.
    """
    kwargs: Dict[str, Any] = {
        "dt": 0.1,
        "goal": jnp.zeros(3),
        "footprint": t_shape_footprint(),
    }
    # One unit-of-limit step in each channel in turn, so each squared
    # difference is exactly 1 and the penalty *is* the weight.
    limit = PlanarPushingObject(**kwargs).wrench_limit
    wrenches = jnp.diag(limit)

    weights = [2.0, 5.0, 11.0]
    obj = PlanarPushingObject(**kwargs, w_rate=weights)
    # Anchored at zero: diffs are 0->fx, fx->fy, fy->tau, so channel i is
    # crossed twice except tau, which is only entered.
    expected = 2 * weights[0] + 2 * weights[1] + weights[2]
    assert float(obj.rate_cost(wrenches, jnp.zeros(3))) == pytest.approx(
        expected
    )

    scalar = PlanarPushingObject(**kwargs, w_rate=3.0)
    assert np.asarray(scalar.w_rate).tolist() == [3.0, 3.0, 3.0]
    assert float(
        scalar.rate_cost(wrenches, jnp.zeros(3))
    ) == pytest.approx(3.0 * 5)


def test_series_are_one_per_control_step(task: PushT) -> None:
    """One cost per control, not per state -- states run one longer."""
    poses = _sample_poses(7)
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert all(len(v) == 6 for v in series.values())


def test_totals_sum_the_series(task: PushT) -> None:
    """`cost_totals` accumulates each term, and `total` sums those."""
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    totals = cost_totals(series)
    for name, values in series.items():
        assert totals[name] == pytest.approx(float(np.sum(values)))
    assert totals["total"] == pytest.approx(
        sum(float(np.sum(v)) for v in series.values())
    )


def test_terms_are_ordered_and_known(task: PushT) -> None:
    """Keys come back in `TERM_ORDER`, so the legend order is stable."""
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert list(series) == [k for k in TERM_ORDER if k in series]


def test_3d_gets_tilt_and_tip_z_and_no_robot_clearance(
    task: PushT,
) -> None:
    """A log carrying `tip_tilt` is scored as 3D, whatever produced it.

    `cost_series` branches on the log, not on the task class, so a task
    whose own run would not carry `tip_tilt` is still scored as 3D when
    the log does. The weights still come from the task.
    """
    # Pinned, not inherited: the fixture is built with the shipped
    # `costs:` block, whose values for these differ from the arithmetic
    # this test asserts.
    task.shaping_fade_dist = 0.0
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    # `tip_quadratic_target_z` is a real config key that defaults to
    # `tip_target_z` only when absent; pinned so moving one moves both.
    task.tip_quadratic_target_z = task.tip_target_z
    poses = _sample_poses()
    log = _log(task, poses, poses[:, :2])
    log["tip_tilt"] = [0.1] * 5
    log["tip_z"] = [0.03] * 5
    series = cost_series(task, log)
    assert "tilt" in series and "tip_z" in series
    # 1 - cos(psi), not psi: the log stores the angle, the cost is the
    # cosine form (see `PushT._tilt`).
    assert series["tilt"][0] == pytest.approx(30.0 * (1.0 - np.cos(0.1)))
    # 100x: `_tip_height_cost` works in centimetres. This assertion read
    # `8.0 * (0.03 - 0.025) ** 2` and had been failing before the 2D world
    # was removed -- the formula it pins has always been in cm^2.
    assert series["tip_z"][0] == pytest.approx(
        8.0 * (100.0 * (0.03 - 0.025)) ** 2
    )


def test_tip_z_is_piecewise_and_the_below_branch_never_fades(
    task: PushT,
) -> None:
    """`cost_series`'s `tip_z` reproduces `PushT._tip_height_cost`.

    The faded quadratic above `tip_target_z`, the unfaded exponential below
    it -- pins the diagnostic against the real formula so the two
    cannot drift.
    """
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    tip_z_val = 0.01  # below tip_target_z=0.025
    gap_cm = 100.0 * (task.tip_target_z - tip_z_val)
    exp_below = task.w_z_tip_exp * np.exp(gap_cm**2)

    goal_np = np.asarray(task.goal)  # float32; offset from this exactly,
    # not from retyped decimal literals, so offset=0.0 gives pos_err
    # *exactly* 0.0 rather than a float32-vs-float64 rounding residual.

    def _series_at(offset: float):
        pose = goal_np.copy()
        pose[0] += offset  # x alone, so pos_err is exactly |offset|
        poses = np.tile(pose, (5, 1))
        log = _log(task, poses, poses[:, :2])
        log["tip_tilt"] = [0.1] * 5
        log["tip_z"] = [tip_z_val] * 5
        return cost_series(task, log)

    # Below the threshold the exponential is a safety guarantee, so it is
    # full strength at every distance from the goal -- including at it.
    for offset in (1.0, 0.2, 0.1, 0.0):
        assert _series_at(offset)["tip_z"][0] == pytest.approx(exp_below)


def test_tip_z_above_threshold_fades_in_cost_series(task: PushT) -> None:
    """`cost_series`'s `tip_z` fades the true above-threshold branch.

    Linearly, over shaping_fade_dist, the same way
    `PushT._tip_height_cost` does -- pins the diagnostic against the
    real formula.
    """
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    task.shaping_fade_dist = 0.2
    # `tip_quadratic_target_z` is a real config key that defaults to
    # `tip_target_z` only when absent; pinned so moving one moves both.
    task.tip_quadratic_target_z = task.tip_target_z
    tip_z_val = 0.03  # above tip_target_z=0.025
    quad_ref = task.w_z_tip * (
        100.0 * (tip_z_val - task.tip_target_z)
    ) ** 2  # cm^2

    goal_np = np.asarray(task.goal)

    def _series_at(offset: float):
        pose = goal_np.copy()
        pose[0] += offset
        poses = np.tile(pose, (5, 1))
        log = _log(task, poses, poses[:, :2])
        log["tip_tilt"] = [0.1] * 5
        log["tip_z"] = [tip_z_val] * 5
        return cost_series(task, log)

    # At/beyond shaping_fade_dist: full, unfaded quadratic.
    assert _series_at(0.2)["tip_z"][0] == pytest.approx(quad_ref)
    assert _series_at(1.0)["tip_z"][0] == pytest.approx(quad_ref)
    # At the goal: faded to exactly 0.
    assert _series_at(0.0)["tip_z"][0] == pytest.approx(0.0, abs=1e-9)
    # Halfway: exact linear fade.
    assert _series_at(0.1)["tip_z"][0] == pytest.approx(0.5 * quad_ref)


def test_approach_fades_in_cost_series(task: PushT) -> None:
    """`cost_series`'s `approach` fades linearly with `shaping_fade_dist`.

    The same way `PushT._ell_r` does -- previously exempt entirely.
    """
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    task.shaping_fade_dist = 0.2

    goal_np = np.asarray(task.goal)

    def _series_at(offset: float):
        pose = goal_np.copy()
        pose[0] += offset
        poses = np.tile(pose, (5, 1))
        # Fixed offset from pose (not equal to it), so d_ee -- and the
        # *unfaded* approach cost -- stays the same nonzero value
        # regardless of pos_err; only the fade should change the total.
        robot = poses[:, :2] + np.array([0.1, 0.1])
        log = _log(task, poses, robot)
        log["tip_tilt"] = [0.1] * 5
        log["tip_z"] = [0.03] * 5
        return cost_series(task, log)

    raw_approach = _series_at(0.2)["approach"][0]
    assert raw_approach > 0.0
    assert _series_at(0.0)["approach"][0] == pytest.approx(0.0, abs=1e-9)
    assert _series_at(0.1)["approach"][0] == pytest.approx(
        0.5 * raw_approach
    )


def test_effort_fades_in_cost_series(task: PushT) -> None:
    """`cost_series`'s `effort` fades linearly with `shaping_fade_dist`.

    The same way `running_cost` does.
    """
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    task.shaping_fade_dist = 0.2

    goal_np = np.asarray(task.goal)

    def _series_at(offset: float):
        pose = goal_np.copy()
        pose[0] += offset
        poses = np.tile(pose, (5, 1))
        controls = np.ones((4, 2))
        log = _log(task, poses, poses[:, :2])
        log["robot_control"] = controls
        log["tip_tilt"] = [0.1] * 5
        log["tip_z"] = [0.03] * 5
        return cost_series(task, log)

    raw_effort = task.w_robot_effort * np.sum(np.ones(2) ** 2)
    assert raw_effort > 0.0
    assert _series_at(0.2)["effort"][0] == pytest.approx(raw_effort)
    assert _series_at(0.0)["effort"][0] == pytest.approx(0.0, abs=1e-9)
    assert _series_at(0.1)["effort"][0] == pytest.approx(0.5 * raw_effort)


def test_goal_pos_and_theta_ramp_with_real_time(task: PushT) -> None:
    """`cost_series`'s `goal_pos`/`goal_theta` ramp with elapsed time.

    The same way `PushT.running_cost` does -- pins the diagnostic
    against the real formula. A log with no "time" key (every other test
    in this file) stays fully unramped, the "absent beats zero" fallback
    `_common_terms` documents.
    """
    task.q_ramp_per_step = 0.002
    task.q_ramp_max = 4.0
    poses = _sample_poses()
    robot = poses[:, :2] - 0.05
    log = _log(task, poses, robot)
    n = len(poses) - 1

    baseline = cost_series(task, log)  # no "time" key: unramped

    ramped_log = dict(log)
    ramped_log["time"] = np.arange(len(poses)) * task.dt  # 0, dt, 2dt, ...
    ramped = cost_series(task, ramped_log)

    # Linear (see `costs._q_ramp_mult`), not the compounding law this
    # replaced -- 1.002**n, which at n = 5 differs by only 2e-5, so the
    # rtol below is what actually separates the two.
    expected_mult = np.minimum(1.0 + 0.002 * np.arange(1, n + 1), 4.0)
    assert np.allclose(
        ramped["goal_pos"], baseline["goal_pos"] * expected_mult, rtol=1e-5
    )
    assert np.allclose(
        ramped["goal_theta"],
        baseline["goal_theta"] * expected_mult,
        rtol=1e-5,
    )


def test_3d_gets_contact_z_hover_slab_when_the_log_carries_it(
    task: PushT,
) -> None:
    """`contact_z`'s kinematic hover-slab matches `PushT._contact_z_cost`.

    Same borrowed-2D-task pattern as used above. Fires inside a
    2cm slab straddling the block's true top surface (1cm below it and
    1cm above, symmetric -- moved 1cm into the block, per Shahid,
    2026-08-19: the surface-to-+1cm-only version let a run's tip
    oscillate a couple mm below the surface -- still plainly
    top-riding -- and read as outside the slab, scoring exactly 0 for
    long stretches despite never actually leaving the top of the
    block), and only where the tip's (x, y) falls inside the block's
    real footprint -- reuses the fixture's own `t_shape_footprint()`,
    whose stem covers the origin, so an object pose and tip both at the
    origin is inside.
    """
    # Pinned, not inherited: the fixture is built with the shipped
    # `costs:` block, whose values for these differ from the arithmetic
    # this test asserts.
    task.w_tilt, task.w_z_tip, task.w_z_tip_exp = 0.0, 0.0, 0.0
    task.tip_target_z, task.block_half_height = 0.025, 0.025
    task.w_contact_z_exp = 1.0
    task.contact_z_slab = task.contact_z_slab_above = 0.01
    task.contact_z_margin, task.contact_z_cap = 0.0, 0.0
    task.contact_z_below_mult = 1.0
    poses = np.zeros((6, 3))  # object at the origin, no rotation
    log = _log(task, poses, np.zeros((6, 2)))  # tip at the world origin too
    log["tip_tilt"] = [0.0] * 5

    log["tip_z"] = [0.055] * 5  # top_z=0.05 -> dz_cm=+0.5, inside the slab
    series = cost_series(task, log)
    assert "contact_z" in series
    gap = 1.0 - 0.5
    assert series["contact_z"][0] == pytest.approx(np.exp((2.0 * gap) ** 2))

    log["tip_z"] = [0.045] * 5  # dz_cm=-0.5, symmetric: same cost as +0.5
    assert cost_series(task, log)["contact_z"][0] == pytest.approx(
        np.exp((2.0 * gap) ** 2)
    )

    log["tip_z"] = [0.07] * 5  # dz_cm=+2.0 -> outside the slab
    assert cost_series(task, log)["contact_z"][0] == pytest.approx(0.0)

    log["tip_z"] = [0.03] * 5  # dz_cm=-2.0 -> outside the slab
    assert cost_series(task, log)["contact_z"][0] == pytest.approx(0.0)

    log["tip_z"] = [0.055] * 5
    log["robot_pos"] = np.full((6, 2), 1.0)  # outside the footprint's xy
    assert cost_series(task, log)["contact_z"][0] == pytest.approx(0.0)


def test_summarize_returns_none_on_an_unusable_log(task: PushT) -> None:
    """The plotting path must degrade, not crash, on a log it cannot score."""
    assert summarize(task, {}) is None
    assert summarize(task, {"object_pose": np.zeros((0, 3))}) is None
    poses = _sample_poses(1)
    empty = _log(task, poses, poses[:, :2])
    assert summarize(task, empty) is None


def test_a_3d_task_without_tip_data_does_not_crash() -> None:
    """A 3D task has no `w_obstacle_robot`, so the 2D branch must not run.

    `PushT` keeps its clearance settings on `object_model`, not on itself.
    Reading `tip_tilt` off a log that has none would be a KeyError
    raised inside plotting, at the end of a finished run.
    """
    from oim.tasks.pusht import PushT  # noqa: PLC0415

    task = PushT(clutter=True, robot="xarm6", env="open_table")
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert "tilt" not in series
    assert "goal_pos" in series
