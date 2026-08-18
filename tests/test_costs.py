"""The cost breakdown must agree with the cost the planner actually used.

`oim.utils.costs` restates the task's cost terms in numpy so a finished run
can be decomposed without re-entering `jit`. That restatement is the thing
most likely to drift: someone retunes a weight on the task and the figure
keeps reporting the old one, silently. These tests pin the two together.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from oim.objects import Circle
from oim.objects.planar_pushing import PlanarPushingObject, t_shape_footprint
from oim.utils.costs import TERM_ORDER, cost_series, cost_totals, summarize
from oim.worlds.sim2d.task import PushT2D


@pytest.fixture
def task() -> PushT2D:
    """A 2D task: pure JAX, so its real cost is cheap to evaluate here.

    Function-scoped because several tests below zero a weight to isolate a
    term, and a shared instance would carry that into the next test.
    """
    return PushT2D(
        footprint=t_shape_footprint(),
        goal=jnp.array([0.3, 0.2, 0.5]),
        obstacles=[Circle(center=jnp.array([0.15, 0.05]), radius=0.04)],
    )


def _log(task: PushT2D, poses: np.ndarray, robot: np.ndarray) -> Dict[str, Any]:
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


def test_terms_match_the_task_object_cost(task: PushT2D) -> None:
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


def test_approach_and_align_match_the_task_robot_cost(task: PushT2D) -> None:
    """The two shaping terms equal what `robot_running_cost` computes.

    Scored with `obj_ref = goal`, which is the reference the breakdown uses
    so ADMM and a flat baseline stay comparable. Every *other* weight in
    `robot_running_cost` is zeroed rather than subtracted off afterwards:
    the goal and clearance terms are orders of magnitude larger than these
    two, so a subtraction would cancel away most of the significant digits
    and the comparison would be testing float32 noise.
    """
    poses = _sample_poses()
    robot = poses[:, :2] - 0.05
    series = cost_series(task, _log(task, poses, robot))
    goal = jnp.asarray(task.goal)

    task.w_robot_effort = 0.0
    task.q_pos = task.q_theta = 0.0
    task.w_obstacle_robot = 0.0

    for i, (pose, rob) in enumerate(zip(poses[1:], robot[1:], strict=True)):
        state = _State(jnp.asarray(pose), jnp.asarray(rob))
        only_shaping = float(
            task.robot_running_cost(state, jnp.zeros(2), goal)
        )
        mine = series["approach"][i] + series["align"][i]
        assert mine == pytest.approx(only_shaping, rel=1e-5, abs=1e-6)


class _State:
    """The two fields `PushT2D.robot_running_cost` reads off a `Sim2DState`."""

    def __init__(self, object_pose: jnp.ndarray, robot_pos: jnp.ndarray):
        self.object_pose = object_pose
        self.robot_pos = robot_pos


def test_effort_uses_the_task_weight(task: PushT2D) -> None:
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


def test_series_are_one_per_control_step(task: PushT2D) -> None:
    """One cost per control, not per state -- states run one longer."""
    poses = _sample_poses(7)
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert all(len(v) == 6 for v in series.values())


def test_totals_sum_the_series(task: PushT2D) -> None:
    """`cost_totals` accumulates each term, and `total` sums those."""
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    totals = cost_totals(series)
    for name, values in series.items():
        assert totals[name] == pytest.approx(float(np.sum(values)))
    assert totals["total"] == pytest.approx(
        sum(float(np.sum(v)) for v in series.values())
    )


def test_terms_are_ordered_and_known(task: PushT2D) -> None:
    """Keys come back in `TERM_ORDER`, so the legend order is stable."""
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert list(series) == [k for k in TERM_ORDER if k in series]


def test_2d_gets_a_robot_clearance_term_and_no_tilt(task: PushT2D) -> None:
    """The embodiment decides which terms exist; absent beats zero."""
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert "robot_obstacle" in series
    assert "tilt" not in series and "tip_z" not in series


def test_3d_gets_tilt_and_tip_z_and_no_robot_clearance(
    task: PushT2D,
) -> None:
    """A log carrying `tip_tilt` is scored as 3D, whatever produced it.

    `cost_series` branches on the log, not on the task class, so the same
    routine serves both worlds. The weights still come from the task, so
    this borrows the 2D task's and only checks the branch.
    """
    task.w_tilt, task.tip_target_z = 30.0, 0.025
    task.w_z_tip, task.w_z_tip_exp = 8.0, 1.0
    poses = _sample_poses()
    log = _log(task, poses, poses[:, :2])
    log["tip_tilt"] = [0.1] * 5
    log["tip_z"] = [0.03] * 5
    series = cost_series(task, log)
    assert "tilt" in series and "tip_z" in series
    assert "robot_obstacle" not in series
    # 1 - cos(psi), not psi: the log stores the angle, the cost is the
    # cosine form (see `PushT._tilt`).
    assert series["tilt"][0] == pytest.approx(30.0 * (1.0 - np.cos(0.1)))
    assert series["tip_z"][0] == pytest.approx(8.0 * (0.03 - 0.025) ** 2)


def test_summarize_returns_none_on_an_unusable_log(task: PushT2D) -> None:
    """The plotting path must degrade, not crash, on a log it cannot score."""
    assert summarize(task, {}) is None
    assert summarize(task, {"object_pose": np.zeros((0, 3))}) is None
    poses = _sample_poses(1)
    empty = _log(task, poses, poses[:, :2])
    assert summarize(task, empty) is None


def test_a_3d_task_without_tip_data_does_not_crash() -> None:
    """A 3D task has no `obstacle_margin`, so the 2D branch must not run.

    `PushT` keeps its clearance settings on `object_model`, not on itself.
    Falling through to the robot-clearance branch would be an
    AttributeError raised inside plotting, at the end of a finished run.
    """
    from oim.tasks.pusht import PushT  # noqa: PLC0415

    task = PushT(clutter=True, robot="xarm6", env="open_table")
    poses = _sample_poses()
    series = cost_series(task, _log(task, poses, poses[:, :2]))
    assert "robot_obstacle" not in series
    assert "tilt" not in series
    assert "goal_pos" in series
