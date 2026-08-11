"""The cost breakdown must agree with the cost the planner actually used.

`oim.utils.costs` restates the task's cost terms in numpy so a finished run
can be decomposed without re-entering `jit`. That restatement is the thing
most likely to drift: someone retunes a weight on the task and the figure
keeps reporting the old one, silently. These tests pin the two together.
"""

from typing import Any, Dict

import jax.numpy as jnp
import numpy as np
import pytest

from oim.objects import Circle
from oim.objects.planar_pushing import t_shape_footprint
from oim.sim2d.task import PushT2D
from oim.utils.costs import TERM_ORDER, cost_series, cost_totals, summarize


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

    task.r_r = 0.0
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
    """Effort is the task's own `r_r` times the squared command."""
    poses = _sample_poses(3)
    log = _log(task, poses, poses[:, :2])
    log["robot_control"] = np.array([[1.0, 2.0], [0.5, 0.0]])
    series = cost_series(task, log)
    assert series["effort"][0] == pytest.approx(task.r_r * 5.0)
    assert series["effort"][1] == pytest.approx(task.r_r * 0.25)


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
    task.w_tilt, task.w_tip_z, task.tip_target_z = 30.0, 8.0, 0.025
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
