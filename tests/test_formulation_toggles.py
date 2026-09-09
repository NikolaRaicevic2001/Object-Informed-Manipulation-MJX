"""The sim/real formulation switches added when debug/real-exp merged.

Each key defaults to what every sim config ran before the merge and
selects the real rig's variant when set; these tests pin both sides of
every switch, so neither can silently drift back into the other.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from conftest import mjx_forward

from oim.objects import ObstacleField
from oim.objects.planar_pushing import PlanarPushingObject, t_shape_footprint
from oim.objects.sdf import Box
from oim.tasks.pusht import DEFAULT_COSTS, PushT

PLAN_DT = 0.05


def _object(**kwargs: object) -> PlanarPushingObject:
    return PlanarPushingObject(
        dt=PLAN_DT,
        goal=[0.5, 0.0, 0.0],
        footprint=t_shape_footprint(),
        obstacles=ObstacleField(
            [Box(center=[0.3, 0.0], half_extents=[0.05, 0.05], angle=0.0)]
        ),
        w_obstacle=10.0,
        obstacle_decay=0.02,
        **kwargs,
    )


def test_defaults_are_the_unified_forms() -> None:
    """The unified set: one formulation for sim and real."""
    assert DEFAULT_COSTS["tip_z_form"] == "piecewise"
    assert DEFAULT_COSTS["align_ref"] == "plan_end"
    assert DEFAULT_COSTS["w_effort_normalized"] == 1.0
    assert DEFAULT_COSTS["robot_block_support"] == 0.0
    obj = _object()
    assert obj.effort_normalized
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    assert task.align_ref == "plan_end"


def test_plant_subtracts_friction_rather_than_gating() -> None:
    """Eq. 5 with friction subtracted: zero inside the cone, continuous
    at the boundary, growing beyond it.

    The selectable quasi-static map (wrench ON the surface, speed fixed
    at `push_speed`) was removed with `plant_form`. The property that
    motivated subtracting rather than gating is what matters and is
    pinned here: motion goes CONTINUOUSLY to zero at the cone, so there
    is no hole in the reachable set the size of the goal tolerance.
    """
    pose = jnp.zeros(3)
    obj = _object()
    limit = np.asarray(obj.wrench_limit)
    w_on = jnp.asarray([limit[0], 0.0, 0.0])   # exactly on the surface
    w_in = 0.5 * w_on                          # inside: sticking
    w_just = 1.05 * w_on                       # just outside
    w_out = 3.0 * w_on                         # well outside

    assert np.allclose(np.asarray(obj.step(pose, w_in)), 0.0)
    assert np.allclose(np.asarray(obj.step(pose, w_on)), 0.0)
    just = float(obj.step(pose, w_just)[0])
    out = float(obj.step(pose, w_out)[0])
    # Continuous at the boundary, and monotone beyond it.
    assert 0.0 < just < out
    assert just < 0.05 * out       # a small excess gives a small step


def test_object_action_is_the_raw_box() -> None:
    """Nothing projects the sampled wrench any more.

    Paper eq. 18's projection (Pi_F) belonged to the quasi-static plant.
    With eq. 5 the raw action box IS the decision -- a wrench beyond the
    cone is meaningful input, not something to scale back.
    """
    task = PushT(clutter=True, planning_dt=PLAN_DT, wrench_fraction=1.5)
    action = jnp.asarray([1.0, 1.0, 1.0])   # box corner, outside the cone
    w = task.object_action_to_consensus(jnp.zeros(3), action)
    assert float(jnp.linalg.norm(w / task.object_model.wrench_limit)) > 1.0
    assert not hasattr(task.object_model, "project_wrench")


def test_obstacle_cost_is_margin_gated() -> None:
    """One obstacle equation: zero past the margin, a barrier inside it.

    The always-on `w * exp(-d/decay)` alternative was removed with the
    sim/real unification -- a penalty non-zero at every distance is not a
    stand-off. This pins the surviving shape, including the property the
    removal was for: EXACTLY zero clearance cost far from an obstacle.
    """
    obj = _object(obstacle_margin=0.03)
    far = jnp.asarray([-0.5, 0.0, 0.0])          # boundary sdf +0.66 m
    assert float(obj.obstacle_cost(far)) == 0.0
    inside = jnp.asarray([0.15, 0.0, 0.0])       # boundary sdf +0.01 m
    assert float(obj.obstacle_cost(inside)) > 0.0
    outside = jnp.asarray([0.12, 0.0, 0.0])      # boundary sdf +0.04 m
    assert float(obj.obstacle_cost(outside)) == 0.0
    # A zero margin leaves no avoidance signal at ANY positive clearance:
    # `gap = clip(-d, 0, inf)` is zero wherever the footprint is clear, so
    # the term only charges once it already overlaps. That is why
    # `point.yaml` needed an explicit margin when it was migrated -- it
    # carried none, having had no use for one under the removed form.
    assert float(_object(obstacle_margin=0.0).obstacle_cost(inside)) == 0.0
    penetrating = jnp.asarray([0.1855, 0.0, 0.0])  # boundary sdf -0.0255 m
    assert float(_object(obstacle_margin=0.0).obstacle_cost(penetrating)) > 0.0


def test_effort_normalized_toggle() -> None:
    raw = _object(effort_normalized=False)
    normalized = _object(effort_normalized=True)
    pose = jnp.asarray([0.5, 0.0, 0.0])  # at the goal: goal terms are 0
    w = jnp.asarray([4.0, 0.0, 0.0])
    limit = float(raw.wrench_limit[0])
    # Same goal/obstacle terms on both objects, so the difference is the
    # effort term alone: w_effort * ((w/L)^2 - w^2).
    diff = float(normalized.running_cost(pose, w)) - float(
        raw.running_cost(pose, w)
    )
    assert diff == pytest.approx(
        raw.w_effort * ((4.0 / limit) ** 2 - 16.0), rel=1e-3
    )


def test_tip_z_form_on_the_task() -> None:
    piecewise = PushT(clutter=True, planning_dt=PLAN_DT, robot="xarm6")
    symmetric = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        costs={"tip_z_form": "symmetric_exp"},
    )
    assert piecewise.tip_z_form == "piecewise"
    assert symmetric.tip_z_form == "symmetric_exp"
    # Tip height is priced by `w_z_tip`/`w_z_tip_exp` alone -- `approach`
    # is purely xy, and no longer folds the height error into its own
    # distance (`approach_z`), which priced the same error twice.
    assert not hasattr(piecewise, "approach_z")
    state = mjx_forward(piecewise.model, piecewise.make_data())
    pos_err = jnp.asarray(0.3)
    # Symmetric form: identical cost the same height above and below.
    z = float(state.site_xpos[symmetric.trace_site_ids[0], 2])
    mid = symmetric.tip_target_z
    up = state.replace(
        site_xpos=state.site_xpos.at[symmetric.trace_site_ids[0], 2].set(
            mid + 0.01
        )
    )
    down = state.replace(
        site_xpos=state.site_xpos.at[symmetric.trace_site_ids[0], 2].set(
            mid - 0.01
        )
    )
    del z
    assert float(symmetric._tip_height_cost(up, pos_err)) == pytest.approx(
        float(symmetric._tip_height_cost(down, pos_err))
    )
    # Piecewise form is asymmetric (quadratic above, exponential below).
    assert float(piecewise._tip_height_cost(up, pos_err)) != pytest.approx(
        float(piecewise._tip_height_cost(down, pos_err))
    )
    # Both `_ell_r` forms evaluate.
    pose = piecewise._block_pose(state)
    pusher = piecewise._pusher_pos(state)
    assert np.isfinite(float(piecewise._ell_r(state, pose, pusher, pose)))
    assert np.isfinite(float(symmetric._ell_r(state, pose, pusher, pose)))
    with pytest.raises(ValueError):
        PushT(clutter=True, planning_dt=PLAN_DT, costs={"tip_z_form": "x"})


def test_robot_block_support_toggle() -> None:
    without = PushT(clutter=True, planning_dt=PLAN_DT, robot="xarm6")
    with_support = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        costs={"robot_block_support": 1.0},
    )
    assert not without.robot_block_support and with_support.robot_block_support
    state = mjx_forward(without.model, without.make_data())
    u = jnp.zeros(without.model.nu)
    base = float(without.robot_running_cost(state, u))
    plus = float(with_support.robot_running_cost(state, u))
    pose = without._block_pose(state)
    support = float(with_support.object_model.support_cost(pose))
    assert plus - base == pytest.approx(support, abs=1e-4)


def test_a_r_is_measured_from_the_rollout_never_estimated() -> None:
    """A^r is the rollout's own contact forces, on both rigs.

    The three twist/contact estimators that used to be selectable via
    `consensus_source` are gone: they inverted the plant from an OBSERVED
    velocity, which is the thing we do not want to do -- on hardware or
    in sim. The planner reads the simulation it planned in, and the arm's
    real contact forces are never read.
    """
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    state = mjx_forward(task.model, task.make_data())
    # A block sliding at 0.02 m/s with NO contact. An estimator would
    # invert that velocity into a full cone-sized wrench; measuring the
    # rollout correctly reports zero, because nothing is touching it.
    qvel = state.qvel.at[task.block_dofs[0]].set(0.02)
    moving = state.replace(qvel=qvel)
    assert np.allclose(np.asarray(task.realized_consensus(moving)), 0.0)
    # The selector and every estimator are gone from the API.
    assert not hasattr(task, "consensus_source")
    for gone in ("_consensus_from_twist", "_consensus_from_twist_exact",
                 "_consensus_from_contact"):
        assert not hasattr(task, gone), gone
    with pytest.raises(TypeError):
        PushT(clutter=True, planning_dt=PLAN_DT, consensus_source="twist")
