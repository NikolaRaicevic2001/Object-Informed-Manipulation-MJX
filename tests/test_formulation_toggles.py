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
    assert DEFAULT_COSTS["obstacle_form"] == "margin"
    assert DEFAULT_COSTS["tip_z_form"] == "piecewise"
    assert DEFAULT_COSTS["align_ref"] == "plan_end"
    assert DEFAULT_COSTS["approach_mode"] == 1.0
    assert DEFAULT_COSTS["w_effort_normalized"] == 1.0
    assert DEFAULT_COSTS["robot_block_support"] == 0.0
    obj = _object()
    assert obj.plant_form == "excess"
    assert obj.obstacle_form == "margin"
    assert obj.effort_normalized
    task = PushT(clutter=True, planning_dt=PLAN_DT)
    assert task.plant_form == "excess"
    assert task.approach_mode == 1 and task.align_ref == "plan_end"


def test_plant_form_excess_vs_quasi_static() -> None:
    """Excess: speed grows with |w| past the cone. Quasi-static: fixed."""
    pose = jnp.zeros(3)
    excess = _object(plant_form="excess")
    quasi = _object(plant_form="quasi_static", push_speed=0.05)
    limit = np.asarray(excess.wrench_limit)
    w_on = jnp.asarray([limit[0], 0.0, 0.0])  # exactly on the surface
    w_out = 3.0 * w_on  # well outside
    w_in = 0.5 * w_on  # inside: sticking under both forms

    assert np.allclose(np.asarray(excess.step(pose, w_in)), 0.0)
    assert np.allclose(np.asarray(quasi.step(pose, w_in)), 0.0)
    # Excess form: on the surface exactly zero motion, outside it grows.
    assert np.allclose(np.asarray(excess.step(pose, w_on)), 0.0)
    d_out = float(excess.step(pose, w_out)[0])
    assert d_out > 0.0
    # Quasi-static: on the surface already moving at push_speed, and the
    # same displacement however far outside the wrench is.
    step_on = float(quasi.step(pose, w_on)[0])
    assert step_on == pytest.approx(PLAN_DT * 0.05)
    assert float(quasi.step(pose, w_out)[0]) == pytest.approx(step_on)


def test_project_wrench_gates_only_the_quasi_static_plant() -> None:
    # wrench_fraction 1.5: the real rig's box, whose corner (1.5*sqrt(3)
    # in normalized units) lies well outside the cone.
    task_excess = PushT(
        clutter=True, planning_dt=PLAN_DT, wrench_fraction=1.5,
        plant_form="excess",
    )
    task_quasi = PushT(
        clutter=True, planning_dt=PLAN_DT, plant_form="quasi_static",
        wrench_fraction=1.5,
    )
    action = jnp.asarray([1.0, 1.0, 1.0])  # the box corner, outside the cone
    obj_state = jnp.zeros(3)
    raw = task_excess.object_action_to_consensus(obj_state, action)
    projected = task_quasi.object_action_to_consensus(obj_state, action)
    limit = task_quasi.object_model.wrench_limit
    assert float(jnp.linalg.norm(raw / limit)) > 1.0
    assert float(jnp.linalg.norm(projected / limit)) == pytest.approx(1.0)


def test_obstacle_form_exp_vs_margin() -> None:
    far = jnp.asarray([-0.5, 0.0, 0.0])  # 0.75 m from the box
    exp_form = _object(obstacle_form="exp")
    margin_form = _object(obstacle_form="margin", obstacle_margin=0.03)
    # Always-on exponential: a gradient at every distance.
    assert float(exp_form.obstacle_cost(far)) > 0.0
    # Margin form: exactly zero outside the margin, positive inside it.
    assert float(margin_form.obstacle_cost(far)) == 0.0
    near = jnp.asarray([0.3 - 0.05 - 0.02 - 0.0445, 0.0, 0.0])
    assert float(margin_form.obstacle_cost(near)) > 0.0
    with pytest.raises(ValueError):
        _object(obstacle_form="hinge")


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


def test_tip_z_form_and_approach_mode_on_the_task() -> None:
    piecewise = PushT(
        clutter=True, planning_dt=PLAN_DT, robot="xarm6",
        costs={"approach_mode": 0.0, "approach_z": 0.0},
    )
    symmetric = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        costs={"tip_z_form": "symmetric_exp", "approach_mode": 1.0,
               "approach_z": 1.0},
    )
    assert piecewise.tip_z_form == "piecewise" and piecewise.approach_mode == 0
    assert symmetric.tip_z_form == "symmetric_exp"
    assert symmetric.approach_mode == 1 and symmetric.approach_z
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
    with pytest.raises(ValueError):
        PushT(clutter=True, planning_dt=PLAN_DT, costs={"approach_mode": 2})


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


def test_consensus_source_dispatch() -> None:
    measured = PushT(clutter=True, planning_dt=PLAN_DT)
    exact = PushT(
        clutter=True, planning_dt=PLAN_DT, consensus_source="twist_exact"
    )
    assert measured.consensus_source == "measured"
    state = mjx_forward(exact.model, exact.make_data())
    # A block sliding at 0.02 m/s in +x: twist_exact pins |A^r| to L.
    qvel = state.qvel.at[exact.block_dofs[0]].set(0.02)
    moving = state.replace(qvel=qvel)
    a_r = np.asarray(exact.realized_consensus(moving))
    limit = np.asarray(exact.object_model.wrench_limit)
    assert a_r[0] == pytest.approx(limit[0], rel=1e-5)
    assert np.allclose(a_r[1:], 0.0)
    # No contact in this state: the measured wrench is zero.
    assert np.allclose(np.asarray(measured.realized_consensus(moving)), 0.0)
    with pytest.raises(ValueError):
        PushT(clutter=True, planning_dt=PLAN_DT, consensus_source="guess")
    with pytest.raises(ValueError):
        PushT(
            clutter=True, planning_dt=PLAN_DT, robot="xarm6",
            consensus_source="contact",
        )
