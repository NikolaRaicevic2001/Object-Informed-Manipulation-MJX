import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from oim.tasks.pusht import PushT


@pytest.mark.parametrize("impl", ["jax", "warp"])
@pytest.mark.parametrize("clutter", [False, True])
def test_task(impl: str, clutter: bool) -> None:
    """Set up the push T task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
        clutter: Whether to set up the full ConsensusTask/cost machinery
            (`self.goal`, `w_ee`, `w_align`, ...) -- despite the name, this
            does not pick a different scene (no `env=`/`scene=` is passed
            either way); every real run in this codebase passes
            `clutter=True` regardless of which scene it's for (see
            `oim/sim3d/build.py`). `False` is a bare construction that can
            still build data, but not evaluate costs -- `running_cost`
            needs those config-driven weights (`_ell_r`, paper eq. 21),
            so it is only checked when `clutter=True`.
    """
    task = PushT(impl=impl, clutter=clutter)

    state = task.make_data()
    assert isinstance(state, mjx.Data)
    state = state.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]]))
    state = jax.jit(mjx.forward)(task.model, state)

    pos = task._get_position_err(state)
    assert pos.shape == (3,)

    ori = task._get_orientation_err(state)
    assert ori.shape == (3,)

    if clutter:
        ell = task.running_cost(state, jnp.zeros(2))
        assert ell.shape == ()

        phi = task.terminal_cost(state)
        assert phi.shape == ()


def test_clutter_consensus_task_methods() -> None:
    """Check shapes of the ConsensusTask (ADMM) methods, clutter=True only."""
    task = PushT(clutter=True, planning_dt=0.05)
    assert task.consensus_dim == 3

    state = task.make_data()
    state = jax.jit(mjx.forward)(task.model, state)

    obj_state = task.object_state_from_robot(state)
    assert obj_state.shape == (3,)

    w = jnp.array([1.0, 2.0, 0.1])
    next_state = task.object_dynamics(obj_state, w)
    assert next_state.shape == (3,)

    running = task.object_running_cost(obj_state, w)
    assert running.shape == ()

    terminal = task.object_terminal_cost(obj_state)
    assert terminal.shape == ()

    assert task.object_action_scale().shape == (3,)
    assert task.consensus_scale().shape == (3,)

    ell = task.robot_running_cost(state, jnp.zeros(2), jnp.zeros(3))
    assert ell.shape == ()

    phi = task.robot_terminal_cost(state)
    assert phi.shape == ()

    consensus_val = task.realized_consensus(state)
    assert consensus_val.shape == (3,)


def test_clutter_analytic_model_matches_mjcf() -> None:
    """The analytic object model must describe the same physics as the MJCF.

    The block joints' `frictionloss` in the MJCF is what makes the simulated
    block resist motion; the analytic limit-surface model's friction-cone
    limit `mu*m*g` plays the same role. If they drift apart, the two ADMM
    blocks are silently modelling different objects.
    """
    task = PushT(clutter=True, planning_dt=0.05)
    mj = task.mj_model
    limit = task.object_model.wrench_limit

    fx = mj.dof_frictionloss[mj.joint("T_x").dofadr[0]]
    fy = mj.dof_frictionloss[mj.joint("T_y").dofadr[0]]
    ftheta = mj.dof_frictionloss[mj.joint("T_z").dofadr[0]]

    assert jnp.allclose(limit[0], fx, rtol=1e-3)
    assert jnp.allclose(limit[1], fy, rtol=1e-3)
    assert jnp.allclose(limit[2], ftheta, rtol=1e-3)


def test_clutter_planning_model_is_stable() -> None:
    """The planning model must not diverge at the planning timestep.

    The pusher's velocity actuator is only stable under explicit Euler for
    dt < 2*m/kv; at the 0.05 s planning timestep that bound is violated, so
    the model needs an implicit integrator. Without this the planner's
    rollouts blow up and the robot appears to "drift away" from the object
    even though the fine-timestep execution model looks fine.
    """
    task = PushT(clutter=True, planning_dt=0.05)
    state = task.make_data().replace(
        qpos=jnp.array([0.0, 0.0, 0.0, -0.4, -0.4]), qvel=jnp.zeros(5)
    )
    step = jax.jit(mjx.step)
    for _ in range(20):  # 1 second of constant commanded velocity
        state = step(task.model, state.replace(ctrl=jnp.array([0.0, 0.6])))

    # Commanded 0.6 m/s for 1 s: the pusher should travel ~0.6 m, not diverge.
    assert jnp.all(jnp.isfinite(state.qpos))
    assert abs(float(state.qvel[4])) < 2.0, (
        f"pusher velocity {float(state.qvel[4])} diverged from the "
        "commanded 0.6 m/s -- planning model is numerically unstable"
    )
    assert abs(float(state.qpos[4]) - (-0.4)) < 1.5


def test_xarm6_requires_clutter() -> None:
    """robot='xarm6' has no non-cluttered scene -- must fail loudly."""
    with pytest.raises(ValueError):
        PushT(robot="xarm6")


def test_xarm6_task() -> None:
    """Set up the push-T task with the xArm6 embodiment (jax impl only --
    unlike `test_task`, not parametrized over "warp" since this is meant to
    be runnable without a GPU for a basic sanity check).
    """
    task = PushT(clutter=True, planning_dt=0.05, robot="xarm6")
    assert task.model.nu == 5

    state = task.make_data()
    assert isinstance(state, mjx.Data)
    state = jax.jit(mjx.forward)(task.model, state)

    pos = task._get_position_err(state)
    assert pos.shape == (3,)

    ell = task.running_cost(state, jnp.zeros(5))
    assert ell.shape == ()
    assert jnp.isfinite(ell)

    phi = task.terminal_cost(state)
    assert phi.shape == ()
    assert jnp.isfinite(phi)


def test_xarm6_block_qpos_addresses() -> None:
    """Regression test for the qpos-ordering trap: unlike the point-mass
    scene (block declared before the pusher, so its pose is qpos[:3]), the
    composed xarm6 scene compiles the arm's 5 joints first, so the block's
    pose must NOT be read from qpos[:3].
    """
    task = PushT(clutter=True, robot="xarm6")
    assert task.model.nu == 5
    assert list(task.block_qpos_adr) != [0, 1, 2]

    state = task.make_data()
    known_block_pose = jnp.array([0.11, -0.22, 0.33])
    state = state.replace(
        qpos=state.qpos.at[task.block_qpos_adr].set(known_block_pose)
    )
    assert jnp.allclose(task._block_pose(state), known_block_pose)


def test_xarm6_consensus_task_methods() -> None:
    """Check shapes of the ConsensusTask (ADMM) methods for robot='xarm6'.

    Mirrors `test_clutter_consensus_task_methods`, with nu=5 instead of 2.
    `realized_consensus` is a known, documented placeholder (real contact-
    force extraction is deferred, see PushT.realized_consensus's docstring
    and XARM6_ADMM_INTEGRATION_PLAN.md) -- checked here for shape and for
    actually being the zero stub, not for physical correctness.
    """
    task = PushT(clutter=True, planning_dt=0.05, robot="xarm6")
    assert task.consensus_dim == 3

    state = task.make_data()
    state = jax.jit(mjx.forward)(task.model, state)

    obj_state = task.object_state_from_robot(state)
    assert obj_state.shape == (3,)

    w = jnp.array([1.0, 2.0, 0.1])
    next_state = task.object_dynamics(obj_state, w)
    assert next_state.shape == (3,)

    assert task.object_action_scale().shape == (3,)
    assert task.consensus_scale().shape == (3,)

    ell = task.robot_running_cost(state, jnp.zeros(5), jnp.zeros(3))
    assert ell.shape == ()
    assert jnp.isfinite(ell)

    phi = task.robot_terminal_cost(state)
    assert phi.shape == ()

    consensus_val = task.realized_consensus(state)
    assert consensus_val.shape == (3,)
    assert jnp.allclose(consensus_val, jnp.zeros(3))


def test_xarm6_planning_model_is_stable() -> None:
    """The xarm6 planning model must not diverge at the planning timestep.

    Mirrors `test_clutter_planning_model_is_stable`'s point-mass check:
    `xarm6_pusht_clutter.xml` starts from `integrator="implicitfast"` (not
    "Euler") from the outset specifically to avoid repeating that exact
    bug with the arm's own velocity actuators.
    """
    task = PushT(clutter=True, planning_dt=0.05, robot="xarm6")
    state = task.make_data()
    step = jax.jit(mjx.step)
    ctrl = jnp.zeros(5).at[0].set(1.0)
    for _ in range(20):  # 1 second of constant commanded velocity
        state = step(task.model, state.replace(ctrl=ctrl))

    assert jnp.all(jnp.isfinite(state.qpos))
    assert jnp.all(jnp.isfinite(state.qvel))


if __name__ == "__main__":
    test_task("jax", False)
    test_task("jax", True)
