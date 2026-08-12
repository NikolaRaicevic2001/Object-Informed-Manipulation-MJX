import copy
import inspect
from typing import Optional, Type

import jax
import jax.numpy as jnp
import mujoco

from oim.alg_base import SamplingBasedController
from oim.algs import (
    ADMM,
    CBO,
    MPPI,
    PoseConsensus,
    WrenchConsensus,
    make_object_shim,
)
from oim.algs.admm import ObjectSubproblem
from oim.tasks.pusht import PushT

PLAN_DT = 0.05
HORIZON = 15


def _build_task(robot: str = "point") -> PushT:
    return PushT(clutter=True, planning_dt=PLAN_DT, robot=robot)


def _build_admm(
    task: PushT,
    n_admm: int = 4,
    proximal_weight: float = 0.05,
    noise_min: float = 0.0,
    noise_kappa: float = 0.0,
    noise_max: float = 0.0,
    object_iterations: int = 1,
    object_cls: Type[SamplingBasedController] = MPPI,
    object_kwargs: Optional[dict] = None,
) -> ADMM:
    consensus = WrenchConsensus(max_dual=15.0)
    robot_optimizer = MPPI(
        task,
        num_samples=8,
        noise_level=0.4,
        temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT,
        spline_type="linear",
        num_knots=4,
        seed=5,
    )
    shim = make_object_shim(task, dt=PLAN_DT)
    default_object_kwargs = dict(
        num_samples=8,
        plan_horizon=HORIZON * PLAN_DT,
        spline_type="zero",
        num_knots=HORIZON,
        seed=5,
        iterations=object_iterations,
    )
    if object_cls is MPPI:
        default_object_kwargs.update(noise_level=1.0, temperature=1.0)
    elif object_cls is CBO:
        default_object_kwargs.update(
            initial_noise_level=1.0,
            temperature=0.1,
            consensus_weight=1.0,
            noise_weight=1.0,
            step_size=0.1,
        )
    if object_kwargs:
        default_object_kwargs.update(object_kwargs)
    object_optimizer = object_cls(shim, **default_object_kwargs)

    return ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=n_admm,
        eps_r=1.0,
        eps_s=1.0,
        proximal_weight=proximal_weight,
        rho_init=1.0,
        noise_min=noise_min,
        noise_kappa=noise_kappa,
        noise_max=noise_max,
    )


def test_wrench_consensus_math() -> None:
    """Unit tests for WrenchConsensus against hand-computed values."""
    consensus = WrenchConsensus(max_dual=10.0)

    # z_update: simple average of both blocks' extracted values + duals.
    a_o = jnp.array([1.0, 1.0, 1.0])
    a_r = jnp.array([3.0, 3.0, 3.0])
    zero = jnp.zeros(3)
    z = consensus.z_update(a_o, a_r, zero, zero, zero)
    assert jnp.allclose(z, jnp.array([2.0, 2.0, 2.0]))

    # ...and the base point it is taken about must cancel, so that
    # `PoseConsensus`'s tangent-space formulation left the wrench space's
    # own behaviour bit-for-bit unchanged rather than merely close.
    for base in (zero, jnp.array([7.0, -2.0, 0.5]), a_r):
        assert jnp.allclose(
            consensus.z_update(a_o, a_r, zero, zero, base),
            jnp.array([2.0, 2.0, 2.0]),
        )

    # dual_update, no clipping.
    dual = consensus.dual_update(
        jnp.array([5.0, 0.0, 0.0]), jnp.array([2.0, 0.0, 0.0]), zero
    )
    assert jnp.allclose(dual, jnp.array([3.0, 0.0, 0.0]))

    # dual_update, with anti-windup clipping.
    dual_clipped = consensus.dual_update(
        jnp.array([100.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), zero
    )
    assert jnp.allclose(dual_clipped, jnp.array([10.0, 0.0, 0.0]))

    # penalty_cost is zero when actual == z - dual.
    zero_penalty = consensus.penalty_cost(a_o, a_o, zero, rho=5.0)
    assert jnp.allclose(zero_penalty, 0.0)

    # penalty_cost scales linearly with rho.
    diff = jnp.array([1.0, 0.0, 0.0])
    p1 = consensus.penalty_cost(diff, zero, zero, rho=1.0)
    p2 = consensus.penalty_cost(diff, zero, zero, rho=2.0)
    assert jnp.allclose(p2, 2.0 * p1)
    assert jnp.allclose(p1, 0.5 * jnp.sum(diff**2))


def test_pose_consensus_wraps_the_heading() -> None:
    """SE(2) consensus must wrap theta, or it breaks exactly at the goal.

    Every tabletop scene's goal is a 180-degree flip, theta_g = pi, which
    sits precisely on the (-pi, pi] branch cut. Two poses a hair either
    side of it are physically the same orientation, and a plain
    subtraction would report them as 2*pi apart -- so the residual would
    blow up at the one place the run is supposed to be converging.
    """
    consensus = PoseConsensus(
        max_dual=jnp.array([0.2, 0.2, 2.0]),
        scale=jnp.array([0.1, 0.1, 1.0]),
    )
    near_plus = jnp.array([0.0, 0.0, 3.14])
    near_minus = jnp.array([0.0, 0.0, -3.14])

    # The two headings differ by ~0.0032 rad, not by ~6.28.
    delta = consensus.difference(near_plus, near_minus)
    assert jnp.abs(delta[2]) < 0.01
    assert consensus.residual_norm(delta) < 0.01

    # `increment` must land back inside (-pi, pi].
    stepped = consensus.increment(near_plus, jnp.array([0.0, 0.0, 0.01]))
    assert -jnp.pi < stepped[2] <= jnp.pi

    # A pose sequence shifts by repeating its last entry: zero-filling
    # would drag the horizon's tail to the world origin, which is a
    # specific pose rather than the absence of one.
    seq = jnp.array([[1.0, 1.0, 0.1], [2.0, 2.0, 0.2], [3.0, 3.0, 0.3]])
    assert jnp.allclose(consensus.shift(seq)[-1], seq[-1])


def test_object_consensus_selects_wrench_or_pose() -> None:
    """A^o is the wrench by default and the resulting pose when asked."""
    wrench_task = _build_task()
    pose_task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus_variable="pose",
    )
    obj_state = jnp.array([0.3, -0.2, 0.5])
    w = jnp.array([1.0, 2.0, 0.3])

    assert jnp.allclose(wrench_task.object_consensus(obj_state, w), w)
    assert jnp.allclose(pose_task.object_consensus(obj_state, w), obj_state)

    # And the normalization follows the variable: the friction-cone limit
    # for a wrench, the object's own size for a pose.
    assert jnp.allclose(
        wrench_task.consensus_scale(), wrench_task.object_model.wrench_limit
    )
    pose_scale = pose_task.consensus_scale()
    assert jnp.allclose(pose_scale[2], 1.0)
    assert 0.0 < float(pose_scale[0]) < 1.0


def test_pose_consensus_admm_jit() -> None:
    """The whole ADMM loop must jit and stay finite under pose consensus."""
    task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus_variable="pose",
    )
    scale = task.consensus_scale()
    consensus = PoseConsensus(max_dual=2.0 * scale, scale=scale)
    robot_optimizer = MPPI(
        task,
        num_samples=8,
        noise_level=0.4,
        temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT,
        spline_type="linear",
        num_knots=4,
        seed=5,
    )
    object_optimizer = MPPI(
        make_object_shim(task, dt=PLAN_DT),
        num_samples=8,
        noise_level=1.0,
        temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT,
        spline_type="zero",
        num_knots=HORIZON,
        seed=5,
    )
    ctrl = ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=3,
        eps_r=0.5,
        eps_s=0.5,
        proximal_weight=0.1,
        rho_init=1.0,
    )
    params, rollouts = jax.jit(ctrl.optimize)(
        task.make_data(), ctrl.init_params()
    )

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert jnp.all(jnp.isfinite(params.mean))
    assert jnp.all(jnp.isfinite(params.z))
    # z is now a pose trajectory, so every heading must be wrapped.
    assert jnp.all(jnp.abs(params.z[:, 2]) <= jnp.pi + 1e-5)


def test_consensus_scale_normalizes_penalty_and_residual() -> None:
    """`scale` must normalize both the penalty and the residual norm.

    Without it the penalty (contact forces ~10 N, squared) dwarfs the task
    costs (~1) and the robot optimizes wrench matching to the exclusion of
    reaching the object.
    """
    scale = jnp.array([8.0, 8.0, 0.5])
    raw = WrenchConsensus(max_dual=10.0)
    scaled = WrenchConsensus(max_dual=10.0, scale=scale)

    v = jnp.array([8.0, 0.0, 0.0])
    zero = jnp.zeros(3)

    # A residual of exactly one "scale" is 1.0 in normalized units.
    assert jnp.allclose(scaled.normalize(scale), jnp.ones(3))
    assert jnp.allclose(scaled.residual_norm(v), 1.0)
    assert jnp.allclose(raw.residual_norm(v), 8.0)

    # The penalty shrinks by scale^2, bringing it onto the task cost's scale.
    assert jnp.allclose(
        scaled.penalty_cost(v, zero, zero, rho=1.0),
        raw.penalty_cost(v, zero, zero, rho=1.0) / 64.0,
    )


def test_both_blocks_use_identical_consensus_penalty() -> None:
    """Object and robot blocks must score the consensus variable identically.

    Both must route through `ConsensusSpace.penalty_cost` rather than each
    hand-rolling a copy, otherwise the two blocks can silently drift into
    optimizing different things.
    """
    task = _build_task()
    ctrl = _build_admm(task)

    assert ctrl.object_subproblem.consensus is ctrl.consensus
    assert ctrl.robot_subproblem.consensus is ctrl.consensus

    # The task must NOT add a penalty of its own: robot_running_cost takes
    # only (state, control, obj_ref_t) -- no z / dual / rho.
    sig = inspect.signature(task.robot_running_cost)
    assert list(sig.parameters) == ["state", "control", "obj_ref_t"]


def test_admm_init_params_shapes() -> None:
    """Check that ADMMParams fields have the expected shapes."""
    task = _build_task()
    ctrl = _build_admm(task)
    params = ctrl.init_params()

    assert params.z.shape == (HORIZON, 3)
    assert params.gamma_o.shape == (HORIZON, 3)
    assert params.gamma_r.shape == (HORIZON, 3)
    assert params.rho.shape == ()
    assert params.primal_residual.shape == ()
    assert params.mean.shape == (4, 2)  # robot's own num_knots/nu
    assert params.tk.shape == (4,)


def test_admm_jit() -> None:
    """`jax.jit(ctrl.optimize)` must succeed and produce finite outputs.

    This is the regression test for the blocking issue that prevented ADMM
    from ever being driven by `run_interactive` (a Python `float()` early
    exit inside the loop cannot be traced under `jax.jit`).
    """
    task = _build_task()
    ctrl = _build_admm(task)
    params = ctrl.init_params()
    state = task.make_data()

    new_params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert jnp.all(jnp.isfinite(new_params.mean))
    assert jnp.all(jnp.isfinite(new_params.z))


def test_admm_jit_xarm6() -> None:
    """Same regression test as `test_admm_jit`, but for `robot="xarm6"`.

    The one thing that changes for this embodiment is a real
    `realized_consensus` (contact-force extraction, see its docstring in
    `oim/tasks/pusht.py` for the verification done and its caveats) in
    place of the `qfrc_constraint` trick -- this is the first test that
    exercises that path end-to-end under `jax.jit`, inside the full ADMM
    loop rather than in isolation.
    """
    task = _build_task(robot="xarm6")
    ctrl = _build_admm(task)
    params = ctrl.init_params()
    state = task.make_data()

    new_params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert jnp.all(jnp.isfinite(new_params.mean))
    assert jnp.all(jnp.isfinite(new_params.z))


def test_admm_pluggability() -> None:
    """ADMM must work with different optimizer types on each block.

    Uses MPPI for the robot side and CBO for the object side (with the
    object optimizer's `iterations` > 1), which would have caught the
    latent bug where the object subproblem silently ignored
    `optimizer.iterations`.
    """
    task = _build_task()
    ctrl = _build_admm(task, object_cls=CBO, object_iterations=3)
    params = ctrl.init_params()
    state = task.make_data()

    new_params, rollouts = jax.jit(ctrl.optimize)(state, params)

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert new_params.object_params.samples.shape == (8, HORIZON, 3)


def test_proximal_term_pulls_toward_previous_iterate() -> None:
    """A higher proximal weight should keep the mean closer to `prev_knots`.

    Isolated on the closed-form object subproblem (no MJX contact chaos),
    so the effect is analytically clean rather than swamped by
    contact-dynamics noise as it would be on the full robot rollout.
    """
    task = _build_task()
    consensus = WrenchConsensus(max_dual=15.0)
    shim = make_object_shim(task, dt=PLAN_DT)
    optimizer = MPPI(
        shim,
        num_samples=64,
        noise_level=1.0,
        temperature=1.0,
        plan_horizon=HORIZON * PLAN_DT,
        spline_type="zero",
        num_knots=HORIZON,
        seed=0,
    )

    obj_state0 = jnp.array([0.0, 0.0, 0.0])
    z = jnp.zeros((HORIZON, 3))
    dual_o = jnp.zeros((HORIZON, 3))
    rho = jnp.asarray(1.0)
    # An arbitrary anchor point, far from wherever the goal-tracking cost
    # would naturally pull the mean on its own.
    prev_knots = 5.0 * jnp.ones((HORIZON, 3))

    low = ObjectSubproblem(task, optimizer, consensus, proximal_weight=0.0)
    high = ObjectSubproblem(task, optimizer, consensus, proximal_weight=50.0)

    params0 = optimizer.init_params(seed=0)
    rng = jax.random.key(0)
    noise_scale = jnp.asarray(0.0)

    params_low, _, _, _ = low.optimize(
        obj_state0, params0, z, dual_o, rho, prev_knots, noise_scale, rng
    )
    params_high, _, _, _ = high.optimize(
        obj_state0, params0, z, dual_o, rho, prev_knots, noise_scale, rng
    )

    dist_low = jnp.sum((params_low.mean - prev_knots) ** 2)
    dist_high = jnp.sum((params_high.mean - prev_knots) ** 2)
    assert dist_high < dist_low


def test_admm_closed_loop_smoke() -> None:
    """Run a short closed loop and check for numerical stability."""
    task = _build_task()
    ctrl = _build_admm(
        task, n_admm=8, noise_min=0.05, noise_kappa=0.05, noise_max=1.0
    )

    exec_model = copy.deepcopy(task.mj_model)
    exec_model.opt.timestep = 0.002
    exec_data = mujoco.MjData(exec_model)
    mujoco.mj_forward(exec_model, exec_data)

    jit_optimize = jax.jit(ctrl.optimize)
    params = ctrl.init_params()

    pos_errs = []
    for _ in range(20):
        robot_data = task.make_data().replace(
            qpos=jnp.array(exec_data.qpos),
            qvel=jnp.array(exec_data.qvel),
        )
        params, rollouts = jit_optimize(robot_data, params)
        assert jnp.all(jnp.isfinite(rollouts.costs))

        u0 = jax.jit(ctrl.get_action)(params, robot_data.time)
        exec_data.ctrl[:] = jax.device_get(u0)
        for _ in range(int(round(PLAN_DT / exec_model.opt.timestep))):
            mujoco.mj_step(exec_model, exec_data)

        pos_err = float(
            jnp.linalg.norm(jnp.asarray(exec_data.qpos[:2]) - task.goal[:2])
        )
        pos_errs.append(pos_err)

    assert all(e < 10.0 for e in pos_errs)  # bounded, no blow-up
