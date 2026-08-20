import copy
import inspect
from typing import Optional, Type

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import mjx_forward
from mujoco import mjx

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
from oim.objects import se2_distance_sq
from oim.runtime.logs import local_goal_marker
from oim.runtime.mjcf import mocap_id
from oim.tasks.pusht import PushT

PLAN_DT = 0.05
# Do not shrink this to speed the suite up. It looks like free savings --
# these tests are compile-bound and no assertion depends on the length --
# and it is the opposite: measured on `test_admm_jit_xarm6` alone,
# HORIZON=6 takes 286.5 s against 72.4 s at 15. A short `lax.scan` is
# cheap enough for XLA to unroll, so the graph it must compile grows
# instead of shrinking.
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

    # The task must NOT add a penalty of its own: no z / dual / rho.
    # `local_goal` (x^{o*}_H) and `weight_scale` (`time_ramp` at the
    # horizon start) are references/weights, not consensus quantities.
    sig = inspect.signature(task.robot_running_cost)
    assert list(sig.parameters) == [
        "state",
        "control",
        "obj_ref_t",
        "local_goal",
        "weight_scale",
    ]


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
    # 8 rather than 20: this is a smoke test for blow-up, and divergence
    # under these dynamics shows up in the first few steps or not at all.
    # The loop is jitted after the first pass, so the steps are cheap --
    # but each one still round-trips through CPU MuJoCo.
    for _ in range(8):
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


def test_local_goal_off_by_default_and_ignores_the_plan() -> None:
    """Default `PushT` tracks the global goal, whatever plan it is handed.

    The flag is off so that every config and recorded run predating it
    keeps its meaning; this pins that, rather than trusting the default in
    the signature to stay put.
    """
    task = _build_task()
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)

    assert task.use_local_goal is False
    # A local goal far from the global one must change nothing.
    elsewhere = jnp.array([-1.0, -1.0, 0.0])
    ref = jnp.zeros(3)
    assert float(
        task.robot_running_cost(state, jnp.zeros(2), ref, elsewhere)
    ) == float(task.robot_running_cost(state, jnp.zeros(2), ref))
    assert float(task.robot_terminal_cost(state, elsewhere)) == float(
        task.robot_terminal_cost(state)
    )


def test_local_goal_retargets_only_the_tracking_terms() -> None:
    """With the flag on, ell_o and the terminal term aim at x^{o*}_H.

    The two are checked against `se2_distance_sq` at the *local* goal
    directly, so this fails if either silently keeps tracking `task.goal`.
    """
    task = _build_task()
    task.use_local_goal = True
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    pose = task._block_pose(state)
    local = jnp.array([0.2, 0.1, 0.3])

    # Terminal cost is pure tracking, so it must equal the distance exactly.
    assert float(task.robot_terminal_cost(state, local)) == pytest.approx(
        float(
            se2_distance_sq(pose, local, task.qf_pos, task.qf_theta)
        ),
        rel=1e-6,
    )

    # The stage cost carries other terms, so compare the *difference*
    # between two local goals against the difference of the ell_o terms
    # alone -- everything else cancels.
    other = jnp.array([-0.3, 0.4, -0.2])
    ref = jnp.zeros(3)
    delta = float(
        task.robot_running_cost(state, jnp.zeros(2), ref, local)
    ) - float(task.robot_running_cost(state, jnp.zeros(2), ref, other))
    expected = float(
        se2_distance_sq(pose, local, task.q_pos, task.q_theta)
        - se2_distance_sq(pose, other, task.q_pos, task.q_theta)
    )
    assert delta == pytest.approx(expected, rel=1e-6)


def test_local_goal_leaves_shaping_fade_on_the_global_goal() -> None:
    """`shaping_fade` must not follow the local goal.

    It means "the task is nearly over"; against a target H steps away it
    would read ~0 every step and switch off align/tilt/tip height for the
    whole run. Nothing else in the cost path guards this, so it is pinned
    here.
    """
    task = _build_task()
    task.shaping_fade_dist = 0.15
    task.use_local_goal = True
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    pose = task._block_pose(state)

    # A local goal *at* the block would zero the fade if it were used.
    at_block = pose
    assert float(task.shaping_fade(pose)) == pytest.approx(1.0)
    ref = jnp.zeros(3)
    # Cost still contains full-weight shaping: compare against the same
    # call with the fade forced off, which must differ.
    with_fade = float(
        task.robot_running_cost(state, jnp.zeros(2), ref, at_block)
    )
    task.shaping_fade_dist = 0.0
    without = float(
        task.robot_running_cost(state, jnp.zeros(2), ref, at_block)
    )
    assert with_fade == pytest.approx(without, rel=1e-6)


def test_local_goal_snaps_back_to_the_global_goal_inside_the_fade_radius() -> (
    None
):
    """Within `shaping_fade_dist` of g, both tracking terms revert to g.

    The one thing the flag must not do is cost the run its last few
    centimetres: x^{o*}_H carries the object block's own residual error, so
    tracking it near the goal stops the robot short by exactly that
    residual. Checked at two fade radii around the *same* geometry, so the
    only thing that changes between the two halves is whether the gate
    fires.
    """
    task = _build_task()
    task.use_local_goal = True
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    pose = task._block_pose(state)
    dist = float(jnp.linalg.norm(pose[:2] - task.goal[:2]))
    assert dist > 0.0  # or neither half of this test means anything
    local = jnp.array([-1.0, -1.0, 0.0])

    # Radius short of the block: the gate is open, the plan endpoint wins.
    task.shaping_fade_dist = 0.5 * dist
    assert float(task.shaping_fade(pose)) == pytest.approx(1.0)
    assert float(task.robot_terminal_cost(state, local)) == pytest.approx(
        float(se2_distance_sq(pose, local, task.qf_pos, task.qf_theta)),
        rel=1e-6,
    )

    # Radius past it: identical to the flag being off entirely.
    task.shaping_fade_dist = 2.0 * dist
    assert float(task.robot_terminal_cost(state, local)) == pytest.approx(
        float(task.robot_terminal_cost(state)), rel=1e-6
    )
    # And the stage cost stops depending on which plan endpoint it is
    # handed -- everything but `ell_o` is independent of it and cancels.
    ref = jnp.zeros(3)
    other = jnp.array([0.3, -0.4, 0.2])
    assert float(
        task.robot_running_cost(state, jnp.zeros(2), ref, local)
    ) == pytest.approx(
        float(task.robot_running_cost(state, jnp.zeros(2), ref, other)),
        rel=1e-6,
    )


def test_local_goal_marker_draws_the_resolved_target_not_the_plan_end() -> (
    None
):
    """The ghost marker must agree with the cost, including the snap.

    It did not, for two independent reasons: `ADMM.local_goal` returns the
    raw x^{o*}_H, and both viewer paths hand `local_goal_marker` the object
    plan's last entry directly as `plan_endpoint` to avoid a second
    rollout -- so the gate was skipped twice over and the ghost sat on the
    plan endpoint while the cost tracked g.

    The gate keys on where the *block* is, which is the case that actually
    bites: a plan overshooting past g has its endpoint outside the radius
    while the object is well inside, so an endpoint-keyed gate strands the
    ghost through exactly the phase the snap exists for. That pairing --
    block inside, endpoint outside -- is the first case below.
    """
    task = _build_task()
    task.use_local_goal = True
    task.shaping_fade_dist = 0.15
    ctrl = _build_admm(task)

    mj_model = copy.deepcopy(task.mj_model)
    index = mocap_id(mj_model, "local_goal")
    assert index >= 0  # otherwise the marker no-ops and proves nothing
    mj_data = mujoco.MjData(mj_model)
    draw = local_goal_marker(ctrl, mj_model)

    goal = np.asarray(task.goal)

    def state_at(pose: np.ndarray) -> mjx.Data:
        qpos = jnp.zeros(task.mj_model.nq).at[:3].set(jnp.asarray(pose))
        return mjx_forward(task.model, task.make_data().replace(qpos=qpos))

    # Block 5 cm from g (inside the radius) but a plan that overshoots to
    # 50 cm past it. The ghost must be on g.
    near_goal = state_at(np.array([goal[0] + 0.05, goal[1], goal[2]]))
    overshoot = np.array([goal[0] + 0.5, goal[1], goal[2]])
    draw(mj_data, near_goal, None, overshoot)
    assert mj_data.mocap_pos[index][:2] == pytest.approx(goal[:2], abs=1e-6)

    # Block far from g: the ghost tracks the plan endpoint as before.
    far = state_at(np.array([goal[0] + 0.6, goal[1], goal[2]]))
    endpoint = np.array([goal[0] + 0.3, goal[1] + 0.1, goal[2]])
    draw(mj_data, far, None, endpoint)
    assert mj_data.mocap_pos[index][:2] == pytest.approx(
        endpoint[:2], abs=1e-6
    )

    # With tracking off the ghost is the global goal at any distance --
    # the cost never looks at the plan, so neither may the marker.
    task.use_local_goal = False
    draw = local_goal_marker(ctrl, copy.deepcopy(task.mj_model))
    draw(mj_data, far, None, endpoint)
    assert mj_data.mocap_pos[index][:2] == pytest.approx(goal[:2], abs=1e-6)


def test_local_goal_snap_is_inert_without_a_fade_radius() -> None:
    """`shaping_fade_dist = 0` must leave local-goal tracking untouched.

    That is the `DEFAULT_COSTS` value and what every config without the
    knob gets, so the gate has to be invisible there however close to the
    goal the block sits.
    """
    task = _build_task()
    task.use_local_goal = True
    task.shaping_fade_dist = 0.0
    # Put the block *at* the goal: the strongest form of "inside" there is.
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    pose = task._block_pose(state)
    task.goal = pose

    local = jnp.array([-1.0, -1.0, 0.0])
    assert float(task.robot_terminal_cost(state, local)) == pytest.approx(
        float(se2_distance_sq(pose, local, task.qf_pos, task.qf_theta)),
        rel=1e-6,
    )


def test_admm_local_goal_matches_the_object_plan_endpoint() -> None:
    """`ADMM.local_goal` is exactly what the robot block was handed.

    The marker is driven by `ADMM.local_goal` while the cost reads
    `obj_ref[-1]` inside the rollout. They are computed in two places, so
    if they ever diverge the picture stops describing the run. What the
    *task* does with the value is separate -- see
    `test_local_goal_snaps_back_to_the_global_goal_inside_the_fade_radius`.
    """
    task = _build_task()
    ctrl = _build_admm(task)
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    params, _ = jax.jit(ctrl.optimize)(state, ctrl.init_params())

    marker = jax.jit(ctrl.local_goal)(state, params)
    obj_state0 = task.object_state_from_robot(state)
    plan = ctrl.object_subproblem.nominal_plan(obj_state0, params.object_params)

    assert marker.shape == (3,)
    assert jnp.allclose(marker, plan[-1])


def test_substepping_the_robot_rollout_keeps_the_planning_step() -> None:
    """`substeps` refines the integration, it does not shorten the step.

    One `MJXRollout.step` must still advance exactly `planning_dt`,
    whatever `substeps` is: the horizon's length in seconds, the
    `dt`-weighted running costs and the spline knot times are all built
    around that, so a step that advanced `planning_dt / substeps` would
    silently shrink the horizon by the same factor.

    Also pins the refinement itself -- a finer integration has to CHANGE
    the result, or the substeps are being spent for nothing.
    """
    from oim.algs import MJXRollout

    task = _build_task()
    model = mjx.put_model(task.mj_model)
    data = mjx.put_data(task.mj_model, mujoco.MjData(task.mj_model))
    control = jnp.full((task.model.nu,), 0.5)

    out = {}
    for n in (1, 5):
        out[n] = jax.jit(MJXRollout(substeps=n).step)(model, data, control)
        assert float(out[n].time) == pytest.approx(PLAN_DT, rel=1e-5), (
            f"substeps={n} advanced {float(out[n].time)}, not {PLAN_DT}"
        )
    assert not np.allclose(
        np.asarray(out[1].qpos), np.asarray(out[5].qpos), atol=1e-9
    ), "substeps=5 integrated to the same answer as one coarse step"

    with pytest.raises(ValueError, match="substeps"):
        MJXRollout(substeps=0)
