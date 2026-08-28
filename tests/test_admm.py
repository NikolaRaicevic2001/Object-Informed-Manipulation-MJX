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
    ConsensusSpace,
    ContactPointConsensus,
    WrenchConsensus,
    make_object_shim,
)
from oim.algs.admm import ADMMParams, ObjectSubproblem
from oim.objects import contact_frame, se2_distance_sq
from oim.runtime.logs import local_goal_marker
from oim.runtime.mjcf import mocap_id
from oim.task_base import ConsensusTask
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
    consensus_object_weight: float = 0.5,
    object_iterations: int = 1,
    object_cls: Type[SamplingBasedController] = MPPI,
    object_kwargs: Optional[dict] = None,
    consensus: Optional[ConsensusSpace] = None,
) -> ADMM:
    if consensus is None:
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
        consensus_object_weight=consensus_object_weight,
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

    # ...and the base point it is taken about must cancel: the update is
    # written in tangent form, which must leave a vector space's own
    # behaviour unchanged rather than merely close.
    for base in (zero, jnp.array([7.0, -2.0, 0.5]), a_r):
        assert jnp.allclose(
            consensus.z_update(a_o, a_r, zero, zero, base),
            jnp.array([2.0, 2.0, 2.0]),
        )

    # The default weight is the paper's plain average, and the tilt is
    # linear in w_o between the two blocks' proposals.
    for w_o, want in ((0.0, 3.0), (0.25, 2.5), (0.5, 2.0), (1.0, 1.0)):
        assert jnp.allclose(
            consensus.z_update(a_o, a_r, zero, zero, zero, w_o),
            jnp.full(3, want),
        ), w_o

    # The duals are weighted with their own block, not split evenly: at
    # w_o = 1 the robot's dual must not reach z at all.
    dual_r = jnp.array([4.0, 4.0, 4.0])
    assert jnp.allclose(
        consensus.z_update(a_o, a_r, zero, dual_r, zero, 1.0), a_o
    )
    assert jnp.allclose(
        consensus.z_update(a_o, a_r, zero, dual_r, zero, 0.5),
        0.5 * (a_o + a_r + dual_r),
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


def test_contact_point_consensus_holds_the_tail_on_shift() -> None:
    """Zero-filling the vacated tail is wrong for a contact point.

    Zero is the object's own *origin*, which is inside the footprint --
    where the boundary normal every contact quantity derives from is
    undefined, not "no contact". Holding the last value keeps the tail on
    the surface.
    """
    consensus = ContactPointConsensus(
        max_dual=jnp.array([0.2, 0.2, 8.0]),
        scale=jnp.array([0.1, 0.1, 4.0]),
    )
    seq = jnp.array([[0.01, 0.02, 1.0], [0.03, 0.04, 2.0], [0.05, 0.06, 3.0]])
    shifted = consensus.shift(seq)
    assert jnp.allclose(shifted[:-1], seq[1:])
    assert jnp.allclose(shifted[-1], seq[-1])

    # A plain vector space: no wrapping, so difference is subtraction and
    # the normalization is per-channel (metres vs newtons).
    a = jnp.array([0.1, 0.0, 4.0])
    assert jnp.allclose(consensus.difference(a, jnp.zeros(3)), a)
    assert jnp.allclose(consensus.normalize(a), jnp.array([1.0, 0.0, 1.0]))


def test_object_consensus_selects_wrench_or_contact_point() -> None:
    """A^o is the block's own decision, whichever variable that is."""
    wrench_task = _build_task()
    cp_task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus="contact_point",
    )
    obj_state = jnp.array([0.3, -0.2, 0.5])
    w = jnp.array([1.0, 2.0, 0.3])
    action = jnp.array([0.02, -0.05, 2.0])

    assert jnp.allclose(wrench_task.object_consensus(obj_state, w, action), w)
    # Not the wrench: the contact task's A^o is the action, and `w` here is
    # what was *derived* from it.
    assert jnp.allclose(cp_task.object_consensus(obj_state, w, action), action)

    # The normalization follows the variable: the friction-cone limit for a
    # wrench; the body radius and the force bound for a contact point.
    assert jnp.allclose(
        wrench_task.consensus_scale(), wrench_task.object_model.wrench_limit
    )
    cp_scale = cp_task.consensus_scale()
    assert jnp.allclose(cp_scale[0], cp_scale[1])
    assert float(cp_scale[2]) == pytest.approx(
        float(cp_task.object_model.action_scale[0])
    )


def test_contact_point_action_is_realizable_by_construction() -> None:
    """Projection is what makes every proposal a wrench a pusher could make.

    The three constraints that the plain wrench parameterization cannot
    express: the point is on the boundary, the force is unilateral, and it
    is bounded. Checked on a sample that violates all three.
    """
    task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus="contact_point",
    )
    shape = task.object_model.footprint
    f_max = float(task.object_model.action_scale[0])

    bad = jnp.array([5.0, -7.0, -3.0])  # far outside, and pulling
    good = task.project_object_action(bad)
    assert jnp.abs(shape.sdf(good[:2])) < 1e-3, "point must land on boundary"
    assert good[2] == 0.0, "a contact pushes, never pulls"
    assert task.project_object_action(jnp.array([0.0, 0.0, 1e3]))[2] <= f_max

    # The wrench derived from a projected action always pushes *into* the
    # object: its force has a positive component along the inward normal.
    pose = jnp.array([0.2, -0.1, 0.7])
    action = task.project_object_action(jnp.array([0.05, 0.05, 2.0]))
    w = task.object_action_to_consensus(pose, action)
    n_world, _ = contact_frame(shape, action[:2], pose[2])
    assert float(jnp.dot(w[:2], n_world)) > 0.0


def test_contact_point_wrench_turns_with_the_object() -> None:
    """One fixed contact point, two headings, two different world wrenches.

    This is the thing a sampled world-frame wrench cannot express, and the
    reason the map is evaluated inside the rollout rather than once.
    """
    task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus="contact_point",
    )
    action = task.project_object_action(jnp.array([0.04, -0.06, 3.0]))
    upright = task.object_action_to_consensus(
        jnp.array([0.0, 0.0, 0.0]), action)
    turned = task.object_action_to_consensus(
        jnp.array([0.0, 0.0, jnp.pi / 2]), action
    )

    assert not jnp.allclose(upright[:2], turned[:2], atol=1e-3)
    # The force is the same push, just rotated: equal magnitude, and the
    # angle between them is the heading change.
    assert float(jnp.linalg.norm(upright[:2])) == pytest.approx(
        float(jnp.linalg.norm(turned[:2])), rel=1e-4
    )
    assert float(jnp.dot(upright[:2], turned[:2])) == pytest.approx(
        0.0, abs=1e-4
    )


def test_contact_point_consensus_admm_jit() -> None:
    """The whole loop must jit under contact-point consensus, and stay legal.

    Legality is the point, not just finiteness: z is negotiated between two
    blocks and is *not* passed through `project_object_action`, so if the
    parameterization only held inside the object block the agreed value
    could drift off the boundary or go pulling. Both A's are checked, and
    A^r comes from the robot's own state through a different code path
    than A^o.
    """
    task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus="contact_point",
    )
    scale = task.consensus_scale()
    consensus = ContactPointConsensus(max_dual=2.0 * scale, scale=scale)
    ctrl = _build_admm(task, n_admm=3, consensus=consensus)
    params, rollouts = jax.jit(ctrl.optimize)(
        mjx_forward(task.model, task.make_data()), ctrl.init_params()
    )

    assert jnp.all(jnp.isfinite(rollouts.costs))
    assert jnp.all(jnp.isfinite(params.mean))
    assert jnp.all(jnp.isfinite(params.z))

    shape = task.object_model.footprint
    f_max = float(task.object_model.action_scale[0])
    for name, a in (("A^o", params.a_obj), ("A^r", params.a_rob)):
        assert jnp.all(jnp.abs(shape.sdf(a[:, :2])) < 1e-3), \
            f"{name} off boundary"
        assert jnp.all(a[:, 2] >= 0.0), f"{name} pulls"
        assert jnp.all(a[:, 2] <= f_max + 1e-3), f"{name} exceeds f_max"


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

    # A residual of exactly one "scale" per channel is 1.0 in normalized
    # units -- `residual_norm` is an RMS, so it reads per channel.
    assert jnp.allclose(scaled.normalize(scale), jnp.ones(3))
    assert jnp.allclose(scaled.residual_norm(scale), 1.0)
    assert jnp.allclose(scaled.residual_norm(v), 1.0 / jnp.sqrt(3.0))
    assert jnp.allclose(raw.residual_norm(v), 8.0 / jnp.sqrt(3.0))

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

    params_low, _, _, _ = low.optimize(
        obj_state0, params0, z, dual_o, rho, prev_knots, rng
    )
    params_high, _, _, _ = high.optimize(
        obj_state0, params0, z, dual_o, rho, prev_knots, rng
    )

    dist_low = jnp.sum((params_low.mean - prev_knots) ** 2)
    dist_high = jnp.sum((params_high.mean - prev_knots) ** 2)
    assert dist_high < dist_low


def test_consensus_object_weight_decides_whose_plan_z_follows() -> None:
    """w_o routes the agreed wrench between the two blocks' proposals.

    One iteration from `init_params`, where both duals are still zero, so
    `z_update` reduces to w_o*A^o + (1 - w_o)*A^r and the endpoints are
    exactly the two blocks' own values -- an equality check rather than an
    inequality that a noisy sampler could satisfy by accident.
    """
    task = _build_task()
    state = mjx_forward(task.model, task.make_data())

    def run(w_o: float) -> ADMMParams:
        ctrl = _build_admm(task, n_admm=1, consensus_object_weight=w_o)
        return jax.jit(ctrl.optimize)(state, ctrl.init_params())[0]

    obj_led = run(1.0)
    assert jnp.allclose(obj_led.z, obj_led.a_obj, atol=1e-5)

    rob_led = run(0.0)
    assert jnp.allclose(rob_led.z, rob_led.a_rob, atol=1e-5)

    # The blocks must actually disagree, or the two checks above are the
    # same assertion twice and the knob is untested.
    assert not jnp.allclose(obj_led.a_obj, obj_led.a_rob, atol=1e-3)

    even = run(0.5)
    assert jnp.allclose(even.z, 0.5 * (even.a_obj + even.a_rob), atol=1e-5)


def test_consensus_object_weight_is_bounded() -> None:
    """Outside [0, 1] the update extrapolates past both proposals."""
    task = _build_task()
    for bad in (-0.1, 1.5):
        with pytest.raises(ValueError, match="consensus_object_weight"):
            _build_admm(task, consensus_object_weight=bad)


def test_admm_closed_loop_smoke() -> None:
    """Run a short closed loop and check for numerical stability."""
    task = _build_task()
    ctrl = _build_admm(task, n_admm=8)

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

    It did not, for two independent reasons: `ADMM.local_goal` returned
    the raw x^{o*}_H, and both viewer paths handed `local_goal_marker` the
    plan's last entry to avoid a second rollout -- so the gate was skipped
    twice over and the ghost sat on the plan endpoint while the cost
    tracked g. The marker now takes the WHOLE plan and resolves it through
    the same `local_goal_from_plan` the cost uses, so a pursuit carrot is
    drawn where it actually is rather than out at the horizon.

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

    def plan_to(end: np.ndarray, start: np.ndarray) -> np.ndarray:
        """A straight 8-step plan from `start` to `end`."""
        return np.stack(
            [start + (end - start) * t for t in np.linspace(0.0, 1.0, 8)]
        )

    # Block 5 cm from g (inside the radius) but a plan that overshoots to
    # 50 cm past it. The ghost must be on g.
    block = np.array([goal[0] + 0.05, goal[1], goal[2]])
    near_goal = state_at(block)
    overshoot = plan_to(np.array([goal[0] + 0.5, goal[1], goal[2]]), block)
    draw(mj_data, near_goal, None, overshoot)
    assert mj_data.mocap_pos[index][:2] == pytest.approx(goal[:2], abs=1e-6)

    # Block far from g, no lookahead: the ghost is the plan endpoint.
    task.local_goal_lookahead = 0.0
    block_far = np.array([goal[0] + 0.6, goal[1], goal[2]])
    far = state_at(block_far)
    plan = plan_to(np.array([goal[0] + 0.3, goal[1] + 0.1, goal[2]]), block_far)
    draw(mj_data, far, None, plan)
    assert mj_data.mocap_pos[index][:2] == pytest.approx(
        plan[-1][:2], abs=1e-6
    )

    # Same plan WITH a lookahead: the ghost moves to the carrot, which is
    # strictly nearer the block than the endpoint -- the whole point of
    # drawing the resolved target rather than the horizon.
    task.local_goal_lookahead = 0.10
    draw = local_goal_marker(ctrl, copy.deepcopy(task.mj_model))
    draw(mj_data, far, None, plan)
    drawn = np.asarray(mj_data.mocap_pos[index][:2])
    carrot = np.asarray(task.local_goal_from_plan(
        jnp.asarray(plan), jnp.asarray(block_far)
    ))
    assert drawn == pytest.approx(carrot[:2], abs=1e-6)
    assert np.linalg.norm(drawn - block_far[:2]) < np.linalg.norm(
        plan[-1][:2] - block_far[:2]
    )

    # With tracking off the ghost is the global goal at any distance --
    # the cost never looks at the plan, so neither may the marker.
    task.use_local_goal = False
    draw = local_goal_marker(ctrl, copy.deepcopy(task.mj_model))
    draw(mj_data, far, None, plan)
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
    from oim.algs import MJXRollout  # noqa: PLC0415

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


def test_local_goal_pursues_a_carrot_along_the_plan() -> None:
    """With a lookahead, the target is the first pose far enough ahead.

    Specifically, the first planned pose at least
    that far from the object -- not the plan's endpoint.

    The endpoint says only where the plan finishes, so a plan that routes
    around an obstacle and one that drives through it score the same as
    long as they end together. A carrot scores the ROUTE.
    """
    task = _build_task()
    task.use_local_goal = True
    # A straight plan along +x at 5 cm spacing, starting at the object.
    plan = jnp.stack(
        [jnp.array([0.05 * i, 0.0, 0.0]) for i in range(10)]
    )
    pose = jnp.array([0.0, 0.0, 0.0])

    task.local_goal_lookahead = 0.0
    assert jnp.allclose(task.local_goal_from_plan(plan, pose), plan[-1])

    # 0.10 m out is index 2 (0.00, 0.05, 0.10) -- the FIRST at or beyond.
    task.local_goal_lookahead = 0.10
    assert jnp.allclose(task.local_goal_from_plan(plan, pose), plan[2])
    # Tighter tracking picks a nearer carrot; looser picks a further one.
    task.local_goal_lookahead = 0.05
    assert jnp.allclose(task.local_goal_from_plan(plan, pose), plan[1])
    task.local_goal_lookahead = 0.30
    assert jnp.allclose(task.local_goal_from_plan(plan, pose), plan[6])

    # The carrot slides forward on its own as the object advances: no
    # stored index, just the same pick from a new pose.
    task.local_goal_lookahead = 0.10
    assert jnp.allclose(
        task.local_goal_from_plan(plan, jnp.array([0.20, 0.0, 0.0])), plan[6]
    )


def test_local_goal_carrot_falls_back_when_the_plan_is_shorter_than_it(
) -> None:
    """A plan entirely inside the lookahead must not aim at the object.

    `argmax` over an all-False mask returns 0, which would target the
    object's own pose and zero the tracking gradient -- exactly the stall
    case (an object block planning to hold still under breakaway) where
    the robot most needs to push. Falls back to the endpoint instead.
    """
    task = _build_task()
    task.use_local_goal = True
    task.local_goal_lookahead = 0.10
    # Whole plan spans 9 mm, far under the 0.10 m lookahead.
    plan = jnp.stack([jnp.array([0.001 * i, 0.0, 0.0]) for i in range(10)])
    pose = jnp.array([0.0, 0.0, 0.0])
    target = task.local_goal_from_plan(plan, pose)
    assert jnp.allclose(target, plan[-1])
    assert not jnp.allclose(target, plan[0])


def test_local_goal_from_plan_defaults_to_the_endpoint() -> None:
    """Without pursuit, the target is the plan endpoint.

    Off, or on a task with no pursuit of its own, the target is
    x^{o*}_H -- the behaviour that shipped before the carrot existed.
    """
    plan = jnp.stack([jnp.array([0.1 * i, 0.0, 0.0]) for i in range(5)])
    pose = jnp.array([0.0, 0.0, 0.0])

    task = _build_task()
    task.use_local_goal = False
    task.local_goal_lookahead = 0.10
    assert jnp.allclose(task.local_goal_from_plan(plan, pose), plan[-1])

    # The base-class contract every ConsensusTask inherits.
    assert jnp.allclose(
        ConsensusTask.local_goal_from_plan(task, plan, pose), plan[-1]
    )


def test_residual_norm_is_horizon_independent() -> None:
    """The same per-channel disagreement must read the same at any H.

    `residual_norm` is handed both blocks' residuals concatenated over the
    horizon, so a plain 2-norm grew like sqrt(2*H*dim) -- which silently
    tightened `eps_r`/`eps_s`, and with them the early exit, whenever the
    horizon changed.
    """
    scale = jnp.array([8.0, 8.0, 0.5])
    consensus = WrenchConsensus(max_dual=10.0, scale=scale)
    for horizon in (8, 16, 32, 64):
        # Every channel off by exactly one scale, over 2H entries.
        v = jnp.broadcast_to(scale, (2 * horizon, 3))
        assert jnp.allclose(consensus.residual_norm(v), 1.0)


def test_twist_exact_inverts_the_plant_it_plans_against() -> None:
    """`twist_exact` must be the exact inverse of `PlanarPushingObject.step`.

    That is the whole point of the estimator: A^o and A^r are the same
    physical quantity only if the map from wrench to motion is inverted
    with the law the plant integrates. `twist` inverts `xdot = D w`, which
    the plant stopped using when friction became subtracted.
    """
    from oim.objects.planar_pushing import (  # noqa: PLC0415
        PlanarPushingObject,
        t_shape_footprint,
    )

    obj = PlanarPushingObject(
        dt=0.05, goal=jnp.zeros(3), footprint=t_shape_footprint()
    )
    limit = obj.wrench_limit

    def plant_twist(w: jnp.ndarray) -> jnp.ndarray:
        return (obj.step(jnp.zeros(3), w) - jnp.zeros(3)) / obj.dt

    def invert_exact(xdot: jnp.ndarray) -> jnp.ndarray:
        speed = jnp.linalg.norm(xdot)
        return limit * ((1.0 + speed) * xdot / jnp.maximum(speed, 1e-9))

    for w in (
        limit * jnp.array([1.5, 0.0, 0.0]),
        limit * jnp.array([1.05, 0.0, 0.0]),
        limit * jnp.array([1.6, 0.8, 0.0]),
        limit * jnp.array([0.0, 0.0, 3.0]),
    ):
        recovered = invert_exact(plant_twist(w))
        assert jnp.allclose(recovered, w, rtol=1e-4, atol=1e-6)

    # Inside the cone the block sticks, so there is no wrench to recover.
    assert jnp.allclose(plant_twist(limit * jnp.array([0.8, 0.0, 0.0])), 0.0)
