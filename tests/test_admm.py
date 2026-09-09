import copy
import inspect
from typing import Optional, Type

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
import pytest
from conftest import mjx_forward

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
from oim.algs.admm import ADMMParams, ObjectSubproblem, _finite_or
from oim.objects import contact_frame
from oim.runtime.logs import object_plan_marker
from oim.runtime.samplers import consensus_space
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
    lagged_consensus: str = "off",
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
        lagged_consensus=lagged_consensus,
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
    # `weight_scale` (`time_ramp` at the horizon start) is a weight, not
    # a consensus quantity, and `ref_pose` is the object plan's endpoint
    # for the shaping reference (opt-in, `align_ref`), not a penalty.
    sig = inspect.signature(task.robot_running_cost)
    assert list(sig.parameters) == [
        "state", "control", "weight_scale", "ref_pose"
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

    params_low, _, _, _, _ = low.optimize(
        obj_state0, params0, z, dual_o, rho, prev_knots, rng
    )
    params_high, _, _, _, _ = high.optimize(
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


def test_robot_cost_reads_the_global_goal_not_the_object_plan() -> None:
    """`align` and both tracking terms aim at g, as the baselines do.

    The bug this pins: the ADMM robot block used to score `align` against
    the object block's pointwise plan x^{o*}_t while `ell_o` scored the
    global goal, so the two halves of one cost pulled at different
    targets -- and a plan index that lagged the rollout could flip
    `align`'s reference to point backwards. `PushT.running_cost` (the
    flat baseline) always used g; this is the ADMM path agreeing with it.
    """
    task = PushT(
        clutter=True, planning_dt=PLAN_DT, costs={"align_ref": "goal"}
    )
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)

    # The plan reaches the cost through one optional keyword only
    # (`ref_pose`, the endpoint, read solely under `align_ref="plan_end"`
    # -- see `test_align_ref_plan_end_reads_the_plan_endpoint`); no
    # positional slot for a pointwise plan or a local goal exists.
    sig = inspect.signature(task.robot_running_cost)
    assert list(sig.parameters) == [
        "state", "control", "weight_scale", "ref_pose"
    ]
    assert list(
        inspect.signature(task.robot_terminal_cost).parameters
    ) == ["state", "weight_scale"]
    # Under `align_ref="goal"` a handed-in endpoint changes nothing: the
    # two blocks couple through z alone.
    assert task.align_ref == "goal"
    u = jnp.zeros(task.model.nu)
    plain = float(task.robot_running_cost(state, u))
    with_ref = float(
        task.robot_running_cost(
            state, u, ref_pose=jnp.asarray([0.6, 0.4, 1.0])
        )
    )
    assert plain == pytest.approx(with_ref)

    # `_ell_r`'s reference IS the global goal: perturbing g moves the
    # cost, and the value matches feeding g in by hand.
    pose = task._block_pose(state)
    pusher = task._pusher_pos(state)
    by_hand = float(task._ell_r(state, pose, pusher, task.goal))
    task_goal = task.goal
    task.goal = jnp.asarray([0.6, 0.4, 1.0])
    moved = float(task._ell_r(state, pose, pusher, task.goal))
    task.goal = task_goal
    assert by_hand != pytest.approx(moved, rel=1e-6)


def test_local_goal_is_gone_from_the_task_and_the_controller() -> None:
    """The removed feature leaves no attribute behind to be read again."""
    task = _build_task()
    ctrl = _build_admm(task)
    for name in (
        "local_goal_from_plan", "tracking_goal", "use_local_goal",
        "local_goal_lookahead",
    ):
        assert not hasattr(task, name), name
    assert not hasattr(ctrl, "local_goal")
    assert not hasattr(ConsensusTask, "local_goal_from_plan")


def test_object_plan_endpoint_is_the_plans_last_pose() -> None:
    """`ADMM.object_plan_endpoint` is x^{o*}_H, drawn by the ghost."""
    task = _build_task()
    ctrl = _build_admm(task)
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    params = ctrl.init_params()
    endpoint = np.asarray(jax.jit(ctrl.object_plan_endpoint)(state, params))
    object_plan, _, _ = jax.jit(ctrl.nominal_plans)(state, params)
    assert np.allclose(endpoint, np.asarray(object_plan)[-1])


def test_object_plan_marker_draws_the_plan_endpoint() -> None:
    """The ghost shows what the object block asked for, both code paths.

    Supplying `object_plan` is the fast path (the caller already rolled
    the block out); omitting it makes the marker roll it out itself. The
    two must agree, or the ghost would move when a caller optimized.
    """
    task = _build_task()
    ctrl = _build_admm(task)
    mj_model = copy.deepcopy(task.mj_model)
    index = mocap_id(mj_model, "object_plan")
    if index < 0:
        pytest.skip("scene declares no object_plan marker")
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    params = ctrl.init_params()
    mj_data = mujoco.MjData(mj_model)

    draw = object_plan_marker(ctrl, mj_model)
    plan = np.asarray(jax.jit(ctrl.nominal_plans)(state, params)[0])
    draw(mj_data, state, params, plan)
    supplied = mj_data.mocap_pos[index].copy()
    draw(mj_data, state, params, None)
    assert np.allclose(mj_data.mocap_pos[index][:2], plan[-1][:2], atol=1e-6)
    assert np.allclose(supplied[:2], plan[-1][:2], atol=1e-6)


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


def test_lagged_consensus_is_validated() -> None:
    """An unknown mode must fail at construction, not silently plan wrong.

    `False` is accepted because YAML 1.1 reads a bare `off` as a boolean,
    so a config written the obvious way arrives here as `False`.
    """
    task = _build_task()
    with pytest.raises(ValueError, match="lagged_consensus"):
        _build_admm(task, lagged_consensus="yes")
    assert _build_admm(task, lagged_consensus=False).lagged_consensus == "off"


def test_lagged_consensus_wires_only_the_blocks_it_names() -> None:
    """`robot` must leave the object block re-rolling its own nominal."""
    task = _build_task()
    off = _build_admm(task)
    assert not off.robot_subproblem.lagged
    assert not off.object_subproblem.lagged

    robot = _build_admm(task, lagged_consensus="robot")
    assert robot.robot_subproblem.lagged
    assert not robot.object_subproblem.lagged

    both = _build_admm(task, lagged_consensus="both")
    assert both.robot_subproblem.lagged
    assert both.object_subproblem.lagged


def test_lagged_robot_block_reads_a_off_the_incoming_mean() -> None:
    """The extra batch row must be the rollout it replaces, one round back.

    Two things have to hold at once for the saving to be sound. The A^r
    the block returns has to be the one `nominal_realized_consensus`
    would compute for the mean the call was ENTERED with -- an off-by-one
    here (reading the last MPPI pass, or the outgoing mean) would silently
    change which trajectory the consensus is built from. And the sampled
    population has to be untouched: the extra row is sliced off before
    the optimizer update, so the knots that get reweighted must be
    bit-identical to the unlagged block's.
    """
    task = _build_task()
    ctrl = _build_admm(task, lagged_consensus="robot")
    plain = _build_admm(task)
    params = ctrl.init_params()
    state = mjx_forward(task.model, task.make_data())
    args = (
        state,
        params.robot_params,
        params.z,
        params.gamma_r,
        params.rho,
        params.robot_params.mean,
        jax.random.key(0),
    )

    _, lagged_rollouts, a_rob = jax.jit(ctrl.robot_subproblem.optimize)(*args)
    _, plain_rollouts, none = jax.jit(plain.robot_subproblem.optimize)(*args)
    assert none is None, "the unlagged block must not return an A^r"

    expected = jax.jit(plain.robot_subproblem.nominal_realized_consensus)(
        state, params.robot_params
    )
    assert a_rob.shape == expected.shape
    # Loose: the row rides in a batch, and MJX's batched solver is not
    # bit-reproducible against a lone rollout. Tight enough to catch the
    # wrong trajectory, which differs by order 1.
    np.testing.assert_allclose(a_rob, expected, rtol=2e-2, atol=2e-2)
    np.testing.assert_array_equal(lagged_rollouts.knots, plain_rollouts.knots)


def test_lagged_consensus_jit() -> None:
    """Both lagged modes must trace, and produce finite plans."""
    task = _build_task()
    for mode in ("robot", "both"):
        ctrl = _build_admm(task, lagged_consensus=mode)
        new_params, rollouts = jax.jit(ctrl.optimize)(
            task.make_data(), ctrl.init_params()
        )
        assert jnp.all(jnp.isfinite(rollouts.costs)), mode
        assert jnp.all(jnp.isfinite(new_params.mean)), mode
        assert jnp.all(jnp.isfinite(new_params.z)), mode
        assert jnp.all(jnp.isfinite(new_params.a_rob)), mode
        assert jnp.all(jnp.isfinite(new_params.a_obj)), mode


def test_finite_or_passes_healthy_values_through_unchanged() -> None:
    """The guard is a no-op when nothing is wrong.

    Asserted rather than assumed: it sits on the consensus path of every
    ADMM round, so if it perturbed a healthy value it would reprice every
    recorded run.
    """
    value = jnp.array([[1.0, -2.0, 3.0], [0.0, 4.0, -5.0]])
    fallback = jnp.zeros_like(value)
    out, ok = _finite_or(value, fallback)
    assert bool(ok)
    assert jnp.array_equal(out, value)


@pytest.mark.parametrize("bad", [jnp.nan, jnp.inf, -jnp.inf])
def test_finite_or_substitutes_the_fallback(bad: float) -> None:
    """One bad entry discards the whole array, not just that entry.

    All-or-nothing on purpose: a consensus value is a horizon-length
    sequence the blocks negotiated jointly, so half of one is not a
    meaningful proposal.
    """
    value = jnp.ones((3, 3)).at[1, 2].set(bad)
    fallback = jnp.full((3, 3), 7.0)
    out, ok = _finite_or(value, fallback)
    assert not bool(ok)
    assert jnp.array_equal(out, fallback)


def test_a_nan_does_not_outlive_the_step_that_produced_it() -> None:
    """A poisoned `ADMMParams` recovers on the next solve.

    The regression this guards is specific and was observed on a real run:
    `z` and both duals are carried across control steps and never reset, so
    before the guard a single non-finite rollout made `penalty_cost` NaN
    for every sample from then on, `MPPI.update_params` saw no finite cost
    and held its mean, and the arm froze for the rest of the episode --
    `primal=nan dual=nan` with the pose bit-identical for hundreds of
    steps.

    Every field is poisoned at an index that SURVIVES the receding-horizon
    warm-start shift. Index 0 does not: `shift` drops it, which would make
    this pass without the guard doing anything.
    """
    task = _build_task()
    ctrl = _build_admm(task)
    state = mjx_forward(task.model, task.make_data())
    optimize = jax.jit(ctrl.optimize)

    params = ctrl.init_params(seed=0)
    for _ in range(2):
        params, _ = optimize(state, params)

    poisoned = {
        "z one entry": params.replace(z=params.z.at[4, 0].set(jnp.nan)),
        "z all": params.replace(z=jnp.full_like(params.z, jnp.nan)),
        "gamma_o": params.replace(
            gamma_o=params.gamma_o.at[4, 0].set(jnp.nan)
        ),
        "gamma_r": params.replace(
            gamma_r=jnp.full_like(params.gamma_r, jnp.inf)
        ),
        "rho": params.replace(rho=jnp.full_like(params.rho, jnp.nan)),
        "primal_residual": params.replace(
            primal_residual=jnp.asarray(jnp.nan, dtype=jnp.float32)
        ),
    }
    for label, bad in poisoned.items():
        out, _ = optimize(state, bad)
        for name in ("z", "gamma_o", "gamma_r", "rho", "primal_residual"):
            assert jnp.all(jnp.isfinite(getattr(out, name))), (
                f"{label}: {name} is still non-finite after one solve"
            )
        assert jnp.all(jnp.isfinite(out.robot_params.mean)), (
            f"{label}: the commanded mean is non-finite, i.e. still frozen"
        )


def test_align_ref_plan_end_reads_the_plan_endpoint() -> None:
    """`align_ref="plan_end"` measures `_ell_r` against the handed-in
    endpoint, and only then; the goal terms keep aiming at g either way.
    """
    task = PushT(
        clutter=True, planning_dt=PLAN_DT, costs={"align_ref": "plan_end"}
    )
    state = task.make_data().replace(qpos=jnp.zeros(task.mj_model.nq))
    state = mjx_forward(task.model, state)
    u = jnp.zeros(task.model.nu)
    ref = jnp.asarray([0.6, 0.4, 1.0])
    pose = task._block_pose(state)
    pusher = task._pusher_pos(state)
    # Without an endpoint the block falls back to the global goal.
    assert float(task.robot_running_cost(state, u)) == pytest.approx(
        float(task.robot_running_cost(state, u, ref_pose=task.goal))
    )
    # With one, exactly `_ell_r` moves -- by the same amount feeding the
    # endpoint to `_ell_r` by hand does.
    delta_cost = float(task.robot_running_cost(state, u, ref_pose=ref)) - (
        float(task.robot_running_cost(state, u))
    )
    delta_ell_r = float(task._ell_r(state, pose, pusher, ref)) - float(
        task._ell_r(state, pose, pusher, task.goal)
    )
    assert delta_cost == pytest.approx(delta_ell_r, rel=1e-3)
    assert delta_cost != pytest.approx(0.0)
    # The ADMM layer hands the endpoint in only for this key: the carry's
    # EMA slot exists either way and stays NaN before the first round.
    ctrl = _build_admm(task)
    params = ctrl.init_params()
    assert params.ref_ema.shape == (3,)
    assert bool(jnp.all(jnp.isnan(params.ref_ema)))


def test_ref_ema_is_identity_at_alpha_zero_and_blends_otherwise() -> None:
    """`wia_ref_alpha` smooths the endpoint the robot block is handed."""
    task = PushT(
        clutter=True, planning_dt=PLAN_DT,
        costs={"align_ref": "plan_end", "wia_ref_alpha": 0.0},
    )
    ctrl = _build_admm(task, n_admm=1)
    state = mjx_forward(task.model, task.make_data())
    params = ctrl.init_params()
    params, _ = ctrl.optimize(state, params)
    endpoint = np.asarray(ctrl.object_plan_endpoint(state, params))
    assert np.allclose(np.asarray(params.ref_ema), endpoint, atol=1e-5)

    task.wia_ref_alpha = 0.9
    ctrl = _build_admm(task, n_admm=1)
    params = ctrl.init_params()
    params, _ = ctrl.optimize(state, params)  # first round: fresh -> raw
    first = np.asarray(params.ref_ema)
    params, _ = ctrl.optimize(state, params)
    second = np.asarray(params.ref_ema)
    endpoint = np.asarray(ctrl.object_plan_endpoint(state, params))
    # Blended: between the previous EMA and the new endpoint in xy.
    for i in range(2):
        lo, hi = sorted([first[i], endpoint[i]])
        assert lo - 1e-6 <= second[i] <= hi + 1e-6


def test_one_rho_serves_both_blocks() -> None:
    """There is no per-block penalty weight any more.

    `rho_object` used to scale the OBJECT block's penalty independently,
    making this a weighted ADMM that the paper's equal-rho convergence
    argument does not cover. It was removed so the single shared rho is
    structural: passing one is a TypeError, not a quietly different
    algorithm.
    """
    task = _build_task()
    ctrl = _build_admm(task)
    assert not hasattr(ctrl, "rho_object_scale")
    with pytest.raises(TypeError):
        ADMM(
            task,
            ctrl.robot_subproblem.optimizer,
            ctrl.object_subproblem.optimizer,
            ctrl.consensus,
            n_admm=4,
            eps_r=1.0,
            eps_s=1.0,
            proximal_weight=0.05,
            rho_init=1.0,
            rho_object=0.25,
        )


def test_dual_clip_is_per_channel_in_every_space() -> None:
    """`factor * scale[i]`, never a scalar taken from scale[0].

    The scalar form was selectable (`max_dual_per_channel`) and was the
    DEFAULT, which left the small-scale channels effectively unclipped --
    scale[0] is a force, ~25x the torque scale in normalized units, so
    the wrench space's torque dual was never really bounded. All three
    spaces now clip per channel and the flag is gone.
    """
    task = _build_task()
    scale = np.asarray(task.consensus_scale())
    assert scale.shape == (3,)
    space = consensus_space(task, "wrench")
    assert np.allclose(np.asarray(space.max_dual), 2.0 * scale)
    # Per channel, so the torque bound differs from the force bound.
    assert not np.isclose(float(space.max_dual[2]), float(space.max_dual[0]))
    # `factor` reaches every space, not just wrench.
    tight = consensus_space(task, "wrench", max_dual_factor=0.5)
    assert np.allclose(np.asarray(tight.max_dual), 0.5 * scale)
    with pytest.raises(TypeError):
        consensus_space(task, "wrench", max_dual_per_channel=True)
