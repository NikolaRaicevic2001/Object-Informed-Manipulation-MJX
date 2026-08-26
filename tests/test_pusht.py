import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from oim.tasks.pusht import EXP_ARG_MAX, PushT


@pytest.mark.parametrize("impl", ["jax", "warp"])
@pytest.mark.parametrize("clutter", [False, True])
def test_task(impl: str, clutter: bool) -> None:
    """Set up the push T task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
        clutter: Whether to set up the full ConsensusTask/cost machinery
            (`self.goal`, `w_approach`, `w_align`, ...) -- despite the name, this
            does not pick a different scene (no `env=`/`scene=` is passed
            either way); every real run in this codebase passes
            `clutter=True` regardless of which scene it's for (see
            `oim/worlds/sim3d/build.py`). `False` is a bare construction
            that can
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

    Two ways a scene can carry the object's support friction, and the
    analytic limit-surface limit `mu*m*g` has to match whichever it uses,
    or the two ADMM blocks are silently modelling different objects:

    * `frictionloss` on the block's joints -- a constant force bound. What
      `clutter` (this test's scene) and the real-table scenes use.
    * the table CONTACT, `mu*N` -- what the tabletop scenes switched to, so
      that pressing down on the block raises its friction. Their joints
      carry no frictionloss and this check would read zero; they are
      covered by `test_scenes.py` against their measured breakaway
      instead.
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


def test_xarm6_flat_terminal_uses_qf_and_shaping() -> None:
    """Flat MPPI terminal = heavier goal (qf_*) + the same ℓ_r as running.

    Goal-only terminals let MPPI abandon align/tilt at the horizon end;
    stage costs alone are too weak there because they are multiplied by dt.
    """
    from oim.objects import se2_distance_sq

    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={
            "q_pos": 10.0,
            "q_theta": 10.0,
            "qf_pos": 1000.0,
            "qf_theta": 1000.0,
            # Isolated from `_tip_height_cost`'s exponential branch:
            # `make_data()`'s default (all-zero) qpos puts the tip well
            # below the block's mid-height, so with the default weight
            # that branch dominates at ~1e21 and swallows the qf_*
            # difference this test is actually about at float32
            # precision. Not what this test checks.
            "w_z_tip_exp": 0.0,
        },
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    pose = task._block_pose(state)
    pusher = task._pusher_pos(state)

    ell_f = se2_distance_sq(pose, task.goal, task.qf_pos, task.qf_theta)
    ell_r = task._ell_r(state, pose, pusher, task.goal)
    assert jnp.allclose(task.terminal_cost(state), ell_f + ell_r)

    # Heavier than a running-cost clone would be (same ℓ_r, lighter q_*).
    running_clone = se2_distance_sq(
        pose, task.goal, task.q_pos, task.q_theta
    ) + ell_r
    assert float(task.terminal_cost(state)) > float(running_clone)


def test_xarm6_shaping_fade_scales_with_goal_distance() -> None:
    """align fades to 0 as ||p-p_g|| → 0; approach/tilt/tip_z are not faded.

    Only `align` is gated by `shaping_fade` -- tilt and tip height used to
    fade too, but that let the tip sink into the table near the goal, so
    both are now always fully active regardless of distance to goal.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        # Isolate the fade check from tilt/tip_z, which are intentionally
        # unfaded now and would otherwise contribute whatever nonzero
        # value the (unset-up) arm pose happens to produce.
        costs={
            "shaping_fade_dist": 0.20,
            "w_approach": 0.0,
            "w_tilt": 0.0,
            "w_z_tip": 0.0,
            "w_z_tip_exp": 0.0,
        },
    )
    assert float(task.shaping_fade(task.goal)) == pytest.approx(0.0)
    far = task.goal + jnp.array([0.40, 0.0, 0.0])
    assert float(task.shaping_fade(far)) == pytest.approx(1.0)
    mid = task.goal + jnp.array([0.10, 0.0, 0.0])
    assert float(task.shaping_fade(mid)) == pytest.approx(0.5)

    state = jax.jit(mjx.forward)(task.model, task.make_data())
    qpos = state.qpos.at[task.block_qpos_adr].set(task.goal)
    state = jax.jit(mjx.forward)(task.model, state.replace(qpos=qpos))
    pose = task._block_pose(state)
    # With tilt/tip_z zeroed out above, ell_r = approach + fade * align;
    # approach is 0 (w_approach=0), so this isolates align -> 0 at the goal.
    ell_r = task._ell_r(state, pose, task._pusher_pos(state), task.goal)
    assert float(ell_r) == pytest.approx(0.0, abs=1e-5)


def test_xarm6_shaping_fade_is_linear_down_to_the_goal() -> None:
    """1 at/beyond shaping_fade_dist, linear in, exactly 0 at the goal."""
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={"shaping_fade_dist": 0.20},
    )
    far = task.goal + jnp.array([0.20, 0.0, 0.0])
    assert float(task.shaping_fade(far)) == pytest.approx(1.0)
    beyond = task.goal + jnp.array([0.40, 0.0, 0.0])
    assert float(task.shaping_fade(beyond)) == pytest.approx(1.0)
    midpoint = task.goal + jnp.array([0.10, 0.0, 0.0])
    assert float(task.shaping_fade(midpoint)) == pytest.approx(0.5)
    # abs=1e-6: float32 arithmetic on task.goal + an offset lands within
    # ~1e-7 of exact, not bit-exact -- pytest.approx(0.0)'s default
    # tolerance (1e-12) is far tighter than float32 itself.
    assert float(task.shaping_fade(task.goal)) == pytest.approx(0.0, abs=1e-6)


def test_xarm6_tip_height_is_not_faded_but_tilt_now_is() -> None:
    """tip_height stays fully active at the goal; align and tilt both fade.

    Regression test for the table-strike freeze `shaping_fade` originally
    fixed: tip_height must never be scaled down near the goal, or nothing
    holds the tip at height and it can sink into the table. Tilt is a
    quadratic shaping cost, not a hard safety veto like tip_height, so
    fading it alongside align near the goal is safe.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={"shaping_fade_dist": 0.20, "w_approach": 0.0, "w_align": 0.0},
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    qpos = state.qpos.at[task.block_qpos_adr].set(task.goal)
    state = jax.jit(mjx.forward)(task.model, state.replace(qpos=qpos))
    pose = task._block_pose(state)
    pusher = task._pusher_pos(state)

    # This default pose's real z_tip is below tip_target_z, so tip_height
    # would otherwise be in its exponential branch (~1e21) and swamp
    # everything else in a floating-point comparison, including tilt --
    # moving the target below the real z_tip forces the small quadratic
    # branch instead, same trick `test_xarm6_tip_height_cost_is_piecewise`
    # uses, so tilt's contribution is actually distinguishable below.
    task.tip_target_z = float(state.site_xpos[task.trace_site_ids[0], 2]) - 0.01
    task.tip_quadratic_target_z = task.tip_target_z

    # align=0 (weight zeroed), approach=0 (w_approach=0), and tilt is now
    # faded to 0 at the goal too -- whatever is left is exactly tip_height
    # (plus contact_z, which reads 0 at this pose).
    ell_r_at_goal = task._ell_r(state, pose, pusher, task.goal)
    # pose was set to task.goal above, so pos_err is exactly 0 here --
    # matches what _ell_r computes internally.
    tip_height = task._tip_height_cost(state, jnp.asarray(0.0))
    contact_z = task._contact_z_cost(state, pose)
    assert jnp.allclose(ell_r_at_goal, tip_height + contact_z)

    # And explicitly: tilt itself is nonzero here, but fully faded out of
    # ell_r at the goal -- the thing this test is actually pinning.
    tilt = task.w_tilt * task._tilt(state)
    assert tilt > 0.0
    assert not jnp.allclose(ell_r_at_goal, tilt + tip_height)


def test_xarm6_tip_height_cost_is_piecewise() -> None:
    """Quadratic at/above the block's mid-height, exponential (cm) below.

    `make_data()`'s default (all-zero) qpos happens to put the real tip
    below `tip_target_z` -- used directly to exercise the exponential
    branch; `tip_target_z` is then overridden below the real z_tip to
    exercise the quadratic branch instead, without needing a second
    hand-built arm pose.

    The quadratic branch centers on `tip_quadratic_target_z`, not
    necessarily `tip_target_z` (see that attribute's docstring) -- moved
    together here since the default has them equal and this test isn't
    exercising their independence, only that the piecewise split itself
    (branch boundary vs. quadratic center, whichever `tip_target_z` is
    at) still works.
    """
    task = PushT(clutter=True, planning_dt=0.05, robot="xarm6")
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    z_tip = state.site_xpos[task.trace_site_ids[0], 2]
    assert z_tip < task.tip_target_z  # below mid-height at this pose

    # Below-mid-height branch: exponential in centimeters, never faded,
    # so pos_err's value here is irrelevant -- passed anyway for a
    # realistic signature.
    gap_cm = 100.0 * (task.tip_target_z - z_tip)
    # Saturated at `EXP_ARG_MAX`, as the cost itself is -- the tip is far
    # enough below mid-height at this pose that the raw exponent exceeds it.
    expected_below = task.w_z_tip_exp * jnp.exp(
        jnp.minimum(gap_cm**2, EXP_ARG_MAX)
    )
    pos_err = jnp.asarray(0.3)  # arbitrary; the branch is never faded
    assert jnp.allclose(
        task._tip_height_cost(state, pos_err), expected_below
    )

    # At/above-mid-height branch: ordinary quadratic. Force it by moving
    # the target (and the quadratic center, kept equal to it here) below
    # the real z_tip.
    task.tip_target_z = float(z_tip) - 0.01
    task.tip_quadratic_target_z = task.tip_target_z
    expected_above = task.w_z_tip * (
        100.0 * (z_tip - task.tip_quadratic_target_z)
    ) ** 2  # cm^2
    assert jnp.allclose(
        task._tip_height_cost(state, pos_err), expected_above
    )


def test_xarm6_tip_height_above_threshold_fades_linearly() -> None:
    """The true above-threshold branch fades (linearly, same
    shaping_fade_dist radius as align/approach/tilt). The
    below-threshold softening blend target must stay the plain, unfaded
    quadratic regardless -- see `_tip_height_cost`'s safety note.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={"shaping_fade_dist": 0.2},
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    z_tip = float(state.site_xpos[task.trace_site_ids[0], 2])
    # Force the above-threshold branch: target 5cm below the real z_tip.
    task.tip_target_z = z_tip - 0.05
    task.tip_quadratic_target_z = task.tip_target_z
    quad_ref = task.w_z_tip * (
        100.0 * (z_tip - task.tip_quadratic_target_z)
    ) ** 2  # cm^2

    # At/beyond shaping_fade_dist: full, unfaded quadratic.
    assert jnp.allclose(
        task._tip_height_cost(state, jnp.asarray(0.2)), quad_ref
    )
    assert jnp.allclose(
        task._tip_height_cost(state, jnp.asarray(1.0)), quad_ref
    )
    # At the goal (pos_err=0): faded to exactly 0.
    assert jnp.allclose(
        task._tip_height_cost(state, jnp.asarray(0.0)), 0.0, atol=1e-6
    )
    # Halfway: exact linear fade.
    assert jnp.allclose(
        task._tip_height_cost(state, jnp.asarray(0.1)), 0.5 * quad_ref
    )

    # Whatever this fade does to the above branch, the below-threshold
    # exponential must stay unfaded -- the safety property
    # `_tip_height_cost`'s docstring describes. Force that branch and
    # confirm it is full-strength at the goal, not 0.
    task.tip_target_z = z_tip + 0.05  # now below-threshold
    task.tip_quadratic_target_z = task.tip_target_z
    gap_cm = 100.0 * (task.tip_target_z - z_tip)
    exp_below = task.w_z_tip_exp * jnp.exp(
        jnp.minimum(gap_cm**2, EXP_ARG_MAX)
    )
    assert jnp.allclose(
        task._tip_height_cost(state, jnp.asarray(0.0)), exp_below
    )


def test_xarm6_approach_fades_linearly() -> None:
    """approach fades the same way align always has -- previously exempt
    entirely.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={
            "shaping_fade_dist": 0.2,
            "w_align": 0.0,
            "w_tilt": 0.0,
            "w_z_tip": 0.0,
            "w_z_tip_exp": 0.0,
            "r0": 0.0,
        },
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    qpos = state.qpos.at[task.block_qpos_adr].set(task.goal)
    state = jax.jit(mjx.forward)(task.model, state.replace(qpos=qpos))
    pose = task._block_pose(state)
    pusher = task._pusher_pos(state)
    d_ee = float(jnp.sum((pusher - pose[:2]) ** 2))
    raw_approach = task.w_approach * max(d_ee, 0.0)  # r0=0, so clip is a no-op
    assert raw_approach > 0.0

    # At the goal: approach faded to exactly 0. Every
    # other _ell_r term is zeroed above (w_contact_z_exp is already 0.0
    # by default), so this isolates approach.
    ell_r_at_goal = task._ell_r(state, pose, pusher, task.goal)
    assert jnp.allclose(ell_r_at_goal, 0.0, atol=1e-5)


def test_xarm6_q_ramp_mult_grows_and_caps() -> None:
    """`_q_ramp_mult`: 1.0 at step 0, growing LINEARLY by
    `q_ramp_per_step` per real control step (`state.time / self.dt`),
    capped at `q_ramp_max`. Inert (always 1.0) when either is left at its
    default.

    Linear, not compounding: this used to be `(1 + per_step) ** steps`
    while the ADMM track read the same two keys through `time_ramp`'s
    linear law, so the two disagreed on how fast the ramp opened. Pinned
    here against `time_ramp` in
    `test_flat_and_admm_ramps_agree`.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={"q_ramp_per_step": 0.002, "q_ramp_max": 4.0},
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())

    def _mult_at_step(n: int) -> float:
        s = state.replace(time=jnp.asarray(n * task.dt))
        return float(task._q_ramp_mult(s))

    assert _mult_at_step(0) == pytest.approx(1.0)
    assert _mult_at_step(1) == pytest.approx(1.002)
    assert _mult_at_step(10) == pytest.approx(1.02)
    # Linear, so 10 steps is 1 + 0.002*10, NOT (1.002)**10 -- the two
    # differ by 9e-5 here, which is exactly the kind of gap that let the
    # old compounding law pass a test written for the linear one.
    assert _mult_at_step(10) != pytest.approx(1.002**10, abs=1e-9)
    # The cap binds at (4.0 - 1) / 0.002 = 1500 steps.
    assert _mult_at_step(1499) == pytest.approx(3.998)
    assert _mult_at_step(2000) == pytest.approx(4.0)

    # Inert when q_ramp_per_step is left at its default (0.0), regardless
    # of q_ramp_max, and vice versa.
    task.q_ramp_per_step, task.q_ramp_max = 0.0, 4.0
    assert _mult_at_step(500) == pytest.approx(1.0)
    task.q_ramp_per_step, task.q_ramp_max = 0.002, 1.0
    assert _mult_at_step(500) == pytest.approx(1.0)


def test_flat_and_admm_ramps_agree() -> None:
    """`_q_ramp_mult` (flat) and `time_ramp` (ADMM) are the same curve.

    They read the same two config keys on disjoint call paths, and used
    to disagree: the flat one compounded and hit `q_ramp_max` in 646
    steps where the ADMM one, linear, needed 4800. A task built for one
    path and driven through the other silently picked up the other
    formula's ramp; this pins them together so they cannot drift again.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={"q_ramp_per_step": 0.005, "q_ramp_max": 25.0},
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    for n in (0, 1, 10, 646, 1000, 4800, 6000):
        t = jnp.asarray(n * task.dt)
        flat = float(task._q_ramp_mult(state.replace(time=t)))
        admm = float(task.time_ramp(t))
        assert flat == pytest.approx(admm), f"step {n}: {flat} != {admm}"
    # And the cap is reached where the linear law says, not sooner.
    cap_step = (25.0 - 1.0) / 0.005
    assert float(task.time_ramp(jnp.asarray(cap_step * task.dt))) == (
        pytest.approx(25.0)
    )
    assert float(task.time_ramp(jnp.asarray((cap_step - 1) * task.dt))) < 25.0


def test_xarm6_running_cost_q_ramp_scales_goal_tracking() -> None:
    """`running_cost`'s `q_pos`/`q_theta` scale with `_q_ramp_mult`.

    Flat baseline only -- `robot_running_cost` (ADMM's robot block)
    reaches the same formula through `time_ramp`/`weight_scale`, which
    `test_flat_and_admm_ramps_agree` pins against this one; see
    `PushT._q_ramp_mult`'s own docstring.

    Isolated via zeroed weights elsewhere, matching the pattern
    `test_xarm6_approach_fades_linearly` uses.
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        costs={
            "q_ramp_per_step": 0.002,
            "q_ramp_max": 4.0,
            "w_approach": 0.0,
            "w_align": 0.0,
            "w_tilt": 0.0,
            "w_z_tip": 0.0,
            "w_z_tip_exp": 0.0,
            "w_obstacle": 0.0,
        },
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    control = jnp.zeros(task.model.nu)

    cost_step0 = float(task.running_cost(state, control))
    assert cost_step0 > 0.0  # object starts away from the goal

    state_step10 = state.replace(time=jnp.asarray(10 * task.dt))
    cost_step10 = float(task.running_cost(state_step10, control))
    # Linear: 1 + 0.002*10. The compounding law this replaced gives
    # 1.002**10 = 1.02018, which is within 2e-4 of it -- close enough that
    # `rel=1e-5` is what actually separates them.
    expected_mult = 1.0 + 0.002 * 10
    assert cost_step10 == pytest.approx(cost_step0 * expected_mult, rel=1e-5)


def test_xarm6_effort_fades_near_goal() -> None:
    """`running_cost`'s control-effort term fades the same way align
    does (linearly), reaching exactly 0 at the goal.

    Isolated from every other term via zeroed weights, on open_table
    (no obstacles beyond the always-present robot-base circle, which
    the pusher isn't near at either test pose).
    """
    task = PushT(
        clutter=True,
        planning_dt=0.05,
        robot="xarm6",
        env="open_table",
        costs={
            "shaping_fade_dist": 0.2,
            "q_pos": 0.0,
            "q_theta": 0.0,
            "w_approach": 0.0,
            "w_align": 0.0,
            "w_tilt": 0.0,
            "w_z_tip": 0.0,
            "w_z_tip_exp": 0.0,
            "w_contact_z_exp": 0.0,
            "w_obstacle": 0.0,
        },
    )
    state = jax.jit(mjx.forward)(task.model, task.make_data())
    qpos = state.qpos.at[task.block_qpos_adr].set(task.goal)
    state = jax.jit(mjx.forward)(task.model, state.replace(qpos=qpos))
    control = jnp.ones(task.model.nu)

    raw_effort = task.w_robot_effort * jnp.sum(control**2)
    assert raw_effort > 0.0

    # At the goal: everything else zeroed out, effort faded to 0 (fade=0,
    # -> total running_cost is exactly 0.
    assert jnp.allclose(task.running_cost(state, control), 0.0, atol=1e-5)

    # Displaced along the table's LONG axis, not across it: the lab table is
    # only 0.763 m deep, so +0.4 m in x puts the block over the far edge and
    # `w_support` -- which this test does not zero, and cannot, since it is
    # the keep-ON-the-table term -- swamps the effort term being measured.
    # 0.4 m in y is the same distance with 0.36 m of table still to spare.
    far_qpos = state.qpos.at[task.block_qpos_adr].set(
        task.goal + jnp.array([0.0, 0.4, 0.0])
    )
    far_state = jax.jit(mjx.forward)(task.model, state.replace(qpos=far_qpos))
    # Far away (fade=1): full, unfaded effort, and nothing else.
    assert jnp.allclose(task.running_cost(far_state, control), raw_effort)


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


def test_contact_rate_is_weighted_in_its_own_units() -> None:
    """`w_rate` and `w_contact_rate` weight different physical quantities.

    Under `wrench` the channels are [f_x, f_y, tau], normalized by the
    friction-cone limit; under `contact_point` they are [p_x, p_y, lambda],
    normalized by [r_body, r_body, f_max]. Reusing one key would silently
    apply a force weight to a position, which is how the contact ends up
    able to teleport across the object between horizon steps for a price
    the object block is happy to pay.
    """
    costs = {"w_rate": [2.0, 2.0, 1.0], "w_contact_rate": [16.0, 16.0, 1.0]}
    wrench = PushT(clutter=True, robot="point", costs=costs)
    contact = PushT(
        clutter=True,
        robot="point",
        consensus="contact_point",
        costs=costs,
    )

    # Each mode reads its own key, and neither reads the other's.
    assert jnp.allclose(wrench.object_model.w_rate, jnp.array([2.0, 2.0, 1.0]))
    assert jnp.allclose(contact._w_contact_rate, jnp.array([16.0, 16.0, 1.0]))

    # A full-scale step in channel 0 costs exactly the weight, in both --
    # which is what "normalized by the channel's own scale" has to mean.
    def step_of(task, size):
        return float(
            task.object_rate_cost(jnp.zeros((2, 3)).at[1, 0].set(size))
        )

    assert step_of(wrench, float(wrench.consensus_scale()[0])) == pytest.approx(
        2.0, rel=1e-5
    )
    assert step_of(
        contact, float(contact.consensus_scale()[0])
    ) == pytest.approx(16.0, rel=1e-5)

    # Quadratic, so the price of relocating is strongly superlinear: a tenth
    # of the body radius must cost far less than a tenth of the full hop.
    r = float(contact.consensus_scale()[0])
    assert step_of(contact, 0.1 * r) < 0.05 * step_of(contact, r)


def test_contact_rate_is_inert_under_wrench_consensus() -> None:
    """Setting it must not change a wrench run -- the two modes stay apart."""
    base = PushT(clutter=True, robot="point", costs={"w_rate": [2.0, 2.0, 1.0]})
    with_key = PushT(
        clutter=True,
        robot="point",
        costs={"w_rate": [2.0, 2.0, 1.0], "w_contact_rate": [99.0, 99.0, 99.0]},
    )
    seq = jnp.array([[1.0, 0.0, 0.0], [4.0, -2.0, 0.1]])
    assert float(base.object_rate_cost(seq)) == pytest.approx(
        float(with_key.object_rate_cost(seq)), rel=1e-6
    )
