"""Per-term cost of a *realized* trajectory, for the diagnostics figure.

The planners' own cost functions score candidate rollouts, one scalar per
sample per step, inside `jit`. This scores the single trajectory that was
actually executed, and keeps the terms apart instead of summing them -- so a
run that stalls can be read as "the obstacle hinge dominated from step 40"
rather than just "cost went up".

Two properties make the numbers comparable across algorithms, which is the
whole point of the figure:

* **The reference is always the goal.** ADMM's own `ell_r` measures approach
  and alignment against the object block's nominal `x^o*`, which a flat
  baseline does not have. Scoring both against the goal is the only way a
  bar from an MPPI run and a bar from an ADMM run mean the same thing.
* **Nothing here is on the hot path.** Every term is recomputed from the
  finished log, so switching the figure on cannot perturb what the
  optimizer did. `tests/test_costs.py` pins the decomposition against the
  task's own cost so the two cannot drift.

`coupling` and the ADMM consensus penalty are deliberately absent: both are
defined against quantities that exist only for ADMM, so including them would
put a column on one algorithm's chart and not another's.
"""

from typing import Any, Dict, Optional

import numpy as np

from oim.objects.planar_pushing import wrap_angle

# Draw order, and the order the legend reads. Roughly "task progress first,
# then robot shaping", which is how the terms are usually interrogated.
TERM_ORDER = (
    "goal_pos",
    "goal_theta",
    "obstacle",
    "support",  # keep-IN: the table edge, mirror of `obstacle`
    "rate",
    "approach",
    "align",
    "tilt",
    "tip_z",
    "contact_z",
    "robot_obstacle",  # 2D: the robot clearance hinge
    "robot_contact",  # 3D: robot-obstacle contact force
    "effort",  # robot: w_robot_effort*||u||^2; object: w_effort*||w||^2
    "admm_penalty",  # (rho/2)||A (-) z + y||^2, both blocks
)


def _q_ramp_mult(task: Any, time: np.ndarray) -> np.ndarray:
    """Matches `PushT._q_ramp_mult`: ``min((1 + q_ramp_per_step) **
    steps, q_ramp_max)``, ``steps = time / task.dt``, computed here from
    the logged ``time`` series (real elapsed simulator clock, one entry
    per control step) rather than a live `mjx.Data.time`. Inert (all
    1.0) when either `q_ramp_per_step` or `q_ramp_max` is absent or at
    its default -- `getattr` rather than a plain attribute read, so run
    files/tasks predating this mechanism decompose unchanged.

    Only ever applied to `goal_pos`/`goal_theta` in `_common_terms` (the
    flat baseline's own tracking, `running_cost`'s `ell_o`) -- never in
    `object_cost_series`, whose `_se2_terms` call scores the *object*
    block's own tracking, which `running_cost` does not touch and so this
    ramp never reaches. Also never in `robot_running_cost`'s own tracking
    (ADMM) -- that method reads `q_ramp_per_step`/`q_ramp_max` through a
    different, non-compounding mechanism (`time_ramp`/`weight_scale`);
    see `PushT._q_ramp_mult`'s own docstring.
    """
    per_step = float(getattr(task, "q_ramp_per_step", 0.0) or 0.0)
    q_max = float(getattr(task, "q_ramp_max", 1.0) or 1.0)
    if per_step <= 0.0 or q_max <= 1.0:
        return np.ones_like(time)
    steps = time / task.dt
    return np.minimum((1.0 + per_step) ** steps, q_max)


def _obstacle_cost(
    obj: Any, obstacles: Any, boundary: np.ndarray
) -> np.ndarray:
    """Object-vs-obstacle clearance, matching `obj.obstacle_cost`.

    `obj.w_obstacle * exp(-d / obj.obstacle_decay)`. See
    `PlanarPushingObject.obstacle_cost`.
    """
    return _exp(obstacles, boundary, obj.w_obstacle, obj.obstacle_decay)


def _support_cost(obj: Any, boundary: np.ndarray) -> Optional[np.ndarray]:
    """Object-vs-table-edge, matching `PlanarPushingObject.support_cost`.

    Returns None -- so the bar is absent, not flat -- when the scene has no
    support region (`clutter` has no table) or the weight is zero, and for
    run files predating the term.
    """
    support = getattr(obj, "support", None)
    weight = float(getattr(obj, "w_support", 0.0) or 0.0)
    if support is None or weight == 0.0:
        return None
    margin = float(getattr(obj, "support_margin", 0.0) or 0.0)
    d = np.asarray(support.sdf(boundary))
    return weight * np.sum(np.clip(d + margin, 0.0, None) ** 2, axis=-1)


def _se2_terms(
    poses: np.ndarray, goal: np.ndarray, w_pos: float, w_theta: float
) -> Dict[str, np.ndarray]:
    """`se2_distance_sq` kept split into its translational and angular half."""
    d_pos = poses[:, :2] - np.asarray(goal)[:2]
    d_theta = np.asarray(wrap_angle(poses[:, 2] - float(goal[2])))
    return {
        "goal_pos": w_pos * np.sum(d_pos**2, axis=1),
        "goal_theta": w_theta * d_theta**2,
    }


def _hinge(
    obstacles: Any, points: np.ndarray, weight: float, margin: float
) -> np.ndarray:
    """The clearance hinge over a series of point sets, one value per step."""
    if not obstacles.shapes or weight == 0.0:
        return np.zeros(len(points))
    return np.asarray(
        [float(obstacles.hinge_cost(p, weight, margin)) for p in points]
    )


def _exp(
    obstacles: Any, points: np.ndarray, weight: float, decay: float
) -> np.ndarray:
    """The exponential proximity cost, one value per step."""
    if not obstacles.shapes or weight == 0.0:
        return np.zeros(len(points))
    return np.asarray(
        [float(obstacles.exp_cost(p, weight, decay)) for p in points]
    )


def _approach_and_align(
    poses: np.ndarray,
    robot: np.ndarray,
    goal: np.ndarray,
    w_approach: float,
    r0: float,
    w_align: float,
    gamma0: float,
) -> Dict[str, np.ndarray]:
    """The two shaping terms both worlds share, scored against the goal."""
    d_ee = np.sum((robot - poses[:, :2]) ** 2, axis=1)
    to_object = poses[:, :2] - robot
    to_ref = np.asarray(goal)[:2] - poses[:, :2]
    cos_angle = np.sum(to_object * to_ref, axis=1) / (
        np.linalg.norm(to_object, axis=1) * np.linalg.norm(to_ref, axis=1)
        + 1e-6
    )
    return {
        "approach": w_approach * np.clip(d_ee - r0**2, 0.0, None),
        "align": w_align * np.clip(gamma0 - cos_angle, 0.0, None),
    }


def _exp_arg_max() -> float:
    """`PushT.EXP_ARG_MAX`, so the replay saturates where the planner does.

    Deferred import: `oim.tasks.pusht` imports this module's siblings, so
    a module-level import here is circular.
    """
    from oim.tasks.pusht import EXP_ARG_MAX  # noqa: PLC0415

    return EXP_ARG_MAX


def _goal_ramps(
    task: Any, log: Dict[str, Any], n: int
) -> Dict[str, np.ndarray]:
    """Per-step multipliers the planner applies to the goal terms.

    Neither is a property of the state, so nothing in a run file records
    them -- both are rebuilt here from `time`/`object_pose` and the task's
    own weights. Without them the goal bars are the *unramped* stage cost
    and understate what actually drove the plan: at `q_ramp_per_step: 0.05`
    / `q_ramp_max: 20.0` the planner is weighting goal tracking 20x
    everything else from step 380 on.

    `time`: `PushT.time_ramp`, on BOTH goal terms, ADMM path only
    (`RobotSubproblem._eval_rollouts_one` reads it once per horizon, at the
    state the control was chosen *from* -- entry i, not the i+1 the cost is
    scored at).

    Which path ran is read off the log rather than asked of the caller:
    `primal_residual` exists only when `init_log` was given `admm=True`,
    so it is the one key that is structurally tied to the controller
    rather than to the scene.

    Args:
        task: The task whose weights the ramps are built from.
        log: The run log, for `time` and the ADMM marker.
        n: Number of control steps.

    `q_ramp_mult` (`_q_ramp_mult`, see its own docstring): on BOTH goal
    terms, flat path only. A *different* mechanism from `time`'s
    time_ramp above despite both reading `q_ramp_per_step`/`q_ramp_max`
    -- compounding (``(1+rate)**steps``) vs. `time_ramp`'s linear
    (``1+rate*steps``), and scoped to the opposite controller
    (`PushT.running_cost` only, never `robot_running_cost`).

    Returns:
        `{"pos": (n,), "theta": (n,)}`, ones wherever a ramp is inert.
    """
    ones = np.ones(n)
    ramps = {"pos": ones, "theta": ones}
    is_admm = "primal_residual" in log

    per_step = float(getattr(task, "q_ramp_per_step", 0.0))
    if is_admm and per_step != 0.0 and len(log.get("time", ())) >= n:
        t = np.asarray(log["time"], dtype=float)[:n]
        time_ramp = np.clip(
            1.0 + per_step * (t / float(task.dt)),
            1.0,
            float(getattr(task, "q_ramp_max", 1.0)),
        )
        ramps = {"pos": time_ramp, "theta": time_ramp}
    elif not is_admm and len(log.get("time", ())) >= n + 1:
        # `[1:][:n]`, not `[:n]` -- the flat path's `_q_ramp_mult` reads
        # `state.time` *after* the step (matching `_common_terms`'s own
        # poses/robot alignment above), unlike `time_ramp`'s ADMM
        # convention just above, which reads time *before* the step.
        t = np.asarray(log["time"], dtype=float)[1:][:n]
        q_ramp = _q_ramp_mult(task, t)
        ramps = {"pos": q_ramp, "theta": q_ramp.copy()}

    return ramps


def _consensus_fade(task: Any, log: Dict[str, Any], n: int) -> np.ndarray:
    """`PushT.shaping_fade` at the pose each solve *started* from.

    Entry i, not the i+1 that `_common_terms` fades approach/align at:
    `ADMM._admm_iteration` reads this once off `obj_state0`, the pose the
    horizon begins at, the same way it reads `time_ramp` off `state.time`.
    Approach and align instead fade against the rolled-out pose of the step
    being scored, so the two really are offset by one -- one step of
    difference in a quantity that is 0 or 1 nearly everywhere.

    Ones when the task has no fade (`PushT2D`, or `shaping_fade_dist <= 0`),
    which is the same off-switch the task itself uses.
    """
    fade_dist = float(getattr(task, "shaping_fade_dist", 0.0) or 0.0)
    if fade_dist <= 0.0 or "object_pose" not in log:
        return np.ones(n)
    poses = np.asarray(log["object_pose"], dtype=float)[:n]
    goal = np.asarray(task.goal, dtype=float)
    pos_err = np.linalg.norm(poses[:, :2] - goal[:2], axis=1)
    return np.clip(pos_err / fade_dist, 0.0, 1.0)


def _admm_penalty(
    task: Any,
    log: Dict[str, Any],
    n: int,
    actual_key: str,
    dual_key: str,
) -> Optional[np.ndarray]:
    """`(rho/2)||A (-) z + y||^2` per control step, in normalized units.

    Restates `ConsensusSpace.penalty_cost` in numpy, at horizon step 0 --
    the entry the executed control was scored against. This is the one term
    of either block's cost that is not a function of the trajectory alone,
    so it can only be plotted because the runners log `z`, both duals and
    `rho` alongside the two blocks' extracted values.

    `None` when the run is not ADMM, or predates those keys: absent rather
    than zero, so an old run file plots what it can instead of claiming the
    penalty was nil.

    The logged `rho` is a scalar even where the task uses the paper's
    anisotropic diag(rho_f, rho_f, rho_tau) -- `log_step` records its mean
    -- so a per-dimension rho is reported at its average weight. It is also
    the *base* rho: the near-goal fade is applied here, since what the
    runner records is what `rho_adapt` carries, not what the subproblems
    were charged.
    """
    needed = ("wrench_consensus", "rho", actual_key, dual_key)
    if any(k not in log or len(log[k]) < n for k in needed):
        return None
    actual = np.asarray(log[actual_key], dtype=float)[:n]
    z = np.asarray(log["wrench_consensus"], dtype=float)[:n]
    dual = np.asarray(log[dual_key], dtype=float)[:n]
    rho = np.asarray(log["rho"], dtype=float)[:n]
    diff = actual - z + dual
    if getattr(task, "consensus_variable", "wrench") == "pose":
        diff[:, 2] = np.asarray(wrap_angle(diff[:, 2]))
    scale = np.asarray(task.consensus_scale(), dtype=float)
    penalty = 0.5 * rho * np.sum((diff / scale) ** 2, axis=1)
    return _consensus_fade(task, log, n) * penalty


def _common_terms(
    task: Any, log: Dict[str, Any], obstacles: Any
) -> Dict[str, np.ndarray]:
    """Everything a 2D and a 3D run decompose the same way.

    Series are trimmed to the number of *control steps*: state series carry
    the initial condition and so run one longer than the input series (see
    `oim.utils.results._SCHEMA`). Scoring the state each control produced --
    entries 1.. of the state series against entries 0.. of the inputs --
    is what makes a per-step cost line up with the control that caused it.
    """
    poses = np.asarray(log["object_pose"])[1:]
    robot = np.asarray(log["robot_pos"])[1:]
    controls = np.asarray(log["robot_control"])
    n = min(len(poses), len(controls))
    poses, robot, controls = poses[:n], robot[:n], controls[:n]

    obj = task.object_model
    goal = np.asarray(task.goal)
    terms = _se2_terms(poses, goal, obj.w_pos, obj.w_theta)
    # The goal bars must show what the planner actually weighted, not the
    # raw quadratic -- see `_goal_ramps` (folds in both ADMM's time_ramp
    # and the flat baseline's theta_ramp/q_ramp_mult).
    ramps = _goal_ramps(task, log, n)
    terms["goal_pos"] = terms["goal_pos"] * ramps["pos"]
    terms["goal_theta"] = terms["goal_theta"] * ramps["theta"]
    terms.update(
        _approach_and_align(
            poses,
            robot,
            goal,
            task.w_approach,
            task.r0,
            task.w_align,
            float(task.gamma0),
        )
    )
    boundary = np.asarray(
        [np.asarray(obj.world_boundary(p)) for p in poses]
    )
    terms["obstacle"] = _obstacle_cost(obj, obstacles, boundary)
    support = _support_cost(obj, boundary)
    if support is not None:
        terms["support"] = support
    terms["effort"] = task.w_robot_effort * np.sum(controls**2, axis=1)
    # The ADMM penalty: not a function of the trajectory alone, hence the
    # extra logged keys. Present for both blocks or neither.
    penalty = _admm_penalty(task, log, n, "wrench", "dual_robot")
    if penalty is not None:
        terms["admm_penalty"] = penalty
    return terms


def cost_series(task: Any, log: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-control-step value of every cost term, keyed by name.

    Args:
        task: The `PushT` or `PushT2D` the run was produced with.
        log: A finished run log from `oim.worlds.sim3d.run` or
            `oim.worlds.sim2d.run_2d`.

    Returns:
        An ordered mapping term name -> array of shape (steps_run,). Terms
        that do not apply to this embodiment are absent rather than zero,
        so the legend never advertises a term that cannot fire.
    """
    obstacles = task.object_model.obstacles
    terms = _common_terms(task, log, obstacles)
    n = len(terms["goal_pos"])

    # Which embodiment-specific terms exist is decided by what is actually
    # available, never assumed from the caller: a task without `w_tilt` has
    # no end-effector pose to shape, and one without `w_obstacle_robot`
    # leaves robot clearance to the simulator's own contact. Asking for
    # either unconditionally is an AttributeError deep inside plotting, at
    # the end of a run that has already been paid for.
    has_tip = "tip_tilt" in log and len(log["tip_tilt"]) > 0
    if has_tip and hasattr(task, "w_tilt"):
        tilt = np.asarray(log["tip_tilt"])[:n]
        tip_z = np.asarray(log["tip_z"])[:n]
        goal = np.asarray(task.goal)
        obj_pos = np.asarray(log["object_pose"])[1:][:n, :2]
        pos_err = np.linalg.norm(obj_pos - goal[:2], axis=1)
        # 1 - cos(psi), matching `PushT._tilt`. The log stores the angle
        # because that is the readable unit; the cost is the cosine form.
        # Omitted at zero weight: for `robot="point"` `_tilt` is a
        # constant 2.0, so the bar was a flat 27% of total cost.
        if task.w_tilt != 0.0:
            terms["tilt"] = task.w_tilt * (1.0 - np.cos(tilt))
        # Matches `PushT._tip_height_cost`: the plain quadratic above
        # `tip_target_z`, faded (linearly, same shaping_fade_dist radius
        # as align/approach/tilt -- see the fade block below), and the
        # unfaded exponential below it. Dropped entirely at zero weight:
        # the point robot has no z DOF, so this is identically 0.
        if task.w_z_tip != 0.0 or task.w_z_tip_exp != 0.0:
            quad_center = getattr(task, "tip_quadratic_target_z", task.tip_target_z)
            quad_ref = task.w_z_tip * (tip_z - quad_center) ** 2
            fade_dist_tip = float(getattr(task, "shaping_fade_dist", 0.0) or 0.0)
            if fade_dist_tip > 0.0:
                tip_fade = np.clip(
                    pos_err / fade_dist_tip, 0.0, 1.0
                )
            else:
                tip_fade = 1.0
            above = tip_fade * quad_ref

            gap_cm = 100.0 * (task.tip_target_z - tip_z)
            exp_below = task.w_z_tip_exp * np.exp(
                np.minimum(gap_cm**2, _exp_arg_max())
            )
            terms["tip_z"] = np.where(
                tip_z >= task.tip_target_z, above, exp_below
            )
        # Matches `PushT._contact_z_cost` (2026-08-19: rewritten to a
        # kinematic hover-slab barrier, replacing the old force-based
        # version this comment used to describe). Purely kinematic --
        # tip position and block SE(2) pose only, both logged at the
        # same fidelity the optimizer itself reads -- so unlike the old
        # force-based version there is no planning/execution fidelity
        # gap to approximate around; this reconstructs the exact
        # formula rather than a scaled-down stand-in. Needs the tip's
        # (x, y) and the block's full pose (not just `obj_pos`'s
        # position above), so both are read fresh from the log here.
        # Absent for run files predating `tip_z`/`robot_pos`, or for a
        # task built before `block_half_height` existed, or 0.0
        # wherever w_contact_z_exp itself was 0 (the mechanism inert for
        # that run) -- same convention as the terms above.
        if (
            hasattr(task, "w_contact_z_exp")
            and hasattr(task, "block_half_height")
            and "robot_pos" in log
        ):
            pose_full = np.asarray(log["object_pose"])[1:][:n]
            tip_xy = np.asarray(log["robot_pos"])[1:][:n]
            top_z = task.tip_target_z + task.block_half_height
            slab = float(getattr(task, "contact_z_slab", 0.01))
            dz = np.abs(tip_z - top_z)

            theta = pose_full[:, 2]
            rel = tip_xy - pose_full[:, :2]
            c, s = np.cos(theta), np.sin(theta)
            local_xy = np.stack(
                [c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]],
                axis=-1,
            )
            inside_footprint = (
                np.asarray(task.object_model.footprint.sdf(local_xy)) <= 0.0
            )
            in_slab = inside_footprint & (dz <= slab)

            gap = 1.0 - np.clip(dz / slab, 0.0, 1.0)
            raw = task.w_contact_z_exp * np.exp((2.0 * gap) ** 2)
            terms["contact_z"] = np.where(in_slab, raw, 0.0)
        # Matches `_ell_r`'s suppression of `align` while
        # `PushT._top_contact_gate` reads 1 (added 2026-08-20). Without
        # this the panel drew the *raw* align, so the figure showed align
        # firing at full value on exactly the steps where the real cost had
        # already zeroed it -- making the escape-gate fix look broken from
        # the plot when it was in fact working (caught by Shahid reading
        # the figure, 2026-08-20; on one open_table run the panel showed
        # 5763 cost-units of align across contact_z's active steps where
        # the applied value was exactly 0.0).
        #
        # Computed in its own block rather than inside the `contact_z`
        # branch above because the gate is deliberately independent of
        # `w_contact_z_exp` and of which top-riding experiment is live --
        # it must still be reconstructed when contact_z is inert. Note the
        # band is NOT the same as the slab above: -0.5cm..+5cm, not
        # +/-1cm, mirroring `_top_contact_gate` exactly.
        if (
            "align" in terms
            and getattr(task, "robot", None) == "xarm6"
            and hasattr(task, "block_half_height")
            and "robot_pos" in log
            and "tip_z" in log
        ):
            pose_g = np.asarray(log["object_pose"])[1:][:n]
            tip_xy_g = np.asarray(log["robot_pos"])[1:][:n]
            tip_z_g = np.asarray(log["tip_z"])[:n]
            top_z_g = task.tip_target_z + task.block_half_height
            th_g = pose_g[:, 2]
            rel_g = tip_xy_g - pose_g[:, :2]
            cg, sg = np.cos(th_g), np.sin(th_g)
            local_g = np.stack(
                [
                    cg * rel_g[:, 0] + sg * rel_g[:, 1],
                    -sg * rel_g[:, 0] + cg * rel_g[:, 1],
                ],
                axis=-1,
            )
            inside_g = (
                np.asarray(task.object_model.footprint.sdf(local_g)) <= 0.0
            )
            near_top_g = (tip_z_g >= top_z_g - 0.005) & (
                tip_z_g <= top_z_g + 0.05
            )
            terms["align"] = terms["align"] * (
                1.0 - (inside_g & near_top_g).astype(float)
            )
        # Robot-vs-obstacle contact, both embodiments -- see
        # `PushT._robot_contact_cost`. From `robot_contact_force`, so this
        # is EXECUTION fidelity, well above the planning value the
        # optimizer weights: read the bar as scale, not as a replay.
        # Never faded.
        if (
            "robot_contact_force" in log
            and getattr(task, "w_robot_contact", 0.0) != 0.0
        ):
            f_rob = np.asarray(log["robot_contact_force"])[:n]
            terms["robot_contact"] = task.w_robot_contact * f_rob**2
    elif hasattr(task, "w_obstacle_robot"):
        robot = np.asarray(log["robot_pos"])[1:][:n]
        terms["robot_obstacle"] = _hinge(
            obstacles,
            robot[:, None, :],
            task.w_obstacle_robot,
            task.obstacle_margin,
        )

    # Matches `PushT._ell_r`/`running_cost`: approach, align, tilt and
    # effort all fade linearly (tip_height's above-threshold branch is
    # faded the same way too, but already handled above, inline with the
    # rest of that term's formula). tip_height's below-threshold branch,
    # contact_z, and the robot-obstacle term (either
    # embodiment's) are never faded -- real safety guarantees or, for the
    # last one, a deliberate choice mirrored from `shaping_fade`.
    # `shaping_fade_dist` is a PushT (3D)-only config key, so this whole
    # block is a no-op for PushT2D by construction (fade_dist stays 0).
    # `admm_penalty` fades too, but against the pose the solve started
    # from, so it is scaled inside `_admm_penalty` rather than listed here.
    _FADED = ("approach", "align", "tilt", "effort")
    fade_dist = float(getattr(task, "shaping_fade_dist", 0.0) or 0.0)
    if fade_dist > 0.0:
        poses = np.asarray(log["object_pose"])[1:][:n]
        goal = np.asarray(task.goal)
        pos_err = np.linalg.norm(poses[:, :2] - goal[:2], axis=1)
        fade = np.clip(pos_err / fade_dist, 0.0, 1.0)
        for key in _FADED:
            if key in terms:
                terms[key] = terms[key] * fade

    return {k: terms[k] for k in TERM_ORDER if k in terms}


def object_cost_series(task: Any, log: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-step value of the *object* block's own cost terms.

    The counterpart of `cost_series` for a run with no robot in it (see
    `oim.worlds.object_only.build`). `cost_series` decomposes the robot
    block's cost --
    approach, align, tilt, effort on `robot_control` -- none of which
    exists here; this decomposes `PlanarPushingObject.running_cost`
    instead, which is goal tracking, the clearance hinge on the object's
    own footprint, and effort on the *wrench* rather than on a control.

    Kept a separate function rather than a branch inside `cost_series`
    because the two decompose genuinely different cost functions, and a
    silent switch on "is `robot_pos` in the log" would make a mislogged
    run quietly report the wrong breakdown.

    Args:
        task: The `PushT` the run was produced with, for `object_model`.
        log: A finished log from `oim.worlds.object_only.build.run_object`.

    Returns:
        Term name -> array of shape (steps_run,), in `TERM_ORDER`.
    """
    obj = task.object_model
    # Entries 1.. of the state series against entries 0.. of the wrench
    # series, the same alignment `_common_terms` uses: score the state each
    # decision produced, not the one it was made from.
    poses = np.asarray(log["object_pose"])[1:]
    wrenches = np.asarray(log["wrench"])
    n = min(len(poses), len(wrenches))
    poses, wrenches = poses[:n], wrenches[:n]

    terms = _se2_terms(poses, np.asarray(task.goal), obj.w_pos, obj.w_theta)
    # The object block's goal terms carry the same time ramp the robot
    # block's do (`ADMM._admm_iteration` hands one `weight_scale` to
    # both), so the bars must show the ramped value.
    ramps = _goal_ramps(task, log, n)
    terms["goal_pos"] = terms["goal_pos"] * ramps["pos"]
    terms["goal_theta"] = terms["goal_theta"] * ramps["theta"]
    boundary = np.asarray([np.asarray(obj.world_boundary(p)) for p in poses])
    terms["obstacle"] = _obstacle_cost(obj, obj.obstacles, boundary)
    support = _support_cost(obj, boundary)
    if support is not None:
        terms["support"] = support
    # Per *executed* step, so this is the realized jitter rather than the
    # within-horizon term the planner scored. Leading zero: nothing
    # precedes the first wrench.
    normalized = wrenches / np.asarray(obj.wrench_limit)
    steps = np.sum(
        np.asarray(obj.w_rate) * np.diff(normalized, axis=0) ** 2, axis=1
    )
    terms["rate"] = np.concatenate([[0.0], steps])
    # Same key as the robot block's control effort, a different quantity:
    # `w_effort*||w||^2` on the wrench. The two never share a panel.
    terms["effort"] = obj.w_effort * np.sum(wrenches**2, axis=1)
    # The object block's half of the same penalty the robot block pays --
    # same z and rho, its own A^o and dual.
    penalty = _admm_penalty(task, log, n, "object_consensus", "dual_object")
    if penalty is not None:
        terms["admm_penalty"] = penalty
    return {k: terms[k] for k in TERM_ORDER if k in terms}


def cost_totals(series: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Accumulated value of each term over the whole run, plus `total`."""
    totals = {k: float(np.sum(v)) for k, v in series.items()}
    totals["total"] = float(sum(totals.values()))
    return totals


def summarize(
    task: Any, log: Dict[str, Any]
) -> Optional[Dict[str, np.ndarray]]:
    """`cost_series`, or None if the log lacks what it needs.

    Used by the plotting path, which must degrade to the old two-panel
    figure rather than crash on a log shape it was not expecting.
    """
    required = ("object_pose", "robot_pos", "robot_control")
    if any(k not in log or len(log[k]) == 0 for k in required):
        return None
    series = cost_series(task, log)
    return series if series and len(next(iter(series.values()))) else None
