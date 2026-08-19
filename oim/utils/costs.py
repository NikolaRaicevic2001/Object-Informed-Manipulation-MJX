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


def _goal_ramps(
    task: Any,
    log: Dict[str, Any],
    poses: np.ndarray,
    n: int,
    theta_ramp: bool = True,
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

    `theta`: `PushT._theta_ramp`, on goal_theta only, flat path only
    (`PushT.running_cost`) and so robot-block only. Which path ran is read
    off the log rather than asked of the caller: `primal_residual` exists
    only when `init_log` was given `admm=True`, so it is the one key that
    is structurally tied to the controller rather than to the scene.

    Args:
        task: The task whose weights the ramps are built from.
        log: The run log, for `time` and the ADMM marker.
        poses: Object poses, one per control step.
        n: Number of control steps.
        theta_ramp: Include `_theta_ramp`. False for the object block,
            which has no such term -- `PushT.running_cost` is the robot's.

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

    q_theta_ramp = float(getattr(task, "q_theta_ramp", 1.0))
    fade_dist = float(getattr(task, "theta_ramp_dist", 0.0))
    if theta_ramp and not is_admm and q_theta_ramp > 1.0 and fade_dist > 0.0:
        pos_err = np.linalg.norm(
            poses[:, :2] - np.asarray(task.goal)[:2], axis=1
        )
        closeness = 1.0 - np.clip(pos_err / fade_dist, 0.0, 1.0)
        ramps["theta"] = ramps["theta"] * (
            1.0 + (q_theta_ramp - 1.0) * closeness
        )
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
    # raw quadratic -- see `_goal_ramps`.
    ramps = _goal_ramps(task, log, poses, n)
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
    terms["obstacle"] = _exp(
        obstacles, boundary, obj.w_obstacle, obj.obstacle_decay
    )
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
        # 1 - cos(psi), matching `PushT._tilt`. The log stores the angle
        # because that is the readable unit; the cost is the cosine form.
        # Omitted at zero weight: for `robot="point"` `_tilt` is a
        # constant 2.0, so the bar was a flat 27% of total cost.
        if task.w_tilt != 0.0:
            terms["tilt"] = task.w_tilt * (1.0 - np.cos(tilt))
        # Matches `PushT._tip_height_cost`. Dropped at zero weight: the
        # point robot has no z DOF, so this is identically 0.
        if task.w_z_tip != 0.0 or task.w_z_tip_exp != 0.0:
            gap_cm = 100.0 * (task.tip_target_z - tip_z)
            terms["tip_z"] = np.where(
                tip_z >= task.tip_target_z,
                task.w_z_tip * (tip_z - task.tip_target_z) ** 2,
                task.w_z_tip_exp * np.exp(gap_cm**2),
            )
        # NOT `PushT._contact_z_cost`'s own formula -- that exponential
        # (clip at 0.6 N in real units) is calibrated for the *planning*
        # model's own force scale (~0.1-0.15 N typical), which is the
        # only one the optimizer ever weights. `contact_normal_force_z`
        # here is logged at *execution* fidelity instead (real Newtons,
        # tens to hundreds -- see `PushT._contact_normal_force_z_mujoco`'s
        # docstring), ~2-3 orders of magnitude larger; reusing the same
        # clip/exponential just saturates every real contact event at
        # the ceiling (exp(36) ~ 4.3e15) and swamps every other term on
        # a shared symlog axis, which is what happened before this was
        # changed to a plain quadratic -- unbounded but not explosively
        # so, comparable in scale to `w_obstacle`'s 60000 the panel
        # already accommodates, and actually shows the trajectory's
        # shape instead of a wall of saturated spikes. Absent (0.0
        # series) for run files predating this log key, and 0.0 wherever
        # w_contact_z_exp itself was 0 (the mechanism inert for that
        # run), so the bar reads flat rather than missing for older logs.
        if "contact_normal_force_z" in log and hasattr(task, "w_contact_z_exp"):
            fz = np.asarray(log["contact_normal_force_z"])[:n]
            terms["contact_z"] = task.w_contact_z_exp * fz**2
        # Robot-vs-obstacle contact, from `robot_contact_force`. Skipped
        # when the log key or the weight is absent. The logged force is
        # EXECUTION fidelity, well above the planning value the optimizer
        # weights -- read the bar as scale, not as a replay.
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

    # Must match `PushT.shaping_fade` exactly, or the figure disagrees
    # with what the optimizer paid. No obstacle term fades, nor tilt/tip_z.
    # `admm_penalty` fades too, but against the pose the solve started
    # from, so it is scaled inside `_admm_penalty` rather than listed here.
    _FADED = ("approach", "align", "effort")
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
    # block's do (`ADMM._admm_iteration` hands one `weight_scale` to both),
    # so the bars must show the ramped value. `theta_ramp=False`: that
    # second ramp is `PushT.running_cost`'s, a robot-block term.
    ramps = _goal_ramps(task, log, poses, n, theta_ramp=False)
    terms["goal_pos"] = terms["goal_pos"] * ramps["pos"]
    terms["goal_theta"] = terms["goal_theta"] * ramps["theta"]
    boundary = np.asarray([np.asarray(obj.world_boundary(p)) for p in poses])
    terms["obstacle"] = _exp(
        obj.obstacles, boundary, obj.w_obstacle, obj.obstacle_decay
    )
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
