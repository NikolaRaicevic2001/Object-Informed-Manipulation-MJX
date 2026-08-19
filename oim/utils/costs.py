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
    "joint3_cave",
    "robot_obstacle",  # 2D: the robot clearance hinge
    "robot_contact",  # 3D point robot: robot-obstacle contact force
    "effort",
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
    """Object-vs-obstacle clearance, matching `obj.obstacle_cost`'s mode.

    "hinge": `_hinge`, weighted by `obj.w_obstacle`/`obj.obstacle_margin`.
    "exp" (both robots as of 2026-08-19, also the default for any task
    predating the mode split): `_exp`, `obj.w_obstacle`/
    `obj.obstacle_decay`. See `PlanarPushingObject.obstacle_cost`.
    """
    if getattr(obj, "obstacle_cost_mode", "exp") == "hinge":
        return _hinge(obstacles, boundary, obj.w_obstacle, obj.obstacle_margin)
    return _exp(obstacles, boundary, obj.w_obstacle, obj.obstacle_decay)


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
    w_ee: float,
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
        "approach": w_ee * np.clip(d_ee - r0**2, 0.0, None),
        "align": w_align * np.clip(gamma0 - cos_angle, 0.0, None),
    }


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
    # goal_pos/goal_theta stand in for task.q_pos/task.q_theta here (see
    # this function's own module-level design note: obj.w_pos/w_theta are
    # shared, by convention, with the robot block's own weights, so the
    # diagnostic reads the same for a flat MPPI run and an ADMM run).
    # `running_cost` ramps q_pos/q_theta with real elapsed time; this
    # reproduces that here, against the logged time series, so the plot
    # reflects the same growing weight the optimizer actually used. "time"
    # absent (run files/synthetic logs predating it in the schema) leaves
    # goal_pos/goal_theta unramped, same "absent beats zero" convention as
    # every other optional key in this module.
    if "time" in log:
        time = np.asarray(log["time"])[1:][:n]
        q_ramp = _q_ramp_mult(task, time)
        terms["goal_pos"] = terms["goal_pos"] * q_ramp
        terms["goal_theta"] = terms["goal_theta"] * q_ramp
    terms.update(
        _approach_and_align(
            poses,
            robot,
            goal,
            task.w_ee,
            task.r0,
            task.w_align,
            float(task.gamma0),
        )
    )
    boundary = np.asarray(
        [np.asarray(obj.world_boundary(p)) for p in poses]
    )
    terms["obstacle"] = _obstacle_cost(obj, obstacles, boundary)
    terms["effort"] = task.r_r * np.sum(controls**2, axis=1)
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
        floor = float(getattr(task, "fade_floor", 0.0) or 0.0)
        goal = np.asarray(task.goal)
        obj_pos = np.asarray(log["object_pose"])[1:][:n, :2]
        pos_err = np.linalg.norm(obj_pos - goal[:2], axis=1)
        # 1 - cos(psi), matching `PushT._tilt`. The log stores the angle
        # because that is the readable unit; the cost is the cosine form.
        # Omitted at zero weight: for `robot="point"` `_tilt` is a
        # constant 2.0, so the bar was a flat 27% of total cost.
        if task.w_tilt != 0.0:
            terms["tilt"] = task.w_tilt * (1.0 - np.cos(tilt))
        # Matches `PushT._tip_height_cost`. `quad_ref` is the plain
        # quadratic -- always the below-threshold softening blend target,
        # and the above-threshold cost too, just faded there (linearly,
        # same shaping_fade_dist/fade_floor radius as align/approach/tilt
        # -- see the fade block below) rather than left unfaded. Dropped
        # entirely at zero weight: the point robot has no z DOF, so this
        # is identically 0.
        if task.w_z_tip != 0.0 or task.w_z_tip_exp != 0.0:
            quad_center = getattr(task, "tip_quadratic_target_z", task.tip_target_z)
            quad_ref = task.w_z_tip * (tip_z - quad_center) ** 2
            fade_dist_tip = float(getattr(task, "shaping_fade_dist", 0.0) or 0.0)
            if fade_dist_tip > 0.0:
                tip_fade = np.clip(
                    (pos_err - floor) / (fade_dist_tip - floor), 0.0, 1.0
                )
            else:
                tip_fade = 1.0
            above = tip_fade * quad_ref

            gap_cm = 100.0 * (task.tip_target_z - tip_z)
            exp_below = task.w_z_tip_exp * np.exp(gap_cm**2)
            softening = getattr(task, "tip_softening_dist", 0.0)
            if softening > 0.0:
                soften = np.clip(
                    (pos_err - floor) / (softening - floor), 0.0, 1.0
                )
            else:
                soften = 1.0
            below = soften * exp_below + (1.0 - soften) * quad_ref
            terms["tip_z"] = np.where(tip_z >= task.tip_target_z, above, below)
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
        # Matches `PushT._joint3_cave_cost`: zero at/above the threshold,
        # exponential (cm) below it. Absent for run files predating this
        # log key and 0.0 for `robot="point"` (see `log_step`'s own
        # no-op convention for `joint3_z`), so the bar reads flat rather
        # than missing for those logs.
        if "joint3_z" in log and hasattr(task, "w_joint3_cave_exp"):
            z3 = np.asarray(log["joint3_z"])[:n]
            gap_cm = 100.0 * (task.joint3_cave_z_threshold - z3)
            terms["joint3_cave"] = np.where(
                z3 >= task.joint3_cave_z_threshold,
                0.0,
                task.w_joint3_cave_exp * np.exp(gap_cm**2),
            )
        # Robot-vs-obstacle avoidance, robot-conditional -- see
        # `PushT._robot_obstacle_cost`. xarm6: the pusher-vs-obstacle
        # hinge, from `robot_pos` (planning fidelity, the same quantity
        # the optimizer itself reads). Point: `robot_contact`, from
        # `robot_contact_force` (EXECUTION fidelity, well above the
        # planning value the optimizer weights -- read that bar as scale,
        # not as a replay). Never faded, either way.
        if getattr(task, "robot", None) == "xarm6":
            if getattr(task, "pusher_obstacle_weight", 0.0) != 0.0:
                pusher = np.asarray(log["robot_pos"])[1:][:n]
                terms["robot_obstacle"] = _hinge(
                    obstacles,
                    pusher[:, None, :],
                    task.pusher_obstacle_weight * task.object_model.w_obstacle,
                    task.pusher_obstacle_margin,
                )
        elif (
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
    # contact_z, joint3_cave, and the robot-obstacle term (either
    # embodiment's) are never faded -- real safety guarantees or, for the
    # last one, a deliberate choice mirrored from `shaping_fade`.
    # `shaping_fade_dist` is a PushT (3D)-only config key, so this whole
    # block is a no-op for PushT2D by construction (fade_dist stays 0).
    _FADED = ("approach", "align", "tilt", "effort")
    fade_dist = float(getattr(task, "shaping_fade_dist", 0.0) or 0.0)
    if fade_dist > 0.0:
        floor = float(getattr(task, "fade_floor", 0.0) or 0.0)
        poses = np.asarray(log["object_pose"])[1:][:n]
        goal = np.asarray(task.goal)
        pos_err = np.linalg.norm(poses[:, :2] - goal[:2], axis=1)
        fade = np.clip((pos_err - floor) / (fade_dist - floor), 0.0, 1.0)
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
    boundary = np.asarray([np.asarray(obj.world_boundary(p)) for p in poses])
    terms["obstacle"] = _obstacle_cost(obj, obj.obstacles, boundary)
    terms["effort"] = obj.w_effort * np.sum(wrenches**2, axis=1)
    # Per *executed* step, so this is the realized jitter rather than the
    # within-horizon term the planner scored. Leading zero: nothing
    # precedes the first wrench.
    normalized = wrenches / np.asarray(obj.wrench_limit)
    steps = np.sum(
        np.asarray(obj.w_rate) * np.diff(normalized, axis=0) ** 2, axis=1
    )
    terms["rate"] = np.concatenate([[0.0], steps])
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
