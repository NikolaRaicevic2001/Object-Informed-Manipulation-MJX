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
    "robot_obstacle",
    "obstacle_contact",
    "effort",
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
    terms["obstacle"] = _hinge(
        obstacles, boundary, obj.w_obstacle, obj.obstacle_margin
    )
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
        # 1 - cos(psi), matching `PushT._tilt`. The log stores the angle
        # because that is the readable unit; the cost is the cosine form.
        #
        # Omitted at zero weight, the same rule `_hinge` already follows:
        # `robot="point"` cannot rotate its site, so `_tilt` is a constant
        # 2.0 and the term is a flat `2*w_tilt` bar that fires on nothing.
        # At the shipped 30.0 that bar was 27% of the run's total cost and
        # the second-largest term in the figure, which reads as a finding
        # rather than as the constant it is. `PushT` now zeroes the weight
        # for that embodiment, and this drops the bar rather than drawing
        # it flat at zero.
        if task.w_tilt != 0.0:
            terms["tilt"] = task.w_tilt * (1.0 - np.cos(tilt))
        # Matches `PushT._tip_height_cost`: quadratic at/above mid-height,
        # exponential (in cm) below it. Dropped when both weights are zero,
        # the same rule `tilt` above follows: `robot="point"` has no z DOF,
        # so its tip sits exactly at `tip_target_z` in every scene and this
        # term is identically 0 -- a bar that can only ever draw flat.
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
        # The pusher's own clearance hinge, which `PushT` charges in BOTH
        # `running_cost` and `robot_running_cost`. It was missing from this
        # figure entirely: `robot_obstacle` below is reached only through
        # the `elif`, and a 3D task always takes the `if` (it has tip data),
        # so no 3D run has ever drawn it. Not one of the deliberately-
        # omitted terms named in the module docstring -- those are the ones
        # defined only for ADMM, and this is charged by the flat baseline
        # too. Reconstructable from the finished log at no cost, since
        # `robot_pos` is exactly the `task._pusher_pos` the cost reads.
        #
        # Effective weight is `pusher_obstacle_weight * w_obstacle`, the
        # product `PushT` forms, not the fraction alone -- the fraction is
        # ~0.5 and the base is 60000, so passing the fraction would draw
        # the bar five orders of magnitude too small.
        if getattr(task, "pusher_obstacle_weight", 0.0) != 0.0:
            pusher = np.asarray(log["robot_pos"])[1:][:n]
            terms["robot_obstacle"] = _hinge(
                obstacles,
                pusher[:, None, :2],
                task.pusher_obstacle_weight * task.object_model.w_obstacle,
                task.pusher_obstacle_margin,
            )
        # The object-obstacle contact-force penalty (paper eq. 19), from
        # `obstacle_contact_force`. Absent for run files predating that log
        # key, and skipped when the mechanism was inert for the run, so a
        # bar never appears for a weight that charged nothing.
        #
        # The same fidelity caveat `contact_z` above carries, and for the
        # same reason: the logged force is execution-fidelity newtons while
        # `_obstacle_contact_cost` only ever weights the planning model's
        # much smaller figure. The formula is applied literally here --
        # unlike `contact_z`, this one is already a plain quadratic hinge
        # and so grows without saturating -- but the bar should be read as
        # "how hard was the object really pressed into an obstacle, priced
        # at this weight", not as a replay of the optimizer's own value.
        if (
            "obstacle_contact_force" in log
            and getattr(task, "w_obstacle_contact", 0.0) != 0.0
        ):
            f_obs = np.asarray(log["obstacle_contact_force"])[:n]
            excess = np.clip(f_obs - task.obstacle_contact_deadband, 0.0, None)
            terms["obstacle_contact"] = task.w_obstacle_contact * excess**2
    elif hasattr(task, "w_obstacle_robot"):
        robot = np.asarray(log["robot_pos"])[1:][:n]
        terms["robot_obstacle"] = _hinge(
            obstacles,
            robot[:, None, :],
            task.w_obstacle_robot,
            task.obstacle_margin,
        )

    # Match `PushT.shaping_fade` exactly -- approach, align, effort and
    # the PUSHER hinge (`robot_obstacle`) fade. The OBJECT's clearance
    # hinge (`obstacle`) and the object/obstacle contact force
    # (`obstacle_contact`) do not, nor do tilt/tip_z. Keeping this in step
    # with that method is the whole point of the tuple: a term faded in
    # the cost but not here (or vice versa) makes the figure disagree with
    # what the optimizer actually paid, which is the one thing this module
    # exists to get right.
    _FADED = ("approach", "align", "effort", "robot_obstacle")
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
    boundary = np.asarray([np.asarray(obj.world_boundary(p)) for p in poses])
    terms["obstacle"] = _hinge(
        obj.obstacles, boundary, obj.w_obstacle, obj.obstacle_margin
    )
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
