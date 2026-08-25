"""The MuJoCo run log: what every world records, and in what shape.

One writer for the three MuJoCo-stepping runners -- headless 3D
(`oim.worlds.sim3d.run`), the interactive viewer (`oim.runtime.viewer`) and
the real robot (`oim.worlds.real3d.run_real`). They already shared it; they
just reached into `oim.worlds.sim3d.run` for the private names to do so, which
made a package-private helper the de facto interface of three packages.

Key names match `oim.worlds.sim2d.run`'s and
`oim.worlds.object_only.run`'s, so `oim.utils.metrics` and
`oim.utils.plotting` read every world's log without knowing which produced
it. Those two worlds do not step MuJoCo, so they build their own smaller
logs rather than calling these -- what is shared is the *schema*, and the
run file `oim.utils.results.save_run` writes from it.
"""

from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim.runtime.mjcf import hide_body_geoms, mocap_id, set_mocap_se2
from oim.tasks.pusht import PushT
from oim.utils.series import finite_difference


def local_goal_marker(
    ctrl: Any, mj_model: mujoco.MjModel
) -> Callable[..., None]:
    """Build the per-step update for the `local_goal` ghost marker.

    Returns a callable rather than taking the two "is this available?"
    tests every control step: whether the controller has an object block
    and whether the scene declares the marker are both fixed for a run, so
    they are answered once here and the loop just calls what it is given.

    Driven whenever both are true, *not* only under `local_goal` cost
    tracking: x^{o*}_H exists either way, and watching it before switching
    tracking on is how you judge whether it is worth tracking.

    What it draws is the *resolved* target -- `task.tracking_goal` applied
    to the plan endpoint, so with `local_goal=False` it sits on the global
    goal and with the flag on it follows the plan until the shaping-fade
    radius snaps it back to g. Drawing the raw endpoint instead would make
    the ghost disagree with the cost exactly where the two diverge, and
    the endpoint is already on screen as the end of the object-plan
    overlay, so the marker earns its place by showing what the plan
    endpoint is *used as*.

    When it will *not* be driven -- a flat baseline, which has no object
    block -- the marker's geoms are made fully transparent here rather than
    left parked wherever `execution_model` put them. A ghost frozen at the
    block's start pose for a whole run is worse than no ghost: it reads as
    a plan that never updated. Alpha is edited on the execution model only
    (a deepcopy), so the planner's own model is untouched.

    A caller that is *already* computing the object block's nominal plan
    for an overlay should hand the WHOLE plan in as `object_plan`:
    `ADMM.local_goal` is that plan resolved through
    `task.local_goal_from_plan`, so recomputing it here rolls the object
    block out a second time for an array the caller is holding. Free to
    ignore under the analytic backend, where a rollout is three multiplies
    and a norm; ~14 ms per control step under the MJX one, which is where
    the duplication started to matter.

    The plan, not its endpoint: under pure pursuit the target is a carrot
    partway along the route, so the endpoint alone is no longer enough to
    reconstruct it.

    Args:
        ctrl: The controller. Anything without `local_goal` (every flat
            baseline) gets the no-op, and hides the marker.
        mj_model: The execution model, whose mocap table is searched.

    Returns:
        `update(mj_data, mjx_data, params, object_plan=None)`, a no-op
        when unavailable.
    """
    index = mocap_id(mj_model, "local_goal")
    if index < 0 or not hasattr(ctrl, "local_goal"):
        hide_body_geoms(mj_model, "local_goal")
        return lambda mj_data, mjx_data, params, object_plan=None: None

    jit_local_goal = jax.jit(ctrl.local_goal)
    # What the robot block *tracks*, which is not always the plan endpoint:
    # `PushT.tracking_goal` reverts to the global goal inside the
    # shaping-fade radius. Applied here rather than inside `ADMM.local_goal`
    # so the ADMM layer keeps meaning "what the plan offers" (its
    # `_eval_rollouts_one` hands the task exactly that, and the task decides
    # what a goal is) -- but applied on *both* paths below, because a marker
    # that skipped the gate whenever the caller supplied `object_plan`
    # would silently disagree with the cost only on the fast path.
    #
    # The gate reads where the object is *now*, off `mjx_data`. The cost
    # gates per rollout step, so no single pose reproduces it exactly, and
    # the endpoint is the tempting choice because it is already in hand --
    # but it is the wrong one. A plan that overshoots past g leaves its
    # endpoint outside the radius while the object sits well inside it, and
    # the ghost then stays out at the overshoot through exactly the phase
    # the snap exists for. "Where is the object" is also the rule the radius
    # is stated by everywhere else: `shaping_fade`, `--shaping-fade-dist`,
    # and the README all measure it from the block.
    task = getattr(ctrl, "task", None)
    resolve = getattr(task, "tracking_goal", None)
    carrot = getattr(task, "local_goal_from_plan", None)
    block_pose = getattr(task, "_block_pose", None)
    if resolve is None or block_pose is None:
        jit_resolve = jit_carrot = jit_block_pose = None
    else:
        jit_resolve = jax.jit(resolve)
        jit_carrot = None if carrot is None else jax.jit(carrot)
        jit_block_pose = jax.jit(block_pose)

    def _update(
        mj_data: mujoco.MjData,
        mjx_data: mjx.Data,
        params: Any,
        object_plan: Optional[np.ndarray] = None,
    ) -> None:
        # One read of where the block is, shared by the carrot pick and
        # the fade gate -- both are stated from the block, not the plan.
        obj_pose = None if jit_block_pose is None else jit_block_pose(mjx_data)
        if object_plan is None:
            # `ADMM.local_goal` already resolves the plan itself.
            pose = np.asarray(jit_local_goal(mjx_data, params))
        elif jit_carrot is not None and obj_pose is not None:
            pose = np.asarray(jit_carrot(jnp.asarray(object_plan), obj_pose))
        else:
            pose = np.asarray(object_plan)[-1]
        if jit_resolve is not None:
            pose = np.asarray(jit_resolve(obj_pose, pose))
        set_mocap_se2(mj_data, index, pose)

    return _update


def init_log(
    task: PushT,
    mj_data: mujoco.MjData,
    mjx_data: mjx.Data,
    show_plans: bool,
    admm: bool = True,
) -> Dict[str, Any]:
    """Seed the log with the initial state and empty per-step series.

    Key names match `run_2d`'s so the two worlds' state logs line up entry
    for entry. qpos/qvel are the full MuJoCo state, kept so a run can be
    resumed or replayed exactly, not just plotted.
    """
    log: Dict[str, Any] = {
        "time": [float(mj_data.time)],
        "object_pose": [np.array(task._block_pose(mjx_data))],
        "object_velocity": [np.array(mjx_data.qvel[task.block_dofs])],
        "robot_pos": [np.array(task._pusher_pos(mjx_data))],
        "qpos": [np.array(mj_data.qpos)],
        "qvel": [np.array(mj_data.qvel)],
        "robot_control": [],
        "compute_time": [],
        # Derived, kept in memory for the console and the diagnostics plot
        # but filtered out by `save_run` -- a run file records only what
        # cannot be recomputed from it.
        "pos_err": [],
        "theta_err": [],
        # The end-effector pose quantities `oim.utils.costs` needs, which
        # no other series carries. Logged for both embodiments: the point
        # pusher has a trace site too, its tilt is simply constant.
        "tip_tilt": [],
        "tip_z": [],
        # Pusher-block contact's pure normal-force z-component, execution
        # fidelity -- see `PushT._contact_normal_force_z_mujoco`. 0.0
        # whenever there is no stick-block contact, same convention as
        # the quantity it logs.
        "contact_normal_force_z": [],
        # Robot-obstacle contact normal force, execution fidelity -- see
        # `PushT._robot_obstacle_force_mujoco`. 0.0 when not touching.
        "robot_contact_force": [],
        # C3 outer-loop mode (1=push, 0=reposition) and pursued target xy.
        "c3_is_c3": [],
        "c3_target": [],
    }
    if admm:
        # Meaningless for a flat controller: no consensus, no residuals.
        log.update(
            wrench=[],
            wrench_consensus=[],
            primal_residual=[],
            dual_residual=[],
            rho=[],
            # The two blocks' scaled duals and the object block's extracted
            # consensus value, all at horizon step 0 -- the entry the
            # executed control was scored against. With `wrench` (A^r),
            # `wrench_consensus` (z) and `rho` these complete the ADMM
            # penalty, which is otherwise the one term of either block's
            # cost that a run file cannot reconstruct.
            dual_object=[],
            dual_robot=[],
            object_consensus=[],
        )
    if show_plans:
        # Only allocated when asked for: (H, 3) per block per step is a
        # different order of magnitude from the rest of the log.
        log["object_plan"] = []
        log["robot_plan"] = []
    return log


def log_step(
    log: Dict[str, Any],
    task: PushT,
    mj_data: mujoco.MjData,
    params: Any,
    us: np.ndarray,
    admm: bool = True,
) -> np.ndarray:
    """Append one control step's state and diagnostics; return the pose."""
    block_pose = np.array(task._block_pose(mj_data))
    log["time"].append(float(mj_data.time))
    log["object_pose"].append(block_pose)
    log["object_velocity"].append(np.array(mj_data.qvel[task.block_dofs]))
    log["robot_pos"].append(np.array(task._pusher_pos(mj_data)))
    log["qpos"].append(np.array(mj_data.qpos))
    log["qvel"].append(np.array(mj_data.qvel))
    log["robot_control"].append(np.array(us[-1]))
    # The two end-effector quantities the cost breakdown needs that no other
    # logged series carries. Free: `mj_step` has already run forward
    # kinematics, so this is two array reads, not a second solve. Derived,
    # so `save_run` drops them -- `qpos` is recorded and the tip pose is a
    # forward-kinematics call away from it.
    site = int(task.trace_site_ids[0])
    r_mat = np.asarray(mj_data.site_xmat[site]).reshape(3, 3)
    log["tip_tilt"].append(float(task.tilt_angle(r_mat)))
    log["tip_z"].append(float(mj_data.site_xpos[site][2]))
    # These read execution-fidelity contact forces via mujoco.mj_contactForce,
    # which needs a plain mujoco.MjData. The real/mock driver (run_real) logs an
    # mjx.Data, whose .contact is not subscriptable -- so compute only when we
    # were handed a real MjData (the sim driver); log 0.0 otherwise.
    _is_mjdata = isinstance(mj_data, mujoco.MjData)
    log["contact_normal_force_z"].append(
        float(task._contact_normal_force_z_mujoco(mj_data))
        if _is_mjdata and hasattr(task, "_contact_normal_force_z_mujoco")
        else 0.0
    )
    log["robot_contact_force"].append(
        float(task._robot_obstacle_force_mujoco(mj_data))
        if _is_mjdata and hasattr(task, "_robot_obstacle_force_mujoco")
        else 0.0
    )
    # C3 outer-loop diagnostics (flat C3 only): its mode (1 = pushing, 0 =
    # repositioning) and the world-xy target it is pursuing, so a run can be
    # replayed to see WHEN it pushes vs approaches and WHERE it aims.
    _samp = getattr(params, "samp", None)
    if _samp is not None and hasattr(_samp, "is_c3"):
        log["c3_is_c3"].append(float(_samp.is_c3))
        log["c3_target"].append(np.array(_samp.target))
    else:
        log["c3_is_c3"].append(0.0)
        log["c3_target"].append(np.zeros(2))
    if admm:
        log["wrench"].append(np.array(task.realized_consensus(mj_data)))
        log["wrench_consensus"].append(np.array(params.z[0]))
        log["primal_residual"].append(float(params.primal_residual))
        log["dual_residual"].append(float(params.dual_residual))
        # `rho` is a scalar (paper's Algorithm 4) or a per-dimension vector
        # (force/torque split, see `WrenchConsensus.penalty_cost`); the
        # log keeps one number, so a vector logs its mean.
        log["rho"].append(float(np.mean(np.asarray(params.rho))))
        log["dual_object"].append(np.array(params.gamma_o[0]))
        log["dual_robot"].append(np.array(params.gamma_r[0]))
        log["object_consensus"].append(np.array(params.a_obj[0]))
    return block_pose


def finalize_log(
    log: Dict[str, Any],
    task: PushT,
    reached: bool,
    show_plans: bool,
    admm: bool = True,
) -> Dict[str, Any]:
    """Stack the per-step lists into arrays and derive robot velocity."""
    log["reached"] = reached
    for key in (
        "time",
        "object_pose",
        "object_velocity",
        "robot_pos",
        "qpos",
        "qvel",
        "robot_control",
        *(
            (
                "wrench_consensus",
                "dual_object",
                "dual_robot",
                "object_consensus",
            )
            if admm
            else ()
        ),
        *(("object_plan", "robot_plan") if show_plans else ()),
    ):
        log[key] = np.array(log[key])
    if admm:
        log["wrench"] = (
            np.array(log["wrench"]) if log["wrench"] else np.zeros((0, 3))
        )
    # Realized world-frame velocity of the contact point, by difference --
    # the arm's tip has no qvel entry of its own, and this is the quantity
    # the 2D world reports, so the two logs stay comparable.
    log["robot_vel"] = finite_difference(log["robot_pos"], task.dt)
    return log
