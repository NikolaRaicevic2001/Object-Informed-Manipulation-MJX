"""Headless closed-loop driver for the MJX/MuJoCo world, with ADMM logging.

The 3D counterpart of `oim.worlds.sim2d.run.run_2d`: steps the real
`mujoco.MjData` and the MJX planning model in lockstep, returning the same
kind of log dict
`run_2d` does (trajectories, wrenches, primal/dual residuals, goal errors),
for `oim.tasks.pusht.PushT` under either `robot` embodiment. Headless -- no
viewer -- mirrors `oim.runtime.viewer.run_interactive`'s stepping, but
that function is generic over any controller/task and has no way to return
a log, which is what this fills in for the ADMM+PushT case specifically.
Both write through `oim.runtime.logs`, so the two paths cannot disagree
about what a run recorded.

Video is still available headless (`record_dir`/`record_name`):
`run_interactive` only needs the viewer for its *camera*, since frames come
from an offscreen `mujoco.Renderer` either way. Here that camera is
constructed directly, so no display is involved.

`run_3d_plain` is the flat-baseline counterpart, for any non-ADMM
`SamplingBasedController` -- see `oim.utils.metrics` for comparing the two.
"""

import time
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim.alg_base import SamplingBasedController, quiet_mjx_cast_overflow
from oim.algs.admm import ADMM
from oim.objects import wrap_angle
from oim.runtime.logs import finalize_log, init_log, local_goal_marker, log_step
from oim.runtime.overlay import (
    CONTACT_POINT_HEIGHT,
    PlanOverlay,
    contact_points_world,
    traces_for,
)
from oim.runtime.video import OffscreenRecorder
from oim.tasks.pusht import PushT


def _arm_ctrl_from_ee_vel(
    task: PushT,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    v_xy: np.ndarray,
    alpha: float = 5.0,
    damping: float = 1e-4,
) -> np.ndarray:
    """Map C3's planar EE velocity `[vx, vy]` to a full ctrl-sized joint-
    velocity command for the xarm6 arm, reusing the repo's task-space mapping.

    C3-on-xarm6 plans a 2-DOF PLANAR EE (see `C3MJXSampling.emits_ee_velocity`)
    and so, unlike MPPI/ADMM which sample all 6 joints against a 3-D cost, it
    cannot plan the tip's height or tilt -- those must be REGULATED, not
    planned. This reuses `_task_space_jac_bias_and_null` (built for MPPI's
    task-space noise): its `bias` drives the tip down to `tip_target_z` (the
    block mid-height) and holds the stick vertical, and `noise_map` routes
    `[vx, vy]` through the null space of the z/tilt rows so pushing never
    disturbs height/tilt. `alpha` is the z/tilt feedback gain (1/s). Uses
    `mujoco.mj_jacSite` on the real model, so it lives here, not in jit. Any
    joint without a matching actuator (e.g. the locked wrist roll) stays zero.
    """
    jac_inv, bias, noise_map = _task_space_jac_bias_and_null(
        task, mj_model, mj_data, alpha, damping
    )
    jac_inv = np.asarray(jac_inv)
    bias = np.asarray(bias)
    noise_map = np.asarray(noise_map)
    # z/tilt held by the bias; planar push added in the z/tilt null space.
    qdot = jac_inv @ bias + noise_map @ np.asarray(v_xy)  # (n_arm,)
    ctrl = np.zeros(mj_model.nu)
    dof = np.asarray(task.robot_dof_adr)
    trn_joint = mj_model.actuator_trnid[:, 0]        # joint id per actuator
    jnt_dofadr = mj_model.jnt_dofadr                 # first dof adr per joint
    for k, d in enumerate(dof):
        act = np.where(jnt_dofadr[trn_joint] == d)[0]
        if act.size:
            ctrl[act[0]] = qdot[k]
    return ctrl


def _configure_arm_torque(mj_model: mujoco.MjModel, task: PushT) -> None:
    """Flip the xarm6 arm actuators from velocity servos to torque passthrough
    (force = ctrl) for the C3 operational-space controller.

    A MuJoCo velocity actuator is gain=kv, bias=-kv*qvel; zeroing the bias and
    setting gain=1 makes the applied force equal the commanded value, i.e. a
    joint torque. Gravity stays compensated by the model's per-link gravcomp=1
    (the arm holds itself at zero torque), and joint damping/armature remain for
    passive stability, so no gravity term is needed downstream. `ctrllimited` is
    cleared because the velocity ctrlrange (+-0.5 rad/s) would otherwise clip the
    torque; `forcerange` is kept as the real per-joint torque ceiling. Only the
    arm actuators are touched -- the samplers use their own model instances.
    """
    dof = np.asarray(task.robot_dof_adr)
    trn_joint = mj_model.actuator_trnid[:, 0]
    jnt_dofadr = mj_model.jnt_dofadr
    for d in dof:
        for a in np.where(jnt_dofadr[trn_joint] == d)[0]:
            mj_model.actuator_gaintype[a] = mujoco.mjtGain.mjGAIN_FIXED
            mj_model.actuator_gainprm[a, :] = 0.0
            mj_model.actuator_gainprm[a, 0] = 1.0
            mj_model.actuator_biastype[a] = mujoco.mjtBias.mjBIAS_NONE
            mj_model.actuator_biasprm[a, :] = 0.0
            mj_model.actuator_ctrllimited[a] = 0


def _arm_osc_torque(
    task: PushT,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    v_xy: np.ndarray,
    f_c3: np.ndarray,
    gains: Tuple[float, float, float, float, float, float],
) -> np.ndarray:
    """Operational-space (Khatib) torque that tracks C3's planar EE velocity
    while holding the tip at `tip_target_z` and the stick vertical -- the
    faithful analog of dairlib's OSC end-effector tracking.

    The arm's 5 joints and a 5-D tip task [dx, dy, dz, tilt_x, tilt_y] are
    square (no redundancy, no null-space term needed). tau = J^T Lambda xddot*,
    with Lambda = (J M^-1 J^T)^-1 the operational-space inertia and xddot* the
    task PD: xy tracks the C3 velocity, z holds the pushing height, tilt holds
    vertical. Gravity is NOT added -- the model's gravcomp compensates it, so
    adding it here would double-count (per the actuator flip above). Runs on the
    real (non-MJX) model via mj_jacSite/mj_fullM, so it lives here, not in jit.
    """
    dof = np.asarray(task.robot_dof_adr)
    jacp = np.zeros((3, mj_model.nv))
    jacr = np.zeros((3, mj_model.nv))
    mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, int(task.tip_site_id))
    Jp = jacp[:, dof]
    Jr = jacr[:, dof]
    J = np.vstack([Jp, Jr[:2]])                       # (5, n): dx,dy,dz,wx,wy
    M = np.zeros((mj_model.nv, mj_model.nv))
    mujoco.mj_fullM(mj_model, M, mj_data.qM)
    Marm = M[np.ix_(dof, dof)]
    Minv = np.linalg.inv(Marm)
    Lam = np.linalg.inv(J @ Minv @ J.T + 1e-6 * np.eye(5))  # op-space inertia
    qd = np.asarray(mj_data.qvel)[dof]
    xdot = Jp @ qd                                    # tip linear velocity (3,)
    wdot = Jr @ qd                                    # tip angular velocity (3,)
    z = float(mj_data.site_xpos[int(task.tip_site_id), 2])
    r_mat = np.asarray(mj_data.site_xmat[int(task.tip_site_id)]).reshape(3, 3)
    tilt = np.array([r_mat[0, 2], r_mat[1, 2]])       # 0 when vertical
    kv_xy, kp_z, kd_z, kp_r, kd_r, z_vmax = gains
    # z as a velocity-LIMITED approach, not a raw position spring: a large
    # initial height error would otherwise dominate the xy push and make the
    # arm plunge straight down before moving toward the block. Capping the
    # descent speed lets the xy velocity move the tip diagonally toward the
    # block while it lowers, and still holds tip_target_z once there.
    zdot_des = np.clip(kp_z * (task.tip_target_z - z), -z_vmax, z_vmax)
    acc = np.array([
        kv_xy * (float(v_xy[0]) - xdot[0]),           # xy: track C3 velocity
        kv_xy * (float(v_xy[1]) - xdot[1]),
        kd_z * (zdot_des - xdot[2]),                  # z: velocity-limited hold
        -kp_r * tilt[0] - kd_r * wdot[0],             # tilt: hold vertical
        -kp_r * tilt[1] - kd_r * wdot[1],
    ])
    # Stage 2: C3-solved contact force fed forward as an EE wrench (J^T F),
    # the operational-space analog of dairlib's OSC force-tracking term. Inert
    # until the pusher is actually in contact (F_c3 = 0 out of contact).
    tau_arm = J.T @ (Lam @ acc) + Jp[:2].T @ np.asarray(f_c3)   # (n,)
    ctrl = np.zeros(mj_model.nu)
    trn_joint = mj_model.actuator_trnid[:, 0]
    jnt_dofadr = mj_model.jnt_dofadr
    for k, d in enumerate(dof):
        a = np.where(jnt_dofadr[trn_joint] == d)[0]
        if a.size:
            ctrl[a[0]] = tau_arm[k]
    return ctrl


def _task_space_jac_bias_and_null(
    task: PushT,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    alpha: float,
    damping: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Damped-inverse tip Jacobian, z/tilt feedback bias, and an x/y noise
    map confined to the null space of the z/tilt rows, this step's q.

    For `oim.algs.mppi.MPPI`'s task-space noise mechanism (see
    `MPPIParams.task_jac_inv`/`task_bias`/`task_noise_map`'s docstrings)
    -- computed here, not inside the jitted `optimize`, because it needs
    `mujoco.mj_jacSite` on the real (non-MJX) model. Only ever called
    from `_run_plain` (xarm6 flat baseline); ADMM's `_run` never calls
    this, so its own MPPI robot block -- built from the same config but
    never given real values here -- must have `use_task_space_noise=
    False`, or every one of its samples collapses to zero perturbation
    (see `MPPIParams.task_jac_inv`'s "zeros as a structural no-op"
    default).

    Why a SEPARATE map for x/y (2026-08-16, per Shahid): the general
    inverse (`jac_inv` below) generically couples every task direction,
    because several joints each move more than one task direction at
    once -- e.g. joint 4 alone drives `wx` (tilt about x) with
    coefficient ~1.0 *and* contributes to `dy` at the very same
    configuration; there is no way to move joint 4 at all without both
    happening together, a fact about this arm's kinematic chain, not a
    modeling choice. So x/y noise routed through `jac_inv` was itself a
    real source of z/tilt disturbance (confirmed directly: the single
    best-approach sample in a 128-sample population still spiked to an
    8.9M tip-height cost from a 1.5cm dip it accumulated purely from its
    own x/y noise). The fix: restrict x/y exploration to the null space
    of just the z/tilt rows (`jac_zt`, 3x5) -- joint-space directions
    that are mathematically guaranteed to produce EXACTLY zero z/tilt
    velocity, not merely small. Checked this null space is not
    degenerate at the real starting configuration before building this
    (`check_null_space.py`): it is 2-dimensional and still produces
    real x/y motion (~0.15-0.2 m/s per unit coefficient), not near-zero.
    `null_basis` is taken as the LAST 2 right-singular-vectors of
    `jac_zt` unconditionally (not gated on a rank/threshold check) so
    its shape never varies step to step even if the arm passes near a
    configuration where `jac_zt`'s true rank drops -- degrades
    gracefully there rather than changing shape, which a jitted
    `MPPIParams` field cannot tolerate anyway.

    v = [dx, dy, dz, d(tilt_x), d(tilt_y)] is 5-dimensional, not 6: the
    xarm6 stick end-effector is an axisymmetric capsule (confirmed from
    the model, `oim/models/xarm6/xarm6.xml`), so spin about its own axis
    (what a 6th, yaw, task-space direction would be) does nothing
    physically -- matching the real vendor's own `xarm6_stick.urdf`,
    which locks that joint for the same reason. Dropping it makes the
    Jacobian square (5 task-space rows, 5 robot joints) rather than a
    6x5 that would need a least-squares pseudo-inverse to even define;
    the damped inverse below is still used, purely for numerical safety
    near kinematic singularities, not to resolve a dimension mismatch.
    tilt_x/tilt_y are small-angle proxies (the tip site's local z-axis's
    world x/y components -- 0 exactly when vertical), not literal roll/
    pitch Euler angles; see Tasks.md for the full derivation.

    Args:
        task: The `PushT` task (`robot="xarm6"`) this is built against.
        mj_model: The execution model.
        mj_data: Its current state.
        alpha: Feedback gain (1/s), see `MPPI.__init__`'s
            `task_space_alpha`.
        damping: Levenberg-Marquardt damping, see `MPPI.__init__`'s
            `task_space_damping`.

    Returns:
        task_jac_inv, (nu, 5). task_bias, (5,). task_noise_map, (nu, 2).
    """
    nv = mj_model.nv
    jacp = np.zeros((3, nv))
    jacr = np.zeros((3, nv))
    mujoco.mj_jacSite(mj_model, mj_data, jacp, jacr, task.tip_site_id)
    dof = task.robot_dof_adr
    jac = np.vstack([jacp[:, dof], jacr[:2, dof]])  # (5, 5): dx,dy,dz,wx,wy
    jac_inv = np.linalg.solve(
        jac.T @ jac + damping * np.eye(jac.shape[1]), jac.T
    )  # (5, 5)

    jac_zt = jac[2:, :]  # (3, 5): just dz, wx, wy
    _, _, vt = np.linalg.svd(jac_zt)
    null_basis = vt[-2:].T  # (5, 2): joint directions with ~0 dz/wx/wy
    jac_xy = jac[:2, :]  # (2, 5): dx, dy
    m = jac_xy @ null_basis  # (2, 2): xy velocity per null-space coeff
    m_inv = np.linalg.solve(m.T @ m + damping * np.eye(2), m.T)  # (2, 2)
    noise_map = null_basis @ m_inv  # (5, 2): desired [dx,dy] -> safe qdot

    z = mj_data.site_xpos[task.tip_site_id, 2]
    r_mat = mj_data.site_xmat[task.tip_site_id].reshape(3, 3)
    tilt_x, tilt_y = r_mat[0, 2], r_mat[1, 2]  # 0 exactly when vertical

    # omega_desired = alpha * (n x n_target), n_target = (0, 0, -1) --
    # the exact (not small-angle) attitude-control law that rotates a
    # body axis n toward a target direction n_target at rate alpha
    # (standard result: d(n)/dt = omega x n, so this gives d(n)/dt =
    # alpha*(n_target - n*(n.n_target)), the component of n_target
    # orthogonal to n, i.e. genuine convergence, not an approximation
    # valid only very close to vertical). n_target = (0, 0, -1), not
    # (0, 0, 1): confirmed from PushT._tilt's own formula, `1 +
    # R[2,2]`, documented as 0 at vertical -- only true if R[2,2] = -1
    # there, i.e. the tip's local z-axis points *down* when vertical.
    #
    # Two earlier versions of this line were both wrong, in the same
    # underlying way (a hand-derived small-angle sign/axis mapping),
    # caught from real telemetry rather than by inspection alone:
    #   1) omega_x <- -alpha*tilt_x, omega_y <- -alpha*tilt_y (the
    #      "obvious" diagonal guess): produced a persistent, non-
    #      decaying *rotation* of (tilt_x, tilt_y) at rate alpha
    #      instead of decay -- all 3 test seeds froze at the exact
    #      start pose for the full 2000 steps (byte-identical
    #      pos_err/theta_err across different seeds, only possible if
    #      something deterministic broke before seed-dependent noise
    #      could matter) with real NaN/Inf CTRL warnings.
    #   2) omega_x <- +alpha*tilt_y, omega_y <- -alpha*tilt_x (the
    #      axis swap corrected, but derived assuming n_target=(0,0,1)):
    #      opposite sign from what's below -- mean tilt rose to
    #      96-131 degrees across the 3 seeds, worse than doing
    #      nothing, consistent with a positive- rather than negative-
    #      feedback sign. Verified directly (not just re-derived) via
    #      verify_tilt_bias_full.py before trusting this version: from
    #      a real, meaningfully tilted state, this sign shows a genuine
    #      net decrease in 1-cos(tilt) over a short closed-loop
    #      rollout, past a brief initial overshoot from momentum.
    bias = np.array(
        [
            0.0,
            0.0,
            -alpha * (z - task.tip_target_z),
            -alpha * tilt_y,
            alpha * tilt_x,
        ]
    )
    return jnp.asarray(jac_inv), jnp.asarray(bias), jnp.asarray(noise_map)


def run_3d_admm(
    task: PushT,
    ctrl: ADMM,
    params: Any,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    verbose: bool = True,
    record_dir: Optional[str] = None,
    record_name: Optional[str] = None,
    video_fps: float = 30.0,
    video_size: Tuple[int, int] = (720, 480),
    camera: Optional[Union[str, int]] = None,
    show_samples: bool = False,
    show_optimal: bool = False,
    show_contact_point: bool = False,
) -> Dict[str, Any]:
    """Run the MJX closed loop under ADMM and return a log, headless.

    Args:
        task: The `PushT` task (`robot="point"` or `"xarm6"`).
        ctrl: The ADMM controller, built against `task`.
        params: Its initial policy parameters.
        mj_model: The (fine-timestep) execution model.
        mj_data: Its initial state.
        frequency: Replanning frequency (Hz).
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Whether to print progress.
        record_dir: Directory for an mp4. None disables recording.
        record_name: Filename stem for the mp4, no extension -- pass the
            same base name used for the run's plot and results JSON.
            Required when `record_dir` is given.
        video_fps: Target playback frame rate.
        video_size: (width, height) of the video in pixels.
        camera: Model camera name or id to render from. None uses the
            default free camera, which frames the whole scene.
        show_samples: Composite each block's sampled candidate rollouts
            into the recorded frames, as the live viewer's `show_samples`
            does.
        show_optimal: Composite each block's chosen trajectory. Independent
            of `show_samples` -- either, both, or neither. Either one also
            logs `object_plan`/`robot_plan` per step, so that comparison
            survives into the states file and can be re-examined without
            the video. Needs `record_dir` to be of any visual use, but the
            logging happens either way.

    Returns:
        A dict with the block/pusher trajectories, per-step wrenches,
        primal/dual residuals, goal errors, and whether the goal was reached.

    Raises:
        ValueError: If `record_dir` is given without `record_name`.
    """
    show_plans = show_samples or show_optimal
    # Gated on the *task*, not just the flag: the dot is the agreed
    # [p_x, p_y, lambda] placed on the object, so under a wrench consensus
    # there is no such quantity to draw -- z's first two entries are
    # forces, and putting them on the block would be a plausible-looking
    # lie. Warn rather than fail: a sweep may set the flag once and vary
    # `consensus` across cells.
    draw_contacts = show_contact_point and _contact_consensus(task)
    # The height `w_z_tip` holds the tip at, so the dot marks a place the
    # tip is actually asked to reach and the two never look like they
    # disagree. Scenes without the attribute fall back to block mid-height.
    contact_height = getattr(task, "tip_target_z", CONTACT_POINT_HEIGHT)
    if show_contact_point and not draw_contacts:
        print(
            "[warn] --show-contact-point ignored: it needs "
            "consensus='contact_point', got "
            f"{getattr(task, 'consensus', 'wrench')!r}"
        )
    # Four slots: both blocks' predictions for the object, the
    # end-effector's own path, and the contact dots. See
    # `oim.runtime.overlay`.
    overlay = (
        PlanOverlay(horizon=ctrl.ctrl_steps, max_blocks=4)
        if show_plans
        else None
    )
    recorder = None
    if record_dir is not None:
        if record_name is None:
            raise ValueError("record_dir requires record_name")
        recorder = OffscreenRecorder(
            mj_model,
            output_dir=record_dir,
            base_name=record_name,
            target_fps=video_fps,
            size=video_size,
            camera=camera,
            overlay=overlay,
        )
    try:
        return _run(
            task,
            ctrl,
            params,
            mj_model,
            mj_data,
            frequency,
            max_steps,
            goal_pos_tol,
            goal_theta_tol,
            verbose,
            recorder,
            show_plans,
            show_samples,
            show_optimal,
            draw_contacts,
            contact_height,
        )
    finally:
        if recorder is not None:
            recorder.close()


# A path shorter than this spans well under a pixel at any sane camera
# distance, so it is drawn but cannot be seen. Reported rather than left
# silent: an overlay that renders nothing looks identical to an overlay
# that was never switched on, and telling those apart by eye cost a while.
_VISIBLE_SPAN_M = 5e-3


def _contact_consensus(task: Any) -> bool:
    """Whether this task's consensus variable is the contact point."""
    return getattr(task, "consensus", "wrench") == "contact_point"


def _report_plan_spans(
    object_plan: np.ndarray, robot_plan: np.ndarray, robot_trace: np.ndarray
) -> None:
    """Print each overlaid path's extent once, flagging invisible ones.

    A block that plans no motion produces a plan whose every pose is the
    same point; the overlay then draws H zero-length segments, which is
    indistinguishable from not drawing at all. That is a real failure --
    the object block collapsing into `PlanarPushingObject.step`'s breakaway
    dead zone does exactly this -- and it is worth naming at the top of a
    run rather than leaving someone to infer it from an empty video.

    Args:
        object_plan: The object block's predicted poses, (H, >=2).
        robot_plan: The robot block's predicted object poses, (H, >=2).
        robot_trace: The end-effector's own path, (H, 3).
    """
    spans = {
        "object block -> object": object_plan,
        "robot block  -> object": robot_plan,
        "robot block  -> tip": robot_trace,
    }
    print("plan overlay, first step:")
    for name, path in spans.items():
        span = float(np.linalg.norm(path[-1, :2] - path[0, :2]))
        flag = "  <-- degenerate, will not be visible" if (
            span < _VISIBLE_SPAN_M
        ) else ""
        print(f"  {name}: span {span:.4f} m{flag}")


def _draw_plans(
    jit_plans: Optional[Any],
    mjx_data: mjx.Data,
    params: Any,
    rollouts: Any,
    log: Dict[str, Any],
    recorder: Optional[Any],
    *,
    show_optimal: bool,
    show_samples: bool,
    report: bool,
    draw_contacts: bool = False,
    contact_height: float = CONTACT_POINT_HEIGHT,
) -> Optional[np.ndarray]:
    """Log and overlay both blocks' plans; return the object block's endpoint.

    Split out of the control loop for two reasons: the loop was at its
    statement budget, and the endpoint this returns is what stops the
    caller from asking the controller for `local_goal` separately -- the
    same object rollout, run twice.

    Args:
        jit_plans: `ADMM.nominal_plans`, compiled, or None to skip.
        mjx_data: The state the plans start from.
        params: What `optimize` just returned.
        rollouts: That step's sampled robot rollouts, for the overlay.
        log: The run log, appended to in place.
        recorder: The mp4 recorder, or None.
        show_optimal: Overlay each block's chosen trajectory.
        show_samples: Overlay the sampled populations.
        report: Print the spans (first step only).
        draw_contacts: Mark the agreed contact point on the object at each
            planned pose. Already resolved against `consensus` by
            the caller -- see `run_3d_admm`.
        contact_height: World z for those dots, the task's `tip_target_z`.

    Returns:
        The object block's planned trajectory, (H, 3), or None when plans
        are not being computed. The whole plan, not its endpoint: under
        pure pursuit the tracked target is a carrot partway along it, so
        `local_goal_marker` needs the route to reconstruct the marker.
    """
    if jit_plans is None:
        return None
    object_plan, robot_plan, robot_trace = jit_plans(mjx_data, params)
    object_plan = np.asarray(object_plan)
    robot_plan = np.asarray(robot_plan)
    robot_trace = np.asarray(robot_trace)
    log["object_plan"].append(object_plan)
    log["robot_plan"].append(robot_plan)
    if report:
        _report_plan_spans(object_plan, robot_plan, robot_trace)
    if recorder is not None:
        recorder.set_plans(
            traces_for(
                robot_chosen=robot_trace if show_optimal else None,
                object_chosen=object_plan if show_optimal else None,
                # The same object as `object_chosen`, under the other
                # block's plan -- their separation is the consensus
                # disagreement drawn rather than summed.
                robot_object_chosen=robot_plan if show_optimal else None,
                # trace_sites: (num_samples, H+1, num_trace_sites, 3) --
                # this task has exactly one trace site (the pusher tip).
                robot_samples=(
                    np.asarray(rollouts.trace_sites)[:, :, 0, :]
                    if show_samples
                    else None
                ),
                object_samples=(
                    np.asarray(params.object_samples)
                    if show_samples
                    else None
                ),
                # z, not A^o: the dot marks what the two blocks *agreed*
                # the contact should be, which is what the robot block is
                # penalized against.
                contact_points=(
                    contact_points_world(
                        object_plan, np.asarray(params.z), contact_height
                    )
                    if draw_contacts
                    else None
                ),
            )
        )
    return object_plan


def _run(
    task: PushT,
    ctrl: ADMM,
    params: Any,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    max_steps: int,
    goal_pos_tol: float,
    goal_theta_tol: float,
    verbose: bool,
    recorder: Optional[OffscreenRecorder],
    show_plans: bool,
    show_samples: bool,
    show_optimal: bool,
    draw_contacts: bool,
    contact_height: float,
) -> Dict[str, Any]:
    """The closed loop itself; see `run_3d_admm` for the arguments."""
    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)

    mjx_data = task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )
    # Populate site_xpos etc. before the first log entry reads them
    # (a freshly made mjx.Data hasn't run forward kinematics yet).
    with quiet_mjx_cast_overflow():
        mjx_data = mjx.forward(task.model, mjx_data)
    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp_func = jax.jit(ctrl.interp_func)

    log = init_log(task, mj_data, mjx_data, show_plans)
    jit_plans = jax.jit(ctrl.nominal_plans) if show_plans else None
    draw_local_goal = local_goal_marker(ctrl, mj_model)
    reached = False

    for step in range(max_steps):
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )
        t0 = time.perf_counter()
        params, rollouts = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)

        # After optimize (the plans come from the params it just produced,
        # and the samples from the rollouts that produced them) and before
        # the substep loop (the recorder draws them into every frame of the
        # step they belong to).
        object_plan = _draw_plans(
            jit_plans, mjx_data, params, rollouts, log, recorder,
            show_optimal=show_optimal, show_samples=show_samples,
            draw_contacts=draw_contacts,
            contact_height=contact_height,
            report=step == 0 and verbose,
        )
        # Also before the substeps, so every frame of the step shows the
        # target that step's plan was scored against. Outside the
        # `compute_time` measurement on purpose -- it is visualization, and
        # folding it in would depress the reported planning rate.
        #
        # Fed the plan when there is one: `ADMM.local_goal` resolves that
        # same array, so recomputing it rolls the object block out a second
        # time per control step for something already in hand. Free under
        # the analytic backend, ~14 ms/step under MJX.
        draw_local_goal(mj_data, mjx_data, params, object_plan)

        tq = (
            jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
            + mj_data.time
        )
        tk = params.tk
        knots = params.mean[None, ...]
        us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)
            if recorder is not None:
                recorder.capture(mj_data)

        # mj_step = mj_forward (contacts/forces at the PRE-step qpos) +
        # integration (advances qpos) -- so immediately after the loop,
        # mj_data.contact/efc_force describe the geometry from BEFORE the
        # last substep's integration, one substep (exec_timestep) stale
        # relative to the qpos they're paired with here. A contact that
        # only becomes active during that final integration is invisible
        # to log_step's read unless forced fresh. Confirmed directly
        # (2026-08-20, per Shahid): a real, sustained 7-15N xarm6_stick/
        # block_stem contact -- persisting for 50+ steps, found by
        # replaying a run's own logged qpos through a fresh mj_forward --
        # logged as exactly 0.0 in contact_normal_force_z/
        # robot_contact_force at every one of those same steps in the
        # original run. Does not change what was simulated, only what
        # gets recorded: MPPI's own planning-fidelity contact reads
        # (`_contact_normal_force_z`, `_contact_z_cost`) are computed
        # separately, on the MJX/Warp planning model, and were never
        # affected by this.
        mujoco.mj_forward(mj_model, mj_data)
        block_pose = log_step(log, task, mj_data, params, us)

        goal = np.asarray(task.goal)
        pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
        theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
        log["pos_err"].append(pos_err)
        log["theta_err"].append(theta_err)
        if verbose:
            print(
                f"step {step:4d}  pos_err={pos_err:.4f}  "
                f"theta_err={theta_err:.4f}  "
                f"primal={log['primal_residual'][-1]:.3f}  "
                f"dual={log['dual_residual'][-1]:.3f}  "
                f"rho={log['rho'][-1]:.2f}"
            )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

    return finalize_log(log, task, reached, show_plans)


def run_3d_plain(
    task: PushT,
    ctrl: SamplingBasedController,
    params: Any,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    verbose: bool = True,
    record_dir: Optional[str] = None,
    record_name: Optional[str] = None,
    video_fps: float = 30.0,
    video_size: Tuple[int, int] = (720, 480),
    camera: Optional[Union[str, int]] = None,
    show_samples: bool = False,
    show_optimal: bool = False,
) -> Dict[str, Any]:
    """Run a plain (non-ADMM) controller headlessly and return a log.

    The flat-baseline counterpart of `run_3d_admm`, generic over any
    `SamplingBasedController` since plain params carry no residual/rho
    fields. Pass the same `task` the ADMM side runs, for a fair comparison
    on the identical scene.

    Logs the same recorded state `run_3d_admm` does, minus the consensus
    quantities a flat controller has none of, so both can be written by
    `oim.utils.results.save_run` and compared by `oim.utils.metrics`.

    Args:
        task: The `PushT` task.
        ctrl: Any `SamplingBasedController` built against `task`.
        params: Its initial policy parameters.
        mj_model: The (fine-timestep) execution model.
        mj_data: Its initial state.
        frequency: Replanning frequency (Hz).
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Whether to print progress.
        record_dir: Directory for an mp4. None disables recording.
        record_name: Filename stem for the mp4, no extension. Required
            when `record_dir` is given.
        video_fps: Target playback frame rate.
        video_size: (width, height) of the video in pixels.
        camera: Model camera name or id to render from. None uses the
            default free camera.
        show_samples: Composite this controller's sampled candidate
            rollouts into the recorded frames. A flat controller has one
            population, in robot space, so one block is drawn where ADMM
            draws two.
        show_optimal: Composite the trajectory it chose, thicker. Independent
            of `show_samples` -- either, both, or neither.

    Returns:
        A log dict with the same trajectory keys as `run_3d_admm`.

    Raises:
        ValueError: If `record_dir` is given without `record_name`.
    """
    show_plans = show_samples or show_optimal
    overlay = (
        PlanOverlay(horizon=ctrl.ctrl_steps, max_blocks=1)
        if show_plans
        else None
    )
    recorder = None
    if record_dir is not None:
        if record_name is None:
            raise ValueError("record_dir requires record_name")
        recorder = OffscreenRecorder(
            mj_model,
            output_dir=record_dir,
            base_name=record_name,
            target_fps=video_fps,
            size=video_size,
            camera=camera,
            overlay=overlay,
        )
    try:
        return _run_plain(
            task,
            ctrl,
            params,
            mj_model,
            mj_data,
            frequency,
            max_steps,
            goal_pos_tol,
            goal_theta_tol,
            verbose,
            recorder,
            show_samples,
            show_optimal,
        )
    finally:
        if recorder is not None:
            recorder.close()


def _run_plain(
    task: PushT,
    ctrl: SamplingBasedController,
    params: Any,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    max_steps: int,
    goal_pos_tol: float,
    goal_theta_tol: float,
    verbose: bool,
    recorder: Optional[OffscreenRecorder],
    show_samples: bool = False,
    show_optimal: bool = False,
) -> Dict[str, Any]:
    """The flat closed loop itself; see `run_3d_plain` for the arguments."""
    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)
    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp_func = jax.jit(ctrl.interp_func)
    # Only the chosen path needs a rollout of its own; the candidates come
    # free with the `Trajectory` `optimize` already returns.
    jit_trace = jax.jit(ctrl.nominal_trace) if show_optimal else None
    # A flat controller has no object block, so this only hides the ghost
    # marker -- otherwise it would sit frozen at the block's start pose for
    # the whole run, in scenes that declare it.
    local_goal_marker(ctrl, mj_model)

    mjx_data = task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )
    with quiet_mjx_cast_overflow():
        mjx_data = mjx.forward(task.model, mjx_data)

    log = init_log(task, mj_data, mjx_data, show_plans=False, admm=False)
    reached = False
    goal = np.asarray(task.goal)

    # Detect-and-kick only ever touches `params` (the traced pytree),
    # never a `self.` attribute on `ctrl` -- `jit_optimize` is a jitted
    # bound method, so `self.x` is baked in as a constant at first trace
    # and later mutation is silently ignored. `ctrl.stuck_kick_steps` is
    # only *read* here, in plain Python between jitted calls, to decide
    # what to write into `params`. A no-op for any controller that
    # doesn't carry the attribute (CEM/PS/CBO, or MPPI built with the
    # config defaults, which are inert).
    stuck_kick_steps = getattr(ctrl, "stuck_kick_steps", 0)
    stuck_kick_scale = getattr(ctrl, "stuck_kick_scale", 0.0)
    # Task-space noise (see oim.algs.mppi.MPPI's `task_space_noise`):
    # needs a fresh tip Jacobian/feedback bias/null-space map every step,
    # from the real (non-MJX) model -- `_task_space_jac_bias_and_null`
    # cannot run inside `jit_optimize`. `getattr` defaults keep this a
    # no-op for any
    # controller without the attribute (CEM/PS/CBO, or an MPPI instance
    # built with `task_space_noise=None`, the default).
    use_task_space_noise = getattr(ctrl, "use_task_space_noise", False)
    task_space_alpha = getattr(ctrl, "task_space_alpha", 0.0)
    task_space_damping = getattr(ctrl, "task_space_damping", 1e-4)
    # C3-on-xarm6 operational-space torque control: flip the arm actuators to
    # torque passthrough once, up front, and cache the OSC gains.
    arm_torque_osc = getattr(ctrl, "arm_torque_osc", False)
    if arm_torque_osc:
        _configure_arm_torque(mj_model, task)
    osc_gains = (
        getattr(ctrl, "osc_kv_xy", 20.0), getattr(ctrl, "osc_kp_z", 8.0),
        getattr(ctrl, "osc_kd_z", 60.0), getattr(ctrl, "osc_kp_rot", 100.0),
        getattr(ctrl, "osc_kd_rot", 20.0), getattr(ctrl, "osc_z_vmax", 0.3),
    )
    # Matches the exact-zero signature real stiction produces in MJX/Warp
    # (traced directly in run files: object_velocity goes bit-exact 0.0,
    # not a gradual decay) -- not a tolerance chosen to catch "slow"
    # progress, which would also flag a genuinely converging run.
    stuck_eps = 1e-4
    stuck_count = 0
    prev_pos_err: Optional[float] = None
    prev_theta_err: Optional[float] = None

    for step in range(max_steps):
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )
        if use_task_space_noise:
            jac_inv, bias, noise_map = _task_space_jac_bias_and_null(
                task, mj_model, mj_data, task_space_alpha, task_space_damping
            )
            params = params.replace(
                task_jac_inv=jac_inv, task_bias=bias, task_noise_map=noise_map
            )
        t0 = time.perf_counter()
        params, rollouts = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)

        # After optimize and before the substep loop, so the recorder draws
        # this step's plan into every frame of the step it belongs to --
        # the same placement, and the same overlay, the ADMM path uses.
        if recorder is not None and (show_samples or show_optimal):
            recorder.set_plans(
                traces_for(
                    robot_chosen=(
                        np.asarray(jit_trace(mjx_data, params))
                        if jit_trace is not None
                        else None
                    ),
                    robot_samples=(
                        np.asarray(rollouts.trace_sites)[:, :, 0, :]
                        if show_samples
                        else None
                    ),
                )
            )

        tq = (
            jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
            + mj_data.time
        )
        us = np.asarray(jit_interp_func(tq, params.tk, params.mean[None, ...]))[
            0
        ]
        # C3-on-xarm6 emits a planar EE velocity, not a joint command. The
        # operational-space controller maps it to arm TORQUES each substep
        # (holding tip height + tilt); the older kinematic path IK-maps it to
        # joint velocities. Any other controller's action already equals ctrl.
        emits_ee_vel = getattr(ctrl, "emits_ee_velocity", False)
        for i in range(sim_steps_per_replan):
            if arm_torque_osc:
                mj_data.ctrl[:] = _arm_osc_torque(
                    task, mj_model, mj_data, us[i],
                    np.asarray(params.samp.last_force), osc_gains
                )
            elif emits_ee_vel:
                mj_data.ctrl[:] = _arm_ctrl_from_ee_vel(
                    task, mj_model, mj_data, us[i],
                    alpha=getattr(ctrl, "task_space_alpha", 5.0),
                    damping=getattr(ctrl, "task_space_damping", 1e-4),
                )
            else:
                mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)
            if recorder is not None:
                recorder.capture(mj_data)

        # See the identical comment at this same point in _run_admm above
        # -- mj_step leaves mj_data.contact one substep stale relative to
        # the qpos it just integrated to, so a contact only entered on
        # the final substep reads as 0 force without this.
        mujoco.mj_forward(mj_model, mj_data)
        block_pose = log_step(log, task, mj_data, params, us, admm=False)
        pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
        theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
        log["pos_err"].append(pos_err)
        log["theta_err"].append(theta_err)
        if verbose:
            print(
                f"step {step:4d}  pos_err={pos_err:.4f}  "
                f"theta_err={theta_err:.4f}"
            )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

        # Detect-and-kick: MPPI's softmax-weighted mean update can settle
        # into a blend that commits to neither "found the contact angle
        # that breaks stiction" nor "stay put" once the cost gap between
        # those outcomes shrinks near the goal. Unlike ADMM's object-block
        # fix, there is no wrench to rescale here -- the action is joint
        # velocities, with no direct, invertible map to a desired contact
        # force -- so this perturbs the sampling distribution's mean
        # directly instead, to push more of the next population toward a
        # genuinely different contact angle rather than more noise around
        # the same stuck one.
        if stuck_kick_steps > 0:
            no_progress = (
                prev_pos_err is not None
                and abs(pos_err - prev_pos_err) < stuck_eps
                and abs(theta_err - prev_theta_err) < stuck_eps
            )
            stuck_count = stuck_count + 1 if no_progress else 0
            if stuck_count >= stuck_kick_steps:
                kick_rng, rng = jax.random.split(params.rng)
                kick = stuck_kick_scale * jax.random.normal(
                    kick_rng, params.mean.shape
                )
                params = params.replace(mean=params.mean + kick, rng=rng)
                stuck_count = 0
                if verbose:
                    print(f"step {step:4d}  stuck -- kicked")
            prev_pos_err, prev_theta_err = pos_err, theta_err

    return finalize_log(log, task, reached, show_plans=False, admm=False)
