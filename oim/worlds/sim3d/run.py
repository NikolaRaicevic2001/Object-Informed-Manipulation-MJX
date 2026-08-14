"""Headless closed-loop driver for the MJX/MuJoCo world, with ADMM logging.

The 3D counterpart of `oim.worlds.sim2d.run.run_2d`: steps the real
`mujoco.MjData` and the MJX planning model in lockstep, returning the same
kind of log dict
`run_2d` does (trajectories, wrenches, primal/dual residuals, goal errors),
for `oim.tasks.pusht.PushT` under either `robot` embodiment. Headless -- no
viewer -- reuses `oim.runtime.viewer.run_interactive`'s stepping logic,
but that function is generic over any controller/task and has no way to
return a log, which is what this fills in for the ADMM+PushT case
specifically.

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

from oim.alg_base import SamplingBasedController
from oim.algs.admm import ADMM
from oim.objects import wrap_angle
from oim.runtime.logs import finalize_log, init_log, local_goal_marker, log_step
from oim.runtime.overlay import PlanOverlay, traces_for
from oim.runtime.video import OffscreenRecorder
from oim.tasks.pusht import PushT


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
    # Three paths, not two: both blocks' predictions for the object, plus
    # the end-effector's own. See `oim.runtime.overlay`.
    overlay = (
        PlanOverlay(horizon=ctrl.ctrl_steps, max_blocks=3)
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
        )
    finally:
        if recorder is not None:
            recorder.close()


# A path shorter than this spans well under a pixel at any sane camera
# distance, so it is drawn but cannot be seen. Reported rather than left
# silent: an overlay that renders nothing looks identical to an overlay
# that was never switched on, and telling those apart by eye cost a while.
_VISIBLE_SPAN_M = 5e-3


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

        # Move the ghost before the substeps, so every frame of the step
        # shows the endpoint that step's plan was scored against. Outside
        # the `compute_time` measurement above on purpose -- it is
        # visualization, and folding it in would depress the reported
        # planning rate.
        draw_local_goal(mj_data, mjx_data, params)

        # After optimize (the plans come from the params it just produced,
        # and the samples from the rollouts that produced them) and before
        # the substep loop (the recorder draws them into every frame of the
        # step they belong to).
        if jit_plans is not None:
            object_plan, robot_plan, robot_trace = jit_plans(mjx_data, params)
            object_plan = np.asarray(object_plan)
            robot_plan = np.asarray(robot_plan)
            robot_trace = np.asarray(robot_trace)
            log["object_plan"].append(object_plan)
            log["robot_plan"].append(robot_plan)
            if step == 0 and verbose:
                _report_plan_spans(object_plan, robot_plan, robot_trace)
            if recorder is not None:
                recorder.set_plans(
                    traces_for(
                        robot_chosen=robot_trace if show_optimal else None,
                        object_chosen=object_plan if show_optimal else None,
                        # The same object as `object_chosen`, under the
                        # other block's plan -- their separation is the
                        # consensus disagreement drawn rather than summed.
                        robot_object_chosen=(
                            robot_plan if show_optimal else None
                        ),
                        # trace_sites: (num_samples, H+1, num_trace_sites,
                        # 3) -- this task has exactly one trace site (the
                        # pusher tip).
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
                    )
                )

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
    mjx_data = mjx.forward(task.model, mjx_data)

    log = init_log(task, mj_data, mjx_data, show_plans=False, admm=False)
    reached = False
    goal = np.asarray(task.goal)

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
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)
            if recorder is not None:
                recorder.capture(mj_data)

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

    return finalize_log(log, task, reached, show_plans=False, admm=False)
