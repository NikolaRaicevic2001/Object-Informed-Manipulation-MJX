"""Headless closed-loop driver for the MJX/MuJoCo world, with ADMM logging.

The 3D counterpart of `oim.sim2d.run.run_2d`: steps the real `mujoco.MjData`
and the MJX planning model in lockstep, returning the same kind of log dict
`run_2d` does (trajectories, wrenches, primal/dual residuals, goal errors),
for `oim.tasks.pusht.PushT` under either `robot` embodiment. Headless -- no
viewer -- reuses `oim.sim3d.deterministic.run_interactive`'s stepping logic,
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim.alg_base import SamplingBasedController
from oim.algs.admm import ADMM
from oim.objects import wrap_angle
from oim.sim3d.plan_overlay import BlockTrace, PlanOverlay, traces_for
from oim.tasks.pusht import PushT
from oim.utils.video import VideoRecorder


class _OffscreenRecorder:
    """Renders `mujoco.MjData` to an mp4 with no viewer and no display.

    Frames are strided so playback is real time: the simulator produces
    `1/timestep` frames per simulated second (100 at the push-T timestep),
    far more than a video needs, so every `stride`-th one is kept and the
    encoder is told the matching frame rate. Recording a frame per physics
    step and encoding at the *replanning* rate -- which is what
    `run_interactive` does -- yields a video that plays back
    `sim_steps_per_replan` times slower than reality.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        output_dir: str,
        base_name: str,
        target_fps: float,
        size: Tuple[int, int],
        camera: Optional[Union[str, int]],
        overlay: Optional[PlanOverlay] = None,
    ) -> None:
        """Set up the offscreen renderer and start the encoder.

        Args:
            mj_model: The execution model to render.
            output_dir: Directory for the mp4.
            base_name: Filename stem, shared with the run's plot/results.
            target_fps: Desired playback frame rate; the achieved rate is
                the nearest one an integer stride allows.
            size: (width, height) in pixels.
            camera: Model camera name or id. None uses the default free
                camera, which frames the whole scene.
            overlay: If given, the ADMM plan overlay to composite into every
                frame. Feed it with `set_plans` once per control step.

        Raises:
            RuntimeError: If no OpenGL backend is available.
        """
        width, height = size
        step_fps = 1.0 / mj_model.opt.timestep
        self.stride = max(1, round(step_fps / target_fps))
        self._i = 0

        # Must be set before the Renderer allocates its buffer.
        mj_model.vis.global_.offwidth = max(
            width, mj_model.vis.global_.offwidth
        )
        mj_model.vis.global_.offheight = max(
            height, mj_model.vis.global_.offheight
        )
        try:
            self.renderer = mujoco.Renderer(
                mj_model, height=height, width=width
            )
        except Exception as e:  # no GL context available
            raise RuntimeError(
                f"Could not create an offscreen renderer ({e}). On a machine "
                "with no display, set MUJOCO_GL=egl (or osmesa for software "
                "rendering) before running."
            ) from e

        self.camera = mujoco.MjvCamera()
        if camera is None:
            mujoco.mjv_defaultFreeCamera(mj_model, self.camera)
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = (
                camera
                if isinstance(camera, int)
                else mujoco.mj_name2id(
                    mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                )
            )
            if self.camera.fixedcamid < 0:
                raise ValueError(f"No camera named {camera!r} in the model")

        self.recorder = VideoRecorder(
            output_dir=output_dir,
            width=width,
            height=height,
            fps=step_fps / self.stride,
            filename=base_name,
        )
        self.active = self.recorder.start()
        self.overlay = overlay
        self._traces: List[BlockTrace] = []

    def set_plans(self, traces: Sequence[BlockTrace]) -> None:
        """Hold the blocks to composite into subsequent frames.

        Called once per control step, while frames are captured once per
        physics step -- so the same plans are drawn across the substeps they
        were computed for, which is exactly their period of validity.

        Args:
            traces: One `BlockTrace` per block, from
                `oim.sim3d.plan_overlay.traces_for`.
        """
        self._traces = list(traces)

    def capture(self, mj_data: mujoco.MjData) -> None:
        """Render and encode one frame, if this physics step is on-stride."""
        on_stride = self._i % self.stride == 0
        self._i += 1
        if not (self.active and on_stride):
            return
        self.renderer.update_scene(mj_data, self.camera)
        # After update_scene, not before: it rebuilds the scene from the
        # model and resets ngeom, discarding anything added earlier. This
        # is what puts the trajectories in the mp4 as well as the viewer.
        if self.overlay is not None and self._traces:
            self.overlay.draw(self.renderer.scene, self._traces)
        self.recorder.add_frame(self.renderer.render().tobytes())

    def close(self) -> None:
        """Finalize the mp4 and release the GL context."""
        if self.active:
            self.recorder.stop()
            self.active = False
        self.renderer.close()


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
    # the end-effector's own. See `oim.sim3d.plan_overlay`.
    overlay = (
        PlanOverlay(horizon=ctrl.ctrl_steps, max_blocks=3)
        if show_plans
        else None
    )
    recorder = None
    if record_dir is not None:
        if record_name is None:
            raise ValueError("record_dir requires record_name")
        recorder = _OffscreenRecorder(
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


def _init_log(
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
    }
    if admm:
        # Meaningless for a flat controller: no consensus, no residuals.
        log.update(
            wrench=[],
            wrench_consensus=[],
            primal_residual=[],
            dual_residual=[],
            rho=[],
        )
    if show_plans:
        # Only allocated when asked for: (H, 3) per block per step is a
        # different order of magnitude from the rest of the log.
        log["object_plan"] = []
        log["robot_plan"] = []
    return log


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
    recorder: Optional[_OffscreenRecorder],
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

    log = _init_log(task, mj_data, mjx_data, show_plans)
    jit_plans = jax.jit(ctrl.nominal_plans) if show_plans else None
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

        block_pose = _log_step(log, task, mj_data, params, us)

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

    return _finalize_log(log, task, reached, show_plans)


def _log_step(
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
    if admm:
        log["wrench"].append(np.array(task.realized_consensus(mj_data)))
        log["wrench_consensus"].append(np.array(params.z[0]))
        log["primal_residual"].append(float(params.primal_residual))
        log["dual_residual"].append(float(params.dual_residual))
        # `rho` is a scalar (paper's Algorithm 4) or a per-dimension vector
        # (force/torque split, see `WrenchConsensus.penalty_cost`); the
        # log keeps one number, so a vector logs its mean.
        log["rho"].append(float(np.mean(np.asarray(params.rho))))
    return block_pose


def _finalize_log(
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
        *(("wrench_consensus",) if admm else ()),
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
    log["robot_vel"] = _finite_difference(log["robot_pos"], task.dt)
    return log


def _finite_difference(series: np.ndarray, dt: float) -> np.ndarray:
    """Per-step velocity of a logged position series, same length as it.

    The first entry is zero (nothing precedes it); entry `i` thereafter is
    the average velocity over the step that produced it.
    """
    if len(series) < 2:
        return np.zeros_like(series)
    return np.vstack([np.zeros_like(series[:1]), np.diff(series, axis=0) / dt])


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
        recorder = _OffscreenRecorder(
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
    recorder: Optional[_OffscreenRecorder],
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

    mjx_data = task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )
    mjx_data = mjx.forward(task.model, mjx_data)

    log = _init_log(task, mj_data, mjx_data, show_plans=False, admm=False)
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

        block_pose = _log_step(log, task, mj_data, params, us, admm=False)
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

    return _finalize_log(log, task, reached, show_plans=False, admm=False)
