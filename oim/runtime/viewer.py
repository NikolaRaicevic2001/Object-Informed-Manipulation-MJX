import os
import time
from typing import Any, Callable, Dict, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from oim import ROOT
from oim.alg_base import SamplingBasedController, quiet_mjx_cast_overflow
from oim.objects import wrap_angle
from oim.runtime.logs import finalize_log, init_log, local_goal_marker, log_step
from oim.runtime.overlay import PlanOverlay, traces_for
from oim.runtime.video import VideoRecorder

"""
Tools for deterministic (synchronous) simulation, with the simulator and
controller running one after the other in the same thread.
"""


def run_interactive(  # noqa: PLR0912, PLR0915
    controller: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    initial_knots: jax.Array = None,
    fixed_camera_id: int = None,
    show_traces: bool = True,
    max_traces: int = 5,
    trace_width: float = 5.0,
    trace_color: Sequence = [1.0, 1.0, 1.0, 0.1],
    reference: np.ndarray = None,
    reference_fps: float = 30.0,
    record_video: bool = False,
    recording_prefix: str = "simulation",
    recording_name: Optional[str] = None,
    show_samples: bool = False,
    show_optimal: bool = False,
    terminate_fn: Optional[Callable[[mujoco.MjData], bool]] = None,
) -> Optional[Dict[str, Any]]:
    """Run an interactive simulation with the MPC controller.

    This is a deterministic simulation, with the controller and simulation
    running in the same thread. This is useful for repeatability, but is less
    realistic than asynchronous simulation.

    Note: the actual control frequency may be slightly different than what is
    requested, because the control period must be an integer multiple of the
    simulation time step.

    Args:
        controller: The controller instance, which includes the task
                    (e.g., model, cost) definition.
        mj_model: The MuJoCo model for the system to use for simulation. Could
                  be slightly different from the model used by the controller.
        mj_data: A MuJoCo data object containing the initial system state.
        frequency: The requested control frequency (Hz) for replanning.
        initial_knots: The initial knot points for the control spline at t=0
        fixed_camera_id: The camera ID to use for the fixed camera view.
        show_traces: Whether to show traces for the site positions.
        max_traces: The maximum number of traces to show at once.
        trace_width: The width of the trace lines (in pixels).
        trace_color: The RGBA color of the trace lines.
        reference: The reference trajectory (qs) to visualize.
        reference_fps: The frame rate of the reference trajectory.
        record_video: Whether to record a video of the simulation.
        recording_prefix: Filename prefix for the recording, before the
            timestamp. Defaults to "simulation" for callers that don't
            care; pass something identifying the task and method for a
            more informative filename. Ignored when `recording_name` is set.
        recording_name: Exact mp4 stem (no extension), shared with the run
            file / plot so live artifacts share one timestamp. Prefer this
            over `recording_prefix` from `experiment.main`.
        show_samples: Overlay the controller's sampled candidate rollouts
            (see `oim.runtime.overlay`). Works for any sampling-based
            controller: a flat one draws its single robot-space population,
            ADMM draws its object and robot blocks separately.
        show_optimal: Overlay the chosen trajectory, thicker. Independent
            of `show_samples` -- either, both, or neither. Both off by
            default: when both are off, nothing here runs at all, so it
            cannot affect timing.
        terminate_fn: Called with the simulator state after each control
            step; returning True closes the viewer and returns. `None`
            (the default) runs until the window is closed, which is what
            every task without a notion of "done" wants -- a walker or a
            humanoid is never finished. The headless runners in
            `oim.worlds.sim3d.run` have this test built in against the
            task's goal
            tolerances; this is the same idea for the viewer, kept as an
            injected predicate so this function stays generic over tasks.

    Returns:
        For PushT-like tasks (those exposing `_block_pose` / `goal` /
        `tilt_angle`), the same log dict the headless runners return, so
        the caller can write a run file and plot. `None` for tasks that
        have no such log (pendulum, walker, …).
    """
    show_plans = show_samples or show_optimal
    # ADMM has a second block, in object-pose space, that no flat
    # controller has; everything else is drawn from the generic interface.
    has_blocks = hasattr(controller, "nominal_plans")
    task = controller.task
    # Same series the headless runners write -- only when the task can
    # supply them. Legacy Hydrax demos keep returning None.
    can_log = all(
        hasattr(task, name)
        for name in ("_block_pose", "goal", "tilt_angle", "block_dofs", "dt")
    )
    is_admm = has_blocks
    # Report the planning horizon in seconds for debugging
    print(
        f"Planning with {controller.ctrl_steps} steps "
        f"over a {controller.plan_horizon} second horizon "
        f"with {controller.num_knots} knots."
    )

    # Figure out how many sim steps to run before replanning
    replan_period = 1.0 / frequency
    sim_steps_per_replan = int(replan_period / mj_model.opt.timestep)
    sim_steps_per_replan = max(sim_steps_per_replan, 1)
    step_dt = sim_steps_per_replan * mj_model.opt.timestep
    actual_frequency = 1.0 / step_dt
    print(
        f"Planning at {actual_frequency} Hz, "
        f"simulating at {1.0 / mj_model.opt.timestep} Hz"
    )

    # Create a data structure for the controller to run rollouts from.
    mjx_data = task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )
    # site_xpos etc. before the first log entry (fresh mjx.Data has none).
    with quiet_mjx_cast_overflow():
        mjx_data = mjx.forward(task.model, mjx_data)

    # Initialize the controller
    policy_params = controller.init_params(initial_knots=initial_knots)
    jit_optimize = jax.jit(controller.optimize)
    jit_interp_func = jax.jit(controller.interp_func)

    # The object block's horizon endpoint, as a ghost object -- ADMM only
    # (a flat controller has no object block to read it from) and only in
    # scenes declaring the marker; `local_goal_marker` resolves both and
    # hands back a no-op otherwise.
    draw_local_goal = local_goal_marker(controller, mj_model)

    log: Optional[Dict[str, Any]] = None
    reached = False
    if can_log:
        log = init_log(
            task, mj_data, mjx_data, show_plans=False, admm=is_admm
        )
        goal = np.asarray(task.goal)

    # Warm-up the controller
    print("Jitting the controller...")
    st = time.time()
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)
    policy_params, rollouts = jit_optimize(mjx_data, policy_params)

    tq = jnp.arange(0, sim_steps_per_replan) * mj_model.opt.timestep
    tk = policy_params.tk
    knots = policy_params.mean[None, ...]
    _ = jit_interp_func(tq, tk, knots)
    _ = jit_interp_func(tq, tk, knots)
    print(f"Time to jit: {time.time() - st:.3f} seconds")
    num_traces = min(rollouts.controls.shape[1], max_traces)

    # Ghost reference setup
    if reference is not None:
        ref_data = mujoco.MjData(mj_model)
        assert reference.shape[1] == mj_model.nq
        ref_data.qpos[:] = reference[0, :]
        mujoco.mj_forward(mj_model, ref_data)

        vopt = mujoco.MjvOption()
        vopt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True  # Transparent.
        pert = mujoco.MjvPerturb()
        catmask = mujoco.mjtCatBit.mjCAT_DYNAMIC  # only show dynamic bodies

    # Initialize video recording if enabled
    recorder = None
    if record_video:
        # Video dimensions
        width, height = 720, 480
        # Create the video recorder
        recorder = VideoRecorder(
            output_dir=os.path.join(ROOT, "recordings"),
            width=width,
            height=height,
            fps=actual_frequency,
            prefix=recording_prefix,
            filename=recording_name,
        )
        # Ensure model visual offscreen buffer is compatible with video
        # recording
        mj_model.vis.global_.offwidth = width
        mj_model.vis.global_.offheight = height
        if not recorder.start():
            record_video = False
        renderer = mujoco.Renderer(mj_model, height=height, width=width)

    # Start the simulation
    with mujoco.viewer.launch_passive(mj_model, mj_data) as viewer:
        if fixed_camera_id is not None:
            # Set the custom camera
            viewer.cam.fixedcamid = fixed_camera_id
            viewer.cam.type = 2

        # Set up rollout traces
        if show_traces:
            num_trace_sites = len(controller.task.trace_site_ids)
            for i in range(
                num_trace_sites * num_traces * controller.ctrl_steps
            ):
                mujoco.mjv_initGeom(
                    viewer.user_scn.geoms[i],
                    type=mujoco.mjtGeom.mjGEOM_LINE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(trace_color),
                )
                viewer.user_scn.ngeom += 1

        # Give the overlay a fixed slot after the traces', so the two claim
        # disjoint geoms and can be shown together. user_scn persists
        # between frames, so the slot is reserved once here.
        overlay = None
        if show_plans:
            overlay = PlanOverlay(
                horizon=controller.ctrl_steps,
                # ADMM draws three paths: both blocks' predictions for the
                # object, plus the end-effector's own. A flat controller
                # has only the last.
                max_blocks=3 if has_blocks else 1,
            )
            overlay_base = viewer.user_scn.ngeom
            jit_plans = (
                jax.jit(controller.nominal_plans)
                if has_blocks
                else jax.jit(controller.nominal_trace)
            )

        # Add geometry for the ghost reference
        if reference is not None:
            mujoco.mjv_addGeoms(
                mj_model, ref_data, vopt, pert, catmask, viewer.user_scn
            )

        while viewer.is_running():
            start_time = time.time()

            # Set the start state for the controller
            mjx_data = mjx_data.replace(
                qpos=jnp.array(mj_data.qpos),
                qvel=jnp.array(mj_data.qvel),
                mocap_pos=jnp.array(mj_data.mocap_pos),
                mocap_quat=jnp.array(mj_data.mocap_quat),
                time=mj_data.time,
            )

            # Do a replanning step
            plan_start = time.time()
            policy_params, rollouts = jit_optimize(mjx_data, policy_params)
            plan_time = time.time() - plan_start
            if log is not None:
                # Match headless: wall time of optimize only, after device sync.
                jax.block_until_ready(policy_params)
                log["compute_time"].append(time.time() - plan_start)

            # Visualize the rollouts
            if show_traces:
                ii = 0
                for k in range(num_trace_sites):
                    for i in range(num_traces):
                        for j in range(controller.ctrl_steps):
                            mujoco.mjv_connector(
                                viewer.user_scn.geoms[ii],
                                mujoco.mjtGeom.mjGEOM_LINE,
                                trace_width,
                                rollouts.trace_sites[i, j, k],
                                rollouts.trace_sites[i, j + 1, k],
                            )
                            ii += 1

            # Overlay each block's sampled and/or chosen trajectories (the
            # robot samples come from the same `rollouts` this step's
            # `show_traces` block, if enabled, also reads).
            # Bound before the branch: the ghost update below reads it
            # whether or not an overlay is being drawn.
            object_plan = None
            if overlay is not None:
                robot_plan = None
                if has_blocks:
                    object_plan, robot_plan, robot_trace = jit_plans(
                        mjx_data, policy_params
                    )
                else:
                    robot_trace = jit_plans(mjx_data, policy_params)
                overlay.draw(
                    viewer.user_scn,
                    traces_for(
                        robot_chosen=(
                            np.asarray(robot_trace) if show_optimal else None
                        ),
                        object_chosen=(
                            np.asarray(object_plan)
                            if show_optimal and object_plan is not None
                            else None
                        ),
                        # What the robot block's controls would do to the
                        # object, against what the object block asked for.
                        robot_object_chosen=(
                            np.asarray(robot_plan)
                            if show_optimal and robot_plan is not None
                            else None
                        ),
                        robot_samples=(
                            np.asarray(rollouts.trace_sites)[:, :, 0, :]
                            if show_samples
                            else None
                        ),
                        object_samples=(
                            np.asarray(policy_params.object_samples)
                            if show_samples and has_blocks
                            else None
                        ),
                    ),
                    base=overlay_base,
                )

            # Before the substeps below, so the ghost is in place for the
            # viewer syncs and recorded frames of the step whose plan
            # produced it. Fed `object_plan` when the overlay already
            # computed one -- the marker resolves that same array, so
            # asking the controller separately would roll the object block
            # out twice per step. The whole plan, not its endpoint: under
            # pure pursuit the target sits partway along it. See
            # `oim.runtime.logs.local_goal_marker`.
            draw_local_goal(
                mj_data,
                mjx_data,
                policy_params,
                None if object_plan is None else np.asarray(object_plan),
            )

            # Update the ghost reference
            if reference is not None:
                t_ref = mj_data.time * reference_fps
                i_ref = int(t_ref)
                i_ref = min(i_ref, reference.shape[0] - 1)
                ref_data.qpos[:] = reference[i_ref]
                mujoco.mj_forward(mj_model, ref_data)
                mujoco.mjv_updateScene(
                    mj_model,
                    ref_data,
                    vopt,
                    pert,
                    viewer.cam,
                    catmask,
                    viewer.user_scn,
                )

            # query the control spline at the sim frequency
            # (we assume the sim freq is the same as the low-level ctrl freq)
            sim_dt = mj_model.opt.timestep
            t_curr = mj_data.time

            tq = jnp.arange(0, sim_steps_per_replan) * sim_dt + t_curr
            tk = policy_params.tk
            knots = policy_params.mean[None, ...]
            us = np.asarray(jit_interp_func(tq, tk, knots))[0]  # (ss, nu)

            # simulate the system between spline replanning steps
            for i in range(sim_steps_per_replan):
                mj_data.ctrl[:] = np.array(us[i])
                mujoco.mj_step(mj_model, mj_data)
                viewer.sync()

                # Capture frame if recording
                if record_video and recorder.is_recording:
                    renderer.update_scene(mj_data, viewer.cam)
                    frame = renderer.render()
                    recorder.add_frame(frame.tobytes())

            # Checked after the substeps, so the state tested is the one the
            # step actually produced -- the same placement as the headless
            # runners' own goal test.
            if log is not None:
                block_pose = log_step(
                    log, task, mj_data, policy_params, us, admm=is_admm
                )
                pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
                theta_err = float(
                    abs(float(wrap_angle(block_pose[2] - goal[2])))
                )
                log["pos_err"].append(pos_err)
                log["theta_err"].append(theta_err)

            if terminate_fn is not None and terminate_fn(mj_data):
                print(f"\ngoal reached at t = {mj_data.time:.2f}s")
                reached = True
                break

            # Try to run in roughly realtime
            elapsed = time.time() - start_time
            if elapsed < step_dt:
                time.sleep(step_dt - elapsed)

            # Print some timing information
            rtr = step_dt / (time.time() - start_time)
            print(
                f"Realtime rate: {rtr:.2f}, plan time: {plan_time:.4f}s",
                end="\r",
            )

    # Preserve the last printout
    print("")

    # Close the video recorder if recording was enabled
    if record_video and recorder is not None:
        recorder.stop()

    if log is None:
        return None
    return finalize_log(
        log, task, reached, show_plans=False, admm=is_admm
    )
