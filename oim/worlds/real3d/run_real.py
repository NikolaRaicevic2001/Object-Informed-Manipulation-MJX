"""Hardware closed-loop driver: the real-robot counterpart of `sim3d/run.py`.

The planner (`ADMM.optimize`), the task cost and the MJX rollouts are reused
unchanged -- MJX is still the planner's internal predictive model on real
hardware; the real world only replaces *execution* and *state*:

    sim3d._run                       real3d.run_real
    ----------------------------     --------------------------------------
    mjx_data <- mj_data.qpos/qvel    mjx_data <- interface.read_state()
    mj_data.ctrl = u ; mj_step(...)  interface.send_velocity(u)

The planner is a plain jitted JAX function, so it is called directly in this
process -- no zerorpc, no separate planner server.

REAL-TIME MODEL. A solve is longer than the planning timestep, so a
single-thread "solve, then publish" loop leaves the arm stalled during every
solve. For hardware (`real_time=True`) the two are split: the main thread
reads state, solves and posts the plan; a publisher thread samples that plan
at `control_rate` and publishes one velocity command. They overlap, and the
plan horizon (0.75 s) must exceed the solve time for a valid sample to
always exist -- see the README's real-time section. We stay on the velocity
topic because the CBF filter only sits there.

The state log uses the same keys/schema as `sim3d/run.py`, so a hardware run
and a simulation run compare entry-for-entry.

WHAT IS WHERE. This module is the setup: resolve the configuration, compile
the jitted entry points, build the log and the rendering destinations, and
hand the lot to a loop as one `LoopContext`. The parts it assembles live
next to it, each answering one question:

    build.py         the task, the controller and the interface
    loops.py         the two control loops, and the context they share
    diagnostics.py   the sample/contact statistics and the cost split
    reconstruct.py   the log series rebuilt after the loop, not during it
    timing.py        loop clocks, transport stamps, the execution window
    visualize.py     camera framing, overlay and recorder
    kicker.py        the no-progress kick
    calibration.py   moving obstacles to where the cameras say they are
"""

from __future__ import annotations

import dataclasses
import functools
import math
import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from oim.runtime.logs import init_log, object_plan_marker
from oim.runtime.mjcf import hide_body_geoms, mocap_id
from oim.runtime.overlay import PlanOverlay
from oim.runtime.video import OffscreenRecorder
from oim.tasks.pusht import PushT
from oim.worlds.real3d.calibration import (
    _OBSTACLE_NAMES,
    _load_obstacle_calibration,
    apply_obstacle_calibration,
    apply_obstacle_calibration_to_planner,
)
from oim.worlds.real3d.diagnostics import (
    _CONTACT_STAT_KEYS,
    _OBJECT_STAT_KEYS,
    _SAMPLE_STAT_KEYS,
    _cost_terms_jnp,
    _init_cost_terms,
    _init_sample_stats,
    _StatsReducer,
)
from oim.worlds.real3d.interface import RobotWorldInterface, SceneAddresses
from oim.worlds.real3d.kicker import _StuckKicker
from oim.worlds.real3d.loops import (
    LoopContext,
    _assemble_state,
    _run_overlapped,
    _run_serial,
)
from oim.worlds.real3d.timing import execution_window
from oim.worlds.real3d.visualize import (
    _VIEW_AZIMUTH,
    _VIEW_ELEVATION,
    _VIEW_FALLBACK_ASPECT,
    _frame_table,
)

# Forward kinematics for the assembled state, JIT-compiled once and reused --
# calling mjx.forward un-jitted every control step dispatches thousands of tiny
# GPU kernels eagerly (~150 s/step); jitted it is milliseconds.
_jit_forward = jax.jit(mjx.forward)










































































def run_real(
    task: PushT,
    ctrl: Any,  # ADMM
    params: Any,
    interface: RobotWorldInterface,
    replan_rate: float = 2.5,
    control_rate: float = 50.0,
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    real_time: bool = False,
    vel_limit: float = 0.2,
    admm: bool = True,
    verbose: bool = True,
    preflight: float = 0.0,
    preflight_min_fps: float = 5.0,
    record_dir: Optional[str] = None,
    record_name: Optional[str] = None,
    video_fps: float = 30.0,
    video_size: Tuple[int, int] = (720, 480),
    camera: Optional[Union[str, int]] = None,
    live: bool = False,
    show_samples: bool = True,
    show_optimal: bool = True,
    show_object_plan: bool = False,
    obstacle_calibration: Optional[str] = None,
    view_azimuth: float = _VIEW_AZIMUTH,
    view_elevation: float = _VIEW_ELEVATION,
    view_distance: Optional[float] = None,
    latency_comp: float = 0.0,
    print_every: int = 10,
    handoff: str = "responsive",
    t_c: float = 0.5,
    actuation_delay: float = 0.0,
) -> Dict[str, Any]:
    """Run the push-T ADMM controller against a `RobotWorldInterface`.

    Args:
        task: the `PushT` task, built with `robot="xarm6"`.
        ctrl: the ADMM controller, built against `task`.
        params: initial policy parameters (`ctrl.init_params(...)`).
        interface: hardware or mock world.
        replan_rate: mock only -- how much sim time one solve covers.
        control_rate: rate (Hz) at which velocity commands are published.
        max_steps: maximum solves.
        goal_pos_tol, goal_theta_tol: success tolerances.
        real_time: True -> hardware (threaded, overlapped); False -> mock
            (single-threaded, deterministic).
        verbose: print progress.
        preflight: hardware only -- watch the RAW FoundationPose stream for
            this many seconds (block still) after warm-up and refuse to
            send the first command on a bad fit; 0 skips the check.
        record_dir: Directory for an mp4, mirroring every sim world's
            `OffscreenRecorder`. `None` disables recording.
        record_name: Filename stem for the mp4, no extension -- pass the
            same base name used for the run's JSON/plot. Required when
            `record_dir` is given.
        video_fps: Target playback frame rate, and (see `_run_serial` /
            `_run_overlapped`) the assumed real-world seconds between one
            `capture()` call and the next -- real calls it once per
            REPLAN step, not once per physics step the way sim worlds do,
            so there is no `mj_model.opt.timestep` to derive this from.
            Matching `replan_rate` makes the mock path play back true to
            real time; hardware's true interval is the solve time itself,
            which varies step to step, so this is an approximation there.
        video_size: (width, height) of the video in pixels.
        camera: Model camera name or id to render from. `None` uses the
            default free camera, which frames the whole scene.
        live: Open a `mujoco.viewer` passive window for the run's
            duration, mirroring `oim.runtime.viewer.run_interactive`.
            Independent of `record_dir` -- either, both, or neither.
        show_samples: Overlay each block's sampled candidate rollouts, in
            whichever of `record_dir`/`live` are active. Off has zero
            cost: `_visualize_step` never runs when both this and
            `show_optimal` are off and neither destination is set.
        show_optimal: Overlay each block's chosen trajectory. With this
            and `show_object_plan` both off, the ADMM block plans are not
            rolled out during the loop even with a viewer or recorder
            open; the run file's `object_plan`/`robot_plan` are rebuilt
            after the loop instead.
        show_object_plan: Draw the object block's plan-endpoint ghost
            marker (ADMM only). Off by default: it sits on the global
            goal for most of a run and duplicates the goal marker.
        view_azimuth, view_elevation: Where `--live`'s free camera starts,
            in degrees. Azimuth 180 looks back along -x from over the
            table's +x end, elevation is negative looking down. Ignored
            when `camera` names a model camera.
        view_distance: How far back that camera stands, in metres. `None`
            solves it from the table's width and the window's aspect so
            the table just fills the frame.
        obstacle_calibration: `"live"` samples obs_1/2/3's current pose
            directly off `interface`'s own TF connection (requires a real
            `Ros2Interface` -- see `_sample_obstacle_tf_live`; the mock
            has no TF tree). Any other string is a path to a
            calibrate_obstacles.py JSON file (works with the mock, so a
            calibrated layout can be rehearsed). None keeps the MJCF's
            own hardcoded obstacle poses. Scenes without `obs_N` mocap
            bodies skip calibration entirely, whatever is passed.
        print_every: Console step summary every this many control steps;
            0 = none (the run file is unaffected; it always holds every
            step).
        latency_comp: Hardware loop only. 0 (default) keeps today's
            behaviour: every solve starts from the state read at `t_loop`,
            while the arm keeps executing the previous plan for the whole
            solve (~250-300 ms here), so the plan's first ~0.3 s is never
            executed and its rollouts branch from a state the arm has
            already left -- 20-40 mm of tip travel at the velocity limit,
            i.e. one crossbar thickness. A positive value is the initial
            guess [s] of that solve latency: the arm's joint state is then
            advanced by the plan the publisher is streaming over the next
            `latency` seconds (exact for velocity-controlled revolute
            joints -- the same dead reckoning the display thread does),
            the plan's clock is anchored at `t_loop + latency`, and the
            guess is tracked per solve as an EMA of the measured read-to-
            publish time. The object pose is held (nothing to integrate
            it with). Logged state stays the MEASURED one.

            This is HARDWARE COMPENSATION, not a change to the algorithm.
            In sim the solve is instantaneous, so a plan always starts from
            the state it was planned for; on the arm the solve takes ~0.3 s
            and the arm keeps moving, so without the prediction the plan
            starts from a state that is already gone. It restores the
            assumption sim satisfies for free rather than adding anything to
            the formulation, which is why it lives here and nothing under
            `oim/algs` reads it. Worth one sentence in the paper's
            implementation section.
        handoff: Hardware loop only; see `_run_overlapped`. ``"responsive"``
            (default, today's behaviour plus the negative-`elapsed` clamp)
            anchors on the EMA and publishes as soon as a plan is ready.
            ``"deterministic"`` anchors on the fixed `t_c` and waits for it,
            giving a constant period at the cost of latency.
        t_c: Anchor offset [s] under ``handoff="deterministic"``. Ignored
            otherwise. Must be at or above the worst-case READ-TO-PUBLISH
            time, which is the solve, not the loop period -- a `t_c` below
            it makes every solve late and the mode degenerates to
            ``"responsive"`` with a stale constant anchor.
        actuation_delay: Hardware loop only. Seconds from our publishing a
            velocity command to the arm's joint velocity following it:
            ~50 ms on the xArm6 behind the CBF (CBF pass-through ~14 ms
            plus the arm's own ~35 ms: 18 ms dead time and an
            acceleration-limited ramp; 2026-09-11 step test and probes).
            So the arm is always executing commands sent `actuation_delay`
            ago, and the state the solve should start from is the one at
            `t_publish + actuation_delay`, the instant the new plan's
            first command takes effect. Integrating the measured state to
            there covers the commands from `actuation_delay` before the
            measurement up to the publish instant (`window_split`); the
            predicted state's clock is `lat + actuation_delay` ahead of
            the read. 0 keeps the previous behaviour (measured state plus
            the commands sent from now). The record of ALREADY-sent
            commands comes from the interface's `executed_stream()` when
            it has one (the CBF's output, so whatever the safety filter
            changed is integrated as executed) and from the publisher's
            own send log otherwise. With the executed record the CBF's
            ~14 ms is counted on the wrong side of "now" (the window
            starts 14 ms early and the plan samples are attributed 14 ms
            early); the two cancel to first order and the residual is
            below 0.2 deg at the velocity limit.

    Returns:
        A log dict with the same schema as `sim3d.run.run_3d_admm`.
    """
    addresses = SceneAddresses.from_model(task.mj_model)
    control_dt = 1.0 / control_rate

    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp = jax.jit(ctrl.interp_func)
    # Only ADMM exposes nominal_plans (object/robot block plans); a flat MPPI
    # baseline has neither, so plan logging is gated on admm.
    jit_plans = jax.jit(ctrl.nominal_plans) if admm else None
    # The flat path's counterpart, for the overlay's chosen end-effector
    # path only (nothing here is logged, unlike jit_plans) -- every
    # controller has nominal_trace (oim.alg_base.SamplingBasedController),
    # ADMM's own override just reuses the rollout nominal_plans already
    # pays for. Built only when visualization can actually use it, so a
    # flat run with neither --record nor --live traces nothing extra.
    show_plans = show_samples or show_optimal
    jit_trace = (
        jax.jit(ctrl.nominal_trace)
        if (not admm and show_plans and (record_dir is not None or live))
        else None
    )

    # First state + JIT warm-up before any timed loop.
    t = time.perf_counter()
    base_data = task.make_data()
    # The goal ghost marker is a mocap body placed by the scene file; when
    # the run overrides the goal (`PushT(goal=...)`) move the marker with
    # it so the viewer, the recording and the logged mocap agree with the
    # pose actually being scored.
    _gid = mocap_id(task.mj_model, "goal")
    if _gid >= 0:
        _g = np.asarray(task.goal, dtype=float)
        _mp = np.array(base_data.mocap_pos, copy=True)
        _mq = np.array(base_data.mocap_quat, copy=True)
        _mp[_gid, :2] = _g[:2]
        _mq[_gid] = [math.cos(_g[2] / 2), 0.0, 0.0, math.sin(_g[2] / 2)]
        base_data = base_data.replace(
            mocap_pos=jnp.asarray(_mp), mocap_quat=jnp.asarray(_mq)
        )
    if obstacle_calibration is not None and not any(
        mocap_id(task.mj_model, n) >= 0 for n in _OBSTACLE_NAMES
    ):
        # Only box_clutter_real declares obs_N as MOCAP bodies, which is
        # what apply_obstacle_calibration can actually write to. Bail out
        # before sampling, for two reasons:
        #
        #   Cost. "live" spends ~1.5 s per obstacle on TF, and on a scene
        #   with no mocap obstacle every result is discarded anyway.
        #
        #   Correctness. single_obstacle_real keeps obs_1 as a plain
        #   worldbody geom but DOES carry a planner Box at shapes[0]. The
        #   two appliers would then disagree: apply_obstacle_calibration
        #   skips it (mocap_id < 0, geom stays at the MJCF pose) while
        #   apply_obstacle_calibration_to_planner happily moves shapes[0]
        #   to the detected pose -- avoidance cost centred somewhere the
        #   collision geometry is not.
        if verbose:
            print("[calibration] no obs_N mocap body in this scene -- "
                  "skipping calibration entirely (only box_clutter_real "
                  "declares obs_N as mocap)")
        obstacle_calibration = None
    if obstacle_calibration is not None:
        # Loaded once (a live sample takes ~1.5s/obstacle) and reused for
        # both destinations, rather than each re-sampling TF independently.
        calibration = _load_obstacle_calibration(
            obstacle_calibration, interface, verbose
        )
        base_data = apply_obstacle_calibration(
            task, base_data, calibration, verbose=verbose
        )
        apply_obstacle_calibration_to_planner(
            task, calibration, verbose=verbose
        )
    world0 = interface.read_state()
    mjx_data = _assemble_state(task, base_data, addresses, world0)
    if verbose:
        print(f"[jit] initial state assembled in {time.perf_counter() - t:.1f}s; "
              "warming up -- the first optimize traces + XLA-compiles the whole "
              "ADMM+MJX graph (minutes; cached across runs)...")
    # Split the two warm-up calls: the first pays compile + run, the second is
    # a warm run -- so the log shows compile time vs pure execution time.
    # Discard the output (`_warm`, not `params`): these calls exist only to
    # compile/time the graph. Keeping `params` at the caller's init_params means
    # the loop starts from the same point as the sim's run_3d_admm (which never
    # pre-optimizes), so their first control matches.
    t = time.perf_counter()
    _warm, _ = jit_optimize(mjx_data, params)
    jax.block_until_ready(_warm)
    if verbose:
        print(f"[jit] optimize compiled + first run: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    _warm, _ = jit_optimize(mjx_data, params)
    jax.block_until_ready(_warm)
    if verbose:
        print(f"[jit] optimize warm run: {time.perf_counter() - t:.3f}s "
              "(this is the real per-step cost)")

    _ = jit_interp(jnp.array([world0.time]), _warm.tk, _warm.mean[None, ...])
    jax.block_until_ready(_warm)

    # Warm up on the real loop path as well: the two calls above reuse one
    # `mjx_data`, so the first solve that actually goes read_state ->
    # _assemble_state -> optimize pays a one-off cost the loop should not
    # (1.7 s at 16 samples, 6.9 s at 64 -- it scales with num_samples, so it
    # looks like an allocation, not a recompile). Do it here, where the
    # publisher has not started and the arm is still.
    t = time.perf_counter()
    _p = params
    for _ in range(3):
        _w = interface.read_state()
        _md = _assemble_state(task, base_data, addresses, _w)
        # Chain like the loop does: the one-off cost lands on the first solve
        # fed a *returned* params, not the first solve overall. _p is discarded
        # -- the loop must still start from `params`, or the pollution returns.
        _p, _ = jit_optimize(_md, _p)
        jax.block_until_ready(_p)
    # `nominal_plans` compiles on ITS first call, which used to happen inside
    # the loop's first iteration -- after the step-0 plan was already handed
    # to the publisher. The publisher exhausted the 1.6 s seed/step-0 plan
    # while that compile blocked the main thread, zero-filled, and the arm
    # did the signature twitch / ~1 s freeze / restart. (Measured on the
    # 2026-08-28 13:59 real run: step0->step1 wall gap was 3.9 s, of which
    # optimize was only 0.6 s -- the rest was this compile.) Same class of
    # bug as the stale-seed fix in `_run_overlapped`: warm every jitted
    # function the loop calls while the publisher has not started and the
    # arm is still.
    _p, _r = jit_optimize(_md, _p)
    if jit_plans is not None and (record_dir is not None or live) and (
        show_optimal or show_object_plan
    ):
        _pl = jit_plans(_md, _p)
        jax.block_until_ready(_pl)
    # The per-step statistics reduce on the device; compile that kernel
    # here too, and the cost decomposition the console print uses.
    reducer = _StatsReducer(admm, task.consensus_scale() if admm else None)
    _warm_log = {k: [] for k in (*_SAMPLE_STAT_KEYS, *_CONTACT_STAT_KEYS,
                                 *_OBJECT_STAT_KEYS)}
    reducer(_warm_log, _r, _p, np.asarray(task._block_pose(_md)))
    jit_cost_terms = jax.jit(functools.partial(_cost_terms_jnp, task))
    jax.block_until_ready(jit_cost_terms(_md))
    if verbose:
        print(f"[jit] loop-path warm-up: {time.perf_counter() - t:.1f}s")

    # FP pre-flight, hardware only, AFTER warm-up (so it grades the stream
    # closest to the first command): watch the raw pose for a few seconds
    # while the block is still, and abort on an upside-down/mirror fit, a
    # fit hopping between minima, or a floated bbox (z/tilt wobble) --
    # each of which cost a full run on 2026-08-29. Raises before any
    # command is published; the arm never moves on a FAIL.
    if real_time and preflight > 0.0:
        from .fp_preflight import preflight_gate  # noqa: PLC0415
        preflight_gate(interface, seconds=preflight,
                       min_fps=preflight_min_fps, verbose=verbose)

    if verbose:
        print(f"[jit] ready; {'overlapped' if real_time else 'serial'} loop, "
              f"control {control_rate:.0f} Hz, stream")
        # The execution window, in each unit it gets discussed in, so a mock
        # and a hardware run can be compared by reading their headers.
        # `responsive` on hardware has no fixed window -- the period is the
        # solve -- so there is nothing honest to print for it.
        if handoff == "deterministic" or not real_time:
            window = execution_window(handoff, t_c, replan_rate)
            print(f"[jit] window {window:.2f}s "
                  f"({max(1, round(window / control_dt))} ticks, "
                  f"{window / float(task.dt):.0f} planning steps), "
                  f"handoff={handoff}")
        else:
            print("[jit] window solve-paced, handoff=responsive")

    log = init_log(task, mjx_data, mjx_data, show_plans=admm, admm=admm)
    _init_sample_stats(log, admm)
    _init_cost_terms(log)

    # Three slots for ADMM (object block, robot block's object prediction,
    # end-effector path); one for a flat controller, which has no object
    # block. See oim.runtime.overlay's module docstring.
    overlay = (
        PlanOverlay(horizon=ctrl.ctrl_steps, max_blocks=4 if admm else 1)
        if show_plans and (record_dir is not None or live)
        else None
    )
    # One deepcopy, shared by every rendering destination -- never
    # task.mj_model itself, matching hide_body_geoms's own rule ("pass the
    # execution model, which is a deepcopy, not the task's own"). Built
    # only when something will actually render, so a run with neither
    # flag pays nothing.
    vis_model = (
        deepcopy(task.mj_model) if (record_dir is not None or live) else None
    )
    recorder = None
    if record_dir is not None:
        if record_name is None:
            raise ValueError("record_dir requires record_name")
        # OffscreenRecorder assumes capture() is called once per PHYSICS
        # step and strides down from mj_model.opt.timestep to hit
        # video_fps -- true for every sim world's own mj_step loop, not
        # here (see video_fps above). Overriding vis_model's timestep to
        # 1/video_fps makes the recorder keep every call (stride=1) and
        # hand VideoRecorder exactly video_fps. Harmless to the live
        # viewer/display thread sharing vis_model: neither ever steps
        # physics, so opt.timestep means nothing to them.
        vis_model.opt.timestep = 1.0 / video_fps
        recorder = OffscreenRecorder(
            vis_model, output_dir=record_dir, base_name=record_name,
            target_fps=video_fps, size=video_size, camera=camera,
            overlay=overlay,
        )
        if camera is None:
            # The same framing --live gets, so the mp4 and the window that
            # produced it are one shot rather than two. OffscreenRecorder's
            # own default is `mjv_defaultFreeCamera` (oim/runtime/video.py)
            # -- right for the sim worlds it was written for, far too wide
            # for the real table. Overwritten here rather than taught to
            # the recorder, which those sim worlds share.
            #
            # The aspect is exact here, unlike the viewer's: an mp4 is
            # `video_size`, a window is whatever the user dragged it to.
            _frame_table(
                vis_model, recorder.camera,
                video_size[0] / video_size[1],
                view_azimuth, view_elevation, view_distance,
            )
    mj_data_cpu = mujoco.MjData(vis_model) if vis_model is not None else None
    # Serialises every touch of `mj_data_cpu`. `launch_passive` below binds
    # the viewer to that exact MjData, so `viewer.sync()` copies it -- and
    # `_run_overlapped` calls sync from its display thread while the solve
    # thread is inside `_visualize_step`'s `mj_forward` on the same object.
    # MuJoCo catches the overlap and aborts the process with
    #   mj_copyDataVisual: attempting to copy mjData while stack is in use
    # which on a CUDA build takes the GL/CUDA context down with it, so the
    # next JAX call (finalize_log) dies too and the run is never saved.
    vis_lock = threading.Lock()

    # The object-plan ghost marker (the object block's horizon endpoint)
    # that sim drives every step. On the live path the display thread's
    # mocap refresh used to leave it parked at the MJCF origin (the robot
    # base) as a translucent T, and under ADMM it mostly duplicates the
    # goal marker -- so it is drawn only on request (`show_object_plan`)
    # and hidden otherwise. A flat controller or a scene without the
    # mocap body makes the marker a no-op either way.
    if vis_model is not None and show_object_plan:
        draw_object_plan = object_plan_marker(ctrl, vis_model)
    else:
        if vis_model is not None:
            hide_body_geoms(vis_model, "object_plan")
        draw_object_plan = lambda *a, **k: None  # noqa: E731

    ctx = LoopContext(
        task=task, interface=interface, addresses=addresses,
        base_data=base_data,
        jit_optimize=jit_optimize, jit_interp=jit_interp, jit_plans=jit_plans,
        jit_trace=jit_trace, jit_cost_terms=jit_cost_terms,
        control_dt=control_dt, replan_rate=replan_rate, max_steps=max_steps,
        vel_limit=vel_limit, goal_pos_tol=goal_pos_tol,
        goal_theta_tol=goal_theta_tol, admm=admm, log=log, verbose=verbose,
        kicker=_StuckKicker(ctrl),
        recorder=recorder, overlay=overlay, mj_data_cpu=mj_data_cpu,
        vis_model=vis_model, vis_lock=vis_lock,
        draw_object_plan=draw_object_plan,
        show_samples=show_samples, show_optimal=show_optimal,
        show_object_plan=show_object_plan,
        reducer=reducer, print_every=print_every,
        latency_comp=latency_comp, handoff=handoff, t_c=t_c,
        actuation_delay=actuation_delay,
    )

    def _run_loop(ctx: LoopContext) -> Dict[str, Any]:
        if real_time:
            return _run_overlapped(ctx, params)
        return _run_serial(ctx, params)

    # Holds the finished log across the viewer's `__exit__`. A passive
    # viewer tearing down its GL context can raise, and on a CUDA build it
    # can take the process's CUDA context with it -- either way the run is
    # already over by then and its log is already complete, so losing it to
    # a teardown fault is never the right outcome.
    result = None

    try:
        if live:
            try:
                # Pin only when a camera was explicitly named -- `camera` is
                # None by default (see pusht_real.py), so this is opt-in.
                fixed_cam = None
                if camera is not None:
                    fixed_cam = (
                        camera if isinstance(camera, int) else
                        mujoco.mj_name2id(
                            vis_model, mujoco.mjtObj.mjOBJ_CAMERA, camera
                        )
                    )
                with mujoco.viewer.launch_passive(
                    vis_model, mj_data_cpu
                ) as viewer:
                    if fixed_cam is not None and fixed_cam >= 0:
                        viewer.cam.fixedcamid = fixed_cam
                        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                    else:
                        # Framed on the table rather than on the whole
                        # model (see _frame_table). Still a FREE camera --
                        # scroll and drag from here exactly as before, this
                        # only changes where the view starts.
                        vp = viewer.viewport
                        aspect = (
                            vp.width / vp.height
                            if vp is not None and vp.height > 0
                            else _VIEW_FALLBACK_ASPECT
                        )
                        _frame_table(
                            vis_model, viewer.cam, aspect,
                            view_azimuth, view_elevation, view_distance,
                        )
                    # The viewer exists only inside this context manager,
                    # which is why it is the one field added late.
                    result = _run_loop(dataclasses.replace(
                        ctx, viewer=viewer,
                        overlay_base=(viewer.user_scn.ngeom
                                      if overlay is not None else None),
                    ))
            except BaseException:
                # Only a fault raised AFTER the loop finished is survivable
                # -- `result` is set exactly then. Anything earlier (a
                # viewer that would not open, an error out of the loop
                # itself) still propagates untouched.
                if result is None:
                    raise
                print("[live] viewer teardown raised after the run "
                      "finished; the log is complete and still saved")
        else:
            result = _run_loop(ctx)
    finally:
        # Never let closing the mp4 lose a completed run. The recorder is a
        # diagnostic; the log is the experiment.
        if recorder is not None:
            try:
                recorder.close()
            except Exception as exc:  # noqa: BLE001
                print(f"[record] closing the mp4 failed, run still saved: "
                      f"{exc}")
    return result
