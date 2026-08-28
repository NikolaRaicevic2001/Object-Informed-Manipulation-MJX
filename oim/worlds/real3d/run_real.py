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
"""

from __future__ import annotations

import threading
import time
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from oim.objects import wrap_angle
from oim.runtime.logs import finalize_log, init_log, local_goal_marker, log_step
from oim.runtime.overlay import BlockTrace, PlanOverlay, traces_for
from oim.runtime.video import OffscreenRecorder
from oim.tasks.pusht import PushT
from oim.worlds.real3d.interface import (
    RobotWorldInterface,
    SceneAddresses,
    clamp_velocity,
)

# Forward kinematics for the assembled state, JIT-compiled once and reused --
# calling mjx.forward un-jitted every control step dispatches thousands of tiny
# GPU kernels eagerly (~150 s/step); jitted it is milliseconds.
_jit_forward = jax.jit(mjx.forward)

# Per-control-step statistics of the sample population MPPI's softmax just
# consumed. Allocated only on the flat path (see `_init_sample_stats`).
_SAMPLE_STAT_KEYS = (
    "sample_cost_min",
    "sample_cost_mean",
    "sample_cost_max",
    "sample_cost_std",
    "sample_eta",
    "sample_nonfinite",
)

# --live's refresh rate on `_run_overlapped`'s display thread, between
# solves. Not tied to control_rate: this is how often a human can usefully
# perceive an update, not a control-loop constraint like control_rate is.
_DISPLAY_HZ = 30.0


def _init_sample_stats(log: Dict[str, Any], admm: bool) -> None:
    """Allocate the sample-statistics series, flat MPPI only.

    Here rather than in `oim.runtime.logs.init_log` so the sim world's log
    layout is untouched: this is a real-driver diagnostic, and `init_log` is
    the contract that keeps a hardware log comparable to a simulation one
    entry-for-entry.
    """
    if not admm:
        log.update({k: [] for k in _SAMPLE_STAT_KEYS})


def _log_sample_stats(
    log: Dict[str, Any], rollouts: Any, temperature: Any
) -> None:
    """Append this step's sample-population statistics, if they exist.

    Why record these at all: the flat MPPI update is a softmax-weighted mean
    over the sampled knot sequences, so it is only as decisive as the SPREAD
    of that population's costs. A run that stalls looks identical in every
    series we already log -- the object sits still, the arm keeps commanding
    -- whether the planner has found a clear best sample it cannot execute,
    or every sample scores the same and the mean is random-walking. The two
    call for opposite fixes, and only the population tells them apart:

      sample_eta       effective sample size, `sum(exp(-(c - c_min) / T))`,
                       in [1, num_samples]. At num_samples the weights are
                       uniform -- the update carries no information at all.
                       At 1 a single sample owns the mean.
      sample_cost_std  the absolute spread the temperature is dividing. eta
                       near num_samples with a large std means the
                       temperature is too high for this cost scale; with a
                       tiny std it means the samples genuinely do not
                       differ, i.e. no reachable sample improves anything.
      min/mean/max     the scale itself, so a term's share can be checked
                       against the population rather than inferred.
      sample_nonfinite how many samples scored inf or NaN. Any nonzero value
                       is a bug in a cost term, not a property of the task
                       -- a single NaN makes every weight NaN.

    No-ops for a controller whose second `optimize` return carries no
    per-sample costs (the ADMM path), so both loops stay algorithm-agnostic.
    """
    if "sample_eta" not in log:
        return
    costs = getattr(rollouts, "costs", None)
    if costs is None:
        return
    raw = np.asarray(costs, dtype=float)
    if raw.ndim != 2:
        return
    total = raw.sum(axis=1)  # (num_samples,), summed over the horizon
    good = total[np.isfinite(total)]
    log["sample_nonfinite"].append(int(total.size - good.size))
    if good.size == 0:
        for key in _SAMPLE_STAT_KEYS[:-1]:
            log[key].append(float("nan"))
        return
    # Same decomposition `MPPI.update_params` uses, on the same numbers:
    # shift by the population minimum before exponentiating, so the largest
    # term is exactly 1 and the sum cannot overflow.
    temp = max(float(np.asarray(temperature)), 1e-9)
    log["sample_cost_min"].append(float(good.min()))
    log["sample_cost_mean"].append(float(good.mean()))
    log["sample_cost_max"].append(float(good.max()))
    log["sample_cost_std"].append(float(good.std()))
    log["sample_eta"].append(float(np.exp(-(good - good.min()) / temp).sum()))


def _visualize_step(
    vis_model: mujoco.MjModel,
    mjx_data: mjx.Data,
    mj_data_cpu: Optional[mujoco.MjData],
    recorder: Optional[OffscreenRecorder],
    overlay: Optional[PlanOverlay],
    viewer: Any,
    overlay_base: Optional[int],
    rollouts: Any,
    params: Any,
    admm: bool,
    show_samples: bool,
    show_optimal: bool,
    obj_plan: Optional[np.ndarray] = None,
    rob_plan: Optional[np.ndarray] = None,
    robot_trace: Optional[np.ndarray] = None,
    sync_viewer: bool = True,
) -> List[BlockTrace]:
    """Push one frame to whichever of `recorder`/`viewer` are active.

    Real has no standing CPU `mujoco.MjData` the way the sim worlds do --
    the whole state is `mjx_data` -- so `mj_data_cpu` is written from it
    here, once, and shared by both destinations: the offscreen recorder
    (`OffscreenRecorder.capture`, exactly what every sim world already
    uses) and the live passive viewer (`viewer.sync`). A no-op if neither
    is set, so a run with neither `--record` nor `--live` pays nothing
    beyond the `is None` checks.

    Args:
        vis_model: The CPU model `mj_data_cpu` mirrors -- the shared
            deepcopy `run_real` builds, never `task.mj_model` itself.
        mjx_data: This control step's assembled state.
        mj_data_cpu: Reused every call; `None` iff both destinations are.
        recorder: The mp4 recorder, or `None`.
        overlay: The shared candidate/chosen-trajectory drawer, or `None`
            if neither `show_samples` nor `show_optimal` was asked for.
        viewer: A `mujoco.viewer` passive-viewer handle, or `None`.
        overlay_base: Fixed geom slot for the viewer's persistent scene
            (see `PlanOverlay.draw`); unused when drawing into the
            recorder's own scene, which is rebuilt every frame.
        rollouts: This step's sampled robot rollouts, for `show_samples`.
        params: What `optimize` just returned, for the object block's
            sampled population (ADMM only).
        admm: Whether `obj_plan`/`rob_plan` are meaningful -- a flat
            controller has no object block to draw one for.
        show_samples: Composite the sample population, as on `run_real`.
        show_optimal: Composite the chosen trajectory, as on `run_real`.
        obj_plan: The object block's predicted object trajectory (ADMM
            only), from `ADMM.nominal_plans`.
        rob_plan: The robot block's, from the same call.
        robot_trace: The chosen end-effector path, from
            `ADMM.nominal_plans` or the flat controller's `nominal_trace`.
        sync_viewer: Draw and sync `viewer` here. `_run_overlapped` passes
            `False`: it still needs `viewer is not None` to trigger trace
            computation below, but its own display thread owns
            `viewer.sync()` exclusively (see there for why one caller).

    Returns:
        The traces just drawn (possibly `[]`) -- `_run_overlapped` reuses
        these for its own display thread, which redraws them at a higher
        rate than one per solve without recomputing `traces_for` itself.
    """
    if recorder is None and viewer is None:
        return []
    mj_data_cpu.qpos[:] = np.asarray(mjx_data.qpos)
    mj_data_cpu.qvel[:] = np.asarray(mjx_data.qvel)
    mj_data_cpu.time = float(mjx_data.time)
    mujoco.mj_forward(vis_model, mj_data_cpu)

    traces = []
    if overlay is not None:
        traces = traces_for(
            robot_chosen=robot_trace if show_optimal else None,
            object_chosen=obj_plan if (show_optimal and admm) else None,
            robot_object_chosen=rob_plan if (show_optimal and admm) else None,
            robot_samples=(
                np.asarray(rollouts.trace_sites)[:, :, 0, :]
                if show_samples
                else None
            ),
            object_samples=(
                np.asarray(params.object_samples)
                if show_samples and admm
                and getattr(params, "object_samples", None) is not None
                else None
            ),
        )
    if recorder is not None:
        recorder.set_plans(traces)
        recorder.capture(mj_data_cpu)
    if viewer is not None and sync_viewer:
        if overlay is not None:
            overlay.draw(viewer.user_scn, traces, base=overlay_base)
        viewer.sync()
    return traces


class _StuckKicker:
    """Detect-and-kick, ported from `oim.worlds.sim3d.run._run_plain`.

    Same logic, same 1e-4 threshold, same perturbation. Wrapped in a class
    only because this module has two loops (`_run_serial` and
    `_run_overlapped`) where the sim has one, so inlining it would mean two
    copies. If sim and real should ever share one implementation, this is the
    piece to hoist into `oim/runtime/`.

    Why it exists: MPPI's softmax-weighted mean update can settle into a blend
    that commits to neither "found the contact angle that breaks stiction" nor
    "back off and re-approach" -- the object stops moving entirely while the
    arm keeps jiggling around the same pose. The sim's flat loop perturbs the
    sampling mean after `stuck_kick_steps` consecutive no-progress control
    steps; this driver had no equivalent, so the same controller stalls here
    where it recovers there.

    Like the sim's version, this only ever touches the traced `params` pytree,
    never a `self.` attribute on the controller -- `jit_optimize` is a jitted
    bound method, so a mutated `self.x` would be silently ignored.

    Reads its two numbers off the controller, so it is inert for ADMM and for
    any optimizer built without them (`stuck_kick_steps` defaults to 0).
    """

    # Matches the exact-zero signature real stiction produces in MJX/Warp
    # (object_velocity goes bit-exact 0.0, not a gradual decay) -- not a
    # tolerance chosen to catch merely "slow" progress.
    EPS = 1e-4

    def __init__(self, ctrl: Any) -> None:
        self.steps = int(getattr(ctrl, "stuck_kick_steps", 0) or 0)
        self.scale = float(getattr(ctrl, "stuck_kick_scale", 0.0) or 0.0)
        self.count = 0
        self.kicks = 0
        self._prev = None

    def maybe_kick(self, params: Any, pos_err: float, theta_err: float,
                   step: int, verbose: bool) -> Any:
        """Return `params`, perturbed if the run has been frozen long enough."""
        if self.steps <= 0 or not hasattr(params, "mean"):
            return params
        if self._prev is not None:
            no_progress = (
                abs(pos_err - self._prev[0]) < self.EPS
                and abs(theta_err - self._prev[1]) < self.EPS
            )
            self.count = self.count + 1 if no_progress else 0
        self._prev = (pos_err, theta_err)
        if self.count < self.steps:
            return params
        kick_rng, rng = jax.random.split(params.rng)
        kick = self.scale * jax.random.normal(kick_rng, params.mean.shape)
        self.count = 0
        self.kicks += 1
        if verbose:
            print(f"step {step:4d}  stuck -- kicked ({self.kicks})")
        return params.replace(mean=params.mean + kick, rng=rng)


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
    record_dir: Optional[str] = None,
    record_name: Optional[str] = None,
    video_fps: float = 30.0,
    video_size: Tuple[int, int] = (720, 480),
    camera: Optional[Union[str, int]] = None,
    live: bool = False,
    show_samples: bool = True,
    show_optimal: bool = True,
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
        goal_pos_tol: success tolerance on position [m].
        goal_theta_tol: success tolerance on yaw [rad].
        real_time: True -> hardware (threaded, overlapped); False -> mock
            (single-threaded, deterministic).
        vel_limit: peak joint velocity a command may carry [rad/s]; every
            joint is scaled together to respect it.
        admm: whether `ctrl` is ADMM, so the two block plans are logged
            and drawn. A flat baseline has neither.
        verbose: print progress.
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
        show_optimal: Overlay each block's chosen trajectory.

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
    world0 = interface.read_state()
    mjx_data = _assemble_state(task, base_data, addresses, world0)
    if verbose:
        print(f"[jit] initial state in {time.perf_counter() - t:.1f}s; "
              "warming up -- the first optimize traces + XLA-compiles the "
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
        print(
            f"[jit] optimize compiled + first run: "
            f"{time.perf_counter() - t:.1f}s"
        )

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
    if verbose:
        print(f"[jit] loop-path warm-up: {time.perf_counter() - t:.1f}s")

    if verbose:
        print(f"[jit] ready; {'overlapped' if real_time else 'serial'} loop, "
              f"control {control_rate:.0f} Hz, stream")

    log = init_log(task, mjx_data, mjx_data, show_plans=admm, admm=admm)
    _init_sample_stats(log, admm)

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
    mj_data_cpu = mujoco.MjData(vis_model) if vis_model is not None else None

    # The `local_goal` ghost marker sim drives every step and real never
    # has -- so on real it just sits wherever the MJCF parked it (world
    # origin, which happens to be at the robot base) instead of being
    # hidden or moved. Built unconditionally: a flat controller (no
    # `local_goal`) or a scene with no such mocap body both make this a
    # no-op that hides the marker instead, exactly the case that was
    # previously silently wrong.
    draw_local_goal = (
        local_goal_marker(ctrl, vis_model)
        if vis_model is not None else lambda *a, **k: None
    )

    common = dict(
        task=task, interface=interface, addresses=addresses,
        base_data=base_data,
        jit_optimize=jit_optimize, jit_interp=jit_interp, jit_plans=jit_plans,
        jit_trace=jit_trace,
        control_dt=control_dt, max_steps=max_steps, goal_pos_tol=goal_pos_tol,
        goal_theta_tol=goal_theta_tol, vel_limit=vel_limit, admm=admm, log=log,
        verbose=verbose, kicker=_StuckKicker(ctrl),
        recorder=recorder, overlay=overlay, mj_data_cpu=mj_data_cpu,
        show_samples=show_samples, show_optimal=show_optimal,
        vis_model=vis_model, draw_local_goal=draw_local_goal,
    )

    def _run_loop() -> Dict[str, Any]:
        if real_time:
            return _run_overlapped(params=params, **common)
        return _run_serial(params=params, replan_rate=replan_rate, **common)

    try:
        if live:
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
                    # The same auto-fit OffscreenRecorder already gives
                    # camera=None (mujoco.Renderer's own default), so a
                    # live view starts framed the same way an mp4 is.
                    # Still a free camera, not fixed -- orbits by hand
                    # from here exactly as it would from any other start.
                    mujoco.mjv_defaultFreeCamera(vis_model, viewer.cam)
                common["viewer"] = viewer
                common["overlay_base"] = (
                    viewer.user_scn.ngeom if overlay is not None else None
                )
                result = _run_loop()
        else:
            common["viewer"] = None
            common["overlay_base"] = None
            result = _run_loop()
    finally:
        if recorder is not None:
            recorder.close()
    return result


def _run_serial(
    task: PushT,
    interface: RobotWorldInterface,
    addresses: SceneAddresses,
    base_data: mjx.Data,
    jit_optimize: Callable[..., Any],
    jit_interp: Callable[..., Any],
    jit_plans: Callable[..., Any],
    jit_trace: Callable[..., Any],
    control_dt: float,
    replan_rate: float,
    max_steps: int,
    goal_pos_tol: float,
    goal_theta_tol: float,
    vel_limit: float,
    admm: bool,
    log: Dict[str, Any],
    verbose: bool,
    params: Any,
    kicker: Any,
    recorder: Any,
    overlay: Optional[PlanOverlay],
    mj_data_cpu: mujoco.MjData,
    show_samples: bool,
    show_optimal: bool,
    viewer: Any,
    overlay_base: Any,
    vis_model: mujoco.MjModel,
    draw_local_goal: bool,
) -> Dict[str, Any]:
    """Single-threaded loop: solve, then publish the window, then repeat.

    Used for the mock (deterministic, MuJoCo not thread-safe). The arm stalls
    on the last command during each solve, which is fine off-hardware.
    """
    replan_period = 1.0 / replan_rate
    num_ticks = max(1, round(replan_period / control_dt))
    reached = False

    for step in range(max_steps):
        if viewer is not None and not viewer.is_running():
            break
        world = interface.read_state()
        mjx_data = _assemble_state(task, base_data, addresses, world)

        t0 = time.perf_counter()
        # The second return -- the sampled rollouts -- used to be dropped on
        # the floor here. It is the only place the sample population is ever
        # visible; see `_log_sample_stats`.
        params, rollouts = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)
        # After the timer: this is diagnostics, not planning, and it forces a
        # device-to-host copy of the (num_samples, H+1) cost array.
        _log_sample_stats(log, rollouts, getattr(params, "temperature", 1.0))

        sample_times = jnp.arange(num_ticks) * control_dt + world.time
        plan_samples = np.asarray(
            jit_interp(sample_times, params.tk, params.mean[None, ...])
        )[0]
        applied = np.empty_like(plan_samples)
        for i in range(num_ticks):
            applied[i] = clamp_velocity(plan_samples[i], vel_limit)
            interface.send_velocity(applied[i])
        obj_plan = rob_plan = robot_trace = None
        if admm:
            obj_plan, rob_plan, robot_trace = jit_plans(mjx_data, params)
            log["object_plan"].append(np.asarray(obj_plan))
            log["robot_plan"].append(np.asarray(rob_plan))
        elif jit_trace is not None:
            robot_trace = jit_trace(mjx_data, params)
        if mj_data_cpu is not None:
            # No-op (and the marker stays hidden) unless ctrl has
            # local_goal and the scene declares the mocap body -- see
            # local_goal_marker's own resolution of both, done once.
            draw_local_goal(mj_data_cpu, mjx_data, params, obj_plan)
        _visualize_step(
            vis_model, mjx_data, mj_data_cpu, recorder, overlay, viewer,
            overlay_base, rollouts, params, admm, show_samples, show_optimal,
            obj_plan=None if obj_plan is None else np.asarray(obj_plan),
            rob_plan=None if rob_plan is None else np.asarray(rob_plan),
            robot_trace=(
                None if robot_trace is None else np.asarray(robot_trace)
            ),
        )
        reached = _log_and_check(log, task, mjx_data, params, applied,
                                 goal_pos_tol, goal_theta_tol, step,
                                 verbose, admm)
        if reached:
            break
        # Same placement the sim's flat loop uses: after the success check,
        # reading the errors that check just used.
        params = kicker.maybe_kick(params, log["pos_err"][-1],
                                   log["theta_err"][-1], step, verbose)

    interface.send_velocity(np.zeros(len(addresses.arm_dof_adr)))
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _run_overlapped(
    task: PushT,
    interface: RobotWorldInterface,
    addresses: SceneAddresses,
    base_data: mjx.Data,
    jit_optimize: Callable[..., Any],
    jit_interp: Callable[..., Any],
    jit_plans: Callable[..., Any],
    jit_trace: Callable[..., Any],
    control_dt: float,
    max_steps: int,
    goal_pos_tol: float,
    goal_theta_tol: float,
    vel_limit: float,
    admm: bool,
    log: Dict[str, Any],
    verbose: bool,
    params: Any,
    kicker: Any,
    recorder: Any,
    overlay: Optional[PlanOverlay],
    mj_data_cpu: mujoco.MjData,
    show_samples: bool,
    show_optimal: bool,
    viewer: Any,
    overlay_base: Any,
    vis_model: mujoco.MjModel,
    draw_local_goal: bool,
) -> Dict[str, Any]:
    """Hardware loop: planning and execution overlap.

    A publisher thread streams the latest plan while the main thread keeps
    solving.
    """

    def _sample_plan(plan: Any) -> np.ndarray:
        """Materialise the plan into a numpy table.

        Sampled on the plan's own time base (`tk`), not the caller's clock:
        the two are set from different reads of the state clock, and querying
        outside [tk[0], tk[-1]] silently returns the last knot -- which is what
        the publisher used to send on every tick.

        The publisher thread must never call into JAX: doing so concurrently
        with the solver is what makes the Warp backend segfault (CUDA graph
        capture is not safe across threads), and it also costs a dispatch on
        every control tick.
        """
        tk = np.asarray(plan.tk)
        span = float(tk[-1] - tk[0])
        n = max(1, int(span / control_dt) + 1)
        ts = jnp.arange(n) * control_dt + float(tk[0])
        return np.asarray(jit_interp(ts, plan.tk, plan.mean[None, ...]))[0]

    # Shared latest plan, guarded by a lock. `samples` is the plan already
    # materialised on a control-tick grid; `t_perf` is the wall clock when it
    # was published, so the publisher can index into it by elapsed time.
    # `qpos`/`traces` are for the display thread below, not the publisher --
    # set to real values on the loop's first iteration, before either thread
    # that reads them can start.
    lock = threading.Lock()
    shared = {"samples": _sample_plan(params),
              "t_perf": time.perf_counter(),
              "qpos": None, "traces": [], "mocap": None}
    stop = threading.Event()

    def _publisher() -> None:
        next_tick = time.perf_counter()
        while not stop.is_set():
            with lock:
                s = shared["samples"]
                t_perf = shared["t_perf"]

            elapsed = time.perf_counter() - t_perf
            if elapsed > len(s) * control_dt:
                # The plan has run out. A stalled solver must not leave the arm
                # executing the tail of an old plan: the interface watchdog
                # cannot catch that, because the publisher is still sending.
                u = np.zeros_like(s[0])
            else:
                u = s[min(int(elapsed / control_dt), len(s) - 1)]

            interface.send_velocity(clamp_velocity(u, vel_limit))
            next_tick += control_dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:  # publisher fell behind; resync rather than spiral
                next_tick = time.perf_counter()

    pub = threading.Thread(target=_publisher, daemon=True)
    pub.start()

    # A live viewer synced only when a solve finishes updates once every
    # solve -- a second or more, at production sample sizes -- even though
    # the arm is genuinely moving the whole time between solves, streamed
    # by the publisher above. This thread redraws in between, without
    # adding a second caller of interface.read_state(): that method
    # advances a finite-difference/low-pass filter (see Ros2Interface's
    # own read_state) that the controller's twist consensus depends on,
    # and a second caller would corrupt it. Instead it dead-reckons the
    # ARM alone -- integrating the exact commanded-velocity plan the
    # publisher is already streaming, exact for revolute joints -- from
    # the qpos the last solve actually measured. There is no equivalent
    # model for the OBJECT, so its pose is simply held at that last
    # measurement until the next real one arrives, same as today.
    disp_data = mujoco.MjData(vis_model) if viewer is not None else None
    disp_stop = threading.Event()

    def _display_loop() -> None:
        period = 1.0 / _DISPLAY_HZ
        while not disp_stop.is_set():
            t_tick = time.perf_counter()
            with lock:
                s = shared["samples"]
                t_perf = shared["t_perf"]
                qpos = shared["qpos"]
                traces = shared["traces"]
                mocap = shared["mocap"]
            if qpos is not None:
                n = len(s)
                plan_span = n * control_dt
                elapsed = min(time.perf_counter() - t_perf, plan_span)
                idx_full = min(int(elapsed / control_dt), n)
                partial = elapsed - idx_full * control_dt
                v_partial = s[idx_full] if idx_full < n else np.zeros_like(s[0])
                integral = (s[:idx_full].sum(axis=0) * control_dt
                            + v_partial * partial)

                disp_data.qpos[:] = qpos
                disp_data.qpos[addresses.arm_qpos_adr] += integral
                # The local_goal ghost (if any) only ever changes once per
                # solve too, same as the object -- copied in, not
                # recomputed: recomputing calls into JAX (see
                # local_goal_marker), which this thread must never do.
                if mocap is not None:
                    disp_data.mocap_pos[:] = mocap[0]
                    disp_data.mocap_quat[:] = mocap[1]
                mujoco.mj_forward(vis_model, disp_data)
                if overlay is not None:
                    overlay.draw(viewer.user_scn, traces, base=overlay_base)
                viewer.sync()
            sleep = period - (time.perf_counter() - t_tick)
            if sleep > 0:
                time.sleep(sleep)

    disp = threading.Thread(target=_display_loop, daemon=True)
    if viewer is not None:
        disp.start()

    reached = False
    try:
        for step in range(max_steps):
            if viewer is not None and not viewer.is_running():
                break
            t_loop = time.perf_counter()
            world = interface.read_state()
            mjx_data = _assemble_state(task, base_data, addresses, world)

            t0 = time.perf_counter()
            params, rollouts = jit_optimize(mjx_data, params)
            jax.block_until_ready(params)
            log["compute_time"].append(time.perf_counter() - t0)

            # Hand the fresh plan to the publisher (and the display thread's
            # dead-reckoning base -- same anchor time, same reasoning).
            samples = _sample_plan(params)
            with lock:
                shared["samples"] = samples
                # The plan's s[0] is the control for the state read at
                # `t_loop`, one solve ago -- the arm has been executing the
                # previous plan since. Anchor plan time to that read, not to
                # now, so the publisher enters the plan where the present
                # actually is instead of replaying a moment that has passed.
                shared["t_perf"] = t_loop
                shared["qpos"] = np.asarray(mjx_data.qpos)

            # Deliberately after the hand-off above: this forces a device-to-
            # host copy of the (num_samples, H+1) cost array, and the
            # publisher must not wait on a diagnostic.
            _log_sample_stats(
                log, rollouts, getattr(params, "temperature", 1.0)
            )

            # Log the command the publisher would send at the solve instant.
            first = samples[:1]
            obj_plan = rob_plan = robot_trace = None
            if admm:
                obj_plan, rob_plan, robot_trace = jit_plans(mjx_data, params)
                log["object_plan"].append(np.asarray(obj_plan))
                log["robot_plan"].append(np.asarray(rob_plan))
            elif jit_trace is not None:
                robot_trace = jit_trace(mjx_data, params)
            if mj_data_cpu is not None:
                draw_local_goal(mj_data_cpu, mjx_data, params, obj_plan)
            # After the hand-off above and _log_sample_stats, same rule:
            # rendering is a diagnostic, and the publisher must not wait on
            # one. Measured safe from this thread against the Warp/JAX
            # solver on the mock -- see the mock diagnostic in Tasks.md.
            # sync_viewer=False: the display thread above owns
            # viewer.sync() exclusively, so this call only feeds the
            # recorder directly; its returned traces are handed to that
            # thread instead of it recomputing traces_for itself.
            traces = _visualize_step(
                vis_model, mjx_data, mj_data_cpu, recorder, overlay, viewer,
                overlay_base, rollouts, params, admm, show_samples,
                show_optimal,
                obj_plan=None if obj_plan is None else np.asarray(obj_plan),
                rob_plan=None if rob_plan is None else np.asarray(rob_plan),
                robot_trace=(
                    None if robot_trace is None else np.asarray(robot_trace)
                ),
                sync_viewer=False,
            )
            if viewer is not None:
                with lock:
                    shared["traces"] = traces
                    # local_goal's ghost pose, same hand-off reasoning as
                    # qpos above -- the display thread copies these rather
                    # than ever calling draw_local_goal itself.
                    shared["mocap"] = (
                        mj_data_cpu.mocap_pos.copy(),
                        mj_data_cpu.mocap_quat.copy(),
                    )
            reached = _log_and_check(log, task, mjx_data, params, first,
                                     goal_pos_tol, goal_theta_tol, step,
                                     verbose, admm)
            if reached:
                break
            # The kick only rewrites the sampling mean the NEXT solve starts
            # from; the publisher keeps streaming the plan already handed to
            # it, so nothing the arm is executing changes discontinuously.
            params = kicker.maybe_kick(params, log["pos_err"][-1],
                                       log["theta_err"][-1], step, verbose)
    finally:
        stop.set()
        pub.join(timeout=1.0)
        if viewer is not None:
            disp_stop.set()
            disp.join(timeout=1.0)
        interface.send_velocity(np.zeros(len(addresses.arm_dof_adr)))
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _log_and_check(
    log: Dict[str, Any],
    task: PushT,
    mjx_data: mjx.Data,
    params: Any,
    applied: np.ndarray,
    goal_pos_tol: float,
    goal_theta_tol: float,
    step: int,
    verbose: bool,
    admm: bool = True,
) -> bool:
    """Append one step to the log and return whether the goal was reached."""
    block_pose = log_step(log, task, mjx_data, params, applied, admm=admm)
    goal = np.asarray(task.goal)
    pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
    theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)
    if verbose and step % 10 == 0:
        primal = f"primal={log['primal_residual'][-1]:.3f}  " if admm else ""
        # eta on the console, not only in the run file: a flat run that has
        # gone uninformative (eta at num_samples, or any nonfinite sample)
        # otherwise looks exactly like one that is working, and there is no
        # point letting 1000 steps finish before finding that out.
        pop = ""
        if log.get("sample_eta"):
            bad = log["sample_nonfinite"][-1]
            pop = (f"eta={log['sample_eta'][-1]:.1f}  "
                   f"cost={log['sample_cost_min'][-1]:.2f}"
                   f"+-{log['sample_cost_std'][-1]:.2f}  "
                   + (f"NONFINITE={bad}  " if bad else ""))
        print(f"step {step:4d}  pos_err={pos_err:.4f}  "
              f"theta_err={theta_err:.4f}  "
              f"{primal}{pop}plan={log['compute_time'][-1] * 1e3:.0f}ms")
    if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
        if verbose:
            print(f"goal reached at step {step}")
        return True
    return False


def _assemble_state(
    task: PushT,
    base_data: mjx.Data,
    addresses: SceneAddresses,
    world: Any,  # WorldState
) -> mjx.Data:
    """Inject measured arm + object state into a full MJX state.

    Hardware measures only the arm (encoders) and the object (FoundationPose);
    the static obstacles are baked into the model. We write the two moving
    parts into their qpos/qvel slots (looked up by joint name) and run forward
    kinematics so `site_xpos` (the stick tip) is populated for cost + logging.
    """
    nq = task.mj_model.nq
    qpos = np.asarray(base_data.qpos).copy()
    qvel = np.zeros(task.mj_model.nv)
    qpos[addresses.arm_qpos_adr] = world.arm_qpos
    qpos[addresses.block_qpos_adr] = world.object_se2
    qvel[addresses.arm_dof_adr] = world.arm_qvel
    # Block twist feeds realized_consensus
    # (w = wrench_limit * qvel[block_dofs]).
    qvel[addresses.block_dof_adr] = world.object_twist
    assert qpos.shape[0] == nq, (qpos.shape, nq)
    mjx_data = base_data.replace(
        qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel), time=float(world.time)
    )
    return _jit_forward(task.model, mjx_data)
