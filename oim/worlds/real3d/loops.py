"""The two control loops, and the state they share.

`_run_serial` solves and then publishes its window, one after the other --
the mock's loop, where MuJoCo is not thread-safe and the arm stalling
during a solve costs nothing. `_run_overlapped` is the hardware loop: a
publisher thread streams the plan already in hand while the main thread
solves the next one, so the arm is never idle. They share `LoopContext`
(everything `run_real` resolved), the state assembly, and the per-step log.
"""

from __future__ import annotations

import dataclasses
import math
import signal
import threading
import time
from collections import deque
from typing import Any, Deque, Dict, List, Tuple

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx

from oim.objects import wrap_angle
from oim.runtime.logs import log_step
from oim.tasks.pusht import PushT
from oim.worlds.real3d.command_stream import integrate_stream, window_split
from oim.worlds.real3d.diagnostics import _cost_terms
from oim.worlds.real3d.interface import (
    ARM_JOINT_NAMES,
    MujocoMockInterface,
    SceneAddresses,
    clamp_velocity,
)
from oim.worlds.real3d.reconstruct import _finish, _plan_knots
from oim.worlds.real3d.timing import (
    _log_publish_stamps,
    _log_read_stamps,
    _phase_summary,
    _PhaseTimer,
    _stamp_summary,
    execution_window,
    publish_index,
)
from oim.worlds.real3d.visualize import _DISPLAY_HZ, _visualize_step

# Forward kinematics for the assembled state, JIT-compiled once and reused --
# calling mjx.forward un-jitted every control step dispatches thousands of
# tiny GPU kernels eagerly (~150 s/step); jitted it is milliseconds.
_jit_forward = jax.jit(mjx.forward)


class _InterruptFlag:
    """Ctrl-C as a FLAG the loop reads, not an exception that unwinds.

    A hardware run normally ends with Ctrl-C, and the run file is written
    after the loop returns -- so the runs worth keeping were the ones that
    saved nothing. Catching `KeyboardInterrupt` did not fix it: with `--warp`
    the interrupt lands inside `wp_cuda_graph_launch`, the C++ runtime throws
    `std::bad_alloc` and the process aborts before Python unwinds a single
    frame. No `except` or `finally` in this file ever runs.

    A signal handler runs between bytecodes in the MAIN thread, so it cannot
    interrupt a CUDA launch. It sets a flag; the loop checks it at the top of
    the next iteration and breaks normally, and everything downstream --
    `finalize_log`, `save_run`, the arm stop -- happens the way it does on a
    clean finish. Worst case the break is one solve late.

    A second Ctrl-C restores the old behaviour, so a genuinely wedged run can
    still be killed.
    """

    def __init__(self) -> None:
        self.requested = False
        self._prev = None

    def install(self) -> "_InterruptFlag":
        # Only ever called from the main thread, which is the only thread
        # allowed to install a handler.
        self._prev = signal.signal(signal.SIGINT, self._on_sigint)
        return self

    def restore(self) -> None:
        if self._prev is not None:
            signal.signal(signal.SIGINT, self._prev)
            self._prev = None

    def _on_sigint(self, signum, frame) -> None:
        if self.requested:
            raise KeyboardInterrupt  # second one: let it through
        self.requested = True
        print("\n[stop] interrupt received -- finishing this solve, then "
              "stopping the arm and saving the run")

@dataclasses.dataclass(frozen=True)
class LoopContext:
    """Everything a control loop needs that does not change between steps.

    `run_real` resolves all of it -- the task and its jitted entry points,
    the interface, the log, the diagnostics, the rendering destinations,
    the handoff policy -- and hands it over whole. It used to be 36
    positional parameters threaded through `run_real` -> an untyped dict ->
    both loop signatures, so a new knob meant four edits in lockstep and
    nothing caught a mismatch until the run started. Frozen, because a loop
    must not silently retune its own configuration mid-run: the one thing
    that legitimately arrives late is the viewer (it exists only inside
    `launch_passive`'s context manager), which `dataclasses.replace` adds.

    The per-step state -- `params`, the plan, the log's contents -- is NOT
    here; that is what the loops own.
    """

    task: Any
    interface: Any
    addresses: Any
    base_data: Any
    # Jitted entry points, compiled once by `run_real`.
    jit_optimize: Any
    jit_interp: Any
    jit_plans: Any
    jit_trace: Any
    # Execution and success criteria.
    control_dt: float
    replan_rate: float
    max_steps: int
    vel_limit: float
    goal_pos_tol: float
    goal_theta_tol: float
    admm: bool
    log: Dict[str, Any]
    verbose: bool
    kicker: Any
    # Rendering: all None when neither --record nor --live is on.
    recorder: Any = None
    overlay: Any = None
    overlay_base: Any = None
    mj_data_cpu: Any = None
    vis_model: Any = None
    vis_lock: Any = None
    viewer: Any = None
    draw_object_plan: Any = None
    show_samples: bool = False
    show_optimal: bool = False
    show_object_plan: bool = False
    # Diagnostics.
    reducer: Any = None
    print_every: int = 10
    jit_cost_terms: Any = None
    # Handoff policy -- see `_run_overlapped`.
    latency_comp: float = 0.0
    handoff: str = "responsive"
    t_c: float = 0.5
    actuation_delay: float = 0.0

def _run_serial(ctx: LoopContext, params: Any) -> Dict[str, Any]:
    """Single-threaded loop: solve, then publish the window, then repeat.

    `latency_comp` is accepted for signature parity with `_run_overlapped`
    and ignored: the serial loop has no overlap to compensate. `handoff`
    and `t_c` are NOT ignored -- they set the execution window, so the mock
    executes as much plan per solve as hardware does. NOTE for anyone
    testing the handoff work: `--mock` runs THIS loop, so the overlapped
    publisher, the anchor wait and `_clip_plan_to_joint_range` are still
    unexercised by a mock run -- only the window is shared.

    Used for the mock (deterministic, MuJoCo not thread-safe). The arm stalls
    on the last command during each solve, which is fine off-hardware.
    """
    replan_period = execution_window(ctx.handoff, ctx.t_c, ctx.replan_rate)
    num_ticks = max(1, round(replan_period / ctx.control_dt))
    reached = False
    # Plans are rolled out live only when something draws them; otherwise
    # the block means are kept (tiny) and the plans rebuilt after the loop.
    live_plans = ctx.vis_model is not None and (
        ctx.show_optimal or ctx.show_object_plan
    )
    plan_knots: List[Any] = []
    ctx.log.setdefault("loop_time", [])
    t_solve_prev = None
    phases = _PhaseTimer(ctx.log)

    t_run0 = None
    for step in range(ctx.max_steps):
        if ctx.viewer is not None and not ctx.viewer.is_running():
            break
        phases.step_start()
        world = ctx.interface.read_state()
        if t_run0 is None:
            t_run0 = float(world.time)
        world = _rebase_time(world, t_run0)
        _log_read_stamps(ctx.log, world)
        phases.mark("t_read")
        mjx_data = _assemble_state(
            ctx.task, ctx.base_data, ctx.addresses, world
        )
        phases.mark("t_assemble")

        t0 = time.perf_counter()
        # Solve-start to solve-start: the whole control period, i.e. the
        # solve plus everything the loop does around it.
        ctx.log["loop_time"].append(
            float("nan") if t_solve_prev is None else t0 - t_solve_prev
        )
        t_solve_prev = t0
        # The second return -- the sampled rollouts -- used to be dropped on
        # the floor here. It is the only place the sample population is ever
        # visible; see `_log_sample_stats`.
        params, rollouts = ctx.jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        phases.mark("compute_time")
        # After the timer: diagnostics, reduced on the device to scalars.
        # Also where the solve's own tail lands -- `rollouts` is still in
        # flight above, and this is the first read of it.
        ctx.reducer(ctx.log, rollouts, params, ctx.task._block_pose(mjx_data))
        phases.mark("t_reduce")

        sample_times = jnp.arange(num_ticks) * ctx.control_dt + world.time
        plan_samples = np.asarray(
            ctx.jit_interp(sample_times, params.tk, params.mean[None, ...])
        )[0]
        applied = np.empty_like(plan_samples)
        for i in range(num_ticks):
            applied[i] = clamp_velocity(plan_samples[i], ctx.vel_limit)
            ctx.interface.send_velocity(applied[i])
            if isinstance(ctx.interface, MujocoMockInterface):
                applied[i] = ctx.interface.last_applied_velocity
        phases.mark("t_send")
        obj_plan = rob_plan = robot_trace = None
        if ctx.admm and live_plans:
            obj_plan, rob_plan, robot_trace = ctx.jit_plans(mjx_data, params)
            ctx.log["object_plan"].append(np.asarray(obj_plan))
            ctx.log["robot_plan"].append(np.asarray(rob_plan))
        elif ctx.admm:
            plan_knots.append(_plan_knots(params))
        elif ctx.jit_trace is not None and ctx.vis_model is not None:
            robot_trace = ctx.jit_trace(mjx_data, params)
        if ctx.mj_data_cpu is not None:
            # No-op (and the marker stays hidden) unless ctrl has
            # object_plan and the scene declares the mocap body -- see
            # object_plan_marker's own resolution of both, done once.
            ctx.draw_object_plan(ctx.mj_data_cpu, mjx_data, params, obj_plan)
        _visualize_step(
            ctx.vis_model, mjx_data, ctx.mj_data_cpu, ctx.recorder,
            ctx.overlay, ctx.viewer, ctx.overlay_base, rollouts, params,
            ctx.admm, ctx.show_samples, ctx.show_optimal,
            obj_plan=None if obj_plan is None else np.asarray(obj_plan),
            rob_plan=None if rob_plan is None else np.asarray(rob_plan),
            robot_trace=(
                None if robot_trace is None else np.asarray(robot_trace)
            ),
            vis_lock=ctx.vis_lock,
        )
        phases.mark("t_plan")
        reached = _log_and_check(ctx.log, ctx.task, mjx_data, params, applied,
                                 ctx.goal_pos_tol, ctx.goal_theta_tol, step,
                                 ctx.verbose, ctx.admm, ctx.print_every,
                                 ctx.jit_cost_terms)
        phases.mark("t_log")
        if reached:
            break
        # Same placement the sim's flat loop uses: after the success check,
        # reading the errors that check just used.
        params = ctx.kicker.maybe_kick(params, ctx.log["pos_err"][-1],
                                   ctx.log["theta_err"][-1], step, ctx.verbose,
                                   tip_xy=np.asarray(ctx.log["robot_pos"][-1]))

    ctx.interface.stop()
    phases.finish()
    _log_publish_stamps(ctx.log, {})
    if ctx.verbose:
        line = _phase_summary(ctx.log)
        if line:
            print(line)
    return _finish(ctx.log, ctx.task, ctx.base_data, reached, ctx.admm,
                   params, ctx.jit_plans,
                   None if live_plans else plan_knots, ctx.verbose)

def _rebase_time(world, t_run0):
    """The world state with its clock restarted at the run's own step 0.

    The interface clock starts when the interface is created, which is
    BEFORE the JIT warm-up -- so the planner's `state.time` at the first
    control step was 47 s on 2026-09-05 and 407 s on 2026-09-06 (194 s
    compile). Every time-keyed cost read it: `time_ramp` / `_q_ramp_mult`
    (goal gains, `q_ramp_per_step` per control step of 0.05 s) sat at
    1 + 0.023 * 407 / 0.05 = 188 at step 1, and at its cap of 25 from
    step 1 on every earlier hardware run. Rebasing to the run's own start
    makes the ramp count control steps, as in sim, and makes results
    independent of compile time. Logged `time` follows the same clock.
    """
    return dataclasses.replace(world, time=float(world.time) - t_run0)

def _run_overlapped(ctx: LoopContext, params: Any) -> Dict[str, Any]:
    """Hardware loop: a publisher thread streams the latest plan while the main
    thread keeps solving, so execution and planning overlap.

    With `latency_comp > 0` (see `run_real`) each solve starts from the arm
    state PREDICTED at the moment its plan will start being executed, and
    the plan's clock is anchored there, so the plan's head is what the arm
    runs instead of a segment ~0.3 s in. That anchor is `t_loop + lat`: the
    plan's index 0 is the command for the state the solve was given, which
    is where the arm is predicted to be one `lat` after the read.

    `handoff` picks how the anchor is chosen and what happens when the solve
    and the anchor disagree. The two are the same in the LATE case (solve
    slower than the anchor: the arm has genuinely moved past the predicted
    state, so the publisher enters the plan `elapsed` in). They differ in
    the EARLY case:

    * ``"responsive"`` -- `lat` is the EMA of measured solve time, and a
      plan is published the moment it is ready. An early solve means the
      arm has NOT yet reached the predicted state, so `elapsed` is negative
      and the publisher clamps it to 0, entering at index 0. The arm is
      then slightly behind the state the plan assumed, bounded by the EMA's
      own error and corrected by the next solve. Period = solve time.
    * ``"deterministic"`` -- `lat` is pinned to `t_c` (no EMA), and an
      early solve WAITS until the anchor before publishing. Prediction
      horizon and switch point are then the same constant, so every cycle
      is `t_c` long and `elapsed` is never negative. Costs latency: the
      arm runs `t_c` behind even when a solve took half that.

    Mixing the two -- an EMA anchor that is also waited for -- is the one
    combination to avoid: the period becomes `lat`, which drifts, so it
    buys neither reproducibility nor responsiveness.

    Args:
        ctx: The run's fixed configuration. `ctx.handoff` picks the policy
            above and `ctx.t_c` is its fixed anchor offset [s] -- also the
            value `lat` is pinned to, and ignored under ``"responsive"``.
        params: The warm-started policy parameters to solve from.

    Returns:
        The finished log.
    """
    if ctx.handoff not in ("responsive", "deterministic"):
        raise ValueError(
            "handoff must be 'responsive' or 'deterministic', got "
            f"{ctx.handoff!r}"
        )

    # Live only when the overlay or the ghost marker draws them (see
    # `_run_serial`); otherwise rebuilt after the loop.
    live_plans = ctx.vis_model is not None and (
        ctx.show_optimal or ctx.show_object_plan
    )
    plan_knots: List[Any] = []
    ctx.log.setdefault("loop_time", [])
    t_solve_prev = None

    def _plan_displacement(s, t0, t1):
        """Joint displacement the publisher's plan `s` produces between
        plan-times t0 and t1 [s] (zero beyond the plan's end, same as the
        publisher sends). Exact integral of the piecewise-constant stream."""
        n = len(s)
        dq = np.zeros_like(s[0])
        if t1 <= t0:
            return dq
        lo = max(t0, 0.0)
        hi = min(t1, n * ctx.control_dt)
        if hi <= lo:
            return dq
        i0 = int(lo / ctx.control_dt)
        i1 = int(hi / ctx.control_dt)
        if i0 == i1:
            return s[i0] * (hi - lo)
        dq = s[i0] * ((i0 + 1) * ctx.control_dt - lo)
        if i1 > i0 + 1:
            dq = dq + s[i0 + 1:i1].sum(axis=0) * ctx.control_dt
        if i1 < n:
            dq = dq + s[i1] * (hi - i1 * ctx.control_dt)
        return dq

    def _clip_stream_to_limit(samples):
        """The plan as the publisher will send it: `clamp_velocity` per
        sample. `_plan_displacement` used to integrate the raw samples,
        over-predicting whenever a plan ran past `vel_limit`."""
        return np.stack([clamp_velocity(u, ctx.vel_limit) for u in samples])

    def _sample_plan(plan):
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
        n = max(1, int(span / ctx.control_dt) + 1)
        ts = jnp.arange(n) * ctx.control_dt + float(tk[0])
        return np.asarray(ctx.jit_interp(ts, plan.tk, plan.mean[None, ...]))[0]

    # The model's joint ranges, minus a margin, as the executed plan's own
    # limit. The rollouts already stop at the model range (limited="true"),
    # but nothing stopped the real arm: a plan solved AT the limit still
    # commands velocity into it, the arm keeps integrating, and the next
    # solve starts from a qpos outside the model's range -- where the limit
    # constraint's spring-back dominated the rollouts and sent the tip
    # 0.2-0.6 m up (2026-09-05 16:03 / 16:27, J3 measured -148 against a
    # -120 range). Clipping the executed plan to the same range keeps model
    # and arm in agreement at the limit; a plan that never nears it is
    # returned bit-identical.
    _jnt_lo = np.array([
        ctx.task.mj_model.jnt_range[ctx.task.mj_model.joint(n).id][0]
        for n in ARM_JOINT_NAMES
    ])
    _jnt_hi = np.array([
        ctx.task.mj_model.jnt_range[ctx.task.mj_model.joint(n).id][1]
        for n in ARM_JOINT_NAMES
    ])
    _jnt_margin = math.radians(2.0)

    def _clip_plan_to_joint_range(samples, q0):
        """Sequentially clip joint velocities so the integrated joint
        trajectory, starting from `q0` (the arm state the plan was solved
        for), stays inside [lo + margin, hi - margin]."""
        q = np.asarray(q0, dtype=float).copy()
        out = np.array(samples, dtype=float, copy=True)
        lo = _jnt_lo + _jnt_margin
        hi = _jnt_hi - _jnt_margin
        for i in range(out.shape[0]):
            v_lo = (lo - q) / ctx.control_dt
            v_hi = (hi - q) / ctx.control_dt
            out[i] = np.clip(out[i], np.minimum(v_lo, 0.0), np.maximum(v_hi, 0.0))
            q = q + out[i] * ctx.control_dt
        return out

    # Seed the publisher with a plan solved from the state the arm is in RIGHT
    # NOW, not the one `params` carries out of warm-up.
    #
    # The warm-up plan was solved against the state assembled before the JIT
    # passes -- by the time the loop starts that is 13+ seconds stale, and its
    # mean is close to the zero seed, so the arm stood still for one whole
    # solve period between "[jit] ready" and step 0. Visible on hardware as a
    # pause right after the run announces itself.
    #
    # This costs one extra solve before the thread starts (~0.15 s here) and
    # removes the gap: the publisher's first tick carries a plan for the
    # current state. `t_perf` is stamped at the READ, matching what the loop
    # does with every plan after it.
    t_seed = time.perf_counter()
    _world0 = ctx.interface.read_state()
    # The run's clock starts here: the seed plan is the first one executed.
    t_run0 = float(_world0.time)
    _world0 = _rebase_time(_world0, t_run0)
    _seed_params, _ = ctx.jit_optimize(
        _assemble_state(ctx.task, ctx.base_data, ctx.addresses, _world0), params
    )
    jax.block_until_ready(_seed_params)
    params = _seed_params

    # Shared latest plan, guarded by a lock. `samples` is the plan already
    # materialised on a control-tick grid; `t_perf` is the wall clock when it
    # was published, so the publisher can index into it by elapsed time.
    # `qpos`/`traces` are for the display thread below, not the publisher --
    # set to real values on the loop's first iteration, before either thread
    # that reads them can start.
    lock = threading.Lock()
    shared = {"samples": _sample_plan(params),
              "t_perf": t_seed,
              "qpos": None, "traces": [], "mocap": None,
              "gen": -1}  # step index of the plan in `samples`; -1 = seed
    stop = threading.Event()
    # step -> (ros, perf, index) of the FIRST command the publisher sent
    # out of that step's plan. Written by the publisher, read after the loop.
    pub_first: Dict[int, Tuple[float, float, int]] = {}
    # Every command the publisher sent, (perf, clamped u), last ~10 s: the
    # fallback record of what the arm has been told when the interface
    # cannot see the executed (post-CBF) stream. Written by the publisher
    # under `lock`, read by the solve loop.
    sent_log: Deque[Tuple[float, np.ndarray]] = deque(maxlen=512)

    def _publisher() -> None:
        next_tick = time.perf_counter()
        gen_seen = -1
        while not stop.is_set():
            with lock:
                s = shared["samples"]
                t_perf = shared["t_perf"]
                gen = shared["gen"]

            # Both the negative-`elapsed` clamp and the exhausted-plan zero
            # live in `publish_index` -- module-level and pure, so the
            # handoff policies are testable without hardware (a `--mock`
            # run executes `_run_serial`, never this loop).
            idx = publish_index(
                time.perf_counter() - t_perf, len(s), ctx.control_dt
            )
            u = np.zeros_like(s[0]) if idx is None else s[idx]

            u_sent = clamp_velocity(u, ctx.vel_limit)
            ctx.interface.send_velocity(u_sent)
            with lock:
                sent_log.append((time.perf_counter(), np.asarray(u_sent, dtype=float)))
            if gen != gen_seen and idx is not None:
                st = ctx.interface.last_publish_stamps()
                if st is not None:
                    pub_first[gen] = (st[0], st[1], int(idx))
                gen_seen = gen
            next_tick += ctx.control_dt
            sleep = next_tick - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)
            else:  # publisher fell behind; resync rather than spiral
                print("[WARN]: publisher falling behind")
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
                plan_span = n * ctx.control_dt
                elapsed = min(time.perf_counter() - t_perf, plan_span)
                idx_full = min(int(elapsed / ctx.control_dt), n)
                partial = elapsed - idx_full * ctx.control_dt
                v_partial = s[idx_full] if idx_full < n else np.zeros_like(s[0])
                integral = (s[:idx_full].sum(axis=0) * ctx.control_dt
                            + v_partial * partial)

                # Into `mj_data_cpu`, not a private copy: the viewer is
                # bound to THAT MjData, so anything written elsewhere is
                # never displayed -- the dead reckoning below used to be
                # computed and then thrown away. Under `vis_lock` because
                # the solve thread writes the same object from
                # `_visualize_step`; the two now alternate whole updates
                # instead of interleaving halves of one.
                # Live sources when the interface offers them (Ros2Interface
                # `peek_*`: read-only copies, no filter state, no solver
                # involvement): the arm at the encoder rate instead of the
                # dead-reckoned plan, the block at the TF rate instead of
                # the pose held since the last solve. Either falls back to
                # the previous behaviour when unavailable.
                arm_live = getattr(
                    ctx.interface, "peek_arm_qpos", lambda: None)()
                obj_live = getattr(
                    ctx.interface, "peek_object_se2", lambda: None)()
                with ctx.vis_lock:
                    ctx.mj_data_cpu.qpos[:] = qpos
                    if arm_live is not None:
                        ctx.mj_data_cpu.qpos[
                            ctx.addresses.arm_qpos_adr] = arm_live
                    else:
                        ctx.mj_data_cpu.qpos[
                            ctx.addresses.arm_qpos_adr] += integral
                    if obj_live is not None:
                        ctx.mj_data_cpu.qpos[
                            ctx.addresses.block_qpos_adr] = obj_live
                    # The object_plan ghost (if any) only ever changes once
                    # per solve too, same as the object -- copied in, not
                    # recomputed: recomputing calls into JAX (see
                    # object_plan_marker), which this thread must never do.
                    if mocap is not None:
                        ctx.mj_data_cpu.mocap_pos[:] = mocap[0]
                        ctx.mj_data_cpu.mocap_quat[:] = mocap[1]
                    mujoco.mj_forward(ctx.vis_model, ctx.mj_data_cpu)
                    if ctx.overlay is not None:
                        ctx.overlay.draw(
                            ctx.viewer.user_scn, traces, base=ctx.overlay_base
                        )
                    ctx.viewer.sync()
            sleep = period - (time.perf_counter() - t_tick)
            if sleep > 0:
                time.sleep(sleep)

    disp = threading.Thread(target=_display_loop, daemon=True)
    if ctx.viewer is not None:
        disp.start()

    reached = False
    # Latency compensation state: the current estimate of read-to-publish
    # time. Under `responsive` this is an EMA of what each iteration
    # measures; under `deterministic` it is pinned to `t_c` and never
    # updated, so the prediction horizon and the publish instant are the
    # same constant.
    deterministic = ctx.handoff == "deterministic"
    if deterministic:
        lat = float(ctx.t_c)
    else:
        lat = float(ctx.latency_comp) if ctx.latency_comp > 0.0 else 0.0
    ctx.log.setdefault("latency_pred", [])
    # Command-to-motion delay of the arm itself; constant, see run_real().
    tau = max(float(ctx.actuation_delay), 0.0)
    # displacement from commands already sent
    ctx.log.setdefault("pred_dq_past", [])
    # ... and from the plan about to be sent
    ctx.log.setdefault("pred_dq_future", [])
    # 0 = own send log, 1 = executed (CBF output)
    ctx.log.setdefault("pred_stream", [])
    # Under `deterministic`, how long each iteration sat idle waiting for
    # its anchor. Zero everywhere under `responsive`.
    ctx.log.setdefault("handoff_wait", [])
    # Collision-stop watchdog state -- see the check at the top of the loop.
    stall_solves = 0
    # Tilt watchdog state -- see the check after _log_and_check below.
    tilt_solves = 0
    tilt_stop_rad = np.radians(45.0)
    prev_samples = shared["samples"]
    step = -1  # defined before the try, so the handlers below can name it even
    #            if the interrupt lands on the very first iteration
    interrupt = _InterruptFlag().install()
    try:
        for step in range(ctx.max_steps):
            if interrupt.requested or (
                ctx.viewer is not None and not ctx.viewer.is_running()
            ):
                break
            t_loop = time.perf_counter()
            world = _rebase_time(ctx.interface.read_state(), t_run0)
            _log_read_stamps(ctx.log, world)
            # Collision-stop watchdog. The xArm's own protection freezes the
            # motors on impact but tells this process nothing, so a run used
            # to keep solving and publishing at a frozen arm until a human
            # hit Ctrl-C (2026-08-28 16:29 run: 5+ solves after the stop,
            # with the logged pose still drifting). Signature: a plainly
            # nonzero command stream against measured joint speeds at zero,
            # for several consecutive solves. 0.05 rad/s commanded is well
            # above deliberate stillness, 0.005 rad/s measured is the
            # encoder noise floor, and 3 solves (~2 s) rides out one stale
            # /joint_states read. A CBF hard-block produces the same
            # signature and the same conclusion: the run cannot continue.
            cmd_mag = float(np.max(np.abs(prev_samples[:40])))
            meas_mag = float(np.max(np.abs(np.asarray(world.arm_qvel))))
            if cmd_mag > 0.05 and meas_mag < 0.005:
                stall_solves += 1
            else:
                stall_solves = 0
            if stall_solves >= 3:
                print("[stop] arm not tracking its commands "
                      f"(|u|={cmd_mag:.2f} rad/s commanded, "
                      f"|qvel|={meas_mag:.4f} rad/s measured, 3 consecutive "
                      "solves) -- collision stop assumed; stopping and saving")
                break
            mjx_data = _assemble_state(
            ctx.task, ctx.base_data, ctx.addresses, world
        )
            # Predicted start state for the solve. `mjx_data` (measured)
            # is what gets logged; `mjx_solve` is what the planner sees.
            mjx_solve = mjx_data
            if lat > 0.0 or tau > 0.0:
                with lock:
                    s_exec = shared["samples"]
                    t_exec = shared["t_perf"]
                    sent_t = np.array([t for t, _ in sent_log])
                    sent_u = (np.stack([u for _, u in sent_log])
                              if sent_log else np.zeros((0, len(ARM_JOINT_NAMES))))
                # The measured qpos is valid at the joint_states receipt,
                # a few ms before t_loop; the arm has been executing the
                # commands sent up to `tau` before that (see
                # `window_split`). Past window: the executed record (CBF
                # output) when the interface has one, else our own send
                # log. Future window: the plan samples about to be sent,
                # clamped the way the publisher clamps them.
                st = getattr(world, "stamps", None) or {}
                t_state = float(st.get("perf_js_recv", t_loop))
                if not np.isfinite(t_state):
                    t_state = t_loop
                (p_lo, p_hi), (f_lo, f_hi) = window_split(
                    t_state, t_loop, t_loop + lat, tau)
                stream = getattr(
                    ctx.interface, "executed_stream", lambda: None)()
                if stream is None:
                    stream = (sent_t, sent_u)
                dq_past = integrate_stream(stream[0], stream[1], p_lo, p_hi)
                e0 = f_lo - t_exec
                dq_future = _plan_displacement(
                    _clip_stream_to_limit(s_exec), e0, e0 + (f_hi - f_lo))
                dq = dq_past + dq_future
                world_pred = dataclasses.replace(
                    world,
                    arm_qpos=np.asarray(world.arm_qpos) + dq,
                    time=float(world.time) + lat + tau,
                )
                mjx_solve = _assemble_state(ctx.task, ctx.base_data,
                                            ctx.addresses, world_pred)
                ctx.log["pred_dq_past"].append(np.asarray(dq_past).tolist())
                ctx.log["pred_dq_future"].append(np.asarray(dq_future).tolist())
                ctx.log["pred_stream"].append(
                    0 if stream is not None and stream[0] is sent_t else 1)

                # Publish the prediction, stamped with the instant it is
                # for, so a probe (contactmpc/analysis/step_test/
                # prediction_probe.py) or a bag can score it against
                # /joint_states. 2-3 Hz, one small message; harmless.
                send_pred = getattr(ctx.interface, "send_joint_state", None)
                st_now = st.get("now")
                if send_pred is not None and st_now is not None:
                    send_pred(world_pred.arm_qpos, float(st_now) + lat + tau)
            ctx.log["latency_pred"].append(lat)

            t0 = time.perf_counter()
            # Solve-start to solve-start: how often a fresh plan reaches the
            # publisher, i.e. the solve plus the loop's tail around it.
            ctx.log["loop_time"].append(
                float("nan") if t_solve_prev is None else t0 - t_solve_prev
            )
            t_solve_prev = t0
            params, rollouts = ctx.jit_optimize(mjx_solve, params)
            jax.block_until_ready(params)
            ctx.log["compute_time"].append(time.perf_counter() - t0)

            # Hand the fresh plan to the publisher (and the display thread's
            # dead-reckoning base -- same anchor time, same reasoning).
            samples = _clip_plan_to_joint_range(
                _sample_plan(params),
                np.asarray(mjx_solve.qpos)[ctx.addresses.arm_qpos_adr],
            )
            prev_samples = samples
            # `deterministic`: hold the finished plan until its own anchor.
            # The plan's index 0 is the command for the state predicted at
            # `t_loop + t_c`; publishing before the arm has reached that
            # state would mean commanding index 0 to a state that is not yet
            # index 0's state. Waiting makes the prediction and the switch
            # the same instant, which is the whole point of this mode -- and
            # it is what fixes the negative-`elapsed` case at the source
            # rather than clamping it. A solve that OVERRAN `t_c` skips the
            # wait entirely and enters the plan `elapsed` in, exactly as
            # `responsive` does.
            wait = 0.0
            if deterministic:
                wait = max((t_loop + lat) - time.perf_counter(), 0.0)
                if wait > 0.0:
                    time.sleep(wait)
            ctx.log["handoff_wait"].append(wait)
            t_pub = time.perf_counter()
            with lock:
                shared["samples"] = samples
                # The plan's s[0] is the control for the state read at
                # `t_loop`, one solve ago -- the arm has been executing the
                # previous plan since. Anchor plan time to that read, not to
                # now, so the publisher enters the plan where the present
                # actually is instead of replaying a moment that has passed.
                # With latency compensation the plan was solved for the
                # state predicted at `t_loop + lat`, so that is its t = 0.
                shared["t_perf"] = t_loop + lat
                shared["qpos"] = np.asarray(mjx_solve.qpos)
                shared["gen"] = step
            if lat > 0.0 and not deterministic:
                # Track the latency the plan actually experienced. The EMA
                # keeps one slow solve (JIT recompile, GC pause) from
                # throwing the next prediction. Skipped under
                # `deterministic`, where `lat` IS the constant `t_c` -- an
                # EMA there would drift the anchor and turn a fixed period
                # back into a variable one.
                lat = 0.8 * lat + 0.2 * (t_pub - t_loop)

            # Deliberately after the hand-off above: a diagnostic, and the
            # publisher must not wait on one. Reduced on the device.
            ctx.reducer(ctx.log, rollouts, params,
                        ctx.task._block_pose(mjx_solve))

            # Log the command the publisher would send at the solve instant.
            first = samples[:1]
            obj_plan = rob_plan = robot_trace = None
            if ctx.admm and live_plans:
                obj_plan, rob_plan, robot_trace = ctx.jit_plans(
                    mjx_data, params)
                ctx.log["object_plan"].append(np.asarray(obj_plan))
                ctx.log["robot_plan"].append(np.asarray(rob_plan))
            elif ctx.admm:
                plan_knots.append(_plan_knots(params))
            elif ctx.jit_trace is not None and ctx.vis_model is not None:
                robot_trace = ctx.jit_trace(mjx_data, params)
            if ctx.mj_data_cpu is not None:
                ctx.draw_object_plan(ctx.mj_data_cpu, mjx_data, params,
                                     obj_plan)
            # After the hand-off above and _log_sample_stats, same rule:
            # rendering is a diagnostic, and the publisher must not wait on
            # one. Measured safe from this thread against the Warp/JAX
            # solver on the mock -- see the mock diagnostic in Tasks.md.
            # sync_viewer=False: the display thread above owns
            # viewer.sync() exclusively, so this call only feeds the
            # recorder directly; its returned traces are handed to that
            # thread instead of it recomputing traces_for itself.
            traces = _visualize_step(
                ctx.vis_model, mjx_data, ctx.mj_data_cpu, ctx.recorder,
                ctx.overlay, ctx.viewer, ctx.overlay_base, rollouts, params,
                ctx.admm, ctx.show_samples,
                ctx.show_optimal,
                obj_plan=None if obj_plan is None else np.asarray(obj_plan),
                rob_plan=None if rob_plan is None else np.asarray(rob_plan),
                robot_trace=(
                    None if robot_trace is None else np.asarray(robot_trace)
                ),
                sync_viewer=False,
                vis_lock=ctx.vis_lock,
            )
            if ctx.viewer is not None:
                # object_plan's ghost pose, same hand-off reasoning as qpos
                # above -- the display thread copies these rather than ever
                # calling draw_object_plan itself. Read under `vis_lock` and
                # OUTSIDE `lock`: the display thread takes `lock` first and
                # `vis_lock` second, so taking them in that order here too
                # is what keeps the pair acyclic.
                with ctx.vis_lock:
                    mocap_snapshot = (
                        ctx.mj_data_cpu.mocap_pos.copy(),
                        ctx.mj_data_cpu.mocap_quat.copy(),
                    )
                with lock:
                    shared["traces"] = traces
                    shared["mocap"] = mocap_snapshot
            reached = _log_and_check(ctx.log, ctx.task, mjx_data, params, first,
                                     ctx.goal_pos_tol, ctx.goal_theta_tol, step,
                                     ctx.verbose, ctx.admm, ctx.print_every,
                                     ctx.jit_cost_terms)
            # The previous step's plan has certainly been published by now;
            # this step's may not have. Report one step behind.
            if ctx.verbose and int(ctx.print_every) > 0 and step > 0 \
                    and step % int(ctx.print_every) == 0:
                _log_publish_stamps(ctx.log, dict(pub_first))
                line = _stamp_summary(ctx.log, step - 1)
                if line:
                    print("           " + line)
            if reached:
                break
            # Tilt watchdog. A tool laid past ~45 deg cannot push, and once
            # the wrist folds the sampler cannot find its way back (23:14
            # run: tilt 54-82 deg for 120 steps, block never moved).
            # Sustained, not instantaneous -- good pushes brush 35-42 deg
            # for a step or two.
            tilt = float(ctx.log["tip_tilt"][-1])
            tilt_solves = tilt_solves + 1 if tilt > tilt_stop_rad else 0
            if tilt_solves >= 6:
                print(f"[stop] tip tilt {np.degrees(tilt):.0f} deg for "
                      f"{tilt_solves} consecutive solves -- wrist folded, "
                      "unrecoverable; stopping and saving")
                break
            # The kick only rewrites the sampling mean the NEXT solve starts
            # from; the publisher keeps streaming the plan already handed to
            # it, so nothing the arm is executing changes discontinuously.
            params = ctx.kicker.maybe_kick(params, ctx.log["pos_err"][-1],
                                       ctx.log["theta_err"][-1], step,
                                       ctx.verbose,
                                       tip_xy=np.asarray(ctx.log["robot_pos"][-1]))
    except RuntimeError as exc:
        # The interface gave up on the object -- see `Ros2Interface._hold`.
        # Handled exactly like Ctrl-C rather than propagating: `finally`
        # below stops the arm either way, but only falling through here
        # reaches `finalize_log`/`save_run`, and the steps leading up to a
        # lost block are the ones worth keeping.
        print(f"\n[stop] {exc}")
        print("[stop] stopping the arm and saving what ran")
    except KeyboardInterrupt:
        # Ctrl-C is how a hardware run normally ENDS -- nobody waits out
        # `--steps 1500` once the answer is visible. Letting the exception
        # leave this function skipped `finalize_log` and every `save_run`
        # below it, so the runs worth looking at were exactly the ones with no
        # run file. Swallowed here, at the loop, rather than in `main`: the log
        # lives in this frame, and the `finally` below still stops the arm.
        if ctx.verbose:
            print(f"\ninterrupted at step {step}; finalising the log")
    finally:
        interrupt.restore()
        stop.set()
        pub.join(timeout=1.0)
        if ctx.viewer is not None:
            disp_stop.set()
            # Joined WITHOUT a timeout: on timeout the daemon thread keeps
            # running straight into the viewer teardown below and syncs a
            # half-destroyed viewer. It only ever waits one 30 Hz tick.
            disp.join()
        ctx.interface.stop()
    _log_publish_stamps(ctx.log, dict(pub_first))
    if ctx.verbose:
        print(f"stopped at step {step}; "
              f"{'goal reached' if reached else 'saving'}")
        line = _stamp_summary(ctx.log)
        if line:
            print(line)
    return _finish(ctx.log, ctx.task, ctx.base_data, reached, ctx.admm,
                   params, ctx.jit_plans,
                   None if live_plans else plan_knots, ctx.verbose)

def _log_and_check(
    log, task, mjx_data, params, applied, goal_pos_tol, goal_theta_tol, step,
    verbose, admm=True, print_every=10, jit_cost_terms=None,
) -> bool:
    """Append one step to the log and return whether the goal was reached.

    The `c_*` cost decomposition is NOT appended here any more: it is
    reconstructed from the logged states after the loop (`_finish`), and
    only evaluated live for the console print.
    """
    block_pose = log_step(log, task, mjx_data, params, applied, admm=admm)
    goal = np.asarray(task.goal)
    pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
    theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)
    print_every = int(print_every)
    if verbose and print_every > 0 and step % print_every == 0:
        primal = ""
        if admm:
            # The residuals alone say the two blocks disagree; the DUALS say
            # what that disagreement is doing. `y <- y + rho*(A - z)` every
            # iteration, so a residual that never shrinks makes them grow
            # without bound, and the consensus penalty they carry then swamps
            # both blocks' own costs. A rising |y| is the signal that ADMM has
            # stopped being a solver and become a constant bias. Norms, not
            # the vectors: the direction is in the run file, the magnitude is
            # what has to be watched live.
            y_o = float(np.linalg.norm(np.asarray(log["dual_object"][-1])))
            y_r = float(np.linalg.norm(np.asarray(log["dual_robot"][-1])))
            # rho PER CHANNEL. `log["rho"]` stores `np.mean(params.rho)`, and
            # rho_init is [rho, rho, rho_torque] = [1, 1, 10] here, so that
            # mean reads 4.0 and looks like a value nobody configured.
            rho = np.atleast_1d(np.asarray(params.rho, dtype=float))
            primal = (f"primal={log['primal_residual'][-1]:.3f} "
                      f"dual={log['dual_residual'][-1]:.3f}  "
                      f"|y_o|={y_o:.2f} |y_r|={y_r:.2f} "
                      f"rho=[{' '.join(f'{v:g}' for v in rho)}]  ")
        # eta on the console, not only in the run file: a flat run that has
        # gone uninformative (eta at num_samples, or any nonfinite sample)
        # otherwise looks exactly like one that is working, and there is no
        # point letting 1000 steps finish before finding that out.
        pop = ""
        if log.get("sample_eta"):
            bad = log["sample_nonfinite"][-1]
            obj_pop = ""
            if log.get("object_eta"):
                obj_pop = (f"\n           obj_eta={log['object_eta'][-1]:.1f} "
                           f"obj_move={log['object_moving_frac'][-1] * 100:3.0f}%  ")
            pop = (obj_pop + f"eta={log['sample_eta'][-1]:.1f}  "
                   f"T*={log['sample_temp_star'][-1]:.0f}  "
                   f"cost={log['sample_cost_min'][-1]:.2f}"
                   f"+-{log['sample_cost_std'][-1]:.2f}  "
                   + (f"NONFINITE={bad}  " if bad else ""))
        # The one line that says whether a stall is an exploration failure or
        # a ranking failure. `touch` at 0% means no sampled rollout reached
        # the object at all -- no weight change can fix that. Above 0, `gap`
        # and `rank` say whether the softmax then preferred those samples.
        con = ""
        if log.get("sample_contact_frac"):
            frac = log["sample_contact_frac"][-1]
            gap, rank = log["sample_contact_gap"][-1], log["sample_contact_rank"][-1]
            con = (f"touch={frac * 100:3.0f}%  "
                   + ("" if frac <= 0.0 else
                      f"gap={gap:+.1f} rank={rank:.2f}  "))
        print(f"step {step:4d}  pos_err={pos_err:.4f}  theta_err={theta_err:.4f}  "
              f"{primal}{pop}{con}plan={log['compute_time'][-1] * 1e3:.0f}ms"
              + (f"  loop={log['loop_time'][-1] * 1e3:.0f}ms"
                 if log.get("loop_time")
                 and np.isfinite(log["loop_time"][-1]) else "")
              + (f"  lat={log['latency_pred'][-1] * 1e3:.0f}ms"
                 if log.get("latency_pred") else ""))
        # `block_pose` is the SE(2) read back out of the ASSEMBLED MJX state,
        # i.e. what the cost function is actually optimising against -- not the
        # TF reading. If this disagrees with tf2_echo, the bug is in
        # _lookup_object_se2 or _assemble_state, not in the planner.
        # tip (x, y), world frame
        tip = np.asarray(log["robot_pos"][-1])
        d_tip = float(np.linalg.norm(tip[:2] - np.asarray(block_pose)[:2]))
        u = np.asarray(log["robot_control"][-1])
        fz = (float(log["contact_normal_force_z"][-1])
              if log.get("contact_normal_force_z") else float("nan"))
        ov = np.asarray(log["object_velocity"][-1])
        obj_speed = float(np.linalg.norm(ov[:2]))
        print(f"           obj=({block_pose[0]:+.4f},{block_pose[1]:+.4f},"
              f"{np.degrees(block_pose[2]):+6.1f}d)"
              f"  tip=({tip[0]:+.4f},{tip[1]:+.4f})"
              f"  z={log['tip_z'][-1] * 1e3:5.1f}mm"
              f"  tilt={np.degrees(log['tip_tilt'][-1]):4.1f}d"
              f"  d_tip={d_tip:.4f}  Fz={fz:6.2f}N")
        c = _cost_terms(task, mjx_data, jit_cost_terms)
        if c:
            print(f"           cost: goal={c.get('c_goal', float('nan')):8.1f}"
                  f"  approach={c.get('c_approach', float('nan')):7.2f}"
                  f"  align={c.get('c_align', float('nan')):6.2f}"
                  f"  tilt={c.get('c_tilt', float('nan')):6.2f}"
                  f"  ztip={c.get('c_ztip', float('nan')):8.2f}"
                  f"  fade={c.get('c_fade', float('nan')):.2f}")
        print(f"           |u|max={np.max(np.abs(u)):.3f}"
              f"  u=[{' '.join(f'{v:+.2f}' for v in u)}]"
              f"  obj_speed={obj_speed * 1e3:6.2f}mm/s"
              f"  obj_w={np.degrees(ov[2]):+6.1f}d/s")
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
    # Block twist feeds realized_consensus (w = wrench_limit * qvel[block_dofs]).
    qvel[addresses.block_dof_adr] = world.object_twist
    assert qpos.shape[0] == nq, (qpos.shape, nq)
    mjx_data = base_data.replace(
        qpos=jnp.asarray(qpos), qvel=jnp.asarray(qvel), time=float(world.time)
    )
    return _jit_forward(task.model, mjx_data)
