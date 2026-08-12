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

REAL-TIME MODEL. `ADMM.optimize` is ~0.36 s/step, longer than the planning
timestep (`plan_dt` = 0.05 s), so a single-thread "solve, then publish, then
solve" loop leaves the arm stalled on the last command during every solve.
For hardware (`real_time=True`) we therefore split the two:

    - main thread:   read state -> optimize -> post the fresh plan; repeat.
    - publisher thread: at `control_rate`, sample the latest plan at the
      current time and publish one velocity command.

The publisher keeps feeding the arm from the current plan *while* the main
thread computes the next one, so execution and planning overlap (this is the
frame-stacking Brian described: the number of commands sent per solve is
`control_rate * compute_time`, and the plan horizon `H*plan_dt` = 0.75 s
comfortably exceeds the ~0.36 s compute time, so there is always a valid
sample). We stay on the velocity path (`commands_nominal` -> distance_cbf ->
`commands`) rather than a trajectory controller, because the CBF filter only
sits on the velocity topic.

`command_mode="stream"` follows the plan's time-varying velocity; `"hold"`
sends the plan's first velocity until the next solve (OI-MPPI's behaviour).
The mock path (`real_time=False`) stays single-threaded and deterministic --
MuJoCo is not thread-safe -- which is all the plumbing validation needs.

The state log uses the same keys/schema as `sim3d/run.py`, so a hardware run
and a simulation run compare entry-for-entry.
"""

from __future__ import annotations

import threading
import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from oim.objects import wrap_angle
from oim.real3d.interface import RobotWorldInterface, SceneAddresses, clamp_velocity
from oim.sim3d.run import _finalize_log, _init_log, _log_step
from oim.tasks.pusht import PushT

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
    command_mode: str = "stream",
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    real_time: bool = False,
    vel_limit: float = 0.2,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the push-T ADMM controller against a `RobotWorldInterface`.

    Args:
        task: the `PushT` task, built with `robot="xarm6"`.
        ctrl: the ADMM controller, built against `task`.
        params: initial policy parameters (`ctrl.init_params(...)`).
        interface: hardware or mock world.
        replan_rate: mock only -- how much sim time one solve covers.
        control_rate: rate (Hz) at which velocity commands are published.
        command_mode: "stream" follows the plan's velocity profile; "hold"
            sends the plan's first velocity until the next solve.
        max_steps: maximum solves.
        goal_pos_tol, goal_theta_tol: success tolerances.
        real_time: True -> hardware (threaded, overlapped); False -> mock
            (single-threaded, deterministic).
        verbose: print progress.

    Returns:
        A log dict with the same schema as `sim3d.run.run_3d_admm`.
    """
    if command_mode not in ("hold", "stream"):
        raise ValueError(f"command_mode must be 'hold' or 'stream', got {command_mode!r}")
    addresses = SceneAddresses.from_model(task.mj_model)
    control_dt = 1.0 / control_rate

    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp = jax.jit(ctrl.interp_func)

    # First state + JIT warm-up before any timed loop.
    t = time.perf_counter()
    base_data = task.make_data()
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
    if verbose:
        print(f"[jit] loop-path warm-up: {time.perf_counter() - t:.1f}s")

    if verbose:
        print(f"[jit] ready; {'overlapped' if real_time else 'serial'} loop, "
              f"control {control_rate:.0f} Hz, {command_mode}")

    log = _init_log(task, mjx_data, mjx_data, show_plans=False)
    common = dict(
        task=task, interface=interface, addresses=addresses, base_data=base_data,
        jit_optimize=jit_optimize, jit_interp=jit_interp, command_mode=command_mode,
        control_dt=control_dt, max_steps=max_steps, goal_pos_tol=goal_pos_tol,
        goal_theta_tol=goal_theta_tol, vel_limit=vel_limit, log=log, verbose=verbose,
    )
    if real_time:
        return _run_overlapped(params=params, **common)
    return _run_serial(params=params, replan_rate=replan_rate, **common)


def _run_serial(
    task, interface, addresses, base_data, jit_optimize, jit_interp, command_mode,
    control_dt, replan_rate, max_steps, goal_pos_tol, goal_theta_tol, vel_limit, log,
    verbose, params,
) -> Dict[str, Any]:
    """Single-threaded loop: solve, then publish the window, then repeat.

    Used for the mock (deterministic, MuJoCo not thread-safe). The arm stalls
    on the last command during each solve, which is fine off-hardware.
    """
    replan_period = 1.0 / replan_rate
    num_ticks = max(1, round(replan_period / control_dt))
    reached = False

    for step in range(max_steps):
        world = interface.read_state()
        mjx_data = _assemble_state(task, base_data, addresses, world)

        t0 = time.perf_counter()
        params, _ = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        log["compute_time"].append(time.perf_counter() - t0)

        sample_times = jnp.arange(num_ticks) * control_dt + world.time
        plan_samples = np.asarray(
            jit_interp(sample_times, params.tk, params.mean[None, ...])
        )[0]
        applied = np.empty_like(plan_samples)
        for i in range(num_ticks):
            velocity = plan_samples[0] if command_mode == "hold" else plan_samples[i]
            applied[i] = clamp_velocity(velocity, vel_limit)
            interface.send_velocity(applied[i])

        reached = _log_and_check(log, task, mjx_data, params, applied,
                                 goal_pos_tol, goal_theta_tol, step, verbose)
        if reached:
            break

    interface.send_velocity(np.zeros(len(addresses.arm_dof_adr)))
    return _finalize_log(log, task, reached, show_plans=False)


def _run_overlapped(
    task, interface, addresses, base_data, jit_optimize, jit_interp, command_mode,
    control_dt, max_steps, goal_pos_tol, goal_theta_tol, vel_limit, log, verbose,
    params,
) -> Dict[str, Any]:
    """Hardware loop: a publisher thread streams the latest plan while the main
    thread keeps solving, so execution and planning overlap."""

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
        n = max(1, int(span / control_dt) + 1)
        ts = jnp.arange(n) * control_dt + float(tk[0])
        return np.asarray(jit_interp(ts, plan.tk, plan.mean[None, ...]))[0]

    # Shared latest plan, guarded by a lock. `samples` is the plan already
    # materialised on a control-tick grid; `t_perf` is the wall clock when it
    # was published, so the publisher can index into it by elapsed time.
    lock = threading.Lock()
    shared = {"samples": _sample_plan(params),
              "t_perf": time.perf_counter()}
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
            elif command_mode == "hold":
                u = s[0]
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
    reached = False
    try:
        for step in range(max_steps):
            t_loop = time.perf_counter()
            world = interface.read_state()
            t_read = time.perf_counter()
            mjx_data = _assemble_state(task, base_data, addresses, world)

            t0 = time.perf_counter()
            params, _ = jit_optimize(mjx_data, params)
            jax.block_until_ready(params)
            log["compute_time"].append(time.perf_counter() - t0)
            t_solve = time.perf_counter()

            # Hand the fresh plan to the publisher.
            samples = _sample_plan(params)
            print(f"[cmd] {np.round(samples[0], 3)}")
            with lock:
                shared["samples"] = samples
                # The plan's s[0] is the control for the state read at
                # `t_loop`, one solve ago -- the arm has been executing the
                # previous plan since. Anchor plan time to that read, not to
                # now, so the publisher enters the plan where the present
                # actually is instead of replaying a moment that has passed.
                shared["t_perf"] = t_loop

            # TEMP: the loop is much slower than `optimize` alone; find where.
            print(f"[t] read {t_read - t_loop:.3f}  assemble+solve "
                  f"{t_solve - t_read:.3f}  total {time.perf_counter() - t_loop:.3f}")

            # Log the command the publisher would send at the solve instant.
            first = samples[:1]
            reached = _log_and_check(log, task, mjx_data, params, first,
                                     goal_pos_tol, goal_theta_tol, step, verbose)
            if reached:
                break
    finally:
        stop.set()
        pub.join(timeout=1.0)
        interface.send_velocity(np.zeros(len(addresses.arm_dof_adr)))
    return _finalize_log(log, task, reached, show_plans=False)


def _log_and_check(
    log, task, mjx_data, params, applied, goal_pos_tol, goal_theta_tol, step, verbose,
) -> bool:
    """Append one step to the log and return whether the goal was reached."""
    block_pose = _log_step(log, task, mjx_data, params, applied)
    goal = np.asarray(task.goal)
    pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
    theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)
    if verbose and step % 10 == 0:
        print(f"step {step:4d}  pos_err={pos_err:.4f}  theta_err={theta_err:.4f}  "
              f"primal={log['primal_residual'][-1]:.3f}  "
              f"plan={log['compute_time'][-1] * 1e3:.0f}ms")
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
