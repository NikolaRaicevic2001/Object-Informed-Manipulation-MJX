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
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from oim.objects import wrap_angle
from oim.runtime.logs import finalize_log, init_log, log_step
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
    "sample_temp_star",
    "sample_nonfinite",
)

# What fraction of the sample population should carry meaningful softmax
# weight. 1/N is a degenerate argmin; 1 is a uniform average that carries no
# information at all. Anywhere in the middle works; 0.4 is the middle.
_ETA_TARGET_FRAC = 0.4


def _temperature_for_eta(costs: Any, frac: float) -> float:
    """The `temperature` that would put eta at `frac` of this population.

    `MPPI.update_params` divides RAW, unnormalised horizon-summed costs by
    `temperature` -- there is no scaling anywhere in that path -- so the right
    value is in cost units and moves with whatever the cost scale happens to
    be that step. Nothing derives it a priori; it has to be measured, and this
    measures it on the same numbers the softmax just consumed.

    eta(T) = sum_i exp(-(c_i - c_min) / T) is continuous and strictly
    increasing in T, from 1 as T -> 0 to N as T -> inf, so bisection in log T
    finds the crossing. Reported only -- never applied. Microseconds on an
    array already copied to the host for the statistics beside it.
    """
    d = np.asarray(costs, dtype=float)
    d = d - d.min()
    n = d.size
    if n < 2 or d.max() <= 0.0:
        return float("nan")
    target = min(max(frac * n, 1.0 + 1e-9), n - 1e-9)
    lo, hi = 1e-9, 1.0
    while float(np.exp(-d / hi).sum()) < target and hi < 1e15:
        hi *= 10.0
    for _ in range(60):
        mid = float(np.sqrt(lo * hi))
        if float(np.exp(-d / mid).sum()) < target:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def _init_sample_stats(log: Dict[str, Any], admm: bool) -> None:
    """Allocate the sample-statistics series, flat MPPI only.

    Here rather than in `oim.runtime.logs.init_log` so the sim world's log
    layout is untouched: this is a real-driver diagnostic, and `init_log` is
    the contract that keeps a hardware log comparable to a simulation one
    entry-for-entry.
    """
    if not admm:
        log.update({k: [] for k in _SAMPLE_STAT_KEYS})


def _log_sample_stats(log: Dict[str, Any], rollouts: Any, temperature: Any) -> None:
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
    log["sample_temp_star"].append(
        _temperature_for_eta(good, _ETA_TARGET_FRAC)
    )


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
    EPS = 2e-3

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
              f"control {control_rate:.0f} Hz, stream")

    log = init_log(task, mjx_data, mjx_data, show_plans=admm, admm=admm)
    _init_sample_stats(log, admm)
    common = dict(
        task=task, interface=interface, addresses=addresses, base_data=base_data,
        jit_optimize=jit_optimize, jit_interp=jit_interp, jit_plans=jit_plans,
        control_dt=control_dt, max_steps=max_steps, goal_pos_tol=goal_pos_tol,
        goal_theta_tol=goal_theta_tol, vel_limit=vel_limit, admm=admm, log=log,
        verbose=verbose, kicker=_StuckKicker(ctrl),
    )
    if real_time:
        return _run_overlapped(params=params, **common)
    return _run_serial(params=params, replan_rate=replan_rate, **common)


def _run_serial(
    task, interface, addresses, base_data, jit_optimize, jit_interp, jit_plans,
    control_dt, replan_rate, max_steps, goal_pos_tol, goal_theta_tol, vel_limit,
    admm, log, verbose, params, kicker,
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
        if admm:
            obj_plan, rob_plan, _ = jit_plans(mjx_data, params)
            log["object_plan"].append(np.asarray(obj_plan))
            log["robot_plan"].append(np.asarray(rob_plan))
        reached = _log_and_check(log, task, mjx_data, params, applied,
                                 goal_pos_tol, goal_theta_tol, step, verbose, admm)
        if reached:
            break
        # Same placement the sim's flat loop uses: after the success check,
        # reading the errors that check just used.
        params = kicker.maybe_kick(params, log["pos_err"][-1],
                                   log["theta_err"][-1], step, verbose)

    interface.stop()
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _run_overlapped(
    task, interface, addresses, base_data, jit_optimize, jit_interp, jit_plans,
    control_dt, max_steps, goal_pos_tol, goal_theta_tol, vel_limit, admm, log,
    verbose, params, kicker,
) -> Dict[str, Any]:
    """Hardware loop: a publisher thread streams the latest plan while the main
    thread keeps solving, so execution and planning overlap.
    """

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
    step = -1  # defined before the try, so the KeyboardInterrupt handler below
    #            can name it even if the interrupt lands on the first iteration
    try:
        for step in range(max_steps):
            t_loop = time.perf_counter()
            world = interface.read_state()
            mjx_data = _assemble_state(task, base_data, addresses, world)

            t0 = time.perf_counter()
            params, rollouts = jit_optimize(mjx_data, params)
            jax.block_until_ready(params)
            log["compute_time"].append(time.perf_counter() - t0)

            # Hand the fresh plan to the publisher.
            samples = _sample_plan(params)
            with lock:
                shared["samples"] = samples
                # The plan's s[0] is the control for the state read at
                # `t_loop`, one solve ago -- the arm has been executing the
                # previous plan since. Anchor plan time to that read, not to
                # now, so the publisher enters the plan where the present
                # actually is instead of replaying a moment that has passed.
                shared["t_perf"] = t_loop

            # Deliberately after the hand-off above: this forces a device-to-
            # host copy of the (num_samples, H+1) cost array, and the
            # publisher must not wait on a diagnostic.
            _log_sample_stats(log, rollouts, getattr(params, "temperature", 1.0))

            # Log the command the publisher would send at the solve instant.
            first = samples[:1]
            if admm:
                obj_plan, rob_plan, _ = jit_plans(mjx_data, params)
                log["object_plan"].append(np.asarray(obj_plan))
                log["robot_plan"].append(np.asarray(rob_plan))
            reached = _log_and_check(log, task, mjx_data, params, first,
                                     goal_pos_tol, goal_theta_tol, step, verbose, admm)
            if reached:
                break
            # The kick only rewrites the sampling mean the NEXT solve starts
            # from; the publisher keeps streaming the plan already handed to
            # it, so nothing the arm is executing changes discontinuously.
            params = kicker.maybe_kick(params, log["pos_err"][-1],
                                       log["theta_err"][-1], step, verbose)
    except KeyboardInterrupt:
        # Ctrl-C is how a hardware run normally ENDS -- nobody waits out
        # `--steps 1500` once the answer is visible. Letting the exception
        # leave this function skipped `finalize_log` and every `save_run`
        # below it, so the runs worth looking at were exactly the ones with no
        # run file. Swallowed here, at the loop, rather than in `main`: the log
        # lives in this frame, and the `finally` below still stops the arm.
        if verbose:
            print(f"\ninterrupted at step {step}; finalising the log")
    finally:
        stop.set()
        pub.join(timeout=1.0)
        interface.stop()
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _log_and_check(
    log, task, mjx_data, params, applied, goal_pos_tol, goal_theta_tol, step, verbose, admm=True,
) -> bool:
    """Append one step to the log and return whether the goal was reached."""
    block_pose = log_step(log, task, mjx_data, params, applied, admm=admm)
    goal = np.asarray(task.goal)
    pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
    theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
    log["pos_err"].append(pos_err)
    log["theta_err"].append(theta_err)
    if verbose and step % 10 == 0:
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
            primal = (f"primal={log['primal_residual'][-1]:.3f} "
                      f"dual={log['dual_residual'][-1]:.3f}  "
                      f"|y_o|={y_o:.2f} |y_r|={y_r:.2f} "
                      f"rho={log['rho'][-1]:.2f}  ")
        # eta on the console, not only in the run file: a flat run that has
        # gone uninformative (eta at num_samples, or any nonfinite sample)
        # otherwise looks exactly like one that is working, and there is no
        # point letting 1000 steps finish before finding that out.
        pop = ""
        if log.get("sample_eta"):
            bad = log["sample_nonfinite"][-1]
            pop = (f"eta={log['sample_eta'][-1]:.1f}  "
                   f"T*={log['sample_temp_star'][-1]:.0f}  "
                   f"cost={log['sample_cost_min'][-1]:.2f}"
                   f"+-{log['sample_cost_std'][-1]:.2f}  "
                   + (f"NONFINITE={bad}  " if bad else ""))
        print(f"step {step:4d}  pos_err={pos_err:.4f}  theta_err={theta_err:.4f}  "
              f"{primal}{pop}plan={log['compute_time'][-1] * 1e3:.0f}ms")
        # `block_pose` is the SE(2) read back out of the ASSEMBLED MJX state,
        # i.e. what the cost function is actually optimising against -- not the
        # TF reading. If this disagrees with tf2_echo, the bug is in
        # _lookup_object_se2 or _assemble_state, not in the planner.
        tip = np.asarray(log["robot_pos"][-1])          # tip (x, y), world frame
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
