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

import signal
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
# consumed. Allocated on BOTH paths -- see `_init_sample_stats` for why that
# changed.
_SAMPLE_STAT_KEYS = (
    "sample_cost_min",
    "sample_cost_mean",
    "sample_cost_max",
    "sample_cost_std",
    "sample_eta",
    "sample_temp_star",
    "sample_nonfinite",
)

# Contact statistics of the same population, ADMM only (they are read off
# `ADMMTrajectory.consensus_values`, which the flat path has no analogue of).
#
# These exist to split the one question weight tuning cannot answer from the
# outside. A stall looks identical either way -- the object sits still, the
# arm keeps commanding, `||A^r_plan|| = 0` -- but the cause is one of:
#
#   (a) NO sampled rollout makes contact.  The planner is not rejecting
#       contact, it never saw any. The levers are exploration: `noise`,
#       `num_samples`, `stuck_kick_scale`, horizon.
#   (b) Rollouts DO make contact and the softmax ranks them badly. Then some
#       cost term is charging for contact, and `sample_contact_cost_gap`
#       says how much.
#
# The two call for opposite fixes and were indistinguishable in every series
# logged before this, which is how three days of runs went into moving
# weights that (a) would have made no difference to.
_CONTACT_STAT_KEYS = (
    "sample_contact_frac",   # share of sampled rollouts touching the object
    "sample_contact_gap",    # mean cost of touching samples MINUS mean cost
                             # of the rest. < 0 = touching is cheaper, so the
                             # softmax should already prefer it
    "sample_contact_rank",   # best touching sample's rank in the population,
                             # normalized: 0 = it WAS the cheapest sample,
                             # 1 = the most expensive. NaN when none touch
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


def _init_cost_terms(log: Dict[str, Any]) -> None:
    """Allocate the per-step cost decomposition series, both algorithms."""
    log.update({k: [] for k in _COST_TERM_KEYS})


def _sampler_temperature(params: Any) -> float:
    """The temperature the softmax that just ran actually used.

    `ADMMParams` has none of its own -- the sampling happens in its ROBOT
    sub-optimizer, whose params it holds. Reading `params.temperature` there
    silently returned the 1.0 default and made `sample_eta` meaningless.
    """
    inner = getattr(params, "robot_params", None)
    if inner is not None and hasattr(inner, "temperature"):
        return float(inner.temperature)
    return float(getattr(params, "temperature", 1.0))


def _init_sample_stats(log: Dict[str, Any], admm: bool) -> None:
    """Allocate the sample-statistics series.

    Both paths now. The ADMM path was excluded on the belief that its
    `optimize` returns no per-sample costs -- it does: the second return is
    the ROBOT block's last `ADMMTrajectory`, whose `costs` is
    (num_samples, H+1) and whose `consensus_values` is (num_samples, H, dim),
    both straight off `RobotSubproblem.rollout_with_randomizations`. Nothing
    was missing but the allocation.

    Here rather than in `oim.runtime.logs.init_log` so the sim world's log
    layout is untouched: this is a real-driver diagnostic, and `init_log` is
    the contract that keeps a hardware log comparable to a simulation one
    entry-for-entry.
    """
    log.update({k: [] for k in _SAMPLE_STAT_KEYS})
    if admm:
        log.update({k: [] for k in _CONTACT_STAT_KEYS})


def _log_contact_stats(log: Dict[str, Any], rollouts: Any, total: Any,
                       scale: Any) -> None:
    """Split "no sample touched" from "touching samples were ranked badly".

    `consensus_values` is (num_samples, H, dim) -- each sampled rollout's own
    A^r at every horizon step. A sample counts as touching if its largest
    |A^r| over the horizon exceeds 1% of the consensus scale in any channel;
    below that it is the estimator's own floor, not contact.

    `total` is the same horizon-summed per-sample cost the softmax ranked, so
    the gap and the rank are computed on exactly the numbers that decided the
    update -- not on a re-scored proxy.
    """
    if "sample_contact_frac" not in log:
        return
    vals = getattr(rollouts, "consensus_values", None)
    nan = float("nan")
    if vals is None:
        for key in _CONTACT_STAT_KEYS:
            log[key].append(nan)
        return
    a = np.asarray(vals, dtype=float)
    if a.ndim != 3 or a.shape[0] != total.shape[0]:
        for key in _CONTACT_STAT_KEYS:
            log[key].append(nan)
        return
    s = np.abs(np.asarray(scale, dtype=float))
    s = np.where(s > 0, s, 1.0)
    touch = (np.abs(a) / s).max(axis=(1, 2)) > 0.01     # (num_samples,)
    finite = np.isfinite(total)
    touch &= finite
    log["sample_contact_frac"].append(float(touch.mean()))
    rest = finite & ~touch
    if touch.any() and rest.any():
        log["sample_contact_gap"].append(
            float(total[touch].mean() - total[rest].mean()))
    else:
        log["sample_contact_gap"].append(nan)
    if touch.any():
        order = np.argsort(total[finite])
        ranks = np.empty(order.size, dtype=float)
        ranks[order] = np.arange(order.size)
        best = ranks[touch[finite]].min()
        log["sample_contact_rank"].append(
            float(best / max(order.size - 1, 1)))
    else:
        log["sample_contact_rank"].append(nan)


def _log_sample_stats(log: Dict[str, Any], rollouts: Any, temperature: Any,
                      scale: Any = None) -> None:
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
    per-sample costs, so both loops stay algorithm-agnostic. The ADMM path
    DOES carry them (see `_init_sample_stats`); pass `scale` there and the
    contact statistics beside these are filled in too.
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
    if scale is not None:
        _log_contact_stats(log, rollouts, total, scale)


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


# Cost terms reported per step. Read from the TASK's own methods wherever one
# exists, so this cannot drift from what the planner optimises; `approach` and
# `align` have no method of their own (they are inline in `_ell_r`) and are the
# only two recomputed here.
_COST_TERM_KEYS = ("c_goal", "c_approach", "c_align", "c_tilt", "c_ztip",
                   "c_contactz", "c_fade")


def _cost_terms(task: Any, mjx_data: Any) -> Dict[str, float]:
    """Decompose this step's cost on the state the arm is ACTUALLY in.

    One evaluation on one state, not a rollout -- microseconds. It answers the
    only question weight tuning ever asks: which term is moving the arm right
    now. Without it, a tip that climbs to 110 mm and a tip that sits at 30 mm
    look the same in the log, and the weight that caused it is a guess.

    NOT the planner's objective. It is the running cost's shaping terms
    evaluated at the current state, so it does not include the horizon, the
    terminal term, or (under ADMM) the consensus penalty -- which is exactly
    why a large unexplained gap between this and the sampled cost is itself
    informative on the ADMM path.
    """
    out = {k: float("nan") for k in _COST_TERM_KEYS}
    try:
        pose = task._block_pose(mjx_data)
        pusher = task._pusher_pos(mjx_data)
        goal = jnp.asarray(task.goal)

        fade = float(task.shaping_fade(pose))
        ramp = float(task._q_ramp_mult(mjx_data))
        out["c_fade"] = fade
        out["c_goal"] = float(task._se2_cost(
            pose, task.q_pos * ramp,
            task.q_theta * task._theta_ramp(pose) * ramp))

        # Must track `PushT._ell_r`'s own branch on `approach_power`, or the
        # diagnostic silently reports the OTHER form's number. It did: with
        # approach_power = 1 and w_approach = 200 at d_tip = 78 mm the
        # optimizer sees 8.56 while this printed 0.97, a factor of 9 -- on
        # the one term being tuned at the time.
        d_ee = float(jnp.sum((pusher - pose[:2]) ** 2))
        if float(getattr(task, "approach_power", 2.0)) == 1.0:
            gap = max(d_ee ** 0.5 - task.r0, 0.0)
        else:
            gap = max(d_ee - task.r0 ** 2, 0.0)
        out["c_approach"] = fade * task.w_approach * gap

        to_object = pose[:2] - pusher
        to_ref = task._align_reference(pose, pusher, to_object, goal)
        cos_angle = float(
            jnp.sum(to_object * to_ref)
            / (jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6))
        out["c_align"] = fade * task.w_align * max(float(task.gamma0) - cos_angle, 0.0)

        # Faded, like `_ell_r` does it. Reporting it unfaded made tilt look
        # like a bigger competitor to approach than it is wherever fade < 1.
        out["c_tilt"] = fade * float(task.w_tilt * task._tilt(mjx_data))
        pos_err = float(jnp.linalg.norm(pose[:2] - goal[:2]))
        out["c_ztip"] = float(task._tip_height_cost(mjx_data, pos_err))
        out["c_contactz"] = float(task._contact_z_cost(mjx_data, pose))
    except Exception:  # noqa: BLE001 -- a diagnostic must never end a run
        pass
    return out


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

    Reads its two numbers off the controller. `MPPI` carries them; `ADMM`
    does not -- it holds an MPPI as its ROBOT sub-optimizer, and that is where
    the two live, so a plain `getattr(ctrl, ...)` read 0 and this class was
    silently inert on the whole ADMM path. Measured cost of that on the two
    2026-08-27 mock runs: 131 and 124 consecutive frozen control steps, 31% of
    each run, in exactly the state it exists to break -- while the setup dump
    printed `stuck_kick=100x2.0` as though it were armed.
    """

    # Matches the exact-zero signature real stiction produces in MJX/Warp
    # (object_velocity goes bit-exact 0.0, not a gradual decay) -- not a
    # tolerance chosen to catch merely "slow" progress.
    EPS = 2e-3

    def __init__(self, ctrl: Any) -> None:
        source = ctrl
        if not hasattr(source, "stuck_kick_steps"):
            # ADMM: the knobs belong to the robot block's own optimizer.
            source = getattr(
                getattr(ctrl, "robot_subproblem", None), "optimizer", ctrl
            )
        self.steps = int(getattr(source, "stuck_kick_steps", 0) or 0)
        self.scale = float(getattr(source, "stuck_kick_scale", 0.0) or 0.0)
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
        inner = getattr(params, "robot_params", None)
        self.count = 0
        self.kicks += 1
        if verbose:
            print(f"step {step:4d}  stuck -- kicked ({self.kicks})")
        if inner is not None and hasattr(inner, "mean"):
            # ADMM: `ADMMParams.mean` is a read-only PROPERTY forwarding to
            # `robot_params.mean`, so `params.replace(mean=...)` would raise
            # -- it is not a field. Kick the field it delegates to. The
            # consensus variable and both duals are deliberately left alone:
            # the robot block is the one that has stopped exploring, and
            # resetting z would also discard whatever the object block has
            # agreed to.
            kick = self.scale * jax.random.normal(kick_rng, inner.mean.shape)
            return params.replace(
                robot_params=inner.replace(mean=inner.mean + kick), rng=rng
            )
        kick = self.scale * jax.random.normal(kick_rng, params.mean.shape)
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
    if jit_plans is not None:
        _pl = jit_plans(_md, _p)
        jax.block_until_ready(_pl)
    # The eager per-step cost decomposition dispatches its small kernels on
    # its first call too -- cheap, but free to pay here rather than at step 0.
    _cost_terms(task, _md)
    if verbose:
        print(f"[jit] loop-path warm-up: {time.perf_counter() - t:.1f}s")

    if verbose:
        print(f"[jit] ready; {'overlapped' if real_time else 'serial'} loop, "
              f"control {control_rate:.0f} Hz, stream")

    log = init_log(task, mjx_data, mjx_data, show_plans=admm, admm=admm)
    _init_sample_stats(log, admm)
    _init_cost_terms(log)
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
        _log_sample_stats(log, rollouts, _sampler_temperature(params),
                          task.consensus_scale() if admm else None)

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
    _world0 = interface.read_state()
    _seed_params, _ = jit_optimize(
        _assemble_state(task, base_data, addresses, _world0), params
    )
    jax.block_until_ready(_seed_params)
    params = _seed_params

    # Shared latest plan, guarded by a lock. `samples` is the plan already
    # materialised on a control-tick grid; `t_perf` is the wall clock when it
    # was published, so the publisher can index into it by elapsed time.
    lock = threading.Lock()
    shared = {"samples": _sample_plan(params), "t_perf": t_seed}
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
    step = -1  # defined before the try, so the handlers below can name it even
    #            if the interrupt lands on the very first iteration
    interrupt = _InterruptFlag().install()
    try:
        for step in range(max_steps):
            if interrupt.requested:
                break
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
            _log_sample_stats(log, rollouts, _sampler_temperature(params),
                              task.consensus_scale() if admm else None)

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
        if verbose:
            print(f"\ninterrupted at step {step}; finalising the log")
    finally:
        interrupt.restore()
        stop.set()
        pub.join(timeout=1.0)
        interface.stop()
    if verbose:
        print(f"stopped at step {step}; {'goal reached' if reached else 'saving'}")
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _log_and_check(
    log, task, mjx_data, params, applied, goal_pos_tol, goal_theta_tol, step, verbose, admm=True,
) -> bool:
    """Append one step to the log and return whether the goal was reached."""
    block_pose = log_step(log, task, mjx_data, params, applied, admm=admm)
    for key, value in _cost_terms(task, mjx_data).items():
        log[key].append(value)
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
            pop = (f"eta={log['sample_eta'][-1]:.1f}  "
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
              f"{primal}{pop}{con}plan={log['compute_time'][-1] * 1e3:.0f}ms")
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
        c = {k: log[k][-1] for k in _COST_TERM_KEYS if log.get(k)}
        if c:
            print(f"           cost: goal={c.get('c_goal', float('nan')):8.1f}"
                  f"  approach={c.get('c_approach', float('nan')):7.2f}"
                  f"  align={c.get('c_align', float('nan')):6.2f}"
                  f"  tilt={c.get('c_tilt', float('nan')):6.2f}"
                  f"  ztip={c.get('c_ztip', float('nan')):8.2f}"
                  f"  contactz={c.get('c_contactz', float('nan')):8.2f}"
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
