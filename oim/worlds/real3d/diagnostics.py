"""Per-step diagnostics of a hardware run: sample statistics and costs.

None of this steers the robot. It is what makes a finished run readable --
the sample population the softmax consumed (which the optimizer discards),
the consensus contact statistics, and the executed state's cost split by
term. `_StatsReducer` is the one that matters for speed: it reduces the
whole population ON the device, so the arrays never cross the bus.
"""

from __future__ import annotations

from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np

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
# The object block's own population (ADMM only): `ADMMParams.object_costs`
# is the horizon-summed cost of each object sample the last ADMM round
# ranked, `object_samples` their trajectories. eta at the OBJECT optimizer's
# temperature, and the share of samples that move the block at all -- the
# quasi-static plant returns zero motion below the friction limit, so a
# population whose mean has decayed can be 100% "hold still" (151025 steps
# 304-456: a_obj 0.04-0.2, plan displacement 0.1 mm, for 150 steps).
_OBJECT_STAT_KEYS = (
    "object_eta",
    "object_cost_min",
    "object_cost_std",
    "object_moving_frac",
)

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
        log.update({k: [] for k in _OBJECT_STAT_KEYS})

class _StatsReducer:
    """Sample / contact / object-block statistics reduced ON the device.

    Same numbers `_log_sample_stats`, `_log_contact_stats` and
    `_log_object_stats` computed, but the (num_samples, H+1) cost array,
    the (num_samples, H, dim) consensus array and the object block's
    (num_samples, H, 3) population never leave the GPU: one jitted kernel
    reduces them to a dozen scalars, and one `device_get` copies those.
    Per step this is ~1 ms where the host-side versions cost tens of ms
    of transfer alone, on the same thread the next solve waits on.
    """

    def __init__(self, admm: bool, scale: Any) -> None:
        self.admm = admm
        if scale is None:
            self._scale = None
        else:
            sc = np.abs(np.asarray(scale, dtype=float))
            self._scale = jnp.asarray(np.where(sc > 0, sc, 1.0))
        self._fn = jax.jit(self._reduce)

    @staticmethod
    def _temperature_for_eta(d: jax.Array, finite: jax.Array, n_good: Any,
                             frac: float) -> jax.Array:
        """`_temperature_for_eta`, as a fixed-length bisection in log T."""
        target = jnp.clip(frac * n_good, 1.0 + 1e-9, n_good - 1e-9)

        def eta_at(log_t: jax.Array) -> jax.Array:
            return jnp.sum(jnp.where(finite, jnp.exp(-d / jnp.exp(log_t)), 0.0))

        def body(_i: int, lh: Any) -> Any:
            lo, hi = lh
            mid = 0.5 * (lo + hi)
            below = eta_at(mid) < target
            return jnp.where(below, mid, lo), jnp.where(below, hi, mid)

        # 40 halvings of a 55-nat bracket leave 5e-11, and float32 stops
        # resolving `mid` well before that: bit-identical to 100 over 200
        # random populations, at 0.60 ms instead of 1.04 -- and the
        # bisection IS this reducer's whole kernel cost.
        lo, hi = jax.lax.fori_loop(
            0, 40, body, (jnp.log(1e-9), jnp.log(1e15))
        )
        t_star = jnp.exp(0.5 * (lo + hi))
        usable = (n_good >= 2) & (jnp.max(jnp.where(finite, d, 0.0)) > 0.0)
        return jnp.where(usable, t_star, jnp.nan)

    def _reduce(self, costs: jax.Array, temperature: jax.Array,
                consensus_values: Any, object_costs: Any,
                object_samples: Any, obj_temperature: Any,
                obj_pose: Any) -> Dict[str, jax.Array]:
        nan = jnp.nan
        total = jnp.sum(costs, axis=1)                  # (S,)
        finite = jnp.isfinite(total)
        n_good = jnp.sum(finite)
        n1 = jnp.maximum(n_good, 1)
        t_min = jnp.min(jnp.where(finite, total, jnp.inf))
        t_max = jnp.max(jnp.where(finite, total, -jnp.inf))
        mean = jnp.sum(jnp.where(finite, total, 0.0)) / n1
        var = jnp.sum(jnp.where(finite, (total - mean) ** 2, 0.0)) / n1
        temp = jnp.maximum(temperature, 1e-9)
        d = jnp.where(finite, total - t_min, 0.0)
        eta = jnp.sum(jnp.where(finite, jnp.exp(-d / temp), 0.0))
        empty = n_good == 0
        out = {
            "sample_nonfinite": (total.shape[0] - n_good).astype(jnp.float32),
            "sample_cost_min": jnp.where(empty, nan, t_min),
            "sample_cost_mean": jnp.where(empty, nan, mean),
            "sample_cost_max": jnp.where(empty, nan, t_max),
            "sample_cost_std": jnp.where(empty, nan, jnp.sqrt(var)),
            "sample_eta": jnp.where(empty, nan, eta),
            "sample_temp_star": self._temperature_for_eta(
                d, finite, n_good, _ETA_TARGET_FRAC
            ),
        }
        if consensus_values is not None:
            a = jnp.abs(consensus_values) / self._scale       # (S, H, dim)
            touch = (jnp.max(a, axis=(1, 2)) > 0.01) & finite
            rest = finite & ~touch
            n_t = jnp.sum(touch)
            n_r = jnp.sum(rest)
            gap = (
                jnp.sum(jnp.where(touch, total, 0.0)) / jnp.maximum(n_t, 1)
                - jnp.sum(jnp.where(rest, total, 0.0)) / jnp.maximum(n_r, 1)
            )
            # Rank among the finite samples: non-finite ones sort last and
            # are excluded from `touch`, so they never win.
            order = jnp.argsort(jnp.where(finite, total, jnp.inf))
            ranks = jnp.argsort(order).astype(jnp.float32)
            best = jnp.min(jnp.where(touch, ranks, jnp.inf))
            out["sample_contact_frac"] = jnp.mean(touch.astype(jnp.float32))
            out["sample_contact_gap"] = jnp.where(
                (n_t > 0) & (n_r > 0), gap, nan
            )
            out["sample_contact_rank"] = jnp.where(
                n_t > 0, best / jnp.maximum(n_good - 1, 1), nan
            )
        if object_costs is not None:
            c = object_costs
            fin = jnp.isfinite(c)
            n = jnp.sum(fin)
            n1o = jnp.maximum(n, 1)
            c_min = jnp.min(jnp.where(fin, c, jnp.inf))
            c_mean = jnp.sum(jnp.where(fin, c, 0.0)) / n1o
            c_var = jnp.sum(jnp.where(fin, (c - c_mean) ** 2, 0.0)) / n1o
            o_temp = jnp.maximum(obj_temperature, 1e-9)
            o_eta = jnp.sum(jnp.where(fin, jnp.exp(-(c - c_min) / o_temp), 0.0))
            none = n == 0
            out["object_eta"] = jnp.where(none, nan, o_eta)
            out["object_cost_min"] = jnp.where(none, nan, c_min)
            out["object_cost_std"] = jnp.where(none, nan, jnp.sqrt(c_var))
            disp = jnp.linalg.norm(
                object_samples[:, -1, :2] - obj_pose[:2], axis=1
            )
            out["object_moving_frac"] = jnp.mean((disp > 0.002).astype(jnp.float32))
        return out

    def __call__(self, log: Dict[str, Any], rollouts: Any, params: Any,
                 obj_pose: Any) -> None:
        """Append this step's statistics to `log` (NaN where unavailable)."""
        if "sample_eta" not in log:
            return
        costs = getattr(rollouts, "costs", None)
        if costs is None or costs.ndim != 2:
            return
        cv = getattr(rollouts, "consensus_values", None) if self.admm else None
        if cv is not None and (cv.ndim != 3 or cv.shape[0] != costs.shape[0]):
            cv = None
        oc = getattr(params, "object_costs", None) if self.admm else None
        osm = getattr(params, "object_samples", None) if self.admm else None
        if oc is None or osm is None or osm.ndim != 3 or osm.shape[-1] < 2:
            oc = osm = None
        inner = getattr(params, "object_params", None)
        o_temp = float(getattr(inner, "temperature", 1.0))
        stats = jax.device_get(self._fn(
            costs, _sampler_temperature(params), cv, oc, osm, o_temp,
            jnp.asarray(obj_pose),
        ))
        nan = float("nan")
        for key in _SAMPLE_STAT_KEYS:
            log[key].append(float(stats.get(key, nan)))
        if self.admm:
            for key in (*_CONTACT_STAT_KEYS, *_OBJECT_STAT_KEYS):
                log[key].append(float(stats.get(key, nan)))

def _log_object_stats(log: Dict[str, Any], params: Any, obj_pose: Any) -> None:
    """Append the object block's population statistics for this step."""
    if "object_eta" not in log:
        return
    nan = float("nan")
    costs = getattr(params, "object_costs", None)
    samples = getattr(params, "object_samples", None)
    inner = getattr(params, "object_params", None)
    if costs is None or samples is None:
        for key in _OBJECT_STAT_KEYS:
            log[key].append(nan)
        return
    c = np.asarray(costs, dtype=float)
    c = c[np.isfinite(c)]
    temp = max(float(getattr(inner, "temperature", 1.0)), 1e-9)
    if c.size == 0:
        log["object_eta"].append(nan)
        log["object_cost_min"].append(nan)
        log["object_cost_std"].append(nan)
    else:
        log["object_eta"].append(float(np.exp(-(c - c.min()) / temp).sum()))
        log["object_cost_min"].append(float(c.min()))
        log["object_cost_std"].append(float(c.std()))
    s = np.asarray(samples, dtype=float)
    if s.ndim == 3 and s.shape[-1] >= 2:
        disp = np.linalg.norm(s[:, -1, :2] - np.asarray(obj_pose)[:2], axis=1)
        log["object_moving_frac"].append(float(np.mean(disp > 0.002)))
    else:
        log["object_moving_frac"].append(nan)

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

# Cost terms reported per step. Read from the TASK's own methods wherever one
# exists, so this cannot drift from what the planner optimises; `approach` and
# `align` have no method of their own (they are inline in `_ell_r`) and are the
# only two recomputed here.
_COST_TERM_KEYS = ("c_goal", "c_approach", "c_align", "c_tilt", "c_ztip",
                   "c_contactz", "c_fade")

def _cost_terms_jnp(task: Any, mjx_data: Any) -> jax.Array:
    """`_cost_terms` as one traceable expression, `(len(_COST_TERM_KEYS),)`.

    Reads the TASK's own methods (`shaping_fade`, `_q_ramp_mult`,
    `_se2_cost`, `_tilt`, `_tip_height_cost`, `_contact_z_cost`) so it
    cannot drift from what the planner optimises; `approach` and `align`
    are inline in `_ell_r` and are the only two mirrored here.
    """
    pose = task._block_pose(mjx_data)
    pusher = task._pusher_pos(mjx_data)
    goal = jnp.asarray(task.goal)
    fade = task.shaping_fade(pose)
    ramp = task._q_ramp_mult(mjx_data)
    c_goal = task._se2_cost(
        pose, task.q_pos * ramp, task.q_theta * task._theta_ramp(pose) * ramp
    )
    # Must mirror `PushT._ell_r`'s approach term exactly, or the
    # diagnostic silently reports a different number than the cost the
    # planner minimised: purely xy, distance to the WALL (the xy SDF's
    # outside component) past the `r0` stand-off.
    from oim.objects.sdf import rotate  # noqa: PLC0415
    local = rotate(-pose[2], pusher - pose[:2])
    sd = jnp.maximum(task.object_model.footprint.sdf(local), 0.0)
    gap = jnp.clip(sd - task.r0, 0.0, None)
    c_approach = fade * task.w_approach * gap**2
    # Align against the GLOBAL goal (the same convention `shaping_fade`
    # uses), through the task's own reference so `align_theta_gain` is
    # honoured; under `align_ref: plan_end` the planner measures this
    # against the object block's plan endpoint instead.
    to_object = pose[:2] - pusher
    to_ref = task._align_reference(pose, pusher, to_object, goal)
    cos_angle = jnp.sum(to_object * to_ref) / (
        jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
    )
    c_align = fade * task.w_align * jnp.clip(task.gamma0 - cos_angle, 0.0, None)
    c_tilt = fade * task.w_tilt * task._tilt(mjx_data)
    pos_err = jnp.linalg.norm(pose[:2] - goal[:2])
    c_ztip = task._tip_height_cost(mjx_data, pos_err)
    if float(getattr(task, "w_contact_z_exp", 0.0)) and hasattr(
        task, "_contact_z_cost"
    ):
        c_contactz = task._contact_z_cost(mjx_data, pose)
    else:
        c_contactz = jnp.nan
    return jnp.stack([
        jnp.asarray(c_goal, dtype=jnp.float32),
        jnp.asarray(c_approach, dtype=jnp.float32),
        jnp.asarray(c_align, dtype=jnp.float32),
        jnp.asarray(c_tilt, dtype=jnp.float32),
        jnp.asarray(c_ztip, dtype=jnp.float32),
        jnp.asarray(c_contactz, dtype=jnp.float32),
        jnp.asarray(fade, dtype=jnp.float32),
    ])

def _cost_terms(task: Any, mjx_data: Any, fn: Any = None) -> Dict[str, float]:
    """Decompose this step's cost on the state the arm is ACTUALLY in.

    One evaluation on one state, not a rollout. It answers the only
    question weight tuning ever asks: which term is moving the arm right
    now. NOT the planner's objective: the running cost's shaping terms at
    the current state, without the horizon, the terminal term or (under
    ADMM) the consensus penalty.

    For the console print. `fn` is `_cost_terms_jnp` compiled for this
    task (one dispatch); without it the expression runs eagerly, op by op,
    which is fine for tests and costs tens of ms per call in a loop. The
    per-step series in the run file is produced after the loop by
    `_reconstruct_cost_terms`, from the logged states.
    """
    out = {k: float("nan") for k in _COST_TERM_KEYS}
    try:
        raw = _cost_terms_jnp(task, mjx_data) if fn is None else fn(mjx_data)
        vals = np.asarray(raw, dtype=float)
        out.update(zip(_COST_TERM_KEYS, (float(v) for v in vals)))
    except Exception:  # noqa: BLE001 -- a diagnostic must never end a run
        pass
    return out
