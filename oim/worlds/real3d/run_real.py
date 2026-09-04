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

import contextlib
import math
import signal
import threading
import time
from collections import deque
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union

import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
import numpy as np
from mujoco import mjx
from scipy.spatial.transform import Rotation

from oim.objects import Box, wrap_angle
from oim.runtime.logs import finalize_log, init_log, local_goal_marker, log_step
from oim.runtime.mjcf import mocap_id
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


def _free_camera_distance(
    model: mujoco.MjModel, aspect: float, elevation: float, lookat_z: float,
    half_span: float, half_depth: float,
) -> float:
    """How far back to stand so `half_span` just fills the frame width.

    At distance `d` the frustum is `d * tan(fovy/2) * aspect` wide, and the
    table's near edge sits `half_depth * cos(elevation)` closer than the aim
    point while raising the aim by `lookat_z` pulls it
    `lookat_z * sin(elevation)` nearer still -- so both shift the distance
    the width has to be solved at, not just the framing.
    """
    tan_h = math.tan(math.radians(model.vis.global_.fovy / 2.0)) * aspect
    tilt = math.radians(abs(elevation))
    return (
        half_span / tan_h
        - math.sin(tilt) * lookat_z
        + math.cos(tilt) * half_depth
    ) * _VIEW_NEAR_CORNER


def _frame_table(
    model: mujoco.MjModel, cam: Any, aspect: float,
    azimuth: float, elevation: float, distance: Optional[float],
) -> None:
    """Aim a free camera at the table, filling the width with its long axis.

    Falls back to `mjv_defaultFreeCamera` for any model without a `table`
    geom, so this stays safe for scenes it was not measured on.
    """
    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    if gid < 0:
        mujoco.mjv_defaultFreeCamera(model, cam)
        return
    pos = model.geom_pos[gid]      # the table is a worldbody geom, so this
    size = model.geom_size[gid]    # is already in world coordinates
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.azimuth = azimuth
    cam.elevation = elevation
    cam.lookat[:] = [float(pos[0]), float(pos[1]), _VIEW_LOOKAT_Z]
    cam.distance = (
        distance if distance is not None else
        _free_camera_distance(
            model, aspect, elevation, _VIEW_LOOKAT_Z,
            half_span=float(size[1]), half_depth=float(size[0]),
        )
    )


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
# --live's refresh rate on `_run_overlapped`'s display thread, between
# solves. Not tied to control_rate: this is how often a human can usefully
# perceive an update, not a control-loop constraint like control_rate is.
_DISPLAY_HZ = 30.0

# --live's default framing, used when no --camera picks a model camera.
# `mjv_defaultFreeCamera` frames the whole MODEL, which on the real scenes
# means the 0.91 m of table leg and a wide margin of floor -- the table top,
# the only part anything happens on, ends up a small patch in the middle.
#
# Azimuth 180 stands the camera off the +x end of the table looking back
# along -x, which puts screen-right on +y: the table's long 1.523 m axis
# lies across the width, and the arm base (world origin) is at the far
# edge. That is the same standpoint the scenes' own "front" camera uses.
_VIEW_AZIMUTH = 180.0
_VIEW_ELEVATION = -27.0
# Raises the aim point off the table top so the frame holds the arm as well
# as the tabletop. Measured: at 0.10 the union of the table and every
# non-floor geom centres on the horizon (NDC 0.00) and spans y[-0.65,+0.65],
# so nothing is clipped and neither half of the frame is left empty. Aiming
# at the tabletop itself (0.0) rides the content high; past ~0.2 the table
# slides into the bottom third.
_VIEW_LOOKAT_Z = 0.10
# `_free_camera_distance` places the table's FAR corners on the frame edge;
# the NEAR corners, being closer, project 4.5% wider. Measured against
# mjv_updateScene and constant -- it does not move with aspect, elevation
# or lookat height.
_VIEW_NEAR_CORNER = 1.045
# Only used when the viewer has not sized its window yet (`Handle.viewport`
# reads back 0). Any real window replaces this on the first frame.
_VIEW_FALLBACK_ASPECT = 1.5

# Stand-in for `vis_lock` on the paths that have no second thread touching
# `mj_data_cpu` (the serial loop, and any run with neither --record nor
# --live). `contextlib.nullcontext` semantics, spelled out so
# `_visualize_step` needs no branch of its own.
_NULL_LOCK = contextlib.nullcontext()


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
        if bool(getattr(task, "approach_sdf", False)):
            # Mirror `PushT._ell_r`'s SDF branch, or this diagnostic
            # reports the origin-distance number for the one term whose
            # FORM is being changed.
            from oim.objects.sdf import rotate  # noqa: PLC0415
            _local = rotate(-pose[2], pusher - pose[:2])
            _sd_raw = float(task.object_model.footprint.sdf(_local))
            _sd = max(_sd_raw, 0.0)
            gap = max(_sd - task.r0, 0.0)
            if bool(getattr(task, "approach_z", False)) and _sd_raw > 0.0:
                _dz = (float(mjx_data.site_xpos[task.trace_site_ids[0], 2])
                       - task.tip_quadratic_target_z)
                gap = (gap ** 2 + _dz ** 2) ** 0.5
            if float(getattr(task, "approach_power", 2.0)) != 1.0:
                gap = gap ** 2
        elif float(getattr(task, "approach_power", 2.0)) == 1.0:
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
    vis_lock: Any = None,
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
        show_samples, show_optimal: As on `run_real`.
        obj_plan, rob_plan: The two blocks' predicted object trajectories
            (ADMM only), from `ADMM.nominal_plans`.
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
    # Held across the write AND the mj_forward: the display thread's
    # viewer.sync() copies this same MjData, and a copy that lands
    # mid-mj_forward is the "stack is in use" abort.
    with vis_lock if vis_lock is not None else _NULL_LOCK:
        mj_data_cpu.qpos[:] = np.asarray(mjx_data.qpos)
        mj_data_cpu.qvel[:] = np.asarray(mjx_data.qvel)
        mj_data_cpu.mocap_pos[:] = np.asarray(mjx_data.mocap_pos)
        mj_data_cpu.mocap_quat[:] = np.asarray(mjx_data.mocap_quat)
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
    with vis_lock if vis_lock is not None else _NULL_LOCK:
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
    # tolerance chosen to catch merely "slow" progress. On HARDWARE that
    # signature never occurs: FoundationPose jitter moves pos_err/theta_err
    # by more than EPS on most steps, so the consecutive-step streak reset
    # every time and the kicker was silently inert -- the 2026-08-29 15:57
    # success run dwelled for 185 s of its 263 s (70%), including 5-8 s
    # frozen episodes, and fired exactly ONE kick. The WINDOW test below
    # replaces the streak for that reason: it asks whether NET progress
    # over the last `stuck_kick_steps` solves is under the noise scale,
    # which jitter cannot fool in either direction. A genuinely slow push
    # (2 mm/s over the ~3.5 s window = 7 mm) still clears it.
    EPS = 2e-3
    WINDOW_POS = 5e-3      # net |d pos_err| under 5 mm over the window ...
    WINDOW_THETA = 3.5e-2  # ... AND net |d theta_err| under ~2 deg = stuck
    # ... AND the tip itself went nowhere. The block not moving is NOT
    # stuck while the ARM is travelling: without this gate the first
    # hardware run fired every ~15 solves DURING THE INITIAL APPROACH
    # (block untouched, tip covering centimetres per window) and knocked
    # the arm off its own approach each time (2026-08-29, kicks at steps
    # 14/29/46/63/78). 30 mm net over the ~3.5 s window is above hover
    # wiggle's net drift but far below any real approach or repositioning
    # leg, so only a genuinely parked arm still counts as stuck.
    WINDOW_TIP = 3e-2

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
        # Rolling window of (pos_err, theta_err), one entry per solve.
        self._hist: deque = deque(maxlen=max(self.steps, 1))

    def maybe_kick(self, params: Any, pos_err: float, theta_err: float,
                   step: int, verbose: bool,
                   tip_xy: Any = None) -> Any:
        """Return `params`, perturbed if the run has been frozen long enough."""
        if self.steps <= 0 or not hasattr(params, "mean"):
            return params
        if tip_xy is not None:
            tx, ty = float(tip_xy[0]), float(tip_xy[1])
        else:  # caller without tip logging: gate passes, old behaviour
            tx = ty = float("nan")
        self._hist.append((pos_err, theta_err, tx, ty))
        if len(self._hist) < self._hist.maxlen:
            return params
        p0, t0, x0, y0 = self._hist[0]
        block_stuck = (
            abs(pos_err - p0) < self.WINDOW_POS
            and abs(theta_err - t0) < self.WINDOW_THETA
        )
        tip_moved = (
            not np.isnan(x0)
            and float(np.hypot(tx - x0, ty - y0)) >= self.WINDOW_TIP
        )
        if (not block_stuck) or tip_moved:
            return params
        # Fire, then start a fresh window so the kick gets `steps` solves
        # to show progress before it can fire again.
        self._hist.clear()
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


_OBSTACLE_NAMES = ("obs_1", "obs_2", "obs_3")


def _sample_obstacle_tf_live(
    interface: Any,
    names: Tuple[str, ...] = _OBSTACLE_NAMES,
    base_frame: str = "xarm_device",
    window_s: float = 1.5,
) -> Dict[str, Tuple[float, float, float]]:
    """The same xarm_device -> obs_N_center averaging
    Fork_FoundationPose/calibrate_obstacles.py does, but read directly off
    `interface`'s own already-running TF listener instead of a separate
    script + JSON file handoff.

    `Ros2Interface.__init__` already builds a `tf2_ros.Buffer` +
    `TransformListener` and spins them on a background thread before
    `run_real` ever reaches this call (see interface.py) -- the same TF
    tree the object pose is already read from. As long as
    `aruco_obstacle_node.py` + `aruco_tf_broadcaster.py` are running on the
    perception laptop and on the same ROS 2 domain (`setup_dds_env.sh` on
    both machines), `obs_N_center` is already arriving in that buffer for
    free; this just samples it. No new subscriptions, no rclpy.init(), no
    file ever touches disk.
    """
    import rclpy  # noqa: PLC0415
    from tf2_ros import (  # noqa: PLC0415
        ConnectivityException,
        ExtrapolationException,
        LookupException,
    )

    buffer = interface._tf_buffer  # noqa: SLF001 -- same listener object_pose uses
    result: Dict[str, Tuple[float, float, float]] = {}
    for name in names:
        target = f"{name}_center"
        samples = []
        t_end = time.monotonic() + window_s
        while time.monotonic() < t_end:
            try:
                tf = buffer.lookup_transform(base_frame, target, rclpy.time.Time())
                t = tf.transform.translation
                q = tf.transform.rotation
                yaw = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler("xyz")[2]
                samples.append((t.x, t.y, yaw))
            except (LookupException, ConnectivityException, ExtrapolationException):
                pass
            time.sleep(0.02)
        if not samples:
            continue
        arr = np.asarray(samples)
        mean_xy = arr[:, :2].mean(axis=0)
        # Circular mean -- a plain average breaks across the +-pi wrap.
        mean_yaw = np.arctan2(np.sin(arr[:, 2]).mean(), np.cos(arr[:, 2]).mean())
        result[name] = (float(mean_xy[0]), float(mean_xy[1]), float(mean_yaw))
    return result


def _load_obstacle_calibration(
    interface: Any = None, verbose: bool = True,
) -> Dict[str, Tuple[float, float, float]]:
    """Sample the obstacles' current pose directly off `interface`'s own
    TF connection (see `_sample_obstacle_tf_live`). Requires a real
    `Ros2Interface` (the mock has no TF tree to sample). JSON-file
    calibration support was removed -- live ArUco sampling is the only
    source now, `--obstacle-calibration` is a plain on/off flag.
    """
    if interface is None or not hasattr(interface, "_tf_buffer"):
        raise ValueError(
            "--obstacle-calibration live requires a real Ros2Interface "
            "(the mock has no TF tree to sample)"
        )
    if verbose:
        print("[calibration] sampling obs_1/2/3 live over TF "
              "(xarm_device -> obs_N_center)...")
    calibration = _sample_obstacle_tf_live(interface)
    missing = set(_OBSTACLE_NAMES) - calibration.keys()
    if missing and verbose:
        print(f"[calibration] no live TF for {sorted(missing)} -- is "
              f"aruco_obstacle_node.py running on the perception laptop, "
              f"and are those tags in view? Continuing with the "
              f"{len(calibration)}/3 obstacles that did resolve.")
    return calibration


def apply_obstacle_calibration(
    task: PushT,
    base_data: mjx.Data,
    calibration: Dict[str, Tuple[float, float, float]],
    verbose: bool = True,
) -> mjx.Data:
    """Overwrite base_data's mocap_pos/mocap_quat from a loaded
    calibration (see `_load_obstacle_calibration` -- a live TF sample).

    Called once, before the control loop starts. `_assemble_state`
    builds every step's mjx_data via `base_data.replace(qpos=...,
    qvel=..., time=...)` -- it never touches mocap_pos/mocap_quat, so
    whatever is written here reaches every step of the run, both
    planning and rendering, for free. Mock runs the identical path (no
    camera involved): this just replaces the MJCF's own hardcoded
    default with a measured one.

    An obstacle with no matching mocap body in this scene (or vice
    versa -- calibration ran against a different scene, or a tag wasn't
    in view) is skipped, not fatal: it simply keeps the MJCF default,
    same as if this were never called.
    """
    # np.asarray on a JAX array can hand back a read-only view -- copy=True
    # forces a writable buffer.
    mocap_pos = np.array(base_data.mocap_pos, copy=True)
    mocap_quat = np.array(base_data.mocap_quat, copy=True)
    for name, (x, y, yaw) in calibration.items():
        idx = mocap_id(task.mj_model, name)
        if idx < 0:
            if verbose:
                print(f"[calibration] '{name}' has no mocap body in this "
                      f"scene -- skipped, keeping the MJCF default")
            continue
        mocap_pos[idx, 0] = x
        mocap_pos[idx, 1] = y
        # z untouched: calibration is SE(2) (see calibrate_obstacles.py),
        # same as every other planar pose in this task.
        qx, qy, qz, qw = Rotation.from_euler("z", yaw).as_quat()
        mocap_quat[idx] = [qw, qx, qy, qz]  # MuJoCo's wxyz, not scipy's xyzw
        if verbose:
            print(f"[calibration] {name}: mocap[{idx}] <- "
                  f"pos=({x:.4f}, {y:.4f})  yaw={np.degrees(yaw):.1f}deg")

    return base_data.replace(
        mocap_pos=jnp.asarray(mocap_pos), mocap_quat=jnp.asarray(mocap_quat)
    )


def apply_obstacle_calibration_to_planner(
    task: PushT,
    calibration: Dict[str, Tuple[float, float, float]],
    verbose: bool = True,
) -> None:
    """Mutate `task.object_model.obstacles.shapes` in place from the same
    loaded calibration `apply_obstacle_calibration` applies to `base_data`
    (see `_load_obstacle_calibration`).

    That function only overwrites `base_data.mocap_pos/mocap_quat`, which
    fixes collision physics but not this -- the planner's own analytic
    avoidance cost (`obstacle_cost`) reads `task.object_model.obstacles`
    directly, a *separate* obstacle list (`oim/utils/scenes.py`) that MJX
    collisions never touch. Must be called before `jit_optimize`'s first
    call: `task` is closed over (not passed as a traced argument), so any
    mutation after the first trace has no effect on the compiled planner.
    """
    shapes = task.object_model.obstacles.shapes
    for name, (x, y, yaw) in calibration.items():
        if not name.startswith("obs_"):
            continue
        idx = int(name.removeprefix("obs_")) - 1
        if not (0 <= idx < len(shapes)) or not isinstance(shapes[idx], Box):
            if verbose:
                print(f"[calibration] '{name}' has no matching planner Box "
                      f"-- skipped, keeping the scene default")
            continue
        old = shapes[idx]
        shapes[idx] = Box(center=[x, y], half_extents=old.half_extents, angle=yaw)
        if verbose:
            print(f"[calibration] {name}: planner obstacle[{idx}] <- "
                  f"center=({x:.4f}, {y:.4f})  angle={np.degrees(yaw):.1f}deg")


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
    record_dir: Optional[str] = None,
    record_name: Optional[str] = None,
    video_fps: float = 30.0,
    video_size: Tuple[int, int] = (720, 480),
    camera: Optional[Union[str, int]] = None,
    live: bool = False,
    show_samples: bool = True,
    show_optimal: bool = True,
    obstacle_calibration: bool = False,
    view_azimuth: float = _VIEW_AZIMUTH,
    view_elevation: float = _VIEW_ELEVATION,
    view_distance: Optional[float] = None,
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
        show_optimal: Overlay each block's chosen trajectory.
        view_azimuth, view_elevation: Where `--live`'s free camera starts,
            in degrees. Azimuth 180 looks back along -x from over the
            table's +x end, elevation is negative looking down. Ignored
            when `camera` names a model camera.
        view_distance: How far back that camera stands, in metres. `None`
            solves it from the table's width and the window's aspect so
            the table just fills the frame.
        obstacle_calibration: True samples obs_1/2/3's current pose
            directly off `interface`'s own TF connection (requires a real
            `Ros2Interface` -- see `_sample_obstacle_tf_live`; the mock
            has no TF tree, so this must be False on a mock run). False
            keeps the MJCF's own hardcoded obstacle poses (the only
            option before ArUco calibration existed). JSON-file
            calibration support was removed -- this is a plain on/off
            flag now, not a source string.

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
    if obstacle_calibration and not any(
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
        obstacle_calibration = False
    if obstacle_calibration:
        # Loaded once (a live sample takes ~1.5s/obstacle) and reused for
        # both destinations, rather than each re-sampling TF independently.
        calibration = _load_obstacle_calibration(interface, verbose)
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
    if jit_plans is not None:
        _pl = jit_plans(_md, _p)
        jax.block_until_ready(_pl)
    # The eager per-step cost decomposition dispatches its small kernels on
    # its first call too -- cheap, but free to pay here rather than at step 0.
    _cost_terms(task, _md)
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
        preflight_gate(interface, seconds=preflight, verbose=verbose)

    if verbose:
        print(f"[jit] ready; {'overlapped' if real_time else 'serial'} loop, "
              f"control {control_rate:.0f} Hz, stream")

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
        task=task, interface=interface, addresses=addresses, base_data=base_data,
        jit_optimize=jit_optimize, jit_interp=jit_interp, jit_plans=jit_plans,
        jit_trace=jit_trace,
        control_dt=control_dt, max_steps=max_steps, goal_pos_tol=goal_pos_tol,
        goal_theta_tol=goal_theta_tol, vel_limit=vel_limit, admm=admm, log=log,
        verbose=verbose, kicker=_StuckKicker(ctrl),
        recorder=recorder, overlay=overlay, mj_data_cpu=mj_data_cpu,
        show_samples=show_samples, show_optimal=show_optimal,
        vis_model=vis_model, draw_local_goal=draw_local_goal,
        vis_lock=vis_lock,
    )

    def _run_loop() -> Dict[str, Any]:
        if real_time:
            return _run_overlapped(params=params, **common)
        return _run_serial(params=params, replan_rate=replan_rate, **common)

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
                    common["viewer"] = viewer
                    common["overlay_base"] = (
                        viewer.user_scn.ngeom if overlay is not None else None
                    )
                    result = _run_loop()
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
            common["viewer"] = None
            common["overlay_base"] = None
            result = _run_loop()
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


def _run_serial(
    task, interface, addresses, base_data, jit_optimize, jit_interp, jit_plans,
    jit_trace, control_dt, replan_rate, max_steps, goal_pos_tol, goal_theta_tol,
    vel_limit, admm, log, verbose, params, kicker,
    recorder, overlay, mj_data_cpu, show_samples, show_optimal, viewer,
    overlay_base, vis_model, draw_local_goal, vis_lock,
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
            vis_lock=vis_lock,
        )
        reached = _log_and_check(log, task, mjx_data, params, applied,
                                 goal_pos_tol, goal_theta_tol, step, verbose, admm)
        if reached:
            break
        # Same placement the sim's flat loop uses: after the success check,
        # reading the errors that check just used.
        params = kicker.maybe_kick(params, log["pos_err"][-1],
                                   log["theta_err"][-1], step, verbose,
                                   tip_xy=np.asarray(log["robot_pos"][-1]))

    interface.stop()
    return finalize_log(log, task, reached, show_plans=admm, admm=admm)


def _run_overlapped(
    task, interface, addresses, base_data, jit_optimize, jit_interp, jit_plans,
    jit_trace, control_dt, max_steps, goal_pos_tol, goal_theta_tol, vel_limit,
    admm, log, verbose, params, kicker,
    recorder, overlay, mj_data_cpu, show_samples, show_optimal, viewer,
    overlay_base, vis_model, draw_local_goal, vis_lock,
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
    # `qpos`/`traces` are for the display thread below, not the publisher --
    # set to real values on the loop's first iteration, before either thread
    # that reads them can start.
    lock = threading.Lock()
    shared = {"samples": _sample_plan(params),
              "t_perf": t_seed,
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
                integral = s[:idx_full].sum(axis=0) * control_dt + v_partial * partial

                # Into `mj_data_cpu`, not a private copy: the viewer is
                # bound to THAT MjData, so anything written elsewhere is
                # never displayed -- the dead reckoning below used to be
                # computed and then thrown away. Under `vis_lock` because
                # the solve thread writes the same object from
                # `_visualize_step`; the two now alternate whole updates
                # instead of interleaving halves of one.
                with vis_lock:
                    mj_data_cpu.qpos[:] = qpos
                    mj_data_cpu.qpos[addresses.arm_qpos_adr] += integral
                    # The local_goal ghost (if any) only ever changes once
                    # per solve too, same as the object -- copied in, not
                    # recomputed: recomputing calls into JAX (see
                    # local_goal_marker), which this thread must never do.
                    if mocap is not None:
                        mj_data_cpu.mocap_pos[:] = mocap[0]
                        mj_data_cpu.mocap_quat[:] = mocap[1]
                    mujoco.mj_forward(vis_model, mj_data_cpu)
                    if overlay is not None:
                        overlay.draw(
                            viewer.user_scn, traces, base=overlay_base
                        )
                    viewer.sync()
            sleep = period - (time.perf_counter() - t_tick)
            if sleep > 0:
                time.sleep(sleep)

    disp = threading.Thread(target=_display_loop, daemon=True)
    if viewer is not None:
        disp.start()

    reached = False
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
        for step in range(max_steps):
            if interrupt.requested or (
                viewer is not None and not viewer.is_running()
            ):
                break
            t_loop = time.perf_counter()
            world = interface.read_state()
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
            mjx_data = _assemble_state(task, base_data, addresses, world)

            t0 = time.perf_counter()
            params, rollouts = jit_optimize(mjx_data, params)
            jax.block_until_ready(params)
            log["compute_time"].append(time.perf_counter() - t0)

            # Hand the fresh plan to the publisher (and the display thread's
            # dead-reckoning base -- same anchor time, same reasoning).
            samples = _sample_plan(params)
            prev_samples = samples
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
            _log_sample_stats(log, rollouts, _sampler_temperature(params),
                              task.consensus_scale() if admm else None)

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
                vis_lock=vis_lock,
            )
            if viewer is not None:
                # local_goal's ghost pose, same hand-off reasoning as qpos
                # above -- the display thread copies these rather than ever
                # calling draw_local_goal itself. Read under `vis_lock` and
                # OUTSIDE `lock`: the display thread takes `lock` first and
                # `vis_lock` second, so taking them in that order here too
                # is what keeps the pair acyclic.
                with vis_lock:
                    mocap_snapshot = (
                        mj_data_cpu.mocap_pos.copy(),
                        mj_data_cpu.mocap_quat.copy(),
                    )
                with lock:
                    shared["traces"] = traces
                    shared["mocap"] = mocap_snapshot
            reached = _log_and_check(log, task, mjx_data, params, first,
                                     goal_pos_tol, goal_theta_tol, step, verbose, admm)
            if reached:
                break
            # Tilt watchdog. A tool laid past ~45 deg cannot push, and once
            # the wrist folds the sampler cannot find its way back (23:14
            # run: tilt 54-82 deg for 120 steps, block never moved).
            # Sustained, not instantaneous -- good pushes brush 35-42 deg
            # for a step or two.
            tilt = float(log["tip_tilt"][-1])
            tilt_solves = tilt_solves + 1 if tilt > tilt_stop_rad else 0
            if tilt_solves >= 6:
                print(f"[stop] tip tilt {np.degrees(tilt):.0f} deg for "
                      f"{tilt_solves} consecutive solves -- wrist folded, "
                      "unrecoverable; stopping and saving")
                break
            # The kick only rewrites the sampling mean the NEXT solve starts
            # from; the publisher keeps streaming the plan already handed to
            # it, so nothing the arm is executing changes discontinuously.
            params = kicker.maybe_kick(params, log["pos_err"][-1],
                                       log["theta_err"][-1], step, verbose,
                                       tip_xy=np.asarray(log["robot_pos"][-1]))
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
        if viewer is not None:
            disp_stop.set()
            # Joined WITHOUT a timeout: on timeout the daemon thread keeps
            # running straight into the viewer teardown below and syncs a
            # half-destroyed viewer. It only ever waits one 30 Hz tick.
            disp.join()
        interface.stop()
    if verbose:
        print(f"stopped at step {step}; "
              f"{'goal reached' if reached else 'saving'}")
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
