#!/usr/bin/env python3
"""Is the top-riding barrier not firing, or firing and being outvoted?

`check_riding.py` answers "how much riding happened". This answers why, by
splitting that number two ways and putting a number on the barrier itself.

  (a) riding INSIDE the cost's own keep-out -- the barrier fired and the
      planner paid it anyway. A weight problem.
  (b) riding inside the SAFETY band but outside the cost's -- the barrier
      never fired. A band-geometry problem.

For (b) it also names WHICH of the barrier's two gates let the step through,
because they have different knobs:

  height   the tip is higher than the slab reaches   -> contact_z_slab_above
  edge     the tip's (x, y) is outside the footprint -> contact_z_margin
           outline, catching the top EDGE from just
           beyond it
  both     neither gate held

and prints the dz / sdf percentiles of those steps, which is the number to
set the knob to.

Every row also carries the tip's own HEIGHT distribution over the moving
steps. That is what separates two fixes that would otherwise be confounded:
`contact_z_margin` changes whether the barrier fires (visible in `cost-band`),
while `w_z_tip` changes where the tip sits in the first place (visible here).
`target` is the block's mid-height, `over` is the share of moving steps with
the tip ABOVE the block's top face -- where the stick cannot be touching the
block at all -- and `on-target` is the share within 5 mm of mid-height, which
is where a side push is solid.

The `tip-block` line is the horizontal distance from the tip to the block's
outline, over EVERY step, not just the moving ones -- drifting away happens
precisely while the block is NOT moving, so filtering on motion would hide it.
`away` is the share of steps more than 100 mm clear of the block, and
`longest` is the longest UNBROKEN stretch of them. The share alone cannot tell
the two kinds of excursion apart: swapping push side means going round the
block, which is productive and takes a handful of steps, while losing the
block and wandering takes tens. The share counts both; `longest` separates
them, the same way `check_riding.py`'s `max_dwell` separates a two-step
transit from thirty steps of riding.

Note on what this is NOT measuring: tipping. The block cannot tip. Its joints
are T_x / T_y slides, a T_z yaw hinge and a T_zs vertical slide -- no roll or
pitch -- so no push at any height can roll it over in this model. Nor would it
in reality at this friction: tipping needs the push height to exceed a / mu,
where a = 34.7 mm is the distance from the block's centre of mass to the
nearest edge of its support polygon; at mu = 0.3 that is 116 mm, and the block
is 59.8 mm tall. It would take mu >= 0.58 to be possible at all.

The two bands are deliberately different. The safety band is what is unsafe
for the hardware and stays fixed; the cost band is a tunable that is supposed
to cover it. Slaving one to the other would mean narrowing the cost band
"improves" the score without changing any behaviour.

Then, for the steps in (a), it reconstructs `PushT._contact_z_cost` exactly
and compares it against the two goal terms of the same steps. That is the
whole question in one row: if the barrier's median is a few percent of
`goal_pos`, it is being outvoted and the weight is too low.

NOT reported, deliberately: the distribution of `contact_normal_force_z`.
On this path it is structurally 0.0 at every step -- `oim/runtime/logs.py`
computes it only when handed a plain `mujoco.MjData`, and `run_real` logs an
`mjx.Data`. This is also why `_contact_z_cost` was rewritten on 2026-08-19
from a force-based penalty to the kinematic slab barrier reconstructed here.
Any config comment still describing `f10 = clip(10*|f_z|, 0, 6)` is stale.

    python oim/worlds/real3d/scripts/analyze_riding.py cz_base_seed
    python oim/worlds/real3d/scripts/analyze_riding.py cz_base_seed \
        cz_parity_seed

Each argument is a sweep-name prefix under oim/results/sweeps/, same as
check_riding.py. Runs in seconds: no physics, no JAX, just the saved JSON.

Pinned to the CPU backend below. Nothing here computes with JAX, but importing
`oim.utils.scenes` pulls it in, and JAX grabs the GPU at import time -- so on a
machine whose GPU is busy this script would die with CUDA_ERROR_OUT_OF_MEMORY
before reading a single run file. `check_riding.py` has the same import and the
same exposure.
"""
from __future__ import annotations

import glob
import json
import os
import sys
from typing import Tuple

# Before anything that imports JAX: see the module docstring. `setdefault`, so
# an explicit JAX_PLATFORMS in the environment still wins.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from oim import ROOT  # noqa: E402
from oim.utils.scenes import SCENES  # noqa: E402

# The hardware keep-out. Identical to check_riding.py on purpose, so the
# `safety` column below reproduces that script's `pushed_from_top` exactly.
MARGIN = 0.012   # outward inflation of the footprint [m]
LO = -0.010      # lower edge, relative to the block's top face [m]
HI = 0.015       # upper edge; above this the tip is transiting

# `PushT`'s own defaults, for runs whose config left a knob unset.
COST_DEFAULTS = {
    "w_contact_z_exp": 0.0,
    "contact_z_slab": 0.01,
    "contact_z_slab_above": 0.0,   # 0 -> falls back to contact_z_slab
    "contact_z_below_mult": 1.0,
    "contact_z_margin": 0.0,
    "contact_z_cap": 0.0,
    "q_pos": 0.0,
    "q_theta": 0.0,
    "q_ramp_per_step": 0.0,
    "q_ramp_max": 1.0,
}


_TOP_CACHE: dict = {}


def poly_sdf(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance from each point to the footprint outline.

    Negative inside. Ray casting for the sign, per-edge point-segment
    distance for the magnitude -- the T is non-convex. Copied from
    check_riding.py so the two
    scripts cannot drift on the geometry.
    """
    x, y = pts[:, 0], pts[:, 1]
    ins = np.zeros(len(pts), bool)
    n = len(poly)
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        ins ^= ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / (yj - yi + 1e-12) + xi
        )
        j = i
    d = np.full(len(pts), np.inf)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        e = b - a
        t = np.clip(((pts - a) @ e) / (e @ e + 1e-12), 0.0, 1.0)
        d = np.minimum(d, np.linalg.norm(pts - (a + t[:, None] * e), axis=1))
    return np.where(ins, -d, d)


def top_face_z(scene: str) -> float:
    """World z of the block's top face, over its colliding geoms only.

    This is what `_contact_z_cost` calls `tip_target_z + block_half_height`:
    `tip_target_z` is read off the model as the block's mid-height and
    `block_half_height` is the same half-extent, so the sum is the top face.
    Computing it from the model keeps this script independent of whether the
    config happens to pin `tip_target_z`.
    """
    if scene in _TOP_CACHE:
        return _TOP_CACHE[scene]
    model = mujoco.MjModel.from_xml_path(
        os.path.join(ROOT, "models", SCENES[scene].mjcf_by_robot["xarm6"]))
    bid = model.body("block").id
    tops = [
        float(model.geom_pos[g][2] + model.geom_size[g][2])
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == bid
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    _TOP_CACHE[scene] = float(model.body("block").pos[2]) + max(tops)
    return _TOP_CACHE[scene]


def slab_edges(c: dict) -> Tuple[float, float]:
    """The slab's (below, above) half-thicknesses.

    `contact_z_slab_above`'s fall-back to a symmetric slab is applied the
    way `PushT.__init__` does.
    """
    below = float(c["contact_z_slab"])
    return below, (float(c["contact_z_slab_above"] or 0.0) or below)


def contact_z_cost(dz: np.ndarray, sdf: np.ndarray, c: dict) -> np.ndarray:
    """`PushT._contact_z_cost`, reconstructed. Zero outside the keep-out.

    No clamp on the exponent: `gap` is a clipped ratio in [0, 1], so the
    argument is bounded by 4 and `EXP_ARG_MAX` never binds here.
    """
    below, above = slab_edges(c)
    in_slab = (sdf <= c["contact_z_margin"]) & (dz >= -below) & (dz <= above)
    edge = np.where(dz < 0.0, below, above)
    gap = 1.0 - np.clip(np.abs(dz) / edge, 0.0, 1.0)
    raw = c["w_contact_z_exp"] * np.exp((2.0 * gap) ** 2)
    raw = np.where(dz < 0.0, raw * c["contact_z_below_mult"], raw)
    if c["contact_z_cap"] > 0.0:
        raw = np.minimum(raw, c["contact_z_cap"])
    return np.where(in_slab, raw, 0.0), in_slab


def goal_terms(
    pose: np.ndarray,
    goal: np.ndarray,
    t: np.ndarray,
    dt: float,
    c: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """`goal_pos` and `goal_theta` as the flat path scores them, ramp included.

    The ramp is `PushT._q_ramp_mult`: min((1 + per_step)**(t/dt), q_ramp_max),
    applied to both goal terms. Inert (all 1.0) at the defaults, which is what
    a config without the two keys gets.
    """
    per_step = float(c["q_ramp_per_step"] or 0.0)
    q_max = float(c["q_ramp_max"] or 1.0)
    if per_step > 0.0 and q_max > 1.0 and dt > 0.0:
        ramp = np.minimum((1.0 + per_step) ** (t / dt), q_max)
    else:
        ramp = np.ones_like(t)
    d_pos = pose[:, :2] - np.asarray(goal, float)[:2]
    d_th = np.abs(np.arctan2(np.sin(pose[:, 2] - float(goal[2])),
                             np.cos(pose[:, 2] - float(goal[2]))))
    return (ramp * c["q_pos"] * np.sum(d_pos ** 2, axis=1),
            ramp * c["q_theta"] * d_th ** 2)


def pct(a: np.ndarray, q: float) -> float:
    """The q-th percentile, 0.0 for an empty array."""
    return float(np.percentile(a, q)) if len(a) else float("nan")


def report_row(line: str, acc: list, gate: list, drift: list) -> None:
    """Score one run file and append its row to the accumulators."""
    row = line.split("\t")
    name, scene, run_json = row[0], row[1], row[4]
    if not run_json or not os.path.exists(run_json):
        print(f"{name:14s} {scene:22s} (run file missing)")
        return
    payload = json.load(open(run_json))
    dyn, static = payload["dynamic"], payload["static"]
    c = dict(COST_DEFAULTS)
    c.update(payload["hyperparameters"].get("costs") or {})

    top = top_face_z(payload["run"]["task"])
    poly = np.asarray(static["object_footprint_body"], float)[:, :2]
    pose = np.asarray(dyn["object_pose"], float)
    tip_xy = np.asarray(dyn["robot_pos"], float)[:, :2]
    tip_z = np.asarray(dyn["tip_z"], float)
    obj_v = np.asarray(dyn["object_velocity"], float)
    time = np.asarray(dyn["time"], float)
    n = min(len(pose), len(tip_xy), len(tip_z), len(obj_v), len(time))
    pose, tip_xy, tip_z = pose[:n], tip_xy[:n], tip_z[:n]
    obj_v, time = obj_v[:n], time[:n]

    # Tip in the block's own frame. Same indexing as check_riding.py (no [1:]
    # offset), so the `safety` column reproduces its numbers exactly.
    cs, sn = np.cos(-pose[:, 2]), np.sin(-pose[:, 2])
    r = tip_xy - pose[:, :2]
    local = np.stack([cs * r[:, 0] - sn * r[:, 1],
                      sn * r[:, 0] + cs * r[:, 1]], 1)
    sdf = poly_sdf(poly, local)
    dz = tip_z - top

    safety = (sdf <= MARGIN) & (dz >= LO) & (dz <= HI)
    cz, in_cost = contact_z_cost(dz, sdf, c)
    moving = np.linalg.norm(obj_v[:, :2], axis=1) > 1e-4
    m = max(int(moving.sum()), 1)

    s_share = 100.0 * int((safety & moving).sum()) / m
    c_share = 100.0 * int((in_cost & moving).sum()) / m
    gap_share = 100.0 * int((safety & ~in_cost & moving).sum()) / m

    dt = float(static.get("sim_timestep", 0.05))
    gp, gt = goal_terms(pose, static["goal"], time, dt, c)
    hot = in_cost & moving
    acc.append((s_share, c_share, gap_share))

    print(f"{name:14s} {scene:22s} "
          f"safety {s_share:5.1f}%  cost-band {c_share:5.1f}%  "
          f"unpenalized {gap_share:5.1f}%")
    if hot.any():
        print(f"{'':14s} {'':22s} "
              f"contact_z  p50 {pct(cz[hot], 50):8.1f}  "
              f"p90 {pct(cz[hot], 90):8.1f}  max {cz[hot].max():8.1f}"
              f"   (w={c['w_contact_z_exp']:g}, peak w*e^4="
              f"{c['w_contact_z_exp'] * np.e ** 4:.0f})")
        print(f"{'':14s} {'':22s} "
              f"goal_pos   p50 {pct(gp[hot], 50):8.1f}   "
              f"goal_theta p50 {pct(gt[hot], 50):8.1f}"
              f"   -> barrier is {100 * pct(cz[hot], 50) / max(
                  pct(gp[hot], 50) + pct(gt[hot], 50), 1e-9):.1f}% "
              "of the two goal terms")
    else:
        print(f"{'':14s} {'':22s} "
              f"barrier never fired on a moving step")

    # The block rests on the table, so its mid-height -- which is what
    # `tip_target_z` is read off the model as -- is exactly half the top face.
    target = top / 2.0
    tz = tip_z[moving]
    if len(tz):
        print(f"{'':14s} {'':22s} "
              f"tip height   p50 {1000 * pct(tz, 50):6.1f}  "
              f"p90 {1000 * pct(tz, 90):6.1f} mm   "
              f"(target {1000 * target:.1f}, top face {1000 * top:.1f})   "
              f"over {100 * float((tz > top).mean()):5.1f}%   "
              "on-target "
              f"{100 * float((np.abs(tz - target) <= 0.005).mean()):5.1f}%")

    far = sdf > 0.100
    away = float(far.mean())
    run = best = 0
    for flag in far:
        run = run + 1 if flag else 0
        best = max(best, run)
    # `static.control_dt`, not the hyperparameter: the logged series advance
    # one entry per REPLAN, which is what that field records. Same convention
    # check_riding.py uses for max_dwell.
    dt = float(static.get("control_dt", 0.4))
    print(f"{'':14s} {'':22s} "
          f"tip-block    p50 {1000 * pct(sdf, 50):6.1f}  "
          f"p90 {1000 * pct(sdf, 90):6.1f}  max {1000 * sdf.max():6.1f} mm"
          f"   away>100mm {100 * away:5.1f}%   longest {best:>3d} steps "
          f"({best * dt:5.2f}s)")
    drift.append((away * 100.0, float(best), float(best) * dt))

    # Which gate let the escaping steps through. `in_z` and `in_xy` are the
    # barrier's own two conditions, evaluated separately.
    esc = safety & ~in_cost & moving
    if esc.any():
        below, above = slab_edges(c)
        in_z = (dz >= -below) & (dz <= above)
        in_xy = sdf <= c["contact_z_margin"]
        only_z = int((esc & in_xy & ~in_z).sum())    # in footprint, too high
        only_xy = int((esc & ~in_xy & in_z).sum())   # right height, outside
        both = int((esc & ~in_xy & ~in_z).sum())
        e = max(int(esc.sum()), 1)
        print(f"{'':14s} {'':22s} "
              f"escapes via  height {100 * only_z / e:5.1f}%  "
              f"edge {100 * only_xy / e:5.1f}%  both {100 * both / e:5.1f}%"
              f"   (slab -{below * 1000:.0f}/+{above * 1000:.0f}mm, "
              f"margin {c['contact_z_margin'] * 1000:.0f}mm)")
        print(f"{'':14s} {'':22s} "
              f"of those:    dz  p50 {1000 * pct(dz[esc], 50):6.1f}  "
              f"p90 {1000 * pct(dz[esc], 90):6.1f} mm    "
              f"sdf p50 {1000 * pct(sdf[esc], 50):6.1f}  "
              f"p90 {1000 * pct(sdf[esc], 90):6.1f} mm")
        gate.append((100 * only_z / e, 100 * only_xy / e, 100 * both / e))


def main() -> None:
    """Report on every sweep prefix named on the command line."""
    prefixes = sys.argv[1:] or ["cz_base_seed"]
    summary = []
    for prefix in prefixes:
        found = sorted(glob.glob(f"oim/results/sweeps/{prefix}*/manifest.tsv"))
        if not found:
            print(f"no sweeps matching oim/results/sweeps/{prefix}*")
            continue
        acc: list = []
        gate: list = []
        drift: list = []
        for manifest in found:
            for line in open(manifest).read().splitlines()[1:]:
                if line.strip():
                    report_row(line, acc, gate, drift)
        summary.append((prefix, acc, gate, drift))
        print()

    if not summary:
        return

    # One row per prefix, so two sweeps can be read side by side. Medians:
    # every column here is a per-run percentage, and a couple of runs that
    # barely move would drag a mean around.
    print(f"{'sweep':>22} {'runs':>5} {'safety':>8} {'cost-band':>10} "
          f"{'unpenalized':>12} {'away>100mm':>11} {'longest':>9}")
    print("-" * 84)
    for prefix, acc, _gate, drift in summary:
        if not acc:
            continue
        a = np.asarray(acc)
        dr = np.asarray(drift) if drift else np.zeros((1, 3))
        print(f"{prefix:>22} {len(acc):>5} "
              f"{np.median(a[:, 0]):>7.1f}% {np.median(a[:, 1]):>9.1f}% "
              f"{np.median(a[:, 2]):>11.1f}% {np.median(dr[:, 0]):>10.1f}% "
              f"{np.median(dr[:, 2]):>8.2f}s")

    print("\nReading it:")
    print("  unpenalized high      -> the cost's keep-out is too small; "
          "widen it along whichever")
    print("                           gate the escapes go through: "
          "height -> contact_z_slab_above,")
    print("                           edge -> contact_z_margin")
    print("  cost-band high and    -> the barrier fires and loses; raise "
          "w_contact_z_exp")
    print("  contact_z small vs goal")
    print("  away>100mm high but   -> the arm is going ROUND the block to swap "
          "push side. Productive.")
    print("  longest short")
    print("  longest tens of steps -> it lost the block and wandered. That is "
          "an approach/fade")
    print("                           problem, not a contact_z one")


if __name__ == "__main__":
    main()
