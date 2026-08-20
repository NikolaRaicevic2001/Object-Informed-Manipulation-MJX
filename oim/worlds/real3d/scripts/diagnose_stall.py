#!/usr/bin/env python3
"""What is the arm actually doing while the object sits still?

Every `open_table_real` run so far stops with the object 6-9 cm from the goal
and then freezes bit-exact for hundreds of steps, at every seed and at both
q_theta values tried. That rules out the heading/position cost trade as the
cause and leaves the question this script answers: during the frozen window,
is the arm commanding nothing, or commanding something that fails to move the
block -- and if the latter, why.

    python oim/worlds/real3d/scripts/diagnose_stall.py RUN.json [RUN2.json ...]
    python oim/worlds/real3d/scripts/diagnose_stall.py --manifest .../manifest.tsv

Per run it reports, for the longest frozen stretch:

  arm motion      joint command magnitude and joint travel -- separates
                  "the planner gave up" from "the planner is pushing and the
                  block will not move"
  coherence       how much of the commanded motion points one way rather
                  than cancelling itself out. A mean that alternates
                  between two competing pushes is decisive at every single
                  step and moves nothing over hundreds of them; no
                  per-step diagnostic can see that, only this one
  joint limits    how close each joint sits to its own range, since an arm
                  pinned against a limit cannot produce the push direction
                  the task needs no matter what the cost says
  push geometry   where the tip sits relative to the block and to the
                  block->goal direction. `align_deg` is the angle between
                  (block -> goal) and (tip -> block): 0 means the tip is
                  exactly behind the block for the push it needs to make,
                  180 means it is on the far side pushing the wrong way
  contact         the object's own speed and the tip-block normal force,
                  which separates "not touching" from "touching but under
                  the block's breakaway friction"
  sample pop.     the cost population MPPI's softmax actually saw, if the
                  run was logged with it -- separates "the planner found a
                  best sample it cannot execute" from "every sample scored
                  the same, so the mean is random-walking". `eta` is the
                  effective sample size, in [1, num_samples]: at
                  num_samples the weights are uniform and the update
                  carries no information at all

Only mujoco + numpy are needed for the joint limits; without `oim` on the path
the limit check is skipped and everything else still prints.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

STALL_EPS = 1e-4        # the same bit-exact threshold the run analysis uses
BREAKAWAY_N = 0.2943    # mu * m * g for the lab block; its slide frictionloss


def wrap_angle(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def longest_frozen_window(pos_err, theta_err):
    """(start, end) of the longest run of steps with no measurable motion."""
    frozen = (np.abs(np.diff(pos_err)) < STALL_EPS) & (
        np.abs(np.diff(theta_err)) < STALL_EPS
    )
    best_len = best_end = run = 0
    for i, f in enumerate(frozen):
        run = run + 1 if f else 0
        if run > best_len:
            best_len, best_end = run, i + 1
    return best_end - best_len, best_end


def joint_ranges(scene):
    """Per-joint (low, high) in radians, or None if the model cannot load."""
    try:
        import mujoco
        from oim import ROOT
        from oim.utils.scenes import SCENES
    except Exception:  # noqa: BLE001 -- optional, everything else still works
        return None
    try:
        spec = SCENES[scene]
        path = os.path.join(ROOT, "models", spec.mjcf_by_robot["xarm6"])
        model = mujoco.MjModel.from_xml_path(path)
    except Exception:  # noqa: BLE001
        return None
    out = []
    for j in range(5):
        jnt = model.joint(f"xarm6_joint{j + 1}")
        out.append((float(jnt.range[0]), float(jnt.range[1])))
    return out


def report(path):
    with open(path) as f:
        payload = json.load(f)
    dyn, static, run = payload["dynamic"], payload["static"], payload["run"]
    hyp = payload["hyperparameters"]

    pose = np.asarray(dyn["object_pose"], float)
    goal = np.asarray(static["goal"], float)
    pos_err = np.linalg.norm(pose[:, :2] - goal[:2], axis=1)
    theta_err = np.abs(wrap_angle(pose[:, 2] - goal[2]))
    a, b = longest_frozen_window(pos_err, theta_err)

    print(f"\n=== {os.path.basename(path)}")
    print(f"    scene {run['task']}  seed {run['seed']}  "
          f"steps {len(pos_err) - 1}  q_theta {hyp['costs']['q_theta']}")
    print(f"    frozen window: steps {a}..{b}  ({b - a} steps)   "
          f"pos_err {pos_err[a]:.4f} m   theta_err {theta_err[a]:.4f} rad")
    if b - a < 20:
        print("    (no meaningful freeze -- nothing to diagnose here)")
        return

    # --- is the arm commanding anything, and is it moving?
    ctrl = np.asarray(dyn["robot_control"], float)[a:b]
    qpos = np.asarray(dyn["qpos"], float)[a:b, :5]
    print(f"    |u| per joint, mean over the freeze : "
          f"{np.abs(ctrl).mean(axis=0).round(4)}")
    print(f"    joint travel over the freeze [rad]  : "
          f"{(qpos.max(axis=0) - qpos.min(axis=0)).round(4)}")

    # --- joint limits: an arm on a stop cannot push where the cost wants
    limits = joint_ranges(run["task"])
    if limits:
        print("    joint headroom at the freeze (fraction of range used):")
        for j, (lo, hi) in enumerate(limits):
            q = qpos[:, j]
            lo_gap, hi_gap = q.min() - lo, hi - q.max()
            flag = "  <-- AT A LIMIT" if min(lo_gap, hi_gap) < 0.05 else ""
            print(f"      joint{j + 1}: [{lo:+.3f}, {hi:+.3f}]  "
                  f"held [{q.min():+.3f}, {q.max():+.3f}]  "
                  f"gap lo {lo_gap:+.3f} hi {hi_gap:+.3f}{flag}")

    # --- push geometry: is the tip behind the block, facing the goal?
    tip = np.asarray(dyn["robot_pos"], float)[a:b, :2]
    blk = pose[a:b, :2]
    to_goal = goal[:2] - blk
    to_goal /= np.maximum(np.linalg.norm(to_goal, axis=1, keepdims=True), 1e-9)
    to_blk = blk - tip
    dist = np.linalg.norm(to_blk, axis=1)
    to_blk /= np.maximum(dist[:, None], 1e-9)
    align_deg = np.degrees(
        np.arccos(np.clip(np.sum(to_goal * to_blk, axis=1), -1.0, 1.0))
    )
    print(f"    tip-to-block distance [m]  : "
          f"mean {dist.mean():.4f}  min {dist.min():.4f}  max {dist.max():.4f}")
    print(f"    align_deg (0 = tip exactly behind the block for this push):")
    print(f"      mean {align_deg.mean():5.1f}   min {align_deg.min():5.1f}   "
          f"max {align_deg.max():5.1f}   "
          f"share under 30 deg: {(align_deg < 30).mean():.0%}")
    print(f"    tip xy wander over the freeze [m] : "
          f"{(tip.max(axis=0) - tip.min(axis=0)).round(4)}")

    # --- is the commanded motion going anywhere, or cancelling itself?
    #
    # `mean(|u|)` above says the arm is commanding something. `|mean(u)|`
    # says how much of that command points ONE way. Their ratio is the
    # number MPPI's own diagnostics cannot show: the effective sample size
    # measures how decisive a SINGLE update is, not whether consecutive
    # updates agree with each other. A weighted mean that alternates
    # between two competing pushes -- go around the left side, go around
    # the right side -- looks perfectly decisive at every step and moves
    # the object nowhere. Same idea on the tip: net displacement over path
    # length, so "travelled 30 cm, ended up 1 cm away" reads as 0.03.
    coherence = np.abs(ctrl.mean(axis=0)) / np.maximum(
        np.abs(ctrl).mean(axis=0), 1e-9
    )
    flips = (np.diff(np.sign(ctrl), axis=0) != 0).mean(axis=0)
    print("    command coherence |mean(u)|/mean(|u|) per joint:")
    print(f"      {coherence.round(3)}   "
          f"(1 = one sustained push, 0 = pure back-and-forth)")
    print(f"    sign flips per step, per joint    : {flips.round(3)}")
    step_len = np.linalg.norm(np.diff(tip, axis=0), axis=1)
    net, path = float(np.linalg.norm(tip[-1] - tip[0])), float(step_len.sum())
    print(f"    tip path efficiency               : {net / max(path, 1e-9):.3f}"
          f"   (net {net:.4f} m over {path:.4f} m travelled)")

    # --- contact: not touching, or touching below breakaway?
    if "contact_normal_force_z" in dyn:
        fz = np.abs(np.asarray(dyn["contact_normal_force_z"], float)[a:b])
        print(f"    |tip-block normal force z| : mean {fz.mean():.4f} N  "
              f"max {fz.max():.4f} N   (block breakaway {BREAKAWAY_N:.4f} N)")
    if "object_velocity" in dyn:
        ov = np.abs(np.asarray(dyn["object_velocity"], float)[a:b])
        print(f"    |object velocity| mean     : {ov.mean(axis=0).round(6)}")
    if "tip_z" in dyn:
        tz = np.asarray(dyn["tip_z"], float)[a:b]
        tt = np.degrees(np.asarray(dyn["tip_tilt"], float)[a:b])
        print(f"    tip z [m] {tz.min():.4f}..{tz.max():.4f}   "
              f"tilt [deg] {tt.mean():.1f} mean / {tt.max():.1f} max")

    # --- the sample population: is the update informative at all?
    if "sample_eta" in dyn:
        eta = np.asarray(dyn["sample_eta"], float)[a:b]
        c_min = np.asarray(dyn["sample_cost_min"], float)[a:b]
        c_std = np.asarray(dyn["sample_cost_std"], float)[a:b]
        c_max = np.asarray(dyn["sample_cost_max"], float)[a:b]
        bad = np.asarray(dyn["sample_nonfinite"], float)[a:b]
        n = int(hyp.get("samples", 0)) or None
        # Free (pre-freeze) reference, so eta reads as "changed from" rather
        # than an absolute number whose scale depends on the temperature.
        free = np.asarray(dyn["sample_eta"], float)[:a]
        print("    sample population over the freeze:")
        print(f"      eta            mean {eta.mean():7.2f}  "
              f"min {eta.min():7.2f}  max {eta.max():7.2f}"
              + (f"   (of {n} samples -> "
                 f"{eta.mean() / n:.0%} carry weight)" if n else ""))
        if free.size:
            print(f"      eta before the freeze, mean {free.mean():7.2f}")
        print(f"      cost min       mean {c_min.mean():10.3f}   "
              f"spread (max-min) mean {(c_max - c_min).mean():10.3f}")
        print(f"      cost std       mean {c_std.mean():10.3f}   "
              f"std/min {np.mean(c_std / np.maximum(np.abs(c_min), 1e-9)):.4f}")
        if bad.max() > 0:
            print(f"      NONFINITE samples: max {int(bad.max())} in a step "
                  f"-- a cost term is producing inf/NaN, fix that first")
    elif "robot_control" in dyn:
        print("    (no sample-population series -- run predates "
              "log_sample_costs, or this is an ADMM run)")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="*", help="run JSON path(s); globs are fine")
    p.add_argument("--manifest", help="a sweep manifest.tsv to read paths from")
    args = p.parse_args()

    paths = []
    for raw in args.runs:
        paths.extend(sorted(glob.glob(raw)) or [raw])
    if args.manifest:
        with open(args.manifest) as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                if line.strip():
                    row = dict(zip(header, line.rstrip("\n").split("\t")))
                    if row.get("result_json"):
                        paths.append(row["result_json"])
    if not paths:
        sys.exit("no run files given")
    for path in paths:
        if os.path.exists(path):
            report(path)
        else:
            print(f"skip (missing): {path}", file=sys.stderr)


if __name__ == "__main__":
    main()
