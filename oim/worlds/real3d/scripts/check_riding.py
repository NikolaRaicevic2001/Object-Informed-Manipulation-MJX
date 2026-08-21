#!/usr/bin/env python3
"""How much of the pushing was done by pressing on the block from above?

The single safety question for hardware: a stick that catches the block's
top face or its top EDGE applies a tipping torque, not a push -- it flips
the block or snaps the stick. Pushing from the side is what the task wants,
and passing OVER the block at a clear height to reach a new contact point is
fine too. Only the height in between -- too low to clear, too high to push
the side -- is forbidden, and this measures time spent there.

    python oim/worlds/real3d/scripts/check_riding.py M_guard_seed
    python oim/worlds/real3d/scripts/check_riding.py Q1_seed H2_seed

Each argument is a sweep-name prefix under oim/results/sweeps/. Matches
`<prefix>*/manifest.tsv`, so `M_guard_seed` covers M_guard_seed1..5, and
every scene in each manifest is scored, not just the first.

Two numbers per run:

  pushed_from_top  of the steps where the OBJECT actually moved, the share
                   with the tip in the keep-out. This is the risk itself:
                   how much of the real pushing was done from on top.
  max_dwell        longest UNBROKEN stretch inside the keep-out. Descending
                   to side-push height means crossing the band, so a share
                   alone cannot tell a two-step transit from thirty steps of
                   riding; this can.

The keep-out matched here is the one `PushT._contact_z_cost` enforces: the
tip's (x, y), rotated into the block frame, within MARGIN of its real
footprint, AND its height inside [top face + LO, top face + HI]. Change
these three constants whenever that cost changes.
"""

from __future__ import annotations

import glob
import json
import os
import sys

import mujoco
import numpy as np

from oim import ROOT
from oim.utils.scenes import SCENES

MARGIN = 0.012   # outward inflation of the footprint [m]
LO = -0.010      # keep-out lower edge, relative to the block's top face [m]
HI = 0.015       # keep-out upper edge; above this the tip is transiting


def poly_sdf(poly: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Signed distance from each point to the footprint outline.

    Negative inside. Ray casting for the sign, per-edge point-segment
    distance for the magnitude -- the T is non-convex, so neither a
    circle nor a convex hull would do.
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


_TOP_CACHE: dict = {}


def top_face_z(scene: str) -> float:
    """World z of the block's top face, over its colliding geoms only."""
    if scene in _TOP_CACHE:
        return _TOP_CACHE[scene]
    path = os.path.join(ROOT, "models", SCENES[scene].mjcf_by_robot["xarm6"])
    model = mujoco.MjModel.from_xml_path(path)
    bid = model.body("block").id
    tops = [
        float(model.geom_pos[g][2] + model.geom_size[g][2])
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == bid
        and (model.geom_contype[g] or model.geom_conaffinity[g])
    ]
    _TOP_CACHE[scene] = float(model.body("block").pos[2]) + max(tops)
    return _TOP_CACHE[scene]


def report(manifest: str) -> None:
    rows = open(manifest).read().splitlines()
    for line in rows[1:]:
        if line.strip():
            _report_row(line)


def _report_row(line: str) -> None:
    row = line.split("\t")
    name, scene, run_json = row[0], row[1], row[4]
    if not run_json or not os.path.exists(run_json):
        print(f"{name:14s} {scene:22s} (run file missing)")
        return
    payload = json.load(open(run_json))
    dyn, static = payload["dynamic"], payload["static"]

    top = top_face_z(payload["run"]["task"])
    poly = np.asarray(static["object_footprint_body"], float)[:, :2]
    pose = np.asarray(dyn["object_pose"], float)
    tip_xy = np.asarray(dyn["robot_pos"], float)[:, :2]
    tip_z = np.asarray(dyn["tip_z"], float)
    obj_v = np.asarray(dyn["object_velocity"], float)
    n = min(len(pose), len(tip_xy), len(tip_z), len(obj_v))
    pose, tip_xy, tip_z, obj_v = pose[:n], tip_xy[:n], tip_z[:n], obj_v[:n]

    # Tip position in the block's own frame.
    c, s = np.cos(-pose[:, 2]), np.sin(-pose[:, 2])
    r = tip_xy - pose[:, :2]
    local = np.stack([c * r[:, 0] - s * r[:, 1], s * r[:, 0] + c * r[:, 1]], 1)

    near = poly_sdf(poly, local) <= MARGIN
    dz = tip_z - top
    bad = near & (dz >= LO) & (dz <= HI)
    moving = np.linalg.norm(obj_v[:, :2], axis=1) > 1e-4
    m = max(int(moving.sum()), 1)

    # Longest unbroken stretch in the keep-out: the run length of the
    # longest block of True in `bad`.
    dwell = best = 0
    for flag in bad:
        dwell = dwell + 1 if flag else 0
        best = max(best, dwell)
    dt = float(payload["hyperparameters"].get("control_dt", 0.05))

    print(
        f"{name:14s} {scene:22s} "
        f"pushed_from_top {100 * int((bad & moving).sum()) / m:5.1f}%   "
        f"max_dwell {best:4d} steps ({best * dt:5.2f}s)"
    )


def main() -> None:
    prefixes = sys.argv[1:] or ["H3_seed"]
    for prefix in prefixes:
        found = sorted(glob.glob(f"oim/results/sweeps/{prefix}*/manifest.tsv"))
        if not found:
            print(f"no sweeps matching oim/results/sweeps/{prefix}*")
            continue
        for manifest in found:
            report(manifest)
        print()


if __name__ == "__main__":
    main()
