#!/usr/bin/env python3
"""Is a real-table scene physically sane to run? Seconds, no GPU, no JAX rollout.

`tests/test_scenes.py` runs `test_start_pose_is_free_and_clear` and
`test_objects_rest_on_the_tabletop` only over `_TABLETOP_SCENES`, which it
builds with `"tabletop/" in path` -- and the real scenes live under
`xarm6_pusht_tabletop_real/`, which does not contain that substring. So the
real scenes have never been through either check. This runs the same
assertions over them, without touching the test file.

    python scripts/check_real_scenes.py
    python scripts/check_real_scenes.py open_table_real box_clutter

Reports, per scene:
  * start pose penetration (arm/block/obstacle/table interpenetration)
  * tip height and tilt from vertical at the start pose
  * distance from the tip to the block at the start (should not start on it)
  * block start and goal distance from the arm base, against the planar reach
  * every obstacle's clearance to the straight start->goal path, in units of
    the block's own crossbar length
"""

from __future__ import annotations

import math
import os
import sys

import mujoco
import numpy as np

from oim import ROOT
from oim.utils.scenes import SCENES

# tests/test_scenes.py's own number: max planar tip radius from the base,
# swept over joints 1-5. Necessary, not sufficient.
XARM6_PLANAR_REACH = 0.84

DEFAULT_SCENES = ["open_table_real", "single_obstacle_real", "box_clutter"]


def load(scene: str, robot: str = "xarm6") -> mujoco.MjModel:
    spec = SCENES[scene]
    path = os.path.join(ROOT, "models", spec.mjcf_by_robot[robot])
    model = mujoco.MjModel.from_xml_path(path)
    base = model.body("xarm6_link_base").id
    model.body_pos[base] = [*spec.xarm6_base_pos, spec.xarm6_base_z]
    yaw = math.radians(spec.xarm6_base_yaw_deg)
    model.body_quat[base] = [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]
    return model


def crossbar_len(scene: str) -> float:
    """Full crossbar length of the pushed object, from the spec's footprint."""
    kw = SCENES[scene].footprint_kwargs or {}
    half = kw.get("crossbar_half")
    return 2.0 * float(half[0]) if half else float("nan")


def check(scene: str) -> bool:
    spec = SCENES[scene]
    print(f"\n=== {scene} ===")
    print(f"  world_frame={spec.world_frame}  base={spec.xarm6_base_pos} "
          f"yaw={spec.xarm6_base_yaw_deg}  base_z={spec.xarm6_base_z}")

    try:
        model = load(scene)
    except Exception as exc:  # noqa: BLE001 -- report, don't abort the sweep
        print(f"  FAIL  MJCF did not load: {exc}")
        return False

    ok = True
    start = np.asarray(spec.object_start, dtype=float)
    goal = np.asarray(spec.goal, dtype=float)
    base = np.asarray(spec.xarm6_base_pos, dtype=float)
    tee = crossbar_len(scene)

    # ---- reach. tests/test_scenes.py hardcodes the object start at the
    # origin, which is only true for the sim tee scenes; use the spec's own.
    for label, point in (("block start", start[:2]), ("goal", goal[:2])):
        d = float(np.linalg.norm(point - base))
        mark = "ok  " if d < XARM6_PLANAR_REACH else "FAIL"
        if d >= XARM6_PLANAR_REACH:
            ok = False
        print(f"  {mark}  {label:11s} {d:.3f} m from base "
              f"(planar reach {XARM6_PLANAR_REACH})")

    # ---- start pose, from the MJCF's own "start" keyframe.
    if model.nkey < 1:
        print("  FAIL  no 'start' keyframe")
        return False
    data = mujoco.MjData(model)
    data.qpos[:] = model.key_qpos[0]
    mujoco.mj_forward(model, data)

    pen = [
        (model.geom(data.contact.geom1[i]).name,
         model.geom(data.contact.geom2[i]).name,
         float(data.contact.dist[i]))
        for i in range(data.ncon)
        if data.contact.dist[i] < -1e-4
    ]
    if pen:
        ok = False
        print(f"  FAIL  start pose penetrates: {pen}")
    else:
        print("  ok    start pose free of penetration")

    site = model.site("xarm6_tip").id
    tip = data.site_xpos[site]
    z_axis = data.site_xmat[site].reshape(3, 3)[:, 2]
    tilt = math.degrees(math.acos(float(np.clip(-z_axis[2], -1.0, 1.0))))
    to_block = float(np.linalg.norm(tip[:2] - start[:2]))
    print(f"  {'ok  ' if tip[2] > 0.02 else 'FAIL'}  tip z {tip[2]:.4f} m")
    print(f"  {'ok  ' if tilt < 30 else 'FAIL'}  tip tilt {tilt:.1f} deg off vertical")
    print(f"  {'ok  ' if to_block > 0.10 else 'WARN'}  tip is {to_block:.3f} m "
          f"from the block at the start ({to_block / tee:.2f} x crossbar)")
    ok = ok and tip[2] > 0.02 and tilt < 30

    # ---- obstacle clearance to the straight start->goal path, in object units.
    seg = goal[:2] - start[:2]
    seg_len = float(np.linalg.norm(seg))
    unit = seg / seg_len
    print(f"  path {start[:2]} -> {goal[:2]}  len {seg_len:.3f} m, "
          f"crossbar {tee:.4f} m")
    for shape in spec.obstacles.shapes:
        centre = np.asarray(getattr(shape, "center", np.zeros(2)), dtype=float)[:2]
        along = float(np.clip((centre - start[:2]) @ unit, 0.0, seg_len))
        nearest = start[:2] + along * unit
        d = float(np.linalg.norm(centre - nearest))
        kind = type(shape).__name__
        extent = float(np.max(np.abs(
            getattr(shape, "half_extents", [getattr(shape, "radius", 0.0)])
        )))
        gap = d - extent - tee / 2.0
        flag = "ok  " if gap > 0.0 else "TIGHT"
        print(f"  {flag}  {kind:8s} at ({centre[0]:+.3f},{centre[1]:+.3f}) "
              f"s={along / seg_len:.2f}L  centre-to-path {d:.3f} m, "
              f"edge gap {gap:+.3f} m ({gap / tee:+.2f} x crossbar)")

    return ok


def main() -> None:
    scenes = sys.argv[1:] or DEFAULT_SCENES
    results = {s: check(s) for s in scenes}
    print("\n=== summary ===")
    for s, good in results.items():
        print(f"  {'PASS' if good else 'FAIL'}  {s}")
    sys.exit(0 if all(results.values()) else 1)


if __name__ == "__main__":
    main()
