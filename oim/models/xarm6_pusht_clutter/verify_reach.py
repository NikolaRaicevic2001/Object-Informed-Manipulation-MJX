"""Standalone reachability check for the xArm6 + pusht-clutter scene.

Not a unit test, no oim imports -- a dev-only script (mirrors
`oim/models/xarm6/verify_model.py`'s standalone-checking style) used to
pick the arm's ground-mounted base placement (x, y, yaw; z is fixed at 0 --
see the "arm base stays on the ground" decision in
`XARM6_ADMM_INTEGRATION_PLAN.md`) and a "vertical stylus" wrist
configuration (joint4/joint5) before wiring any of this into a `Task`.

Workspace being checked against (from `xarm6_pusht_clutter.xml`, matching
`pusht_clutter.xml`'s original layout): block starts near (0, 0), goal at
(0.50, 0.48), obstacles spread over roughly x in [0.08, 0.38], y in
[0.10, 0.47]. The candidate offset must let the stick tip reach across
that whole footprint at close to the block's half-height (z ~ 0.03),
while keeping the stick close to vertical (small tilt from straight down)
so the paper's ell_r "approach the object squarely" cost has something
sensible to reward, and the arm's own base/links don't overlap the block's
travel path.
"""

import itertools
import pathlib

import mujoco
import numpy as np

HERE = pathlib.Path(__file__).parent

# Candidate ground-level base placements (x, y, yaw_deg). Placed outside
# the obstacle/goal footprint so the arm's own body doesn't sit in the
# block's way, with joint1's full rotation range sweeping into it.
CANDIDATES = [
    (-0.35, 0.20, 0.0),
    (-0.30, 0.10, 15.0),
    (0.20, -0.35, 90.0),
    (0.20, 0.75, -90.0),
]

# Workspace corners to check reachability against (x, y), target z ~ 0.03.
WORKSPACE_PTS = [
    (0.0, 0.0),   # block start
    (0.50, 0.48),  # goal
    (0.08, 0.32),  # obstacle (circle)
    (0.38, 0.10),  # obstacle (box)
    (0.15, 0.47),  # obstacle (tri)
]

TARGET_Z = 0.03


def load_model(
    base_x: float, base_y: float, base_yaw_deg: float
) -> mujoco.MjModel:
    """The clutter scene with the arm base moved to (base_x, base_y)."""
    m = mujoco.MjModel.from_xml_path(
        str((HERE / "xarm6_pusht_clutter.xml").resolve()))
    base_id = m.body("xarm6_link_base").id
    m.body_pos[base_id] = [base_x, base_y, 0.0]
    yaw = np.deg2rad(base_yaw_deg)
    m.body_quat[base_id] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    return m


def tip_pose(
    m: mujoco.MjModel,
    d: mujoco.MjData,
    q1: float,
    q2: float,
    q3: float,
    q4: float,
    q5: float,
) -> np.ndarray:
    """Forward-kinematic tip position for one arm pose.

    Sets the 5 arm joints (block/obstacles left at their defaults) and
    return the stick tip's world position and its local z-axis tilt from
    vertical (0 = perfectly straight down/up, matching "stick points along
    world z" -- sign doesn't matter, only magnitude).
    """
    d.qpos[:5] = [q1, q2, q3, q4, q5]
    mujoco.mj_forward(m, d)
    tip_id = m.site("xarm6_tip").id
    tip_pos = d.site_xpos[tip_id].copy()
    tip_z_axis = d.site_xmat[tip_id].reshape(3, 3)[:, 2]
    tilt = np.degrees(np.arccos(np.clip(abs(tip_z_axis[2]), -1, 1)))
    return tip_pos, tilt


def main() -> None:
    """Sweep base placements and report the reachable fraction."""
    j1_range = np.linspace(-np.pi, np.pi, 24)
    j2_range = np.deg2rad(np.linspace(-100, 100, 9))
    j3_range = np.deg2rad(np.linspace(-200, 5, 9))
    # A handful of wrist (joint4/joint5) pairs to try at each shoulder/elbow
    # sample, looking for whichever keeps the stick most vertical.
    wrist_candidates = [
        (0.0, np.deg2rad(a)) for a in (-90, -60, -30, 0, 30, 60, 90)
    ] + [(np.deg2rad(b), np.deg2rad(a))
         for b in (-90, 90) for a in (-30, 0, 30)]

    for base_x, base_y, base_yaw in CANDIDATES:
        m = load_model(base_x, base_y, base_yaw)
        d = mujoco.MjData(m)

        best_per_target = {pt: None for pt in WORKSPACE_PTS}
        n_checked = 0
        n_finite = 0
        for q1, q2, q3 in itertools.product(j1_range, j2_range, j3_range):
            for q4, q5 in wrist_candidates:
                n_checked += 1
                tip_pos, tilt = tip_pose(m, d, q1, q2, q3, q4, q5)
                if not np.all(np.isfinite(tip_pos)):
                    continue
                n_finite += 1
                for pt in WORKSPACE_PTS:
                    dist2 = (
                        (tip_pos[0] - pt[0]) ** 2
                        + (tip_pos[1] - pt[1]) ** 2
                        + (tip_pos[2] - TARGET_Z) ** 2
                    )
                    prev = best_per_target[pt]
                    # Prefer close position, then low tilt as a tiebreaker.
                    score = (dist2, tilt)
                    if prev is None or score < prev[0]:
                        best_per_target[pt] = (score, (q1, q2, q3, q4, q5))

        print(f"\n=== base=({base_x}, {base_y}), yaw={base_yaw} deg ===")
        print(f"  poses checked: {n_checked}, finite: {n_finite}")
        worst_dist = 0.0
        for pt, best in best_per_target.items():
            (dist2, tilt), _cfg = best
            dist = np.sqrt(dist2)
            worst_dist = max(worst_dist, dist)
            print(
                f"  target {pt}: closest reach {dist * 100:.1f} cm away, "
                f"tilt {tilt:.1f} deg from vertical"
            )
        print("  worst-case miss across all targets: "
              f"{worst_dist * 100:.1f} cm")


if __name__ == "__main__":
    main()
