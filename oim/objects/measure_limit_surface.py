"""Measure a `PushObject`'s `limit_surface_radius` against the compiled scene.

Support friction on the tabletop scenes is the table contact's mu*N, so the
torque a footprint transmits before it starts to turn is an outcome of its
geometry rather than a number to choose. This ramps a pure yaw torque on the
resting block, bisects for the breakaway value tau*, and reports

    r = tau* / (mu * m * g)

which is what `PlanarPushingObject` uses for its torque budget. The force
channel is swept the same way as a check that it still breaks away near
mu*m*g, the number `wrench_limit` assumes.

    python -m oim.objects.measure_limit_surface T_large_block
    python -m oim.objects.measure_limit_surface A_block --scene open_table_real

The T-block's 0.0422 and the icra C's 0.0548 were produced this way.
"""

import argparse
from typing import Callable, Tuple

import numpy as np

HOLD_S = 1.0          # how long each trial wrench is held
# Breakaway is the ONSET of rotation. 0.005 rad in one second reproduces
# the real T's documented measurement (tau* = 0.012405 N*m, r = 0.0422 m,
# scenes.py) to 0.5%; a "clearly turning" threshold like 0.05 rad lands
# 20% higher, because a box footprint creeps for a while before it lets
# go. The same criterion on every object is what keeps their torque
# budgets comparable.
YAW_MOVED = 0.005
POS_MOVED = 0.02      # m in HOLD_S: the plant test pins creep under 0.01
REL_TOL = 0.005       # bisection stops at this relative width


def bisect_breakaway(moved_at: Callable[[float], float], lo: float, hi: float,
                     threshold: float) -> Tuple[float, list]:
    """Smallest magnitude whose response exceeds `threshold`.

    `moved_at(m)` returns the displacement after holding magnitude `m`.
    `hi` must already move; `lo` must not. Returns the estimate and the
    trials, so the transition can be read rather than trusted.
    """
    trials = []
    while (hi - lo) / hi > REL_TOL:
        mid = 0.5 * (lo + hi)
        d = moved_at(mid)
        trials.append((mid, d))
        if d > threshold:
            hi = mid
        else:
            lo = mid
    return hi, trials


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("object", help="a key of oim.objects.library.PUSH_OBJECTS")
    ap.add_argument("--scene", default="open_table_real")
    ap.add_argument("--robot", default="xarm6")
    ap.add_argument("--config", default="xarm6_real")
    args = ap.parse_args()

    import mujoco  # noqa: PLC0415
    from oim.experiment import load_config  # noqa: PLC0415
    from oim.objects.library import PUSH_OBJECTS  # noqa: PLC0415
    from oim.tasks.pusht import PushT  # noqa: PLC0415
    from oim.worlds.object_only.plant import build_plant  # noqa: PLC0415

    obj = PUSH_OBJECTS[args.object]
    cfg = load_config(args.config)
    task = PushT(
        impl="jax", clutter=True, planning_dt=cfg["world3d"]["planning_dt"],
        robot=args.robot, env=args.scene, costs=cfg.get("costs"),
        push_object=args.object,
    )
    plant = build_plant("mujoco", task, args.robot, cfg["world3d"],
                        control_dt=float(task.dt))
    start = np.asarray(task.start, dtype=float)
    steps = int(round(HOLD_S / float(task.dt)))
    budget = obj.mu * obj.mass * 9.81            # mu*m*g, the force budget

    def hold(wrench: np.ndarray) -> np.ndarray:
        # Full reset, not just the plant's: `MujocoPlant.reset` leaves the
        # solver warm start and the block's vertical slide as the previous
        # trial left them, and at the onset threshold that is enough to
        # move the answer between runs.
        mujoco.mj_resetData(plant.mj_model, plant.mj_data)
        plant.reset(start)
        pose = start
        for _ in range(steps):
            pose = plant.step(wrench)
        return np.asarray(pose) - start

    def yaw_after(tau: float) -> float:
        return abs(float(hold(np.array([0.0, 0.0, tau]))[2]))

    def travel_after(force: float) -> float:
        return abs(float(hold(np.array([force, 0.0, 0.0]))[0]))

    print(f"{args.object} on {args.scene}: m={obj.mass:.4f} kg  mu={obj.mu}  "
          f"mu*m*g={budget:.4f} N  boxes={len(obj.boxes)}  "
          f"hold={HOLD_S}s ({steps} steps of {float(task.dt):.3f}s)")

    # Torque. A footprint of reach R cannot transmit more than mu*m*g*R,
    # so that bounds the search from above; the lower bound is zero.
    reach = float(np.max(np.abs(np.asarray(obj.footprint().vertices))))
    tau_hi = budget * reach
    assert yaw_after(tau_hi) > YAW_MOVED, "footprint reach bound did not move"
    tau_star, trials = bisect_breakaway(yaw_after, 0.0, tau_hi, YAW_MOVED)
    print("\n  torque sweep (N*m -> yaw rad after hold):")
    for m, d in trials:
        print(f"    {m:8.5f}  ->  {d:7.4f}  {'MOVED' if d > YAW_MOVED else 'held'}")
    r = tau_star / budget
    print(f"  breakaway tau* = {tau_star:.5f} N*m   ->   "
          f"limit_surface_radius = {r:.4f} m  (footprint reach {reach:.4f})")

    # Force, as a check on the budget the whole formulation is scaled by.
    f_star, ftrials = bisect_breakaway(travel_after, 0.0, 2.0 * budget, POS_MOVED)
    print("\n  force sweep (N -> travel m after hold):")
    for m, d in ftrials:
        print(f"    {m:8.4f}  ->  {d:7.4f}  {'MOVED' if d > POS_MOVED else 'held'}")
    print(f"  breakaway F* = {f_star:.4f} N = {f_star / budget:.3f} x mu*m*g")
    print(f"\n  -> limit_surface_radius={r:.4f},")


if __name__ == "__main__":
    main()
