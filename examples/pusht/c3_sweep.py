"""One-shot parameter sweep for the C3+ baseline on a PushT scene.

Runs several (q_theta, num_random, qf_theta) combos back-to-back on the SAME
scene and prints a summary table of the final errors, so we pick working
parameters from data in a single lab-machine run instead of guessing one knob
at a time.

Put this at examples/pusht/c3_sweep.py, then on the lab machine:
    MUJOCO_GL=egl uv run python examples/pusht/c3_sweep.py --scene open_table --steps 800

No video, no per-step spam -- just the final table.
"""

import argparse
import copy
import os
import time

import mujoco
import numpy as np
import yaml

from oim.algs.c3_dynamic import C3MJXSampling
from oim.runtime.mjcf import execution_model
from oim.tasks.pusht import PushT
from oim.worlds.sim3d.run import run_3d_plain


# Round 2: lock the winner (q_theta=6, num_random=8) and sweep horizon N
# (and one progress_window variant). "N" is popped out and sets num_knots.
CONFIGS = [
    # P5 directed-sampling check: same winner params as G (was 0.161 without P5)
    ("P5 q_th=6 nr=8 N=10",  dict(q_theta=6.0, num_random=8, N=10)),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", default="open_table")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--config", default=None,
                    help="robot yaml (default oim/configs/robots/point.yaml)")
    args = ap.parse_args()

    cfg_path = args.config or os.path.join(
        os.path.dirname(__file__), "..", "..", "oim", "configs", "robots", "point.yaml")
    cfg = yaml.safe_load(open(cfg_path))
    w3 = cfg["world3d"]
    control_dt = float(w3["planning_dt"])
    costs = cfg.get("costs", {})

    task = PushT(impl="jax", clutter=True, planning_dt=control_dt,
                 robot="point", env=args.scene, costs=costs)
    goal = np.asarray(task.goal)
    print(f"scene={args.scene}  goal={np.round(goal,3)}  steps={args.steps}\n")

    rows = []
    for label, kw in CONFIGS:
        # Fresh execution state for every config.
        mj_model, mj_data = execution_model(task, "point", w3, None, None)
        N = kw.pop("N", 10)
        ctrl = C3MJXSampling(
            task, plan_horizon=N * control_dt, num_knots=N, seed=5,
            q_pos=float(costs.get("q_pos", 200.0)), **kw)
        params = ctrl.init_params(seed=5)
        t0 = time.perf_counter()
        log = run_3d_plain(
            task, ctrl, params, mj_model, mj_data,
            frequency=1.0 / control_dt, max_steps=args.steps,
            goal_pos_tol=0.05, goal_theta_tol=0.1, verbose=False)
        dt = time.perf_counter() - t0
        pe = np.asarray(log["pos_err"])
        te = np.asarray(log["theta_err"])
        rows.append((label, pe[-1], te[-1], pe.min(), te.min(),
                     log["reached"], dt))
        print(f"  done {label:22s} final_pos={pe[-1]:.3f} final_th={te[-1]:.3f} "
              f"reached={log['reached']} ({dt:.0f}s)")

    print("\n" + "=" * 78)
    print(f"{'config':22s} {'final_pos':>9} {'final_th':>9} "
          f"{'best_pos':>9} {'best_th':>8} {'reached':>8}")
    print("-" * 78)
    for label, fpe, fte, bpe, bte, reached, _ in rows:
        print(f"{label:22s} {fpe:9.3f} {fte:9.3f} {bpe:9.3f} {bte:8.3f} "
              f"{str(reached):>8}")
    print("=" * 78)
    print("Pick the row with the lowest final_pos (theta almost always converges).")
    print("goal tolerance: pos<0.05, theta<0.10.")


if __name__ == "__main__":
    main()
