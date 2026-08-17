"""v4 (P1 goal-stop + P2 progress cutoff + P3 sticky hysteresis) on T/R/combined.

Expect: T stops at the goal (no more 0.37 overshoot), R rotates via push->stall->
reposition cycles, combined reaches both. Progress window shortened to 40 for
these short runs (faithful value is 100 loops at ~14 Hz).

Run:  python tests/test_c3_v4.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

PUSHER0 = (0.0, -0.15)
MAX_STEPS = 180


def run(name, goal):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(task, num_random=4, horizon=15, admm_iters=8,
                      progress_window=40)
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=PUSHER0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy, goal_th = jnp.asarray(goal[:2]), goal[2]
    reached_at, best = -1, jnp.inf
    for step in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        pe = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
        te = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
        best = min(best, pe + te)
        if pe < 0.025 and te < 0.12 and reached_at < 0:
            reached_at = step
    print(f"[{name:12s}] pos_err={pe:.4f} theta_err={te:.4f} "
          f"best(p+t)={best:.4f} reached_at={reached_at} "
          f"final={jnp.round(state.object_pose, 3)}")


def main():
    run("T translate", (0.0, 0.06, 0.0))
    run("R rotate",    (0.0, 0.0, 0.5))
    run("C combined",  (0.08, 0.0, 0.35))
    print("\nRead: T should hold near the goal (no overshoot). R/C should get "
          "both pos_err and theta_err small. reached_at >= 0 means it hit "
          "tolerance (pos<0.025, theta<0.12) at some point.")


if __name__ == "__main__":
    main()
