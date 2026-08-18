"""Faithful sampling-C3 (v3, hysteresis + modes) on translation/rotation/combined.

v3 follows Venkatesh Algorithm 1: 3 samples (current EE + previous switch +
one random), hysteresis-gated switching, commit-to-target while repositioning.
This should stop the v2 flip-flop that made rotation a coin toss.

Run:  python tests/test_c3_v3.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

PUSHER0 = (0.0, -0.15)
MAX_STEPS = 160


def run(name, goal, hysteresis_ratio=0.7):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(
        task, horizon=6, admm_iters=8, w_travel=200.0,
        rho_scale=1.0, hysteresis_ratio=hysteresis_ratio,
    )
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=PUSHER0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy, goal_th = jnp.asarray(goal[:2]), goal[2]
    best = jnp.inf
    for _ in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        pe = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
        te = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
        best = min(best, pe + te)
    pe = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
    te = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
    print(f"[{name:12s}] pos_err={pe:.4f}  theta_err={te:.4f}  "
          f"best(p+t)={best:.4f}  final={jnp.round(state.object_pose, 3)}")
    return pe, te


def main():
    run("T translate", (0.0, 0.06, 0.0))
    run("R rotate",    (0.0, 0.0, 0.5))
    run("C combined",  (0.08, 0.0, 0.35))
    print("\nRead: with hysteresis, all three should be stable (no wrong-way "
          "spin, no runaway). Want pos_err and theta_err both small per row.")


if __name__ == "__main__":
    main()
