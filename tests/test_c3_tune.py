"""Tune q_theta (rotation weight) for combined reorientation (C) and spin (R).

Pre-check (horizon 15) brought combined goal C to pos_err 0.035, theta_err 0.139
-- just short of tolerance because it under-rotates. Boosting q_theta should
close the rotation gap on C. Also see how far pure in-place spin R gets.

Run:  python tests/test_c3_tune.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

PUSHER0 = (0.0, -0.15)
MAX_STEPS = 200


def run(goal, q_theta):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(task, num_random=4, horizon=15, admm_iters=8,
                      progress_window=40, q_theta=q_theta)
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=PUSHER0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    gxy, gth = jnp.asarray(goal[:2]), goal[2]
    reached, best = -1, jnp.inf
    for step in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        pe = float(jnp.linalg.norm(state.object_pose[:2] - gxy))
        te = float(jnp.abs(wrap_angle(state.object_pose[2] - gth)))
        best = min(best, pe + te)
        if pe < 0.025 and te < 0.12 and reached < 0:
            reached = step
    return pe, te, reached, state.object_pose


def main():
    print(f"{'scen':>5} {'q_th':>5} | {'pos_err':>8} {'theta_err':>9} "
          f"{'reached':>7} | final")
    for q_theta in [100.0, 300.0, 600.0]:
        pe, te, r, obj = run((0.08, 0.0, 0.35), q_theta)   # C combined
        print(f"{'C':>5} {q_theta:>5.0f} | {pe:>8.4f} {te:>9.4f} {r:>7d} "
              f"| {jnp.round(obj, 3)}")
    for q_theta in [100.0, 300.0, 600.0]:
        pe, te, r, obj = run((0.0, 0.0, 0.5), q_theta)     # R in-place spin
        print(f"{'R':>5} {q_theta:>5.0f} | {pe:>8.4f} {te:>9.4f} {r:>7d} "
              f"| {jnp.round(obj, 3)}")
    print("\nRead: for C, want a q_theta where pos_err<0.025 AND theta_err<0.12 "
          "(reached>=0). R is the hard corner; see if higher q_theta helps at "
          "all without blowing up position.")


if __name__ == "__main__":
    main()
