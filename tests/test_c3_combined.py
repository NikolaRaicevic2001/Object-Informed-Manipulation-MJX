"""Combined translate+rotate goals -- the realistic reorientation task.

Pure in-place rotation (pos 0, theta 0.5) is the hardest primitive for a single
pusher and rho does not crack it (bimodal). The real pusht task is a combined
pose goal, where the translation budget gives the controller room. Check a few
combined goals at a couple of configs.

Run:  python tests/test_c3_combined.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

PUSHER0 = (0.0, -0.15)
MAX_STEPS = 160


def run(goal, rho_scale):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(
        task, num_candidates=6, horizon=6, admm_iters=8,
        w_travel=50.0, rho_scale=rho_scale,
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
        best = min(best, pe + te)  # crude combined closeness
    pe = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
    te = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
    return pe, te, state.object_pose


def main():
    goals = [
        (0.08, 0.00, 0.35),   # push right + rotate
        (0.06, 0.05, 0.30),   # diagonal + rotate
        (0.10, 0.02, 0.45),   # bigger move + bigger rotate
    ]
    print(f"{'goal':>22} {'rho_s':>6} | {'pos_err':>8} {'theta_err':>9} "
          f"| final_obj")
    for goal in goals:
        for rho_scale in [1.0, 1.2]:
            pe, te, obj = run(goal, rho_scale)
            print(f"{str(goal):>22} {rho_scale:>6.1f} | {pe:>8.4f} "
                  f"{te:>9.4f} | {jnp.round(obj, 3)}")

    print("\nRead: for a combined pose goal, want BOTH pos_err and theta_err "
          "small in the same row. If several rows reach ~(0.02, 0.1), the "
          "baseline handles realistic reorientation and in-place spin is the "
          "only hard corner.")


if __name__ == "__main__":
    main()
