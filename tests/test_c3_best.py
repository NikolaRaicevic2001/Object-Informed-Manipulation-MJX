"""Re-check translation AND rotation at the promising configs from the sweep.

The w_travel sweep showed low w_travel (~50) + rho adaptation (rho_scale 2.0)
helps the translation stall. Now verify both scenarios at these configs, and
in particular whether rotation (R) -- which went the WRONG direction before --
is fixed. We run rho_scale 1.0 vs 2.0 at w_travel=50 to isolate the adaptation
effect on rotation.

Run:  python tests/test_c3_best.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

MAX_STEPS = 140


def run(goal, pusher0, w_travel, rho_scale):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(
        task, num_candidates=6, horizon=6, admm_iters=8,
        w_travel=w_travel, rho_scale=rho_scale,
    )
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=pusher0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy, goal_th = jnp.asarray(goal[:2]), goal[2]

    max_abs_theta = 0.0
    for _ in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        max_abs_theta = max(max_abs_theta, abs(float(state.object_pose[2])))
    pe = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
    te = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
    return pe, te, max_abs_theta, state.object_pose


def main():
    configs = [(50.0, 1.0), (50.0, 2.0)]
    print(f"{'scenario':>6} {'w_tr':>5} {'rho_s':>6} | {'pos_err':>8} "
          f"{'theta_err':>9} {'max|th|':>8} | final_obj")
    for w_travel, rho_scale in configs:
        pe, te, mt, obj = run((0.0, 0.06, 0.0), (0.0, -0.15), w_travel, rho_scale)
        print(f"{'T':>6} {w_travel:>5.0f} {rho_scale:>6.1f} | {pe:>8.4f} "
              f"{te:>9.4f} {mt:>8.4f} | {jnp.round(obj, 3)}")
        pe, te, mt, obj = run((0.0, 0.0, 0.5), (0.0, -0.15), w_travel, rho_scale)
        print(f"{'R':>6} {w_travel:>5.0f} {rho_scale:>6.1f} | {pe:>8.4f} "
              f"{te:>9.4f} {mt:>8.4f} | {jnp.round(obj, 3)}")

    print("\nRead: T should reach the goal (pos_err small, theta_err small). "
          "For R (goal theta=+0.5): max|th| should be large AND the final theta "
          "POSITIVE (right direction). Compare rho_s 1.0 vs 2.0.")


if __name__ == "__main__":
    main()
