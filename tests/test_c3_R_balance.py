"""Find a rho_scale that balances rotation vs position holding on R.

At w_travel=50: rho_scale=1.0 rotates strongly (theta 0.77) but drifts position
(0.42); rho_scale=2.0 holds position (0.023) but under-rotates (theta 0.20).
Sweep the middle to find a knee that rotates near the +0.5 goal while keeping
position error small.

Run:  python tests/test_c3_R_balance.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

GOAL = (0.0, 0.0, 0.5)      # rotate +0.5 rad in place
PUSHER0 = (0.0, -0.15)
MAX_STEPS = 140


def run(rho_scale):
    task = PushT2D(footprint=t_shape_footprint(), goal=GOAL)
    ctrl = C3Sampling(
        task, num_candidates=6, horizon=6, admm_iters=8,
        w_travel=50.0, rho_scale=rho_scale,
    )
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=PUSHER0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    max_th = 0.0
    for _ in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        max_th = max(max_th, abs(float(state.object_pose[2])))
    pe = float(jnp.linalg.norm(state.object_pose[:2]))
    th = float(state.object_pose[2])
    te = float(jnp.abs(wrap_angle(th - GOAL[2])))
    return pe, te, th, max_th, state.object_pose


def main():
    print(f"{'rho_s':>6} | {'pos_err':>8} {'theta':>7} {'theta_err':>9} "
          f"{'max|th|':>8} | final_obj   (goal theta=+0.50)")
    for rho_scale in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        pe, te, th, mt, obj = run(rho_scale)
        print(f"{rho_scale:>6.1f} | {pe:>8.4f} {th:>7.3f} {te:>9.4f} "
              f"{mt:>8.4f} | {jnp.round(obj, 3)}")
    print("\nRead: want a row with theta near +0.5 AND pos_err small. That "
          "rho_scale is the balance point for in-place rotation.")


if __name__ == "__main__":
    main()
