"""Sweep w_travel (and check rho adaptation) on the translation stall case.

The T-translation run stalled at pos_err ~0.17 with drift. Hypothesis: the
travel penalty w_travel was too high, so the controller would not relocate to
the contact needed to correct the residual x/theta error. Sweep w_travel and
see where it converges; also compare rho_scale 1.0 vs 2.0 (growing-rho, the C3
paper's residual-balancing trick).

Small settings (K=6, horizon=6, admm_iters=8) so each config compiles fast.
Each config is a separate jit compile -- expect ~30-60 s per row.

Run:  python tests/test_c3_wtravel_sweep.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

GOAL = (0.0, 0.06, 0.0)
PUSHER0 = (0.0, -0.15)
MAX_STEPS = 90


def run(w_travel, rho_scale):
    task = PushT2D(footprint=t_shape_footprint(), goal=GOAL)
    ctrl = C3Sampling(
        task, num_candidates=6, horizon=6, admm_iters=8,
        w_travel=w_travel, rho_scale=rho_scale,
    )
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=PUSHER0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy = jnp.asarray(GOAL[:2])

    best_pos = jnp.inf
    for _ in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        best_pos = min(best_pos, float(jnp.linalg.norm(state.object_pose[:2] - goal_xy)))
    pos_err = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
    theta_err = float(jnp.abs(wrap_angle(state.object_pose[2] - GOAL[2])))
    return pos_err, theta_err, best_pos, state.object_pose


def main():
    print(f"{'w_travel':>9} {'rho_s':>6} | {'final_pos':>9} {'best_pos':>9} "
          f"{'theta_err':>9} | final_obj")
    rows = [
        (50.0, 1.0), (150.0, 1.0), (500.0, 1.0), (1500.0, 1.0),
        (150.0, 2.0),  # rho adaptation check at a mid w_travel
    ]
    for w_travel, rho_scale in rows:
        pe, te, bp, obj = run(w_travel, rho_scale)
        print(f"{w_travel:>9.0f} {rho_scale:>6.1f} | {pe:>9.4f} {bp:>9.4f} "
              f"{te:>9.4f} | {jnp.round(obj, 3)}")

    print("\nRead: lower w_travel should let the controller relocate and drive "
          "final_pos below the ~0.17 stall. Watch theta_err too (over-relocation "
          "can add spin). rho_s=2.0 row shows whether growing-rho helps.")


if __name__ == "__main__":
    main()
