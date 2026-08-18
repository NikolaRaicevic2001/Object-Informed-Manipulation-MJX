"""Sampling-C3: can the outer loop achieve rotation a single contact cannot?

Scenario T (regression): pure translation must still work with sampling on.
Scenario R (rotation): a rotation-dominant goal. Without the outer loop the
    controller reached |theta| ~ 0.04 (see test_c3_rotate_diag). If sampling
    relocates to high-lever contacts, |theta| should get far larger.

The first optimize() compiles a vmap over K candidate C3 solves -- expect a
minute or two before the first step prints.

Run:  python tests/test_c3_sampling.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D


def run(name, goal, pusher0, max_steps=200):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3Sampling(task, num_candidates=8, horizon=8, admm_iters=20)
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=pusher0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy, goal_th = jnp.asarray(goal[:2]), goal[2]

    max_abs_theta = 0.0
    for step in range(max_steps):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        max_abs_theta = max(max_abs_theta, abs(float(state.object_pose[2])))
        pos_err = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
        theta_err = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
        if step % 20 == 0:
            print(f"  [{name}] step {step:3d}  pos_err={pos_err:.4f}  "
                  f"theta_err={theta_err:.4f}  obj={jnp.round(state.object_pose, 3)}")
        if pos_err < 0.02 and theta_err < 0.12:
            print(f"  [{name}] reached at step {step}")
            break

    print(f"[{name}] final obj={jnp.round(state.object_pose, 4)}  "
          f"pos_err={pos_err:.4f}  theta_err={theta_err:.4f}  "
          f"max|theta|={max_abs_theta:.4f}")
    return pos_err, theta_err, max_abs_theta


def test_sampling():
    print("=== T: translation (regression) ===")
    tp, tt, _ = run("T", (0.0, 0.06, 0.0), (0.0, -0.15))

    print("\n=== R: rotation-dominant goal ===")
    rp, rt, r_maxth = run("R", (0.0, 0.0, 0.5), (0.0, -0.15))

    print("\n[summary]")
    print(f"  T translation: pos_err={tp:.4f} theta_err={tt:.4f}")
    print(f"  R rotation:    theta_err={rt:.4f} max|theta|={r_maxth:.4f} "
          f"(goal 0.50)")

    assert tp < 0.03, "sampling broke basic translation"
    # Sampling should rotate FAR more than the ~0.04 a fixed contact managed.
    assert r_maxth > 0.25, "sampling did not unlock meaningful rotation"


if __name__ == "__main__":
    test_sampling()
    print("\nDone. T must reach; R shows whether the outer loop unlocks rotation.")
