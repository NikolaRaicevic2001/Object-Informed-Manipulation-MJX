"""Diagnose the overshoot seen in the off-axis / rotation scenarios.

Separates the controller's INTENT (its open-loop plan) from what the sim
actually does, and checks whether the on-axis case overshoots too once the
success-break is removed. Pure diagnostic: prints, no asserts.

Run:  python tests/test_c3_diag.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3, build_contact_lcs, c3_solve
from oim.objects.planar_pushing import t_shape_footprint
from oim.worlds.sim2d.task import PushT2D


def open_loop_plan(goal, object0, pusher0):
    """Print the object trajectory the controller PLANS from the start state."""
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3(task, horizon=10, admm_iters=25, rho=0.1)
    obj = jnp.asarray(object0, dtype=float)
    ee = jnp.asarray(pusher0, dtype=float)
    lcs = build_contact_lcs(
        ctrl.shape, ctrl.D, ctrl.robot_radius, obj, ee, ctrl.dt,
        mu_c=ctrl.mu_c, slide_sign=0.0,
    )
    x_init = jnp.concatenate([obj, ee])
    xs, us, _ = c3_solve(
        lcs, x_init, ctrl.x_ref, ctrl.Q, ctrl.R, ctrl.Qf,
        rho=ctrl.rho, horizon=ctrl.horizon, admm_iters=ctrl.admm_iters,
    )
    print(f"  goal            = {goal}")
    print(f"  planned obj x   = {jnp.round(xs[:, 0], 4)}")
    print(f"  planned obj y   = {jnp.round(xs[:, 1], 4)}")
    print(f"  planned obj th  = {jnp.round(xs[:, 2], 4)}")
    print(f"  planned |u|     = {jnp.round(jnp.linalg.norm(us, axis=1), 4)}")


def closed_loop_nobreak(goal, object0, pusher0, steps=40):
    """Run closed loop with NO success-break; print obj y every 5 steps."""
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3(task, horizon=10, admm_iters=25, rho=0.1)
    params = ctrl.init_params()
    state = task.make_data(object_pose=object0, robot_pos=pusher0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    for step in range(steps):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        if step % 5 == 0 or step == steps - 1:
            print(f"  step {step:3d}  obj={jnp.round(state.object_pose, 4)}  "
                  f"ee={jnp.round(state.robot_pos, 4)}  "
                  f"u={jnp.round(u, 3)}")
    print(f"  goal was {goal}")


if __name__ == "__main__":
    print("=== (1) OPEN-LOOP PLAN from B's start (goal y=-0.06) ===")
    open_loop_plan((0.0, -0.06, 0.0), (0.0, 0.0, 0.0), (0.05, 0.060))

    print("\n=== (2) A on-axis, NO break, 40 steps (does +y overshoot 0.06?) ===")
    closed_loop_nobreak((0.0, 0.06, 0.0), (0.0, 0.0, 0.0), (0.0, -0.120))

    print("\n=== (3) B off-axis, NO break, 40 steps (watch y run past -0.06) ===")
    closed_loop_nobreak((0.0, -0.06, 0.0), (0.0, 0.0, 0.0), (0.05, 0.060))
