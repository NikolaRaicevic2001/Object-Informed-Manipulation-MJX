"""Stress-test the C3 controller: off-axis pushing and rotation goals.

Scenario A (regression): on-axis push, pure +y translation -- must still reach.
Scenario B (rotation rejection): an off-center downward push naturally induces
    a torque; the goal asks for pure -y translation, so the controller must keep
    theta near zero. Diagnostic -- a single fixed contact point per horizon
    cannot fully cancel the torque, so we report how much theta drifts.
Scenario C (rotation goal): the goal is aligned with what the off-center push
    naturally produces (-y translation AND -theta rotation), an achievable
    combined motion. Tests whether C3 can hit a moderate rotation goal.

theta goals are kept near 0.4 rad, away from the +-pi branch cut (the cost does
not wrap angles yet -- that is a separate item).

Run:  python tests/test_c3_sim2d_hard.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

POS_TOL = 0.02
THETA_TOL = 0.12


def run_scenario(name, goal, object0, pusher0, max_steps=150, mu_c=0.0):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3(task, horizon=10, admm_iters=25, rho=0.1, mu_c=mu_c)
    params = ctrl.init_params()
    state = task.make_data(object_pose=object0, robot_pos=pusher0)

    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy = jnp.asarray(goal[:2])
    goal_th = goal[2]

    reached_at = -1
    pos_err = theta_err = 0.0
    max_theta = 0.0
    for step in range(max_steps):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)

        pos_err = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
        theta_err = float(jnp.abs(wrap_angle(state.object_pose[2] - goal_th)))
        max_theta = max(max_theta, abs(float(state.object_pose[2])))
        if pos_err < POS_TOL and theta_err < THETA_TOL and reached_at < 0:
            reached_at = step
            break

    ok = reached_at >= 0
    print(
        f"[{name}] reached={ok} at step {reached_at:4d} | "
        f"pos_err={pos_err:.4f} theta_err={theta_err:.4f} "
        f"| final obj={jnp.round(state.object_pose, 4)}"
    )
    return {"name": name, "reached": ok, "pos_err": pos_err,
            "theta_err": theta_err}


def test_hard_scenarios():
    results = []
    # A: on-axis translation (regression).
    results.append(run_scenario(
        "A on-axis", goal=(0.0, 0.06, 0.0),
        object0=(0.0, 0.0, 0.0), pusher0=(0.0, -0.120),
    ))
    # B: off-center downward push, goal is pure -y translation (reject torque).
    results.append(run_scenario(
        "B reject-rot", goal=(0.0, -0.06, 0.0),
        object0=(0.0, 0.0, 0.0), pusher0=(0.05, 0.060),
    ))
    # C: combined goal aligned with the off-center push (translate -y + rotate).
    results.append(run_scenario(
        "C rotate", goal=(0.0, -0.03, -0.40),
        object0=(0.0, 0.0, 0.0), pusher0=(0.05, 0.060),
    ))

    print("\n[summary]")
    for r in results:
        print(f"  {r['name']:14s} reached={r['reached']} "
              f"pos_err={r['pos_err']:.4f} theta_err={r['theta_err']:.4f}")

    # Only the on-axis regression is a hard requirement; B and C are diagnostic.
    assert results[0]["reached"], "on-axis regression failed"


if __name__ == "__main__":
    test_hard_scenarios()
    print("\nDone. A must pass; B/C are diagnostics for off-axis + rotation.")
