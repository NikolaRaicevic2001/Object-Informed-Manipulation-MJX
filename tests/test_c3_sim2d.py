"""First closed-loop run of the C3 controller in the 2D world.

Pushes a T-shaped object from below along its symmetry axis toward a +y goal --
a near-pure translation, so success does not hinge on the controller balancing
rotation on the very first run. We replan every step (receding horizon) and
execute on the analytic 2D rollout, exactly as oim.worlds.sim2d.run would.

This is the milestone: does the C3 planner, driving the pusher through the
contact LCS, actually move the object toward the goal in closed loop?

Run:  python tests/test_c3_sim2d.py
(The first optimize() call compiles a sizeable unrolled graph -- give it a
few tens of seconds before the first step prints.)
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3
from oim.objects.planar_pushing import t_shape_footprint, wrap_angle
from oim.worlds.sim2d.task import PushT2D

MAX_STEPS = 120
GOAL = (0.0, 0.06, 0.0)          # translate +y by 6 cm
OBJECT0 = (0.0, 0.0, 0.0)
PUSHER0 = (0.0, -0.120)          # just below the T's stem, on the symmetry axis


def test_c3_closed_loop_pushes_to_goal():
    task = PushT2D(footprint=t_shape_footprint(), goal=GOAL)
    ctrl = C3(task, horizon=10, admm_iters=25, rho=0.1)
    params = ctrl.init_params()
    state = task.make_data(object_pose=OBJECT0, robot_pos=PUSHER0)

    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)
    goal_xy = jnp.array(GOAL[:2])

    errs, thetas = [], []
    for step in range(MAX_STEPS):
        params, _ = optimize(state, params)
        u = get_action(params, state.time)
        u = jnp.clip(u, task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)

        pos_err = float(jnp.linalg.norm(state.object_pose[:2] - goal_xy))
        theta = float(jnp.abs(wrap_angle(state.object_pose[2])))
        errs.append(pos_err)
        thetas.append(theta)
        if step % 10 == 0 or pos_err < 0.02:
            print(
                f"step {step:3d}  pos_err={pos_err:.4f}  |theta|={theta:.4f}  "
                f"obj={jnp.round(state.object_pose, 4)}  "
                f"ee={jnp.round(state.robot_pos, 4)}"
            )
        if pos_err < 0.02:
            print(f"goal reached at step {step}")
            break

    e0, e_best = errs[0], min(errs)
    print(f"\n[c3-2d] start pos_err {e0:.4f} -> best {e_best:.4f} "
          f"(final {errs[-1]:.4f}); max |theta| seen {max(thetas):.4f}")

    assert e_best < 0.5 * e0, (
        "C3 did not make meaningful progress toward the goal in closed loop"
    )


if __name__ == "__main__":
    test_c3_closed_loop_pushes_to_goal()
    print("\nOK: the C3 controller drives the object toward the goal in the "
          "2D closed loop.")
