"""Is C3 unable to rotate, or is scenario C just infeasible for one contact?

C asks for a large rotation (theta=-0.40) with almost no translation (y=-0.03),
which a single fixed contact point cannot deliver -- rotating needs sustained
off-center pushing, which necessarily translates. This probe separates the two
explanations by (1) re-weighting toward rotation and (2) giving a goal whose
translation is compatible with the rotation the push naturally produces, and
(3) using a longer lever (push near the crossbar end).

Pure diagnostic: prints final pose and the largest |theta| achieved.

Run:  python tests/test_c3_rotate_diag.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3
from oim.objects.planar_pushing import t_shape_footprint
from oim.worlds.sim2d.task import PushT2D


def probe(name, goal, pusher0, q_pos, q_theta, w_ee, steps=120):
    task = PushT2D(footprint=t_shape_footprint(), goal=goal)
    ctrl = C3(task, q_pos=q_pos, q_theta=q_theta, w_ee=w_ee)
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=pusher0)
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)

    max_abs_theta = 0.0
    for _ in range(steps):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)
        max_abs_theta = max(max_abs_theta, abs(float(state.object_pose[2])))

    obj = state.object_pose
    print(
        f"[{name}] goal={goal}\n"
        f"    final obj = {jnp.round(obj, 4)}   "
        f"max|theta| seen = {max_abs_theta:.4f}   "
        f"(theta goal {goal[2]:+.2f})"
    )


if __name__ == "__main__":
    # 1) Baseline C: position-weighted, tiny translation, big rotation demand.
    probe("1 baseline", (0.0, -0.03, -0.40), (0.05, 0.060),
          q_pos=1000.0, q_theta=100.0, w_ee=400.0)

    # 2) Same goal, but weight rotation far above position.
    probe("2 rot-weighted", (0.0, -0.03, -0.40), (0.05, 0.060),
          q_pos=200.0, q_theta=3000.0, w_ee=400.0)

    # 3) Compatible goal (rotation WITH the translation it implies) + long lever
    #    (push near the crossbar's right end for more torque).
    probe("3 compatible", (0.0, -0.12, -0.30), (0.085, 0.055),
          q_pos=1000.0, q_theta=300.0, w_ee=400.0)

    print("\nRead: if (2)/(3) reach a much larger |theta| than (1), the "
          "machinery rotates fine and C was simply infeasible for one contact "
          "-> motivates the sampling outer loop. If |theta| stays ~0.03 "
          "everywhere, rotation itself is too weak and needs a code fix.")
