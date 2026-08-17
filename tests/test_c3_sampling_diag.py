"""Confirm the +x runaway is travel-phase plowing, not a bad push.

Prints, per step: the chosen contact, the pusher-to-contact distance vs the
contact threshold (i.e. is it TRAVELING or PUSHING), the applied control, and
the object pose. If it is almost always TRAVELING and the object drifts while
traveling, the runaway is the pusher plowing through the object between
contacts -- a relocation problem, not a C3 problem.

Run:  python tests/test_c3_sampling_diag.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import C3Sampling
from oim.objects.planar_pushing import t_shape_footprint
from oim.worlds.sim2d.task import PushT2D


def main():
    task = PushT2D(footprint=t_shape_footprint(), goal=(0.0, 0.06, 0.0))
    ctrl = C3Sampling(task, num_candidates=8, horizon=8, admm_iters=20)
    params = ctrl.init_params()
    state = task.make_data(object_pose=(0.0, 0.0, 0.0), robot_pos=(0.0, -0.15))
    optimize = jax.jit(ctrl.optimize)
    get_action = jax.jit(ctrl.get_action)

    print(f"contact_thresh = {ctrl.contact_thresh}")
    for step in range(15):
        params, _ = optimize(state, params)
        u = jnp.clip(get_action(params, state.time), task.u_min, task.u_max)
        pusher = state.robot_pos
        best = params.best_contact
        dist = float(jnp.linalg.norm(pusher - best))
        mode = "PUSH " if dist < ctrl.contact_thresh else "travel"
        print(
            f"step {step:2d} [{mode}] dist={dist:.4f}  "
            f"best_contact={jnp.round(best, 3)}  u={jnp.round(u, 3)}  "
            f"obj={jnp.round(state.object_pose, 3)}  "
            f"ee={jnp.round(pusher, 3)}"
        )
        state = task.rollout.step(task.sim_model, state, u)


if __name__ == "__main__":
    main()
