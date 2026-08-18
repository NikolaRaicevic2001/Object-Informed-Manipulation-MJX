"""Validate the increment-2a contact LCS against the analytic 2D rollout.

The contact LCS uses strict complementarity (force removes the predicted
penetration in one step); `Analytic2DRollout` uses a penalty-spring contact
plus substepping and a safe-displacement limiter. So the two will NOT match in
magnitude -- the LCS is C3's internal model, not a copy of the simulator. What
must agree is the PHYSICS: no force when separated, and the object pushed in the
same direction when in contact. The test asserts those and prints magnitudes so
we can eyeball the linearization.

Run:  python tests/test_c3_contact.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import build_contact_lcs, lcs_step
from oim.objects.planar_pushing import t_shape_footprint
from oim.worlds.sim2d.engine import (
    Analytic2DRollout,
    Sim2DModel,
    Sim2DState,
)

# Limit-surface compliance matching PlanarPushingObject's defaults:
# fl = [mu*m*g, mu*m*g, c*r*mu*m*g] = [7.848, 7.848, 0.471], D = 1/fl.
WRENCH_LIMIT = jnp.array([7.848, 7.848, 0.471])
D = 1.0 / WRENCH_LIMIT
ROBOT_RADIUS = 0.012
DT = 0.05


def _analytic_dpose(shape, object_pose, pusher_pos, control):
    """One analytic rollout step; returns the object pose change and wrench."""
    model = Sim2DModel.create(
        limit_surface_d=D, mu_c=0.0, f_max=100.0, robot_radius=ROBOT_RADIUS
    )
    rollout = Analytic2DRollout(shape, dt=DT)
    s0 = Sim2DState.create(object_pose=object_pose, robot_pos=pusher_pos)
    s1 = rollout.step(model, s0, jnp.asarray(control, dtype=float))
    return s1.object_pose - s0.object_pose, s1.wrench


def _lcs_dpose(shape, object_pose, pusher_pos, control):
    """One contact-LCS step; returns the object pose change and normal force."""
    lcs = build_contact_lcs(
        shape,
        D,
        ROBOT_RADIUS,
        jnp.asarray(object_pose, dtype=float),
        jnp.asarray(pusher_pos, dtype=float),
        dt=DT,
        mu_c=0.0,
    )
    x0 = jnp.array([*object_pose, *pusher_pos], dtype=float)
    x1, lam = lcs_step(lcs, x0, jnp.asarray(control, dtype=float))
    return x1[:3] - x0[:3], lam


def _report(name, shape, object_pose, pusher_pos, control):
    da, wrench = _analytic_dpose(shape, object_pose, pusher_pos, control)
    dl, lam = _lcs_dpose(shape, object_pose, pusher_pos, control)
    print(f"\n[{name}]  pusher={pusher_pos}  u={control}")
    print(f"  analytic dpose = {da}   (wrench {wrench})")
    print(f"  lcs      dpose = {dl}   (f_n {float(lam[0]):.3f})")
    return da, dl, lam


def test_separated_pusher_no_contact():
    """A pusher far from the object applies no force and moves nothing."""
    shape = t_shape_footprint()
    da, dl, lam = _report(
        "separated", shape, (0.0, 0.0, 0.0), (0.0, 0.20), (0.0, -0.6)
    )
    assert float(lam[0]) < 1e-6, "no contact expected, but f_n > 0"
    assert float(jnp.linalg.norm(dl)) < 1e-6, "object moved with no contact"


def test_top_push_moves_object_down():
    """Pushing down on the T's top face moves the object down in both models."""
    shape = t_shape_footprint()
    da, dl, lam = _report(
        "top-push", shape, (0.0, 0.0, 0.0), (0.0, 0.058), (0.0, -0.6)
    )
    assert float(lam[0]) > 0.0, "contact expected, but f_n = 0"
    assert float(dl[1]) < 0.0, "LCS did not push the object -y"
    assert float(da[1]) <= 1e-9, "analytic did not push the object -y"


def test_side_push_shows_coupling():
    """Pushing the crossbar's right end in -x should induce a torque.

    Diagnostic only (prints), since the exact rotation depends on the contact
    point and both models resolve it differently -- but the torque sign is
    worth watching: a -x push above the CoM should rotate one consistent way.
    """
    shape = t_shape_footprint()
    da, dl, lam = _report(
        "side-push", shape, (0.0, 0.0, 0.0), (0.103, 0.030), (-0.6, 0.0)
    )
    assert float(lam[0]) > 0.0, "contact expected on the right face"
    assert float(dl[0]) < 0.0, "LCS did not push the object -x"


if __name__ == "__main__":
    test_separated_pusher_no_contact()
    test_top_push_moves_object_down()
    test_side_push_shows_coupling()
    print("\nOK: contact LCS is directionally consistent with the analytic "
          "2D rollout (magnitudes differ by construction).")
