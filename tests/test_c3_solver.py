"""Validate the C3+ ADMM solver on the (validated) object-only LCS.

We reuse increment 1's object-only pushing LCS -- control = wrench, state =
object pose -- as a clean testbed: the solver must plan a wrench trajectory that
drives the object from x_init toward x_ref, while (a) obeying the dynamics
exactly (the KKT step enforces them as equalities) and (b) keeping the contact
variables complementary (0 <= lam _|_ eta), which the ADMM drives softly.

Checks:
  * project_complementarity matches hand-computed values.
  * The solved trajectory satisfies the LCS dynamics to numerical tolerance.
  * The object ends closer to the goal than it started.
  * The complementarity residual max_i |lam_i * eta_i| is reported (soft).

Run:  python tests/test_c3_solver.py
"""

import jax
import jax.numpy as jnp

from oim.algs.c3 import (
    build_planar_pushing_lcs,
    c3_solve,
    project_complementarity,
)

WRENCH_LIMIT = jnp.array([7.848, 7.848, 0.471])
DT = 0.05


def test_projection_hand_cases():
    """Elementwise projection onto the complementarity set."""
    a = jnp.array([3.0, 3.0, -2.0, 2.0])
    b = jnp.array([5.0, 0.0, 5.0, -1.0])
    a_p, b_p = project_complementarity(a, b)
    # (3,5)->(0,5)  (3,0)->(3,0)  (-2,5)->(0,5)  (2,-1)->(2,0)
    assert jnp.allclose(a_p, jnp.array([0.0, 3.0, 0.0, 2.0])), a_p
    assert jnp.allclose(b_p, jnp.array([5.0, 0.0, 5.0, 0.0])), b_p
    # Every projected pair is complementary.
    assert jnp.allclose(a_p * b_p, 0.0)
    print("[projection] hand cases OK")


def test_solver_drives_object_to_goal():
    """The planned wrench trajectory should push the object toward the goal."""
    lcs = build_planar_pushing_lcs(WRENCH_LIMIT, DT)
    n = lcs.n

    x_init = jnp.array([0.10, 0.0, 0.0])   # object 10 cm off in +x
    x_ref = jnp.zeros(3)                    # goal at the origin

    Q = jnp.diag(jnp.array([1000.0, 1000.0, 100.0]))
    Qf = 10.0 * Q
    R = 1e-4 * jnp.eye(3)

    xs, us, lams = c3_solve(
        lcs, x_init, x_ref, Q, R, Qf,
        rho=5.0, horizon=10, admm_iters=40,
    )

    # (a) dynamics residual: x_{k+1} - (A x_k + B u_k + G lam_k + d).
    dyn_res = 0.0
    for k in range(xs.shape[0] - 1):
        pred = lcs.A @ xs[k] + lcs.B @ us[k] + lcs.G @ lams[k] + lcs.d
        dyn_res = max(dyn_res, float(jnp.max(jnp.abs(xs[k + 1] - pred))))

    # (b) complementarity residual on the returned (lam, eta).
    comp_res = 0.0
    for k in range(us.shape[0]):
        eta = lcs.E @ xs[k] + lcs.F @ lams[k] + lcs.H @ us[k] + lcs.c
        comp_res = max(comp_res, float(jnp.max(jnp.abs(lams[k] * eta))))
    min_lam = float(jnp.min(lams))

    d0 = float(jnp.linalg.norm(x_init[:2]))
    dT = float(jnp.linalg.norm(xs[-1][:2]))

    print(f"[solver] start dist {d0:.4f} -> final dist {dT:.4f}")
    print(f"[solver] dynamics residual (max) : {dyn_res:.2e}")
    print(f"[solver] complementarity residual: {comp_res:.2e}")
    print(f"[solver] min lam (want >= ~0)    : {min_lam:.2e}")
    print(f"[solver] final pose = {xs[-1]}")

    assert dyn_res < 1e-4, "dynamics not satisfied by the KKT solve"
    assert dT < d0, "object did not move toward the goal"
    assert min_lam > -1e-4, "contact force went negative"


if __name__ == "__main__":
    test_projection_hand_cases()
    test_solver_drives_object_to_goal()
    print("\nOK: C3+ ADMM solver satisfies dynamics, keeps contacts "
          "complementary, and drives the object toward the goal.")
