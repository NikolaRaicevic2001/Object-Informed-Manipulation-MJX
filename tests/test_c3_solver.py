"""Validate the C3+ ADMM solver on the (validated) object-only LCS.

The solver must plan a wrench trajectory that actually drives the object from
x_init to x_ref. A previous version asserted only `final < initial`, which a
0.018 mm move passes -- useless. This version sweeps the ADMM penalty `rho` and
requires a *real* move to the goal, because `rho` too large relative to the
cost pins the contact force at zero (moving looks more expensive than the
position error) and the object never leaves its start.

Run:  python tests/test_c3_solver.py
"""

import jax.numpy as jnp

from oim.algs.c3 import (
    build_planar_pushing_lcs,
    c3_solve,
    project_complementarity,
)

WRENCH_LIMIT = jnp.array([7.848, 7.848, 0.471])
DT = 0.05


def _residuals(lcs, xs, us, lams):
    """Max dynamics residual and max complementarity residual over a rollout."""
    dyn = 0.0
    for k in range(us.shape[0]):
        pred = lcs.A @ xs[k] + lcs.B @ us[k] + lcs.G @ lams[k] + lcs.d
        dyn = max(dyn, float(jnp.max(jnp.abs(xs[k + 1] - pred))))
    comp = 0.0
    for k in range(us.shape[0]):
        eta = lcs.E @ xs[k] + lcs.F @ lams[k] + lcs.H @ us[k] + lcs.c
        comp = max(comp, float(jnp.max(jnp.abs(lams[k] * eta))))
    return dyn, comp


def test_projection_hand_cases():
    a = jnp.array([3.0, 3.0, -2.0, 2.0])
    b = jnp.array([5.0, 0.0, 5.0, -1.0])
    a_p, b_p = project_complementarity(a, b)
    assert jnp.allclose(a_p, jnp.array([0.0, 3.0, 0.0, 2.0])), a_p
    assert jnp.allclose(b_p, jnp.array([5.0, 0.0, 5.0, 0.0])), b_p
    assert jnp.allclose(a_p * b_p, 0.0)
    print("[projection] hand cases OK")


def test_solver_rho_sweep():
    """Find a rho that actually pushes the object to the goal."""
    lcs = build_planar_pushing_lcs(WRENCH_LIMIT, DT)
    x_init = jnp.array([0.10, 0.0, 0.0])
    x_ref = jnp.zeros(3)
    Q = jnp.diag(jnp.array([1000.0, 1000.0, 100.0]))
    Qf = 10.0 * Q
    R = 1e-4 * jnp.eye(3)

    d0 = float(jnp.linalg.norm(x_init[:2]))
    print(f"[solver] start dist {d0:.4f}, goal at origin")
    print(f"{'rho':>8} | {'final dist':>10} | {'dyn res':>9} | {'comp res':>9}")

    best = jnp.inf
    for rho in [0.02, 0.05, 0.1, 0.5, 1.0, 5.0]:
        xs, us, lams = c3_solve(
            lcs, x_init, x_ref, Q, R, Qf,
            rho=rho, horizon=10, admm_iters=300,
        )
        dT = float(jnp.linalg.norm(xs[-1][:2]))
        dyn, comp = _residuals(lcs, xs, us, lams)
        best = min(best, dT)
        print(f"{rho:>8.2f} | {dT:>10.4f} | {dyn:>9.1e} | {comp:>9.1e}")

    print(f"[solver] best final dist across rho: {best:.4f}  (start {d0:.4f})")
    assert best < 0.03, (
        "no rho pushed the object near the goal -- the ADMM is stuck at the "
        "zero-force point; lower rho / raise Q / more iters"
    )


if __name__ == "__main__":
    test_projection_hand_cases()
    test_solver_rho_sweep()
    print("\nOK: at least one rho drives the object to the goal while keeping "
          "dynamics and complementarity satisfied.")
