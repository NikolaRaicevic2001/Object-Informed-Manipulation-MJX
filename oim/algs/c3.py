"""Contact-implicit MPC (C3+ style) baseline, built as one integrated module.

This mirrors the reference algorithm from "Push Anything: Single- and
Multi-Object Pushing From First Sight with Contact-Implicit MPC" (C3+), but
runs on an analytic Linear Complementarity System (LCS) we build ourselves for
planar pushing, rather than on Drake's autodiff of a MultibodyPlant.

The module is built incrementally. This is INCREMENT 1: the LCS data structure,
a builder for single-object planar pushing, a small LCP solver, and a forward
rollout. The C3+ three-step ADMM solver (QP z-step + closed-form projection +
dual) and the `C3(SamplingBasedController)` controller wrapper land in later
increments, once the LCS itself is validated against the known-good analytic
model (`oim.objects.planar_pushing.PlanarPushingObject`).

LCS convention (per timestep k):

    x_{k+1} = A x_k + B u_k + G lam_k + d
    0 <= lam_k  _|_  E x_k + F lam_k + H u_k + c >= 0

NOTE ON NAMING: the coefficient of `lam` in the state update is written `G`
here, NOT `D`. The letter D is already the limit-surface compliance in
`PlanarPushingObject`, and reusing it inside the LCS would be a constant source
of confusion. So: `G` = the LCS lambda-to-state map; the limit-surface
compliance stays `D` wherever the pushing physics is discussed.
"""

from typing import Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from oim.objects.planar_pushing import wrap_angle


@dataclass
class LCS:
    """A time-invariant Linear Complementarity System.

    Shapes, with n = state dim, m = control dim, k = complementarity dim:

        A: (n, n)   B: (n, m)   G: (n, k)   d: (n,)
        E: (k, n)   F: (k, k)   H: (k, m)   c: (k,)

    `n`, `m`, `k` are static (they set array shapes), so they are marked as
    non-pytree fields and can be read at trace time.
    """

    A: jax.Array
    B: jax.Array
    G: jax.Array
    d: jax.Array
    E: jax.Array
    F: jax.Array
    H: jax.Array
    c: jax.Array
    n: int = field(pytree_node=False)
    m: int = field(pytree_node=False)
    k: int = field(pytree_node=False)


def solve_lcp(M: jax.Array, q: jax.Array, iters: int = 60) -> jax.Array:
    """Solve the LCP  0 <= lam _|_ (M lam + q) >= 0  by projected iteration.

    Uses a projected Jacobi sweep, which converges for the P-matrix LCPs that
    the planar-pushing LCS produces. This is a placeholder solver good enough
    to *validate the LCS*; the C3+ increment replaces the whole solve with the
    QP + closed-form-projection ADMM loop, which is both faster and the point
    of the baseline.

    Args:
        M: The LCP matrix, (k, k).
        q: The LCP offset, (k,).
        iters: Number of projected sweeps.

    Returns:
        The complementarity variable lam, (k,).
    """
    diag = jnp.diag(M)
    # Guard against a zero on the diagonal (a decoupled, always-inactive row).
    inv_diag = jnp.where(jnp.abs(diag) > 1e-12, 1.0 / diag, 0.0)

    def _sweep(lam: jax.Array, _: jax.Array) -> Tuple[jax.Array, None]:
        residual = M @ lam + q
        lam = jnp.maximum(0.0, lam - inv_diag * residual)
        return lam, None

    lam, _ = jax.lax.scan(_sweep, jnp.zeros_like(q), None, length=iters)
    return lam


def lcs_step(
    lcs: LCS, x: jax.Array, u: jax.Array, wrap_theta_index: int = 2
) -> Tuple[jax.Array, jax.Array]:
    """Advance the LCS one step: solve for lam, then apply the state update.

    Args:
        lcs: The system.
        x: Current state, (n,).
        u: Current control, (m,).
        wrap_theta_index: Index of an SE(2) heading in the state to wrap to
            (-pi, pi], or a negative value to skip wrapping.

    Returns:
        The next state x_{k+1}, (n,), and the solved lam, (k,).
    """
    # For a time-invariant LCS the LCP offset is q = E x + H u + c.
    q = lcs.E @ x + lcs.H @ u + lcs.c
    lam = solve_lcp(lcs.F, q)
    x_next = lcs.A @ x + lcs.B @ u + lcs.G @ lam + lcs.d
    if wrap_theta_index >= 0:
        x_next = x_next.at[wrap_theta_index].set(
            wrap_angle(x_next[wrap_theta_index])
        )
    return x_next, lam


def lcs_rollout(
    lcs: LCS, x0: jax.Array, controls: jax.Array, wrap_theta_index: int = 2
) -> jax.Array:
    """Roll the LCS forward under a control sequence.

    Args:
        lcs: The system.
        x0: Initial state, (n,).
        controls: Control sequence, (H, m).
        wrap_theta_index: See `lcs_step`.

    Returns:
        States x_0 .. x_H, shape (H + 1, n).
    """

    def _body(x: jax.Array, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x_next, _ = lcs_step(lcs, x, u, wrap_theta_index)
        return x_next, x_next

    _, xs = jax.lax.scan(_body, x0, controls)
    return jnp.concatenate([x0[None, :], xs], axis=0)


def build_planar_pushing_lcs(wrench_limit: jax.Array, dt: float) -> LCS:
    """Build the object-only, wrench-driven planar-pushing LCS (INCREMENT 1).

    Physical model. The object slides quasi-statically on its support surface.
    Along each axis i of the world-frame wrench w = [f_x, f_y, tau], the object
    moves only by the force that EXCEEDS the friction limit `wrench_limit[i]`
    -- a per-axis (box) friction cone:

        dx_i = dt * D_i * ( relu(w_i - fl_i) - relu(-w_i - fl_i) ),
        D_i  = 1 / fl_i,   fl = wrench_limit.

    This is the polyhedral counterpart of `PlanarPushingObject.step`'s smooth
    ellipsoidal limit surface. The two AGREE EXACTLY for axis-aligned wrenches
    (the ellipse and the box coincide on each axis) and differ only for
    off-axis wrenches, which is the property the unit test checks.

    LCS encoding. State x = [p_x, p_y, theta] (n = 3), control u = w (m = 3).
    Two complementarity variables per axis, for the positive and negative
    excess:  lam = [lp_x, ln_x, lp_y, ln_y, lp_theta, ln_theta] (k = 6), with

        lp_i = relu(w_i - fl_i),   ln_i = relu(-w_i - fl_i).

    Each is a scalar relu, which is the LCP  0 <= lam _|_ (lam - a) >= 0  with
    a = (+/- w_i - fl_i). So F = I_6, and the offset rows carry -/+ w_i + fl_i.
    The state update maps the net excess (lp_i - ln_i) through dt * D_i.

    Args:
        wrench_limit: Friction-cone limit [fl_x, fl_y, fl_theta], (3,). This is
            exactly `PlanarPushingObject.wrench_limit`.
        dt: Planning timestep, matching the object model's dt.

    Returns:
        The assembled `LCS`.
    """
    fl = jnp.asarray(wrench_limit, dtype=float)
    D = 1.0 / fl  # limit-surface compliance, per axis
    n, m, k = 3, 3, 6

    # State update: x_{k+1} = x_k + G lam  (no direct A-offdiag / B / d terms;
    # all motion is carried by the excess forces in lam).
    A = jnp.eye(n)
    B = jnp.zeros((n, m))
    d = jnp.zeros(n)

    # G maps [lp_x, ln_x, lp_y, ln_y, lp_th, ln_th] -> dt * D_i * (lp_i - ln_i).
    G = jnp.zeros((n, k))
    for i in range(n):
        G = G.at[i, 2 * i].set(dt * D[i])  # + excess drives +motion
        G = G.at[i, 2 * i + 1].set(-dt * D[i])  # - excess drives -motion

    # Complementarity: 0 <= lam _|_ (F lam + H u + c) >= 0, with F = I.
    #   row lp_i:  lp_i - (w_i - fl_i) = lp_i - w_i + fl_i   >= 0
    #   row ln_i:  ln_i - (-w_i - fl_i) = ln_i + w_i + fl_i  >= 0
    E = jnp.zeros((k, n))
    F = jnp.eye(k)
    H = jnp.zeros((k, m))
    c = jnp.zeros(k)
    for i in range(n):
        H = H.at[2 * i, i].set(-1.0)  # -w_i on the positive-excess row
        H = H.at[2 * i + 1, i].set(1.0)  # +w_i on the negative-excess row
        c = c.at[2 * i].set(fl[i])
        c = c.at[2 * i + 1].set(fl[i])

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=n, m=m, k=k)
