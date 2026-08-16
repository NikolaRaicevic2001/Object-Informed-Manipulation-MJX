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

# =====================================================================
# INCREMENT 2a: single-point pusher contact LCS.
#
# Append this to oim/algs/c3.py. It adds the end-effector (pusher) to the
# state and derives the contact normal force as the LCS complementarity
# variable, so the object is driven by a *contact* wrench (with force-torque
# coupling) rather than by a wrench handed in as the control.
#
# State   x = [obj_x, obj_y, obj_theta, ee_x, ee_y]     (n = 5)
# Control u = [ee_vx, ee_vy]                              (m = 2)
# Compl.  lam = [f_n]  (contact normal force >= 0)        (k = 1)
#
# Everything is linearized once at the current (object_pose, pusher_pos), the
# way C3 relinearizes its LCS each control step. Friction is optional and, when
# on, uses a fixed slide direction (the sign of the current tangential motion),
# matching resolve_contact's Coulomb law f_t = -mu_c f_n sign(v_t).
# =====================================================================

from oim.objects.sdf import rotate  # noqa: E402
from oim.objects.contact import contact_force_to_com_wrench  # noqa: E402


def build_contact_lcs(
    shape,
    limit_surface_d: jax.Array,
    robot_radius: float,
    object_pose: jax.Array,
    pusher_pos: jax.Array,
    dt: float,
    mu_c: float = 0.0,
    slide_sign: float = 0.0,
) -> LCS:
    """Linearize the single-point pusher-object contact into an LCS.

    Args:
        shape: The object footprint (an `oim.objects.sdf.Shape`), body frame.
        limit_surface_d: Limit-surface compliance D = [D_x, D_y, D_theta], (3,),
            i.e. `Sim2DModel.limit_surface_d` / `PlanarPushingObject.D`.
        robot_radius: Pusher disc radius (0 for a point pusher).
        object_pose: Linearization pose [x, y, theta], (3,).
        pusher_pos: Linearization pusher world position [x, y], (2,).
        dt: Planning timestep.
        mu_c: Coulomb friction coefficient at the pusher-object interface.
        slide_sign: Sign (+1/0/-1) of the current tangential sliding, fixing
            the friction direction for this linearization.

    Returns:
        The assembled `LCS` with n=5, m=2, k=1.
    """
    D = jnp.asarray(limit_surface_d, dtype=float)
    theta = object_pose[2]

    # --- Contact geometry at the linearization point (mirrors resolve_contact).
    q = rotate(-theta, pusher_pos - object_pose[:2])  # pusher in body frame
    dist, grad = shape.sdf_and_grad(q)
    gap0 = dist - robot_radius
    n_body = -grad  # inward: pusher presses in
    t_body = jnp.stack([-n_body[1], n_body[0]])
    n_world = rotate(theta, n_body)
    t_world = rotate(theta, t_body)
    n_out = -n_world  # outward: +gap direction

    contact_body = q - gap0 * grad
    p_world = object_pose[:2] + rotate(theta, contact_body)
    r = p_world - object_pose[:2]  # lever arm about CoM

    # --- World contact force per unit normal force, with fixed-direction friction.
    a_hat = n_world - mu_c * slide_sign * t_world
    W = contact_force_to_com_wrench(object_pose, p_world,
                                    a_hat)  # (3,): [fx,fy,tau]

    # Object pose delta per unit normal force, through the limit surface.
    gvec = dt * D * W  # (3,)

    # Contact-point displacement per unit normal force (rigid-body kinematics):
    #   d(contact) = [gvec_x - gvec_theta * r_y,  gvec_y + gvec_theta * r_x]
    dcontact = jnp.array([gvec[0] - gvec[2] * r[1], gvec[1] + gvec[2] * r[0]])

    n, m, k = 5, 2, 1

    # --- State update. Object holds pose and is moved only by f_n through gvec;
    # the pusher integrates its velocity command.
    A = jnp.eye(n)
    B = jnp.zeros((n, m))
    B = B.at[3, 0].set(dt)
    B = B.at[4, 1].set(dt)
    G = jnp.zeros((n, k))
    G = G.at[0:3, 0].set(gvec)
    d = jnp.zeros(n)

    # --- Complementarity: 0 <= f_n _|_ gap_{k+1} >= 0, with
    #   gap_{k+1} = gap0 + dt * (n_out . u) + f_n * (-(n_out . dcontact)).
    # The f_n coefficient is positive (more force -> object recedes -> gap grows),
    # so the LCP picks the f_n that just removes the predicted penetration.
    E = jnp.zeros((k, n))
    F = jnp.array([[-jnp.dot(n_out, dcontact)]])
    H = jnp.zeros((k, m))
    H = H.at[0, 0].set(dt * n_out[0])
    H = H.at[0, 1].set(dt * n_out[1])
    c = jnp.array([gap0])

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=n, m=m, k=k)

# =====================================================================
# INCREMENT 2b: the C3+ three-step ADMM solver.
#
# Append to oim/algs/c3.py. Solves the contact-implicit trajectory optimization
#
#   min  sum_k [ 0.5 (x_k - x*)' Q (x_k - x*) + 0.5 u_k' R u_k ]
#              + 0.5 (x_N - x*)' Qf (x_N - x*)
#   s.t. x_{k+1} = A x_k + B u_k + G lam_k + d          (dynamics)
#        eta_k   = E x_k + F lam_k + H u_k + c           (slack, C3+ style)
#        0 <= lam_k  _|_  eta_k >= 0                     (complementarity)
#
# via ADMM, splitting the smooth QP (dynamics + slack + cost) from the
# per-timestep complementarity, which becomes a closed-form projection.
#
#   z-step:      solve the equality-constrained QP (a KKT linear solve),
#                with (lam, eta) pulled toward the current consensus copy.
#   projection:  snap each (lam, eta) onto {a>=0, b>=0, a*b=0}, closed form.
#   dual update: accumulate the disagreement.
# =====================================================================


def project_complementarity(
    a: jax.Array, b: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Project each pair (a_i, b_i) onto {a>=0, b>=0, a*b=0}, elementwise.

    The complementarity set is the union of the two non-negative axes. The
    nearest point either zeroes b and clamps a >= 0, or zeroes a and clamps
    b >= 0; pick whichever is closer. For two positive inputs this keeps the
    larger coordinate and zeroes the smaller. This is C3+'s per-contact
    closed-form projection.

    Args:
        a: First member of each pair (e.g. contact force lam), any shape.
        b: Second member (e.g. slack eta), same shape.

    Returns:
        The projected (a, b), same shapes.
    """
    dist_to_a_axis = jnp.where(a < 0, a**2, 0.0) + b**2  # project onto b = 0
    dist_to_b_axis = a**2 + jnp.where(b < 0, b**2, 0.0)  # project onto a = 0
    use_a_axis = dist_to_a_axis <= dist_to_b_axis
    a_proj = jnp.where(use_a_axis, jnp.maximum(a, 0.0), 0.0)
    b_proj = jnp.where(use_a_axis, 0.0, jnp.maximum(b, 0.0))
    return a_proj, b_proj


def c3_solve(
    lcs: LCS,
    x_init: jax.Array,
    x_ref: jax.Array,
    Q: jax.Array,
    R: jax.Array,
    Qf: jax.Array,
    rho: float = 1.0,
    horizon: int = 10,
    admm_iters: int = 20,
    reg: float = 1e-6,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Solve the contact-implicit trajectory optimization with C3+ ADMM.

    Args:
        lcs: The linearized system (constant over the horizon here).
        x_init: Initial state, (n,).
        x_ref: Target state tracked at every stage and the terminal, (n,).
        Q: Stage state cost, (n, n).
        R: Control cost, (m, m), positive definite.
        Qf: Terminal state cost, (n, n).
        rho: ADMM penalty weight.
        horizon: Number of control steps N (static).
        admm_iters: ADMM iterations (static).
        reg: Tikhonov regularization added to the QP Hessian.

    Returns:
        xs: States x_0 .. x_N, (N + 1, n).
        us: Controls u_0 .. u_{N-1}, (N, m).
        lams: Contact variables lam_0 .. lam_{N-1}, (N, kd).
    """
    N = horizon
    n, m, kd = lcs.n, lcs.m, lcs.k
    bs = n + m + 2 * kd  # per-step block: [x, u, lam, eta]
    Z = N * bs + n  # + terminal state x_N

    # Block offsets within the stacked decision vector z.
    def xk(k: int) -> int:
        return k * bs

    def uk(k: int) -> int:
        return k * bs + n

    def lk(k: int) -> int:
        return k * bs + n + m

    def ek(k: int) -> int:
        return k * bs + n + m + kd

    xN = N * bs

    # --- Equality constraints C z = bvec: initial + dynamics + slack.
    n_rows = n + N * n + N * kd
    C = jnp.zeros((n_rows, Z))
    bvec = jnp.zeros((n_rows,))

    row = 0
    # x_0 = x_init
    C = C.at[row:row + n, xk(0):xk(0) + n].set(jnp.eye(n))
    bvec = bvec.at[row:row + n].set(x_init)
    row += n
    # x_{k+1} - A x_k - B u_k - G lam_k = d
    for k in range(N):
        nxt = xN if k == N - 1 else xk(k + 1)
        C = C.at[row:row + n, nxt:nxt + n].set(jnp.eye(n))
        C = C.at[row:row + n, xk(k):xk(k) + n].set(-lcs.A)
        C = C.at[row:row + n, uk(k):uk(k) + m].set(-lcs.B)
        C = C.at[row:row + n, lk(k):lk(k) + kd].set(-lcs.G)
        bvec = bvec.at[row:row + n].set(lcs.d)
        row += n
    # eta_k - E x_k - F lam_k - H u_k = c
    for k in range(N):
        C = C.at[row:row + kd, ek(k):ek(k) + kd].set(jnp.eye(kd))
        C = C.at[row:row + kd, xk(k):xk(k) + n].set(-lcs.E)
        C = C.at[row:row + kd, lk(k):lk(k) + kd].set(-lcs.F)
        C = C.at[row:row + kd, uk(k):uk(k) + m].set(-lcs.H)
        bvec = bvec.at[row:row + kd].set(lcs.c)
        row += kd

    # --- QP Hessian P (constant). Q, R on x, u; rho*I on the (lam, eta) copies.
    P = jnp.zeros((Z, Z))
    for k in range(N):
        P = P.at[xk(k):xk(k) + n, xk(k):xk(k) + n].set(Q)
        P = P.at[uk(k):uk(k) + m, uk(k):uk(k) + m].set(R)
        P = P.at[lk(k):lk(k) + kd, lk(k):lk(k) + kd].set(rho * jnp.eye(kd))
        P = P.at[ek(k):ek(k) + kd, ek(k):ek(k) + kd].set(rho * jnp.eye(kd))
    P = P.at[xN:xN + n, xN:xN + n].set(Qf)
    P = P + reg * jnp.eye(Z)

    # KKT left-hand side [[P, C'], [C, 0]] is constant; only the RHS moves.
    kkt = jnp.block([[P, C.T], [C, jnp.zeros((n_rows, n_rows))]])

    def build_linear_term(
        lam_hat: jax.Array,
        eta_hat: jax.Array,
        w_lam: jax.Array,
        w_eta: jax.Array,
    ) -> jax.Array:
        """The gradient term q of 0.5 z'P z + q'z, with the ADMM penalty."""
        q = jnp.zeros((Z,))
        for k in range(N):
            q = q.at[xk(k):xk(k) + n].set(-Q @ x_ref)
            q = q.at[lk(k):lk(k) + kd].set(rho * (-lam_hat[k] + w_lam[k]))
            q = q.at[ek(k):ek(k) + kd].set(rho * (-eta_hat[k] + w_eta[k]))
        q = q.at[xN:xN + n].set(-Qf @ x_ref)
        return q

    def stack(z: jax.Array, idx_fn, dim: int) -> jax.Array:
        return jnp.stack([z[idx_fn(k):idx_fn(k) + dim] for k in range(N)])

    lam_hat = jnp.zeros((N, kd))
    eta_hat = jnp.zeros((N, kd))
    w_lam = jnp.zeros((N, kd))
    w_eta = jnp.zeros((N, kd))

    z = jnp.zeros((Z,))
    for _ in range(admm_iters):
        q = build_linear_term(lam_hat, eta_hat, w_lam, w_eta)
        rhs = jnp.concatenate([-q, bvec])
        z = jnp.linalg.solve(kkt, rhs)[:Z]

        lam = stack(z, lk, kd)
        eta = stack(z, ek, kd)
        lam_hat, eta_hat = project_complementarity(lam + w_lam, eta + w_eta)
        w_lam = w_lam + lam - lam_hat
        w_eta = w_eta + eta - eta_hat

    xs = jnp.stack([z[xk(k):xk(k) + n] for k in range(N)] + [z[xN:xN + n]])
    us = stack(z, uk, m)
    lams = stack(z, lk, kd)
    return xs, us, lams

# =====================================================================
# INCREMENT 2b (final): the C3 controller wrapper for the 2D world.
#
# Append to oim/algs/c3.py. Wraps c3_solve into the optimize / get_action /
# init_params interface that oim.worlds.sim2d.run drives, relinearizing the
# contact LCS at the current (object_pose, pusher_pos) every control step
# (receding horizon).
#
# State  x = [obj_x, obj_y, obj_theta, ee_x, ee_y]   control u = [ee_vx, ee_vy].
# The stage cost tracks the object to the goal AND keeps the pusher on the
# object (an approach term), both pure quadratics in x, so they live in one Q.
# =====================================================================


@dataclass
class C3ControllerParams:
    """Warm-startable policy state for the C3 controller.

    Attributes:
        us: The last planned control sequence, (H, m).
        t0: The time the plan was computed (for the receding-horizon index).
    """

    us: jax.Array
    t0: jax.Array


def _state_cost_hessian(
    q_pos: float, q_theta: float, w_ee: float
) -> jax.Array:
    """Hessian of the stage state cost in the 0.5 (x - x_ref)' Q (x - x_ref) form.

    Encodes  q_pos * ||obj_xy - goal_xy||^2 + q_theta * (obj_theta - goal_theta)^2
             + w_ee * ||ee_xy - obj_xy||^2,
    the last term being the approach cost that keeps the pusher on the object.
    With x_ref = [gx, gy, gtheta, gx, gy] the approach cross-terms reproduce
    (ee - obj)^2 exactly (the shared reference cancels). Factor 2 because the
    solver's objective carries the 0.5.

    Args:
        q_pos: Object translational tracking weight.
        q_theta: Object rotational tracking weight.
        w_ee: Approach weight pulling the pusher onto the object.

    Returns:
        The (5, 5) cost Hessian.
    """
    Q = jnp.zeros((5, 5))
    Q = Q.at[0, 0].add(2.0 * q_pos)
    Q = Q.at[1, 1].add(2.0 * q_pos)
    Q = Q.at[2, 2].add(2.0 * q_theta)
    for o, e in ((0, 3), (1, 4)):  # (obj_x, ee_x) and (obj_y, ee_y)
        Q = Q.at[o, o].add(2.0 * w_ee)
        Q = Q.at[e, e].add(2.0 * w_ee)
        Q = Q.at[o, e].add(-2.0 * w_ee)
        Q = Q.at[e, o].add(-2.0 * w_ee)
    return Q


class C3:
    """Contact-implicit MPC (C3+ style) controller for the 2D push task.

    Standalone (like the way `ADMM` sidesteps the MJX-specific base __init__):
    it exposes exactly the `init_params` / `optimize` / `get_action` the 2D
    runner calls, and plans by relinearizing the contact LCS each step and
    running the C3+ ADMM solver.
    """

    def __init__(
        self,
        task,
        rho: float = 0.1,
        horizon: int = 10,
        admm_iters: int = 25,
        q_pos: float = 1000.0,
        q_theta: float = 100.0,
        w_ee: float = 400.0,
        qf_pos: float = 10000.0,
        qf_theta: float = 1000.0,
        r_r: float = 0.05,
        mu_c: float = 0.0,
    ) -> None:
        """Configure the controller against a `PushT2D` task.

        Args:
            task: The 2D task (supplies geometry, limit surface, goal, dt).
            rho: ADMM penalty weight (0.1 from the solver sweep).
            horizon: Planning horizon N in steps of task.dt.
            admm_iters: ADMM iterations per solve (fewer than the offline
                sweep, since only u_0 is applied and we replan each step).
            q_pos, q_theta: Object goal-tracking weights (stage).
            w_ee: Approach weight keeping the pusher on the object.
            qf_pos, qf_theta: Terminal object-tracking weights.
            r_r: Control-effort weight on the pusher velocity.
            mu_c: Coulomb friction used when linearizing the contact
                (0 = frictionless linearization for now).
        """
        self.task = task
        self.dt = float(task.dt)
        self.rho = rho
        self.horizon = horizon
        self.admm_iters = admm_iters
        self.mu_c = mu_c

        # Geometry / physics read off the task.
        self.shape = task.footprint
        self.D = task.model.limit_surface_d
        self.robot_radius = task.model.robot_radius

        # Cost matrices (constant). EE reference = object goal xy, so the
        # approach cross-terms reduce to (ee - obj)^2.
        g = task.goal
        self.x_ref = jnp.array([g[0], g[1], g[2], g[0], g[1]])
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0)  # no approach term
        self.R = r_r * jnp.eye(2)

        # Attributes the runner may read off a controller.
        self.u_min = task.u_min
        self.u_max = task.u_max

    def init_params(self, seed: int = 0) -> C3ControllerParams:
        """Zero plan, time zero."""
        del seed
        return C3ControllerParams(
            us=jnp.zeros((self.horizon, 2)), t0=jnp.asarray(0.0)
        )

    def optimize(self, state, params):
        """Relinearize at the current state and solve one C3 trajectory opt."""
        object_pose = state.object_pose
        pusher_pos = state.robot_pos

        lcs = build_contact_lcs(
            self.shape,
            self.D,
            self.robot_radius,
            object_pose,
            pusher_pos,
            self.dt,
            mu_c=self.mu_c,
            slide_sign=0.0,
        )
        x_init = jnp.concatenate([object_pose, pusher_pos])
        _, us, _ = c3_solve(
            lcs,
            x_init,
            self.x_ref,
            self.Q,
            self.R,
            self.Qf,
            rho=self.rho,
            horizon=self.horizon,
            admm_iters=self.admm_iters,
        )
        return params.replace(us=us, t0=state.time), us

    def get_action(self, params, t) -> jax.Array:
        """Receding-horizon query: the plan's control for the current time."""
        idx = jnp.clip(
            jnp.floor((t - params.t0) / self.dt).astype(jnp.int32),
            0,
            self.horizon - 1,
        )
        return params.us[idx]
