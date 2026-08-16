"""Contact-implicit MPC (C3+ style) baseline, as one integrated module.

Mirrors the reference C3+ algorithm ("Push Anything") but runs on an analytic
Linear Complementarity System (LCS) we build for planar pushing, instead of on
Drake's autodiff of a MultibodyPlant.

Contents:
  * LCS                      -- the linear complementarity system container.
  * solve_lcp / lcs_step / lcs_rollout -- forward simulation of an LCS.
  * build_planar_pushing_lcs -- object-only, wrench-driven LCS (validation).
  * build_contact_lcs        -- single-point pusher contact LCS (the real one).
  * project_complementarity  -- closed-form projection onto {a>=0, b>=0, ab=0}.
  * c3_solve                 -- the C3+ ADMM solver (KKT z-step + projections +
                                duals), with optional input box constraints.
  * C3                       -- the receding-horizon controller for the 2D task.

LCS convention (per step k):
    x_{k+1} = A x_k + B u_k + G lam_k + d
    0 <= lam_k  _|_  E x_k + F lam_k + H u_k + c >= 0
The lam-to-state map is named G (not D) to avoid clashing with the
limit-surface compliance D used in the pushing physics.
"""

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field

from oim.objects.contact import contact_force_to_com_wrench
from oim.objects.planar_pushing import wrap_angle
from oim.objects.sdf import rotate


# =====================================================================
# LCS container and forward simulation
# =====================================================================


@dataclass
class LCS:
    """A time-invariant Linear Complementarity System (see module docstring)."""

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
    """Solve 0 <= lam _|_ (M lam + q) >= 0 by projected Jacobi iteration.

    Good enough to validate the LCS forward dynamics; the C3 solver below does
    not use it (it handles complementarity by ADMM projection instead).
    """
    diag = jnp.diag(M)
    inv_diag = jnp.where(jnp.abs(diag) > 1e-12, 1.0 / diag, 0.0)

    def _sweep(lam: jax.Array, _: jax.Array) -> Tuple[jax.Array, None]:
        residual = M @ lam + q
        return jnp.maximum(0.0, lam - inv_diag * residual), None

    lam, _ = jax.lax.scan(_sweep, jnp.zeros_like(q), None, length=iters)
    return lam


def lcs_step(
    lcs: LCS, x: jax.Array, u: jax.Array, wrap_theta_index: int = 2
) -> Tuple[jax.Array, jax.Array]:
    """Advance the LCS one step: solve for lam, then apply the state update."""
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
    """Roll the LCS forward under a control sequence; returns (H+1, n) states."""

    def _body(x: jax.Array, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        x_next, _ = lcs_step(lcs, x, u, wrap_theta_index)
        return x_next, x_next

    _, xs = jax.lax.scan(_body, x0, controls)
    return jnp.concatenate([x0[None, :], xs], axis=0)


# =====================================================================
# LCS builders
# =====================================================================


def build_planar_pushing_lcs(wrench_limit: jax.Array, dt: float) -> LCS:
    """Object-only, wrench-driven planar-pushing LCS (validation testbed).

    Per axis the object moves by the force exceeding the friction limit (a box
    limit surface): dx_i = dt * D_i * (relu(w_i - fl_i) - relu(-w_i - fl_i)).
    State x = pose (3), control u = wrench (3), lam = +/- excess per axis (6).
    """
    fl = jnp.asarray(wrench_limit, dtype=float)
    D = 1.0 / fl
    n, m, k = 3, 3, 6

    A = jnp.eye(n)
    B = jnp.zeros((n, m))
    d = jnp.zeros(n)
    G = jnp.zeros((n, k))
    for i in range(n):
        G = G.at[i, 2 * i].set(dt * D[i])
        G = G.at[i, 2 * i + 1].set(-dt * D[i])

    E = jnp.zeros((k, n))
    F = jnp.eye(k)
    H = jnp.zeros((k, m))
    c = jnp.zeros(k)
    for i in range(n):
        H = H.at[2 * i, i].set(-1.0)
        H = H.at[2 * i + 1, i].set(1.0)
        c = c.at[2 * i].set(fl[i])
        c = c.at[2 * i + 1].set(fl[i])

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=n, m=m, k=k)


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

    State x = [obj_x, obj_y, obj_theta, ee_x, ee_y] (n=5), control u = pusher
    velocity (m=2), lam = contact normal force (k=1). Friction (when mu_c > 0)
    uses a fixed slide direction.
    """
    D = jnp.asarray(limit_surface_d, dtype=float)
    theta = object_pose[2]

    q = rotate(-theta, pusher_pos - object_pose[:2])
    dist, grad = shape.sdf_and_grad(q)
    gap0 = dist - robot_radius
    n_body = -grad
    t_body = jnp.stack([-n_body[1], n_body[0]])
    n_world = rotate(theta, n_body)
    t_world = rotate(theta, t_body)
    n_out = -n_world

    contact_body = q - gap0 * grad
    p_world = object_pose[:2] + rotate(theta, contact_body)
    r = p_world - object_pose[:2]

    a_hat = n_world - mu_c * slide_sign * t_world
    W = contact_force_to_com_wrench(object_pose, p_world, a_hat)  # (3,)
    gvec = dt * D * W
    dcontact = jnp.array(
        [gvec[0] - gvec[2] * r[1], gvec[1] + gvec[2] * r[0]]
    )

    n, m, k = 5, 2, 1
    A = jnp.eye(n)
    B = jnp.zeros((n, m))
    B = B.at[3, 0].set(dt)
    B = B.at[4, 1].set(dt)
    G = jnp.zeros((n, k))
    G = G.at[0:3, 0].set(gvec)
    d = jnp.zeros(n)

    E = jnp.zeros((k, n))
    F = jnp.array([[-jnp.dot(n_out, dcontact)]])
    H = jnp.zeros((k, m))
    H = H.at[0, 0].set(dt * n_out[0])
    H = H.at[0, 1].set(dt * n_out[1])
    c = jnp.array([gap0])

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=n, m=m, k=k)


# =====================================================================
# C3+ ADMM solver
# =====================================================================


def project_complementarity(
    a: jax.Array, b: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """Project each pair (a_i, b_i) onto {a>=0, b>=0, a*b=0}, elementwise."""
    dist_to_a_axis = jnp.where(a < 0, a**2, 0.0) + b**2
    dist_to_b_axis = a**2 + jnp.where(b < 0, b**2, 0.0)
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
    admm_iters: int = 40,
    reg: float = 1e-6,
    u_min: Optional[jax.Array] = None,
    u_max: Optional[jax.Array] = None,
    rho_u: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Solve the contact-implicit trajectory optimization with C3+ ADMM.

    The smooth QP (dynamics + slack + cost) is solved as a KKT linear system;
    the complementarity is enforced by a closed-form projection each iteration;
    an optional input box [u_min, u_max] is enforced by a SECOND ADMM
    consensus (clip projection + dual), so no external QP solver is needed.

    Returns states (N+1, n), controls (N, m), contact vars (N, kd).
    """
    N = horizon
    n, m, kd = lcs.n, lcs.m, lcs.k
    bounded = u_min is not None
    bs = n + m + 2 * kd
    Z = N * bs + n

    def xk(k):
        return k * bs

    def uk(k):
        return k * bs + n

    def lk(k):
        return k * bs + n + m

    def ek(k):
        return k * bs + n + m + kd

    xN = N * bs

    # --- Equality constraints: initial + dynamics + slack.
    n_rows = n + N * n + N * kd
    C = jnp.zeros((n_rows, Z))
    bvec = jnp.zeros((n_rows,))
    row = 0
    C = C.at[row:row + n, xk(0):xk(0) + n].set(jnp.eye(n))
    bvec = bvec.at[row:row + n].set(x_init)
    row += n
    for k in range(N):
        nxt = xN if k == N - 1 else xk(k + 1)
        C = C.at[row:row + n, nxt:nxt + n].set(jnp.eye(n))
        C = C.at[row:row + n, xk(k):xk(k) + n].set(-lcs.A)
        C = C.at[row:row + n, uk(k):uk(k) + m].set(-lcs.B)
        C = C.at[row:row + n, lk(k):lk(k) + kd].set(-lcs.G)
        bvec = bvec.at[row:row + n].set(lcs.d)
        row += n
    for k in range(N):
        C = C.at[row:row + kd, ek(k):ek(k) + kd].set(jnp.eye(kd))
        C = C.at[row:row + kd, xk(k):xk(k) + n].set(-lcs.E)
        C = C.at[row:row + kd, lk(k):lk(k) + kd].set(-lcs.F)
        C = C.at[row:row + kd, uk(k):uk(k) + m].set(-lcs.H)
        bvec = bvec.at[row:row + kd].set(lcs.c)
        row += kd

    # --- QP Hessian. rho on (lam, eta) copies; rho_u on u copies if bounded.
    P = jnp.zeros((Z, Z))
    u_block = R + (rho_u * jnp.eye(m) if bounded else 0.0 * jnp.eye(m))
    for k in range(N):
        P = P.at[xk(k):xk(k) + n, xk(k):xk(k) + n].set(Q)
        P = P.at[uk(k):uk(k) + m, uk(k):uk(k) + m].set(u_block)
        P = P.at[lk(k):lk(k) + kd, lk(k):lk(k) + kd].set(rho * jnp.eye(kd))
        P = P.at[ek(k):ek(k) + kd, ek(k):ek(k) + kd].set(rho * jnp.eye(kd))
    P = P.at[xN:xN + n, xN:xN + n].set(Qf)
    P = P + reg * jnp.eye(Z)

    kkt = jnp.block([[P, C.T], [C, jnp.zeros((n_rows, n_rows))]])

    def stack(z, idx_fn, dim):
        return jnp.stack([z[idx_fn(k):idx_fn(k) + dim] for k in range(N)])

    def build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u):
        q = jnp.zeros((Z,))
        for k in range(N):
            q = q.at[xk(k):xk(k) + n].set(-Q @ x_ref)
            q = q.at[lk(k):lk(k) + kd].set(rho * (-lam_hat[k] + w_lam[k]))
            q = q.at[ek(k):ek(k) + kd].set(rho * (-eta_hat[k] + w_eta[k]))
            if bounded:
                q = q.at[uk(k):uk(k) + m].set(rho_u * (-u_hat[k] + w_u[k]))
        q = q.at[xN:xN + n].set(-Qf @ x_ref)
        return q

    lam_hat = jnp.zeros((N, kd))
    eta_hat = jnp.zeros((N, kd))
    w_lam = jnp.zeros((N, kd))
    w_eta = jnp.zeros((N, kd))
    u_hat = jnp.zeros((N, m))
    w_u = jnp.zeros((N, m))

    z = jnp.zeros((Z,))
    for _ in range(admm_iters):
        q = build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u)
        z = jnp.linalg.solve(kkt, jnp.concatenate([-q, bvec]))[:Z]

        lam = stack(z, lk, kd)
        eta = stack(z, ek, kd)
        lam_hat, eta_hat = project_complementarity(lam + w_lam, eta + w_eta)
        w_lam = w_lam + lam - lam_hat
        w_eta = w_eta + eta - eta_hat

        if bounded:
            us_it = stack(z, uk, m)
            u_hat = jnp.clip(us_it + w_u, u_min, u_max)
            w_u = w_u + us_it - u_hat

    xs = jnp.stack([z[xk(k):xk(k) + n] for k in range(N)] + [z[xN:xN + n]])
    us = stack(z, uk, m)
    lams = stack(z, lk, kd)
    if bounded:
        us = jnp.clip(us, u_min, u_max)
    return xs, us, lams


# =====================================================================
# C3 controller (2D task)
# =====================================================================


@dataclass
class C3ControllerParams:
    """Warm-startable policy state: last plan and the time it was computed."""

    us: jax.Array
    t0: jax.Array


def _state_cost_hessian(
    q_pos: float, q_theta: float, w_ee: float
) -> jax.Array:
    """Hessian of q_pos||obj-goal||^2 + q_theta*dtheta^2 + w_ee||ee-obj||^2.

    In the 0.5 (x - x_ref)' Q (x - x_ref) convention, with x_ref's ee entries
    set to the object goal xy so the approach cross-terms reduce to (ee-obj)^2.
    """
    Q = jnp.zeros((5, 5))
    Q = Q.at[0, 0].add(2.0 * q_pos)
    Q = Q.at[1, 1].add(2.0 * q_pos)
    Q = Q.at[2, 2].add(2.0 * q_theta)
    for o, e in ((0, 3), (1, 4)):
        Q = Q.at[o, o].add(2.0 * w_ee)
        Q = Q.at[e, e].add(2.0 * w_ee)
        Q = Q.at[o, e].add(-2.0 * w_ee)
        Q = Q.at[e, o].add(-2.0 * w_ee)
    return Q


class C3:
    """Contact-implicit MPC (C3+ style) controller for the 2D push task."""

    def __init__(
        self,
        task,
        rho: float = 0.1,
        horizon: int = 10,
        admm_iters: int = 40,
        q_pos: float = 1000.0,
        q_theta: float = 100.0,
        w_ee: float = 400.0,
        qf_pos: float = 10000.0,
        qf_theta: float = 1000.0,
        r_r: float = 0.05,
        mu_c: float = 0.0,
        rho_u: float = 1.0,
    ) -> None:
        """Configure the controller against a `PushT2D` task."""
        self.task = task
        self.dt = float(task.dt)
        self.rho = rho
        self.rho_u = rho_u
        self.horizon = horizon
        self.admm_iters = admm_iters
        self.mu_c = mu_c

        self.shape = task.footprint
        self.D = task.model.limit_surface_d
        self.robot_radius = task.model.robot_radius

        g = task.goal
        self.x_ref = jnp.array([g[0], g[1], g[2], g[0], g[1]])
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0)
        self.R = r_r * jnp.eye(2)

        self.u_min = task.u_min
        self.u_max = task.u_max

    def init_params(self, seed: int = 0) -> C3ControllerParams:
        del seed
        return C3ControllerParams(
            us=jnp.zeros((self.horizon, 2)), t0=jnp.asarray(0.0)
        )

    def optimize(self, state, params):
        object_pose = state.object_pose
        pusher_pos = state.robot_pos
        lcs = build_contact_lcs(
            self.shape, self.D, self.robot_radius,
            object_pose, pusher_pos, self.dt,
            mu_c=self.mu_c, slide_sign=0.0,
        )
        x_init = jnp.concatenate([object_pose, pusher_pos])
        _, us, _ = c3_solve(
            lcs, x_init, self.x_ref, self.Q, self.R, self.Qf,
            rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
            u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
        )
        return params.replace(us=us, t0=state.time), us

    def get_action(self, params, t) -> jax.Array:
        idx = jnp.clip(
            jnp.floor((t - params.t0) / self.dt).astype(jnp.int32),
            0, self.horizon - 1,
        )
        return params.us[idx]
