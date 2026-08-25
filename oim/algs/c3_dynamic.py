"""Contact-implicit MPC (C3+ / "Push Anything") baseline -- dynamic planar LCS.

Model 2: the object is the sim3d plant's own 3-DOF planar body (T_x, T_y slides
+ T_z hinge) and the pusher is the 2-DOF point robot's velocity servo. The
object-ground friction is the MJCF `frictionloss` (set-valued Coulomb, fixed
bound), and the single pusher-object contact is linearized with the ANITESCU
convex contact model -- the exact contact_model dairlib's push_t / anything
C3+ uses (sampling_c3plus_options.yaml: contact_model: 'anitescu',
num_friction_directions: 2, N: 10, admm_iter: 3, rho_scale: 3).

Why this and not a literal 3D (SE(3)) LCS: C3+'s LCS dimensionality follows the
plant. dairlib push_t is 3D only because its object can tip; the sim3d object
is joint-constrained to the plane, so the model-error-free (= strongest) C3+
model is planar. This keeps the baseline on the *same* plant both methods run.

State  x = [ox, oy, oth, ex, ey,  vox, voy, voth, vex, vey]  (n = 10)
Input  u = [ux, uy]  (pusher velocity servo target, m = 2)
Lambda lam = [ lam1_p, lam2_p,                       # pusher Anitescu cone edges
               g_gx, gx+, gx-,                       # ground box friction, x
               g_gy, gy+, gy-,                        #                       y
               g_gth, gth+, gth- ]  (k = 11)          #                       theta

LCS convention (per step):
    x_{k+1} = A x + B u + G lam + d
    0 <= lam  _|_  E x + F lam + H u + c >= 0
G is the lam->state map (not the limit-surface D).
"""

from typing import Optional

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from mujoco import mjx

from oim.alg_base import SamplingBasedController, Trajectory
from oim.objects.planar_pushing import wrap_angle
from oim.objects.sdf import rotate


# =====================================================================
# LCS container
# =====================================================================
@dataclass
class LCS:
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


# =====================================================================
# Plant parameters. Defaults mirror the point-robot MJCF; build_lcs_from_mjx
# (next step) will read M, frictionloss and kv straight from the MJX model so
# the LCS is provably the same plant the run executes in.
# =====================================================================
@dataclass
class PlantParams:
    mo: float = field(pytree_node=False)      # object mass (T_x, T_y)
    Io: float = field(pytree_node=False)      # object z-inertia (T_z)
    me: float = field(pytree_node=False)      # pusher mass
    kv: float = field(pytree_node=False)      # velocity-servo gain
    bv: float = field(pytree_node=False)      # pusher joint damping
    mu_p: float = field(pytree_node=False)    # pusher-object friction
    bx: float = field(pytree_node=False)      # ground Coulomb bound, x
    by: float = field(pytree_node=False)      # ground Coulomb bound, y
    bth: float = field(pytree_node=False)     # ground Coulomb bound, theta
    dt: float = field(pytree_node=False)


def default_plant(dt=0.05):
    """From tee.xml / open_table_point.xml / common_point.xml."""
    return PlantParams(
        mo=2.0, Io=0.005, me=1.0, kv=100.0, bv=0.05, mu_p=0.5,
        bx=7.848, by=7.848, bth=0.47088, dt=dt,
    )


def _M_diag(pp):
    return jnp.array([pp.mo, pp.mo, pp.Io, pp.me, pp.me])


def _default_Minv(pp):
    """Diagonal M^-1 from the scalar masses (used when no full M is given)."""
    return jnp.diag(1.0 / _M_diag(pp))


def _ground_bounds(pp):
    return jnp.array([pp.bx, pp.by, pp.bth])


def _smooth_discrete(pp):
    """Implicit (stable) discrete contact-free velocity map: v_next = A_v v + B_v u.

    The pusher is a stiff velocity servo (force = kv*(u - v_e) - bv*v_e).
    MuJoCo integrates it with `implicitfast`, so integrating it explicitly here
    would be unstable (kv*dt/me can far exceed 2). The implicit update is
        v_e_next = (v_e + dt*kv*u/me) / (1 + dt*(kv+bv)/me),
    stable for any gain. The object velocity has no smooth force (ground
    friction is a contact), so it persists -- that is the object's momentum.
    """
    dt = pp.dt
    denom = 1.0 + dt * (pp.kv + pp.bv) / pp.me
    alpha = 1.0 / denom
    beta = (dt * pp.kv / pp.me) * alpha
    A_v = jnp.diag(jnp.array([1.0, 1.0, 1.0, alpha, alpha]))
    B_v = jnp.zeros((5, 2))
    B_v = B_v.at[3, 0].set(beta)
    B_v = B_v.at[4, 1].set(beta)
    return A_v, B_v


# =====================================================================
# Contact geometry.
#   contact_fn(q) -> (phi, n_world(2,), r_world(2,))
# n points from the object surface toward the pusher; r is the lever arm from
# the object COM to the contact point. The T-shape uses shape.sdf_and_grad; a
# disc is provided for standalone validation.
# =====================================================================
def make_shape_contact(shape, robot_radius):
    """Contact function for an oim footprint `shape` (body-frame SDF)."""
    def contact_fn(q):
        ox, oy, oth = q[0], q[1], q[2]
        p_ee = q[3:5]
        p_body = rotate(-oth, p_ee - jnp.array([ox, oy]))
        dist, grad = shape.sdf_and_grad(p_body)
        phi = dist - robot_radius
        n_body = grad                     # outward surface normal (toward ee)
        n_world = rotate(oth, n_body)
        contact_body = p_body - dist * grad
        p_world = jnp.array([ox, oy]) + rotate(oth, contact_body)
        r = p_world - jnp.array([ox, oy])
        return phi, n_world, r
    return contact_fn


def make_disc_contact(Ro, re):
    """Contact function for a disc object of radius Ro (validation only)."""
    def contact_fn(q):
        d = jnp.array([q[3] - q[0], q[4] - q[1]])
        dist = jnp.linalg.norm(d) + 1e-12
        n = d / dist
        phi = dist - (Ro + re)
        r = Ro * n
        return phi, n, r
    return contact_fn


def _pusher_jac(n, r):
    """Normal / tangent contact Jacobian rows (d(gap-rate)/dv), planar."""
    nx, ny = n
    tx, ty = -ny, nx
    rxn = r[0] * ny - r[1] * nx
    rxt = r[0] * ty - r[1] * tx
    Jn = jnp.array([-nx, -ny, -rxn, nx, ny])
    Jt = jnp.array([-tx, -ty, -rxt, tx, ty])
    return Jn, Jt


# =====================================================================
# LCS builder: Anitescu pusher contact + Coulomb box ground friction.
# =====================================================================
def build_dynamic_lcs(pp, contact_fn, x0, u0, Minv=None, obs=None):
    """Anitescu pusher + Coulomb box ground friction, plus optional frictionless
    object-obstacle normal contacts, linearized at x0.

    obs: None, or (phi (No,), n (No,2), r (No,2)) -- object-obstacle contacts,
    n = world push-away normal, r = lever arm from object COM. Object DOFs only
    (the obstacle is static), one frictionless normal variable each.
    """
    dt = pp.dt
    Minv = _default_Minv(pp) if Minv is None else Minv
    A_v, B_v = _smooth_discrete(pp)
    q0 = x0[:5]
    phi, n, r = contact_fn(q0)
    Jn, Jt = _pusher_jac(n, r)
    b = _ground_bounds(pp)

    if obs is not None and int(obs[0].shape[0]) > 0:
        o_phi, o_n, o_r = obs
        No = int(o_phi.shape[0])
        rxn = o_r[:, 0] * o_n[:, 1] - o_r[:, 1] * o_n[:, 0]              # (No,)
        Jo = jnp.stack([o_n[:, 0], o_n[:, 1], rxn,
                        jnp.zeros(No), jnp.zeros(No)], axis=1)           # (No,5)
    else:
        No = 0
        o_phi = jnp.zeros(0)
        Jo = jnp.zeros((0, 5))
    kd = 11 + No

    A = jnp.zeros((10, 10))
    A = A.at[:5, :5].set(jnp.eye(5))
    A = A.at[:5, 5:].set(dt * A_v)
    A = A.at[5:, 5:].set(A_v)
    B = jnp.zeros((10, 2))
    B = B.at[:5, :].set(dt * B_v)
    B = B.at[5:, :].set(B_v)
    d = jnp.zeros(10)

    d1 = Jn + pp.mu_p * Jt
    d2 = Jn - pp.mu_p * Jt
    e0 = jnp.array([1.0, 0, 0, 0, 0])
    e1 = jnp.array([0, 1.0, 0, 0, 0])
    e2 = jnp.array([0, 0, 1.0, 0, 0])

    def col(J):
        return dt * (Minv @ J)

    zero = jnp.zeros(5)
    base_cols = [col(d1), col(d2),
                 zero, col(e0), -col(e0),
                 zero, col(e1), -col(e1),
                 zero, col(e2), -col(e2)]
    obs_cols = [col(Jo[o]) for o in range(No)]
    Gv = jnp.stack(base_cols + obs_cols, axis=1)                        # (5, kd)

    G = jnp.zeros((10, kd))
    G = G.at[:5, :].set(dt * Gv)
    G = G.at[5:, :].set(Gv)

    Vfree_x = jnp.concatenate([jnp.zeros((5, 5)), A_v], axis=1)
    Vfree_u = B_v

    E = jnp.zeros((kd, 10))
    F = jnp.zeros((kd, kd))
    H = jnp.zeros((kd, 2))
    c = jnp.zeros(kd)

    def vel_row(Jrow):
        return (Jrow @ Vfree_x, Jrow @ Vfree_u, Jrow @ Gv)

    for i, dj in ((0, d1), (1, d2)):
        ex, hu, fl = vel_row(dj)
        E = E.at[i, :].set(ex); H = H.at[i, :].set(hu)
        F = F.at[i, :].set(fl); c = c.at[i].set(phi / dt)

    def ground(base, e_axis, bound):
        nonlocal E, F, H, c
        g, pl, mn = base, base + 1, base + 2
        F = F.at[g, pl].set(-1.0); F = F.at[g, mn].set(-1.0)
        c = c.at[g].set(bound)
        ex, hu, fl = vel_row(e_axis)
        F = F.at[pl, g].set(1.0); F = F.at[pl, :].add(fl)
        E = E.at[pl, :].add(ex); H = H.at[pl, :].add(hu)
        ex, hu, fl = vel_row(-e_axis)
        F = F.at[mn, g].set(1.0); F = F.at[mn, :].add(fl)
        E = E.at[mn, :].add(ex); H = H.at[mn, :].add(hu)

    ground(2, e0, b[0]); ground(5, e1, b[1]); ground(8, e2, b[2])

    # object-obstacle rows: 0 <= lam _|_ (Jo . v_next + phi/dt)
    for o in range(No):
        ex, hu, fl = vel_row(Jo[o])
        E = E.at[11 + o, :].set(ex); H = H.at[11 + o, :].set(hu)
        F = F.at[11 + o, :].set(fl); c = c.at[11 + o].set(o_phi[o] / dt)

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=10, m=2, k=kd)


# =====================================================================
# Faithful forward simulator (P4 "simulate the plan" cost / validation).
# The Anitescu pusher LCP is PSD -> projected Gauss-Seidel; the ground box
# friction has a fixed bound and a diagonal mass block -> per-DOF clamp. A
# short splitting iteration couples them. This replaces the projected-Jacobi
# solve_lcp, which cannot move the zero-diagonal friction-cone variables.
# =====================================================================
def _pgs_psd(W, w, iters=60):
    diag = jnp.diag(W)
    inv = 1.0 / jnp.where(diag > 1e-12, diag, 1.0)
    ncol = w.shape[0]

    def sweep(z, _):
        def body(i, z):
            r = w[i] + W[i] @ z
            return z.at[i].set(jnp.maximum(0.0, z[i] - inv[i] * r))
        return jax.lax.fori_loop(0, ncol, body, z), None

    z, _ = jax.lax.scan(sweep, jnp.zeros_like(w), None, length=iters)
    return z


def simulate_step(pp, contact_fn, x, u, splits=8, Minv=None, obs_fn=None):
    dt = pp.dt
    Minv = _default_Minv(pp) if Minv is None else Minv
    M_obj = jnp.linalg.inv(Minv)[:3, :3]
    q, v = x[:5], x[5:]
    A_v, B_v = _smooth_discrete(pp)
    v_free = A_v @ v + B_v @ u
    phi, n, r = contact_fn(q)
    Jn, Jt = _pusher_jac(n, r)
    rows = [Jn + pp.mu_p * Jt, Jn - pp.mu_p * Jt]
    cvec = [phi / dt, phi / dt]
    if obs_fn is not None:
        o_phi, o_n, o_r = obs_fn(q)
        for o in range(int(o_phi.shape[0])):
            rxn = o_r[o, 0] * o_n[o, 1] - o_r[o, 1] * o_n[o, 0]
            rows.append(jnp.array([o_n[o, 0], o_n[o, 1], rxn, 0.0, 0.0]))
            cvec.append(o_phi[o] / dt)
    Jc = jnp.stack(rows)                    # (nc,5)
    cvec = jnp.stack(cvec)                   # (nc,)
    Fp = dt * (Jc @ (Minv @ Jc.T))
    b = _ground_bounds(pp)

    def pad(vg):
        return dt * (Minv @ jnp.concatenate([vg, jnp.zeros(2)]))

    def body(v_ground, _):
        v_pre = v_free + pad(v_ground)
        lam = _pgs_psd(Fp, Jc @ v_pre + cvec)
        v_after = v_pre + dt * (Minv @ (Jc.T @ lam))
        f_arrest = -(M_obj @ v_after[:3]) / dt
        return jnp.clip(f_arrest, -b, b), None

    v_ground, _ = jax.lax.scan(body, jnp.zeros(3), None, length=splits)
    v_pre = v_free + pad(v_ground)
    lam = _pgs_psd(Fp, Jc @ v_pre + cvec)
    v_next = v_pre + dt * (Minv @ (Jc.T @ lam))
    q_next = q + dt * v_next
    q_next = q_next.at[2].set(wrap_angle(q_next[2]))
    return jnp.concatenate([q_next, v_next])


def simulate_rollout(pp, contact_fn, x0, us, Minv=None, obs_fn=None):
    def step(x, u):
        xn = simulate_step(pp, contact_fn, x, u, Minv=Minv, obs_fn=obs_fn)
        return xn, xn
    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None], xs], axis=0)


# =====================================================================
# C3+ ADMM solver (unchanged: KKT z-step + complementarity projection +
# input-box projection + growing-rho). Handles the zero-diagonal friction-cone
# rows fine because complementarity is done by PROJECTION, not matrix inverse.
# =====================================================================
def project_complementarity(a, b):
    dist_a = jnp.where(a < 0, a**2, 0.0) + b**2
    dist_b = a**2 + jnp.where(b < 0, b**2, 0.0)
    use_a = dist_a <= dist_b
    return (
        jnp.where(use_a, jnp.maximum(a, 0.0), 0.0),
        jnp.where(use_a, 0.0, jnp.maximum(b, 0.0)),
    )


def c3_solve(
    lcs, x_init, x_ref, Q, R, Qf,
    rho=0.1, horizon=10, admm_iters=3, reg=1e-6,
    u_min=None, u_max=None, rho_u=1.0, rho_scale=3.0,
):
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
    zeros_mm = jnp.zeros((n_rows, n_rows))

    def make_P(rho_l, rho_u_l):
        P = jnp.zeros((Z, Z))
        u_blk = R + (rho_u_l * jnp.eye(m) if bounded else jnp.zeros((m, m)))
        for k in range(N):
            P = P.at[xk(k):xk(k) + n, xk(k):xk(k) + n].set(Q)
            P = P.at[uk(k):uk(k) + m, uk(k):uk(k) + m].set(u_blk)
            P = P.at[lk(k):lk(k) + kd, lk(k):lk(k) + kd].set(rho_l * jnp.eye(kd))
            P = P.at[ek(k):ek(k) + kd, ek(k):ek(k) + kd].set(rho_l * jnp.eye(kd))
        P = P.at[xN:xN + n, xN:xN + n].set(Qf)
        return P + reg * jnp.eye(Z)

    def stack(z, idx_fn, dim):
        return jnp.stack([z[idx_fn(k):idx_fn(k) + dim] for k in range(N)])

    def build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u, rho_l, rho_u_l):
        q = jnp.zeros((Z,))
        for k in range(N):
            q = q.at[xk(k):xk(k) + n].set(-Q @ x_ref)
            q = q.at[lk(k):lk(k) + kd].set(rho_l * (-lam_hat[k] + w_lam[k]))
            q = q.at[ek(k):ek(k) + kd].set(rho_l * (-eta_hat[k] + w_eta[k]))
            if bounded:
                q = q.at[uk(k):uk(k) + m].set(rho_u_l * (-u_hat[k] + w_u[k]))
        q = q.at[xN:xN + n].set(-Qf @ x_ref)
        return q

    def update(z, lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u):
        lam = stack(z, lk, kd)
        eta = stack(z, ek, kd)
        lam_hat, eta_hat = project_complementarity(lam + w_lam, eta + w_eta)
        w_lam = w_lam + lam - lam_hat
        w_eta = w_eta + eta - eta_hat
        if bounded:
            us_it = stack(z, uk, m)
            u_hat = jnp.clip(us_it + w_u, u_min, u_max)
            w_u = w_u + us_it - u_hat
        return lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u

    lam_hat = jnp.zeros((N, kd))
    eta_hat = jnp.zeros((N, kd))
    w_lam = jnp.zeros((N, kd))
    w_eta = jnp.zeros((N, kd))
    u_hat = jnp.zeros((N, m))
    w_u = jnp.zeros((N, m))
    z = jnp.zeros((Z,))

    if rho_scale == 1.0:
        kkt = jnp.block([[make_P(rho, rho_u), C.T], [C, zeros_mm]])
        for _ in range(admm_iters):
            q = build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u, rho, rho_u)
            z = jnp.linalg.solve(kkt, jnp.concatenate([-q, bvec]))[:Z]
            lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u = update(
                z, lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u)
    else:
        for it in range(admm_iters):
            rho_l = rho * (rho_scale ** it)
            rho_u_l = rho_u * (rho_scale ** it)
            kkt = jnp.block([[make_P(rho_l, rho_u_l), C.T], [C, zeros_mm]])
            q = build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u, rho_l, rho_u_l)
            z = jnp.linalg.solve(kkt, jnp.concatenate([-q, bvec]))[:Z]
            lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u = update(
                z, lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u)

    xs = jnp.stack([z[xk(k):xk(k) + n] for k in range(N)] + [z[xN:xN + n]])
    us = stack(z, uk, m)
    lams = stack(z, lk, kd)
    if bounded:
        us = jnp.clip(us, u_min, u_max)
    return xs, us, lams


# =====================================================================
# Cost Hessian on the 10-dim state x = [q(5); v(5)].
#   object pose error (q_pos on x,y; q_theta on theta),
#   EE-tracking coupling (w_ee, pull tip toward object as in the ref approach
#   cost -- here just a light tip regularizer), velocity regularizer w_v.
# =====================================================================
def _state_cost_hessian(q_pos, q_theta, w_ee, w_v):
    Q = jnp.zeros((10, 10))
    Q = Q.at[0, 0].add(2.0 * q_pos)
    Q = Q.at[1, 1].add(2.0 * q_pos)
    Q = Q.at[2, 2].add(2.0 * q_theta)
    for o, e in ((0, 3), (1, 4)):
        Q = Q.at[o, o].add(2.0 * w_ee)
        Q = Q.at[e, e].add(2.0 * w_ee)
        Q = Q.at[o, e].add(-2.0 * w_ee)
        Q = Q.at[e, o].add(-2.0 * w_ee)
    for i in range(5, 10):
        Q = Q.at[i, i].add(2.0 * w_v)
    return Q


# ---------------------------------------------------------------------
# Minimal single-mode C3+ controller (translation/rotation core). The
# sampling outer loop (P1-P5) wraps this next, once the LCS is confirmed on
# the T-shape in the repo.
# ---------------------------------------------------------------------
@dataclass
class C3Params:
    us: jax.Array
    q_prev: jax.Array
    t0: jax.Array


class C3Dynamic:
    """C3+ MPC on the dynamic planar LCS. Faithful defaults from
    dairlib push_t sampling_c3plus_options.yaml."""

    def __init__(
        self, contact_fn, goal, u_min, u_max, plant,
        horizon=10, admm_iters=3, rho=0.1, rho_scale=3.0, rho_u=1.0,
        q_pos=200.0, q_theta=40.0, w_ee=10.0, w_v=0.05,
        qf_pos=2000.0, qf_theta=400.0, r_r=0.05, Minv=None,
    ):
        self.contact_fn = contact_fn
        self.plant = plant
        self.Minv = Minv
        self.dt = plant.dt
        self.horizon, self.admm_iters = horizon, admm_iters
        self.rho, self.rho_scale, self.rho_u = rho, rho_scale, rho_u
        g = jnp.asarray(goal, dtype=float)
        self.goal = g
        self.x_ref = jnp.array(
            [g[0], g[1], g[2], g[0], g[1], 0, 0, 0, 0, 0], dtype=float)
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee, w_v)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0, w_v)
        self.R = r_r * jnp.eye(2)
        self.u_min, self.u_max = u_min, u_max

    def init_params(self, q0):
        return C3Params(us=jnp.zeros((self.horizon, 2)),
                        q_prev=jnp.asarray(q0, dtype=float),
                        t0=jnp.asarray(0.0))

    def optimize(self, q, params, t=0.0):
        v = (q - params.q_prev) / self.dt
        x_init = jnp.concatenate([q, v])
        lcs = build_dynamic_lcs(self.plant, self.contact_fn, x_init,
                                params.us[0], Minv=self.Minv)
        _, us, _ = c3_solve(
            lcs, x_init, self.x_ref, self.Q, self.R, self.Qf,
            rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
            u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
            rho_scale=self.rho_scale)
        return params.replace(us=us, q_prev=q, t0=t), us

    def get_action(self, params, t):
        idx = jnp.clip(jnp.floor((t - params.t0) / self.dt).astype(jnp.int32),
                       0, self.horizon - 1)
        return params.us[idx]


# =====================================================================
# MJX adapter: C3+ as a drop-in flat baseline, run by the SAME path as
# mppi/cem/ps/cbo (oim.worlds.sim3d.run.run_3d_plain via build_flat_3d).
#
# Subclasses SamplingBasedController and overrides `optimize` wholesale (the
# ADMM pattern): the C3+ plan is stashed in `params.mean` as the spline knots,
# so the inherited zero-order-hold `interp_func`, `get_action`, `init_params`
# and `nominal_trace` all work unchanged and run_3d_plain needs no C3 branch.
#
# The LCS is built from the MJX plant itself: only the 5x5 inverse mass matrix
# Minv is configuration-dependent (recomputed from mjx.full_m each step); every
# other constant (kv, damping, ground friction bounds, mu) is read once at
# construction. Requires robot="point" (Push Anything's simple end-effector):
#   object SE(2) = qpos[block_dofs];  EE xy = xpos[pusher_body_id].


class C3MJX(SamplingBasedController):
    """C3+ (dynamic planar LCS) as a flat baseline on the sim3d MJX plant."""

    def __init__(
        self, task, *, plan_horizon, num_knots, seed=0,
        robot_radius=0.02, mu_p=0.5, kv=100.0,
        admm_iters=3, rho=0.1, rho_scale=3.0, rho_u=1.0,
        q_pos=200.0, q_theta=40.0, w_ee=10.0, w_v=0.05,
        qf_pos=2000.0, qf_theta=400.0, r_r=0.05,
        # Accepted-and-ignored so build_sub_optimizer-style kwargs are safe:
        spline_type="zero", iterations=1, num_samples=None, **_ignored,
    ):
        super().__init__(
            task, num_randomizations=1, risk_strategy=None, seed=seed,
            plan_horizon=plan_horizon, spline_type="zero",
            num_knots=num_knots, iterations=1)
        if task.model.nu != 2:
            raise ValueError("C3MJX targets robot='point' (nu=2)")

        import numpy as np
        self.horizon = num_knots
        self.admm_iters = admm_iters
        self.rho, self.rho_scale, self.rho_u = rho, rho_scale, rho_u
        self.block_dofs = jnp.asarray(task.block_dofs)
        self.pusher_dofs = jnp.asarray(task.pusher_dofs)
        self.pusher_bid = int(task.pusher_body_id)
        self.idx5 = jnp.concatenate([self.block_dofs, self.pusher_dofs])

        m = task.model
        # EE world xy from LIVE qpos (run_3d_plain updates qpos but not xpos):
        #   ee = pusher body's declared pos + its slide-joint displacement.
        self.pusher_offset = jnp.asarray(
            np.asarray(m.body_pos)[self.pusher_bid][:2])
        # The object's nominal ground-friction budget, read from the
        # analytic model rather than from `dof_frictionloss`: the tabletop
        # scenes now take their support friction from the table CONTACT
        # (mu*N, so it rises when the block is pressed -- see tee.xml) and
        # their joints carry none, so reading the joints there would hand
        # this LCS a frictionless ground. Same triple those joints used to
        # hold, and still correct for the scenes that keep them.
        #
        # The LCS bound is constant either way, so it models the nominal
        # load only and cannot represent that press-down coupling.
        fl = np.asarray(task.object_model.wrench_limit)
        bv = float(np.asarray(m.dof_damping)[int(self.pusher_dofs[0])])
        me = float(np.asarray(m.body_mass)[self.pusher_bid])
        self.plant = PlantParams(
            mo=2.0, Io=0.005, me=me, kv=kv, bv=bv, mu_p=mu_p,
            bx=float(fl[0]), by=float(fl[1]), bth=float(fl[2]),
            dt=float(task.dt))
        self.contact_fn = make_shape_contact(
            task.object_model.footprint, robot_radius)

        g = jnp.asarray(task.goal, dtype=float)
        self.x_ref = jnp.array(
            [g[0], g[1], g[2], g[0], g[1], 0, 0, 0, 0, 0], dtype=float)
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee, w_v)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0, w_v)
        self.R = r_r * jnp.eye(2)
        self.u_min, self.u_max = task.u_min, task.u_max
        sites = getattr(task, "trace_site_ids", None)
        self._n_sites = int(sites.shape[0]) if sites is not None else 1

    def _state_from_data(self, data):
        q = data.qpos[self.block_dofs]                # object SE(2)
        ee = self.pusher_offset + data.qpos[self.pusher_dofs]  # live, not xpos
        v = jnp.concatenate(
            [data.qvel[self.block_dofs], data.qvel[self.pusher_dofs]])
        return jnp.concatenate([q, ee, v])

    def optimize(self, state, params):
        new_tk = jnp.linspace(
            0.0, self.plan_horizon, self.num_knots) + state.time
        M = mjx.full_m(self.model, state)             # (nv, nv)
        Minv = jnp.linalg.inv(M[self.idx5][:, self.idx5])
        x0 = self._state_from_data(state)
        lcs = build_dynamic_lcs(
            self.plant, self.contact_fn, x0, params.mean[0], Minv=Minv)
        _, us, _ = c3_solve(
            lcs, x0, self.x_ref, self.Q, self.R, self.Qf,
            rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
            u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
            rho_scale=self.rho_scale)
        params = params.replace(tk=new_tk, mean=us)
        H = self.ctrl_steps
        dummy = Trajectory(
            controls=jnp.zeros((1, H, 2)), knots=us[None],
            costs=jnp.zeros((1, H + 1)),
            trace_sites=jnp.zeros((1, H + 1, self._n_sites, 3)))
        return params, dummy

    # optimize is overridden wholesale; these only satisfy the ABC.
    def sample_knots(self, params):
        return params.mean[None, ...], params

    def update_params(self, params, rollouts):
        return params


# =====================================================================
# Sampling / repositioning outer loop (Push Anything, P1-P4). The local C3
# solver alone cannot choose or reach a contact face, so on any non-trivial
# task the EE glues to the nearest point and stalls. This wraps it with the
# dairlib sampling_based_c3_controller logic:
#   P1 goal-met stop; P2 config-cost progress cutoff; P3 sticky asymmetric
#   hysteresis (c3->repos 0.8, repos->repos 0.9, repos->c3 0.5); P4 rank
#   candidates by the plan simulated through the faithful LCS stepper.
# Ported from the 2D-validated C3Sampling, adapted to the 10-dim dynamic LCS.
# =====================================================================
@dataclass
class C3SampState:
    is_c3: jax.Array          # 1.0 = pushing (C3), 0.0 = repositioning
    target: jax.Array         # current repositioning target (world EE xy)
    target_body: jax.Array    # target contact in body frame (for the unsucc buffer)
    cost_hist: jax.Array      # (W,) object config-cost history
    n_prog: jax.Array         # steps since last progress reset
    unsucc: jax.Array         # (U, 2) body-frame contacts that made no progress
    good_buf: jax.Array       # (G, 2) body-frame contacts that made progress (N_sample_buffer)
    last_force: jax.Array     # (2,) C3-solved pusher contact force (world xy), for the OSC
    rng: jax.Array
    crossed: jax.Array        # 1.0 once object XY entered the pose-tracking band


class C3SamplingCore:
    """The outer-loop logic, operating on (obj, ee, v_obj). Framework-free so
    it is testable without MJX; C3MJXSampling wraps it with state extraction."""

    def __init__(
            self,
            footprint,
            plant,
            goal,
            u_min,
            u_max,
            robot_radius=0.02,
            num_random=3,
            horizon=10,
            admm_iters=3,
            rho=0.1,
            rho_scale=3.0,
            rho_u=1.0,
            q_pos=200.0,
            q_theta=40.0,
            w_ee=10.0,
            w_v=0.05,
            qf_pos=2000.0,
            qf_theta=400.0,
            r_r=0.05,
            pos_success=0.03,
            theta_success=0.09,
            progress_window=16,
            progress_drop=0.5,  # dairlib kConfigCostDrop: 0.5 over 16 loops
            cost_switching_threshold_distance=0.05,  # ignore orientation until within 5 cm (position-first)
            hyst_c3_to_repos_frac=0.6,
            hyst_c3_to_repos_frac_position=0.7,
            hyst_repos_to_c3_frac=0.9,
            hyst_repos_to_c3_frac_position=0.5,
            hyst_repos_to_repos_frac=0.7,
            hyst_repos_to_repos_frac_position=0.7,
            contact_thresh=0.02,
            contact_margin=0.02,   # push only when the pusher is within this of contact
            force_scale=1.0,       # scale on the C3 contact force fed to the OSC
            safe_margin=0.02,
            align_tol=0.35,
            max_dphi=0.6,
            straight_line_angle=0.3,
            shell_clearance=0.027,   # dairlib sample_projection_clearance (kMeshNormal / kRandomOnShell)
            stall_widen=1.0,         # scale the shell jitter when stalled (exploration widening)
            n_boundary_per_edge=8,
            n_unsuccessful=10,       # dairlib N_unsuccessful_sample_buffer
            unsucc_radius=0.01,      # dairlib unsuccessful_radius
            n_good=8,                # dairlib N_sample_buffer (good-sample memory)
            obstacles=(),
            n_obstacles=2,
            obs_margin=0.01,
    ):
        self.footprint = footprint
        self.contact_fn = make_shape_contact(footprint, robot_radius)
        self.plant, self.dt = plant, plant.dt
        self.robot_radius = robot_radius
        self.bounding_radius = float(footprint.bounding_radius)
        self.horizon, self.admm_iters = horizon, admm_iters
        self.rho, self.rho_scale, self.rho_u = rho, rho_scale, rho_u
        self.num_random = num_random
        g = jnp.asarray(goal, dtype=float)
        self.goal = g
        self.q_pos, self.q_theta = q_pos, q_theta
        self.x_ref = jnp.array([g[0], g[1], g[2], g[0], g[1], 0, 0, 0, 0, 0.0])
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee, w_v)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0, w_v)
        self.R = r_r * jnp.eye(2)
        self.u_min, self.u_max = u_min, u_max
        self.pos_success, self.theta_success = pos_success, theta_success
        self.progress_window, self.progress_drop = progress_window, progress_drop
        # dairlib relative hysteresis fractions, split by position / pose mode.
        self.frac_c3repos, self.frac_c3repos_pos = (
            hyst_c3_to_repos_frac, hyst_c3_to_repos_frac_position)
        self.frac_reposc3, self.frac_reposc3_pos = (
            hyst_repos_to_c3_frac, hyst_repos_to_c3_frac_position)
        self.frac_reposrepos, self.frac_reposrepos_pos = (
            hyst_repos_to_repos_frac, hyst_repos_to_repos_frac_position)
        self.cost_switch_dist = cost_switching_threshold_distance
        # Position-only cost matrices (q_theta = 0) for the far-field phase.
        self.Q_pos = _state_cost_hessian(q_pos, 0.0, w_ee, w_v)
        self.Qf_pos = _state_cost_hessian(qf_pos, 0.0, 0.0, w_v)
        self.contact_thresh, self.safe_margin = contact_thresh, safe_margin
        self.contact_margin, self.force_scale = contact_margin, force_scale
        self.align_tol, self.max_dphi = align_tol, max_dphi
        self.straight_line_angle = straight_line_angle
        self.shell_clearance = shell_clearance
        self.stall_widen = stall_widen
        # P5: a DENSE mesh-normal contact set (body-frame points + outward
        # normals + lever arms), precomputed once. The step() heuristic ranks
        # all of them by how well pushing there reduces BOTH the position and
        # orientation error, then C3-solves the top few -- the real
        # sampling_strategy=kMeshNormal + N_sample_buffer role.
        self.cand_body = footprint.sample_boundary(n_boundary_per_edge)  # (M,2)
        self.num_boundary = self.cand_body.shape[0]
        normals = []
        for i in range(self.num_boundary):
            _, gr = footprint.sdf_and_grad(self.cand_body[i])
            normals.append(gr)
        self.cand_normal = jnp.stack(normals)  # (M,2)
        self.n_unsucc = n_unsuccessful
        self.unsucc_radius = unsucc_radius
        self.n_good = n_good
        self.obs_shapes = list(obstacles)
        self.n_obs = min(n_obstacles, len(self.obs_shapes))
        self.obs_margin = obs_margin

    def init_state(self, seed=0):
        W = self.progress_window
        return C3SampState(is_c3=jnp.asarray(0.0),
                           target=jnp.zeros(2),
                           target_body=jnp.zeros(2),
                           cost_hist=jnp.full((W,), 1e12),
                           n_prog=jnp.asarray(0),
                           unsucc=jnp.full((self.n_unsucc, 2), 1e3),
                           good_buf=jnp.full((self.n_good, 2), 1e3),
                           last_force=jnp.zeros(2),
                           rng=jax.random.key(seed),
                           crossed=jnp.asarray(0.0))

    def _plan_cost(self, xs):
        dpos = xs[:, :2] - self.goal[:2]
        dth = wrap_angle(xs[:, 2] - self.goal[2])
        return jnp.sum(self.q_pos * jnp.sum(dpos**2, axis=1) +
                       self.q_theta * dth**2)

    def _config_cost(self, obj):
        return (self.q_pos * jnp.sum((obj[:2] - self.goal[:2])**2) +
                self.q_theta * wrap_angle(obj[2] - self.goal[2])**2)

    def _obs_contacts(self, q):
        """N-closest object-obstacle contacts: (phi (No,), n (No,2), r (No,2)).
        Empty when the scene has no obstacles. n is the world push-away normal."""
        if self.n_obs == 0:
            return (jnp.zeros(0), jnp.zeros((0, 2)), jnp.zeros((0, 2)))
        oxy, oth = q[:2], q[2]
        cw = oxy[None, :] + jax.vmap(lambda pb: rotate(oth, pb))(self.cand_body)
        phis, ns, rs = [], [], []
        for s in self.obs_shapes:
            d, g = s.sdf_and_grad(cw)                # (M,), (M,2)
            j = jnp.argmin(d)
            gj = g[j] / (jnp.linalg.norm(g[j]) + 1e-9)
            phis.append(d[j] - self.obs_margin)
            ns.append(gj)
            rs.append(cw[j] - oxy)
        phis, ns, rs = jnp.stack(phis), jnp.stack(ns), jnp.stack(rs)
        sel = jax.lax.top_k(-phis, self.n_obs)[1]     # the N closest
        return phis[sel], ns[sel], rs[sel]

    def _ee_from_body(self, pb, oxy, theta):
        cw = oxy + rotate(theta, pb)
        _, gr = self.footprint.sdf_and_grad(pb)
        return cw + self.robot_radius * rotate(theta, gr)

    def _reposition_move(self, q_ee, target, c):
        """dairlib kCircular reposition (planar): if the new contact is only a
        small angle around the object from the current EE, go straight to it
        (use_straight_line_traj_within_angle); otherwise retreat to the ring,
        arc around, then approach -- so the EE never drags the (non-convex)
        object on a large-angle switch."""
        r_safe = self.bounding_radius + self.robot_radius + self.safe_margin
        v_ee = q_ee - c
        r_ee = jnp.linalg.norm(v_ee) + 1e-9
        phi_ee = jnp.arctan2(v_ee[1], v_ee[0])
        v_t = target - c
        phi_t = jnp.arctan2(v_t[1], v_t[0])
        dphi = wrap_angle(phi_t - phi_ee)  # angle to sweep around object
        straight = jnp.abs(
            dphi) < self.straight_line_angle  # near -> no retreat/arc
        aligned = jnp.abs(dphi) < self.align_tol
        inside = r_ee < r_safe
        retreat = c + r_safe * v_ee / r_ee  # out to the ring
        phi_next = phi_ee + jnp.clip(dphi, -self.max_dphi, self.max_dphi)
        orbit = c + r_safe * jnp.array([jnp.cos(phi_next), jnp.sin(phi_next)])
        circ_tgt = jnp.where(aligned, target, jnp.where(inside, retreat, orbit))
        tgt = jnp.where(straight, target,
                        circ_tgt)  # straight shortcut for small angle
        return jnp.clip((tgt - q_ee) / self.dt, self.u_min, self.u_max)

    def _ee_and_normal(self, pb, oxy, theta):
        """World EE placement for a body-frame boundary point, and the world
        outward surface normal there."""
        cw = oxy + rotate(theta, pb)
        _, gr = self.footprint.sdf_and_grad(pb)
        n_world = rotate(theta, gr)
        return cw + self.robot_radius * n_world, n_world

    def step(self, obj, ee, v_obj, s, Minv=None):
        theta, oxy = obj[2], obj[:2]

        # Position-first staging (dairlib cost_switching_threshold_distance):
        # while the object XY is farther than cost_switch_dist from the goal,
        # ignore orientation entirely (position-only costs, ranking, hysteresis,
        # progress). Latches once crossed, as crossed_cost_switching_threshold_.
        pose_diff = jnp.linalg.norm(oxy - self.goal[:2])
        crossing_now = (s.crossed <= 0.5) & (pose_diff < self.cost_switch_dist)
        crossed = (s.crossed > 0.5) | (pose_diff < self.cost_switch_dist)
        q_theta_eff = jnp.where(crossed, self.q_theta, 0.0)
        Q_eff = jnp.where(crossed, self.Q, self.Q_pos)
        Qf_eff = jnp.where(crossed, self.Qf, self.Qf_pos)

        # --- kMeshNormal sampling (dairlib sampling_strategy=kMeshNormal) ---
        # Draw contact faces RANDOMLY from the mesh-normal candidate pool. This
        # is faithful to Push Anything, which samples contacts randomly rather
        # than greedily toward the goal; the best of the solved samples is then
        # picked downstream by C3 cost. (The previous greedy top-k ranking was
        # our divergence from the original.)
        cw = oxy[None, :] + jax.vmap(lambda pb: rotate(theta, pb))(
            self.cand_body)
        nw = jax.vmap(lambda gr: rotate(theta, gr))(self.cand_normal)
        ee_cand = cw + self.robot_radius * nw

        # avoid_choosing_unsuccessful_samples: drop faces whose body-frame
        # contact lies inside the unsuccessful-buffer radius by zeroing their
        # draw probability (kept at 1e-6 so the distribution stays normalizable).
        d_bad = jnp.linalg.norm(self.cand_body[:, None, :] -
                                s.unsucc[None, :, :],
                                axis=-1)
        ok = jnp.min(d_bad, axis=1) >= self.unsucc_radius
        probs = ok.astype(jnp.float32) + 1e-6
        probs = probs / jnp.sum(probs)

        # stall-triggered exploration: widen the shell jitter once the config
        # cost has stalled (uses last step's progress counter), so a jammed
        # pusher is shaken toward a fresh contact instead of re-picking the
        # same face -- a substitute for the original's buffer-driven diversity.
        rng, k_idx, k_jit = jax.random.split(s.rng, 3)
        widen = (s.n_prog >= self.progress_window - 1).astype(jnp.float32)
        idx = jax.random.choice(k_idx, self.num_boundary,
                                shape=(self.num_random,), replace=False,
                                p=probs)
        samp_ees = ee_cand[idx]
        # reposition jitter: offset each sampled EE outward along its face
        # normal by a random shell clearance (dairlib sample_projection_clearance
        # / kRandomOnShell), widened when stalled.
        clr = self.shell_clearance * (1.0 + self.stall_widen * widen)
        jit = jax.random.uniform(k_jit, shape=(self.num_random, 1),
                                 minval=0.0, maxval=clr)
        samp_ees = samp_ees + jit * nw[idx]

        # N_sample_buffer: re-propose contacts that previously made progress.
        # Reconstruct each buffered body-frame contact's world EE from the
        # nearest precomputed mesh-face normal (no per-step SDF needed);
        # invalid (sentinel) entries map far away and lose on cost.
        def _good_ee(pb):
            j = jnp.argmin(jnp.linalg.norm(self.cand_body - pb[None, :],
                                           axis=1))
            cwp = oxy + rotate(theta, pb)
            return cwp + self.robot_radius * rotate(theta, self.cand_normal[j])
        good_ees = jax.vmap(_good_ee)(s.good_buf)

        samples = jnp.concatenate(
            [ee[None, :], s.target[None, :], samp_ees, good_ees], axis=0)
        v5 = jnp.concatenate([v_obj, jnp.zeros(2)])

        def plan_cost(xs):
            # Faithful C3+ sample cost: object goal error only. Obstacle
            # avoidance comes from the object-obstacle CONTACT in the LCS
            # (see _obs_contacts), exactly as in dairlib -- the original has
            # no obstacle cost term.
            dpos = xs[:, :2] - self.goal[:2]
            dth = wrap_angle(xs[:, 2] - self.goal[2])
            return jnp.sum(self.q_pos * jnp.sum(dpos**2, axis=1) +
                           q_theta_eff * dth**2)

        obs = self._obs_contacts(obj)      # same for all candidates (depends on object pose only)

        def solve_one(p_i):
            x_init = jnp.concatenate([obj, p_i, v5])
            lcs = build_dynamic_lcs(self.plant, self.contact_fn, x_init,
                                    jnp.zeros(2), Minv=Minv, obs=obs)
            _, us, lams = c3_solve(
                lcs, x_init, self.x_ref, Q_eff, self.R, Qf_eff,
                rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
                u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
                rho_scale=self.rho_scale)
            sim_xs = simulate_rollout(self.plant, self.contact_fn, x_init, us,
                                      Minv=Minv, obs_fn=self._obs_contacts)
            return plan_cost(sim_xs), us[0], lams[0][:2]

        costs, first_us, first_lams = jax.vmap(solve_one)(samples)
        curr_cost, push_u, push_lam = costs[0], first_us[0], first_lams[0]

        # C3-solved pusher contact force at the current EE, for the OSC force
        # feedforward (Stage 2). The two Anitescu cone-edge multipliers give a
        # normal force (l1+l2) along the contact normal and friction mu*(l1-l2)
        # along the tangent; the pusher applies this INTO the object (-n).
        phi_ee, n_ee, r_ee = self.contact_fn(jnp.concatenate([obj, ee]))
        t_ee = jnp.array([-n_ee[1], n_ee[0]])
        F_c3 = self.force_scale * (
            -(push_lam[0] + push_lam[1]) * n_ee
            - self.plant.mu_p * (push_lam[0] - push_lam[1]) * t_ee)
        repos_target_cost = costs[1]
        new_costs = costs[2:]
        new_i = jnp.argmin(new_costs)
        best_new, best_new_cost = samples[2 + new_i], new_costs[new_i]
        other_costs = costs[1:]
        other_i = jnp.argmin(other_costs)
        best_other, best_other_cost = samples[1 + other_i], other_costs[other_i]

        # P1 goal met (full pose)
        pos_err = jnp.linalg.norm(oxy - self.goal[:2])
        th_err = jnp.abs(wrap_angle(theta - self.goal[2]))
        goal_met = (pos_err < self.pos_success) & (th_err < self.theta_success)

        # P2 progress: kConfigCostDrop -- stall if the (position-only until
        # crossed) config cost has not dropped by progress_drop over the window.
        config_cost = (self.q_pos * jnp.sum((oxy - self.goal[:2])**2) +
                       q_theta_eff * wrap_angle(theta - self.goal[2])**2)
        cost_hist = jnp.concatenate([s.cost_hist[1:], config_cost[None]])
        n_prog = s.n_prog + 1
        full = n_prog >= self.progress_window
        frac = (cost_hist[-1] - cost_hist[0]) / (cost_hist[0] + 1e-9)
        stalled = full & (frac > -self.progress_drop)

        # P3 hysteresis (relative; position vs pose fraction by `crossed`)
        fr_c3repos = jnp.where(crossed, self.frac_c3repos,
                               self.frac_c3repos_pos)
        fr_reposc3 = jnp.where(crossed, self.frac_reposc3,
                               self.frac_reposc3_pos)
        fr_reposrepos = jnp.where(crossed, self.frac_reposrepos,
                                  self.frac_reposrepos_pos)
        is_c3 = s.is_c3 > 0.5
        reached = jnp.linalg.norm(ee - s.target) < self.contact_thresh
        c3_cost_switch = best_other_cost < (1.0 - fr_c3repos) * curr_cost
        leave_c3 = stalled | c3_cost_switch
        repos_back = reached | (curr_cost
                                < (1.0 - fr_reposc3) * best_other_cost)
        switch_target = best_new_cost < (1.0 -
                                         fr_reposrepos) * repos_target_cost
        new_is_c3 = jnp.where(is_c3, ~leave_c3, repos_back).astype(jnp.float32)
        target_if_c3 = jnp.where(leave_c3, best_other, s.target)
        target_if_repos = jnp.where(switch_target, best_new, s.target)
        new_target = jnp.where(is_c3, target_if_c3, target_if_repos)

        # Contact gate on push ENTRY only (matches the original): out of contact
        # the push QP is blind to the object and drives u -> 0, so never ENTER
        # push from out of contact -- reposition toward the best goal-reducing
        # contact until actually within contact_margin, then push. But once
        # pushing, STAY in push even if contact briefly loosens as the block
        # moves; the original keeps pushing as long as it makes progress and its
        # OSC re-presses to hold contact. Exit is left to the progress-stall
        # logic. Forcing exit on every phi > margin (as a symmetric gate would)
        # breaks a sustained push the instant the arm lags a few cm behind.
        out_of_contact = phi_ee >= self.contact_margin
        was_push = s.is_c3 > 0.5
        new_is_c3 = jnp.where(out_of_contact & (~was_push), 0.0, new_is_c3)
        # NOTE: the reposition TARGET is intentionally left to the dairlib
        # latched-target hysteresis computed above (switch_target /
        # target_if_repos, lines ~1019-1024): once a repositioning contact is
        # chosen it is HELD until a fresh sample beats its cost by
        # frac_reposrepos. An earlier revision overrode it here with
        # `new_target = best_new` on every out-of-contact step, which
        # re-randomized the target each control step and made a large-angle
        # orbit incoherent -- the arm dithered in place and never arced from
        # the +x push side round to the -y/+y-pushing contact. Do NOT re-chase
        # best_new here; the entry-mode gate above is the only contact
        # correction, matching the original (mode is gated, target is latched).

        # Reset progress history on a mode flip, on goal, or when crossing the
        # position band (the cost definition changes, so old history is stale).
        mode_flipped = (new_is_c3 > 0.5) != is_c3
        reset = mode_flipped | goal_met | crossing_now
        cost_hist = jnp.where(reset, jnp.full_like(cost_hist, 1e12), cost_hist)
        n_prog = jnp.where(reset, 0, n_prog)

        new_target_body = rotate(-theta, new_target - oxy)
        stalled_in_c3 = is_c3 & stalled
        unsucc = jnp.where(
            stalled_in_c3,
            jnp.concatenate([s.unsucc[1:], s.target_body[None, :]], axis=0),
            s.unsucc)

        # N_sample_buffer retention: remember the pushing contact whenever it
        # improved the object config cost (position/orientation retention
        # analog; body-frame contact stands in for the full sample).
        progressed = is_c3 & (config_cost < s.cost_hist[-1])
        good_buf = jnp.where(
            progressed,
            jnp.concatenate([s.good_buf[1:], s.target_body[None, :]], axis=0),
            s.good_buf)

        repos_action = self._reposition_move(ee, new_target, oxy)
        u0 = jnp.where(new_is_c3 > 0.5, push_u, repos_action)
        u0 = jnp.where(goal_met, jnp.zeros(2), u0)
        F_c3 = jnp.where(goal_met, jnp.zeros(2), F_c3)
        return u0, s.replace(is_c3=new_is_c3,
                             target=new_target,
                             target_body=new_target_body,
                             last_force=F_c3,
                             cost_hist=cost_hist,
                             n_prog=n_prog,
                             unsucc=unsucc,
                             good_buf=good_buf,
                             rng=rng,
                             crossed=crossed.astype(jnp.float32))


@dataclass
class C3SamplingParams:
    tk: jax.Array
    mean: jax.Array
    rng: jax.Array
    samp: C3SampState


class C3MJXSampling(SamplingBasedController):
    """Push Anything C3+ (local C3 + sampling/reposition) as a flat baseline."""

    def __init__(self,
                 task,
                 *,
                 plan_horizon,
                 num_knots,
                 seed=0,
                 robot_radius=0.02,
                 mu_p=0.5,
                 kv=100.0,
                 num_random=3,
                 admm_iters=3,
                 rho=0.1,
                 rho_scale=3.0,
                 **core_kwargs):
        super().__init__(task,
                         num_randomizations=1,
                         risk_strategy=None,
                         seed=seed,
                         plan_horizon=plan_horizon,
                         spline_type="zero",
                         num_knots=num_knots,
                         iterations=1)
        import numpy as np
        m = task.model
        self.block_dofs = jnp.asarray(task.block_dofs)   # object qvel DOFs (both)
        # Embodiment: the point task sets pusher_dofs/pusher_body_id; the xarm6
        # task instead exposes tip_site_id/block_qpos_adr and its EE is a
        # 6-joint arm. C3 ALWAYS plans a 2-DOF planar EE against the object; on
        # the arm that EE is the tip site (FK) and its 2-D velocity is mapped to
        # joint velocities in the execution loop (run_3d_plain, mj_jacSite) --
        # see `emits_ee_velocity`. The point path drives the 2 slide joints
        # directly, so its action already equals ctrl.
        self.is_xarm6 = not hasattr(task, "pusher_body_id")
        self.emits_ee_velocity = self.is_xarm6
        # The object's nominal ground-friction budget, from the analytic model
        # (not `dof_frictionloss`): tabletop scenes take support friction from
        # the table CONTACT and their joints carry none, so the LCS reads the
        # nominal load here. Constant bound -- models the nominal load only.
        fl = np.asarray(task.object_model.wrench_limit)
        if self.is_xarm6:
            self.tip_site_id = int(task.tip_site_id)
            self.block_qpos_adr = jnp.asarray(task.block_qpos_adr)  # [x,y,yaw]
            # Drive the arm with an operational-space (Khatib) torque controller
            # in the execution loop (run_3d_plain): C3's planar EE velocity is
            # the xy task, tip height and tilt are held. run_3d_plain reads
            # these to flip the arm actuators to torque and pick OSC gains.
            # Gravity needs no term here -- the model's gravcomp handles it.
            self.arm_torque_osc = True
            self.osc_kv_xy = 20.0    # xy op-space velocity gain (tracks C3 vel)
            self.osc_kp_z = 8.0      # tip-height error -> desired descent speed
            self.osc_z_vmax = 0.3    # cap on descent speed (diagonal approach)
            self.osc_kd_z = 60.0     # descent-velocity tracking gain
            self.osc_kp_rot = 100.0  # tilt stiffness (hold stick vertical)
            self.osc_kd_rot = 20.0
            # The C3 contact model treats the pusher as a disk of radius
            # `robot_radius`; the point robot's is 0.02, but the xarm6 stick is
            # a thin capsule. Using the wrong (larger) radius makes C3 stop the
            # EE a standoff short of the block that the thin stick cannot span,
            # so it never makes contact -- read the real stick radius instead.
            robot_radius = float(
                np.asarray(m.geom_size)[int(task.stick_geoms[0])][0])
            me, bv = 1.0, 0.0                 # nominal planar point-mass EE (v1)
            ev = float(getattr(task, "ee_speed_limit", 0.5))  # EE Cartesian m/s
            u_min, u_max = jnp.array([-ev, -ev]), jnp.array([ev, ev])
        else:
            self.pusher_dofs = jnp.asarray(task.pusher_dofs)
            self.pusher_bid = int(task.pusher_body_id)
            self.idx5 = jnp.concatenate([self.block_dofs, self.pusher_dofs])
            self.pusher_offset = jnp.asarray(
                np.asarray(m.body_pos)[self.pusher_bid][:2])  # live EE from qpos
            bv = float(np.asarray(m.dof_damping)[int(self.pusher_dofs[0])])
            me = float(np.asarray(m.body_mass)[self.pusher_bid])
            u_min, u_max = task.u_min, task.u_max
        plant = PlantParams(mo=2.0,
                            Io=0.005,
                            me=me,
                            kv=kv,
                            bv=bv,
                            mu_p=mu_p,
                            bx=float(fl[0]),
                            by=float(fl[1]),
                            bth=float(fl[2]),
                            dt=float(task.dt))
        om = getattr(task, "object_model", None)
        field = getattr(om, "obstacles", None) if om is not None else None
        obstacles = tuple(field.shapes) if field is not None else ()
        self.core = C3SamplingCore(task.object_model.footprint,
                                   plant,
                                   task.goal,
                                   u_min,
                                   u_max,
                                   robot_radius=robot_radius,
                                   num_random=num_random,
                                   horizon=num_knots,
                                   admm_iters=admm_iters,
                                   rho=rho,
                                   rho_scale=rho_scale,
                                   obstacles=obstacles,
                                   **core_kwargs)
        self._seed = seed
        sites = getattr(task, "trace_site_ids", None)
        self._n_sites = int(sites.shape[0]) if sites is not None else 1

    def init_params(self, initial_knots=None, seed=0):
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        mean = jnp.zeros((self.num_knots, 2))
        return C3SamplingParams(tk=tk,
                                mean=mean,
                                rng=jax.random.key(seed),
                                samp=self.core.init_state(seed))

    def optimize(self, state, params):
        new_tk = jnp.linspace(0.0, self.plan_horizon,
                              self.num_knots) + state.time
        if self.is_xarm6:
            # FK inside jit: the exec loop updates qpos but not site_xpos, so
            # recompute the tip's world pose from the live qpos.
            kd = mjx.kinematics(self.model, state)
            obj = state.qpos[self.block_qpos_adr]           # object [x, y, yaw]
            ee = kd.site_xpos[self.tip_site_id, :2]         # planar EE (FK)
            v_obj = state.qvel[self.block_dofs]
            u0, samp = self.core.step(obj, ee, v_obj, params.samp, Minv=None)
        else:
            M = mjx.full_m(self.model, state)
            Minv = jnp.linalg.inv(M[self.idx5][:, self.idx5])
            obj = state.qpos[self.block_dofs]
            ee = self.pusher_offset + state.qpos[self.pusher_dofs]  # live, not xpos
            v_obj = state.qvel[self.block_dofs]
            u0, samp = self.core.step(obj, ee, v_obj, params.samp, Minv=Minv)
        mean = jnp.broadcast_to(u0, (self.num_knots, 2))
        params = params.replace(tk=new_tk, mean=mean, samp=samp)
        H = self.ctrl_steps
        dummy = Trajectory(controls=jnp.zeros((1, H, 2)),
                           knots=mean[None],
                           costs=jnp.zeros((1, H + 1)),
                           trace_sites=jnp.zeros((1, H + 1, self._n_sites, 3)))
        return params, dummy

    def nominal_trace(self, state, params):
        # The base overlay clips `params.mean` by `task.u_min/u_max` and rolls
        # it out as ctrl -- valid for the point robot (mean IS the 2-DOF ctrl),
        # but not on xarm6, where mean is a 2-D EE velocity while the task
        # bounds are 5-D task space and ctrl is 6 joints. Return a degenerate
        # trace at the current tip so the overlay never drives that mismatch.
        if self.is_xarm6:
            H = self.ctrl_steps
            kd = mjx.kinematics(self.model, state)
            return jnp.broadcast_to(kd.site_xpos[self.tip_site_id], (H + 1, 3))
        return super().nominal_trace(state, params)

    def sample_knots(self, params):
        return params.mean[None, ...], params

    def update_params(self, params, rollouts):
        return params
