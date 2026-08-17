"""Contact-implicit MPC (C3+ style) baseline, as one integrated module.

Mirrors the reference C3+ algorithm ("Push Anything") but runs on an analytic
Linear Complementarity System (LCS) we build for planar pushing.

LCS convention (per step k):
    x_{k+1} = A x_k + B u_k + G lam_k + d
    0 <= lam_k  _|_  E x_k + F lam_k + H u_k + c >= 0
The lam-to-state map is G (not D) to avoid clashing with the limit-surface
compliance D used in the pushing physics.
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
    """Solve 0 <= lam _|_ (M lam + q) >= 0 by projected Jacobi iteration."""
    diag = jnp.diag(M)
    inv_diag = jnp.where(jnp.abs(diag) > 1e-12, 1.0 / diag, 0.0)

    def _sweep(lam, _):
        return jnp.maximum(0.0, lam - inv_diag * (M @ lam + q)), None

    lam, _ = jax.lax.scan(_sweep, jnp.zeros_like(q), None, length=iters)
    return lam


def lcs_step(lcs, x, u, wrap_theta_index=2):
    q = lcs.E @ x + lcs.H @ u + lcs.c
    lam = solve_lcp(lcs.F, q)
    x_next = lcs.A @ x + lcs.B @ u + lcs.G @ lam + lcs.d
    if wrap_theta_index >= 0:
        x_next = x_next.at[wrap_theta_index].set(
            wrap_angle(x_next[wrap_theta_index])
        )
    return x_next, lam


def lcs_rollout(lcs, x0, controls, wrap_theta_index=2):
    def _body(x, u):
        x_next, _ = lcs_step(lcs, x, u, wrap_theta_index)
        return x_next, x_next

    _, xs = jax.lax.scan(_body, x0, controls)
    return jnp.concatenate([x0[None, :], xs], axis=0)


# =====================================================================
# LCS builders
# =====================================================================


def build_planar_pushing_lcs(wrench_limit, dt):
    """Object-only, wrench-driven planar-pushing LCS (validation testbed)."""
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
    shape, limit_surface_d, robot_radius, object_pose, pusher_pos, dt,
    mu_c=0.0, slide_sign=0.0,
):
    """Linearize the single-point pusher-object contact into an LCS.

    State x = [obj_x, obj_y, obj_theta, ee_x, ee_y] (n=5), control u = pusher
    velocity (m=2), lam = contact normal force (k=1).
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
    W = contact_force_to_com_wrench(object_pose, p_world, a_hat)
    gvec = dt * D * W
    dcontact = jnp.array([gvec[0] - gvec[2] * r[1], gvec[1] + gvec[2] * r[0]])

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
# C3+ ADMM solver (with optional input box + growing-rho adaptation)
# =====================================================================


def project_complementarity(a, b):
    """Project each pair (a_i, b_i) onto {a>=0, b>=0, a*b=0}, elementwise."""
    dist_a = jnp.where(a < 0, a**2, 0.0) + b**2
    dist_b = a**2 + jnp.where(b < 0, b**2, 0.0)
    use_a = dist_a <= dist_b
    return (
        jnp.where(use_a, jnp.maximum(a, 0.0), 0.0),
        jnp.where(use_a, 0.0, jnp.maximum(b, 0.0)),
    )


def c3_solve(
    lcs, x_init, x_ref, Q, R, Qf,
    rho=1.0, horizon=10, admm_iters=40, reg=1e-6,
    u_min=None, u_max=None, rho_u=1.0, rho_scale=1.0,
):
    """C3+ ADMM: KKT z-step + complementarity projection + input-box projection.

    rho_scale > 1 grows the ADMM penalty each iteration (rho <- rho*rho_scale),
    the residual-balancing trick from the C3 paper; rho_scale == 1 keeps rho
    fixed and reuses a single KKT factorization (the fast path).

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

    # --- Equality constraints (constant): initial + dynamics + slack.
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
        # Fast path: one KKT factorization reused across iterations.
        kkt = jnp.block([[make_P(rho, rho_u), C.T], [C, zeros_mm]])
        for _ in range(admm_iters):
            q = build_q(lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u, rho, rho_u)
            z = jnp.linalg.solve(kkt, jnp.concatenate([-q, bvec]))[:Z]
            lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u = update(
                z, lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u
            )
    else:
        # Adaptive path: rho grows each iteration, so rebuild the KKT.
        for it in range(admm_iters):
            rho_l = rho * (rho_scale ** it)
            rho_u_l = rho_u * (rho_scale ** it)
            kkt = jnp.block([[make_P(rho_l, rho_u_l), C.T], [C, zeros_mm]])
            q = build_q(
                lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u, rho_l, rho_u_l
            )
            z = jnp.linalg.solve(kkt, jnp.concatenate([-q, bvec]))[:Z]
            lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u = update(
                z, lam_hat, eta_hat, w_lam, w_eta, u_hat, w_u
            )

    xs = jnp.stack([z[xk(k):xk(k) + n] for k in range(N)] + [z[xN:xN + n]])
    us = stack(z, uk, m)
    lams = stack(z, lk, kd)
    if bounded:
        us = jnp.clip(us, u_min, u_max)
    return xs, us, lams


# =====================================================================
# Controllers
# =====================================================================


@dataclass
class C3ControllerParams:
    us: jax.Array
    t0: jax.Array


def _state_cost_hessian(q_pos, q_theta, w_ee):
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
    """C3+ controller for the 2D push task (single fixed contact per step)."""

    def __init__(
        self, task, rho=0.1, horizon=10, admm_iters=40,
        q_pos=1000.0, q_theta=100.0, w_ee=400.0,
        qf_pos=10000.0, qf_theta=1000.0, r_r=0.05, mu_c=0.0,
        rho_u=1.0, rho_scale=1.0,
    ):
        self.task = task
        self.dt = float(task.dt)
        self.rho, self.rho_u, self.rho_scale = rho, rho_u, rho_scale
        self.horizon, self.admm_iters = horizon, admm_iters
        self.mu_c = mu_c
        self.shape = task.footprint
        self.D = task.model.limit_surface_d
        self.robot_radius = task.model.robot_radius
        g = task.goal
        self.x_ref = jnp.array([g[0], g[1], g[2], g[0], g[1]])
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0)
        self.R = r_r * jnp.eye(2)
        self.u_min, self.u_max = task.u_min, task.u_max

    def init_params(self, seed=0):
        del seed
        return C3ControllerParams(
            us=jnp.zeros((self.horizon, 2)), t0=jnp.asarray(0.0)
        )

    def optimize(self, state, params):
        lcs = build_contact_lcs(
            self.shape, self.D, self.robot_radius,
            state.object_pose, state.robot_pos, self.dt,
            mu_c=self.mu_c, slide_sign=0.0,
        )
        x_init = jnp.concatenate([state.object_pose, state.robot_pos])
        _, us, _ = c3_solve(
            lcs, x_init, self.x_ref, self.Q, self.R, self.Qf,
            rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
            u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
            rho_scale=self.rho_scale,
        )
        return params.replace(us=us, t0=state.time), us

    def get_action(self, params, t):
        idx = jnp.clip(
            jnp.floor((t - params.t0) / self.dt).astype(jnp.int32),
            0, self.horizon - 1,
        )
        return params.us[idx]


# =====================================================================
# sampling-C3 outer loop, v4: P1 (goal-met stop) + P2 (progress cutoff) +
# P3 (sticky asymmetric hysteresis), matched to dairlib
# systems/controllers/sampling_based_c3_controller.cc.
#
#   P1  line 870 : object within pos/ang success thresholds -> stop pushing.
#   P2  line 2071: object config-cost fails to drop >=10% over W loops -> reposition.
#   P3  line 1151-1265: relative hysteresis, c3->repos 0.8, repos->repos 0.9,
#                       repos->c3 0.5.
# Cost is still the raw plan cost (P4 -- simulated rollout -- deferred).
# =====================================================================


@dataclass
class C3SamplingParams:
    u0: jax.Array
    is_c3: jax.Array          # 1.0 = pushing (C3), 0.0 = repositioning
    target: jax.Array         # current repositioning target (world EE pos)
    cost_hist: jax.Array      # (W,) object config-cost history for progress
    n_prog: jax.Array         # steps since last progress reset
    key: jax.Array


class C3Sampling:
    """Sampling-C3 with goal-met stop, progress cutoff, and sticky hysteresis."""

    def __init__(
        self, task, num_random=3, rho=0.1, horizon=8, admm_iters=20,
        q_pos=1000.0, q_theta=100.0, w_ee=400.0,
        qf_pos=10000.0, qf_theta=1000.0, r_r=0.05, mu_c=0.0,
        rho_u=1.0, rho_scale=1.0, contact_thresh=0.02,
        safe_margin=0.02, align_tol=0.35, max_dphi=0.6,
        # P1 goal thresholds:
        pos_success=0.02, theta_success=0.10,
        # P2 progress cutoff:
        progress_window=40, progress_drop=0.1,
        # P3 hysteresis fractions (from progress_params_c3plus.yaml):
        hyst_c3_to_repos_frac=0.8, hyst_repos_to_repos_frac=0.9,
        hyst_repos_to_c3_frac=0.5,
    ):
        self.task = task
        self.dt = float(task.dt)
        self.rho, self.rho_u, self.rho_scale = rho, rho_u, rho_scale
        self.horizon, self.admm_iters = horizon, admm_iters
        self.mu_c = mu_c
        self.contact_thresh = contact_thresh
        self.safe_margin, self.align_tol, self.max_dphi = (
            safe_margin, align_tol, max_dphi)
        self.pos_success, self.theta_success = pos_success, theta_success
        self.progress_window, self.progress_drop = progress_window, progress_drop
        self.h_c3_repos = hyst_c3_to_repos_frac
        self.h_repos_repos = hyst_repos_to_repos_frac
        self.h_repos_c3 = hyst_repos_to_c3_frac
        self.shape = task.footprint
        self.D = task.model.limit_surface_d
        self.robot_radius = task.model.robot_radius
        self.bounding_radius = float(task.footprint.bounding_radius)
        g = task.goal
        self.goal = jnp.asarray(g, dtype=float)
        self.x_ref = jnp.array([g[0], g[1], g[2], g[0], g[1]])
        self.q_pos, self.q_theta = q_pos, q_theta
        self.Q = _state_cost_hessian(q_pos, q_theta, w_ee)
        self.Qf = _state_cost_hessian(qf_pos, qf_theta, 0.0)
        self.R = r_r * jnp.eye(2)
        self.u_min, self.u_max = task.u_min, task.u_max
        self.cand_body = task.footprint.sample_boundary(3)
        self.num_boundary = self.cand_body.shape[0]
        self.num_random = num_random

    def init_params(self, seed=0):
        W = self.progress_window
        return C3SamplingParams(
            u0=jnp.zeros(2), is_c3=jnp.asarray(1.0), target=jnp.zeros(2),
            cost_hist=jnp.full((W,), 1e12), n_prog=jnp.asarray(0),
            key=jax.random.key(seed),
        )

    def _plan_cost(self, xs):
        dpos = xs[:, :2] - self.goal[:2]
        dth = wrap_angle(xs[:, 2] - self.goal[2])
        return jnp.sum(
            self.q_pos * jnp.sum(dpos**2, axis=1) + self.q_theta * dth**2)

    def _config_cost(self, obj):
        return (self.q_pos * jnp.sum((obj[:2] - self.goal[:2]) ** 2)
                + self.q_theta * wrap_angle(obj[2] - self.goal[2]) ** 2)

    def _ee_from_body(self, pb, oxy, theta):
        cw = oxy + rotate(theta, pb)
        _, gr = self.shape.sdf_and_grad(pb)
        return cw + self.robot_radius * rotate(theta, gr)

    def _reposition_move(self, q_ee, target, c):
        r_safe = self.bounding_radius + self.robot_radius + self.safe_margin
        v_ee, v_t = q_ee - c, target - c
        phi_ee = jnp.arctan2(v_ee[1], v_ee[0])
        phi_t = jnp.arctan2(v_t[1], v_t[0])
        dphi = wrap_angle(phi_t - phi_ee)
        aligned = jnp.abs(dphi) < self.align_tol
        phi_next = phi_ee + jnp.clip(dphi, -self.max_dphi, self.max_dphi)
        orbit = c + r_safe * jnp.array([jnp.cos(phi_next), jnp.sin(phi_next)])
        tgt = jnp.where(aligned, target, orbit)
        return jnp.clip((tgt - q_ee) / self.dt, self.u_min, self.u_max)

    def optimize(self, state, params):
        obj = state.object_pose
        q_ee = state.robot_pos
        theta, oxy = obj[2], obj[:2]

        key, sub = jax.random.split(params.key)
        idxs = jax.random.randint(sub, (self.num_random,), 0, self.num_boundary)
        rand_ees = jax.vmap(
            lambda i: self._ee_from_body(self.cand_body[i], oxy, theta))(idxs)
        # samples: [current EE, current target, random...]
        samples = jnp.concatenate(
            [q_ee[None, :], params.target[None, :], rand_ees], axis=0)

        def solve_one(p_i):
            lcs = build_contact_lcs(
                self.shape, self.D, self.robot_radius, obj, p_i, self.dt,
                mu_c=self.mu_c, slide_sign=0.0)
            x_init = jnp.concatenate([obj, p_i])
            xs, us, _ = c3_solve(
                lcs, x_init, self.x_ref, self.Q, self.R, self.Qf,
                rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
                u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
                rho_scale=self.rho_scale)
            return self._plan_cost(xs), us[0]

        costs, first_us = jax.vmap(solve_one)(samples)
        curr_cost = costs[0]
        push_u = first_us[0]
        repos_target_cost = costs[1]
        new_costs = costs[2:]
        new_i = jnp.argmin(new_costs)
        best_new = samples[2 + new_i]
        best_new_cost = new_costs[new_i]
        # best "other" (non-current) sample over target + randoms.
        other_costs = costs[1:]
        other_i = jnp.argmin(other_costs)
        best_other = samples[1 + other_i]
        best_other_cost = other_costs[other_i]

        # --- P1: goal met?
        pos_err = jnp.linalg.norm(oxy - self.goal[:2])
        th_err = jnp.abs(wrap_angle(theta - self.goal[2]))
        goal_met = (pos_err < self.pos_success) & (th_err < self.theta_success)

        # --- P2: progress over the window.
        config_cost = self._config_cost(obj)
        cost_hist = jnp.concatenate([params.cost_hist[1:], config_cost[None]])
        n_prog = params.n_prog + 1
        full = n_prog >= self.progress_window
        frac = (cost_hist[-1] - cost_hist[0]) / (cost_hist[0] + 1e-9)
        stalled = full & (frac > -self.progress_drop)

        is_c3 = params.is_c3 > 0.5
        reached = jnp.linalg.norm(q_ee - params.target) < self.contact_thresh

        # --- P3: mode transitions with sticky asymmetric hysteresis.
        # In C3: leave to repos on stall OR a dramatically cheaper alternative.
        c3_cost_switch = best_other_cost < (1.0 - self.h_c3_repos) * curr_cost
        leave_c3 = stalled | c3_cost_switch
        # In repos: return to C3 on reaching target OR current clearly good.
        repos_back_to_c3 = reached | (curr_cost < self.h_repos_c3 * best_other_cost)
        # In repos: switch target only if a new sample is drastically better.
        switch_target = best_new_cost < (1.0 - self.h_repos_repos) * repos_target_cost

        new_is_c3 = jnp.where(
            is_c3, ~leave_c3, repos_back_to_c3).astype(jnp.float32)
        # target: entering repos -> best_other; staying repos -> maybe switch.
        target_if_c3 = jnp.where(leave_c3, best_other, params.target)
        target_if_repos = jnp.where(switch_target, best_new, params.target)
        new_target = jnp.where(is_c3, target_if_c3, target_if_repos)

        # Reset progress history whenever the mode flips.
        mode_flipped = (new_is_c3 > 0.5) != is_c3
        reset = mode_flipped | goal_met
        cost_hist = jnp.where(reset, jnp.full_like(cost_hist, 1e12), cost_hist)
        n_prog = jnp.where(reset, 0, n_prog)

        # --- Action.
        push_action = push_u
        repos_action = self._reposition_move(q_ee, new_target, oxy)
        u0 = jnp.where(new_is_c3 > 0.5, push_action, repos_action)
        # P1: goal met overrides everything -> hold still (stop pushing).
        u0 = jnp.where(goal_met, jnp.zeros(2), u0)

        return params.replace(
            u0=u0, is_c3=new_is_c3, target=new_target,
            cost_hist=cost_hist, n_prog=n_prog, key=key), u0

    def get_action(self, params, t):
        del t
        return params.u0
