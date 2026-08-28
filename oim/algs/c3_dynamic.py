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
Lambda lam = [ lam1_p, lam2_p,                       # pusher Anitescu edges
               g_gx, gx+, gx-,                       # ground box friction, x
               g_gy, gy+, gy-,                        #                       y
               g_gth, gth+, gth- ]  (k = 11)          #                    theta

LCS convention (per step):
    x_{k+1} = A x + B u + G lam + d
    0 <= lam  _|_  E x + F lam + H u + c >= 0
G is the lam->state map (not the limit-surface D).
"""

from typing import Any, Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass, field
from mujoco import mjx

from oim.alg_base import SamplingBasedController, Trajectory

# The solver and its LCS container are shared with the object-only C3
# baseline; only the plant differs. Defaults there are the 2D ones, so
# every call below passes its own.
from oim.algs.c3 import LCS, c3_solve
from oim.objects.planar_pushing import wrap_angle
from oim.objects.sdf import rotate


# =====================================================================
# Plant parameters. Defaults mirror the point-robot MJCF; build_lcs_from_mjx
# (next step) will read M, frictionloss and kv straight from the MJX model so
# the LCS is provably the same plant the run executes in.
# =====================================================================
@dataclass
class PlantParams:
    """The planar plant: masses, servo gains and ground friction bounds."""

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


def _M_diag(pp: PlantParams) -> jax.Array:
    return jnp.array([pp.mo, pp.mo, pp.Io, pp.me, pp.me])


def _default_Minv(pp: PlantParams) -> jax.Array:
    """Diagonal M^-1 from the scalar masses (used when no full M is given)."""
    return jnp.diag(1.0 / _M_diag(pp))


def _ground_bounds(pp: PlantParams) -> jax.Array:
    return jnp.array([pp.bx, pp.by, pp.bth])


def _smooth_discrete(pp: PlantParams) -> Tuple[jax.Array, jax.Array]:
    """Implicit discrete contact-free velocity map, v_next = A_v v + B_v u.

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
# the object COM to the contact point, via shape.sdf_and_grad.
# =====================================================================
def make_shape_contact(
    shape: Any, robot_radius: float
) -> Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]]:
    """Contact function for an oim footprint `shape` (body-frame SDF)."""

    def contact_fn(
        q: jax.Array,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
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


def _pusher_jac(n: jax.Array, r: jax.Array) -> Tuple[jax.Array, jax.Array]:
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
def build_dynamic_lcs(
    pp: PlantParams,
    contact_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]],
    x0: jax.Array,
    u0: jax.Array,
    Minv: Optional[jax.Array] = None,
    obs: Optional[Tuple[jax.Array, jax.Array, jax.Array]] = None,
) -> LCS:
    """Anitescu pusher and Coulomb box ground friction, linearized at x0.

    Plus optional frictionless object-obstacle normal contacts.

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
                        jnp.zeros(No), jnp.zeros(No)], axis=1)   # (No,5)
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

    def col(J: jax.Array) -> jax.Array:
        return dt * (Minv @ J)

    zero = jnp.zeros(5)
    base_cols = [col(d1), col(d2),
                 zero, col(e0), -col(e0),
                 zero, col(e1), -col(e1),
                 zero, col(e2), -col(e2)]
    obs_cols = [col(Jo[o]) for o in range(No)]
    Gv = jnp.stack(base_cols + obs_cols, axis=1)                # (5, kd)

    G = jnp.zeros((10, kd))
    G = G.at[:5, :].set(dt * Gv)
    G = G.at[5:, :].set(Gv)

    Vfree_x = jnp.concatenate([jnp.zeros((5, 5)), A_v], axis=1)
    Vfree_u = B_v

    E = jnp.zeros((kd, 10))
    F = jnp.zeros((kd, kd))
    H = jnp.zeros((kd, 2))
    c = jnp.zeros(kd)

    def vel_row(Jrow: jax.Array) -> Tuple[jax.Array, ...]:
        return (Jrow @ Vfree_x, Jrow @ Vfree_u, Jrow @ Gv)

    for i, dj in ((0, d1), (1, d2)):
        ex, hu, fl = vel_row(dj)
        E = E.at[i, :].set(ex)
        H = H.at[i, :].set(hu)
        F = F.at[i, :].set(fl)
        c = c.at[i].set(phi / dt)

    def ground(base: int, e_axis: jax.Array, bound: jax.Array) -> None:
        nonlocal E, F, H, c
        g, pl, mn = base, base + 1, base + 2
        F = F.at[g, pl].set(-1.0)
        F = F.at[g, mn].set(-1.0)
        c = c.at[g].set(bound)
        ex, hu, fl = vel_row(e_axis)
        F = F.at[pl, g].set(1.0)
        F = F.at[pl, :].add(fl)
        E = E.at[pl, :].add(ex)
        H = H.at[pl, :].add(hu)
        ex, hu, fl = vel_row(-e_axis)
        F = F.at[mn, g].set(1.0)
        F = F.at[mn, :].add(fl)
        E = E.at[mn, :].add(ex)
        H = H.at[mn, :].add(hu)

    ground(2, e0, b[0])
    ground(5, e1, b[1])
    ground(8, e2, b[2])

    # object-obstacle rows: 0 <= lam _|_ (Jo . v_next + phi/dt)
    for o in range(No):
        ex, hu, fl = vel_row(Jo[o])
        E = E.at[11 + o, :].set(ex)
        H = H.at[11 + o, :].set(hu)
        F = F.at[11 + o, :].set(fl)
        c = c.at[11 + o].set(o_phi[o] / dt)

    return LCS(A=A, B=B, G=G, d=d, E=E, F=F, H=H, c=c, n=10, m=2, k=kd)


# =====================================================================
# Faithful forward simulator (P4 "simulate the plan" cost / validation).
# The Anitescu pusher LCP is PSD -> projected Gauss-Seidel; the ground box
# friction has a fixed bound and a diagonal mass block -> per-DOF clamp. A
# short splitting iteration couples them. This replaces the projected-Jacobi
# solve_lcp, which cannot move the zero-diagonal friction-cone variables.
# =====================================================================
def _pgs_psd(W: jax.Array, w: jax.Array, iters: int = 60) -> jax.Array:
    diag = jnp.diag(W)
    inv = 1.0 / jnp.where(diag > 1e-12, diag, 1.0)
    ncol = w.shape[0]

    def sweep(z: jax.Array, _: Any) -> Tuple[jax.Array, None]:
        def body(i: jax.Array, z: jax.Array) -> jax.Array:
            r = w[i] + W[i] @ z
            return z.at[i].set(jnp.maximum(0.0, z[i] - inv[i] * r))
        return jax.lax.fori_loop(0, ncol, body, z), None

    z, _ = jax.lax.scan(sweep, jnp.zeros_like(w), None, length=iters)
    return z


def simulate_step(
    pp: PlantParams,
    contact_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]],
    x: jax.Array,
    u: jax.Array,
    splits: int = 8,
    Minv: Optional[jax.Array] = None,
    obs_fn: Optional[Callable[[jax.Array], Tuple[jax.Array, ...]]] = None,
) -> jax.Array:
    """One faithful forward step: PGS on the pusher LCP, then ground box."""
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

    def pad(vg: jax.Array) -> jax.Array:
        return dt * (Minv @ jnp.concatenate([vg, jnp.zeros(2)]))

    def body(v_ground: jax.Array, _: Any) -> Tuple[jax.Array, None]:
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


def simulate_rollout(
    pp: PlantParams,
    contact_fn: Callable[[jax.Array], Tuple[jax.Array, jax.Array, jax.Array]],
    x0: jax.Array,
    us: jax.Array,
    Minv: Optional[jax.Array] = None,
    obs_fn: Optional[Callable[[jax.Array], Tuple[jax.Array, ...]]] = None,
) -> jax.Array:
    """States from `x0` under `us`, shape (len(us) + 1, 10)."""

    def step(x: jax.Array, u: jax.Array) -> Tuple[jax.Array, jax.Array]:
        xn = simulate_step(pp, contact_fn, x, u, Minv=Minv, obs_fn=obs_fn)
        return xn, xn
    _, xs = jax.lax.scan(step, x0, us)
    return jnp.concatenate([x0[None], xs], axis=0)

# =====================================================================
# Cost Hessian on the 10-dim state x = [q(5); v(5)].
#   object pose error (q_pos on x,y; q_theta on theta),
#   EE-tracking coupling (w_ee, pull tip toward object as in the ref approach
#   cost -- here just a light tip regularizer), velocity regularizer w_v.
# =====================================================================
def _state_cost_hessian(
    q_pos: float, q_theta: float, w_ee: float, w_v: float
) -> jax.Array:
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
    """Push/reposition state, the progress window, and failed contacts."""

    is_c3: jax.Array          # 1.0 = pushing (C3), 0.0 = repositioning
    target: jax.Array         # current repositioning target (world EE xy)
    target_body: jax.Array    # target contact in body frame (unsucc buffer)
    cost_hist: jax.Array      # (W,) object config-cost history
    n_prog: jax.Array         # steps since last progress reset
    unsucc: jax.Array         # (U, 2) body-frame contacts that made no progress
    rng: jax.Array
    crossed: jax.Array        # 1.0 once object XY entered the pose band


class C3SamplingCore:
    """The outer-loop logic, operating on (obj, ee, v_obj).

    Framework-free so it is testable without MJX; `C3MJXSampling` wraps it
    with the state extraction.
    """

    def __init__(
            self,
            footprint: Any,
            plant: PlantParams,
            goal: jax.Array,
            u_min: jax.Array,
            u_max: jax.Array,
            robot_radius: float = 0.02,
            num_random: int = 3,
            horizon: int = 10,
            admm_iters: int = 3,
            rho: float = 0.1,
            rho_scale: float = 3.0,
            rho_u: float = 1.0,
            q_pos: float = 200.0,
            q_theta: float = 40.0,
            w_ee: float = 10.0,
            w_v: float = 0.05,
            qf_pos: float = 2000.0,
            qf_theta: float = 400.0,
            r_r: float = 0.05,
            pos_success: float = 0.03,
            theta_success: float = 0.09,
            progress_window: int = 16,
            # dairlib kConfigCostDrop: 0.5 over 16 loops
            progress_drop: float = 0.5,
            # position-first: ignore orientation until within 5 cm
            cost_switching_threshold_distance: float = 0.05,
            hyst_c3_to_repos_frac: float = 0.6,
            hyst_c3_to_repos_frac_position: float = 0.7,
            hyst_repos_to_c3_frac: float = 0.9,
            hyst_repos_to_c3_frac_position: float = 0.5,
            hyst_repos_to_repos_frac: float = 0.7,
            hyst_repos_to_repos_frac_position: float = 0.7,
            contact_thresh: float = 0.02,
            safe_margin: float = 0.02,
            align_tol: float = 0.35,
            max_dphi: float = 0.6,
            straight_line_angle: float = 0.3,
            n_boundary_per_edge: int = 8,
            n_unsuccessful: int = 8,
            unsucc_radius: float = 0.03,
            obstacles: Tuple[Any, ...] = (),
            n_obstacles: int = 2,
            obs_margin: float = 0.01,
    ):
        """Read the plant, the P1-P4 thresholds, and the candidate set."""
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
        self.progress_window = progress_window
        self.progress_drop = progress_drop
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
        self.align_tol, self.max_dphi = align_tol, max_dphi
        self.straight_line_angle = straight_line_angle
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
        self.obs_shapes = list(obstacles)
        self.n_obs = min(n_obstacles, len(self.obs_shapes))
        self.obs_margin = obs_margin

    def init_state(self, seed: int = 0) -> C3SampState:
        """Start in repositioning, with an empty progress window."""
        W = self.progress_window
        return C3SampState(is_c3=jnp.asarray(0.0),
                           target=jnp.zeros(2),
                           target_body=jnp.zeros(2),
                           cost_hist=jnp.full((W,), 1e12),
                           n_prog=jnp.asarray(0),
                           unsucc=jnp.full((self.n_unsucc, 2), 1e3),
                           rng=jax.random.key(seed),
                           crossed=jnp.asarray(0.0))

    def _plan_cost(self, xs: jax.Array) -> jax.Array:
        dpos = xs[:, :2] - self.goal[:2]
        dth = wrap_angle(xs[:, 2] - self.goal[2])
        return jnp.sum(self.q_pos * jnp.sum(dpos**2, axis=1) +
                       self.q_theta * dth**2)

    def _config_cost(self, obj: jax.Array) -> jax.Array:
        return (self.q_pos * jnp.sum((obj[:2] - self.goal[:2])**2) +
                self.q_theta * wrap_angle(obj[2] - self.goal[2])**2)

    def _obs_contacts(
        self, q: jax.Array
    ) -> Optional[Tuple[jax.Array, jax.Array, jax.Array]]:
        """N-closest object-obstacle contacts, or None with no obstacles.

        Returns (phi (No,), n (No,2), r (No,2)); n is the world push-away
        normal.
        """
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

    def _ee_from_body(
        self, pb: jax.Array, oxy: jax.Array, theta: jax.Array
    ) -> jax.Array:
        cw = oxy + rotate(theta, pb)
        _, gr = self.footprint.sdf_and_grad(pb)
        return cw + self.robot_radius * rotate(theta, gr)

    def _reposition_move(
        self, q_ee: jax.Array, target: jax.Array, c: jax.Array
    ) -> jax.Array:
        """Dairlib kCircular reposition, planar.

        If the new contact is only a small angle around the object from the
        current EE, go straight to it
        (use_straight_line_traj_within_angle); otherwise retreat to the ring,
        arc around, then approach -- so the EE never drags the (non-convex)
        object on a large-angle switch.
        """
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

    def step(
        self,
        obj: jax.Array,
        ee: jax.Array,
        v_obj: jax.Array,
        s: C3SampState,
        Minv: Optional[jax.Array] = None,
    ) -> Tuple[jax.Array, C3SampState]:
        """One outer-loop pass: rank contacts, then push or reposition."""
        theta, oxy = obj[2], obj[:2]
        rng, _ = jax.random.split(s.rng)

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

        # --- mesh-normal contact ranking (position-only until crossed) ---
        cw = oxy[None, :] + jax.vmap(lambda pb: rotate(theta, pb))(
            self.cand_body)
        nw = jax.vmap(lambda gr: rotate(theta, gr))(self.cand_normal)
        ee_cand = cw + self.robot_radius * nw
        r_lev = cw - oxy[None, :]
        push = -nw
        e_t = self.goal[:2] - oxy
        e_r = wrap_angle(self.goal[2] - theta)
        f_trans = push @ e_t
        f_rot = (r_lev[:, 0] * push[:, 1] - r_lev[:, 1] * push[:, 0]) * e_r
        score = self.q_pos * f_trans + q_theta_eff * f_rot
        d_bad = jnp.linalg.norm(self.cand_body[:, None, :] -
                                s.unsucc[None, :, :],
                                axis=-1)
        score = jnp.where(
            jnp.min(d_bad, axis=1) < self.unsucc_radius, -1e9, score)
        topk = jax.lax.top_k(score, self.num_random)[1]
        heur_ees = ee_cand[topk]

        samples = jnp.concatenate([ee[None, :], s.target[None, :], heur_ees],
                                  axis=0)
        v5 = jnp.concatenate([v_obj, jnp.zeros(2)])

        def plan_cost(xs: jax.Array) -> jax.Array:
            dpos = xs[:, :2] - self.goal[:2]
            dth = wrap_angle(xs[:, 2] - self.goal[2])
            return jnp.sum(self.q_pos * jnp.sum(dpos**2, axis=1) +
                           q_theta_eff * dth**2)

        # Same for every candidate: depends on the object pose alone.
        obs = self._obs_contacts(obj)

        def solve_one(p_i: jax.Array) -> Tuple[jax.Array, jax.Array]:
            x_init = jnp.concatenate([obj, p_i, v5])
            lcs = build_dynamic_lcs(self.plant, self.contact_fn, x_init,
                                    jnp.zeros(2), Minv=Minv, obs=obs)
            _, us, _ = c3_solve(
                lcs, x_init, self.x_ref, Q_eff, self.R, Qf_eff,
                rho=self.rho, horizon=self.horizon, admm_iters=self.admm_iters,
                u_min=self.u_min, u_max=self.u_max, rho_u=self.rho_u,
                rho_scale=self.rho_scale)
            sim_xs = simulate_rollout(self.plant, self.contact_fn, x_init, us,
                                      Minv=Minv, obs_fn=self._obs_contacts)
            return plan_cost(sim_xs), us[0]

        costs, first_us = jax.vmap(solve_one)(samples)
        curr_cost, push_u = costs[0], first_us[0]
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

        repos_action = self._reposition_move(ee, new_target, oxy)
        u0 = jnp.where(new_is_c3 > 0.5, push_u, repos_action)
        u0 = jnp.where(goal_met, jnp.zeros(2), u0)
        return u0, s.replace(is_c3=new_is_c3,
                             target=new_target,
                             target_body=new_target_body,
                             cost_hist=cost_hist,
                             n_prog=n_prog,
                             unsucc=unsucc,
                             rng=rng,
                             crossed=crossed.astype(jnp.float32))


@dataclass
class C3SamplingParams:
    """Spline knots for the base class, plus the outer loop's own state."""

    tk: jax.Array
    mean: jax.Array
    rng: jax.Array
    samp: C3SampState


class C3MJXSampling(SamplingBasedController):
    """Push Anything C3+ (local C3 + sampling/reposition) as a flat baseline."""

    def __init__(self,
                 task: Any,
                 *,
                 plan_horizon: float,
                 num_knots: int,
                 seed: int = 0,
                 robot_radius: float = 0.02,
                 mu_p: float = 0.5,
                 kv: float = 100.0,
                 num_random: int = 3,
                 admm_iters: int = 3,
                 rho: float = 0.1,
                 rho_scale: float = 3.0,
                 **core_kwargs):
        """Read the MJX plant's constants, then build the outer loop."""
        super().__init__(task,
                         num_randomizations=1,
                         risk_strategy=None,
                         seed=seed,
                         plan_horizon=plan_horizon,
                         spline_type="zero",
                         num_knots=num_knots,
                         iterations=1)
        if task.model.nu != 2:
            raise ValueError("C3MJXSampling targets robot='point' (nu=2)")
        import numpy as np  # noqa: PLC0415
        self.block_dofs = jnp.asarray(task.block_dofs)
        self.pusher_dofs = jnp.asarray(task.pusher_dofs)
        self.pusher_bid = int(task.pusher_body_id)
        self.idx5 = jnp.concatenate([self.block_dofs, self.pusher_dofs])
        m = task.model
        self.pusher_offset = jnp.asarray(
            np.asarray(m.body_pos)[self.pusher_bid][:2])  # live EE from qpos
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
                                   task.u_min,
                                   task.u_max,
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

    def init_params(
        self,
        initial_knots: Optional[jax.Array] = None,
        seed: int = 0,
    ) -> "C3SamplingParams":
        """A zero plan and a fresh outer-loop state."""
        tk = jnp.linspace(0.0, self.plan_horizon, self.num_knots)
        mean = jnp.zeros((self.num_knots, 2))
        return C3SamplingParams(tk=tk,
                                mean=mean,
                                rng=jax.random.key(seed),
                                samp=self.core.init_state(seed))

    def optimize(
        self, state: mjx.Data, params: "C3SamplingParams"
    ) -> Tuple["C3SamplingParams", Trajectory]:
        """Run the outer loop, then stash its action as the spline knots."""
        new_tk = jnp.linspace(0.0, self.plan_horizon,
                              self.num_knots) + state.time
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

    def sample_knots(
        self, params: "C3SamplingParams"
    ) -> Tuple[jax.Array, "C3SamplingParams"]:
        """The single C3 plan; there is no sample population to draw."""
        return params.mean[None, ...], params

    def update_params(
        self, params: "C3SamplingParams", rollouts: Trajectory
    ) -> "C3SamplingParams":
        """Nothing to update: `optimize` already wrote the plan."""
        return params
