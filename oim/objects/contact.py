"""Contact-point parameterization of the object-level consensus decision.

The plain object subproblem in `oim.algs.admm.ObjectSubproblem` samples the
consensus wrench w = [f_x, f_y, tau] directly. Nothing then constrains w to be
a wrench a *point contact somewhere on the object's boundary* could actually
apply: the sampler is free to propose a pure torque, a pulling force, or a
wrench outside the friction cone, none of which the robot block can ever
realize. The consensus then converges to something physically meaningless.

This module provides the alternative used by `oim.worlds.sim2d`: the object
block
decides a **contact action** a = [p_x, p_y, f_n, f_t] -- where to touch the
object (in its body frame) and with what normal/tangential force -- and the
wrench is *derived* through the contact Jacobian, w = J_c^T f. Every wrench
reachable this way is realizable by construction, because

* p is projected onto the object's boundary,
* f_n >= 0 is unilateral (contacts push, never pull), and
* |f_t| <= mu_c f_n keeps the force inside the friction cone.

The maps are plain JAX functions on arrays so they work identically inside a
jitted MJX rollout and in eager 2D debugging.
"""

from typing import Any, Tuple

import jax
import jax.numpy as jnp

from oim.objects.sdf import Shape, rotate

# Layout of the packed contact action a = [p_x, p_y, f_n, f_t].
CONTACT_ACTION_DIM = 4


def contact_force_to_com_wrench(
    pose: jax.Array, p_world: jax.Array, f_world: jax.Array
) -> jax.Array:
    """Map a world-frame contact force to a world-frame CoM wrench.

    This is w = J_c^T f for a planar point contact: the force passes straight
    through, and the torque is the moment of that force about the object's
    pose origin.

    Args:
        pose: Object SE(2) pose(s) [x, y, theta], shape (..., 3).
        p_world: Contact point(s) in the world frame, shape (..., 2).
        f_world: Contact force(s) in the world frame, shape (..., 2).

    Returns:
        Wrench [f_x, f_y, tau] of shape (..., 3), in the world frame about
        the object's pose origin -- the same frame and reference point the
        robot block's A^r reports, so both ADMM blocks agree on what z means.
    """
    r = p_world - pose[..., :2]
    tau = r[..., 0] * f_world[..., 1] - r[..., 1] * f_world[..., 0]
    return jnp.concatenate([f_world, tau[..., None]], axis=-1)


def contact_frame(
    shape: Shape, p_body: jax.Array, theta: jax.Array
) -> Tuple[jax.Array, jax.Array]:
    """World-frame inward normal and tangent at a body-frame contact point.

    Args:
        shape: The object's footprint, supplying the boundary normal.
        p_body: Contact point(s) in the object body frame, shape (..., 2).
        theta: Object heading(s), shape (...,).

    Returns:
        Inward unit normal and unit tangent, both shape (..., 2), in world
        coordinates. "Inward" means pointing into the object, i.e. the
        direction a pusher presses.
    """
    _, grad = shape.sdf_and_grad(p_body)
    n_body = -grad  # sdf gradient points outward; a contact pushes inward
    t_body = jnp.stack([-n_body[..., 1], n_body[..., 0]], axis=-1)
    return rotate(theta, n_body), rotate(theta, t_body)


def contact_action_to_wrench(
    shape: Shape, pose: jax.Array, action: jax.Array
) -> jax.Array:
    """Map a packed contact action to the world CoM wrench it applies.

    Args:
        shape: The object's footprint.
        pose: Object SE(2) pose [x, y, theta], shape (..., 3).
        action: Packed [p_x, p_y, f_n, f_t], shape (..., 4). The contact
            point is in the object's *body* frame; forces are magnitudes
            along the inward normal and the tangent.

    Returns:
        Wrench [f_x, f_y, tau] of shape (..., 3).
    """
    p_body, f_n, f_t = unpack_contact_action(action)
    n_world, t_world = contact_frame(shape, p_body, pose[..., 2])
    f_world = f_n[..., None] * n_world + f_t[..., None] * t_world
    p_world = pose[..., :2] + rotate(pose[..., 2], p_body)
    return contact_force_to_com_wrench(pose, p_world, f_world)


def pack_contact_action(
    p_body: jax.Array, f_n: jax.Array, f_t: jax.Array
) -> jax.Array:
    """Stack a body contact point and its force magnitudes into (..., 4)."""
    return jnp.concatenate([p_body, f_n[..., None], f_t[..., None]], axis=-1)


def unpack_contact_action(
    action: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Split (..., 4) into the contact point (..., 2) and f_n, f_t (...,)."""
    return action[..., :2], action[..., 2], action[..., 3]


def project_contact_action(
    shape: Shape, action: jax.Array, mu_c: float, f_max: float
) -> jax.Array:
    """Project a contact action onto the physically realizable set.

    Enforces the three constraints that make the derived wrench meaningful:
    the contact point lies on the object's boundary, the normal force is
    unilateral and bounded, and the tangential force respects the friction
    cone. Applied to *every* sample, so the object block can never score a
    wrench the robot has no hope of producing.

    Args:
        shape: The object's footprint.
        action: Packed contact action(s), shape (..., 4).
        mu_c: Coulomb friction coefficient between pusher and object.
        f_max: Maximum normal force.

    Returns:
        The projected action, same shape.
    """
    p_body, f_n, f_t = unpack_contact_action(action)
    p_body = shape.project_to_boundary(p_body)
    f_n = jnp.clip(f_n, 0.0, f_max)
    f_t = jnp.clip(f_t, -mu_c * f_n, mu_c * f_n)
    return pack_contact_action(p_body, f_n, f_t)


def estimate_contact_point(
    shape: Shape,
    pose: jax.Array,
    goal_cost_fn: Any,
    rng: jax.Array,
    candidates: jax.Array,
    f_n: float,
    num_samples: int = 48,
    iterations: int = 4,
    horizon: int = 5,
    sigma_init: float = 0.006,
    sigma_min: float = 0.004,
    sigma_decay: float = 0.6,
    temperature: float = 0.35,
) -> jax.Array:
    """Pick where on the boundary to push, by a short CEM search.

    `sample_contact_actions` is deliberately *local*: its rejection filter
    keeps every sample on the same face as the nominal, which is what stops
    a Gaussian step from flipping the contact to the opposite side of the
    object and reversing the wrench. The cost is that the object block can
    only ever slide the contact along one face -- it cannot decide to go
    around and push from somewhere else, which is exactly what is needed to
    route an object around an obstacle.

    This closes that gap. Each call scores a spread of candidate contact
    points by rolling the object forward a few steps under the wrench each
    would produce, then softmax-averages toward the good ones and shrinks
    the spread. Because the candidates start from `candidates` (a set of
    points covering the whole boundary) rather than from the incumbent, the
    search is global over faces rather than local to one.

    Args:
        shape: The object's footprint.
        pose: The object's current SE(2) pose.
        goal_cost_fn: Scores a batch of poses, `(N, 3) -> (N,)`. Typically
            the task's own object-level stage cost.
        rng: PRNG key.
        candidates: Boundary points to seed the search, shape (M, 2).
        f_n: Normal force to assume while scoring; the search is over
            *where* to push, not how hard.
        num_samples: Candidates evaluated per iteration.
        iterations: CEM refinement rounds.
        horizon: Steps to roll the object forward when scoring.
        sigma_init: Initial candidate spread.
        sigma_min: Floor on the spread.
        sigma_decay: Per-iteration spread decay.
        temperature: Softmax temperature over candidate costs.

    Returns:
        The chosen body-frame contact point, shape (2,).
    """
    # The estimator only needs a *relative* ranking of candidates, so the
    # object's compliance enters as a fixed positive scaling rather than the
    # task's exact dt*D; using a constant keeps this independent of the
    # object model and avoids threading dynamics through the signature.
    shape_step_scale = 0.01

    def _score(points: jax.Array) -> jax.Array:
        """Roll the object forward under each candidate's wrench."""
        n = points.shape[0]
        action = pack_contact_action(points, jnp.full(n, f_n), jnp.zeros(n))
        poses = jnp.broadcast_to(pose, (n, 3))

        def _step(p: jax.Array, _: jax.Array) -> Tuple[jax.Array, jax.Array]:
            # The wrench is recomputed at the *current* pose each step, so a
            # candidate is scored on the trajectory it actually induces.
            w = jax.vmap(contact_action_to_wrench, in_axes=(None, 0, 0))(
                shape, p, action
            )
            p = p + shape_step_scale * w
            return p, goal_cost_fn(p)

        _, costs = jax.lax.scan(_step, poses, jnp.arange(horizon))
        return jnp.sum(costs, axis=0)

    # Global step: score every candidate spanning the whole boundary and take
    # the outright best. `argmin`, not a softmax average -- averaging points
    # that lie on *different* faces is geometrically meaningless (the mean of
    # a left-face and a right-face contact is somewhere inside the object,
    # and its normal is unrelated to either), which is the same reason the
    # local sampler rejection-filters on normal alignment. Choosing the face
    # is inherently a discrete decision.
    best = candidates[jnp.argmin(_score(candidates))]

    # Local step: CEM refinement within the chosen face, where averaging is
    # meaningful because every sample shares a normal direction.
    def _iterate(
        carry: Tuple[jax.Array, jax.Array, jax.Array], _: jax.Array
    ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], None]:
        mean, sigma, key = carry
        key, sub = jax.random.split(key)
        pts = shape.project_to_boundary(
            mean[None, :] + sigma * jax.random.normal(sub, (num_samples, 2))
        )
        weights = jax.nn.softmax(-_score(pts) / temperature)
        mean = shape.project_to_boundary(
            jnp.sum(weights[:, None] * pts, axis=0)
        )
        return (mean, jnp.maximum(sigma * sigma_decay, sigma_min), key), None

    (mean, _, _), _ = jax.lax.scan(
        _iterate,
        (best, jnp.asarray(sigma_init), rng),
        jnp.arange(iterations),
    )
    return mean


def sample_contact_actions(
    shape: Shape,
    nominal: jax.Array,
    rng: jax.Array,
    num_samples: int,
    sigma_p: float,
    sigma_fn: float,
    sigma_ft: float,
    mu_c: float,
    f_max: float,
    tau_n: float,
    max_tries: int = 8,
) -> jax.Array:
    """Sample contact actions around a nominal, staying on the boundary.

    The contact point is perturbed and re-projected onto the boundary, then
    *rejection-filtered* on normal alignment: a candidate is kept only if its
    inward normal agrees with the nominal's to within `tau_n`. Without that
    filter a Gaussian step along the boundary can hop to a completely
    different face -- e.g. from the front of the T's stem to its back -- and
    the resulting wrench reverses sign, which reads to the consensus update
    as violent disagreement rather than as exploration.

    The rejection loop runs a fixed `max_tries` rounds with masking rather
    than looping until acceptance, so the whole thing stays jit-compatible.

    Args:
        shape: The object's footprint.
        nominal: Nominal action trajectory, shape (H, 4).
        rng: PRNG key.
        num_samples: Number of samples K to draw.
        sigma_p: Std-dev of the contact-point perturbation.
        sigma_fn: Std-dev of the normal-force perturbation.
        sigma_ft: Std-dev of the tangential-force perturbation.
        mu_c: Coulomb friction coefficient.
        f_max: Maximum normal force.
        tau_n: Minimum cosine between a candidate normal and the nominal's.
        max_tries: Rejection-sampling rounds.

    Returns:
        Sampled actions of shape (K, H, 4), each already projected onto the
        realizable set.
    """
    p_nom, f_n_nom, f_t_nom = unpack_contact_action(nominal)
    h = nominal.shape[0]

    # Reference normal per horizon step, from the nominal contact point.
    _, grad_nom = shape.sdf_and_grad(p_nom)
    n_nom = -grad_nom  # (H, 2)

    p_rng, fn_rng, ft_rng = jax.random.split(rng, 3)

    def _draw(key: jax.Array) -> jax.Array:
        noise = sigma_p * jax.random.normal(key, (num_samples, h, 2))
        return shape.project_to_boundary(p_nom[None] + noise)

    def _body(
        carry: Tuple[jax.Array, jax.Array, jax.Array], _: jax.Array
    ) -> Tuple[Tuple[jax.Array, jax.Array, jax.Array], None]:
        accepted, cand, key = carry
        key, sub = jax.random.split(key)
        fresh = _draw(sub)
        # Keep whatever was already accepted; retry the rest.
        cand = jnp.where(accepted[..., None], cand, fresh)
        _, grad_c = shape.sdf_and_grad(cand)
        aligned = jnp.sum(-grad_c * n_nom[None], axis=-1) >= tau_n
        return (accepted | aligned, cand, key), None

    key0, key1 = jax.random.split(p_rng)
    init = (
        jnp.zeros((num_samples, h), dtype=bool),
        _draw(key0),
        key1,
    )
    (accepted, cand, _), _ = jax.lax.scan(_body, init, jnp.arange(max_tries))
    # Anything still unaligned after `max_tries` falls back to the nominal
    # point, which is aligned with itself by construction.
    p_k = jnp.where(accepted[..., None], cand, p_nom[None])

    f_n_k = f_n_nom[None] + sigma_fn * jax.random.normal(
        fn_rng, (num_samples, h)
    )
    f_t_k = f_t_nom[None] + sigma_ft * jax.random.normal(
        ft_rng, (num_samples, h)
    )
    return project_contact_action(
        shape, pack_contact_action(p_k, f_n_k, f_t_k), mu_c, f_max
    )


# ---------------------------------------------------------------------------
# Frictionless (x, y, lambda) parameterization
#
# The 3-component form used as an ADMM *consensus variable*: where to touch
# and how hard along the inward normal, with no tangential component. Thin
# wrappers over the 4D action above with f_t == 0 rather than a parallel
# implementation, so the boundary projection and the normal-alignment
# rejection filter stay in one place.
#
# Dropping f_t is what makes it a consensus variable at all: with a
# tangential component, (p, f) -> w is many-to-one in a way the robot block
# cannot invert from what it observes. Frictionless, (p, lambda) and the
# pose determine w exactly, and w and the pose determine (p, lambda) back.
# ---------------------------------------------------------------------------

# Layout of a contact point c = [p_x, p_y, lambda].
CONTACT_POINT_DIM = 3


def pack_contact_point(p_body: jax.Array, lam: jax.Array) -> jax.Array:
    """Stack a body contact point and its normal force into (..., 3)."""
    return jnp.concatenate([p_body, lam[..., None]], axis=-1)


def unpack_contact_point(cp: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """Split (..., 3) into the contact point (..., 2) and lambda (...,)."""
    return cp[..., :2], cp[..., 2]


def contact_point_to_wrench(
    shape: Shape, pose: jax.Array, cp: jax.Array
) -> jax.Array:
    """Map [p_x, p_y, lambda] to the world CoM wrench it applies.

    Evaluated at the object's *current* pose, so one fixed contact point
    tracks the same material point as the object rotates and the wrench it
    produces turns with it.

    Args:
        shape: The object's footprint.
        pose: Object SE(2) pose [x, y, theta], shape (..., 3).
        cp: Packed [p_x, p_y, lambda], shape (..., 3), p in the body frame.

    Returns:
        Wrench [f_x, f_y, tau] of shape (..., 3).
    """
    p_body, lam = unpack_contact_point(cp)
    return contact_action_to_wrench(
        shape, pose, pack_contact_action(p_body, lam, jnp.zeros_like(lam))
    )


def wrench_to_contact_point(
    shape: Shape, pose: jax.Array, p_body: jax.Array, w: jax.Array
) -> jax.Array:
    """Read [p_x, p_y, lambda] off a wrench applied at a known point.

    The inverse of `contact_point_to_wrench` given where the contact is:
    lambda is the component of the wrench's force along the inward normal
    at `p_body`. Clipped at zero because a contact pushes and never pulls
    -- a negative projection means the observed force did not come from a
    contact there, and reporting it as a negative "normal force" would put
    the consensus value outside the set the object block can propose.

    Args:
        shape: The object's footprint.
        pose: Object SE(2) pose [x, y, theta], shape (..., 3).
        p_body: Body-frame contact point(s), shape (..., 2).
        w: Wrench [f_x, f_y, tau], shape (..., 3).

    Returns:
        Packed [p_x, p_y, lambda] of shape (..., 3).
    """
    n_world, _ = contact_frame(shape, p_body, pose[..., 2])
    lam = jnp.maximum(jnp.sum(w[..., :2] * n_world, axis=-1), 0.0)
    return pack_contact_point(p_body, lam)


def project_contact_point(
    shape: Shape, cp: jax.Array, f_max: float
) -> jax.Array:
    """Project onto the realizable set: p on the boundary, 0 <= lambda <= f_max."""
    p_body, lam = unpack_contact_point(cp)
    return pack_contact_point(
        shape.project_to_boundary(p_body), jnp.clip(lam, 0.0, f_max)
    )


def sample_contact_points(
    shape: Shape,
    nominal: jax.Array,
    rng: jax.Array,
    num_samples: int,
    sigma_p: float,
    sigma_lambda: float,
    f_max: float,
    tau_n: float,
    max_tries: int = 8,
) -> jax.Array:
    """Sample [p_x, p_y, lambda] around a nominal, staying on the boundary.

    Delegates to `sample_contact_actions` at `mu_c = 0`, which zeroes the
    tangential channel through its own projection, so the normal-alignment
    rejection filter (the thing that stops a Gaussian step hopping to the
    opposite face and reversing the wrench) is shared rather than copied.

    Args:
        shape: The object's footprint.
        nominal: Nominal trajectory, shape (H, 3).
        rng: PRNG key.
        num_samples: Number of samples K to draw.
        sigma_p: Std-dev of the contact-point perturbation.
        sigma_lambda: Std-dev of the normal-force perturbation.
        f_max: Maximum normal force.
        tau_n: Minimum cosine between a candidate normal and the nominal's.
        max_tries: Rejection-sampling rounds.

    Returns:
        Sampled contact points of shape (K, H, 3), already projected.
    """
    p_nom, lam_nom = unpack_contact_point(nominal)
    actions = sample_contact_actions(
        shape,
        pack_contact_action(p_nom, lam_nom, jnp.zeros_like(lam_nom)),
        rng,
        num_samples,
        sigma_p=sigma_p,
        sigma_fn=sigma_lambda,
        sigma_ft=0.0,
        mu_c=0.0,
        f_max=f_max,
        tau_n=tau_n,
        max_tries=max_tries,
    )
    p_k, f_n_k, _ = unpack_contact_action(actions)
    return pack_contact_point(p_k, f_n_k)
