"""Analytic object model for quasi-static planar pushing.

Implements the object-level subproblem of the object-informed MPPI
formulation: closed-form SE(2) limit-surface dynamics (paper eq. 4-5) plus
the object-level stage costs (paper eq. 18-19). This is deliberately
independent of MuJoCo -- a task supplies the goal, footprint, and obstacle
field, and gets dynamics and costs back, rather than re-deriving the
geometry in every task file.
"""

from typing import Optional, Sequence, Union

import jax
import jax.numpy as jnp

from oim.objects.sdf import ObstacleField, Polygon, Shape, rotate

WrenchWeights = Union[float, Sequence[float]]


def wrench_weights(value: WrenchWeights) -> jax.Array:
    """Broadcast a scalar or an `[f_x, f_y, tau]` triple to a (3,) vector.

    Lets a per-channel weight be written as one number when the channels
    are meant to share it, without the reader having to know which form is
    in play downstream.

    Args:
        value: A scalar, or three values in wrench order.

    Returns:
        The weights, shape (3,).
    """
    return jnp.broadcast_to(jnp.asarray(value, dtype=float), (3,))


def wrap_angle(a: jax.Array) -> jax.Array:
    """Wrap angle(s) to (-pi, pi]."""
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def se2_distance_sq(
    pose: jax.Array, goal: jax.Array, w_pos: float, w_theta: float
) -> jax.Array:
    """Weighted squared SE(2) distance d^2(x, g), paper eq. 19.

    Args:
        pose: SE(2) pose(s) [x, y, theta], shape (..., 3).
        goal: Goal pose [x, y, theta], shape (3,).
        w_pos: Weight on the translational error.
        w_theta: Weight on the (wrapped) rotational error.

    Returns:
        The weighted squared distance, shape (...,).
    """
    diff_pos = pose[..., :2] - goal[:2]
    diff_theta = wrap_angle(pose[..., 2] - goal[2])
    return w_pos * jnp.sum(diff_pos**2, axis=-1) + w_theta * diff_theta**2


class PlanarPushingObject:
    """Quasi-static planar object driven by a contact wrench.

    Dynamics are the limit-surface relation (paper eq. 4-5):

        x^o_{t+1} = x^o_t + dt * D w^o_t,
        D = diag(mu*m*g, mu*m*g, c*r*mu*m*g)^{-1},

    where w^o = [f_x, f_y, tau] is the contact wrench applied to the object.
    D is isotropic in translation and the torque is a scalar, so in 2D the
    relation is identical whether the wrench is expressed in the world frame
    or the object body frame -- the implementation uses the world frame
    throughout, matching what the simulator reports.
    """

    def __init__(
        self,
        dt: float,
        goal: jax.Array,
        footprint: Polygon,
        obstacles: Optional[ObstacleField] = None,
        mu: float = 0.4,
        mass: float = 2.0,
        gravity: float = 9.81,
        pressure_coeff: float = 1.0,
        limit_surface_radius: float = 0.06,
        w_pos: float = 40.0,
        w_theta: float = 10.0,
        wf_pos: float = 500.0,
        wf_theta: float = 150.0,
        w_effort: float = 0.01,
        w_rate: WrenchWeights = 0.0,
        w_obstacle: float = 10.0,
        obstacle_decay: float = 0.02,
        support: Optional[Shape] = None,
        w_support: float = 0.0,
        support_margin: float = 0.0,
        boundary_samples_per_edge: int = 4,
        wrench_sample_fraction: float = 1.0,
    ) -> None:
        """Configure the object's physics, goal, geometry, and cost weights.

        Args:
            dt: Planning timestep for the forward-Euler integration.
            goal: Goal pose [x, y, theta].
            footprint: The object's outline in its body frame, used for
                obstacle clearance checks.
            obstacles: Static obstacles to avoid. None means no obstacles.
            mu: Friction coefficient between object and support surface.
            mass: Object mass.
            gravity: Gravitational acceleration.
            pressure_coeff: The limit-surface pressure coefficient `c`.
            limit_surface_radius: The characteristic radius `r`.
            w_pos: Running-cost weight on translational goal error.
            w_theta: Running-cost weight on rotational goal error.
            wf_pos: Terminal-cost weight on translational goal error.
            wf_theta: Terminal-cost weight on rotational goal error.
            w_effort: Weight on the squared wrench (control effort).
            w_rate: Weight on the squared *change* in wrench between
                consecutive steps, normalized by `wrench_limit`. Either
                one number for all three channels or `[f_x, f_y, tau]`;
                0 disables it. See `rate_cost`.
            w_obstacle: Cost of a boundary point at zero clearance.
            obstacle_decay: e-folding length of that cost, in metres. The
                penalty has no cutoff, so a sampler sees which way is away
                at every distance -- xarm6 briefly used a hinge instead
                (2026-08-18) and real single_obstacle/YCB-clutter runs
                showed the object getting stuck against an obstacle rather
                than routing around it, since a hinge gives zero avoidance
                signal outside its margin.
            support: The region the object must stay ON -- the tabletop.
                A KEEP-IN region, the mirror of `obstacles`: those are
                shapes the footprint must stay out of, this is one it must
                stay inside. None (default) means unbounded support, which
                is what every scene assumed before this existed and is
                still correct for the ones with no table (`clutter`).
            w_support: Weight on leaving `support`. 0 disables the term
                even when a support shape is given.
            support_margin: How far inside the edge the penalty starts, in
                metres. The cost is zero while the whole footprint is at
                least this far in, so the object is free to work anywhere
                on the table except a strip along the rim.
            boundary_samples_per_edge: Footprint boundary sampling density.
            wrench_sample_fraction: A unit sample from the object optimizer
                maps to this fraction of the friction-cone limit. Sets
                `action_scale`.

                1.0, so a unit action *is* the friction-cone limit and the
                optimizer's box (the unit box, see
                `ConsensusTask.object_action_bounds`) is exactly the set of
                wrenches the support surface can transmit -- the paper's
                eq. 18 projection, as a box rather than its inscribed
                ellipsoid. Was 0.5, which combined with a bound that was
                itself the friction-cone limit to give a realized box of
                +/- (mu*m*g)^2 / 2: 3.92x the transmissible force, and only
                0.235x the transmissible torque. That torque figure is
                below `step`'s own breakaway threshold, so a pure rotation
                was unreachable by construction -- on a scene set that is
                entirely 90- and 180-degree turns.
        """
        self.dt = dt
        self.goal = jnp.asarray(goal)
        self.footprint = footprint
        self.obstacles = obstacles or ObstacleField([])

        # Limit-surface compliance D and its inverse, the friction-cone
        # limit (the largest wrench the support surface can transmit).
        f_limit = mu * mass * gravity
        tau_limit = pressure_coeff * limit_surface_radius * f_limit
        self.wrench_limit = jnp.array([f_limit, f_limit, tau_limit])
        # NOTE: PushT._consensus_from_twist inverts this by hand, as
        # `wrench_limit * qvel`. Any change to D has to be made there too.
        self.D = 1.0 / self.wrench_limit

        # A unit sample from the object optimizer -> physical wrench.
        self.action_scale = wrench_sample_fraction * self.wrench_limit

        self.w_pos, self.w_theta = w_pos, w_theta
        self.wf_pos, self.wf_theta = wf_pos, wf_theta
        self.w_effort = w_effort
        self.w_rate = wrench_weights(w_rate)
        self.w_obstacle = w_obstacle
        self.obstacle_decay = obstacle_decay
        self.support = support
        self.w_support = w_support
        self.support_margin = support_margin
        self.boundary_samples = footprint.sample_boundary(
            boundary_samples_per_edge
        )

    @property
    def dim(self) -> int:
        """Dimension of the wrench / consensus variable."""
        return 3

    def step(self, pose: jax.Array, wrench: jax.Array) -> jax.Array:
        """One forward-Euler step of the limit-surface dynamics (eq. 5).

        Friction is *subtracted*, not gated on:

            s = ||w / D^-1||,   x_{t+1} = x_t + dt * D w * max(0, 1 - 1/s)

        A limit surface's own definition is the boundary between sticking
        (wrenches inside it produce zero relative motion) and slipping
        (wrenches at or beyond it do), but eq. 5's plain proportional
        formula extends across that boundary with no such cutoff, so a
        wrench well inside `wrench_limit` still predicts a small nonzero
        step. Diagnosed (2026-08-09/10) as the root cause of the near-goal
        stall: as position error shrinks the optimal wrench shrinks with
        it, continuously, with nothing to stop it settling below the real
        breakaway force -- confirmed against real runs, where the realized
        wrench is genuinely near-zero on 96-98% of steps in that regime.

        The first fix for that zeroed sub-threshold wrenches and passed the
        *full* wrench above threshold. That restored sticking but made the
        map discontinuous: one-step displacement jumped from 0 to
        `dt * 1.0` (0.05 m at the shipped dt) the instant `s` crossed 1, so
        the reachable set had a hole in `(0, 0.05)` -- which is exactly the
        goal tolerance. The object could not make a correction smaller than
        the ball it was aiming at, and the two available behaviours near
        the goal were freeze (below threshold) and overshoot (above). A
        goal-proximity snap on the object action existed to pick the
        latter; it was removed once this form made it unnecessary.

        Subtracting instead is both the standard Coulomb form and what
        MuJoCo's `frictionloss` already does -- its acceleration under an
        over-threshold push is `(|w| - mu m g) / m`, the excess, which is
        why the simulator moved ~0.003 m where this model predicted 0.075.
        Motion now goes continuously to zero as the wrench approaches the
        cone boundary, so a smaller sampled force really does produce a
        smaller step: `s = 1.05` gives 2.5 mm, 20x finer than the 0.05
        tolerance, where the gated form gave 52.5 mm.
        """
        # Double `where` throughout: `w = 0` is both a perfectly ordinary
        # input here (the deadzone's interior) and a singularity of both
        # `norm` and the reciprocal below. Guarding only the output leaves
        # a nan in the *gradient* -- `jnp.linalg.norm` is not
        # differentiable at the origin, and `0 * inf` is nan -- which would
        # propagate silently, since nothing on the sampling path
        # differentiates through the dynamics today but plenty could.
        squared = jnp.sum((wrench / self.wrench_limit) ** 2)
        positive = squared > 0.0
        normalized_mag = jnp.where(
            positive, jnp.sqrt(jnp.where(positive, squared, 1.0)), 0.0
        )
        slipping = normalized_mag > 1.0
        slip = jnp.where(
            slipping,
            1.0 - 1.0 / jnp.where(slipping, normalized_mag, 1.0),
            0.0,
        )
        new_pose = pose + self.dt * self.D * wrench * slip
        return new_pose.at[2].set(wrap_angle(new_pose[2]))

    def world_boundary(self, pose: jax.Array) -> jax.Array:
        """The footprint boundary samples transformed into the world frame."""
        return pose[:2] + rotate(pose[2], self.boundary_samples)

    def obstacle_cost(self, pose: jax.Array) -> jax.Array:
        """Object-vs-obstacle clearance: `w_obstacle * exp(-d/obstacle_decay)`.

        Split out from `running_cost` so `PushT`'s flat (non-ADMM)
        `running_cost` -- which has no wrench decision to score, only a
        pose, and so reconstructs this one term of eq. 18 by hand rather
        than calling `running_cost` itself -- reads the exact same formula
        this method uses, instead of duplicating it at a second call site
        that could drift out of sync.
        """
        return self.obstacles.exp_cost(
            self.world_boundary(pose), self.w_obstacle, self.obstacle_decay
        )

    def support_cost(self, pose: jax.Array) -> jax.Array:
        """Penalty for the footprint leaving the support surface.

        The mirror of `ObstacleField.hinge_cost`, sign-flipped. A `Shape`
        SDF is negative inside and positive outside, so `hinge_cost`
        charges `clip(margin - d, 0, inf)**2` for being NEAR a shape, and
        this charges `clip(d + margin, 0, inf)**2` for being near LEAVING
        one -- zero while every boundary point is at least `support_margin`
        inside the table, then quadratic, and still rising once the point
        is off the edge entirely.

        Quadratic rather than the exponential `obstacle_cost` defaults to,
        because the two failure modes are not alike. An obstacle needs a
        gradient from far away, since the object has to route around it
        before it ever gets close. The table edge does not: everywhere the
        object is legitimately working is inside, so a cost that is nonzero
        across the whole tabletop would bias every plan toward the middle
        and fight the goal term. Zero until the rim, then hard, is the
        shape that says "anywhere on the table is fine, the edge is not".

        Summed over boundary points, like `hinge_cost` and unlike
        `exp_cost`'s min: a corner hanging over the edge is a real loss of
        support, and the more of the footprint that overhangs the closer
        the block is to tipping off, so more points outside should cost
        more.

        Returns 0 when the object has no support shape (`clutter`, which
        has no table) or `w_support` is 0.
        """
        if self.support is None or self.w_support == 0.0:
            return jnp.zeros(())
        d = self.support.sdf(self.world_boundary(pose))
        overhang = jnp.clip(d + self.support_margin, 0.0, None)
        return self.w_support * jnp.sum(overhang**2)

    def running_cost(
        self,
        pose: jax.Array,
        wrench: jax.Array,
        weight_scale: jax.Array = 1.0,
    ) -> jax.Array:
        """Object stage cost: goal tracking + proximity + effort (eq. 18).

        `weight_scale` multiplies the goal term ONLY -- the same time ramp
        the robot block applies to its own `ell_o`, so as a run goes on both
        blocks agree that reaching the goal matters more than the shaping
        around it. Proximity and effort keep their weights: letting the ramp
        run away with them would buy goal error by driving into an obstacle.
        """
        cost = weight_scale * se2_distance_sq(
            pose, self.goal, self.w_pos, self.w_theta
        )
        cost += self.obstacle_cost(pose)
        cost += self.support_cost(pose)
        cost += self.w_effort * jnp.sum(wrench**2)
        return cost

    def rate_cost(
        self, wrenches: jax.Array, w_prev: Optional[jax.Array] = None
    ) -> jax.Array:
        """Per-channel sum of squared step-to-step changes in the wrench.

            sum_t sum_i w_rate[i] * (w_{t+1,i}/limit_i - w_{t,i}/limit_i)^2

        Not in the paper. The object block samples one *independent* knot
        per timestep under a zero-order hold, so nothing couples w_t to
        w_{t+1} and the effort term -- which sees only |w_t| -- is happy
        with a sequence that reverses every step. Measured on icra_sign:
        the executed wrench changed by 118% of its own magnitude per step
        and turned the force direction by a median 73 degrees, reversing
        by more than 90 degrees on 43% of steps.

        That is not free in reality: reversing the push means relocating
        the contact to the opposite face. This is the cheapest stand-in
        for that cost the object model can carry, since the block has no
        notion of *where* it is being pushed (that is the robot block's
        concern -- see the withdrawn contact parameterization in
        README_ADMM.md §9).

        Being quadratic is the point: spreading a given change over k
        steps costs 1/k of taking it in one, so the cheapest way to reach
        a new wrench is to ramp into it rather than jump. Weighted per
        channel because the three do not cost the same to change -- a
        force reversal relocates the contact, while a torque change can
        often be had by sliding along the same face -- and because the
        torque limit is ~17x smaller than the force limit on these
        scenes, so a shared weight taxes rotation hardest exactly where
        rotation is what the goal needs.

        Normalized by `wrench_limit` before squaring, like the ADMM
        penalty, so `w_rate` is scale-free and reads against the other
        weights rather than against newtons.

        Args:
            wrenches: The horizon's wrenches, (H, 3).
            w_prev: The wrench the previous solve already intended for the
                first step, anchoring the sequence to it. `None` scores
                only the differences within the horizon.

        Returns:
            The scalar penalty for this sequence.
        """
        normalized = wrenches / self.wrench_limit
        if w_prev is not None:
            normalized = jnp.concatenate(
                [(w_prev / self.wrench_limit)[None, :], normalized], axis=0
            )
        return jnp.sum(self.w_rate * jnp.diff(normalized, axis=0) ** 2)

    def terminal_cost(
        self, pose: jax.Array, weight_scale: jax.Array = 1.0
    ) -> jax.Array:
        """Object terminal cost: heavier goal tracking only (ell_f)."""
        return weight_scale * se2_distance_sq(
            pose, self.goal, self.wf_pos, self.wf_theta
        )


def _boundary_edges(inside: list, nx: int, ny: int) -> dict:
    """Directed grid edges separating inside from out, interior on the left.

    Keyed by start node so the walk in `boxes_footprint` is a lookup. One
    node starts at most one edge for a region whose cells meet edge-to-edge
    rather than only at a corner, which is what a union of overlapping
    axis-aligned boxes always gives.
    """

    def within(i: int, j: int) -> bool:
        return 0 <= i < nx and 0 <= j < ny and inside[i][j]

    step = {}
    for i in range(nx):
        for j in range(ny):
            if not inside[i][j]:
                continue
            for occupied, node, nxt in (
                (within(i, j - 1), (i, j), (i + 1, j)),
                (within(i + 1, j), (i + 1, j), (i + 1, j + 1)),
                (within(i, j + 1), (i + 1, j + 1), (i, j + 1)),
                (within(i - 1, j), (i, j + 1), (i, j)),
            ):
                if not occupied:
                    step[node] = nxt
    return step


def _walk_boundary(step: dict) -> list:
    """The closed loop through `step`, starting from its lowest node.

    Raises:
        ValueError: If the loop does not use every edge -- i.e. the region
            has a hole or a second component, which one `Polygon` cannot
            describe.
    """
    start = min(step)
    loop, node = [start], step[start]
    while node != start:
        loop.append(node)
        node = step[node]
    if len(loop) != len(step):
        raise ValueError(
            "boxes_footprint: the union is not one simply-connected region "
            f"({len(step)} boundary edges, outer loop uses {len(loop)}); a "
            "Polygon is a single closed loop, so it cannot describe a hole "
            "or a second component"
        )
    return loop


def boxes_footprint(
    boxes: Sequence[Sequence[float]], tol: float = 1e-9
) -> Polygon:
    """Exact outline of a union of AXIS-ALIGNED boxes, as one polygon.

    The generic form of `t_shape_footprint`/`c_shape_footprint`, which are
    this same computation written out by hand for two particular shapes --
    it reproduces both of them vertex for vertex. Use it for an object
    whose collision geometry is a convex decomposition into boxes: the
    MJCF geoms and the analytic footprint are then derived from ONE list
    of numbers and cannot drift, which is exactly what
    `tests/test_scenes.py::test_footprint_matches_the_block_geoms` checks
    geom by geom.

    Works by cutting the plane on every box edge, marking each cell of the
    resulting grid inside or out, and walking the boundary edges -- so the
    result is exact rather than sampled, and carries a vertex only where
    the outline actually turns.

    Args:
        boxes: `(centre_x, centre_y, half_x, half_y)` per box, in the
            object's own body frame.
        tol: Coordinate rounding, and the collinearity threshold used to
            drop redundant vertices.

    Returns:
        The union outline as a `Polygon`, wound counter-clockwise.

    Raises:
        ValueError: If `boxes` is empty, or the union is not one connected
            region without holes (which this cannot describe -- a
            `Polygon` is a single closed loop).
    """
    if not len(boxes):
        raise ValueError("boxes_footprint needs at least one box")
    # Plain Python floats, never `jnp.asarray`: jax defaults to float32, and
    # at these magnitudes two edges that coincide exactly in the source
    # numbers come out ~4e-9 apart, which splits the grid below into
    # zero-width columns and corrupts the walk.
    b = [tuple(float(v) for v in row) for row in boxes]

    def _merge(values: Sequence[float]) -> list:
        """Sorted, with values within `tol` of each other collapsed."""
        out = []
        for v in sorted(values):
            if not out or v - out[-1] > tol:
                out.append(v)
        return out

    xs = _merge([c[0] - c[2] for c in b] + [c[0] + c[2] for c in b])
    ys = _merge([c[1] - c[3] for c in b] + [c[1] + c[3] for c in b])
    nx, ny = len(xs) - 1, len(ys) - 1
    mid_x = [(xs[i] + xs[i + 1]) / 2.0 for i in range(nx)]
    mid_y = [(ys[j] + ys[j + 1]) / 2.0 for j in range(ny)]
    inside = [[False] * ny for _ in range(nx)]
    for cx, cy, hx, hy in b:
        for i, mx in enumerate(mid_x):
            if not (cx - hx - tol <= mx <= cx + hx + tol):
                continue
            for j, my in enumerate(mid_y):
                if cy - hy - tol <= my <= cy + hy + tol:
                    inside[i][j] = True

    loop = _walk_boundary(_boundary_edges(inside, nx, ny))

    pts = [(xs[i], ys[j]) for i, j in loop]
    kept = []
    for k, (x, y) in enumerate(pts):
        ax, ay = pts[k - 1]
        bx, by = pts[(k + 1) % len(pts)]
        if abs((x - ax) * (by - y) - (bx - x) * (y - ay)) > tol:
            kept.append((x, y))
    return Polygon(jnp.array(kept))


def t_shape_footprint(
    crossbar_half: Sequence[float] = (0.090, 0.015),
    stem_half: Sequence[float] = (0.015, 0.060),
    crossbar_y: float = 0.030,
    stem_y: float = -0.045,
) -> Polygon:
    """Build the outline of a capital-T footprint from its two box halves.

    Defaults match the `pusht_clutter` MJCF's `block_crossbar`/`block_stem`
    geoms, so the analytic footprint and the simulated geometry agree.

    Args:
        crossbar_half: Half-extents (x, y) of the crossbar box.
        stem_half: Half-extents (x, y) of the stem box.
        crossbar_y: Crossbar center offset along y.
        stem_y: Stem center offset along y.

    Returns:
        The T outline as a `Polygon`, wound counter-clockwise.
    """
    cx, cy = crossbar_half
    sx, sy = stem_half
    top, bot = crossbar_y + cy, crossbar_y - cy
    stem_bot = stem_y - sy
    return Polygon(
        jnp.array(
            [
                [-cx, top],
                [cx, top],
                [cx, bot],
                [sx, bot],
                [sx, stem_bot],
                [-sx, stem_bot],
                [-sx, bot],
                [-cx, bot],
            ]
        )
    )


def c_shape_footprint(
    half_width: float = 0.0483,
    half_height: float = 0.0515,
    half_stroke: float = 0.016,
) -> Polygon:
    """Build the outline of a block capital-C from its stroke dimensions.

    The C is a full-height spine down the left with a full-width bar at the
    top and another at the bottom. This is the exact union outline of the
    three boxes making up the `block` body in
    `models/xarm6_pusht_tabletop/icra_sign.xml`, so the analytic footprint
    and the simulated geometry agree there the same way `t_shape_footprint`
    agrees with the T scenes.

    Defaults match that MJCF: 0.0966 x 0.103 m with a 0.032 m stroke,
    taken from the `glyph_c` mesh the scene renders -- its spine spans
    x in [-0.0482, -0.0163] at mid-height. That mesh is a smoothly curved
    C; this is a block-letter stand-in with the same bounding box, stroke
    width and single concavity, and it is what the simulator actually
    collides. See `icra_sign.xml` for why the collision geometry is three
    boxes rather than the mesh, and why the C rather than another letter.

    Was 0.070 x 0.103 with a 0.020 stroke, sized to the older glyph set
    (0.64-0.71 aspect). The row was re-cut from a font running 0.80-0.98,
    where a 0.070-wide C would have been the narrowest thing in it.

    Args:
        half_width: Half the letter's overall width (x).
        half_height: Half the letter's overall height (y).
        half_stroke: Half the thickness of the spine and each bar.

    Returns:
        The C outline as a `Polygon`, wound counter-clockwise.
    """
    x, y, t = half_width, half_height, half_stroke
    # Inner faces: where the two bars end, and the spine's right face.
    top_bar_bot = y - 2.0 * t
    bot_bar_top = -y + 2.0 * t
    spine_inner = -x + 2.0 * t
    return Polygon(
        jnp.array(
            [
                [-x, -y],
                [x, -y],
                # Up the bottom bar's right end, then left along its top
                # face as far as the spine.
                [x, bot_bar_top],
                [spine_inner, bot_bar_top],
                # Up the spine's right face and out along the top bar.
                [spine_inner, top_bar_bot],
                [x, top_bar_bot],
                [x, y],
                [-x, y],
            ]
        )
    )
